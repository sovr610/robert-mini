import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
from reasoning_llm import ReasoningLLM, ModelRegistry
from data_loader import create_main_dataset
import os
from tqdm import tqdm
import gc
import math
import json
from datetime import datetime

# Use 8-bit AdamW to reduce optimizer memory footprint by 75%
try:
    import bitsandbytes as bnb
    USE_8BIT_OPTIM = True
except ImportError:
    USE_8BIT_OPTIM = False
    print("Warning: bitsandbytes not found. Will use SGD optimizer instead of AdamW.")

# Optimize CUDA memory allocation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def log_memory(stage: str):
    """Log detailed GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[MEMORY] {stage}:")
        print(f"         Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {max_allocated:.2f} GB | Total: {total:.2f} GB")

def count_parameters(model):
    """Count and categorize model parameters."""
    total = sum(p.numel() for p in model.parameters())
    embedding_params = model.tok_embeddings.weight.numel()
    core_params = total - embedding_params
    
    print(f"[PARAMS] Total: {total:,} ({total * 2 / 1024**3:.2f} GB in fp16)")
    print(f"         Embedding/Output (tied): {embedding_params:,} ({embedding_params * 2 / 1024**3:.2f} GB)")
    print(f"         Core Transformer: {core_params:,} ({core_params * 2 / 1024**3:.2f} GB)")
    print(f"         AdamW would need: {total * 2 * 4 / 1024**3:.2f} GB additional for states")

TOKENIZER_NAME = "Xenova/gpt-4o"  # Constant for tokenizer name

def save_checkpoint(model, config, optimizer, epoch: int, loss: float, path: str):
    """Save model checkpoint with metadata."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'epoch': epoch,
        'loss': loss,
        'tokenizer_name': TOKENIZER_NAME,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(checkpoint, path)

class TrainingLogger:
    """Simple JSON-based training logger for tracking metrics."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.jsonl")
        self.metrics_history = []
        
        print(f"[LOGGER] Logging to {self.log_file}")
    
    def log(self, metrics: dict, step: int = None):
        """Log metrics to file."""
        entry = {"timestamp": datetime.now().isoformat(), "step": step, **metrics}
        self.metrics_history.append(entry)
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_config(self, config: dict):
        """Log training configuration."""
        config_file = os.path.join(self.log_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)

def train():
    # 1. Configuration
    # Using verse-3b which fits on a single GPU with standard optimizer
    model_name = "verse-3b" 
    batch_size = 1 # Reduced to 1 to fit in memory
    gradient_accumulation_steps = 16 # Simulate batch size 16
    learning_rate = 1e-4  # Slightly lower for stability with warmup
    min_lr = 1e-6  # Minimum LR for cosine schedule
    warmup_steps = 100  # Warmup steps for stable training start
    max_grad_norm = 1.0  # Gradient clipping threshold
    epochs = 3 # Increased epochs for better training
    max_seq_len = 1024 # Reduced from 2048 to prevent OOM with large vocab
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize logger
    logger = TrainingLogger()

    # 2. Tokenizer
    # We use the o200k tokenizer (GPT-4o) via the Xenova/gpt-4o repository
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4o")
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Dataset
    print("Preparing dataset...")
    # Use larger dataset for meaningful learning. Set sample_size=None for full training.
    # 500 samples is too few - model will just memorize/repeat.
    dataset = create_main_dataset(sample_size=10000) 
    
    def collate_fn(batch):
        # Extract text from the "messages" format
        texts = []
        for item in batch:
            # Simple concatenation of user/assistant for training
            # In production, you'd apply a chat template
            full_text = ""
            for msg in item['messages']:
                full_text += f"{msg['role']}: {msg['content']}\n"
            texts.append(full_text)
            
        # Tokenize
        encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=max_seq_len, 
            return_tensors="pt"
        )
        
        input_ids = encodings['input_ids']
        # Create targets (same as input for causal LM)
        targets = input_ids.clone()
        
        return input_ids, targets

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    total_steps = len(dataloader) * epochs // gradient_accumulation_steps

    # 4. Model
    print(f"Initializing model: {model_name}")
    config = ModelRegistry.get_config(model_name)
    
    # Update config vocab size to match tokenizer
    config.vocab_size = len(tokenizer)
    config.max_seq_len = max_seq_len
    
    print(f"[CONFIG] dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}")
    print(f"[CONFIG] vocab_size={config.vocab_size}, max_seq_len={config.max_seq_len}")
    
    # Log configuration
    logger.log_config({
        "model_name": model_name,
        "dim": config.dim,
        "n_layers": config.n_layers,
        "vocab_size": config.vocab_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "epochs": epochs,
        "max_seq_len": max_seq_len,
        "total_steps": total_steps
    })
    
    log_memory("Before model creation")
    model = ReasoningLLM(config).to(device)
    log_memory("After model.to(device)")
    count_parameters(model)
    
    # 5. Optimizer with LR Schedule
    import torch.optim as optim
    if USE_8BIT_OPTIM:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        print("[OPTIM] Using 8-bit AdamW optimizer (memory efficient)")
    else:
        # SGD uses minimal additional memory for optimizer states
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01)
        print("[OPTIM] Using SGD optimizer (minimal memory footprint)")
    
    # Learning rate schedulers: warmup then cosine decay
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    print(f"[SCHEDULE] Warmup: {warmup_steps} steps, Total: {total_steps} steps, Min LR: {min_lr}")
    
    # Use ignore_index to not compute loss on padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)
    scaler = GradScaler('cuda') # Initialize Mixed Precision Scaler
    
    log_memory("After optimizer creation")

    # 6. Training Loop
    print("Starting training...")
    model.train()
    
    # Training tracking variables
    global_step = 0
    best_loss = float('inf')
    
    # Clear memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    log_memory("Before training loop")
    
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        optimizer.zero_grad(set_to_none=True) # Initialize gradients with None to save memory
        
        for batch_idx, (input_ids, targets) in enumerate(progress_bar):
            input_ids, targets = input_ids.to(device), targets.to(device)
            
            # Log memory at specific intervals
            if batch_idx == 0:
                log_memory(f"Epoch {epoch+1} - After first batch to GPU")
            
            # Mixed Precision Forward pass
            with autocast('cuda'):
                logits = model(input_ids)
                
                if batch_idx == 0:
                    log_memory(f"Epoch {epoch+1} - After forward pass")
                
                # Shift logits and targets for Causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                
                loss = criterion(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
                loss = loss / gradient_accumulation_steps # Normalize loss
            
            # Delete intermediate tensors to free memory
            del shift_logits, shift_labels, logits
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            if batch_idx == 0:
                log_memory(f"Epoch {epoch+1} - After backward pass")
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if batch_idx < gradient_accumulation_steps * 2:
                    log_memory(f"Before optimizer.step() (batch {batch_idx})")
                
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Log gradient norm occasionally for debugging
                if global_step % 100 == 0:
                    print(f"  [Step {global_step}] grad_norm: {grad_norm:.4f}")
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad(set_to_none=True) # set_to_none=True saves memory
                torch.cuda.empty_cache() # Clear cache to prevent fragmentation
                
                global_step += 1
                current_lr = scheduler.get_last_lr()[0]
                
                # Log training step
                logger.log_step(global_step, loss.item() * gradient_accumulation_steps, current_lr)
                
                if batch_idx < gradient_accumulation_steps * 2:
                    log_memory(f"After optimizer.step() (batch {batch_idx})")
            
            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({
                "loss": loss.item() * gradient_accumulation_steps,
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint at end of each epoch
        save_checkpoint(model, config, optimizer, epoch, avg_loss, f"checkpoints/epoch_{epoch+1}.pt")
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, config, optimizer, epoch, avg_loss, "checkpoints/best_model.pt")
            print(f"  → New best model saved! (loss: {best_loss:.4f})")

    # 7. Save final model
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/reasoning_llm_sft.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_name': TOKENIZER_NAME  # Must match training tokenizer
    }, save_path)
    print(f"Final model saved to {save_path}")
    
    # Save training log
    logger.save()
    print(f"Training log saved to {logger.log_file}")

if __name__ == "__main__":
    train()
