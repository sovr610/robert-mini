import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from reasoning_llm import ReasoningLLM, ModelRegistry
from data_loader import create_main_dataset
import os
from tqdm import tqdm
import gc

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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = model.tok_embeddings.weight.numel()
    output_params = embedding_params  # Weight tied
    core_params = total - embedding_params
    
    print(f"[PARAMS] Total: {total:,} ({total * 2 / 1024**3:.2f} GB in fp16)")
    print(f"         Embedding/Output: {embedding_params:,} ({embedding_params * 2 / 1024**3:.2f} GB)")
    print(f"         Core Transformer: {core_params:,} ({core_params * 2 / 1024**3:.2f} GB)")
    print(f"         AdamW would need: {total * 2 * 4 / 1024**3:.2f} GB additional for states")

def train():
    # 1. Configuration
    # Using verse-3b which fits on a single GPU with standard optimizer
    model_name = "verse-3b" 
    batch_size = 1 # Reduced to 1 to fit in memory
    gradient_accumulation_steps = 16 # Simulate batch size 16
    learning_rate = 3e-4
    epochs = 3 # Increased epochs for better training
    max_seq_len = 1024 # Reduced from 2048 to prevent OOM with large vocab
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 4. Model
    print(f"Initializing model: {model_name}")
    config = ModelRegistry.get_config(model_name)
    
    # Update config vocab size to match tokenizer
    config.vocab_size = len(tokenizer)
    config.max_seq_len = max_seq_len
    
    print(f"[CONFIG] dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}")
    print(f"[CONFIG] vocab_size={config.vocab_size}, max_seq_len={config.max_seq_len}")
    
    log_memory("Before model creation")
    model = ReasoningLLM(config).to(device)
    log_memory("After model.to(device)")
    count_parameters(model)
    
    # 5. Optimizer - Use SGD as fallback (no per-param state buffers)
    import torch.optim as optim
    if USE_8BIT_OPTIM:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        print("[OPTIM] Using 8-bit AdamW optimizer (memory efficient)")
    else:
        # SGD uses no additional memory for optimizer states
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        print("[OPTIM] Using SGD optimizer (minimal memory footprint)")
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda') # Initialize Mixed Precision Scaler
    
    log_memory("After optimizer creation")

    # 6. Training Loop
    print("Starting training...")
    model.train()
    
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
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # set_to_none=True saves memory
                torch.cuda.empty_cache() # Clear cache to prevent fragmentation
                
                if batch_idx < gradient_accumulation_steps * 2:
                    log_memory(f"After optimizer.step() (batch {batch_idx})")
            
            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")

    # 7. Save
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/reasoning_llm_sft.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_name': "Xenova/gpt-4o"  # Must match training tokenizer
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
