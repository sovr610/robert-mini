import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from reasoning_llm import ReasoningLLM, ModelRegistry
from data_loader import create_main_dataset
import os
from tqdm import tqdm

def train():
    # 1. Configuration
    # Switched to mistral-7b which fits on a single A100 80GB
    model_name = "mistral-7b" 
    batch_size = 1 # Reduced to 1 to fit in memory
    gradient_accumulation_steps = 16 # Simulate batch size 16
    learning_rate = 3e-4
    epochs = 3 # Increased epochs for better training
    max_seq_len = 2048 # Increased context length for reasoning
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Tokenizer
    # We use the o200k tokenizer (GPT-4o) via the Xenova/gpt-4o repository
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4o")
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Dataset
    print("Preparing dataset...")
    # Load a small sample for demonstration. Set sample_size=None for full training.
    dataset = create_main_dataset(sample_size=500) 
    
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
    
    model = ReasoningLLM(config).to(device)
    
    # 5. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda') # Initialize Mixed Precision Scaler

    # 6. Training Loop
    print("Starting training...")
    model.train()
    
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        optimizer.zero_grad() # Initialize gradients
        
        for batch_idx, (input_ids, targets) in enumerate(progress_bar):
            input_ids, targets = input_ids.to(device), targets.to(device)
            
            # Mixed Precision Forward pass
            with autocast('cuda'):
                logits = model(input_ids)
                
                # Shift logits and targets for Causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                
                loss = criterion(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
                loss = loss / gradient_accumulation_steps # Normalize loss
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache() # Clear cache to prevent fragmentation
            
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
        'tokenizer_name': "gpt2"
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
