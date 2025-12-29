import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from reasoning_llm import ReasoningLLM, ModelRegistry
from data_loader import create_main_dataset
import os
from tqdm import tqdm

def train():
    # 1. Configuration
    # We'll use a small model for the demo, but you can switch to "llama-2-7b" etc.
    model_name = "test-tiny" 
    batch_size = 4 # Small batch size for CPU/Demo
    learning_rate = 3e-4
    epochs = 1
    max_seq_len = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Tokenizer
    # We use GPT-2 tokenizer as a standard choice. 
    # In production, you might use LlamaTokenizer (requires sentencepiece)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
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

    # 6. Training Loop
    print("Starting training...")
    model.train()
    
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        for batch_idx, (input_ids, targets) in enumerate(progress_bar):
            input_ids, targets = input_ids.to(device), targets.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Shift logits and targets for Causal LM loss
            # Logits: [B, T, V] -> [B, T-1, V]
            # Targets: [B, T] -> [B, T-1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            # Flatten
            loss = criterion(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

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
