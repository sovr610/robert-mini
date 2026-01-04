import torch
from transformers import AutoTokenizer
from reasoning_llm import ReasoningLLM
from reasoning_llm.config import ModelConfig

# Configuration
CHECKPOINT_PATH = "checkpoints/reasoning_llm_sft.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PyTorch 2.6+ requires allowlisting custom classes for safe loading
torch.serialization.add_safe_globals([ModelConfig])

def load_model():
    """Load the trained model and tokenizer."""
    print(f"Loading model from {CHECKPOINT_PATH}...")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    config = checkpoint['config']
    
    model = ReasoningLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Load the o200k tokenizer (GPT-4o)
    tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-4o")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer, config

def generate(
    model: ReasoningLLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50
) -> str:
    """Generate a response from the model."""
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for the last token
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode and return
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response

def main():
    model, tokenizer, config = load_model()
    
    print("\n--- Verse Mini Inference ---")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        response = generate(model, tokenizer, user_input)
        print(f"Model: {response}\n")

if __name__ == "__main__":
    main()