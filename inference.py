import torch
from typing import Optional, List
from transformers import AutoTokenizer
from reasoning_llm import ReasoningLLM
from reasoning_llm.config import ModelConfig

# Configuration
CHECKPOINT_PATH = "checkpoints/reasoning_llm_sft.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PyTorch 2.6+ requires allowlisting custom classes for safe loading
torch.serialization.add_safe_globals([ModelConfig])

def load_model(checkpoint_path: str = CHECKPOINT_PATH):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint['config']
    tokenizer_name = checkpoint.get('tokenizer_name', 'Xenova/gpt-4o')
    
    model = ReasoningLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer, config

def generate(
    model: ReasoningLLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    stop_tokens: Optional[List[str]] = None,
    device: str = None
) -> str:
    """
    Generate a response from the model with improved sampling.
    
    Args:
        model: The trained ReasoningLLM model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k filtering (0 to disable)
        top_p: Nucleus sampling threshold (1.0 to disable)
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        stop_tokens: List of strings to stop generation at
        device: Device to run on (defaults to global DEVICE)
    
    Returns:
        Generated response text
    """
    # Use global DEVICE if not specified
    _device = device if device is not None else DEVICE
    
    # Format as chat
    formatted_prompt = f"user: {prompt}\nassistant:"
    
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(_device)
    prompt_length = input_ids.shape[1]
    
    generated_ids = input_ids.clone()
    
    # Track generated tokens for repetition penalty
    generated_token_set = set()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context if too long
            context = generated_ids[:, -model.config.max_seq_len:] if generated_ids.size(1) > model.config.max_seq_len else generated_ids
            
            # Get logits for the last token
            logits = model(context)
            next_token_logits = logits[:, -1, :].clone()
            
            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated_token_set) > 0:
                for token_id in generated_token_set:
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty
            
            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_values = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0]
                indices_to_remove = next_token_logits < top_k_values[..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Track for repetition penalty
            generated_token_set.add(next_token.item())
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Check for stop tokens
            if stop_tokens:
                current_text = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
                if any(stop in current_text for stop in stop_tokens):
                    break
    
    # Decode only the generated part
    response = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
    
    # Clean up stop tokens from response
    if stop_tokens:
        for stop in stop_tokens:
            if stop in response:
                response = response.split(stop)[0]
    
    return response.strip()

def main():
    model, tokenizer, config = load_model()
    
    print("\n" + "="*50)
    print("     VERSE MINI - Reasoning LLM Inference")
    print("="*50)
    print(f"Model: {config.n_layers} layers, {config.dim} dim")
    print(f"Device: {DEVICE}")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        response = generate(
            model, 
            tokenizer, 
            user_input,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            stop_tokens=["\nuser:", "\n\n\n"]
        )
        print(f"Model: {response}\n")

if __name__ == "__main__":
    main()