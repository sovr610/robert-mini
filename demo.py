import torch
from reasoning_llm import ReasoningLLM, ModelRegistry

def main():
    print("Reasoning LLM Demo")
    print("------------------")

    # 1. Select a model size
    model_name = "test-tiny"  # Using a small model for demonstration
    print(f"Loading configuration for: {model_name}")
    
    try:
        config = ModelRegistry.get_config(model_name)
    except ValueError as e:
        print(e)
        return

    print(f"Configuration: {config}")

    # 2. Instantiate the model
    print("\nInitializing model...")
    model = ReasoningLLM(config)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 3. Run a dummy forward pass
    print("\nRunning forward pass...")
    batch_size = 2
    seq_len = 10
    
    # Create dummy input tokens (integers)
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\nSuccess! The model is built and runnable.")

    print("\nTo use other sizes, change 'model_name' to one of:")
    print("- llama-2-7b")
    print("- llama-2-70b")
    print("- mistral-7b")

if __name__ == "__main__":
    main()
