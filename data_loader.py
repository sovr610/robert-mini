import torch
from datasets import load_dataset, concatenate_datasets, Dataset
import random

def format_chat_instruction(example):
    """
    Standardizes different dataset formats into a common chat format:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    # Handle 'ultrachat' format (list of messages)
    if 'messages' in example:
        return example
    
    # Handle 'instruction'/'response' format (e.g., CodeAlpaca)
    if 'instruction' in example and 'output' in example:
        input_text = example['instruction']
        if 'input' in example and example['input']:
            input_text += "\nInput: " + example['input']
            
        return {
            "messages": [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": example['output']}
            ]
        }
    
    # Handle 'question'/'answer' format (e.g., Orca Math)
    if 'question' in example and 'answer' in example:
        return {
            "messages": [
                {"role": "user", "content": example['question']},
                {"role": "assistant", "content": example['answer']}
            ]
        }

    return None

def create_main_dataset(sample_size=1000):
    """
    Loads and combines subsets of key datasets to create a 'Main Dataset'.
    Args:
        sample_size: Number of examples to take from each source (for demo purposes).
                     Set to None to use full datasets (Warning: Huge).
    """
    print("Loading datasets...")
    
    # 1. General Conversation (UltraChat)
    print("Loading General Conversation data...")
    ds_chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    if sample_size: ds_chat = ds_chat.select(range(sample_size))
    ds_chat = ds_chat.map(format_chat_instruction, remove_columns=ds_chat.column_names)
    
    # 2. Coding (CodeAlpaca)
    print("Loading Coding data...")
    ds_code = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    if sample_size: ds_code = ds_code.select(range(sample_size))
    ds_code = ds_code.map(format_chat_instruction, remove_columns=ds_code.column_names)
    
    # 3. Reasoning/Math (Orca Math)
    print("Loading Reasoning data...")
    ds_math = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    if sample_size: ds_math = ds_math.select(range(sample_size))
    ds_math = ds_math.map(format_chat_instruction, remove_columns=ds_math.column_names)

    # Combine
    print("Combining datasets...")
    combined_dataset = concatenate_datasets([ds_chat, ds_code, ds_math])
    
    # Shuffle
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    print(f"Created Main Dataset with {len(combined_dataset)} examples.")
    return combined_dataset

if __name__ == "__main__":
    # Requires: pip install datasets
    try:
        dataset = create_main_dataset(sample_size=100) # Small sample for testing
        
        print("\nSample Example:")
        print(dataset[0]['messages'])
        
        # Save to disk
        # dataset.save_to_disk("main_dataset")
        print("\nDataset ready for training!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have the 'datasets' library installed: pip install datasets")
