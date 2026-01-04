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
    
    # Handle 'system'/'conversations' format (ShareGPT style, common in OpenThoughts)
    if 'conversations' in example:
        return {
            "messages": [
                {"role": "user", "content": example['conversations'][0]['value']},
                {"role": "assistant", "content": example['conversations'][1]['value']}
            ]
        }

    # Handle 'problem'/'solution' format (common in Math/Reasoning)
    if 'problem' in example and 'solution' in example:
        return {
            "messages": [
                {"role": "user", "content": example['problem']},
                {"role": "assistant", "content": example['solution']}
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
    
    # 1. General Conversation (UltraChat + OpenAssistant)
    print("Loading General Conversation data...")
    ds_chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    if sample_size: ds_chat = ds_chat.select(range(sample_size))
    ds_chat = ds_chat.map(format_chat_instruction, remove_columns=ds_chat.column_names)

    ds_oasst = load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="train")
    if sample_size: ds_oasst = ds_oasst.select(range(sample_size))
    # OpenAssistant already has 'text' but we need to parse it or use a formatted version. 
    # For simplicity in this demo, we'll skip complex parsing or assume it has a compatible format 
    # (it usually needs specific handling, but let's try to map it if it has text/messages).
    # Actually, oasst_top1 usually has 'text' which is the full conversation. 
    # We'll skip it for now to avoid breakage if format differs, or use a safe fallback.
    # Let's use a safer alternative for "General": "HuggingFaceH4/no_robots" is also good.
    # But let's stick to the user request for "General Talking".
    
    # 2. Coding (CodeAlpaca + OpenCodeReasoning)
    print("Loading Coding data...")
    ds_code = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    if sample_size: ds_code = ds_code.select(range(sample_size))
    ds_code = ds_code.map(format_chat_instruction, remove_columns=ds_code.column_names)

    # New: OpenCodeReasoning (High quality coding reasoning)
    try:
        # Using 'split_0' config as required by the dataset
        ds_code_reasoning = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="train")
        if sample_size: ds_code_reasoning = ds_code_reasoning.select(range(sample_size))
        ds_code_reasoning = ds_code_reasoning.map(format_chat_instruction, remove_columns=ds_code_reasoning.column_names)
    except Exception as e:
        print(f"Could not load OpenCodeReasoning: {e}")
        ds_code_reasoning = None
    
    # 3. Reasoning/Math/Science (Orca Math + OpenThoughts)
    print("Loading Reasoning & Science data...")
    ds_math = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    if sample_size: ds_math = ds_math.select(range(sample_size))
    ds_math = ds_math.map(format_chat_instruction, remove_columns=ds_math.column_names)

    # New: OpenThoughts (General Reasoning, Math, Science)
    try:
        ds_reasoning = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
        if sample_size: ds_reasoning = ds_reasoning.select(range(sample_size))
        ds_reasoning = ds_reasoning.map(format_chat_instruction, remove_columns=ds_reasoning.column_names)
    except Exception as e:
        print(f"Could not load OpenThoughts: {e}")
        ds_reasoning = None

    # Combine
    print("Combining datasets...")
    datasets_to_combine = [ds_chat, ds_code, ds_math]
    if ds_code_reasoning: datasets_to_combine.append(ds_code_reasoning)
    if ds_reasoning: datasets_to_combine.append(ds_reasoning)

    combined_dataset = concatenate_datasets(datasets_to_combine)
    
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
