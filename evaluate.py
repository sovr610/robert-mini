"""
Verse-Mini Model Evaluation Script
===================================
Evaluates the trained model on various metrics and test prompts.
Generates a report of model quality and saves results to JSON.
"""

import torch
import json
import os
from datetime import datetime
from transformers import AutoTokenizer
from reasoning_llm import ReasoningLLM
from reasoning_llm.config import ModelConfig
from inference import generate

# PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals([ModelConfig])

# Test prompts for different capabilities
TEST_PROMPTS = {
    "general_knowledge": [
        "What is the capital of France?",
        "Explain photosynthesis in simple terms.",
        "Who wrote Romeo and Juliet?",
    ],
    "reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
    ],
    "coding": [
        "Write a Python function to check if a number is prime.",
        "Explain what a binary search algorithm does.",
        "What is the difference between a list and a tuple in Python?",
    ],
    "math": [
        "What is 15% of 240?",
        "Solve for x: 2x + 5 = 17",
        "Calculate the area of a circle with radius 7.",
    ],
    "conversation": [
        "Hello! How are you today?",
        "Can you help me with a problem?",
        "Thank you for your help!",
    ]
}


def load_model(checkpoint_path: str, device: str):
    """Load the trained model and tokenizer."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    tokenizer_name = checkpoint.get('tokenizer_name', 'Xenova/gpt-4o')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = ReasoningLLM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {config.n_layers} layers, {config.dim} dim")
    return model, tokenizer, config


def evaluate_response_quality(response: str) -> dict:
    """Evaluate basic quality metrics of a response."""
    metrics = {
        "length": len(response),
        "word_count": len(response.split()),
        "has_content": len(response.strip()) > 10,
        "is_repetitive": False,
        "coherence_score": 0.0
    }
    
    # Check for repetition
    words = response.lower().split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        metrics["is_repetitive"] = unique_ratio < 0.3
        metrics["coherence_score"] = unique_ratio
    
    return metrics


def run_evaluation(
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> dict:
    """Run full evaluation across all test categories."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty
        },
        "categories": {},
        "summary": {}
    }
    
    total_prompts = 0
    total_quality = 0
    repetitive_count = 0
    
    for category, prompts in TEST_PROMPTS.items():
        print(f"\n{'='*50}")
        print(f"Testing: {category.upper()}")
        print('='*50)
        
        category_results = []
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            
            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                device=device
            )
            
            print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
            
            quality = evaluate_response_quality(response)
            
            category_results.append({
                "prompt": prompt,
                "response": response,
                "quality": quality
            })
            
            total_prompts += 1
            total_quality += quality["coherence_score"]
            if quality["is_repetitive"]:
                repetitive_count += 1
        
        results["categories"][category] = category_results
    
    # Calculate summary statistics
    results["summary"] = {
        "total_prompts": total_prompts,
        "avg_coherence_score": total_quality / total_prompts if total_prompts > 0 else 0,
        "repetitive_responses": repetitive_count,
        "repetition_rate": repetitive_count / total_prompts if total_prompts > 0 else 0
    }
    
    return results


def print_summary(results: dict):
    """Print a formatted summary of evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    summary = results["summary"]
    print(f"Total Prompts Tested: {summary['total_prompts']}")
    print(f"Average Coherence Score: {summary['avg_coherence_score']:.2%}")
    print(f"Repetitive Responses: {summary['repetitive_responses']} ({summary['repetition_rate']:.1%})")
    
    print("\nPer-Category Breakdown:")
    for category, items in results["categories"].items():
        avg_coherence = sum(i["quality"]["coherence_score"] for i in items) / len(items)
        repetitive = sum(1 for i in items if i["quality"]["is_repetitive"])
        print(f"  {category}: coherence={avg_coherence:.2%}, repetitive={repetitive}/{len(items)}")


def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "checkpoints/reasoning_llm_sft.pt"
    OUTPUT_PATH = "evaluation_results.json"
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        # Try best model
        CHECKPOINT_PATH = "checkpoints/best_model.pt"
        if not os.path.exists(CHECKPOINT_PATH):
            print("No checkpoint found. Please train the model first with 'python train_sft.py'")
            return
    
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    
    # Load model
    model, tokenizer, config = load_model(CHECKPOINT_PATH, DEVICE)
    
    # Run evaluation
    results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    # Add model info to results
    results["model_info"] = {
        "dim": config.dim,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len
    }
    
    # Print summary
    print_summary(results)
    
    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
