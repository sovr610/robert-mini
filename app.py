import torch
import os
import gradio as gr
from transformers import AutoTokenizer
from reasoning_llm import ReasoningLLM
from reasoning_llm.config import ModelConfig
from inference import generate  # Use improved inference module

# PyTorch 2.6+ requires allowlisting custom classes for safe loading
torch.serialization.add_safe_globals([ModelConfig])

# 1. Load Model & Tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/reasoning_llm_sft.pt"

print(f"Loading model from {CHECKPOINT_PATH} on {DEVICE}...")

if not torch.cuda.is_available() and not os.path.exists(CHECKPOINT_PATH):
    print("Checkpoint not found. Please run train_sft.py first.")

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    config = checkpoint['config']
    tokenizer_name = checkpoint.get('tokenizer_name', 'Xenova/gpt-4o')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = ReasoningLLM(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you have run 'python train_sft.py' to generate the checkpoint.")
    exit(1)

# 2. Inference Function (using improved generate with top-p and repetition penalty)
def generate_response(user_input, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=user_input,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        device=DEVICE
    )
    return response

# 3. Gradio Interface
with gr.Blocks(title="Verse-Mini LLM Demo") as demo:
    gr.Markdown("# 🧠 Verse-Mini Reasoning LLM")
    gr.Markdown("Interact with your custom-trained Transformer model optimized for reasoning tasks.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Your Input", placeholder="Ask a question or request code...", lines=5)
            with gr.Accordion("Generation Settings", open=False):
                max_tokens = gr.Slider(minimum=10, maximum=512, value=150, step=10, label="Max New Tokens")
                temp = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-K")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-P (Nucleus Sampling)")
                rep_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.05, label="Repetition Penalty")
            
            submit_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Model Response", lines=10, interactive=False)
    
    submit_btn.click(
        fn=generate_response,
        inputs=[input_text, max_tokens, temp, top_k, top_p, rep_penalty],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
