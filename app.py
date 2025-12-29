import torch
import gradio as gr
from transformers import AutoTokenizer
from reasoning_llm import ReasoningLLM, ModelConfig

# 1. Load Model & Tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/reasoning_llm_sft.pt"

print(f"Loading model from {CHECKPOINT_PATH} on {DEVICE}...")

if not torch.cuda.is_available() and not os.path.exists(CHECKPOINT_PATH):
    print("Checkpoint not found. Please run train_sft.py first.")
    # Create a dummy model for interface testing if file doesn't exist
    # In a real scenario, we'd exit or download a model
    pass

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    config = checkpoint['config']
    tokenizer_name = checkpoint.get('tokenizer_name', 'gpt2')
    
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

# 2. Inference Function
def generate_response(user_input, max_new_tokens, temperature, top_k):
    # Format input (simple chat format)
    prompt = f"user: {user_input}\nassistant: "
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # Use the generate method we added to the model
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=int(max_new_tokens), 
            temperature=float(temperature), 
            top_k=int(top_k)
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    # (Simple splitting, might need robustness for production)
    try:
        response = generated_text.split("assistant: ")[-1]
    except:
        response = generated_text
        
    return response

# 3. Gradio Interface
with gr.Blocks(title="Reasoning LLM Demo") as demo:
    gr.Markdown("# Reasoning LLM Interface")
    gr.Markdown("Interact with your custom-trained Transformer model.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Your Input", placeholder="Ask a question or request code...", lines=5)
            with gr.Accordion("Advanced Settings", open=False):
                max_tokens = gr.Slider(minimum=10, maximum=512, value=100, step=10, label="Max New Tokens")
                temp = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
                top_k = gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top-K")
            
            submit_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Model Response", lines=10, interactive=False)
    
    submit_btn.click(
        fn=generate_response,
        inputs=[input_text, max_tokens, temp, top_k],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
