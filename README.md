# Verse Mini

Verse Mini is a custom Large Language Model (LLM) project designed for reasoning and coding tasks. It features a modern Transformer architecture built from scratch in PyTorch, including Rotary Positional Embeddings (RoPE), Grouped Query Attention (GQA), and SwiGLU activations.

## Features

*   **Custom Architecture**: A decoder-only Transformer (`ReasoningLLM`) implementing state-of-the-art techniques found in Llama 2 and Mistral.
*   **Training Pipelines**:
    *   **SFT (Supervised Fine-Tuning)**: `train_sft.py` for training on instruction datasets.
    *   **GRPO (Reinforcement Learning)**: `train.py` for optimizing reasoning capabilities (experimental).
*   **Data Pipeline**: `data_loader.py` aggregates high-quality datasets for chat, coding, and math/reasoning.
*   **Inference**:
    *   `demo.py`: A simple CLI demo to verify model architecture.
    *   `app.py`: A Gradio web interface for interacting with the trained model.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install `torch` with CUDA support separately depending on your hardware.*

2.  **Prepare Data**:
    The `data_loader.py` script automatically downloads and formats datasets from Hugging Face.
    *   General Chat: `HuggingFaceH4/ultrachat_200k`, `OpenAssistant/oasst_top1_2023-08-25`
    *   Coding: `sahil2801/CodeAlpaca-20k`, `nvidia/OpenCodeReasoning`
    *   Reasoning: `microsoft/orca-math-word-problems-200k`, `open-thoughts/OpenThoughts-114k`

## Usage

### 1. Training (SFT)
To train the model using Supervised Fine-Tuning:

```bash
python train_sft.py
```

*   **Configuration**: You can modify `train_sft.py` to change the model size (e.g., `mistral-7b`, `test-tiny`), batch size, and learning rate.
*   **Output**: Checkpoints are saved to the `checkpoints/` directory.

### 2. Inference Demo
To run a quick test of the model architecture (random weights):

```bash
python demo.py
```

### 3. Web Interface
To chat with your trained model:

```bash
python app.py
```
This launches a Gradio interface in your browser.

## Model Configuration

Model architectures are defined in `reasoning_llm/config.py`. Available configurations include:
*   `test-tiny`: A small model for debugging (CPU friendly).
*   `llama-2-7b`: Standard 7B parameter configuration.
*   `mistral-7b`: 7B configuration with larger context window.
*   `llama-2-70b`: Large 70B parameter configuration.

## License

[MIT](LICENSE)
