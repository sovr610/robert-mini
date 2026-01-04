# Verse Mini - Copilot Instructions

This file provides custom instructions for GitHub Copilot to understand the specific context, architecture, and coding standards of the **Verse Mini** project.

## Project Overview
**Verse Mini** is a custom Large Language Model (LLM) project built from scratch using PyTorch. The goal is to train a reasoning-focused model (7B+ parameters) on a single GPU environment, utilizing high-quality datasets like OpenThoughts, OpenCodeReasoning, and UltraChat.

## Tech Stack
- **Language**: Python 3.10+
- **Framework**: PyTorch (latest stable)
- **Libraries**: Hugging Face Transformers, Datasets, TRL, Gradio
- **Hardware Context**: Single NVIDIA A100 (80GB) - *Memory efficiency is paramount.*

## Coding Guidelines

### 1. Model Architecture (`reasoning_llm/`)
- **Core Class**: `ReasoningLLM` in `reasoning_llm/model.py`.
- **Key Features**:
    - Rotary Positional Embeddings (RoPE)
    - SwiGLU Activations
    - Grouped Query Attention (GQA)
    - RMSNorm
- **Memory Optimization**:
    - Ensure `gradient_checkpointing` is implemented using `torch.utils.checkpoint.checkpoint` for transformer layers.
    - Avoid unnecessary tensor duplication in the `forward` pass.

### 2. Training Pipeline (`train_sft.py`)
- **Mixed Precision**: Use `torch.amp` (Automatic Mixed Precision) for all training loops. *Do not use the deprecated `torch.cuda.amp`.*
- **Batching Strategy**:
    - Due to memory constraints, prefer `batch_size=1`.
    - Use `gradient_accumulation_steps` (e.g., 16 or 32) to simulate larger effective batch sizes.
- **Tokenizer**:
    - **MUST** use `Xenova/gpt-4o` (o200k_base).
    - Do not default to `gpt2` or `llama` tokenizers unless explicitly requested for debugging.

### 3. Data Pipeline (`data_loader.py`)
- **Dataset Sources**: Hugging Face Hub.
- **Formatting**: All datasets must be tokenized and formatted into `input_ids` and `labels`.
- **Handling Large Datasets**: Use streaming or iterable datasets where possible if disk/RAM is limited, though current implementation loads to memory.
- **Specific Configs**:
    - `nvidia/OpenCodeReasoning` requires `split="train"` and config `split_0`.

## Preferred Patterns

### Error Handling
- When encountering `CUDA out of memory`:
    1. Suggest reducing `max_seq_len` (default is often 2048).
    2. Suggest increasing `gradient_accumulation_steps`.
    3. Verify `torch.cuda.empty_cache()` usage.

### Code Style
- **Type Hinting**: Use Python type hints (`List`, `Optional`, `Tensor`) for function signatures.
- **Documentation**: Add docstrings to new model layers or complex data processing functions.
- **Imports**: Keep imports organized (Standard Lib -> Third Party -> Local).

## Key Files Map
- `train_sft.py`: Main entry point for Supervised Fine-Tuning.
- `reasoning_llm/config.py`: Hyperparameter configurations (`mistral-7b`, `llama-2-70b`, etc.).
- `reasoning_llm/model.py`: The neural network definition.
- `data_loader.py`: Dataset aggregation and preprocessing.
