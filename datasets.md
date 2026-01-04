# Recommended Datasets for Reasoning & Coding LLM

Based on the latest research (2024-2025), here is a curated list of datasets to combine for training a production-grade reasoning and coding model.

## 1. General Instruction & Conversation (The "Chat" Ability)
*   **Dataset**: `HuggingFaceH4/ultrachat_200k`
    *   **Description**: A heavily filtered version of UltraChat, providing high-quality multi-turn conversations.
    *   **Use Case**: General conversation, question answering, and instruction following.
*   **Dataset**: `OpenAssistant/oasst_top1_2023-08-25`
    *   **Description**: High-quality, human-annotated assistant-style conversations.
    *   **Use Case**: Human-like interaction and helpfulness.

## 2. Coding & Technical Reasoning (The "Code" Ability)
*   **Dataset**: `nvidia/OpenCodeReasoning` (New 2025)
    *   **Description**: High-quality coding reasoning dataset.
    *   **Use Case**: Complex code logic and problem solving.
*   **Dataset**: `nampdn-ai/tiny-codes`
    *   **Description**: A collection of ~1.6M code snippets and programming problems.
    *   **Use Case**: Syntax learning, code generation, and logic.
*   **Dataset**: `bigcode/the-stack-smol`
    *   **Description**: A smaller, curated subset of "The Stack" (permissive license code).
    *   **Use Case**: Exposure to diverse programming languages.
*   **Dataset**: `sahil2801/CodeAlpaca-20k`
    *   **Description**: Instruction-following dataset specifically for code tasks.
    *   **Use Case**: Answering "How do I write a function to..." questions.

## 3. Mathematical, Scientific & Logical Reasoning (The "Reasoning" Ability)
*   **Dataset**: `open-thoughts/OpenThoughts-114k` (New 2025)
    *   **Description**: State-of-the-art reasoning dataset covering math, science, and code.
    *   **Use Case**: Deep reasoning and Chain-of-Thought (CoT).
*   **Dataset**: `microsoft/orca-math-word-problems-200k`
    *   **Description**: High-quality math word problems with step-by-step solutions.
    *   **Use Case**: Teaching the model to think step-by-step (Chain of Thought).
*   **Dataset**: `TIGER-Lab/MathInstruct`
    *   **Description**: A compilation of various math instruction datasets.

## 4. Modern "All-in-One" Mixtures (Recommended Starting Point)
If you want a pre-mixed high-quality dataset:
*   **Dataset**: `mlabonne/FineTome-100k`
    *   **Description**: A highly curated mix of Arcee's Tome, ShareGPT, and other top-tier datasets. Known for training high-performance 7B models.
*   **Dataset**: `nvidia/OpenMathInstruct-1`
    *   **Description**: Excellent for math/reasoning heavy models.

## Strategy for "One Main Dataset"
To create your "Main Dataset", you should:
1.  **Load** a portion of each dataset above.
2.  **Format** them into a standard structure (e.g., `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`).
3.  **Interleave** them so the model sees a mix of code, chat, and math in every batch.

See `data_loader.py` for a script to do this. It is currently set to download a small sample (1000 examples) for testing. To create the full dataset, set `sample_size=None` in the `create_main_dataset` function.
