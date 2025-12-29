from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    hidden_dim: int
    vocab_size: int
    max_seq_len: int
    norm_eps: float = 1e-6

class ModelRegistry:
    @staticmethod
    def get_config(model_name: str) -> ModelConfig:
        configs = {
            "llama-2-7b": ModelConfig(
                dim=4096,
                n_layers=32,
                n_heads=32,
                n_kv_heads=32,
                hidden_dim=11008,
                vocab_size=32000,
                max_seq_len=4096
            ),
            "llama-2-70b": ModelConfig(
                dim=8192,
                n_layers=80,
                n_heads=64,
                n_kv_heads=8,
                hidden_dim=28672,
                vocab_size=32000,
                max_seq_len=4096
            ),
            "mistral-7b": ModelConfig(
                dim=4096,
                n_layers=32,
                n_heads=32,
                n_kv_heads=8,
                hidden_dim=14336,
                vocab_size=32000,
                max_seq_len=32768
            ),
            "test-tiny": ModelConfig(
                dim=256,
                n_layers=4,
                n_heads=8,
                n_kv_heads=4,
                hidden_dim=1024,
                vocab_size=1000,
                max_seq_len=512
            )
        }
        
        if model_name not in configs:
            raise ValueError(f"Model {model_name} not found. Available models: {list(configs.keys())}")
            
        return configs[model_name]
