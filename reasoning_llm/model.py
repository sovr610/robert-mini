import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

class RMSNorm(nn.Module):
    """Root Mean Square Normalization - faster than LayerNorm, used in LLaMA/Mistral."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

class SwiGLU(nn.Module):
    """SwiGLU: Swish-Gated Linear Unit. Formula: SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)
    Note: Uses 3 matrices, so hidden_dim is scaled to 2/3 of standard FFN to maintain params.
    """
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256  # Round for efficiency
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class RotaryPositionalEmbeddings(nn.Module):
    """RoPE encodes positions through rotation, enabling extrapolation beyond training length.
    Formula: RoPE(x, pos) = x * cos(pos*θ) + rotate_half(x) * sin(pos*θ)
    """
    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 32768):
        super().__init__()
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', positions, theta)
        
        self.register_buffer('cos_cached', freqs.cos()[None, :, None, :])
        self.register_buffer('sin_cached', freqs.sin()[None, :, None, :])

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        # Slice cached cos/sin to the current sequence length
        # Note: This assumes seq_len <= max_seq_len. 
        # In production, you might need to dynamically recompute if seq_len > max_seq_len
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        
        # Repeat cos/sin to match q/k last dimension
        # cos is [1, seq_len, 1, dim/2] -> [1, seq_len, 1, dim]
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        
        def rotate(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
            return rotated
        
        # Apply RoPE
        # q, k shape: [bs, seq_len, n_heads, head_dim]
        # cos, sin shape: [1, seq_len, 1, head_dim/2] -> broadcast to match
        
        # We need to ensure dimensions match for broadcasting
        # The provided snippet had cos/sin as [1, seq_len, 1, dim/2] (from freqs)
        # But q/k are [bs, seq_len, n_heads, head_dim]
        # The rotation happens on the head_dim.
        
        q_rot = q * cos + rotate(q) * sin
        k_rot = k * cos + rotate(k) * sin
        return q_rot, k_rot

class GroupedQueryAttention(nn.Module):
    """GQA reduces KV-cache memory by sharing key-value heads across query groups.
    Used in LLaMA 2 70B (8 KV heads, 64 query heads) and Mistral 7B (8 KV heads, 32 query heads).
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, rope_module: RotaryPositionalEmbeddings):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # How many query heads share each KV head
        self.head_dim = d_model // n_heads
        self.rope = rope_module
        
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        
        q = self.wq(x).view(bs, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        q, k = self.rope(q, k, seq_len)

        # Expand KV heads to match query heads
        # k: [bs, seq_len, n_kv_heads, head_dim] -> [bs, seq_len, n_heads, head_dim]
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Use Flash Attention when available (PyTorch 2.0+)
        # Note: mask handling might need adjustment depending on F.scaled_dot_product_attention implementation details
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True if mask is None else False)
        
        return self.wo(out.transpose(1, 2).contiguous().view(bs, seq_len, -1))

class TransformerBlock(nn.Module):
    """Modern transformer block: Pre-Norm + GQA + RoPE + SwiGLU."""
    def __init__(self, config, rope_module):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(config.dim, config.n_heads, config.n_kv_heads, rope_module)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = SwiGLU(config.dim, config.hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), mask)
        return h + self.feed_forward(self.ffn_norm(h))

class ReasoningLLM(nn.Module):
    """Complete decoder-only transformer for reasoning tasks."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        # Initialize RoPE once and share across layers
        self.rope = RotaryPositionalEmbeddings(config.dim // config.n_heads, max_seq_len=config.max_seq_len)
        
        self.layers = nn.ModuleList([TransformerBlock(config, self.rope) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight  # Weight tying

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(tokens)
        
        # Create causal mask if needed, though F.scaled_dot_product_attention handles is_causal=True
        # If we pass is_causal=True to the attention layer, we don't strictly need to pass a mask here
        # unless we have padding tokens to mask out.
        # For simplicity, we'll let the attention layer handle causality.
        
        for layer in self.layers:
            if self.training:
                # Use gradient checkpointing to save memory
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            
        return self.output(self.norm(x))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        idx: [batch_size, seq_len] indices
        """
        for _ in range(max_new_tokens):
            # If the sequence context is too long, crop it
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # Forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
