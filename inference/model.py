"""Small Llama-style decoder used by the inference service."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.kernels.cuda_ops import fused_rmsnorm
from inference.kernels.triton_rope import apply_rope_triton


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_layers: int = 8
    num_heads: int = 16
    intermediate_size: int = 2816
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return fused_rmsnorm(x, self.weight, self.eps)


def rope_tables(cfg: ModelConfig, device="cuda"):
    head_dim = cfg.hidden_size // cfg.num_heads
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(cfg.max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = apply_rope_triton(q, cos[:s], sin[:s])
        k = apply_rope_triton(k, cos[:s], sin[:s])
        # Force the FlashAttention backend on NVIDIA GPUs.
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, -1))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        return self.lm_head(self.norm(x))


def load_model(cfg: ModelConfig = ModelConfig(), dtype=torch.float16) -> Decoder:
    torch.manual_seed(0)
    model = Decoder(cfg).to(device="cuda", dtype=dtype).eval()
    return model
