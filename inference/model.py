"""Small Llama-style decoder used by the inference service."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.device import resolve_device, resolve_dtype, torch_device
from inference.kernels.dispatch import KernelConfig, apply_rope, rmsnorm


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
    def __init__(self, dim: int, eps: float, impl: str = "reference"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.impl = impl

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps, self.impl)


def causal_attention_math(q, k, v):
    """Explicit causal attention: matmul -> mask -> softmax(fp32) -> matmul.

    Used on Neuron instead of a fused SDPA backend: every op has a static shape and a
    direct XLA lowering, so neuronx-cc sees (and fuses) the whole pattern.
    """
    s = q.shape[-2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    mask = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(scores.float().softmax(dim=-1).to(q.dtype), v)


def rope_tables(cfg: ModelConfig, device="cpu"):
    head_dim = cfg.hidden_size // cfg.num_heads
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(cfg.max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, kernels: KernelConfig = KernelConfig()):
        super().__init__()
        self.rope_impl = kernels.rope
        self.attn_impl = kernels.attention
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = apply_rope(q, cos[:s], sin[:s], self.rope_impl)
        k = apply_rope(k, cos[:s], sin[:s], self.rope_impl)
        if self.attn_impl == "flash":
            # Force the FlashAttention backend on NVIDIA GPUs.
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.attn_impl == "math":
            out = causal_attention_math(q, k, v)
        else:
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
    def __init__(self, cfg: ModelConfig, kernels: KernelConfig = KernelConfig()):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.eps, kernels.rmsnorm)
        self.attn = Attention(cfg, kernels)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.eps, kernels.rmsnorm)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig, kernels: KernelConfig = KernelConfig()):
        super().__init__()
        self.cfg = cfg
        self.kernels = kernels
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg, kernels) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.eps, kernels.rmsnorm)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids, last_pos=None):
        """Returns logits [b, s, vocab], or [b, vocab] at ``last_pos`` (shape [b]) if given.

        ``last_pos`` lets a right-padded, fixed-length (bucketed) input return only the
        next-token logits of the real last token; causal masking keeps padding from
        influencing earlier positions.
        """
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        if last_pos is not None:
            idx = last_pos.view(-1, 1, 1).expand(-1, 1, x.shape[-1])
            x = x.gather(1, idx).squeeze(1)
        return self.lm_head(self.norm(x))


def load_model(cfg: ModelConfig = ModelConfig(), dtype=None, device: str | None = None,
               kernels: KernelConfig | None = None) -> Decoder:
    """Build the decoder for ``device`` (cuda | neuron | cpu; default from INFERENCE_DEVICE).

    For ``neuron`` the eager model stays on the host; ``inference.neuron`` compiles it.
    """
    device = resolve_device(device)
    dtype = resolve_dtype(device, dtype)
    kernels = kernels or KernelConfig.for_device(device)
    torch.manual_seed(0)
    model = Decoder(cfg, kernels).to(device=torch_device(device), dtype=dtype).eval()
    return model
