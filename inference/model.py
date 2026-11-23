"""Small Llama-style decoder used by the inference service.

Hardware-specific kernels are selected through ``KernelConfig`` and imported lazily, so the
model can be built on hosts without CUDA or Triton (CPU, AWS Neuron).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.backends import KernelConfig, get_backend
from inference.kernels.reference import apply_rope_reference, rmsnorm_reference


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
        if impl == "cuda":
            from inference.kernels.cuda_ops import fused_rmsnorm
            self._fn = fused_rmsnorm
        elif impl == "reference":
            self._fn = rmsnorm_reference
        else:
            raise ValueError(f"unknown rmsnorm impl {impl!r}")

    def forward(self, x):
        return self._fn(x, self.weight, self.eps)


def causal_attention_math(q, k, v):
    """Causal attention from matmul/softmax only; numerically equivalent to SDPA(is_causal=True)."""
    s = q.shape[-2]
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    mask = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


def rope_tables(cfg: ModelConfig, device="cuda"):
    head_dim = cfg.hidden_size // cfg.num_heads
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(cfg.max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, kernels: KernelConfig = KernelConfig()):
        super().__init__()
        if kernels.rope == "triton":
            from inference.kernels.triton_rope import apply_rope_triton
            self.rope = apply_rope_triton
        elif kernels.rope == "reference":
            self.rope = apply_rope_reference
        else:
            raise ValueError(f"unknown rope impl {kernels.rope!r}")
        if kernels.attention not in ("flash", "sdpa", "math"):
            raise ValueError(f"unknown attention impl {kernels.attention!r}")
        self.attention_impl = kernels.attention
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self.rope(q, cos[:s], sin[:s])
        k = self.rope(k, cos[:s], sin[:s])
        if self.attention_impl == "flash":
            # Force the FlashAttention backend on NVIDIA GPUs.
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.attention_impl == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = causal_attention_math(q.contiguous(), k.contiguous(), v.contiguous())
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
        cos, sin = rope_tables(cfg, device="cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids, last_pos=None):
        """Returns logits ``[b, s, vocab]``, or ``[b, vocab]`` at ``last_pos`` (``[b]`` long tensor).

        ``last_pos`` lets fixed-length (right-padded) inputs return next-token logits without a
        data-dependent slice, keeping shapes static for ahead-of-time compilers such as Neuron.
        """
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        if last_pos is not None:
            idx = last_pos.view(-1, 1, 1).expand(-1, 1, x.shape[-1])
            x = torch.gather(x, 1, idx).squeeze(1)
        return self.lm_head(self.norm(x))


def load_model(cfg: ModelConfig = ModelConfig(), dtype=None, backend=None) -> Decoder:
    """Build the decoder with deterministic weights on ``backend`` (see ``inference.backends``).

    Weights are initialised on CPU from a fixed seed, so every backend gets identical parameters.
    """
    backend = get_backend(backend)
    torch.manual_seed(0)
    model = Decoder(cfg, backend.kernels)
    return model.to(device=backend.device, dtype=dtype or backend.default_dtype).eval()
