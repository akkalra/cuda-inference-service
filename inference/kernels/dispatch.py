"""Pick a kernel implementation per backend.

| op      | cuda              | neuron                      | cpu       |
|---------|-------------------|-----------------------------|-----------|
| rmsnorm | CUDA cpp_extension| NKI kernel                  | reference |
| rope    | Triton            | reference (native XLA ops)  | reference |

RoPE on Neuron defaults to the native-op path: it is purely elementwise, so neuronx-cc fuses
it with the surrounding QKV reshape; an NKI custom call is a fusion barrier that forces q/k
through HBM. The NKI RoPE kernel is kept, simulator-validated, and selectable with
``INFERENCE_ROPE_IMPL=nki`` so the two can be compared with a profiler on real hardware.

Overrides: ``INFERENCE_RMSNORM_IMPL`` in {reference, cuda, nki}, ``INFERENCE_ROPE_IMPL`` in
{reference, triton, nki}, ``INFERENCE_ATTENTION_IMPL`` in {flash, sdpa, math}.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from inference.kernels.reference import apply_rope_reference, rmsnorm_reference

RMSNORM_IMPLS = ("reference", "cuda", "nki")
ROPE_IMPLS = ("reference", "triton", "nki")
ATTENTION_IMPLS = ("flash", "sdpa", "math")

_DEFAULTS = {
    "cuda": ("cuda", "triton", "flash"),
    "neuron": ("nki", "reference", "math"),
    "cpu": ("reference", "reference", "sdpa"),
}


@dataclass(frozen=True)
class KernelConfig:
    rmsnorm: str = "reference"
    rope: str = "reference"
    attention: str = "sdpa"

    def __post_init__(self):
        for value, allowed, name in ((self.rmsnorm, RMSNORM_IMPLS, "rmsnorm"),
                                     (self.rope, ROPE_IMPLS, "rope"),
                                     (self.attention, ATTENTION_IMPLS, "attention")):
            if value not in allowed:
                raise ValueError(f"unsupported {name} impl {value!r}; expected one of {allowed}")

    @classmethod
    def for_device(cls, device: str) -> "KernelConfig":
        rms, rope, attn = _DEFAULTS[device]
        return cls(rmsnorm=os.environ.get("INFERENCE_RMSNORM_IMPL", rms),
                   rope=os.environ.get("INFERENCE_ROPE_IMPL", rope),
                   attention=os.environ.get("INFERENCE_ATTENTION_IMPL", attn))


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, impl: str) -> torch.Tensor:
    if impl == "cuda":
        from inference.kernels.cuda_ops import fused_rmsnorm
        return fused_rmsnorm(x, weight, eps)
    if impl == "nki":
        from inference.kernels.nki_ops import nki_rmsnorm
        return nki_rmsnorm(x, weight, eps)
    return rmsnorm_reference(x, weight, eps)


def apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, impl: str) -> torch.Tensor:
    if impl == "triton":
        from inference.kernels.triton_rope import apply_rope_triton
        return apply_rope_triton(q, cos, sin)
    if impl == "nki":
        from inference.kernels.nki_ops import nki_rope
        return nki_rope(q, cos, sin)
    return apply_rope_reference(q, cos, sin)
