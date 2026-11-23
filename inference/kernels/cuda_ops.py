"""JIT-compiled custom CUDA ops."""
import os

import torch

from inference.kernels.reference import rmsnorm_reference  # noqa: F401  (re-exported)

_CSRC = os.path.join(os.path.dirname(__file__), "..", "..", "csrc")
_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        _ext = load(
            name="fused_ops",
            sources=[os.path.join(_CSRC, "bindings.cpp"), os.path.join(_CSRC, "fused_rmsnorm_kernel.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_80,code=sm_80",
                               "-gencode=arch=compute_90,code=sm_90"],
            verbose=False,
        )
    return _ext


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    out = torch.empty_like(x)
    _load().fused_rmsnorm(out, x.contiguous(), weight, eps)
    return out
