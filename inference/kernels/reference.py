"""Pure-PyTorch kernels: numerical ground truth, and the implementation used on CPU and AWS Neuron."""
import torch


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


def apply_rope_reference(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = q.shape[-1] // 2
    x1, x2 = q[..., :half].float(), q[..., half:].float()
    c, s_ = cos[None, None], sin[None, None]
    return torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], dim=-1).to(q.dtype)
