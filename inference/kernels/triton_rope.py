"""Rotary position embedding (RoPE) implemented as a Triton kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(q_ptr, cos_ptr, sin_ptr, out_ptr,
                 seq_len, head_dim: tl.constexpr, half: tl.constexpr,
                 stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = row % seq_len
    offs = tl.arange(0, BLOCK)
    mask = offs < half
    base = q_ptr + row * stride_row
    x1 = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base + half + offs, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + pos * half + offs, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + pos * half + offs, mask=mask, other=0.0)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    out = out_ptr + row * stride_row
    tl.store(out + offs, o1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out + half + offs, o2.to(out_ptr.dtype.element_ty), mask=mask)


def apply_rope_triton(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """q: [batch, heads, seq, head_dim] on CUDA; cos/sin: [seq, head_dim // 2] float32."""
    assert q.is_cuda, "Triton RoPE requires CUDA"
    b, h, s, d = q.shape
    q2 = q.contiguous().view(-1, d)
    out = torch.empty_like(q2)
    half = d // 2
    _rope_kernel[(q2.shape[0],)](q2, cos.contiguous(), sin.contiguous(), out,
                                 s, d, half, q2.stride(0), BLOCK=triton.next_power_of_2(half))
    return out.view(b, h, s, d)


def apply_rope_reference(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = q.shape[-1] // 2
    x1, x2 = q[..., :half].float(), q[..., half:].float()
    c, s_ = cos[None, None], sin[None, None]
    return torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], dim=-1).to(q.dtype)
