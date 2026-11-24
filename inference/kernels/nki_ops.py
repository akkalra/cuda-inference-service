"""NKI (Neuron Kernel Interface) ports of the custom CUDA RMSNorm and Triton RoPE kernels.

Written against the current NKI API (``import nki``; ``nki.isa`` instructions with explicit
``dst=`` outputs and ``nisa.dma_copy`` for HBM <-> SBUF movement), targeting NeuronCore-v2
(trn1 / inf2). Both kernels tile the token/sequence axis onto the 128 SBUF partitions and keep
the hidden/head dimension in the free axis, so every row is processed by one partition lane.

The same ``@nki.jit`` functions are used three ways:

* inside a model compiled with ``torch_neuronx.trace`` (XLA tensors -> NKI custom call),
* on the NKI CPU simulator (``nki.simulate(kernel)(numpy arrays)``) in tests/CI,
* directly on a Neuron device.

The ``nki`` package is optional: on CUDA/CPU installs this module imports cleanly and
``HAS_NKI`` is False.
"""
from __future__ import annotations

import torch

try:  # pragma: no cover - exercised only where the Neuron toolchain is installed
    import nki
    import nki.isa as nisa
    import nki.language as nl

    HAS_NKI = True
except ImportError:  # CUDA / CPU installs
    HAS_NKI = False


if HAS_NKI:

    @nki.jit
    def rmsnorm_kernel(x, weight, eps):
        """out = (x * rsqrt(mean(x^2, -1) + eps)) * weight.

        x: [N, H] in HBM, weight: [1, H]. Rows are tiled 128 at a time onto partitions.
        Matches ``rmsnorm_reference``: statistics and normalization in fp32 (the Scalar
        Engine always computes in fp32), cast to x.dtype, then scaled by ``weight``.
        """
        n_rows, hidden = x.shape
        pmax = nl.tile_size.pmax
        out = nl.ndarray((n_rows, hidden), dtype=x.dtype, buffer=nl.shared_hbm)

        # Broadcast gamma across all partitions once; reused by every row tile.
        w_tile = nl.ndarray((pmax, hidden), dtype=weight.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_tile, src=weight.broadcast(0, pmax))

        for t in nl.affine_range((n_rows + pmax - 1) // pmax):
            r0 = t * pmax
            rows = min(pmax, n_rows - r0)

            x_tile = nl.ndarray((pmax, hidden), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile[0:rows, :], src=x[r0:r0 + rows, :])

            # square + row-sum fused in one Scalar Engine instruction.
            sq = nl.ndarray((pmax, hidden), dtype=nl.float32, buffer=nl.sbuf)
            sum_sq = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sq[0:rows, :], op=nl.square, data=x_tile[0:rows, :],
                            reduce_op=nl.add, reduce_res=sum_sq[0:rows, :],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)

            # mean + eps, then rsqrt -> one scale per row.
            mean_sq = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mean_sq[0:rows, :], data=sum_sq[0:rows, :],
                               op0=nl.multiply, operand0=1.0 / hidden, op1=nl.add, operand1=eps)
            rstd = nl.ndarray((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rstd[0:rows, :], op=nl.rsqrt, data=mean_sq[0:rows, :])

            # x * rstd (per-partition scale), cast to x.dtype like the reference.
            normed = nl.ndarray((pmax, hidden), dtype=x.dtype, buffer=nl.sbuf)
            nisa.activation(dst=normed[0:rows, :], op=nl.copy, data=x_tile[0:rows, :],
                            scale=rstd[0:rows, :])

            y_tile = nl.ndarray((pmax, hidden), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y_tile[0:rows, :], data1=normed[0:rows, :],
                               data2=w_tile[0:rows, :], op=nl.multiply)
            nisa.dma_copy(dst=out[r0:r0 + rows, :], src=y_tile[0:rows, :])
        return out

    @nki.jit
    def rope_kernel(q, cos, sin):
        """Rotate-half RoPE. q: [B, NH, S, D]; cos/sin: [S, D/2].

        Sequence positions go on partitions so each cos/sin tile is loaded once per
        128-token block and reused for every (batch, head).
        """
        batch, n_heads, seq, head_dim = q.shape
        half = head_dim // 2
        pmax = nl.tile_size.pmax
        out = nl.ndarray((batch, n_heads, seq, head_dim), dtype=q.dtype, buffer=nl.shared_hbm)

        for t in nl.affine_range((seq + pmax - 1) // pmax):
            s0 = t * pmax
            rows = min(pmax, seq - s0)
            c = nl.ndarray((pmax, half), dtype=cos.dtype, buffer=nl.sbuf)
            s = nl.ndarray((pmax, half), dtype=sin.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=c[0:rows, :], src=cos[s0:s0 + rows, :])
            nisa.dma_copy(dst=s[0:rows, :], src=sin[s0:s0 + rows, :])

            for b in nl.affine_range(batch):
                for h in nl.affine_range(n_heads):
                    x = nl.ndarray((pmax, head_dim), dtype=q.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=x[0:rows, :], src=q[b, h, s0:s0 + rows, :])
                    x1, x2 = x[0:rows, 0:half], x[0:rows, half:head_dim]
                    y = nl.ndarray((pmax, head_dim), dtype=q.dtype, buffer=nl.sbuf)

                    # y1 = x1*cos - x2*sin (fp32 intermediates, one cast on write)
                    a = nl.ndarray((pmax, half), dtype=nl.float32, buffer=nl.sbuf)
                    b_ = nl.ndarray((pmax, half), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=a[0:rows, :], data1=x1, data2=c[0:rows, :], op=nl.multiply)
                    nisa.tensor_tensor(dst=b_[0:rows, :], data1=x2, data2=s[0:rows, :], op=nl.multiply)
                    nisa.tensor_tensor(dst=y[0:rows, 0:half], data1=a[0:rows, :], data2=b_[0:rows, :],
                                       op=nl.subtract)

                    # y2 = x2*cos + x1*sin
                    a2 = nl.ndarray((pmax, half), dtype=nl.float32, buffer=nl.sbuf)
                    b2 = nl.ndarray((pmax, half), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=a2[0:rows, :], data1=x2, data2=c[0:rows, :], op=nl.multiply)
                    nisa.tensor_tensor(dst=b2[0:rows, :], data1=x1, data2=s[0:rows, :], op=nl.multiply)
                    nisa.tensor_tensor(dst=y[0:rows, half:head_dim], data1=a2[0:rows, :],
                                       data2=b2[0:rows, :], op=nl.add)

                    nisa.dma_copy(dst=out[b, h, s0:s0 + rows, :], src=y[0:rows, :])
        return out


def _require_nki():
    if not HAS_NKI:
        raise RuntimeError("NKI kernels need the Neuron toolchain: pip install -r requirements-neuron.txt")


def nki_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Drop-in for ``fused_rmsnorm``; call on XLA tensors (i.e. under ``torch_neuronx.trace``)."""
    _require_nki()
    shape = x.shape
    out = rmsnorm_kernel(x.reshape(-1, shape[-1]), weight.reshape(1, -1), float(eps))
    return out.reshape(shape)


def nki_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Drop-in for ``apply_rope_triton``; call on XLA tensors (i.e. under ``torch_neuronx.trace``)."""
    _require_nki()
    return rope_kernel(q.contiguous(), cos.contiguous(), sin.contiguous())
