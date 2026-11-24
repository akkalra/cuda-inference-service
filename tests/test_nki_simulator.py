"""NKI kernels vs. the pure-PyTorch references, on the NKI CPU simulator.

Runs wherever the ``nki`` package is installed (x86 Linux, from pip.repos.neuron.amazonaws.com);
skipped otherwise. Simulates a trn1 NeuronCore-v2 unless NEURON_PLATFORM_TARGET_OVERRIDE is set.
This validates kernel *numerics and API usage*, not performance.
"""
import os

import numpy as np
import pytest
import torch

nki = pytest.importorskip("nki")
ml_dtypes = pytest.importorskip("ml_dtypes")

from inference.kernels.nki_ops import rmsnorm_kernel, rope_kernel  # noqa: E402
from inference.kernels.reference import apply_rope_reference, rmsnorm_reference  # noqa: E402
from inference.model import ModelConfig, rope_tables  # noqa: E402

pytestmark = pytest.mark.nki_simulator
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "1")  # NeuronCore-v2 (trn1/inf2) is LNC=1 only

DTYPES = {"float32": (np.float32, torch.float32, 1e-5), "bfloat16": (ml_dtypes.bfloat16, torch.bfloat16, 2e-2)}


def _to_torch(a, tdtype):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(tdtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows,hidden", [(128, 1024), (300, 1024), (7, 256)])
def test_nki_rmsnorm_matches_reference(rows, hidden, dtype):
    np_dt, t_dt, tol = DTYPES[dtype]
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((rows, hidden)) * 3).astype(np.float32)
    w = (1 + 0.1 * rng.standard_normal((1, hidden))).astype(np.float32)
    out = nki.simulate(rmsnorm_kernel)(x.astype(np_dt), w.astype(np_dt), 1e-6)
    ref = rmsnorm_reference(_to_torch(x, t_dt), _to_torch(w[0], t_dt), 1e-6)
    torch.testing.assert_close(_to_torch(out, torch.float32), ref.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch,heads,seq,head_dim", [(1, 16, 128, 64), (2, 4, 200, 64), (1, 2, 5, 128)])
def test_nki_rope_matches_reference(batch, heads, seq, head_dim, dtype):
    np_dt, t_dt, tol = DTYPES[dtype]
    cfg = ModelConfig(hidden_size=heads * head_dim, num_heads=heads)
    cos, sin = rope_tables(cfg, device="cpu")
    cos, sin = cos[:seq], sin[:seq]
    q = np.random.default_rng(1).standard_normal((batch, heads, seq, head_dim)).astype(np.float32)
    out = nki.simulate(rope_kernel)(q.astype(np_dt), cos.numpy(), sin.numpy())
    ref = apply_rope_reference(_to_torch(q, t_dt), cos, sin)
    torch.testing.assert_close(_to_torch(out, torch.float32), ref.float(), atol=tol, rtol=tol)
