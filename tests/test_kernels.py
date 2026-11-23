import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA GPU")


@requires_cuda
def test_fused_rmsnorm_matches_reference():
    from inference.kernels.cuda_ops import fused_rmsnorm, rmsnorm_reference
    x = torch.randn(64, 1024, device="cuda", dtype=torch.float16)
    w = torch.randn(1024, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(fused_rmsnorm(x, w), rmsnorm_reference(x, w), atol=1e-2, rtol=1e-2)


@requires_cuda
def test_triton_rope_matches_reference():
    from inference.kernels.triton_rope import apply_rope_reference, apply_rope_triton
    from inference.model import ModelConfig, rope_tables
    cos, sin = rope_tables(ModelConfig())
    q = torch.randn(2, 16, 128, 64, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(apply_rope_triton(q, cos[:128], sin[:128]),
                               apply_rope_reference(q, cos[:128], sin[:128]), atol=1e-2, rtol=1e-2)


@requires_cuda
def test_generate_smoke():
    from inference.engine import InferenceEngine
    from inference.model import ModelConfig
    eng = InferenceEngine(ModelConfig(num_layers=2, hidden_size=256, num_heads=4, intermediate_size=512))
    out = eng.generate([1, 2, 3], max_new_tokens=4)
    assert len(out["output_ids"]) == 7
