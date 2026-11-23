"""CPU correctness of the model and of the code paths the Neuron backend relies on."""
import pytest
import torch
import torch.nn.functional as F

from inference.backends import KernelConfig
from inference.model import Decoder, ModelConfig, causal_attention_math, load_model


def test_load_model_cpu_defaults(tiny_cfg):
    model = load_model(tiny_cfg, backend="cpu")
    p = next(model.parameters())
    assert p.device.type == "cpu" and p.dtype == torch.float32
    assert not model.training


def test_forward_shape(tiny_cfg):
    model = load_model(tiny_cfg, backend="cpu")
    ids = torch.randint(0, tiny_cfg.vocab_size, (3, 17))
    with torch.inference_mode():
        assert model(ids).shape == (3, 17, tiny_cfg.vocab_size)


def test_default_config_param_count_unchanged():
    model = Decoder(ModelConfig())
    assert sum(p.numel() for p in model.parameters()) == 168_313_856


def test_weights_are_deterministic(tiny_cfg):
    a, b = load_model(tiny_cfg, backend="cpu"), load_model(tiny_cfg, backend="cpu")
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


@pytest.mark.parametrize("shape", [(2, 4, 1, 32), (2, 4, 9, 32), (1, 2, 64, 16)])
def test_math_attention_matches_sdpa(shape):
    torch.manual_seed(0)
    q, k, v = (torch.randn(shape) for _ in range(3))
    torch.testing.assert_close(causal_attention_math(q, k, v),
                               F.scaled_dot_product_attention(q, k, v, is_causal=True),
                               atol=1e-5, rtol=1e-5)


def test_neuron_kernel_config_matches_cpu_model(tiny_cfg):
    """The Neuron graph (math attention) computes the same function as the CPU graph."""
    torch.manual_seed(0)
    cpu = Decoder(tiny_cfg, KernelConfig(attention="sdpa")).eval()
    torch.manual_seed(0)
    neuron_like = Decoder(tiny_cfg, KernelConfig(attention="math")).eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 24))
    with torch.inference_mode():
        torch.testing.assert_close(neuron_like(ids), cpu(ids), atol=1e-4, rtol=1e-4)


def test_last_pos_gather_equals_slice(tiny_cfg):
    model = load_model(tiny_cfg, backend="cpu")
    ids = torch.randint(0, tiny_cfg.vocab_size, (3, 20))
    last = torch.tensor([0, 7, 19])
    with torch.inference_mode():
        full = model(ids)
        gathered = model(ids, last)
    assert gathered.shape == (3, tiny_cfg.vocab_size)
    torch.testing.assert_close(gathered, full[torch.arange(3), last])


def test_right_padding_does_not_change_real_positions(tiny_cfg):
    """Causal masking makes static right-padding exact — the basis of the Neuron generate loop."""
    model = load_model(tiny_cfg, backend="cpu")
    ids = torch.randint(1, tiny_cfg.vocab_size, (1, 11))
    padded = torch.zeros(1, 64, dtype=torch.long)
    padded[:, :11] = ids
    with torch.inference_mode():
        torch.testing.assert_close(model(padded)[:, :11], model(ids), atol=1e-5, rtol=1e-5)


def test_bf16_close_to_fp32_reference(tiny_cfg):
    """Neuron runs in bfloat16; check bf16 numerics against the fp32 reference on CPU."""
    ref = load_model(tiny_cfg, backend="cpu")
    bf16 = load_model(tiny_cfg, dtype=torch.bfloat16, backend="cpu")
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 32))
    with torch.inference_mode():
        a, b = ref(ids), bf16(ids).float()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    assert cos > 0.999
    assert (a.argmax(-1) == b.argmax(-1)).float().mean() > 0.9


def test_unknown_kernel_impl_rejected(tiny_cfg):
    with pytest.raises(ValueError):
        Decoder(tiny_cfg, KernelConfig(attention="cudnn"))
    with pytest.raises(ValueError):
        Decoder(tiny_cfg, KernelConfig(rope="cuda"))
