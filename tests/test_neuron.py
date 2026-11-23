"""AWS Neuron (Trainium2) tests.

* ``neuron`` tests run only where TorchNeuron sees a real device (``torch.neuron.is_available()``).
* ``neuron_compile`` compiles the decoder graph to a Trn2 NEFF without a device. It needs
  torch-neuronx + neuronx-cc and these env vars, set before Python starts::

    TORCH_NEURONX_COMPILE_ONLY=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_LOGICAL_NC_CONFIG=2 \\
    NEURON_RT_FAKE_INSTANCE_TYPE=trn2.3xlarge python -m pytest -m neuron_compile
"""
import os

import pytest
import torch
import torch.nn.functional as F

from inference.backends import NeuronBackend
from inference.engine import InferenceEngine
from inference.model import load_model

requires_neuron = pytest.mark.skipif(not NeuronBackend.is_available(),
                                     reason="requires an AWS Neuron device (Trn2/Trn3)")


def _compare_to_cpu_reference(cfg, neuron_logits, ids):
    ref = load_model(cfg, backend="cpu")
    with torch.inference_mode():
        expected = ref(ids)
    actual = neuron_logits.float().cpu()
    cos = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    top1 = (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()
    return cos, top1


@pytest.mark.neuron
@requires_neuron
@pytest.mark.parametrize("compile", [True, False], ids=["torch.compile", "eager"])
def test_neuron_forward_matches_cpu_reference(tiny_cfg, compile):
    backend = NeuronBackend(compile=compile)
    model = backend.prepare_model(load_model(tiny_cfg, backend=backend))
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 64))
    with torch.inference_mode():
        logits = model(ids.to("neuron"))
        backend.synchronize()
    assert logits.device.type == "neuron"
    assert logits.shape == (2, 64, tiny_cfg.vocab_size)
    cos, top1 = _compare_to_cpu_reference(tiny_cfg, logits, ids)
    assert cos > 0.99, cos
    assert top1 > 0.9, top1


@pytest.mark.neuron
@requires_neuron
def test_neuron_generate(tiny_cfg):
    eng = InferenceEngine(tiny_cfg, backend="neuron")
    out = eng.generate([1, 2, 3], max_new_tokens=8)
    assert out["backend"] == "neuron" and out["device"].startswith("AWS Neuron")
    assert len(out["output_ids"]) == 11
    assert all(0 <= t < tiny_cfg.vocab_size for t in out["output_ids"])
    assert eng.generate([1, 2, 3], max_new_tokens=8)["output_ids"] == out["output_ids"]


@pytest.mark.neuron_compile
@pytest.mark.skipif(os.environ.get("TORCH_NEURONX_COMPILE_ONLY") != "1",
                    reason="device-free Trn2 compile; see the module docstring for the env vars")
def test_trn2_compile_only(tiny_cfg):
    """Compile the decoder at the static generate shape to a Trn2 NEFF, with no Neuron device."""
    import torch_neuronx.compiler

    with torch_neuronx.compiler.compile_only():
        backend = NeuronBackend(compile=True)
        model = backend.prepare_model(load_model(tiny_cfg, backend=backend))
        ids = torch.zeros(1, 64, dtype=torch.long, device="neuron")
        last_pos = torch.tensor([2], dtype=torch.long, device="neuron")
        with torch.inference_mode():
            out = model(ids, last_pos)
        torch.neuron.synchronize()
    assert out.shape == (1, tiny_cfg.vocab_size)
    artifact = out._neff_artifact
    neff_info = torch_neuronx._C.get_neff_info(artifact.cache_key)
    assert neff_info and neff_info[1], f"no NEFF for cache key {artifact.cache_key}"
