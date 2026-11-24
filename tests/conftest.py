import os

import pytest
import torch

TINY = dict(vocab_size=512, hidden_size=128, num_layers=2, num_heads=4, intermediate_size=256, max_seq_len=256)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("INFERENCE_DEVICE", "INFERENCE_DTYPE", "INFERENCE_RMSNORM_IMPL", "INFERENCE_ROPE_IMPL",
                "INFERENCE_ATTENTION_IMPL", "NEURON_SEQ_BUCKETS", "NEURON_ARTIFACT_DIR"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def tiny_cfg():
    from inference.model import ModelConfig
    return ModelConfig(**TINY)


def pytest_configure(config):
    config.addinivalue_line("markers", "nki_simulator: NKI kernels on the NKI CPU simulator (needs nki)")
    config.addinivalue_line("markers", "neuron_compile: torch_neuronx.trace CPU-backend compile (needs torch-neuronx)")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
