import pytest
import torch

from inference import backends
from inference.backends import (BackendUnavailableError, CPUBackend, CUDABackend, KernelConfig,
                                NeuronBackend, get_backend)


def test_explicit_cpu():
    b = get_backend("cpu")
    assert isinstance(b, CPUBackend)
    assert b.device == "cpu" and b.default_dtype == torch.float32
    assert not b.requires_static_shapes


def test_env_var_selects_backend(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "CPU")
    assert isinstance(get_backend(), CPUBackend)


def test_backend_instance_passthrough():
    b = CPUBackend()
    assert get_backend(b) is b


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend 'tpu'"):
        get_backend("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="host has CUDA")
def test_cuda_unavailable_raises_instead_of_falling_back():
    with pytest.raises(BackendUnavailableError, match="cuda"):
        get_backend("cuda")


@pytest.mark.skipif(NeuronBackend.is_available(), reason="host has Neuron")
def test_neuron_unavailable_raises_instead_of_falling_back():
    with pytest.raises(BackendUnavailableError, match="neuron-ls"):
        get_backend("neuron")


def test_auto_order_prefers_cuda_then_neuron_then_cpu(monkeypatch):
    def fake(cuda, neuron):
        monkeypatch.setattr(CUDABackend, "is_available", classmethod(lambda cls: cuda))
        monkeypatch.setattr(NeuronBackend, "is_available", classmethod(lambda cls: neuron))

    fake(False, False)
    assert isinstance(get_backend("auto"), CPUBackend)
    fake(False, True)
    assert isinstance(get_backend("auto"), NeuronBackend)
    fake(True, True)
    assert isinstance(get_backend("auto"), CUDABackend)


def test_available_backends_always_includes_cpu():
    assert "cpu" in backends.available_backends()


def test_kernel_configs_preserve_cuda_path_and_avoid_cuda_on_others():
    assert CUDABackend.kernels == KernelConfig(rmsnorm="cuda", rope="triton", attention="flash")
    assert CPUBackend.kernels == KernelConfig(rmsnorm="reference", rope="reference", attention="sdpa")
    assert NeuronBackend.kernels == KernelConfig(rmsnorm="reference", rope="reference", attention="math")
    assert CUDABackend.default_dtype == torch.float16
    assert NeuronBackend.default_dtype == torch.bfloat16
    assert NeuronBackend.requires_static_shapes


def test_distributed_backends():
    assert CUDABackend().distributed_backend() == "nccl"
    assert NeuronBackend(compile=False).distributed_backend() == "neuron"
    assert CPUBackend().distributed_backend() == "gloo"


def test_neuron_compile_flag(monkeypatch):
    monkeypatch.setenv("INFERENCE_NEURON_COMPILE", "0")
    assert NeuronBackend().compile is False
    monkeypatch.delenv("INFERENCE_NEURON_COMPILE")
    assert NeuronBackend().compile is True
    model = torch.nn.Linear(2, 2)
    assert NeuronBackend(compile=False).prepare_model(model) is model


def test_cpu_timer_reports_elapsed_ms():
    with CPUBackend().timer() as t:
        pass
    assert t["elapsed_ms"] >= 0


def test_neuron_compile_only_mode_is_not_a_device(monkeypatch):
    import sys
    import types

    monkeypatch.setattr(torch, "neuron", types.SimpleNamespace(is_available=lambda: True), raising=False)
    fake = types.ModuleType("torch_neuronx")
    fake._C = types.SimpleNamespace(_is_compile_only=lambda: True)
    monkeypatch.setitem(sys.modules, "torch_neuronx", fake)
    assert not NeuronBackend.is_available()
    fake._C = types.SimpleNamespace(_is_compile_only=lambda: False)
    assert NeuronBackend.is_available()
