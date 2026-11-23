"""Hardware backends: CPU, NVIDIA CUDA, and AWS Neuron (Trainium2 via TorchNeuron).

All device-specific behaviour (device placement, dtype, kernel choice, runtime flags,
compilation, synchronization, timing, distributed init) lives here so the model and
engine stay hardware-agnostic.
"""
from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass

import torch

BACKEND_ENV_VAR = "INFERENCE_BACKEND"
NEURON_COMPILE_ENV_VAR = "INFERENCE_NEURON_COMPILE"
SUPPORTED_BACKENDS = ("cpu", "cuda", "neuron")


class BackendUnavailableError(RuntimeError):
    """A backend was requested explicitly but its hardware/software stack is not present."""


@dataclass(frozen=True)
class KernelConfig:
    """Which implementation the model uses for each op that has a hardware-specific kernel."""

    rmsnorm: str = "reference"  # "cuda" (cpp_extension) | "reference"
    rope: str = "reference"  # "triton" | "reference"
    attention: str = "sdpa"  # "flash" (forced FlashAttention) | "sdpa" | "math"


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


class Backend:
    name: str = ""
    device: str = ""
    default_dtype: torch.dtype = torch.float32
    kernels: KernelConfig = KernelConfig()
    # Neuron compiles one graph per input shape, so the engine pads to a fixed length.
    requires_static_shapes: bool = False

    @classmethod
    def is_available(cls) -> bool:
        raise NotImplementedError

    def configure_runtime(self) -> None:
        """Global runtime flags (TF32, cuDNN, ...). Called once before the model is loaded."""

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the callable used for inference (eager model or a compiled wrapper)."""
        return model

    def synchronize(self) -> None:
        """Block until all queued device work has finished."""

    def stream(self):
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def timer(self):
        """Yields a dict; on exit ``elapsed_ms`` holds the synchronized elapsed time."""
        result: dict = {}
        self.synchronize()
        t0 = time.perf_counter()
        try:
            yield result
        finally:
            self.synchronize()
            result["elapsed_ms"] = (time.perf_counter() - t0) * 1000

    def device_name(self) -> str:
        return self.name

    def distributed_backend(self) -> str:
        return "gloo"

    def set_local_device(self, local_rank: int) -> None:
        """Bind this process to its local accelerator before distributed init."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(device={self.device!r}, dtype={self.default_dtype})"


class CPUBackend(Backend):
    name = device = "cpu"
    default_dtype = torch.float32
    kernels = KernelConfig(rmsnorm="reference", rope="reference", attention="sdpa")

    @classmethod
    def is_available(cls) -> bool:
        return True

    def device_name(self) -> str:
        return "cpu"


class CUDABackend(Backend):
    """The original NVIDIA execution path, unchanged in behaviour."""

    name = device = "cuda"
    default_dtype = torch.float16
    kernels = KernelConfig(rmsnorm="cuda", rope="triton", attention="flash")

    def __init__(self):
        self._stream = None

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()

    def configure_runtime(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def stream(self):
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return torch.cuda.stream(self._stream)

    @contextlib.contextmanager
    def timer(self):
        result: dict = {}
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield result
        finally:
            end.record()
            torch.cuda.synchronize()
            result["elapsed_ms"] = start.elapsed_time(end)

    def device_name(self) -> str:
        return torch.cuda.get_device_name(0)

    def distributed_backend(self) -> str:
        return "nccl"

    def set_local_device(self, local_rank: int) -> None:
        torch.cuda.set_device(local_rank)


class NeuronBackend(Backend):
    """AWS Trainium2/3 via TorchNeuron (torch-neuronx >= 2.13, native ``neuron`` device).

    * Kernels: the CUDA cpp_extension RMSNorm and the Triton RoPE have no Neuron build, so the
      pure-PyTorch reference implementations are used; neuronx-cc fuses them. Attention uses
      explicit matmul/softmax ("math"), which lowers to native ops with contiguous inputs.
    * Precision: bfloat16 weights (Trainium matmul engines are fastest in BF16).
    * Compilation: ``torch.compile(model, backend="neuron", dynamic=False, fullgraph=True)``.
      Set ``INFERENCE_NEURON_COMPILE=0`` to run in TorchNeuron eager mode instead.
    """

    name = device = "neuron"
    default_dtype = torch.bfloat16
    kernels = KernelConfig(rmsnorm="reference", rope="reference", attention="math")
    requires_static_shapes = True

    def __init__(self, compile: bool | None = None):
        self.compile = _env_flag(NEURON_COMPILE_ENV_VAR, True) if compile is None else compile

    @staticmethod
    def _torch_neuron():
        # torch-neuronx registers ``torch.neuron`` through PyTorch's device-backend autoload.
        return getattr(torch, "neuron", None)

    @classmethod
    def is_available(cls) -> bool:
        """True only for a real device: compile-only mode reports a fake Trn2 as available."""
        mod = cls._torch_neuron()
        try:
            if mod is None or not mod.is_available():
                return False
            import torch_neuronx

            return not torch_neuronx._C._is_compile_only()
        except Exception:
            return False

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.compile:
            return model
        return torch.compile(model, backend="neuron", dynamic=False, fullgraph=True)

    def synchronize(self) -> None:
        self._torch_neuron().synchronize()

    def device_name(self) -> str:
        instance = os.environ.get("NEURON_RT_FAKE_INSTANCE_TYPE") or _ec2_instance_type()
        cores = self._torch_neuron().device_count()
        return f"AWS Neuron ({instance or 'unknown instance'}, {cores} NeuronCore(s) visible)"

    def distributed_backend(self) -> str:
        return "neuron"

    def set_local_device(self, local_rank: int) -> None:
        self._torch_neuron().set_device(local_rank)


def _ec2_instance_type() -> str | None:
    path = "/sys/devices/virtual/dmi/id/product_name"
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


_BACKENDS: dict[str, type[Backend]] = {"cpu": CPUBackend, "cuda": CUDABackend, "neuron": NeuronBackend}
_AUTO_ORDER = ("cuda", "neuron", "cpu")


def available_backends() -> list[str]:
    return [name for name in SUPPORTED_BACKENDS if _BACKENDS[name].is_available()]


def get_backend(name: str | None = None) -> Backend:
    """Resolve a backend from ``name`` or ``$INFERENCE_BACKEND`` (default ``auto``).

    ``auto`` picks the first available of cuda -> neuron -> cpu. An explicitly requested
    backend that is not available raises :class:`BackendUnavailableError` instead of silently
    falling back, so a misconfigured accelerator host fails loudly.
    """
    if isinstance(name, Backend):
        return name
    requested = (name or os.environ.get(BACKEND_ENV_VAR) or "auto").strip().lower()
    if requested == "auto":
        for candidate in _AUTO_ORDER:
            if _BACKENDS[candidate].is_available():
                return _BACKENDS[candidate]()
    if requested not in _BACKENDS:
        raise ValueError(f"Unknown backend {requested!r}; expected one of {('auto',) + SUPPORTED_BACKENDS}")
    cls = _BACKENDS[requested]
    if not cls.is_available():
        hint = {
            "cuda": "no NVIDIA GPU visible to PyTorch (torch.cuda.is_available() is False)",
            "neuron": "torch-neuronx is not installed or no Neuron device is visible "
                      "(check `neuron-ls` on a Trn2 instance)",
        }.get(requested, "")
        raise BackendUnavailableError(f"Backend {requested!r} is not available: {hint}")
    return cls()
