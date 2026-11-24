"""Device selection and device-specific runtime helpers.

The service runs on one of three backends, selected by ``INFERENCE_DEVICE``
(or an explicit argument):

* ``cuda``   - the original NVIDIA path (CUDA/Triton kernels, FlashAttention, CUDA events).
* ``neuron`` - AWS Trainium / Inferentia2 via torch-neuronx. The eager model lives on the
  host and is compiled into fixed-shape Neuron graphs with ``torch_neuronx.trace``.
* ``cpu``    - pure-PyTorch reference path, used by CI and for numerical ground truth.

``auto`` (the default) picks cuda, then neuron, then cpu.
"""
from __future__ import annotations

import glob
import os
import platform
import time
from contextlib import nullcontext

import torch

DEVICE_ENV = "INFERENCE_DEVICE"
DTYPE_ENV = "INFERENCE_DTYPE"
SUPPORTED_DEVICES = ("cuda", "neuron", "cpu")

# fp16 matches the original CUDA service. Trainium/Inferentia2 are bf16-native, and bf16
# avoids the fp16 overflow risk without needing loss-scaling-style care at inference time.
DEFAULT_DTYPES = {"cuda": torch.float16, "neuron": torch.bfloat16, "cpu": torch.float32}
_DTYPES = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16,
           "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32}


def neuron_available() -> bool:
    """True when a Neuron device is visible (``/dev/neuron*`` is created by the Neuron driver)."""
    return bool(glob.glob("/dev/neuron[0-9]*"))


def resolve_device(requested: str | None = None) -> str:
    name = (requested or os.environ.get(DEVICE_ENV) or "auto").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if neuron_available():
            return "neuron"
        return "cpu"
    if name not in SUPPORTED_DEVICES:
        raise ValueError(f"unsupported device {name!r}; expected one of {SUPPORTED_DEVICES} or 'auto'")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("INFERENCE_DEVICE=cuda but no CUDA device is available")
    return name


def resolve_dtype(device: str, requested: str | torch.dtype | None = None) -> torch.dtype:
    if isinstance(requested, torch.dtype):
        return requested
    name = requested or os.environ.get(DTYPE_ENV)
    if not name:
        return DEFAULT_DTYPES[device]
    try:
        return _DTYPES[name.strip().lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(_DTYPES)}") from None


def torch_device(device: str) -> torch.device:
    """Where eager tensors live. Neuron graphs take host (CPU) tensors at their boundary."""
    return torch.device("cuda") if device == "cuda" else torch.device("cpu")


def configure_runtime(device: str) -> None:
    """Backend-wide numerics/tuning flags.

    TF32 and cuDNN autotuning only exist on NVIDIA. On Neuron the equivalent knobs are
    compiler flags (``--auto-cast``, ``--model-type``), set in ``inference.neuron``.
    """
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def synchronize(device: str) -> None:
    # Neuron traced graphs are invoked synchronously and return host tensors, so there is
    # no device queue to drain; CPU is synchronous by definition.
    if device == "cuda":
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "neuron":
        return f"AWS Neuron ({os.environ.get('NEURON_COMPILE_TARGET', 'trn1/inf2')})"
    return platform.processor() or platform.machine() or "cpu"


class DeviceTimer:
    """Times a region in milliseconds.

    CUDA uses events on the given stream (device time, as in the original service).
    Neuron and CPU use host wall time around synchronous calls.
    """

    def __init__(self, device: str, stream: "torch.cuda.Stream | None" = None):
        self.device = device
        self.stream = stream
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self.device == "cuda":
            self._ctx = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
            self._ctx.__enter__()
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device == "cuda":
            self._end.record()
            self._ctx.__exit__(*exc)
            torch.cuda.synchronize()
            self.elapsed_ms = self._start.elapsed_time(self._end)
        else:
            synchronize(self.device)
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000
        return False
