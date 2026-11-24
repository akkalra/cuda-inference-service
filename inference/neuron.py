"""Compile the decoder for AWS Trainium / Inferentia2 with ``torch_neuronx.trace``.

Neuron graphs are compiled ahead of time for fixed input shapes. Generation produces a new
sequence length every step, so we compile one graph per *sequence bucket* and right-pad each
request up to the smallest bucket that fits. The traced graph takes ``(input_ids [b, S],
last_pos [b])`` and returns next-token logits ``[b, vocab]``; causal masking guarantees the
padded tail cannot change the logits at ``last_pos``.

Compiled graphs are saved with ``torch.jit.save`` so a fleet loads NEFFs instead of
recompiling at startup (``NEURON_ARTIFACT_DIR``). File names carry a fingerprint of the model
config, dtype, kernel selection, compiler flags and torch-neuronx version, so a config change
never silently loads a stale graph. Compile on any x86 host with
``cpu_backend=True`` + ``--target``; run on trn1/inf2.
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable, Iterable

import torch

DEFAULT_SEQ_BUCKETS = (128, 256, 512, 1024, 2048)
SEQ_BUCKETS_ENV = "NEURON_SEQ_BUCKETS"
ARTIFACT_DIR_ENV = "NEURON_ARTIFACT_DIR"
TARGET_ENV = "NEURON_COMPILE_TARGET"


def parse_buckets(value: str | Iterable[int] | None, max_seq_len: int | None = None) -> tuple[int, ...]:
    if value is None:
        value = os.environ.get(SEQ_BUCKETS_ENV) or DEFAULT_SEQ_BUCKETS
    if isinstance(value, str):
        value = [int(v) for v in value.split(",") if v.strip()]
    buckets = tuple(sorted(set(int(v) for v in value)))
    if max_seq_len is not None:
        buckets = tuple(b for b in buckets if b <= max_seq_len)
    if not buckets or buckets[0] <= 0:
        raise ValueError(f"invalid sequence buckets: {value!r}")
    return buckets


def pick_bucket(length: int, buckets: tuple[int, ...]) -> int:
    for b in buckets:
        if length <= b:
            return b
    raise ValueError(f"sequence length {length} exceeds the largest compiled bucket {buckets[-1]}")


def pad_to_bucket(input_ids: torch.Tensor, bucket: int, pad_id: int = 0):
    """Right-pad [b, n] ids to [b, bucket]; returns (padded_ids, last_pos [b])."""
    b, n = input_ids.shape
    padded = torch.full((b, bucket), pad_id, dtype=input_ids.dtype)
    padded[:, :n] = input_ids
    last_pos = torch.full((b,), n - 1, dtype=torch.long)
    return padded, last_pos


def compiler_args(target: str | None = None) -> list[str]:
    # --model-type=transformer enables neuronx-cc's transformer-specific optimizations.
    # --auto-cast=none: weights are already bf16; never silently downcast fp32 ops.
    args = ["--model-type=transformer", "--auto-cast=none"]
    target = target or os.environ.get(TARGET_ENV)
    if target:
        args += ["--target", target]
    return args


def artifact_tag(model, target: str | None = None) -> str:
    """Fingerprint of everything that changes the compiled graph besides its shape."""
    import hashlib

    try:
        import torch_neuronx
        neuronx_version = torch_neuronx.__version__
    except ImportError:
        neuronx_version = "none"
    dtype = next(model.parameters()).dtype
    key = repr((model.cfg, model.kernels, dtype, compiler_args(target), neuronx_version))
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def artifact_path(artifact_dir: str, batch: int, seq: int, tag: str) -> str:
    return os.path.join(artifact_dir, f"decoder_{tag}_b{batch}_s{seq}.pt")


def trace_decoder(model: torch.nn.Module, batch: int, seq: int, target: str | None = None,
                  cpu_backend: bool = False, compiler_workdir: str | None = None):
    import torch_neuronx

    example = (torch.zeros(batch, seq, dtype=torch.long), torch.full((batch,), seq - 1, dtype=torch.long))
    if cpu_backend and not (target or os.environ.get(TARGET_ENV)):
        raise ValueError("cpu_backend=True requires a compile target (trn1 or inf2)")
    # Keep neuronx-cc scratch output out of the working directory.
    compiler_workdir = compiler_workdir or tempfile.mkdtemp(prefix="neuronx-cc-")
    return torch_neuronx.trace(model, example, compiler_args=compiler_args(target),
                               cpu_backend=cpu_backend, compiler_workdir=compiler_workdir)


class BucketedDecoder:
    """Routes variable-length requests to fixed-shape graphs (one per sequence bucket)."""

    def __init__(self, graphs: dict[int, Callable], batch_size: int = 1, pad_id: int = 0):
        self.graphs = dict(sorted(graphs.items()))
        self.buckets = tuple(self.graphs)
        self.batch_size = batch_size
        self.pad_id = pad_id

    @property
    def max_seq_len(self) -> int:
        return self.buckets[-1]

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] != self.batch_size:
            raise ValueError(f"graphs were compiled for batch {self.batch_size}, got {input_ids.shape[0]}")
        bucket = pick_bucket(input_ids.shape[1], self.buckets)
        padded, last_pos = pad_to_bucket(input_ids.cpu(), bucket, self.pad_id)
        return self.graphs[bucket](padded, last_pos)

    @classmethod
    def compile(cls, model, buckets, batch_size: int = 1, target: str | None = None,
                cpu_backend: bool = False, artifact_dir: str | None = None) -> "BucketedDecoder":
        graphs = {}
        tag = artifact_tag(model, target)
        for seq in buckets:
            graphs[seq] = trace_decoder(model, batch_size, seq, target=target, cpu_backend=cpu_backend)
            if artifact_dir:
                os.makedirs(artifact_dir, exist_ok=True)
                torch.jit.save(graphs[seq], artifact_path(artifact_dir, batch_size, seq, tag))
        return cls(graphs, batch_size)

    @classmethod
    def load(cls, artifact_dir: str, buckets, tag: str, batch_size: int = 1) -> "BucketedDecoder":
        import torch_neuronx  # noqa: F401  registers the Neuron ops needed to deserialize graphs

        return cls({seq: torch.jit.load(artifact_path(artifact_dir, batch_size, seq, tag)) for seq in buckets},
                   batch_size)

    @classmethod
    def load_or_compile(cls, model, buckets, batch_size: int = 1, artifact_dir: str | None = None,
                        target: str | None = None) -> "BucketedDecoder":
        artifact_dir = artifact_dir or os.environ.get(ARTIFACT_DIR_ENV)
        tag = artifact_tag(model, target)
        if artifact_dir and all(os.path.exists(artifact_path(artifact_dir, batch_size, s, tag)) for s in buckets):
            return cls.load(artifact_dir, buckets, tag, batch_size)
        return cls.compile(model, buckets, batch_size, target=target, artifact_dir=artifact_dir)


def main():
    """Precompile bucketed graphs: python -m inference.neuron --target trn1 --out ./neuron_artifacts"""
    import argparse

    from inference.model import ModelConfig, load_model

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--out", required=True, help="artifact directory (set NEURON_ARTIFACT_DIR to this at serve time)")
    p.add_argument("--buckets", default=None, help="comma-separated sequence buckets")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--target", default=None, choices=["trn1", "inf2"],
                   help="compile target; required with --cpu-backend")
    p.add_argument("--cpu-backend", action="store_true",
                   help="compile on a host without Neuron devices (CI / build machines)")
    args = p.parse_args()
    cfg = ModelConfig()
    model = load_model(cfg, device="neuron")
    buckets = parse_buckets(args.buckets, cfg.max_seq_len)
    BucketedDecoder.compile(model, buckets, args.batch, target=args.target,
                            cpu_backend=args.cpu_backend, artifact_dir=args.out)
    print(f"compiled buckets {buckets} (batch={args.batch}) -> {args.out}")


if __name__ == "__main__":
    main()
