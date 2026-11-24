"""Greedy generation engine for CUDA, Neuron, or CPU."""
from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from inference.device import DeviceTimer, configure_runtime, device_name, resolve_device, resolve_dtype, torch_device
from inference.distributed import all_reduce_logits, init_distributed
from inference.model import ModelConfig, load_model


class InferenceEngine:
    def __init__(self, cfg: ModelConfig = ModelConfig(), dtype=None, device: str | None = None,
                 seq_buckets=None, artifact_dir: str | None = None):
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        self.rank, self.world_size = init_distributed(self.device)
        configure_runtime(self.device)
        self.model = load_model(cfg, self.dtype, self.device)
        self.max_seq_len = cfg.max_seq_len
        self.stream = torch.cuda.Stream() if self.device == "cuda" else None
        if self.device == "neuron":
            from inference.neuron import BucketedDecoder, parse_buckets

            buckets = parse_buckets(seq_buckets, cfg.max_seq_len)
            self._next_logits = BucketedDecoder.load_or_compile(self.model, buckets, artifact_dir=artifact_dir)
            self.max_seq_len = self._next_logits.max_seq_len
        else:
            self._next_logits = lambda ids: self.model(ids)[:, -1, :]

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int = 32) -> dict:
        if not input_ids:
            raise ValueError("input_ids must be non-empty")
        if len(input_ids) + max_new_tokens > self.max_seq_len:
            raise ValueError(f"prompt + max_new_tokens = {len(input_ids) + max_new_tokens} exceeds "
                             f"max sequence length {self.max_seq_len}")
        ids = torch.tensor([input_ids], device=torch_device(self.device), dtype=torch.long)
        t0 = time.perf_counter()
        stream_ctx = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        with stream_ctx, DeviceTimer(self.device, self.stream) as timer:
            for _ in range(max_new_tokens):
                logits = all_reduce_logits(self._next_logits(ids))
                next_id = logits.argmax(-1, keepdim=True).to(ids.device)
                ids = torch.cat([ids, next_id], dim=1)
        out = {
            "output_ids": ids[0].tolist(),
            "device_ms": timer.elapsed_ms,
            "wall_ms": (time.perf_counter() - t0) * 1000,
            "device": device_name(self.device),
            "backend": self.device,
        }
        if self.device == "cuda":
            out["gpu_ms"] = timer.elapsed_ms  # kept for existing clients
        return out
