"""Greedy generation engine; hardware specifics come from ``inference.backends``."""
import time

import torch

from inference.backends import Backend, get_backend
from inference.distributed import all_reduce_logits, init_distributed
from inference.model import ModelConfig, load_model


def static_length(cfg: ModelConfig, needed: int, min_len: int = 64) -> int:
    """Smallest power-of-two length >= ``needed`` (and ``min_len``), capped at ``max_seq_len``."""
    if needed > cfg.max_seq_len:
        raise ValueError(f"prompt + max_new_tokens = {needed} exceeds max_seq_len={cfg.max_seq_len}")
    length = min_len
    while length < needed:
        length *= 2
    return min(length, cfg.max_seq_len)


class InferenceEngine:
    def __init__(self, cfg: ModelConfig = ModelConfig(), dtype=None, backend: "Backend | str | None" = None):
        self.backend = get_backend(backend)
        self.cfg = cfg
        self.rank, self.world_size = init_distributed(self.backend)
        self.backend.configure_runtime()
        self.model = load_model(cfg, dtype, self.backend)
        self.forward = self.backend.prepare_model(self.model)

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int = 32) -> dict:
        t0 = time.perf_counter()
        with self.backend.stream(), self.backend.timer() as timing:
            if self.backend.requires_static_shapes:
                output_ids = self._generate_static(input_ids, max_new_tokens)
            else:
                output_ids = self._generate_dynamic(input_ids, max_new_tokens)
        result = {
            "output_ids": output_ids,
            "device_ms": timing["elapsed_ms"],
            "wall_ms": (time.perf_counter() - t0) * 1000,
            "device": self.backend.device_name(),
            "backend": self.backend.name,
        }
        if self.backend.name == "cuda":
            result["gpu_ms"] = timing["elapsed_ms"]  # kept for existing clients
        return result

    def _generate_dynamic(self, input_ids: list[int], max_new_tokens: int) -> list[int]:
        """Original loop: the sequence grows by one token per step (fine for eager CPU/CUDA)."""
        ids = torch.tensor([input_ids], device=self.backend.device, dtype=torch.long)
        for _ in range(max_new_tokens):
            logits = all_reduce_logits(self.forward(ids)[:, -1, :])
            next_id = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids[0].tolist()

    def _generate_static(self, input_ids: list[int], max_new_tokens: int) -> list[int]:
        """Fixed-shape loop for compiled backends: one compiled graph serves every step.

        The sequence is right-padded to a static length and next-token logits are gathered at the
        last real position. Causal attention means padding never influences earlier positions, so
        the generated tokens match the dynamic loop exactly.
        """
        # The last generated token is never fed back, so it needs no slot in the buffer.
        length = static_length(self.cfg, len(input_ids) + max_new_tokens - 1)
        ids = list(input_ids)
        buf = torch.zeros(1, length, dtype=torch.long)
        buf[0, : len(ids)] = torch.tensor(ids)
        for _ in range(max_new_tokens):
            last_pos = torch.tensor([len(ids) - 1], dtype=torch.long)
            logits = self.forward(buf.to(self.backend.device), last_pos.to(self.backend.device))
            next_id = int(all_reduce_logits(logits).argmax(-1).item())
            if len(ids) < length:
                buf[0, len(ids)] = next_id
            ids.append(next_id)
        return ids
