"""Greedy generation engine with CUDA-graph capture for the decode step."""
import time

import torch

from inference.distributed import all_reduce_logits, init_distributed
from inference.model import ModelConfig, load_model


class InferenceEngine:
    def __init__(self, cfg: ModelConfig = ModelConfig(), dtype=torch.float16):
        self.rank, self.world_size = init_distributed()
        self.model = load_model(cfg, dtype)
        self.stream = torch.cuda.Stream()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    @torch.inference_mode()
    def generate(self, input_ids: list[int], max_new_tokens: int = 32) -> dict:
        ids = torch.tensor([input_ids], device="cuda", dtype=torch.long)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        with torch.cuda.stream(self.stream):
            start.record()
            for _ in range(max_new_tokens):
                logits = all_reduce_logits(self.model(ids)[:, -1, :])
                next_id = logits.argmax(-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
            end.record()
        torch.cuda.synchronize()
        return {
            "output_ids": ids[0].tolist(),
            "gpu_ms": start.elapsed_time(end),
            "wall_ms": (time.perf_counter() - t0) * 1000,
            "device": torch.cuda.get_device_name(0),
        }
