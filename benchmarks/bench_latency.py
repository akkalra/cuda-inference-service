"""Latency / throughput benchmark using CUDA events."""
import argparse
import json

import torch

from inference.model import ModelConfig, load_model


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    model = load_model(ModelConfig())
    x = torch.randint(0, 32000, (args.batch, args.seq), device="cuda")
    for _ in range(3):
        model(x)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        model(x)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters
    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "batch": args.batch, "seq": args.seq,
        "latency_ms": round(ms, 2),
        "tokens_per_s": round(args.batch * args.seq / (ms / 1000), 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }))


if __name__ == "__main__":
    main()
