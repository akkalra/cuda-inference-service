"""Device-agnostic latency / throughput / correctness / cost benchmark.

Measures the prefill forward pass (``batch x seq`` tokens -> next-token logits) on
cuda, neuron or cpu and reports:

* latency (mean / p50 / p90 ms per forward) and throughput (tokens/s),
* correctness vs. the pure-PyTorch reference model (fp32, CPU, same weights),
* estimated $ per 1M tokens = instance $/hr / (tokens/s x replicas x 3600) x 1e6.

Examples::

    python -m benchmarks.bench_latency --device cuda   --instance g5.xlarge
    python -m benchmarks.bench_latency --device neuron --instance inf2.xlarge
    python -m benchmarks.bench_latency --device cpu --model tiny --price-per-hour 0.17

Numbers are only meaningful on the hardware they were measured on.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from inference.device import DeviceTimer, configure_runtime, device_name, resolve_device, resolve_dtype, torch_device
from inference.kernels.dispatch import KernelConfig
from inference.model import ModelConfig, load_model

PRICES_PATH = os.path.join(os.path.dirname(__file__), "instance_prices.json")
MODELS = {
    "default": ModelConfig(),
    "tiny": ModelConfig(vocab_size=512, hidden_size=128, num_layers=2, num_heads=4,
                        intermediate_size=256, max_seq_len=256),
}


def resolve_price(instance: str | None, price_per_hour: float | None):
    """Returns (usd_per_hour | None, replicas, source)."""
    entry = None
    if instance:
        with open(PRICES_PATH) as f:
            table = json.load(f)["instances"]
        if instance not in table:
            raise SystemExit(f"unknown instance {instance!r}; known: {', '.join(table)} (or pass --price-per-hour)")
        entry = table[instance]
    replicas = entry["replicas"] if entry else 1
    if price_per_hour is not None:
        return price_per_hour, replicas, "--price-per-hour"
    if entry:
        return entry["usd_per_hour"], replicas, "benchmarks/instance_prices.json"
    return None, replicas, None


def cost_per_million_tokens(usd_per_hour: float, tokens_per_s: float, replicas: int = 1) -> float:
    return usd_per_hour / 3600.0 / (tokens_per_s * replicas) * 1e6


def build_forward(model, device: str, batch: int, seq: int, target: str | None):
    if device == "neuron":
        from inference.neuron import trace_decoder
        graph = trace_decoder(model, batch, seq, target=target)
        return lambda ids, last: graph(ids, last)
    return lambda ids, last: model(ids, last)


@torch.inference_mode()
def check_correctness(cfg: ModelConfig, logits: torch.Tensor, ids: torch.Tensor, last: torch.Tensor) -> dict:
    """Compare against the reference model (fp32, CPU, reference kernels) on the same input."""
    ref_model = load_model(cfg, torch.float32, "cpu", KernelConfig())
    ref = ref_model(ids.cpu(), last.cpu())
    got = logits.float().cpu()
    return {
        "reference": "fp32 CPU, reference kernels",
        "max_abs_err": round((got - ref).abs().max().item(), 5),
        "mean_abs_err": round((got - ref).abs().mean().item(), 6),
        "top1_agreement": (got.argmax(-1) == ref.argmax(-1)).float().mean().item(),
    }


@torch.inference_mode()
def main(argv=None):
    p = argparse.ArgumentParser(description="Device-agnostic latency/throughput/cost benchmark")
    p.add_argument("--device", default=None, help="cuda | neuron | cpu | auto (default: INFERENCE_DEVICE or auto)")
    p.add_argument("--dtype", default=None, help="float16 | bfloat16 | float32 (default per device)")
    p.add_argument("--model", default="default", choices=sorted(MODELS))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--instance", default=None, help="price preset from benchmarks/instance_prices.json")
    p.add_argument("--price-per-hour", type=float, default=None, help="instance $/hr (overrides preset)")
    p.add_argument("--target", default=None, choices=["trn1", "inf2"], help="Neuron compile target")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    cfg = MODELS[args.model]
    configure_runtime(device)
    model = load_model(cfg, dtype, device)
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), generator=gen).to(torch_device(device))
    last = torch.full((args.batch,), args.seq - 1, dtype=torch.long, device=ids.device)

    forward = build_forward(model, device, args.batch, args.seq, args.target)
    for _ in range(args.warmup):
        logits = forward(ids, last)
    stream = torch.cuda.Stream() if device == "cuda" else None
    times = []
    for _ in range(args.iters):
        with DeviceTimer(device, stream) as t:
            logits = forward(ids, last)
        times.append(t.elapsed_ms)

    mean_ms = statistics.fmean(times)
    tokens_per_s = args.batch * args.seq / (mean_ms / 1000)
    usd_hr, replicas, price_source = resolve_price(args.instance, args.price_per_hour)
    result = {
        "device": device_name(device), "backend": device, "dtype": str(dtype).replace("torch.", ""),
        "kernels": vars(model.kernels) if hasattr(model.kernels, "__dict__") else str(model.kernels),
        "model": args.model, "mode": "prefill -> next-token logits",
        "batch": args.batch, "seq": args.seq, "iters": args.iters,
        "latency_ms": {"mean": round(mean_ms, 3), "p50": round(statistics.median(times), 3),
                       "p90": round(sorted(times)[int(0.9 * (len(times) - 1))], 3)},
        "tokens_per_s": round(tokens_per_s, 1),
        "correctness": check_correctness(cfg, logits, ids, last),
    }
    if device == "cuda":
        result["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if usd_hr is not None:
        result.update({"instance": args.instance, "usd_per_hour": usd_hr, "price_source": price_source,
                       "replicas_per_instance": replicas,
                       "usd_per_1m_tokens": round(cost_per_million_tokens(usd_hr, tokens_per_s, replicas), 4)})
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
