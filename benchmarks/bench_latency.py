"""Backend-agnostic latency / throughput / accuracy benchmark (CPU, CUDA, AWS Neuron).

Reports, for one fixed (batch, seq) shape:
  * setup_s        model build + transfer to device
  * compile_s      first forward (includes torch.compile / neuronx-cc on Neuron, cuDNN autotune on CUDA)
  * warmup_ms      latency of each warmup iteration after the first
  * latency_ms     steady-state mean / p50 / p90 / p99 over --iters synchronized forwards
  * throughput     requests/s (one request = one sequence) and tokens/s
  * accuracy       vs an fp32 CPU reference model with identical weights: max/mean abs error,
                   cosine similarity and top-1 agreement of logits at the last position

Example:
  python -m benchmarks.bench_latency --backend neuron --batch 8 --seq 512
  python -m benchmarks.bench_latency --backend cpu --preset small --batch 1 --seq 128
"""
import argparse
import json
import platform
import statistics
import time

import torch
import torch.nn.functional as F

from inference.backends import get_backend
from inference.model import ModelConfig, load_model

PRESETS = {
    "default": ModelConfig(),
    "small": ModelConfig(num_layers=2, hidden_size=256, num_heads=4, intermediate_size=512),
}
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _percentile(values, q):
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _accuracy(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.float(), expected.float()
    diff = (actual - expected).abs()
    return {
        "max_abs_err": round(diff.max().item(), 5),
        "mean_abs_err": round(diff.mean().item(), 6),
        "cosine_similarity": round(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(), 6),
        "top1_agreement": round((actual.argmax(-1) == expected.argmax(-1)).float().mean().item(), 4),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default=None, help="auto|cpu|cuda|neuron (default: $INFERENCE_BACKEND or auto)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="default")
    p.add_argument("--dtype", choices=sorted(DTYPES), default=None, help="default: the backend's dtype")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-reference", action="store_true", help="skip the fp32 CPU accuracy check")
    p.add_argument("--json-out", default=None, help="also write the result to this file")
    args = p.parse_args(argv)
    cfg = PRESETS[args.preset]
    if not 1 <= args.seq <= cfg.max_seq_len:
        p.error(f"--seq must be in [1, {cfg.max_seq_len}]")
    if args.batch < 1 or args.iters < 1 or args.warmup < 0:
        p.error("--batch and --iters must be >= 1, --warmup >= 0")
    return args


@torch.inference_mode()
def run(args) -> dict:
    backend = get_backend(args.backend)
    cfg = PRESETS[args.preset]
    dtype = DTYPES[args.dtype] if args.dtype else backend.default_dtype
    if backend.kernels.attention == "flash" and dtype == torch.float32:
        raise ValueError("FlashAttention does not support float32. Use --dtype float16 or bfloat16.")
    backend.configure_runtime()

    t0 = time.perf_counter()
    model = load_model(cfg, dtype, backend)
    forward = backend.prepare_model(model)
    backend.synchronize()
    setup_s = time.perf_counter() - t0

    torch.manual_seed(1234)
    x_cpu = torch.randint(0, cfg.vocab_size, (args.batch, args.seq))
    x = x_cpu.to(backend.device)

    with backend.timer() as t:
        out = forward(x)
    compile_s = t["elapsed_ms"] / 1000

    warmup_ms = []
    for _ in range(args.warmup):
        with backend.timer() as t:
            forward(x)
        warmup_ms.append(t["elapsed_ms"])

    latencies = []
    for _ in range(args.iters):
        with backend.timer() as t:
            out = forward(x)
        latencies.append(t["elapsed_ms"])

    mean_ms = statistics.fmean(latencies)
    result = {
        "backend": backend.name,
        "device": backend.device_name(),
        "compiled": forward is not model,
        "preset": args.preset,
        "dtype": str(dtype).replace("torch.", ""),
        "batch": args.batch,
        "seq": args.seq,
        "iters": args.iters,
        "setup_s": round(setup_s, 3),
        "compile_s": round(compile_s, 3),
        "warmup_ms": [round(v, 2) for v in warmup_ms],
        "latency_ms": {
            "mean": round(mean_ms, 2),
            "p50": round(_percentile(latencies, 0.5), 2),
            "p90": round(_percentile(latencies, 0.9), 2),
            "p99": round(_percentile(latencies, 0.99), 2),
        },
        "throughput": {
            "requests_per_s": round(args.batch / (mean_ms / 1000), 2),
            "tokens_per_s": round(args.batch * args.seq / (mean_ms / 1000), 1),
        },
        "host": {"python": platform.python_version(), "torch": torch.__version__, "machine": platform.machine()},
    }
    if backend.name == "cuda":
        result["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    if not args.no_reference:
        if backend.name == "cpu" and dtype == torch.float32:
            ref_model = model
        else:
            ref_model = load_model(cfg, torch.float32, "cpu")
        expected = ref_model(x_cpu)[:, -1]
        result["accuracy_vs_fp32_cpu"] = _accuracy(out[:, -1].cpu(), expected)
    return result


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")
    return result


if __name__ == "__main__":
    main()
