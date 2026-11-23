"""Compile the decoder for Trainium2 without a Neuron device and report compile time.

This checks that the model graph lowers to a Trn2 NEFF (torch.compile backend="neuron",
dynamic=False, fullgraph=True). It does not execute the model, so it reports no latency,
throughput or accuracy. Run it on any x86_64 Linux host with torch-neuronx + neuronx-cc::

  TORCH_NEURONX_COMPILE_ONLY=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_LOGICAL_NC_CONFIG=2 \\
  NEURON_RT_FAKE_INSTANCE_TYPE=trn2.3xlarge python -m benchmarks.neuron_compile_only --batch 1 --seq 128
"""
import argparse
import json
import os
import time

import torch

from benchmarks.bench_latency import DTYPES, PRESETS
from inference.backends import NeuronBackend
from inference.model import load_model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="default")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--last-pos", action="store_true", help="compile the generate graph (next-token logits)")
    args = p.parse_args(argv)
    if os.environ.get("TORCH_NEURONX_COMPILE_ONLY") != "1":
        p.error("TORCH_NEURONX_COMPILE_ONLY=1 is not set. Set the env vars in the module docstring.")

    import torch_neuronx.compiler

    cfg = PRESETS[args.preset]
    backend = NeuronBackend(compile=True)
    with torch_neuronx.compiler.compile_only(), torch.inference_mode():
        t0 = time.perf_counter()
        model = backend.prepare_model(load_model(cfg, DTYPES[args.dtype], backend))
        setup_s = time.perf_counter() - t0
        ids = torch.zeros(args.batch, args.seq, dtype=torch.long, device="neuron")
        extra = (torch.full((args.batch,), args.seq - 1, dtype=torch.long, device="neuron"),) if args.last_pos else ()
        t0 = time.perf_counter()
        out = model(ids, *extra)
        capture_s = time.perf_counter() - t0
        # neuronx-cc runs asynchronously; synchronize waits for the NEFF.
        torch.neuron.synchronize()
        compile_s = time.perf_counter() - t0
    artifact = out._neff_artifact
    neff_info = torch_neuronx._C.get_neff_info(artifact.cache_key)
    if not (neff_info and neff_info[1]):
        raise SystemExit(f"Compilation failed: no NEFF for cache key {artifact.cache_key}.")
    result = {
        "mode": "compile-only (no execution)",
        "target": os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"),
        "instance_type": os.environ.get("NEURON_RT_FAKE_INSTANCE_TYPE"),
        "preset": args.preset,
        "dtype": args.dtype,
        "batch": args.batch,
        "seq": args.seq,
        "graph": "generate (last_pos)" if args.last_pos else "forward (full logits)",
        "output_shape": list(out.shape),
        "neff_bytes": len(neff_info[1]),
        "setup_s": round(setup_s, 3),
        "graph_capture_s": round(capture_s, 3),
        "compile_s": round(compile_s, 3),
        "versions": {"torch": torch.__version__, "torch_neuronx": torch_neuronx.__version__},
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
