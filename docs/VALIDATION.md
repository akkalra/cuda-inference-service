# Validation report

All results below come from this branch. Nothing was run on an NVIDIA GPU or on a Trainium device.

## Environments

| Environment | Hardware | Software | Used for |
|---|---|---|---|
| Dev VM | 2 vCPU Intel Xeon, 7 GB RAM, no GPU, no Neuron device | Ubuntu 22.04, Python 3.11.18, torch 2.4.1+cpu | CPU tests and CPU benchmark |
| Compile container (`docker/Dockerfile.neuron-compile`) on the same VM | Same, no Neuron device | Ubuntu 26.04, Python 3.14, torch 2.14.0+cpu, torch-neuronx 2.14.3.0.282, neuronx-cc 2.28.7631.0, Neuron runtime from apt | Device-free Trn2 compile (`TORCH_NEURONX_COMPILE_ONLY=1`, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, `NEURON_LOGICAL_NC_CONFIG=2`, `NEURON_RT_FAKE_INSTANCE_TYPE=trn2.3xlarge`) |
| NVIDIA GPU | **Not used** | | |
| Trn2 instance | **Not used** (needs approval for paid infrastructure) | | |

## 1. CPU test suite

```bash
python -m pytest -v
```

Result: **39 passed, 7 skipped**.

- Passed: backend selection and errors (including compile-only is not a device), kernel configs,
  model shapes, parameter count (168,313,856), math attention vs SDPA, Neuron kernel config vs
  CPU kernel config, `last_pos` gather, right-padding invariance, bf16 vs fp32 closeness,
  CPU generation, static generation = dynamic generation, `static_length`, FastAPI `/healthz` and
  `/generate`, benchmark output and argument checks.
- Skipped (no hardware): 3 CUDA tests (`test_fused_rmsnorm_matches_reference`,
  `test_triton_rope_matches_reference`, `test_generate_smoke`), 3 Neuron execution tests
  (`test_neuron_forward_matches_cpu_reference[torch.compile|eager]`, `test_neuron_generate`) and
  `test_trn2_compile_only` (it runs only in the compile container).

## 2. Trainium2 compile-only (no device, no execution)

```bash
python -m pytest -v -m "neuron_compile or neuron" tests/test_neuron.py   # in the compile container
```

Result: **`test_trn2_compile_only` PASSED**, 3 Neuron execution tests skipped. TorchNeuron
compile-only mode makes `torch.neuron.is_available()` return True with a fake device;
`NeuronBackend.is_available()` returns False there, so the execution tests correctly skip.

Full default model (8 layers, hidden 1024, 168M parameters, bf16),
`torch.compile(backend="neuron", dynamic=False, fullgraph=True)`:

```bash
python -m benchmarks.neuron_compile_only --batch 1 --seq 128
python -m benchmarks.neuron_compile_only --batch 1 --seq 128 --last-pos
python -m benchmarks.neuron_compile_only --batch 8 --seq 512
```

| Graph | Batch | Seq | Output shape | neuronx-cc status | NEFF size | Graph capture | Capture + compile |
|---|---|---|---|---|---|---|---|
| forward (full logits) | 1 | 128 | [1, 128, 32000] | PASS | 918,604 B | 2.5 s | 14.6 s |
| generate (`last_pos`) | 1 | 128 | [1, 32000] | PASS | 965,517 B | 3.1 s | 19.2 s |
| forward (full logits) | 8 | 512 | [8, 512, 32000] | PASS | 4,318,714 B | 2.6 s | 39.8 s |

`fullgraph=True` means there was no graph break, so the whole decoder lowers to one graph.
Compile times are for this 2-vCPU VM; a Trn2 host compiles faster. Weights are graph inputs, so
they are not in the NEFF size.

This proves that the model lowers and compiles for Trn2. It does **not** prove that it runs
correctly or quickly on a device.

## 3. CPU benchmark (reference only)

```bash
python -m benchmarks.bench_latency --backend cpu --batch 1 --seq 128 --warmup 2 --iters 10
python -m benchmarks.bench_latency --backend cpu --dtype bfloat16 --batch 1 --seq 128 --warmup 2 --iters 10
```

| dtype | setup | first forward | mean | p50 | p99 | req/s | tokens/s | top-1 vs fp32 | cosine vs fp32 |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | 0.98 s | 1.27 s | 907 ms | 748 ms | 1443 ms | 1.1 | 141 | 1.000 | 1.000000 |
| bf16 | 1.18 s | 10.1 s | 9715 ms | 9550 ms | 10695 ms | 0.1 | 13 | 1.000 | 0.999976 |

The CPU numbers show that the harness works and that bf16 stays accurate (top-1 agreement 1.0).
They are not a baseline for Trainium or GPU performance. This CPU has no fast bf16 path.

## 4. Not validated

| Item | How to validate |
|---|---|
| Neuron execution and accuracy vs CPU | `pytest -m neuron` on `trn2.3xlarge` (docs/DEPLOY_TRN2.md) |
| Neuron latency and throughput | `bench_latency --backend neuron` on `trn2.3xlarge` |
| Neuron eager mode | `INFERENCE_NEURON_COMPILE=0 pytest -m neuron` |
| CUDA path after the refactor | `pytest -q` and `bench_latency --backend cuda` on an NVIDIA GPU host |
| Multi-device (`torchrun`) | `torchrun --nproc-per-node 2` on Trn2 or GPU |
