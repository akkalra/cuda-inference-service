# CUDA to AWS Trainium2 migration (TorchNeuron)

This change adds an AWS Trainium2 execution path to the inference service. The CUDA path and the
model architecture stay the same. All hardware-specific code is now behind one backend interface in
`inference/backends.py`, with three backends: `cpu`, `cuda` and `neuron`.

## 1. Original CUDA dependency inventory

| # | Location | CUDA-specific dependency | Can it run on Trainium? | Migration |
|---|---|---|---|---|
| 1 | `csrc/fused_rmsnorm_kernel.cu`, `inference/kernels/cuda_ops.py` | Custom CUDA C++ RMSNorm (warp shuffles), built at import with `torch.utils.cpp_extension.load` for `sm_80/sm_90` | No | Neuron uses the existing PyTorch reference RMSNorm. neuronx-cc fuses it. `cpp_extension` is now imported only when the kernel is used. |
| 2 | `inference/kernels/triton_rope.py` | OpenAI Triton RoPE kernel | No (Triton targets GPUs) | Neuron uses the existing PyTorch reference RoPE. Triton is imported only by the CUDA backend. |
| 3 | `inference/model.py` | `torch.backends.cuda.sdp_kernel(enable_flash=True, ...)` forces FlashAttention | No | Neuron uses explicit causal attention (matmul, mask, fp32 softmax, matmul). CUDA keeps forced FlashAttention. |
| 4 | `inference/model.py`, `inference/engine.py` | Hard-coded `device="cuda"`, `.cuda()`, fp16 | No | The backend gives the device and dtype: `cuda`/fp16, `neuron`/bf16, `cpu`/fp32. |
| 5 | `inference/engine.py` | `torch.cuda.Stream`, `torch.cuda.Event` timing, TF32 and cuDNN flags | No | `Backend.stream()`, `Backend.timer()` and `Backend.configure_runtime()`. CUDA keeps streams, events and flags. |
| 6 | `inference/engine.py` | The token loop grows the sequence by one each step (a new shape per step) | Works, but forces one Neuron compile per length | The Neuron path right-pads to a power-of-two bucket and gathers logits at `last_pos`. One compiled graph serves every step in a bucket. |
| 7 | `inference/distributed.py` | `init_process_group("nccl")`, `torch.cuda.set_device` | No | `nccl` / `neuron` / `gloo` and the matching `set_device`, chosen by the backend. |
| 8 | `benchmarks/bench_latency.py` | CUDA events, `max_memory_allocated`, CUDA only | No | Backend-agnostic benchmark (section 5). |
| 9 | `tests/test_kernels.py` | All tests need a GPU | n/a | Kept as is. New CPU, backend, benchmark and Neuron tests were added. |
| 10 | `Dockerfile`, `requirements.txt` | `nvidia/cuda` base image, `triton` | No | Kept for CUDA. Added `requirements-neuron.txt`, `requirements-cpu.txt` and `docker/Dockerfile.neuron-compile`. The Trn2 runtime is the TorchNeuron DLAMI. |

## 2. Architecture

```
            INFERENCE_BACKEND=auto|cpu|cuda|neuron
                          |
                 get_backend()  (inference/backends.py)
        +-----------------+-------------------+
   CUDABackend        NeuronBackend         CPUBackend
   fp16               bf16                  fp32
   kernels:           kernels:              kernels:
    rmsnorm=cuda       rmsnorm=reference     rmsnorm=reference
    rope=triton        rope=reference        rope=reference
    attention=flash    attention=math        attention=sdpa
   eager + streams    torch.compile(         eager
                        backend="neuron",
                        dynamic=False,
                        fullgraph=True)
                      static shapes
        +-----------------+-------------------+
                          |
   Decoder(cfg, KernelConfig)   InferenceEngine   init_distributed   bench_latency
   (same layers, same weights, same parameter count: 168,313,856)
```

Design rules:

- **One place for hardware code.** The model reads only a `KernelConfig`. The engine, distributed
  init and benchmark call `Backend` methods. Nothing outside `backends.py` checks the device type.
- **No silent fallback.** `auto` selects `cuda`, then `neuron`, then `cpu`. An explicitly requested
  backend that is not present raises `BackendUnavailableError`. A misconfigured Trn2 host fails at
  startup and does not run on CPU.
- **Lazy CUDA imports.** CPU and Neuron hosts do not need the CUDA toolkit, `setuptools` or Triton.
- **Compile with eager fallback.** `INFERENCE_NEURON_COMPILE=0` runs the Neuron backend in
  TorchNeuron eager mode. Use it to isolate a graph that does not compile.
- **Compile-only mode is not a device.** `NeuronBackend.is_available()` returns False in
  TorchNeuron compile-only mode, so hardware tests do not run against a fake device.

## 3. Implementation notes

- **BF16 on Trainium.** The Trn2 matrix engines are fastest in BF16. RMSNorm and softmax compute
  in fp32 inside the kernels, as in the original reference code.
- **Explicit attention on Neuron.** The math attention is numerically equal to SDPA with
  `is_causal=True` (tested on CPU). It uses only `matmul`, `masked_fill` and `softmax`.
- **Static shapes.** TorchNeuron does not support dynamic shapes. `static_length()` picks a
  power-of-two length from 64 up to `max_seq_len`. Causal attention means right-padding cannot
  change earlier positions, so static generation gives the same tokens as the original loop
  (tested on CPU).
- **`Decoder.forward(input_ids, last_pos=None)`.** With `last_pos`, the model returns `[b, vocab]`
  next-token logits through `torch.gather`. This avoids a data-dependent slice in the graph.
  Without it, the output is the same `[b, s, vocab]` as before.
- **API compatibility.** `/generate` keeps all response fields (`gpu_ms` stays on CUDA) and adds
  `backend` and `device_ms`. `/healthz` adds `backend`.

## 4. Preservation of existing behavior

| Behavior | Status |
|---|---|
| CUDA kernels, FlashAttention, fp16, streams, events, TF32/cuDNN flags, NCCL | Same code paths, selected by `CUDABackend`. **Not executed in this change: no GPU was available.** |
| Model architecture and parameter count | Unchanged (tested: 168,313,856 parameters). |
| Reference kernels (numerical ground truth) | Moved to `inference/kernels/reference.py`. They are still re-exported from the original modules. |
| `tests/test_kernels.py` (CUDA kernel tests) | Unchanged. Skipped without a GPU. |
| HTTP API | Backward-compatible additions only. |

## 5. Benchmark

`python -m benchmarks.bench_latency --backend <cpu|cuda|neuron> --batch B --seq S` reports, for one
fixed shape:

| Field | Meaning |
|---|---|
| `setup_s` | Build the model and move it to the device. |
| `compile_s` | The first forward. On Neuron this includes `torch.compile` and neuronx-cc. |
| `warmup_ms` | Each warmup iteration after the first. |
| `latency_ms` | Steady state, synchronized: mean, p50, p90, p99 over `--iters`. |
| `throughput` | Requests/s (1 request = 1 sequence) and tokens/s. |
| `accuracy_vs_fp32_cpu` | Last-position logits against an fp32 CPU model with the same weights: max/mean absolute error, cosine similarity, top-1 agreement. |

`python -m benchmarks.neuron_compile_only` compiles for Trn2 without a device. It reports compile
time and NEFF size only. It does not measure performance.

## 6. Validation status

See [docs/VALIDATION.md](docs/VALIDATION.md) for the commands and full results.

| Item | Implemented | Tested here | Validated on hardware |
|---|---|---|---|
| CPU backend (fp32, bf16) | Yes | Yes: unit tests, server tests, benchmark | x86_64 CPU only |
| Backend selection and errors | Yes | Yes | n/a |
| Static-shape generation | Yes | Yes, on CPU (same tokens as the dynamic loop) | No |
| Neuron graph lowers to a Trn2 NEFF (`torch.compile`, `fullgraph=True`) | Yes | Yes, compile-only, no device | **No** |
| Neuron execution, accuracy vs CPU, latency, throughput | Yes | **No** | **No: needs a Trn2 instance** |
| Neuron eager mode (`INFERENCE_NEURON_COMPILE=0`) | Yes | No | No |
| CUDA backend | Yes (original code) | No | **No: no GPU** |
| Multi-device (`torchrun`, `neuron` process group) | Yes (wiring only) | Backend name only | No |

## 7. Limitations and remaining work

- **No Trainium execution yet.** The NEFFs compile for Trn2, but no Trainium2 device ran them. Run
  `docs/DEPLOY_TRN2.md` on a `trn2.3xlarge` to get accuracy and performance numbers.
- **CUDA was not re-run.** The CUDA logic is the original code behind a selector. Run
  `pytest -q` and the benchmark on a GPU host before you merge.
- **No KV cache.** Like the original, each step recomputes the full sequence. On Neuron each step
  uses the padded bucket length. That is correct but does more work for short prompts. A KV-cache
  decode graph is the next optimization.
- **Generic kernels on Neuron.** RMSNorm, RoPE and attention use PyTorch ops that neuronx-cc
  compiles. NKI kernels (for example NKI flash attention) can replace them after a profile on
  hardware shows that they matter.
- **One NeuronCore.** Tensor parallelism across NeuronCores is wired (`neuron` process group) but
  not tested. The model fits on one logical NeuronCore.
- **Synthetic model.** Random weights and a small decoder. The numbers show that the path works.
  They do not predict production cost or performance.
