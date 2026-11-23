# CUDA → AWS Neuron (Trainium / Inferentia2) migration

This port adds an AWS Neuron backend for trn1 / inf2 (NeuronCore-v2). The CUDA path stays as
it was, and a pure-PyTorch CPU path is now the reference. The backend is chosen with
`INFERENCE_DEVICE=neuron|cuda|cpu|auto`.

**Validation scope.** The checks were run without Neuron hardware. Correctness was checked on CPU,
the NKI kernels were checked on the NKI CPU simulator, and the full model was compiled to NEFF
with `torch_neuronx.trace(cpu_backend=True)` for `--target trn1` and `--target inf2`. No
Trainium/Inferentia performance numbers appear anywhere in this repo, and none were measured.

## 1. Dependency inventory

| File | CUDA construct | Neuron replacement | Status |
|---|---|---|---|
| `csrc/fused_rmsnorm_kernel.cu`, `inference/kernels/cuda_ops.py` | Hand-written CUDA RMSNorm (warp shuffles, `cpp_extension` JIT, sm_80/90) | `inference/kernels/nki_ops.py::rmsnorm_kernel`: NKI kernel. Rows go on the 128 SBUF partitions; square and row-sum are fused into one `nisa.activation(reduce_op=add)`; `rsqrt` runs on the Scalar Engine. This is the default on Neuron. | ✅ Simulator parity vs `rmsnorm_reference` (fp32/bf16, row counts that are and are not multiples of 128). ✅ Compiles inside the traced model. ⏳ Perf on trn1/inf2 not measured |
| `inference/kernels/triton_rope.py` | OpenAI Triton RoPE kernel | `nki_ops.py::rope_kernel`: NKI kernel. Sequence is on partitions and cos/sin are loaded once per 128-token block. The **default on Neuron is native ops** (`apply_rope_reference`); see §4. Opt in with `INFERENCE_ROPE_IMPL=nki`. | ✅ Simulator parity vs `apply_rope_reference`. ✅ Compiles in the traced model (both impls). ⏳ Which one is faster must be profiled on hardware |
| `inference/model.py` | `torch.backends.cuda.sdp_kernel(enable_flash=True, …)` | `causal_attention_math`: explicit matmul → causal mask → fp32 softmax → matmul. Every op is static-shape and lowers directly to XLA, so neuronx-cc fuses it. The CUDA path still forces FlashAttention. | ✅ Matches SDPA on CPU (fp32/bf16, op and full-model level). ✅ Compiles |
| `inference/model.py` | `.to("cuda")`, `rope_tables(device="cuda")` | `load_model(device=…)`. On Neuron the eager model stays on the host and is compiled ahead of time. | ✅ |
| `inference/engine.py` | `torch.cuda.Stream`, `torch.cuda.Event` timing | `inference/device.py::DeviceTimer`: CUDA events on CUDA; host wall time around synchronous Neuron graph calls elsewhere. | ✅ CPU. ⏳ Neuron timing on hardware |
| `inference/engine.py` | TF32 / `cudnn.benchmark` flags | `configure_runtime()`. These flags apply on CUDA only. On Neuron the equivalents are compiler flags `--auto-cast=none` (weights already bf16) and `--model-type=transformer` in `inference/neuron.py`. | ✅ |
| `inference/engine.py` (implicit) | Dynamic shapes: every decode step runs a new sequence length | `inference/neuron.py::BucketedDecoder`. `torch_neuronx.trace` builds one graph per sequence bucket (`NEURON_SEQ_BUCKETS`, default 128…2048). Requests are right-padded and `last_pos` selects the real next-token logits. Graphs are saved to `NEURON_ARTIFACT_DIR`. | ✅ The invariant "padding doesn't change logits" and identical greedy tokens across bucket boundaries are tested on CPU. ✅ Compile and save tested. ⏳ `torch.jit.load` and execution of the NEFF need a device |
| `inference/distributed.py` | NCCL process group, `torch.cuda.set_device` | `xla` process group (`torch_xla.distributed.xla_backend`, `init_method="xla://"`) on Neuron, `gloo` on CPU, NCCL unchanged | ⏳ Needs a multi-core trn1 to verify (see §4) |
| `benchmarks/bench_latency.py` | `torch.cuda.Event`, `max_memory_allocated` | Device-agnostic harness. It reports latency (mean/p50/p90), tokens/s, correctness vs the fp32 CPU reference model, and $/1M tokens from `--instance` presets or `--price-per-hour`. | ✅ CPU. ⏳ Needs real numbers from trn1/inf2/g5 |
| `Dockerfile` | `nvidia/cuda` base | Kept for CUDA. New `Dockerfile.neuron` on `public.ecr.aws/neuron/pytorch-inference-neuronx:2.9.0-neuronx-py312-sdk2.31.1-ubuntu24.04`, port 8080, with SageMaker `/ping` and `/invocations` routes | ⏳ Image build and run on an instance |
| `requirements.txt` | `torch==2.4.1`, `triton` | Split into `requirements-base.txt` (service), `requirements.txt` (CUDA, unchanged pins), and `requirements-neuron.txt` (torch-neuronx 2.9.0.2.15, neuronx-cc 2.28, nki 0.7) | ✅ CI installs both stacks |

## 2. What was validated, and how

All of the following runs in GitHub Actions (`.github/workflows/ci.yml`) on `ubuntu-22.04` x86 runners with no accelerator:

| Check | Where | Evidence |
|---|---|---|
| Unit and integration tests (device selection, reference kernels, math attention vs SDPA, bucketing invariants, engine `generate`, HTTP routes, benchmark cost math, end-to-end benchmark incl. correctness) | `cpu-tests` job, torch 2.4.1 (CUDA pin) and torch 2.9.1 (Neuron pin) | `tests/test_cpu.py` |
| NKI RMSNorm and RoPE vs `*_reference` on the **NKI CPU simulator** (`nki.simulate`, trn1 target), fp32 and bf16, 6 shapes each, including partial 128-row tiles | `neuron-toolchain` job | `tests/test_nki_simulator.py`. Tolerances: fp32 1e-5, bf16 2e-2 |
| Full decoder (NKI RMSNorm custom call + math attention + `(ids, last_pos)` static signature) **compiles to NEFF for trn1 and inf2** with `torch_neuronx.trace(cpu_backend=True)` | `neuron-toolchain` job | `tests/test_neuron_compile.py`, which asserts a `.neff` is produced |

The NKI kernels were also run locally against **nki 0.5.0** (the version bundled in the 2.31.1
DLC), and all simulator tests passed. So the kernels do not depend on the newest NKI release.

The answer to the question "can the NKI packages be installed on x86 CPU?" is **yes**. `neuronx-cc`, `nki`, and
`torch-neuronx` install from `pip.repos.neuron.amazonaws.com` on Python ≥ 3.11. Importing
`torch_xla` also needs the `aws-neuronx-runtime-lib` apt package (for `libnrt.so`); no driver or
device is required.

## 3. Requires a real trn1 / inf2 instance to verify

1. **Execution and numerics on silicon.** Load the saved NEFFs (`torch.jit.load` initializes the Neuron runtime and fails without a device), run `generate`, and compare tokens with the CPU reference (`bench_latency` reports `correctness`).
2. **Performance.** Latency, throughput and $/1M tokens with `python -m benchmarks.bench_latency --device neuron --instance inf2.xlarge` (or `trn1.2xlarge`), compared with the same command on `g5.xlarge --device cuda`.
3. **Kernel choices.** Profile NKI vs native RMSNorm, and native vs NKI RoPE (`INFERENCE_RMSNORM_IMPL` / `INFERENCE_ROPE_IMPL`), with Neuron Explorer / `neuron-profile`.
4. **Compile time and memory** for the default model across all 5 buckets. Pick buckets from the real prompt-length distribution.
5. **Distributed:** `xla` process-group init and `all_reduce` under `torchrun` on a multi-core trn1.
6. **Container:** build `Dockerfile.neuron` and run it with `--device=/dev/neuron0` (EC2) or as a SageMaker endpoint (`docs/DEPLOY_NEURON.md`).

## 4. Risks and limitations

- **Platform lifecycle.** trn1/inf2 inference uses torch-neuronx 2.9 (PyTorch/XLA) and `torch_neuronx.trace`. Neuron docs list Torch/XLA on Inf2/Trn1 as maintenance mode from Neuron 2.31. The newer native TorchNeuron backend and NxD Inference target trn2/trn3, and the `pytorch-inference-neuronx` DLC was last published for SDK 2.31.x. Moving to trn2 means re-targeting to NxD Inference or vLLM-Neuron rather than reusing this trace flow.
- **No KV cache (inherited).** Like the original service, every decode step re-runs the full prefix. Static shapes make this more expensive on Neuron (the cost is the padded bucket length, not the real length). A production LLM on Neuron should use NxD Inference (KV cache, continuous batching, on-device sampling). This PR is a like-for-like port, not that rewrite.
- **Bucketing trade-offs.** Each bucket is a separate compile and NEFF, and padding wastes compute up to the next bucket. Batch size is fixed at compile time (the service uses batch 1).
- **Kernel portability.** The NKI kernels are written for NeuronCore-v2 with LNC=1. They are simple and will run on later cores, but they are not tuned for them. The RMSNorm kernel handles any row count but keeps a full `[128, H]` fp32 row tile in SBUF, which is fine for this model (H=1024) but needs H-tiling for very wide models. The CUDA kernel was fp16; Neuron defaults to bf16, so numerics differ from the GPU path within bf16 tolerance.
- **Why RoPE defaults to native ops.** RoPE is elementwise. neuronx-cc can fuse it with the QKV split and transpose, while an NKI custom call is a fusion boundary that forces q/k through HBM. The NKI version is kept, simulator-validated and switchable, so the decision can be made from a profile rather than an assumption.
- **Tensor parallelism.** The original code only all-reduces logits; the weights are never sharded. Real TP on Neuron should use `neuronx-distributed` parallel layers so collectives are compiled into the graph. The current `xla` all-reduce copies host logits to the device for each step.
- **Cost estimates** use AWS on-demand us-east-1 list prices (`benchmarks/instance_prices.json`, from AWS's public pricing data). They are only as good as the measured throughput, and Savings Plans or Spot change them materially.
