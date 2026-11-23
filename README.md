# cuda-inference-service

A small, production-shaped inference service for a Llama-style decoder. It runs on **NVIDIA GPUs
(CUDA)**, **AWS Trainium2 (AWS Neuron / TorchNeuron)** and **CPU** behind one backend abstraction
(`inference/backends.py`). See [MIGRATION.md](MIGRATION.md) for the CUDA to Trainium2 migration
and [docs/DEPLOY_TRN2.md](docs/DEPLOY_TRN2.md) for the `trn2.3xlarge` runbook.

## Original CUDA dependencies

| Area | File | CUDA-specific piece |
|---|---|---|
| Custom kernel | `csrc/fused_rmsnorm_kernel.cu`, `inference/kernels/cuda_ops.py` | Hand-written CUDA RMSNorm (warp shuffles, `cpp_extension` JIT, `sm_80/sm_90`) |
| Triton kernel | `inference/kernels/triton_rope.py` | Rotary embeddings written in OpenAI Triton |
| Attention | `inference/model.py` | `torch.backends.cuda.sdp_kernel` forcing FlashAttention |
| Runtime | `inference/engine.py` | `.cuda()`, CUDA streams, CUDA events, TF32/cuDNN flags |
| Distributed | `inference/distributed.py` | NCCL process group, `torch.cuda.set_device` |
| Benchmark | `benchmarks/bench_latency.py` | `torch.cuda.Event` timing, `max_memory_allocated` |
| Container | `Dockerfile` | `nvidia/cuda` base image |

## Backends

| Backend | Selected by | dtype | RMSNorm | RoPE | Attention | Execution |
|---|---|---|---|---|---|---|
| `cuda` | `INFERENCE_BACKEND=cuda` | fp16 | CUDA C++ kernel | Triton | FlashAttention (SDPA) | eager, CUDA streams |
| `neuron` | `INFERENCE_BACKEND=neuron` | bf16 | PyTorch reference | PyTorch reference | explicit math | `torch.compile(backend="neuron", dynamic=False)`, eager with `INFERENCE_NEURON_COMPILE=0` |
| `cpu` | `INFERENCE_BACKEND=cpu` | fp32 | PyTorch reference | PyTorch reference | SDPA | eager |

`auto` (the default) picks `cuda`, then `neuron`, then `cpu`. Asking for an unavailable backend
raises `BackendUnavailableError`; it never silently falls back.

## Run

```bash
# NVIDIA GPU (unchanged)
pip install -r requirements.txt
# AWS Trainium2: use the TorchNeuron DLAMI venv, see docs/DEPLOY_TRN2.md
# CPU only
pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

uvicorn inference.server:app --port 8000
curl -s localhost:8000/healthz
curl -s localhost:8000/generate -H 'content-type: application/json' -d '{"input_ids":[1,2,3],"max_new_tokens":8}'
python -m benchmarks.bench_latency --backend <cpu|cuda|neuron> --batch 8 --seq 512
python -m pytest -q   # CUDA and Neuron tests skip on hosts without that hardware
```

Each custom kernel ships with a pure-PyTorch reference implementation
(`inference/kernels/reference.py`). It is the numerical ground truth and the CPU/Neuron kernel.
