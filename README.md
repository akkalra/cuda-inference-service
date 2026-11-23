# cuda-inference-service

A small, production-shaped inference service for a Llama-style decoder. It runs on **NVIDIA GPUs (CUDA)**,
**AWS Trainium / Inferentia2 (AWS Neuron)**, or **CPU**, selected with `INFERENCE_DEVICE=cuda|neuron|cpu|auto`.
See [MIGRATION.md](MIGRATION.md) for the CUDA → Neuron port: inventory, validation, open items.

## CUDA dependencies (original; each now has a Neuron equivalent, see MIGRATION.md)

| Area | File | CUDA-specific piece |
|---|---|---|
| Custom kernel | `csrc/fused_rmsnorm_kernel.cu`, `inference/kernels/cuda_ops.py` | Hand-written CUDA RMSNorm (warp shuffles, `cpp_extension` JIT, `sm_80/sm_90`) |
| Triton kernel | `inference/kernels/triton_rope.py` | Rotary embeddings written in OpenAI Triton |
| Attention | `inference/model.py` | `torch.backends.cuda.sdp_kernel` forcing FlashAttention |
| Runtime | `inference/engine.py` | `.cuda()`, CUDA streams, CUDA events, TF32/cuDNN flags |
| Distributed | `inference/distributed.py` | NCCL process group, `torch.cuda.set_device` |
| Benchmark | `benchmarks/bench_latency.py` | `torch.cuda.Event` timing, `max_memory_allocated` |
| Container | `Dockerfile` | `nvidia/cuda` base image |

## Run on AWS Trainium / Inferentia2

```bash
pip install --extra-index-url https://pip.repos.neuron.amazonaws.com -r requirements-neuron.txt
python -m inference.neuron --out ./neuron_artifacts            # compile seq buckets (on the instance)
INFERENCE_DEVICE=neuron NEURON_ARTIFACT_DIR=./neuron_artifacts uvicorn inference.server:app --port 8000
python -m benchmarks.bench_latency --device neuron --instance inf2.xlarge
```

Container and EC2/SageMaker steps: [`Dockerfile.neuron`](Dockerfile.neuron), [docs/DEPLOY_NEURON.md](docs/DEPLOY_NEURON.md).

## Run on CPU (reference path, what CI runs)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-base.txt -r requirements-dev.txt
pytest -q                                   # NKI-simulator / Neuron-compile tests skip without the Neuron toolchain
python -m benchmarks.bench_latency --device cpu --model tiny --price-per-hour 0.17
```

## Run (NVIDIA GPU)

```bash
pip install -r requirements.txt
uvicorn inference.server:app --port 8000
curl -s localhost:8000/generate -H 'content-type: application/json' -d '{"input_ids":[1,2,3],"max_new_tokens":8}'
python -m benchmarks.bench_latency --device cuda --batch 8 --seq 512 --instance g5.xlarge
pytest -q   # GPU tests are skipped on machines without CUDA
```

Each custom kernel ships with a pure-PyTorch `*_reference` implementation, used as the
numerical ground truth.
