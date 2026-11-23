# cuda-inference-service

A small, production-shaped **NVIDIA GPU** inference service for a Llama-style decoder. It is the
"before" state for a CUDA → **AWS Trainium / Inferentia (AWS Neuron)** migration demo run by Devin.

## CUDA dependencies (what a migration has to deal with)

| Area | File | CUDA-specific piece |
|---|---|---|
| Custom kernel | `csrc/fused_rmsnorm_kernel.cu`, `inference/kernels/cuda_ops.py` | Hand-written CUDA RMSNorm (warp shuffles, `cpp_extension` JIT, `sm_80/sm_90`) |
| Triton kernel | `inference/kernels/triton_rope.py` | Rotary embeddings written in OpenAI Triton |
| Attention | `inference/model.py` | `torch.backends.cuda.sdp_kernel` forcing FlashAttention |
| Runtime | `inference/engine.py` | `.cuda()`, CUDA streams, CUDA events, TF32/cuDNN flags |
| Distributed | `inference/distributed.py` | NCCL process group, `torch.cuda.set_device` |
| Benchmark | `benchmarks/bench_latency.py` | `torch.cuda.Event` timing, `max_memory_allocated` |
| Container | `Dockerfile` | `nvidia/cuda` base image |

## Run (NVIDIA GPU)

```bash
pip install -r requirements.txt
uvicorn inference.server:app --port 8000
curl -s localhost:8000/generate -H 'content-type: application/json' -d '{"input_ids":[1,2,3],"max_new_tokens":8}'
python -m benchmarks.bench_latency --batch 8 --seq 512
pytest -q   # GPU tests are skipped on machines without CUDA
```

Each custom kernel ships with a pure-PyTorch `*_reference` implementation, used as the
numerical ground truth.
