# Deploy and validate on AWS Trainium2 (`trn2.3xlarge`)

This runbook takes the service from a fresh AWS account to verified Trainium2 inference, then
deletes every billable resource. It uses the TorchNeuron single-framework DLAMI (Neuron SDK 2.33+,
PyTorch 2.14, `torch-neuronx` 2.14).

> **Cost warning:** a Trn2 instance bills per second while it is running, and its EBS volume bills
> until you delete it. Do section 8 (Clean up) when you finish.

## 1. Prerequisites

- An AWS account with Trn2 quota in the target Region. Check the **Running On-Demand Trn instances**
  quota (vCPU-based) in Service Quotas. `trn2.3xlarge` has 12 vCPUs. If it is not offered On-Demand
  in your Region/AZ, buy an **EC2 Capacity Block for ML** for `trn2.3xlarge`.
- AWS CLI v2, configured with IAM credentials (`aws sts get-caller-identity` works).
- An EC2 key pair and a security group that allows SSH (22/tcp) from your IP only.

## 2. Find the TorchNeuron DLAMI

```bash
export AWS_REGION=us-east-2   # pick a Region that offers trn2.3xlarge
aws ec2 describe-images --region $AWS_REGION --owners amazon \
  --filters "Name=name,Values=*Neuron*PyTorch*2.14*Ubuntu 24.04*" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].[ImageId,Name]' --output text
```

Alternatively, read the SSM parameter for the DLAMI listed in the Neuron docs
(<https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/dlami.html>).

## 3. Launch the instance

```bash
aws ec2 run-instances --region $AWS_REGION \
  --image-id <ami-id> --instance-type trn2.3xlarge \
  --key-name <key-pair> --security-group-ids <sg-id> --subnet-id <subnet-in-a-trn2-az> \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=trn2-migration-demo}]' \
  --query 'Instances[0].InstanceId' --output text
# For a Capacity Block add: --instance-market-options MarketType=capacity-block \
#   --capacity-reservation-specification CapacityReservationTarget={CapacityReservationId=<cr-id>}
aws ec2 wait instance-running --region $AWS_REGION --instance-ids <instance-id>
ssh -i <key>.pem ubuntu@<public-dns>
```

## 4. Verify the Neuron hardware and environment

```bash
neuron-ls                       # expect: instance-type: trn2.3xlarge, 1 device, 4 NeuronCores, 96 GB
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
python -c "import torch, torch_neuronx; print(torch.__version__, torch_neuronx.__version__, torch.neuron.is_available(), torch.neuron.device_count())"
```

## 5. Set up the repository

```bash
git clone https://github.com/akkalra/cuda-inference-service.git && cd cuda-inference-service
git checkout <this PR branch>
pip install fastapi uvicorn pydantic pytest httpx   # torch / torch-neuronx come from the DLAMI venv
export INFERENCE_BACKEND=neuron
```

## 6. Run inference, tests and benchmarks

```bash
# Tests: the CPU suite plus the hardware-gated Neuron tests (they skip on non-Neuron hosts)
python -m pytest -v
python -m pytest -v -m neuron            # Neuron-only: torch.compile + eager vs fp32 CPU reference

# Inference server
uvicorn inference.server:app --port 8000 &
curl -s localhost:8000/healthz           # {"ok":true,"backend":"neuron"}
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"input_ids":[1,2,3],"max_new_tokens":8}'

# Benchmark (compile time is reported separately as compile_s)
python -m benchmarks.bench_latency --backend neuron --batch 1 --seq 128 --json-out trn2_b1_s128.json
python -m benchmarks.bench_latency --backend neuron --batch 8 --seq 512 --json-out trn2_b8_s512.json
INFERENCE_NEURON_COMPILE=0 python -m benchmarks.bench_latency --backend neuron --batch 1 --seq 128  # eager mode

# Confirm the NeuronCores are busy while a benchmark runs (second terminal)
neuron-top
```

Copy the JSON files and the `pytest` output into the PR. They are the hardware evidence.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `BackendUnavailableError: Backend 'neuron' is not available` | The DLAMI venv is not active, or the host has no Neuron device. Run `neuron-ls`, then activate the venv. |
| `neuron-ls` lists no devices | Wrong instance type or the driver is not loaded. Check `lsmod \| grep neuron` and reinstall `aws-neuronx-dkms`. |
| First request or benchmark iteration is slow | That is neuronx-cc compiling the graph (`compile_s`). Compiled NEFFs are cached in `TORCH_NEURONX_NEFF_CACHE_DIR` (default `/tmp/neff_cache`). |
| Recompiles on every request | The input shape changed. The engine pads to a power-of-two length; keep `prompt + max_new_tokens` within one bucket, or warm each bucket once. |
| `dynamic=True` / dynamic shape errors | Not supported by the Neuron backend. The code uses `dynamic=False`. |
| A graph fails to compile | Run eager with `INFERENCE_NEURON_COMPILE=0` to isolate it, or set `TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=0` to fall back to eager per graph. Keep artifacts with `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1 TORCH_NEURONX_DEBUG_DIR=./neuron_debug`. |
| An op runs on CPU (slow) | Check the torch-neuronx operator compatibility table; `TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1` limits CPU fallback. |
| Out of device memory | Reduce `--batch`/`--seq`. A logical NeuronCore on Trn2 has 24 GB. |

## 8. Clean up (avoid charges)

```bash
aws ec2 terminate-instances --region $AWS_REGION --instance-ids <instance-id>
aws ec2 wait instance-terminated --region $AWS_REGION --instance-ids <instance-id>
aws ec2 describe-volumes --region $AWS_REGION --filters Name=status,Values=available \
  --query 'Volumes[].VolumeId'          # delete any leftover volumes: aws ec2 delete-volume --volume-id <id>
aws ec2 describe-addresses --region $AWS_REGION   # release any Elastic IP you allocated
# Delete the security group and key pair if you created them only for this test.
```

A Capacity Block bills for the whole reserved window, even after you terminate the instance.
