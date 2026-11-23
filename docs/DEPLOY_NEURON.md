# Deploying on trn1 / inf2

Instance choice: **inf2.xlarge** (1 Inferentia2, 2 NeuronCores) for low-cost serving of this
model size; **trn1.2xlarge** (1 Trainium, 2 NeuronCores) if the same fleet also trains. Both
are NeuronCore-v2, use the same NEFFs per `--target`, and use the same code path.

## 1. Compile once, ship artifacts

Compile with the **same Neuron SDK as the serving image** (NEFFs are tied to the compiler and runtime versions):

```bash
docker build -f Dockerfile.neuron -t inference-neuron .
# On any x86 host (no Neuron device needed):
docker run --rm -v $PWD/neuron_artifacts:/out inference-neuron \
  python -m inference.neuron --target inf2 --cpu-backend --out /out
# Or on the instance itself (no --cpu-backend / --target needed):
python -m inference.neuron --out ./neuron_artifacts
```

At startup the server loads `decoder_<fingerprint>_b1_s{bucket}.pt` from `NEURON_ARTIFACT_DIR`. The fingerprint covers the model config, dtype, kernel selection, compiler flags and torch-neuronx version, so serve with the same settings you compiled with. If the files are missing, it compiles them, which is slow, so don't rely on that in production.

## 2. EC2

1. Launch inf2/trn1 with a **Deep Learning AMI Neuron** (Ubuntu 22.04/24.04); it has the driver (`aws-neuronx-dkms`) and Docker.
2. Run:
   ```bash
   docker run -d -p 8080:8080 --device=/dev/neuron0 \
     -v $PWD/neuron_artifacts:/opt/ml/model/neuron_artifacts inference-neuron
   curl -s localhost:8080/generate -H 'content-type: application/json' -d '{"input_ids":[1,2,3],"max_new_tokens":8}'
   ```
3. `neuron-top` / `neuron-ls` show core utilization. To use both NeuronCores, run two containers with `NEURON_RT_VISIBLE_CORES=0` and `=1`.
4. Benchmark: `docker run --device=/dev/neuron0 inference-neuron python -m benchmarks.bench_latency --device neuron --instance inf2.xlarge --batch 1 --seq 512`.

## 3. SageMaker (bring your own container)

The image follows the SageMaker container contract: port 8080, `GET /ping`, `POST /invocations`.

1. Push `inference-neuron` to ECR.
2. Package `neuron_artifacts/` as `model.tar.gz` in S3. SageMaker extracts it to `/opt/ml/model/`, which matches the default `NEURON_ARTIFACT_DIR=/opt/ml/model/neuron_artifacts`.
3. Create a model and endpoint with instance type `ml.inf2.xlarge` (or `ml.trn1.2xlarge`) and invoke it with the same JSON body as `/generate`.
