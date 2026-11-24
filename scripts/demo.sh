#!/usr/bin/env bash
# Terminal walkthrough of the CUDA -> AWS Neuron migration.
#
#   ./scripts/demo.sh            # pause between steps (live demo; press Enter to continue)
#   DEMO_PAUSE=0 ./scripts/demo.sh   # run straight through (recording / CI)
#
# Steps 1-3 and 6 run anywhere with the CPU requirements installed (Mac or Linux).
# Steps 4-5 need the Neuron toolchain, which only installs on x86-64 Linux; they are
# skipped elsewhere. Override interpreters with PYTHON=... and NEURON_PYTHON=...
set -euo pipefail
cd "$(dirname "$0")/.."

# Resolve to absolute paths up front so later PATH changes cannot swap interpreters.
PYTHON="$(command -v "${PYTHON:-python3}")"
NEURON_PYTHON="$(command -v "${NEURON_PYTHON:-$HOME/.venvs/neuron/bin/python}" || true)"
PAUSE="${DEMO_PAUSE:-1}"
export INFERENCE_DEVICE=cpu PYTHONWARNINGS=ignore

bold=$(tput bold 2>/dev/null || true); dim=$(tput dim 2>/dev/null || true)
cyan=$(tput setaf 6 2>/dev/null || true); green=$(tput setaf 2 2>/dev/null || true)
yellow=$(tput setaf 3 2>/dev/null || true); reset=$(tput sgr0 2>/dev/null || true)

step() {
  echo; echo "${bold}${cyan}== $1 ==${reset}"; echo "${dim}$2${reset}"
  if [ "$PAUSE" = "1" ]; then read -r -p "${dim}[Enter]${reset} " _; fi
}
show() {  # print the command with interpreter paths shortened for readability
  local cmd="$*"
  cmd="${cmd//"$PYTHON"/python}"
  [ -n "$NEURON_PYTHON" ] && cmd="${cmd//"$NEURON_PYTHON"/python[neuron]}"
  echo "${green}\$ ${cmd}${reset}" >&2
}
run() { show "$@"; "$@"; }

step "1. Before: the CUDA-only pieces in main" \
     "Every NVIDIA-specific construct the original service depended on."
run git --no-pager grep -n -E "torch\.cuda|nccl|import triton|sdp_kernel|cpp_extension|nvidia/cuda" \
    origin/main -- inference benchmarks Dockerfile | sed 's/^origin\/main://' | cut -c1-120

step "2. After: one codebase, three backends" \
     "INFERENCE_DEVICE picks cuda, neuron or cpu; each gets its own kernels and attention."
run "$PYTHON" - <<'PY'
from inference.kernels.dispatch import KernelConfig
for d in ("cuda", "neuron", "cpu"):
    k = KernelConfig.for_device(d)
    print(f"  {d:7s} rmsnorm={k.rmsnorm:9s} rope={k.rope:9s} attention={k.attention}")
PY

step "3. CPU test suite (what every PR runs in GitHub Actions)" \
     "Device selection, reference kernels, attention parity, bucketing, engine, HTTP API, benchmark."
run "$PYTHON" -m pytest -q tests/test_cpu.py

if [ -n "$NEURON_PYTHON" ] && [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ]; then
  (  # subshell: the Neuron env settings below must not leak into later steps
  NEURON_BIN="$(dirname "$NEURON_PYTHON")"
  export PATH="$NEURON_BIN:$PATH" NEURON_PLATFORM_TARGET_OVERRIDE=trn1 NEURON_LOGICAL_NC_CONFIG=1
  LD_LIBRARY_PATH="$("$NEURON_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH

  step "4. NKI kernels vs. the PyTorch references, on the NKI CPU simulator" \
       "RMSNorm and RoPE NKI kernels, fp32 + bf16, simulating a trn1 NeuronCore. No hardware."
  run "$NEURON_PYTHON" -m pytest -v -p no:warnings tests/test_nki_simulator.py

  step "5. Compile the whole model for Trainium and Inferentia2" \
       "torch_neuronx.trace(cpu_backend=True) -> a NEFF binary for --target trn1 and inf2."
  run "$NEURON_PYTHON" -m pytest -v -p no:warnings tests/test_neuron_compile.py
  )
else
  step "4-5. NKI simulator + Neuron compile: skipped" \
       "${yellow}The Neuron toolchain installs only on x86-64 Linux. See the 'Neuron toolchain' job in CI.${reset}"
fi

step "6. Benchmark: latency, throughput, correctness, \$ per 1M tokens" \
     "Same command on every backend; here on CPU with a tiny model and a c7i.xlarge price."
run "$PYTHON" -m benchmarks.bench_latency --device cpu --model tiny --batch 4 --seq 128 \
    --iters 10 --instance c7i.xlarge | "$PYTHON" -m json.tool

echo; echo "${bold}${green}Done.${reset} On trn1/inf2 the same benchmark runs with: --device neuron --instance inf2.xlarge"
echo "${dim}What still needs real hardware: MIGRATION.md, section 3.${reset}"
