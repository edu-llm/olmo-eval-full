#!/usr/bin/env bash
# Run the five-judge reliability validation suite on ONE dedicated GPU
# (default GPU 2) on the shared B200 node. Nothing here runs until invoked.
#
# It wraps the frozen study runner (run_judge_suite.sh -> run_judge_validation.py)
# without touching any checksummed study artifact. It only supplies environment
# the shared node needs: a single-GPU pin, the venv, the HF token for the gated
# Gemma judge, and the same attention-backend / CUDA settings that the MCQ sweep
# required on this Blackwell node.
#
# Each judge runs its six waves sequentially and checkpoints every wave to S3.
# Judges themselves are run one after another because they share the single GPU.
#
# Env knobs (all optional except S3_ROOT):
#   GPU=2                 single GPU index to pin (leaves training GPUs alone)
#   S3_ROOT=s3://.../blinded   blinded-output root (or pass as arg 1)
#   JUDGES="selene flow prometheus qwen gemma"   subset / order of judges
#   LIMIT=                smoke-test only N cases per wave (unset = full 261)
#   GMU=0.90              gpu_memory_utilization for the (dedicated) GPU
#   VENV=<path>           override venv (default: shared adaptive-inference venv)
#   BUNDLE=<path>         override bundle dir
set -euo pipefail

ROOT="/opt/dlami/nvme/adaptive-inference"
VENV="${VENV:-$ROOT/venv}"
BUNDLE="${BUNDLE:-$ROOT/judge/aws_judge_handoff}"
export HF_HOME="${HF_HOME:-$ROOT/hf-cache}"

GPU="${GPU:-2}"
S3_ROOT="${S3_ROOT:-${1:-}}"
JUDGES="${JUDGES:-selene flow prometheus qwen gemma}"
GMU="${GMU:-0.90}"

if [[ -z "${S3_ROOT}" ]]; then
  echo "error: S3_ROOT is required (set env S3_ROOT or pass as first argument)" >&2
  echo "usage: [GPU=2] [LIMIT=3] bash aws_run_gpu.sh s3://BUCKET/edu-judge-validation/v2/blinded" >&2
  exit 2
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "error: venv not found at ${VENV}" >&2
  exit 1
fi
if [[ ! -f "${BUNDLE}/run_judge_suite.sh" ]]; then
  echo "error: bundle not found at ${BUNDLE}" >&2
  exit 1
fi

# Hard single-GPU pin: never touches the training GPUs.
export CUDA_VISIBLE_DEVICES="${GPU}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# CUDA toolkit ships in the pip wheels; some vLLM JIT paths need nvcc on PATH.
NVCC_BIN="$(find "${VENV}/lib" -name nvcc -type f 2>/dev/null | head -1)"
if [[ -n "${NVCC_BIN}" ]]; then
  export CUDA_HOME="$(dirname "$(dirname "${NVCC_BIN}")")"
  export PATH="$(dirname "${NVCC_BIN}"):${PATH}"
fi
# Use prebuilt FlashAttention kernels; avoid the FlashInfer JIT path that failed
# to compile against the pip CUDA headers on this node. vLLM 0.26 has no env var
# for the attention backend, so a PYTHONPATH sitecustomize shim injects it as the
# LLM() kwarg (without touching any checksummed study file).
export JUDGE_ATTENTION_BACKEND="${JUDGE_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTHONPATH="${BUNDLE}/shim${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_FLASHINFER_SAMPLER=0

# HF token for the gated Gemma judge; read from the instance role. Never printed.
if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$(python - <<'PY' 2>/dev/null || true
import boto3, sys
try:
    v = boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(
        SecretId="hf-token")["SecretString"]
    sys.stdout.write(v.strip())
except Exception:
    pass
PY
)"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  echo "HF token loaded (len ${#HF_TOKEN})"
else
  echo "WARN: no HF token; the gated Gemma judge will fail to download" >&2
fi

cd "${BUNDLE}"
TS="$(date -u +%Y%m%d-%H%M%S)"
LOGDIR="${ROOT}/logs/judge_${TS}"
mkdir -p "${LOGDIR}"

# Pass-through options the frozen launcher permits (it blocks the study-critical
# ones like --seed/--temperature/--prompt-variant itself).
EXTRA=(--gpu-memory-utilization "${GMU}")
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "${LIMIT}")

echo "gpu=${GPU} judges=[${JUDGES}] s3=${S3_ROOT} gmu=${GMU} limit=${LIMIT:-none}"
echo "logs -> ${LOGDIR}"

rc=0
for j in ${JUDGES}; do
  echo "=== judge ${j}: six waves -> ${LOGDIR}/${j}.log ==="
  t0=${SECONDS}
  if bash run_judge_suite.sh "${j}" "${S3_ROOT}" "${EXTRA[@]}" > "${LOGDIR}/${j}.log" 2>&1; then
    echo "=== judge ${j}: done in $((SECONDS - t0))s ==="
  else
    rc=1
    echo "=== judge ${j}: FAILED after $((SECONDS - t0))s (see ${LOGDIR}/${j}.log) ==="
  fi
done

echo "ALL JUDGES COMPLETE (rc=${rc}). logs in ${LOGDIR}"
exit "${rc}"
