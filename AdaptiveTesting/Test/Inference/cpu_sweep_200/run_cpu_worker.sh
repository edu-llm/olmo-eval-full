#!/usr/bin/env bash
# One machine / one shard of the 200-model CPU sweep.
#
# Required env:
#   SHARD_INDEX   0-based shard id for this machine
#   NUM_SHARDS    total machines (round-robin over models_200.yaml)
#
# Optional env:
#   S3_URI              s3://bucket/prefix  (continuous sync if set)
#   S3_SYNC_SECONDS     default 120
#   MAX_SAMPLES         default 2000
#   PROBE_QUESTIONS     default 100
#   PROBE_MAX_SECONDS   default 900  (skip model if 100Q takes >15min on CPU)
#   MODELS_YAML         default AdaptiveTesting/Inputs/Models/models_200.yaml
#   HF_TOKEN            for gated models (Llama/Mistral/Gemma/Pedagogy);
#                       falls back to ${ROOT}/.hf_token, then AdaptiveTesting/.env
#   ROOT                working root (code + outputs + hf cache)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INF_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ADAPTIVE="$(cd "${INF_DIR}/../.." && pwd)"
ROOT="${ROOT:-${ADAPTIVE}}"
CODE="${INF_DIR}"
MODELS_YAML="${MODELS_YAML:-${ADAPTIVE}/Inputs/Models/models_200.yaml}"
CFG="${CODE}/configs/inference.cpu.yaml"
VENV_PY="${VENV_PY:-python3}"

SHARD_INDEX="${SHARD_INDEX:?set SHARD_INDEX}"
NUM_SHARDS="${NUM_SHARDS:?set NUM_SHARDS}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
PROBE_QUESTIONS="${PROBE_QUESTIONS:-100}"
PROBE_MAX_SECONDS="${PROBE_MAX_SECONDS:-900}"
S3_URI="${S3_URI:-}"
S3_SYNC_SECONDS="${S3_SYNC_SECONDS:-120}"
BENCHMARKS="${BENCHMARKS:-cpu_sweep}"

export FORCE_CPU=1
export HF_HOME="${HF_HOME:-${ROOT}/hf-cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

# Gated-model token. bootstrap_cpu.sh writes it to ${ROOT}/.hf_token but exports
# it only in its own shell, so pick it back up here. run_benchmark.py separately
# reads AdaptiveTesting/.env; this covers hosts that have no .env.
if [[ -z "${HF_TOKEN:-}" && -r "${ROOT}/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "${ROOT}/.hf_token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

LOGDIR="${ROOT}/logs/cpu_sweep_shard${SHARD_INDEX}"
mkdir -p "${LOGDIR}" "${HF_HOME}"
LOG="${LOGDIR}/worker.log"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"; }

sync_s3() {
  if [[ -z "${S3_URI}" ]]; then
    return 0
  fi
  # Non-blocking-ish: best-effort; never abort the worker on sync failure.
  aws s3 sync "${ADAPTIVE}/Outputs/" "${S3_URI}/Outputs/" \
    --only-show-errors \
    --exclude "*.tmp" --exclude ".mplcache/*" \
    >>"${LOGDIR}/s3_sync.log" 2>&1 || log "s3 sync warning (continuing)"
  aws s3 sync "${ADAPTIVE}/Outputs/_manifests/" "${S3_URI}/Outputs/_manifests/" \
    --only-show-errors >>"${LOGDIR}/s3_sync.log" 2>&1 || true
}

# Background continuous S3 sync
SYNC_PID=""
if [[ -n "${S3_URI}" ]]; then
  (
    while true; do
      sync_s3
      sleep "${S3_SYNC_SECONDS}"
    done
  ) &
  SYNC_PID=$!
  log "S3 continuous sync -> ${S3_URI} every ${S3_SYNC_SECONDS}s (pid=${SYNC_PID})"
fi

cleanup() {
  if [[ -n "${SYNC_PID}" ]]; then
    kill "${SYNC_PID}" 2>/dev/null || true
  fi
  sync_s3
}
trap cleanup EXIT

log "shard=${SHARD_INDEX}/${NUM_SHARDS} models_yaml=${MODELS_YAML}"
log "benchmarks=${BENCHMARKS} max_samples=${MAX_SAMPLES}"
log "probe=${PROBE_QUESTIONS}Q / ${PROBE_MAX_SECONDS}s  FORCE_CPU=1"
if [[ -n "${HF_TOKEN:-}" ]]; then
  log "HF token loaded (len ${#HF_TOKEN})"
else
  log "WARN: no HF_TOKEN; gated models will 401 and be skipped"
fi

cd "${CODE}"
"${VENV_PY}" run_benchmark.py \
  --benchmarks "${BENCHMARKS}" \
  --models-yaml "${MODELS_YAML}" \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  --backend hf \
  --no-judge \
  --max-samples "${MAX_SAMPLES}" \
  --probe-questions "${PROBE_QUESTIONS}" \
  --probe-max-seconds "${PROBE_MAX_SECONDS}" \
  --inference-config "${CFG}" \
  2>&1 | tee -a "${LOG}"

log "worker finished"
