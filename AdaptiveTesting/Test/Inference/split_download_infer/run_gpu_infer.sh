#!/usr/bin/env bash
# GPU side: pull the warm cache, then run the EXISTING sweep driver against it.
# Weights load from the synced cache (no GPU-time downloads). This script does
# not install anything - stand up the GPU venv with the repo's own
# aws/node_bootstrap.sh (or requirements.txt) first.
#
# Env:
#   ROOT            prefix dir     (default /opt/dlami/nvme/adaptive-infer)
#   VENV            python venv    (default ${ROOT}/venv)
#   S3_CACHE        cache prefix   (default s3://edullm-adaptive-inference-056956104102/hf-cache)
#   S3_OUT          outputs prefix (default s3://edullm-adaptive-inference-056956104102/split_infer)
#   REGION          AWS region     (default us-east-1)
#   MODELS_YAML     roster path    (default: registry MODELS_YAML)
#   MAX_PARAMS_B    size filter    (optional; must match downloader)
#   SHARD_INDEX/NUM_SHARDS   this worker's shard (optional)
#   BENCHMARKS      default "all"
#   MAX_SAMPLES     per-benchmark cap (optional)
#   OFFLINE         1 => export HF_HUB_OFFLINE=1 (strict; requires datasets pre-cached)
#   SYNC_SECONDS    background output sync period (default 120)
set -euo pipefail

ROOT="${ROOT:-/opt/dlami/nvme/adaptive-infer}"
VENV="${VENV:-${ROOT}/venv}"
S3_CACHE="${S3_CACHE:-s3://edullm-adaptive-inference-056956104102/hf-cache}"
S3_OUT="${S3_OUT:-s3://edullm-adaptive-inference-056956104102/split_infer}"
REGION="${REGION:-us-east-1}"
BENCHMARKS="${BENCHMARKS:-all}"
SYNC_SECONDS="${SYNC_SECONDS:-120}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INF="$(cd "${HERE}/.." && pwd)"

export HF_HOME="${ROOT}/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
[[ "${OFFLINE:-0}" == "1" ]] && export HF_HUB_OFFLINE=1

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# 1) pull this shard's weights from the warm cache.
S3_CACHE="${S3_CACHE}" REGION="${REGION}" ROOT="${ROOT}" \
  MODELS_YAML="${MODELS_YAML:-}" MAX_PARAMS_B="${MAX_PARAMS_B:-}" \
  SHARD_INDEX="${SHARD_INDEX:-}" NUM_SHARDS="${NUM_SHARDS:-}" \
  bash "${HERE}/pull_cache_from_s3.sh"

# 2) background durability sync of outputs.
OUT_LOCAL="$(cd "${INF}/../.." && pwd)/Outputs"
mkdir -p "${OUT_LOCAL}"
( while true; do sleep "${SYNC_SECONDS}"; aws s3 sync "${OUT_LOCAL}/" "${S3_OUT}/Outputs/" --region "${REGION}" --only-show-errors || true; done ) &
SYNC_PID=$!
trap 'kill ${SYNC_PID} 2>/dev/null || true; aws s3 sync "${OUT_LOCAL}/" "${S3_OUT}/Outputs/" --region "${REGION}" --only-show-errors || true' EXIT

# 3) run the existing driver (unchanged) with the warm cache.
cd "${INF}"
run=(python run_benchmark.py --backend vllm --benchmarks "${BENCHMARKS}")
[[ -n "${MODELS_YAML:-}" ]] && run+=(--models-yaml "${MODELS_YAML}")
[[ -n "${SHARD_INDEX:-}" && -n "${NUM_SHARDS:-}" ]] && run+=(--shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}")
[[ -n "${MAX_SAMPLES:-}" ]] && run+=(--max-samples "${MAX_SAMPLES}")

echo "[$(date -u +%H:%M:%S)] ${run[*]}"
"${run[@]}"
echo "[$(date -u +%H:%M:%S)] inference complete"
