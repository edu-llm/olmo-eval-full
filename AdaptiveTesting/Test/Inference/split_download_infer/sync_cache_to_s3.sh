#!/usr/bin/env bash
# Downloader side: fetch the roster's weights into HF_HOME, then push the warm
# cache to S3. Idempotent - re-running only transfers new/changed objects.
#
# Env:
#   ROOT            prefix dir                (default /opt/adaptive-dl)
#   S3_CACHE        cache prefix in S3        (default s3://edullm-adaptive-inference-056956104102/hf-cache)
#   REGION          AWS region                (default us-east-1)
#   MODELS_YAML     roster path               (default: registry MODELS_YAML)
#   MAX_PARAMS_B    size filter (B)           (optional)
#   SHARD_INDEX/NUM_SHARDS  split the roster across N downloaders (optional)
#   WORKERS         repo-level parallelism    (default 4)
set -euo pipefail

ROOT="${ROOT:-/opt/adaptive-dl}"
S3_CACHE="${S3_CACHE:-s3://edullm-adaptive-inference-056956104102/hf-cache}"
REGION="${REGION:-us-east-1}"
WORKERS="${WORKERS:-4}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${ROOT}/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}" "${ROOT}/logs"

# shellcheck disable=SC1091
[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate" || true
[[ -f "${ROOT}/.hf_token" ]] && export HF_TOKEN="$(cat "${ROOT}/.hf_token")"

# Stream each model to S3 as it completes and (by default) delete it locally so
# disk stays flat even for a TB-scale roster. Set DELETE_AFTER=0 to keep a local
# copy too.
args=(--workers "${WORKERS}" --s3-prefix "${S3_CACHE}" --region "${REGION}"
      --manifest "${ROOT}/logs/prefetch_manifest.json")
[[ "${DELETE_AFTER:-1}" == "1" ]] && args+=(--delete-after)
[[ -n "${MODELS_YAML:-}" ]] && args+=(--models-yaml "${MODELS_YAML}")
[[ -n "${MAX_PARAMS_B:-}" ]] && args+=(--max-params-b "${MAX_PARAMS_B}")
[[ -n "${SHARD_INDEX:-}" && -n "${NUM_SHARDS:-}" ]] && args+=(--shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}")

echo "[$(date -u +%H:%M:%S)] downloading + streaming weights -> ${S3_CACHE}"
python "${HERE}/prefetch_weights.py" "${args[@]}"

aws s3 cp "${ROOT}/logs/prefetch_manifest.json" "${S3_CACHE}/_manifests/prefetch_$(date -u +%Y%m%d-%H%M%S).json" \
  --region "${REGION}" --only-show-errors || true
# Readiness marker GPU workers can poll before they launch.
date -u +%Y%m%dT%H%M%SZ | aws s3 cp - "${S3_CACHE}/_READY" --region "${REGION}" --only-show-errors
echo "[$(date -u +%H:%M:%S)] cache ready at ${S3_CACHE}"
