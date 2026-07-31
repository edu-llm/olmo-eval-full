#!/usr/bin/env bash
# GPU side: pull the warm HF cache from S3 into HF_HOME. With SHARD_INDEX/
# NUM_SHARDS set, only this shard's model folders are synced (smaller disk +
# faster start); otherwise the whole cache is pulled.
#
# Env:
#   ROOT            prefix dir      (default /opt/dlami/nvme/adaptive-infer)
#   S3_CACHE        cache prefix    (default s3://edullm-adaptive-inference-056956104102/hf-cache)
#   REGION          AWS region      (default us-east-1)
#   MODELS_YAML     roster path     (default: registry MODELS_YAML)
#   MAX_PARAMS_B    size filter (B) (optional; must match the downloader's filter)
#   SHARD_INDEX/NUM_SHARDS  selective per-shard pull (optional)
#   CODE_DIR        Inference dir with list_repos.py (default: this folder's parent)
set -euo pipefail

ROOT="${ROOT:-/opt/dlami/nvme/adaptive-infer}"
S3_CACHE="${S3_CACHE:-s3://edullm-adaptive-inference-056956104102/hf-cache}"
REGION="${REGION:-us-east-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${ROOT}/hf-cache"
mkdir -p "${HF_HOME}/hub"

if [[ -n "${SHARD_INDEX:-}" && -n "${NUM_SHARDS:-}" ]]; then
  echo "[$(date -u +%H:%M:%S)] selective pull for shard ${SHARD_INDEX}/${NUM_SHARDS}"
  lr=(python "${HERE}/list_repos.py" --cache-dirs --include-tokenizers
      --shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}")
  [[ -n "${MODELS_YAML:-}" ]] && lr+=(--models-yaml "${MODELS_YAML}")
  [[ -n "${MAX_PARAMS_B:-}" ]] && lr+=(--max-params-b "${MAX_PARAMS_B}")
  includes=()
  while IFS= read -r d; do
    [[ -n "$d" ]] && includes+=(--include "hub/${d}/*")
  done < <("${lr[@]}")
  if [[ ${#includes[@]} -eq 0 ]]; then
    echo "no models in this shard; nothing to pull"; exit 0
  fi
  aws s3 sync "${S3_CACHE}/" "${HF_HOME}/" --region "${REGION}" \
    --exclude "*" "${includes[@]}" --only-show-errors
else
  echo "[$(date -u +%H:%M:%S)] full cache pull"
  aws s3 sync "${S3_CACHE}/" "${HF_HOME}/" --region "${REGION}" --only-show-errors
fi

echo "[$(date -u +%H:%M:%S)] cache present under ${HF_HOME}/hub:"
ls -1 "${HF_HOME}/hub" 2>/dev/null | head -n 20 || true
