#!/usr/bin/env bash
# End-to-end smoke for the split-cost setup. Two modes:
#
#   smoke_split.sh cpu   # parallel-download 5 small models -> HF_HOME -> S3, _READY
#   smoke_split.sh gpu   # pull those 5 from S3, run a few real inferences (smoke_test.py)
#
# The CPU side needs only huggingface_hub (bootstrap_downloader.sh). The GPU side
# needs the vLLM venv (installed by the launcher) and never downloads weights -
# they come from the S3 cache the CPU box produced.
#
# Env (shared):
#   REGION        default us-east-1
#   SMOKE_BASE    default s3://edullm-adaptive-inference-056956104102/smoke_split
#   MODELS        comma list (default: 5 small, ungated, vLLM-friendly models)
#   WORKERS       CPU repo-level parallelism (default 5)
set -euo pipefail

MODE="${1:?usage: smoke_split.sh cpu|gpu}"
REGION="${REGION:-us-east-1}"
SMOKE_BASE="${SMOKE_BASE:-s3://edullm-adaptive-inference-056956104102/smoke_split}"
CACHE="${SMOKE_BASE}/hf-cache"
MODELS="${MODELS:-Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-0.5B-Instruct,HuggingFaceTB/SmolLM2-135M,HuggingFaceTB/SmolLM2-360M,Qwen/Qwen2.5-1.5B-Instruct}"
WORKERS="${WORKERS:-5}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INF="$(cd "${HERE}/.." && pwd)"

case "${MODE}" in
cpu)
  ROOT="${ROOT:-/opt/adaptive-dl}"
  export HF_HOME="${ROOT}/hf-cache"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  mkdir -p "${HF_HOME}" "${ROOT}/logs"
  # shellcheck disable=SC1091
  [[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate" || true
  [[ -f "${ROOT}/.hf_token" ]] && export HF_TOKEN="$(cat "${ROOT}/.hf_token")"

  # Stream each model to S3 the moment it finishes, then delete it locally, so
  # disk stays flat and every completed model is persisted (crash-safe).
  echo "[cpu $(date -u +%H:%M:%S)] downloading+streaming (workers=${WORKERS}): ${MODELS}"
  python "${HERE}/prefetch_weights.py" --models "${MODELS}" --workers "${WORKERS}" \
    --s3-prefix "${CACHE}" --region "${REGION}" --delete-after \
    --manifest "${ROOT}/logs/smoke_prefetch.json"

  aws s3 cp "${ROOT}/logs/smoke_prefetch.json" "${CACHE}/_manifests/smoke_prefetch.json" \
    --region "${REGION}" --only-show-errors || true
  date -u +%Y%m%dT%H%M%SZ | aws s3 cp - "${CACHE}/_READY" --region "${REGION}" --only-show-errors
  echo "[cpu $(date -u +%H:%M:%S)] cache READY at ${CACHE}"
  ;;

gpu)
  ROOT="${ROOT:-/opt/dlami/nvme/adaptive-infer}"
  STAMP="$(date -u +%Y%m%d-%H%M%S)"
  RES="${SMOKE_BASE}/results/${STAMP}"
  export HF_HOME="${ROOT}/hf-cache"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  export FLASHINFER_DISABLE_VERSION_CHECK=1
  # eduLLM-Evals ships beside olmo-eval-full in the tarball; caller may override.
  export EDULLM_EVALS_ROOT="${EDULLM_EVALS_ROOT:-$(cd "${INF}/../../.." && pwd)/eduLLM-Evals}"
  mkdir -p "${HF_HOME}/hub"
  # shellcheck disable=SC1091
  [[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate" || true

  # Selective pull: only these 5 models' cache folders from S3.
  includes=()
  while IFS= read -r d; do
    [[ -n "$d" ]] && includes+=(--include "hub/${d}/*")
  done < <(python "${HERE}/list_repos.py" --cache-dirs --include-tokenizers --models "${MODELS}")
  echo "[gpu $(date -u +%H:%M:%S)] pulling ${#includes[@]} model dir(s) from ${CACHE}"
  aws s3 sync "${CACHE}/" "${HF_HOME}/" --region "${REGION}" --exclude "*" "${includes[@]}" --only-show-errors
  echo "[gpu] warm cache now on box:"; ls -1 "${HF_HOME}/hub" 2>/dev/null || true

  cd "${INF}"
  set +e
  python smoke_test.py --models "${MODELS}" --num-models 5 --num-mcq 3 --num-frq 2 \
    --max-new-tokens 64 --backend vllm 2>&1 | tee "${ROOT}/smoke_split.log"
  RC=${PIPESTATUS[0]}
  set -e
  echo "[gpu $(date -u +%H:%M:%S)] smoke rc=${RC}"

  REPORT="$(ls -t "${INF}"/../../Outputs/_smoke/*/smoke_report.json 2>/dev/null | head -1 || true)"
  aws s3 cp "${ROOT}/smoke_split.log" "${RES}/smoke_test.log" --region "${REGION}" || true
  [[ -n "${REPORT}" ]] && aws s3 cp "${REPORT}" "${RES}/smoke_report.json" --region "${REGION}" || true
  printf 'rc=%s stamp=%s report=%s\n' "${RC}" "${STAMP}" "${REPORT:-none}" | \
    aws s3 cp - "${RES}/DONE" --region "${REGION}" || true
  echo "[gpu $(date -u +%H:%M:%S)] results -> ${RES}"
  ;;

*)
  echo "unknown mode '${MODE}' (want cpu|gpu)" >&2
  exit 2
  ;;
esac
