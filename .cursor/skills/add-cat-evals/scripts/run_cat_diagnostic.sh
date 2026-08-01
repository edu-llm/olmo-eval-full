#!/usr/bin/env bash
# In-process ATLAS CAT diagnostic for an OLMo-core training checkpoint.
#
#   convert (OLMo-core -> HF) -> run atlas_arc CAT via vLLM -> sync results to S3
#
# Usage:
#   run_cat_diagnostic.sh --checkpoint DIR --run-id ID --s3-out s3://bucket/prefix
#
# The checkpoint may be a native OLMo-core dir (config.json + model_and_optim/),
# or an HF dir / HF id with --skip-convert. Nothing is pushed to the HF Hub.
set -euo pipefail

CHECKPOINT=""
RUN_ID=""
S3_OUT=""
TOKENIZER=""
SE_STOP="0.3"
MIN_ITEMS="8"
MAX_ITEMS="40"
TP="1"
# CAT benchmarks to administer (space-separated online eval names). Default is the
# four wired ATLAS benchmarks with calibrated banks in this repo. Override with
# --evals "atlas_arc atlas_hellaswag" to run a subset.
EVALS="${ATLAS_EVALS:-atlas_arc atlas_hellaswag atlas_winogrande atlas_gsm8k}"
SKIP_CONVERT="0"
KEEP_HF="0"
DRY_RUN="0"
REGION="${AWS_REGION:-us-east-1}"
# Path to OLMo-core's converter; override if it lives elsewhere.
CONVERT_SCRIPT="${OLMO_CORE_CONVERT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --s3-out) S3_OUT="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --se-stop) SE_STOP="$2"; shift 2 ;;
    --min-items) MIN_ITEMS="$2"; shift 2 ;;
    --max-items) MAX_ITEMS="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --evals) EVALS="$2"; shift 2 ;;
    --skip-convert) SKIP_CONVERT="1"; shift ;;
    --keep-hf) KEEP_HF="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${CHECKPOINT}" ]] || { echo "--checkpoint required" >&2; exit 2; }
[[ -n "${RUN_ID}" ]] || { echo "--run-id required" >&2; exit 2; }
[[ -n "${S3_OUT}" ]] || { echo "--s3-out required" >&2; exit 2; }
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--run-id must match [A-Za-z0-9._-]+ (got ${RUN_ID})" >&2; exit 2
fi

S3_DEST="${S3_OUT%/}/${RUN_ID}"
OUT_LOCAL="$(mktemp -d)"
LOG="${OUT_LOCAL}/worker.log"
# Mirror stdout/stderr into worker.log. Fall back to plain redirection if the
# environment forbids process substitution.
if ! exec > >(tee -a "${LOG}") 2>&1; then
  exec >>"${LOG}" 2>&1
fi
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

HF_CKPT="${CHECKPOINT}"
CONVERTED_DIR=""

# Detect a native OLMo-core checkpoint (needs conversion) vs HF/HF-id.
needs_convert() {
  [[ "${SKIP_CONVERT}" == "1" ]] && return 1
  [[ -d "${CHECKPOINT}" ]] || return 1   # HF id -> no conversion
  [[ -d "${CHECKPOINT}/model_and_optim" || -f "${CHECKPOINT}/.metadata" ]]
}

log "checkpoint=${CHECKPOINT}"
log "run_id=${RUN_ID}"
log "s3_dest=${S3_DEST}"
log "params: se_stop=${SE_STOP} min_items=${MIN_ITEMS} max_items=${MAX_ITEMS} tp=${TP}"
log "evals: ${EVALS}"

if [[ "${DRY_RUN}" == "1" ]]; then
  if needs_convert; then log "DRY_RUN would convert OLMo-core -> HF, then run: ${EVALS}"
  else log "DRY_RUN would run ${EVALS} directly (no conversion)"; fi
  log "DRY_RUN results -> ${S3_DEST}/"
  rm -rf "${OUT_LOCAL}"; exit 0
fi

if needs_convert; then
  CONVERTED_DIR="${OUT_LOCAL}/hf_ckpt"
  if [[ -z "${CONVERT_SCRIPT}" ]]; then
    echo "OLMo-core converter not found. Set OLMO_CORE_CONVERT to the path of" >&2
    echo "convert_checkpoint_to_hf.py (see CONVERSION.md)." >&2
    exit 3
  fi
  log "converting OLMo-core -> HF at ${CONVERTED_DIR}"
  conv_args=(-i "${CHECKPOINT}" -o "${CONVERTED_DIR}" --dtype bfloat16 --skip-validation)
  [[ -n "${TOKENIZER}" ]] && conv_args+=(-t "${TOKENIZER}")
  python "${CONVERT_SCRIPT}" "${conv_args[@]}"
  HF_CKPT="${CONVERTED_DIR}"
else
  log "no conversion (HF checkpoint or --skip-convert)"
fi

log "running CAT (${EVALS}) on ${HF_CKPT}"
tp_args=()
[[ "${TP}" != "1" ]] && tp_args+=(--tp "${TP}")
eval_args=()
for e in ${EVALS}; do eval_args+=(-e "${e}"); done
# One vLLM boot serves every -e; each benchmark runs its own adaptive loop.
uv run olmo-eval run-external \
  -m "${HF_CKPT}" \
  "${eval_args[@]}" \
  --provider vllm_server \
  -a "se_stop=${SE_STOP}" \
  -a "min_items=${MIN_ITEMS}" \
  -a "max_items=${MAX_ITEMS}" \
  -O "${OUT_LOCAL}" \
  "${tp_args[@]}"

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
OUT_LOCAL="${OUT_LOCAL}" S3_DEST="${S3_DEST}" CHECKPOINT="${CHECKPOINT}" \
RUN_ID="${RUN_ID}" SE_STOP="${SE_STOP}" MIN_ITEMS="${MIN_ITEMS}" MAX_ITEMS="${MAX_ITEMS}" \
EVALS="${EVALS}" \
GIT_SHA="${GIT_SHA}" python3 - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_LOCAL"])
prov = {
    "checkpoint": os.environ["CHECKPOINT"],
    "run_id": os.environ["RUN_ID"],
    "s3_dest": os.environ["S3_DEST"],
    "se_stop": float(os.environ["SE_STOP"]),
    "min_items": int(os.environ["MIN_ITEMS"]),
    "max_items": int(os.environ["MAX_ITEMS"]),
    "git_sha": os.environ.get("GIT_SHA", "unknown"),
    "evals": os.environ.get("EVALS", "").split(),
}
(out / "pipeline_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
print("wrote", out / "pipeline_provenance.json")
PY

log "syncing results -> ${S3_DEST}"
aws s3 sync "${OUT_LOCAL}" "${S3_DEST}/" --region "${REGION}" \
  --exclude "hf_ckpt/*"
echo ok | aws s3 cp - "${S3_DEST}/_READY" --region "${REGION}"
log "done -> ${S3_DEST}/"

if [[ "${KEEP_HF}" == "1" && -n "${CONVERTED_DIR}" ]]; then
  DEST_HF="$(dirname "${CHECKPOINT}")/$(basename "${CHECKPOINT}")-hf"
  log "keeping converted HF checkpoint at ${DEST_HF}"
  cp -r "${CONVERTED_DIR}" "${DEST_HF}" || true
fi
rm -rf "${OUT_LOCAL}"
