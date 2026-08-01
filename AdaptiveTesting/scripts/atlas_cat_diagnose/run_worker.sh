#!/usr/bin/env bash
# Runs on the g6.xlarge worker: online ATLAS CAT → sync results to S3.
#
# Env (set by launch_g6.sh user-data):
#   CHECKPOINT   s3://... or HF model id
#   RUN_ID       results key segment
#   S3_OUT_ROOT  e.g. s3://edullm-adaptive-inference-056956104102/smoke/atlas_cat
#   EVALS        space-separated online eval names (default: the 4 wired ATLAS evals)
#   SE_STOP      default 0.3
#   MAX_ITEMS    default 40
#   MIN_ITEMS    default 8
#   REPO_ROOT    checkout path
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?CHECKPOINT required}"
RUN_ID="${RUN_ID:?RUN_ID required}"
S3_OUT_ROOT="${S3_OUT_ROOT:-s3://edullm-adaptive-inference-056956104102/smoke/atlas_cat}"
EVALS="${EVALS:-atlas_arc atlas_hellaswag atlas_winogrande atlas_gsm8k}"
SE_STOP="${SE_STOP:-0.3}"
MAX_ITEMS="${MAX_ITEMS:-40}"
MIN_ITEMS="${MIN_ITEMS:-8}"
REPO_ROOT="${REPO_ROOT:-/opt/dlami/nvme/atlas-cat/code}"
OUT_LOCAL="${OUT_LOCAL:-/opt/dlami/nvme/atlas-cat/out}"
REGION="${REGION:-us-east-1}"

S3_DEST="${S3_OUT_ROOT%/}/${RUN_ID}"
mkdir -p "${OUT_LOCAL}"
LOG="${OUT_LOCAL}/worker.log"
exec > >(tee -a "${LOG}") 2>&1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "checkpoint=${CHECKPOINT}"
log "run_id=${RUN_ID}"
log "s3_dest=${S3_DEST}"
log "repo=${REPO_ROOT}"

cd "${REPO_ROOT}"

if command -v uv >/dev/null 2>&1; then
  log "installing olmo-eval extras via uv"
  uv sync --extra vllm --extra hf 2>&1 | tail -20 || uv pip install -e ".[vllm,hf]"
  PY=(uv run)
else
  VENV=/opt/dlami/nvme/atlas-cat/venv
  if [[ ! -x "${VENV}/bin/python" ]]; then
    python3 -m venv "${VENV}"
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  log "pip install olmo-eval[vllm,hf] (slow on first boot)"
  pip install -U pip wheel
  pip install -e ".[vllm,hf]"
  PY=(python)
fi

export HF_HOME="${HF_HOME:-/opt/dlami/nvme/atlas-cat/hf-cache}"
mkdir -p "${HF_HOME}"

log "starting ATLAS CAT (${EVALS})"
eval_args=()
for e in ${EVALS}; do eval_args+=(-e "${e}"); done
"${PY[@]}" olmo-eval run-external \
  -m "${CHECKPOINT}" \
  "${eval_args[@]}" \
  -a "se_stop=${SE_STOP}" \
  -a "min_items=${MIN_ITEMS}" \
  -a "max_items=${MAX_ITEMS}" \
  -O "${OUT_LOCAL}" \
  --provider vllm_server

INSTANCE_ID="$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id || echo unknown)"
GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"

OUT_LOCAL="${OUT_LOCAL}" S3_DEST="${S3_DEST}" CHECKPOINT="${CHECKPOINT}" \
RUN_ID="${RUN_ID}" SE_STOP="${SE_STOP}" MIN_ITEMS="${MIN_ITEMS}" MAX_ITEMS="${MAX_ITEMS}" \
EVALS="${EVALS}" \
INSTANCE_ID="${INSTANCE_ID}" GIT_SHA="${GIT_SHA}" \
python3 - <<'PY'
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
    "instance_id": os.environ.get("INSTANCE_ID", "unknown"),
    "git_sha": os.environ.get("GIT_SHA", "unknown"),
    "evals": os.environ.get("EVALS", "").split(),
}
(out / "pipeline_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
print("wrote", out / "pipeline_provenance.json")
PY

log "syncing results → ${S3_DEST}"
aws s3 sync "${OUT_LOCAL}" "${S3_DEST}/" --region "${REGION}"
echo ok | aws s3 cp - "${S3_DEST}/_READY" --region "${REGION}"
log "done"
