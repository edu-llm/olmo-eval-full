#!/usr/bin/env bash
# One-time environment setup for the adaptive-inference sweep on the shared
# ms-135m B200 node. Idempotent: safe to re-run. Does NOT start the sweep and
# does NOT touch the training GPUs (0,1) - it only builds a Python venv and
# installs wheels on the already-running (pre-paid) instance, so it incurs no
# additional AWS spend.
#
# Layout (all on the 26 TB NVMe scratch, isolated from the MemorySplit dirs):
#   /opt/dlami/nvme/adaptive-inference/
#     code/        <- Test/Inference + Inputs (pushed separately)
#     venv/        <- dedicated Python venv
#     hf-cache/    <- HF_HOME (model + dataset cache)
#     logs/        <- bootstrap + sweep logs
set -euo pipefail

ROOT="/opt/dlami/nvme/adaptive-inference"
VENV="${ROOT}/venv"
export HF_HOME="${ROOT}/hf-cache"
mkdir -p "${ROOT}/logs" "${HF_HOME}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- Python venv (system python3 = 3.12) ----------------------------------
if [[ ! -x "${VENV}/bin/python" ]]; then
  log "creating venv at ${VENV}"
  if ! python3 -m venv "${VENV}" 2>/dev/null; then
    log "python3-venv missing; installing via apt"
    apt-get update -y && apt-get install -y python3.12-venv
    python3 -m venv "${VENV}"
  fi
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- Dependencies ----------------------------------------------------------
# vLLM pulls a Blackwell-capable torch. hf_transfer accelerates HF downloads.
# prometheus-eval provides the judge model plumbing (we call it through vLLM).
log "installing python deps (this is the long step; ~minutes)"
python -m pip install -U \
  vllm \
  "transformers>=4.44" \
  datasets \
  huggingface_hub \
  hf_transfer \
  accelerate \
  prometheus-eval \
  boto3 \
  pyyaml

# --- Sanity: confirm Blackwell (sm_100) is visible to torch/vLLM -----------
log "verifying GPU/toolchain"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device_count", torch.cuda.device_count())
if torch.cuda.device_count():
    print("capability", torch.cuda.get_device_capability(0))
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:  # noqa: BLE001
    print("vllm import FAILED:", e)
PY

touch "${ROOT}/.bootstrap_ok"
log "bootstrap complete -> ${ROOT}/.bootstrap_ok"
