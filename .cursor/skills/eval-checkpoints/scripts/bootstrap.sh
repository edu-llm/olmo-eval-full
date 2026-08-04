#!/usr/bin/env bash
# Bring a bare Linux + CUDA GPU box to a state where run_eval_sweep.sh can run.
#
# This is what lets the skill work without a pre-built platform container image:
# rather than assuming an environment was baked in ahead of time, it installs one
# at run time. Slower to first result than a pre-built image, but it needs nothing
# to exist beforehand except the repo, a GPU, and network access.
#
# Usage:
#   bootstrap.sh                 # sync into $OLMO_EVAL_ROOT (inferred from this file)
#   OLMO_EVAL_ROOT=/path bootstrap.sh
#   bootstrap.sh --no-olmo-core  # skip the conversion extra (see the note below)
#
# Safe to re-run. `uv sync` is declarative, so a second run against an already
# synced tree is close to a no-op.
set -euo pipefail

SKIP_OLMO_CORE="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-olmo-core) SKIP_OLMO_CORE="1"; shift ;;
    -h|--help) awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${OLMO_EVAL_ROOT:-$(cd "${SKILL_DIR}/../../.." && pwd)}"

log() { echo "[bootstrap] $*"; }

[[ -f "${REPO_ROOT}/pyproject.toml" ]] || {
  echo "no pyproject.toml at ${REPO_ROOT}; set OLMO_EVAL_ROOT to the olmo-eval checkout" >&2
  exit 2
}

log "repo root: ${REPO_ROOT}"

# --- uv -----------------------------------------------------------------------
# The official installer drops uv in ~/.local/bin, which is not on PATH under
# cloud-init or a bare ssh session. Export it here and tell the caller, since a
# child shell will not inherit this one's PATH.
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  export HOME="${HOME:-/root}"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || {
  echo "uv still not on PATH after install; add \$HOME/.local/bin to PATH" >&2
  exit 3
}
log "uv: $(uv --version 2>&1)"

# --- HF cache -----------------------------------------------------------------
# Benchmark datasets download on first use and the default cache is under $HOME,
# which on many GPU AMIs is a small root volume. Prefer an instance-store or data
# mount when one is present.
if [[ -z "${HF_HOME:-}" ]]; then
  for candidate in /opt/dlami/nvme /mnt /data; do
    if [[ -d "${candidate}" && -w "${candidate}" ]]; then
      export HF_HOME="${candidate}/hf-cache"
      break
    fi
  done
  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
fi
mkdir -p "${HF_HOME}"
log "HF_HOME: ${HF_HOME}"

# --- dependencies -------------------------------------------------------------
# vllm  : the inference backend the sweep runs through
# hf    : transformers, for tokenizers and HF-format checkpoints
# s3    : boto3/smart_open, for reading checkpoints and writing results
# olmo_core : only needed to convert native OLMo-core checkpoints. It pins
#   ai2-olmo-core==2.4.0, which conflicts with the openhands dependency, which is
#   why it is not in the default sync. If resolution fails, re-run with
#   --no-olmo-core and either pre-convert the checkpoints elsewhere or point
#   $OLMO_CORE_CONVERT at a converter in its own environment.
EXTRAS=(--extra vllm --extra hf --extra s3)
if [[ "${SKIP_OLMO_CORE}" == "0" ]]; then
  EXTRAS+=(--extra olmo_core)
fi

log "syncing: ${EXTRAS[*]}"
if ! (cd "${REPO_ROOT}" && uv sync "${EXTRAS[@]}"); then
  if [[ "${SKIP_OLMO_CORE}" == "0" ]]; then
    echo >&2
    echo "sync failed. If the conflict involves ai2-olmo-core, re-run as:" >&2
    echo "  bash ${BASH_SOURCE[0]} --no-olmo-core" >&2
    echo "and handle conversion separately (see SKILL.md)." >&2
  fi
  exit 4
fi

# --- record what we actually got ----------------------------------------------
# A sweep's log should say which versions produced its numbers. Resolved
# versions, not the pins, because extras and markers can move them.
log "resolved versions:"
(cd "${REPO_ROOT}" && uv run python - <<'PY' || true
import importlib.metadata as md

for dist in ("olmo-eval", "vllm", "transformers", "ai2-olmo-core", "boto3"):
    try:
        print(f"  {dist}: {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"  {dist}: not installed")
PY
)

# `aws` is not a Python dependency, so uv sync cannot supply it. The sweep shells
# out to it for every S3 operation.
if command -v aws >/dev/null 2>&1; then
  log "aws: $(aws --version 2>&1 | head -1)"
else
  log "WARNING: aws CLI not found. The sweep needs it for S3 discovery, staging"
  log "         and upload. Results always go to S3 even when the checkpoints are"
  log "         local, so install it before running the sweep."
fi

log "done. If uv was just installed, export PATH before running the sweep:"
log "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
