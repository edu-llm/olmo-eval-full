#!/usr/bin/env bash
# CPU-ONLY Hugging Face cache pre-warmer.
#
# Fills a Hugging Face cache from a models manifest, in parallel, on cheap CPU
# compute, then mirrors it to a same-region S3 cache. GPU boxes later restore it
# with `aws s3 sync` and load OFFLINE, so an L40S never spends GPU time on a cold
# HF download (see the HF_CACHE_S3 path in run_respgen_small_l4.sh).
#
# This script imports NO torch/vllm/transformers and needs NO GPU. It is meant to
# run on a small CPU instance (good ENA bandwidth, e.g. c7i.2xlarge / m7i.2xlarge)
# that carries an IAM role allowed to write the S3 cache (edullm-lane-instance).
# The `aws s3 sync` runs on THIS box under that instance role — never laptop keys.
#
# Env:
#   MANIFEST=models_small_l4/all.yaml   # any respgen-style YAML (reads each `id`)
#   HF_CACHE_S3=s3://edullm-scratch/hf_cache   # destination cache prefix
#   HF_HOME=$HOME/.cache/huggingface    # local cache dir to fill and sync
#   WORKERS=8                           # models downloaded concurrently
#   HF_TOKEN=<token>                    # optional; keep set for tokenizer pulls / rate limits
#   RESTORE_FIRST=1                     # (default) pull existing cache down first, so a
#                                       # re-run/second box only fills gaps (resumable)
#   SKIP_SYNC=0                         # 1 = download only, don't push to S3
#
# Run (inside tmux so a disconnect doesn't kill it):
#   tmux new -s prewarm
#   HF_TOKEN=<token> bash scripts/aws/prewarm_hf_cache.sh
#   # a different roster:
#   MANIFEST=models.yaml HF_TOKEN=<token> bash scripts/aws/prewarm_hf_cache.sh

set -euo pipefail

cd "$(dirname "$0")/../.."          # -> repo root (eduLLM-Evals/)

MANIFEST="${MANIFEST:-models_small_l4/all.yaml}"
HF_CACHE_S3="${HF_CACHE_S3:-s3://edullm-scratch/hf_cache}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1   # multi-threaded high-throughput transfer
WORKERS="${WORKERS:-8}"
RESTORE_FIRST="${RESTORE_FIRST:-1}"
SKIP_SYNC="${SKIP_SYNC:-0}"

[[ -f "$MANIFEST" ]] || { echo "no such manifest: $MANIFEST" >&2; exit 1; }

# Refuse to run on a GPU box by mistake — this job is for CPU compute only.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi present: this is a GPU box. The pre-warmer is CPU-only; run it" >&2
  echo "on a cheap CPU instance so the GPU box isn't billed for downloads." >&2
  exit 1
fi

echo "== HF cache pre-warm =="
echo "   manifest=$MANIFEST  HF_HOME=$HF_HOME  workers=$WORKERS"
echo "   s3=$HF_CACHE_S3  restore_first=$RESTORE_FIRST  skip_sync=$SKIP_SYNC"

# --- CPU-only deps: huggingface_hub (+ hf_transfer) and pyyaml. No torch/vllm. --
# Prefer uv (per repo convention); fall back to pip. Installed into an isolated
# venv so this box needs nothing pre-provisioned.
PYBIN="python3"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv-prewarm >/dev/null 2>&1 || true
  # shellcheck disable=SC1091
  source .venv-prewarm/bin/activate
  uv pip install --quiet "huggingface_hub[hf_transfer]>=0.24" "pyyaml>=6"
  PYBIN="python"
else
  python3 -m venv .venv-prewarm
  # shellcheck disable=SC1091
  source .venv-prewarm/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet "huggingface_hub[hf_transfer]>=0.24" "pyyaml>=6"
  PYBIN="python"
fi

mkdir -p "$HF_HOME"

# --- resume: pull whatever is already cached in S3 so we only fill gaps --------
# `aws s3 sync` copies only new/changed keys, so this is cheap on a warm bucket
# and a no-op on a cold one. Makes the whole job idempotent across boxes/re-runs.
if [[ "$RESTORE_FIRST" == "1" ]]; then
  echo "== restoring existing cache from $HF_CACHE_S3 (gap-fill) =="
  aws s3 sync "$HF_CACHE_S3" "$HF_HOME" --only-show-errors || \
    echo "   (nothing to restore, or S3 read failed — continuing to download)"
fi

# --- parallel download ---------------------------------------------------------
SUMMARY="prewarm_summary.json"
"$PYBIN" scripts/aws/prewarm_hf_cache.py \
  --manifest "$MANIFEST" --workers "$WORKERS" --summary "$SUMMARY"

# --- push to S3 ----------------------------------------------------------------
# Exclude blobs/: HF's cache stores each file once in blobs/ and symlinks it from
# snapshots/<sha>/. `aws s3 sync` dereferences the snapshot symlinks (uploading
# real content there), so also shipping blobs/ would double every byte — and on
# restore. Snapshot files come back as regular files, which HF loads fine offline
# without blobs/, so excluding them halves the cache size and the GPU restore time.
if [[ "$SKIP_SYNC" != "1" ]]; then
  echo "== syncing cache -> $HF_CACHE_S3 (blobs/ excluded, snapshots dereferenced) =="
  aws s3 sync "$HF_HOME" "$HF_CACHE_S3" \
    --exclude "*/blobs/*" \
    --exclude "*.lock" \
    --exclude "*/.no_exist/*" \
    --only-show-errors
  echo "== done. cache in $HF_CACHE_S3 =="
  aws s3 ls --summarize --human-readable --recursive "$HF_CACHE_S3/" | tail -3
else
  echo "SKIP_SYNC=1: left cache in $HF_HOME, not pushed to S3."
fi
