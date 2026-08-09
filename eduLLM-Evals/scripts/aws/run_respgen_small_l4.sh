#!/usr/bin/env bash
# Response generation (NO grading) for the 27 small-cluster (<2B) tutor models on
# TutorEval + BiGGen, sized for a SINGLE-GPU L40S node (g6e.xlarge, single L40S 48GB).
# Run one SHARD per L40S box; a few L40S boxes in parallel cover the whole roster.
#
#   SHARD 1..4  -> models_small_l4/shard<N>.yaml   (size-balanced, ~7 models each)
#
# Each model is loaded ONCE and answers BOTH benchmarks in that load (model load
# dominates wall clock). Only TutorEval + BiGGen are generated; grading is a
# separate step (different rubrics/verifiers) and is intentionally NOT run here.
#
# Prereqs (once per box; see scripts/aws/setup_respgen.sh):
#   source .venv/bin/activate
#   export HF_TOKEN=<token>          # all 27 are ungated, but keep it set for the tokenizer pulls
# Required:
#   SHARD=<1..4>                     # which shard this L4 runs
# Optional:
#   S3_URI=s3://<bucket>/<prefix>    # per-shard upload (instance IAM); highly recommended
#   OUT_DIR=runs/responses           # local shard dir (default)
#   GPU=0                            # CUDA index to pin (default 0 = the L40S)
#   ONLY=TutorEval,BiGGen            # override the benchmark subset if ever needed
#
# Run (inside tmux so a disconnect doesn't kill it):
#   tmux new -s respgen
#   SHARD=1 S3_URI=s3://edullm-scratch/tutor_responses/small_lt2b bash scripts/aws/run_respgen_small_l4.sh
#
# Resumable: re-running skips scenarios already in each (benchmark, model) shard.
# Nothing here provisions AWS resources; it uses the L40S box you already pay for.

set -euo pipefail

cd "$(dirname "$0")/../.."          # -> tutor_cat project root (eduLLM-Evals/)

SHARD="${SHARD:?set SHARD=1..4 (which models_small_l4/shard<N>.yaml this L4 runs)}"
MANIFEST="models_small_l4/shard${SHARD}.yaml"
[[ -f "$MANIFEST" ]] || { echo "no such shard manifest: $MANIFEST (SHARD must be 1..4)" >&2; exit 1; }

GPU="${GPU:-0}"
OUT_DIR="${OUT_DIR:-runs/responses}"
ONLY="${ONLY:-TutorEval,BiGGen}"
BENCHMARKS="benchmarks.yaml"

: "${HF_TOKEN:?set HF_TOKEN in the environment; never commit it}"

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found — are you on the GPU node?" >&2; exit 1; }

# --- GPU index must exist on this node -------------------------------------
NGPU="$(nvidia-smi -L | wc -l)"
if (( GPU >= NGPU )); then
  echo "This node has $NGPU GPU(s), valid indices 0..$((NGPU-1)); GPU $GPU does not exist." >&2
  exit 1
fi

# --- refuse to start if the target GPU is already busy (never disturb a job) --
BUSY="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
if (( BUSY > 0 )); then
  echo "GPU $GPU already has $BUSY running compute process(es); refusing to start." >&2
  echo "Free GPU $GPU first, or set GPU=<a free index> — do NOT kill the existing job." >&2
  nvidia-smi --id="$GPU"
  exit 1
fi

# --- optional warm HF cache from S3 (zero GPU-idle download) ------------------
# When HF_CACHE_S3 is set, restore the pre-warmed Hugging Face cache (filled on
# cheap CPU compute by scripts/aws/prewarm_hf_cache.sh) and load models OFFLINE,
# so the L40S spends no GPU time on cold HF downloads. Unset => current behavior:
# each model downloads inside the GPU process on first load. The sync runs under
# the box's instance IAM role (same as the shard upload), never laptop keys.
if [[ -n "${HF_CACHE_S3:-}" ]]; then
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  mkdir -p "$HF_HOME"
  echo "== warm cache: aws s3 sync ${HF_CACHE_S3} -> ${HF_HOME} =="
  aws s3 sync "$HF_CACHE_S3" "$HF_HOME" --only-show-errors
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  echo "   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 — models load from the warm cache"
fi

echo "== response generation: shard ${SHARD} (${MANIFEST}) on GPU index ${GPU} =="
echo "   benchmarks=${ONLY}  out_dir=${OUT_DIR}  s3=${S3_URI:-<none>}  warm_cache=${HF_CACHE_S3:-<none>}  resume=on"

ARGS=(generate --models "$MANIFEST" --benchmarks "$BENCHMARKS" --only "$ONLY" \
      --gpu-ids "$GPU" --out-dir "$OUT_DIR")
[[ -n "${S3_URI:-}" ]] && ARGS+=(--s3-uri "$S3_URI")

exec tutor-cat "${ARGS[@]}"
