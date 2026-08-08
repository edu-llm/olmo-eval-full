#!/usr/bin/env bash
# Run the multi-benchmark judge grading pipeline on the AWS P6 node, PINNED TO
# GPU INDEX 4 ONLY (each judge run gets CUDA_VISIBLE_DEVICES=4), so jobs on the
# other GPUs are left completely alone. Modeled on scripts/aws/run_respgen_gpu2.sh
# (same env-var / S3 / GPU-safety conventions).
#
# Flow (emit -> judge -> ingest), driven by scripts/run_all_judge_grading.py and
# the FROZEN judge runner aws_judge_handoff/scripts/run_judge_validation.py:
#   (a) EMIT   : stage per-benchmark blinded cases under runs/judge/<Benchmark>/
#                (shard-aware; see NUM_SHARDS/SHARD_INDEX below).
#   (b) PUSH   : aws s3 cp each benchmark's cases file up to S3.
#   (c) JUDGE  : run the frozen judge (Qwen) for each in-scope benchmark on GPU 4,
#                writing verdicts locally and to S3 (--require-s3-upload).
#   (d) PULL   : sync each shard's verdict file back down into a FLAT inbox so the
#                ingest glob (<inbox>/<Benchmark>/*.jsonl) finds every shard.
#   (e) INGEST : run the driver with --ingest to assemble per-benchmark verdicts +
#                response matrices (merges all shards).
#
# S3 DIVISION OF LABOR: this wrapper OWNS the on-node S3 transfers (push cases in
# (b), pull verdicts in (d)) because it runs on the GPU box, loops per shard, and
# knows the frozen runner's upload layout. The driver (run_all_judge_grading.py)
# is S3-aware but does NOT duplicate this: it derives its GPU hand-off command's
# --s3-output-prefix from $S3_GRADING_PREFIX (exported below, so both agree on the
# same layout), and its --ingest can take an s3:// root for standalone (no-wrapper)
# runs. Here we still pull to a flat local inbox and ingest from that local path.
#
# Prereqs (see scripts/aws/setup_respgen.sh, and `uv sync`):
#   export HF_TOKEN=<token>
# S3 (bucket/prefix configurable; NEVER hardcode a real bucket here):
#   export S3_GRADING_PREFIX=s3://<bucket>/edu-tutor-grading
#
# Optional env:
#   export GPU=4                    # override the pinned index if ever needed
#   export JUDGE=qwen               # frozen judge name (judge_frozen.yaml default)
#   export ONLY=TutorBench,Bridge   # narrow the in-scope benchmark set
#   export RESPONSES_ROOT=runs/responses
#   export JUDGE_ROOT=runs/judge
#   export TP=1                     # judge tensor-parallel size
#   export REPLICATE_ID=r1
#   export PROMPT_VARIANT=canonical
#   export SKIP_INGEST=1            # emit+judge only (for per-shard array workers)
#
# Sharding across an AWS Batch array job:
#   Set NUM_SHARDS to the array size; each task's SHARD_INDEX defaults to
#   AWS_BATCH_JOB_ARRAY_INDEX. Emit partitions whole (model, scenario) blocks by a
#   stable hash of response_id, so all criteria for a block stay in ONE shard and
#   remain contiguous (prefix caching relies on this). Each task grades its shard
#   and uploads canonical_<rid>.shard<i>.jsonl; ingest merges all shards.

set -euo pipefail

cd "$(dirname "$0")/../.."          # -> repo root (eduLLM-Evals/)

GPU="${GPU:-4}"
JUDGE="${JUDGE:-qwen}"
RESPONSES_ROOT="${RESPONSES_ROOT:-runs/responses}"
JUDGE_ROOT="${JUDGE_ROOT:-runs/judge}"
INBOX="${INBOX:-$JUDGE_ROOT/_verdicts_inbox}"
S3_GRADING_PREFIX="${S3_GRADING_PREFIX:-s3://YOUR-BUCKET/edu-tutor-grading}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-${AWS_BATCH_JOB_ARRAY_INDEX:-0}}"
TP="${TP:-1}"
REPLICATE_ID="${REPLICATE_ID:-r1}"
PROMPT_VARIANT="${PROMPT_VARIANT:-canonical}"
SKIP_INGEST="${SKIP_INGEST:-0}"
ONLY="${ONLY:-}"

# In-scope benchmarks (mirror IN_SCOPE_BENCHMARKS in run_all_judge_grading.py).
DEFAULT_BENCHMARKS="TutorBench TutorEval InFoBench Bridge BiGGen WildBench"
if [[ -n "$ONLY" ]]; then
  BENCHMARKS="${ONLY//,/ }"
else
  BENCHMARKS="$DEFAULT_BENCHMARKS"
fi

: "${HF_TOKEN:?set HF_TOKEN in the environment (gated repos); never commit it}"
command -v uv >/dev/null 2>&1 || { echo "uv not found — install uv first" >&2; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found — needed for S3 push/pull" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found — are you on the GPU node?" >&2; exit 1; }

if [[ "$S3_GRADING_PREFIX" == *YOUR-BUCKET* ]]; then
  echo "WARNING: S3_GRADING_PREFIX is the placeholder ($S3_GRADING_PREFIX); set it to a real bucket/prefix." >&2
fi

# --- GPU index must exist on this node -------------------------------------
NGPU="$(nvidia-smi -L | wc -l)"
if (( GPU >= NGPU )); then
  echo "This node has $NGPU GPU(s), valid indices 0..$((NGPU-1)); GPU $GPU does not exist." >&2
  echo "(CUDA indices are 0-based: the 8th GPU is index 7, not 8.)" >&2
  exit 1
fi

# --- refuse to start if GPU 4 is already in use (never disturb existing jobs) ---
BUSY="$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
if (( BUSY > 0 )); then
  echo "GPU $GPU already has $BUSY running compute process(es); refusing to start so I don't disturb them." >&2
  echo "Free GPU $GPU first, or set GPU=<a free index> — do NOT kill the existing job." >&2
  nvidia-smi --id="$GPU"
  exit 1
fi

# Shard-aware file naming (matches scripts/run_all_judge_grading.py).
if (( NUM_SHARDS > 1 )); then
  SHARD_SUFFIX=".shard${SHARD_INDEX}"
else
  SHARD_SUFFIX=""
fi
CASES_NAME="cases${SHARD_SUFFIX}.jsonl"
VERDICT_BASE="canonical_${REPLICATE_ID}${SHARD_SUFFIX}"
VERDICT_NAME="${VERDICT_BASE}.jsonl"

echo "== multi-benchmark judge grading on GPU index $GPU =="
echo "   judge=$JUDGE  responses=$RESPONSES_ROOT  judge_root=$JUDGE_ROOT"
echo "   s3=$S3_GRADING_PREFIX  shard=$SHARD_INDEX/$NUM_SHARDS  tp=$TP"
echo "   benchmarks: $BENCHMARKS"

# --- (a) EMIT: stage per-benchmark blinded cases (shard-aware) --------------
echo "== (a) emit cases =="
EMIT_ARGS=(scripts/run_all_judge_grading.py --emit-cases-only
           --responses-root "$RESPONSES_ROOT" --out-root "$JUDGE_ROOT" --judge "$JUDGE"
           --s3-prefix "$S3_GRADING_PREFIX"
           --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX")
[[ -n "$ONLY" ]] && EMIT_ARGS+=(--only "$ONLY")
uv run "${EMIT_ARGS[@]}"

# --- (b) PUSH cases + (c) JUDGE each in-scope benchmark ---------------------
for b in $BENCHMARKS; do
  cases="$JUDGE_ROOT/$b/$CASES_NAME"
  if [[ ! -f "$cases" ]]; then
    echo "-- $b: no $cases (nothing staged for this shard); skipping"
    continue
  fi

  echo "== (b) push cases: $b =="
  aws s3 cp "$cases" "$S3_GRADING_PREFIX/$b/$CASES_NAME" --only-show-errors

  echo "== (c) judge: $b (GPU $GPU) =="
  out="$INBOX/$b/$VERDICT_NAME"
  mkdir -p "$(dirname "$out")"
  CUDA_VISIBLE_DEVICES="$GPU" uv run python aws_judge_handoff/scripts/run_judge_validation.py run \
    --cases "$cases" --judge "$JUDGE" \
    --output "$out" \
    --backend vllm --prompt-variant "$PROMPT_VARIANT" --replicate-id "$REPLICATE_ID" --resume \
    --tensor-parallel-size "$TP" \
    --s3-output-prefix "$S3_GRADING_PREFIX/$b/$VERDICT_BASE" \
    --require-s3-upload
done

# --- (d) PULL every shard's verdict into a FLAT inbox for ingest ------------
# The frozen runner uploads to <prefix>/<filename>, i.e.
# <S3_GRADING_PREFIX>/<Benchmark>/canonical_<rid>.shard<i>/canonical_<rid>.shard<i>.jsonl.
# Ingest globs <inbox>/<Benchmark>/*.jsonl (non-recursive), so download each shard
# file to that flat level.
echo "== (d) sync verdicts back down -> $INBOX =="
for b in $BENCHMARKS; do
  mkdir -p "$INBOX/$b"
  for (( i=0; i<NUM_SHARDS; i++ )); do
    if (( NUM_SHARDS > 1 )); then
      base="canonical_${REPLICATE_ID}.shard${i}"
    else
      base="canonical_${REPLICATE_ID}"
    fi
    aws s3 cp "$S3_GRADING_PREFIX/$b/$base/$base.jsonl" "$INBOX/$b/$base.jsonl" \
      --only-show-errors 2>/dev/null || true
  done
done

# --- (e) INGEST: assemble verdicts + response matrices (merges all shards) --
if [[ "$SKIP_INGEST" == "1" ]]; then
  echo "== (e) ingest SKIPPED (SKIP_INGEST=1); run a final ingest pass once all shards are up =="
  exit 0
fi
echo "== (e) ingest =="
INGEST_ARGS=(scripts/run_all_judge_grading.py --ingest "$INBOX"
             --responses-root "$RESPONSES_ROOT" --out-root "$JUDGE_ROOT" --judge "$JUDGE")
[[ -n "$ONLY" ]] && INGEST_ARGS+=(--only "$ONLY")
uv run "${INGEST_ARGS[@]}"

echo "== grading pipeline complete; per-benchmark outputs under $JUDGE_ROOT/ =="
