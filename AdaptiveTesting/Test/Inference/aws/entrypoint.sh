#!/usr/bin/env bash
# In-container entrypoint for one AWS Batch array task.
#
# Reads AWS_BATCH_JOB_ARRAY_INDEX to pick this worker's model shard, runs the
# sweep (benchmarks in a fixed order so the fleet advances benchmark-first),
# and syncs outputs to S3 after each pair (the runner writes incrementally;
# this loop provides periodic durability off-box).
#
# Required env:
#   S3_BUCKET            e.g. s3://my-bucket/adaptivetesting
#   NUM_SHARDS           total array size (usually = number of models)
# Optional env:
#   BENCHMARKS           default "all"
#   BACKEND              default "vllm"
#   MAX_SAMPLES          per-benchmark cap
#   HF_TOKEN             for gated models
#   SYNC_INTERVAL        seconds between S3 syncs (default 60)
set -euo pipefail

cd /app/Inference

SHARD_INDEX="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
BENCHMARKS="${BENCHMARKS:-all}"
BACKEND="${BACKEND:-vllm}"
S3_BUCKET="${S3_BUCKET:?set S3_BUCKET}"
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"

OUT_LOCAL="/cache/Outputs"
mkdir -p "${OUT_LOCAL}"

# Pull any prior outputs for this shard so we resume instead of recompute.
aws s3 sync "${S3_BUCKET}/outputs/" "${OUT_LOCAL}/" --quiet || true

# Background periodic sync of partial results to S3.
(
  while true; do
    sleep "${SYNC_INTERVAL}"
    aws s3 sync "${OUT_LOCAL}/" "${S3_BUCKET}/outputs/" --quiet || true
  done
) &
SYNC_PID=$!
trap 'kill ${SYNC_PID} 2>/dev/null || true; aws s3 sync "${OUT_LOCAL}/" "${S3_BUCKET}/outputs/" --quiet || true' EXIT

extra=()
[[ -n "${MAX_SAMPLES:-}" ]] && extra+=(--max-samples "${MAX_SAMPLES}")

python3.12 run_benchmark.py \
  --benchmarks "${BENCHMARKS}" \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  --backend "${BACKEND}" \
  "${extra[@]}"

python3.12 aggregate.py

# Final sync happens via the EXIT trap.
echo "shard ${SHARD_INDEX}/${NUM_SHARDS} complete"
