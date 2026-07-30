#!/usr/bin/env bash
# Print / optionally launch NUM_SHARDS CPU workers.
#
# Example (print commands only):
#   NUM_SHARDS=8 S3_URI=s3://edullm-adaptive-inference-056956104102/cpu_sweep_200 \
#     ./launch_fleet.sh
#
# Example (local multi-process on one fat CPU box — 4 processes):
#   LOCAL=1 NUM_SHARDS=4 S3_URI=... ./launch_fleet.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_SHARDS="${NUM_SHARDS:-8}"
LOCAL="${LOCAL:-0}"
S3_URI="${S3_URI:-}"
ROOT="${ROOT:-}"

echo "# CPU sweep fleet plan: NUM_SHARDS=${NUM_SHARDS}"
echo "# Set on every machine: same NUM_SHARDS, unique SHARD_INDEX, same S3_URI"
echo

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  cmd="SHARD_INDEX=${i} NUM_SHARDS=${NUM_SHARDS}"
  [[ -n "${S3_URI}" ]] && cmd+=" S3_URI=${S3_URI}"
  [[ -n "${ROOT}" ]] && cmd+=" ROOT=${ROOT}"
  cmd+=" bash ${SCRIPT_DIR}/run_cpu_worker.sh"
  echo "# --- shard ${i} ---"
  echo "${cmd}"
  if [[ "${LOCAL}" == "1" ]]; then
    eval "${cmd}" >"${ROOT:-.}/logs_shard${i}.out" 2>&1 &
    echo "  started pid=$!"
  fi
  echo
done

if [[ "${LOCAL}" == "1" ]]; then
  echo "waiting for local shards …"
  wait
fi
