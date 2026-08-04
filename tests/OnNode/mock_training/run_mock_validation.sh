#!/usr/bin/env bash
# Local, no-GPU, no-AWS-cost validation of Flow 1 (on-node checkpoint inference).
#
# Spins up a pure-Python moto S3 endpoint on localhost, sets dummy AWS creds and
# the checkpoint_infer.py environment, creates the results bucket, runs the mock
# trainer (which fires the Flow 1 hook per checkpoint), then lists and validates
# the S3 outputs. No Docker, no real AWS, no network model downloads.
#
# To point the SAME mock trainer at REAL S3 instead: skip this script, export
# real AWS creds + RESULTS_BUCKET, unset S3_ENDPOINT_URL, then run
#   uv run --no-sync python tests/OnNode/mock_training/mock_training_run.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$REPO_ROOT"

PORT="${MOTO_PORT:-5001}"
UV="uv run --no-sync"

# --- Environment shared by the trainer and the checkpoint_infer.py hook ---
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-test}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-test}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export S3_ENDPOINT_URL="http://127.0.0.1:${PORT}"

export RESULTS_BUCKET="${RESULTS_BUCKET:-mock-training-bucket}"
export RUN_NAME="${RUN_NAME:-mock-run}"
export RESULTS_PREFIX="${RESULTS_PREFIX:-checkpoint-infer}"
export CHECKPOINT_KIND="hf"
export NUM_PROMPTS="${NUM_PROMPTS:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
export TEMPERATURE="0.0"
export NUM_GPUS="0"            # CPU only: disables device_map="auto" in the hook
export BLOCKING="false"       # background the hook so training is not blocked

export MOCK_NUM_STEPS="${MOCK_NUM_STEPS:-3}"
export MOCK_STEP_SIZE="${MOCK_STEP_SIZE:-100}"

MOTO_PID=""
cleanup() {
  if [[ -n "$MOTO_PID" ]] && kill -0 "$MOTO_PID" 2>/dev/null; then
    kill "$MOTO_PID" 2>/dev/null || true
    wait "$MOTO_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo ">>> starting moto S3 server on 127.0.0.1:${PORT}"
$UV moto_server -p "$PORT" >/tmp/moto_server.log 2>&1 &
MOTO_PID=$!

# Wait for the endpoint to accept connections.
for _ in $(seq 1 50); do
  if $UV python -c "import socket,sys; s=socket.socket(); s.settimeout(0.3); sys.exit(0 if s.connect_ex(('127.0.0.1',${PORT}))==0 else 1)"; then
    break
  fi
  sleep 0.2
done

echo ">>> creating bucket s3://${RESULTS_BUCKET} up front"
$UV python -c "
import boto3, os
c = boto3.client('s3', region_name=os.environ['AWS_REGION'], endpoint_url=os.environ['S3_ENDPOINT_URL'])
c.create_bucket(Bucket=os.environ['RESULTS_BUCKET'])
print('bucket ready:', os.environ['RESULTS_BUCKET'])
"

echo
echo ">>> [1/3] dry-run of checkpoint_infer.py against a fake URI"
$UV python tests/OnNode/checkpoint_infer.py \
  "s3://${RESULTS_BUCKET}/checkpoints/mockowner/${RUN_NAME}/step999/" --dry-run

echo
echo ">>> [2/3] running the mock training loop end-to-end"
$UV python tests/OnNode/mock_training/mock_training_run.py

echo
echo ">>> [3/3] validating S3 outputs"
$UV python tests/OnNode/mock_training/validate_results.py

echo
echo ">>> done"
