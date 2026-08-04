#!/usr/bin/env bash
#
# rung1_smoketest.sh — on-NODE helper for the olmo-eval "Rung 1" smoke test.
#
# Runs ON the GPU EC2 instance (the sbsandbox g5.xlarge), NOT on your laptop.
# It installs olmo-eval with uv (vLLM comes from the default groups) and runs a
# tiny built-in eval, uploading results to the team S3 bucket.
#
# See RUNBOOK_rung1_smoketest.md (same folder) for the full context, how to get
# AWS access via the sb-aws-creds broker, and how to launch/stage/teardown.
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW IT GETS INVOKED (from your laptop, via SSM, fully detached):
#
#   aws ssm send-command \
#     --instance-ids i-XXXXXXXXXXXXXXXXX \
#     --document-name AWS-RunShellScript \
#     --parameters 'commands=["setsid nohup bash /opt/dlami/nvme/rung1/olmo-eval-full/eduLLM-Evals/scripts/aws/rung1_smoketest.sh > /opt/dlami/nvme/rung1/smoketest.log 2>&1 &"]' \
#     --profile sbsandbox --region us-east-1
#
# `setsid nohup ... &` fully detaches so SSM doesn't hang at InProgress.
# Watch progress by tailing /opt/dlami/nvme/rung1/smoketest.log via a separate
# SSM command — do NOT wait on the SSM status.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config (override by exporting before launch; defaults match the RUNBOOK) ──
# The team bucket already exists. Leave as-is unless you know why you're changing it.
S3_BUCKET="${S3_BUCKET:-edullm-adaptive-inference-056956104102}"
S3_PREFIX="${S3_PREFIX:-smoketest}"
S3_GROUP="${S3_GROUP:-rung1}"
S3_REGION="${S3_REGION:-us-east-1}"

# Cheapest thing that works: ~0.5B public model + small logprob-scored MCQ task,
# capped to 10 instances via the task override `-o limit=10`.
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TASK="${TASK:-arc_easy}"
LIMIT="${LIMIT:-10}"

# Where the repo was staged (Section 5 of the RUNBOOK) and where local outputs go.
REPO_DIR="${REPO_DIR:-/opt/dlami/nvme/rung1/olmo-eval-full}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/dlami/nvme/rung1/results}"

echo "=== Rung 1 smoke test ==="
echo "repo:    $REPO_DIR"
echo "model:   $MODEL"
echo "task:    $TASK (limit=$LIMIT)"
echo "S3 dest: s3://$S3_BUCKET/$S3_PREFIX/$S3_GROUP/ (region $S3_REGION)"
echo "=========================="

# ── Sanity: repo present? ─────────────────────────────────────────────────────
if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: repo not found at $REPO_DIR." >&2
  echo "Stage it first (RUNBOOK Section 5: S3 + presigned URL + SSM), or git clone" >&2
  echo "https://github.com/allenai/olmo-eval into that path." >&2
  exit 1
fi
cd "$REPO_DIR"

# ── Install uv if missing, then sync deps from the lockfile ──────────────────
# The `vllm` group is in the project default-groups, so `uv sync --frozen`
# installs vLLM (Linux/CUDA-only) + torch + transformers, plus boto3/smart_open
# (the s3 extra) needed for S3 uploads. Python 3.12+ is required.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Ensuring Python 3.12 + syncing dependencies (this can take a few minutes)..."
uv python install 3.12
uv sync --frozen

echo "olmo-eval CLI check:"
uv run olmo-eval --help | head -n 20

# ── Run the eval → upload to S3 ───────────────────────────────────────────────
# NOTE: `-o limit=$LIMIT` MUST come right after `-t $TASK` (override applies to
# the preceding -t). All three S3 flags are required together. --num-gpus
# defaults to 1, so we don't pass it on a single-GPU box.
echo "Launching eval..."
uv run olmo-eval run \
  -m "$MODEL" \
  -t "$TASK" -o "limit=$LIMIT" \
  -O "$OUTPUT_DIR" \
  --s3-bucket "$S3_BUCKET" \
  --s3-prefix "$S3_PREFIX" \
  --s3-group "$S3_GROUP" \
  --s3-region "$S3_REGION"

echo "=== DONE. Verify results with (from your laptop): ==="
echo "aws s3 ls s3://$S3_BUCKET/$S3_PREFIX/$S3_GROUP/ --recursive --profile sbsandbox --region $S3_REGION"
echo "Then TERMINATE the instance (RUNBOOK Section 8)."
