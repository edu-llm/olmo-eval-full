#!/usr/bin/env bash
# Launch the adaptive-inference sweep on the shared B200 node, pinned to a
# SINGLE free GPU so it never disturbs the ms-135m training job (GPUs 0,1).
#
# Nothing here runs until you invoke it. It detaches with nohup and streams to
# a timestamped log so you can disconnect safely.
#
# Env knobs (all optional):
#   GPU=6                 which single GPU index to use (default 6; free ones are 2-7)
#   BENCHMARKS=all        all | mcq | open | comma list
#   BACKEND=vllm          vllm | hf | mock
#   MAX_SAMPLES=2000      per-benchmark cap
#   ORDER=model|benchmark model-outer (efficient, 1 load/model) vs benchmark-first
#   LIMIT_MODELS=         cap number of models (smoke test)
#   NO_JUDGE=0            set 1 to skip Prometheus judging
#   S3_BUCKET=s3://edullm-adaptive-inference-056956104102   output bucket
#   S3_PREFIX=outputs     key prefix under the bucket for this sweep's Outputs tree
#   SYNC_INTERVAL=60      seconds between incremental S3 syncs of partial results
#   S3_REGION=us-east-1   region for the output bucket / instance-role S3 calls
#   NO_S3=0               set 1 to disable S3 sync entirely (local-only run)
set -euo pipefail

ROOT="/opt/dlami/nvme/adaptive-inference"
VENV="${ROOT}/venv"
CODE="${ROOT}/code/Test/Inference"
export HF_HOME="${ROOT}/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

GPU="${GPU:-6}"
BENCHMARKS="${BENCHMARKS:-all}"
BACKEND="${BACKEND:-vllm}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
ORDER="${ORDER:-model}"
NO_JUDGE="${NO_JUDGE:-0}"

# --- S3 output sync. This script runs ON the p6 node, so S3 calls use the node's
# instance role (the same role used for Secrets Manager below). The runner writes
# Outputs locally (durable per-row); we mirror them to S3 so partial progress
# survives instance loss and is retrievable off-box.
S3_BUCKET="${S3_BUCKET:-s3://edullm-adaptive-inference-056956104102}"
S3_PREFIX="${S3_PREFIX:-outputs}"
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"
S3_REGION="${S3_REGION:-us-east-1}"
NO_S3="${NO_S3:-0}"
S3_DEST="${S3_BUCKET%/}/${S3_PREFIX}"

export CUDA_VISIBLE_DEVICES="${GPU}"   # <-- hard 1-GPU pin

if [[ ! -x "${VENV}/bin/python" || ! -f "${ROOT}/.bootstrap_ok" ]]; then
  echo "error: env not bootstrapped; run node_bootstrap.sh first" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# --- HF token: fetch from Secrets Manager via the instance role (boto3), with
# an aws-CLI fallback and finally a local .hf_env file. Value never printed.
if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$(python - <<'PY' 2>/dev/null || true
import boto3, sys
try:
    v = boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(
        SecretId="hf-token")["SecretString"]
    sys.stdout.write(v.strip())
except Exception:
    pass
PY
)"
fi
if [[ -z "${HF_TOKEN:-}" ]] && command -v aws >/dev/null 2>&1; then
  HF_TOKEN="$(aws secretsmanager get-secret-value --secret-id hf-token \
    --query SecretString --output text --region us-east-1 2>/dev/null || true)"
fi
if [[ -z "${HF_TOKEN:-}" && -f "${ROOT}/.hf_env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.hf_env"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  echo "HF token loaded (len ${#HF_TOKEN})"
else
  echo "WARN: no HF token; gated models will be skipped/fail" >&2
fi

ORDER_FLAG=()
[[ "${ORDER}" == "benchmark" ]] && ORDER_FLAG+=(--resident-all)
JUDGE_FLAG=()
[[ "${NO_JUDGE}" == "1" ]] && JUDGE_FLAG+=(--no-judge)
LIMIT_FLAG=()
[[ -n "${LIMIT_MODELS:-}" ]] && LIMIT_FLAG+=(--limit-models "${LIMIT_MODELS}")

TS="$(date -u +%Y%m%d-%H%M%S)"
LOG="${ROOT}/logs/sweep_${TS}.log"
cd "${CODE}"

# Local Outputs tree the runner writes to (AdaptiveTesting/Outputs; see common.py).
OUT_LOCAL="$(cd "${CODE}/../.." && pwd)/Outputs"
mkdir -p "${OUT_LOCAL}"

# S3 sync needs the aws CLI; fall back to local-only if it is missing.
if [[ "${NO_S3}" != "1" ]] && ! command -v aws >/dev/null 2>&1; then
  echo "WARN: aws CLI not found; disabling S3 sync (outputs stay in ${OUT_LOCAL})" >&2
  NO_S3=1
fi

# Assemble the runner command once so the detached block below stays readable.
RUN_CMD=(python run_benchmark.py
  --benchmarks "${BENCHMARKS}"
  --backend "${BACKEND}"
  --max-samples "${MAX_SAMPLES}"
  --inference-config "${CODE}/configs/inference.node.yaml"
  --judge-config "${CODE}/configs/judge.yaml"
  "${ORDER_FLAG[@]}" "${JUDGE_FLAG[@]}" "${LIMIT_FLAG[@]}")

echo "launching sweep: gpu=${GPU} benchmarks=${BENCHMARKS} backend=${BACKEND} order=${ORDER}"
if [[ "${NO_S3}" == "1" ]]; then
  echo "S3 sync: DISABLED; outputs stay in ${OUT_LOCAL}"
else
  echo "S3 sync: ${OUT_LOCAL} <-> ${S3_DEST} (region ${S3_REGION}, every ${SYNC_INTERVAL}s)"
fi
echo "log -> ${LOG}"

# Detach the whole unit (resume-pull + periodic sync + sweep + aggregate + final
# sync) so the S3 mirroring stays tied to the sweep's lifetime and is NOT killed
# when this launcher returns.
nohup bash -c '
  set -uo pipefail
  out_local="$1"; s3_dest="$2"; s3_region="$3"; sync_interval="$4"; no_s3="$5"
  shift 5

  sync_up() { aws s3 sync "${out_local}/" "${s3_dest}/" --region "${s3_region}" --only-show-errors || true; }

  if [[ "${no_s3}" != "1" ]]; then
    # Resume: pull any prior partial outputs for this sweep before starting.
    aws s3 sync "${s3_dest}/" "${out_local}/" --region "${s3_region}" --only-show-errors || true
    ( while true; do sleep "${sync_interval}"; sync_up; done ) &
    sync_pid=$!
    trap "kill ${sync_pid} 2>/dev/null || true; sync_up" EXIT
  fi

  "$@"; rc=$?
  python aggregate.py || true
  exit ${rc}
' _ "${OUT_LOCAL}" "${S3_DEST}" "${S3_REGION}" "${SYNC_INTERVAL}" "${NO_S3}" \
    "${RUN_CMD[@]}" >"${LOG}" 2>&1 &

echo "PID $! (tail -f ${LOG})"
if [[ "${NO_S3}" != "1" ]]; then
  echo "outputs -> ${S3_DEST}/  (mcq/ open/ _manifests/ _summary/)"
fi
