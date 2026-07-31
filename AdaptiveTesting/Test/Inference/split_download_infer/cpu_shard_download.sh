#!/usr/bin/env bash
# One CPU downloader shard. Streams its zig-zag slice of the roster to S3, one
# model at a time, writing a per-model READY marker the moment each model (and
# its borrowed tokenizer) is fully in S3. Disk stays flat (prefetch --delete-after).
#
# Sharding is by the instance's EC2 ami-launch-index, so a single `run-instances
# --count N` fans out into N shards with no per-instance config.
#
# Env:
#   RUN_BASE     s3 run prefix     (default s3://edullm-adaptive-inference-056956104102/full200)
#   REGION       AWS region        (default us-east-1)
#   ROOT         local prefix      (default /opt/adaptive-dl)
#   NUM_SHARDS   fleet size        (default 16)
#   SHARD_INDEX  this shard        (default: ami-launch-index)
#   WORKERS      file parallelism  (default 4)
#   MODELS_YAML  roster            (default models_200.yaml in the tree)
#   RUN_DEADLINE_EPOCH  unix ts; if set, exit+shutdown when reached (budget cap)
set -uo pipefail

RUN_BASE="${RUN_BASE:-s3://edullm-adaptive-inference-056956104102/full200}"
REGION="${REGION:-us-east-1}"
ROOT="${ROOT:-/opt/adaptive-dl}"
NUM_SHARDS="${NUM_SHARDS:-16}"
WORKERS="${WORKERS:-4}"
CACHE="${RUN_BASE}/hf-cache"
RUN_DEADLINE_EPOCH="${RUN_DEADLINE_EPOCH:-}"

budget_ok() {
  [[ -z "${RUN_DEADLINE_EPOCH}" ]] && return 0
  (( $(date -u +%s) < RUN_DEADLINE_EPOCH ))
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INF="$(cd "${HERE}/.." && pwd)"
MODELS_YAML="${MODELS_YAML:-${INF}/../../Inputs/Models/models_200.yaml}"

if [[ -z "${SHARD_INDEX:-}" ]]; then
  _tok="$(curl -s -X PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' 2>/dev/null || true)"
  SHARD_INDEX="$(curl -s -H "X-aws-ec2-metadata-token: ${_tok}" http://169.254.169.254/latest/meta-data/ami-launch-index 2>/dev/null || echo 0)"
  [[ -z "${SHARD_INDEX}" ]] && SHARD_INDEX=0
fi
export HF_HOME="${ROOT}/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}" "${ROOT}/logs"
# shellcheck disable=SC1091
[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate" || true
[[ -f "${ROOT}/.hf_token" ]] && export HF_TOKEN="$(cat "${ROOT}/.hf_token")" && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

LOG="${ROOT}/logs/cpu-${SHARD_INDEX}.log"
exec > >(tee -a "${LOG}") 2>&1
hb() { aws s3 cp "${LOG}" "${RUN_BASE}/logs/cpu-${SHARD_INDEX}.log" --region "${REGION}" --only-show-errors 2>/dev/null || true; }

echo "[cpu ${SHARD_INDEX}/${NUM_SHARDS} $(date -u +%H:%M:%S)] start; roster=${MODELS_YAML}"

# Shard 0 also builds the 3 MCQ item caches (HF -> normalized JSONL) once and
# publishes them, so GPU workers can score MCQ fully offline.
if [[ "${SHARD_INDEX}" == "0" ]]; then
  echo "[cpu 0 $(date -u +%H:%M:%S)] building MCQ caches (pedagogy/piqa/socialiqa)"
  ( cd "${INF}" && python - <<'PY'
from datasets_registry import load_benchmark
for n in ("pedagogy", "piqa", "socialiqa"):
    try:
        items = load_benchmark(n, 2000, 1234, use_cache=True)
        print(f"  MCQ cache {n}: {len(items)} items")
    except Exception as e:
        print(f"  MCQ cache {n} FAILED: {type(e).__name__}: {e}")
PY
  ) || true
  MCQ_DIR="${INF}/../../Inputs/MCQ/Benchmarks"
  if compgen -G "${MCQ_DIR}/*.jsonl" >/dev/null; then
    aws s3 sync "${MCQ_DIR}/" "${RUN_BASE}/mcq_cache/" --region "${REGION}" --only-show-errors
    date -u +%Y%m%dT%H%M%SZ | aws s3 cp - "${RUN_BASE}/_MCQ_READY" --region "${REGION}" --only-show-errors
    echo "[cpu 0 $(date -u +%H:%M:%S)] MCQ caches published"
  else
    echo "[cpu 0 $(date -u +%H:%M:%S)] WARNING: no MCQ caches produced"
  fi
  hb
fi

mapfile -t ROWS < <(python "${HERE}/order_models.py" --models-yaml "${MODELS_YAML}" \
  --shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}")
echo "[cpu ${SHARD_INDEX} $(date -u +%H:%M:%S)] ${#ROWS[@]} model(s) in this shard"

done_n=0
for row in "${ROWS[@]}"; do
  if ! budget_ok; then
    echo "[cpu ${SHARD_INDEX} $(date -u +%H:%M:%S)] BUDGET_DEADLINE reached -> stop shard"
    hb
    exit 42
  fi
  idx="$(cut -f1 <<<"$row")"; mid="$(cut -f2 <<<"$row")"; cd="$(cut -f3 <<<"$row")"
  if aws s3 ls "${RUN_BASE}/ready/${cd}" --region "${REGION}" >/dev/null 2>&1; then
    echo "[cpu ${SHARD_INDEX}] skip (ready) #${idx} ${mid}"; done_n=$((done_n+1)); continue
  fi
  echo "[cpu ${SHARD_INDEX} $(date -u +%H:%M:%S)] fetch #${idx} ${mid}"
  if python "${HERE}/prefetch_weights.py" --models "${mid}" --models-yaml "${MODELS_YAML}" \
       --workers "${WORKERS}" --s3-prefix "${CACHE}" --region "${REGION}" --delete-after; then
    date -u +%Y%m%dT%H%M%SZ | aws s3 cp - "${RUN_BASE}/ready/${cd}" --region "${REGION}" --only-show-errors
    done_n=$((done_n+1))
    echo "[cpu ${SHARD_INDEX} $(date -u +%H:%M:%S)] READY #${idx} ${mid} (${done_n}/${#ROWS[@]})"
  else
    echo "[cpu ${SHARD_INDEX} $(date -u +%H:%M:%S)] FAIL #${idx} ${mid} (left for a retry/relaunch)"
  fi
  hb
done

date -u +%Y%m%dT%H%M%SZ | aws s3 cp - "${RUN_BASE}/_SHARD_DONE/${SHARD_INDEX}" --region "${REGION}" --only-show-errors
echo "[cpu ${SHARD_INDEX} $(date -u +%H:%M:%S)] shard complete: ${done_n}/${#ROWS[@]} ready"
hb
