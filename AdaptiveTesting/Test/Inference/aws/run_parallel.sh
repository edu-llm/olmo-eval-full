#!/usr/bin/env bash
# Parallel, benchmark-by-benchmark sweep.
#
# Ordering: the fleet finishes one benchmark across ALL models before starting
# the next. Within a benchmark, the model list is split into WORKERS shards that
# run concurrently, so several models are doing inference at the same time.
#
#   for benchmark in <list>:
#       launch WORKERS workers (shard i of WORKERS)   <- concurrent
#       wait for all                                  <- barrier
#   then: one Prometheus judging pass over all open-ended responses
#
# Open-ended generation runs with judging disabled; a single resident judge
# handles all of it afterwards (judge_all.py). One judge instead of WORKERS
# judges is what keeps everything inside one GPU.
#
# Env knobs:
#   GPUS=2              comma list of GPU indices to spread workers over
#   WORKERS=6           concurrent worker processes (total, across GPUS)
#   MAX_SAMPLES=2000    per-benchmark cap
#   MCQ_BENCHMARKS=...  override MCQ list
#   OPEN_BENCHMARKS=... override open-ended list
#   SKIP_JUDGE=0        set 1 to generate open responses but not judge
#   DRY_RUN=0           set 1 to print the plan and exit
set -uo pipefail

ROOT="/opt/dlami/nvme/adaptive-inference"
VENV="${ROOT}/venv"
CODE="${CODE_DIR:-${ROOT}/code}/Test/Inference"
export HF_HOME="${ROOT}/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

GPUS="${GPUS:-2}"
WORKERS="${WORKERS:-6}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Initial benchmark list minus the four already covered by the MCQ job
# (arc_easy, arc_challenge, openbookqa, sciq), plus pedagogy.
MCQ_BENCHMARKS="${MCQ_BENCHMARKS:-hellaswag,piqa,boolq,winogrande,mathqa,educationq,pedagogy}"
OPEN_BENCHMARKS="${OPEN_BENCHMARKS:-squad_v2,svamp,mathdial,tutoreval,tutorbench,edubench}"

CFG="${CODE}/configs/inference.parallel.yaml"
JCFG="${CODE}/configs/judge.yaml"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOGDIR="${ROOT}/logs/parallel_${STAMP}"
mkdir -p "${LOGDIR}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPU="${#GPU_ARR[@]}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "${LOGDIR}/coordinator.log"; }

log "gpus=${GPUS} workers=${WORKERS} max_samples=${MAX_SAMPLES}"
log "mcq: ${MCQ_BENCHMARKS}"
log "open: ${OPEN_BENCHMARKS}"
log "logs: ${LOGDIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1 - plan only, exiting"
  exit 0
fi

if [[ ! -x "${VENV}/bin/python" || ! -f "${ROOT}/.bootstrap_ok" ]]; then
  echo "error: env not bootstrapped" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# CUDA toolkit lives in the pip wheels; needed by any JIT path.
NVCC_BIN=$(find "${VENV}"/lib -name nvcc -type f 2>/dev/null | head -1)
if [ x"$NVCC_BIN" != x ]; then
  export CUDA_HOME=$(dirname "$(dirname "$NVCC_BIN")")
  export PATH=$(dirname "$NVCC_BIN"):$PATH
fi
export VLLM_ATTN_BACKEND="${VLLM_ATTN_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0

# HF token via the instance role; value is never echoed.
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
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  log "HF token loaded (len ${#HF_TOKEN})"
else
  log "WARN: no HF token; gated datasets/models will be skipped"
fi

cd "${CODE}"

ALL_BENCH="${MCQ_BENCHMARKS},${OPEN_BENCHMARKS}"

# Warm the normalized-dataset cache once, single-process. Without this the
# workers would race to download and write the same cache files.
log "warming dataset cache (single process)"
python - "$ALL_BENCH" "$MAX_SAMPLES" <<'PY' 2>&1 | tee -a "${LOGDIR}/warm.log"
import sys
from datasets_registry import load_benchmark

names = [b for b in sys.argv[1].split(",") if b]
cap = int(sys.argv[2])
usable = []
for b in names:
    try:
        qs = load_benchmark(b, cap, 1234, use_cache=True)
        print(f"  [warm ok] {b}: {len(qs)} items", flush=True)
        usable.append(b)
    except Exception as exc:
        print(f"  [warm FAIL] {b}: {type(exc).__name__}: {exc}", flush=True)
print("USABLE=" + ",".join(usable), flush=True)
PY

USABLE="$(grep -h '^USABLE=' "${LOGDIR}/warm.log" | tail -1 | cut -d= -f2-)"
if [[ -z "${USABLE}" ]]; then
  log "no usable benchmarks after warm-up; aborting"
  exit 1
fi
log "usable benchmarks: ${USABLE}"

run_benchmark_parallel() {
  local bench="$1"
  local pids=()
  local i gpu
  for (( i=0; i<WORKERS; i++ )); do
    gpu="${GPU_ARR[$(( i % NGPU ))]}"
    CUDA_VISIBLE_DEVICES="${gpu}" python run_benchmark.py \
      --benchmarks "${bench}" \
      --shard-index "${i}" --num-shards "${WORKERS}" \
      --backend vllm \
      --max-samples "${MAX_SAMPLES}" \
      --no-judge \
      --inference-config "${CFG}" \
      --judge-config "${JCFG}" \
      > "${LOGDIR}/${bench}_w${i}.log" 2>&1 &
    pids+=("$!")
  done
  local rc=0
  for p in "${pids[@]}"; do
    wait "$p" || rc=1
  done
  return $rc
}

IFS=',' read -r -a USABLE_ARR <<< "${USABLE}"
for bench in "${USABLE_ARR[@]}"; do
  log "=== benchmark ${bench}: launching ${WORKERS} workers ==="
  t0=$SECONDS
  if run_benchmark_parallel "${bench}"; then
    log "=== benchmark ${bench}: complete in $((SECONDS - t0))s ==="
  else
    log "=== benchmark ${bench}: finished WITH ERRORS in $((SECONDS - t0))s (see ${LOGDIR}/${bench}_w*.log) ==="
  fi
done

if [[ "${SKIP_JUDGE}" != "1" ]]; then
  log "=== judging pass (single resident judge) ==="
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python judge_all.py \
    --benchmarks "${OPEN_BENCHMARKS}" \
    --backend vllm \
    --inference-config "${CFG}" \
    --judge-config "${JCFG}" \
    > "${LOGDIR}/judge_all.log" 2>&1
  log "judging exit=$?"
fi

log "ALL DONE. logs in ${LOGDIR}"
