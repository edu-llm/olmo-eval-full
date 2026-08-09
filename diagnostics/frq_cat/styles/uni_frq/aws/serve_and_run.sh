#!/usr/bin/env bash
# On-node entrypoint for the uni_frq FRQ CAT flow: serve the tutor, wait until it is
# actually answering, run the pipeline, tear the servers down.
#
# This deliberately does NOT fork tests/aws/run_checkpoint_diag.sh. That script owns EC2
# lifecycle (launch, SSM, teardown, billing guard) and none of that is FRQ-specific. The
# only thing it is missing for FRQ is that it never brings up the model endpoints, so this
# script fills exactly that gap and nothing else. Point the shared launcher at this file,
# or run it directly on any box that already has the repo and its dependencies.
#
#   CHECKPOINT=allenai/OLMo-2-1124-7B-Instruct \
#   OUT=s3://bucket/frq_cat \
#   JUDGE_API_KEY_ENV=OPENAI_API_KEY \
#   diagnostics/frq_cat/styles/uni_frq/aws/serve_and_run.sh
#
# JUDGE_MODE=api (default) keeps the judge on a hosted API, so only the tutor needs a GPU
# and a single 24 GB instance is enough. JUDGE_MODE=local serves the frozen Qwen judge on a
# second port, which needs a second GPU (the 9B judge alone is ~18 GB in bf16).
#
# Cost safety: every server is started in its own process group and the group is killed on
# any exit, the pipeline runs under a hard timeout, and a port that is already serving is
# refused rather than mistaken for our own model.
set -euo pipefail
set -m  # own process group per background job, so teardown can kill the whole tree

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STYLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"

# ---- required ---------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT (HF id or local path of the model under test)}"
OUT="${OUT:?set OUT (s3://bucket/prefix or a local directory)}"

# ---- tutor ------------------------------------------------------------------------
PYBIN="${PYBIN:-python3}"
TUTOR_PORT="${TUTOR_PORT:-8000}"
TUTOR_SERVED_NAME="${TUTOR_SERVED_NAME:-tutor}"
# 16384 is a prerequisite, not a preference: 16.3% of TutorEval scenarios have a prompt
# that alone exceeds ~90% of a 4096-token window (see FLOW_PACKAGE.md).
TUTOR_MAX_MODEL_LEN="${TUTOR_MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"

# ---- judge ------------------------------------------------------------------------
JUDGE_MODE="${JUDGE_MODE:-api}"                                  # api | local
JUDGE_CONFIG="${JUDGE_CONFIG:-${STYLE_DIR}/judge_frontier.yaml}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-}"        # api mode: else taken from the judge config
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-}"  # env var NAME holding the key, never the key
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-9B}"   # local mode only
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"

# ---- run --------------------------------------------------------------------------
MAX_ITEMS="${MAX_ITEMS:-40}"
SE_THRESHOLD="${SE_THRESHOLD:-0.3}"
RUN_ID="${RUN_ID:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
LOG_DIR="${LOG_DIR:-${TMPDIR:-/tmp}/uni_frq_logs}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"          # seconds to wait for weights to load
PIPELINE_TIMEOUT="${PIPELINE_TIMEOUT:-10800}"   # hard cap on the CAT run itself
VLLM_CMD="${VLLM_CMD:-vllm serve}"              # overridden by the test with a stub server
DRY_RUN="${DRY_RUN:-false}"

PIDS=""
LAST_PID=""

die() { echo "!! $*" >&2; exit 1; }
log() { echo ">> $*"; }

cleanup() {
  local rc=$? pid
  for pid in ${PIDS}; do
    # Kill the whole process group: vLLM spawns engine/worker children that would
    # otherwise survive and keep holding the GPU (and the bill).
    kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  done
  if [ -n "${PIDS}" ]; then
    sleep 2
    for pid in ${PIDS}; do
      kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    done
  fi
  return "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

port_is_open() {  # port
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3>&- && return 0
  return 1
}

start_server() {  # name model port max_model_len
  local name="$1" model="$2" port="$3" maxlen="$4"
  if port_is_open "${port}"; then
    die "port ${port} is already in use; refusing to start ${name} (a stale server there would be mistaken for ours)"
  fi
  log "starting ${name} (${model}) on :${port} with max_model_len=${maxlen}"
  # VLLM_CMD is intentionally word-split so it can be "vllm serve" or a test stub.
  # shellcheck disable=SC2086
  ${VLLM_CMD} "${model}" \
    --served-model-name "${name}" \
    --port "${port}" \
    --max-model-len "${maxlen}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    >"${LOG_DIR}/${name}.log" 2>&1 &
  LAST_PID="$!"
  PIDS="${PIDS} ${LAST_PID}"
}

wait_ready() {  # name port pid
  local name="$1" port="$2" pid="$3" waited=0
  local url="http://127.0.0.1:${port}/v1/models"
  while [ "${waited}" -lt "${READY_TIMEOUT}" ]; do
    if curl -sf "${url}" >/dev/null 2>&1; then
      # Confirm it is *our* model answering, not something else on the port.
      if curl -sf "${url}" 2>/dev/null | grep -q "${name}"; then
        log "${name} ready after ${waited}s"
        return 0
      fi
      die "port ${port} answers /v1/models but does not serve '${name}'"
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "---- ${name} log tail ----" >&2
      tail -n 40 "${LOG_DIR}/${name}.log" >&2 || true
      die "${name} died before becoming ready"
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "---- ${name} log tail ----" >&2
  tail -n 40 "${LOG_DIR}/${name}.log" >&2 || true
  die "${name} not ready within ${READY_TIMEOUT}s"
}

mkdir -p "${LOG_DIR}"

# ---- preflight: fail before paying for a GPU, not 30 minutes into the run ----------
command -v curl >/dev/null 2>&1 || die "curl is required for the readiness probe"
command -v "${PYBIN}" >/dev/null 2>&1 || die "PYBIN not executable: ${PYBIN}"
set -- ${VLLM_CMD}
command -v "$1" >/dev/null 2>&1 || die "VLLM_CMD not executable: $1"
[ -f "${JUDGE_CONFIG}" ] || die "judge config not found: ${JUDGE_CONFIG}"
[ -d "${REPO_DIR}/diagnostics/frq_cat" ] || die "REPO_DIR does not contain diagnostics/frq_cat: ${REPO_DIR}"

if [ "${TUTOR_MAX_MODEL_LEN}" -lt 16384 ]; then
  echo "!! warning: TUTOR_MAX_MODEL_LEN=${TUTOR_MAX_MODEL_LEN} < 16384;" \
       "long scenarios will be skipped rather than scored" >&2
fi

log "repo=${REPO_DIR} checkpoint=${CHECKPOINT} judge_mode=${JUDGE_MODE} out=${OUT}"

if [ "${DRY_RUN}" = true ]; then
  log "[dry-run] tutor :${TUTOR_PORT}; judge_mode=${JUDGE_MODE}; config=${JUDGE_CONFIG}"
  log "[dry-run] judge_endpoint=${JUDGE_ENDPOINT:-<from config>}; pipeline_timeout=${PIPELINE_TIMEOUT}s"
  exit 0
fi

start_server "${TUTOR_SERVED_NAME}" "${CHECKPOINT}" "${TUTOR_PORT}" "${TUTOR_MAX_MODEL_LEN}"
TUTOR_PID="${LAST_PID}"
wait_ready "${TUTOR_SERVED_NAME}" "${TUTOR_PORT}" "${TUTOR_PID}"

if [ "${JUDGE_MODE}" = local ]; then
  # Started only after the tutor is up, so a single-GPU box fails fast on the tutor
  # instead of OOMing both models at once.
  start_server "judge" "${JUDGE_MODEL}" "${JUDGE_PORT}" "${JUDGE_MAX_MODEL_LEN}"
  JUDGE_PID="${LAST_PID}"
  wait_ready "judge" "${JUDGE_PORT}" "${JUDGE_PID}"
  JUDGE_ENDPOINT="http://127.0.0.1:${JUDGE_PORT}/v1"
  JUDGE_CONFIG="${JUDGE_CONFIG_LOCAL:-${STYLE_DIR}/judge_frozen.yaml}"
fi

set -- \
  --checkpoint "${CHECKPOINT}" \
  --tutor-endpoint "http://127.0.0.1:${TUTOR_PORT}/v1" \
  --served-model "${TUTOR_SERVED_NAME}" \
  --judge-config "${JUDGE_CONFIG}" \
  --out "${OUT}" \
  --max-items "${MAX_ITEMS}" \
  --se-threshold "${SE_THRESHOLD}" \
  --aws-region "${AWS_REGION}"
if [ -n "${JUDGE_ENDPOINT}" ]; then set -- "$@" --judge-endpoint "${JUDGE_ENDPOINT}"; fi
if [ -n "${JUDGE_API_KEY_ENV}" ]; then set -- "$@" --judge-api-key-env "${JUDGE_API_KEY_ENV}"; fi
if [ -n "${RUN_ID}" ]; then set -- "$@" --run-id "${RUN_ID}"; fi

TIMEOUT_PREFIX=""
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_PREFIX="timeout ${PIPELINE_TIMEOUT}"
else
  echo "!! warning: 'timeout' not found; the pipeline runs without a hard time cap" >&2
fi

log "running the FRQ CAT pipeline (cap ${PIPELINE_TIMEOUT}s)"
cd "${REPO_DIR}"
set +e
# shellcheck disable=SC2086
${TIMEOUT_PREFIX} "${PYBIN}" -m diagnostics.frq_cat.styles.uni_frq.run_uni_frq "$@"
RC=$?
set -e
if [ "${RC}" -eq 124 ]; then
  echo "!! pipeline exceeded PIPELINE_TIMEOUT=${PIPELINE_TIMEOUT}s and was killed" >&2
fi
if [ "${RC}" -ne 0 ]; then
  for name in "${TUTOR_SERVED_NAME}" judge; do
    [ -f "${LOG_DIR}/${name}.log" ] || continue
    echo "---- ${name} log tail ----" >&2
    tail -n 30 "${LOG_DIR}/${name}.log" >&2 || true
  done
fi
log "pipeline exit=${RC}"
exit "${RC}"
