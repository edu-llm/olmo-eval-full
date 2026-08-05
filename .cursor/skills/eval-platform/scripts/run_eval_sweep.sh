#!/usr/bin/env bash
# Evaluate a sweep of training checkpoints on S3 across full benchmarks.
#
#   discover checkpoints -> per checkpoint:
#     fetch -> convert if needed -> eval every benchmark -> upload -> _READY/_FAILED
#   ...then aggregate every checkpoint into accuracy.csv.
#
# Usage:
#   run_eval_sweep.sh --checkpoint-root s3://B/checkpoints/EXP \
#     --s3-out s3://B/evals/EXP [--group NAME | --benchmarks "a b c"]
#
# ---------------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------------
# Required:
#   --s3-out s3://BUCKET/PREFIX     where results are written
#   and one of:
#   --checkpoint-root s3://.../checkpoints
#                                   a prefix whose immediate children are the
#                                   checkpoints (stepN/ ...). A local directory
#                                   works too; children are found with find.
#   --checkpoint PATH               exactly one checkpoint, s3:// or local. For
#                                   several, use --checkpoint-root. (The former
#                                   plural --checkpoints is rejected with a hint.)
#
# Each checkpoint must contain config.json plus either model_and_optim/ (native
# OLMo-core, converted automatically) or *.safetensors (HF, used as-is).
#
# Optional -- what to run:
#   --group NAME                    a named benchmark set (default: "default").
#                                   "smoke" runs the default set at 2 instances
#                                   per benchmark -- a plumbing check only.
#   --benchmarks "a b c"            explicit tasks; conflicts with --group
#   --allow-any-task                permit tasks outside the registry
# Optional -- which checkpoints:
#   --latest N                      keep the N highest-step checkpoints
#   --pattern REGEX                 filter on the checkpoint directory name
# Optional -- how to run:
#   --tp N                          vLLM tensor-parallel size (default 1)
#   --gpu-memory-utilization 0.N    VRAM fraction; lower it to share a GPU
#   --limit N                       cap instances per task; overrides a group's
#                                   own limit. Smoke tests only, see BENCHMARKS.md
#   --tokenizer ID                  HF tokenizer, if the config names none
#   --run-id-prefix STR             prefix for result subdirectories
#   --bootstrap                     install the environment first
#   --keep-local                    keep the local work tree
#   --dry-run                       print the plan and cost, then exit
#
# From the environment, not flags:
#   a checkout of this repo (or $OLMO_EVAL_ROOT), the aws CLI, AWS credentials
#   with read on the checkpoints and s3:PutObject on --s3-out, one GPU, and Hub
#   access with room in $HF_HOME. $AWS_REGION defaults to us-east-1.
#   $OLMO_CORE_CONVERT is an optional override for the bundled converter.
#
# ---------------------------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------------------------
# Per checkpoint, under <s3-out>/<run-id>/:
#   metrics.json                    one entry per benchmark
#   predictions/, requests/         per-instance JSONL
#   logs/                           vLLM server log
#   run_provenance.json             checkpoint, benchmarks, status, args, git sha
#   _READY | _FAILED                always one or the other
# At the sweep root:
#   accuracy_wide.csv               one row per checkpoint  <- the deliverable
#   accuracy.csv                    long form, one row per metric
#   accuracy.json, sweep.log
#
# Exit codes: 0 all succeeded, 1 some checkpoint failed, 2 bad arguments,
# 3 preflight failed (missing dependency, before anything spends).
#
# ---------------------------------------------------------------------------
# Every benchmark runs its complete evaluation split, which is expensive. Always
# preview with --dry-run first: it prints the resolved list and a cost estimate
# without spending anything. Use --latest N to cap the sweep.
#
# Which benchmarks exist, which groups they belong to, what they cost, and which
# splits they score all come from scripts/benchmarks.json, documented in
# BENCHMARKS.md. With neither --group nor --benchmarks, the registry's "default"
# group runs.
#
# No pre-built container image is required. On a bare Linux + CUDA box, run once
# with --bootstrap (or run scripts/bootstrap.sh yourself) to install the
# environment, then sweep as normal.
set -euo pipefail

CHECKPOINT=""
CHECKPOINT_ROOT=""
BENCHMARKS=""
GROUP=""
S3_OUT=""
LIMIT=""
TP="1"
GPU_MEM=""
TOKENIZER=""
PATTERN=""
LATEST=""
RUN_ID_PREFIX=""
ALLOW_ANY_TASK="0"
BOOTSTRAP="0"
KEEP_LOCAL="0"
DRY_RUN="0"
REGION="${AWS_REGION:-us-east-1}"

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${OLMO_EVAL_ROOT:-$(cd "${SKILL_DIR}/../../.." && pwd)}"
REGISTRY="${SKILL_DIR}/scripts/benchmarks.json"
RESOLVER="${SKILL_DIR}/scripts/resolve_benchmarks.py"
BUNDLED_CONVERT="${SKILL_DIR}/scripts/convert_to_hf.py"

# $OLMO_CORE_CONVERT overrides; otherwise the bundled library-only converter is
# used, so nothing has to clone OLMo-core at run time.
CONVERT_SCRIPT="${OLMO_CORE_CONVERT:-${BUNDLED_CONVERT}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    # Named explicitly rather than falling to the catch-all: the plural took a
    # list, so anyone reaching for it wants the sweep form and should be sent to
    # --checkpoint-root instead of just being told the flag is unknown.
    --checkpoints)
      echo "--checkpoints was replaced by --checkpoint, which takes exactly one." >&2
      echo "To evaluate several, point --checkpoint-root at the prefix above them." >&2
      exit 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --group) GROUP="$2"; shift 2 ;;
    --s3-out) S3_OUT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    --latest) LATEST="$2"; shift 2 ;;
    --run-id-prefix) RUN_ID_PREFIX="$2"; shift 2 ;;
    --allow-any-task) ALLOW_ANY_TASK="1"; shift ;;
    --bootstrap) BOOTSTRAP="1"; shift ;;
    --keep-local) KEEP_LOCAL="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    # Print the header block: every comment line after the shebang, stopping at
    # the first line of code. Derived rather than a fixed line range, which drifts
    # silently every time the header changes length.
    -h|--help) awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${S3_OUT}" ]] || { echo "--s3-out required" >&2; exit 2; }
if [[ -z "${CHECKPOINT}" && -z "${CHECKPOINT_ROOT}" ]]; then
  echo "one of --checkpoint or --checkpoint-root required" >&2; exit 2
fi
# --checkpoint is deliberately singular. Catching a space-separated list here
# gives a usable message; without this the whole string becomes one path and the
# failure surfaces much later as a confusing fetch error.
if [[ "${CHECKPOINT}" =~ [[:space:]] ]]; then
  echo "--checkpoint takes exactly one checkpoint, got: ${CHECKPOINT}" >&2
  echo "To evaluate several, point --checkpoint-root at the prefix above them." >&2
  exit 2
fi
[[ -f "${REGISTRY}" ]] || { echo "missing benchmark registry: ${REGISTRY}" >&2; exit 2; }

# Reject non-numeric knobs here rather than deep in the loop. These values are
# parsed as numbers when run_provenance.json is written, which happens *after*
# the eval has already run -- a bad value discovered there would waste the
# inference and abort the sweep before the terminal marker.
integer() { [[ "$2" =~ ^[0-9]+$ ]] || { echo "$1 must be an integer, got '$2'" >&2; exit 2; }; }
integer --tp "${TP}"
[[ -z "${LIMIT}" ]] || integer --limit "${LIMIT}"
[[ -z "${LATEST}" ]] || integer --latest "${LATEST}"
if [[ -n "${GPU_MEM}" ]]; then
  [[ "${GPU_MEM}" =~ ^0?\.[0-9]+$|^1(\.0+)?$ ]] || {
    echo "--gpu-memory-utilization must be a fraction in (0,1], got '${GPU_MEM}'" >&2
    exit 2
  }
fi

# Bootstrap before anything else, since the preflight below tests what it installs.
if [[ "${BOOTSTRAP}" == "1" ]]; then
  echo "running bootstrap (this installs the environment; see scripts/bootstrap.sh)"
  OLMO_EVAL_ROOT="${REPO_ROOT}" bash "${SKILL_DIR}/scripts/bootstrap.sh"
  # bootstrap.sh installs uv into ~/.local/bin and adds that to *its own* PATH,
  # but it runs as a child process, so the export dies with it. Re-apply it here
  # or the preflight below reports uv missing on exactly the bare box that
  # --bootstrap exists to serve, forcing a pointless second invocation.
  if ! command -v uv >/dev/null 2>&1 && [[ -x "${HOME:-/root}/.local/bin/uv" ]]; then
    export PATH="${HOME:-/root}/.local/bin:${PATH}"
    echo "added ${HOME:-/root}/.local/bin to PATH for this run"
  fi
fi

# Preflight. The sweep's first real work is an S3 listing followed by a
# multi-gigabyte checkpoint download, and only then does it invoke olmo-eval. A
# missing dependency discovered at that point has already cost time and transfer,
# so check the whole toolchain up front.
missing=()
command -v uv >/dev/null 2>&1 || missing+=("uv")
command -v aws >/dev/null 2>&1 || missing+=("aws CLI")
command -v python3 >/dev/null 2>&1 || missing+=("python3")
if command -v uv >/dev/null 2>&1; then
  if ! (cd "${REPO_ROOT}" && uv run python -c 'import olmo_eval' >/dev/null 2>&1); then
    missing+=("an importable olmo_eval in ${REPO_ROOT}")
  fi
fi
if [[ ${#missing[@]} -gt 0 ]]; then
  {
    echo "preflight failed; missing: ${missing[*]}"
    echo
    echo "This skill needs no container image, but it does need its environment"
    echo "installed once on this box. Run:"
    echo "  bash ${SKILL_DIR}/scripts/bootstrap.sh"
    echo "or re-run this script with --bootstrap."
    echo
    echo "If uv was just installed, its bin directory may not be on PATH yet:"
    echo "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
  } >&2
  exit 3
fi

# Resolve the request against the registry: pick an explicit --benchmarks list, a
# named --group, or the registry's "default" group; reject typos; total up the
# cost. Validating here rather than letting olmo-eval reject an unknown task
# matters because olmo-eval only does so after the checkpoint has been downloaded
# and possibly converted, by which point the time is already spent.
#
# THE RESOLUTION ITSELF LIVES IN scripts/resolve_benchmarks.py, AND IT IS SHARED.
# submit_eval_run.sh has to make the same judgement before dispatching a platform
# job, and two implementations would be two answers to "is this benchmark real"
# -- which is the one thing the registry exists to make single. It emits a JSON
# document, and the fields read below are its whole interface.
resolver_args=(--registry "${REGISTRY}")
if [[ -n "${GROUP}" ]]; then resolver_args+=(--group "${GROUP}"); fi
if [[ -n "${BENCHMARKS}" ]]; then resolver_args+=(--benchmarks "${BENCHMARKS}"); fi
if [[ -n "${LIMIT}" ]]; then resolver_args+=(--limit "${LIMIT}"); fi
if [[ "${ALLOW_ANY_TASK}" != "0" ]]; then resolver_args+=(--allow-any-task); fi

if ! RESOLVED="$(python3 "${RESOLVER}" "${resolver_args[@]}")"; then
  exit 2
fi

# tr -d '\r': python writes CRLF when stdout is a pipe on a Windows host, and a
# surviving CR would ride along on the last benchmark name -- producing a
# `-t name<CR>` argument -- and break every numeric comparison below.
_resolved_field() {
  printf '%s' "${RESOLVED}" |
    python3 -c "import json,sys;v=json.load(sys.stdin).get(sys.argv[1]);print('' if v is None else v)" "$1" |
    tr -d '\r'
}
read -r -a BENCH_LIST <<< "$(
  printf '%s' "${RESOLVED}" |
    python3 -c "import json,sys;print(' '.join(json.load(sys.stdin)['benchmarks']))" | tr -d '\r'
)"
TOTAL_INSTANCES="$(_resolved_field instances)"
TOTAL_PROMPTS="$(_resolved_field prompts)"
N_UNKNOWN="$(
  printf '%s' "${RESOLVED}" |
    python3 -c "import json,sys;print(len(json.load(sys.stdin)['unknown']))" | tr -d '\r'
)"
BENCH_SOURCE="$(_resolved_field source)"
# The resolver applied the precedence rule (an explicit --limit beats a group's),
# so whatever it returns is the effective limit.
LIMIT="$(_resolved_field effective_limit)"

WORK="$(mktemp -d)"
LOG="${WORK}/sweep.log"
if ! exec > >(tee -a "${LOG}") 2>&1; then
  exec >>"${LOG}" 2>&1
fi
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
cleanup() { [[ "${KEEP_LOCAL}" == "1" ]] || rm -rf "${WORK}"; }
trap cleanup EXIT

if [[ "${N_UNKNOWN}" -gt 0 ]]; then
  log "WARNING: ${N_UNKNOWN} benchmark(s) are not in the registry (--allow-any-task);"
  log "         the cost estimate below excludes them"
fi

# ---------------------------------------------------------------------------
# 1. Discover checkpoints
# ---------------------------------------------------------------------------
# Immediate child directories of a root, whether it is an S3 prefix or a local
# path. Both branches emit one absolute checkpoint path per line, so everything
# downstream is indifferent to which was used.
discover() {
  local root="${1%/}"
  if [[ "${root}" == s3://* ]]; then
    # `aws s3 ls` emits child prefixes as "PRE <name>/" lines; olmo-eval has no
    # checkpoint enumeration of its own.
    aws s3 ls "${root}/" --region "${REGION}" \
      | awk '/^ *PRE /{print $2}' \
      | sed 's:/$::' \
      | while read -r seg; do
          if [[ -n "${seg}" ]]; then echo "${root}/${seg}"; fi
        done
  else
    # Local root -- pre-converted checkpoints on disk, or a test fixture. Without
    # this branch the path would be handed to `aws s3 ls`, which reports no
    # prefixes, and the sweep would fail with a misleading "no checkpoints found".
    if [[ ! -d "${root}" ]]; then
      echo "--checkpoint-root is neither an s3:// URI nor an existing directory: ${root}" >&2
      return 1
    fi
    find "${root}" -mindepth 1 -maxdepth 1 -type d | sort
  fi
}

CKPT_LIST=()
if [[ -n "${CHECKPOINT}" ]]; then
  CKPT_LIST=("${CHECKPOINT%/}")
else
  log "discovering checkpoints under ${CHECKPOINT_ROOT}"
  readarray -t CKPT_LIST < <(discover "${CHECKPOINT_ROOT}")
  # readarray on empty input yields one empty element; drop it.
  if [[ ${#CKPT_LIST[@]} -eq 1 && -z "${CKPT_LIST[0]}" ]]; then
    CKPT_LIST=()
  fi
fi

if [[ -n "${PATTERN}" ]]; then
  FILTERED=()
  for c in "${CKPT_LIST[@]}"; do
    if [[ "$(basename "${c}")" =~ ${PATTERN} ]]; then
      FILTERED+=("${c}")
    fi
  done
  CKPT_LIST=("${FILTERED[@]+"${FILTERED[@]}"}")
fi

if [[ ${#CKPT_LIST[@]} -eq 0 ]]; then
  echo "no checkpoints found (root=${CHECKPOINT_ROOT} pattern=${PATTERN})" >&2
  exit 2
fi

# Order by training step so --latest and the summary follow training progress
# rather than lexical order (step9 before step10). Only the final path segment is
# examined, and the digits must end it: otherwise "latest" would pick up the "3"
# from "s3://" and sort as step 3. Segments with no trailing step sort last.
# tr -d '\r': python writes CRLF on a Windows host, and `readarray -t` strips
# only the LF, which would leave a stray CR inside every checkpoint path.
readarray -t CKPT_LIST < <(printf '%s\n' "${CKPT_LIST[@]}" | python3 -c '
import re, sys
def key(p):
    seg = p.rstrip("/").rsplit("/", 1)[-1]
    m = re.search(r"(\d+)(?:-hf)?$", seg)
    return (0, int(m.group(1)), p) if m else (1, 0, p)
print("\n".join(sorted((l.strip() for l in sys.stdin if l.strip()), key=key)))' | tr -d '\r')

if [[ -n "${LATEST}" && ${#CKPT_LIST[@]} -gt ${LATEST} ]]; then
  CKPT_LIST=("${CKPT_LIST[@]: -${LATEST}}")
fi

log "benchmarks (${#BENCH_LIST[@]}) from ${BENCH_SOURCE}: ${BENCH_LIST[*]}"
log "checkpoints (${#CKPT_LIST[@]}):"
for c in "${CKPT_LIST[@]}"; do log "  ${c}"; done

S3_ROOT="${S3_OUT%/}"
SWEEP_INSTANCES=$((TOTAL_INSTANCES * ${#CKPT_LIST[@]}))
SWEEP_PROMPTS=$((TOTAL_PROMPTS * ${#CKPT_LIST[@]}))
log "cost estimate: ~${TOTAL_INSTANCES} instances / ~${TOTAL_PROMPTS} vLLM prompts per checkpoint"
log "               ~${SWEEP_INSTANCES} instances / ~${SWEEP_PROMPTS} vLLM prompts for the whole sweep"
if [[ -n "${LIMIT}" ]]; then
  log "NOTE: capped at ${LIMIT} instances per benchmark; the estimate above reflects that."
  # Some tasks change which rows they load once a limit is set, which makes a
  # limited run score a population the unlimited run never touches. The registry
  # flags those; the table is absent when none do.
  AFFECTED="$(
    BENCHES="${BENCH_LIST[*]}" REGISTRY="${REGISTRY}" python3 - <<'PY'
import json, os
reg = json.load(open(os.environ["REGISTRY"], encoding="utf-8"))
unsafe = reg.get("limit_unsafe", {})
hits = [b for b in os.environ["BENCHES"].split() if b in unsafe]
for b in hits:
    print(f"{b}: {unsafe[b]}")
PY
  )"
  if [[ -n "${AFFECTED}" ]]; then
    log "      These score a different population when limited, so this run is a"
    log "      plumbing check and not a measurement (see BENCHMARKS.md):"
    while IFS= read -r line; do log "        ${line}"; done <<< "${AFFECTED}"
  fi
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN would run, per checkpoint, one vLLM boot covering: ${BENCH_LIST[*]}"
  log "DRY_RUN results -> ${S3_ROOT}/<run-id>/metrics.json"
  log "DRY_RUN summary -> ${S3_ROOT}/accuracy.{csv,json} and accuracy_wide.csv"
  exit 0
fi

# ---------------------------------------------------------------------------
# 2. Per-checkpoint evaluation
# ---------------------------------------------------------------------------
run_id_for() {
  local base
  base="$(basename "${1%/}")"
  printf '%s%s' "${RUN_ID_PREFIX}" "${base}" | tr -c 'A-Za-z0-9._-' '-'
}

# A native OLMo-core checkpoint carries sharded state; HF format does not.
is_native_olmo_core() {
  [[ -d "$1/model_and_optim" || -f "$1/.metadata" ]]
}

# Record what happened to a checkpoint. Called for every outcome, including the
# ones that abort before the eval runs, so a failed checkpoint carries a reason
# into the summary instead of showing up as an unexplained blank row.
write_provenance() {
  local status="$1"
  RUN_ID="${RUN_ID}" CKPT="${CKPT}" DEST="${DEST}" OUT="${OUT}" \
  BENCHES="${BENCH_LIST[*]}" BENCH_SOURCE="${BENCH_SOURCE}" STATUS="${status}" \
  GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)" \
  LIMIT="${LIMIT}" TP="${TP}" GPU_MEM="${GPU_MEM}" \
  python3 - <<'PY' || log "WARNING: could not write run_provenance.json"
import json, os
from pathlib import Path
env = os.environ
prov = {
    "run_id": env["RUN_ID"],
    "checkpoint": env["CKPT"],
    "s3_dest": env["DEST"],
    "benchmarks": env["BENCHES"].split(),
    "benchmark_selection": env["BENCH_SOURCE"],
    "status": env["STATUS"],
    "limit": int(env["LIMIT"]) if env["LIMIT"] else None,
    "tensor_parallel_size": int(env["TP"]),
    "gpu_memory_utilization": float(env["GPU_MEM"]) if env["GPU_MEM"] else None,
    "git_sha": env["GIT_SHA"],
}
Path(env["OUT"], "run_provenance.json").write_text(
    json.dumps(prov, indent=2), encoding="utf-8"
)
PY
}

# Terminal state for a checkpoint that never reached the eval. The local
# provenance is what puts it in the summary; the S3 marker is what stops a poller
# waiting forever.
fail_early() {
  local reason="$1"
  write_provenance "${reason}"
  echo "${reason}" | aws s3 cp - "${DEST}/_FAILED" --region "${REGION}" || true
  aws s3 cp "${OUT}/run_provenance.json" "${DEST}/run_provenance.json" \
    --region "${REGION}" --only-show-errors || true
  FAILURES=$((FAILURES + 1))
}

FAILURES=0
for CKPT in "${CKPT_LIST[@]}"; do
  RUN_ID="$(run_id_for "${CKPT}")"
  DEST="${S3_ROOT}/${RUN_ID}"
  OUT="${WORK}/out/${RUN_ID}"
  STAGED_CKPT=""
  CONVERTED_DIR=""
  mkdir -p "${OUT}"
  STATUS="skipped"

  log "=== ${RUN_ID} :: ${CKPT}"

  # Fetch locally. vLLM cannot load an s3:// path directly (olmo-eval has no S3
  # download on the vLLM path), so materialize the weights first.
  if [[ "${CKPT}" == s3://* ]]; then
    STAGED_CKPT="${WORK}/ckpt/${RUN_ID}"
    mkdir -p "${STAGED_CKPT}"
    LOCAL_CKPT="${STAGED_CKPT}"
    log "[${RUN_ID}] fetching weights"
    if ! aws s3 sync "${CKPT}" "${LOCAL_CKPT}" --region "${REGION}" --only-show-errors; then
      log "[${RUN_ID}] FAILED to fetch; skipping"
      fail_early "fetch_failed"
      continue
    fi
  else
    LOCAL_CKPT="${CKPT}"
  fi

  MODEL="${LOCAL_CKPT}"
  if is_native_olmo_core "${LOCAL_CKPT}"; then
    # CONVERT_SCRIPT defaults to the bundled library-only converter, so this only
    # trips if that file is missing or an $OLMO_CORE_CONVERT override points
    # somewhere that does not exist.
    if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
      log "[${RUN_ID}] native OLMo-core checkpoint but no converter at ${CONVERT_SCRIPT}; skipping"
      fail_early "no_converter"
      continue
    fi
    log "[${RUN_ID}] converting OLMo-core -> HF"
    # Convert inside the work tree: a "<checkpoint>-hf" sibling would write into
    # (and later delete from) the caller's directory for a local checkpoint.
    CONVERTED_DIR="${WORK}/hf/${RUN_ID}"
    conv=(-i "${LOCAL_CKPT}" -o "${CONVERTED_DIR}" --dtype bfloat16 --skip-validation)
    if [[ -n "${TOKENIZER}" ]]; then
      conv+=(-t "${TOKENIZER}")
    fi
    # Through `uv run` so the converter sees the synced environment: it imports
    # olmo_core and torch, which are project dependencies rather than system ones.
    if ! (cd "${REPO_ROOT}" && uv run python "${CONVERT_SCRIPT}" "${conv[@]}"); then
      log "[${RUN_ID}] FAILED conversion; skipping"
      fail_early "conversion_failed"
      continue
    fi
    MODEL="${CONVERTED_DIR}"
  fi

  # Every benchmark goes in one invocation so they share a single vLLM boot.
  #
  # `run` has no top-level --provider, and each -o binds to the *preceding*
  # --harness or -t. So provider overrides must sit between --harness and the
  # first -t, and each task's overrides directly after its own -t. Tensor
  # parallelism goes through provider.kwargs: ProviderConfig has no
  # tensor_parallel_size field, and ProviderConfig.from_dict silently drops
  # keys it does not recognize, so the shorter path would vanish without error.
  log "[${RUN_ID}] evaluating: ${BENCH_LIST[*]}"
  args=(--harness default -o provider.kind=vllm_server)
  if [[ "${TP}" != "1" ]]; then
    args+=(-o "provider.kwargs.tensor_parallel_size=${TP}")
  fi
  if [[ -n "${GPU_MEM}" ]]; then
    args+=(-o "provider.kwargs.gpu_memory_utilization=${GPU_MEM}")
  fi
  for b in "${BENCH_LIST[@]}"; do
    args+=(-t "${b}")
    if [[ -n "${LIMIT}" ]]; then
      args+=(-o "limit=${LIMIT}")
    fi
  done

  if (cd "${REPO_ROOT}" && uv run olmo-eval run \
        -m "${MODEL}" "${args[@]}" -O "${OUT}"); then
    STATUS="ok"
  else
    STATUS="failed"
    log "[${RUN_ID}] eval failed (continuing to next checkpoint)"
  fi

  # --- provenance, upload, terminal marker ---
  # Nothing from here to the marker may abort the loop. `set -e` would otherwise
  # kill the sweep between a finished eval and its _READY/_FAILED, which is the
  # one failure mode a poller cannot recover from.
  write_provenance "${STATUS}"

  log "[${RUN_ID}] uploading -> ${DEST}/"
  aws s3 sync "${OUT}" "${DEST}/" --region "${REGION}" --only-show-errors || true

  # Always write a terminal marker, so a poller can tell a failed run from one
  # still in progress.
  if [[ "${STATUS}" == "failed" ]]; then
    echo "status=failed" | aws s3 cp - "${DEST}/_FAILED" --region "${REGION}" || true
    FAILURES=$((FAILURES + 1))
  else
    echo "status=${STATUS}" | aws s3 cp - "${DEST}/_READY" --region "${REGION}" || true
  fi

  # Only ever remove paths this script created, never a caller-supplied checkpoint.
  if [[ "${KEEP_LOCAL}" != "1" ]]; then
    if [[ -n "${STAGED_CKPT}" ]]; then rm -rf "${STAGED_CKPT}"; fi
    if [[ -n "${CONVERTED_DIR}" ]]; then rm -rf "${CONVERTED_DIR}"; fi
  fi
done

# ---------------------------------------------------------------------------
# 3. Cross-checkpoint accuracy table
# ---------------------------------------------------------------------------
log "aggregating ${#CKPT_LIST[@]} checkpoint(s)"
if python3 "${SKILL_DIR}/scripts/summarize_accuracy.py" \
     --runs-dir "${WORK}/out" \
     --out-csv "${WORK}/accuracy.csv" \
     --out-wide-csv "${WORK}/accuracy_wide.csv" \
     --out-json "${WORK}/accuracy.json"; then
  for f in accuracy.csv accuracy_wide.csv accuracy.json; do
    aws s3 cp "${WORK}/${f}" "${S3_ROOT}/${f}" --region "${REGION}" || true
  done
fi
aws s3 cp "${LOG}" "${S3_ROOT}/sweep.log" --region "${REGION}" || true

log "done: ${#CKPT_LIST[@]} checkpoint(s), ${FAILURES} with failures -> ${S3_ROOT}/"
if [[ "${FAILURES}" -ne 0 ]]; then
  exit 1
fi
