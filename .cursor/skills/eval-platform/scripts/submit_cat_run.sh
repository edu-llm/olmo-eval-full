#!/usr/bin/env bash
# Run one adaptive-testing (CAT) session against one OLMo-core checkpoint, on the
# eduLLM platform, inside the OLMo-core image.
#
#   validate locally -> dispatch submit-run.yml -> report where results will land
#
# Usage:
#   submit_cat_run.sh --checkpoint s3://.../runs/RUN/checkpoints/step305176 \
#     --team eval-inference --experiment native-cat --wandb-project edullm-evals \
#     --benchmark arc_challenge --dry-run
#
# ---------------------------------------------------------------------------
# WHAT THIS IS, AND WHY IT IS NOT submit_eval_run.sh
# ---------------------------------------------------------------------------
# submit_eval_run.sh beside this file runs olmo-eval's own harness over a list of
# benchmarks, through `python -m olmo_eval.platform.run_eval`. This runs the CAT
# in `diagnostics/mcq_cat/`, which administers 13-40 adaptively chosen items from
# one calibrated bank and reports an IRT ability estimate rather than an accuracy.
# One benchmark, not a list, because a CAT session is a session over one bank.
#
# Everything about the SUBMISSION is copied from that script deliberately -- the
# same platform repository, the same workload profile, the same reasons a
# checkpoint must be in the outputs bucket and the machine must be a GPU. Those
# are properties of the platform, not of what runs on it, and a second answer to
# any of them would be a second answer that can drift.
#
# ---------------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------------
# Required:
#   --checkpoint s3://...   one OLMo-core checkpoint directory, under the outputs
#                           bucket. Read natively; nothing is converted.
#   --team NAME             one of the platform's eight teams.
#   --experiment SLUG       groups related runs. Lower-case with hyphens.
#   --wandb-project NAME    free text; the platform form has no default.
#
# Optional -- what to run:
#   --benchmark NAME        one calibrated MCQ bank (default: arc_challenge).
#   --max-items N           ceiling on items administered (default: the runner's 40).
#   --se-threshold X        standard-error stop (default: the runner's 0.3).
#   --dtype NAME            bfloat16 (default) | float16 | float32. See WHY DTYPE.
#
# Optional -- how to run it:
#   --compute-profile NAME  machine (default: gpu-1xl4).
#   --runtime-hours N       (default: 2). Forfeits automatic approval; see WHY.
#   --eval-ref SHA          full 40-hex commit of THIS repo. Defaults to HEAD.
#
# Optional -- how to behave:
#   --dry-run               validate, print the submission, dispatch nothing
#   --no-wait               dispatch and exit without polling for the run id
#
# Exit codes: 0 submitted (or dry run), 2 refused locally, 3 missing dependency,
# 4 dispatched but the run id could not be read back.
#
# ---------------------------------------------------------------------------
# WHY repository=OLMo-core, WHICH IS THE WHOLE POINT OF THIS FILE
# ---------------------------------------------------------------------------
# The CAT's native path needs three things on the machine: CUDA torch, an
# `ai2-olmo-core` whose config schema can parse the checkpoint, and this
# repository's code. Two images could carry them.
#
# olmo-eval-full's image is built from `.edullm/Dockerfile` in this repository and
# has this repository's code by construction -- but its `ai2-olmo-core` comes from
# `uv.lock`, which pins PyPI 2.4.0. That release has no `sequence_mixer` field on
# TransformerBlockConfig and no `partial_rotary_factor` or `no_global_rope` on
# RoPEConfig, all three of which this checkpoint's config.json carries, so
# `TransformerConfig.from_dict` raises before a weight is read. Making that image
# work means pinning a fork into the Dockerfile and asserting the schema at build
# time. It is doable -- it was written and then withdrawn in favour of this -- but
# it is a second olmo_core to keep correct.
#
# OLMo-core's image installs olmo_core from its own checkout (`COPY . .` then
# `pip install ".[wandb]"`), so the library in that image *is* the repository at
# the commit the image was built from. Read at origin/main 08df5aa0 on 2026-08-08:
# `sequence_mixer` is on TransformerBlockConfig and `no_global_rope` and
# `partial_rotary_factor` are on RoPEConfig, version 2.5.0. So main parses this
# checkpoint's schema with no fork and no pin.
#
# The only difference between main and the `edullm/memory-split-135m` fork in
# those two files is three `smollm2_*` convenience constructors, which are builders
# used at training time to *create* a config. `TransformerConfig.from_dict` never
# calls them: it reads the serialized dict the checkpoint already holds. Their
# absence is therefore not a blocker for reading a checkpoint, only for writing
# `TransformerConfig.smollm2_135M(...)` in new code.
#
# The cost is that the platform's lineage record names OLMo-core's commit rather
# than this repository's. `--eval-ref` is what the container checks out and is
# printed below, and the CAT writes it into its own report; the two have to be
# reconciled by hand.
#
# ---------------------------------------------------------------------------
# WHY THE CONTAINER CLONES INSTEAD OF pip install-ing THIS REPOSITORY
# ---------------------------------------------------------------------------
# submit_eval_run.sh installs a GitHub tarball as a wheel, which is right for it:
# everything it runs is `olmo_eval.platform.run_eval`, and `pyproject.toml`
# packages `olmo_eval` from `src/`.
#
# The CAT is not in that wheel. `diagnostics/` is at the repository root, outside
# `src/`, and so is `calibrated_datasets/` -- 18 MB of item banks and IRT
# parameters that `resolve.py` finds through `Path(__file__).parents[4]`, a repo
# root rather than a package directory. `python -m diagnostics.mcq_cat.runner`
# inside a pip-installed image is a `ModuleNotFoundError`, and it would be a
# `DatasetNotAvailable` one line later even if the module were found.
#
# THREE WAYS TO FIX THAT, AND THE CHEAPEST ONE IS NOT THE PACKAGING CHANGE.
#
#   (a) Fetch the repository into the container and run from it, installing
#       `olmo-eval` from the same checkout for its dependencies. Touches no
#       packaging. `resolve.REPO_ROOT` keeps pointing at a real repository root,
#       which is what it is written to expect, and the banks arrive for free.
#   (b) Add `diagnostics` to the wheel via `pyproject.toml`. Needs explicit
#       `packages.find` over two roots, and it does not solve the banks:
#       `calibrated_datasets/` is not a package, has no `__init__.py`, and cannot
#       be package-data of one. It would have to move under `diagnostics/`, and
#       `REPO_ROOT` would have to stop meaning repo root -- so the "packaging
#       change" is really a layout change plus a resolver change.
#   (c) Add an `olmo_eval.platform.run_cat` entry point. This does not help on its
#       own: the entry point would be in the wheel and the CAT it drives would not.
#       It only works on top of (b), or by moving `diagnostics/` under
#       `src/olmo_eval/`, which is every import in the diagnostic and every test.
#
# This script does (a). `git clone --filter=blob:none` then `git checkout <sha>`
# rather than a tarball, for one reason: the whole command is a single-quoted
# argument, so nothing inside it may contain a single quote, and the quote-free
# ways to unpack a tarball all need a tool the base may not carry. `curl` is not
# on a bare python base, and `python -c` needs a string literal for the URL. git
# is already installed by the bootstrap line -- olmo-eval declares `ifbench` as a
# git+https dependency -- so using it costs nothing that was not already paid.
#
# NOT `--extra olmo_core` ON THE INSTALL, AND THIS ONE IS A TRAP. That extra is
# `ai2-olmo-core[torchao,transformers]==2.4.0`, so asking for it would install the
# exact PyPI release that cannot parse this checkpoint, on top of the image's
# working 2.5.0, and undo the only reason this rides on OLMo-core's image. `hf`
# and `s3` are what is needed: transformers for AutoTokenizer, boto3 for reading
# the checkpoint and writing the report. `transformers>=5.4.0` is satisfied by the
# 5.14.1 the image pins, so pip leaves it alone.
#
# ---------------------------------------------------------------------------
# WHY DTYPE IS ON THE COMMAND WHEN NOTHING IS CONVERTED
# ---------------------------------------------------------------------------
# `--checkpoint-prep none` converts nothing, so `--dtype` writes no weights and
# the runner logs that it had no effect. It is on the command anyway because the
# platform's `bfloat16_not_in_the_hardware` guard reads the words of the command
# and nothing else, and the hardware requirement is real: the weights are bfloat16
# and the native loader runs them on the card in that format. Without the flag a
# T4 -- Turing, no bfloat16 in the silicon -- is priced, admitted, placed, and dies
# on the first kernel that wants the format. Verified 2026-08-08 against
# `edullm check`: with the flag, gpu-1xt4, gpu-4xt4 and gpu-8xt4 are refused and
# quote it back; without it, none of them is.
#
# That is a flag used for its effect on admission rather than for what it names,
# and it is better said here than discovered. `--dtype float16` is a real escape
# onto a T4 rather than a way to quiet the refusal: the banks were calibrated
# behind bfloat16 scoring, so it is a measurement change.
#
# ---------------------------------------------------------------------------
# WHY TWO HOURS, KNOWING IT FORFEITS AUTOMATIC APPROVAL
# ---------------------------------------------------------------------------
# The same argument submit_eval_run.sh makes, and it binds harder here. The CAT
# itself is minutes: 13-40 items against a 135M model. Everything before it is not
# -- apt-get git, clone this repository, pip install olmo-eval and its
# dependencies, and download the checkpoint, which is 1.74 GB across 140 objects
# and is downloaded in full even under `--checkpoint-prep none`, because the
# native loader reads the shards. The bound is a hard timeout the job is killed
# at, not a budget it is refunded from, and the workload profile allows one
# attempt. A run killed at the ceiling writes _FAILED and the money is gone.
set -euo pipefail

PLATFORM_REPO="edu-llm/platform"
SUBMIT_WORKFLOW="submit-run.yml"
EVAL_REPO="edu-llm/olmo-eval-full"
OUTPUTS_BUCKET="sbsandbox-intern-edullm-outputs"
# Bound to OLMo-core because the image is. See WHY above.
RESEARCH_REPOSITORY="OLMo-core"
RESEARCH_COMMIT="main"
WORKLOAD_PROFILE="olmo-core-check"

# Where the container puts the checkout. Fixed rather than mktemp'd so a log line
# naming a path is the same path on every run.
CONTAINER_ROOT="/opt/olmo-eval-full"

TEAMS="platform memory-split input-core pre-training post-training data-prep eval-inference scratch"
GPU_PROFILES="gpu-1xt4 gpu-4xt4 gpu-8xt4 gpu-1xa10g gpu-4xa10g gpu-8xa10g gpu-1xl4 gpu-4xl4 gpu-8xl4 gpu-1xl40s gpu-4xl40s gpu-8xl40s gpu-1xh100 gpu-8xa100 gpu-8xh100"
UNRELIABLE_PROFILES="gpu-4xa10g gpu-1xh100 gpu-8xa100 gpu-8xh100"

# The MCQ banks. Named here rather than read out of Python because this script has
# to run on a machine with no numpy, and stale is cheap: `grading.check_bank_modality`
# and `check_checkpoint_kind` both refuse a generative bank on the MCQ scorer inside
# the container, before the checkpoint is fetched. So a name missing from this list
# costs a local refusal and a name wrongly on it costs a fast remote one -- neither
# costs a measurement. The source of truth is `modality=` in
# diagnostics/mcq_cat/styles/uni_mcq/datasets.py.
MCQ_BENCHMARKS="arc_challenge hellaswag winogrande musr bbh"
# Generative banks exist and are calibrated, and the CAT can administer them -- but
# only through a completer, and the only registered generative backend is `hf`.
# Named separately so asking for one gets an explanation rather than "unknown".
GENERATIVE_BENCHMARKS="gsm8k leaderboard_math ifeval gpqa"

CHECKPOINT=""
TEAM=""
EXPERIMENT=""
WANDB_PROJECT=""
BENCHMARK="arc_challenge"
MAX_ITEMS=""
SE_THRESHOLD=""
DTYPE="bfloat16"
COMPUTE_PROFILE="gpu-1xl4"
RUNTIME_HOURS="2"
EVAL_REF=""
DRY_RUN="0"
NO_WAIT="0"

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${OLMO_EVAL_ROOT:-$(cd "${SKILL_DIR}/../../.." && pwd)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --team) TEAM="$2"; shift 2 ;;
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --max-items) MAX_ITEMS="$2"; shift 2 ;;
    --se-threshold) SE_THRESHOLD="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --compute-profile) COMPUTE_PROFILE="$2"; shift 2 ;;
    --runtime-hours) RUNTIME_HOURS="$2"; shift 2 ;;
    --eval-ref) EVAL_REF="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --no-wait) NO_WAIT="1"; shift ;;
    # Named rather than left to the catch-all, because it is the flag someone
    # arrives with from the script next door and its absence is deliberate.
    --benchmarks)
      {
        echo "--benchmarks is plural and a CAT session runs over exactly one bank."
        echo
        echo "Adaptive item selection conditions every next item on the ability"
        echo "estimate so far, and that estimate is per-bank: the IRT parameters"
        echo "that make it meaningful were fitted on one bank at a time. A list"
        echo "would be several sessions, so run this once per bank with the same"
        echo "--experiment and they stay grouped."
        echo
        echo "The plural flag belongs to submit_eval_run.sh, which runs olmo-eval's"
        echo "fixed-form harness and for which a list is the natural unit."
      } >&2
      exit 2 ;;
    -h|--help) awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

die() { echo "$@" >&2; exit 2; }
in_list() { case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# --- required inputs -------------------------------------------------------
[[ -n "${CHECKPOINT}" ]] || die "--checkpoint required: the S3 directory of one OLMo-core checkpoint"
[[ -n "${TEAM}" ]] || die "--team required (one of: ${TEAMS})"
[[ -n "${EXPERIMENT}" ]] || die "--experiment required: a lower-case slug grouping related runs"
[[ -n "${WANDB_PROJECT}" ]] || die "--wandb-project required: the platform form has no default for it"

# --- the checkpoint has to be readable ------------------------------------
CHECKPOINT="${CHECKPOINT%/}"
if [[ "${CHECKPOINT}" != s3://* ]]; then
  die "--checkpoint must be an s3:// uri, got: ${CHECKPOINT}"
fi
if [[ "${CHECKPOINT}" != "s3://${OUTPUTS_BUCKET}/teams/"*"/runs/"* ]]; then
  {
    echo "--checkpoint is not somewhere the eval job can read."
    echo
    echo "  given: ${CHECKPOINT}"
    echo "  needs: s3://${OUTPUTS_BUCKET}/teams/<team>/runs/<run-id>/..."
    echo
    echo "The GPU workload role permits s3:GetObject on that prefix and nothing"
    echo "else, so a checkpoint outside it fails with AccessDenied after the run"
    echo "has been approved and placed. See CHECKPOINTS.md."
  } >&2
  exit 2
fi

in_list "${TEAM}" "${TEAMS}" || die "--team must be one of: ${TEAMS}"
[[ "${EXPERIMENT}" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] ||
  die "--experiment must be lower-case with hyphens, got: ${EXPERIMENT}"

# --- the bank has to exist, and be one this scorer can grade ---------------
if in_list "${BENCHMARK}" "${GENERATIVE_BENCHMARKS}"; then
  {
    echo "--benchmark ${BENCHMARK} is a generative bank, and this path scores MCQ."
    echo
    echo "Its items have no answer choices, so they are graded by generating a"
    echo "completion and matching it. The generative completer has exactly one"
    echo "registered backend, \`hf\`, and the native OLMo-core reader is not it:"
    echo "the completer needs a distinct EOS to stop on and every checkpoint this"
    echo "training setup writes has bos == eos == pad == 0."
    echo
    echo "MCQ banks: ${MCQ_BENCHMARKS}"
  } >&2
  exit 2
fi
in_list "${BENCHMARK}" "${MCQ_BENCHMARKS}" ||
  die "--benchmark must be one of: ${MCQ_BENCHMARKS} (got: ${BENCHMARK})"
# The bank travels in the checkout, so its absence here is its absence there.
BANK_DIR="${REPO_ROOT}/calibrated_datasets/${BENCHMARK}"
if [[ ! -f "${BANK_DIR}/manifest.json" ]]; then
  {
    echo "--benchmark ${BENCHMARK} names a supported dataset with no vendored bank."
    echo
    echo "  looked for: ${BANK_DIR}/manifest.json"
    echo
    echo "The container clones this repository at --eval-ref and reads the bank"
    echo "from the checkout, so a bank missing here is missing there, and the run"
    echo "would fail after the image was pulled rather than now."
  } >&2
  exit 2
fi

# --- the machine has to be able to read ------------------------------------
if ! in_list "${COMPUTE_PROFILE}" "${GPU_PROFILES}"; then
  {
    echo "--compute-profile must be a GPU profile, got: ${COMPUTE_PROFILE}"
    echo
    echo "The CPU workload role holds no s3:GetObject at all, so a CPU run is"
    echo "admitted, approved, placed, and then dies on its first read. Choose one"
    echo "of: ${GPU_PROFILES}"
  } >&2
  exit 2
fi
if in_list "${COMPUTE_PROFILE}" "${UNRELIABLE_PROFILES}"; then
  echo "NOTE: config/capacity.yaml marks ${COMPUTE_PROFILE} as placing unreliably." >&2
  echo "      A job that cannot get capacity waits in RUNNABLE and nothing watches." >&2
fi

in_list "${DTYPE}" "bfloat16 float16 float32" ||
  die "--dtype must be bfloat16, float16 or float32, got: ${DTYPE}"
[[ "${RUNTIME_HOURS}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "--runtime-hours must be a number, got: ${RUNTIME_HOURS}"
[[ -z "${MAX_ITEMS}" || "${MAX_ITEMS}" =~ ^[1-9][0-9]*$ ]] ||
  die "--max-items must be a positive integer, got: ${MAX_ITEMS}"
[[ -z "${SE_THRESHOLD}" || "${SE_THRESHOLD}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "--se-threshold must be a number, got: ${SE_THRESHOLD}"

# --- dependencies ----------------------------------------------------------
if [[ "${DRY_RUN}" != "1" ]]; then
  command -v gh >/dev/null 2>&1 || {
    echo "the gh CLI is required to dispatch the workflow, and is not installed." >&2
    echo "No AWS credentials are needed; the platform assumes its own roles." >&2
    exit 3
  }
  gh auth status >/dev/null 2>&1 || { echo "gh is not authenticated; run: gh auth login" >&2; exit 3; }
fi

# --- which commit of this repo runs ---------------------------------------
# A full sha rather than a branch. The platform's manifest will name OLMo-core's
# commit, so this is the only record of what code ran, and a moving ref would make
# that unrecoverable.
if [[ -z "${EVAL_REF}" ]]; then
  EVAL_REF="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
fi
[[ "${EVAL_REF}" =~ ^[0-9a-f]{40}$ ]] ||
  die "--eval-ref must be a full 40-character commit sha of this repo, got: ${EVAL_REF:-<empty>}"
if ! git -C "${REPO_ROOT}" merge-base --is-ancestor "${EVAL_REF}" "@{upstream}" 2>/dev/null; then
  echo "NOTE: ${EVAL_REF:0:12} may not be pushed. The container clones from GitHub," >&2
  echo "      so an unpushed commit fails at git checkout rather than at submission." >&2
fi

# --- the command the container runs ---------------------------------------
# bash -lc because the line needs && and quoting; the platform execs argv verbatim
# and the first word must be a program, which `bash` is. Nothing below may contain
# a single quote: the whole thing is one single-quoted argument.
#
# git first, conditionally, so a future image that ships it pays nothing. It is
# needed twice here -- to clone this repository, and by pip, because olmo-eval
# declares `ifbench` as a git+https dependency.
#
# --filter=blob:none is a partial clone: every commit and tree, no file contents
# until something asks. The checkout below then fetches only the blobs at that one
# commit, which is the 18 MB of banks plus the source rather than the history of
# both.
BOOTSTRAP="command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq --no-install-recommends git; }"
FETCH="git clone --filter=blob:none https://github.com/${EVAL_REPO}.git ${CONTAINER_ROOT} && cd ${CONTAINER_ROOT} && git checkout ${EVAL_REF}"
INSTALL="python -m pip install --no-cache-dir \".[hf,s3]\""
RUNNER="python -m diagnostics.mcq_cat.runner --cat-style uni_mcq"
RUNNER+=" --checkpoint ${CHECKPOINT}"
RUNNER+=" --benchmark ${BENCHMARK}"
RUNNER+=" --checkpoint-prep none --checkpoint-kind olmo_core"
RUNNER+=" --dtype ${DTYPE}"
[[ -z "${MAX_ITEMS}" ]] || RUNNER+=" --max-items ${MAX_ITEMS}"
[[ -z "${SE_THRESHOLD}" ]] || RUNNER+=" --se-threshold ${SE_THRESHOLD}"
RUNNER+=" --s3-out \"\$EDULLM_OUTPUT_PREFIX\""
COMMAND="bash -lc '${BOOTSTRAP} && ${FETCH} && ${INSTALL} && ${RUNNER}'"

# --- report the plan -------------------------------------------------------
echo "cat session:    ${BENCHMARK} (one calibrated MCQ bank, adaptively administered)"
echo "  bank:         ${BANK_DIR}"
echo "  stop:         se<=${SE_THRESHOLD:-0.3 (runner default)} or ${MAX_ITEMS:-40 (runner default)} items"
echo "  path:         --checkpoint-prep none --checkpoint-kind olmo_core (no conversion)"
echo "checkpoint:     ${CHECKPOINT}"
echo "eval commit:    ${EVAL_REF}"
echo "submission:     repository=${RESEARCH_REPOSITORY} commit=${RESEARCH_COMMIT}"
echo "                workload=${WORKLOAD_PROFILE} compute=${COMPUTE_PROFILE} runtime=${RUNTIME_HOURS}h"
echo "                team=${TEAM} experiment=${EXPERIMENT} wandb=${WANDB_PROJECT}"
echo "approval:       the platform classifies this from the runtime bound above and"
echo "                the profile's hourly rate. This script does not read the"
echo "                thresholds and cannot tell you the class; the compiled"
echo "                submission carries it, and it is printed below unless"
echo "                --dry-run or --no-wait."
echo "results will land at:"
echo "  s3://${OUTPUTS_BUCKET}/teams/${TEAM}/runs/<run-id>/cat_report.json"
echo
echo "command:"
echo "  ${COMMAND}"
echo
echo "NOT PROVEN: the native olmo_core scorer has never executed against a real"
echo "            checkpoint. What model_forward returns for a batch of one, what"
echo "            this tokenizer prepends by default, and whether bfloat16"
echo "            log_softmax agrees with the provider's cast are all open. Two of"
echo "            the three fail loudly -- there is a permanent shape assertion and"
echo "            the tokenizer defaults are measured at load and written into"
echo "            cat_report.json under run.tokenization. This run is the test."

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "DRY_RUN: nothing was dispatched."
  exit 0
fi

# --- dispatch --------------------------------------------------------------
DISPATCHED_AFTER="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "dispatching ${SUBMIT_WORKFLOW} in ${PLATFORM_REPO} ..."
gh workflow run "${SUBMIT_WORKFLOW}" -R "${PLATFORM_REPO}" \
  -f "repository=${RESEARCH_REPOSITORY}" \
  -f "commit_sha=${RESEARCH_COMMIT}" \
  -f "workload_profile=${WORKLOAD_PROFILE}" \
  -f "compute_profile=${COMPUTE_PROFILE}" \
  -f "dataset_release=none" \
  -f "team=${TEAM}" \
  -f "experiment=${EXPERIMENT}" \
  -f "wandb_project=${WANDB_PROJECT}" \
  -f "maximum_runtime_hours=${RUNTIME_HOURS}" \
  -f "command=${COMMAND}"

echo "dispatched at ${DISPATCHED_AFTER}."
if [[ "${NO_WAIT}" == "1" ]]; then
  echo "Not waiting, so neither the run id nor the approval class is known here."
  echo "Both are in the workflow run:"
  echo "  https://github.com/${PLATFORM_REPO}/actions/workflows/${SUBMIT_WORKFLOW}"
  exit 0
fi

# The run id is minted by the platform's compile job, so it cannot be known before
# dispatch. Read it, and the approval class beside it, out of the machine-readable
# `compiled-submission` artifact rather than re-deriving either here.
echo "waiting for the workflow to appear ..."
WORKFLOW_RUN_ID=""
for _ in $(seq 1 30); do
  WORKFLOW_RUN_ID="$(gh run list -R "${PLATFORM_REPO}" --workflow "${SUBMIT_WORKFLOW}" \
    --created ">=${DISPATCHED_AFTER}" --limit 1 --json databaseId \
    --jq '.[0].databaseId // empty' 2>/dev/null || true)"
  [[ -n "${WORKFLOW_RUN_ID}" ]] && break
  sleep 4
done
if [[ -z "${WORKFLOW_RUN_ID}" ]]; then
  echo "the workflow was dispatched but did not appear in the run list." >&2
  echo "It is probably running; look for it under:" >&2
  echo "  https://github.com/${PLATFORM_REPO}/actions/workflows/${SUBMIT_WORKFLOW}" >&2
  exit 4
fi
echo "workflow run: https://github.com/${PLATFORM_REPO}/actions/runs/${WORKFLOW_RUN_ID}"

echo "waiting for the submission to compile ..."
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
RUN_ID=""
APPROVAL_CLASS=""
for _ in $(seq 1 60); do
  if gh run download "${WORKFLOW_RUN_ID}" -R "${PLATFORM_REPO}" \
      -n compiled-submission -D "${WORK}" >/dev/null 2>&1; then
    RUN_ID="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['run_id'])" \
      "${WORK}/compiled-submission.json" 2>/dev/null | tr -d '\r' || true)"
    APPROVAL_CLASS="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('approval_class') or '')" \
      "${WORK}/compiled-submission.json" 2>/dev/null | tr -d '\r' || true)"
    [[ -n "${RUN_ID}" ]] && break
  fi
  sleep 5
done

if [[ -z "${RUN_ID}" ]]; then
  {
    echo
    echo "The submission was dispatched but its run id could not be read back."
    echo "That does not mean it failed. The artifact appears only once the compile"
    echo "job finishes, so either it is still running or it refused the submission"
    echo "on the merits -- waiting for an approver is a later gate and does not"
    echo "withhold this file."
    echo "Read the workflow run above; its summary names the run id and the prefix."
  } >&2
  exit 4
fi

echo
echo "run id:   ${RUN_ID}"
echo "results:  s3://${OUTPUTS_BUCKET}/teams/${TEAM}/runs/${RUN_ID}/cat_report.json"
if [[ -n "${APPROVAL_CLASS}" ]]; then
  echo "approval: ${APPROVAL_CLASS} -- the platform's classification of this submission,"
  echo "          read from the compiled submission rather than guessed here."
  case "${APPROVAL_CLASS}" in
    automatic) echo "          Nobody releases it. It starts on its own." ;;
    routine)   echo "          It starts when a team lead releases it, and not before." ;;
    exception) echo "          It starts when a platform admin releases it, and not before." ;;
    *)         echo "          This script does not know what that class implies; read the run." ;;
  esac
else
  echo "approval: not carried by the compiled submission. Read the workflow run"
  echo "          above for it rather than assuming either way."
fi
echo
echo "To watch it or stop it, dispatch 'Look at a run, or stop it' with that run id:"
echo "  https://github.com/${PLATFORM_REPO}/actions/workflows/cancel-run.yml"
echo "Nothing watches the queue, so if it has not started within an hour, ask."
