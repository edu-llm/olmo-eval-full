#!/usr/bin/env bash
# Evaluate one checkpoint by submitting a job to the eduLLM platform.
#
#   validate locally -> dispatch submit-run.yml -> report where results will land
#
# Usage:
#   submit_eval_run.sh --checkpoint s3://.../runs/RUN/checkpoints/step2000 \
#     --team pre-training --experiment my-eval --wandb-project edullm-evals \
#     --group smoke --dry-run
#
# ---------------------------------------------------------------------------
# INPUTS
# ---------------------------------------------------------------------------
# Required, and none of them can be inferred:
#   --checkpoint s3://...           one OLMo-core checkpoint directory. Must live
#                                   under the outputs bucket; see WHY below.
#   --team NAME                     one of the platform's eight teams. Routes the
#                                   approval and fixes the output prefix.
#   --experiment SLUG               groups related runs. Lower-case with hyphens.
#   --wandb-project NAME            free text; the entity is always eduLLM.
#
# Optional -- what to run:
#   --group NAME                    a named benchmark set (default: "default").
#                                   "smoke" is the 2-instance plumbing check.
#   --benchmarks "a b c"            explicit tasks; conflicts with --group
#   --limit N                       instances per task; overrides a group's own
#   --allow-any-task                permit tasks outside the registry
#   --tokenizer ID                  HF tokenizer, if the checkpoint names none
#   --batch-size N                  prompts per forward pass. SET THIS for anything
#                                   past a smoke run: the provider's default is
#                                   every prompt at once, which runs out of memory
#                                   once the machine has already been allocated.
#                                   SKILL.md has a table keyed on model size.
#   --override KEY=VALUE            any other harness override, repeatable. The
#                                   escape hatch for provider kwargs --batch-size
#                                   does not cover.
# Optional -- how to run it:
#   --compute-profile NAME          machine (default: gpu-1xa10g). Must be a GPU
#                                   profile; see WHY below.
#   --runtime-hours N               overrides the workload profile's 1-hour bound
#                                   (default: 2). See WHY TWO HOURS below; every
#                                   value this script can sensibly pass waits for
#                                   a team lead.
#   --eval-ref SHA                  full 40-hex commit of THIS repo to run.
#                                   Defaults to HEAD. Must be pushed.
#   --research-commit SHA|main      commit of the OLMo-core research repo whose
#                                   image the job rides on (default: main). Set
#                                   this to a training commit to eval on the exact
#                                   image that produced a checkpoint -- required
#                                   when the checkpoint's config imports modules
#                                   (e.g. olmo_core.nn.memory.*) absent from main.
# Optional -- how to behave:
#   --dry-run                       validate, print the submission, dispatch nothing
#   --no-wait                       dispatch and exit without polling for the run id
#
# From the environment: the `gh` CLI, authenticated against the platform repo.
# No AWS credentials: the platform assumes its own roles through GitHub OIDC.
#
# ---------------------------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------------------------
# Results land at a prefix the PLATFORM chooses, not one you pass:
#   s3://sbsandbox-intern-edullm-outputs/teams/<team>/runs/<run-id>/
# This script prints that path once the run id is known. Under it:
#   metrics.json, predictions/, requests/   from olmo-eval
#   eval_provenance.json                    what ran, including this repo's commit
#   _READY | _FAILED                        terminal marker
#
# Exit codes: 0 submitted (or dry run), 2 refused locally, 3 missing dependency,
# 4 dispatched but the run id could not be read back.
#
# ---------------------------------------------------------------------------
# WHY THE CHECKPOINT MUST BE IN THE OUTPUTS BUCKET. The GPU workload role permits
# s3:GetObject on sbsandbox-intern-edullm-outputs/teams/*/runs/* and nothing else,
# so a checkpoint anywhere else is unreadable. That failure arrives after the
# submission has compiled, been approved, reached a queue and pulled an image --
# the platform's own tests record it costing a real run -- so it is checked here.
#
# WHY A GPU PROFILE. The CPU workload role holds no s3:GetObject at all. A CPU
# submission is admitted, approved, placed, and then dies on its first read.
#
# WHY repository=OLMo-core. The olmo-eval-full image leaves torch and vllm out and
# can only run provider.kind=mock. The OLMo-core image carries CUDA torch and
# ai2-olmo-core, which is what the olmo_core provider needs, so this rides on that
# image and installs olmo-eval into it at start-up. The cost is that the lineage
# record names OLMo-core's commit; --eval-ref is recorded in the run's own
# provenance so the two can be reconciled. See SKILL.md.
#
# WHY TWO HOURS, KNOWING IT FORFEITS AUTOMATIC APPROVAL. The platform releases a
# run without a person only when the requested bound is strictly under one hour
# and the estimated cost strictly under five dollars -- the live numbers are the
# two automatic_below_ bounds in the platform's config/policy.yaml, and this
# script does not read them, so check there rather than trusting this line. Cost
# is never the binding half here, gpu-1xa10g being $1.006/hour at one attempt, so
# the bound alone decides, and two hours means every submission this script makes
# waits for a team lead. That is deliberate. The bound is a hard timeout the job is killed
# at, not a budget it is refunded from, and before this job scores its first
# prompt it apt-gets git, pip-installs olmo-eval and transformers, downloads the
# checkpoint, and pulls the benchmark data -- roughly 17k rows on a cold HF_HOME
# even for a 36-prompt smoke run. The workload profile allows one attempt, so a
# run killed at the ceiling is not retried; it writes _FAILED and the money is
# gone. Buying automatic approval would mean asking for under an hour on the
# same work, and the thing it saves is one click.
#
# The same default also covers full sweeps, which is the other reason not to
# shrink it: --group defaults to the whole reasoning set, not to smoke.
set -euo pipefail

PLATFORM_REPO="edu-llm/platform"
SUBMIT_WORKFLOW="submit-run.yml"
EVAL_REPO="edu-llm/olmo-eval-full"
OUTPUTS_BUCKET="sbsandbox-intern-edullm-outputs"
# Bound to OLMo-core because the image is. See WHY above.
RESEARCH_REPOSITORY="OLMo-core"
# Default image is built from OLMo-core@main; --research-commit overrides it so a
# checkpoint can be evaluated on the exact image that trained it.
RESEARCH_COMMIT="main"
WORKLOAD_PROFILE="olmo-core-check"

# The eight team_ids in config/organization.yaml. A closed list on the form, so a
# typo is refused there too -- but refused here it costs no dispatch.
TEAMS="platform memory-split input-core pre-training post-training data-prep eval-inference scratch"
# GPU profiles from submit-run.yml. cpu-32vcpu is deliberately absent.
GPU_PROFILES="gpu-1xt4 gpu-4xt4 gpu-8xt4 gpu-1xa10g gpu-4xa10g gpu-8xa10g gpu-1xl4 gpu-4xl4 gpu-8xl4 gpu-1xl40s gpu-4xl40s gpu-8xl40s gpu-1xh100 gpu-8xa100 gpu-8xh100"
# config/capacity.yaml marks these as placing unreliably; a job that cannot get
# capacity waits in RUNNABLE with nothing watching.
UNRELIABLE_PROFILES="gpu-4xa10g gpu-1xh100 gpu-8xa100 gpu-8xh100"

CHECKPOINT=""
TEAM=""
EXPERIMENT=""
WANDB_PROJECT=""
GROUP=""
BENCHMARKS=""
LIMIT=""
TOKENIZER=""
BATCH_SIZE=""
OVERRIDES=()
COMPUTE_PROFILE="gpu-1xa10g"
# Two, not something under one that would auto-approve. See WHY TWO HOURS above.
RUNTIME_HOURS="2"
EVAL_REF=""
ALLOW_ANY_TASK="0"
DRY_RUN="0"
NO_WAIT="0"

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${OLMO_EVAL_ROOT:-$(cd "${SKILL_DIR}/../../.." && pwd)}"
REGISTRY="${SKILL_DIR}/scripts/benchmarks.json"
RESOLVER="${SKILL_DIR}/scripts/resolve_benchmarks.py"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --team) TEAM="$2"; shift 2 ;;
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --group) GROUP="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --override) OVERRIDES+=("$2"); shift 2 ;;
    --compute-profile) COMPUTE_PROFILE="$2"; shift 2 ;;
    --runtime-hours) RUNTIME_HOURS="$2"; shift 2 ;;
    --eval-ref) EVAL_REF="$2"; shift 2 ;;
    --research-commit) RESEARCH_COMMIT="$2"; shift 2 ;;
    --allow-any-task) ALLOW_ANY_TASK="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --no-wait) NO_WAIT="1"; shift ;;
    # Two flags that belong to the other path and reach this one in real client
    # prompts. Both are named here rather than left to the catch-all, because
    # "unknown arg" sends someone to --help to look for a flag that is missing on
    # purpose, and neither absence is an oversight worth rediscovering.
    --bootstrap)
      {
        echo "--bootstrap belongs to run_eval_sweep.sh, and there is nothing here"
        echo "for it to do. That flag installs a Python environment on a GPU box you"
        echo "own. This path owns no box: the container installs git, olmo-eval and"
        echo "its dependencies from scratch on every run, which is the couple of"
        echo "minutes of start-up SKILL.md tells you to budget for."
        echo
        echo "Drop the flag. If you meant to prepare a local machine for the sweep,"
        echo "that is: bash scripts/bootstrap.sh"
      } >&2
      exit 2 ;;
    --checkpoints)
      {
        echo "--checkpoints is plural and this script evaluates exactly one."
        echo
        echo "One submission is one checkpoint on one machine, so a list has no"
        echo "spelling here -- run this script once per checkpoint, each with its"
        echo "own --checkpoint and the same --experiment so the runs stay grouped."
        echo "Each mints its own run id and its own output prefix; record them as"
        echo "they print, because they are not derivable afterwards."
        echo
        echo "The sweep's --checkpoint-root, which discovers every checkpoint under"
        echo "a prefix, exists only on run_eval_sweep.sh and needs your own GPU."
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
    echo "has been approved and placed. Training runs on this platform already"
    echo "write there; see CHECKPOINTS.md."
  } >&2
  exit 2
fi

in_list "${TEAM}" "${TEAMS}" || die "--team must be one of: ${TEAMS}"
[[ "${EXPERIMENT}" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] ||
  die "--experiment must be lower-case with hyphens, got: ${EXPERIMENT}"

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

[[ "${RUNTIME_HOURS}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  die "--runtime-hours must be a number, got: ${RUNTIME_HOURS}"
[[ -z "${LIMIT}" || "${LIMIT}" =~ ^[0-9]+$ ]] || die "--limit must be an integer, got: ${LIMIT}"

# --- dependencies ----------------------------------------------------------
command -v python3 >/dev/null 2>&1 || { echo "python3 is required to resolve benchmarks" >&2; exit 3; }
[[ -f "${REGISTRY}" ]] || { echo "missing benchmark registry: ${REGISTRY}" >&2; exit 3; }
[[ -f "${RESOLVER}" ]] || { echo "missing resolver: ${RESOLVER}" >&2; exit 3; }
if [[ "${DRY_RUN}" != "1" ]]; then
  command -v gh >/dev/null 2>&1 || {
    echo "the gh CLI is required to dispatch the workflow, and is not installed." >&2
    echo "No AWS credentials are needed; the platform assumes its own roles." >&2
    exit 3
  }
  gh auth status >/dev/null 2>&1 || { echo "gh is not authenticated; run: gh auth login" >&2; exit 3; }
fi

# --- what to run -----------------------------------------------------------
# The registry is the single place benchmark names live, and this is the same
# resolver the local sweep uses, so a name valid here is valid there.
resolver_args=(--registry "${REGISTRY}")
[[ -z "${GROUP}" ]] || resolver_args+=(--group "${GROUP}")
[[ -z "${BENCHMARKS}" ]] || resolver_args+=(--benchmarks "${BENCHMARKS}")
[[ -z "${LIMIT}" ]] || resolver_args+=(--limit "${LIMIT}")
[[ "${ALLOW_ANY_TASK}" == "0" ]] || resolver_args+=(--allow-any-task)

if ! RESOLVED="$(python3 "${RESOLVER}" "${resolver_args[@]}")"; then
  exit 2
fi
read_field() { printf '%s' "${RESOLVED}" | python3 -c "import json,sys;print(json.load(sys.stdin).get(sys.argv[1]) or '')" "$1" | tr -d '\r'; }
RESOLVED_BENCHMARKS="$(printf '%s' "${RESOLVED}" | python3 -c "import json,sys;print(' '.join(json.load(sys.stdin)['benchmarks']))" | tr -d '\r')"
BENCH_SOURCE="$(read_field source)"
EFFECTIVE_LIMIT="$(read_field effective_limit)"
EST_INSTANCES="$(read_field instances)"
EST_PROMPTS="$(read_field prompts)"

# --- which commit of this repo runs ---------------------------------------
# A full sha rather than a branch, because the tarball the job fetches is the one
# thing recording what code actually ran: the platform's manifest will name
# OLMo-core's commit instead. A moving ref here would make that unrecoverable.
if [[ -z "${EVAL_REF}" ]]; then
  EVAL_REF="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
fi
[[ "${EVAL_REF}" =~ ^[0-9a-f]{40}$ ]] ||
  die "--eval-ref must be a full 40-character commit sha of this repo, got: ${EVAL_REF:-<empty>}"
if ! git -C "${REPO_ROOT}" merge-base --is-ancestor "${EVAL_REF}" "@{upstream}" 2>/dev/null; then
  echo "NOTE: ${EVAL_REF:0:12} may not be pushed. The job fetches it from GitHub," >&2
  echo "      so an unpushed commit fails at pip install rather than at submission." >&2
fi
TARBALL="https://github.com/${EVAL_REPO}/archive/${EVAL_REF}.tar.gz"

# The research image commit: either the moving default or a full sha pinning the
# exact image a checkpoint was trained on. A branch other than main is refused
# because the image the job rides on must be reproducible from what is printed.
[[ "${RESEARCH_COMMIT}" == "main" || "${RESEARCH_COMMIT}" =~ ^[0-9a-f]{40}$ ]] ||
  die "--research-commit must be 'main' or a full 40-character commit sha, got: ${RESEARCH_COMMIT}"

# --- batch size and overrides ---------------------------------------------
# Both end up inside a single-quoted `bash -lc '...'`, so a value carrying a
# single quote would close that quoting early and a value carrying whitespace
# would split into two arguments. Neither is worth supporting; both are refused.
if [[ -n "${BATCH_SIZE}" ]]; then
  [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] ||
    die "--batch-size must be a positive integer, got: ${BATCH_SIZE}"
  OVERRIDES+=("provider.kwargs.batch_size=${BATCH_SIZE}")
fi
for ov in ${OVERRIDES+"${OVERRIDES[@]}"}; do
  [[ "${ov}" == *=* ]] || die "--override wants KEY=VALUE, got: ${ov}"
  case "${ov}" in
    *\'*|*[[:space:]]*) die "--override cannot contain quotes or whitespace: ${ov}" ;;
  esac
done

# Both selection flags are optional and the resolver falls back to the full
# default set, so the command that deliberately sweeps 65,315 prompts and the
# command that forgot to say `--group smoke` are byte-for-byte identical. SKILL.md
# tells the agent to propose smoke for a first run and nothing downstream held it
# to that; the batch-size warning below is about memory and fires on a real sweep
# too, so it cannot carry this.
#
# SILENCE IS THE TRIGGER, NOT SIZE. `--group default` resolves to exactly the same
# benchmarks and says nothing here, because someone who named the group answered
# the question. Only an unanswered one is worth interrupting.
if [[ -z "${GROUP}" && -z "${BENCHMARKS}" ]]; then
  {
    echo "WARNING: neither --group nor --benchmarks was given, so this is the"
    echo "         registry's full default set: ~${EST_INSTANCES} instances / ~${EST_PROMPTS} prompts."
    echo "         THIS IS NOT A SMOKE TEST. A first run against a checkpoint"
    echo "         should prove the plumbing before it buys a measurement:"
    echo "             --group smoke"
    echo "         is the same five benchmarks at 2 instances each, ~36 prompts."
    echo "         If you did mean the full set, pass --group default to say so"
    echo "         and this stops asking."
    echo
  } >&2
fi

# The provider batches every prompt at once when nothing says otherwise, so the
# run that most needs a batch size is the one that says nothing. Warn in
# proportion to the cost of finding out the hard way.
if [[ -z "${BATCH_SIZE}" ]] && [[ "${EST_PROMPTS}" -gt 1000 ]]; then
  echo "WARNING: no --batch-size on a ~${EST_PROMPTS}-prompt run." >&2
  echo "         The olmo_core provider defaults to one forward pass for every" >&2
  echo "         prompt, which runs out of memory after the machine has been" >&2
  echo "         allocated and the image pulled. SKILL.md has a table keyed on" >&2
  echo "         model size; 64 is a reasonable start for a model under 3B on" >&2
  echo "         the 24 GB default profile." >&2
  echo >&2
fi

# --- the command the container runs ---------------------------------------
# bash -lc because the line needs && and quoting; the platform execs argv
# verbatim and the first word must be a program, which `bash` is.
#
# git is installed first because olmo-eval declares `ifbench` as a git+https
# dependency and the OLMo-core image is built on a bare python base that carries
# no git -- pip would shell out to it and fail. Conditional so a future image
# that ships git pays nothing.
#
# The [hf] extra brings transformers, which the olmo_core provider imports for
# AutoTokenizer and which ai2-olmo-core declares only as an extra of its own.
BOOTSTRAP="command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq --no-install-recommends git; }"
INSTALL="python -m pip install --no-cache-dir \"olmo-eval[hf] @ ${TARBALL}\""
RUNNER="python -m olmo_eval.platform.run_eval --checkpoint ${CHECKPOINT} --benchmarks \"${RESOLVED_BENCHMARKS}\""
[[ -z "${EFFECTIVE_LIMIT}" ]] || RUNNER+=" --limit ${EFFECTIVE_LIMIT}"
[[ -z "${TOKENIZER}" ]] || RUNNER+=" --tokenizer ${TOKENIZER}"
for ov in ${OVERRIDES+"${OVERRIDES[@]}"}; do
  RUNNER+=" --override ${ov}"
done
COMMAND="bash -lc '${BOOTSTRAP} && ${INSTALL} && ${RUNNER}'"

# --- report the plan -------------------------------------------------------
echo "benchmarks (${RESOLVED_BENCHMARKS// /,}) from ${BENCH_SOURCE}"
echo "  resolved:     ${RESOLVED_BENCHMARKS}"
[[ -z "${EFFECTIVE_LIMIT}" ]] || echo "  limit:        ${EFFECTIVE_LIMIT} instances per benchmark"
echo "  cost shape:   ~${EST_INSTANCES} instances / ~${EST_PROMPTS} prompts"
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
echo "  s3://${OUTPUTS_BUCKET}/teams/${TEAM}/runs/<run-id>/"
echo
echo "command:"
echo "  ${COMMAND}"

if [[ -n "$(printf '%s' "${RESOLVED}" | python3 -c "import json,sys;print(' '.join(json.load(sys.stdin)['limit_unsafe']))" | tr -d '\r')" ]]; then
  echo
  echo "NOTE: a limit is in force and these benchmarks score a different population"
  echo "      when limited, so this run is a plumbing check and not a measurement:"
  printf '%s' "${RESOLVED}" |
    python3 -c "import json,sys
for name, why in json.load(sys.stdin)['limit_unsafe'].items():
    print(f'        {name}: {why}')"
fi

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

# The run id is minted by the platform's compile job, so it cannot be known
# before dispatch. It is published two ways: in the workflow's step summary,
# which no REST endpoint exposes, and in the `compiled-submission` artifact,
# which is machine-readable. Read the artifact.
#
# That artifact also carries `approval_class`, which is the only honest source
# for it: the platform computed it for this submission, from thresholds this
# script cannot see and which are versioned and deployed separately from the
# file they are written in. Re-deriving the class here would be a second answer
# to a question that already has one, and it would be the wrong answer on the
# day the policy moves. So it is read, not computed.
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
    # Defaulted rather than indexed: an older compile job that wrote no such key
    # should cost the class, not the run id sitting beside it.
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
echo "results:  s3://${OUTPUTS_BUCKET}/teams/${TEAM}/runs/${RUN_ID}/"
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
