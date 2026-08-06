#!/usr/bin/env bash
#
# run_checkpoint_eval.sh — the per-checkpoint eval ATOM (P0). ONE-SHOT laptop-side
# orchestrator that evaluates ONE checkpoint on AWS and lands results in S3. It
# generalizes run_rung1_smoketest.sh from "one fixed smoke model" to "the checkpoint
# I was handed", and is the reusable unit the retroactive batch driver (scoping doc
# §3.5) repeats across a whole run.
#
# It drives the ENTIRE flow from your machine using the AWS CLI: skip-if-done check ->
# pre-flight safety checks -> launch ONE GPU box -> stage the repo -> install deps ->
# run `olmo-eval run` against the checkpoint -> verify results in S3 -> terminate.
#
# CHECKPOINT FORMAT (scoping doc §3.6 decision):
#   CHECKPOINT_KIND=hf   (DEFAULT, validated) — the olmo-eval HF/vLLM provider loads an
#       HF-format checkpoint (config.json + model.safetensors) or a public HF id. Lean
#       install; no ai2-olmo-core. This is the common, proven path.
#   CHECKPOINT_KIND=olmo_core (optional native path, wired but NOT yet validated) — the
#       install step CONDITIONALLY adds the `olmo_core` extra (ai2-olmo-core) and the run
#       selects the real OlmoCoreProvider via `-o provider.kind=olmo_core`. This is the
#       safety valve for raw, sharded OLMo-Core checkpoints (model_and_optim/ + .metadata)
#       that skip the OLMo->HF converter. Coherent and runnable, left to be exercised in P1.
#
# IDEMPOTENCY (scoping doc §3.5 point 4):
#   Before launching a (billed) GPU box, the orchestrator checks S3 for the checkpoint's
#   result marker (metrics.json under the result prefix). If present it SKIPS the whole
#   launch and exits success, so a re-run / resume after a crash or partial sweep only
#   evaluates the unfinished checkpoints. Override with FORCE=true / SKIP_IF_DONE=false.
#
# STAGING MODES (how olmo-eval reaches the GPU node — see STAGE_MODE):
#   clone (DEFAULT) — `git clone` the PUBLIC olmo-eval repo directly on the node. Enough
#       for stock tasks (e.g. arc_easy) that live in the public repo.
#   tar             — tar a LOCAL checkout -> S3 -> presigned URL -> SSM curl+untar. Use
#       this to ship uncommitted/private code (custom eduLLM tasks + the Qwen LLM-judge).
#
# ─────────────────────────────────────────────────────────────────────────────
# PREREQUISITES (do these once, on your laptop)
#   1. Install the AWS CLI v2  (`aws --version` should work).
#   2. Get team credentials through the broker and install a local profile:
#          sb-aws-creds login              # browser login (~30-day session)
#          sb-aws-creds install-profiles   # writes the `sbsandbox` profile into ~/.aws/config
#      After that, `aws sts get-caller-identity --profile sbsandbox` must succeed.
#
# USAGE
#   ./run_checkpoint_eval.sh                                  # P0 smoke: tiny HF model, arc_easy
#   CHECKPOINT=s3://bucket/checkpoints/run/step_1000 \
#     TASKS=arc_easy ./run_checkpoint_eval.sh                 # a real HF checkpoint (still g5/smoke)
#   CHECKPOINT_KIND=olmo_core CHECKPOINT=s3://.../step_1000 \
#     ./run_checkpoint_eval.sh                                # native path (wired, unvalidated)
#   ./run_checkpoint_eval.sh --dry-run                        # PREVIEW: print every aws command
#   ./run_checkpoint_eval.sh -h                               # full help / all env vars
#
# EVERYTHING is overridable via environment variables (see the CONFIG block / `-h`).
#
# COST / SAFETY: a GPU box bills per hour. This script installs a trap that TERMINATES the
# launched instance on ANY exit (success, error, Ctrl-C), the box self-terminates via
# `shutdown -h +N`, and we terminate explicitly at the end — three independent safety nets.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — env var with default. Defaults are the exact team values from the
# runbook. Override by exporting before you run.
# ══════════════════════════════════════════════════════════════════════════════

# --- AWS access (also settable with --profile / --region flags) --------------
AWS_PROFILE="${AWS_PROFILE:-sbsandbox}"     # named profile from install-profiles
AWS_REGION="${AWS_REGION:-us-east-1}"       # team region (N. Virginia)

# --- EC2 / instance identity (team-provisioned; do not invent new ones) ------
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"                # smoke default (1x A10G, 24 GB)
AMI_ID="${AMI_ID:-ami-0b6f2229ad14c9323}"                 # Ubuntu 24.04 DLAMI
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-087218d8c87aa8576}"
SUBNET_ID="${SUBNET_ID:-subnet-0a4235fb98b63930f}"        # us-east-1b
INSTANCE_PROFILE="${INSTANCE_PROFILE:-EswManagedInstance}" # IAM "badge" -> S3 access

# --- The checkpoint + eval ----------------------------------------------------
# CHECKPOINT: the thing under test. For P0 this defaults to a tiny public HF model so the
# atom runs end-to-end with no real checkpoint and no AWS change. Point it at an HF id, a
# local dir, or an s3://.../checkpoints/step_N URI for a real run.
CHECKPOINT="${CHECKPOINT:-Qwen/Qwen2.5-0.5B-Instruct}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-hf}"       # hf (default, validated) | olmo_core (native)
TASKS="${TASKS:-arc_easy}"                     # task spec or suite (space-separated for many)
LIMIT="${LIMIT:-10}"                           # `-o limit=N` task override (empty = no limit)

# --- S3 destination for results (team bucket already exists) -----------------
# S3_PREFIX defaults to 'smoke' because the EswManagedInstance role's existing s3:PutObject
# grant covers only smoke/* (+ smoke_split/*, full200/*). Results land under
# smoke/checkpoints/ so auto-upload succeeds with NO IAM change (scoping doc §5 interim).
S3_BUCKET="${S3_BUCKET:-edullm-adaptive-inference-056956104102}"
S3_PREFIX="${S3_PREFIX:-smoke}"
S3_GROUP="${S3_GROUP:-checkpoints}"

# --- Idempotency (skip-if-done) ----------------------------------------------
# Before launching a billed box, look for the result marker under the result prefix. If
# it's already there, the checkpoint was evaluated — skip the whole launch. FORCE re-runs.
SKIP_IF_DONE="${SKIP_IF_DONE:-true}"
FORCE="${FORCE:-false}"
DONE_MARKER="${DONE_MARKER:-metrics.json}"     # object name that marks a finished eval

# --- How the repo reaches the node (staging mode) ----------------------------
STAGE_MODE="${STAGE_MODE:-clone}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/allenai/olmo-eval}"  # clone mode source
GIT_REF="${GIT_REF:-}"   # optional branch/tag to check out (empty = repo default branch)

# --- Safety timer + local repo location (LOCAL_* used by STAGE_MODE=tar ONLY) -
SHUTDOWN_MINUTES="${SHUTDOWN_MINUTES:-90}"                  # user-data self-destruct
LOCAL_REPO_PARENT="${LOCAL_REPO_PARENT:-/Users/cat/alpha-projects}"  # tar -C here
REPO_DIRNAME="${REPO_DIRNAME:-olmo-eval-full}"             # dir to tar/untar

# --- On-node paths (DLAMI fast NVMe scratch) ---------------------------------
NODE_STAGE_DIR="${NODE_STAGE_DIR:-/opt/dlami/nvme/checkpoint-eval}"
NODE_RESULTS_DIR="${NODE_RESULTS_DIR:-${NODE_STAGE_DIR}/results}"
NODE_LOG="${NODE_LOG:-${NODE_STAGE_DIR}/checkpoint_eval.log}"

# On-node env the SSM commands must set (SSM's AWS-RunShellScript runs as root but does NOT
# set HOME, and the DLAMI root disk is only ~19 GB — too small for torch + vLLM + model
# downloads — so HOME and all caches point at the big NVMe scratch).
NODE_HOME="${NODE_HOME:-/root}"
UV_CACHE_DIR_NODE="${UV_CACHE_DIR_NODE:-${NODE_STAGE_DIR}/uv-cache}"
HF_HOME_NODE="${HF_HOME_NODE:-${NODE_STAGE_DIR}/hf}"

# On-node repo directory (per staging mode: a clone lands in a dir named after the git URL
# basename; a tar untars into REPO_DIRNAME). An explicit NODE_REPO_DIR override still wins.
case "$STAGE_MODE" in
  tar) REPO_BASENAME="$REPO_DIRNAME" ;;
  *)   REPO_BASENAME="${GIT_REPO_URL##*/}"; REPO_BASENAME="${REPO_BASENAME%.git}" ;;
esac
NODE_REPO_DIR="${NODE_REPO_DIR:-${NODE_STAGE_DIR}/${REPO_BASENAME}}"

# --- Timeouts / behaviour knobs ----------------------------------------------
SSM_ONLINE_TIMEOUT="${SSM_ONLINE_TIMEOUT:-300}"   # wait for SSM agent Online (s)
STAGE_TIMEOUT="${STAGE_TIMEOUT:-300}"             # curl+untar on node (s)
INSTALL_TIMEOUT="${INSTALL_TIMEOUT:-1800}"        # uv sync can be slow (s)
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"              # checkpoint dl + provider + score (s)
PRESIGN_EXPIRY="${PRESIGN_EXPIRY:-3600}"          # presigned URL lifetime (s)
CLEANUP_STAGING="${CLEANUP_STAGING:-true}"        # rm staging tgz from S3 at end

# Allowlisted instance types — the team hard rule: never above g6.xlarge.
ALLOWED_INSTANCE_TYPES="g5.xlarge g6.xlarge"

# Sentinel appended to the on-node log after the eval process exits, so we can reliably
# detect completion + read its exit code by tailing the log.
DONE_SENTINEL="CHECKPOINT_DONE_EXIT"

# Runtime state (populated as we go).
DRY_RUN=false
INSTANCE_ID=""
TARBALL=""
TEARDOWN_DONE=false

# S3 key/uri for the staged code tarball, and the result prefix for verify/skip-if-done.
STAGING_KEY="${S3_PREFIX}/staging/olmo-eval.tgz"
STAGING_URI="s3://${S3_BUCKET}/${STAGING_KEY}"
RESULT_PREFIX="s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/"

# ══════════════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════════════

# Pretty-join a command's argv for display (quote args that contain spaces).
quote_cmd() {
  local a out=""
  for a in "$@"; do
    case "$a" in
      *[[:space:]]*) out="${out}'${a}' " ;;
      *)             out="${out}${a} " ;;
    esac
  done
  printf '%s' "${out% }"
}

# run(): echo the command (always) then execute it — UNLESS --dry-run, in which case it
# only echoes. Its stdout is passed through so callers can capture it with $( ... ); the
# "+ ..." trace line goes to stderr so it never pollutes captured output.
run() {
  printf '    + %s\n' "$(quote_cmd "$@")" >&2
  if [ "$DRY_RUN" = true ]; then
    return 0
  fi
  "$@"
}

log()  { printf '\n=== %s\n' "$*" >&2; }
info() { printf '    %s\n' "$*" >&2; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# Escape an arbitrary string so it can be embedded inside a JSON string literal.
json_escape() {
  local s=$1
  s=${s//\\/\\\\}   # backslash -> \\   (must be first)
  s=${s//\"/\\\"}   # "         -> \"
  printf '%s' "$s"
}

# Build the AWS-RunShellScript --parameters value as JSON: {"commands":["..."]}.
ssm_params() { printf '{"commands":["%s"]}' "$(json_escape "$1")"; }

# ssm_invoke <payload> <description> [timeout_seconds]
# Sends ONE shell payload to the node via SSM, waits until the SSM command reaches a
# terminal state, prints StandardOutputContent to stdout (for the caller to capture), and
# returns non-zero on any non-Success status or timeout.
ssm_invoke() {
  local payload=$1 desc=${2:-"ssm command"} timeout=${3:-600}
  local params cid status waited=0 interval=5
  params=$(ssm_params "$payload")
  log "SSM: ${desc}"
  cid=$(run aws ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name AWS-RunShellScript \
    --comment "$desc" \
    --parameters "$params" \
    --query 'Command.CommandId' --output text \
    --profile "$AWS_PROFILE" --region "$AWS_REGION") \
    || { echo "ERROR: ssm send-command failed (${desc})" >&2; return 1; }

  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would poll: aws ssm get-command-invocation --command-id <id> --instance-id ${INSTANCE_ID}"
    return 0
  fi

  while :; do
    status=$(aws ssm get-command-invocation \
      --command-id "$cid" --instance-id "$INSTANCE_ID" \
      --query 'Status' --output text \
      --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || echo "Pending")
    case "$status" in
      Success) break ;;
      Failed|Cancelled|TimedOut|Undeliverable|Terminated)
        echo "ERROR: SSM '${desc}' -> ${status}. StandardErrorContent:" >&2
        aws ssm get-command-invocation \
          --command-id "$cid" --instance-id "$INSTANCE_ID" \
          --query 'StandardErrorContent' --output text \
          --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null >&2 || true
        return 1 ;;
      *) : ;;  # Pending / InProgress / Delayed -> keep waiting
    esac
    if [ "$waited" -ge "$timeout" ]; then
      echo "ERROR: SSM '${desc}' timed out after ${timeout}s (last status=${status})" >&2
      return 1
    fi
    sleep "$interval"
    waited=$(( waited + interval ))
  done

  aws ssm get-command-invocation \
    --command-id "$cid" --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' --output text \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null
}

# ══════════════════════════════════════════════════════════════════════════════
# Teardown — guaranteed to run on ANY exit so we never leak a billing instance.
# ══════════════════════════════════════════════════════════════════════════════

# Explicit terminate + confirm (belt-and-suspenders end-of-run teardown).
terminate_and_confirm() {
  [ -z "${INSTANCE_ID}" ] && return 0
  log "Teardown: terminating ${INSTANCE_ID}"
  run aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" >/dev/null || true
  if [ "$DRY_RUN" = true ]; then return 0; fi
  aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || true
  local state
  state=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[].Instances[].State.Name' --output text \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || echo unknown)
  info "Instance ${INSTANCE_ID} state: ${state}"
}

# trap handler: fires on EXIT/INT/TERM/ERR. Best-effort terminate of any launched
# instance, then remove the local tarball. No-op under --dry-run or if nothing was
# launched. Detaches itself first so it runs exactly once.
cleanup() {
  local rc=$?
  trap - EXIT INT TERM ERR

  if [ -n "${TARBALL}" ] && [ -f "${TARBALL}" ]; then
    rm -f "$TARBALL" 2>/dev/null || true
  fi

  if [ "$DRY_RUN" = true ]; then exit "$rc"; fi

  if [ "$TEARDOWN_DONE" != true ] && [ -n "${INSTANCE_ID}" ]; then
    echo "" >&2
    echo "!!! Exiting (rc=${rc}). Terminating ${INSTANCE_ID} to avoid leaked billing." >&2
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
      --profile "$AWS_PROFILE" --region "$AWS_REGION" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM ERR

# ══════════════════════════════════════════════════════════════════════════════
# Usage
# ══════════════════════════════════════════════════════════════════════════════
usage() {
  cat <<EOF
run_checkpoint_eval.sh — per-checkpoint eval atom (P0). Evaluate ONE checkpoint on AWS and
land results in S3, with an idempotent skip-if-done guard.

USAGE:
  ./run_checkpoint_eval.sh [--dry-run] [--profile NAME] [--region REGION]
  ./run_checkpoint_eval.sh -h | --help

FLAGS:
  --dry-run          Print every aws command that would run; execute nothing.
  --profile NAME     AWS CLI profile (default: \$AWS_PROFILE or 'sbsandbox').
  --region REGION    AWS region       (default: \$AWS_REGION  or 'us-east-1').
  -h, --help         Show this help.

PREREQUISITES:
  aws CLI installed, and 'sb-aws-creds login' + 'sb-aws-creds install-profiles' run once
  so 'aws --profile ${AWS_PROFILE}' works.

OVERRIDABLE ENV VARS (current effective defaults shown):
  AWS_PROFILE=${AWS_PROFILE}        AWS_REGION=${AWS_REGION}
  INSTANCE_TYPE=${INSTANCE_TYPE}    AMI_ID=${AMI_ID}
  SECURITY_GROUP_ID=${SECURITY_GROUP_ID}
  SUBNET_ID=${SUBNET_ID}
  INSTANCE_PROFILE=${INSTANCE_PROFILE}
  CHECKPOINT=${CHECKPOINT}
  CHECKPOINT_KIND=${CHECKPOINT_KIND}   # hf (default, validated) | olmo_core (native, unvalidated)
  TASKS=${TASKS}    LIMIT=${LIMIT:-<none>}
  S3_BUCKET=${S3_BUCKET}
  S3_PREFIX=${S3_PREFIX}    S3_GROUP=${S3_GROUP}
  SKIP_IF_DONE=${SKIP_IF_DONE}   FORCE=${FORCE}   DONE_MARKER=${DONE_MARKER}
  SHUTDOWN_MINUTES=${SHUTDOWN_MINUTES}

STAGING (how olmo-eval gets onto the node):
  STAGE_MODE=${STAGE_MODE}   # 'clone' (default) or 'tar'
    clone — git clone GIT_REPO_URL on the node; needs NO local checkout. Uses:
      GIT_REPO_URL=${GIT_REPO_URL}
      GIT_REF=${GIT_REF:-<repo default branch>}   # optional branch/tag
    tar   — tar a local checkout -> S3 -> presign -> untar on node. Only this mode uses:
      LOCAL_REPO_PARENT=${LOCAL_REPO_PARENT}    REPO_DIRNAME=${REPO_DIRNAME}

  NODE_STAGE_DIR=${NODE_STAGE_DIR}
  SSM_ONLINE_TIMEOUT=${SSM_ONLINE_TIMEOUT}  STAGE_TIMEOUT=${STAGE_TIMEOUT}
  INSTALL_TIMEOUT=${INSTALL_TIMEOUT}        EVAL_TIMEOUT=${EVAL_TIMEOUT}
  PRESIGN_EXPIRY=${PRESIGN_EXPIRY}          CLEANUP_STAGING=${CLEANUP_STAGING}

Allowed instance types (hard safety cap): ${ALLOWED_INSTANCE_TYPES}
EOF
}

# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --profile) AWS_PROFILE="${2:?--profile needs a value}"; shift 2 ;;
    --region)  AWS_REGION="${2:?--region needs a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1  (try -h)" ;;
  esac
done

# ══════════════════════════════════════════════════════════════════════════════
# Banner
# ══════════════════════════════════════════════════════════════════════════════
log "Checkpoint-eval atom (P0)"
info "profile/region : ${AWS_PROFILE} / ${AWS_REGION}"
info "instance       : ${INSTANCE_TYPE}  ami=${AMI_ID}"
info "checkpoint     : ${CHECKPOINT}  (kind=${CHECKPOINT_KIND})"
info "eval           : -t ${TASKS}${LIMIT:+ -o limit=${LIMIT}}"
info "stage mode     : ${STAGE_MODE}"
if [ "$STAGE_MODE" = clone ]; then
  info "repo source    : git ${GIT_REPO_URL}${GIT_REF:+ (ref ${GIT_REF})}"
else
  info "repo source    : local ${LOCAL_REPO_PARENT}/${REPO_DIRNAME}"
fi
info "results dest   : ${RESULT_PREFIX}"
info "skip-if-done   : ${SKIP_IF_DONE} (marker=${DONE_MARKER}, force=${FORCE})"
info "self-destruct  : shutdown -h +${SHUTDOWN_MINUTES} (+ terminate-on-shutdown)"
[ "$DRY_RUN" = true ] && info ">>> DRY-RUN: no AWS commands will be executed <<<"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PRE-FLIGHT CHECKLIST (MANDATORY). Do not launch if any check fails.
# ══════════════════════════════════════════════════════════════════════════════
log "Pre-flight checks"

# 3.0 — aws CLI present.
command -v aws >/dev/null 2>&1 || die "aws CLI not found. Install AWS CLI v2 first."

# 3.0 — checkpoint kind must be one of the two supported formats (scoping doc §3.6).
case "$CHECKPOINT_KIND" in
  hf|olmo_core) info "checkpoint kind OK: ${CHECKPOINT_KIND}" ;;
  *) die "CHECKPOINT_KIND='${CHECKPOINT_KIND}' invalid. Use 'hf' (default) or 'olmo_core'." ;;
esac

# 3.0 — staging mode must be one of the two supported flows.
case "$STAGE_MODE" in
  clone|tar) info "stage mode OK: ${STAGE_MODE}" ;;
  *) die "STAGE_MODE='${STAGE_MODE}' invalid. Use 'clone' (default) or 'tar'." ;;
esac

# 3.0 — profile actually works (identity resolves). Read-only; skipped under --dry-run.
if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would verify: aws sts get-caller-identity --profile ${AWS_PROFILE}"
else
  who=$(aws sts get-caller-identity --query 'Arn' --output text \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null) \
    || die "Profile '${AWS_PROFILE}' can't authenticate. Did you run 'sb-aws-creds login' + 'sb-aws-creds install-profiles'?"
  info "authenticated as: ${who}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3.5 — IDEMPOTENCY / SKIP-IF-DONE (before we pay for a box)
# ══════════════════════════════════════════════════════════════════════════════
# The atom is safe to re-run: if this checkpoint's result marker is already in S3, there is
# nothing to do, so we skip the whole launch (no GPU minutes spent). This is exactly what
# lets the batch driver (§3.5) resume after a crash / spot reclaim / partial sweep.
log "Idempotency check (skip-if-done)"
if [ "$SKIP_IF_DONE" = true ] && [ "$FORCE" != true ]; then
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would check: aws s3 ls --recursive ${RESULT_PREFIX} for ${DONE_MARKER}"
  else
    existing=$(aws s3 ls "$RESULT_PREFIX" --recursive \
      --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || true)
    if printf '%s' "$existing" | grep -q "$DONE_MARKER"; then
      log "SKIP ✅  ${DONE_MARKER} already present under ${RESULT_PREFIX} — checkpoint already evaluated."
      info "Re-run with FORCE=true (or SKIP_IF_DONE=false) to evaluate anyway."
      exit 0
    fi
    info "no existing ${DONE_MARKER} under ${RESULT_PREFIX}; proceeding to launch"
  fi
else
  info "skip-if-done disabled (SKIP_IF_DONE=${SKIP_IF_DONE}, FORCE=${FORCE}); will evaluate unconditionally"
fi

# 3.1 — instance-type guard: reject anything not in the allowlist (never > g6.xlarge).
type_ok=false
for t in $ALLOWED_INSTANCE_TYPES; do
  [ "$INSTANCE_TYPE" = "$t" ] && type_ok=true
done
[ "$type_ok" = true ] || die "INSTANCE_TYPE='${INSTANCE_TYPE}' is not allowed. Use one of: ${ALLOWED_INSTANCE_TYPES}. If a checkpoint needs more than 24 GB, drop/restructure it — do NOT upsize."
info "instance-type OK: ${INSTANCE_TYPE} (<= g6.xlarge)"

# 3.2 — running/pending GPU-worker count. Launching must not exceed 3 total.
if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would count instances tagged Name=edullm-gpu-worker (running,pending)"
else
  running=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=edullm-gpu-worker" \
              "Name=instance-state-name,Values=running,pending" \
    --query 'length(Reservations[].Instances[])' --output text \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || echo "ERR")
  [ "$running" = "ERR" ] && die "Could not query running GPU instances."
  info "running/pending edullm-gpu-worker instances: ${running}"
  if [ "$running" -ge 3 ]; then
    die "Already ${running} GPU workers up; launching would exceed the 3-instance cap. Aborting."
  fi
fi

# 3.3 — the shutdown timer is baked into user-data below (verified by construction).
info "auto-shutdown timer: shutdown -h +${SHUTDOWN_MINUTES} will be in user-data"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LAUNCH ONE INSTANCE
# ══════════════════════════════════════════════════════════════════════════════
log "Launch"
USER_DATA=$(printf '#!/bin/bash\n# Checkpoint-eval auto-terminate safety net.\nshutdown -h +%s\n' "$SHUTDOWN_MINUTES")

INSTANCE_ID=$(run aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --count 1 \
  --iam-instance-profile "Name=${INSTANCE_PROFILE}" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --subnet-id "$SUBNET_ID" \
  --instance-initiated-shutdown-behavior terminate \
  --user-data "$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=edullm-gpu-worker},{Key=Purpose,Value=checkpoint-eval}]" \
  --query 'Instances[0].InstanceId' --output text \
  --profile "$AWS_PROFILE" --region "$AWS_REGION")

if [ "$DRY_RUN" = true ]; then
  INSTANCE_ID="i-DRYRUNXXXXXXXXXX"
  info "(dry-run) pretend InstanceId=${INSTANCE_ID}"
else
  [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ] || die "run-instances did not return an InstanceId."
  info "launched InstanceId=${INSTANCE_ID}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4b — WAIT FOR running, THEN FOR SSM Online
# ══════════════════════════════════════════════════════════════════════════════
log "Wait for 'running' + SSM Online"

run aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" \
  --profile "$AWS_PROFILE" --region "$AWS_REGION"

if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would poll: aws ssm describe-instance-information ... PingStatus == Online"
else
  info "instance running; waiting for SSM agent to report Online (up to ${SSM_ONLINE_TIMEOUT}s)..."
  waited=0
  while :; do
    ping=$(aws ssm describe-instance-information \
      --filters "Key=InstanceIds,Values=${INSTANCE_ID}" \
      --query 'InstanceInformationList[0].PingStatus' --output text \
      --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || echo "None")
    [ "$ping" = "Online" ] && { info "SSM PingStatus=Online"; break; }
    if [ "$waited" -ge "$SSM_ONLINE_TIMEOUT" ]; then
      die "SSM did not reach Online within ${SSM_ONLINE_TIMEOUT}s (PingStatus=${ping}). The instance profile or subnet may lack SSM reachability."
    fi
    sleep 10
    waited=$(( waited + 10 ))
  done
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STAGE the repo onto the node (clone default, tar opt-in)
# ══════════════════════════════════════════════════════════════════════════════
log "Stage repo onto node (mode=${STAGE_MODE})"

if [ "$STAGE_MODE" = clone ]; then
  BRANCH_OPT=""
  [ -n "$GIT_REF" ] && BRANCH_OPT="--branch ${GIT_REF} "
  STAGE_CMD="mkdir -p ${NODE_STAGE_DIR} && cd ${NODE_STAGE_DIR} && git clone --depth 1 ${BRANCH_OPT}${GIT_REPO_URL} && ls -d ${NODE_REPO_DIR}"
  ssm_invoke "$STAGE_CMD" "stage: git clone repo" "$STAGE_TIMEOUT" >/dev/null \
    || die "Staging (git clone) failed on the node."
  info "repo cloned at ${NODE_REPO_DIR}"
else
  # tar mode (opt-in): ship a LOCAL checkout that may contain uncommitted/private code.
  TARBALL="${TMPDIR:-/tmp}/olmo-eval-checkpoint-$$.tgz"
  [ -d "${LOCAL_REPO_PARENT}/${REPO_DIRNAME}" ] \
    || die "Local repo not found at ${LOCAL_REPO_PARENT}/${REPO_DIRNAME} (check LOCAL_REPO_PARENT / REPO_DIRNAME)."
  run tar czf "$TARBALL" -C "$LOCAL_REPO_PARENT" "$REPO_DIRNAME"

  run aws s3 cp "$TARBALL" "$STAGING_URI" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION"

  PRESIGNED_URL=$(run aws s3 presign "$STAGING_URI" \
    --expires-in "$PRESIGN_EXPIRY" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION")
  if [ "$DRY_RUN" = true ]; then
    PRESIGNED_URL="https://DRY-RUN-PRESIGNED-URL"
  else
    [ -n "$PRESIGNED_URL" ] || die "aws s3 presign returned an empty URL."
  fi

  STAGE_CMD="mkdir -p ${NODE_STAGE_DIR} && cd ${NODE_STAGE_DIR} && curl -sSL -o olmo-eval.tgz \"${PRESIGNED_URL}\" && tar xzf olmo-eval.tgz && ls -d ${NODE_REPO_DIR}"
  ssm_invoke "$STAGE_CMD" "stage: curl + untar repo" "$STAGE_TIMEOUT" >/dev/null \
    || die "Staging (curl+untar) failed on the node."
  info "repo staged at ${NODE_REPO_DIR}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5b — INSTALL deps with uv (synchronous, so failures are LOUD)
# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT_KIND drives the install (scoping doc §3.6): the HF default stays lean, and the
# olmo_core native path CONDITIONALLY adds the `olmo_core` extra (ai2-olmo-core) so the real
# OlmoCoreProvider can load a raw distributed checkpoint. The extra is only pulled when it's
# actually needed, keeping the common path light.
INSTALL_EXTRA_OPT=""
if [ "$CHECKPOINT_KIND" = olmo_core ]; then
  INSTALL_EXTRA_OPT=" --extra olmo_core"
  info "olmo_core kind: install will add the 'olmo_core' extra (ai2-olmo-core)"
fi
log "Install deps with uv${INSTALL_EXTRA_OPT}"
# SSM gives us no HOME (so uv at /root/.local/bin is unreachable and $HOME/.local/bin
# expands to /.local/bin), and the ~19 GB root disk can't hold torch+vLLM, so we pin the uv
# and HF caches onto the big NVMe scratch (NODE_STAGE_DIR).
INSTALL_CMD="cd ${NODE_REPO_DIR} && export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=${UV_CACHE_DIR_NODE} && export HF_HOME=${HF_HOME_NODE} && mkdir -p \"\$UV_CACHE_DIR\" \"\$HF_HOME\" && curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12 && uv sync --frozen${INSTALL_EXTRA_OPT} && uv run olmo-eval --help | head -n 20"
ssm_invoke "$INSTALL_CMD" "install: uv + python 3.12 + uv sync --frozen${INSTALL_EXTRA_OPT}" "$INSTALL_TIMEOUT" >/dev/null \
  || die "Dependency install failed on the node (see StandardErrorContent above)."
info "dependencies installed; olmo-eval CLI present"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RUN the eval DETACHED (avoids the SSM InProgress hang)
# ══════════════════════════════════════════════════════════════════════════════
log "Run eval detached"
# Provider selection (scoping doc §3.6): the HF default needs no override (the default
# provider loads HF-format checkpoints); the olmo_core native path selects the real
# OlmoCoreProvider via a harness override placed right after `-H default`.
RUN_PROVIDER_OPT=""
if [ "$CHECKPOINT_KIND" = olmo_core ]; then
  RUN_PROVIDER_OPT="-H default -o provider.kind=olmo_core "
fi
LIMIT_OPT=""
[ -n "$LIMIT" ] && LIMIT_OPT=" -o limit=${LIMIT}"

# We wrap the eval in `bash -c '... ; echo SENTINEL=$? >> log'` so that after the eval
# process exits, a line like "CHECKPOINT_DONE_EXIT=0" is appended to the log — a reliable
# completion marker AND the exit code when we poll. `setsid nohup ... &` fully detaches.
RUN_INNER="export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=${UV_CACHE_DIR_NODE} && export HF_HOME=${HF_HOME_NODE} && mkdir -p \"\$UV_CACHE_DIR\" \"\$HF_HOME\" && uv run olmo-eval run ${RUN_PROVIDER_OPT}-m ${CHECKPOINT} -t ${TASKS}${LIMIT_OPT} -O ${NODE_RESULTS_DIR} --s3-bucket ${S3_BUCKET} --s3-prefix ${S3_PREFIX} --s3-group ${S3_GROUP} --s3-region ${AWS_REGION} > ${NODE_LOG} 2>&1; echo \"${DONE_SENTINEL}=\$?\" >> ${NODE_LOG}"
RUN_CMD="cd ${NODE_REPO_DIR} && export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=${UV_CACHE_DIR_NODE} && export HF_HOME=${HF_HOME_NODE} && setsid nohup bash -c '${RUN_INNER}' >/dev/null 2>&1 &"
ssm_invoke "$RUN_CMD" "run: launch eval (detached)" 120 >/dev/null \
  || die "Failed to launch the eval on the node."
info "eval launched; logging to ${NODE_LOG}"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6b — POLL the on-node log until the eval finishes (or times out)
# ══════════════════════════════════════════════════════════════════════════════
log "Poll on-node log for completion"
# The completion sentinel MUST survive SSM's ~24 KB StandardOutputContent cap: olmo-eval's
# log is full of large Rich tables, so a plain `tail -n 200` can exceed 24 KB and truncate
# the trailing "${DONE_SENTINEL}=N" line. Fix: emit the sentinel line FIRST (always within
# the first bytes), then a small byte-bounded tail purely for a human heartbeat.
TAIL_CMD="{ grep -aE \"${DONE_SENTINEL}=[0-9]+\" ${NODE_LOG} 2>/dev/null | tail -n 1; tail -c 1500 ${NODE_LOG} 2>/dev/null; } 2>/dev/null || true"

if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would repeatedly SSM: ${TAIL_CMD} and wait for '${DONE_SENTINEL}=0'"
else
  waited=0; interval=15; eval_rc=""
  while :; do
    out=$(ssm_invoke "$TAIL_CMD" "poll: tail checkpoint_eval.log" 120 || true)
    last_line=$(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -n 1 || true)
    [ -n "$last_line" ] && info "log> ${last_line}"

    if printf '%s' "$out" | grep -q "${DONE_SENTINEL}=0"; then
      eval_rc=0; break
    fi
    if printf '%s' "$out" | grep -Eq "${DONE_SENTINEL}=[1-9][0-9]*"; then
      eval_rc=$(printf '%s' "$out" | grep -Eo "${DONE_SENTINEL}=[0-9]+" | tail -n1 | cut -d= -f2)
      break
    fi
    if [ "$waited" -ge "$EVAL_TIMEOUT" ]; then
      die "Eval did not complete within ${EVAL_TIMEOUT}s. Last log line: ${last_line:-<empty>}"
    fi
    sleep "$interval"
    waited=$(( waited + interval ))
  done

  if [ "$eval_rc" != 0 ]; then
    printf '%s\n' "$out" | tail -n 40 >&2
    die "Eval process exited non-zero (${DONE_SENTINEL}=${eval_rc}). See log tail above."
  fi
  info "eval finished cleanly (${DONE_SENTINEL}=0)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — VERIFY results in S3 (assert the result marker exists)
# ══════════════════════════════════════════════════════════════════════════════
log "Verify results in S3"

if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would verify: aws s3 ls --recursive ${RESULT_PREFIX} contains ${DONE_MARKER}"
  VERIFY_PASS=true
else
  listing=$(aws s3 ls "$RESULT_PREFIX" --recursive \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || true)
  printf '%s\n' "$listing" >&2
  if printf '%s' "$listing" | grep -q "$DONE_MARKER"; then
    VERIFY_PASS=true
    info "found ${DONE_MARKER} under ${RESULT_PREFIX}"
  else
    VERIFY_PASS=false
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TEARDOWN (explicit) + optional staging cleanup
# ══════════════════════════════════════════════════════════════════════════════
terminate_and_confirm
TEARDOWN_DONE=true

if [ "$STAGE_MODE" = tar ] && [ "$CLEANUP_STAGING" = true ]; then
  log "Remove staging tarball from S3"
  run aws s3 rm "$STAGING_URI" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" >/dev/null || true
fi

# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════
if [ "$DRY_RUN" = true ]; then
  log "DRY-RUN COMPLETE — nothing was executed. Re-run without --dry-run to go live."
  exit 0
fi

if [ "${VERIFY_PASS:-false}" = true ]; then
  log "CHECKPOINT EVAL: PASS ✅  ${DONE_MARKER} present at ${RESULT_PREFIX}"
  info "Inspect it with:"
  info "  aws s3 sync ${RESULT_PREFIX} ./checkpoint-results --profile ${AWS_PROFILE} --region ${AWS_REGION}"
  exit 0
else
  die "CHECKPOINT EVAL: FAIL — no ${DONE_MARKER} under ${RESULT_PREFIX}. Instance already terminated."
fi
