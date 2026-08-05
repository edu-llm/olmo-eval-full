#!/usr/bin/env bash
#
# run_rung1_smoketest.sh — ONE-SHOT laptop-side orchestrator for the olmo-eval
# "Rung 1" AWS smoke test. It drives the ENTIRE flow from your machine using the
# AWS CLI: pre-flight safety checks -> launch ONE GPU box -> stage the repo ->
# install deps -> run a tiny eval -> verify results in S3 -> terminate the box.
#
# This is the scripted equivalent of the manual steps in
#   RUNBOOK_rung1_smoketest.md   (authoritative step-by-step)
#   SMOKETEST_EXPLAINED.md       (plain-English background / glossary)
# Read those first if any term below (SSM, presigned URL, DLAMI, ...) is new.
#
# STAGING MODES (how olmo-eval reaches the GPU node — see STAGE_MODE):
#   clone (DEFAULT) — `git clone` the PUBLIC olmo-eval repo directly on the node.
#       The smoke test only exercises the built-in `arc_easy` task, which already
#       lives in the public repo, so a clone is all Rung 1 needs. This is the
#       default precisely so the smoke test requires NO local checkout/path.
#   tar             — tar a LOCAL checkout -> S3 -> presigned URL -> SSM curl+untar.
#       Opt in with STAGE_MODE=tar. Reserve it for later, when you must ship
#       uncommitted/private code that is NOT in the public repo (e.g. the custom
#       eduLLM tasks + the Qwen LLM-judge). Only this mode uses LOCAL_REPO_PARENT
#       / REPO_DIRNAME and the S3 staging object.
#
# ─────────────────────────────────────────────────────────────────────────────
# PREREQUISITES (do these once, on your laptop)
#   1. Install the AWS CLI v2  (`aws --version` should work).
#   2. Get team credentials through the broker and install a local profile:
#          sb-aws-creds login              # browser login (~30-day session)
#          sb-aws-creds install-profiles   # writes the `sbsandbox` profile into
#                                           # ~/.aws/config so `aws --profile
#                                           # sbsandbox ...` works transparently
#      After that, `aws sts get-caller-identity --profile sbsandbox` must succeed.
#      (This script runs raw `aws` locally with a named profile; it does NOT go
#      through the sb_aws MCP tool. `install-profiles` is what makes that work.)
#
# USAGE
#   ./run_rung1_smoketest.sh                 # real run with team defaults
#   ./run_rung1_smoketest.sh --dry-run       # PREVIEW: print every aws command,
#                                            # execute nothing (safe first step)
#   ./run_rung1_smoketest.sh -h              # full help / all env vars
#
# EVERYTHING is overridable. Flags: --dry-run, --profile, --region, -h/--help.
# All other knobs are environment variables with runbook-matching defaults (see
# the CONFIG block below and `-h`). Example:
#   TASK=hellaswag LIMIT=5 ./run_rung1_smoketest.sh
#
# COST / SAFETY: a GPU box bills per hour. This script installs a trap that
# TERMINATES the launched instance on ANY exit (success, error, Ctrl-C), the
# launched box also self-terminates via `shutdown -h +N`, and we terminate
# explicitly at the end. Three independent safety nets — you should not leak a
# billing instance. Still, if you ever see an `edullm-gpu-worker` running later,
# terminate it.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — env var with default. Defaults are the exact team values from the
# runbook (Section 2 / Appendix). Override by exporting before you run, e.g.
#   MODEL=... TASK=... ./run_rung1_smoketest.sh
# ══════════════════════════════════════════════════════════════════════════════

# --- AWS access (also settable with --profile / --region flags) --------------
AWS_PROFILE="${AWS_PROFILE:-sbsandbox}"     # named profile from install-profiles
AWS_REGION="${AWS_REGION:-us-east-1}"       # team region (N. Virginia)

# --- EC2 / instance identity (team-provisioned; do not invent new ones) ------
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"                 # 1x A10G, 24 GB
AMI_ID="${AMI_ID:-ami-0b6f2229ad14c9323}"                  # Ubuntu 24.04 DLAMI
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-087218d8c87aa8576}"
SUBNET_ID="${SUBNET_ID:-subnet-0a4235fb98b63930f}"         # us-east-1b
INSTANCE_PROFILE="${INSTANCE_PROFILE:-EswManagedInstance}" # IAM "badge" -> S3 access

# --- The eval itself (cheapest thing that works) -----------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"   # ~0.5B, public/ungated
TASK="${TASK:-arc_easy}"                        # small logprob-scored MCQ
LIMIT="${LIMIT:-10}"                            # `-o limit=N` task override

# --- S3 destination for results (team bucket already exists) -----------------
# S3_PREFIX defaults to 'smoke' because the EswManagedInstance role's existing
# s3:PutObject grant covers only smoke/* (+ smoke_split/*, full200/*); results and
# tar-staging both land under smoke/ so auto-upload succeeds with no IAM change.
S3_BUCKET="${S3_BUCKET:-edullm-adaptive-inference-056956104102}"
S3_PREFIX="${S3_PREFIX:-smoke}"
S3_GROUP="${S3_GROUP:-rung1}"

# --- How the repo reaches the node (staging mode) ----------------------------
# clone (default): git clone the public repo on the node — no local path needed.
# tar:             ship a local checkout via S3 (for uncommitted/private code).
# See the STAGING MODES note in the header for when to use each.
STAGE_MODE="${STAGE_MODE:-clone}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/allenai/olmo-eval}"  # clone mode source
GIT_REF="${GIT_REF:-}"   # optional branch/tag to check out (empty = repo default branch)

# --- Safety timer + local repo location (LOCAL_* used by STAGE_MODE=tar ONLY) -
SHUTDOWN_MINUTES="${SHUTDOWN_MINUTES:-90}"                  # user-data self-destruct
LOCAL_REPO_PARENT="${LOCAL_REPO_PARENT:-/Users/cat/alpha-projects}"  # tar -C here
REPO_DIRNAME="${REPO_DIRNAME:-olmo-eval-full}"             # dir to tar/untar

# --- On-node paths (DLAMI fast NVMe scratch) ---------------------------------
NODE_STAGE_DIR="${NODE_STAGE_DIR:-/opt/dlami/nvme/rung1}"  # everything lives here
NODE_RESULTS_DIR="${NODE_RESULTS_DIR:-${NODE_STAGE_DIR}/results}"
NODE_LOG="${NODE_LOG:-${NODE_STAGE_DIR}/smoketest.log}"

# On-node env the SSM commands must set (SSM's AWS-RunShellScript runs as root but
# does NOT set HOME, and the DLAMI root disk is only ~19 GB — too small for torch +
# vLLM + model downloads — so HOME and all caches point at the big NVMe scratch).
NODE_HOME="${NODE_HOME:-/root}"                                   # SSM leaves HOME empty
UV_CACHE_DIR_NODE="${UV_CACHE_DIR_NODE:-${NODE_STAGE_DIR}/uv-cache}"  # keep uv cache off root disk
HF_HOME_NODE="${HF_HOME_NODE:-${NODE_STAGE_DIR}/hf}"             # keep HF downloads off root disk

# On-node repo directory. The name depends on the staging mode: a clone lands in
# a dir named after the git URL basename (e.g. olmo-eval), whereas a tar untars
# into REPO_DIRNAME (e.g. olmo-eval-full). Compute the per-mode default here so
# the later `cd ${NODE_REPO_DIR}` is correct; an explicit NODE_REPO_DIR override
# still wins. (Invalid STAGE_MODE is rejected in pre-flight before this is used.)
case "$STAGE_MODE" in
  tar) REPO_BASENAME="$REPO_DIRNAME" ;;
  *)   REPO_BASENAME="${GIT_REPO_URL##*/}"; REPO_BASENAME="${REPO_BASENAME%.git}" ;;
esac
NODE_REPO_DIR="${NODE_REPO_DIR:-${NODE_STAGE_DIR}/${REPO_BASENAME}}"

# --- Timeouts / behaviour knobs ----------------------------------------------
SSM_ONLINE_TIMEOUT="${SSM_ONLINE_TIMEOUT:-300}"   # wait for SSM agent Online (s)
STAGE_TIMEOUT="${STAGE_TIMEOUT:-300}"             # curl+untar on node (s)
INSTALL_TIMEOUT="${INSTALL_TIMEOUT:-1800}"        # uv sync can be slow (s)
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"              # model dl + vLLM + score (s)
PRESIGN_EXPIRY="${PRESIGN_EXPIRY:-3600}"          # presigned URL lifetime (s)
CLEANUP_STAGING="${CLEANUP_STAGING:-true}"        # rm staging tgz from S3 at end

# Allowlisted instance types — the team hard rule: never above g6.xlarge.
ALLOWED_INSTANCE_TYPES="g5.xlarge g6.xlarge"

# Sentinel we append to the on-node log after the eval process exits, so we can
# reliably detect completion + read its exit code by tailing the log (instead of
# guessing the accuracy line format or waiting on the SSM status, which hangs).
DONE_SENTINEL="RUNG1_DONE_EXIT"

# Runtime state (populated as we go).
DRY_RUN=false
INSTANCE_ID=""
TARBALL=""
TEARDOWN_DONE=false

# S3 key/uri for the staged code tarball.
STAGING_KEY="${S3_PREFIX}/staging/olmo-eval.tgz"
STAGING_URI="s3://${S3_BUCKET}/${STAGING_KEY}"

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

# run(): echo the command (always) then execute it — UNLESS --dry-run, in which
# case it only echoes. Its stdout is passed through so callers can capture it
# with $( ... ); the "+ ..." trace line goes to stderr so it never pollutes
# captured output.
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
# We only need to handle backslash and double-quote (our payloads have no raw
# newlines/control chars). Used to build the SSM --parameters JSON safely.
json_escape() {
  local s=$1
  s=${s//\\/\\\\}   # backslash -> \\   (must be first)
  s=${s//\"/\\\"}   # "         -> \"
  printf '%s' "$s"
}

# Build the AWS-RunShellScript --parameters value as JSON: {"commands":["..."]}.
ssm_params() { printf '{"commands":["%s"]}' "$(json_escape "$1")"; }

# ssm_invoke <payload> <description> [timeout_seconds]
# Sends ONE shell payload to the node via SSM, waits until the SSM command
# reaches a terminal state, prints the command's StandardOutputContent to stdout
# (for the caller to capture), and returns non-zero on any non-Success status or
# timeout. This is the correct SSM async pattern: send-command returns a
# CommandId, then we poll get-command-invocation for Status/output.
ssm_invoke() {
  local payload=$1 desc=${2:-"ssm command"} timeout=${3:-600}
  local params cid status out waited=0 interval=5
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
  # `wait instance-terminated` blocks until the box is fully gone.
  aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || true
  local state
  state=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[].Instances[].State.Name' --output text \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || echo unknown)
  info "Instance ${INSTANCE_ID} state: ${state}"
}

# trap handler: fires on EXIT/INT/TERM/ERR. Best-effort terminate of any
# launched instance, then remove the local tarball. No-op under --dry-run or if
# nothing was launched. Detaches itself first so it runs exactly once.
cleanup() {
  local rc=$?
  trap - EXIT INT TERM ERR

  # Always clean the local tarball if we made one.
  if [ -n "${TARBALL}" ] && [ -f "${TARBALL}" ]; then
    rm -f "$TARBALL" 2>/dev/null || true
  fi

  if [ "$DRY_RUN" = true ]; then exit "$rc"; fi

  # If we already tore down cleanly at the end, nothing to do.
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
run_rung1_smoketest.sh — one-shot orchestrator for the olmo-eval Rung 1 AWS smoke test.

USAGE:
  ./run_rung1_smoketest.sh [--dry-run] [--profile NAME] [--region REGION]
  ./run_rung1_smoketest.sh -h | --help

FLAGS:
  --dry-run          Print every aws command that would run; execute nothing.
  --profile NAME     AWS CLI profile (default: \$AWS_PROFILE or 'sbsandbox').
  --region REGION    AWS region       (default: \$AWS_REGION  or 'us-east-1').
  -h, --help         Show this help.

PREREQUISITES:
  aws CLI installed, and 'sb-aws-creds login' + 'sb-aws-creds install-profiles'
  run once so 'aws --profile ${AWS_PROFILE}' works.

OVERRIDABLE ENV VARS (current effective defaults shown):
  AWS_PROFILE=${AWS_PROFILE}        AWS_REGION=${AWS_REGION}
  INSTANCE_TYPE=${INSTANCE_TYPE}    AMI_ID=${AMI_ID}
  SECURITY_GROUP_ID=${SECURITY_GROUP_ID}
  SUBNET_ID=${SUBNET_ID}
  INSTANCE_PROFILE=${INSTANCE_PROFILE}
  MODEL=${MODEL}
  TASK=${TASK}    LIMIT=${LIMIT}
  S3_BUCKET=${S3_BUCKET}
  S3_PREFIX=${S3_PREFIX}    S3_GROUP=${S3_GROUP}
  SHUTDOWN_MINUTES=${SHUTDOWN_MINUTES}

STAGING (how olmo-eval gets onto the node):
  STAGE_MODE=${STAGE_MODE}   # 'clone' (default) or 'tar'
    clone — git clone GIT_REPO_URL on the node; needs NO local checkout. Uses:
      GIT_REPO_URL=${GIT_REPO_URL}
      GIT_REF=${GIT_REF:-<repo default branch>}   # optional branch/tag
    tar   — tar a local checkout -> S3 -> presign -> untar on node. Only this
            mode uses these (ignored in clone mode):
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
log "Rung 1 smoke-test orchestrator"
info "profile/region : ${AWS_PROFILE} / ${AWS_REGION}"
info "instance       : ${INSTANCE_TYPE}  ami=${AMI_ID}"
info "eval           : ${MODEL}  -t ${TASK} -o limit=${LIMIT}"
info "stage mode     : ${STAGE_MODE}"
if [ "$STAGE_MODE" = clone ]; then
  info "repo source    : git ${GIT_REPO_URL}${GIT_REF:+ (ref ${GIT_REF})}"
else
  info "repo source    : local ${LOCAL_REPO_PARENT}/${REPO_DIRNAME}"
fi
info "results dest   : s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/"
info "self-destruct  : shutdown -h +${SHUTDOWN_MINUTES} (+ terminate-on-shutdown)"
[ "$DRY_RUN" = true ] && info ">>> DRY-RUN: no AWS commands will be executed <<<"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PRE-FLIGHT CHECKLIST (MANDATORY). Do not launch if any check fails.
# ══════════════════════════════════════════════════════════════════════════════
log "Pre-flight checks (runbook Section 3)"

# 3.0 — aws CLI present.
command -v aws >/dev/null 2>&1 || die "aws CLI not found. Install AWS CLI v2 first."

# 3.0 — staging mode must be one of the two supported flows. In clone mode there
#        is NO local-repo requirement; the tar branch (Section 5) enforces the
#        local-repo-exists check itself.
case "$STAGE_MODE" in
  clone|tar) info "stage mode OK: ${STAGE_MODE}" ;;
  *) die "STAGE_MODE='${STAGE_MODE}' invalid. Use 'clone' (default) or 'tar'." ;;
esac

# 3.0 — profile actually works (identity resolves). Read-only; echoed+skipped
#        under --dry-run.
if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would verify: aws sts get-caller-identity --profile ${AWS_PROFILE}"
else
  who=$(aws sts get-caller-identity --query 'Arn' --output text \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null) \
    || die "Profile '${AWS_PROFILE}' can't authenticate. Did you run 'sb-aws-creds login' + 'sb-aws-creds install-profiles'?"
  info "authenticated as: ${who}"
fi

# 3.1 — instance-type guard: reject anything not in the allowlist (never > g6.xlarge).
type_ok=false
for t in $ALLOWED_INSTANCE_TYPES; do
  [ "$INSTANCE_TYPE" = "$t" ] && type_ok=true
done
[ "$type_ok" = true ] || die "INSTANCE_TYPE='${INSTANCE_TYPE}' is not allowed. Use one of: ${ALLOWED_INSTANCE_TYPES}. If a model needs more than 24 GB, drop the model — do NOT upsize."
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
log "Launch (runbook Section 4)"

# user-data: AWS runs this once at first boot. Its only job is to arm the
# self-destruct timer. Combined with --instance-initiated-shutdown-behavior
# terminate, "halt" becomes "terminate" (delete + stop billing).
USER_DATA=$(printf '#!/bin/bash\n# Rung 1 auto-terminate safety net.\nshutdown -h +%s\n' "$SHUTDOWN_MINUTES")

INSTANCE_ID=$(run aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --count 1 \
  --iam-instance-profile "Name=${INSTANCE_PROFILE}" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --subnet-id "$SUBNET_ID" \
  --instance-initiated-shutdown-behavior terminate \
  --user-data "$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=edullm-gpu-worker},{Key=Purpose,Value=rung1-smoketest}]" \
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
log "Wait for 'running' + SSM Online (runbook Section 4b)"

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
# SECTION 5 — STAGE the repo onto the node. Two flows, selected by STAGE_MODE:
#   clone (default): git clone the public repo on the node (no local repo needed).
#   tar:             tar local checkout -> S3 -> presign -> SSM curl+untar.
# ══════════════════════════════════════════════════════════════════════════════
log "Stage repo onto node (runbook Section 5, mode=${STAGE_MODE})"

if [ "$STAGE_MODE" = clone ]; then
  # Clone the public repo directly on the node — the built-in arc_easy task lives
  # there, so no local checkout or S3 staging is required. --depth 1 keeps it fast.
  # NOTE: `--depth 1 --branch <ref>` works for a branch or tag, but NOT for a full
  # commit sha (a shallow clone can't fetch an arbitrary sha). For a pinned commit,
  # use STAGE_MODE=tar (or a non-shallow clone). No fallback is wired up here.
  BRANCH_OPT=""
  [ -n "$GIT_REF" ] && BRANCH_OPT="--branch ${GIT_REF} "
  STAGE_CMD="mkdir -p ${NODE_STAGE_DIR} && cd ${NODE_STAGE_DIR} && git clone --depth 1 ${BRANCH_OPT}${GIT_REPO_URL} && ls -d ${NODE_REPO_DIR}"
  ssm_invoke "$STAGE_CMD" "stage: git clone repo" "$STAGE_TIMEOUT" >/dev/null \
    || die "Staging (git clone) failed on the node."
  info "repo cloned at ${NODE_REPO_DIR}"
else
  # tar mode (opt-in): ship a LOCAL checkout that may contain uncommitted/private
  # code the public repo lacks. This is the only path that touches LOCAL_REPO_*,
  # the S3 staging object, and the presigned URL.

  # 5.1 — tar the repo locally (from the PARENT of the repo dir).
  TARBALL="${TMPDIR:-/tmp}/olmo-eval-rung1-$$.tgz"
  [ -d "${LOCAL_REPO_PARENT}/${REPO_DIRNAME}" ] \
    || die "Local repo not found at ${LOCAL_REPO_PARENT}/${REPO_DIRNAME} (check LOCAL_REPO_PARENT / REPO_DIRNAME)."
  run tar czf "$TARBALL" -C "$LOCAL_REPO_PARENT" "$REPO_DIRNAME"

  # 5.2 — upload to the team bucket under the smoke-test staging path.
  run aws s3 cp "$TARBALL" "$STAGING_URI" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION"

  # 5.3 — presign a short-lived download URL (no login needed to fetch that one object).
  PRESIGNED_URL=$(run aws s3 presign "$STAGING_URI" \
    --expires-in "$PRESIGN_EXPIRY" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION")
  if [ "$DRY_RUN" = true ]; then
    PRESIGNED_URL="https://DRY-RUN-PRESIGNED-URL"
  else
    [ -n "$PRESIGNED_URL" ] || die "aws s3 presign returned an empty URL."
  fi

  # 5.4 — on the node: curl the URL + untar into the fast NVMe scratch dir.
  STAGE_CMD="mkdir -p ${NODE_STAGE_DIR} && cd ${NODE_STAGE_DIR} && curl -sSL -o olmo-eval.tgz \"${PRESIGNED_URL}\" && tar xzf olmo-eval.tgz && ls -d ${NODE_REPO_DIR}"
  ssm_invoke "$STAGE_CMD" "stage: curl + untar repo" "$STAGE_TIMEOUT" >/dev/null \
    || die "Staging (curl+untar) failed on the node."
  info "repo staged at ${NODE_REPO_DIR}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5b — INSTALL deps with uv (synchronous, so failures are LOUD)
# ══════════════════════════════════════════════════════════════════════════════
# NOTE ON REUSE: the repo ships an on-node helper `rung1_smoketest.sh` that does
# install + run in one detached process. We deliberately DO NOT invoke it here;
# instead we inline the equivalent commands so we can (a) run install
# SYNCHRONOUSLY and fail loudly if deps break, and (b) launch the eval DETACHED
# as its own step. Bundling both in one detached process would hide install
# failures behind a log poll. The inlined commands are byte-aligned with the
# helper and with runbook Sections 5-6; the helper remains on the node for
# manual use.
log "Install deps with uv (runbook Section 5)"
# Prelude the SSM shell must run before any uv call: SSM's AWS-RunShellScript gives
# us no HOME (so uv at /root/.local/bin is unreachable and $HOME/.local/bin expands
# to /.local/bin), and the ~19 GB root disk can't hold torch+vLLM, so we pin the uv
# and HF caches onto the big NVMe scratch (NODE_STAGE_DIR).
INSTALL_CMD="cd ${NODE_REPO_DIR} && export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=${UV_CACHE_DIR_NODE} && export HF_HOME=${HF_HOME_NODE} && mkdir -p \"\$UV_CACHE_DIR\" \"\$HF_HOME\" && curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12 && uv sync --frozen && uv run olmo-eval --help | head -n 20"
ssm_invoke "$INSTALL_CMD" "install: uv + python 3.12 + uv sync --frozen" "$INSTALL_TIMEOUT" >/dev/null \
  || die "Dependency install failed on the node (see StandardErrorContent above)."
info "dependencies installed; olmo-eval CLI present"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RUN the eval DETACHED (avoids the SSM InProgress hang)
# ══════════════════════════════════════════════════════════════════════════════
log "Run eval detached (runbook Section 6)"
# We wrap the eval in `bash -c '... ; echo SENTINEL=$? >> log'` so that after the
# eval process exits, a line like "RUNG1_DONE_EXIT=0" is appended to the log.
# That gives us a reliable completion marker AND the exit code when we poll.
# `setsid nohup ... &` fully detaches so the SSM command returns immediately.
# Same reason as the install prelude: SSM sets no HOME and the root disk is too
# small, so the detached eval process needs HOME/PATH/UV_CACHE_DIR/HF_HOME on the
# NVMe scratch too (the model download honours HF_HOME). We export inside the
# `bash -c` payload so the actual eval subprocess sees them, and also in the outer
# shell for good measure.
RUN_INNER="export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=${UV_CACHE_DIR_NODE} && export HF_HOME=${HF_HOME_NODE} && mkdir -p \"\$UV_CACHE_DIR\" \"\$HF_HOME\" && uv run olmo-eval run -m ${MODEL} -t ${TASK} -o limit=${LIMIT} -O ${NODE_RESULTS_DIR} --s3-bucket ${S3_BUCKET} --s3-prefix ${S3_PREFIX} --s3-group ${S3_GROUP} --s3-region ${AWS_REGION} > ${NODE_LOG} 2>&1; echo \"${DONE_SENTINEL}=\$?\" >> ${NODE_LOG}"
RUN_CMD="cd ${NODE_REPO_DIR} && export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=${UV_CACHE_DIR_NODE} && export HF_HOME=${HF_HOME_NODE} && setsid nohup bash -c '${RUN_INNER}' >/dev/null 2>&1 &"
ssm_invoke "$RUN_CMD" "run: launch eval (detached)" 120 >/dev/null \
  || die "Failed to launch the eval on the node."
info "eval launched; logging to ${NODE_LOG}"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6b — POLL the on-node log until the eval finishes (or times out)
# ══════════════════════════════════════════════════════════════════════════════
log "Poll on-node log for completion (runbook Section 6)"
# The completion sentinel MUST survive SSM's ~24 KB StandardOutputContent cap.
# olmo-eval's log is full of large Rich tables, so a plain `tail -n 200` can exceed
# 24 KB and be truncated BEFORE the trailing "${DONE_SENTINEL}=N" line — which makes a
# finished eval look like it never completes (the poll loop then spins to EVAL_TIMEOUT).
# Fix: emit the sentinel line FIRST (always within the first bytes, so it can't be
# truncated away), then a small byte-bounded tail purely for a human heartbeat.
TAIL_CMD="{ grep -aE \"${DONE_SENTINEL}=[0-9]+\" ${NODE_LOG} 2>/dev/null | tail -n 1; tail -c 1500 ${NODE_LOG} 2>/dev/null; } 2>/dev/null || true"

if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would repeatedly SSM: ${TAIL_CMD} and wait for '${DONE_SENTINEL}=0'"
else
  waited=0; interval=15; eval_rc=""
  while :; do
    out=$(ssm_invoke "$TAIL_CMD" "poll: tail smoketest.log" 120 || true)
    # Show the last non-blank line as a heartbeat (|| true: grep exits 1 on an
    # empty log early on, which would otherwise trip set -e/pipefail).
    last_line=$(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -n 1 || true)
    [ -n "$last_line" ] && info "log> ${last_line}"

    if printf '%s' "$out" | grep -q "${DONE_SENTINEL}=0"; then
      eval_rc=0; break
    fi
    # Non-zero sentinel -> the eval process failed.
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
    # Dump the tail for debugging, then fail.
    printf '%s\n' "$out" | tail -n 40 >&2
    die "Eval process exited non-zero (${DONE_SENTINEL}=${eval_rc}). See log tail above."
  fi
  info "eval finished cleanly (${DONE_SENTINEL}=0)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — VERIFY results in S3 (assert metrics.json exists)
# ══════════════════════════════════════════════════════════════════════════════
log "Verify results in S3 (runbook Section 7)"
RESULT_PREFIX="s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/"

if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would verify: aws s3 ls --recursive ${RESULT_PREFIX} contains metrics.json"
  VERIFY_PASS=true
else
  listing=$(aws s3 ls "$RESULT_PREFIX" --recursive \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || true)
  printf '%s\n' "$listing" >&2
  if printf '%s' "$listing" | grep -q 'metrics.json'; then
    VERIFY_PASS=true
    info "found metrics.json under ${RESULT_PREFIX}"
  else
    VERIFY_PASS=false
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TEARDOWN (explicit) + optional staging cleanup
# ══════════════════════════════════════════════════════════════════════════════
terminate_and_confirm
TEARDOWN_DONE=true

# Only tar mode uploads a staging object; clone mode never puts one in S3, so
# there is nothing to remove there (and no TARBALL for the trap to delete).
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
  log "RUNG 1: PASS ✅  metrics.json present at ${RESULT_PREFIX}"
  info "Inspect it with:"
  info "  aws s3 sync ${RESULT_PREFIX} ./rung1-results --profile ${AWS_PROFILE} --region ${AWS_REGION}"
  exit 0
else
  die "RUNG 1: FAIL — no metrics.json under ${RESULT_PREFIX}. Instance already terminated."
fi
