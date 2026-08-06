#!/usr/bin/env bash
#
# run_checkpoint_diag.sh — the style-agnostic CAT-diagnostics invocation seam.
#
# This is the shared entry point that runs a CAT diagnostic (MCQ or FRQ, any style)
# on ONE checkpoint on AWS and lands the report in S3. It is the diagnostics sibling
# of run_checkpoint_eval.sh: it REUSES the atom's proven mechanics (pre-flight caps,
# launch, wait-for-SSM, stage, install, detached run + 24 KB-safe sentinel poll,
# verify, triple self-terminate teardown) and swaps ONLY the on-node run command from
# `olmo-eval run` to the diagnostics runner:
#
#     python -m diagnostics.${DIAGNOSTIC_MODALITY}.runner --cat-style ${CAT_STYLE} ...
#
# Because the runner resolves --cat-style through its auto-discovering registry, this
# script never changes when a style is added: DIAGNOSTIC_MODALITY and CAT_STYLE are
# just values. All four style branches (flow/uni-mcq, flow/mirt-mcq, flow/uni-frq,
# flow/mirt-frq) invoke this identical script; none edits it. See
# Plan/checkpoint_diagnostics_wiring/README.md.
#
# MCQ scores in-process (checkpoint loaded on the box). FRQ needs two served vLLM
# endpoints — the tutor (checkpoint under test) and the frozen judge; on-node serving
# is an integration point, so for FRQ this seam requires TUTOR_ENDPOINT + JUDGE_ENDPOINT
# to be reachable from the node (pass them in), and threads --judge-config through.
#
# EVERYTHING is overridable via environment variables (see the CONFIG block / -h).
# --dry-run prints every aws command and the resolved on-node run command; runs nothing.

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — env var with default (team values from the runbook). Override by exporting.
# ══════════════════════════════════════════════════════════════════════════════

AWS_PROFILE="${AWS_PROFILE:-sbsandbox}"
AWS_REGION="${AWS_REGION:-us-east-1}"

INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
AMI_ID="${AMI_ID:-ami-0b6f2229ad14c9323}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-087218d8c87aa8576}"
SUBNET_ID="${SUBNET_ID:-subnet-0a4235fb98b63930f}"
INSTANCE_PROFILE="${INSTANCE_PROFILE:-EswManagedInstance}"

# --- The diagnostic + checkpoint ---------------------------------------------
DIAGNOSTIC_MODALITY="${DIAGNOSTIC_MODALITY:-mcq_cat}"   # mcq_cat | frq_cat
CAT_STYLE="${CAT_STYLE:-}"                              # registered style name (required)
CHECKPOINT="${CHECKPOINT:-Qwen/Qwen2.5-0.5B-Instruct}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-hf}"               # hf (default) | olmo_core
BENCHMARK="${BENCHMARK:-}"                              # optional; else the style default
IRT_PARAMS="${IRT_PARAMS:-}"                            # optional; else the style's bundled bank
SE_THRESHOLD="${SE_THRESHOLD:-0.3}"
MAX_ITEMS="${MAX_ITEMS:-50}"

# --- FRQ-only served endpoints (reachable from the node) ---------------------
TUTOR_ENDPOINT="${TUTOR_ENDPOINT:-}"                    # checkpoint-under-test /v1 base URL
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-}"                    # frozen judge /v1 base URL
SERVED_MODEL="${SERVED_MODEL:-tutor}"
JUDGE_CONFIG="${JUDGE_CONFIG:-}"                        # judge_frozen.yaml path on the node

# --- S3 result namespace (same grant story as the atom, §5) ------------------
S3_BUCKET="${S3_BUCKET:-edullm-adaptive-inference-056956104102}"
S3_PREFIX="${S3_PREFIX:-smoke}"
S3_GROUP="${S3_GROUP:-diagnostics}"

# --- Idempotency (skip-if-done) ----------------------------------------------
SKIP_IF_DONE="${SKIP_IF_DONE:-true}"
FORCE="${FORCE:-false}"
DONE_MARKER="${DONE_MARKER:-cat_report.json}"

# --- Staging (how the repo reaches the node) ---------------------------------
STAGE_MODE="${STAGE_MODE:-clone}"                      # clone | tar
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/allenai/olmo-eval}"
GIT_REF="${GIT_REF:-}"
LOCAL_REPO_PARENT="${LOCAL_REPO_PARENT:-/Users/cat/alpha-projects}"
REPO_DIRNAME="${REPO_DIRNAME:-olmo-eval-full}"

SHUTDOWN_MINUTES="${SHUTDOWN_MINUTES:-90}"

# --- On-node paths (DLAMI fast NVMe scratch) ---------------------------------
NODE_STAGE_DIR="${NODE_STAGE_DIR:-/opt/dlami/nvme/checkpoint-diag}"
NODE_RESULTS_DIR="${NODE_RESULTS_DIR:-${NODE_STAGE_DIR}/results}"
NODE_LOG="${NODE_LOG:-${NODE_STAGE_DIR}/checkpoint_diag.log}"
NODE_HOME="${NODE_HOME:-/root}"
UV_CACHE_DIR_NODE="${UV_CACHE_DIR_NODE:-${NODE_STAGE_DIR}/uv-cache}"
HF_HOME_NODE="${HF_HOME_NODE:-${NODE_STAGE_DIR}/hf}"

case "$STAGE_MODE" in
  tar) REPO_BASENAME="$REPO_DIRNAME" ;;
  *)   REPO_BASENAME="${GIT_REPO_URL##*/}"; REPO_BASENAME="${REPO_BASENAME%.git}" ;;
esac
NODE_REPO_DIR="${NODE_REPO_DIR:-${NODE_STAGE_DIR}/${REPO_BASENAME}}"

SSM_ONLINE_TIMEOUT="${SSM_ONLINE_TIMEOUT:-300}"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-300}"
INSTALL_TIMEOUT="${INSTALL_TIMEOUT:-1800}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
PRESIGN_EXPIRY="${PRESIGN_EXPIRY:-3600}"
CLEANUP_STAGING="${CLEANUP_STAGING:-true}"

ALLOWED_INSTANCE_TYPES="g5.xlarge g6.xlarge"
DONE_SENTINEL="DIAG_DONE_EXIT"

DRY_RUN=false
INSTANCE_ID=""
TARBALL=""
TEARDOWN_DONE=false

STAGING_KEY="${S3_PREFIX}/staging/olmo-eval.tgz"
STAGING_URI="s3://${S3_BUCKET}/${STAGING_KEY}"
RESULT_PREFIX="s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/${CAT_STYLE}/"

# ══════════════════════════════════════════════════════════════════════════════
# Small helpers (mirror run_checkpoint_eval.sh)
# ══════════════════════════════════════════════════════════════════════════════
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

run() {
  printf '    + %s\n' "$(quote_cmd "$@")" >&2
  [ "$DRY_RUN" = true ] && return 0
  "$@"
}

log()  { printf '\n=== %s\n' "$*" >&2; }
info() { printf '    %s\n' "$*" >&2; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

json_escape() {
  local s=$1
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  printf '%s' "$s"
}

ssm_params() { printf '{"commands":["%s"]}' "$(json_escape "$1")"; }

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
    info "(dry-run) would poll: aws ssm get-command-invocation --command-id <id>"
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
      *) : ;;
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
# Teardown — guaranteed on ANY exit so we never leak a billing instance.
# ══════════════════════════════════════════════════════════════════════════════
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
run_checkpoint_diag.sh — style-agnostic CAT-diagnostics seam. Run one CAT diagnostic
(MCQ or FRQ, any registered style) on ONE checkpoint on AWS and land the report in S3.

USAGE:
  DIAGNOSTIC_MODALITY=mcq_cat CAT_STYLE=uni_2pl \\
    CHECKPOINT=s3://bucket/checkpoints/run/step_1000 ./run_checkpoint_diag.sh
  DIAGNOSTIC_MODALITY=frq_cat CAT_STYLE=uni_frq \\
    TUTOR_ENDPOINT=http://host:8000/v1 JUDGE_ENDPOINT=http://host:8001/v1 \\
    JUDGE_CONFIG=/path/judge_frozen.yaml ./run_checkpoint_diag.sh
  ./run_checkpoint_diag.sh --dry-run | -h

FLAGS:
  --dry-run          Print every aws command + the on-node run command; execute nothing.
  --profile NAME     AWS CLI profile (default: \$AWS_PROFILE or 'sbsandbox').
  --region REGION    AWS region       (default: \$AWS_REGION  or 'us-east-1').
  -h, --help         Show this help.

KEY ENV VARS (current effective defaults shown):
  DIAGNOSTIC_MODALITY=${DIAGNOSTIC_MODALITY}   # mcq_cat | frq_cat
  CAT_STYLE=${CAT_STYLE:-<required>}           # any registered style (auto-discovered)
  CHECKPOINT=${CHECKPOINT}    CHECKPOINT_KIND=${CHECKPOINT_KIND}
  BENCHMARK=${BENCHMARK:-<style default>}   IRT_PARAMS=${IRT_PARAMS:-<style bank>}
  SE_THRESHOLD=${SE_THRESHOLD}   MAX_ITEMS=${MAX_ITEMS}
  S3_BUCKET=${S3_BUCKET}   S3_PREFIX=${S3_PREFIX}   S3_GROUP=${S3_GROUP}
  results dest -> ${RESULT_PREFIX} (marker: ${DONE_MARKER})
  FRQ only: TUTOR_ENDPOINT / JUDGE_ENDPOINT / JUDGE_CONFIG (must be reachable from the node)

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
# Banner + validation
# ══════════════════════════════════════════════════════════════════════════════
log "Checkpoint CAT-diagnostics seam"
info "profile/region : ${AWS_PROFILE} / ${AWS_REGION}"
info "instance       : ${INSTANCE_TYPE}  ami=${AMI_ID}"
info "diagnostic     : ${DIAGNOSTIC_MODALITY} / style=${CAT_STYLE:-<unset>}"
info "checkpoint     : ${CHECKPOINT}  (kind=${CHECKPOINT_KIND})"
info "results dest   : ${RESULT_PREFIX}"
[ "$DRY_RUN" = true ] && info ">>> DRY-RUN: no AWS commands will be executed <<<"

command -v aws >/dev/null 2>&1 || die "aws CLI not found. Install AWS CLI v2 first."
[ -n "$CAT_STYLE" ] || die "CAT_STYLE is required (the registered style name)."
case "$DIAGNOSTIC_MODALITY" in
  mcq_cat|frq_cat) : ;;
  *) die "DIAGNOSTIC_MODALITY='${DIAGNOSTIC_MODALITY}' invalid. Use 'mcq_cat' or 'frq_cat'." ;;
esac
case "$CHECKPOINT_KIND" in
  hf|olmo_core) : ;;
  *) die "CHECKPOINT_KIND='${CHECKPOINT_KIND}' invalid. Use 'hf' or 'olmo_core'." ;;
esac
case "$STAGE_MODE" in
  clone|tar) : ;;
  *) die "STAGE_MODE='${STAGE_MODE}' invalid. Use 'clone' or 'tar'." ;;
esac
if [ "$DIAGNOSTIC_MODALITY" = frq_cat ]; then
  { [ -n "$TUTOR_ENDPOINT" ] && [ -n "$JUDGE_ENDPOINT" ] && [ -n "$JUDGE_CONFIG" ]; } \
    || die "FRQ needs TUTOR_ENDPOINT + JUDGE_ENDPOINT + JUDGE_CONFIG (served endpoints reachable from the node)."
fi

type_ok=false
for t in $ALLOWED_INSTANCE_TYPES; do [ "$INSTANCE_TYPE" = "$t" ] && type_ok=true; done
[ "$type_ok" = true ] || die "INSTANCE_TYPE='${INSTANCE_TYPE}' not allowed. Use one of: ${ALLOWED_INSTANCE_TYPES}."

if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would verify: aws sts get-caller-identity --profile ${AWS_PROFILE}"
else
  who=$(aws sts get-caller-identity --query 'Arn' --output text \
    --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null) \
    || die "Profile '${AWS_PROFILE}' can't authenticate. Run 'sb-aws-creds login' + 'install-profiles'."
  info "authenticated as: ${who}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Idempotency / skip-if-done (before we pay for a box)
# ══════════════════════════════════════════════════════════════════════════════
log "Idempotency check (skip-if-done)"
if [ "$SKIP_IF_DONE" = true ] && [ "$FORCE" != true ]; then
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would check: aws s3 ls --recursive ${RESULT_PREFIX} for ${DONE_MARKER}"
  else
    existing=$(aws s3 ls "$RESULT_PREFIX" --recursive \
      --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || true)
    if printf '%s' "$existing" | grep -q "$DONE_MARKER"; then
      log "SKIP ✅  ${DONE_MARKER} already present under ${RESULT_PREFIX} — already run."
      exit 0
    fi
    info "no existing ${DONE_MARKER}; proceeding to launch"
  fi
else
  info "skip-if-done disabled (SKIP_IF_DONE=${SKIP_IF_DONE}, FORCE=${FORCE})"
fi

# ≤3-worker cap
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
  [ "$running" -ge 3 ] && die "Already ${running} GPU workers up; launching would exceed the 3-instance cap."
fi

# ══════════════════════════════════════════════════════════════════════════════
# Launch + wait for SSM Online
# ══════════════════════════════════════════════════════════════════════════════
log "Launch"
USER_DATA=$(printf '#!/bin/bash\n# Checkpoint-diag auto-terminate safety net.\nshutdown -h +%s\n' "$SHUTDOWN_MINUTES")
INSTANCE_ID=$(run aws ec2 run-instances \
  --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" --count 1 \
  --iam-instance-profile "Name=${INSTANCE_PROFILE}" \
  --security-group-ids "$SECURITY_GROUP_ID" --subnet-id "$SUBNET_ID" \
  --instance-initiated-shutdown-behavior terminate --user-data "$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=edullm-gpu-worker},{Key=Purpose,Value=checkpoint-diag}]" \
  --query 'Instances[0].InstanceId' --output text \
  --profile "$AWS_PROFILE" --region "$AWS_REGION")

if [ "$DRY_RUN" = true ]; then
  INSTANCE_ID="i-DRYRUNXXXXXXXXXX"
  info "(dry-run) pretend InstanceId=${INSTANCE_ID}"
else
  [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ] || die "run-instances did not return an InstanceId."
  info "launched InstanceId=${INSTANCE_ID}"
fi

log "Wait for 'running' + SSM Online"
run aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" \
  --profile "$AWS_PROFILE" --region "$AWS_REGION"
if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would poll SSM PingStatus == Online"
else
  waited=0
  while :; do
    ping=$(aws ssm describe-instance-information \
      --filters "Key=InstanceIds,Values=${INSTANCE_ID}" \
      --query 'InstanceInformationList[0].PingStatus' --output text \
      --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || echo "None")
    [ "$ping" = "Online" ] && { info "SSM PingStatus=Online"; break; }
    [ "$waited" -ge "$SSM_ONLINE_TIMEOUT" ] && die "SSM not Online within ${SSM_ONLINE_TIMEOUT}s (PingStatus=${ping})."
    sleep 10; waited=$(( waited + 10 ))
  done
fi

# ══════════════════════════════════════════════════════════════════════════════
# Stage repo + install deps (mirrors the atom)
# ══════════════════════════════════════════════════════════════════════════════
log "Stage repo onto node (mode=${STAGE_MODE})"
if [ "$STAGE_MODE" = clone ]; then
  BRANCH_OPT=""; [ -n "$GIT_REF" ] && BRANCH_OPT="--branch ${GIT_REF} "
  STAGE_CMD="mkdir -p ${NODE_STAGE_DIR} && cd ${NODE_STAGE_DIR} && git clone --depth 1 ${BRANCH_OPT}${GIT_REPO_URL} && ls -d ${NODE_REPO_DIR}"
  ssm_invoke "$STAGE_CMD" "stage: git clone repo" "$STAGE_TIMEOUT" >/dev/null || die "Staging (git clone) failed."
else
  TARBALL="${TMPDIR:-/tmp}/olmo-eval-diag-$$.tgz"
  [ -d "${LOCAL_REPO_PARENT}/${REPO_DIRNAME}" ] || die "Local repo not found at ${LOCAL_REPO_PARENT}/${REPO_DIRNAME}."
  run tar czf "$TARBALL" -C "$LOCAL_REPO_PARENT" "$REPO_DIRNAME"
  run aws s3 cp "$TARBALL" "$STAGING_URI" --profile "$AWS_PROFILE" --region "$AWS_REGION"
  PRESIGNED_URL=$(run aws s3 presign "$STAGING_URI" --expires-in "$PRESIGN_EXPIRY" \
    --profile "$AWS_PROFILE" --region "$AWS_REGION")
  [ "$DRY_RUN" = true ] && PRESIGNED_URL="https://DRY-RUN-PRESIGNED-URL"
  [ -n "$PRESIGNED_URL" ] || die "aws s3 presign returned an empty URL."
  STAGE_CMD="mkdir -p ${NODE_STAGE_DIR} && cd ${NODE_STAGE_DIR} && curl -sSL -o olmo-eval.tgz \"${PRESIGNED_URL}\" && tar xzf olmo-eval.tgz && ls -d ${NODE_REPO_DIR}"
  ssm_invoke "$STAGE_CMD" "stage: curl + untar repo" "$STAGE_TIMEOUT" >/dev/null || die "Staging (curl+untar) failed."
fi
info "repo staged at ${NODE_REPO_DIR}"

# olmo_core checkpoints need the extra so the provider can load raw sharded weights.
INSTALL_EXTRA_OPT=""
[ "$CHECKPOINT_KIND" = olmo_core ] && INSTALL_EXTRA_OPT=" --extra olmo_core"
log "Install deps with uv${INSTALL_EXTRA_OPT}"
INSTALL_CMD="cd ${NODE_REPO_DIR} && export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=${UV_CACHE_DIR_NODE} && export HF_HOME=${HF_HOME_NODE} && mkdir -p \"\$UV_CACHE_DIR\" \"\$HF_HOME\" && curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12 && uv sync --frozen${INSTALL_EXTRA_OPT}"
ssm_invoke "$INSTALL_CMD" "install: uv + python 3.12 + uv sync --frozen${INSTALL_EXTRA_OPT}" "$INSTALL_TIMEOUT" >/dev/null \
  || die "Dependency install failed on the node."
info "dependencies installed"

# ══════════════════════════════════════════════════════════════════════════════
# Build the style-agnostic on-node run command + run detached
# ══════════════════════════════════════════════════════════════════════════════
log "Run diagnostic detached (${DIAGNOSTIC_MODALITY} / ${CAT_STYLE})"
S3_OUT="${RESULT_PREFIX}"
COMMON_OPTS="--cat-style ${CAT_STYLE} --checkpoint ${CHECKPOINT} --checkpoint-kind ${CHECKPOINT_KIND} --s3-out ${S3_OUT} --se-threshold ${SE_THRESHOLD} --max-items ${MAX_ITEMS} --aws-region ${AWS_REGION}"
[ -n "$BENCHMARK" ]  && COMMON_OPTS="${COMMON_OPTS} --benchmark ${BENCHMARK}"
[ -n "$IRT_PARAMS" ] && COMMON_OPTS="${COMMON_OPTS} --irt-params ${IRT_PARAMS}"

if [ "$DIAGNOSTIC_MODALITY" = frq_cat ]; then
  MODALITY_OPTS="--tutor-endpoint ${TUTOR_ENDPOINT} --judge-endpoint ${JUDGE_ENDPOINT} --judge-config ${JUDGE_CONFIG} --served-model ${SERVED_MODEL}"
else
  MODALITY_OPTS=""
fi

RUN_INNER="export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=${UV_CACHE_DIR_NODE} && export HF_HOME=${HF_HOME_NODE} && mkdir -p \"\$UV_CACHE_DIR\" \"\$HF_HOME\" ${NODE_RESULTS_DIR} && uv run python -m diagnostics.${DIAGNOSTIC_MODALITY}.runner ${COMMON_OPTS} ${MODALITY_OPTS} > ${NODE_LOG} 2>&1; echo \"${DONE_SENTINEL}=\$?\" >> ${NODE_LOG}"
RUN_CMD="cd ${NODE_REPO_DIR} && export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && setsid nohup bash -c '${RUN_INNER}' >/dev/null 2>&1 &"
info "on-node run: uv run python -m diagnostics.${DIAGNOSTIC_MODALITY}.runner ${COMMON_OPTS} ${MODALITY_OPTS}"
ssm_invoke "$RUN_CMD" "run: launch diagnostic (detached)" 120 >/dev/null || die "Failed to launch the diagnostic."
info "diagnostic launched; logging to ${NODE_LOG}"

# ══════════════════════════════════════════════════════════════════════════════
# Poll on-node log (24 KB-safe sentinel-first tail)
# ══════════════════════════════════════════════════════════════════════════════
log "Poll on-node log for completion"
TAIL_CMD="{ grep -aE \"${DONE_SENTINEL}=[0-9]+\" ${NODE_LOG} 2>/dev/null | tail -n 1; tail -c 1500 ${NODE_LOG} 2>/dev/null; } 2>/dev/null || true"
if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would poll: ${TAIL_CMD} and wait for '${DONE_SENTINEL}=0'"
else
  waited=0; interval=15; eval_rc=""
  while :; do
    out=$(ssm_invoke "$TAIL_CMD" "poll: tail checkpoint_diag.log" 120 || true)
    last_line=$(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -n 1 || true)
    [ -n "$last_line" ] && info "log> ${last_line}"
    if printf '%s' "$out" | grep -q "${DONE_SENTINEL}=0"; then eval_rc=0; break; fi
    if printf '%s' "$out" | grep -Eq "${DONE_SENTINEL}=[1-9][0-9]*"; then
      eval_rc=$(printf '%s' "$out" | grep -Eo "${DONE_SENTINEL}=[0-9]+" | tail -n1 | cut -d= -f2); break
    fi
    [ "$waited" -ge "$EVAL_TIMEOUT" ] && die "Diagnostic did not complete within ${EVAL_TIMEOUT}s. Last log: ${last_line:-<empty>}"
    sleep "$interval"; waited=$(( waited + interval ))
  done
  if [ "$eval_rc" != 0 ]; then
    printf '%s\n' "$out" | tail -n 40 >&2
    die "Diagnostic exited non-zero (${DONE_SENTINEL}=${eval_rc}). See log tail above."
  fi
  info "diagnostic finished cleanly (${DONE_SENTINEL}=0)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Verify + teardown
# ══════════════════════════════════════════════════════════════════════════════
log "Verify results in S3"
if [ "$DRY_RUN" = true ]; then
  info "(dry-run) would verify: aws s3 ls --recursive ${RESULT_PREFIX} contains ${DONE_MARKER}"
  VERIFY_PASS=true
else
  listing=$(aws s3 ls "$RESULT_PREFIX" --recursive --profile "$AWS_PROFILE" --region "$AWS_REGION" 2>/dev/null || true)
  printf '%s\n' "$listing" >&2
  if printf '%s' "$listing" | grep -q "$DONE_MARKER"; then VERIFY_PASS=true; info "found ${DONE_MARKER}"; else VERIFY_PASS=false; fi
fi

terminate_and_confirm
TEARDOWN_DONE=true
if [ "$STAGE_MODE" = tar ] && [ "$CLEANUP_STAGING" = true ]; then
  log "Remove staging tarball from S3"
  run aws s3 rm "$STAGING_URI" --profile "$AWS_PROFILE" --region "$AWS_REGION" >/dev/null || true
fi

if [ "$DRY_RUN" = true ]; then
  log "DRY-RUN COMPLETE — nothing executed. Re-run without --dry-run to go live."
  exit 0
fi
if [ "${VERIFY_PASS:-false}" = true ]; then
  log "CHECKPOINT DIAG: PASS ✅  ${DONE_MARKER} present at ${RESULT_PREFIX}"
  exit 0
else
  die "CHECKPOINT DIAG: FAIL — no ${DONE_MARKER} under ${RESULT_PREFIX}. Instance already terminated."
fi
