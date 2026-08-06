#!/usr/bin/env bash
#
# run_checkpoint_batch.sh — the retroactive batch / back-fill DRIVER (scoping doc §3.5).
# It sits beside the per-checkpoint atom (run_checkpoint_eval.sh) and, given a run's
# checkpoint root in S3, enumerates every step_N checkpoint and (in later sub-steps)
# dispatches the §3 atom across all of them on a small self-terminating GPU fleet.
#
# ─────────────────────────────────────────────────────────────────────────────
# SUB-STEPS IMPLEMENTED HERE:
#   P2.a — enumeration + identity-agnostic entrypoint (no GPU).
#   P2.b — the K=1 (FLEET_SIZE=1) single-worker path: launch ONE instance, stage +
#          install deps ONCE, then LOOP the enumerated checkpoints on that SAME box
#          (skip-if-done via _SUCCESS → run → verify → _SUCCESS → per-step cleanup),
#          write a worker manifest + summary, and tear down (LAPTOP-driven loop).
#   P2.c — the ≤3-worker FLEET (FLEET_SIZE 2..3): ONE `run-instances --count K_eff`,
#          publish the ordered to-do list to S3 (_batch/todo.json), and ship an on-node
#          slice-loop each worker runs autonomously — read `ami-launch-index` via IMDSv2,
#          install deps ONCE, then loop its static slice (i, i+K_eff, …) running the SAME
#          per-checkpoint sequence as K=1 and writing its shard _batch/worker_<i>.json
#          (NODE-driven loop). The laptop polls each exact id's slice sentinel and tears the
#          whole fleet down.
#   P2.d — shard-merge aggregation: after the fleet finishes, download every worker_<i>.json,
#          merge per-step outcomes, and tally done/skipped/failed across ALL workers PLUS the
#          gate pre-skipped checkpoints into a single _batch/summary.json (missing shards tolerated).
# The K=1 AND the on-node fleet command strings MIRROR the validated atom
# (run_checkpoint_eval.sh) rather than refactoring it, so the atom stays byte-for-byte
# untouched. K=1 (laptop-driven) and the fleet (node-driven) are SEPARATE mechanisms so the
# proven K=1 path is never routed through the new node script. --dry-run launches NOTHING and
# prints the whole planned sequence. See scoping doc §3.5.1–§3.5.6.
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT P2.a DOES
#   1. Parse the batch inputs (RUN_URI or an explicit CHECKPOINT_LIST, CHECKPOINT_KIND,
#      TASKS/LIMIT, STEP_FILTER, FLEET_SIZE, S3 result namespace) — env-overridable in the
#      same style as the atom, plus --dry-run/--profile/--region flags.
#   2. Enumerate the run's checkpoints (scoping doc §3.5.2): list the IMMEDIATE children of
#      RUN_URI (no auto-descend), keep those matching ^step_?[0-9]+$, sort NUMERICALLY by N,
#      apply STEP_FILTER, and per checkpoint run a cheap list-only format detect that WARNS
#      if the on-disk layout disagrees with CHECKPOINT_KIND. Produces the ordered
#      (step, uri, kind) list.
#   3. Resolve AWS credentials AMBIENTLY (scoping doc §3.5.6): AWS_PROFILE is OPTIONAL — the
#      same entrypoint runs laptop-side today and under a service/CI role later, no code
#      change. Nothing here hard-codes a human profile.
#   4. Validate FLEET_SIZE against the ≤3 hard cap (§4) as an inert check (not used yet).
#   5. Print the resolved config + the full ordered enumeration + what it WOULD dispatch.
#
# WHAT P2.b ADDS (K=1 / FLEET_SIZE=1 only)
#   6. Launch ONE self-terminating worker, stage + `uv sync` ONCE (honoring CHECKPOINT_KIND
#      → conditional `--extra olmo_core`), then loop the enumerated (step, uri) list on the
#      SAME instance: skip-if-done (S3 _SUCCESS/metrics.json), detached run + 24 KB-safe
#      sentinel poll per checkpoint, write an explicit _SUCCESS on success, exfil the on-node
#      log on failure (bounded retry, then CONTINUE), fresh per-step output + delete the prior
#      HF/NVMe download between steps, and record every outcome into a worker manifest +
#      _batch/summary.json.
#
# WHAT P2.c ADDS (FLEET_SIZE 2..3 — the node-side static shard fleet, §3.5.3)
#   7. After the SAME pre-launch gate, clamp K_eff = min(FLEET_SIZE, N_TODO, 3), publish the
#      ordered to-do subset to S3 (_batch/todo.json), launch K_eff workers with ONE
#      `run-instances --count K_eff` (reusing the subnet/type capacity fallback; ≤3 preflight
#      across the whole fleet), then ship an on-node slice-loop (base64 heredoc via SSM,
#      detached). Each worker reads its `ami-launch-index` (IMDSv2 token dance), installs deps
#      ONCE, loops its static slice (positions i, i+K_eff, …) of the shared to-do list running
#      the SAME per-checkpoint sequence, writes _batch/worker_<i>.json, appends a whole-slice
#      sentinel to its on-node log, and self-shuts (grace timer). The laptop polls each exact
#      id's sentinel (24 KB-safe) and terminates ALL of OUR ids, then P2.d merges the shards.
#
# PRE-LAUNCH GATE (correctness + cost): a read-only S3 done-marker check runs for EVERY
#   enumerated checkpoint BEFORE anything is launched (honoring SKIP_IF_DONE/FORCE). If all
#   are already done the driver records them 'skipped', writes the manifest/summary and returns
#   with ZERO GPU (no run-instances/stage/install); otherwise it launches and loops ONLY the
#   to-do subset (pre-satisfied ones still counted as skipped). --dry-run prints this split.
#
# CAPACITY RESILIENCE: launch tries an ordered subnet list (SUBNET_IDS, default = single
#   SUBNET_ID) and, on InsufficientInstanceCapacity, retries the next subnet; with
#   INSTANCE_TYPE_FALLBACK=true it also retries the other ALLOWED_INSTANCE_TYPES (never above
#   g6.xlarge). run-instances' exit status is captured explicitly so a real launch failure
#   aborts non-zero (no pipe/tee masking). The ≤3-worker and allowed-type caps are unchanged.
#
# EXPLICIT-RUN_URI RULE (approved, scoping doc §3.5.2)
#   RUN_URI must point DIRECTLY at the parent of the step* dirs — the driver does NOT
#   auto-descend into a checkpoints/ child. The step* dirs may live at a bucket root
#   (s3://edullm-olmo-100m-bpe-ckpts/step7629) or nested
#   (s3://edullm-checkpoints/olmo-370m/<run>/checkpoints/step940); point RUN_URI at
#   whichever prefix directly holds them.
#
# USAGE
#   RUN_URI=s3://edullm-olmo-100m-bpe-ckpts/ ./run_checkpoint_batch.sh --dry-run
#   RUN_URI=s3://edullm-checkpoints/olmo-370m/<run>/checkpoints/ \
#     CHECKPOINT_KIND=olmo_core ./run_checkpoint_batch.sh --dry-run
#   CHECKPOINT_LIST="s3://b/step_100 s3://b/step_200" ./run_checkpoint_batch.sh --dry-run
#   ./run_checkpoint_batch.sh -h
#
# EVERYTHING is overridable via environment variables (see the CONFIG block / -h).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — env var with default. Override by exporting before you run.
# ══════════════════════════════════════════════════════════════════════════════

# --- AWS access (identity-agnostic; scoping doc §3.5.6) ----------------------
# AWS_PROFILE is OPTIONAL: when set it is passed through as one credential source; when
# unset, creds resolve via the standard AWS chain (env vars, SSO, instance/role profile,
# …). Never hard-code a human profile here — the same entrypoint must run laptop-side and
# under a service/CI role unchanged. AWS_REGION is not an identity, so it keeps a default.
AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# --- What to enumerate ---------------------------------------------------------
# RUN_URI: the run's checkpoint ROOT — the S3 prefix that is the DIRECT parent of the
# step* dirs (see the explicit-RUN_URI rule above). Required unless CHECKPOINT_LIST is set.
RUN_URI="${RUN_URI:-}"
# CHECKPOINT_LIST: explicit override — a space/newline-separated list of s3://…/step_N URIs
# that BYPASSES enumeration entirely (re-run a hand-picked subset). Takes precedence.
CHECKPOINT_LIST="${CHECKPOINT_LIST:-}"

# --- Per-BATCH checkpoint format (scoping doc §3.5.5 / §3.6) ------------------
# CHECKPOINT_KIND applies to the WHOLE batch (not per-checkpoint mixed). The per-checkpoint
# auto-detect only WARNS on mismatch; it never switches the kind.
CHECKPOINT_KIND="${CHECKPOINT_KIND:-hf}"   # hf (default, validated) | olmo_core (native)

# --- Eval spec threaded into each atom invocation ----------------------------
TASKS="${TASKS:-arc_easy}"                 # task spec or suite (space-separated for many)
LIMIT="${LIMIT:-10}"                       # `-o limit=N` task override (empty = no limit)

# --- Step selection ------------------------------------------------------------
# STEP_FILTER (optional) restricts the enumerated steps. Two forms:
#   - a numeric window "MIN:MAX" on N (either side optional, e.g. "500:" or ":1000"), or
#   - a regex matched against the "stepN" basename (e.g. "step_?1[0-9]+").
STEP_FILTER="${STEP_FILTER:-}"

# --- Fleet width (validated in P2.a, USED from P2.c) --------------------------
FLEET_SIZE="${FLEET_SIZE:-1}"              # workers; 1 = sequential K=1 case
FLEET_SIZE_MAX=3                           # §4 hard cap: never more than 3 GPU workers

# --- S3 result namespace (matches the atom's §5 grant story) ------------------
S3_BUCKET="${S3_BUCKET:-edullm-adaptive-inference-056956104102}"
S3_PREFIX="${S3_PREFIX:-smoke}"            # smoke/* is the covered interim prefix (§5)
# Per-run group under the result prefix; each checkpoint writes to <S3_GROUP>/step_N/.
S3_GROUP="${S3_GROUP:-checkpoints}"

# --- Instance guard-rails (carried from the atom, §4) ------------------------
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.xlarge}"        # real-run default (1x L4, 24 GB)
ALLOWED_INSTANCE_TYPES="g5.xlarge g6.xlarge"       # hard rule: never above g6.xlarge

# --- Instance launch identity (carried from the atom §3/§4; USED for K=1 in P2.b) ---
# The K=1 path launches ONE real worker with these; the atom's proven values are the
# defaults. FLEET_SIZE>1 (the ≤3 IMDSv2 fleet) remains a P2.c stub.
AMI_ID="${AMI_ID:-ami-0b6f2229ad14c9323}"                   # Ubuntu 24.04 DLAMI (CUDA baked)
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-sg-087218d8c87aa8576}"
SUBNET_ID="${SUBNET_ID:-subnet-0a4235fb98b63930f}"          # us-east-1b (back-compat single value)
# SUBNET_IDS: ordered space/comma-separated subnet list tried on InsufficientInstanceCapacity.
# Defaults to the single SUBNET_ID so existing invocations are unchanged; add more AZs' subnets
# to survive a capacity blip in one AZ (launch retries the next subnet on a capacity error).
SUBNET_IDS="${SUBNET_IDS:-$SUBNET_ID}"
# INSTANCE_TYPE_FALLBACK: when true, if every subnet hits capacity for INSTANCE_TYPE, retry the
# OTHER ALLOWED_INSTANCE_TYPES (never above g6.xlarge) across the same subnets. Off by default.
INSTANCE_TYPE_FALLBACK="${INSTANCE_TYPE_FALLBACK:-false}"
INSTANCE_PROFILE="${INSTANCE_PROFILE:-EswManagedInstance}"  # IAM "badge" -> S3 access
SHUTDOWN_MINUTES="${SHUTDOWN_MINUTES:-180}"                 # user-data self-destruct (sized for a slice)

# --- How the repo reaches the node (staging mode; mirrors the atom) -----------
STAGE_MODE="${STAGE_MODE:-clone}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/allenai/olmo-eval}"  # clone mode source
GIT_REF="${GIT_REF:-}"                                       # optional branch/tag (empty = default)
LOCAL_REPO_PARENT="${LOCAL_REPO_PARENT:-/Users/cat/alpha-projects}"  # tar -C here (tar mode only)
REPO_DIRNAME="${REPO_DIRNAME:-olmo-eval-full}"              # dir to tar/untar (tar mode only)

# --- On-node paths (DLAMI fast NVMe scratch; mirrors the atom) ----------------
# HOME/caches point at the big NVMe (root disk ~19 GB is too small; SSM sets no HOME).
NODE_STAGE_DIR="${NODE_STAGE_DIR:-/opt/dlami/nvme/checkpoint-eval}"
NODE_RESULTS_DIR="${NODE_RESULTS_DIR:-${NODE_STAGE_DIR}/results}"
NODE_HOME="${NODE_HOME:-/root}"
UV_CACHE_DIR_NODE="${UV_CACHE_DIR_NODE:-${NODE_STAGE_DIR}/uv-cache}"
HF_HOME_NODE="${HF_HOME_NODE:-${NODE_STAGE_DIR}/hf}"
# On-node repo dir: a clone lands in the git URL basename; a tar untars into REPO_DIRNAME.
case "$STAGE_MODE" in
  tar) REPO_BASENAME="$REPO_DIRNAME" ;;
  *)   REPO_BASENAME="${GIT_REPO_URL##*/}"; REPO_BASENAME="${REPO_BASENAME%.git}" ;;
esac
NODE_REPO_DIR="${NODE_REPO_DIR:-${NODE_STAGE_DIR}/${REPO_BASENAME}}"

# --- Idempotency / done-markers (scoping doc §3.5.4) --------------------------
# The K=1 loop SKIPS a checkpoint whose result prefix already holds the explicit _SUCCESS
# marker (preferred) or the atom's metrics.json; each finished step WRITES _SUCCESS.
SKIP_IF_DONE="${SKIP_IF_DONE:-true}"
FORCE="${FORCE:-false}"
SUCCESS_MARKER="${SUCCESS_MARKER:-_SUCCESS}"   # explicit per-checkpoint done-marker (§3.5.4)
DONE_MARKER="${DONE_MARKER:-metrics.json}"     # also accepted as "done" (the atom's marker)

# --- Per-checkpoint failure handling -----------------------------------------
# RETRIES transient re-attempts per checkpoint before recording it failed and CONTINUING
# (one bad checkpoint must NOT abort the slice). Total attempts = RETRIES + 1.
RETRIES="${RETRIES:-1}"

# --- Timeouts / behaviour knobs (mirror the atom) ----------------------------
SSM_ONLINE_TIMEOUT="${SSM_ONLINE_TIMEOUT:-300}"   # wait for SSM agent Online (s)
STAGE_TIMEOUT="${STAGE_TIMEOUT:-300}"             # curl/clone+untar on node (s)
INSTALL_TIMEOUT="${INSTALL_TIMEOUT:-1800}"        # uv sync can be slow (s)
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"              # per-checkpoint dl + provider + score (s)
PRESIGN_EXPIRY="${PRESIGN_EXPIRY:-3600}"          # presigned URL lifetime (s, tar mode)
CLEANUP_STAGING="${CLEANUP_STAGING:-true}"        # rm staging tgz from S3 at end (tar mode)

# Completion sentinel appended to each on-node eval log (24 KB-safe poll, like the atom).
DONE_SENTINEL="CHECKPOINT_DONE_EXIT"

# --- Fleet (P2.c) on-node loop knobs -----------------------------------------
# WORKER_DONE_SENTINEL is the WHOLE-SLICE completion marker each fleet worker appends to its
# on-node log after finishing its shard (distinct from the per-checkpoint DONE_SENTINEL). The
# laptop polls each worker's log for it (24 KB-safe grep-first), exactly like the K=1 poll.
WORKER_DONE_SENTINEL="WORKER_DONE_EXIT"
# Fixed on-node path of a fleet worker's slice log (same path on every node; each has its own FS).
NODE_WORKER_LOG="${NODE_WORKER_LOG:-${NODE_STAGE_DIR}/worker.log}"
# WORKER_SHUTDOWN_GRACE: after a worker writes its manifest + sentinel it arms `shutdown -h +GRACE`
# so the laptop's SSM poll reliably reads the sentinel BEFORE the box self-terminates (the laptop
# terminate is still the primary net; the user-data `shutdown -h +SHUTDOWN_MINUTES` is the outer net).
WORKER_SHUTDOWN_GRACE="${WORKER_SHUTDOWN_GRACE:-5}"
# Poll cadence for the whole fleet (laptop-side), mirroring the K=1 per-step poll interval.
FLEET_POLL_INTERVAL="${FLEET_POLL_INTERVAL:-30}"
# Whole-slice wall-clock budget per worker (a worker loops MANY checkpoints, so this is much larger
# than the per-checkpoint EVAL_TIMEOUT). On timeout the laptop tears the fleet down regardless.
FLEET_TIMEOUT="${FLEET_TIMEOUT:-14400}"

# S3 key/uri for the staged code tarball (tar mode only).
STAGING_KEY="${S3_PREFIX}/staging/olmo-eval.tgz"
STAGING_URI="s3://${S3_BUCKET}/${STAGING_KEY}"

# Step-directory matcher (scoping doc §3.5.2): real checkpoints are named WITHOUT an
# underscore (step7629), but the underscore form (step_1000) is also accepted.
STEP_REGEX='^step_?[0-9]+$'

# Runtime state.
DRY_RUN=false
INSTANCE_ID=""          # K=1 single-worker id (P2.b)
FLEET_IDS=()            # P2.c fleet: ALL launched worker ids (space-tracked, exact ids only)
TARBALL=""
TEARDOWN_DONE=false

# ══════════════════════════════════════════════════════════════════════════════
# Small helpers (mirror run_checkpoint_eval.sh conventions)
# ══════════════════════════════════════════════════════════════════════════════

log()  { printf '\n=== %s\n' "$*" >&2; }
info() { printf '    %s\n' "$*" >&2; }
warn() { printf '    WARN: %s\n' "$*" >&2; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# Pretty-join an argv for display (quote args containing spaces).
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

# aws_common_args: region always; profile only when explicitly set (identity-agnostic).
AWS_COMMON_ARGS=(--region "$AWS_REGION")
[ -n "$AWS_PROFILE" ] && AWS_COMMON_ARGS+=(--profile "$AWS_PROFILE")

# aws_ro: run a READ-ONLY aws command with the common args. Unlike a mutating `run`, this
# always executes (even under --dry-run) because enumeration must query S3 to be useful.
aws_ro() {
  aws "$@" "${AWS_COMMON_ARGS[@]}"
}

# Split "s3://bucket/key/prefix" into BUCKET and KEY (KEY has no leading/trailing slash).
# Mirrors diagnostics/mcq_cat/common/s3_io.py: parse_s3_uri.
S3_BUCKET_OUT=""
S3_KEY_OUT=""
parse_s3_uri() {
  local uri=$1 rest
  case "$uri" in
    s3://*) : ;;
    *) die "Not an S3 URI: ${uri}" ;;
  esac
  rest=${uri#s3://}
  S3_BUCKET_OUT=${rest%%/*}
  if [ "$rest" = "$S3_BUCKET_OUT" ]; then
    S3_KEY_OUT=""
  else
    S3_KEY_OUT=${rest#*/}
  fi
  S3_KEY_OUT=${S3_KEY_OUT#/}
  S3_KEY_OUT=${S3_KEY_OUT%/}
}

# Parse the integer N out of a "stepN" / "step_N" basename.
step_num() {
  local name=$1 n
  n=${name#step}
  n=${n#_}
  printf '%s' "$n"
}

# ══════════════════════════════════════════════════════════════════════════════
# Mutating-command + SSM helpers (used by the K=1 dispatch; mirror the atom)
# ══════════════════════════════════════════════════════════════════════════════
# These are only exercised on the FLEET_SIZE=1 path. They honor --dry-run: run() and
# ssm_invoke() PRINT what they would send and execute nothing when DRY_RUN=true. Unlike
# aws_ro (read-only enumeration, always executes), these gate all mutation on DRY_RUN.

# run(): echo the argv (always) then execute it — UNLESS --dry-run, in which case it only
# echoes. stdout passes through so callers can capture with $( ... ); the trace goes to stderr.
run() {
  printf '    + %s\n' "$(quote_cmd "$@")" >&2
  [ "$DRY_RUN" = true ] && return 0
  "$@"
}

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
# Send ONE shell payload to the node via SSM, wait for a terminal state, print
# StandardOutputContent to stdout, and return non-zero on any non-Success/timeout.
# Identity-agnostic: uses AWS_COMMON_ARGS (region always; profile only when set).
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
    "${AWS_COMMON_ARGS[@]}") \
    || { echo "ERROR: ssm send-command failed (${desc})" >&2; return 1; }

  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would poll: aws ssm get-command-invocation --command-id <id> --instance-id ${INSTANCE_ID}"
    return 0
  fi

  while :; do
    status=$(aws ssm get-command-invocation \
      --command-id "$cid" --instance-id "$INSTANCE_ID" \
      --query 'Status' --output text \
      "${AWS_COMMON_ARGS[@]}" 2>/dev/null || echo "Pending")
    case "$status" in
      Success) break ;;
      Failed|Cancelled|TimedOut|Undeliverable|Terminated)
        echo "ERROR: SSM '${desc}' -> ${status}. StandardErrorContent:" >&2
        aws ssm get-command-invocation \
          --command-id "$cid" --instance-id "$INSTANCE_ID" \
          --query 'StandardErrorContent' --output text \
          "${AWS_COMMON_ARGS[@]}" 2>/dev/null >&2 || true
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
    "${AWS_COMMON_ARGS[@]}" 2>/dev/null
}

# ══════════════════════════════════════════════════════════════════════════════
# Teardown — guaranteed to run on ANY exit so we never leak a billing instance.
# ══════════════════════════════════════════════════════════════════════════════

# Explicit terminate + confirm (belt-and-suspenders end-of-run teardown).
terminate_and_confirm() {
  [ -z "${INSTANCE_ID}" ] && return 0
  log "Teardown: terminating ${INSTANCE_ID}"
  run aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
    "${AWS_COMMON_ARGS[@]}" >/dev/null || true
  if [ "$DRY_RUN" = true ]; then return 0; fi
  aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID" \
    "${AWS_COMMON_ARGS[@]}" 2>/dev/null || true
  local state
  state=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[].Instances[].State.Name' --output text \
    "${AWS_COMMON_ARGS[@]}" 2>/dev/null || echo unknown)
  info "Instance ${INSTANCE_ID} state: ${state}"
}

# trap handler: fires on EXIT/INT/TERM/ERR. Best-effort terminate of any launched instance,
# then remove the local tarball. No-op under --dry-run or if nothing was launched. Detaches
# itself first so it runs exactly once.
cleanup() {
  local rc=$?
  trap - EXIT INT TERM ERR

  if [ -n "${TARBALL}" ] && [ -f "${TARBALL}" ]; then
    rm -f "$TARBALL" 2>/dev/null || true
  fi

  if [ "$DRY_RUN" = true ]; then exit "$rc"; fi

  # Best-effort terminate of EVERY exact id we launched — the K=1 single id AND the fleet ids.
  # We only ever act on ids WE captured from our own run-instances (never a shared-tag query).
  if [ "$TEARDOWN_DONE" != true ]; then
    local leaked=()
    [ -n "${INSTANCE_ID}" ] && leaked+=("$INSTANCE_ID")
    [ "${#FLEET_IDS[@]}" -gt 0 ] && leaked+=("${FLEET_IDS[@]}")
    if [ "${#leaked[@]}" -gt 0 ]; then
      echo "" >&2
      echo "!!! Exiting (rc=${rc}). Terminating ${leaked[*]} to avoid leaked billing." >&2
      aws ec2 terminate-instances --instance-ids "${leaked[@]}" \
        "${AWS_COMMON_ARGS[@]}" >/dev/null 2>&1 || true
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM ERR

# ══════════════════════════════════════════════════════════════════════════════
# Usage
# ══════════════════════════════════════════════════════════════════════════════
usage() {
  cat <<EOF
run_checkpoint_batch.sh — retroactive batch DRIVER (scoping doc §3.5). Enumeration +
identity-agnostic entrypoint (P2.a) + the K=1 install-ONCE / loop-many path (P2.b) + the
≤3-worker node-side IMDSv2 shard fleet (P2.c). FLEET_SIZE=1 launches ONE box and loops the
checkpoints (laptop-driven); FLEET_SIZE 2..3 launches a fleet whose workers each self-assign a
static slice and loop it (node-driven). --dry-run launches NOTHING and prints the whole plan.

USAGE:
  RUN_URI=s3://bucket/<run>/checkpoints/ ./run_checkpoint_batch.sh [--dry-run] [--profile NAME] [--region REGION]
  CHECKPOINT_LIST="s3://b/step_100 s3://b/step_200" ./run_checkpoint_batch.sh --dry-run
  ./run_checkpoint_batch.sh -h | --help

FLAGS:
  --dry-run          Print the resolved config + ordered enumeration + planned dispatch; launch nothing.
  --profile NAME     Optional AWS CLI profile (one credential source; default: ambient resolution).
  --region REGION    AWS region (default: \$AWS_REGION or 'us-east-1').
  -h, --help         Show this help.

IDENTITY (scoping doc §3.5.6):
  Credentials resolve AMBIENTLY via the standard AWS chain. AWS_PROFILE is OPTIONAL — set it
  laptop-side if you use a named profile; leave it unset under a service/CI/instance role. The
  same entrypoint runs in both cases with no code change; no human profile is hard-coded.

KEY ENV VARS (current effective defaults shown):
  RUN_URI=${RUN_URI:-<required unless CHECKPOINT_LIST set>}
  CHECKPOINT_LIST=${CHECKPOINT_LIST:-<none>}
  CHECKPOINT_KIND=${CHECKPOINT_KIND}   # hf (default) | olmo_core — per-BATCH, auto-detect warns on mismatch
  TASKS=${TASKS}    LIMIT=${LIMIT:-<none>}
  STEP_FILTER=${STEP_FILTER:-<none>}   # "MIN:MAX" numeric window OR a regex on the stepN name
  FLEET_SIZE=${FLEET_SIZE}   (hard cap ${FLEET_SIZE_MAX}; 1 = K=1 laptop loop, 2..3 = P2.c node-side fleet)
  S3_BUCKET=${S3_BUCKET}
  S3_PREFIX=${S3_PREFIX}    S3_GROUP=${S3_GROUP}
  INSTANCE_TYPE=${INSTANCE_TYPE}   (allowed: ${ALLOWED_INSTANCE_TYPES})
  AWS_PROFILE=${AWS_PROFILE:-<ambient>}   AWS_REGION=${AWS_REGION}

K=1 (FLEET_SIZE=1) LAUNCH/LOOP KNOBS (mirror the atom; only used on the real K=1 path):
  STAGE_MODE=${STAGE_MODE}   # clone (default) | tar     GIT_REPO_URL=${GIT_REPO_URL}
  AMI_ID=${AMI_ID}   SHUTDOWN_MINUTES=${SHUTDOWN_MINUTES}
  SKIP_IF_DONE=${SKIP_IF_DONE}   FORCE=${FORCE}   SUCCESS_MARKER=${SUCCESS_MARKER}   DONE_MARKER=${DONE_MARKER}
  RETRIES=${RETRIES}   EVAL_TIMEOUT=${EVAL_TIMEOUT}   INSTALL_TIMEOUT=${INSTALL_TIMEOUT}
  Per-checkpoint results land at s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/step_N/ ;
  worker manifest + summary at .../${S3_GROUP}/_batch/{worker_0,summary}.json

FLEET (FLEET_SIZE 2..3, P2.c) node-side shard knobs (shared to-do list + on-node slice-loop):
  K_eff = min(FLEET_SIZE, N_TODO, ${FLEET_SIZE_MAX}); ONE run-instances --count K_eff.
  Shared to-do list -> .../${S3_GROUP}/_batch/todo.json ; per-worker shards -> _batch/worker_<i>.json.
  Each worker reads ami-launch-index via IMDSv2 and loops positions i, i+K_eff, … of the to-do list.
  WORKER_SHUTDOWN_GRACE=${WORKER_SHUTDOWN_GRACE}   FLEET_POLL_INTERVAL=${FLEET_POLL_INTERVAL}   FLEET_TIMEOUT=${FLEET_TIMEOUT}
  Laptop polls each EXACT id's '${WORKER_DONE_SENTINEL}' sentinel, then terminates ALL of OUR ids
  (operator creds -> the Purpose=checkpoint-eval self-terminate tag mismatch does NOT block teardown).

PRE-LAUNCH GATE (Bug #1): a read-only S3 done-marker check runs for every checkpoint BEFORE
  launching. If all are done the driver records them skipped, writes the manifest and returns
  with ZERO GPU; otherwise it launches and loops ONLY the to-do subset. --dry-run shows the split.

CAPACITY FALLBACK (Bug #2): run-instances retries across subnets on InsufficientInstanceCapacity.
  SUBNET_IDS=${SUBNET_IDS}   # ordered space/comma list; default = single SUBNET_ID (back-compat)
  INSTANCE_TYPE_FALLBACK=${INSTANCE_TYPE_FALLBACK}   # true = also retry other ALLOWED types (<= g6.xlarge)

EXPLICIT-RUN_URI RULE: RUN_URI must point DIRECTLY at the parent of the step* dirs; the driver
does not auto-descend. Enumeration keeps immediate children matching ${STEP_REGEX}, sorted
numerically by N.
EOF
}

# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing (mirrors the atom's flags)
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

# Rebuild the common args after flag parsing (they may override the env-derived values).
AWS_COMMON_ARGS=(--region "$AWS_REGION")
[ -n "$AWS_PROFILE" ] && AWS_COMMON_ARGS+=(--profile "$AWS_PROFILE")

# ══════════════════════════════════════════════════════════════════════════════
# Input validation
# ══════════════════════════════════════════════════════════════════════════════
command -v aws >/dev/null 2>&1 || die "aws CLI not found. Install AWS CLI v2 first."

case "$CHECKPOINT_KIND" in
  hf|olmo_core) : ;;
  *) die "CHECKPOINT_KIND='${CHECKPOINT_KIND}' invalid. Use 'hf' (default) or 'olmo_core'." ;;
esac

# FLEET_SIZE: positive integer, within the §4 hard cap. Inert in P2.a (not used to launch).
case "$FLEET_SIZE" in
  ''|*[!0-9]*) die "FLEET_SIZE='${FLEET_SIZE}' must be a positive integer." ;;
esac
[ "$FLEET_SIZE" -ge 1 ] || die "FLEET_SIZE must be >= 1."
[ "$FLEET_SIZE" -le "$FLEET_SIZE_MAX" ] \
  || die "FLEET_SIZE=${FLEET_SIZE} exceeds the hard cap of ${FLEET_SIZE_MAX} GPU workers (§4). Never widen the fleet — restructure the sweep."

# RETRIES: non-negative integer (per-checkpoint transient re-attempts on the K=1 path).
case "$RETRIES" in
  ''|*[!0-9]*) die "RETRIES='${RETRIES}' must be a non-negative integer." ;;
esac

# STAGE_MODE must be one of the two supported flows (only used on the K=1 launch path).
case "$STAGE_MODE" in
  clone|tar) : ;;
  *) die "STAGE_MODE='${STAGE_MODE}' invalid. Use 'clone' (default) or 'tar'." ;;
esac

# INSTANCE_TYPE allowlist (inert guard carried from the atom; never above g6.xlarge).
type_ok=false
for t in $ALLOWED_INSTANCE_TYPES; do
  [ "$INSTANCE_TYPE" = "$t" ] && type_ok=true
done
[ "$type_ok" = true ] || die "INSTANCE_TYPE='${INSTANCE_TYPE}' not allowed. Use one of: ${ALLOWED_INSTANCE_TYPES}."

# Exactly one enumeration source.
if [ -z "$RUN_URI" ] && [ -z "$CHECKPOINT_LIST" ]; then
  die "Provide RUN_URI (the step* parent prefix) or CHECKPOINT_LIST (explicit URIs). See -h."
fi

# ══════════════════════════════════════════════════════════════════════════════
# Banner
# ══════════════════════════════════════════════════════════════════════════════
if [ "$FLEET_SIZE" -eq 1 ]; then
  log "Checkpoint-eval BATCH driver — P2.b K=1 (install ONCE, loop checkpoints on ONE box)"
else
  log "Checkpoint-eval BATCH driver — P2.c fleet (FLEET_SIZE=${FLEET_SIZE}; node-side IMDSv2 static shard, install ONCE/worker)"
fi
info "identity       : ${AWS_PROFILE:+profile=${AWS_PROFILE} }region=${AWS_REGION} (creds resolved ambiently)"
if [ -n "$CHECKPOINT_LIST" ]; then
  info "source         : CHECKPOINT_LIST (explicit override; enumeration bypassed)"
else
  info "source         : RUN_URI=${RUN_URI} (immediate step* children; no auto-descend)"
fi
info "batch kind     : ${CHECKPOINT_KIND} (per-batch; auto-detect warns on mismatch)"
info "eval           : -t ${TASKS}${LIMIT:+ -o limit=${LIMIT}}"
info "step filter    : ${STEP_FILTER:-<none>}"
info "fleet size     : ${FLEET_SIZE} (hard cap ${FLEET_SIZE_MAX}; 1=K=1 laptop loop, 2..3=P2.c node-side fleet)"
info "instance type  : ${INSTANCE_TYPE} (hard cap g6.xlarge)"
info "idempotency    : skip-if-done=${SKIP_IF_DONE} force=${FORCE} markers=${SUCCESS_MARKER}/${DONE_MARKER} retries=${RETRIES}"
info "stage/install  : mode=${STAGE_MODE}; install ONCE then loop (ami=${AMI_ID})"
info "capacity       : subnets=${SUBNET_IDS// /,} type-fallback=${INSTANCE_TYPE_FALLBACK} (retry on InsufficientInstanceCapacity)"
if [ "$FLEET_SIZE" -ne 1 ]; then
  info "fleet shard    : K_eff=min(FLEET_SIZE,N_TODO,${FLEET_SIZE_MAX}); todo.json + on-node IMDSv2 slice-loop; poll each exact id"
fi
info "results dest   : s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/step_N/  (manifest under _batch/)"
[ "$DRY_RUN" = true ] && info ">>> DRY-RUN: no instances will be launched <<<"

# ══════════════════════════════════════════════════════════════════════════════
# Format detect (scoping doc §3.5.2 step 5) — list-only, WARN on mismatch only
# ══════════════════════════════════════════════════════════════════════════════
# Classifies one checkpoint by listing its IMMEDIATE children:
#   model_and_optim/ + .metadata layout            -> olmo_core
#   config.json + *.safetensors and NO model_and_optim -> hf
#   otherwise                                        -> unknown
# This never rewrites the batch kind; it only lets the caller warn on disagreement.
detect_kind() {
  local uri=$1 listing
  # Normalize to a trailing slash so `s3 ls` lists the prefix's children.
  case "$uri" in */) : ;; *) uri="${uri}/" ;; esac
  listing=$(aws_ro s3 ls "$uri" 2>/dev/null || true)
  [ -z "$listing" ] && { printf 'unknown'; return 0; }

  local has_mao=false has_safetensors=false
  printf '%s\n' "$listing" | grep -qE '[[:space:]]PRE[[:space:]]+model_and_optim/' && has_mao=true
  printf '%s\n' "$listing" | grep -qE '\.safetensors$' && has_safetensors=true

  if [ "$has_mao" = true ]; then
    printf 'olmo_core'
  elif [ "$has_safetensors" = true ]; then
    printf 'hf'
  else
    printf 'unknown'
  fi
}

# ══════════════════════════════════════════════════════════════════════════════
# Enumeration (scoping doc §3.5.2) — produces the ordered (step, uri, kind) list
# ══════════════════════════════════════════════════════════════════════════════
# Collected, in numeric N order, into three parallel arrays.
ENUM_STEPS=()
ENUM_URIS=()
ENUM_KINDS=()
MISMATCH_COUNT=0

# apply_step_filter <step_name> <N> -> return 0 to keep, 1 to drop.
apply_step_filter() {
  local name=$1 n=$2
  [ -z "$STEP_FILTER" ] && return 0
  # Numeric window "MIN:MAX" (either side optional).
  if printf '%s' "$STEP_FILTER" | grep -qE '^[0-9]*:[0-9]*$'; then
    local min max
    min=${STEP_FILTER%%:*}
    max=${STEP_FILTER##*:}
    [ -n "$min" ] && [ "$n" -lt "$min" ] && return 1
    [ -n "$max" ] && [ "$n" -gt "$max" ] && return 1
    return 0
  fi
  # Otherwise treat STEP_FILTER as a regex on the stepN basename.
  printf '%s' "$name" | grep -qE "$STEP_FILTER"
}

# collect_one <step_name> <step_uri>: filter, detect kind, warn on mismatch, append.
collect_one() {
  local name=$1 uri=$2 n kind
  n=$(step_num "$name")
  apply_step_filter "$name" "$n" || { info "filtered out: ${name} (N=${n})"; return 0; }
  kind=$(detect_kind "$uri")
  if [ "$kind" = unknown ]; then
    warn "${name}: could not classify layout (list-only); leaving as batch kind '${CHECKPOINT_KIND}'"
  elif [ "$kind" != "$CHECKPOINT_KIND" ]; then
    warn "${name}: detected on-disk layout '${kind}' disagrees with CHECKPOINT_KIND='${CHECKPOINT_KIND}' (dispatching as '${CHECKPOINT_KIND}')"
    MISMATCH_COUNT=$(( MISMATCH_COUNT + 1 ))
  fi
  ENUM_STEPS+=("$n")
  ENUM_URIS+=("$uri")
  ENUM_KINDS+=("$CHECKPOINT_KIND")
}

enumerate_from_run_uri() {
  local bucket prefix base_uri listing line name n raw sorted
  parse_s3_uri "$RUN_URI"
  bucket=$S3_BUCKET_OUT
  prefix=$S3_KEY_OUT
  if [ -n "$prefix" ]; then
    base_uri="s3://${bucket}/${prefix}/"
  else
    base_uri="s3://${bucket}/"
  fi

  log "Enumerate immediate children of ${base_uri}"
  # `aws s3 ls <prefix>/` uses delimiter '/' implicitly, listing child prefixes as "PRE name/".
  listing=$(aws_ro s3 ls "$base_uri" 2>/dev/null || true)
  [ -z "$listing" ] && die "No listing returned for ${base_uri} (empty prefix or no read access)."

  # Gather matching step dirs as "N<TAB>uri" lines, then numeric-sort by N.
  raw=""
  while IFS= read -r line; do
    case "$line" in
      *" PRE "*) : ;;   # only common-prefix (directory) lines
      *) continue ;;
    esac
    name=${line##*PRE }
    name=${name%/}
    [[ "$name" =~ $STEP_REGEX ]] || continue
    n=$(step_num "$name")
    raw="${raw}${n}"$'\t'"${name}"$'\n'
  done <<EOF
$listing
EOF

  [ -z "$raw" ] && die "No step directories matching ${STEP_REGEX} directly under ${base_uri}. Point RUN_URI at the step* parent (no auto-descend)."

  # Numeric sort by N (col 1). This is the crux: lexical sort would misorder step1000<step125.
  sorted=$(printf '%s' "$raw" | sort -t$'\t' -k1,1n)

  while IFS=$'\t' read -r n name; do
    [ -z "$name" ] && continue
    collect_one "$name" "${base_uri}${name}"
  done <<EOF
$sorted
EOF
}

enumerate_from_list() {
  local uri name raw sorted n
  log "Enumerate from explicit CHECKPOINT_LIST (enumeration bypassed)"
  # Normalize whitespace/newlines into one URI per line, derive N, numeric-sort.
  raw=""
  for uri in $CHECKPOINT_LIST; do
    uri=${uri%/}
    name=${uri##*/}
    if [[ "$name" =~ $STEP_REGEX ]]; then
      n=$(step_num "$name")
    else
      warn "CHECKPOINT_LIST entry '${uri}' basename '${name}' does not match ${STEP_REGEX}; keeping with N=0 sort key"
      n=0
    fi
    raw="${raw}${n}"$'\t'"${uri}"$'\n'
  done
  [ -z "$raw" ] && die "CHECKPOINT_LIST is set but parsed to no entries."
  sorted=$(printf '%s' "$raw" | sort -t$'\t' -k1,1n)
  while IFS=$'\t' read -r n uri; do
    [ -z "$uri" ] && continue
    collect_one "${uri##*/}" "$uri"
  done <<EOF
$sorted
EOF
}

if [ -n "$CHECKPOINT_LIST" ]; then
  enumerate_from_list
else
  enumerate_from_run_uri
fi

COUNT=${#ENUM_URIS[@]}
[ "$COUNT" -gt 0 ] || die "Enumeration produced 0 checkpoints (after STEP_FILTER). Nothing to do."

# ══════════════════════════════════════════════════════════════════════════════
# Report the ordered (step, uri, kind) enumeration
# ══════════════════════════════════════════════════════════════════════════════
log "Ordered checkpoint enumeration (${COUNT} step(s), numeric order)"
idx=0
while [ "$idx" -lt "$COUNT" ]; do
  printf '    [%2d] step %-8s kind=%-9s %s\n' \
    "$idx" "${ENUM_STEPS[$idx]}" "${ENUM_KINDS[$idx]}" "${ENUM_URIS[$idx]}" >&2
  idx=$(( idx + 1 ))
done
[ "$MISMATCH_COUNT" -gt 0 ] && warn "${MISMATCH_COUNT}/${COUNT} checkpoint(s) had a layout that disagrees with CHECKPOINT_KIND='${CHECKPOINT_KIND}' (see warnings above)."

# ══════════════════════════════════════════════════════════════════════════════
# Dispatch — K=1 single-worker orchestration (P2.b) + P2.c fleet stub
# ══════════════════════════════════════════════════════════════════════════════
# P2.b implements the FLEET_SIZE=1 path: launch ONE box, stage + install deps ONCE,
# then loop the enumerated checkpoints on that SAME box. All on-node command strings
# MIRROR the validated atom (run_checkpoint_eval.sh) so the atom stays untouched.
# Everything below honors --dry-run: it PRINTS the whole planned sequence and launches
# nothing. FLEET_SIZE>1 (the ≤3 IMDSv2 fleet) stays a P2.c stub further down.

# Per-step result group + prefix (canonical step_<N> under our results namespace).
step_group()  { printf '%s/step_%s' "$S3_GROUP" "$1"; }
step_prefix() { printf 's3://%s/%s/%s/step_%s/' "$S3_BUCKET" "$S3_PREFIX" "$S3_GROUP" "$1"; }

# Laptop-side manifest accumulators (one worker for K=1).
MANIFEST_STEPS=()
N_DONE=0
N_SKIP=0
N_FAIL=0

# Pre-launch gate state (Bug #1): DONE_FLAG[idx]=true means the checkpoint's done-marker is
# already in S3, so it is recorded 'skipped' and never dispatched. N_TODO/N_PRESKIP are the
# to-do vs already-done split the gate uses to decide whether to launch at all.
DONE_FLAG=()
N_TODO=0
N_PRESKIP=0

# checkpoint_marker_present <N> -> return 0 if this step's _SUCCESS/metrics.json is already in
# S3. Read-only (aws_ro), so it runs even under --dry-run — the gate must query S3 to be useful.
checkpoint_marker_present() {
  local n=$1 prefix listing
  prefix=$(step_prefix "$n")
  listing=$(aws_ro s3 ls "$prefix" --recursive 2>/dev/null || true)
  printf '%s' "$listing" | grep -q -e "$SUCCESS_MARKER" -e "$DONE_MARKER"
}

# compute_todo_set: the pre-launch gate. For each enumerated checkpoint, decide to-do vs
# already-done via a read-only S3 marker check (honoring SKIP_IF_DONE/FORCE) and populate
# DONE_FLAG + the N_TODO/N_PRESKIP counts. Always runs, including under --dry-run.
compute_todo_set() {
  log "Pre-launch gate: read-only S3 done-marker check for ${COUNT} checkpoint(s)"
  local skip_active=true
  if [ "$SKIP_IF_DONE" != true ] || [ "$FORCE" = true ]; then
    skip_active=false
    info "skip-if-done inactive (SKIP_IF_DONE=${SKIP_IF_DONE}, FORCE=${FORCE}); every checkpoint is to-do"
  fi
  local idx=0 n
  while [ "$idx" -lt "$COUNT" ]; do
    n=${ENUM_STEPS[$idx]}
    if [ "$skip_active" = true ] && checkpoint_marker_present "$n"; then
      DONE_FLAG[$idx]=true
      N_PRESKIP=$(( N_PRESKIP + 1 ))
      info "  [done] step_${n}  (${SUCCESS_MARKER}/${DONE_MARKER} present)"
    else
      DONE_FLAG[$idx]=false
      N_TODO=$(( N_TODO + 1 ))
      info "  [todo] step_${n}"
    fi
    idx=$(( idx + 1 ))
  done
  info "gate result: ${N_TODO} to-do, ${N_PRESKIP} already-done (of ${COUNT})"
}

# record_skipped <idx>: record a pre-satisfied checkpoint as 'skipped' in the manifest so the
# summary counts stay correct even when we never launch / never dispatch it.
record_skipped() {
  local idx=$1
  local n=${ENUM_STEPS[$idx]} uri=${ENUM_URIS[$idx]} kind=${ENUM_KINDS[$idx]}
  info "SKIP (gate) step_${n}: done-marker already present — not dispatched."
  record_step "$n" "$uri" "$kind" skipped 0 0 0
  N_SKIP=$(( N_SKIP + 1 ))
}

# --- Pre-flight: never exceed the ≤3-worker cap (§4) --------------------------
# preflight_worker_cap <n_launch>: assert that launching <n_launch> more workers keeps the TOTAL
# running/pending edullm-gpu-worker count within the ${FLEET_SIZE_MAX} hard cap — applied across
# the WHOLE fleet (K=1 passes 1, the fleet passes K_eff).
preflight_worker_cap() {
  local n_launch=${1:-1}
  log "Pre-flight: ≤${FLEET_SIZE_MAX}-worker cap (launching ${n_launch})"
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would count running/pending Name=edullm-gpu-worker and require running+${n_launch} <= ${FLEET_SIZE_MAX}"
    return 0
  fi
  local running
  running=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=edullm-gpu-worker" \
              "Name=instance-state-name,Values=running,pending" \
    --query 'length(Reservations[].Instances[])' --output text \
    "${AWS_COMMON_ARGS[@]}" 2>/dev/null || echo "ERR")
  [ "$running" = "ERR" ] && die "Could not query running GPU instances."
  info "running/pending edullm-gpu-worker instances: ${running}"
  [ "$(( running + n_launch ))" -gt "$FLEET_SIZE_MAX" ] \
    && die "Already ${running} GPU workers up; launching ${n_launch} would exceed the ${FLEET_SIZE_MAX}-instance cap (§4)."
  return 0
}

# run_instances_once <type> <subnet> <count> <user_data>: single run-instances attempt for a
# whole reservation of <count> workers (count=1 is the K=1 case). Captures stdout+stderr AND the
# command's own exit status explicitly (no tee/pipe in the path) so a genuine launch failure is
# NEVER masked. Prints ALL launched ids (whitespace-separated) to stdout; returns run-instances' rc.
# With --count K, run-instances is all-or-nothing (min=max=K) so a partial fleet never happens —
# InsufficientInstanceCapacity for the whole reservation surfaces to the capacity fallback.
run_instances_once() {
  local itype=$1 subnet=$2 count=$3 user_data=$4
  aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$itype" \
    --count "$count" \
    --iam-instance-profile "Name=${INSTANCE_PROFILE}" \
    --security-group-ids "$SECURITY_GROUP_ID" \
    --subnet-id "$subnet" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$user_data" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=edullm-gpu-worker},{Key=Purpose,Value=checkpoint-eval}]" \
    --query 'Instances[].InstanceId' --output text \
    "${AWS_COMMON_ARGS[@]}" 2>&1
}

# --- Launch ONE self-terminating worker (atom §4 params) ---------------------
# Capacity resilience (Bug #2): iterate the SUBNET_IDS list (and, if INSTANCE_TYPE_FALLBACK,
# the other ALLOWED_INSTANCE_TYPES) and retry run-instances on InsufficientInstanceCapacity.
# Any non-capacity error aborts immediately; exhausting all combos aborts non-zero.
launch_instance() {
  log "Launch ONE worker (${INSTANCE_TYPE}; self-terminate at shutdown -h +${SHUTDOWN_MINUTES})"
  local user_data
  user_data=$(printf '#!/bin/bash\n# Checkpoint-eval batch auto-terminate safety net.\nshutdown -h +%s\n' "$SHUTDOWN_MINUTES")

  # Instance types to try, in order: primary first, then the other ALLOWED types if enabled.
  local types="$INSTANCE_TYPE" t
  if [ "$INSTANCE_TYPE_FALLBACK" = true ]; then
    for t in $ALLOWED_INSTANCE_TYPES; do
      [ "$t" = "$INSTANCE_TYPE" ] || types="${types} ${t}"
    done
  fi
  # Subnets to try, in order (normalize commas -> spaces).
  local subnets="${SUBNET_IDS//,/ }"

  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would try run-instances in order over types='${types}' subnets='${subnets}' (retry next on capacity errors)"
    local first_type="${types%% *}" first_subnet="${subnets%% *}" trace
    # Assign the joined trace to a var first: printing "$(quote_cmd ...)" directly would
    # brace-expand the tag-specifications literal into multiple lines.
    trace=$(quote_cmd aws ec2 run-instances \
      --image-id "$AMI_ID" --instance-type "$first_type" --count 1 \
      --iam-instance-profile "Name=${INSTANCE_PROFILE}" \
      --security-group-ids "$SECURITY_GROUP_ID" --subnet-id "$first_subnet" \
      --instance-initiated-shutdown-behavior terminate --user-data "$user_data" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=edullm-gpu-worker},{Key=Purpose,Value=checkpoint-eval}]" \
      --query 'Instances[0].InstanceId' --output text "${AWS_COMMON_ARGS[@]}")
    printf '    + %s\n' "$trace" >&2
    INSTANCE_ID="i-DRYRUNXXXXXXXXXX"
    info "(dry-run) pretend InstanceId=${INSTANCE_ID} (type=${first_type} subnet=${first_subnet})"
    return 0
  fi

  local itype subnet out rc
  for itype in $types; do
    for subnet in $subnets; do
      info "try run-instances: type=${itype} subnet=${subnet}"
      rc=0
      out=$(run_instances_once "$itype" "$subnet" 1 "$user_data") || rc=$?
      if [ "$rc" -eq 0 ] && [ -n "$out" ] && [ "$out" != "None" ]; then
        INSTANCE_ID="$out"
        info "launched InstanceId=${INSTANCE_ID} (type=${itype} subnet=${subnet})"
        return 0
      fi
      if printf '%s' "$out" | grep -qE 'InsufficientInstanceCapacity|Insufficient capacity'; then
        warn "capacity unavailable for ${itype} in ${subnet}; trying next combo."
        continue
      fi
      die "run-instances failed (type=${itype} subnet=${subnet}, rc=${rc}): ${out}"
    done
  done
  die "run-instances exhausted all subnets/types without capacity (types='${types}' subnets='${subnets}')."
}

# --- Wait for 'running' then SSM Online (atom §4b) ---------------------------
wait_running_and_ssm() {
  log "Wait for 'running' + SSM Online"
  run aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" "${AWS_COMMON_ARGS[@]}"
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would poll: aws ssm describe-instance-information ... PingStatus == Online"
    return 0
  fi
  local waited=0 ping
  while :; do
    ping=$(aws ssm describe-instance-information \
      --filters "Key=InstanceIds,Values=${INSTANCE_ID}" \
      --query 'InstanceInformationList[0].PingStatus' --output text \
      "${AWS_COMMON_ARGS[@]}" 2>/dev/null || echo "None")
    [ "$ping" = "Online" ] && { info "SSM PingStatus=Online"; break; }
    [ "$waited" -ge "$SSM_ONLINE_TIMEOUT" ] \
      && die "SSM did not reach Online within ${SSM_ONLINE_TIMEOUT}s (PingStatus=${ping})."
    sleep 10
    waited=$(( waited + 10 ))
  done
}

# --- Stage the repo ONCE (atom §5) -------------------------------------------
stage_repo() {
  log "Stage repo onto node (mode=${STAGE_MODE}) — ONCE"
  if [ "$STAGE_MODE" = clone ]; then
    local branch_opt=""
    [ -n "$GIT_REF" ] && branch_opt="--branch ${GIT_REF} "
    local cmd="mkdir -p ${NODE_STAGE_DIR} && cd ${NODE_STAGE_DIR} && git clone --depth 1 ${branch_opt}${GIT_REPO_URL} && ls -d ${NODE_REPO_DIR}"
    ssm_invoke "$cmd" "stage: git clone repo" "$STAGE_TIMEOUT" >/dev/null \
      || die "Staging (git clone) failed on the node."
  else
    TARBALL="${TMPDIR:-/tmp}/olmo-eval-batch-$$.tgz"
    [ -d "${LOCAL_REPO_PARENT}/${REPO_DIRNAME}" ] \
      || die "Local repo not found at ${LOCAL_REPO_PARENT}/${REPO_DIRNAME}."
    run tar czf "$TARBALL" -C "$LOCAL_REPO_PARENT" "$REPO_DIRNAME"
    run aws s3 cp "$TARBALL" "$STAGING_URI" "${AWS_COMMON_ARGS[@]}"
    local url
    url=$(run aws s3 presign "$STAGING_URI" --expires-in "$PRESIGN_EXPIRY" "${AWS_COMMON_ARGS[@]}")
    [ "$DRY_RUN" = true ] && url="https://DRY-RUN-PRESIGNED-URL"
    [ -n "$url" ] || die "aws s3 presign returned an empty URL."
    local cmd="mkdir -p ${NODE_STAGE_DIR} && cd ${NODE_STAGE_DIR} && curl -sSL -o olmo-eval.tgz \"${url}\" && tar xzf olmo-eval.tgz && ls -d ${NODE_REPO_DIR}"
    ssm_invoke "$cmd" "stage: curl + untar repo" "$STAGE_TIMEOUT" >/dev/null \
      || die "Staging (curl+untar) failed on the node."
  fi
  info "repo staged at ${NODE_REPO_DIR}"
}

# --- Install deps ONCE (atom §5b) — THE amortized ~10-min cost ---------------
install_deps_once() {
  local extra=""
  if [ "$CHECKPOINT_KIND" = olmo_core ]; then
    extra=" --extra olmo_core"
    info "olmo_core kind: install adds the 'olmo_core' extra (ai2-olmo-core)"
  fi
  log "Install deps with uv${extra} — ONCE (amortized across ${COUNT} checkpoint(s))"
  local cmd="cd ${NODE_REPO_DIR} && export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=${UV_CACHE_DIR_NODE} && export HF_HOME=${HF_HOME_NODE} && mkdir -p \"\$UV_CACHE_DIR\" \"\$HF_HOME\" && curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12 && uv sync --frozen${extra} && uv run olmo-eval --help | head -n 20"
  ssm_invoke "$cmd" "install: uv + python 3.12 + uv sync --frozen${extra}" "$INSTALL_TIMEOUT" >/dev/null \
    || die "Dependency install failed on the node (see StandardErrorContent above)."
  info "dependencies installed ONCE; olmo-eval CLI present"
}

# --- Per-checkpoint pieces (atom §6/§6b/§7, scoped to one step) --------------

# launch_detached_eval <N> <uri>: start the per-checkpoint eval detached with a trailing
# sentinel (fresh per-step output dir; results to .../step_N/ via --s3-group).
launch_detached_eval() {
  local n=$1 uri=$2
  local group results_dir node_log provider_opt="" limit_opt=""
  group=$(step_group "$n")
  results_dir="${NODE_RESULTS_DIR}/step_${n}"
  node_log="${NODE_STAGE_DIR}/eval_step_${n}.log"
  [ "$CHECKPOINT_KIND" = olmo_core ] && provider_opt="-H default -o provider.kind=olmo_core "
  [ -n "$LIMIT" ] && limit_opt=" -o limit=${LIMIT}"
  local inner="export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && export UV_CACHE_DIR=${UV_CACHE_DIR_NODE} && export HF_HOME=${HF_HOME_NODE} && mkdir -p \"\$UV_CACHE_DIR\" \"\$HF_HOME\" ${results_dir} && uv run olmo-eval run ${provider_opt}-m ${uri} -t ${TASKS}${limit_opt} -O ${results_dir} --s3-bucket ${S3_BUCKET} --s3-prefix ${S3_PREFIX} --s3-group ${group} --s3-region ${AWS_REGION} > ${node_log} 2>&1; echo \"${DONE_SENTINEL}=\$?\" >> ${node_log}"
  local cmd="cd ${NODE_REPO_DIR} && export HOME=${NODE_HOME} && export PATH=\"\$HOME/.local/bin:\$PATH\" && setsid nohup bash -c '${inner}' >/dev/null 2>&1 &"
  ssm_invoke "$cmd" "run step_${n}: launch eval (detached)" 120 >/dev/null
}

# poll_step <N>: 24 KB-safe sentinel poll (atom §6b). Prints the eval exit code as its
# return code (0 = clean). Never die()s — a stuck/failed step must not abort the slice.
poll_step() {
  local n=$1 node_log="${NODE_STAGE_DIR}/eval_step_${n}.log"
  local tail_cmd="{ grep -aE \"${DONE_SENTINEL}=[0-9]+\" ${node_log} 2>/dev/null | tail -n 1; tail -c 1500 ${node_log} 2>/dev/null; } 2>/dev/null || true"
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would poll step_${n}: ${tail_cmd}  (await ${DONE_SENTINEL}=0)"
    return 0
  fi
  local waited=0 interval=15 out last_line rc
  while :; do
    out=$(ssm_invoke "$tail_cmd" "poll step_${n}: tail eval log" 120 || true)
    last_line=$(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -n 1 || true)
    [ -n "$last_line" ] && info "log> ${last_line}"
    if printf '%s' "$out" | grep -q "${DONE_SENTINEL}=0"; then return 0; fi
    if printf '%s' "$out" | grep -Eq "${DONE_SENTINEL}=[1-9][0-9]*"; then
      rc=$(printf '%s' "$out" | grep -Eo "${DONE_SENTINEL}=[0-9]+" | tail -n1 | cut -d= -f2)
      return "${rc:-1}"
    fi
    if [ "$waited" -ge "$EVAL_TIMEOUT" ]; then
      warn "step_${n}: eval did not complete within ${EVAL_TIMEOUT}s; treating as failure."
      return 124
    fi
    sleep "$interval"
    waited=$(( waited + interval ))
  done
}

# verify_step <N>: assert a result marker (metrics.json) landed in S3 for this step.
verify_step() {
  local n=$1 prefix listing
  prefix=$(step_prefix "$n")
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would verify: aws s3 ls --recursive ${prefix} contains ${DONE_MARKER}"
    return 0
  fi
  listing=$(aws_ro s3 ls "$prefix" --recursive 2>/dev/null || true)
  printf '%s' "$listing" | grep -q "$DONE_MARKER"
}

# write_success <N>: write the explicit _SUCCESS done-marker after verify passes (§3.5.4).
write_success() {
  local n=$1 uri
  uri="$(step_prefix "$n")${SUCCESS_MARKER}"
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would write done-marker: aws s3 cp - ${uri}"
    return 0
  fi
  if printf 'ok\n' | aws s3 cp - "$uri" "${AWS_COMMON_ARGS[@]}" >/dev/null 2>&1; then
    info "wrote ${SUCCESS_MARKER} at ${uri}"
  else
    warn "could not write ${SUCCESS_MARKER} at ${uri}"
  fi
}

# exfil_log <N>: on failure, push the on-node eval log to .../step_N/eval.log (§3.5.4).
exfil_log() {
  local n=$1 node_log="${NODE_STAGE_DIR}/eval_step_${n}.log" dest
  dest="$(step_prefix "$n")eval.log"
  local cmd="aws s3 cp ${node_log} ${dest} --region ${AWS_REGION} || true"
  ssm_invoke "$cmd" "exfil step_${n}: upload eval.log on failure" 120 >/dev/null \
    || warn "log exfil for step_${n} failed (continuing)."
}

# cleanup_step <N>: per-checkpoint isolation — drop this step's HF/NVMe download + outputs
# before the next -m load so a long slice never fills the ~229 GB scratch. Keep the uv cache
# (that is the install). The detached-run + sentinel boundary already freed the prior VRAM.
cleanup_step() {
  local n=$1 results_dir="${NODE_RESULTS_DIR}/step_${n}"
  local cmd="rm -rf ${HF_HOME_NODE} ${results_dir} && mkdir -p ${HF_HOME_NODE}"
  ssm_invoke "$cmd" "cleanup step_${n}: free HF/NVMe download" 120 >/dev/null \
    || warn "cleanup for step_${n} failed (continuing)."
}

# record_step: append one JSON outcome object to the worker manifest.
record_step() {
  local n=$1 uri=$2 kind=$3 outcome=$4 attempts=$5 dur=$6 rc=$7
  MANIFEST_STEPS+=("$(printf '{"step":%s,"uri":"%s","kind":"%s","outcome":"%s","attempts":%s,"duration_s":%s,"exit_code":%s}' \
    "$n" "$(json_escape "$uri")" "$kind" "$outcome" "$attempts" "$dur" "$rc")")
}

# eval_checkpoint <idx>: the whole per-checkpoint lifecycle. NEVER propagates a failure to
# the caller — a bad checkpoint is recorded 'failed' and the loop continues.
eval_checkpoint() {
  local idx=$1
  local n=${ENUM_STEPS[$idx]} uri=${ENUM_URIS[$idx]} kind=${ENUM_KINDS[$idx]}
  local prefix
  prefix=$(step_prefix "$n")
  log "Checkpoint [$((idx + 1))/${COUNT}] step_${n}  ${uri}"
  info "results -> ${prefix}"

  # skip-if-done: explicit _SUCCESS preferred; the atom's metrics.json is also accepted.
  if [ "$SKIP_IF_DONE" = true ] && [ "$FORCE" != true ]; then
    if [ "$DRY_RUN" = true ]; then
      info "(dry-run) would skip-check: aws s3 ls --recursive ${prefix} for ${SUCCESS_MARKER}/${DONE_MARKER}"
    else
      local listing
      listing=$(aws_ro s3 ls "$prefix" --recursive 2>/dev/null || true)
      if printf '%s' "$listing" | grep -q -e "$SUCCESS_MARKER" -e "$DONE_MARKER"; then
        info "SKIP ✅  ${SUCCESS_MARKER}/${DONE_MARKER} already present — already evaluated."
        record_step "$n" "$uri" "$kind" skipped 0 0 0
        N_SKIP=$(( N_SKIP + 1 ))
        return 0
      fi
      info "not done; evaluating."
    fi
  fi

  local max=$(( RETRIES + 1 )) attempt=0 rc=1 outcome=failed start end dur=0
  while [ "$attempt" -lt "$max" ]; do
    attempt=$(( attempt + 1 ))
    info "attempt ${attempt}/${max} for step_${n}"
    start=$(date +%s)
    rc=0
    launch_detached_eval "$n" "$uri" || rc=$?
    if [ "$rc" -eq 0 ]; then
      poll_step "$n" || rc=$?
    fi
    end=$(date +%s)
    dur=$(( end - start ))
    if [ "$rc" -eq 0 ] && verify_step "$n"; then
      write_success "$n"
      outcome=done
      break
    fi
    warn "step_${n} attempt ${attempt} failed (rc=${rc})."
    exfil_log "$n"
    [ "$attempt" -lt "$max" ] && info "will retry step_${n}."
  done

  if [ "$outcome" = done ]; then
    info "step_${n}: DONE (attempts=${attempt}, ${dur}s)"
    N_DONE=$(( N_DONE + 1 ))
  else
    warn "step_${n}: FAILED after ${attempt} attempt(s); continuing with remaining checkpoints."
    N_FAIL=$(( N_FAIL + 1 ))
  fi
  record_step "$n" "$uri" "$kind" "$outcome" "$attempt" "$dur" "$rc"
  cleanup_step "$n"
  return 0
}

# write_manifests: worker manifest shard + a minimal K=1 summary roll-up (§3.5.4; full
# multi-worker shard-merge is P2.d).
write_manifests() {
  log "Write worker manifest + K=1 summary -> ${S3_PREFIX}/${S3_GROUP}/_batch/"
  local steps_json="" s first=true
  for s in "${MANIFEST_STEPS[@]}"; do
    if [ "$first" = true ]; then steps_json="$s"; first=false; else steps_json="${steps_json},${s}"; fi
  done
  local run_src
  if [ -n "$CHECKPOINT_LIST" ]; then run_src="CHECKPOINT_LIST"; else run_src="$RUN_URI"; fi
  local worker_json summary_json base
  worker_json=$(printf '{"worker":0,"fleet_size":%s,"source":"%s","checkpoint_count":%s,"steps":[%s]}' \
    "$FLEET_SIZE" "$(json_escape "$run_src")" "$COUNT" "$steps_json")
  summary_json=$(printf '{"fleet_size":%s,"total":%s,"done":%s,"skipped":%s,"failed":%s,"workers":["_batch/worker_0.json"]}' \
    "$FLEET_SIZE" "$COUNT" "$N_DONE" "$N_SKIP" "$N_FAIL")
  base="s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/_batch"
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would upload worker manifest -> ${base}/worker_0.json"
    info "  ${worker_json}"
    info "(dry-run) would upload summary        -> ${base}/summary.json"
    info "  ${summary_json}"
    return 0
  fi
  printf '%s\n' "$worker_json" | aws s3 cp - "${base}/worker_0.json" "${AWS_COMMON_ARGS[@]}" >/dev/null 2>&1 \
    && info "wrote ${base}/worker_0.json" || warn "could not upload worker_0.json"
  printf '%s\n' "$summary_json" | aws s3 cp - "${base}/summary.json" "${AWS_COMMON_ARGS[@]}" >/dev/null 2>&1 \
    && info "wrote ${base}/summary.json" || warn "could not upload summary.json"
  info "summary: total=${COUNT} done=${N_DONE} skipped=${N_SKIP} failed=${N_FAIL}"
}

# dispatch_k1: the whole K=1 flow — install ONCE, then loop the checkpoints on that box.
dispatch_k1() {
  log "K=1 single-worker dispatch — install ONCE, then loop ${COUNT} checkpoint(s)"
  info "worker 0 slice: ALL ${COUNT} checkpoint(s) (K=1)"
  info "on-node plan: stage -> install ONCE -> per checkpoint { skip-check -> run(detached)+poll -> verify -> ${SUCCESS_MARKER} -> cleanup }"
  [ "$DRY_RUN" = true ] && info ">>> DRY-RUN: printing the FULL planned on-node sequence; launching NOTHING <<<"

  # Pre-launch gate (Bug #1): decide to-do vs already-done BEFORE paying for a box. Record
  # every pre-satisfied checkpoint as skipped so counts are right in both branches below.
  compute_todo_set
  local i=0
  while [ "$i" -lt "$COUNT" ]; do
    [ "${DONE_FLAG[$i]}" = true ] && record_skipped "$i"
    i=$(( i + 1 ))
  done

  # Empty to-do set: truly ZERO-GPU. Write the manifest/summary and return BEFORE any launch,
  # stage or install — no run-instances at all.
  if [ "$N_TODO" -eq 0 ]; then
    log "All ${COUNT} checkpoint(s) already done — nothing to launch."
    [ "$DRY_RUN" = true ] && info ">>> DRY-RUN: gate reports ALL done; would NOT launch an instance <<<"
    write_manifests
    log "K=1 BATCH COMPLETE — total=${COUNT} done=0 skipped=${N_SKIP} failed=0 (no GPU launched)"
    return 0
  fi

  info "gate: ${N_TODO}/${COUNT} to-do; ${N_PRESKIP} already-done recorded as skipped."
  [ "$DRY_RUN" = true ] && info ">>> DRY-RUN: gate reports ${N_TODO} to-do; WOULD launch and loop ONLY the to-do subset <<<"

  preflight_worker_cap 1
  launch_instance
  wait_running_and_ssm
  stage_repo
  install_deps_once

  # Loop ONLY the to-do subset; the pre-satisfied ones were already recorded as skipped above.
  # eval_checkpoint keeps its own skip-check as a defensive second line.
  i=0
  while [ "$i" -lt "$COUNT" ]; do
    [ "${DONE_FLAG[$i]}" != true ] && eval_checkpoint "$i"
    i=$(( i + 1 ))
  done

  write_manifests
  terminate_and_confirm
  TEARDOWN_DONE=true

  if [ "$DRY_RUN" = true ]; then
    log "DRY-RUN COMPLETE — gate split printed; install/run appeared for the to-do subset ONLY; nothing launched."
  else
    log "K=1 BATCH COMPLETE — total=${COUNT} done=${N_DONE} skipped=${N_SKIP} failed=${N_FAIL}"
    [ "$N_FAIL" -gt 0 ] && warn "${N_FAIL} checkpoint(s) failed; see per-step eval.log in S3."
  fi
}

# ══════════════════════════════════════════════════════════════════════════════
# Dispatch — FLEET (P2.c): ≤3 workers, node-side IMDSv2 static shard, install ONCE
# ══════════════════════════════════════════════════════════════════════════════
# P2.c fans the K=1 install-once/loop-many logic out across a fleet of up to 3 workers using
# node-side static shard assignment (scoping doc §3.5.3): the driver launches K_eff self-
# terminating workers with ONE `run-instances --count K_eff`, writes the ordered to-do list to
# S3 (_batch/todo.json), and ships an on-node slice-loop that each worker runs autonomously —
# each reads its `ami-launch-index` via the IMDSv2 token dance, installs deps ONCE, then loops
# its static slice (positions i, i+K_eff, i+2*K_eff, …) of the shared to-do list, running the
# SAME per-checkpoint sequence as K=1 (skip → run → verify → _SUCCESS → cleanup → bounded retry)
# and writing its shard manifest _batch/worker_<i>.json. The laptop polls each exact instance id
# for the whole-slice sentinel and tears the whole fleet down.
#
# DESIGN CHOICE (K=1 vs fleet): K=1 stays the VALIDATED laptop-driven loop (dispatch_k1),
# byte-for-byte, while the fleet is a SEPARATE node-driven mechanism (dispatch_fleet). The two
# loops are fundamentally different (laptop sends one SSM per step vs. the node runs a self-
# contained slice-loop), so routing the proven K=1 path through the new on-node script would
# risk regressing it. The node slice-loop MIRRORS the atom's command strings (like K=1 does),
# so both stay faithful to the validated atom without the atom itself changing.

# Ordered to-do subset (the shared shard input): the enumerated checkpoints MINUS the pre-
# satisfied ones, in enumeration order. Worker i owns positions i, i+K_eff, … of THIS list.
TODO_STEPS=()
TODO_URIS=()
TODO_KINDS=()
build_todo_arrays() {
  TODO_STEPS=(); TODO_URIS=(); TODO_KINDS=()
  local i=0
  while [ "$i" -lt "$COUNT" ]; do
    if [ "${DONE_FLAG[$i]}" != true ]; then
      TODO_STEPS+=("${ENUM_STEPS[$i]}")
      TODO_URIS+=("${ENUM_URIS[$i]}")
      TODO_KINDS+=("${ENUM_KINDS[$i]}")
    fi
    i=$(( i + 1 ))
  done
}

# print_fleet_slices <K>: show each worker's static round-robin slice as to-do-list positions
# and their steps (worker 0: {0,K,2K,…}, worker 1: {1,1+K,…}, …). Display only; the real
# slicing happens node-side from todo.json + the IMDSv2 ami-launch-index.
print_fleet_slices() {
  local k=$1 w=0
  while [ "$w" -lt "$k" ]; do
    local positions="" steps="" p=$w
    while [ "$p" -lt "$N_TODO" ]; do
      positions="${positions} ${p}"
      steps="${steps} step${TODO_STEPS[$p]}"
      p=$(( p + k ))
    done
    info "worker ${w} slice — positions {${positions# }} -> {${steps# }}"
    w=$(( w + 1 ))
  done
}

# write_todo_list <uri>: publish the ordered to-do subset to S3 as JSON [{pos,step,uri,kind}, …].
# Every worker reads this ONE object; passing the shared work list via S3 is cleaner than cramming
# it into user-data. Read-only under --dry-run (prints the JSON, uploads nothing).
write_todo_list() {
  local uri=$1
  local json="" p=0 first=true obj
  while [ "$p" -lt "$N_TODO" ]; do
    obj=$(printf '{"pos":%s,"step":%s,"uri":"%s","kind":"%s"}' \
      "$p" "${TODO_STEPS[$p]}" "$(json_escape "${TODO_URIS[$p]}")" "${TODO_KINDS[$p]}")
    if [ "$first" = true ]; then json="$obj"; first=false; else json="${json},${obj}"; fi
    p=$(( p + 1 ))
  done
  json="[${json}]"
  log "Publish shared to-do list -> ${uri}"
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would upload shared to-do list -> ${uri}"
    info "  ${json}"
    return 0
  fi
  printf '%s\n' "$json" | aws s3 cp - "$uri" "${AWS_COMMON_ARGS[@]}" >/dev/null 2>&1 \
    && info "wrote ${uri}" || die "could not upload shared to-do list ${uri}"
}

# prepare_fleet_stage: for STAGE_MODE=tar, tar the local checkout -> S3 -> presign ONCE (the same
# presigned URL is handed to every worker). For clone mode this is a no-op (each worker clones).
STAGE_URL=""
prepare_fleet_stage() {
  [ "$STAGE_MODE" != tar ] && return 0
  log "Fleet stage (tar): tar local checkout -> S3 -> presign (one bundle, shared by all workers)"
  TARBALL="${TMPDIR:-/tmp}/olmo-eval-batch-fleet-$$.tgz"
  [ -d "${LOCAL_REPO_PARENT}/${REPO_DIRNAME}" ] \
    || die "Local repo not found at ${LOCAL_REPO_PARENT}/${REPO_DIRNAME}."
  run tar czf "$TARBALL" -C "$LOCAL_REPO_PARENT" "$REPO_DIRNAME"
  run aws s3 cp "$TARBALL" "$STAGING_URI" "${AWS_COMMON_ARGS[@]}"
  STAGE_URL=$(run aws s3 presign "$STAGING_URI" --expires-in "$PRESIGN_EXPIRY" "${AWS_COMMON_ARGS[@]}")
  [ "$DRY_RUN" = true ] && STAGE_URL="https://DRY-RUN-PRESIGNED-URL"
  [ -n "$STAGE_URL" ] || die "aws s3 presign returned an empty URL."
}

# build_node_loop_script <K>: emit the on-node slice-loop bash script to stdout. It has two parts:
# a driver-substituted CONFIG header (values expanded here) and a LITERAL logic body (a quoted
# heredoc, so on-node runtime vars like $TOKEN/$SHARD/$rc are NOT expanded by the driver). The
# body is identical on every worker; only the IMDSv2 ami-launch-index differs, which is what
# statically partitions the work. The per-checkpoint sequence MIRRORS the validated atom.
build_node_loop_script() {
  local k=$1
  local todo_uri="s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/_batch/todo.json"
  local manifest_base="s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/_batch"
  cat <<EOF
#!/bin/bash
# ── Fleet worker slice-loop (P2.c) — CONFIG (driver-substituted) ─────────────
K_EFF=${k}
S3_BUCKET='${S3_BUCKET}'
S3_PREFIX='${S3_PREFIX}'
S3_GROUP='${S3_GROUP}'
export AWS_DEFAULT_REGION='${AWS_REGION}'
TASKS='${TASKS}'
LIMIT='${LIMIT}'
CHECKPOINT_KIND='${CHECKPOINT_KIND}'
RETRIES=${RETRIES}
SUCCESS_MARKER='${SUCCESS_MARKER}'
DONE_MARKER='${DONE_MARKER}'
NODE_STAGE_DIR='${NODE_STAGE_DIR}'
NODE_RESULTS_DIR='${NODE_RESULTS_DIR}'
NODE_REPO_DIR='${NODE_REPO_DIR}'
NODE_HOME='${NODE_HOME}'
UV_CACHE_DIR_NODE='${UV_CACHE_DIR_NODE}'
HF_HOME_NODE='${HF_HOME_NODE}'
STAGE_MODE='${STAGE_MODE}'
GIT_REPO_URL='${GIT_REPO_URL}'
GIT_REF='${GIT_REF}'
STAGE_URL='${STAGE_URL}'
TODO_URI='${todo_uri}'
WORKER_MANIFEST_BASE='${manifest_base}'
WORKER_DONE_SENTINEL='${WORKER_DONE_SENTINEL}'
WORKER_SHUTDOWN_GRACE=${WORKER_SHUTDOWN_GRACE}
EOF
  cat <<'NODE_EOF'
# ── Fleet worker slice-loop (P2.c) — LOGIC (identical on every worker) ───────
set -uo pipefail
export HOME="$NODE_HOME"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$UV_CACHE_DIR_NODE"
export HF_HOME="$HF_HOME_NODE"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$NODE_RESULTS_DIR" "$NODE_STAGE_DIR"

# fail_out <code> <msg>: record the completion sentinel with a nonzero code and arm the self-
# shutdown net, so a fatal setup error still surfaces to the laptop poll and never leaks the box.
fail_out() {
  echo "[worker] FATAL: $2"
  echo "${WORKER_DONE_SENTINEL}=$1"
  shutdown -c >/dev/null 2>&1 || true
  shutdown -h +"$WORKER_SHUTDOWN_GRACE" >/dev/null 2>&1 || true
  exit "$1"
}

echo "[worker] boot $(date -u +%FT%TZ 2>/dev/null || date)"

# 1. shard identity via the IMDSv2 token dance (blank index without the token -> do nothing).
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || true)
SHARD=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/ami-launch-index 2>/dev/null || true)
printf '%s' "$SHARD" | grep -qE '^[0-9]+$' \
  || fail_out 97 "blank/invalid ami-launch-index ('$SHARD') — IMDSv2 token dance failed"
echo "[worker] shard index i=$SHARD of K=$K_EFF"

# 2. stage the repo ONCE.
cd "$NODE_STAGE_DIR" || fail_out 96 "cannot cd $NODE_STAGE_DIR"
if [ "$STAGE_MODE" = tar ]; then
  echo "[worker] stage: curl presigned bundle + untar"
  curl -sSL -o olmo-eval.tgz "$STAGE_URL" && tar xzf olmo-eval.tgz || fail_out 96 "tar staging failed"
else
  echo "[worker] stage: git clone $GIT_REPO_URL"
  BRANCH_OPT=""
  [ -n "$GIT_REF" ] && BRANCH_OPT="--branch $GIT_REF "
  git clone --depth 1 ${BRANCH_OPT}"$GIT_REPO_URL" || fail_out 96 "git clone failed"
fi
cd "$NODE_REPO_DIR" || fail_out 96 "repo not at $NODE_REPO_DIR"

# 3. install deps ONCE — the amortized ~10-min cost paid per worker, not per checkpoint.
EXTRA_OPT=""
[ "$CHECKPOINT_KIND" = olmo_core ] && EXTRA_OPT=" --extra olmo_core"
echo "[worker] install ONCE: uv + python 3.12 + uv sync --frozen$EXTRA_OPT"
curl -LsSf https://astral.sh/uv/install.sh | sh || fail_out 95 "uv install failed"
uv python install 3.12 || fail_out 95 "uv python install failed"
uv sync --frozen${EXTRA_OPT} || fail_out 95 "uv sync failed"
uv run olmo-eval --help >/dev/null 2>&1 || fail_out 95 "olmo-eval CLI missing after install"

# 4. read the shared to-do list and parse it into step<TAB>uri<TAB>kind lines.
echo "[worker] fetch to-do list $TODO_URI"
aws s3 cp "$TODO_URI" ./todo.json || fail_out 94 "cannot fetch $TODO_URI"
TODO_LINES=$(python3 - ./todo.json <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    for e in json.load(fh):
        print("%s\t%s\t%s" % (e["step"], e["uri"], e["kind"]))
PY
)
N_TODO=$(printf '%s\n' "$TODO_LINES" | grep -c . || true)
echo "[worker] to-do=$N_TODO; my slice = positions $SHARD,$((SHARD + K_EFF)),... (static round-robin)"

# 5. per-checkpoint sequence (mirrors run_checkpoint_eval.sh, scoped to one step).
MANIFEST=""
add_manifest() {  # step uri kind outcome attempts exit_code
  local obj
  obj=$(printf '{"step":%s,"uri":"%s","kind":"%s","outcome":"%s","attempts":%s,"exit_code":%s}' \
    "$1" "$2" "$3" "$4" "$5" "$6")
  if [ -z "$MANIFEST" ]; then MANIFEST="$obj"; else MANIFEST="$MANIFEST,$obj"; fi
}
process_one() {  # step uri kind
  local n=$1 uri=$2 kind=$3
  local prefix="s3://$S3_BUCKET/$S3_PREFIX/$S3_GROUP/step_$n/"
  local group="$S3_GROUP/step_$n"
  local results_dir="$NODE_RESULTS_DIR/step_$n"
  local node_log="$NODE_STAGE_DIR/eval_step_$n.log"

  # skip-if-done: explicit _SUCCESS preferred; the atom's metrics.json is also accepted. Capture the
  # listing FIRST, then grep it via a here-string (no live `aws | grep -q` pipe). Under `set -o
  # pipefail`, grep -q's early exit on a match would SIGPIPE a still-writing aws and make the pipeline
  # return nonzero, misreading a genuinely-done step as not-done; a here-string has no writer to break.
  local listing
  listing=$(aws s3 ls "$prefix" --recursive 2>/dev/null || true)
  if grep -q -e "$SUCCESS_MARKER" -e "$DONE_MARKER" <<<"$listing"; then
    echo "[worker] SKIP step_$n (done-marker present)"
    add_manifest "$n" "$uri" "$kind" skipped 0 0
    return 0
  fi

  local provider_opt="" limit_opt=""
  [ "$kind" = olmo_core ] && provider_opt="-H default -o provider.kind=olmo_core "
  [ -n "$LIMIT" ] && limit_opt=" -o limit=$LIMIT"

  local max=$((RETRIES + 1)) attempt=0 rc=1 outcome=failed
  while [ "$attempt" -lt "$max" ]; do
    attempt=$((attempt + 1))
    echo "[worker] step_$n attempt $attempt/$max"
    rm -rf "$results_dir"; mkdir -p "$results_dir"
    rc=0
    uv run olmo-eval run ${provider_opt}-m "$uri" -t $TASKS${limit_opt} \
      -O "$results_dir" --s3-bucket "$S3_BUCKET" --s3-prefix "$S3_PREFIX" \
      --s3-group "$group" --s3-region "$AWS_DEFAULT_REGION" > "$node_log" 2>&1 || rc=$?
    # Verify the atom's metrics.json landed. Grep a CAPTURED listing via a here-string (not a live
    # `aws | grep -q` pipe) so grep's early exit can't SIGPIPE aws into a false-negative under
    # `set -o pipefail` (see the skip-check note above) — this was the step_300 misclassification.
    listing=$(aws s3 ls "$prefix" --recursive 2>/dev/null || true)
    if [ "$rc" -eq 0 ] && grep -q "$DONE_MARKER" <<<"$listing"; then
      printf 'ok\n' | aws s3 cp - "${prefix}${SUCCESS_MARKER}" >/dev/null 2>&1 || true
      outcome=done
      break
    fi
    echo "[worker] step_$n attempt $attempt FAILED (rc=$rc); exfil log -> ${prefix}eval.log"
    aws s3 cp "$node_log" "${prefix}eval.log" >/dev/null 2>&1 || true
  done

  # per-checkpoint isolation: drop this step's HF/NVMe download + outputs before the next -m load
  # so a long slice never fills the ~229 GB scratch; keep the uv cache (that is the install).
  rm -rf "$HF_HOME" "$results_dir"; mkdir -p "$HF_HOME"
  add_manifest "$n" "$uri" "$kind" "$outcome" "$attempt" "$rc"
  echo "[worker] step_$n -> $outcome (attempts=$attempt rc=$rc)"
  [ "$outcome" = done ] && return 0 || return 1
}

# iterate MY static slice: to-do positions SHARD, SHARD+K_EFF, SHARD+2*K_EFF, …
pos=$SHARD
n_fail=0
while [ "$pos" -lt "$N_TODO" ]; do
  line=$(printf '%s\n' "$TODO_LINES" | sed -n "$((pos + 1))p")
  step=$(printf '%s' "$line" | cut -f1)
  uri=$(printf '%s' "$line" | cut -f2)
  kind=$(printf '%s' "$line" | cut -f3)
  if [ -n "$step" ]; then
    process_one "$step" "$uri" "$kind" || n_fail=$((n_fail + 1))
  fi
  pos=$((pos + K_EFF))
done

# 6. write this worker's manifest shard (source of truth for the P2.d reduce).
echo "[worker] upload shard manifest -> ${WORKER_MANIFEST_BASE}/worker_${SHARD}.json"
printf '{"worker":%s,"fleet_size":%s,"steps":[%s]}\n' "$SHARD" "$K_EFF" "$MANIFEST" \
  | aws s3 cp - "${WORKER_MANIFEST_BASE}/worker_${SHARD}.json" >/dev/null 2>&1 || true

# 7. completion sentinel (the laptop's 24 KB-safe poll reads this) + self-shutdown net.
echo "${WORKER_DONE_SENTINEL}=${n_fail}"
echo "[worker] slice complete (failures=$n_fail); arming shutdown -h +${WORKER_SHUTDOWN_GRACE}"
shutdown -c >/dev/null 2>&1 || true
shutdown -h +"$WORKER_SHUTDOWN_GRACE" >/dev/null 2>&1 || true
NODE_EOF
}

# ssm_run_on <instance_id> <payload> <desc> [timeout]: like ssm_invoke but targets an EXPLICIT id
# (the fleet has many), printing StandardOutputContent and returning non-zero on failure/timeout.
# Used only on the live path (ship + poll); callers gate --dry-run themselves.
ssm_run_on() {
  local iid=$1 payload=$2 desc=$3 timeout=${4:-120}
  local params cid status waited=0 interval=5
  params=$(ssm_params "$payload")
  cid=$(aws ssm send-command \
    --instance-ids "$iid" --document-name AWS-RunShellScript \
    --comment "$desc" --parameters "$params" \
    --query 'Command.CommandId' --output text \
    "${AWS_COMMON_ARGS[@]}" 2>/dev/null) \
    || { echo "ERROR: ssm send-command failed (${desc} on ${iid})" >&2; return 1; }
  while :; do
    status=$(aws ssm get-command-invocation \
      --command-id "$cid" --instance-id "$iid" \
      --query 'Status' --output text \
      "${AWS_COMMON_ARGS[@]}" 2>/dev/null || echo "Pending")
    case "$status" in
      Success) break ;;
      Failed|Cancelled|TimedOut|Undeliverable|Terminated) return 1 ;;
      *) : ;;
    esac
    [ "$waited" -ge "$timeout" ] && return 1
    sleep "$interval"
    waited=$(( waited + interval ))
  done
  aws ssm get-command-invocation \
    --command-id "$cid" --instance-id "$iid" \
    --query 'StandardOutputContent' --output text \
    "${AWS_COMMON_ARGS[@]}" 2>/dev/null
}

# launch_fleet <count>: ONE `run-instances --count <count>` capturing ALL ids into FLEET_IDS, with
# the same subnet/type capacity fallback as the K=1 launch. --count is all-or-nothing, so a
# capacity shortfall for the whole reservation retries the next subnet/type.
launch_fleet() {
  local count=$1
  log "Launch FLEET of ${count} worker(s) — ONE run-instances --count ${count} (${INSTANCE_TYPE}; self-terminate at shutdown -h +${SHUTDOWN_MINUTES})"
  local user_data
  user_data=$(printf '#!/bin/bash\n# Checkpoint-eval batch auto-terminate safety net.\nshutdown -h +%s\n' "$SHUTDOWN_MINUTES")

  local types="$INSTANCE_TYPE" t
  if [ "$INSTANCE_TYPE_FALLBACK" = true ]; then
    for t in $ALLOWED_INSTANCE_TYPES; do
      [ "$t" = "$INSTANCE_TYPE" ] || types="${types} ${t}"
    done
  fi
  local subnets="${SUBNET_IDS//,/ }"

  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would issue ONE run-instances --count ${count} over types='${types}' subnets='${subnets}' (retry next on capacity errors)"
    local first_type="${types%% *}" first_subnet="${subnets%% *}" trace
    trace=$(quote_cmd aws ec2 run-instances \
      --image-id "$AMI_ID" --instance-type "$first_type" --count "$count" \
      --iam-instance-profile "Name=${INSTANCE_PROFILE}" \
      --security-group-ids "$SECURITY_GROUP_ID" --subnet-id "$first_subnet" \
      --instance-initiated-shutdown-behavior terminate --user-data "$user_data" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=edullm-gpu-worker},{Key=Purpose,Value=checkpoint-eval}]" \
      --query 'Instances[].InstanceId' --output text "${AWS_COMMON_ARGS[@]}")
    printf '    + %s\n' "$trace" >&2
    FLEET_IDS=()
    local w=0
    while [ "$w" -lt "$count" ]; do FLEET_IDS+=("i-DRYRUNWORKER${w}"); w=$(( w + 1 )); done
    info "(dry-run) pretend fleet ids: ${FLEET_IDS[*]} (type=${first_type} subnet=${first_subnet})"
    return 0
  fi

  local itype subnet out rc
  for itype in $types; do
    for subnet in $subnets; do
      info "try run-instances --count ${count}: type=${itype} subnet=${subnet}"
      rc=0
      out=$(run_instances_once "$itype" "$subnet" "$count" "$user_data") || rc=$?
      if [ "$rc" -eq 0 ] && [ -n "$out" ] && [ "$out" != "None" ]; then
        # shellcheck disable=SC2206  # deliberate word-split of the whitespace-separated id list.
        FLEET_IDS=($out)
        [ "${#FLEET_IDS[@]}" -ne "$count" ] \
          && warn "run-instances returned ${#FLEET_IDS[@]} id(s), expected ${count}: ${FLEET_IDS[*]}"
        info "launched fleet ids: ${FLEET_IDS[*]} (type=${itype} subnet=${subnet})"
        return 0
      fi
      if printf '%s' "$out" | grep -qE 'InsufficientInstanceCapacity|Insufficient capacity'; then
        warn "capacity unavailable for ${count}x ${itype} in ${subnet}; trying next combo."
        continue
      fi
      die "run-instances (fleet) failed (type=${itype} subnet=${subnet}, rc=${rc}): ${out}"
    done
  done
  die "run-instances (fleet) exhausted all subnets/types without capacity for ${count}x (types='${types}' subnets='${subnets}')."
}

# wait_fleet_running_and_ssm: block until EVERY launched id is 'running' then SSM Online.
wait_fleet_running_and_ssm() {
  log "Wait for fleet 'running' + SSM Online (${#FLEET_IDS[@]} worker(s))"
  run aws ec2 wait instance-running --instance-ids "${FLEET_IDS[@]}" "${AWS_COMMON_ARGS[@]}"
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would poll each id: aws ssm describe-instance-information ... PingStatus == Online"
    return 0
  fi
  local iid waited ping
  for iid in "${FLEET_IDS[@]}"; do
    waited=0
    while :; do
      ping=$(aws ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=${iid}" \
        --query 'InstanceInformationList[0].PingStatus' --output text \
        "${AWS_COMMON_ARGS[@]}" 2>/dev/null || echo "None")
      [ "$ping" = "Online" ] && { info "[$iid] SSM PingStatus=Online"; break; }
      [ "$waited" -ge "$SSM_ONLINE_TIMEOUT" ] \
        && die "[$iid] SSM did not reach Online within ${SSM_ONLINE_TIMEOUT}s (PingStatus=${ping})."
      sleep 10
      waited=$(( waited + 10 ))
    done
  done
}

# ship_node_loop <K>: build the on-node slice-loop, base64 it (a single SSM-safe token with no
# quotes/newlines), and SSM-send each worker a detached launcher that decodes + runs it, logging
# to NODE_WORKER_LOG. Each worker self-identifies via IMDSv2, so the SAME payload goes to all.
ship_node_loop() {
  local k=$1 script b64 launch_cmd iid
  script=$(build_node_loop_script "$k")
  b64=$(printf '%s' "$script" | base64 | tr -d '\n')
  launch_cmd="mkdir -p ${NODE_STAGE_DIR} && printf '%s' '${b64}' | base64 -d > ${NODE_STAGE_DIR}/worker_loop.sh && chmod +x ${NODE_STAGE_DIR}/worker_loop.sh && setsid nohup bash ${NODE_STAGE_DIR}/worker_loop.sh > ${NODE_WORKER_LOG} 2>&1 & echo shipped"
  log "Ship node slice-loop to ${#FLEET_IDS[@]} worker(s) (detached; each self-identifies via IMDSv2)"
  info "node script: $(printf '%s\n' "$script" | wc -l | tr -d ' ') lines; base64 payload ${#b64} bytes"
  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) would SSM-send to EACH fleet id this detached launcher:"
    info "  + mkdir -p ${NODE_STAGE_DIR} && printf '%s' '<BASE64 ${#b64} chars>' | base64 -d > ${NODE_STAGE_DIR}/worker_loop.sh && chmod +x ... && setsid nohup bash worker_loop.sh > ${NODE_WORKER_LOG} 2>&1 &"
    info "  --- node slice-loop plan (IMDSv2 shard -> stage -> install ONCE -> per-slice per-checkpoint loop -> shard manifest -> sentinel) ---"
    printf '%s\n' "$script" | sed 's/^/      | /' >&2
    return 0
  fi
  for iid in "${FLEET_IDS[@]}"; do
    ssm_run_on "$iid" "$launch_cmd" "ship+launch worker slice-loop" 120 >/dev/null \
      && info "[$iid] slice-loop launched (detached)" \
      || warn "[$iid] failed to launch slice-loop (poll/teardown still guard this id)."
  done
}

# poll_fleet: poll EACH exact launched id for its whole-slice ${WORKER_DONE_SENTINEL} using the
# 24 KB-safe grep-sentinel-first tail (like K=1). Tracks completion per id; never queries by
# shared tag, so it never acts on another engineer's edullm-gpu-worker box. Returns when all
# workers report done or FLEET_TIMEOUT elapses (teardown handles the rest).
poll_fleet() {
  log "Poll fleet for ${WORKER_DONE_SENTINEL} — tracking each exact id (${FLEET_IDS[*]})"
  local tail_cmd="{ grep -aE \"${WORKER_DONE_SENTINEL}=[0-9]+\" ${NODE_WORKER_LOG} 2>/dev/null | tail -n 1; tail -c 1200 ${NODE_WORKER_LOG} 2>/dev/null; } 2>/dev/null || true"
  if [ "$DRY_RUN" = true ]; then
    local iid
    for iid in "${FLEET_IDS[@]}"; do
      info "(dry-run) would poll ${iid}: ssm tail ${NODE_WORKER_LOG} (await ${WORKER_DONE_SENTINEL}=N)"
    done
    return 0
  fi
  local n=${#FLEET_IDS[@]} done_count waited=0 idx iid out last rc
  local status=()
  idx=0; while [ "$idx" -lt "$n" ]; do status[$idx]=""; idx=$(( idx + 1 )); done
  while :; do
    done_count=0
    idx=0
    while [ "$idx" -lt "$n" ]; do
      iid=${FLEET_IDS[$idx]}
      if [ -n "${status[$idx]}" ]; then done_count=$(( done_count + 1 )); idx=$(( idx + 1 )); continue; fi
      out=$(ssm_run_on "$iid" "$tail_cmd" "poll worker slice tail" 120 || true)
      last=$(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -n 1 || true)
      [ -n "$last" ] && info "[$iid] ${last}"
      if printf '%s' "$out" | grep -Eq "${WORKER_DONE_SENTINEL}=[0-9]+"; then
        rc=$(printf '%s' "$out" | grep -Eo "${WORKER_DONE_SENTINEL}=[0-9]+" | tail -n1 | cut -d= -f2)
        status[$idx]="$rc"
        info "[$iid] slice complete (${WORKER_DONE_SENTINEL}=${rc})"
        done_count=$(( done_count + 1 ))
      fi
      idx=$(( idx + 1 ))
    done
    [ "$done_count" -ge "$n" ] && { info "all ${n} worker(s) reported their slice done."; return 0; }
    if [ "$waited" -ge "$FLEET_TIMEOUT" ]; then
      warn "fleet poll timed out after ${FLEET_TIMEOUT}s; ${done_count}/${n} done — tearing the fleet down."
      return 0
    fi
    sleep "$FLEET_POLL_INTERVAL"
    waited=$(( waited + FLEET_POLL_INTERVAL ))
  done
}

# teardown_fleet: terminate EVERY exact id we launched, then confirm. This runs under the OPERATOR
# creds (laptop-side), so the node-role Purpose=checkpoint-eval self-terminate scoping (§3.5.6)
# does NOT block it — the operator's ec2:TerminateInstances covers our ids regardless of tag.
teardown_fleet() {
  [ "${#FLEET_IDS[@]}" -eq 0 ] && return 0
  log "Teardown: terminating fleet ${FLEET_IDS[*]}"
  run aws ec2 terminate-instances --instance-ids "${FLEET_IDS[@]}" "${AWS_COMMON_ARGS[@]}" >/dev/null || true
  if [ "$DRY_RUN" = true ]; then return 0; fi
  aws ec2 wait instance-terminated --instance-ids "${FLEET_IDS[@]}" "${AWS_COMMON_ARGS[@]}" 2>/dev/null || true
  local states
  states=$(aws ec2 describe-instances --instance-ids "${FLEET_IDS[@]}" \
    --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text \
    "${AWS_COMMON_ARGS[@]}" 2>/dev/null || echo "unknown")
  info "fleet states:"
  printf '%s\n' "$states" >&2
}

# merge_fleet_summary <plan_file> <shard_dir>: the PURE P2.d reduce (no S3/AWS) — print the merged
# summary JSON to stdout. It reads the driver-built PLAN (the gate pre-skipped checkpoints plus the
# to-do assignment "position p -> worker p % k_eff") and whatever worker_<i>.json shards were
# downloaded into <shard_dir>, then tallies done/skipped/failed across ALL workers PLUS the pre-
# skipped checkpoints and lists every step's worker id + outcome. A missing / partial / invalid
# shard (a worker that died before uploading) is tolerated: each of its assigned steps is recorded
# 'unknown' (counted as failed) rather than crashing. Kept AWS-free so it is unit-testable offline.
merge_fleet_summary() {
  local plan_file=$1 shard_dir=$2
  python3 - "$plan_file" "$shard_dir" <<'PY'
import json, os, sys

plan_file, shard_dir = sys.argv[1], sys.argv[2]
with open(plan_file) as fh:
    plan = json.load(fh)

k_eff = int(plan["k_eff"])

# Load each expected worker shard. A shard that is absent, truncated, or not valid JSON is treated
# as missing so its assigned steps fall through to 'unknown' below (worker died before uploading).
shard_steps = {}          # worker_id -> {step_int: outcome}
present, missing = [], []
for w in range(k_eff):
    path = os.path.join(shard_dir, "worker_%d.json" % w)
    try:
        with open(path) as fh:
            data = json.load(fh)
        m = {}
        for s in data.get("steps", []):
            m[int(s["step"])] = s.get("outcome", "unknown")
        shard_steps[w] = m
        present.append(w)
    except Exception:
        missing.append(w)

steps_out = []
done = skipped = failed = 0

# Gate pre-skipped checkpoints (recorded laptop-side, never dispatched to a worker).
for e in plan.get("preskipped", []):
    steps_out.append({"step": int(e["step"]), "worker": -1,
                      "uri": e.get("uri", ""), "outcome": "skipped"})
    skipped += 1

# To-do checkpoints: resolve each from its assigned worker's shard; a missing shard or a step the
# worker never recorded becomes 'unknown' (counted as failed) so nothing is silently dropped.
for e in plan.get("todo", []):
    step = int(e["step"])
    w = int(e["worker"])
    if w in shard_steps and step in shard_steps[w]:
        outcome = shard_steps[w][step]
    else:
        outcome = "unknown"
    steps_out.append({"step": step, "worker": w, "uri": e.get("uri", ""), "outcome": outcome})
    if outcome == "done":
        done += 1
    elif outcome == "skipped":
        skipped += 1
    else:
        failed += 1

steps_out.sort(key=lambda x: x["step"])
summary = {
    "fleet_size": int(plan["fleet_size"]),
    "k_eff": k_eff,
    "total": int(plan["total"]),
    "done": done,
    "skipped": skipped,
    "failed": failed,
    "workers_present": sorted(present),
    "workers_missing": sorted(missing),
    "worker_shards": ["_batch/worker_%d.json" % w for w in range(k_eff)],
    "steps": steps_out,
}
print(json.dumps(summary))
PY
}

# write_fleet_summary <K>: the P2.d aggregation — build the reduce PLAN from the driver's own state
# (gate pre-skipped + the to-do -> worker assignment), download every _batch/worker_<i>.json shard,
# merge_fleet_summary them into a single _batch/summary.json (fleet_size/k_eff/total/done/skipped/
# failed + a per-step worker+outcome list), and upload it. Robust to a dead worker's missing shard
# (its steps count as failed/unknown). --dry-run prints the plan and uploads nothing.
write_fleet_summary() {
  local k=$1
  log "Aggregate fleet shards (P2.d reduce) -> ${S3_PREFIX}/${S3_GROUP}/_batch/summary.json"
  local base="s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/_batch"

  # Reduce PLAN: the expected work map the merge checks each shard against. Pre-skipped checkpoints
  # (gate) + the to-do subset with its static round-robin owner (position p -> worker p % k).
  local preskip_json="" todo_json="" first p w i obj
  first=true; i=0
  while [ "$i" -lt "$COUNT" ]; do
    if [ "${DONE_FLAG[$i]}" = true ]; then
      obj=$(printf '{"step":%s,"uri":"%s","kind":"%s"}' \
        "${ENUM_STEPS[$i]}" "$(json_escape "${ENUM_URIS[$i]}")" "${ENUM_KINDS[$i]}")
      if [ "$first" = true ]; then preskip_json="$obj"; first=false; else preskip_json="${preskip_json},${obj}"; fi
    fi
    i=$(( i + 1 ))
  done
  first=true; p=0
  while [ "$p" -lt "$N_TODO" ]; do
    w=0; [ "$k" -gt 0 ] && w=$(( p % k ))
    obj=$(printf '{"pos":%s,"step":%s,"uri":"%s","kind":"%s","worker":%s}' \
      "$p" "${TODO_STEPS[$p]}" "$(json_escape "${TODO_URIS[$p]}")" "${TODO_KINDS[$p]}" "$w")
    if [ "$first" = true ]; then todo_json="$obj"; first=false; else todo_json="${todo_json},${obj}"; fi
    p=$(( p + 1 ))
  done
  local plan_json
  plan_json=$(printf '{"fleet_size":%s,"k_eff":%s,"total":%s,"preskipped":[%s],"todo":[%s]}' \
    "$FLEET_SIZE" "$k" "$COUNT" "$preskip_json" "$todo_json")

  if [ "$DRY_RUN" = true ]; then
    info "(dry-run) reduce plan: ${plan_json}"
    info "(dry-run) would download _batch/worker_<i>.json (i=0..$(( k - 1 ))), merge, and upload -> ${base}/summary.json"
    return 0
  fi

  # No python3 laptop-side: fall back to a minimal pointer index rather than crash the teardown.
  if ! command -v python3 >/dev/null 2>&1; then
    warn "python3 not found; writing a minimal pointer summary instead of the full merge."
    local shards="" ww=0 f2=true ss
    while [ "$ww" -lt "$k" ]; do
      ss="\"_batch/worker_${ww}.json\""
      if [ "$f2" = true ]; then shards="$ss"; f2=false; else shards="${shards},${ss}"; fi
      ww=$(( ww + 1 ))
    done
    printf '{"fleet_size":%s,"k_eff":%s,"total":%s,"preskipped":%s,"todo":%s,"worker_shards":[%s],"aggregation":"unavailable-no-python3"}\n' \
      "$FLEET_SIZE" "$k" "$COUNT" "$N_PRESKIP" "$N_TODO" "$shards" \
      | aws s3 cp - "${base}/summary.json" "${AWS_COMMON_ARGS[@]}" >/dev/null 2>&1 \
      && info "wrote minimal ${base}/summary.json" || warn "could not upload summary.json"
    return 0
  fi

  local tmp plan_file summary_json ww
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/p2d-merge-XXXXXX")
  plan_file="${tmp}/plan.json"
  printf '%s\n' "$plan_json" > "$plan_file"

  # Download every expected shard; a worker that died before uploading is simply absent here and
  # the merge marks its steps 'unknown'. Read-only (aws_ro), so it's identity-agnostic like the rest.
  ww=0
  while [ "$ww" -lt "$k" ]; do
    if aws_ro s3 cp "${base}/worker_${ww}.json" "${tmp}/worker_${ww}.json" >/dev/null 2>&1; then
      info "downloaded worker_${ww}.json shard"
    else
      warn "worker_${ww}.json shard missing/unreadable — its steps -> 'unknown' (counted failed)."
    fi
    ww=$(( ww + 1 ))
  done

  summary_json=$(merge_fleet_summary "$plan_file" "$tmp") || {
    warn "shard merge failed; leaving shards in ${tmp} for inspection."
    return 0
  }
  rm -rf "$tmp"

  printf '%s\n' "$summary_json" | aws s3 cp - "${base}/summary.json" "${AWS_COMMON_ARGS[@]}" >/dev/null 2>&1 \
    && info "wrote ${base}/summary.json (merged reduce)" \
    || warn "could not upload summary.json"
  info "summary: $(printf '%s' "$summary_json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("total=%s done=%s skipped=%s failed=%s present=%s missing=%s" % (d["total"],d["done"],d["skipped"],d["failed"],d["workers_present"],d["workers_missing"]))' 2>/dev/null || printf '%s' "$summary_json")"
}

# dispatch_fleet: the whole FLEET (P2.c) flow — gate, K_eff clamp, publish the shared to-do list,
# launch K_eff workers, ship the node slice-loop, poll each exact id, tear the fleet down.
dispatch_fleet() {
  log "FLEET dispatch (P2.c) — FLEET_SIZE=${FLEET_SIZE}; node-side IMDSv2 static shard; install ONCE/worker"

  # Pre-launch gate (reuse the fix): to-do vs already-done BEFORE paying for any box.
  compute_todo_set
  local i=0
  while [ "$i" -lt "$COUNT" ]; do
    [ "${DONE_FLAG[$i]}" = true ] && record_skipped "$i"
    i=$(( i + 1 ))
  done

  # Empty to-do set: truly ZERO-GPU — write the index and return BEFORE any launch.
  if [ "$N_TODO" -eq 0 ]; then
    log "All ${COUNT} checkpoint(s) already done — nothing to launch."
    [ "$DRY_RUN" = true ] && info ">>> DRY-RUN: gate reports ALL done; would NOT launch any worker <<<"
    write_fleet_summary 0
    log "FLEET BATCH COMPLETE — total=${COUNT} done=0 skipped=${N_SKIP} failed=0 (no GPU launched)"
    return 0
  fi

  # K_eff = min(FLEET_SIZE, N_TODO, hard cap): never more workers than to-do steps, never > 3.
  local k_eff=$FLEET_SIZE
  [ "$k_eff" -gt "$N_TODO" ] && k_eff=$N_TODO
  [ "$k_eff" -gt "$FLEET_SIZE_MAX" ] && k_eff=$FLEET_SIZE_MAX
  log "Effective fleet width K_eff = min(FLEET_SIZE=${FLEET_SIZE}, N_TODO=${N_TODO}, cap=${FLEET_SIZE_MAX}) = ${k_eff}"

  build_todo_arrays
  info "shared to-do list = ${N_TODO} checkpoint(s); per-worker static slices (positions i,i+K,…):"
  print_fleet_slices "$k_eff"
  [ "$DRY_RUN" = true ] && info ">>> DRY-RUN: WOULD run-instances --count ${k_eff}, publish todo.json, ship the node loop; launching NOTHING <<<"

  local todo_uri="s3://${S3_BUCKET}/${S3_PREFIX}/${S3_GROUP}/_batch/todo.json"

  prepare_fleet_stage
  write_todo_list "$todo_uri"
  preflight_worker_cap "$k_eff"
  launch_fleet "$k_eff"
  wait_fleet_running_and_ssm
  ship_node_loop "$k_eff"
  poll_fleet
  write_fleet_summary "$k_eff"
  teardown_fleet
  TEARDOWN_DONE=true

  if [ "$DRY_RUN" = true ]; then
    log "DRY-RUN COMPLETE — K_eff=${k_eff}; run-instances --count ${k_eff}, todo.json write, per-worker slices + node loop plan printed; NOTHING launched."
  else
    log "FLEET BATCH COMPLETE — K_eff=${k_eff}; per-worker shards at ${S3_PREFIX}/${S3_GROUP}/_batch/worker_<i>.json; total=${COUNT} preskipped=${N_PRESKIP} todo=${N_TODO}"
    info "Shard-merge roll-up (done/skipped/failed across all workers) written to _batch/summary.json."
  fi
}

# ── Branch: K=1 (P2.b, validated laptop-driven loop) vs. FLEET (P2.c) ────────
# K=1 stays byte-for-byte the proven dispatch_k1; FLEET_SIZE>=2 runs the separate node-driven
# dispatch_fleet (see the DESIGN CHOICE note above).
if [ "$FLEET_SIZE" -eq 1 ]; then
  dispatch_k1
else
  dispatch_fleet
fi
exit 0
