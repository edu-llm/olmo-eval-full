#!/usr/bin/env bash
# End-to-end exercise of run_eval_sweep.sh with aws/uv/python3 shimmed.
# No GPU, no AWS, no network. Verifies the loop, markers, and aggregation.
#
# Two skills ship a copy of that script and they have diverged, so this driver is
# written to run against either: lib.sh picks one from $EVALCK_SKILL and
# run_tests.py lists the file once per skill. Nothing below may state a fact that
# is true of only one copy -- every count comes from the registry the script under
# test will read, and anything one copy cannot be asked is probed for rather than
# assumed. The alternative is what this file used to be: a green suite for the
# copy nobody runs on a GPU.
#
# The sweep takes --checkpoint (exactly one) or --checkpoint-root (a prefix whose
# children are checkpoints). Cases wanting a specific set therefore get their own
# root; CASE I pins that the withdrawn plural still explains itself.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${HERE}/../lib.sh"

# Recorded before anything runs, so CASE K can tell what this run created from
# what it merely found. An "s3:" left behind by an older run is worth removing,
# but it is not this run's failure.
LITTER_DIRS=("${TESTS_DIR}" "${TESTS_DIR}/shell" "${REPO}" "${PWD}")
PRE_LITTER=""
for d in "${LITTER_DIRS[@]}"; do
  if [[ -e "${d}/s3:" ]]; then PRE_LITTER="${PRE_LITTER} ${d}"; fi
done

SANDBOX="$(sandbox_dir)"
SHIM="${SANDBOX}/shim"
CKPTS="${SANDBOX}/ckpts"
HF_ROOT="${SANDBOX}/ckpts_hf"
AWSLOG="${SANDBOX}/aws_calls.log"

# emptybin exists so a case can exercise the preflight with no shims on PATH.
mkdir -p "${SHIM}" "${CKPTS}" "${HF_ROOT}" "${SANDBOX}/emptybin"
: > "${AWSLOG}"
echo "skill:   ${SKILL_NAME}"
echo "sweep:   ${SWEEP}"
echo "sandbox: ${SANDBOX}"

# --- shims -------------------------------------------------------------------
# aws: record every call, succeed. `aws s3 ls` returns nothing (every case here
# uses local checkpoints, so S3 discovery is not exercised).
cat > "${SHIM}/aws" <<SH
#!/usr/bin/env bash
echo "aws \$*" >> "${AWSLOG}"
# The sweep syncs in both directions: s3->local to fetch a checkpoint, and
# local->s3 to upload results. Only the fetch may materialize anything. Matching
# on "s3 sync" alone made the upload run mkdir -p on an s3:// URI, which is a
# perfectly valid *relative directory name* -- so every run quietly built an
# "s3:/bucket/evals/..." tree under whatever the working directory happened to
# be. Require the source to be remote and the destination local.
if [[ "\$1 \$2" == "s3 sync" && "\$3" == s3://* && "\$4" != s3://* ]]; then
  dest="\$4"
  mkdir -p "\${dest}"
  echo '{}' > "\${dest}/config.json"
  : > "\${dest}/model.safetensors"
fi
if [[ "\$1 \$2" == "s3 cp" && "\$3" == "-" ]]; then
  cat > /dev/null   # consume stdin for marker writes
fi
exit 0
SH

# uv: stand in for `uv run olmo-eval run`. Writes a metrics.json shaped like the
# real MetricsOutput at whatever -O points to. FAIL_FOR makes one checkpoint fail
# so the isolation path is exercised.
cat > "${SHIM}/uv" <<'SH'
#!/usr/bin/env bash
outdir=""
model=""
tasks=()
prev=""
for a in "$@"; do
  case "${prev}" in
    -O) outdir="${a}" ;;
    -m) model="${a}" ;;
    -t) tasks+=("${a}") ;;
  esac
  prev="${a}"
done
echo "uv $*" >> "${AWS_CALLS_LOG}"

# uv is now called for four distinct things, not just the eval. Dispatch on shape
# so each is exercised independently.
case "$*" in
  --version)          echo "uv 0.0.0-shim"; exit 0 ;;
  # Preflight's importability probe.
  *"python -c"*)      exit "${PREFLIGHT_IMPORT_RC:-0}" ;;
  # Bootstrap's dependency sync.
  "sync "*|"sync")    exit "${SYNC_RC:-0}" ;;
esac

# The bundled converter, invoked as `uv run python <path>/convert_to_hf.py ...`.
if [[ "$*" == *convert_to_hf.py* ]]; then
  if [[ "${CONVERT_RC:-0}" != "0" ]]; then
    echo "simulated conversion failure" >&2
    exit "${CONVERT_RC}"
  fi
  # Emit a plausible HF checkpoint at the -o destination.
  prev=""
  for a in "$@"; do
    if [[ "${prev}" == "-o" ]]; then mkdir -p "${a}"; echo '{}' > "${a}/config.json"; : > "${a}/model.safetensors"; fi
    prev="${a}"
  done
  exit 0
fi

if [[ -n "${FAIL_FOR:-}" && "${model}" == *"${FAIL_FOR}"* ]]; then
  echo "simulated eval failure for ${model}" >&2
  exit 1
fi
# An eval that takes measurable time, which CASE J needs and nothing else wants:
# an interim sync on a timer has nothing to sync if the eval is instantaneous.
if [[ -n "${EVAL_DELAY:-}" ]]; then sleep "${EVAL_DELAY}"; fi
mkdir -p "${outdir}"
{
  echo '{"timestamp":"2026-08-03T00:00:00Z","config":{"model":"m"},"tasks":['
  first=1
  for t in "${tasks[@]}"; do
    if [[ ${first} -eq 0 ]]; then echo ','; fi
    first=0
    if [[ "${t}" == "naturalqs" ]]; then
      printf '{"task":"%s","metrics":{"f1":{"drop_f1":0.31},"accuracy":{"drop_exact_match":0.2}},"num_instances":3610,"primary_metric":"f1:drop_f1"}' "${t}"
    elif [[ "${t}" == "jeopardy" ]]; then
      printf '{"task":"%s","metrics":{"f1":{"f1":0.4},"accuracy":{"squad_exact_match":0.28}},"num_instances":2117,"primary_metric":"f1:f1"}' "${t}"
    else
      printf '{"task":"%s","metrics":{"accuracy":{"logprob":0.5}},"num_instances":100,"primary_metric":"accuracy:logprob"}' "${t}"
    fi
  done
  echo '],"summary":{},"errors":[]}'
} > "${outdir}/metrics.json"
exit 0
SH

# Git Bash has `python` but not always `python3`; the script calls python3.
cat > "${SHIM}/python3" <<'SH'
#!/usr/bin/env bash
exec python "$@"
SH

chmod +x "${SHIM}"/*
export PATH="${SHIM}:${PATH}"
export AWS_CALLS_LOG="${AWSLOG}"

# --- fake checkpoints --------------------------------------------------------
# Two HF-format locals, plus one native OLMo-core to exercise the converter path.
# ckpts/ holds all three; ckpts_hf/ holds only the two HF ones, because a root is
# now the only way to ask for more than one checkpoint and the cases below want
# different counts.
for step in step9 step10; do
  for root in "${CKPTS}" "${HF_ROOT}"; do
    mkdir -p "${root}/${step}"
    echo '{}' > "${root}/${step}/config.json"
    : > "${root}/${step}/model.safetensors"
  done
done
mkdir -p "${CKPTS}/step2000/model_and_optim"
echo '{}' > "${CKPTS}/step2000/config.json"

# --- what this copy of the sweep can be asked --------------------------------
# eval-direct-gpu's sweep pushes partial results to S3 on a timer while a
# checkpoint is still evaluating; eval-platform's has no such flag and would
# reject it as an unknown argument. Probed from the script's own help rather than
# from the skill name, so whichever copy next gains or loses the flag is followed
# without an edit here.
#
# Every case but CASE J turns the loop off, through SWEEP_CMD. Interim syncing is
# CASE J's subject and only CASE J's; anywhere else it would start a background job
# per checkpoint whose first tick an instantly-shimmed eval never reaches -- a job
# that proves nothing, and an aws call log that has to be read more carefully than
# it should be.
SWEEP_CMD=(bash "${SWEEP}")
HAS_INTERIM_SYNC=0
if bash "${SWEEP}" --help 2>/dev/null | grep -qF -- '--sync-interval'; then
  HAS_INTERIM_SYNC=1
  SWEEP_CMD+=(--sync-interval 0)
fi

# Whether this copy can choose a provider. Where it can, the default is olmo_core:
# it reads a native checkpoint as it stands and splits the run so multiple-choice and
# generative work get their own batch size, which means two model loads rather than
# one and no conversion at all. Conversion, tensor parallelism and the VRAM fraction
# are vLLM-only there, so the cases that exercise them ask for vllm_server explicitly
# through SWEEP_VLLM. A copy without --provider is vLLM-only and both arrays are the
# same command, so those cases keep testing exactly what they tested before.
HAS_PROVIDER=0
SWEEP_VLLM=("${SWEEP_CMD[@]}")
if bash "${SWEEP}" --help 2>/dev/null | grep -qF -- '--provider'; then
  HAS_PROVIDER=1
  SWEEP_VLLM+=(--provider vllm_server)
fi

# --- what the registry under test says ---------------------------------------
# Read out of the registry the script will read, never written down here. The two
# differ where it counts: eval-platform registers three ':mc' variants that
# eval-direct-gpu does not, and their smoke groups are built from opposite ends --
# one caps the default set, the other subtracts a group from 'all'. A literal that
# is right for one copy is wrong for the other, and wrong for both the day a
# benchmark is added.
#
# Summing the registry here is still a check on the resolver rather than a copy of
# its answer, because it is a second implementation over the same input. What is
# deliberately not done is expanding a group defined by 'like' or 'exclude': that
# would be a second implementation of the resolver itself, in the file that exists
# to drive the real one. Only groups spelled out as a plain list are expanded
# below, and smoke's membership is read back out of the sweep's own log.
registry_members() {
  # registry_members <group> -- the names, for a group given as a plain list
  python3 -c "
import json, sys
spec = json.load(open(sys.argv[1], encoding='utf-8'))['groups'][sys.argv[2]]
if not isinstance(spec, list):
    sys.exit(f'group {sys.argv[2]!r} is derived, not a plain list')
print(' '.join(spec))" "${REGISTRY}" "$1" | tr -d '\r'
}
registry_cost() {
  # registry_cost "<names>" ["<caps>"] -- "<instances> <prompts>", priced as the
  # sweep prices it: a cap replaces a split size rather than trimming the total.
  # Caps are one per name, in order, because the two registries no longer agree
  # that a limited group has a single cap -- see registry_group_caps.
  python3 -c "
import json, sys
bench = json.load(open(sys.argv[1], encoding='utf-8'))['benchmarks']
caps = sys.argv[3].split()
counts = [(min(bench[b]['instances'], int(caps[i])) if caps else bench[b]['instances'],
           bench[b]['choices']) for i, b in enumerate(sys.argv[2].split())]
print(sum(n for n, _ in counts), sum(n * c for n, c in counts))" \
    "${REGISTRY}" "$1" "${2:-}" | tr -d '\r'
}
registry_group_caps() {
  # registry_group_caps <group> "<names>" -- the instance cap the registry implies
  # for each name, in order, or empty when the group sets none.
  #
  # A second implementation of the resolver's rule over the same input, not a copy
  # of its answer. One registry's smoke group sets a flat `limit`; the other sets a
  # `prompt_budget`, an even share of which becomes a different instance cap per
  # benchmark because prompts are instances times choices. Priced only against a
  # total, a bug that emitted one uniform cap for a budgeted group could still add
  # up, so the caps themselves are compared below.
  python3 -c "
import json, sys
reg = json.load(open(sys.argv[1], encoding='utf-8'))
bench, spec = reg['benchmarks'], reg['groups'][sys.argv[2]]
names = sys.argv[3].split()
budget = spec.get('prompt_budget') if isinstance(spec, dict) else None
flat = spec.get('limit') if isinstance(spec, dict) else None
if budget is not None:
    share = budget / len(names)
    print(' '.join(str(max(1, min(bench[n]['instances'], round(share / bench[n]['choices']))))
                   for n in names))
elif flat is not None:
    print(' '.join(str(flat) for _ in names))
else:
    print('')" "${REGISTRY}" "$1" "$2" | tr -d '\r'
}
registry_unsafe() {
  # registry_unsafe -- the names the registry flags as scoring a different
  # population when limited. Empty today; the assertions below branch on it so
  # they keep testing the warning if one is ever put back.
  python3 -c "
import json, sys
print(' '.join(json.load(open(sys.argv[1], encoding='utf-8')).get('limit_unsafe', {})))" \
    "${REGISTRY}" | tr -d '\r'
}

read -r -a DEFAULT_BENCH <<< "$(registry_members default)"
read -r DEFAULT_INSTANCES DEFAULT_PROMPTS <<< "$(registry_cost "${DEFAULT_BENCH[*]}")"
read -r -a ALL_BENCH <<< "$(registry_members all)"
read -r _ ALL_PROMPTS <<< "$(registry_cost "${ALL_BENCH[*]}")"
read -r -a FACT_BENCH <<< "$(registry_members fact_proxy)"
read -r CSQA_INSTANCES CSQA_PROMPTS <<< "$(registry_cost csqa)"
read -r -a UNSAFE_BENCH <<< "$(registry_unsafe)"
echo "registry: default=${#DEFAULT_BENCH[@]} all=${#ALL_BENCH[@]} limit_unsafe=${#UNSAFE_BENCH[@]}"

echo "======================================================================"
echo "CASE A: --dry-run prints the cost estimate and spends nothing"
echo "======================================================================"
OUT_A="${SANDBOX}/a.log"
"${SWEEP_CMD[@]}" --checkpoint-root "${HF_ROOT}" \
  --s3-out s3://bucket/evals --dry-run > "${OUT_A}" 2>&1
RC=$?
check "exits 0" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"
# Bare invocation resolves the registry's "default" group -- the MCQ reasoning
# tasks -- and prices it, doubled for the two checkpoints under HF_ROOT. Both
# registries happen to define the same five today, which is exactly why the
# numbers are read rather than trusted to stay that way.
check "defaults to the ${#DEFAULT_BENCH[@]}-benchmark default group" \
  "$(has "benchmarks (${#DEFAULT_BENCH[@]})" "${OUT_A}")"
check "names the group it chose" "$(has "group 'default' (registry default)" "${OUT_A}")"
check "estimates the per-checkpoint cost the registry implies" \
  "$(has "~${DEFAULT_INSTANCES} instances / ~${DEFAULT_PROMPTS} vLLM prompts per checkpoint" "${OUT_A}")"
check "estimates whole-sweep prompts" \
  "$(has "~$((DEFAULT_PROMPTS * 2)) vLLM prompts for the whole sweep" "${OUT_A}")"
check "discovered both checkpoints under the root" "$(has 'checkpoints (2)' "${OUT_A}")"
# The log records uv calls too (the preflight import probe), so count aws lines
# specifically rather than testing the file for emptiness.
check "no aws calls made" \
  "$([[ "$(grep -c '^aws ' "${AWSLOG}")" == "0" ]] && echo 1 || echo 0)" \
  "$(grep -c '^aws ' "${AWSLOG}") aws lines"
sed 's/^/    /' "${OUT_A}"

echo
echo "======================================================================"
echo "CASE B: invalid benchmark name rejected before anything runs"
echo "======================================================================"
OUT_B="${SANDBOX}/b.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --benchmarks "commonsense_qa" --dry-run > "${OUT_B}" 2>&1
RC=$?
check "exits 2" "$([[ ${RC} -eq 2 ]] && echo 1 || echo 0)" "rc=${RC}"
check "names the offender" "$(has 'unknown benchmark(s): commonsense_qa' "${OUT_B}")"
check "suggests the right name" "$(has "the olmo-eval task is 'csqa'" "${OUT_B}")"
# Checked name by name rather than as one contiguous run: the list is sorted, so
# adding a variant like arc_easy:mc inserts itself between neighbours and breaks a
# substring match without anything actually being wrong.
check "lists the known names" \
  "$(python3 -c "
import sys
line = next((l for l in open(sys.argv[1], encoding='utf-8') if 'known:' in l), '')
print(1 if all(b in line for b in ('arc_easy', 'csqa', 'hellaswag')) else 0)" \
    "${OUT_B}" | tr -d '\r')"
check "points at BENCHMARKS.md" "$(has 'See BENCHMARKS.md' "${OUT_B}")"

echo
echo "======================================================================"
echo "CASE C: full loop, 3 checkpoints, one of them native OLMo-core"
echo "======================================================================"
# step2000 is native. With the bundled converter it now converts and evaluates
# rather than failing, which is the point of shipping a converter with the skill.
# The unconvertible path is exercised in CASE H via a broken override instead.
: > "${AWSLOG}"
OUT_C="${SANDBOX}/c.log"
unset OLMO_CORE_CONVERT
"${SWEEP_CMD[@]}" --checkpoint-root "${CKPTS}" \
  --benchmarks "hellaswag naturalqs jeopardy" \
  --s3-out s3://bucket/evals --keep-local > "${OUT_C}" 2>&1
RC=$?
check "exits 0, every checkpoint succeeded" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"
check "step9 got _READY" "$(has 's3://bucket/evals/step9/_READY' "${AWSLOG}")"
check "step10 got _READY" "$(has 's3://bucket/evals/step10/_READY' "${AWSLOG}")"
check "native step2000 got _READY, not _FAILED" "$(has 's3://bucket/evals/step2000/_READY' "${AWSLOG}")"
check "no converter complaint" "$(lacks 'no_converter' "${OUT_C}")"
EVAL_CALLS="$(grep -c '^uv run olmo-eval' "${AWSLOG}")"
if [[ "${HAS_PROVIDER}" == "1" ]]; then
  # Default provider, so the native checkpoint is read as it stands and each of the
  # three checkpoints is split in two: hellaswag by itself at the multiple-choice
  # batch size, naturalqs and jeopardy together at the generative one.
  check "nothing was converted on the native path" "$(lacks 'convert_to_hf.py' "${AWSLOG}")"
  check "two loads per checkpoint (6 eval calls)" \
    "$([[ "${EVAL_CALLS}" == "6" ]] && echo 1 || echo 0)" "${EVAL_CALLS} calls"
  check "provider override before first -t" \
    "$(grep '^uv run olmo-eval' "${AWSLOG}" | head -1 | grep -qF -- '--harness default -o provider.kind=olmo_core' && echo 1 || echo 0)"
  check "the multiple-choice half carries only hellaswag" \
    "$(grep '^uv run olmo-eval' "${AWSLOG}" | grep -F 'batch_size=512' | head -1 |
        grep -qE -- '-t hellaswag( |$)' && echo 1 || echo 0)"
  check "and not the generative pair" \
    "$(grep '^uv run olmo-eval' "${AWSLOG}" | grep -F 'batch_size=512' | head -1 |
        grep -qF -- '-t naturalqs' && echo 0 || echo 1)"
  check "the generative half carries naturalqs and jeopardy" \
    "$(grep '^uv run olmo-eval' "${AWSLOG}" | grep -F 'batch_size=192' | head -1 |
        grep -qF -- '-t naturalqs' && echo 1 || echo 0)"
  # The whole reason for two invocations: one batch_size cannot serve both halves.
  check "the two halves really do differ in batch size" \
    "$([[ "$(grep '^uv run olmo-eval' "${AWSLOG}" | grep -oE 'batch_size=[0-9]+' | sort -u | wc -l)" == "2" ]] && echo 1 || echo 0)" \
    "$(grep '^uv run olmo-eval' "${AWSLOG}" | grep -oE 'batch_size=[0-9]+' | sort -u | tr '\n' ' ')"
  # None would put every request in one padded chunk; _iter_chunks treats it as one.
  check "batch_size is never left unset" \
    "$([[ "$(grep -c '^uv run olmo-eval' "${AWSLOG}")" == "$(grep '^uv run olmo-eval' "${AWSLOG}" | grep -c 'batch_size=')" ]] && echo 1 || echo 0)"
  # The halves write two metrics.json files, and write_metrics_json opens with "w", so
  # without the merge the summary would show one half and silently lose the other.
  # One checkpoint reporting a benchmark from each half is the merge working.
  SUMMARY="$(sed -n '/RUN_ID/,/^$/p' "${OUT_C}")"
  check "the summary shows step9's multiple-choice half" \
    "$(printf '%s\n' "${SUMMARY}" | grep -qE '^ *step9 +hellaswag' && echo 1 || echo 0)"
  check "and step9's generative half, from the same merged file" \
    "$(printf '%s\n' "${SUMMARY}" | grep -qE '^ *step9 +jeopardy' && echo 1 || echo 0)"
else
  check "the native one was converted" "$(has 'convert_to_hf.py' "${AWSLOG}")"
  check "one vLLM boot per checkpoint (3 eval calls)" \
    "$([[ "${EVAL_CALLS}" == "3" ]] && echo 1 || echo 0)" "${EVAL_CALLS} calls"
  check "all 3 benchmarks in one invocation" \
    "$(grep '^uv run olmo-eval' "${AWSLOG}" | head -1 | grep -qF -- '-t hellaswag -t naturalqs -t jeopardy' && echo 1 || echo 0)"
  check "provider override before first -t" \
    "$(grep '^uv run olmo-eval' "${AWSLOG}" | head -1 | grep -qF -- '--harness default -o provider.kind=vllm_server -t' && echo 1 || echo 0)"
fi
check "accuracy.csv uploaded" "$(has 's3://bucket/evals/accuracy.csv' "${AWSLOG}")"
check "accuracy_wide.csv uploaded" "$(has 's3://bucket/evals/accuracy_wide.csv' "${AWSLOG}")"
check "sweep.log uploaded" "$(has 's3://bucket/evals/sweep.log' "${AWSLOG}")"
check "all three checkpoints carry scores" \
  "$(sed -n '/RUN_ID/,/^$/p' "${OUT_C}" | grep -cE '0\.[0-9]{4}' | awk '{print ($1>=9)?1:0}')" \
  "$(sed -n '/RUN_ID/,/^$/p' "${OUT_C}" | grep -cE '0\.[0-9]{4}') scored cells"
# The assertions above all passed while every score was silently empty, because
# they only checked that rows EXIST. A successful checkpoint must carry an actual
# number, or metrics.json was never parsed and we are looking at placeholders.
check "successful checkpoint has a real score" \
  "$(grep -E 'step9 +hellaswag +accuracy +logprob +[0-9]' "${OUT_C}" >/dev/null && echo 1 || echo 0)" \
  "$(grep -E 'step9 +hellaswag' "${OUT_C}" | head -1)"
check "generative task reports both metrics with scores" \
  "$(grep -cE 'step9 +naturalqs +(f1|accuracy) +\S+ +[0-9]' "${OUT_C}" | awk '{print ($1==2)?1:0}')" \
  "$(grep -cE 'step9 +naturalqs +\S+ +\S+ +[0-9]' "${OUT_C}") scored rows"
check "no successful row is left unscored" \
  "$(grep -E '(step9|step10) +\S+ +- +- +-' "${OUT_C}" >/dev/null && echo 0 || echo 1)"
echo "    --- emitted olmo-eval command ---"
grep '^uv ' "${AWSLOG}" | head -1 | sed 's/^/    /'
echo "    --- aggregation table ---"
sed -n '/RUN_ID/,/^$/p' "${OUT_C}" | sed 's/^/    /'

echo
echo "======================================================================"
echo "CASE D: eval failure keeps the sweep going and still marks the run"
echo "======================================================================"
: > "${AWSLOG}"
OUT_D="${SANDBOX}/d.log"
FAIL_FOR="step9" "${SWEEP_CMD[@]}" --checkpoint-root "${HF_ROOT}" \
  --benchmarks "hellaswag" \
  --s3-out s3://bucket/evals > "${OUT_D}" 2>&1
RC=$?
check "exits 1" "$([[ ${RC} -eq 1 ]] && echo 1 || echo 0)" "rc=${RC}"
check "failed checkpoint marked _FAILED" "$(has 's3://bucket/evals/step9/_FAILED' "${AWSLOG}")"
check "sweep continued to step10" "$(has 's3://bucket/evals/step10/_READY' "${AWSLOG}")"
check "failure logged, not fatal" "$(has 'continuing to next checkpoint' "${OUT_D}")"
check "summary still uploaded" "$(has 's3://bucket/evals/accuracy.csv' "${AWSLOG}")"
check "failed run kept a row" "$(has 'step9' "${OUT_D}")"

echo
echo "======================================================================"
echo "CASE E: --tp emits the harness-scoped kwargs path"
echo "======================================================================"
# Tensor parallelism is a vLLM engine setting, so this asks for that provider where
# the copy under test has a choice. OlmoCoreProvider requires 1 and refuses --tp.
: > "${AWSLOG}"
OUT_E="${SANDBOX}/e.log"
"${SWEEP_VLLM[@]}" --checkpoint "${CKPTS}/step9" --benchmarks hellaswag \
  --s3-out s3://bucket/evals --tp 2 > "${OUT_E}" 2>&1
check "uses provider.kwargs.tensor_parallel_size" \
  "$(grep '^uv ' "${AWSLOG}" | grep -qF -- '-o provider.kwargs.tensor_parallel_size=2' && echo 1 || echo 0)"
check "does NOT use provider.tensor_parallel_size" \
  "$(grep '^uv ' "${AWSLOG}" | grep -qF -- '-o provider.tensor_parallel_size=2' && echo 0 || echo 1)"
check "tp override precedes the first -t" \
  "$(grep '^uv ' "${AWSLOG}" | grep -qF -- 'tensor_parallel_size=2 -t hellaswag' && echo 1 || echo 0)"

echo
echo "======================================================================"
echo "CASE F: bad numeric flag rejected up front, before any spend"
echo "======================================================================"
: > "${AWSLOG}"
OUT_F="${SANDBOX}/f.log"
"${SWEEP_VLLM[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --tp abc > "${OUT_F}" 2>&1
RC=$?
check "exits 2" "$([[ ${RC} -eq 2 ]] && echo 1 || echo 0)" "rc=${RC}"
check "explains which flag" "$(has '--tp must be an integer' "${OUT_F}")"
check "nothing spent" "$([[ ! -s "${AWSLOG}" ]] && echo 1 || echo 0)"

echo
echo "======================================================================"
echo "CASE G: registry drives the defaults and the --limit warning"
echo "======================================================================"
: > "${AWSLOG}"
OUT_G="${SANDBOX}/g.log"
# No --benchmarks and no --group: must resolve the registry's default group.
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --dry-run > "${OUT_G}" 2>&1
check "empty selection resolves the default group" \
  "$(has "benchmarks (${#DEFAULT_BENCH[@]})" "${OUT_G}")"

# --group must resolve from the registry, and the group's size must follow it.
#
# Both numbers are read back out of benchmarks.json rather than written down
# here. "all" is the group that grows whenever a benchmark is added -- it gained
# popqa and triviaqa, which is what stranded the literals that used to sit here
# -- so pinning a constant tests only how recently someone edited this file. It
# also differs between the two skills: eval-platform's 'all' is nine benchmarks
# out of a registry of twelve, because its three ':mc' variants are deliberately
# outside it.
OUT_GA="${SANDBOX}/ga.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --group all --dry-run > "${OUT_GA}" 2>&1
check "--group all resolves the registry's whole 'all' group (${#ALL_BENCH[@]})" \
  "$(has "benchmarks (${#ALL_BENCH[@]})" "${OUT_GA}")"
check "--group all estimate matches the registry (${ALL_PROMPTS} prompts)" \
  "$(has "${ALL_PROMPTS} vLLM prompts per checkpoint" "${OUT_GA}")"
# 'all' used to mean every entry in the registry, and this file asserted that. It
# stopped being true when the :mc variants were registered: they are deliberately
# outside 'all' so a full sweep does not silently double in cost and start mixing
# two scoring formats. What that identity was standing in for -- that no benchmark
# is defined and then left unreachable -- is worth keeping, but it is not kept
# here. python/test_mc_variants.py pins it, and pins the sharper form beside it
# ("'all' is every benchmark that is not a variant of another"), using the
# resolver's own group_members. A copy here would need its own `like` inheritance
# to walk the groups, which is a second implementation of the resolver in a file
# that exists to drive the real one.
OUT_GF="${SANDBOX}/gf.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --group fact_proxy --dry-run > "${OUT_GF}" 2>&1
check "--group fact_proxy resolves ${#FACT_BENCH[@]}" \
  "$(has "benchmarks (${#FACT_BENCH[@]})" "${OUT_GF}")"
check "--group is named in the log" "$(has "group 'fact_proxy'" "${OUT_GF}")"
# 'smoke' is where the two registries disagree most: one caps the default set,
# the other takes 'all' minus fact_proxy at a larger cap. Which set it resolved to
# is read back out of the sweep's own log and priced against the registry, so the
# estimate is checked for exactly the benchmarks the resolver chose without this
# file having to know how either group is built.
OUT_GS="${SANDBOX}/gs.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --group smoke --dry-run > "${OUT_GS}" 2>&1
SMOKE_BENCH="$(sed -n 's/.*benchmarks ([0-9]*) from .*: //p' "${OUT_GS}" | head -1 | tr -d '\r')"
SMOKE_CAPS="$(registry_group_caps smoke "${SMOKE_BENCH}")"
read -r -a SMOKE_CAP_LIST <<< "${SMOKE_CAPS}"
read -r -a SMOKE_NAMES <<< "${SMOKE_BENCH}"
read -r SMOKE_INSTANCES SMOKE_PROMPTS <<< "$(registry_cost "${SMOKE_BENCH}" "${SMOKE_CAPS}")"
DISTINCT_CAPS="$(printf '%s\n' "${SMOKE_CAP_LIST[@]}" | sort -u | wc -l | tr -d ' \r')"
if [[ "${DISTINCT_CAPS}" -eq 1 ]]; then
  check "--group smoke applies the single cap the registry sets on it (${SMOKE_CAP_LIST[0]})" \
    "$(has "capped at ${SMOKE_CAP_LIST[0]} instances per benchmark" "${OUT_GS}")"
else
  # A prompt budget, so the caps differ and the log has to name them one by one:
  # "capped at N instances per benchmark" would state a cap no benchmark got. Each
  # is checked by name, which is what a bug emitting one uniform cap would fail --
  # the total alone can be right for the wrong reason.
  check "--group smoke says its caps vary rather than naming one" \
    "$(lacks 'capped at ' "${OUT_GS}")"
  for i in "${!SMOKE_NAMES[@]}"; do
    check "--group smoke caps ${SMOKE_NAMES[i]} at ${SMOKE_CAP_LIST[i]}" \
      "$(has "${SMOKE_NAMES[i]}: ${SMOKE_CAP_LIST[i]}" "${OUT_GS}")"
  done
  check "--group smoke's caps are genuinely per benchmark, not one value repeated" \
    "$([[ "${DISTINCT_CAPS}" -gt 1 ]] && echo 1 || echo 0)" "${SMOKE_CAPS}"
fi
check "--group smoke prices the capped set it resolved (${SMOKE_INSTANCES} instances)" \
  "$(has "~${SMOKE_INSTANCES} instances / ~${SMOKE_PROMPTS} vLLM prompts per checkpoint" "${OUT_GS}")"
# An unknown group must be rejected, not silently treated as empty.
OUT_GU="${SANDBOX}/gu.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --group nonsense --dry-run > "${OUT_GU}" 2>&1
RC_GU=$?
check "unknown group exits 2" "$([[ ${RC_GU} -eq 2 ]] && echo 1 || echo 0)" "rc=${RC_GU}"
check "unknown group lists valid ones" "$(has 'known groups:' "${OUT_GU}")"
# --group and --benchmarks together are ambiguous and must be refused.
OUT_GB="${SANDBOX}/gb.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --group all --benchmarks "csqa" --dry-run > "${OUT_GB}" 2>&1
RC_GB=$?
check "--group with --benchmarks exits 2" "$([[ ${RC_GB} -eq 2 ]] && echo 1 || echo 0)" "rc=${RC_GB}"
check "--group with --benchmarks explains" "$(has 'not both' "${OUT_GB}")"
# --limit must name only the registry-flagged tasks, driven by limit_unsafe. That
# table is empty in both registries today -- hellaswag and socialiqa were in it
# until their loaders were changed to sample the split they score -- so what can be
# asserted depends on the registry the script under test will read, and is read
# from it rather than written down here. If a name is ever put back, the first
# branch starts running again without an edit.
OUT_G2="${SANDBOX}/g2.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --benchmarks "hellaswag arc_easy" --limit 50 --dry-run > "${OUT_G2}" 2>&1
if [[ ${#UNSAFE_BENCH[@]} -gt 0 ]]; then
  check "limit warning fires for a flagged task" "$(has "${UNSAFE_BENCH[0]}: " "${OUT_G2}")"
  check "limit warning omits an unflagged task" "$(lacks 'arc_easy: loads' "${OUT_G2}")"
else
  check "a limited run still reports the cap it applied" \
    "$(has 'capped at 50 instances per benchmark' "${OUT_G2}")"
  check "and warns about no benchmark, because none is flagged" \
    "$(lacks 'score a different population when limited' "${OUT_G2}")"
  check "hellaswag in particular is no longer called out" \
    "$(lacks 'hellaswag: loads' "${OUT_G2}")"
fi
OUT_G3="${SANDBOX}/g3.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --benchmarks "arc_easy csqa" --limit 50 --dry-run > "${OUT_G3}" 2>&1
check "no limit warning when no flagged task requested" \
  "$(lacks 'score a different population when limited' "${OUT_G3}")"
# A subset must produce a smaller estimate than the full set.
OUT_G4="${SANDBOX}/g4.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --benchmarks "csqa" --dry-run > "${OUT_G4}" 2>&1
check "subset estimate comes from the registry" \
  "$(has "~${CSQA_INSTANCES} instances / ~${CSQA_PROMPTS} vLLM prompts" "${OUT_G4}")"
check "--allow-any-task permits an unknown name" \
  "$("${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
       --benchmarks "some_other_task" --allow-any-task --dry-run >/dev/null 2>&1 && echo 1 || echo 0)"

echo
echo "======================================================================"
echo "CASE H: preflight, bootstrap, converter fallback, gpu-memory"
echo "======================================================================"
# Preflight must fire before ANY aws call when a dependency is missing.
: > "${AWSLOG}"
OUT_H="${SANDBOX}/h.log"
PATH="${SANDBOX}/emptybin:/usr/bin:/bin" "${SWEEP_CMD[@]}" \
  --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals --dry-run > "${OUT_H}" 2>&1
RC=$?
check "missing deps exit 3" "$([[ ${RC} -eq 3 ]] && echo 1 || echo 0)" "rc=${RC}"
check "preflight names what is missing" "$(has 'preflight failed; missing:' "${OUT_H}")"
check "preflight points at bootstrap.sh" "$(has 'scripts/bootstrap.sh' "${OUT_H}")"
check "preflight says no image is needed" "$(has 'needs no container image' "${OUT_H}")"
check "preflight spent nothing" "$([[ ! -s "${AWSLOG}" ]] && echo 1 || echo 0)"

# A failing import probe must also be caught, not just a missing binary.
: > "${AWSLOG}"
OUT_H2="${SANDBOX}/h2.log"
PREFLIGHT_IMPORT_RC=1 "${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" \
  --s3-out s3://bucket/evals --dry-run > "${OUT_H2}" 2>&1
RC=$?
check "unimportable olmo_eval exits 3" "$([[ ${RC} -eq 3 ]] && echo 1 || echo 0)" "rc=${RC}"
check "names the import problem" "$(has 'importable olmo_eval' "${OUT_H2}")"

# --bootstrap must run before the preflight it satisfies.
: > "${AWSLOG}"
OUT_H3="${SANDBOX}/h3.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --bootstrap --dry-run > "${OUT_H3}" 2>&1
check "--bootstrap runs and the sweep proceeds" "$(has 'DRY_RUN' "${OUT_H3}")"
check "--bootstrap announced" "$(has 'running bootstrap' "${OUT_H3}")"
check "bootstrap invoked uv sync" "$(grep -q '^uv sync' "${AWSLOG}" && echo 1 || echo 0)"

# Converter: bundled script used with OLMO_CORE_CONVERT unset. step2000 is native.
# Conversion exists only to feed vLLM, so these ask for that provider explicitly;
# the native provider reads step2000 as it stands and converts nothing.
: > "${AWSLOG}"
OUT_H4="${SANDBOX}/h4.log"
unset OLMO_CORE_CONVERT
"${SWEEP_VLLM[@]}" --checkpoint "${CKPTS}/step2000" --benchmarks hellaswag \
  --s3-out s3://bucket/evals > "${OUT_H4}" 2>&1
check "native checkpoint no longer fails for want of a converter" \
  "$(lacks 'no_converter' "${OUT_H4}")"
check "bundled converter was invoked" "$(has 'convert_to_hf.py' "${AWSLOG}")"
check "converter ran through uv run python" \
  "$(grep -q '^uv run python .*convert_to_hf.py' "${AWSLOG}" && echo 1 || echo 0)"
check "converted checkpoint then evaluated" \
  "$(grep -q 'olmo-eval run' "${AWSLOG}" && echo 1 || echo 0)"
check "checkpoint reached _READY" "$(has 's3://bucket/evals/step2000/_READY' "${AWSLOG}")"

# A broken override must still fail cleanly rather than silently skipping.
: > "${AWSLOG}"
OUT_H5="${SANDBOX}/h5.log"
OLMO_CORE_CONVERT=/nonexistent/conv.py "${SWEEP_VLLM[@]}" \
  --checkpoint "${CKPTS}/step2000" --benchmarks hellaswag \
  --s3-out s3://bucket/evals > "${OUT_H5}" 2>&1
check "missing override reports no_converter" "$(has 'no_converter' "${OUT_H5}")"
check "and marks _FAILED" "$(has 's3://bucket/evals/step2000/_FAILED' "${AWSLOG}")"

# A failing conversion is distinct from a missing converter.
: > "${AWSLOG}"
OUT_H6="${SANDBOX}/h6.log"
CONVERT_RC=1 "${SWEEP_VLLM[@]}" --checkpoint "${CKPTS}/step2000" --benchmarks hellaswag \
  --s3-out s3://bucket/evals > "${OUT_H6}" 2>&1
check "failing conversion reports conversion_failed" "$(has 'conversion_failed' "${OUT_H6}")"

# --gpu-memory-utilization must reach the harness group, before the first -t. Another
# vLLM engine setting, so another vllm_server case.
: > "${AWSLOG}"
OUT_H7="${SANDBOX}/h7.log"
"${SWEEP_VLLM[@]}" --checkpoint "${CKPTS}/step9" --benchmarks hellaswag \
  --s3-out s3://bucket/evals --gpu-memory-utilization 0.4 > "${OUT_H7}" 2>&1
check "gpu_memory_utilization passed via provider.kwargs" \
  "$(grep '^uv run olmo-eval' "${AWSLOG}" | grep -qF -- '-o provider.kwargs.gpu_memory_utilization=0.4' && echo 1 || echo 0)"
check "and precedes the first -t" \
  "$(grep '^uv run olmo-eval' "${AWSLOG}" | grep -qF -- 'gpu_memory_utilization=0.4 -t hellaswag' && echo 1 || echo 0)"
# Out-of-range values are rejected up front.
OUT_H8="${SANDBOX}/h8.log"
"${SWEEP_VLLM[@]}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
  --gpu-memory-utilization 2 --dry-run > "${OUT_H8}" 2>&1
RC=$?
check "gpu-memory-utilization > 1 rejected" "$([[ ${RC} -eq 2 ]] && echo 1 || echo 0)" "rc=${RC}"

if [[ "${HAS_PROVIDER}" == "1" ]]; then
  echo
  echo "======================================================================"
  echo "CASE H9: the native provider, on the copy that offers one"
  echo "======================================================================"
  # A native checkpoint the vLLM path would have had to convert. The point of this
  # provider is that it does not: conversion is architecture-gated, so a model it
  # refuses can still be scored here.
  : > "${AWSLOG}"
  OUT_H9="${SANDBOX}/h9.log"
  unset OLMO_CORE_CONVERT
  "${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step2000" --benchmarks "hellaswag naturalqs" \
    --s3-out s3://bucket/evals > "${OUT_H9}" 2>&1
  check "a native checkpoint is scored without conversion" \
    "$(lacks 'convert_to_hf.py' "${AWSLOG}")"
  check "and reaches _READY" "$(has 's3://bucket/evals/step2000/_READY' "${AWSLOG}")"
  check "no converter is needed, so none is complained about" \
    "$(lacks 'no_converter' "${OUT_H9}")"
  # A mixed request splits; a single-kind one must not, or every generative-only run
  # would pay a second model load for an empty half.
  check "a mixed request costs two loads" \
    "$([[ "$(grep -c '^uv run olmo-eval' "${AWSLOG}")" == "2" ]] && echo 1 || echo 0)" \
    "$(grep -c '^uv run olmo-eval' "${AWSLOG}") calls"
  : > "${AWSLOG}"
  "${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --benchmarks hellaswag \
    --s3-out s3://bucket/evals > "${SANDBOX}/h10.log" 2>&1
  check "a multiple-choice-only request costs one" \
    "$([[ "$(grep -c '^uv run olmo-eval' "${AWSLOG}")" == "1" ]] && echo 1 || echo 0)" \
    "$(grep -c '^uv run olmo-eval' "${AWSLOG}") calls"
  # Refused rather than forwarded: ProviderConfig.from_dict drops keys it does not
  # know, so a vLLM-only setting accepted here would look honoured and do nothing.
  : > "${AWSLOG}"
  "${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9" --benchmarks hellaswag \
    --s3-out s3://bucket/evals --tp 2 > "${SANDBOX}/h11.log" 2>&1
  RC=$?
  check "--tp is refused on the native path" "$([[ ${RC} -eq 2 ]] && echo 1 || echo 0)" "rc=${RC}"
  check "and names the provider that takes it" "$(has 'vllm_server' "${SANDBOX}/h11.log")"
  check "nothing spent on the refusal" "$([[ ! -s "${AWSLOG}" ]] && echo 1 || echo 0)"
fi

echo
echo "======================================================================"
echo "CASE I: the withdrawn --checkpoints still explains itself"
echo "======================================================================"
# This suite was written against the plural and every case failed opaquely when
# it was withdrawn. The replacement is only discoverable if the old spelling
# says so, so that message is now itself under test.
: > "${AWSLOG}"
OUT_I="${SANDBOX}/i.log"
"${SWEEP_CMD[@]}" --checkpoints "${CKPTS}/step9 ${CKPTS}/step10" \
  --s3-out s3://bucket/evals --dry-run > "${OUT_I}" 2>&1
RC=$?
check "exits 2" "$([[ ${RC} -eq 2 ]] && echo 1 || echo 0)" "rc=${RC}"
check "says the plural is gone" "$(has '--checkpoints was replaced by --checkpoint' "${OUT_I}")"
check "names the replacement for several checkpoints" "$(has '--checkpoint-root' "${OUT_I}")"
check "nothing spent" "$([[ ! -s "${AWSLOG}" ]] && echo 1 || echo 0)"
# A list smuggled through the singular must be caught too, rather than becoming
# one impossible path that fails much later.
OUT_I2="${SANDBOX}/i2.log"
"${SWEEP_CMD[@]}" --checkpoint "${CKPTS}/step9 ${CKPTS}/step10" \
  --s3-out s3://bucket/evals --dry-run > "${OUT_I2}" 2>&1
RC=$?
check "a space-separated --checkpoint exits 2" "$([[ ${RC} -eq 2 ]] && echo 1 || echo 0)" "rc=${RC}"
check "and says it takes exactly one" "$(has '--checkpoint takes exactly one checkpoint' "${OUT_I2}")"

echo
echo "======================================================================"
echo "CASE J: interim syncs, on the copy of the sweep that does them"
echo "======================================================================"
# A box can die mid-eval -- OOM, a spot reclaim, a dropped session -- and whatever
# it had already scored should still be in S3. eval-direct-gpu's sweep therefore
# uploads on a timer while the eval runs and marks the prefix _IN_PROGRESS while it
# does; eval-platform's does not, and this case reports itself as having nothing to
# do rather than pretending to cover it.
#
# Bounded on purpose: the eval is stalled for a few seconds and the interval set to
# one, rather than waiting out the three-minute default. Two seconds of stall would
# be enough for a single interim sync, so three buys a margin without buying a slow
# suite.
if [[ "${HAS_INTERIM_SYNC}" == "0" ]]; then
  echo "  n/a  ${SKILL_NAME}'s sweep has no --sync-interval; nothing to exercise"
else
  : > "${AWSLOG}"
  OUT_J="${SANDBOX}/j.log"
  EVAL_DELAY=3 bash "${SWEEP}" --checkpoint "${CKPTS}/step9" --benchmarks hellaswag \
    --s3-out s3://bucket/evals --sync-interval 1 > "${OUT_J}" 2>&1
  RC=$?
  check "exits 0" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"
  check "marked the prefix _IN_PROGRESS while the eval was still running" \
    "$(has 's3 cp - s3://bucket/evals/step9/_IN_PROGRESS' "${AWSLOG}")"
  # More than one upload to the checkpoint's prefix: the final one plus at least
  # one from the timer. A single upload would mean the loop never fired, which is
  # what a wrong comparison or a lost background job looks like.
  check "uploaded partial results before the checkpoint finished" \
    "$(grep -c 's3 sync .* s3://bucket/evals/step9/' "${AWSLOG}" | awk '{print ($1>=2)?1:0}')" \
    "$(grep -c 's3 sync .* s3://bucket/evals/step9/' "${AWSLOG}") uploads"
  # The interim marker must outlive the eval and be cleared only once a terminal
  # marker exists, or a poller sees a prefix with neither and waits forever.
  READY_LINE="$(grep -n '_READY' "${AWSLOG}" | head -1 | cut -d: -f1)"
  RM_LINE="$(grep -n 's3 rm .*_IN_PROGRESS' "${AWSLOG}" | head -1 | cut -d: -f1)"
  check "cleared _IN_PROGRESS only after writing a terminal marker" \
    "$([[ -n "${READY_LINE}" && -n "${RM_LINE}" && ${READY_LINE} -lt ${RM_LINE} ]] && echo 1 || echo 0)" \
    "_READY at line ${READY_LINE:-none}, removal at ${RM_LINE:-none}"

  # And the off switch has to actually switch it off, with the same stalled eval:
  # otherwise every other case in this file is quietly running the loop as well.
  : > "${AWSLOG}"
  OUT_J2="${SANDBOX}/j2.log"
  EVAL_DELAY=3 bash "${SWEEP}" --checkpoint "${CKPTS}/step9" --benchmarks hellaswag \
    --s3-out s3://bucket/evals --sync-interval 0 > "${OUT_J2}" 2>&1
  check "--sync-interval 0 writes no interim marker" \
    "$(lacks 's3 cp - s3://bucket/evals/step9/_IN_PROGRESS' "${AWSLOG}")"
  check "--sync-interval 0 uploads once, at the end" \
    "$(grep -c 's3 sync .* s3://bucket/evals/step9/' "${AWSLOG}" | awk '{print ($1==1)?1:0}')" \
    "$(grep -c 's3 sync .* s3://bucket/evals/step9/' "${AWSLOG}") uploads"
  # Zero is a meaningful value for this flag where it is not for --latest, so it
  # gets its own validation and its own message.
  OUT_J3="${SANDBOX}/j3.log"
  bash "${SWEEP}" --checkpoint "${CKPTS}/step9" --s3-out s3://bucket/evals \
    --sync-interval abc --dry-run > "${OUT_J3}" 2>&1
  RC=$?
  check "a non-numeric --sync-interval exits 2" "$([[ ${RC} -eq 2 ]] && echo 1 || echo 0)" "rc=${RC}"
  check "and says what it wanted" \
    "$(has '--sync-interval must be a whole number of seconds' "${OUT_J3}")"
fi

echo
echo "======================================================================"
echo "CASE K: the run left nothing behind outside the sandbox"
echo "======================================================================"
# A shim that treats an s3:// URI as a path runs mkdir -p on it, which is legal
# and quietly builds a directory literally named "s3:" wherever the shell was
# standing. Every other assertion in this file passed while that was happening.
# Compared against the pre-run snapshot so an older stray is reported without
# being blamed on this run.
for dir in "${LITTER_DIRS[@]}"; do
  label="${dir#"${REPO}/"}"
  if [[ "${PRE_LITTER}" == *" ${dir}"* ]]; then
    echo "  NOTE ${label}/s3: predates this run -- stray output from an older one, safe to delete"
    continue
  fi
  check "this run created no s3: tree under ${label}" \
    "$([[ ! -e "${dir}/s3:" ]] && echo 1 || echo 0)" "${dir}/s3: appeared"
done
check "the sandbox is outside the repo" \
  "$([[ "${SANDBOX}" != "${REPO}"/* ]] && echo 1 || echo 0)" "${SANDBOX}"

finish "ALL E2E CHECKS PASSED"
