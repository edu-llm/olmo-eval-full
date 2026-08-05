# Shared setup for the shell half of the suite. Source it, do not run it.
#
# Resolves the repo the same way the scripts under test do -- $OLMO_EVAL_ROOT
# with a fixed relative fallback -- so a test and the script it drives can never
# disagree about which checkout is being exercised. It also decides which skill's
# scripts a driver is pointed at, since two of them ship a run_eval_sweep.sh.
#
# Every driver here runs the submitter or the sweep with --dry-run, or against
# shims. Nothing in this suite may reach AWS.

# BASH_SOURCE[0] is this file even when sourced, so the suite root is fixed
# regardless of which driver sourced it or what its working directory is.
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_DIR="$(cd "${TESTS_DIR}/../.." && pwd)"

# Which skill's scripts are under test. The default is the skill this suite lives
# in and has to stay that way: most drivers reach for submit_eval_run.sh or
# resolve_benchmarks.py, and only eval-platform ships either. $EVALCK_SKILL names
# a sibling instead, which is how e2e.sh reaches the second copy of
# run_eval_sweep.sh -- setting it is a driver's assertion that it asks only for
# scripts that skill actually has.
SKILL_NAME="${EVALCK_SKILL:-$(basename "$(dirname "${TESTS_DIR}")")}"
if [[ ! -d "${SKILLS_DIR}/${SKILL_NAME}/scripts" ]]; then
  echo "no skill '${SKILL_NAME}' under ${SKILLS_DIR} -- check \$EVALCK_SKILL" >&2
  exit 2
fi
SKILL_DIR="$(cd "${SKILLS_DIR}/${SKILL_NAME}" && pwd)"

if [[ -n "${OLMO_EVAL_ROOT:-}" ]]; then
  REPO="${OLMO_EVAL_ROOT}"
  # A root exported by a Windows caller arrives as C:\... or C:/..., neither of
  # which is a path this shell can cd into. cygpath is present in both Git Bash
  # and MSYS2; elsewhere the value is already a POSIX path.
  if command -v cygpath >/dev/null 2>&1; then
    REPO="$(cygpath -u "${REPO}")"
  fi
  REPO="${REPO%/}"
else
  REPO="$(cd "${SKILL_DIR}/../../.." && pwd)"
fi
export OLMO_EVAL_ROOT="${REPO}"

# run_eval_sweep.sh and benchmarks.json exist in both skills; the other two are
# eval-platform's alone. They are still named unconditionally, because a driver
# that wants one should say so and get a missing-file error naming the path,
# rather than an empty variable that fails somewhere less obvious.
SUBMIT="${SKILL_DIR}/scripts/submit_eval_run.sh"
SWEEP="${SKILL_DIR}/scripts/run_eval_sweep.sh"
RESOLVER="${SKILL_DIR}/scripts/resolve_benchmarks.py"
REGISTRY="${SKILL_DIR}/scripts/benchmarks.json"

# A checkpoint URI under the outputs bucket, which the submitter requires. It
# names nothing real; every driver that uses it passes --dry-run.
PROBE_CKPT="s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/r-1/checkpoints/step1000"
SUBMIT_BASE=(--checkpoint "${PROBE_CKPT}"
             --team eval-inference --experiment probe --wandb-project p --dry-run)

# Scratch space. Kept outside the repo so a run leaves no untracked files
# behind; $EVALCK_SANDBOX overrides it when you want to inspect the wreckage.
sandbox_dir() {
  if [[ -n "${EVALCK_SANDBOX:-}" ]]; then
    rm -rf "${EVALCK_SANDBOX}"
    mkdir -p "${EVALCK_SANDBOX}"
    echo "${EVALCK_SANDBOX}"
  else
    mktemp -d
  fi
}

FAILURES=0
check() {
  # check <label> <0|1> [detail]
  if [[ "$2" == "1" ]]; then
    echo "  OK   $1"
  else
    echo "  FAIL $1  [${3:-}]"
    FAILURES=$((FAILURES + 1))
  fi
}
bool() { if "$@" >/dev/null 2>&1; then echo 1; else echo 0; fi; }
# `--` so a pattern starting with "--" is not read as a grep option.
has() { if grep -qF -- "$1" "$2"; then echo 1; else echo 0; fi; }
lacks() { if grep -qF -- "$1" "$2"; then echo 0; else echo 1; fi; }

finish() {
  echo
  if [[ ${FAILURES} -eq 0 ]]; then
    echo "${1:-ALL CHECKS PASSED}"
    exit 0
  fi
  echo "${FAILURES} CHECK(S) FAILED"
  exit 1
}
