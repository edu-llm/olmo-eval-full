#!/usr/bin/env bash
# --batch-size and --override validation. Every case here must be refused with
# exit 2 and a message naming the offending value.
#
# These are checked before dispatch on purpose: both end up inside the single
# quoted payload the platform runs, so a bad value is not rejected by anything
# downstream -- it becomes a malformed shell line inside a job that has already
# been approved and placed.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${HERE}/../lib.sh"

ERRF="$(mktemp)"
BASE=("${SUBMIT_BASE[@]}" --group smoke)

refuses() {
  local label="$1" expect="$2"; shift 2
  bash "${SUBMIT}" "${BASE[@]}" "$@" >/dev/null 2>"${ERRF}"
  local rc=$?
  check "${label} exits 2" "$([[ ${rc} -eq 2 ]] && echo 1 || echo 0)" "rc=${rc}"
  check "${label} says why" "$(has "${expect}" "${ERRF}")" "$(head -1 "${ERRF}")"
}

echo "=== --batch-size must be a positive integer ==="
refuses "--batch-size 0"    "--batch-size must be a positive integer" --batch-size 0
refuses "--batch-size -4"   "--batch-size must be a positive integer" --batch-size -4
refuses "--batch-size abc"  "--batch-size must be a positive integer" --batch-size abc
refuses "--batch-size 3.5"  "--batch-size must be a positive integer" --batch-size 3.5

echo
echo "=== --override must be a quote-free, whitespace-free KEY=VALUE ==="
refuses "--override novalue"      "--override wants KEY=VALUE"  --override novalue
refuses "--override 'has space=1'" \
  "--override cannot contain quotes or whitespace" --override "has space=1"
refuses "--override \"quo'te=1\"" \
  "--override cannot contain quotes or whitespace" --override "quo'te=1"

echo
echo "=== flags from the other path are refused with directions, not 'unknown arg' ==="
# Both appear in real requests, and both are absent here on purpose rather than by
# oversight. A bare "unknown arg" sends someone to --help to hunt for a flag that
# was never going to be there, so each names what to do instead. The assertions
# are on the direction given, not on the refusal, which the catch-all would also
# manage.
refuses "--bootstrap" "belongs to run_eval_sweep.sh" --bootstrap
check "--bootstrap explains why the container needs no bootstrap" \
  "$(has 'installs git, olmo-eval and' "${ERRF}")" "$(head -1 "${ERRF}")"
check "--bootstrap names the sweep's script for a local box" \
  "$(has 'scripts/bootstrap.sh' "${ERRF}")" "$(head -1 "${ERRF}")"

refuses "--checkpoints" "this script evaluates exactly one" --checkpoints "a b"
check "--checkpoints says how to do several here" \
  "$(has 'once per checkpoint' "${ERRF}")" "$(head -1 "${ERRF}")"
check "--checkpoints does not offer --checkpoint-root as if it existed here" \
  "$(has 'only on run_eval_sweep.sh' "${ERRF}")" "$(head -1 "${ERRF}")"

echo
echo "=== ...and a well-formed one is accepted ==="
bash "${SUBMIT}" "${BASE[@]}" --batch-size 64 --override foo.bar=1 >/dev/null 2>"${ERRF}"
RC=$?
check "a valid pair exits 0" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"

finish "ALL VALIDATION CHECKS PASSED"
