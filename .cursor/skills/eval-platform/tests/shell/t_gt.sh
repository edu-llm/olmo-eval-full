#!/usr/bin/env bash
# What `[[ X -gt 1000 ]]` does to a non-numeric X under `set -euo pipefail`.
#
# submit_eval_run.sh guards its batch-size warning with:
#
#     if [[ -z "${BATCH_SIZE}" ]] && [[ "${EST_PROMPTS}" -gt 1000 ]]; then
#
# EST_PROMPTS comes from the resolver, and is empty whenever the estimate covers
# nothing -- every requested task was unknown and let through by
# --allow-any-task. So the empty case is reachable in normal use and must not be
# fatal.
#
# The results below are not uniform, which is the point. bash evaluates the
# operand as an arithmetic expression, so a bare word is treated as a *variable
# name*: with `set -u` an unset one aborts the shell. Anything that merely fails
# to parse as a number is a non-fatal error and compares as false. The guard is
# therefore safe for the value it actually sees and unsafe for a value it must
# never be given, which is worth having written down.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${HERE}/../lib.sh"

# Each case runs in a subshell so a fatal one does not take the test with it.
outcome() {
  ( set -euo pipefail
    E="$1"
    if [[ "${E}" -gt 1000 ]]; then echo TRUE; else echo FALSE; fi
  ) 2>/dev/null || echo FATAL
}

expect() {
  local value="$1" want="$2" why="$3"
  local got
  got="$(outcome "${value}")"
  check "<${value}> -> ${want}  (${why})" \
    "$([[ "${got}" == "${want}" ]] && echo 1 || echo 0)" "got ${got}"
}

echo "=== the value the guard actually receives ==="
expect ""       FALSE "empty estimate: the --allow-any-task path, must not be fatal"
expect "37040"  TRUE  "a real estimate over the threshold"
expect "36"     FALSE "a real estimate under it (a smoke run)"

echo
echo "=== values it must never receive ==="
expect "abc"    FATAL "a bare word is read as a variable name, and set -u aborts"
expect "1,234"  FALSE "a thousands separator silently truncates to the last group"
expect "3.5"    FALSE "no decimals in shell arithmetic"
expect "1e3"    FALSE "no scientific notation either"
expect "12 34"  FALSE "an embedded space is a parse error, not two numbers"

finish "ARITHMETIC GUARD BEHAVES AS THE SUBMITTER ASSUMES"
