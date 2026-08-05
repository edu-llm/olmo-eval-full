#!/usr/bin/env bash
# The missing-batch-size warning: it must fire on a run big enough to run out of
# memory, stay quiet otherwise, and never become an error.
#
# It goes to stderr rather than stdout because stdout carries the `command:` line
# that other tests parse; a warning mixed into it would be read as part of the
# command. And it is a warning, not a refusal: a large run without a batch size
# is usually a mistake, but the submitter is not in a position to be sure.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${HERE}/../lib.sh"

OUTF="$(mktemp)"; ERRF="$(mktemp)"
WARN="WARNING: no --batch-size"

probe() {
  local label="$1"; shift
  bash "${SUBMIT}" "${SUBMIT_BASE[@]}" "$@" >"${OUTF}" 2>"${ERRF}"
  RC=$?
  echo "### ${label}"
  echo "    rc=${RC}   cost line: $(grep -o 'cost shape:.*' "${OUTF}" || echo NONE)"
}

probe "--group default, no --batch-size" --group default
check "the big run still succeeds" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"
check "it warns" "$(has "${WARN}" "${ERRF}")"
check "the warning goes to stderr, not stdout" "$(lacks "${WARN}" "${OUTF}")"

probe "--group default WITH --batch-size 64" --group default --batch-size 64
check "no warning once a batch size is given" "$(lacks "${WARN}" "${ERRF}")"
check "still exits 0" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"

probe "--group smoke, no --batch-size" --group smoke
check "no warning on a 36-prompt smoke run" "$(lacks "${WARN}" "${ERRF}")"
check "still exits 0" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"

echo
echo "=== the forgotten-selection warning ==="
# Omitting both selection flags resolves the full default set, which is the same
# command line as deliberately asking for it. The distinguishing signal is that
# nobody answered the question, so silence is what fires this and an explicit
# --group default does not. Without it the only clue that a smoke run was meant
# and not delivered is the batch-size warning, which is about memory and fires on
# a real sweep too.
NOSEL="THIS IS NOT A SMOKE TEST"
probe "neither --group nor --benchmarks"
check "it says the full default set is running" "$(has "${NOSEL}" "${ERRF}")"
check "it names the flag that was meant" "$(has "--group smoke" "${ERRF}")"
check "the warning goes to stderr, not stdout" "$(lacks "${NOSEL}" "${OUTF}")"
check "it is a warning, not a refusal" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"

probe "--group default, which answers the question" --group default
check "no warning once the group is named" "$(lacks "${NOSEL}" "${ERRF}")"

probe "--benchmarks answers it too" --benchmarks "csqa"
check "no warning on an explicit list" "$(lacks "${NOSEL}" "${ERRF}")"

echo
echo "=== the empty-estimate path ==="
# An unknown task under --allow-any-task contributes nothing to the estimate, so
# the resolver's `prompts` field comes back empty and the guard becomes
# [[ "" -gt 1000 ]]. bash reads that as 0 rather than failing, which is the only
# reason `set -u` does not kill the run here -- see t_gt.sh for the values that
# would. Worth pinning: this is a plausible arithmetic crash on a code path that
# only runs for people already doing something unusual.
probe "unknown task under --allow-any-task" --benchmarks nosuchtask --allow-any-task
check "no arithmetic or unbound-variable error" \
  "$(grep -qE 'syntax error|unbound variable' "${ERRF}" && echo 0 || echo 1)" \
  "$(grep -E 'syntax error|unbound variable' "${ERRF}" | head -1)"
check "exits 0" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"

RAW="$(python3 "${RESOLVER}" --registry "${REGISTRY}" \
        --benchmarks nosuchtask --allow-any-task --field prompts | tr -d '\r')"
echo "    resolver prompts field: <${RAW}>"
check "the estimate really is empty, so the guard really is exercised" \
  "$([[ -z "${RAW}" || "${RAW}" == "0" ]] && echo 1 || echo 0)" "<${RAW}>"

finish "ALL WARNING CHECKS PASSED"
