#!/usr/bin/env bash
# Every shell script the skill ships must parse, and its --help must work.
#
# Cheap, but not redundant: the other drivers only reach the code paths they
# exercise, so a syntax error in a branch none of them takes -- an unclosed
# heredoc in the bootstrap path, say -- survives a fully green suite and then
# fails on the box where it matters. `bash -n` reads the whole file.
#
# --help is checked alongside it because both scripts derive their help text by
# scanning their own header comments rather than storing it, so a header edit can
# quietly produce empty output.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${HERE}/../lib.sh"

echo "bash: ${BASH_VERSION}"
echo
echo "=== every shipped script parses ==="
for script in "${SKILL_DIR}"/scripts/*.sh; do
  name="$(basename "${script}")"
  err="$(bash -n "${script}" 2>&1)"
  check "${name} parses" "$([[ -z "${err}" ]] && echo 1 || echo 0)" "${err}"
done

echo
echo "=== ...including this suite's own drivers ==="
for script in "${HERE}"/*.sh "${HERE}/../lib.sh"; do
  name="$(basename "${script}")"
  err="$(bash -n "${script}" 2>&1)"
  check "${name} parses" "$([[ -z "${err}" ]] && echo 1 || echo 0)" "${err}"
done

echo
echo "=== --help produces its header ==="
for script in "${SUBMIT}" "${SWEEP}"; do
  name="$(basename "${script}")"
  out="$(bash "${script}" --help 2>&1)"
  rc=$?
  check "${name} --help exits 0" "$([[ ${rc} -eq 0 ]] && echo 1 || echo 0)" "rc=${rc}"
  check "${name} --help prints an INPUTS block" \
    "$(printf '%s' "${out}" | grep -q 'INPUTS' && echo 1 || echo 0)" \
    "$(printf '%s' "${out}" | wc -l) lines"
done

echo
echo "=== a run with no arguments at all is refused, not defaulted ==="
# Both scripts require inputs that cannot be guessed. Exiting 0 here would mean
# something was silently invented.
for script in "${SUBMIT}" "${SWEEP}"; do
  name="$(basename "${script}")"
  bash "${script}" >/dev/null 2>&1
  rc=$?
  check "${name} with no arguments exits 2" \
    "$([[ ${rc} -eq 2 ]] && echo 1 || echo 0)" "rc=${rc}"
done

finish "ALL SHELL SCRIPTS PARSE AND SELF-DOCUMENT"
