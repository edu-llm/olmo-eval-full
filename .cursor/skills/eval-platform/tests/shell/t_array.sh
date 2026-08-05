#!/usr/bin/env bash
# The bash array idiom the submitter depends on, exercised in isolation.
#
# submit_eval_run.sh iterates its overrides as:
#
#     for ov in ${OVERRIDES+"${OVERRIDES[@]}"}; do
#
# which is doing three things at once, all of them load-bearing, and none of them
# obvious. `${A+...}` suppresses the expansion entirely when the array is empty,
# because under `set -u` a bare "${A[@]}" on an empty array is an unbound
# variable error in bash before 4.4 and the script must run on whatever bash the
# box has. The inner quotes keep an element containing a space as one word. And
# because the whole thing is quoted internally, no element is glob-expanded
# against the working directory.
#
# Any of those three regressing produces a wrong runner invocation rather than an
# error, so they are pinned here rather than left to the drivers that would only
# notice the symptom.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${HERE}/../lib.sh"

echo "bash version: ${BASH_VERSION}"

echo "--- A: empty array under set -u"
A=()
n=0
for ov in ${A+"${A[@]}"}; do n=$((n + 1)); echo "  [$n]=<$ov>"; done
check "empty array iterates zero times without tripping set -u" \
  "$([[ ${n} -eq 0 ]] && echo 1 || echo 0)" "count=${n}"

echo "--- B: element containing whitespace"
B=("has space=1")
m=0
for ov in ${B+"${B[@]}"}; do m=$((m + 1)); echo "  [$m]=<$ov>"; done
check "a value with a space stays one element" \
  "$([[ ${m} -eq 1 ]] && echo 1 || echo 0)" "count=${m}, expected 1"

echo "--- C: element containing a glob character, in a directory with files"
# cd somewhere non-empty first: globbing can only be observed where there is
# something for the pattern to match.
cd "${HERE}"
C=("foo=*")
k=0
kept=""
for ov in ${C+"${C[@]}"}; do k=$((k + 1)); kept="${ov}"; echo "  [$k]=<$ov>"; done
check "a value with a glob character is not expanded" \
  "$([[ ${k} -eq 1 && "${kept}" == 'foo=*' ]] && echo 1 || echo 0)" \
  "count=${k}, value=<${kept}>"

echo "--- D: several elements at once"
D=("has space=1" "foo=*")
j=0
for ov in ${D+"${D[@]}"}; do j=$((j + 1)); done
check "two awkward values stay two elements" \
  "$([[ ${j} -eq 2 ]] && echo 1 || echo 0)" "count=${j}, expected 2"

finish "ARRAY IDIOM HOLDS ON THIS BASH"
