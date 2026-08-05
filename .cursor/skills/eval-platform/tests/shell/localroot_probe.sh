#!/usr/bin/env bash
# Does --checkpoint-root accept a local directory, as its header claims?
#
# The header says "A local directory works too; children are found with find."
# That branch exists only because without it a local path would be handed to
# `aws s3 ls`, which reports no prefixes for it and leaves the sweep failing with
# "no checkpoints found" -- a message that points at the wrong thing entirely.
# It is also the branch every other test in this suite relies on, so it is worth
# testing directly rather than only through them.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${HERE}/../lib.sh"

SB="$(sandbox_dir)"
mkdir -p "${SB}/ckpts/step100" "${SB}/ckpts/step200" "${SB}/shim"
for d in step100 step200; do
  echo '{}' > "${SB}/ckpts/${d}/config.json"
  : > "${SB}/ckpts/${d}/model.safetensors"
done

# Shims: the preflight must pass, and nothing real may be invoked. Any aws call
# is recorded so a dry run that reaches for the network is visible rather than
# merely slow.
AWSLOG="${SB}/aws_calls.log"
: > "${AWSLOG}"
cat > "${SB}/shim/aws" <<SH
#!/usr/bin/env bash
echo "aws \$*" >> "${AWSLOG}"
exit 0
SH
cat > "${SB}/shim/uv" <<'SH'
#!/usr/bin/env bash
exit 0
SH
cat > "${SB}/shim/python3" <<'SH'
#!/usr/bin/env bash
exec python "$@"
SH
chmod +x "${SB}/shim"/*
export PATH="${SB}/shim:${PATH}"

OUT="${SB}/root.log"
echo "=== --checkpoint-root pointed at a LOCAL directory ==="
bash "${SWEEP}" --checkpoint-root "${SB}/ckpts" --s3-out s3://b/out --dry-run \
  > "${OUT}" 2>&1
RC=$?
sed 's/^/    /' "${OUT}"
check "exits 0" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"
check "does not mistake a local path for an S3 prefix" \
  "$(lacks 'no checkpoints found' "${OUT}")"
check "discovers both children" "$(has 'checkpoints (2)' "${OUT}")"
check "names step100" "$(has 'step100' "${OUT}")"
check "names step200" "$(has 'step200' "${OUT}")"
check "orders them by step, not lexically" \
  "$(awk '/step100/{a=NR} /step200/{b=NR} END{print (a && b && a<b) ? 1 : 0}' "${OUT}")"
check "spends nothing on a dry run" \
  "$([[ ! -s "${AWSLOG}" ]] && echo 1 || echo 0)" "$(head -1 "${AWSLOG}")"

echo
echo "=== for contrast: one of the same checkpoints via --checkpoint ==="
OUT1="${SB}/one.log"
bash "${SWEEP}" --checkpoint "${SB}/ckpts/step100" --s3-out s3://b/out --dry-run \
  > "${OUT1}" 2>&1
RC=$?
sed 's/^/    /' "${OUT1}"
check "the singular form exits 0" "$([[ ${RC} -eq 0 ]] && echo 1 || echo 0)" "rc=${RC}"
check "and takes exactly the one it was given" "$(has 'checkpoints (1)' "${OUT1}")"
check "which is step100" "$(has 'step100' "${OUT1}")"
check "not step200" "$(lacks 'step200' "${OUT1}")"

echo
echo "=== a root that does not exist says so ==="
OUT2="${SB}/missing.log"
bash "${SWEEP}" --checkpoint-root "${SB}/nope" --s3-out s3://b/out --dry-run \
  > "${OUT2}" 2>&1
RC=$?
check "exits nonzero" "$([[ ${RC} -ne 0 ]] && echo 1 || echo 0)" "rc=${RC}"
check "and names the directory rather than blaming S3" \
  "$(has 'neither an s3:// URI nor an existing directory' "${OUT2}")" \
  "$(head -2 "${OUT2}" | tr '\n' '|')"

finish "LOCAL --checkpoint-root BEHAVES AS DOCUMENTED"
