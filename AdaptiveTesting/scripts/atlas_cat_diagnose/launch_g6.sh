#!/usr/bin/env bash
# Launch one g6.xlarge (1× L4) to run ATLAS CAT on a training checkpoint.
#
# Usage:
#   bash AdaptiveTesting/scripts/atlas_cat_diagnose/launch_g6.sh \
#     --checkpoint s3://BUCKET/checkpoints/EXP/STEP \
#     --run-id EXP-stepSTEP
#
# Safety: DRY_RUN=1 prints the aws call and exits 0 without spending.
# Defaults match AdaptiveTesting/Test/Inference/split_download_infer infra.
set -euo pipefail

CHECKPOINT=""
RUN_ID=""
SE_STOP="0.3"
MAX_ITEMS="40"
MIN_ITEMS="8"
BRANCH="${BRANCH:-AdaptiveEvals}"
REPO_URL="${REPO_URL:-https://github.com/edu-llm/olmo-eval-full.git}"
REGION="${REGION:-us-east-1}"
AMI="${AMI:-ami-0b6f2229ad14c9323}"
SG="${SG:-sg-087218d8c87aa8576}"
PROFILE="${PROFILE:-EswManagedInstance}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.xlarge}"
# Default lands under smoke/ because the EswManagedInstance role only has
# s3:PutObject on smoke/*, smoke_split/*, full200/*. Point elsewhere only if the
# worker role can write there.
S3_OUT_ROOT="${S3_OUT_ROOT:-s3://edullm-adaptive-inference-056956104102/smoke/atlas_cat}"
DRY_RUN="${DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --se-stop) SE_STOP="$2"; shift 2 ;;
    --max-items) MAX_ITEMS="$2"; shift 2 ;;
    --min-items) MIN_ITEMS="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${CHECKPOINT}" ]] || { echo "--checkpoint required" >&2; exit 2; }
[[ -n "${RUN_ID}" ]] || { echo "--run-id required" >&2; exit 2; }
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--run-id must match [A-Za-z0-9._-]+ (got ${RUN_ID})" >&2
  exit 2
fi

resolve_subnet() {
  if [[ -n "${SUBNET:-}" ]]; then echo "${SUBNET}"; return; fi
  local vpc
  vpc="$(aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG}" \
        --query 'SecurityGroups[0].VpcId' --output text)"
  aws ec2 describe-subnets --region "${REGION}" \
      --filters "Name=vpc-id,Values=${vpc}" "Name=availability-zone,Values=us-east-1a" \
      --query 'Subnets[0].SubnetId' --output text
}

# Prefer an explicit SUBNET=; else try these VPC subnets (capacity often uneven by AZ).
SUBNET_CANDIDATES=(
  "${SUBNET:-}"
  subnet-0bbe2b7870da13713  # us-east-1a
  subnet-0a4235fb98b63930f  # us-east-1b
  subnet-0fd5ed8accae254dc  # us-east-1c
  subnet-08792525c62ba31c0  # us-east-1d
  subnet-01f4bf9a051404a37  # us-east-1f
)
# Also try g5.xlarge (A10G) if g6 is capacity-starved region-wide.
TYPE_CANDIDATES=("${INSTANCE_TYPE}")
if [[ "${INSTANCE_TYPE}" == "g6.xlarge" ]]; then
  TYPE_CANDIDATES+=("g5.xlarge")
fi

NAME="atlas-cat-${RUN_ID}"
S3_DEST="${S3_OUT_ROOT%/}/${RUN_ID}"

echo "region=${REGION} ami=${AMI} types=${TYPE_CANDIDATES[*]}"
echo "sg=${SG} profile=${PROFILE}"
echo "checkpoint=${CHECKPOINT}"
echo "results → ${S3_DEST}/"
echo "dry_run=${DRY_RUN}"

UD_FILE="$(mktemp)"
trap 'rm -f "${UD_FILE}"' EXIT
cat >"${UD_FILE}" <<UD
#!/usr/bin/env bash
set -euo pipefail
export HOME="\${HOME:-/root}"   # cloud-init runs with no HOME; needed for uv + PATH
ROOT=/opt/dlami/nvme/atlas-cat
mkdir -p "\${ROOT}"
git clone --depth 1 --branch ${BRANCH} ${REPO_URL} \${ROOT}/code
chmod +x \${ROOT}/code/AdaptiveTesting/scripts/atlas_cat_diagnose/run_worker.sh
export CHECKPOINT=$(printf '%q' "${CHECKPOINT}")
export RUN_ID=$(printf '%q' "${RUN_ID}")
export S3_OUT_ROOT=$(printf '%q' "${S3_OUT_ROOT}")
export SE_STOP=$(printf '%q' "${SE_STOP}")
export MAX_ITEMS=$(printf '%q' "${MAX_ITEMS}")
export MIN_ITEMS=$(printf '%q' "${MIN_ITEMS}")
export REPO_ROOT=\${ROOT}/code
export REGION=${REGION}
curl -LsSf https://astral.sh/uv/install.sh | sh || true
export PATH="\$HOME/.local/bin:\$PATH"
bash \${ROOT}/code/AdaptiveTesting/scripts/atlas_cat_diagnose/run_worker.sh
shutdown -h now
UD

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN would try types=${TYPE_CANDIDATES[*]} across VPC subnets"
  echo "would write results to ${S3_DEST}/"
  # Keep a representative command printable for debugging.
  printf 'DRY_RUN aws ec2 run-instances --instance-type %q --subnet-id %q --user-data file://%q ...\n' \
    "${TYPE_CANDIDATES[0]}" "${SUBNET_CANDIDATES[1]}" "${UD_FILE}"
  exit 0
fi

IID=""
LAST_ERR=""
for typ in "${TYPE_CANDIDATES[@]}"; do
  for subnet in "${SUBNET_CANDIDATES[@]}"; do
    [[ -n "${subnet}" ]] || continue
    echo "trying type=${typ} subnet=${subnet} ..."
    if IID="$(aws ec2 run-instances \
      --region "${REGION}" \
      --image-id "${AMI}" \
      --instance-type "${typ}" \
      --subnet-id "${subnet}" \
      --security-group-ids "${SG}" \
      --iam-instance-profile "Name=${PROFILE}" \
      --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
      --user-data "file://${UD_FILE}" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}},{Key=Project,Value=atlas-cat-diagnose},{Key=RunId,Value=${RUN_ID}}]" \
      --count 1 \
      --output text \
      --query "Instances[0].InstanceId" 2>/tmp/atlas_cat_launch.err)"; then
      echo "launched ${IID} (${typ} / ${subnet})"
      echo "watch: aws ec2 describe-instances --instance-ids ${IID} --region ${REGION} --query Reservations[0].Instances[0].State.Name"
      echo "results (when _READY appears): aws s3 ls ${S3_DEST}/"
      exit 0
    fi
    LAST_ERR="$(cat /tmp/atlas_cat_launch.err 2>/dev/null || true)"
    echo "  failed: ${LAST_ERR}" | head -c 400; echo
  done
done

echo "exhausted type/subnet candidates" >&2
echo "${LAST_ERR}" >&2
exit 1
