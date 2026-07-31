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
S3_OUT_ROOT="${S3_OUT_ROOT:-s3://edullm-adaptive-inference-056956104102/atlas_cat}"
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

# Default subnet used by AdaptiveTesting GPU jobs in us-east-1a (override with SUBNET=).
SUBNET="${SUBNET:-subnet-0bbe2b7870da13713}"
if [[ "${DRY_RUN}" != "1" && "${SUBNET}" == "subnet-0bbe2b7870da13713" ]]; then
  # Refresh from API when spending money, in case infra moved.
  SUBNET="$(resolve_subnet)"
fi
NAME="atlas-cat-${RUN_ID}"
S3_DEST="${S3_OUT_ROOT%/}/${RUN_ID}"

echo "region=${REGION} type=${INSTANCE_TYPE} ami=${AMI}"
echo "sg=${SG} subnet=${SUBNET} profile=${PROFILE}"
echo "checkpoint=${CHECKPOINT}"
echo "results → ${S3_DEST}/"
echo "dry_run=${DRY_RUN}"

UD_FILE="$(mktemp)"
trap 'rm -f "${UD_FILE}"' EXIT
cat >"${UD_FILE}" <<UD
#!/usr/bin/env bash
set -euo pipefail
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

ARGS=(
  ec2 run-instances
  --region "${REGION}"
  --image-id "${AMI}"
  --instance-type "${INSTANCE_TYPE}"
  --subnet-id "${SUBNET}"
  --security-group-ids "${SG}"
  --iam-instance-profile "Name=${PROFILE}"
  --instance-initiated-shutdown-behavior terminate
  --user-data "file://${UD_FILE}"
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}},{Key=Project,Value=atlas-cat-diagnose},{Key=RunId,Value=${RUN_ID}}]"
  --count 1
  --output text
  --query "Instances[0].InstanceId"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN aws'; printf ' %q' "${ARGS[@]}"; printf '\n'
  echo "would write results to ${S3_DEST}/"
  exit 0
fi

IID="$(aws "${ARGS[@]}")"
echo "launched ${IID}"
echo "watch: aws ec2 describe-instances --instance-ids ${IID} --region ${REGION} --query Reservations[0].Instances[0].State.Name"
echo "results (when _READY appears): aws s3 ls ${S3_DEST}/"
