#!/usr/bin/env bash
# Orchestrate the split: (1) one cheap CPU box downloads all weights and pushes a
# warm cache to S3, then self-terminates; (2) once the cache is READY, a GPU
# fleet pulls its shard and runs inference, each self-terminating on completion.
#
# SAFETY: defaults to DRY_RUN=1 (prints the AWS calls, spends nothing). Set
# DRY_RUN=0 to actually launch. All spend happens on GPU only during inference;
# the download hour is billed at the cheap CPU rate.
#
# Key env (with the account's known infra as defaults):
#   REGION        default us-east-1
#   AMI           default ami-0b6f2229ad14c9323   (DLAMI; has python3+aws+git)
#   SG            default sg-087218d8c87aa8576
#   PROFILE       default EswManagedInstance
#   SUBNET        default: first subnet in the SG's VPC (capacity permitting)
#   CPU_TYPE      default c7i.large
#   GPU_TYPE      default g6.xlarge
#   NUM_SHARDS    default 4
#   REPO_URL      default https://github.com/edu-llm/olmo-eval-full.git
#   BRANCH        default AdaptiveEvals
#   MODELS_YAML   roster path relative to the Inference dir (optional)
#   MAX_PARAMS_B  size filter (optional)
#   S3_CACHE      default s3://edullm-adaptive-inference-056956104102/hf-cache
#   S3_OUT        default s3://edullm-adaptive-inference-056956104102/split_infer
set -euo pipefail

REGION="${REGION:-us-east-1}"
AMI="${AMI:-ami-0b6f2229ad14c9323}"
SG="${SG:-sg-087218d8c87aa8576}"
PROFILE="${PROFILE:-EswManagedInstance}"
CPU_TYPE="${CPU_TYPE:-c7i.large}"
GPU_TYPE="${GPU_TYPE:-g6.xlarge}"
NUM_SHARDS="${NUM_SHARDS:-4}"
REPO_URL="${REPO_URL:-https://github.com/edu-llm/olmo-eval-full.git}"
BRANCH="${BRANCH:-AdaptiveEvals}"
S3_CACHE="${S3_CACHE:-s3://edullm-adaptive-inference-056956104102/hf-cache}"
S3_OUT="${S3_OUT:-s3://edullm-adaptive-inference-056956104102/split_infer}"
DRY_RUN="${DRY_RUN:-1}"

SDI_REL="AdaptiveTesting/Test/Inference/split_download_infer"

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN aws'; printf ' %q' "$@"; printf '\n'
  else
    aws "$@"
  fi
}

resolve_subnet() {
  [[ -n "${SUBNET:-}" ]] && { echo "${SUBNET}"; return; }
  local vpc
  vpc="$(aws ec2 describe-security-groups --region "${REGION}" --group-ids "${SG}" \
        --query 'SecurityGroups[0].VpcId' --output text)"
  aws ec2 describe-subnets --region "${REGION}" --filters "Name=vpc-id,Values=${vpc}" \
      --query 'Subnets[0].SubnetId' --output text
}

# --- user-data builders --------------------------------------------------------
cpu_userdata() {
  cat <<UD
#!/usr/bin/env bash
set -euo pipefail
export ROOT=/opt/adaptive-dl
git clone --depth 1 --branch ${BRANCH} ${REPO_URL} \${ROOT}/code
cd \${ROOT}/code/${SDI_REL}
ROOT=\${ROOT} REGION=${REGION} bash bootstrap_downloader.sh
ROOT=\${ROOT} REGION=${REGION} S3_CACHE=${S3_CACHE} \
  ${MODELS_YAML:+MODELS_YAML=\${ROOT}/code/AdaptiveTesting/Test/Inference/${MODELS_YAML}} \
  ${MAX_PARAMS_B:+MAX_PARAMS_B=${MAX_PARAMS_B}} \
  bash sync_cache_to_s3.sh
shutdown -h now
UD
}

gpu_userdata() {
  local shard="$1"
  cat <<UD
#!/usr/bin/env bash
set -euo pipefail
export ROOT=/opt/dlami/nvme/adaptive-infer
git clone --depth 1 --branch ${BRANCH} ${REPO_URL} \${ROOT}/code
cd \${ROOT}/code/AdaptiveTesting/Test/Inference
ROOT=\${ROOT} bash aws/node_bootstrap.sh || true
cd ${SDI_REL##*/}
ROOT=\${ROOT} REGION=${REGION} S3_CACHE=${S3_CACHE} S3_OUT=${S3_OUT} \
  SHARD_INDEX=${shard} NUM_SHARDS=${NUM_SHARDS} \
  ${MODELS_YAML:+MODELS_YAML=\${ROOT}/code/AdaptiveTesting/Test/Inference/${MODELS_YAML}} \
  ${MAX_PARAMS_B:+MAX_PARAMS_B=${MAX_PARAMS_B}} \
  bash run_gpu_infer.sh
shutdown -h now
UD
}

launch() {
  local type="$1" name="$2" ud_b64="$3"
  run ec2 run-instances --region "${REGION}" --image-id "${AMI}" \
    --instance-type "${type}" --subnet-id "${SUBNET}" \
    --security-group-ids "${SG}" --iam-instance-profile "Name=${PROFILE}" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "${ud_b64}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${name}}]" \
    --count 1 --output text --query 'Instances[0].InstanceId'
}

wait_ready() {
  echo "waiting for ${S3_CACHE}/_READY ..."
  for _ in $(seq 1 240); do   # up to ~2h
    if aws s3 ls "${S3_CACHE}/_READY" --region "${REGION}" >/dev/null 2>&1; then
      echo "cache READY"; return 0
    fi
    sleep 30
  done
  echo "timed out waiting for cache READY" >&2; return 1
}

# --- main ----------------------------------------------------------------------
SUBNET="$(resolve_subnet)"
echo "region=${REGION} ami=${AMI} sg=${SG} subnet=${SUBNET} profile=${PROFILE}"
echo "cpu=${CPU_TYPE} gpu=${GPU_TYPE} shards=${NUM_SHARDS} dry_run=${DRY_RUN}"

echo "== step 1: CPU downloader =="
launch "${CPU_TYPE}" "split-downloader" "$(cpu_userdata | base64 | tr -d '\n')"

if [[ "${DRY_RUN}" != "1" ]]; then
  wait_ready
fi

echo "== step 2: GPU inference fleet (${NUM_SHARDS} shards) =="
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  launch "${GPU_TYPE}" "split-infer-${i}" "$(gpu_userdata "${i}" | base64 | tr -d '\n')"
done

echo "done (${DRY_RUN:+DRY_RUN=}${DRY_RUN})"
