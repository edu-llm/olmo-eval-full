#!/usr/bin/env bash
# Store HF_TOKEN from a local .env into AWS Secrets Manager.
#
# This uses YOUR configured AWS CLI credentials (aws configure / SSO / role).
# The token value is never printed. On success it prints the secret ARN to
# paste into aws/batch_job_def.json (containerProperties.secrets[].valueFrom).
#
# Usage:
#   ./aws/put_hf_secret.sh [--env PATH] [--name SECRET_NAME] [--region REGION]
#
# Defaults:
#   --env    ../../../.env         (repo root .env)
#   --name   hf-token
set -euo pipefail

ENV_FILE="$(cd "$(dirname "$0")/../../.." && pwd)/.env"
SECRET_NAME="hf-token"
REGION_ARG=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_FILE="$2"; shift 2 ;;
    --name) SECRET_NAME="$2"; shift 2 ;;
    --region) REGION_ARG=(--region "$2"); shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "error: env file not found: ${ENV_FILE}" >&2
  exit 1
fi

# Extract HF_TOKEN=... (strip optional quotes); do not export the whole file.
HF_TOKEN="$(grep -E '^HF_TOKEN=' "${ENV_FILE}" | tail -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: HF_TOKEN not found in ${ENV_FILE}" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "error: aws CLI not installed / not on PATH" >&2
  exit 1
fi

# Create the secret, or update it if it already exists. Token passed via stdin
# to avoid it appearing in process args / shell history.
if aws secretsmanager describe-secret "${REGION_ARG[@]}" --secret-id "${SECRET_NAME}" >/dev/null 2>&1; then
  echo "secret '${SECRET_NAME}' exists -> updating value"
  printf '%s' "${HF_TOKEN}" | aws secretsmanager put-secret-value "${REGION_ARG[@]}" \
    --secret-id "${SECRET_NAME}" --secret-string file:///dev/stdin >/dev/null
else
  echo "creating secret '${SECRET_NAME}'"
  printf '%s' "${HF_TOKEN}" | aws secretsmanager create-secret "${REGION_ARG[@]}" \
    --name "${SECRET_NAME}" --description "Hugging Face token for adaptive-inference sweep" \
    --secret-string file:///dev/stdin >/dev/null
fi

ARN="$(aws secretsmanager describe-secret "${REGION_ARG[@]}" --secret-id "${SECRET_NAME}" --query ARN --output text)"
echo "done. secret ARN:"
echo "  ${ARN}"
echo
echo "paste this into aws/batch_job_def.json under containerProperties.secrets:"
echo "  { \"name\": \"HF_TOKEN\", \"valueFrom\": \"${ARN}\" }"
