#!/usr/bin/env bash
# Bootstrap a cheap CPU "downloader" host. NO torch / vllm - weight fetching only
# needs huggingface_hub + hf_transfer, so this stays tiny and fast to stand up.
#
# Env:
#   ROOT        install/prefix dir            (default /opt/adaptive-dl)
#   REGION      AWS region for the HF secret  (default us-east-1)
#   HF_SECRET   Secrets Manager id            (default hf-token)
set -euo pipefail

ROOT="${ROOT:-/opt/adaptive-dl}"
REGION="${REGION:-us-east-1}"
HF_SECRET="${HF_SECRET:-hf-token}"
VENV="${ROOT}/venv"

sudo mkdir -p "${ROOT}" 2>/dev/null || mkdir -p "${ROOT}"
sudo chown "$(id -u):$(id -g)" "${ROOT}" 2>/dev/null || true
mkdir -p "${ROOT}/hf-cache" "${ROOT}/logs"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}" 2>/dev/null || {
    sudo apt-get update -y && sudo apt-get install -y python3-venv
    python3 -m venv "${VENV}"
  }
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -q --upgrade pip wheel setuptools
python -m pip install -q "huggingface_hub>=0.25" hf_transfer "datasets>=3.2" pyyaml boto3

# Cache the HF token to a file the sync step can source (never printed).
if command -v aws >/dev/null 2>&1; then
  if TOK="$(aws secretsmanager get-secret-value --region "${REGION}" \
        --secret-id "${HF_SECRET}" --query SecretString --output text 2>/dev/null)"; then
    if [[ "${TOK}" == \{* ]]; then
      TOK="$(python -c 'import sys,json;print(json.load(sys.stdin).get("HF_TOKEN",""))' <<<"${TOK}")"
    fi
    umask 077
    printf '%s' "${TOK}" > "${ROOT}/.hf_token"
    echo "hf token cached -> ${ROOT}/.hf_token"
  fi
fi

touch "${ROOT}/.bootstrap_ok"
echo "downloader bootstrap ok -> ${ROOT}"
