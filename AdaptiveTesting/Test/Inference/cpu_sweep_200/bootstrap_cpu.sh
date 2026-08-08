#!/usr/bin/env bash
# Bootstrap a CPU worker host (Amazon Linux 2023 / Ubuntu).
set -euo pipefail

ROOT="${ROOT:-/opt/adaptive-cpu}"
REPO_URL="${REPO_URL:-https://github.com/edu-llm/olmo-eval-full.git}"
BRANCH="${BRANCH:-AdaptiveEvals}"
VENV="${ROOT}/venv"

sudo mkdir -p "${ROOT}"
sudo chown "$(id -u):$(id -g)" "${ROOT}" || true
mkdir -p "${ROOT}/hf-cache" "${ROOT}/logs"

if [[ ! -d "${ROOT}/code/.git" ]]; then
  git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${ROOT}/code"
else
  git -C "${ROOT}/code" fetch origin "${BRANCH}"
  git -C "${ROOT}/code" checkout "${BRANCH}"
  git -C "${ROOT}/code" pull --ff-only origin "${BRANCH}" || true
fi

python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel
# CPU torch wheels
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r "${ROOT}/code/AdaptiveTesting/Test/Inference/cpu_sweep_200/requirements-cpu.txt"

# HF token from Secrets Manager if present
if command -v aws >/dev/null 2>&1; then
  if TOK=$(aws secretsmanager get-secret-value --secret-id hf-token --query SecretString --output text 2>/dev/null); then
    echo "${TOK}" > "${ROOT}/.hf_token"
    chmod 600 "${ROOT}/.hf_token"
    export HF_TOKEN="${TOK}"
  fi
fi

touch "${ROOT}/.bootstrap_ok"
echo "bootstrap ok -> ${ROOT}"
