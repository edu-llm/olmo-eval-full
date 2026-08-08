#!/usr/bin/env bash
# Stand up the frozen EduLLM Qwen judge as a persistent OpenAI-compatible vLLM 0.26.0
# server on a dedicated judge machine (GPU). The live two-machine CAT run points its
# judge auxiliary provider's base_url at http://THIS_HOST:${JUDGE_PORT}/v1.
#
# The launch flags mirror olmo_eval.inference.providers.vllm_server_utils._build_server_command
# for the frozen judge kwargs, so a manually served endpoint matches what the managed
# path would have started. The model + revision are frozen and MUST equal the constants
# in src/olmo_eval/edullm/judge.py (QWEN_JUDGE_MODEL / QWEN_JUDGE_REVISION).
#
# vLLM 0.26.0 is required for explicit-token logprob_token_ids; keep it in its own venv
# so it never perturbs the repository's own vLLM pin.
set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-9B}"
JUDGE_REVISION="${JUDGE_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_VENV="${JUDGE_VENV:-.venv-judge-vllm026}"
VLLM_VERSION="${VLLM_VERSION:-0.26.0}"

if [[ ! -d "${JUDGE_VENV}" ]]; then
  echo "Creating isolated vLLM ${VLLM_VERSION} environment at ${JUDGE_VENV}"
  uv venv "${JUDGE_VENV}" --python 3.12
  uv pip install --python "${JUDGE_VENV}/bin/python" "vllm==${VLLM_VERSION}"
fi

JUDGE_PY="${JUDGE_VENV}/bin/python"
echo "vLLM version in judge env:"
"${JUDGE_PY}" -c "from importlib.metadata import version; print(version('vllm'))"

# VLLM_ALLOW_LONG_MAX_MODEL_LEN mirrors the managed path when max_model_len is set.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

echo "Serving ${JUDGE_MODEL}@${JUDGE_REVISION} on 0.0.0.0:${JUDGE_PORT} (base_url http://<host>:${JUDGE_PORT}/v1)"
exec "${JUDGE_PY}" -m vllm.entrypoints.openai.api_server \
  --model "${JUDGE_MODEL}" \
  --revision "${JUDGE_REVISION}" \
  --tokenizer "${JUDGE_MODEL}" \
  --host 0.0.0.0 \
  --port "${JUDGE_PORT}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --language-model-only \
  --enable-prefix-caching \
  --no-use-tqdm-on-load
