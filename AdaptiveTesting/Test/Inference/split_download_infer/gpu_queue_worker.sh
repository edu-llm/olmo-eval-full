#!/usr/bin/env bash
# GPU queue worker for the full-roster sweep. Pulls only models that are already
# in S3 ("ready"), always picking the lowest zig-zag index available, so:
#   * a GPU never starts a model whose weights aren't staged, and
#   * inference proceeds small/medium/large/repeat (good early-stop spread).
# Each model: pull weights -> resume any prior outputs -> run_benchmark ->
# upload outputs -> mark done -> free disk -> next. Exits (box self-terminates)
# once downloads are complete and nothing is left to claim.
#
# Env:
#   RUN_BASE     s3 run prefix   (default s3://edullm-adaptive-inference-056956104102/full200)
#   REGION       AWS region      (default us-east-1)
#   ROOT         local prefix    (default /opt/dlami/nvme/adaptive-infer)
#   NUM_SHARDS   downloader count (default 16; used to detect "downloads done")
#   MODELS_YAML  roster          (default models_200.yaml in the tree)
#   INFCONFIG    inference config (default configs/inference.full200.yaml)
#   RUN_DEADLINE_EPOCH  unix ts; if set, exit+shutdown when reached (budget cap)
set -uo pipefail

RUN_BASE="${RUN_BASE:-s3://edullm-adaptive-inference-056956104102/full200}"
REGION="${REGION:-us-east-1}"
ROOT="${ROOT:-/opt/dlami/nvme/adaptive-infer}"
NUM_SHARDS="${NUM_SHARDS:-16}"
CACHE="${RUN_BASE}/hf-cache"
CLAIM_TTL="${CLAIM_TTL:-1200}"   # a claim older than this (dead worker) is reclaimable
RUN_DEADLINE_EPOCH="${RUN_DEADLINE_EPOCH:-}"

budget_ok() {
  [[ -z "${RUN_DEADLINE_EPOCH}" ]] && return 0
  (( $(date -u +%s) < RUN_DEADLINE_EPOCH ))
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INF="$(cd "${HERE}/.." && pwd)"
MODELS_YAML="${MODELS_YAML:-${INF}/../../Inputs/Models/models_200.yaml}"
INFCONFIG="${INFCONFIG:-${INF}/configs/inference.full200.yaml}"
OUT="$(cd "${INF}/../.." && pwd)/Outputs"
mkdir -p "${OUT}" "${ROOT}/logs"

export HF_HOME="${ROOT}/hf-cache"; mkdir -p "${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
# shellcheck disable=SC1091
[[ -f "${ROOT}/venv/bin/activate" ]] && source "${ROOT}/venv/bin/activate" || true
[[ -f "${ROOT}/.hf_token" ]] && export HF_TOKEN="$(cat "${ROOT}/.hf_token")" && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

TOKEN="$(curl -s -X PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' 2>/dev/null || true)"
IID="$(curl -s -H "X-aws-ec2-metadata-token: ${TOKEN}" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "gpu-$$")"
LOG="${ROOT}/logs/gpu-${IID}.log"
exec > >(tee -a "${LOG}") 2>&1
hb() { aws s3 cp "${LOG}" "${RUN_BASE}/logs/gpu-${IID}.log" --region "${REGION}" --only-show-errors 2>/dev/null || true; }

echo "[gpu ${IID} $(date -u +%H:%M:%S)] start"

# --- MCQ caches (needed for offline scoring); wait briefly for shard 0 -------
for _ in $(seq 1 60); do
  if aws s3 ls "${RUN_BASE}/_MCQ_READY" --region "${REGION}" >/dev/null 2>&1; then break; fi
  echo "[gpu ${IID}] waiting for MCQ caches..."; sleep 15
done
aws s3 sync "${RUN_BASE}/mcq_cache/" "${INF}/../../Inputs/MCQ/Benchmarks/" --region "${REGION}" --only-show-errors || true

# --- full zig-zag order -------------------------------------------------------
mapfile -t ROWS < <(python "${HERE}/order_models.py" --models-yaml "${MODELS_YAML}")
echo "[gpu ${IID}] ${#ROWS[@]} models in order"

have() { aws s3 ls "${RUN_BASE}/$1/$2" --region "${REGION}" >/dev/null 2>&1; }
downloads_done() {
  local n; n="$(aws s3 ls "${RUN_BASE}/_SHARD_DONE/" --region "${REGION}" 2>/dev/null | wc -l)"
  [[ "${n}" -ge "${NUM_SHARDS}" ]]
}
fresh_claim_holder() {  # echo winning (smallest) non-stale claimant IID for a cd
  local cd="$1" now key d t sz; now="$(date -u +%s)"
  aws s3 ls "${RUN_BASE}/claims/${cd}/" --region "${REGION}" 2>/dev/null | while read -r d t sz key; do
    [[ -z "${key}" ]] && continue
    local ts age; ts="$(date -u -d "${d} ${t}" +%s 2>/dev/null || echo 0)"; age=$(( now - ts ))
    (( age < CLAIM_TTL )) && echo "${key}"
  done | sort | head -1
}
claim() {  # try to own cd; 0 on success
  local cd="$1"
  printf '%s %s\n' "$(date -u +%s)" "${IID}" | \
    aws s3 cp - "${RUN_BASE}/claims/${cd}/${IID}" --region "${REGION}" --only-show-errors 2>/dev/null || return 1
  sleep 3
  [[ "$(fresh_claim_holder "${cd}")" == "${IID}" ]]
}

process() {  # idx id cd
  local idx="$1" id="$2" cd="$3" slug rc
  slug="${id//\//__}"
  echo "[gpu ${IID} $(date -u +%H:%M:%S)] === #${idx} ${id} ==="
  local includes=()
  while IFS= read -r d; do [[ -n "$d" ]] && includes+=(--include "hub/${d}/*"); done < <(
    python "${HERE}/list_repos.py" --models "${id}" --models-yaml "${MODELS_YAML}" --cache-dirs --include-tokenizers)
  aws s3 sync "${CACHE}/" "${HF_HOME}/" --region "${REGION}" --exclude "*" "${includes[@]}" --only-show-errors
  aws s3 sync "${RUN_BASE}/results/Outputs/" "${OUT}/" --region "${REGION}" --exclude "*" \
    --include "mcq/*/${slug}.csv" --include "open/*/${slug}.responses.jsonl" \
    --include "_manifests/*__${slug}.done" --only-show-errors || true
  ( while true; do sleep 300; printf '%s %s\n' "$(date -u +%s)" "${IID}" | \
      aws s3 cp - "${RUN_BASE}/claims/${cd}/${IID}" --region "${REGION}" --only-show-errors 2>/dev/null; done ) &
  local ref=$!
  ( cd "${INF}" && python run_benchmark.py --models "${id}" --models-yaml "${MODELS_YAML}" \
      --benchmarks all --backend vllm --inference-config "${INFCONFIG}" ) 2>&1 | tee "${ROOT}/logs/model-${idx}.log"
  rc=${PIPESTATUS[0]}
  kill "${ref}" 2>/dev/null || true
  aws s3 sync "${OUT}/" "${RUN_BASE}/results/Outputs/" --region "${REGION}" --exclude "*" \
    --include "mcq/*/${slug}.csv" --include "open/*/${slug}.responses.jsonl" \
    --include "_manifests/*__${slug}.done" --only-show-errors
  aws s3 cp "${ROOT}/logs/model-${idx}.log" "${RUN_BASE}/logs/model-${idx}-${slug}.log" \
    --region "${REGION}" --only-show-errors || true
  printf 'rc=%s idx=%s id=%s at=%s\n' "${rc}" "${idx}" "${id}" "$(date -u +%Y%m%dT%H%M%SZ)" | \
    aws s3 cp - "${RUN_BASE}/done/${cd}" --region "${REGION}" --only-show-errors
  rm -rf "${HF_HOME}/hub/${cd}" 2>/dev/null || true
  echo "[gpu ${IID} $(date -u +%H:%M:%S)] done #${idx} ${id} rc=${rc}"
  hb
}

idle=0
while true; do
  if ! budget_ok; then
    echo "[gpu ${IID} $(date -u +%H:%M:%S)] BUDGET_DEADLINE reached -> exit"
    hb
    break
  fi
  mapfile -t READY < <(aws s3 ls "${RUN_BASE}/ready/" --region "${REGION}" 2>/dev/null | awk '{print $NF}')
  mapfile -t DONE  < <(aws s3 ls "${RUN_BASE}/done/"  --region "${REGION}" 2>/dev/null | awk '{print $NF}')
  declare -A R=(); for x in "${READY[@]}"; do [[ -n "$x" ]] && R["$x"]=1; done
  declare -A D=(); for x in "${DONE[@]}";  do [[ -n "$x" ]] && D["$x"]=1; done
  selected=""; any_candidate=0
  for row in "${ROWS[@]}"; do
    idx="$(cut -f1 <<<"$row")"; id="$(cut -f2 <<<"$row")"; cd="$(cut -f3 <<<"$row")"
    [[ -n "${R[$cd]:-}" && -z "${D[$cd]:-}" ]] || continue
    any_candidate=1
    holder="$(fresh_claim_holder "${cd}")"
    if [[ -z "${holder}" ]]; then
      if claim "${cd}"; then selected="${idx}	${id}	${cd}"; break; fi
    fi
  done
  if [[ -n "${selected}" ]]; then
    idle=0
    process "$(cut -f1 <<<"$selected")" "$(cut -f2 <<<"$selected")" "$(cut -f3 <<<"$selected")"
    continue
  fi
  if downloads_done; then
    if [[ "${any_candidate}" == "0" ]]; then
      echo "[gpu ${IID} $(date -u +%H:%M:%S)] downloads complete, queue drained -> exit"; break
    fi
    # remaining work is actively claimed by live workers; nothing for us to do.
    idle=$((idle+1)); echo "[gpu ${IID}] all remaining models claimed by peers (${idle})"
    [[ "${idle}" -ge 3 ]] && { echo "[gpu ${IID}] exiting (peers finishing tail)"; break; }
    sleep 30
  else
    echo "[gpu ${IID} $(date -u +%H:%M:%S)] waiting for more ready models..."; sleep 20
  fi
  hb
done
hb
echo "[gpu ${IID} $(date -u +%H:%M:%S)] worker exit"
