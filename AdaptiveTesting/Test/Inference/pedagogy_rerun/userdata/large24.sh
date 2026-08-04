#!/bin/bash
# pedagogy re-run cooperative claim-queue worker (group=large24)
exec > /var/log/ped-worker.log 2>&1
set -x
export HOME=/root
trap 'shutdown -h now' EXIT
REGION=us-east-1
RUN_BASE=s3://edullm-adaptive-inference-056956104102/full200
CACHE=$RUN_BASE/hf-cache
RESULTS=$RUN_BASE/results/Outputs
PR=$RUN_BASE/pedagogy_rerun
DEADLINE=1785782884
GROUP=large24
ORDER=desc
RUN_TAG=pedrerun
CLAIM_TTL=1500
ROOT=/opt/dlami/nvme/adaptive-inference
[ -d /opt/dlami/nvme ] || ROOT=/opt/adaptive-inference
mkdir -p "$ROOT/logs" "$ROOT/hf-cache/hub"

# hard budget watchdog: terminate at the deadline no matter what
( while :; do [ "$(date -u +%s)" -ge "$DEADLINE" ] && shutdown -h now; sleep 20; done ) &

TOKEN=$(curl -sX PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' || true)
IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id || echo unknown-$$)
STATUS="$ROOT/logs/status-$IID.log"
push(){ aws s3 cp "$STATUS" "$PR/status/$RUN_TAG-$GROUP-$IID.log" --region "$REGION" --only-show-errors || true; }
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; push; }

log "boot group=$GROUP iid=$IID deadline=$DEADLINE now=$(date -u +%s)"

# --- fetch driver code from the S3 snapshot (no git; repo is private) ---------
aws s3 cp "$RUN_BASE/CODE.tgz" "$ROOT/code.tgz" --region "$REGION" --only-show-errors
log "code cp rc=$?"
mkdir -p "$ROOT/code"; tar xzf "$ROOT/code.tgz" -C "$ROOT/code"
log "untar rc=$?"
INF="$ROOT/code/olmo-eval-full/AdaptiveTesting/Test/Inference"
[ -f "$INF/run_benchmark.py" ] || INF="$(dirname "$(find "$ROOT/code" -name run_benchmark.py -path '*Test/Inference*' | head -1)")"
log "INF=$INF run_benchmark:$([ -f "$INF/run_benchmark.py" ] && echo yes || echo NO)"
ADAPT="$(cd "$INF/../.." && pwd)"     # .../AdaptiveTesting
mkdir -p "$ADAPT/Inputs/Models" "$ADAPT/Outputs" "$ADAPT/Inputs/MCQ/Benchmarks"
cat > "$ADAPT/Inputs/Models/pedagogy_group.yaml" <<'YAML_EOF'
defaults:
  dtype: bfloat16
  trust_remote_code: true
  tp: 1
  apply_chat_template: null
  scoring_method: loglikelihood
  max_model_len: 2048
models:
- id: HuggingFaceTB/SmolLM3-3B
  params_b: 3
  family: SmolLM3
  arch: transformer
  apply_chat_template: true
  enable_thinking: false
- id: Qwen/Qwen2.5-3B
  params_b: 3
  family: Qwen2.5
  arch: transformer
- id: Qwen/Qwen2.5-3B-Instruct
  params_b: 3
  family: Qwen2.5
  arch: transformer
- id: microsoft/Phi-3-mini-4k-instruct
  params_b: 3.8
  family: Phi3
  arch: transformer
- id: microsoft/Phi-3.5-mini-instruct
  params_b: 3.8
  family: Phi3.5
  arch: transformer
- id: microsoft/Phi-4-mini-instruct
  params_b: 3.8
  family: Phi4
  arch: transformer
- id: Qwen/Qwen1.5-4B
  params_b: 4
  family: Qwen1.5
  arch: transformer
- id: Qwen/Qwen1.5-4B-Chat
  params_b: 4
  family: Qwen1.5
  arch: transformer
- id: Qwen/Qwen3-4B
  params_b: 4
  family: Qwen3
  arch: transformer
  apply_chat_template: true
  enable_thinking: false
- id: nvidia/Llama-3.1-Minitron-4B-Depth-Base
  params_b: 4
  family: Llama3.1
  arch: transformer
- id: nvidia/Minitron-4B-Base
  params_b: 4
  family: Minitron
  arch: transformer
- id: nvidia/Nemotron-Mini-4B-Instruct
  params_b: 4
  family: Nemotron
  arch: transformer
- id: openbmb/MiniCPM3-4B
  params_b: 4
  family: MiniCPM3
  arch: transformer
  apply_chat_template: true
- id: facebook/xglm-4.5B
  params_b: 4.5
  family: XGLM
  arch: transformer
- id: Deci/DeciLM-7B
  params_b: 7
  family: DeciLM
  arch: transformer
- id: Deci/DeciLM-7B-instruct
  params_b: 7
  family: DeciLM
  arch: transformer
- id: HuggingFaceH4/zephyr-7b-gemma-v0.1
  params_b: 7
  family: Gemma1
  arch: transformer
- id: Intel/neural-chat-7b-v3-2
  params_b: 7
  family: NeuralChat
  arch: transformer
- id: Nexusflow/Starling-LM-7B-beta
  params_b: 7
  family: Starling
  arch: transformer
- id: Open-Orca/Mistral-7B-OpenOrca
  params_b: 7
  family: OpenOrca
  arch: transformer
- id: allenai/OLMo-2-1124-7B-Instruct
  params_b: 7
  family: OLMo2
  arch: transformer
- id: argilla/DistilabelBeagle14-7B
  params_b: 7
  family: Beagle
  arch: transformer
- id: argilla/notus-7b-v1
  params_b: 7
  family: Notus
  arch: transformer
- id: deepseek-ai/deepseek-llm-7b-chat
  params_b: 7
  family: DeepSeek-LLM
  arch: transformer
- id: ibm/merlinite-7b
  params_b: 7
  family: Merlinite
  arch: transformer
YAML_EOF

# --- env setup: fresh venv + vLLM, with ALL pip tmp/cache on the 250 GB NVMe
#     (the small root EBS is what filled up: OSError ENOSPC). Full pip log to S3.
cd "$INF"
export ROOT
export HF_HOME="$ROOT/hf-cache"
export HF_HUB_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
mkdir -p "$HF_HOME/hub"
export TMPDIR="$ROOT/tmp"; export PIP_CACHE_DIR="$ROOT/pipcache"; export XDG_CACHE_HOME="$ROOT/xdg"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"
PIPLOG="$ROOT/logs/pip.log"
log "df root: $(df -h / | tail -1) ; df nvme: $(df -h "$ROOT" | tail -1)"
python3 -m venv "$ROOT/venv" >>"$PIPLOG" 2>&1 \
  || { apt-get update -y >>"$PIPLOG" 2>&1; apt-get install -y python3.12-venv >>"$PIPLOG" 2>&1; python3 -m venv "$ROOT/venv" >>"$PIPLOG" 2>&1; }
source "$ROOT/venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools >>"$PIPLOG" 2>&1
python -m pip install vllm transformers huggingface_hub hf_transfer accelerate pyyaml >>"$PIPLOG" 2>&1
piprc=$?
tail -c 6000 "$PIPLOG" > "$ROOT/logs/pip.tail" 2>/dev/null || true
aws s3 cp "$ROOT/logs/pip.tail" "$PR/status/pip-$GROUP-$IID.log" --region "$REGION" --only-show-errors || true
log "pip rc=$piprc vllm=$(python -c 'import vllm;print(vllm.__version__)' 2>&1|tail -1) torch=$(python -c 'import torch;print(torch.__version__)' 2>&1|tail -1) transformers=$(python -c 'import transformers;print(transformers.__version__)' 2>&1|tail -1)"
aws s3 sync "$RUN_BASE/mcq_cache/" "$ADAPT/Inputs/MCQ/Benchmarks/" --region "$REGION" --only-show-errors || true

YAML="$ADAPT/Inputs/Models/pedagogy_group.yaml"
SDI=split_download_infer
OUT="$ADAPT/Outputs"
s3exists(){ aws s3 ls "$1" --region "$REGION" >/dev/null 2>&1; }
holder(){ local slug="$1" now ts age d t sz key; now=$(date -u +%s)
  aws s3 ls "$PR/claims/$slug/" --region "$REGION" 2>/dev/null | while read -r d t sz key; do
    [ -z "$key" ] && continue
    ts=$(date -u -d "$d $t" +%s 2>/dev/null || echo 0); age=$((now-ts))
    [ "$age" -lt "$CLAIM_TTL" ] && echo "$key"
  done | sort | head -1
}

mapfile -t IDS < <(python "$SDI/list_repos.py" --models-yaml "$YAML" 2>>"$STATUS")
[ "$ORDER" = desc ] && mapfile -t IDS < <(printf '%s\n' "${IDS[@]}" | tac)
log "eligible=${#IDS[@]} first=${IDS[0]:-none}"

idle=0
while :; do
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && { log "DEADLINE"; break; }
  progressed=0; pending=0
  for id in "${IDS[@]}"; do
    [ -z "$id" ] && continue
    [ "$(date -u +%s)" -ge "$DEADLINE" ] && break
    slug=$(echo "$id" | sed 's#/#__#g')
    s3exists "$RESULTS/_manifests/pedagogy__$slug.done" && continue
    s3exists "$PR/attempted/$slug" && continue
    pending=1
    h=$(holder "$slug"); [ -n "$h" ] && [ "$h" != "$IID" ] && continue
    echo "$(date -u +%s) $IID" | aws s3 cp - "$PR/claims/$slug/$IID" --region "$REGION" --only-show-errors
    sleep 3
    [ "$(holder "$slug")" = "$IID" ] || continue
    log "START $id"
    mapfile -t CDS < <(python "$SDI/list_repos.py" --models "$id" --models-yaml "$YAML" --cache-dirs --include-tokenizers)
    inc=(); for d in "${CDS[@]}"; do [ -n "$d" ] && inc+=(--include "hub/$d/*"); done
    aws s3 sync "$CACHE/" "$HF_HOME/" --region "$REGION" --exclude "*" "${inc[@]}" --only-show-errors
    aws s3 sync "$RESULTS/" "$OUT/" --region "$REGION" --exclude "*" --include "mcq/pedagogy/$slug.csv" --include "_manifests/pedagogy__$slug.done" --only-show-errors || true
    timeout 900 python run_benchmark.py --models "$id" --models-yaml "$YAML" --benchmarks pedagogy --backend vllm --inference-config configs/inference.full200.yaml > "$ROOT/logs/model-$slug.log" 2>&1
    rc=$?
    aws s3 sync "$OUT/" "$RESULTS/" --region "$REGION" --exclude "*" --include "mcq/pedagogy/$slug.csv" --include "_manifests/pedagogy__$slug.done" --only-show-errors
    aws s3 cp "$ROOT/logs/model-$slug.log" "$PR/logs/$RUN_TAG-$GROUP-$slug.log" --region "$REGION" --only-show-errors || true
    echo "$(date -u +%s) $IID rc=$rc" | aws s3 cp - "$PR/attempted/$slug" --region "$REGION" --only-show-errors
    if s3exists "$RESULTS/_manifests/pedagogy__$slug.done"; then st=OK; else st="FAIL(rc=$rc)"; fi
    log "END $id -> $st"
    for d in "${CDS[@]}"; do [ -n "$d" ] && rm -rf "$HF_HOME/hub/$d"; done
    progressed=1
  done
  [ "$pending" = 0 ] && { log "all done/attempted -> exit"; break; }
  if [ "$progressed" = 0 ]; then idle=$((idle+1)); log "idle $idle"; [ "$idle" -ge 5 ] && break; sleep 20; else idle=0; fi
done
log "worker exit group=$GROUP iid=$IID at $(date -u +%s)"
# trap EXIT -> shutdown -h now (instance terminates)
