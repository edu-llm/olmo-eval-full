#!/usr/bin/env bash
# Local dry-run / smoke driver for the inference sweep.
#
# Usage:
#   ./orchestrate.sh smoke      # tiny mock end-to-end (no GPU, no weights)
#   ./orchestrate.sh mcq        # all MCQ benchmarks, models from manifest (needs GPU)
#   ./orchestrate.sh all        # full sweep (needs GPU)
#
# Env overrides:
#   PY           python launcher (default: "uv run python")
#   MODELS       comma list of models (default: manifest / shard)
#   BACKEND      vllm | hf | mock
#   MAX_SAMPLES  per-benchmark cap
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-uv run python}"
MODE="${1:-smoke}"
BACKEND="${BACKEND:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

args=()
[[ -n "${MODELS:-}" ]] && args+=(--models "${MODELS}")
[[ -n "${BACKEND}" ]] && args+=(--backend "${BACKEND}")
[[ -n "${MAX_SAMPLES}" ]] && args+=(--max-samples "${MAX_SAMPLES}")

case "${MODE}" in
  smoke)
    echo "== smoke: mock backend, synthetic data (offline), 2 small models =="
    ${PY} run_benchmark.py \
      --benchmarks synth_mcq,synth_open \
      --models "Qwen/Qwen2.5-0.5B,EleutherAI/pythia-410m" \
      --backend mock --judge-backend mock \
      --max-samples 5 --no-cache --resident-all
    ${PY} aggregate.py
    ;;
  mcq)
    ${PY} run_benchmark.py --benchmarks mcq "${args[@]}"
    ${PY} aggregate.py
    ;;
  open)
    ${PY} run_benchmark.py --benchmarks open "${args[@]}"
    ${PY} aggregate.py
    ;;
  all)
    ${PY} run_benchmark.py --benchmarks all "${args[@]}"
    ${PY} aggregate.py
    ;;
  *)
    echo "unknown mode: ${MODE} (use smoke|mcq|open|all)" >&2
    exit 1
    ;;
esac
