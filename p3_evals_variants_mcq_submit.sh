#!/usr/bin/env bash
# Submit the 4 "variant" (Engram / Optimized-Lngram) checkpoints for MCQ-only eval.
#
# HOW THE MISSING olmo_core.nn.memory MODULE IS PROVIDED
# ------------------------------------------------------
# These checkpoints' config.json import olmo_core.nn.memory.engram.EngramConfig,
# which does NOT exist on OLMo-core@main (the default eval image), so the run
# crashed at model construction. Instead of reinstalling olmo_core at container
# start (torch>=2.8 + torchao + fla -> a risky re-resolve), each checkpoint is
# evaluated on the EXACT image that trained it, via --research-commit:
#
#   engram   run_019fdeec  -> edu-llm/OLMo-core @ 26b5e361...  (src/scripts/train/engram_experiment/engram_moe.py)
#   optlngram run_019fdf42 -> edu-llm/OLMo-core @ 09c02401...  (.../lngram_moe.py; == 26b5e361 + 1 lngram commit)
#
# Both commits carry src/olmo_core/nn/memory/engram.py with the checkpoints'
# schema (orders/num_hash_heads/table_sizes/embedding_dim/vocab_size/
# tokenizer_compression/conv_kernel_size). Recovered from the training runs'
# compiled-submission manifests in edu-llm/platform.
#
# The olmo-eval code itself is unchanged, so EVAL_REF stays at the pushed
# p3-evals HEAD; the fix is entirely in which research image the job rides on.
#
# Requires: gh CLI authenticated. AWS creds are NOT needed for dispatch.
set -euo pipefail

# olmo-eval-full commit to run inside the container (unchanged; already pushed).
EVAL_REF="${EVAL_REF:-76944f78f1bd0ce38584c5a9f685d25fcd828a1c}"

SUBMIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.cursor/skills/eval-platform/scripts/submit_eval_run.sh"
BUCKET="s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs"
ENGRAM_RUN="run_019fdeec-f05e-704c-abee-2c165e03cd7a"
OPTLNGRAM_RUN="run_019fdf42-f22a-7057-a509-23829b81777b"
ENGRAM_COMMIT="26b5e361d13b9bff43cdcd5be836325226ebd41f"
OPTLNGRAM_COMMIT="09c02401c96c5d1a2aa26a46a6528074db07d9d5"

# "<checkpoint>|<research-commit>"
JOBS=(
  "${BUCKET}/${ENGRAM_RUN}/checkpoints/step17000|${ENGRAM_COMMIT}"
  "${BUCKET}/${ENGRAM_RUN}/checkpoints/step19074|${ENGRAM_COMMIT}"
  "${BUCKET}/${OPTLNGRAM_RUN}/checkpoints/step17000|${OPTLNGRAM_COMMIT}"
  "${BUCKET}/${OPTLNGRAM_RUN}/checkpoints/step19074|${OPTLNGRAM_COMMIT}"
)

DRY_RUN="${DRY_RUN:-0}"
maybe_dry=()
[[ "${DRY_RUN}" == "1" ]] && maybe_dry=(--dry-run)

for job in "${JOBS[@]}"; do
  ckpt="${job%|*}"; rcommit="${job#*|}"
  echo "############################################################"
  echo "# submitting: ${ckpt}"
  echo "#      image: OLMo-core @ ${rcommit}"
  echo "############################################################"
  bash "${SUBMIT}" \
    --checkpoint "${ckpt}" \
    --team input-core \
    --experiment p3-evals-variants-mcq \
    --wandb-project edullm-evals \
    --benchmarks "atlas_arc_challenge atlas_hellaswag" \
    --allow-any-task \
    --compute-profile gpu-1xl40s \
    --batch-size 128 \
    --runtime-hours 2 \
    --eval-ref "${EVAL_REF}" \
    --research-commit "${rcommit}" \
    ${maybe_dry[@]+"${maybe_dry[@]}"}
  echo
done
