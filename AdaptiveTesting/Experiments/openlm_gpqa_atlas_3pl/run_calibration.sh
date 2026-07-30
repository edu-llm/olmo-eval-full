#!/usr/bin/env bash
# ATLAS k-subset 3PL (mirt) + mean–σ linking on OpenLM GPQA train matrix.
set -euo pipefail

EXP="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP/../../.." && pwd)"
ATLAS="$REPO/AdaptiveTesting/Inputs/ATLAS"
DATA="$EXP/data"
CALIB="$EXP/calibration"
LOG="$EXP/logs/calibration.log"

mkdir -p "$CALIB" "$EXP/logs"
: >"$LOG"

TRAIN="$DATA/gpqa_response_matrix_train.csv"
ENDS_FILE="$DATA/chunk_ends.txt"
[[ -f "$TRAIN" ]] || { echo "missing $TRAIN — run prepare_matrix.py first"; exit 1; }
[[ -f "$ENDS_FILE" ]] || { echo "missing $ENDS_FILE"; exit 1; }

CHUNK_ENDS="$(tr -d '[:space:]' <"$ENDS_FILE")"
echo "chunk_ends=$CHUNK_ENDS" | tee -a "$LOG"
echo "train=$TRAIN" | tee -a "$LOG"

cd "$ATLAS"

# Fit each chunk in parallel (ATLAS k-subset calibration).
pids=()
for end in ${CHUNK_ENDS//,/ }; do
  echo "Starting chunk_end=$end" | tee -a "$LOG"
  Rscript scripts/01_fit_irt_custom.r \
    --data_file="$TRAIN" \
    --chunk_end="$end" \
    --chunk_ends="$CHUNK_ENDS" \
    --outdir="$CALIB" \
    --itemtype=3PL \
    >>"$LOG" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "chunk pid $pid failed" | tee -a "$LOG"
    fail=1
  fi
done
[[ "$fail" -eq 0 ]] || { echo "One or more chunk fits failed — see $LOG"; exit 1; }

echo "Linking chunks…" | tee -a "$LOG"
Rscript scripts/02_link_chunks_custom.r \
  --outdir="$CALIB" \
  --chunk_ends="$CHUNK_ENDS" \
  | tee -a "$LOG"

echo "Done. Linked params: $CALIB/irt_item_parameters_combined.csv" | tee -a "$LOG"
