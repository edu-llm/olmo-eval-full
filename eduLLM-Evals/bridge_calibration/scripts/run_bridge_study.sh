#!/usr/bin/env bash
# Reproducible Bridge calibration study runner (RUN ON ORCD, from the qwen_grading_run
# repo root). Produces the full bridge_calibration/ tree -- base package + experiments
# 03..12 -- into $OUT, then you rsync $OUT to the repo's eduLLM-Evals/bridge_calibration/.
#
# Each step is independent; a failure in one (e.g. the heavy jackknife) is logged and the
# rest still run. Locked operating point: SE target 0.15 / floor 20 (see experiment 06).
#
#   MATRIX=runs/judge_open4/Bridge/response_matrix.csv \
#   bash scripts/run_bridge_study.sh
set -u

MATRIX="${MATRIX:-runs/judge_open4/Bridge/response_matrix.csv}"
RUBRICS="${RUBRICS:-data/Bridge/rubrics.jsonl}"
SCEN="${SCEN:-data/Bridge/scenarios.jsonl}"
OUT="${OUT:-runs/calibration/Bridge_study}"
SE="${SE:-0.15}"; FLOOR="${FLOOR:-20}"; RIDGE="${RIDGE:-0.01}"; GRID="${GRID:-41}"; K="${K:-5}"
RUN="${RUN:-uv run --no-sync python}"
E="$OUT/experiments"
mkdir -p "$OUT" "$E" "$OUT/bank" "$OUT/scripts"

banner(){ echo; echo "======================================================================"; echo ">> $*"; echo "======================================================================"; }
step(){ banner "$1"; shift; if "$@"; then echo "[ok]"; else echo "[FAILED: $*] (continuing)"; fi; }

banner "Bridge study config"
cat > "$OUT/study_config.json" <<JSON
{ "matrix": "$MATRIX", "rubrics": "$RUBRICS", "scenarios": "$SCEN",
  "operating_point": {"se_target": $SE, "floor": $FLOOR}, "ridge": $RIDGE,
  "fit_grid": $GRID, "k_folds": $K,
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)" }
JSON
cat "$OUT/study_config.json"

# --- base package: item params, leaderboard (Fisher SE), calibrated bank, manifest ---
step "base package (item_params, leaderboard, rubrics_calibrated)" \
  $RUN scripts/bridge_calibration_study.py --matrix "$MATRIX" --rubrics "$RUBRICS" \
    --scenarios "$SCEN" --out-dir "$OUT" --k "$K" --grid "$GRID" --ridge "$RIDGE"
cp -f "$OUT/rubrics_calibrated.jsonl" "$OUT/bank/bridge_unidim_calibrated.jsonl" 2>/dev/null || true
cp -f "$OUT/CALIBRATION_STUDY_SUMMARY.md" "$OUT/README.md" 2>/dev/null || true
# committed layout = README.md + bank/ (drop the redundant originals)
rm -f "$OUT/CALIBRATION_STUDY_SUMMARY.md" "$OUT/rubrics_calibrated.jsonl" 2>/dev/null || true

# --- 03 dimensionality (1D..5D structures) ---
step "03 dimensionality comparison (1D..5D)" \
  $RUN scripts/bridge_dimensionality.py --matrix "$MATRIX" --rubrics "$RUBRICS" \
    --scenarios "$SCEN" --out-dir "$E/03_structures" --k "$K" --ridge "$RIDGE"

# --- 04 CAT vs random efficiency ---
step "04 CAT-vs-random efficiency" \
  $RUN scripts/bridge_cat_vs_random.py --matrix "$MATRIX" --rubrics "$RUBRICS" \
    --scenarios "$SCEN" --out-dir "$E/04_efficiency_vs_random" --k "$K" --grid "$GRID" \
    --ridge "$RIDGE" --se-target "$SE"

# --- 05 OOS recovery at the locked operating point ---
step "05 OOS recovery (theta + p-IRT) @ SE $SE / floor $FLOOR" \
  $RUN scripts/bridge_recovery_plots.py --matrix "$MATRIX" --rubrics "$RUBRICS" \
    --scenarios "$SCEN" --out-dir "$E/05_oos_recovery" --k "$K" --grid "$GRID" \
    --ridge "$RIDGE" --se-target "$SE" --min-items "$FLOOR"

# --- 06 SE x floor efficiency grid + reconciled best.json (locked op-point) ---
step "06 SE x floor efficiency grid" \
  $RUN scripts/bridge_se_floor_sweep.py --matrix "$MATRIX" --rubrics "$RUBRICS" \
    --scenarios "$SCEN" --out-dir "$E/06_floor_se_grid" --k "$K" --grid "$GRID" --ridge "$RIDGE"
step "06b stamp locked operating point into best.json (SE $SE / floor $FLOOR)" \
  $RUN scripts/bridge_pick_operating_point.py --sweep "$E/06_floor_se_grid/sweep_results.csv" \
    --out "$E/06_floor_se_grid/best.json" --se "$SE" --floor "$FLOOR"

# --- 07 parameter uncertainty / total SE (also rewrites the leaderboard) + 08 figure ---
step "07 parameter uncertainty / total SE (jackknife) + rewrite leaderboard" \
  $RUN scripts/bridge_param_uncertainty.py --matrix "$MATRIX" --rubrics "$RUBRICS" \
    --scenarios "$SCEN" --out-dir "$E/07_parameter_uncertainty" \
    --leaderboard-out "$OUT/model_leaderboard.csv" --grid "$GRID" --ridge "$RIDGE"
mkdir -p "$E/08_leaderboard"
cp -f "$E/07_parameter_uncertainty/figures/leaderboard_general.png" "$E/08_leaderboard/" 2>/dev/null || true
cp -f "$OUT/model_leaderboard.csv" "$E/08_leaderboard/" 2>/dev/null || true

# --- 10 estimator comparison / 11 order-seed / 12 ridge-grid sensitivity ---
step "10 estimator comparison (EAP/MWLE/MLE)" \
  $RUN scripts/bridge_sensitivity.py --matrix "$MATRIX" --rubrics "$RUBRICS" \
    --scenarios "$SCEN" --out-dir "$E/10_estimator_comparison" --k "$K" --grid "$GRID" \
    --ridge "$RIDGE" --se-target "$SE" --min-items "$FLOOR" --only estimator
step "12 ridge/grid sensitivity" \
  $RUN scripts/bridge_sensitivity.py --matrix "$MATRIX" --rubrics "$RUBRICS" \
    --scenarios "$SCEN" --out-dir "$E/12_ridge_grid_sensitivity" --k "$K" --grid "$GRID" \
    --ridge "$RIDGE" --se-target "$SE" --min-items "$FLOOR" --only sensitivity
step "11 order/seed stability" \
  $RUN scripts/bridge_sensitivity.py --matrix "$MATRIX" --rubrics "$RUBRICS" \
    --scenarios "$SCEN" --out-dir "$E/11_order_seed" --k "$K" --grid "$GRID" \
    --ridge "$RIDGE" --se-target "$SE" --min-items "$FLOOR" --only order

# --- vendor the scripts so the study is reproducible from the committed tree ---
for f in bridge_calibration_study.py bridge_dimensionality.py bridge_cat_vs_random.py \
         bridge_recovery_plots.py bridge_se_floor_sweep.py bridge_param_uncertainty.py \
         bridge_sensitivity.py bridge_pick_operating_point.py run_bridge_study.sh; do
  cp -f "scripts/$f" "$OUT/scripts/" 2>/dev/null || true
done

# --- append a study index to README so the top-level doc ties everything together ---
cat >> "$OUT/README.md" <<MD

---

## Study contents (reproducible via \`scripts/run_bridge_study.sh\`)

Locked CAT operating point: **SE target $SE / floor $FLOOR** (experiment 06).

| # | experiment | question | key output |
|---|---|---|---|
| 03 | \`experiments/03_structures\` | 1D vs 2/3/4/5-skill dimensionality | \`structure_comparison.csv\`, \`selection.json\` |
| 04 | \`experiments/04_efficiency_vs_random\` | does adaptive selection save criteria? | \`results.csv\`, \`summary.json\` |
| 05 | \`experiments/05_oos_recovery\` | held-out theta + p-IRT recovery @ locked point | \`recovery_metrics.json\` (headline r) |
| 06 | \`experiments/06_floor_se_grid\` | SE x floor efficiency frontier | \`sweep_results.csv\`, \`best.json\` |
| 07 | \`experiments/07_parameter_uncertainty\` | ability vs calibration SE (total-SE bars) | \`leaderboard_se_components.csv\` |
| 08 | \`experiments/08_leaderboard\` | leaderboard with honest total-SE bars | \`leaderboard_general.png\` |
| 10 | \`experiments/10_estimator_comparison\` | EAP vs MWLE vs MLE | \`estimator_comparison.json\` |
| 11 | \`experiments/11_order_seed\` | is theta robust to administration order? | \`order_seed_stability.json\` |
| 12 | \`experiments/12_ridge_grid_sensitivity\` | fit robustness to ridge/grid | \`ridge_grid_sensitivity.csv\` |

**Headline recovery** is experiment 05 (held-out MWLE-CAT theta at the locked point);
the top-level \`recovery.json\` is a simpler EAP k-fold sanity check. **Leaderboard SE**
in \`model_leaderboard.csv\` is the *total* SE (ability (+) calibration) from experiment 07;
at N=51 the calibration term dominates. **Dimensionality (03) is exploratory**: a 5-skill
MIRT does not identify at this N (latent correlations collapse toward 1; the affective axis
has no informative anchor) -- the committed multi-skill numbers require the 200-model run.
MD

banner "DONE -> $OUT"
echo "experiments:"; ls -1 "$E" 2>/dev/null | sed 's/^/  /'
echo; echo "next (on your Mac):"
echo "  rsync -az 'orcd-login:qwen_grading_run/$OUT/' eduLLM-Evals/bridge_calibration/"
