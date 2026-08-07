# 09 — p-IRT / pass-rate MAE

**Status: TODO — EAP 20/0.27 regeneration pending (STOP-DEPENDENT).**

Predicted-vs-actual pass-rate MAE (and θ-MAE) is **stop-dependent** (the prediction is formed
from the ability estimated at the stop) and is NOT yet available under the locked EAP-posterior
stop at floor=20 / SE=0.27. **No figure is dropped into this slot** to avoid implying a legacy
(online-SE) result is current.

The legacy online-SE version is preserved (clearly superseded) at
`../archive_onlineSE_legacy/frq_mae/` (`frq_passrate_mae.csv`, `frq_mae_summary.csv`,
`figures/frq_pirt_vs_actual_2skill.png`). Under that OLD stop the 2-skill pass-rate MAE was
~0.019 (corr) / ~0.062 (scaff, calibrated); θ-MAE ~0.37 (corr) / ~0.31 (scaff). These must be
re-measured at 20/0.27 under the EAP stop before quoting.

## How to regenerate @20/0.27

Extend the EAP OOS machinery (`scripts/eap_oos_grid_study.py`): at the 20/0.27 stop, for each
held-out model compute the predicted pass probability of each observed criterion from the
MWLE θ-at-stop and the fold-refit item params, average to a predicted pass-rate, and compare to
the actual observed pass-rate (raw + calibrated) → pass-rate MAE + `pirt_pred_vs_actual` scatter.
The fold-refit params and full-exhaustion traces are already produced by the grid study; the
per-model θ-at-stop for the locked cell is in `05_oos_recovery/oos_per_model.csv`. Run with
`--workers 6`.

Deferred as TODO (per the packaging brief's escape hatch) rather than shipping unvalidated
freshly-written numbers.
