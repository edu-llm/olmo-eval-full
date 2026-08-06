"""ANALYSIS-ONLY: deployed-CAT precision-reached rate at the locked op-point 8/0.12.

Reads existing artifacts (no refit / re-bootstrap / engine re-run):
  * per-model deployed SE from experiments/07_parameter_uncertainty/se_post_vs_total.csv
    (regime deployed_cat_8_0.12: SE_posterior = ability-only, SE_total),
  * the weakly_identified flag from model_leaderboard.csv,
  * the engine's aggregate convergence_rate for f8_se0.12 from
    experiments/06_floor_se_grid/recovery_grid.csv (its online normal-approx SE criterion).

precision_reached (honest posterior metric) = SE_posterior <= 0.12. Writes
experiments/07_parameter_uncertainty/precision_reached.{json,csv}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "experiments" / "07_parameter_uncertainty" / "se_post_vs_total.csv"
LEADER = BASE / "model_leaderboard.csv"
GRID = BASE / "experiments" / "06_floor_se_grid" / "recovery_grid.csv"
OUT_JSON = BASE / "experiments" / "07_parameter_uncertainty" / "precision_reached.json"
OUT_CSV = BASE / "experiments" / "07_parameter_uncertainty" / "precision_reached.csv"
TARGET = 0.12

dep = pd.read_csv(SRC)
dep = dep[dep["regime"] == "deployed_cat_8_0.12"].copy()
lead = pd.read_csv(LEADER).set_index("model")
dep["weakly_identified"] = dep["model"].map(
    lambda m: bool(lead.loc[m, "weakly_identified"]) if m in lead.index else False)
dep["precision_reached"] = dep["SE_posterior"] <= TARGET
dep = dep.sort_values("SE_posterior").reset_index(drop=True)
dep[["model", "theta", "SE_posterior", "SE_total", "n_admin_criteria",
     "weakly_identified", "precision_reached"]].to_csv(OUT_CSV, index=False)

n = len(dep)
reached = int(dep["precision_reached"].sum())
missed = dep[~dep["precision_reached"]].sort_values("SE_posterior", ascending=False)
missed_flagged = int(missed["weakly_identified"].sum())
missed_unflagged = missed[~missed["weakly_identified"]]

# engine's own aggregate criterion (online normal-approx SE)
eng_conv = eng_cap = None
if GRID.is_file():
    g = pd.read_csv(GRID)
    row = g[(g["min_scenarios"] == 8) & (np.isclose(g["se_target"], 0.12))]
    if not row.empty:
        eng_conv = float(row.iloc[0]["convergence_rate"])
        eng_cap = float(row.iloc[0]["capped_frac"])


def q(s):
    return {"median": float(s.median()), "mean": float(s.mean()),
            "q1": float(s.quantile(0.25)), "q3": float(s.quantile(0.75)),
            "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
            "min": float(s.min()), "max": float(s.max())}


summary = {
    "purpose": "deployed-CAT precision-reached rate at locked op-point 8/0.12 (analysis-only)",
    "source_per_model_se": SRC.name,
    "target_se_ability": TARGET,
    "n_models": n,
    "precision_reached_posterior_metric": {
        "definition": "SE_posterior (honest fine-grid EAP posterior SE) <= 0.12",
        "n_reached": reached, "pct_reached": round(100.0 * reached / n, 1),
        "n_missed": n - reached, "pct_missed": round(100.0 * (n - reached) / n, 1),
    },
    "engine_online_se_criterion": {
        "note": "the engine's own stop test uses an online normal-approx SE; it is optimistic "
                "vs the honest posterior SE at boundary/low-information theta.",
        "convergence_rate_f8_se0.12": eng_conv, "capped_frac_f8_se0.12": eng_cap,
        "source": GRID.name,
    },
    "se_ability_distribution": q(dep["SE_posterior"]),
    "se_total_distribution": q(dep["SE_total"]),
    "missed_target_models": missed[[
        "model", "theta", "SE_posterior", "SE_total", "weakly_identified"]].round(4)
        .to_dict("records"),
    "n_missed_weakly_identified": missed_flagged,
    "n_missed_not_flagged": int(len(missed_unflagged)),
    "missed_not_flagged_models": missed_unflagged[[
        "model", "theta", "SE_posterior", "SE_total"]].round(4).to_dict("records"),
}
OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("=" * 84)
print("DEPLOYED precision-reached @ 8/0.12 (honest posterior SE_ability <= 0.12)")
print("=" * 84)
print(f"reached: {reached}/{n} ({100.0*reached/n:.1f}%)   missed: {n-reached}")
print(f"engine online-SE convergence_rate (f8_se0.12) = {eng_conv} (capped {eng_cap})")
print(f"SE_ability: median={dep['SE_posterior'].median():.3f} "
      f"IQR=[{dep['SE_posterior'].quantile(.25):.3f},{dep['SE_posterior'].quantile(.75):.3f}] "
      f"mean={dep['SE_posterior'].mean():.3f}")
print(f"SE_total  : median={dep['SE_total'].median():.3f} "
      f"IQR=[{dep['SE_total'].quantile(.25):.3f},{dep['SE_total'].quantile(.75):.3f}] "
      f"mean={dep['SE_total'].mean():.3f}")
print(f"\nmissed target ({n-reached}): {missed_flagged} weakly_identified + "
      f"{len(missed_unflagged)} not-flagged")
for _, r in missed.iterrows():
    tag = "WEAKLY-ID" if r["weakly_identified"] else "not-flagged"
    print(f"  {r['model']:48s} SE_ab={r['SE_posterior']:.3f} SE_tot={r['SE_total']:.3f} "
          f"theta={r['theta']:6.2f}  [{tag}]")
print(f"\nwrote -> {OUT_JSON}\nwrote -> {OUT_CSV}")
