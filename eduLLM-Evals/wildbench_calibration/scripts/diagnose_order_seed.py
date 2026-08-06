"""Task B diagnostic: which models drive the order/seed theta-SD tail? (read-only).

Ranks models by across-seed theta SD (exp-11) and, for the high-SD tail, reports the
degeneracy diagnostics that would explain instability: full-bank pass rate, leaderboard
theta + SE_total, #administrable scenarios/criteria, |theta|, and the fraction of CAT-pool
items that are INFORMATIVE at the model's theta (|a*theta - b| small => p in ~[0.12,0.88]).
Few informative items and/or |theta| near the +/-6.75 leaderboard extremes => short adaptive
tests over a huge bank land on very different scenario sets per seed. Changes NOTHING.

Writes experiments/11_order_seed/order_seed_diagnostics.json + prints a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import wildbench_scenario_lib as L  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402

base = ROOT / "wildbench_calibration"
BANK = base / "wildbench_scenario_fitted_1d_catpool.jsonl"
SPREAD = base / "experiments" / "11_order_seed" / "per_model_spread.csv"
LEADER = base / "model_leaderboard.csv"
OUT = base / "experiments" / "11_order_seed" / "order_seed_diagnostics.json"
OUT_CSV = base / "experiments" / "11_order_seed" / "order_seed_diagnostics.csv"

# "weakly_identified" rule (EAP-of-record): a model that CANNOT reach the SE target under the
# EAP-posterior stop even at the forced cap (deployed hit_cap == True) -- its ability is only
# bounded, not point-identified, because the bank lacks informative items at its theta. This is
# the stop-native definition; the frac_informative / order-seed-SD diagnostics below are context.
WI_INFORMATIVE_CUTOFF = 0.05   # (context) < 5% of CAT-pool items informative at the model's theta
WI_THETA_SD_CUTOFF = 0.50      # (context) across-seed theta SD threshold
DEPLOYED_SE_CSV = base / "experiments" / "07_parameter_uncertainty" / "se_post_vs_total.csv"


def main() -> int:
    records, dims, _ = scat.load_fitted_bank(BANK, "clamp")
    d = dims[0]
    ids, A, b = scat.assemble_arrays(records, dims)
    a = A[:, 0]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    matrix = pd.read_csv(L.DEFAULT_MATRIX, index_col=0)
    sub = matrix.reindex(columns=ids)
    Y = sub.to_numpy(float)
    M = ~np.isnan(Y)
    models = list(matrix.index)

    spread = pd.read_csv(SPREAD).set_index("model")
    lead = pd.read_csv(LEADER).set_index("model")

    rows = []
    for i, m in enumerate(models):
        obs = M[i]
        yv = Y[i][obs]
        n_crit = int(obs.sum())
        scens = {scen_of[ids[j]] for j in np.where(obs)[0]}
        pass_rate = float(np.mean(yv)) if yv.size else np.nan
        theta = float(lead.loc[m, "theta"]) if m in lead.index else np.nan
        se_total = float(lead.loc[m, "se_total"]) if m in lead.index else np.nan
        # informative CAT-pool items at this model's theta: p(theta) in ~[0.12, 0.88]
        eta = a * theta - b
        p = 1.0 / (1.0 + np.exp(-eta))
        informative = np.mean((p > 0.12) & (p < 0.88))
        rows.append({
            "model": m,
            "theta_sd": float(spread.loc[m, "theta_sd"]) if m in spread.index else np.nan,
            "theta_range_seeds": float(spread.loc[m, "theta_range"]) if m in spread.index else np.nan,
            "theta": theta, "abs_theta": abs(theta), "se_total": se_total,
            "pass_rate": pass_rate,
            "pass_extremity": float(min(pass_rate, 1 - pass_rate)),  # 0 => all-pass/all-fail
            "n_admin_criteria": n_crit, "n_admin_scenarios": len(scens),
            "frac_informative_items": round(float(informative), 4),
        })
    df = pd.DataFrame(rows).sort_values("theta_sd", ascending=False).reset_index(drop=True)

    # EAP-of-record weakly-identified flag: deployed EAP hit_cap (can't reach SE target).
    hit_cap = {}
    if DEPLOYED_SE_CSV.is_file():
        sd = pd.read_csv(DEPLOYED_SE_CSV)
        sd = sd[sd["regime"] == "deployed_cat_8_0.12"]
        if "hit_cap" in sd.columns:
            hit_cap = dict(zip(sd["model"], sd["hit_cap"].astype(bool)))
    df["deployed_hit_cap"] = df["model"].map(lambda m: bool(hit_cap.get(m, False)))
    # structural context flag (informative-item-poor AND order/seed-unstable)
    df["low_info_unstable"] = ((df["frac_informative_items"] < WI_INFORMATIVE_CUTOFF)
                               & (df["theta_sd"] > WI_THETA_SD_CUTOFF))
    df["weakly_identified"] = df["deployed_hit_cap"] if hit_cap else df["low_info_unstable"]
    df.round(6).to_csv(OUT_CSV, index=False)
    flagged = df[df["weakly_identified"]].copy()

    top = df.head(8)
    # correlations across all models
    corr_sd_abstheta = float(np.corrcoef(df["theta_sd"], df["abs_theta"])[0, 1])
    corr_sd_passext = float(np.corrcoef(df["theta_sd"], df["pass_extremity"])[0, 1])
    corr_sd_inform = float(np.corrcoef(df["theta_sd"], df["frac_informative_items"])[0, 1])
    theta_ext = df["abs_theta"].quantile(0.90)
    tail = df[df["theta_sd"] > 0.30]
    tail_frac_boundary = float(np.mean(tail["abs_theta"] >= theta_ext)) if len(tail) else 0.0

    summary = {
        "purpose": "order/seed theta-SD tail diagnostics (read-only)",
        "n_models": len(df),
        "weakly_identified_rule": (
            f"weakly_identified = (frac_informative_items < {WI_INFORMATIVE_CUTOFF}) "
            f"AND (theta_sd > {WI_THETA_SD_CUTOFF}). Near-all-fail / very low information "
            "AND order/seed-unstable => ability is only bounded (not point-identified) at a "
            "short adaptive test. Both conditions required so a low-info-but-stable model "
            "is not flagged."),
        "weakly_identified_cutoffs": {"frac_informative_items_max": WI_INFORMATIVE_CUTOFF,
                                      "theta_sd_min": WI_THETA_SD_CUTOFF},
        "weakly_identified_models": flagged[[
            "model", "theta", "theta_sd", "pass_rate", "frac_informative_items",
            "se_total"]].round(4).to_dict("records"),
        "n_weakly_identified": int(len(flagged)),
        "theta_sd_stats": {"mean": float(df["theta_sd"].mean()),
                           "median": float(df["theta_sd"].median()),
                           "max": float(df["theta_sd"].max())},
        "top8_high_sd": top.round(4).to_dict("records"),
        "correlations_with_theta_sd": {
            "abs_theta": round(corr_sd_abstheta, 3),
            "pass_extremity (higher=more central)": round(corr_sd_passext, 3),
            "frac_informative_items": round(corr_sd_inform, 3)},
        "tail_models_sd_gt_0.30": {
            "n": int(len(tail)),
            "models": tail["model"].tolist(),
            "frac_in_top10pct_abs_theta": round(tail_frac_boundary, 3),
            "mean_abs_theta": round(float(tail["abs_theta"].mean()), 3) if len(tail) else None,
            "mean_frac_informative": round(float(tail["frac_informative_items"].mean()), 4)
            if len(tail) else None},
        "median_frac_informative_all_models": round(float(df["frac_informative_items"].median()), 4),
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("=" * 100)
    print("ORDER/SEED theta-SD tail diagnostics (top 8 by across-seed SD)")
    print("=" * 100)
    print(top[["model", "theta_sd", "theta", "pass_rate", "n_admin_scenarios",
               "n_admin_criteria", "frac_informative_items"]].to_string(index=False))
    print("\ncorr(theta_sd, |theta|)            =", round(corr_sd_abstheta, 3))
    print("corr(theta_sd, pass_extremity)     =", round(corr_sd_passext, 3),
          "(pass_extremity: 0=all-pass/fail, 0.5=balanced)")
    print("corr(theta_sd, frac_informative)   =", round(corr_sd_inform, 3))
    print(f"\ntail (SD>0.30): {len(tail)} models; "
          f"{tail_frac_boundary*100:.0f}% are in the top-10% |theta| (boundary); "
          f"median frac_informative all models = {summary['median_frac_informative_all_models']}")
    print(f"\nweakly_identified rule: frac_informative < {WI_INFORMATIVE_CUTOFF} AND "
          f"theta_sd > {WI_THETA_SD_CUTOFF}  ->  {len(flagged)} models:")
    for _, r in flagged.iterrows():
        print(f"  {r['model']:48s} theta={r['theta']:6.2f} SD={r['theta_sd']:.2f} "
              f"pass={r['pass_rate']*100:.2f}% inf={r['frac_informative_items']*100:.1f}%")
    print(f"wrote -> {OUT} and {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
