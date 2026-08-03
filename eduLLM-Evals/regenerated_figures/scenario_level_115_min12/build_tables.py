"""Build floor-12 deliverable tables (A, B, C, D) from the min12 run outputs.

Reads only metrics.json / cat_per_model.csv produced by the floor-12 runs (and the
existing floor-0 runs for the diff). Nothing is invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent          # regenerated_figures/scenario_level_115_min12
FLOOR0 = BASE.parent / "scenario_level_115"      # existing floor-0 runs


def load(p: Path) -> dict:
    with (p).open(encoding="utf-8") as fh:
        return json.load(fh)


def median_final_se(cat_csv: Path, dims: list[str]) -> dict[str, float]:
    df = pd.read_csv(cat_csv)
    return {d: float(df[f"final_se_{d}"].median()) for d in dims}


# ---------------------------------------------------------------------------
# A. Estimator table (OOS, 2-skill), all three estimators
# ---------------------------------------------------------------------------
kf2 = load(BASE / "kfold" / "2_skills" / "metrics.json")
print("=" * 78)
print("A. OOS k-fold estimator recovery (2-skill, floor 12) -- KEEP all three")
print("=" * 78)
print(f"{'estimator':10s} {'corr_r':>8s} {'corr_slope':>11s} {'scaf_r':>8s} {'scaf_slope':>11s}")
for e in ("online", "batch", "mwle"):
    rec = kf2["oos_recovery"][e]
    label = {"online": "online(g)", "batch": "batch", "mwle": "MWLE"}[e]
    print(f"{label:10s} {rec['correctness']['r']:8.3f} {rec['correctness']['slope']:11.3f} "
          f"{rec['scaffolding']['r']:8.3f} {rec['scaffolding']['slope']:11.3f}")
print()


# ---------------------------------------------------------------------------
# B. SE-target table (floor 12), per bank
# ---------------------------------------------------------------------------
def se_table(bank_key: str, dims: list[str]) -> pd.DataFrame:
    rows = []
    for se in ("0.20", "0.25", "0.30", "0.35"):
        d = BASE / "se_sweep" / bank_key / f"se{se}"
        m = load(d / "metrics.json")
        mw = m["recovery"]["theta_mwle"]
        mse = median_final_se(d / "cat_per_model.csv", dims)
        row = {
            "se_target": se,
            "precision": f'{m["precision_reached"]}/{m["n_models"]}',
            "scen_mean": round(m["scenarios_administered"]["mean"], 1),
            "scen_median": round(m["scenarios_administered"]["median"], 1),
            "criteria_mean": round(m["criteria_administered"]["mean"], 1),
            "mwle_corr_r": round(mw["correctness"]["r"], 3),
            "mwle_corr_slope": round(mw["correctness"]["slope"], 3),
            "mwle_scaf_r": round(mw["scaffolding"]["r"], 3),
            "mwle_scaf_slope": round(mw["scaffolding"]["slope"], 3),
        }
        if "presentation" in dims:
            row["mwle_pres_r"] = round(mw["presentation"]["r"], 3)
            row["mwle_pres_slope"] = round(mw["presentation"]["slope"], 3)
        for d2 in dims:
            row[f"medSE_{d2[:4]}"] = round(mse[d2], 3)
        rows.append(row)
    return pd.DataFrame(rows)


b2 = se_table("2_skills", ["correctness", "scaffolding"])
b3 = se_table("3_skills", ["correctness", "scaffolding", "presentation"])
print("=" * 78)
print("B. SE-target sweep table (floor 12)")
print("=" * 78)
print("-- 2-skill --")
print(b2.to_string(index=False))
b2.to_csv(BASE / "se_target_table_2skill.csv", index=False)
print("\n-- 3-skill --")
print(b3.to_string(index=False))
b3.to_csv(BASE / "se_target_table_3skill.csv", index=False)
print()


# ---------------------------------------------------------------------------
# C. Wide MWLE-only 1v2v3 table (one row per model)
# ---------------------------------------------------------------------------
def kf_mwle(path: Path) -> dict:
    return load(path)["oos_recovery"]["mwle"]


def pu_se(path: Path) -> dict:
    return load(path)["se_components"]


def lb(path: Path) -> dict:
    m = load(path)
    return {"precision": f'{m["precision_reached"]}/{m["n_models"]}',
            "scen_mean": round(m["scenarios_administered"]["mean"], 1)}


cols = ["model", "corr_r", "corr_slope", "corr_SEpost", "corr_SEtot",
        "scaf_r", "scaf_slope", "scaf_SEpost", "scaf_SEtot",
        "pres_r", "pres_slope", "pres_SEpost", "pres_SEtot",
        "precision", "scen_mean"]

wide = []

# 1-skill (unidim): single "overall" axis placed in correctness columns
u_kf = kf_mwle(BASE / "kfold" / "unidim" / "metrics.json")["overall"]
u_pu = pu_se(BASE / "param_uncertainty" / "unidim" / "metrics.json")["overall"]
u_lb = lb(BASE / "leaderboard" / "unidim" / "metrics.json")
wide.append({
    "model": "1-skill (unidim)*",
    "corr_r": round(u_kf["r"], 3), "corr_slope": round(u_kf["slope"], 3),
    "corr_SEpost": round(u_pu["median_se_posterior"], 3),
    "corr_SEtot": round(u_pu["median_se_total"], 3),
    "scaf_r": "", "scaf_slope": "", "scaf_SEpost": "", "scaf_SEtot": "",
    "pres_r": "", "pres_slope": "", "pres_SEpost": "", "pres_SEtot": "",
    "precision": u_lb["precision"], "scen_mean": u_lb["scen_mean"],
})

# 2-skill (no presentation)
t2_kf = kf_mwle(BASE / "kfold" / "2_skills" / "metrics.json")
t2_pu = pu_se(BASE / "param_uncertainty" / "2_skills" / "metrics.json")
t2_lb = lb(BASE / "leaderboard" / "2_skills" / "metrics.json")
wide.append({
    "model": "2-skill",
    "corr_r": round(t2_kf["correctness"]["r"], 3), "corr_slope": round(t2_kf["correctness"]["slope"], 3),
    "corr_SEpost": round(t2_pu["correctness"]["median_se_posterior"], 3),
    "corr_SEtot": round(t2_pu["correctness"]["median_se_total"], 3),
    "scaf_r": round(t2_kf["scaffolding"]["r"], 3), "scaf_slope": round(t2_kf["scaffolding"]["slope"], 3),
    "scaf_SEpost": round(t2_pu["scaffolding"]["median_se_posterior"], 3),
    "scaf_SEtot": round(t2_pu["scaffolding"]["median_se_total"], 3),
    "pres_r": "", "pres_slope": "", "pres_SEpost": "", "pres_SEtot": "",
    "precision": t2_lb["precision"], "scen_mean": t2_lb["scen_mean"],
})

# 3-skill (reported at se=0.30 leaderboard for precision/scen_mean)
t3_kf = kf_mwle(BASE / "kfold" / "3_skills" / "metrics.json")
t3_pu = pu_se(BASE / "param_uncertainty" / "3_skills" / "metrics.json")
t3_lb = lb(BASE / "leaderboard" / "3_skills" / "metrics.json")
wide.append({
    "model": "3-skill (se=0.30)",
    "corr_r": round(t3_kf["correctness"]["r"], 3), "corr_slope": round(t3_kf["correctness"]["slope"], 3),
    "corr_SEpost": round(t3_pu["correctness"]["median_se_posterior"], 3),
    "corr_SEtot": round(t3_pu["correctness"]["median_se_total"], 3),
    "scaf_r": round(t3_kf["scaffolding"]["r"], 3), "scaf_slope": round(t3_kf["scaffolding"]["slope"], 3),
    "scaf_SEpost": round(t3_pu["scaffolding"]["median_se_posterior"], 3),
    "scaf_SEtot": round(t3_pu["scaffolding"]["median_se_total"], 3),
    "pres_r": round(t3_kf["presentation"]["r"], 3), "pres_slope": round(t3_kf["presentation"]["slope"], 3),
    "pres_SEpost": round(t3_pu["presentation"]["median_se_posterior"], 3),
    "pres_SEtot": round(t3_pu["presentation"]["median_se_total"], 3),
    "precision": t3_lb["precision"], "scen_mean": t3_lb["scen_mean"],
})

wide_df = pd.DataFrame(wide, columns=cols)
wide_df.to_csv(BASE / "model_comparison_1v2v3_mwle_wide.csv", index=False)
print("=" * 78)
print("C. Wide MWLE-only 1v2v3 table (one row per model)")
print("* 1-skill 'overall' axis in correctness cols; +0.996 w/ correctness, -0.66 w/ scaffolding")
print("=" * 78)
print(wide_df.to_string(index=False))
print()


# ---------------------------------------------------------------------------
# D. Diff floor-0 vs floor-12 (leaderboard scen_mean/precision + in-sample MWLE r)
# ---------------------------------------------------------------------------
def lb_full(path: Path):
    m = load(path)
    return m


floor0_paths = {
    "unidim": FLOOR0 / "unidim" / "leaderboard" / "metrics.json",
    "2-skill": FLOOR0 / "leaderboard" / "2_skills" / "metrics.json",
    "3-skill": FLOOR0 / "leaderboard" / "3_skills" / "metrics.json",
}
floor12_paths = {
    "unidim": BASE / "leaderboard" / "unidim" / "metrics.json",
    "2-skill": BASE / "leaderboard" / "2_skills" / "metrics.json",
    "3-skill": BASE / "leaderboard" / "3_skills" / "metrics.json",
}


def corr_axis(m: dict) -> str:
    # unidim uses 'overall'; multi uses 'correctness'
    return "overall" if "overall" in m["recovery"]["theta_mwle"] else "correctness"


drows = []
for model in ("unidim", "2-skill", "3-skill"):
    m0 = lb_full(floor0_paths[model])
    m1 = lb_full(floor12_paths[model])
    ca0 = corr_axis(m0)
    ca1 = corr_axis(m1)
    has_scaf = "scaffolding" in m0["recovery"]["theta_mwle"]
    drows.append({
        "model": model,
        "scen_mean_f0": round(m0["scenarios_administered"]["mean"], 1),
        "scen_mean_f12": round(m1["scenarios_administered"]["mean"], 1),
        "precision_f0": f'{m0["precision_reached"]}/{m0["n_models"]}',
        "precision_f12": f'{m1["precision_reached"]}/{m1["n_models"]}',
        "mwle_corr_r_f0": round(m0["recovery"]["theta_mwle"][ca0]["r"], 3),
        "mwle_corr_r_f12": round(m1["recovery"]["theta_mwle"][ca1]["r"], 3),
        "mwle_scaf_r_f0": round(m0["recovery"]["theta_mwle"]["scaffolding"]["r"], 3) if has_scaf else "",
        "mwle_scaf_r_f12": round(m1["recovery"]["theta_mwle"]["scaffolding"]["r"], 3) if has_scaf else "",
    })

diff_df = pd.DataFrame(drows)
diff_df.to_csv(BASE / "diff_floor0_vs_floor12.csv", index=False)
print("=" * 78)
print("D. Diff floor-0 vs floor-12 (leaderboard; MWLE r is in-sample recovery)")
print("=" * 78)
print(diff_df.to_string(index=False))
print("\nWROTE:")
for f in ("model_comparison_1v2v3_mwle_wide.csv", "diff_floor0_vs_floor12.csv",
          "se_target_table_2skill.csv", "se_target_table_3skill.csv"):
    print("  ", BASE / f)
