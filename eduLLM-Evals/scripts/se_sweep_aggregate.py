"""Aggregate the se_target sweep into one summary table per bank.

Reads metrics.json (scenarios/criteria/precision/recovery) and cat_per_model.csv
(per-model final online SE) from each sweep output dir, reusing the existing
se=0.30 leaderboard run for the 0.30 row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "regenerated_figures" / "scenario_level_115"
SWEEP = BASE / "se_sweep"

# param-uncertainty SE_param floor (established elsewhere; referenced, not recomputed)
PARAM_FLOOR = {"2_skills": 0.213, "3_skills": 0.254}  # correctness floor

BANKS = {
    "2_skills": ["correctness", "scaffolding"],
    "3_skills": ["correctness", "scaffolding", "presentation"],
}

SE_TARGETS = [0.20, 0.25, 0.30, 0.35]


def dir_for(bank: str, se: float) -> Path:
    if abs(se - 0.30) < 1e-9:
        return BASE / "leaderboard" / bank
    return SWEEP / bank / f"se{se:.2f}"


def build(bank: str) -> pd.DataFrame:
    dims = BANKS[bank]
    rows = []
    for se in SE_TARGETS:
        d = dir_for(bank, se)
        with (d / "metrics.json").open(encoding="utf-8") as fh:
            m = json.load(fh)
        per = pd.read_csv(d / "cat_per_model.csv")
        rec = m["recovery"]
        row = {
            "se_target": se,
            "precision_reached": f"{m['precision_reached']}/{m['n_models']}",
            "scenarios_mean": round(m["scenarios_administered"]["mean"], 1),
            "scenarios_median": m["scenarios_administered"]["median"],
            "criteria_mean": round(m["criteria_administered"]["mean"], 1),
            "batch_corr_slope": round(rec["theta_batch"]["correctness"]["slope"], 3),
            "batch_corr_r": round(rec["theta_batch"]["correctness"]["r"], 3),
            "mwle_corr_slope": round(rec["theta_mwle"]["correctness"]["slope"], 3),
            "mwle_corr_r": round(rec["theta_mwle"]["correctness"]["r"], 3),
            "batch_scaf_slope": round(rec["theta_batch"]["scaffolding"]["slope"], 3),
            "batch_scaf_r": round(rec["theta_batch"]["scaffolding"]["r"], 3),
        }
        if "presentation" in dims:
            row["batch_pres_slope"] = round(rec["theta_batch"]["presentation"]["slope"], 3)
            row["batch_pres_r"] = round(rec["theta_batch"]["presentation"]["r"], 3)
        for dd in dims:
            row[f"median_final_se_{dd}"] = round(float(per[f"final_se_{dd}"].median()), 4)
        # approximate total correctness SE with the se_target-independent param floor
        floor = PARAM_FLOOR[bank]
        mfse_c = float(per["final_se_correctness"].median())
        row["approx_total_se_correctness"] = round((mfse_c**2 + floor**2) ** 0.5, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    frames = {}
    for bank in BANKS:
        df = build(bank)
        df.insert(0, "bank", bank)
        frames[bank] = df

    combined = pd.concat(frames.values(), ignore_index=True)
    out = SWEEP / "summary.csv"
    combined.to_csv(out, index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    for bank, df in frames.items():
        floor = PARAM_FLOOR[bank]
        print("=" * 110)
        print(f"BANK: {bank}   (correctness SE_param floor = {floor})")
        print("=" * 110)
        print(df.drop(columns=["bank"]).to_string(index=False))
        print()
        print("  approx total correctness SE = sqrt(median_final_se_correctness^2 + "
              f"{floor}^2):")
        for _, r in df.iterrows():
            print(f"    se_target={r['se_target']:.2f}: "
                  f"sqrt({r['median_final_se_correctness']:.4f}^2 + {floor}^2) "
                  f"= {r['approx_total_se_correctness']:.4f}")
        print()
    print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
