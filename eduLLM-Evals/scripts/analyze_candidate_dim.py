"""Candidate-dimension health for the experimental [correctness, scaffolding,
<candidate>] M2PL fit -- read-only diagnostic. GENERALIZATION of
``scripts/analyze_motivation_dim.py``.

Reads the experimental Q bank (slot ``scaffolding`` holds the CANDIDATE bit), the
response matrix, and the run's per-criterion CSV (whose ``a_scaffolding`` column
holds the fitted CANDIDATE discrimination), then reports:

* how many criteria were ASSIGNED candidate=1 (over the response-matrix columns),
* how many SURVIVED the empty / zero-variance / all-zero-Q filters,
* candidate items' pass-rate distribution across the fleet, and
* the fitted candidate-discrimination distribution (near-zero / extreme / negative
  tallies) alongside correctness & scaffolding for context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTREME_A = 6.0


def load_bank_q(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qm = rec.get("q_mapping") or {}
            rows.append({
                "criterion_id": rec.get("criterion_id"),
                "correctness": int(qm.get("content", 0)),
                "scaffolding": int(qm.get("diagnosis", 0)),
                "candidate": int(qm.get("scaffolding", 0)),
            })
    return pd.DataFrame(rows)


def dist(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(s.size),
        "min": round(float(s.min()), 4),
        "p25": round(float(s.quantile(0.25)), 4),
        "median": round(float(s.median()), 4),
        "p75": round(float(s.quantile(0.75)), 4),
        "max": round(float(s.max()), 4),
        "mean": round(float(s.mean()), 4),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--matrix", type=Path,
                   default=ROOT / "staging" / "response_matrix.csv")
    p.add_argument("--csv", type=Path, required=True,
                   help="calibration_mirt.csv from the candidate's 3-dim fit.")
    args = p.parse_args()

    name = args.candidate
    bankq = load_bank_q(args.bank)
    mat = pd.read_csv(args.matrix, index_col=0)
    fit = pd.read_csv(args.csv)
    fit = fit.rename(columns={"a_content": "a_correctness",
                              "a_diagnosis": "a_scaffolding_skill",
                              "a_scaffolding": "a_candidate"})

    matrix_cols = set(mat.columns)
    fitted_ids = set(fit["criterion_id"])

    q_cols = bankq[bankq["criterion_id"].isin(matrix_cols)]
    assigned = q_cols[q_cols["candidate"] == 1]["criterion_id"].tolist()
    n_assigned = len(assigned)

    cand_fitted = [c for c in assigned if c in fitted_ids]
    n_survived = len(cand_fitted)
    n_dropped = n_assigned - n_survived

    present = [c for c in assigned if c in matrix_cols]
    pass_rates = mat[present].mean(axis=0, skipna=True)
    cand_survived_present = [c for c in cand_fitted if c in matrix_cols]
    pass_rates_surv = mat[cand_survived_present].mean(axis=0, skipna=True)

    fitm = fit[fit["criterion_id"].isin(cand_fitted)]
    a_cand = fitm["a_candidate"]

    print("=" * 72)
    print(f"{name.upper()}-DIMENSION HEALTH  (candidate [correctness, scaffolding, {name}])")
    print("=" * 72)
    print(f"assigned {name}=1 (matrix columns) : {n_assigned}")
    print(f"survived to the fit                    : {n_survived}")
    print(f"dropped (empty / all-fail / all-zero-Q): {n_dropped}")

    print(f"\n{name} items pass-rate across fleet (per-item mean over 82 models):")
    if len(present):
        print(f"  ASSIGNED  ({len(present)} present): "
              f"mean={float(pass_rates.mean()):.4f} median={float(pass_rates.median()):.4f} "
              f"min={float(pass_rates.min()):.4f} max={float(pass_rates.max()):.4f}")
    if len(cand_survived_present):
        print(f"  SURVIVED  ({len(cand_survived_present)}): "
              f"mean={float(pass_rates_surv.mean()):.4f} "
              f"median={float(pass_rates_surv.median()):.4f} "
              f"min={float(pass_rates_surv.min()):.4f} "
              f"max={float(pass_rates_surv.max()):.4f}")
    n_allfail_assigned = int((pass_rates == 0).sum())
    print(f"  assigned {name} items with 0% pass across ALL models (all-fail): "
          f"{n_allfail_assigned}")

    print(f"\nfitted {name.upper()} discrimination (a_candidate) distribution:")
    d = dist(a_cand)
    print(f"  {d}")
    n_neg = int((a_cand < 0).sum())
    n_near0 = int((a_cand.abs() < 0.2).sum())
    n_extreme = int((a_cand.abs() > EXTREME_A).sum())
    print(f"  negative loadings : {n_neg}/{len(a_cand)}")
    print(f"  |a| < 0.2 (near-zero, no discrimination): {n_near0}/{len(a_cand)}")
    print(f"  |a| > {EXTREME_A} (extreme/unstable): {n_extreme}/{len(a_cand)}")

    print("\nfor context -- correctness & scaffolding discrimination (fitted items):")
    corr_items = fit[fit["a_correctness"] != 0]["a_correctness"]
    scaff_items = fit[fit["a_scaffolding_skill"] != 0]["a_scaffolding_skill"]
    print(f"  correctness : {dist(corr_items)}")
    print(f"  scaffolding : {dist(scaff_items)}")
    print(f"  {name}  : {dist(a_cand)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
