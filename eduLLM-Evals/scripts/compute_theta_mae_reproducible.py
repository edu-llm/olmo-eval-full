"""Theta-MAE for the reproducible TutorEval k-fold export.

Reads the persisted per-model out-of-sample pairs under
``regenerated_figures/scenario_level/kfold_reproducible/tutoreval_{unidim,2skill}/oos_per_model.csv``
(produced with ``--scenarios data/TutorEval/scenarios_final.jsonl``, which reproduces the
published k-fold numbers exactly) and computes, for each estimator (online / batch / mwle)
and each axis/skill,

    Theta-MAE = mean_i | theta_estimate_i - theta_ref_i |   (i over the 52 models).

For the MWLE rows it also reports a 95% percentile CI from a model bootstrap
(B=2000, seed 20260803), matching the CI method used for slope/means elsewhere.

The Theta-MAE rows are appended (idempotently) to
``staging/tutoreval_calibration/tutoreval_confidence_intervals_reproducible.csv`` in the same
column schema; any pre-existing ``theta_mae`` rows are replaced, so re-running is safe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "regenerated_figures" / "scenario_level" / "kfold_reproducible"
OUT_CSV = (
    ROOT / "staging" / "tutoreval_calibration" / "tutoreval_confidence_intervals_reproducible.csv"
)

B = 2000
SEED = 20260803
ESTIMATORS = ("online", "batch", "mwle")
AXES = {
    "unidim": ["ability"],
    "2skill": ["conceptual_understanding", "quantitative_procedural"],
}
COLUMNS = [
    "stat",
    "axis",
    "skill",
    "estimator",
    "point",
    "ci_lo",
    "ci_hi",
    "method",
    "n",
    "notes",
]


def load_regen(axis: str) -> pd.DataFrame:
    name = "tutoreval_unidim" if axis == "unidim" else "tutoreval_2skill"
    return pd.read_csv(REPRO / name / "oos_per_model.csv")


def theta_mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(y - x)))


def boot_mae(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = x.size
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = theta_mae(x[idx], y[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def theta_mae_rows() -> list[dict]:
    rows: list[dict] = []
    for axis, dims in AXES.items():
        df = load_regen(axis)
        for skill in dims:
            x = df[f"theta_ref_{skill}"].to_numpy(float)
            for est in ESTIMATORS:
                y = df[f"theta_{est}_{skill}"].to_numpy(float)
                mae = theta_mae(x, y)
                if est == "mwle":
                    rng = np.random.default_rng(SEED)
                    lo, hi = boot_mae(x, y, rng)
                    rows.append(
                        {
                            "stat": "theta_mae",
                            "axis": axis,
                            "skill": skill,
                            "estimator": est,
                            "point": round(mae, 4),
                            "ci_lo": round(lo, 4),
                            "ci_hi": round(hi, 4),
                            "method": "bootstrap_models",
                            "n": 52,
                            "notes": f"mean|theta_est-theta_ref| over 52 models on "
                            f"kfold_reproducible; B={B}, seed={SEED}",
                        }
                    )
                else:
                    rows.append(
                        {
                            "stat": "theta_mae",
                            "axis": axis,
                            "skill": skill,
                            "estimator": est,
                            "point": round(mae, 4),
                            "ci_lo": "",
                            "ci_hi": "",
                            "method": "point",
                            "n": 52,
                            "notes": "mean|theta_est-theta_ref| over 52 models on "
                            "kfold_reproducible",
                        }
                    )
    return rows


def main() -> int:
    rows = theta_mae_rows()
    new = pd.DataFrame(rows, columns=COLUMNS)

    if OUT_CSV.is_file():
        existing = pd.read_csv(OUT_CSV)
        existing = existing[existing["stat"] != "theta_mae"]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(OUT_CSV, index=False)

    print(f"appended {len(new)} theta_mae rows -> {OUT_CSV}\n")
    hdr = f"{'axis':7} {'skill':25} {'estimator':9} {'Theta-MAE':>9} {'95% CI (MWLE)':>22}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci_lo']}, {r['ci_hi']}]" if r["ci_lo"] != "" else ""
        print(f"{r['axis']:7} {r['skill']:25} {r['estimator']:9} {r['point']:9.4f} {ci:>22}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
