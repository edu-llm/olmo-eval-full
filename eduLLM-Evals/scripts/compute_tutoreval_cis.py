"""Compute 95% confidence intervals for the reported TutorEval calibration statistics.

This is a thin, read-only reporting script. It does not refit any model or touch the
production analysis code; it consumes:

  * the published point estimates in
    ``regenerated_figures/scenario_level/kfold/tutoreval_{unidim,2skill}/metrics.json``;
  * per-model out-of-sample recovery pairs (theta_ref vs. each estimator) produced by a
    faithful deterministic re-run of ``scripts/scenario_kfold_estimator_cv.py`` (same seed
    ``20260729`` and config), stored under
    ``staging/tutoreval_calibration/_kfold_regen/tutoreval_{unidim,2skill}/oos_per_model.csv``;
  * per-criterion verifier votes for the Q-matrix Fleiss kappa in
    ``data/TutorEval/rubrics_qmatrix_verified.jsonl``.

Methods
-------
  * Recovery r (Pearson): Fisher z-transform CI (primary, anchored to the *published* r and
    n=52), plus a nonparametric bootstrap over the 52 models (B, seed) as a cross-check.
  * Slope, gap_worst12, mean scenarios/criteria: percentile bootstrap over the 52 models.
  * Fleiss kappa (conceptual / quantitative): percentile bootstrap over criteria (resample the
    per-criterion 3-rater label vectors).
  * Latent correlation rho: point estimate only; a CI needs person-bootstrap M2PL refits at
    N=52, which is out of scope here (noted, not computed).

Output: ``staging/tutoreval_calibration/tutoreval_confidence_intervals.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KFOLD = ROOT / "regenerated_figures" / "scenario_level" / "kfold"
REGEN = ROOT / "staging" / "tutoreval_calibration" / "_kfold_regen"
VERIFIED = ROOT / "data" / "TutorEval" / "rubrics_qmatrix_verified.jsonl"
OUT_CSV = ROOT / "staging" / "tutoreval_calibration" / "tutoreval_confidence_intervals.csv"

B = 2000
SEED = 20260803
ESTIMATORS = ("online", "batch", "mwle")
SKILLS = ("conceptual_understanding", "quantitative_procedural")
# published Fleiss kappa (from the verification manifest cited in the report)
PUBLISHED_KAPPA = {"conceptual_understanding": 0.686, "quantitative_procedural": 0.722}
PUBLISHED_RHO = 0.981


Z_95 = 1.959963984540054  # two-sided 95% normal critical value


def fisher_z_ci(r: float, n: int) -> tuple[float, float]:
    """Fisher z-transform 95% CI for a Pearson correlation."""
    from math import atanh, sqrt, tanh

    z = atanh(r)
    se = 1.0 / sqrt(n - 3)
    return tanh(z - Z_95 * se), tanh(z + Z_95 * se)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def gap_worst12(x: np.ndarray, y: np.ndarray) -> float:
    w = np.argsort(x)[:12]
    return float(np.mean(y[w] - x[w]))


def boot_pairs(x: np.ndarray, y: np.ndarray, fn, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI for a statistic of paired (x, y) resampled over models."""
    n = x.size
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = fn(x[idx], y[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_mean(counts: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = counts.size
    means = np.array([counts[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_published(axis: str) -> dict:
    name = "tutoreval_unidim" if axis == "unidim" else "tutoreval_2skill"
    return json.loads((KFOLD / name / "metrics.json").read_text())


def load_regen(axis: str) -> pd.DataFrame:
    name = "tutoreval_unidim" if axis == "unidim" else "tutoreval_2skill"
    return pd.read_csv(REGEN / name / "oos_per_model.csv")


def recovery_rows(axis: str, skill: str, dim_col: str, rows: list[dict]) -> None:
    """Append r (Fisher-z + bootstrap), slope, and gap_worst12 CI rows for one axis/skill."""
    pub = load_published(axis)
    df = load_regen(axis)
    x = df[f"theta_ref_{dim_col}"].to_numpy(float)
    for est in ESTIMATORS:
        y = df[f"theta_{est}_{dim_col}"].to_numpy(float)
        pub_stats = pub["oos_recovery"][est][skill]
        pub_r = pub_stats["r"]
        pub_slope = pub_stats["slope"]
        pub_gap = pub_stats["gap_worst12"]
        re_r, re_slope, re_gap = pearson(x, y), slope(x, y), gap_worst12(x, y)

        lo, hi = fisher_z_ci(pub_r, 52)
        rows.append(
            {
                "stat": "recovery_r",
                "axis": axis,
                "skill": skill,
                "estimator": est,
                "point": round(pub_r, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "method": "fisher_z",
                "n": 52,
                "notes": f"primary; anchored to published r; re-run r={re_r:.4f}",
            }
        )

        rng = np.random.default_rng(SEED)
        lo, hi = boot_pairs(x, y, pearson, rng)
        rows.append(
            {
                "stat": "recovery_r",
                "axis": axis,
                "skill": skill,
                "estimator": est,
                "point": round(pub_r, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "method": "bootstrap_models",
                "n": 52,
                "notes": f"cross-check on re-run pairs (B={B}, seed={SEED}); re-run r={re_r:.4f}",
            }
        )

        rng = np.random.default_rng(SEED)
        lo, hi = boot_pairs(x, y, slope, rng)
        rows.append(
            {
                "stat": "slope",
                "axis": axis,
                "skill": skill,
                "estimator": est,
                "point": round(pub_slope, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "method": "bootstrap_models",
                "n": 52,
                "notes": f"B={B}, seed={SEED}; re-run slope={re_slope:.4f}",
            }
        )

        rng = np.random.default_rng(SEED)
        lo, hi = boot_pairs(x, y, gap_worst12, rng)
        rows.append(
            {
                "stat": "gap_worst12",
                "axis": axis,
                "skill": skill,
                "estimator": est,
                "point": round(pub_gap, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "method": "bootstrap_models",
                "n": 52,
                "notes": f"tail statistic (worst-12), wide CI; re-run gap={re_gap:.4f}",
            }
        )


def count_rows(axis: str, rows: list[dict]) -> None:
    """Append mean-scenarios and mean-criteria CI rows for one axis."""
    pub = load_published(axis)
    df = load_regen(axis)
    for stat, col in (
        ("mean_criteria_administered", "criteria_administered"),
        ("mean_scenarios_administered", "scenarios_administered"),
    ):
        counts = df[col].to_numpy(float)
        rng = np.random.default_rng(SEED)
        lo, hi = boot_mean(counts, rng)
        pub_val = pub.get(stat)
        point = pub_val if pub_val is not None else float(counts.mean())
        note = (
            f"B={B}, seed={SEED}; re-run mean={counts.mean():.3f}"
            if pub_val is not None
            else f"not stored in metrics.json; point is re-run mean over 52 models (B={B}, "
            f"seed={SEED})"
        )
        rows.append(
            {
                "stat": stat,
                "axis": axis,
                "skill": "",
                "estimator": "",
                "point": round(float(point), 3),
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
                "method": "bootstrap_models",
                "n": 52,
                "notes": note,
            }
        )


def fleiss_kappa(rows_counts: np.ndarray, raters: int) -> float:
    """Fleiss' kappa for a fixed number of raters over two categories (matches verify_qmatrix)."""
    n = rows_counts.shape[0]
    total = n * raters
    p1 = rows_counts[:, 1].sum() / total
    p0 = rows_counts[:, 0].sum() / total
    pe = p0 * p0 + p1 * p1
    p_i = (rows_counts[:, 0] ** 2 + rows_counts[:, 1] ** 2 - raters) / (raters * (raters - 1))
    p_bar = p_i.mean()
    return float((p_bar - pe) / (1 - pe))


def load_fleiss_rows() -> dict[str, np.ndarray]:
    """Per-skill (n_zeros, n_ones) rows over criteria with all 3 raters present."""
    rows: dict[str, list[tuple[int, int]]] = {s: [] for s in SKILLS}
    with VERIFIED.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            votes = rec.get("verification", {}).get("votes")
            if not votes:
                continue
            for s in SKILLS:
                v = votes.get(s)
                if v is not None and len(v) == 3:
                    ones = int(sum(v))
                    rows[s].append((3 - ones, ones))
    return {s: np.array(v, dtype=int) for s, v in rows.items()}


def kappa_rows(rows: list[dict]) -> None:
    fleiss = load_fleiss_rows()
    for s in SKILLS:
        cr = fleiss[s]
        n = cr.shape[0]
        point = fleiss_kappa(cr, 3)
        rng = np.random.default_rng(SEED)
        vals = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, n)
            vals[b] = fleiss_kappa(cr[idx], 3)
        lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
        pub = PUBLISHED_KAPPA[s]
        rows.append(
            {
                "stat": "fleiss_kappa",
                "axis": "qmatrix",
                "skill": s,
                "estimator": "",
                "point": round(pub, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "method": "bootstrap_criteria",
                "n": n,
                "notes": f"3 raters (generator+2 verifiers); recomputed kappa={point:.4f} "
                f"(published {pub}); B={B}, seed={SEED}",
            }
        )


def rho_row(rows: list[dict]) -> None:
    rows.append(
        {
            "stat": "latent_correlation_rho",
            "axis": "2skill",
            "skill": "",
            "estimator": "",
            "point": PUBLISHED_RHO,
            "ci_lo": "",
            "ci_hi": "",
            "method": "not_computed",
            "n": 52,
            "notes": "CI not well-defined here; needs a person-bootstrap M2PL refit at N=52 "
            "(expensive, out of scope). Reported as a point estimate only.",
        }
    )


def main() -> int:
    rows: list[dict] = []
    recovery_rows("unidim", "ability", "ability", rows)
    for s in SKILLS:
        recovery_rows("2skill", s, s, rows)
    count_rows("unidim", rows)
    count_rows("2skill", rows)
    kappa_rows(rows)
    rho_row(rows)

    df = pd.DataFrame(
        rows,
        columns=[
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
        ],
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"wrote {len(df)} rows -> {OUT_CSV}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.drop(columns=["notes"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
