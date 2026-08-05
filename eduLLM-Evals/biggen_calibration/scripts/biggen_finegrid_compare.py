"""Task B: quantify the coarse -> fine EAP-grid fix and emit additive artifacts.

BiGGen's full-bank EAP reference (exp-05 recovery X-axis and the exp-07/08 leaderboard)
was computed on a COARSE uniform grid (eap_grid=61 over [-6,6] -> node spacing 0.2). With
a razor-sharp full-bank posterior the EAP mean collapses onto the nearest node, so the
recovery reference theta BANDS at multiples of 0.2. The fix re-runs the SAME pipelines
(``scenario_kfold_estimator_cv.py`` and ``scenario_param_uncertainty.py``) with a FINE
uniform grid (eap_grid=3201 over [-8,8], std-normal prior); the coarse mode stays reachable
behind the identical ``--eap-grid/--range`` flags.

This script does NOT re-run any engine. It reads the committed (coarse) outputs and the
fine outputs produced by those two scripts, quantifies the before/after change (banding,
r, slope, theta-MAE for recovery; theta / SE_total deltas for the leaderboard), writes the
comparison JSON, a de-banding before/after scatter, a single fine MWLE recovery scatter,
and a fine leaderboard CSV in the exp-08 schema. All fine artifacts are ADDITIVE
(``*_finegrid.*`` / ``_finegrid/`` subdirs); the committed of-record files are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "biggen_calibration" / "experiments"


def banding_metrics(theta: np.ndarray, spacing: float = 0.2, tol: float = 5e-3) -> dict:
    """How strongly ``theta`` snaps to a ``spacing`` grid (the quantization symptom)."""
    nearest = np.round(theta / spacing) * spacing
    dist = np.abs(theta - nearest)
    return {
        "n": int(theta.size),
        "n_unique_round4": int(np.unique(np.round(theta, 4)).size),
        "frac_within_tol_of_0p2_grid": round(float(np.mean(dist <= tol)), 3),
        "mean_abs_dist_to_0p2_grid": round(float(np.mean(dist)), 5),
        "median_abs_dist_to_0p2_grid": round(float(np.median(dist)), 5),
    }


def recovery_stats(ref: np.ndarray, est: np.ndarray) -> dict:
    ok = np.isfinite(ref) & np.isfinite(est)
    ref, est = ref[ok], est[ok]
    slope, intercept = np.polyfit(ref, est, 1)
    return {
        "r": round(float(np.corrcoef(ref, est)[0, 1]), 4),
        "slope": round(float(slope), 4),
        "theta_mae": round(float(np.mean(np.abs(est - ref))), 4),
        "n": int(ref.size),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", default="general")
    args = p.parse_args()
    d = args.dim

    exp05 = BASE / "05_oos_recovery"
    exp08 = BASE / "08_leaderboard"
    exp07 = BASE / "07_parameter_uncertainty"

    # ---- exp-05 recovery: coarse (committed) vs fine ----
    coarse = pd.read_csv(exp05 / "oos_per_model.csv").sort_values("model").reset_index(drop=True)
    fine = pd.read_csv(exp05 / "_finegrid" / "oos_per_model.csv").sort_values("model").reset_index(drop=True)
    assert (coarse["model"].values == fine["model"].values).all(), "model sets differ"

    ref_c = coarse[f"theta_ref_{d}"].to_numpy()
    ref_f = fine[f"theta_ref_{d}"].to_numpy()
    mwle_c = coarse[f"theta_mwle_{d}"].to_numpy()
    mwle_f = fine[f"theta_mwle_{d}"].to_numpy()

    rec = {
        "before_coarse_grid": {"eap_grid": 61, "range": 6.0,
                               **recovery_stats(ref_c, mwle_c),
                               "reference_banding": banding_metrics(ref_c)},
        "after_fine_grid": {"eap_grid": 3201, "range": 8.0,
                            **recovery_stats(ref_f, mwle_f),
                            "reference_banding": banding_metrics(ref_f)},
        "mwle_theta_unchanged_max_abs_diff": round(float(np.max(np.abs(mwle_f - mwle_c))), 6),
        "reference_theta_max_abs_shift": round(float(np.max(np.abs(ref_f - ref_c))), 4),
        "reference_theta_mean_abs_shift": round(float(np.mean(np.abs(ref_f - ref_c))), 4),
    }

    # ---- exp-08 leaderboard: coarse (committed) vs fine ----
    lb_c = pd.read_csv(exp08 / "results.csv")
    fine_lb = pd.read_csv(exp07 / "_finegrid" / "leaderboard_se_components.csv")
    fine_rank = fine_lb[["model", f"theta_{d}", f"se_posterior_{d}", f"se_total_{d}"]].copy()
    fine_rank.columns = ["model", "theta", "se_ability", "se_total"]
    fine_rank = fine_rank.sort_values("theta", ascending=False).reset_index(drop=True)
    fine_rank.insert(len(fine_rank.columns), "rank", np.arange(1, len(fine_rank) + 1))
    fine_rank.to_csv(exp08 / "results_finegrid.csv", index=False)

    merged = lb_c.merge(fine_rank, on="model", suffixes=("_coarse", "_fine"))
    theta_delta = (merged["theta_fine"] - merged["theta_coarse"]).abs()
    setot_delta = (merged["se_total_fine"] - merged["se_total_coarse"]).abs()
    rank_spearman = float(pd.Series(merged["rank_coarse"]).corr(
        pd.Series(merged["rank_fine"]), method="spearman"))
    n_rank_changed = int((merged["rank_coarse"] != merged["rank_fine"]).sum())
    max_rank_move = int((merged["rank_coarse"] - merged["rank_fine"]).abs().max())

    leaderboard = {
        "note": ("exp-08 leaderboard theta is the CAT-ADMINISTERED-set EAP (n_admin ~30-68 "
                 "criteria), NOT the full-bank reference; its posterior is not razor-sharp, "
                 "so it is already continuous (not banded). The fine grid mainly refines the "
                 "per-model SE (the coarse 0.2 spacing slightly under-resolved SE_param)."),
        "theta_max_abs_delta": round(float(theta_delta.max()), 4),
        "theta_mean_abs_delta": round(float(theta_delta.mean()), 4),
        "se_total_max_abs_delta": round(float(setot_delta.max()), 4),
        "se_total_mean_abs_delta": round(float(setot_delta.mean()), 4),
        "rank_spearman_coarse_vs_fine": round(rank_spearman, 5),
        "n_ranks_changed": n_rank_changed,
        "max_rank_move": max_rank_move,
    }

    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "B -- de-quantize the full-bank EAP reference (coarse uniform -> fine uniform)",
        "diagnosis": {
            "quantized": True,
            "where": ("exp-05 recovery reference theta (full-bank EAP). The full-bank "
                      "posterior is razor-sharp, so EAP on a 0.2-spaced grid collapses onto "
                      "nodes -> visible 0.2 banding on the recovery X-axis."),
            "coarse_grid": {"eap_grid": 61, "range": 6.0, "node_spacing": 0.2,
                            "source_script": "scripts/scenario_kfold_estimator_cv.py "
                                             "(default --eap-grid 61 --range 6)"},
            "fine_grid": {"eap_grid": 3201, "range": 8.0, "node_spacing": 0.005,
                          "prior": "standard-normal", "mirrors": "Bridge scenario_recovery_final "
                          "fine uniform EAP reference"},
        },
        "recovery_exp05": rec,
        "leaderboard_exp08": leaderboard,
        "verdict": (
            "Banding CONFIRMED on the coarse full-bank reference and REMOVED on the fine "
            f"grid (unique reference values {rec['before_coarse_grid']['reference_banding']['n_unique_round4']} "
            f"-> {rec['after_fine_grid']['reference_banding']['n_unique_round4']} of "
            f"{rec['after_fine_grid']['n']}; fraction snapped to the 0.2 grid "
            f"{rec['before_coarse_grid']['reference_banding']['frac_within_tol_of_0p2_grid']} "
            f"-> {rec['after_fine_grid']['reference_banding']['frac_within_tol_of_0p2_grid']}). "
            "The MWLE recovery HEADLINE is robust: r "
            f"{rec['before_coarse_grid']['r']} -> {rec['after_fine_grid']['r']}, slope "
            f"{rec['before_coarse_grid']['slope']} -> {rec['after_fine_grid']['slope']}, "
            f"theta-MAE {rec['before_coarse_grid']['theta_mae']} -> "
            f"{rec['after_fine_grid']['theta_mae']} (MWLE is grid-free; max |dtheta_mwle| "
            f"{rec['mwle_theta_unchanged_max_abs_diff']}). The exp-08 leaderboard ranking is "
            f"essentially unchanged (Spearman {leaderboard['rank_spearman_coarse_vs_fine']})."),
    }
    (exp05 / "finegrid_comparison.json").write_text(json.dumps(comparison, indent=2),
                                                    encoding="utf-8")
    (exp08 / "finegrid_comparison.json").write_text(json.dumps(comparison, indent=2),
                                                    encoding="utf-8")

    _figures(ref_c, mwle_c, ref_f, mwle_f, rec, d, exp05)

    print(json.dumps(comparison, indent=2))
    print(f"\nwrote -> {exp05/'finegrid_comparison.json'}")
    print(f"wrote -> {exp08/'results_finegrid.csv'}")
    return 0


def _figures(ref_c, mwle_c, ref_f, mwle_f, rec, d, exp05: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = exp05 / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # (1) de-banding before/after: reference theta on both grids vs MWLE
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), sharex=True, sharey=True)
    for ax, refv, tag, rr in ((axes[0], ref_c, "coarse (eap_grid=61)", rec["before_coarse_grid"]),
                              (axes[1], ref_f, "fine (eap_grid=3201)", rec["after_fine_grid"])):
        lo = min(refv.min(), mwle_c.min()) - 0.3
        hi = max(refv.max(), mwle_c.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1)
        ax.scatter(refv, mwle_c if tag.startswith("coarse") else mwle_f, s=28, alpha=0.8,
                   edgecolor="k", linewidth=0.3)
        for gx in np.arange(-5, 3.01, 0.2):
            ax.axvline(gx, color="#c1666b", alpha=0.08, lw=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("full-bank EAP reference theta")
        ax.set_title(f"{tag}\nr={rr['r']} slope={rr['slope']} "
                     f"snap-to-0.2={rr['reference_banding']['frac_within_tol_of_0p2_grid']}",
                     fontsize=9)
    axes[0].set_ylabel("CAT MWLE theta")
    fig.suptitle("BiGGen exp-05 reference de-quantization (pink lines = 0.2 grid)", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / "finegrid_debanding_general.png", dpi=140)
    plt.close(fig)

    # (2) single fine MWLE recovery scatter (mirrors oos_recovery_scatter_general.png)
    xs = np.linspace(ref_f.min(), ref_f.max(), 60)
    slope, intercept = np.polyfit(ref_f, mwle_f, 1)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo = min(ref_f.min(), mwle_f.min()) - 0.3
    hi = max(ref_f.max(), mwle_f.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.plot(xs, slope * xs + intercept, color="#d95f0e", lw=1.5, label=f"OLS slope={slope:.3f}")
    ax.scatter(ref_f, mwle_f, s=30, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("full-bank EAP reference theta (fine uniform grid)")
    ax.set_ylabel("CAT MWLE theta")
    ax.set_title(f"BiGGen scenario OOS recovery @ LOCKED (fine EAP ref)\n"
                 f"r={rec['after_fine_grid']['r']}, slope={rec['after_fine_grid']['slope']} "
                 f"(n={ref_f.size})", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(exp05 / "_finegrid" / "figures" / "oos_recovery_scatter_general.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
