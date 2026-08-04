"""Paper figure: the unidimensional ("overall") ability vs the 2-skill axes.

Shows why one dimension is insufficient: the single composite tracks correctness almost
perfectly (r ~ +1.0) but is negatively related to scaffolding (r < 0), so collapsing to one
axis captures content correctness while inverting/obscuring scaffolding. Uses the full-bank
EAP reference ability (floor-independent) so it matches the reported composite-vs-axes
correlations.

Usage
-----
    python scripts/fig_composite_vs_axes.py \
        --unidim regenerated_figures/scenario_level_115_min12/leaderboard/unidim/cat_per_model.csv \
        --twoskill regenerated_figures/scenario_level_115_min12/leaderboard/2_skills/cat_per_model.csv \
        --out regenerated_figures/scenario_level_115_min12/figures/composite_vs_axes.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def _boot_fit(x, y, grid, b=2000, seed=0):
    """Bootstrap the OLS fit + r/slope by resampling model pairs with replacement.
    Returns (band_lo, band_hi over grid), (r_lo, r_hi), (slope_lo, slope_hi)."""
    rng = np.random.default_rng(seed)
    n = x.size
    preds = np.empty((b, grid.size))
    rs = np.empty(b)
    slopes = np.empty(b)
    for i in range(b):
        idx = rng.integers(0, n, n)
        s, c = np.polyfit(x[idx], y[idx], 1)
        preds[i] = s * grid + c
        slopes[i] = s
        rs[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    band = np.percentile(preds, [2.5, 97.5], axis=0)
    return band[0], band[1], np.percentile(rs, [2.5, 97.5]), np.percentile(slopes, [2.5, 97.5])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ROOT / "regenerated_figures" / "scenario_level_115_min12" / "leaderboard"
    p.add_argument("--unidim", type=Path, default=d / "unidim" / "cat_per_model.csv")
    p.add_argument("--twoskill", type=Path, default=d / "2_skills" / "cat_per_model.csv")
    p.add_argument("--out", type=Path,
                   default=ROOT / "regenerated_figures" / "scenario_level_115_min12"
                   / "figures" / "composite_vs_axes.png")
    p.add_argument("--theta-col", default="theta_full",
                   help="which theta to use: theta_full (reference), theta_mwle, ...")
    args = p.parse_args()

    uni = pd.read_csv(args.unidim)
    two = pd.read_csv(args.twoskill)
    m = uni.merge(two, on="model", suffixes=("_uni", "_2sk"))
    overall = m[f"{args.theta_col}_overall"].to_numpy(float)
    corr = m[f"{args.theta_col}_correctness"].to_numpy(float)
    scaf = m[f"{args.theta_col}_scaffolding"].to_numpy(float)
    r_c, r_s = _pearson(overall, corr), _pearson(overall, scaf)
    print(f"n={len(m)}  overall-vs-correctness r={r_c:.3f}  overall-vs-scaffolding r={r_s:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, y, lab, r in ((ax1, corr, "2-skill correctness", r_c),
                          (ax2, scaf, "2-skill scaffolding", r_s)):
        ax.scatter(overall, y, s=26, alpha=0.75, edgecolor="k", linewidth=0.3, zorder=2)
        b, a = np.polyfit(overall, y, 1)
        grid = np.linspace(overall.min(), overall.max(), 100)
        band_lo, band_hi, (r_lo, r_hi), (sl_lo, sl_hi) = _boot_fit(overall, y, grid)
        ax.fill_between(grid, band_lo, band_hi, color="crimson", alpha=0.15, zorder=0,
                        label="95% CI (fit)")
        ax.plot(grid, b * grid + a, ls="--", color="crimson", lw=1.3, zorder=3,
                label=f"fit (r = {r:+.3f})")
        ax.set_xlabel("unidimensional 'overall' ability (logits)")
        ax.set_ylabel(f"{lab} ability (logits)")
        ax.set_title(f"overall vs {lab.split()[-1]}\n"
                     f"r = {r:+.3f} [{r_lo:+.3f}, {r_hi:+.3f}], "
                     f"slope = {b:+.3f} [{sl_lo:+.3f}, {sl_hi:+.3f}]", fontsize=9)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("A single 'overall' axis = correctness, and inverts scaffolding", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
