"""Final-config figure: MWLE-only out-of-sample recovery (correctness + scaffolding).

A single MWLE panel per skill (no online/batch), for the "final config" section — the
estimator comparison (3-panel online/batch/MWLE) lives in the config-decision section.
Reads the k-fold OOS per-model estimates.

Usage
-----
    python scripts/fig_mwle_oos_recovery.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _stats(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    return float(np.corrcoef(x, y)[0, 1]), float(np.polyfit(x, y, 1)[0])


def _boot_fit(x, y, grid, b=2000, seed=0):
    """Bootstrap the OLS fit + r/slope by resampling the model pairs with replacement.
    Returns (band_lo, band_hi over grid), (r_lo, r_hi), (slope_lo, slope_hi)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    rng = np.random.default_rng(seed)
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
    base = ROOT / "regenerated_figures" / "scenario_level_115_min12"
    p.add_argument("--oos", type=Path, default=base / "kfold" / "2_skills" / "oos_per_model.csv")
    p.add_argument("--skills", nargs="+", default=["correctness", "scaffolding"])
    p.add_argument("--out", type=Path, default=base / "figures" / "mwle_oos_recovery_2skill.png")
    args = p.parse_args()

    df = pd.read_csv(args.oos)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(args.skills), figsize=(4.8 * len(args.skills), 4.6),
                             squeeze=False)
    for ax, sk in zip(axes[0], args.skills):
        x = df[f"theta_ref_{sk}"].to_numpy(float)
        y = df[f"theta_mwle_{sk}"].to_numpy(float)
        r, slope = _stats(x, y)
        ok = np.isfinite(x) & np.isfinite(y)
        s_fit, c_fit = np.polyfit(x[ok], y[ok], 1)
        lo = min(np.nanmin(x), np.nanmin(y)) - 0.3
        hi = max(np.nanmax(x), np.nanmax(y)) + 0.3
        grid = np.linspace(lo, hi, 100)
        band_lo, band_hi, (r_lo, r_hi), (sl_lo, sl_hi) = _boot_fit(x, y, grid)
        ax.fill_between(grid, band_lo, band_hi, color="#ff7f0e", alpha=0.18,
                        label="95% CI (fit)", zorder=0)
        ax.scatter(x, y, s=26, alpha=0.75, edgecolor="k", linewidth=0.3, zorder=2)
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x", zorder=1)
        ax.plot(grid, s_fit * grid + c_fit, color="#ff7f0e", lw=1.6,
                label=f"OLS fit (slope {slope:.2f})", zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full-bank EAP ability ({sk})")
        ax.set_ylabel(f"CAT MWLE ability ({sk})")
        ax.set_title(f"{sk}:  r = {r:.3f} [{r_lo:.3f}, {r_hi:.3f}]\n"
                     f"slope = {slope:.3f} [{sl_lo:.3f}, {sl_hi:.3f}]", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("MWLE out-of-sample ability recovery (2-skill, se=0.30, min_scenarios=12)",
                 fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote -> {args.out}")
    for sk in args.skills:
        r, slope = _stats(df[f"theta_ref_{sk}"].to_numpy(float),
                          df[f"theta_mwle_{sk}"].to_numpy(float))
        print(f"  {sk}: r={r:.3f} slope={slope:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
