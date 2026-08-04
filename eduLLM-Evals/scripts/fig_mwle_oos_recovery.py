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
        ax.scatter(x, y, s=26, alpha=0.75, edgecolor="k", linewidth=0.3)
        lo = min(np.nanmin(x), np.nanmin(y)) - 0.3
        hi = max(np.nanmax(x), np.nanmax(y)) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full-bank EAP ability ({sk})")
        ax.set_ylabel(f"CAT MWLE ability ({sk})")
        ax.set_title(f"{sk}:  r = {r:.3f},  slope = {slope:.3f}")
        ax.legend(loc="upper left", fontsize=9)
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
