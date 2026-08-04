"""Per-model test-length histograms for the TutorEval Reduction section.

For the out-of-sample k-fold run, each of the N=52 tutor models stops the adaptive
test after some number of *scenarios* (and *criteria*) administered. This driver reads
the per-model length columns already exported in

    regenerated_figures/scenario_level/kfold_reproducible/tutoreval_{unidim,2skill}/oos_per_model.csv

and draws, per bank, a histogram of ``scenarios_administered`` and a companion of
``criteria_administered``. Each panel marks the mean and shades a 95% CI band from a
model bootstrap (B=2000, seed 20260803) that reproduces the CIs quoted in the report.

This is a thin, read-only plotting driver: it does NOT run the CAT engine and does NOT
modify any existing results, CSVs, or figures.

Usage
-----
    uv run --project .. python scripts/plot_tutoreval_length_hist.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "staging" / "_length_hist" / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCEN = ROOT / "regenerated_figures" / "scenario_level" / "kfold_reproducible"

BOOT_B = 2000
BOOT_SEED = 20260803

BANKS = {
    "tutoreval_unidim": "unidimensional",
    "tutoreval_2skill": "two-skill",
}
UNITS = {
    "scenarios_administered": ("scenarios administered", "#1f77b4"),
    "criteria_administered": ("criteria administered", "#2ca02c"),
}


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """Percentile 95% CI of the mean from a model bootstrap (B=2000, fixed seed)."""
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.array(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(BOOT_B)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def plot_hist(values: np.ndarray, col: str, bank_label: str, out_png: Path) -> dict:
    xlabel, color = UNITS[col]
    mean = float(values.mean())
    lo, hi = bootstrap_ci(values)

    vmin, vmax = int(values.min()), int(values.max())
    bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.hist(values, bins=bins, color=color, alpha=0.8, edgecolor="white")
    ax.axvspan(lo, hi, color=color, alpha=0.18, label=f"95% CI [{lo:.1f}, {hi:.1f}]")
    ax.axvline(mean, color="#333333", lw=2.0, ls="--", label=f"mean = {mean:.1f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of models")
    ax.set_title(
        f"TutorEval {bank_label}: per-model {xlabel}\n"
        f"(out-of-sample k-fold, N={values.size} models)"
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(values.size)}


def main() -> int:
    for bank, bank_label in BANKS.items():
        df = pd.read_csv(SCEN / bank / "oos_per_model.csv")
        for col in UNITS:
            unit = col.split("_")[0]
            out_png = SCEN / bank / "figures" / f"length_hist_{unit}.png"
            stats = plot_hist(df[col].to_numpy(dtype=float), col, bank_label, out_png)
            print(
                f"{bank} {col}: mean={stats['mean']:.3f} "
                f"CI=[{stats['ci_lo']:.2f}, {stats['ci_hi']:.2f}] N={stats['n']}"
            )
            print(f"wrote -> {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
