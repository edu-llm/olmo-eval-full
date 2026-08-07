"""Heatmaps for the EAP-posterior op-point grid study (reads per_cell_grid.csv).

Emits three floor x SE_ability-target heatmaps under the report folder:
  * heatmap_median_se_total_correctness.png
  * heatmap_median_length.png
  * heatmap_pct_reach.png
Kept separate from ``eap_stop_grid_study.py`` so plots can be regenerated without the
~5-min engine + dense-grid recompute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def heatmap(ax, df, value_col, floors, targets, title, fmt, cmap):
    grid = np.full((len(floors), len(targets)), np.nan)
    for i, fl in enumerate(floors):
        for j, se in enumerate(targets):
            sub = df[(df["floor"] == fl) & (np.isclose(df["se_target"], se))]
            if len(sub):
                grid[i, j] = float(sub[value_col].iloc[0])
    im = ax.imshow(grid, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(targets)), [f"{t:.2f}" for t in targets])
    ax.set_yticks(range(len(floors)), [str(f) for f in floors])
    ax.set_xlabel("SE_ability target")
    ax.set_ylabel("min_scenarios floor")
    ax.set_title(title)
    for i in range(len(floors)):
        for j in range(len(targets)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, format(grid[i, j], fmt), ha="center", va="center",
                        color="black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", type=Path,
                    default=ROOT / "reports/eap_stop_grid_tutorbench_2skill")
    args = ap.parse_args()
    df = pd.read_csv(args.report_dir / "per_cell_grid.csv")
    floors = sorted(df["floor"].unique())
    targets = sorted(df["se_target"].unique())

    specs = [
        ("median_se_total_correctness", "Median SE_total (correctness)",
         ".3f", "viridis_r", "heatmap_median_se_total_correctness.png"),
        ("median_len", "Median test length (scenarios)",
         ".0f", "magma_r", "heatmap_median_length.png"),
        ("pct_reach_all_skills", "% models reaching target (all skills, SD<=target)",
         ".2f", "viridis", "heatmap_pct_reach.png"),
    ]
    for value_col, title, fmt, cmap, fname in specs:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        heatmap(ax, df, value_col, floors, targets, title, fmt, cmap)
        fig.tight_layout()
        fig.savefig(args.report_dir / fname, dpi=140)
        plt.close(fig)
        print(f"wrote {args.report_dir / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
