#!/usr/bin/env python3
"""Single-panel grouped bars: held-out Pearson r for 1PL/2PL/3PL by benchmark at SE<=0.30.

Reads the 52-model pl_1_2_3_comparison.csv and plots only the SE=0.3 slice.
"""
import csv
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
D = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/pedagogy_feasibility/pl_1_2_3_comparison"
os.environ.setdefault("MPLCONFIGDIR", str(D / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCHES = ["pedagogy", "piqa", "socialiqa"]
MODELS = ["1PL", "2PL", "3PL"]
COLORS = {"1PL": "#55A868", "2PL": "#DD8452", "3PL": "#4C72B0"}

rows = list(csv.DictReader(open(D / "pl_1_2_3_comparison.csv")))
val_by = {}
for row in rows:
    if abs(float(row["se_stop"]) - 0.3) < 1e-9:
        val_by[(row["benchmark"], row["model_type"])] = row

x = np.arange(len(BENCHES))
w = 0.26


def grouped(key, ylabel, title, fmt, out_name, ylim=None):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for i, mt in enumerate(MODELS):
        vals = [float(val_by[(b, mt)][key]) if (b, mt) in val_by else np.nan for b in BENCHES]
        xs = x + (i - 1) * w
        ax.bar(xs, vals, w, label=mt, color=COLORS[mt], edgecolor="black", linewidth=0.5)
        for xi, v in zip(xs, vals):
            if np.isfinite(v):
                ax.annotate(fmt.format(v), (xi, v), ha="center", va="bottom",
                            fontsize=9, fontweight="bold", xytext=(0, 1), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHES, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="IRT model", loc="upper right")
    fig.tight_layout()
    out = D / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


grouped("r", "Pearson r (CAT-predicted vs actual full accuracy)",
        "Held-out Pearson r at SE <= 0.30 (1PL vs 2PL vs 3PL)\n40-model calibration pool, 12 held-out models",
        "{:.3f}", "pl_comparison_r_se0.3.png", ylim=(0, 1.0))
grouped("mean_items", "mean CAT items administered",
        "Mean CAT items at SE <= 0.30 (1PL vs 2PL vs 3PL)\n40-model calibration pool, 12 held-out models",
        "{:.1f}", "pl_comparison_items_se0.3.png")
