#!/usr/bin/env python3
"""Single-panel SE<=0.30 version of the pedagogy 66-model controlled figure.

40-model vs 66-model calibration pool, Pearson r for 1PL/2PL/3PL, SE<=0.30 only.
"""
import csv
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
D = REPO / ("AdaptiveTesting/Research/01_MCQ_ATLAS/data/pedagogy_feasibility/"
            "pl_1_2_3_comparison/pedagogy_expanded78")
os.environ.setdefault("MPLCONFIGDIR", str(D / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["1PL", "2PL", "3PL"]
rows = [r for r in csv.DictReader(open(D / "pl_1_2_3_expanded_controlled.csv"))
        if abs(float(r["se_stop"]) - 0.3) < 1e-9]
by = {r["model_type"]: r for r in rows}


def plot(k40, k66, ylabel, title, fmt, out_name, ylim=None):
    x = np.arange(len(MODELS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for i, (key, label, color) in enumerate(
        [(k40, "40-model pool", "#BBBBBB"), (k66, "66-model pool", "#4C72B0")]
    ):
        ys = [float(by[mt][key]) for mt in MODELS]
        xs = x + (i - 0.5) * w
        ax.bar(xs, ys, w, label=label, color=color, edgecolor="black", linewidth=0.6)
        for xi, v in zip(xs, ys):
            ax.annotate(fmt.format(v), (xi, v), ha="center", va="bottom",
                        fontsize=9, fontweight="bold", xytext=(0, 1), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("IRT model", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="calibration pool", loc="upper right")
    fig.tight_layout()
    out = D / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


plot("r_40", "r_66", "Pearson r (CAT-predicted vs actual full accuracy)",
     "Pedagogy 1PL vs 2PL vs 3PL at SE <= 0.30\ncontrolled 40 vs 66 calibration pool, 12 held-out",
     "{:.3f}", "pl_1_2_3_expanded78_se0.3.png", ylim=(0, 1.0))
plot("items_40", "items_66", "mean CAT items administered",
     "Pedagogy mean CAT length at SE <= 0.30\ncontrolled 40 vs 66 calibration pool, 12 held-out",
     "{:.1f}", "pl_1_2_3_expanded78_items_se0.3.png")
