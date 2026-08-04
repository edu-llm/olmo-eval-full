#!/usr/bin/env python3
"""Pedagogy item-matched correlation vs test length, linear x-axis, concise text."""
import csv
import os
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
D = REPO / ("AdaptiveTesting/Research/01_MCQ_ATLAS/data/pedagogy_feasibility/"
            "pl_1_2_3_comparison/pedagogy_expanded78/item_matched")
FIG = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/figures"
os.environ.setdefault("MPLCONFIGDIR", str(D / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"1PL": "#55A868", "2PL": "#DD8452", "3PL": "#4C72B0"}
curves = defaultdict(lambda: ([], []))
for row in csv.DictReader(open(D / "pedagogy_item_matched_corr_vs_N.csv")):
    xs, ys = curves[row["bank"]]
    xs.append(int(row["N"]))
    ys.append(float(row["r"]))

fig, ax = plt.subplots(figsize=(8.4, 5.2))
for bank in ("1PL", "2PL", "3PL"):
    xs, ys = curves[bank]
    ax.plot(xs, ys, "-", color=COLORS[bank], lw=1.8, label=bank)

ax.set_xlabel("test length (items)", fontweight="bold")
ax.set_ylabel("Pearson r", fontweight="bold")
ax.set_title("Pedagogy item-matched recovery (66-model pool)", fontweight="bold")
ax.set_ylim(0.2, 1.0)
ax.grid(True, alpha=0.3)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
leg = ax.legend(loc="lower right")
for txt in leg.get_texts():
    txt.set_fontweight("bold")
fig.tight_layout()
out = FIG / "pedagogy_item_matched_corr_vs_items.png"
fig.savefig(out, dpi=150)
print("wrote", out)
