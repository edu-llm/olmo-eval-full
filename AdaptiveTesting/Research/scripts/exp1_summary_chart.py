#!/usr/bin/env python3
"""One-figure summary of Experiment 1 (OpenLM ATLAS-style CAT recovery, SE<=0.3).

Packs accuracy MAE (+/- SEM), Pearson r, adaptive vs random CAT items and savings,
bank size, and held-out test model count into a single combo chart.
Source: AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication/summary_*.csv
Usage: uv run --with matplotlib --with numpy python AdaptiveTesting/Research/scripts/exp1_summary_chart.py
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/figures/exp1_openlm_se0.3_summary.png"

# SE<=0.3, ordered by accuracy MAE ascending
bench = ["MATH", "IFEval", "GPQA", "MuSR", "BBH"]
mae = [0.025, 0.050, 0.070, 0.095, 0.114]
sem = [0.004, 0.005, 0.004, 0.005, 0.007]
r = [0.878, 0.921, 0.737, 0.746, 0.674]
adapt = [77.3, 18.2, 13.0, 13.6, 8.2]
rand = [147.6, 62.1, 45.7, 76.2, 13.3]
save = [1.9, 3.4, 3.5, 5.6, 1.6]
bank = [1183, 511, 579, 432, 3965]
ntest = [90, 110, 110, 110, 110]

x = np.arange(len(bench))
fig, ax1 = plt.subplots(figsize=(11, 6.4))

ax1.bar(
    x, mae, yerr=sem, capsize=4, width=0.55,
    color="#4C72B0", alpha=0.85, label="Accuracy MAE (+/- SEM)", zorder=3,
)
ax1.set_ylabel("Accuracy MAE (p-IRT predicted vs actual)", color="#33517a")
ax1.set_ylim(0, 0.155)
ax1.tick_params(axis="y", labelcolor="#33517a")

for i in range(len(bench)):
    ax1.annotate(
        f"{adapt[i]:.0f} vs {rand[i]:.0f} items\n({save[i]}x fewer)",
        (x[i], mae[i] + sem[i] + 0.006), ha="center", va="bottom",
        fontsize=8, color="#333333",
    )

ax2 = ax1.twinx()
ax2.plot(x, r, "s-", color="#C44E52", lw=2, ms=9, label="Pearson r", zorder=4)
for i in range(len(bench)):
    ax2.annotate(
        f"r={r[i]:.2f}", (x[i], r[i]), textcoords="offset points",
        xytext=(0, 11), ha="center", fontsize=8, color="#8c2f33",
    )
ax2.set_ylabel("Pearson r (out-of-sample)", color="#8c2f33")
ax2.set_ylim(0.5, 1.0)
ax2.tick_params(axis="y", labelcolor="#8c2f33")

ax1.set_xticks(x)
ax1.set_xticklabels(
    [f"{b}\n(bank {bk:,}; n_test={n})" for b, bk, n in zip(bench, bank, ntest, strict=False)]
)
ax1.set_title(
    "OpenLM ATLAS-style CAT recovery at SE<=0.3 (out-of-sample)\n"
    "bars: accuracy MAE (+/- SEM);  line: Pearson r;  labels: adaptive vs random CAT items"
)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
ax1.grid(axis="y", alpha=0.25, zorder=0)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
