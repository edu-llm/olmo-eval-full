#!/usr/bin/env python3
"""ARC balanced-pool random sweep: held-out correlation vs number of calibration models.

Per N (50-model steps), plot the mean Pearson r across 5 random seeds with a shaded
min-max range band, plus the full-pool reference line. All r are out-of-sample on the
417 held-out test models.
Usage: uv run --with numpy --with matplotlib python AdaptiveTesting/Research/scripts/arc_random_range_plot.py
"""
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14, "font.weight": "bold",
    "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 15, "axes.labelweight": "bold",
    "xtick.labelsize": 12, "ytick.labelsize": 12,
})

D = Path(__file__).resolve().parents[3] / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_arc_balanced_sweep"
rows = list(csv.DictReader(open(D / "results.csv")))
full_pool_r = next(float(r["r"]) for r in rows if r["strategy"] == "full_pool")

by_n = defaultdict(list)
for r in rows:
    if r["strategy"] == "random":
        by_n[int(r["N"])].append(float(r["r"]))
Ns = sorted(by_n)
mean = np.array([np.mean(by_n[n]) for n in Ns])
lo = np.array([np.min(by_n[n]) for n in Ns])
hi = np.array([np.max(by_n[n]) for n in Ns])

fig, ax = plt.subplots(figsize=(10.5, 6.4))
ax.fill_between(Ns, lo, hi, color="#4C72B0", alpha=0.22, label="range (min-max over 5 seeds)")
ax.plot(Ns, mean, "o-", color="#2f4b7c", lw=2.4, ms=8, label="mean r (5 seeds)")
ax.axhline(full_pool_r, color="black", ls="--", lw=1.6, label=f"full pool, N=500 (r={full_pool_r:.3f})")
ax.set_xlabel("number of calibration models (N)")
ax.set_ylabel("Pearson r (held-out test set)")
ax.set_title("ATLAS ARC (balanced pool): held-out correlation vs calibration models\nrandom selection, 50-model steps")
ax.set_xticks(Ns)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=11)
for lab in ax.get_xticklabels() + ax.get_yticklabels():
    lab.set_fontweight("bold")
fig.tight_layout()
out = D / "figures" / "arc_random_sweep_range.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
for n in Ns:
    v = by_n[n]
    print(f"N={n:>3}  mean={np.mean(v):.4f}  min={np.min(v):.4f}  max={np.max(v):.4f}")
