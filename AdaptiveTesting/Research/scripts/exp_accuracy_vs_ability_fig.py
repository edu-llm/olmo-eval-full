#!/usr/bin/env python3
"""Flagship 'same accuracy, different ability' pairs (MuSR + IFEval) as dumbbells.

Reads the curated example CSVs from data/accuracy_vs_ability/ and draws, per pair,
the two models' full-bank IRT ability (theta) at their shared full-bank accuracy,
annotated with the held-out-half performance gap.
Usage: uv run --with matplotlib python AdaptiveTesting/Research/scripts/exp_accuracy_vs_ability_fig.py
"""
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/accuracy_vs_ability"
OUT = BASE / "figures" / "flagship_pairs.png"


def short(m, n=26):
    s = m.split("/")[-1]
    return s if len(s) <= n else s[: n - 1] + "..."


def top_pairs(bench, k=2):
    with open(BASE / f"{bench}_examples.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if float(r["heldoutB_gap"]) > 0]
    rows.sort(key=lambda r: float(r["heldoutB_gap"]), reverse=True)
    return rows[:k]


pairs = []
for bench, label in [("ifeval", "IFEval"), ("musr", "MuSR")]:
    for r in top_pairs(bench, 2):
        pairs.append((label, r))

fig, ax = plt.subplots(figsize=(11.5, 5.4))
yticks, ylabels = [], []
for i, (label, r) in enumerate(pairs):
    ta, tb = float(r["theta_a"]), float(r["theta_b"])
    acc = float(r["accuracy"]) * 100
    gap = float(r["heldoutB_gap"]) * 100
    y = len(pairs) - 1 - i
    ax.plot([tb, ta], [y, y], color="#9a9a9a", lw=2.2, zorder=1)
    ax.scatter([tb], [y], s=140, color="#C44E52", zorder=3, label="lower ability" if i == 0 else None)
    ax.scatter([ta], [y], s=140, color="#55A868", zorder=3, label="higher ability" if i == 0 else None)
    ax.annotate(f"held-out +{gap:.1f} pts", (ta, y), textcoords="offset points",
                xytext=(12, 0), va="center", fontsize=9, color="#2f6b3f", fontweight="bold")
    yticks.append(y)
    ylabels.append(f"[{label}]  {short(r['model_a'])}  vs  {short(r['model_b'])}\nsame accuracy = {acc:.1f}%")

ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=8)
ax.axvline(0, color="black", lw=0.8, ls=":")
ax.set_xlabel("full-bank IRT ability (theta)")
ax.set_ylim(-0.6, len(pairs) - 0.4)
ax.set_title(
    "Same full-bank accuracy, different IRT ability\n"
    "the higher-ability model wins on held-out items (annotated), so accuracy hides ability signal"
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
