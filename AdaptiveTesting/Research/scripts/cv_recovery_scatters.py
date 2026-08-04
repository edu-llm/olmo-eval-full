#!/usr/bin/env python3
"""10-fold CV recovery scatters: CAT-predicted vs full-benchmark accuracy, all models.

ATLAS-style recovery panels: each shows the data cloud, the ideal y=x line, AND the
actual regression fit (so the scale deviation is visible). One scatter per OpenLM
benchmark with CV data (IFEval, MATH, GPQA, MuSR; BBH excluded from CV), plus two
horizontally-split combined figures (2 panels each).
Usage: uv run --with numpy --with matplotlib python AdaptiveTesting/Research/scripts/cv_recovery_scatters.py
"""
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 15,
    "font.weight": "bold",
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

REPO = Path(__file__).resolve().parents[3]
CV = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication_cv"
OUT = CV / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    ("IFEval", CV / "ifeval/pirt_vs_actual_cv_se0.3.csv", "#55A868"),
    ("MATH", CV / "math/per_model.csv", "#4C72B0"),
    ("GPQA", CV / "gpqa/per_model_se0.3.csv", "#C44E52"),
    ("MuSR", CV / "musr/pirt_vs_actual_cv_se0.3.csv", "#8172B3"),
]


def load(path):
    pirt, actual = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            pirt.append(float(r["pirt"]))
            actual.append(float(r["actual"]))
    return np.array(actual), np.array(pirt)


def draw(ax, name, actual, pirt, color):
    slope, intercept = (float(v) for v in np.polyfit(actual, pirt, 1))
    r = float(np.corrcoef(actual, pirt)[0, 1])
    mae = float(np.mean(np.abs(pirt - actual)))
    lo = min(actual.min(), pirt.min())
    hi = max(actual.max(), pirt.max())
    pad = 0.03 * (hi - lo)
    lims = [lo - pad, hi + pad]
    xl = np.array(lims)
    ax.plot(lims, lims, "--", color="#555555", lw=1.8, alpha=0.9, label="ideal (y = x)", zorder=1)
    ax.scatter(actual, pirt, s=14, color=color, alpha=0.35, edgecolors="none", zorder=2)
    ax.plot(xl, slope * xl + intercept, "-", color="black", lw=2.6,
            label=f"actual fit (slope {slope:.2f})", zorder=4)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("actual full-benchmark accuracy")
    ax.set_ylabel("CAT-predicted accuracy (p-IRT)")
    ax.set_title(f"{name}: r={r:.3f}, slope={slope:.2f}, MAE={mae:.3f}")
    ax.grid(True, alpha=0.2)
    leg = ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
    for t in leg.get_texts():
        t.set_fontweight("bold")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


# individual figures
data = {}
for name, path, color in FILES:
    actual, pirt = load(path)
    data[name] = (actual, pirt, color)
    fig, ax = plt.subplots(figsize=(7.0, 7.0), layout="constrained")
    draw(ax, name, actual, pirt, color)
    fig.savefig(OUT / f"cv_recovery_{name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# two horizontally-split combined figures (2 panels side by side)
groups = [("1", ["IFEval", "MATH"], True), ("2", ["GPQA", "MuSR"], False)]
for tag, names, show_title in groups:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.6), layout="constrained")
    for ax, name in zip(axes, names, strict=True):
        actual, pirt, color = data[name]
        draw(ax, name, actual, pirt, color)
    if show_title:
        fig.get_layout_engine().set(h_pad=0.35, w_pad=0.06)
        fig.suptitle(
            "OpenLM 10-fold CV recovery\nCAT-predicted vs full-benchmark accuracy",
            fontsize=24, fontweight="bold", y=1.06,
        )
    fig.savefig(OUT / f"cv_recovery_openlm_grid_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

for tag in ("1", "2"):
    print("wrote", OUT / f"cv_recovery_openlm_grid_{tag}.png")
for name, *_ in FILES:
    print("wrote", OUT / f"cv_recovery_{name.lower()}.png")
