"""Regenerate the exp-03 dimensionality figures from cached outputs (no re-fit).

Reads ``experiments/03_structures/{selection.json, structure_comparison.csv,
skill_correlation.csv}`` and writes to ``experiments/03_structures/figures/``:
  * ``scree.png``               -- eigenvalue scree of the 11-skill ability correlation,
  * ``structure_cv.png``        -- held-out log-loss (+/-SE) per structure with 1-SE band,
  * ``skill_correlation.png``   -- the 11x11 latent-ability correlation heatmap.
matplotlib Agg backend.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "experiments" / "03_structures"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
rows = []
with (OUT / "structure_comparison.csv").open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rows.append(r)
skills, corr = [], []
with (OUT / "skill_correlation.csv").open(encoding="utf-8") as fh:
    rd = csv.reader(fh)
    next(rd)
    for line in rd:
        skills.append(line[0])
        corr.append([float(x) for x in line[1:]])
corr = np.array(corr)
eig = sel["scree_efa"]["eigenvalues"]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- scree ---
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(range(1, len(eig) + 1), eig, "o-", color="#4d648d")
ax.axhline(1.0, ls="--", color="gray", label="Kaiser (eigenvalue=1)")
ax.set_xlabel("factor"); ax.set_ylabel("eigenvalue")
ax.set_title(f"WildBench 11-skill ability scree (1st = {eig[0]/sum(eig)*100:.0f}% var, "
             f"{sel['scree_efa']['n_eigen_ge_1']} factor >= 1)")
ax.set_xticks(range(1, len(eig) + 1))
ax.legend()
fig.tight_layout(); fig.savefig(OUT / "figures" / "scree.png", dpi=140); plt.close(fig)

# --- structure comparison (held-out log-loss) ---
order = sorted(rows, key=lambda r: int(r["n_dims"]))
xs = [r["structure"] for r in order]
ys = [float(r["oos_logloss_mean"]) for r in order]
es = [float(r["oos_logloss_se"]) for r in order]
thresh = sel["one_se_threshold"]
fig, ax = plt.subplots(figsize=(7.8, 4.6))
ax.errorbar(range(len(xs)), ys, yerr=es, fmt="o", capsize=4, color="#c1666b")
ax.axhline(thresh, ls="--", color="gray", label="1-SE threshold")
ax.set_xticks(range(len(xs)))
ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("held-out marginal log-loss (lower=better)")
ax.set_title(f"WildBench dimensionality (N=52); selected: {sel['selected']} "
             f"(all within 1 SE)")
ax.legend()
fig.tight_layout(); fig.savefig(OUT / "figures" / "structure_cv.png", dpi=140); plt.close(fig)

# --- skill correlation heatmap ---
fig, ax = plt.subplots(figsize=(7.2, 6.0))
im = ax.imshow(corr, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(skills))); ax.set_xticklabels(skills, rotation=90, fontsize=7)
ax.set_yticks(range(len(skills))); ax.set_yticklabels(skills, fontsize=7)
for i in range(len(skills)):
    for j in range(len(skills)):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                fontsize=5.5, color="white" if corr[i, j] < 0.7 else "black")
fig.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("WildBench 11-skill latent-ability correlation (plug-in, N=52)")
fig.tight_layout(); fig.savefig(OUT / "figures" / "skill_correlation.png", dpi=140); plt.close(fig)

print("wrote exp-03 figures ->", OUT / "figures")
