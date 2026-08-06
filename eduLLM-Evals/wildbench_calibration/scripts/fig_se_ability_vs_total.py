"""PLOT-ONLY: WildBench analog of the BiGGen aggregate SE two-bar chart.

Reads the EXISTING per-model deployed-CAT @ 8/0.12 SE already on disk
(``experiments/07_parameter_uncertainty/se_post_vs_total.csv``, regime
``deployed_cat_8_0.12``: SE_posterior = ability-only, SE_total = +calibration) and renders a
two-bar aggregate chart matching the BiGGen figure:
  * bar 1 = mean SE ability-only, bar 2 = mean SE total (+calibration),
  * error bars = SD ACROSS THE 52 MODELS (not a bootstrap CI),
  * dotted horizontal line at the SE target = 0.12.
NO refit / re-bootstrap / engine re-run -- pure read + plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "experiments" / "07_parameter_uncertainty" / "se_post_vs_total.csv"
FIG = BASE / "experiments" / "07_parameter_uncertainty" / "figures" / "se_ability_vs_total.png"
OUT_CSV = BASE / "experiments" / "07_parameter_uncertainty" / "se_ability_vs_total_summary.csv"
SE_TARGET = 0.12

df = pd.read_csv(SRC)
dep = df[df["regime"] == "deployed_cat_8_0.12"].copy()
n = len(dep)
ability = dep["SE_posterior"].to_numpy()   # ability-only measurement SE
total = dep["SE_total"].to_numpy()          # ability (+) calibration (SE_param)

stats = {
    "SE_ability_only": {"mean": float(ability.mean()), "sd": float(ability.std(ddof=1)),
                        "median": float(np.median(ability))},
    "SE_total": {"mean": float(total.mean()), "sd": float(total.std(ddof=1)),
                 "median": float(np.median(total))},
}
pd.DataFrame([
    {"metric": "SE_ability_only", "mean": stats["SE_ability_only"]["mean"],
     "sd_across_models": stats["SE_ability_only"]["sd"],
     "median": stats["SE_ability_only"]["median"], "n_models": n, "se_target": SE_TARGET,
     "regime": "deployed_cat_8_0.12", "source": SRC.name},
    {"metric": "SE_total", "mean": stats["SE_total"]["mean"],
     "sd_across_models": stats["SE_total"]["sd"],
     "median": stats["SE_total"]["median"], "n_models": n, "se_target": SE_TARGET,
     "regime": "deployed_cat_8_0.12", "source": SRC.name},
]).to_csv(OUT_CSV, index=False)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

means = [stats["SE_ability_only"]["mean"], stats["SE_total"]["mean"]]
sds = [stats["SE_ability_only"]["sd"], stats["SE_total"]["sd"]]
labels = ["SE ability-only", "SE total (+calibration)"]
colors = ["#4d648d", "#c1666b"]

fig, ax = plt.subplots(figsize=(6.2, 5.2))
bars = ax.bar([0, 1], means, 0.58, color=colors, yerr=sds, capsize=6,
              error_kw={"elinewidth": 1.6, "ecolor": "#333333"})
ax.axhline(SE_TARGET, ls=":", color="k", lw=1.6)
ax.text(1.45, SE_TARGET + 0.004, "SE target=0.12", ha="right", va="bottom", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("SE (general-ability theta, 1-D MWLE)")
ax.set_title("WildBench general-ability SE: ability-only vs total\n"
             "(locked SE=0.12 / floor 8; N=52)", fontsize=11)
for i, (mn, sd) in enumerate(zip(means, sds)):
    ax.annotate(f"{mn:.3f}\n(SD {sd:.3f})", (i, mn), ha="center", va="bottom",
                fontsize=9, xytext=(0, 3), textcoords="offset points")
ax.set_ylim(0, max(means[1] + sds[1], SE_TARGET) * 1.25)
fig.tight_layout()
fig.savefig(FIG, dpi=150)
plt.close(fig)

print(f"N={n} models (deployed CAT @ 8/0.12), source={SRC.name}")
print(f"SE ability-only : mean={means[0]:.4f}  SD={sds[0]:.4f}  median={stats['SE_ability_only']['median']:.4f}")
print(f"SE total        : mean={means[1]:.4f}  SD={sds[1]:.4f}  median={stats['SE_total']['median']:.4f}")
print(f"target={SE_TARGET}")
print(f"wrote -> {FIG}\nwrote -> {OUT_CSV}")
