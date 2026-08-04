#!/usr/bin/env python3
"""SE sweep: mean CAT items and correlation vs SE target.

Fig 1 (ARC): dual-axis, 3 IRT models, from data/se_sweep/se_sweep_combined.csv.
Fig 2 (OpenLM): two panels (items | correlation), 5 benchmarks, from the
atlas_replication summary CSVs (SE 0.1/0.2/0.3).
Usage: uv run --with matplotlib python AdaptiveTesting/Research/scripts/se_sweep_items_corr_plots.py
"""
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
D = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS"
FIG = D / "figures"


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


# ---------- Fig 1: ARC dual-axis ----------
rows = read(D / "data/se_sweep/se_sweep_combined.csv")
models = {
    "atlas3pl_own_resp": ("ATLAS 3PL", "#4C72B0"),
    "local_arc_2pl": ("Local 2PL", "#DD8452"),
    "local_arc_1pl": ("Local 1PL", "#55A868"),
}
# Finer 39-point grid: read as smooth curves (thin solid = items, thin dashed = r),
# no markers so the dense grid stays legible, with larger bold labels/ticks/legend.
fig, ax1 = plt.subplots(figsize=(11.0, 6.4), constrained_layout=True)
ax2 = ax1.twinx()
for tag, (label, color) in models.items():
    r = sorted([x for x in rows if x["tag"] == tag], key=lambda x: float(x["se_target"]))
    se = [float(x["se_target"]) for x in r]
    items = [float(x["mean_items"]) for x in r]
    corr = [float(x["corr"]) for x in r]
    ax1.plot(se, items, "-", color=color, lw=2.0, label=f"{label} items")
    ax2.plot(se, corr, "--", color=color, lw=1.8, alpha=0.9, label=f"{label} r")
ax1.set_yscale("log")
ax1.invert_xaxis()
ax1.set_xlabel("SE target (tighter to the right)", fontsize=14, fontweight="bold")
ax1.set_ylabel("mean CAT items (log scale)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Pearson r", fontsize=14, fontweight="bold")
ax2.set_ylim(0.6, 1.0)
ax1.tick_params(axis="both", labelsize=12)
ax2.tick_params(axis="y", labelsize=12)
ax1.set_title("ARC: CAT items and correlation vs SE target\n"
              "solid = items (left, log), dashed = r (right)",
              fontsize=15, fontweight="bold")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=11, ncol=2, framealpha=0.9)
ax1.grid(True, which="both", alpha=0.25)
fig.savefig(FIG / "se_sweep_items_corr_arc.png", dpi=150)
plt.close(fig)

# ---------- Fig 2: OpenLM two panels ----------
items_rows = read(D / "data/atlas_replication/summary_items_adaptive_vs_random.csv")
pirt_rows = read(D / "data/atlas_replication/summary_pirt_mae_sd_se.csv")
items_by = {(x["benchmark"], float(x["se_target"])): float(x["mean_adaptive_items"]) for x in items_rows}
r_by = {(x["benchmark"], float(x["se_target"])): float(x["r"]) for x in pirt_rows}
benches = {
    "math": "#4C72B0", "ifeval": "#55A868", "gpqa": "#C44E52",
    "musr": "#8172B3", "bbh": "#CCB974",
}
ses = [0.3, 0.2, 0.1]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.4))
for b, color in benches.items():
    it = [items_by.get((b, s)) for s in ses]
    rr = [r_by.get((b, s)) for s in ses]
    axa.plot(ses, it, "o-", color=color, label=b)
    axb.plot(ses, rr, "o-", color=color, label=b)
for ax in (axa, axb):
    ax.invert_xaxis()
    ax.set_xlabel("SE target (tighter to the right)")
    ax.set_xticks(ses)
    ax.grid(True, alpha=0.25)
axa.set_yscale("log")
axa.set_ylabel("mean CAT items (log scale)")
axa.set_title("OpenLM: mean CAT items vs SE target")
axb.set_ylabel("Pearson r")
axb.set_ylim(0.5, 1.0)
axb.set_title("OpenLM: correlation vs SE target")
axb.legend(fontsize=9, title="benchmark")
fig.tight_layout()
fig.savefig(FIG / "atlasrep_items_corr_openlm.png", dpi=150)
plt.close(fig)

# ---------- Fig 3: OpenLM finer SE grid (two panels, readable fonts) ----------
# Reads the _fine summary CSVs (produced by a fine-grid rerun of the single-split
# trace driver atlas_error_plots_openlm.py); skipped if they are absent.
items_fine_p = D / "data/atlas_replication/summary_items_adaptive_vs_random_fine.csv"
pirt_fine_p = D / "data/atlas_replication/summary_pirt_mae_sd_se_fine.csv"
if items_fine_p.exists() and pirt_fine_p.exists():
    items_rows_f = read(items_fine_p)
    pirt_rows_f = read(pirt_fine_p)
    items_by_f = {(x["benchmark"], float(x["se_target"])): float(x["mean_adaptive_items"])
                  for x in items_rows_f}
    r_by_f = {(x["benchmark"], float(x["se_target"])): float(x["r"]) for x in pirt_rows_f}
    ses_f = sorted({float(x["se_target"]) for x in items_rows_f}, reverse=True)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for b, color in benches.items():
        it = [items_by_f.get((b, s)) for s in ses_f]
        rr = [r_by_f.get((b, s)) for s in ses_f]
        axa.plot(ses_f, it, "-", marker="o", ms=4, lw=1.7, color=color, label=b)
        axb.plot(ses_f, rr, "-", marker="o", ms=4, lw=1.7, color=color, label=b)
    for ax in (axa, axb):
        ax.invert_xaxis()
        ax.set_xlabel("SE target (tighter to the right)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, which="both", alpha=0.25)
    axa.set_yscale("log")
    axa.set_ylabel("mean CAT items (log scale)", fontsize=14)
    axa.set_title("OpenLM: mean CAT items vs SE target (fine grid)", fontsize=15, fontweight="bold")
    axb.set_ylabel("Pearson r", fontsize=14)
    axb.set_ylim(0.5, 1.0)
    axb.set_title("OpenLM: correlation vs SE target (fine grid)", fontsize=15, fontweight="bold")
    axb.legend(fontsize=12, title="benchmark", title_fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "atlasrep_items_corr_openlm_fine.png", dpi=150)
    plt.close(fig)

print("wrote", FIG / "se_sweep_items_corr_arc.png")
print("wrote", FIG / "atlasrep_items_corr_openlm.png")
if (FIG / "atlasrep_items_corr_openlm_fine.png").exists():
    print("wrote", FIG / "atlasrep_items_corr_openlm_fine.png")
