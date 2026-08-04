"""TutorEval skill-fit figure set.

A thin, read-only plotting driver that visualises whether TutorEval's 2-skill axis
(``conceptual_understanding`` vs ``quantitative_procedural``) was a good fit. It
reads already-computed calibration and labeling-reliability artifacts and renders
four figures; it never re-runs a calibration and never modifies any data/results.

Sources (all pre-computed, read-only):
  * ``staging/tutoreval_calibration/calibration_mirt_manifest.json``
      -- latent correlation, AIC/BIC model comparison, identifiability note,
         fitted-subset q-pattern counts, per-skill loaded counts.
  * ``staging/tutoreval_calibration/calibration_mirt.csv``
      -- per-item discriminations (1,186 fitted items).
  * ``data/TutorEval/rubrics_qmatrix_final.jsonl``
      -- full-bank q-matrix (1,786 criteria) -> q-pattern distribution.
  * ``qmatrix_verify_logs/full/manifest.json``
      -- Fleiss/Cohen kappa per skill/verifier and resolution breakdown.

Outputs (``regenerated_figures/tutoreval_skillfit/``):
  * ``fig1_dimensionality.png``   -- latent correlation + AIC/BIC comparison.
  * ``fig2_skill_loading.png``    -- per-skill discriminations + q-pattern bars.
  * ``fig3_qmatrix_map.png``      -- aggregate TutorEval q-matrix map.
  * ``fig4_labeling_irr.png``     -- inter-rater reliability + resolution breakdown.

Usage
-----
    python scripts/plot_tutoreval_skillfit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "staging" / "tutoreval_calibration" / "calibration_mirt_manifest.json"
ITEM_CSV = ROOT / "staging" / "tutoreval_calibration" / "calibration_mirt.csv"
RUBRICS = ROOT / "data" / "TutorEval" / "rubrics_qmatrix_final.jsonl"
VERIFY = ROOT / "qmatrix_verify_logs" / "full" / "manifest.json"

OUT_DIR = ROOT / "regenerated_figures" / "tutoreval_skillfit"

SKILLS = ("conceptual_understanding", "quantitative_procedural")
SKILL_SHORT = {
    "conceptual_understanding": "conceptual",
    "quantitative_procedural": "quantitative",
}
C_CONCEPT = "#3b6ea5"
C_QUANT = "#c2703d"


# ---------------------------------------------------------------------------
# loading (read-only)
# ---------------------------------------------------------------------------


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def load_verify() -> dict:
    return json.loads(VERIFY.read_text())


def load_discriminations() -> dict[str, np.ndarray]:
    """Per-skill non-zero discriminations from the fitted-item table.

    A criterion loads a skill iff its masked discrimination for that skill is
    non-zero (the confirmatory M2PL fixes ``a_k == 0`` for unmapped skills).
    """
    import csv

    conceptual: list[float] = []
    quant: list[float] = []
    with ITEM_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ac = float(row["a_conceptual_understanding"])
            aq = float(row["a_quantitative_procedural"])
            if ac != 0.0:
                conceptual.append(ac)
            if aq != 0.0:
                quant.append(aq)
    return {
        "conceptual_understanding": np.array(conceptual),
        "quantitative_procedural": np.array(quant),
    }


def load_full_qpatterns() -> dict[str, int]:
    """Full-bank q-pattern distribution over all 1,786 criteria."""
    counts = {"10": 0, "01": 0, "11": 0, "00": 0}
    with RUBRICS.open() as fh:
        for line in fh:
            rec = json.loads(line)
            q = rec.get("q_mapping") or {}
            cu = int(q.get("conceptual_understanding", 0))
            qp = int(q.get("quantitative_procedural", 0))
            counts[f"{cu}{qp}"] += 1
    return counts


# ---------------------------------------------------------------------------
# Figure 1 -- dimensionality panel
# ---------------------------------------------------------------------------


def fig_dimensionality(manifest: dict, out: Path) -> dict:
    rho = float(manifest["latent_correlation"][0][1])
    cmp = manifest["comparison"]
    uni, multi = cmp["uni"], cmp["multi"]
    d_aic = cmp["delta_aic_uni_minus_multi"]
    d_bic = cmp["delta_bic_uni_minus_multi"]
    n_persons = int(manifest["n_persons_fit"])
    min_id = int(manifest["identifiability"]["min_persons_identifiable"])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.6), constrained_layout=True)

    # -- left: latent correlation matrix with prominent rho callout --
    corr = np.array(manifest["latent_correlation"], dtype=float)
    im = ax_l.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_l.set_xticks([0, 1], [SKILL_SHORT[s] for s in SKILLS], fontsize=10)
    ax_l.set_yticks([0, 1], [SKILL_SHORT[s] for s in SKILLS], rotation=90, va="center", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax_l.text(
                j,
                i,
                f"{corr[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if abs(corr[i, j]) > 0.5 else "black",
                fontsize=13,
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax_l, fraction=0.046, pad=0.04, label="latent correlation")
    ax_l.set_title(
        f"Latent skill correlation  \u03c1 = {rho:.3f}\n"
        "nearly collinear \u2192 effectively one axis",
        fontsize=12.5,
        fontweight="bold",
    )

    # -- right: AIC / BIC comparison --
    metrics = ["AIC", "BIC"]
    uni_vals = [uni["aic"], uni["bic"]]
    multi_vals = [multi["aic"], multi["bic"]]
    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax_r.bar(x - w / 2, uni_vals, w, label="Unidimensional 2PL", color=C_CONCEPT)
    b2 = ax_r.bar(x + w / 2, multi_vals, w, label="2-skill M2PL", color=C_QUANT)
    ax_r.bar_label(b1, fmt="%.0f", fontsize=9, padding=2)
    ax_r.bar_label(b2, fmt="%.0f", fontsize=9, padding=2)
    ax_r.set_xticks(x, metrics, fontsize=11)
    ax_r.set_ylabel("information criterion (lower = better)", fontsize=10)
    ax_r.set_ylim(0, max(multi_vals) * 1.22)
    ax_r.set_title(
        "Model fit: unidimensional wins on both criteria", fontsize=12.5, fontweight="bold"
    )
    ax_r.legend(fontsize=9.5, loc="upper left")
    ax_r.text(
        0.46,
        0.82,
        f"\u0394AIC = {d_aic:+.0f}\n\u0394BIC = {d_bic:+.0f}\n"
        "(uni \u2212 multi; both < 0\nfavor unidimensional)",
        transform=ax_r.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6"),
    )

    fig.suptitle(
        "TutorEval is effectively unidimensional\n"
        f"Confirmatory 2-skill M2PL vs unidimensional 2PL  (N = {n_persons} models "
        f"< {min_id} identifiability threshold)",
        fontsize=12.5,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return {
        "rho": rho,
        "aic_uni": uni["aic"],
        "aic_multi": multi["aic"],
        "bic_uni": uni["bic"],
        "bic_multi": multi["bic"],
        "delta_aic": d_aic,
        "delta_bic": d_bic,
        "n_persons": n_persons,
    }


# ---------------------------------------------------------------------------
# Figure 2 -- skill-loading visual
# ---------------------------------------------------------------------------


def fig_skill_loading(
    manifest: dict, discr: dict[str, np.ndarray], full_q: dict[str, int], out: Path
) -> dict:
    fitted_q = manifest["block"]["q_pattern_counts"]  # 10 / 01 / 11 over fitted items
    n_fit = int(manifest["n_items_fit"])
    conceptual = discr["conceptual_understanding"]
    quant = discr["quantitative_procedural"]
    med_c = float(np.median(conceptual))
    med_q = float(np.median(quant))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)

    # -- left: per-skill discrimination distributions --
    data = [conceptual, quant]
    labels = [f"conceptual\nn = {conceptual.size}", f"quantitative\nn = {quant.size}"]
    colors = [C_CONCEPT, C_QUANT]
    parts = ax_l.violinplot(data, showextrema=False)
    for pc, col in zip(parts["bodies"], colors, strict=True):
        pc.set_facecolor(col)
        pc.set_alpha(0.35)
    bp = ax_l.boxplot(
        data,
        widths=0.18,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", lw=1.6),
    )
    for patch, col in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    for i, (arr, col) in enumerate(zip(data, colors, strict=True), start=1):
        jitter = np.random.default_rng(0).normal(0, 0.045, arr.size)
        ax_l.scatter(
            np.full(arr.size, i) + jitter,
            arr,
            s=7,
            color=col,
            alpha=0.25,
            edgecolor="none",
            zorder=3,
        )
    ax_l.set_xticks([1, 2], labels, fontsize=10)
    ax_l.text(1, med_c, f"  median {med_c:.2f}", va="center", ha="left", fontsize=9.5)
    ax_l.text(2, med_q, f"  median {med_q:.2f}", va="center", ha="left", fontsize=9.5)
    ax_l.set_ylabel("item discrimination $a$", fontsize=10)
    ax_l.set_ylim(-1, min(8, np.percentile(np.concatenate(data), 99.5) + 1))
    ax_l.set_title("Per-skill discriminations (fitted items)", fontsize=12.5, fontweight="bold")

    # -- right: q-pattern distribution, full bank vs fitted subset --
    order = ["10", "01", "11", "00"]
    pat_labels = {
        "10": "conceptual\nonly",
        "01": "quant\nonly",
        "11": "both",
        "00": "unmapped",
    }
    full_vals = [full_q[p] for p in order]
    fitted_vals = [fitted_q.get(p, 0) for p in order]  # no "00" in fitted subset
    x = np.arange(len(order))
    w = 0.4
    pat_colors = [C_CONCEPT, C_QUANT, "#6a51a3", "0.6"]
    b_full = ax_r.bar(
        x - w / 2,
        full_vals,
        w,
        label=f"full bank (n = {sum(full_vals)})",
        color=pat_colors,
        edgecolor="black",
        linewidth=0.4,
    )
    b_fit = ax_r.bar(
        x + w / 2,
        fitted_vals,
        w,
        label=f"fitted subset (n = {n_fit})",
        color=pat_colors,
        alpha=0.5,
        edgecolor="black",
        linewidth=0.4,
        hatch="//",
    )
    ax_r.bar_label(b_full, fmt="%d", fontsize=9, padding=2)
    ax_r.bar_label(b_fit, fmt="%d", fontsize=9, padding=2)
    ax_r.set_xticks(x, [pat_labels[p] for p in order], fontsize=10)
    ax_r.set_ylabel("number of criteria", fontsize=10)
    ax_r.set_ylim(0, max(full_vals) * 1.15)
    ax_r.set_title("Q-pattern distribution", fontsize=12.5, fontweight="bold")
    ax_r.legend(fontsize=9.5, loc="upper right")

    quant_share = quant.size / n_fit
    ax_r.text(
        0.5,
        0.55,
        f"Only {quant.size} / {n_fit} fitted items\n"
        f"load quantitative (\u2248 {quant_share:.0%}):\n"
        "the quantitative pool is thin.",
        transform=ax_r.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdf3e7", edgecolor=C_QUANT),
    )

    fig.suptitle(
        "The quantitative skill is a thin, less-populated pool\n"
        f"conceptual median a = {med_c:.2f} (n = {conceptual.size}) vs "
        f"quantitative median a = {med_q:.2f} (n = {quant.size}).",
        fontsize=13,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return {
        "med_conceptual": med_c,
        "med_quant": med_q,
        "n_conceptual": int(conceptual.size),
        "n_quant": int(quant.size),
        "full_q": full_q,
        "fitted_q": fitted_q,
    }


# ---------------------------------------------------------------------------
# Figure 3 -- aggregate q-matrix map
# ---------------------------------------------------------------------------


def fig_qmatrix_map(full_q: dict[str, int], out: Path) -> dict:
    total = sum(full_q.values())
    # rows = q-patterns (with counts), columns = the two skills; a cell is "loaded"
    # if that pattern requires that skill. Aggregate analog of the IFEval map.
    order = ["10", "11", "01", "00"]
    row_labels = {
        "10": f"conceptual only  (n={full_q['10']})",
        "11": f"both skills  (n={full_q['11']})",
        "01": f"quantitative only  (n={full_q['01']})",
        "00": f"unmapped  (n={full_q['00']})",
    }
    # loading matrix: 1 = loads that skill (conceptual, quantitative)
    loads = {
        "10": (1, 0),
        "11": (1, 1),
        "01": (0, 1),
        "00": (0, 0),
    }
    grid = np.array([loads[p] for p in order], dtype=float)

    fig, (ax, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    # -- left: pattern x skill map --
    cmap = matplotlib.colors.ListedColormap(["#eef1f4", C_CONCEPT])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    # overlay quantitative-loaded cells in the quantitative color
    for i, p in enumerate(order):
        c_load, q_load = loads[p]
        if q_load:
            ax.add_patch(plt.Rectangle((1 - 0.5, i - 0.5), 1, 1, color=C_QUANT))
        if c_load:
            ax.add_patch(plt.Rectangle((0 - 0.5, i - 0.5), 1, 1, color=C_CONCEPT))
        for j, load in enumerate((c_load, q_load)):
            ax.text(
                j,
                i,
                "loads" if load else "\u2013",
                ha="center",
                va="center",
                color="white" if load else "0.5",
                fontsize=10,
                fontweight="bold" if load else "normal",
            )
    ax.set_xticks([0, 1], ["conceptual", "quantitative"], fontsize=11)
    ax.set_yticks(range(len(order)), [row_labels[p] for p in order], fontsize=10)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    for edge in range(len(order) + 1):
        ax.axhline(edge - 0.5, color="white", lw=2)
    ax.axvline(0.5, color="white", lw=2)
    ax.set_title(
        "TutorEval Q-matrix: criterion pattern \u2192 skill", fontsize=12.5, fontweight="bold"
    )

    # -- right: proportion of criteria loading each skill (with overlap) --
    n_concept = full_q["10"] + full_q["11"]
    n_quant = full_q["01"] + full_q["11"]
    both = full_q["11"]
    bars = ax_bar.bar(
        ["conceptual", "quantitative", "both\n(overlap)", "unmapped"],
        [n_concept, n_quant, both, full_q["00"]],
        color=[C_CONCEPT, C_QUANT, "#6a51a3", "0.6"],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_bar.bar_label(bars, fmt="%d", fontsize=10, padding=2)
    ax_bar.set_ylabel("criteria loading skill", fontsize=10)
    ax_bar.set_ylim(0, total * 1.05)
    ax_bar.set_title("Criteria loading each skill", fontsize=12.5, fontweight="bold")
    ax_bar.text(
        0.5,
        0.78,
        f"conceptual: {n_concept}/{total} ({n_concept / total:.0%})\n"
        f"quantitative: {n_quant}/{total} ({n_quant / total:.0%})",
        transform=ax_bar.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6"),
    )

    fig.suptitle(
        f"TutorEval Q-matrix map \u2014 {total} criteria across 2 skills\n"
        "Aggregate view: conceptual dominates, quantitative is sparse, overlap is small.",
        fontsize=13,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return {
        "total": total,
        "n_conceptual": n_concept,
        "n_quant": n_quant,
        "both": both,
        "unmapped": full_q["00"],
    }


# ---------------------------------------------------------------------------
# Figure 4 -- labeling reliability (IRR)
# ---------------------------------------------------------------------------


def fig_labeling_irr(verify: dict, out: Path) -> dict:
    per_skill = verify["stats"]["per_skill"]
    fleiss = {s: per_skill[s]["fleiss_kappa_all_raters"] for s in SKILLS}
    cohen = {s: per_skill[s]["cohen_kappa_vs_generator"] for s in SKILLS}
    verifiers = verify["stats"]["verifier_names"]
    res = verify["resolution_counts"]
    review_q = verify["review_queue_size"]
    n_in = verify["input_records"]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(12.5, 5.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.7, 1.0]}
    )

    # -- left: grouped kappa bars (per skill) --
    series = [("Fleiss \u03ba (all raters)", "#2f6f4f", [fleiss[s] for s in SKILLS])]
    v_colors = {"gpt-5.5": C_CONCEPT, "gemini-3.1-pro": C_QUANT}
    for v in verifiers:
        series.append(
            (f"Cohen \u03ba ({v})", v_colors.get(v, "0.5"), [cohen[s][v] for s in SKILLS])
        )

    x = np.arange(len(SKILLS))
    n_series = len(series)
    w = 0.8 / n_series
    for k, (label, color, vals) in enumerate(series):
        offset = (k - (n_series - 1) / 2) * w
        b = ax_l.bar(
            x + offset, vals, w, label=label, color=color, edgecolor="black", linewidth=0.4
        )
        ax_l.bar_label(b, fmt="%.3f", fontsize=8.5, padding=2)
    ax_l.set_xticks(x, [SKILL_SHORT[s] for s in SKILLS], fontsize=11)
    ax_l.set_ylabel("\u03ba agreement", fontsize=10)
    ax_l.set_ylim(0, 1.0)
    ax_l.axhspan(0.61, 0.80, color="green", alpha=0.06)
    ax_l.axhline(0.61, color="0.6", ls="--", lw=0.8)
    ax_l.text(
        ax_l.get_xlim()[1],
        0.61,
        " substantial \u2265 0.61",
        va="bottom",
        ha="right",
        fontsize=8,
        color="0.4",
    )
    ax_l.set_title("Inter-rater reliability per skill", fontsize=12.5, fontweight="bold")
    ax_l.legend(fontsize=9, loc="upper left", ncol=1)

    # -- right: resolution breakdown stacked bar --
    res_order = ["unanimous", "majority", "tie", "generator_only"]
    res_colors = ["#2f6f4f", "#3b6ea5", "#c2703d", "#7d1f1f"]
    bottom = 0.0
    for name, col in zip(res_order, res_colors, strict=True):
        val = res[name]
        ax_r.bar(
            0,
            val,
            bottom=bottom,
            color=col,
            width=0.6,
            label=f"{name} ({val})",
            edgecolor="white",
            linewidth=0.6,
        )
        if val > 15:
            ax_r.text(
                0,
                bottom + val / 2,
                f"{val}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )
        bottom += val
    ax_r.set_xlim(-0.8, 1.9)
    ax_r.set_ylim(0, bottom * 1.28)
    ax_r.set_xticks([])
    ax_r.set_ylabel("criteria", fontsize=10)
    ax_r.set_title("Verifier resolution", fontsize=12.5, fontweight="bold")
    ax_r.legend(fontsize=9, loc="upper right")
    ax_r.text(
        0.5,
        0.44,
        f"total = {int(bottom)}\nreview queue\n= {review_q} of {n_in}",
        transform=ax_r.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6"),
    )

    fig.suptitle(
        "Q-matrix labeling reliability (two LLM verifiers vs generator)\n"
        f"Substantial agreement on both skills; {review_q} of {n_in} criteria "
        "flagged for review.",
        fontsize=13,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return {
        "fleiss": fleiss,
        "cohen": cohen,
        "resolution": dict(res),
        "review_queue": review_q,
        "n_in": n_in,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    verify = load_verify()
    discr = load_discriminations()
    full_q = load_full_qpatterns()

    r1 = fig_dimensionality(manifest, out_dir / "fig1_dimensionality.png")
    r2 = fig_skill_loading(manifest, discr, full_q, out_dir / "fig2_skill_loading.png")
    r3 = fig_qmatrix_map(full_q, out_dir / "fig3_qmatrix_map.png")
    r4 = fig_labeling_irr(verify, out_dir / "fig4_labeling_irr.png")

    print("Wrote figures to", out_dir)
    for name, res in (("fig1", r1), ("fig2", r2), ("fig3", r3), ("fig4", r4)):
        print(f"  {name}: {json.dumps(res, default=float)}")


if __name__ == "__main__":
    main()
