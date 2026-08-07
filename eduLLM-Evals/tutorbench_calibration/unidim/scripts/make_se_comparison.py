"""Assemble the SE-component comparison across three scales and render a grouped bar chart.

STUDY-ONLY / LOCAL. Reads existing metrics.json artifacts (read-only):
  * 2-skill correctness (of-record):  regenerated_figures/scenario_level_115_min12/param_uncertainty/2_skills
  * collapse-all unidim (proxy):       regenerated_figures/scenario_level_115_min12/param_uncertainty/unidim
  * correctness-only unidim (this study): ./param_uncertainty
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # eduLLM-Evals

SOURCES = {
    "2skill_correctness": (
        ROOT / "regenerated_figures/scenario_level_115_min12/param_uncertainty/2_skills/metrics.json",
        "correctness"),
    "collapse_all_unidim": (
        ROOT / "regenerated_figures/scenario_level_115_min12/param_uncertainty/unidim/metrics.json",
        "overall"),
    "correctness_only_unidim": (
        HERE / "param_uncertainty/metrics.json",
        "correctness"),
}

LABELS = {
    "2skill_correctness": "2-skill correctness\n(of-record)",
    "collapse_all_unidim": "collapse-all unidim\n(proxy)",
    "correctness_only_unidim": "correctness-only unidim\n(this study)",
}


def main() -> int:
    rows = []
    data = {}
    for key, (path, dim) in SOURCES.items():
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        c = m["se_components"][dim]
        rec = {
            "scale": key,
            "dim": dim,
            "median_se_posterior": round(c["median_se_posterior"], 4),
            "median_se_param": round(c["median_se_param"], 4),
            "median_se_total": round(c["median_se_total"], 4),
            "median_bar_inflation": round(c["median_bar_inflation"], 4),
            "max_bar_inflation": round(c["max_bar_inflation"], 4),
            "source": str(Path(path).relative_to(ROOT)),
        }
        rows.append(rec)
        data[key] = rec

    # deltas vs 2-skill correctness (the scale we want to beat)
    base = data["2skill_correctness"]
    co = data["correctness_only_unidim"]
    deltas = {
        "se_param_abs_reduction_vs_2skill": round(base["median_se_param"] - co["median_se_param"], 4),
        "se_param_pct_reduction_vs_2skill": round(
            100 * (base["median_se_param"] - co["median_se_param"]) / base["median_se_param"], 1),
        "se_posterior_abs_reduction_vs_2skill": round(
            base["median_se_posterior"] - co["median_se_posterior"], 4),
        "se_posterior_pct_reduction_vs_2skill": round(
            100 * (base["median_se_posterior"] - co["median_se_posterior"]) / base["median_se_posterior"], 1),
        "se_total_abs_reduction_vs_2skill": round(base["median_se_total"] - co["median_se_total"], 4),
        "se_total_pct_reduction_vs_2skill": round(
            100 * (base["median_se_total"] - co["median_se_total"]) / base["median_se_total"], 1),
        "max_inflation_2skill": base["max_bar_inflation"],
        "max_inflation_correctness_only": co["max_bar_inflation"],
    }

    with (HERE / "se_components_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (HERE / "se_components_comparison.json").write_text(
        json.dumps({"rows": rows, "deltas_correctness_only_vs_2skill": deltas}, indent=2),
        encoding="utf-8")

    print("SE-component comparison (median):")
    print(f"{'scale':32s} {'SE_param':>9s} {'SE_post':>9s} {'SE_total':>9s} {'maxInfl':>8s}")
    for r in rows:
        print(f"{r['scale']:32s} {r['median_se_param']:>9.3f} {r['median_se_posterior']:>9.3f} "
              f"{r['median_se_total']:>9.3f} {r['max_bar_inflation']:>8.2f}")
    print("\nCorrectness-only vs 2-skill:")
    for k, v in deltas.items():
        print(f"  {k}: {v}")

    _bar(rows, HERE / "figures" / "se_components_comparison.png")
    print("\nwrote se_components_comparison.csv/json and figures/se_components_comparison.png")
    return 0


def _bar(rows, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    comps = ["median_se_param", "median_se_posterior", "median_se_total"]
    comp_labels = ["SE_param\n(fit stability)", "SE_posterior\n(information)", "SE_total\n(combined)"]
    colors = {"2skill_correctness": "#c1666b", "collapse_all_unidim": "#8aa29e",
              "correctness_only_unidim": "#4d648d"}
    x = np.arange(len(comps))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, r in enumerate(rows):
        vals = [r[c] for c in comps]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=LABELS[r["scale"]],
                      color=colors[r["scale"]], edgecolor="k", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(comp_labels)
    ax.set_ylabel("median SE (ability units)")
    ax.set_title("TutorBench ability-SE decomposition: correctness-only unidim vs 2-skill vs collapse-all\n"
                 "(scenario-level param-uncertainty bootstrap, N=115)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
