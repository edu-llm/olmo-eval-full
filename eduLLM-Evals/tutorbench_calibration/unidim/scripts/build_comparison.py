"""Side-by-side comparison of the two TutorBench candidate scales at floor 20 / SE_ability 0.27.

STUDY-ONLY / LOCAL. Reads existing artifacts (read-only) and, for the rank-agreement panel,
recomputes selection-independent FULL-BANK EAP theta from each fitted bank using the shared
production estimator library (`scripts/scenario_cat_lib`). Nothing is committed; the engine
(`tutor_cat/`, `scripts/scenario_cat_lib.py`) is only imported/called, never modified.

Scales compared, both at the SAME operating point floor 20 / SE_ability 0.27:
  * 2-skill (correctness + scaffolding) -- the of-record scale.
  * correctness-only unidim              -- the alternative.

Outputs (this folder):
  comparison_table_20_0.27.csv / .md
  figures/se_components_grouped_bar.png
  figures/rank_agreement_scatter.png
  figures/recovery_efficiency_comparison.png
  fullbank_eap_theta_per_model.csv
  comparison_summary.json
  README.md   (written separately)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UNIDIM_DIR = HERE.parent                       # eap_unidim_correctness_only_tutorbench
REPORTS = UNIDIM_DIR.parent                    # reports
EVALROOT = REPORTS.parent                      # eduLLM-Evals
sys.path.insert(0, str(EVALROOT / "scripts"))

import scenario_cat_lib as scl  # noqa: E402

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
BANK_2SKILL = EVALROOT / "data/TutorBench/rubrics_qmatrix_calibrated_2skill_115_fitted.jsonl"
BANK_UNIDIM = UNIDIM_DIR / "rubrics_qmatrix_correctness_only_unidim_115_fitted.jsonl"
MATRIX = EVALROOT / "staging/response_matrix_full_nonopt_115.csv"

SE_2SKILL = EVALROOT / "regenerated_figures/scenario_level_115_min12/param_uncertainty/2_skills/metrics.json"
SE_UNIDIM = UNIDIM_DIR / "param_uncertainty/metrics.json"

GRID_2SKILL = REPORTS / "eap_oos_grid_tutorbench_2skill/oos_per_cell_grid.csv"
GRID_UNIDIM = UNIDIM_DIR / "oos_grid_full/oos_per_cell_grid_excl.csv"

OUT_FIG = HERE / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

HEADLINE_EXCLUDE = "Qwen/Qwen1.5-1.8B"          # excluded from headline pool on both scales
REHABILITATED = "BSC-LT/salamandra-7b-instruct"  # rehabilitated under unidim

FLOOR, SE_TARGET = 20, 0.27


# ---------------------------------------------------------------------------
# full-bank EAP theta (selection-independent reference) per bank
# ---------------------------------------------------------------------------
def fullbank_eap_theta(bank_path: Path, matrix: pd.DataFrame, nodes_per_dim: int):
    """Return (models, theta_correctness) full-bank EAP over every graded criterion.

    Uses clamp negative-loading policy (the consume-time default) and the shared
    `eap_all_models` estimator. For the 2-skill bank we return the CORRECTNESS column.
    """
    records, dims, stats = scl.load_fitted_bank(bank_path, negative_policy="clamp")
    ids, A, b = scl.assemble_arrays(records, dims)
    cidx = dims.index("correctness")

    cols = [c for c in ids if c in matrix.columns]
    keep = [i for i, c in enumerate(ids) if c in matrix.columns]
    A, b = A[keep], b[keep]
    Y = matrix[cols].to_numpy(dtype=float)          # (n_models, n_items), may contain NaN
    mask = ~np.isnan(Y)

    grid, log_prior = scl.build_grid(len(dims), nodes_per_dim)
    theta = scl.eap_all_models(Y, mask, A, b, grid, log_prior)
    return list(matrix.index), theta[:, cidx], {
        "n_items_used": len(cols), "n_items_bank": len(ids),
        "dims": dims, "nodes_per_dim": nodes_per_dim, **stats,
    }


def pearson_spearman(x, y):
    from scipy.stats import pearsonr, spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    r = float(pearsonr(x[ok], y[ok])[0])
    rho = float(spearmanr(x[ok], y[ok])[0])
    return r, rho, int(ok.sum())


# ---------------------------------------------------------------------------
# gather numbers
# ---------------------------------------------------------------------------
def load_se_components():
    m2 = json.loads(SE_2SKILL.read_text(encoding="utf-8"))["se_components"]
    mu = json.loads(SE_UNIDIM.read_text(encoding="utf-8"))["se_components"]
    return {
        "2skill_correctness": m2["correctness"],
        "2skill_scaffolding": m2["scaffolding"],
        "unidim": mu["correctness"],
    }


def load_oos_rows():
    g2 = pd.read_csv(GRID_2SKILL)
    r2 = g2[(g2.floor == FLOOR) & (g2.se_target == SE_TARGET)].iloc[0]
    gu = pd.read_csv(GRID_UNIDIM)
    ru = gu[(gu.floor == FLOOR) & (gu.se_target == SE_TARGET)].iloc[0]
    return r2, ru


def build_table(se, r2, ru):
    """rows = metrics; cols = [2-skill correctness, 2-skill scaffolding, correctness-only unidim]."""
    NA = ""
    rows = [
        ("SE_ability (median, param-unc. bootstrap)",
         se["2skill_correctness"]["median_se_posterior"],
         se["2skill_scaffolding"]["median_se_posterior"],
         se["unidim"]["median_se_posterior"]),
        ("SE_param (median)",
         se["2skill_correctness"]["median_se_param"],
         se["2skill_scaffolding"]["median_se_param"],
         se["unidim"]["median_se_param"]),
        ("SE_total (median)",
         se["2skill_correctness"]["median_se_total"],
         se["2skill_scaffolding"]["median_se_total"],
         se["unidim"]["median_se_total"]),
        ("OOS recovery r @20/0.27",
         r2["r_correctness"], r2["r_scaffolding"], ru["r_correctness"]),
        ("slope @20/0.27",
         r2["slope_correctness"], r2["slope_scaffolding"], ru["slope_correctness"]),
        ("theta MAE @20/0.27",
         r2["theta_mae_correctness"], r2["theta_mae_scaffolding"], ru["theta_mae_correctness"]),
        ("median test length (scenarios)",
         r2["median_len"], r2["median_len"], ru["median_len"]),
        ("%reach SE_ability<=0.27",
         100 * r2["pct_reach_se_ability_correctness"],
         100 * r2["pct_reach_se_ability_scaffolding"],
         100 * ru["pct_reach_se_ability_correctness"]),
    ]
    df = pd.DataFrame(rows, columns=[
        "metric", "2skill_correctness", "2skill_scaffolding", "correctness_only_unidim"])
    return df


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_se_components(se, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = ["median_se_ability", "median_se_param", "median_se_total"]
    # map to actual keys (SE_ability == posterior)
    keymap = {"median_se_ability": "median_se_posterior",
              "median_se_param": "median_se_param",
              "median_se_total": "median_se_total"}
    labels = ["SE_ability\n(information)", "SE_param\n(fit stability)", "SE_total\n(combined)"]
    scales = [("2-skill correctness\n(of-record)", "2skill_correctness", "#c1666b"),
              ("correctness-only unidim\n(this study)", "unidim", "#4d648d")]
    x = np.arange(len(comps))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for i, (lab, key, col) in enumerate(scales):
        vals = [se[key][keymap[c]] for c in comps]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=lab,
                      color=col, edgecolor="k", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(SE_TARGET, ls=":", color="k", lw=1.2)
    ax.text(x[-1] + 0.55, SE_TARGET + 0.004, f"SE_ability target {SE_TARGET}",
            ha="right", va="bottom", fontsize=8, style="italic")

    # annotate the SE_param gap
    p2 = se["2skill_correctness"]["median_se_param"]
    pu = se["unidim"]["median_se_param"]
    gx = x[1]
    ax.annotate("", xy=(gx + 0.5 * width, pu), xytext=(gx - 0.5 * width, p2),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.0))
    ax.text(gx, (p2 + pu) / 2 + 0.02,
            f"SE_param gap\n{p2:.3f}->{pu:.3f}\n(-{100*(p2-pu)/p2:.0f}%)",
            ha="center", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff3cd", ec="k", lw=0.4))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("median SE (ability units)")
    ax.set_title("TutorBench ability-SE decomposition @ floor 20 / SE_ability 0.27\n"
                 "2-skill correctness vs correctness-only unidim "
                 "(scenario-level param-uncertainty bootstrap, N=115)", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_rank_agreement(df_theta, r, rho, n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df_theta[~df_theta["is_headline_excluded"]]
    x = d["theta_2skill_correctness"].to_numpy()
    y = d["theta_unidim_correctness"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    lo = float(min(x.min(), y.min())) - 0.3
    hi = float(max(x.max(), y.max())) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.6", lw=1.0, label="y = x")
    s, c = np.polyfit(x, y, 1)
    xs = np.array([lo, hi])
    ax.plot(xs, s * xs + c, color="#4d648d", lw=1.4,
            label=f"OLS slope={s:.3f}")
    ax.scatter(x, y, s=26, color="#c1666b", edgecolor="k", linewidth=0.3, alpha=0.85)

    # flag the rehabilitated model
    if REHABILITATED in set(d["model"]):
        rr = d[d["model"] == REHABILITATED].iloc[0]
        ax.scatter([rr["theta_2skill_correctness"]], [rr["theta_unidim_correctness"]],
                   s=70, facecolor="none", edgecolor="green", linewidth=1.6, zorder=5)
        ax.annotate("salamandra-7b-instruct\n(rehabilitated)",
                    (rr["theta_2skill_correctness"], rr["theta_unidim_correctness"]),
                    textcoords="offset points", xytext=(8, -22), fontsize=7.5, color="green")

    ax.set_xlabel("2-skill correctness  full-bank EAP θ")
    ax.set_ylabel("correctness-only unidim  full-bank EAP θ")
    ax.set_title("Rank agreement: same correctness axis?\n"
                 f"full-bank EAP θ (stop-independent), N={n} (headline pool)", fontsize=10.5)
    ax.text(0.04, 0.96, f"Pearson r = {r:.4f}\nSpearman ρ = {rho:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f0f0f0", ec="k", lw=0.4))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_recovery_efficiency(r2, ru, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("OOS recovery r", r2["r_correctness"], ru["r_correctness"], "{:.3f}"),
        ("θ MAE (lower=better)", r2["theta_mae_correctness"], ru["theta_mae_correctness"], "{:.3f}"),
        ("median length (scenarios)", r2["median_len"], ru["median_len"], "{:.0f}"),
        ("%reach SE_ability<=0.27", 100 * r2["pct_reach_se_ability_correctness"],
         100 * ru["pct_reach_se_ability_correctness"], "{:.0f}%"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
    colors = ["#c1666b", "#4d648d"]
    names = ["2-skill\ncorrectness", "correctness-only\nunidim"]
    for ax, (title, v2, vu, fmt) in zip(axes, panels):
        vals = [v2, vu]
        bars = ax.bar([0, 1], vals, color=colors, edgecolor="k", linewidth=0.4, width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.18)
    fig.suptitle("Recovery + efficiency @ floor 20 / SE_ability 0.27 (correctness axis, headline pool N=113/114)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    matrix = pd.read_csv(MATRIX, index_col=0)

    # --- full-bank EAP theta (rank agreement) ---
    m2, th2, meta2 = fullbank_eap_theta(BANK_2SKILL, matrix, nodes_per_dim=61)
    mu, thu, metau = fullbank_eap_theta(BANK_UNIDIM, matrix, nodes_per_dim=241)
    assert m2 == mu
    df_theta = pd.DataFrame({
        "model": m2,
        "theta_2skill_correctness": th2,
        "theta_unidim_correctness": thu,
    })
    df_theta["is_headline_excluded"] = df_theta["model"] == HEADLINE_EXCLUDE
    df_theta = df_theta.sort_values("theta_2skill_correctness", ascending=False).reset_index(drop=True)
    df_theta.to_csv(HERE / "fullbank_eap_theta_per_model.csv", index=False)

    pool = df_theta[~df_theta["is_headline_excluded"]]
    r_h, rho_h, n_h = pearson_spearman(pool["theta_2skill_correctness"], pool["theta_unidim_correctness"])
    r_all, rho_all, n_all = pearson_spearman(df_theta["theta_2skill_correctness"], df_theta["theta_unidim_correctness"])

    # --- tables + SE + OOS ---
    se = load_se_components()
    r2row, rurow = load_oos_rows()
    table = build_table(se, r2row, rurow)

    # round for display
    disp = table.copy()
    for col in ["2skill_correctness", "2skill_scaffolding", "correctness_only_unidim"]:
        disp[col] = disp[col].map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)
    table.to_csv(HERE / "comparison_table_20_0.27.csv", index=False)

    md = ["# TutorBench scale comparison @ floor 20 / SE_ability 0.27",
          "",
          "| metric | 2-skill correctness | 2-skill scaffolding | correctness-only unidim |",
          "|---|---:|---:|---:|"]
    for _, row in table.iterrows():
        def f(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        md.append(f"| {row['metric']} | {f(row['2skill_correctness'])} | "
                  f"{f(row['2skill_scaffolding'])} | {f(row['correctness_only_unidim'])} |")
    md += ["",
           "Notes:",
           "- SE_ability / SE_param / SE_total are medians from the scenario-level "
           "parameter-uncertainty bootstrap (N=115, eap-grid 61, n-boot 150, min-evals-per-skill 12, "
           "max-se 0.30). SE_ability == posterior SE; SE_total = sqrt(SE_ability^2 + SE_param^2).",
           "- OOS r / slope / θMAE / length / %reach are from the k=5 refit-per-fold OOS grids at "
           "floor 20 / SE_ability 0.27 (headline pool excludes flagged models).",
           "- median test length is the JOINT administered test; for the 2-skill scale one test "
           "measures both skills simultaneously (25 scenarios), unidim measures the single axis (20).",
           f"- %reach shown is SE_ability<=0.27 per axis. 2-skill BOTH-skills reach = "
           f"{100*r2row['pct_reach_se_ability_all']:.1f}%; unidim combined (SE_ability<=0.27 AND "
           f"SE_total<=0.30) = {100*rurow['pct_reach']:.1f}%.",
           ""]
    (HERE / "comparison_table_20_0.27.md").write_text("\n".join(md), encoding="utf-8")

    # --- figures ---
    fig_se_components(se, OUT_FIG / "se_components_grouped_bar.png")
    fig_rank_agreement(df_theta, r_h, rho_h, n_h, OUT_FIG / "rank_agreement_scatter.png")
    fig_recovery_efficiency(r2row, rurow, OUT_FIG / "recovery_efficiency_comparison.png")

    # --- summary json ---
    summary = {
        "operating_point": {"floor": FLOOR, "se_ability_target": SE_TARGET},
        "rank_agreement": {
            "headline_pool": {"n": n_h, "pearson_r": r_h, "spearman_rho": rho_h,
                              "excluded": HEADLINE_EXCLUDE},
            "all_115": {"n": n_all, "pearson_r": r_all, "spearman_rho": rho_all},
            "fullbank_eap_meta": {"2skill": meta2, "unidim": metau},
        },
        "se_components_median": se,
        "oos_2skill_20_0.27": {k: (float(r2row[k]) if k in r2row else None) for k in
                               ["r_correctness", "slope_correctness", "theta_mae_correctness",
                                "r_scaffolding", "slope_scaffolding", "theta_mae_scaffolding",
                                "median_len", "pct_reach_se_ability_correctness",
                                "pct_reach_se_ability_scaffolding", "pct_reach_se_ability_all"]},
        "oos_unidim_20_0.27": {k: float(rurow[k]) for k in
                               ["r_correctness", "slope_correctness", "theta_mae_correctness",
                                "median_len", "pct_reach_se_ability_correctness", "pct_reach"]},
        "artifacts": {
            "table_csv": "comparison_table_20_0.27.csv",
            "table_md": "comparison_table_20_0.27.md",
            "theta_csv": "fullbank_eap_theta_per_model.csv",
            "fig_se": "figures/se_components_grouped_bar.png",
            "fig_rank": "figures/rank_agreement_scatter.png",
            "fig_recovery": "figures/recovery_efficiency_comparison.png",
        },
        "status": "STUDY/LOCAL only; not committed; engine untouched.",
    }
    (HERE / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- console report ---
    print("=== Comparison table @ 20/0.27 ===")
    print(disp.to_string(index=False))
    print("\n=== Rank agreement (2-skill correctness theta vs unidim theta, full-bank EAP) ===")
    print(f"  headline pool (excl {HEADLINE_EXCLUDE}, N={n_h}): "
          f"Pearson r = {r_h:.4f}, Spearman rho = {rho_h:.4f}")
    print(f"  all 115: Pearson r = {r_all:.4f}, Spearman rho = {rho_all:.4f}")
    print("\nWrote:")
    for v in summary["artifacts"].values():
        print(f"  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
