"""Build a correctness-axis leaderboard of the 82 tutor models from the frozen
2-skill TutorBench M2PL bank.

Consumes the per-model output of the definitive 2-skill CAT run
(``reports/cat_eval_policy/2skill/cat_per_model.csv``, produced by
``scripts/cat_eval_tutorbench_multiskill.py --skills 2`` on the frozen ridge=1e-2
bank + ``response_matrix_full_nonopt.csv`` under policy defaults, floor=15). This
is scoring/ranking only -- no re-fit, no parameter changes.

Two abilities per model on the CORRECTNESS axis:
* ``theta_correctness_full`` -- full-response EAP over ALL of a model's observed
  items under the frozen bank. The reference / leaderboard sort key. Computed
  here on a FINE Gauss-Hermite grid (``--grid`` nodes/dim, default 41) via the
  harness's own ``eap_full`` on the exact same eligible-item set (exclusions +
  point-biserial filter) the pilot CAT run used, so the reference is a smooth,
  continuous ability rather than the coarse 7-node quantisation the pilot's
  in-run EAP produces. ``theta_correctness_full_grid7`` (the pilot's default-grid
  value straight from ``cat_per_model.csv``) is retained for provenance.
* ``theta_correctness_cat``  -- adaptive-test estimate (frozen params, policy
  defaults), with its posterior SE and the number of items administered. Taken
  verbatim from the pilot CAT run.

Outputs (all under ``reports/leaderboard_correctness_2skill/``):
* ``leaderboard_correctness_2skill.csv`` -- one row per model, ranked.
* ``figures/leaderboard_correctness_caterpillar.png``
* ``figures/cat_vs_full_correctness_recovery.png``

Reproduce:
    python scripts/cat_eval_tutorbench_multiskill.py --skills 2
    python scripts/build_correctness_leaderboard.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "cat_eval_policy" / "2skill" / "cat_per_model.csv"
OUT_DIR = ROOT / "reports" / "leaderboard_correctness_2skill"
FIG_DIR = OUT_DIR / "figures"


def _load_harness():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    path = ROOT / "scripts" / "cat_eval_tutorbench_multiskill.py"
    spec = importlib.util.spec_from_file_location("cat_eval_tutorbench_multiskill", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cat_eval_tutorbench_multiskill"] = mod
    spec.loader.exec_module(mod)
    return mod


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def fine_full_correctness_theta(grid_nodes: int) -> dict[str, float]:
    """Full-response EAP correctness ability on a fine grid, reusing the harness's
    ``eap_full`` over the SAME eligible item set the pilot CAT run consumed
    (exclude_from_fit removed, then the point-biserial >= 0.05 filter)."""
    mh = _load_harness()
    cfg = mh.skillset_config(2)
    dims = cfg["dims"]  # ("correctness", "scaffolding")
    items, A, b = mh.load_bank(cfg["bank"], cfg["a_cols"])
    excluded = mh.load_excluded_criteria(mh.DEFAULT_CURATED)
    keep = np.array([it not in excluded for it in items])
    items = [it for it, k in zip(items, keep, strict=False) if k]
    A, b = A[keep], b[keep]

    mat = mh.cp.load_matrix(cfg["matrix"])
    models = list(mat.index)
    Yraw = mat.reindex(columns=items).to_numpy(dtype=float)

    pk, _ = mh.apply_pbis_filter(Yraw, items, 0.05)
    A, b, Yraw = A[pk], b[pk], Yraw[:, pk]

    M = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)

    grid = mh.cm.build_grid(len(dims), grid_nodes)
    base_logw = mh.cm.base_log_weights(len(dims), grid_nodes)
    log_prior = mh.cm.prior_log_weights(grid, base_logw, np.eye(len(dims)))

    out: dict[str, float] = {}
    for r, model in enumerate(models):
        theta = mh.eap_full(Y[r], M[r], A, b, grid, log_prior)
        out[model] = float(theta[0])  # correctness dim
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=41,
                    help="Gauss-Hermite nodes/dim for the fine full-response EAP reference.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC)
    n = len(df)

    fine = fine_full_correctness_theta(args.grid)
    df["theta_full_correctness_fine"] = df["model"].map(fine)

    theta_full = df["theta_full_correctness_fine"].to_numpy(float)
    theta_full_g7 = df["theta_full_correctness"].to_numpy(float)
    theta_cat = df["theta_cat_correctness"].to_numpy(float)
    recovery_r = pearson(theta_full, theta_cat)
    recovery_r_g7 = pearson(theta_full_g7, theta_cat)

    mu, sd = float(theta_full.mean()), float(theta_full.std(ddof=0))
    z = (theta_full - mu) / sd if sd > 0 else np.zeros_like(theta_full)
    # percentile rank (0-100) of the full-response correctness ability
    pct = pd.Series(theta_full).rank(pct=True).to_numpy() * 100.0

    lb = pd.DataFrame(
        {
            "model_name": df["model"],
            "theta_correctness_full": theta_full,
            "theta_correctness_cat": theta_cat,
            "cat_se_correctness": df["final_se_correctness"].to_numpy(float),
            "items_administered": df["cat_n_items"].astype(int),
            "cat_items_loading_correctness": df["exposure_correctness"].astype(int),
            "theta_scaffolding_full": df["theta_full_scaffolding"].to_numpy(float),
            "z_correctness": z,
            "percentile_correctness": pct,
            "correctness_converged": df["n_cross_correctness"].notna(),
            "theta_correctness_full_grid7": theta_full_g7,
        }
    )
    # sort by the fine full-response reference; break any residual ties by CAT theta.
    lb = lb.sort_values(
        ["theta_correctness_full", "theta_correctness_cat"], ascending=False
    ).reset_index(drop=True)
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))

    round_cols = {
        "theta_correctness_full": 4,
        "theta_correctness_cat": 4,
        "cat_se_correctness": 4,
        "theta_scaffolding_full": 4,
        "z_correctness": 3,
        "percentile_correctness": 1,
        "theta_correctness_full_grid7": 4,
    }
    for c, nd in round_cols.items():
        lb[c] = lb[c].round(nd)

    csv_path = OUT_DIR / "leaderboard_correctness_2skill.csv"
    lb.to_csv(csv_path, index=False)

    # ---- figure 1: ranked caterpillar of correctness ability -----------------
    order = lb.iloc[::-1].reset_index(drop=True)  # worst at bottom -> best at top
    ypos = np.arange(len(order))
    ci = 1.96 * order["cat_se_correctness"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.5, max(10, 0.22 * n)))
    ax.errorbar(
        order["theta_correctness_cat"], ypos, xerr=ci, fmt="none",
        ecolor="#9bb0d6", elinewidth=1.4, capsize=2, zorder=1,
        label="CAT theta \u00b1 1.96\u00b7SE",
    )
    ax.scatter(order["theta_correctness_cat"], ypos, s=18, color="#dd8452",
               zorder=2, label="CAT theta")
    ax.scatter(order["theta_correctness_full"], ypos, s=26, color="#4c72b0",
               marker="D", zorder=3, label="full-response theta (sort key)")
    ax.axvline(0.0, color="gray", ls=":", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order["model_name"], fontsize=6)
    ax.set_ylim(-1, len(order))
    ax.set_xlabel("correctness ability  \u03b8 (logits)")
    ax.set_title(
        f"TutorBench correctness-axis leaderboard ({n} tutor models)\n"
        f"frozen 2-skill M2PL bank (ridge=1e-2, floor=15); "
        f"CAT\u2013full recovery r = {recovery_r:.3f}"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    cat_fig = FIG_DIR / "leaderboard_correctness_caterpillar.png"
    fig.savefig(cat_fig, dpi=140)
    plt.close(fig)

    # ---- figure 2: CAT vs full recovery scatter ------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(theta_full, theta_cat, s=34, alpha=0.8, edgecolor="k",
               linewidth=0.3, color="#4c72b0")
    lo = min(theta_full.min(), theta_cat.min()) - 0.3
    hi = max(theta_full.max(), theta_cat.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("full-response correctness \u03b8 (reference)")
    ax.set_ylabel("CAT correctness \u03b8 (adaptive)")
    ax.set_title(
        f"CAT recovers the correctness ranking\n"
        f"in-sample Pearson r = {recovery_r:.3f}  (n = {n} models)"
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    sc_fig = FIG_DIR / "cat_vs_full_correctness_recovery.png"
    fig.savefig(sc_fig, dpi=140)
    plt.close(fig)

    # Spearman rank correlation as a ranking-fidelity companion to Pearson.
    rho = pd.Series(theta_full).corr(pd.Series(theta_cat), method="spearman")
    mean_items = float(df["cat_n_items"].mean())
    median_items = float(df["cat_n_items"].median())

    print(f"models                    : {n}")
    print(f"grid (fine EAP reference)  : {args.grid} nodes/dim")
    print(f"CAT-vs-full recovery r     : {recovery_r:.4f} (fine grid)  "
          f"| {recovery_r_g7:.4f} (pilot grid-7)  (Spearman rho {rho:.4f})")
    print(f"CAT items administered     : mean {mean_items:.1f}, median {median_items:.0f}")
    print(f"leaderboard csv            : {csv_path}")
    print(f"caterpillar figure         : {cat_fig}")
    print(f"recovery scatter figure    : {sc_fig}")
    print("\nTop 10:")
    print(lb.head(10)[["rank", "model_name", "theta_correctness_full",
                       "theta_correctness_cat", "cat_se_correctness",
                       "items_administered"]].to_string(index=False))
    print("\nBottom 5:")
    print(lb.tail(5)[["rank", "model_name", "theta_correctness_full",
                      "theta_correctness_cat", "cat_se_correctness",
                      "items_administered"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
