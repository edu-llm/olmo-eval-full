"""Build the full set of CAT ability-recovery scatter variants for the 2-skill
TutorBench M2PL pilot, so the recovery figures are visually consistent and
clearly labelled.

Three variants per axis (correctness, scaffolding):

* in-sample, coarse 7-node grid  -- the pilot's in-run EAP reference (quantised;
  x clamps at +/-3.75). Regenerated IN PLACE with a title that flags it as the
  quantised, non-headline variant. Filenames stay stable:
    figures/recovery_scatter_{dim}.png
* in-sample, fine 41-node grid    -- the full-response EAP reference recomputed
  on the fine Gauss-Hermite grid (same frozen item params, only the quadrature
  density changes), reusing ``build_correctness_leaderboard.fine_full_*``:
    figures/recovery_scatter_{dim}_finegrid.png
* out-of-sample (k-fold), fine grid -- the honest headline. y = OOS CAT theta
  (fold-trained params, from cat_per_model_oos.csv, one held-out point per
  model); x = the SAME fine-grid full-response reference:
    figures/oos_recovery_scatter_{dim}.png

The OOS ``theta_cat_*`` estimates are NOT recomputed -- only their reference
x-axis is de-quantised from the coarse 7-node grid onto the fine 41-node grid.
Nothing is re-fit; only the EAP integration grid density changes.

Reproduce:
    python scripts/cat_eval_tutorbench_multiskill.py --skills 2
    python scripts/build_recovery_figure_variants.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "cat_eval_policy" / "2skill"
FIG_DIR = OUT_DIR / "figures"
IN_CSV = OUT_DIR / "cat_per_model.csv"
OOS_CSV = OUT_DIR / "cat_per_model_oos.csv"

DIMS = ("correctness", "scaffolding")
DIM_COLORS = {"correctness": "#4c72b0", "scaffolding": "#dd8452"}
FINE_GRID = 41


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


def _finite(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite(x, y)
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite(x, y)
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def fine_full_theta(grid_nodes: int) -> dict[str, dict[str, float]]:
    """Full-response EAP ability on a fine grid for BOTH dims, over the SAME
    eligible item set the pilot CAT run consumed (exclude_from_fit removed, then
    point-biserial >= 0.05 filter). Extends
    ``build_correctness_leaderboard.fine_full_correctness_theta`` to scaffolding."""
    mh = _load_harness()
    cfg = mh.skillset_config(2)
    dims = cfg["dims"]
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

    out: dict[str, dict[str, float]] = {}
    for r, model in enumerate(models):
        theta = mh.eap_full(Y[r], M[r], A, b, grid, log_prior)
        out[model] = {dim: float(theta[i]) for i, dim in enumerate(dims)}
    return out


def scatter(
    x: np.ndarray,
    y: np.ndarray,
    dim: str,
    subtitle: str,
    out_path: Path,
    lim: tuple[float, float] | None = None,
) -> tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = pearson(x, y)
    rho = spearman(x, y)
    n = int(np.isfinite(x).sum())

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(x, y, s=34, alpha=0.8, edgecolor="k", linewidth=0.3, color=DIM_COLORS[dim])
    if lim is None:
        lo = min(np.nanmin(x), np.nanmin(y)) - 0.3
        hi = max(np.nanmax(x), np.nanmax(y)) + 0.3
    else:
        lo, hi = lim
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, ls=":", lw=0.5, alpha=0.4)
    ax.set_xlabel(f"full-response {dim} \u03b8 (reference)")
    ax.set_ylabel(f"CAT {dim} \u03b8 (adaptive)")
    ax.set_title(
        f"TutorBench 2-skill CAT recovery: {dim}\n"
        f"{subtitle}\n"
        f"Pearson r = {r:.3f} \u00b7 Spearman \u03c1 = {rho:.3f}  (n = {n} models)",
        fontsize=8.5,
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return r, rho


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fine = fine_full_theta(FINE_GRID)

    df_in = pd.read_csv(IN_CSV)
    df_oos = pd.read_csv(OOS_CSV)

    results: dict[str, dict[str, tuple[float, float]]] = {}

    for dim in DIMS:
        # --- assemble the three references / estimates, aligned by model name ---
        in_models = df_in["model"].tolist()
        coarse_ref = df_in[f"theta_full_{dim}"].to_numpy(float)
        in_cat = df_in[f"theta_cat_{dim}"].to_numpy(float)
        fine_ref_in = np.array([fine[m][dim] for m in in_models])

        oos_models = df_oos["model"].tolist()
        oos_cat = df_oos[f"theta_cat_{dim}"].to_numpy(float)
        fine_ref_oos = np.array([fine[m][dim] for m in oos_models])

        # shared axis limits for the two fine-grid variants of this dim, so the
        # in-sample and OOS panels are directly comparable.
        allx = np.concatenate([fine_ref_in, fine_ref_oos])
        ally = np.concatenate([in_cat, oos_cat])
        lo = float(min(allx.min(), ally.min())) - 0.3
        hi = float(max(allx.max(), ally.max())) + 0.3
        fine_lim = (lo, hi)

        results[dim] = {}
        # 1. coarse 7-node in-sample (relabelled, filename stable)
        results[dim]["coarse_in"] = scatter(
            coarse_ref, in_cat, dim,
            "in-sample \u00b7 coarse 7-node grid (quantized \u2014 see fine-grid/OOS variants)",
            FIG_DIR / f"recovery_scatter_{dim}.png",
        )
        # 2. fine 41-node in-sample
        results[dim]["fine_in"] = scatter(
            fine_ref_in, in_cat, dim,
            "in-sample \u00b7 fine 41-node grid",
            FIG_DIR / f"recovery_scatter_{dim}_finegrid.png",
            lim=fine_lim,
        )
        # 3. OOS k-fold, fine grid (headline)
        results[dim]["oos"] = scatter(
            fine_ref_oos, oos_cat, dim,
            "out-of-sample (k-fold) \u00b7 fine grid",
            FIG_DIR / f"oos_recovery_scatter_{dim}.png",
            lim=fine_lim,
        )

    print("=" * 70)
    print("Recovery figure variants (2-skill) -- Pearson r / Spearman rho")
    print("=" * 70)
    for dim in DIMS:
        for key, label in (
            ("coarse_in", "in-sample coarse 7-node"),
            ("fine_in", "in-sample fine 41-node"),
            ("oos", "OOS k-fold fine     "),
        ):
            r, rho = results[dim][key]
            print(f"{dim:12s} | {label:24s} | r = {r:.4f} | rho = {rho:.4f}")
    print("\nfigures written under:", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
