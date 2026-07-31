"""FULL Computerized-Adaptive-Testing (CAT) evaluation of the 3-skill
[correctness, scaffolding, presentation] TutorBench M2PL calibration, over all 82 models.

Purpose
-------
The empirical companion to ``scripts/cat_eval_tutorbench.py`` (the DEFINITIVE 2-skill
CAT eval). It mirrors that script's structure/columns/figures EXACTLY so the two are
directly comparable, but adds the third **presentation** axis (the optional
``style_surface`` criteria) fitted in Calibration Run 6. The decision question it
answers is *incremental*: on complete data presentation is statistically separable
(Run 6: latent r(presentation,correctness)=0.945, r(presentation,scaffolding)=-0.559;
3-dim beats both 2-skill folds on AIC/BIC; a_presentation cross-fold r=0.712 > scaffolding
0.488). But is it a *useful independent CAT axis*, or does it just track correctness?

Latent axes (Run 6 slot-repurposing of the confirmatory M2PL Q-matrix):

* **correctness** (collapsed content+diagnosis),
* **scaffolding**, and
* **presentation** (``dimension=='style_surface'`` optional criteria; 656 items survive).

For every model we compute:

1. **FULL-BANK ability** ``(theta_correctness, theta_scaffolding, theta_presentation)``
   by EAP over the ENTIRE fitted 3-d item bank from the model's observed responses --
   the recovery "truth" reference.
2. **Adaptive CAT ability**: from the prior ``theta=(0,0,0)``, ``U=diag(1,1,1)``,
   iteratively pick the not-yet-administered observed item with MAX (multidimensional)
   Fisher information ``p(1-p) * (m . m)`` at the current ability, administer the model's
   OBSERVED response, and update ``(theta, U)`` via the exact MIRT Newton/Laplace step
   (``tutor_cat.mirt.update``). Stop when ALL per-dim SEs < ``--se-target`` (after
   >= ``--min-items``) or at ``--max-items``. Records #items to cross SE<target per
   dimension + overall, the SE / ability trajectory, and **which skill each administered
   item loads** (item-exposure by skill -- does the CAT actually "want" presentation?).
3. **Metrics** aggregated over 82 models, per skill: Recovery r = corr(CAT, full-bank);
   CAT items to SE<target; pIRT MAE = mean |predicted - observed| accuracy.

Incremental-value analysis (the decision driver)
------------------------------------------------
* corr(CAT presentation ability, CAT correctness ability) -- does presentation just
  track correctness (Run 6 person-level latent r=0.945 hypothesis)?
* item-exposure share by skill during adaptive testing -- does the selector pick
  presentation items or is it dominated by correctness?
* delta in correctness/scaffolding recovery vs the 2-skill run (adding presentation
  helps / hurts / neutral) -- read from ``reports/cat_eval/cat_metrics.json``.
* presentation recovery (in-sample + OOS) side by side with scaffolding.

Out-of-sample honesty check
---------------------------
In-sample CAT recovery is circular (the bank was fit on the same 82 models it scores).
The OOS variant runs the CAT with each model's k-fold FOLD-TRAINED 3-skill item params
(``staging/kfold_full_3skill/fold_<f>_item_params.csv``; the model was held out of that
fold's fit) and correlates the OOS CAT ability against the model's FULL-BANK ability.
In-sample is the headline; OOS is the honest generalisation check.

Reuse (no reimplementation of the M2PL)
---------------------------------------
* ``scripts/calibrate_mirt.py`` -- ``build_grid``, ``base_log_weights``,
  ``prior_log_weights`` (fixed Gauss-Hermite quadrature + standard-normal prior),
  and its ``cp.load_matrix`` loader.
* ``scripts/kfold_cat_pilot.py`` -- the EAP full-response ability + max-Fisher-info
  adaptive administration pattern (here generalised to 3 dims).
* ``tutor_cat.mirt`` -- ``update`` (PRD Eqs 1-3) and ``standard_errors``.

READ-ONLY: nothing is written back to the bank, the rubric bank, or the fitters.

Usage
-----
    python scripts/cat_eval_tutorbench_3skill.py
    python scripts/cat_eval_tutorbench_3skill.py --se-target 0.3 --max-items 100 --seed 20260730
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


cm = _load_module("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
cp = cm.cp
from tutor_cat import mirt as tc_mirt  # noqa: E402

# Run 6's full 3-dim per-item params (slot-repurposed columns; see load_bank).
DEFAULT_BANK = ROOT / "staging" / "run6_presentation" / "calibration_mirt.csv"
DEFAULT_MATRIX = ROOT / "staging" / "response_matrix_full.csv"
DEFAULT_KFOLD_DIR = ROOT / "staging" / "kfold_full_3skill"
DEFAULT_OUT_DIR = ROOT / "staging" / "cat_eval_3skill"
# The 2-skill headline metrics for the side-by-side comparison (read-only reference).
DEFAULT_REF_2SKILL = ROOT / "reports" / "cat_eval" / "cat_metrics.json"

DIMS = ("correctness", "scaffolding", "presentation")
N_DIMS = 3
DIM_COLORS = {"correctness": "#4c72b0", "scaffolding": "#dd8452", "presentation": "#55a868"}

# Structural facts for this bank (Run 6 manifest calibration_mirt_manifest.json).
ITEMS_KEPT = 4156
ITEMS_TOTAL = 6845
# Cross-fold a-loading stability (Run 6 kfold_presentation_stability.json), used only
# for the descriptive "bank health" column of the summary table.
BANK_HEALTH = {
    "correctness": "healthy (cross-fold a r=0.77)",
    "scaffolding": "needs refit (cross-fold a r=0.49)",
    "presentation": "stable (cross-fold a r=0.71)",
}
BANK_HEALTH_SHORT = {
    "correctness": "healthy",
    "scaffolding": "needs refit",
    "presentation": "stable",
}


# ---------------------------------------------------------------------------
# ability estimation (reused math)
# ---------------------------------------------------------------------------


def eap_full(
    y_obs: np.ndarray,
    mask: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    grid: np.ndarray,
    log_prior: np.ndarray,
) -> np.ndarray:
    """Full-response EAP ability over ALL observed items (the recovery reference).

    Identical E-step math to ``calibrate_mirt.fit_m2pl_em`` / ``kfold_cat_pilot.eap_full``.
    """
    ym = np.where(mask, y_obs, 0.0)
    nm = np.where(mask, 1.0 - y_obs, 0.0)
    eta = A @ grid.T - b[:, None]
    ll = ym @ log_expit(eta) + nm @ log_expit(-eta)
    joint = ll + log_prior
    post = np.exp(joint - logsumexp(joint))
    return post @ grid


def run_cat_person(
    y: np.ndarray,
    mask: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    min_items: int,
    max_items: int,
    se_target: float,
) -> dict:
    """Max-Fisher-info adaptive administration for one person over observed items.

    At each step the not-yet-administered observed item with the largest multidimensional
    Fisher information ``p(1-p) * (m . m)`` is administered (its OBSERVED response is
    revealed) and ``(theta, U)`` are updated by the exact MIRT Newton/Laplace step
    (``tutor_cat.mirt.update``). SEs are monotone non-increasing, so the first step a
    dimension drops below ``se_target`` is its convergence point. Stops when ALL per-dim
    SEs are below target (after ``min_items``) or at ``max_items``.

    Also records, per administered item, WHICH skills it loads (``A[pick, d] > 0``) so
    the caller can measure item-exposure by skill (the incremental-value question).
    """
    obs_idx = np.where(mask)[0]
    theta = np.zeros(N_DIMS)
    U = np.eye(N_DIMS)
    ones_q = np.ones(N_DIMS)
    remaining = set(int(i) for i in obs_idx)

    se_trace: list[list[float]] = []
    theta_trace: list[list[float]] = []
    n_cross: list[int | None] = [None] * N_DIMS
    exposure = np.zeros(N_DIMS, dtype=int)  # #administered items loading each dim
    pattern_counts: dict[str, int] = {}  # 3-bit load pattern -> count

    n_admin = 0
    while remaining and n_admin < max_items:
        rem = np.fromiter(remaining, dtype=int)
        m = A[rem]
        p = expit(m @ theta - b[rem])
        info = p * (1.0 - p) * np.sum(m * m, axis=1)
        pick = int(rem[int(np.argmax(info))])
        loads = A[pick] > 0
        exposure += loads.astype(int)
        key = "".join("1" if loads[d] else "0" for d in range(N_DIMS))
        pattern_counts[key] = pattern_counts.get(key, 0) + 1
        theta, U, _ = tc_mirt.update(theta, U, A[pick], ones_q, float(b[pick]), int(y[pick]))
        remaining.discard(pick)
        n_admin += 1
        se = tc_mirt.standard_errors(U)
        se_trace.append([float(se[d]) for d in range(N_DIMS)])
        theta_trace.append([float(theta[d]) for d in range(N_DIMS)])
        for d in range(N_DIMS):
            if n_cross[d] is None and float(se[d]) < se_target:
                n_cross[d] = n_admin
        if n_admin >= min_items and float(np.max(se)) < se_target:
            break

    se_final = tc_mirt.standard_errors(U)
    all_crossed = all(c is not None for c in n_cross)
    n_overall = max(c for c in n_cross) if all_crossed else None
    return {
        "theta_cat": theta,
        "n_items": n_admin,
        "se_final": [float(se_final[d]) for d in range(N_DIMS)],
        "n_cross": n_cross,
        "n_cross_overall": n_overall,
        "converged_overall": all_crossed,
        "exposure": exposure,
        "pattern_counts": pattern_counts,
        "se_trace": se_trace,
        "theta_trace": theta_trace,
    }


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------


def pearson(x, y) -> float | None:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 2 or xa.std() == 0 or ya.std() == 0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _clean(vals) -> list[float]:
    out = []
    for v in vals:
        if v is None:
            continue
        fv = float(v)
        if np.isnan(fv):
            continue
        out.append(fv)
    return out


def _median(vals) -> float | None:
    vals = _clean(vals)
    return float(np.median(vals)) if vals else None


def _mean(vals) -> float | None:
    vals = _clean(vals)
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(
    df: pd.DataFrame,
    agg: dict,
    A: np.ndarray,
    b: np.ndarray,
    fig_dir: Path,
    se_target: float,
    max_items: int,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def _save(fig, name):
        p = fig_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths[name] = str(p.resolve())

    # --- recovery scatters (per dimension, incl. presentation) ---
    for dim in DIMS:
        full = df[f"theta_full_{dim}"].to_numpy()
        cat = df[f"theta_cat_{dim}"].to_numpy()
        r = agg["in_sample"]["recovery_r"][dim]
        rstr = f"{r:.3f}" if r is not None else "n/a"
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.scatter(full, cat, s=28, alpha=0.75, edgecolor="k", linewidth=0.3, color=DIM_COLORS[dim])
        lo = min(full.min(), cat.min()) - 0.3
        hi = max(full.max(), cat.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full-bank EAP ability ({dim})")
        ax.set_ylabel(f"CAT ability ({dim})")
        ax.set_title(
            f"TutorBench 3-skill CAT recovery: {dim}\nPearson r = {rstr} (n = {len(df)} models)"
        )
        ax.legend(loc="upper left", fontsize=9)
        _save(fig, f"recovery_scatter_{dim}.png")

    # --- CAT length histogram (correctness convergence, matches 2-skill fig) ---
    lens = df["n_cross_correctness"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if lens.size:
        ax.hist(lens, bins=range(0, int(lens.max()) + 2), color="#4c72b0", edgecolor="white")
        mean_v, med_v = float(lens.mean()), float(np.median(lens))
        ax.axvline(mean_v, color="crimson", ls="--", lw=1.5, label=f"mean = {mean_v:.1f}")
        ax.axvline(med_v, color="green", ls=":", lw=1.5, label=f"median = {med_v:.0f}")
        ax.legend()
    ax.set_xlabel(f"# items to reach correctness SE < {se_target}")
    ax.set_ylabel("# models")
    conv = int(df["n_cross_correctness"].notna().sum())
    ax.set_title(
        f"TutorBench 3-skill CAT length (correctness axis)\n"
        f"{conv}/{len(df)} models converged; cap = {max_items} items"
    )
    _save(fig, "cat_length_hist.png")

    # --- SE reduction curve (mean SE vs #items, all 3 dims) ---
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    max_len = int(df["cat_n_items"].max())
    for idx, dim in enumerate(DIMS):
        means, xs = [], []
        for step in range(max_len):
            vals = [tr[step][idx] for tr in df["se_trace"] if len(tr) > step]
            if len(vals) >= 5:
                means.append(float(np.mean(vals)))
                xs.append(step + 1)
        ax.plot(xs, means, label=f"{dim} (mean SE)", color=DIM_COLORS[dim], lw=2)
    ax.axhline(se_target, color="gray", ls="--", lw=1, label=f"SE target = {se_target}")
    ax.set_xlabel("# items administered")
    ax.set_ylabel("mean posterior SE")
    ax.set_title("TutorBench 3-skill CAT: mean SE reduction by dimension")
    ax.legend()
    _save(fig, "se_reduction_curve.png")

    # --- pIRT calibration (predicted vs actual accuracy, from CAT ability) ---
    pred = df["pred_acc_cat"].to_numpy()
    act = df["obs_acc"].to_numpy()
    mae = agg["in_sample"]["pirt_mae_cat"]
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(act, pred, s=28, alpha=0.75, edgecolor="k", linewidth=0.3)
    lo = min(pred.min(), act.min()) - 0.02
    hi = max(pred.max(), act.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("actual observed pass-rate")
    ax.set_ylabel("predicted accuracy (from CAT ability)")
    ax.set_title(f"TutorBench 3-skill pIRT calibration\nMAE = {mae:.4f} (n = {len(df)} models)")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "pirt_calibration.png")

    # --- item info by skill (a_* loadings, all 3 dims) ---
    a_by_dim = {dim: A[:, i][A[:, i] > 0] for i, dim in enumerate(DIMS)}
    amax = max(v.max() for v in a_by_dim.values())
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bins = np.linspace(0, amax + 0.1, 40)
    for dim in DIMS:
        v = a_by_dim[dim]
        ax.hist(
            v,
            bins=bins,
            alpha=0.55,
            label=f"a_{dim} (n={v.size}, med={np.median(v):.2f})",
            color=DIM_COLORS[dim],
        )
    ax.set_xlabel("discrimination loading a (non-zero loadings only)")
    ax.set_ylabel("# items")
    ax.set_title("TutorBench 3-skill item discrimination by skill axis")
    ax.legend()
    _save(fig, "item_info_by_skill.png")

    # --- item-exposure by skill during CAT (the incremental-value figure) ---
    exp = agg["in_sample"]["item_exposure"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    xlabs = list(DIMS)
    cat_share = [exp["exposure_share_by_skill"][d] for d in DIMS]
    bank_share = [exp["bank_loading_share_by_skill"][d] for d in DIMS]
    x = np.arange(len(xlabs))
    w = 0.38
    ax.bar(
        x - w / 2,
        bank_share,
        width=w,
        label="bank availability (share of items)",
        color="#bbbbbb",
        edgecolor="k",
        linewidth=0.3,
    )
    ax.bar(
        x + w / 2,
        cat_share,
        width=w,
        label="CAT selections (share of administered)",
        color=[DIM_COLORS[d] for d in DIMS],
        edgecolor="k",
        linewidth=0.3,
    )
    for xi, cs in zip(x, cat_share, strict=False):
        ax.text(xi + w / 2, cs + 0.01, f"{cs * 100:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabs)
    ax.set_ylabel("share of items loading skill")
    ax.set_title(
        "TutorBench 3-skill CAT: item-exposure by skill\n"
        "(does the selector actually want presentation items?)"
    )
    ax.legend(fontsize=8)
    _save(fig, "item_exposure_by_skill.png")

    return paths


# ---------------------------------------------------------------------------
# core run
# ---------------------------------------------------------------------------


def load_bank(path: Path):
    """Load Run 6's full 3-dim item params. The CSV uses the fitter's SKILLS slot
    column names (``a_content, a_diagnosis, a_scaffolding``); under Run 6's
    slot-repurposing these carry (correctness, scaffolding, presentation)."""
    df = pd.read_csv(path)
    items = df["criterion_id"].tolist()
    A = df[["a_content", "a_diagnosis", "a_scaffolding"]].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    return items, A, b


def run_cat_over_models(models, Y, M, A, b, grid, log_prior, min_items, max_items, se_target):
    rows = []
    for r, model in enumerate(models):
        mask = M[r]
        theta_full = eap_full(Y[r], mask, A, b, grid, log_prior)
        res = run_cat_person(Y[r], mask, A, b, min_items, max_items, se_target)
        theta_cat = res["theta_cat"]
        obs = np.where(mask)[0]
        obs_acc = float(Y[r][obs].mean())
        pred_acc_cat = float(expit(A[obs] @ theta_cat - b[obs]).mean())
        pred_acc_full = float(expit(A[obs] @ theta_full - b[obs]).mean())
        row = {
            "model": model,
            "n_obs_items": int(mask.sum()),
            "cat_n_items": res["n_items"],
            "n_cross_overall": res["n_cross_overall"],
            "converged_overall": res["converged_overall"],
            "obs_acc": obs_acc,
            "pred_acc_cat": pred_acc_cat,
            "pred_acc_full": pred_acc_full,
            "se_trace": res["se_trace"],
            "theta_trace": res["theta_trace"],
        }
        for i, dim in enumerate(DIMS):
            row[f"theta_full_{dim}"] = float(theta_full[i])
            row[f"theta_cat_{dim}"] = float(theta_cat[i])
            row[f"n_cross_{dim}"] = res["n_cross"][i]
            row[f"final_se_{dim}"] = res["se_final"][i]
            row[f"exposure_{dim}"] = int(res["exposure"][i])
        rows.append(row)
    return rows


def aggregate(df: pd.DataFrame, tag: str) -> dict:
    out = {}
    out["recovery_r"] = {d: pearson(df[f"theta_full_{d}"], df[f"theta_cat_{d}"]) for d in DIMS}
    full_all = np.concatenate([df[f"theta_full_{d}"].to_numpy() for d in DIMS])
    cat_all = np.concatenate([df[f"theta_cat_{d}"].to_numpy() for d in DIMS])
    out["recovery_r"]["overall_stacked"] = pearson(full_all, cat_all)

    out["cat_items"] = {"n_models": int(len(df))}
    for d in DIMS:
        out["cat_items"][d] = {
            "mean": _mean(df[f"n_cross_{d}"]),
            "median": _median(df[f"n_cross_{d}"]),
            "converged": int(df[f"n_cross_{d}"].notna().sum()),
        }
    out["cat_items"]["overall"] = {
        "mean": _mean(df["n_cross_overall"]),
        "median": _median(df["n_cross_overall"]),
        "converged": int(df["n_cross_overall"].notna().sum()),
    }
    d_cat = (df["pred_acc_cat"] - df["obs_acc"]).abs()
    d_full = (df["pred_acc_full"] - df["obs_acc"]).abs()
    out["pirt_mae_cat"] = float(d_cat.mean())
    out["pirt_mae_full"] = float(d_full.mean())
    out["pirt_corr_cat"] = pearson(df["pred_acc_cat"], df["obs_acc"])
    out["tag"] = tag
    return out


def a_medians(A: np.ndarray) -> dict:
    out = {}
    for i, dim in enumerate(DIMS):
        col = A[:, i]
        nz = col[col > 0]
        out[f"a_{dim}_median_nonzero"] = float(np.median(nz)) if nz.size else None
        out[f"a_{dim}_n_loading"] = int((col > 0).sum())
    return out


def exposure_summary(df: pd.DataFrame, A: np.ndarray) -> dict:
    """Item-exposure-by-skill during CAT (summed over all models) + bank availability.

    ``exposure_<dim>`` counts, per model, how many ADMINISTERED items load that dim.
    Items can multi-load, so the shares are of (item, load) incidences, not disjoint.
    ``bank_loading_share_by_skill`` is the share of the item BANK that loads each dim --
    the null the selector would match if it ignored information.
    """
    total_admin = int(df["cat_n_items"].sum())
    exp_counts = {d: int(df[f"exposure_{d}"].sum()) for d in DIMS}
    exp_share = {d: (exp_counts[d] / total_admin if total_admin else None) for d in DIMS}
    n_items = A.shape[0]
    bank_share = {d: float((A[:, i] > 0).sum()) / n_items for i, d in enumerate(DIMS)}
    return {
        "total_items_administered": total_admin,
        "exposure_count_by_skill": exp_counts,
        "exposure_share_by_skill": exp_share,
        "bank_loading_share_by_skill": bank_share,
        "presentation_over_bank_ratio": (
            (exp_share["presentation"] / bank_share["presentation"])
            if bank_share["presentation"]
            else None
        ),
    }


def run_oos(
    mat,
    oos_models,
    model_to_fold,
    kfold_dir,
    df_full,
    grid,
    log_prior,
    min_items,
    max_items,
    se_target,
):
    """Out-of-sample CAT: for each model run the CAT with its fold's 3-skill item params
    (the model was held out of that fold), then correlate against the model's full-bank
    ability (already in ``df_full``)."""
    full_theta = {
        r["model"]: {d: r[f"theta_full_{d}"] for d in DIMS} for _, r in df_full.iterrows()
    }
    rows = []
    for f in sorted(set(model_to_fold[m] for m in oos_models)):
        fp = pd.read_csv(kfold_dir / f"fold_{f}_item_params.csv")
        items = fp["criterion_id"].tolist()
        fA = fp[["a_correctness", "a_scaffolding", "a_presentation"]].to_numpy(dtype=float)
        fb = fp["b"].to_numpy(dtype=float)
        test_models = [m for m in oos_models if model_to_fold[m] == f]
        sub = mat.loc[test_models].reindex(columns=items)
        Yraw = sub.to_numpy(dtype=float)
        Mobs = ~np.isnan(Yraw)
        Yv = np.nan_to_num(Yraw, nan=0.0)
        for ridx, model in enumerate(test_models):
            res = run_cat_person(Yv[ridx], Mobs[ridx], fA, fb, min_items, max_items, se_target)
            tc = res["theta_cat"]
            row = {
                "model": model,
                "fold": int(f),
                "cat_n_items": res["n_items"],
                "n_cross_overall": res["n_cross_overall"],
            }
            for i, dim in enumerate(DIMS):
                row[f"theta_full_{dim}"] = full_theta[model][dim]
                row[f"theta_cat_{dim}"] = float(tc[i])
                row[f"n_cross_{dim}"] = res["n_cross"][i]
                row[f"final_se_{dim}"] = res["se_final"][i]
                row[f"exposure_{dim}"] = int(res["exposure"][i])
            rows.append(row)
    oos_df = pd.DataFrame(rows)
    oos = {
        "recovery_r": {
            d: pearson(oos_df[f"theta_full_{d}"], oos_df[f"theta_cat_{d}"]) for d in DIMS
        },
        "cat_items": {
            d: {
                "mean": _mean(oos_df[f"n_cross_{d}"]),
                "median": _median(oos_df[f"n_cross_{d}"]),
                "converged": int(oos_df[f"n_cross_{d}"].notna().sum()),
            }
            for d in DIMS
        },
        "cat_ability_corr_presentation_correctness": pearson(
            oos_df["theta_cat_presentation"], oos_df["theta_cat_correctness"]
        ),
        "n_models": int(len(oos_df)),
    }
    return oos_df, oos


# ---------------------------------------------------------------------------
# incremental-value analysis
# ---------------------------------------------------------------------------


def incremental_value(df: pd.DataFrame, agg: dict, oos: dict | None, ref2: dict | None) -> dict:
    ins = agg["in_sample"]
    iv: dict = {}
    # 1. Does CAT presentation ability just track CAT correctness ability?
    iv["cat_ability_corr"] = {
        "presentation_vs_correctness": pearson(
            df["theta_cat_presentation"], df["theta_cat_correctness"]
        ),
        "presentation_vs_scaffolding": pearson(
            df["theta_cat_presentation"], df["theta_cat_scaffolding"]
        ),
        "correctness_vs_scaffolding": pearson(
            df["theta_cat_correctness"], df["theta_cat_scaffolding"]
        ),
        "full_bank_presentation_vs_correctness": pearson(
            df["theta_full_presentation"], df["theta_full_correctness"]
        ),
        "note": (
            "Run 6 latent (person) r(presentation,correctness)=0.945; compare the "
            "CAT-estimated ability correlation here."
        ),
    }
    # 2. item-exposure by skill (does the CAT want presentation items?)
    iv["item_exposure"] = ins["item_exposure"]
    # 3. recovery vs the 2-skill run (does adding presentation move corr/scaff?)
    iv["recovery_vs_2skill"] = {}
    if ref2 is not None:
        r2_ins = ref2.get("in_sample", {}).get("recovery_r", {})
        r2_oos = ref2.get("out_of_sample", {}).get("recovery_r", {})
        for d in ("correctness", "scaffolding"):
            three_in = ins["recovery_r"].get(d)
            two_in = r2_ins.get(d)
            three_oos = oos["recovery_r"].get(d) if oos else None
            two_oos = r2_oos.get(d)
            iv["recovery_vs_2skill"][d] = {
                "in_sample_2skill": two_in,
                "in_sample_3skill": three_in,
                "in_sample_delta": (
                    None if (three_in is None or two_in is None) else three_in - two_in
                ),
                "oos_2skill": two_oos,
                "oos_3skill": three_oos,
                "oos_delta": (
                    None if (three_oos is None or two_oos is None) else three_oos - two_oos
                ),
            }
    # 4. presentation recovery next to scaffolding
    iv["presentation_vs_scaffolding_recovery"] = {
        "presentation_in_sample": ins["recovery_r"].get("presentation"),
        "scaffolding_in_sample": ins["recovery_r"].get("scaffolding"),
        "presentation_oos": oos["recovery_r"].get("presentation") if oos else None,
        "scaffolding_oos": oos["recovery_r"].get("scaffolding") if oos else None,
    }
    return iv


# ---------------------------------------------------------------------------
# table writers
# ---------------------------------------------------------------------------


def build_summary_table(agg: dict, am: dict, se_target: float) -> pd.DataFrame:
    ins = agg["in_sample"]
    n_models = ins["cat_items"]["n_models"]

    def fmt_items(dim_key):
        c = ins["cat_items"][dim_key]
        if c["mean"] is None:
            return f"n/a (0/{n_models} conv.)"
        return f"{c['mean']:.1f} (median {c['median']:.0f}; {c['converged']}/{n_models} conv.)"

    rows = []
    rows.append(
        {
            "Benchmark": "TutorBench (3-skill, overall)",
            "Models": agg["n_models"],
            "Items kept (fit_block / total)": f"{ITEMS_KEPT} / {ITEMS_TOTAL}",
            "Bank (healthy / needs refit)": (
                "correctness healthy / scaffolding needs refit / presentation stable"
            ),
            "a median": " / ".join(f"{d[:5]} {am[f'a_{d}_median_nonzero']:.2f}" for d in DIMS),
            "Recovery r": " / ".join(
                f"{d[:5]} "
                + (f"{ins['recovery_r'][d]:.3f}" if ins["recovery_r"][d] is not None else "n/a")
                for d in DIMS
            ),
            "CAT items": fmt_items("overall"),
            "pIRT MAE": f"{ins['pirt_mae_cat']:.4f}",
        }
    )
    for dim in DIMS:
        rv = ins["recovery_r"][dim]
        rows.append(
            {
                "Benchmark": f"  - {dim}",
                "Models": agg["n_models"],
                "Items kept (fit_block / total)": f"{am[f'a_{dim}_n_loading']} loading",
                "Bank (healthy / needs refit)": BANK_HEALTH_SHORT[dim],
                "a median": f"{am[f'a_{dim}_median_nonzero']:.3f}",
                "Recovery r": f"{rv:.3f}" if rv is not None else "n/a",
                "CAT items": fmt_items(dim),
                "pIRT MAE": (
                    f"{ins['pirt_mae_cat']:.4f} (overall)" if dim == "correctness" else "-"
                ),
            }
        )
    return pd.DataFrame(rows)


def table_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--kfold-dir", type=Path, default=DEFAULT_KFOLD_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--ref-2skill", type=Path, default=DEFAULT_REF_2SKILL)
    p.add_argument(
        "--grid",
        type=int,
        default=7,
        help="Gauss-Hermite nodes per dim (default 7 -> 343 nodes for 3-d).",
    )
    p.add_argument(
        "--se-target",
        type=float,
        default=0.3,
        help="stop when ALL per-dim SEs < this (after --min-items).",
    )
    p.add_argument("--min-items", type=int, default=1)
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260730)
    p.add_argument("--no-oos", action="store_true", help="skip the OOS k-fold check.")
    args = p.parse_args()

    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"

    print("=" * 72)
    print("FULL CAT evaluation - TutorBench 3-skill M2PL [correctness, scaffolding, presentation]")
    print("=" * 72)

    items, A, b = load_bank(args.bank)
    print(
        f"bank: {len(items)} fitted items "
        f"({(A[:, 0] > 0).sum()} load correctness, {(A[:, 1] > 0).sum()} load scaffolding, "
        f"{(A[:, 2] > 0).sum()} load presentation)"
    )

    mat = cp.load_matrix(args.matrix)
    models = list(mat.index)
    sub = mat.reindex(columns=items)
    Yraw = sub.to_numpy(dtype=float)
    M = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    print(
        f"matrix: {len(models)} models x {len(items)} bank items "
        f"({int(M.sum())} observed cells, {M.mean() * 100:.1f}% filled)"
    )

    grid = cm.build_grid(N_DIMS, args.grid)
    base_logw = cm.base_log_weights(N_DIMS, args.grid)
    log_prior = cm.prior_log_weights(grid, base_logw, np.eye(N_DIMS))

    print(
        f"\nrunning full-bank EAP + adaptive CAT (SE target {args.se_target}, "
        f"cap {args.max_items}) over {len(models)} models ..."
    )
    rows = run_cat_over_models(
        models, Y, M, A, b, grid, log_prior, args.min_items, args.max_items, args.se_target
    )
    df = pd.DataFrame(rows)

    agg = {"n_models": int(len(df))}
    agg["in_sample"] = aggregate(df, "in_sample")
    agg["in_sample"]["item_exposure"] = exposure_summary(df, A)
    am = a_medians(A)

    # --- OOS honesty check ---
    oos_df = None
    oos = None
    if not args.no_oos:
        with (args.kfold_dir / "fold_assignments.json").open(encoding="utf-8") as fh:
            fa = json.load(fh)
        model_to_fold = fa["model_to_fold"]
        missing = [m for m in models if m not in model_to_fold]
        if missing:
            print(
                f"WARNING: {len(missing)} models absent from fold assignments; OOS will skip them."
            )
        oos_models = [m for m in models if m in model_to_fold]
        print(
            f"\nrunning OUT-OF-SAMPLE CAT with fold-trained 3-skill params "
            f"({len(oos_models)} models) ..."
        )
        oos_df, oos = run_oos(
            mat,
            oos_models,
            model_to_fold,
            args.kfold_dir,
            df,
            grid,
            log_prior,
            args.min_items,
            args.max_items,
            args.se_target,
        )
        agg["out_of_sample"] = oos

    # --- 2-skill reference for the side-by-side ---
    ref2 = None
    if args.ref_2skill.is_file():
        with args.ref_2skill.open(encoding="utf-8") as fh:
            ref2 = json.load(fh)

    iv = incremental_value(df, agg, oos, ref2)
    agg["incremental_value"] = iv

    # --- write per-model CSVs (drop bulky traces) ---
    csv_cols = [c for c in df.columns if c not in ("se_trace", "theta_trace")]
    per_model_csv = args.out_dir / "cat_per_model.csv"
    df[csv_cols].to_csv(per_model_csv, index=False)
    if oos_df is not None:
        oos_df.to_csv(args.out_dir / "cat_per_model_oos.csv", index=False)

    # --- figures ---
    print("\ngenerating figures ...")
    fig_paths = make_figures(df, agg, A, b, fig_dir, args.se_target, args.max_items)

    # --- aggregate metrics JSON ---
    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "bank": str(args.bank),
            "matrix": str(args.matrix),
            "kfold_dir": str(args.kfold_dir),
            "grid": args.grid,
            "se_target": args.se_target,
            "min_items": args.min_items,
            "max_items": args.max_items,
            "seed": args.seed,
            "ref_2skill": str(args.ref_2skill),
        },
        "bank": {
            "n_items_fit": ITEMS_KEPT,
            "n_items_total": ITEMS_TOTAL,
            **am,
            "bank_health": BANK_HEALTH,
            "note": (
                "Run 6 slot-repurposed 3-dim M2PL: a_content->a_correctness, "
                "a_diagnosis->a_scaffolding, a_scaffolding->a_presentation."
            ),
        },
        "n_models": agg["n_models"],
        **agg,
        "figures": fig_paths,
    }
    metrics_json = args.out_dir / "cat_metrics.json"
    with metrics_json.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # --- summary table (md + csv) ---
    table_df = build_summary_table(agg, am, args.se_target)
    table_csv = args.out_dir / "cat_summary_table.csv"
    table_df.to_csv(table_csv, index=False)
    md = table_to_markdown(table_df)
    table_md = args.out_dir / "cat_summary_table.md"
    with table_md.open("w", encoding="utf-8") as fh:
        fh.write(
            "# TutorBench 3-skill CAT summary table (correctness, scaffolding, presentation)\n\n"
        )
        fh.write(md + "\n")
        if oos is not None:
            fh.write("\n## Out-of-sample (k-fold fold-trained 3-skill params) recovery\n\n")
            for d in DIMS:
                rv = oos["recovery_r"][d]
                fh.write(f"- Recovery r ({d}): " + (f"{rv:.3f}\n" if rv is not None else "n/a\n"))

    # --- console summary ---
    ins = agg["in_sample"]
    print("\n" + "=" * 72)
    print("RESULTS (in-sample headline)")
    print("=" * 72)
    print(
        "Recovery r  : "
        + "  ".join(
            f"{d}=" + (f"{ins['recovery_r'][d]:.3f}" if ins["recovery_r"][d] is not None else "n/a")
            for d in DIMS
        )
        + f"  overall(stacked)={ins['recovery_r']['overall_stacked']:.3f}"
    )
    n_ci = ins["cat_items"]["n_models"]
    for d in DIMS:
        c = ins["cat_items"][d]
        mv = f"{c['mean']:.1f}" if c["mean"] is not None else "n/a"
        print(
            f"CAT items ({d:12s}): mean={mv}  median={c['median']}  "
            f"converged={c['converged']}/{n_ci}"
        )
    co = ins["cat_items"]["overall"]
    print(
        f"CAT items (overall all): mean={co['mean']}  median={co['median']}  "
        f"converged={co['converged']}/{n_ci}"
    )
    print(
        f"pIRT MAE    : CAT={ins['pirt_mae_cat']:.4f}  "
        f"full-bank(ceiling)={ins['pirt_mae_full']:.4f}"
    )
    exp = ins["item_exposure"]
    print("\nitem-exposure share by skill during CAT (bank availability in parens):")
    for d in DIMS:
        print(
            f"  {d:12s}: {exp['exposure_share_by_skill'][d] * 100:5.1f}%  "
            f"(bank {exp['bank_loading_share_by_skill'][d] * 100:.1f}%)"
        )
    print(
        f"  presentation exposure / bank-availability ratio = "
        f"{exp['presentation_over_bank_ratio']:.2f}x"
    )
    cac = iv["cat_ability_corr"]
    print(
        f"\nCAT-ability corr presentation<->correctness = "
        f"{cac['presentation_vs_correctness']:.3f}  "
        f"(full-bank {cac['full_bank_presentation_vs_correctness']:.3f}; "
        f"Run6 latent 0.945)"
    )
    print(f"CAT-ability corr presentation<->scaffolding = {cac['presentation_vs_scaffolding']:.3f}")
    if iv["recovery_vs_2skill"]:
        print("\nrecovery vs 2-skill run:")
        for d, v in iv["recovery_vs_2skill"].items():
            di = v["in_sample_delta"]
            do = v["oos_delta"]
            print(
                f"  {d:12s}: in-sample {v['in_sample_2skill']:.3f}->"
                f"{v['in_sample_3skill']:.3f} (delta {di:+.3f}); "
                f"OOS {v['oos_2skill']:.3f}->{v['oos_3skill']:.3f} (delta {do:+.3f})"
            )
    if oos is not None:
        print("\nOUT-OF-SAMPLE (fold-trained params) recovery r:")
        print(
            "  "
            + "  ".join(
                f"{d}="
                + (f"{oos['recovery_r'][d]:.3f}" if oos["recovery_r"][d] is not None else "n/a")
                for d in DIMS
            )
        )

    print("\nwrote:")
    for pth in [per_model_csv, metrics_json, table_csv, table_md]:
        print(f"  {pth}")
    for _name, pth in fig_paths.items():
        print(f"  {pth}")
    print("\nSUMMARY TABLE (markdown):\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
