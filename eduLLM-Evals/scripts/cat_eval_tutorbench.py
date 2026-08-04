"""FULL Computerized-Adaptive-Testing (CAT) evaluation of the DEFINITIVE 2-skill
TutorBench M2PL calibration, over all 82 models.

Purpose
-------
Produce a summary table + figures analogous to the unidimensional MCQ-CAT report
(``tutor_cat.mcq_irt.pipeline``), but for the TWO-DIMENSIONAL TutorBench bank whose
latent axes are:

* **correctness** (the collapsed content+diagnosis dimension), and
* **scaffolding**.

For every model we compute:

1. **FULL-BANK ability** ``(theta_correctness, theta_scaffolding)`` by EAP over the
   ENTIRE fitted item bank from the model's observed responses. This is the "truth"
   reference for recovery.
2. **Adaptive CAT ability**: starting from the prior ``theta=(0,0)``, ``U=diag(1,1)``,
   iteratively pick the not-yet-administered observed item with MAX (multidimensional)
   Fisher information ``p(1-p) * (m . m)`` at the current ability, administer the model's
   OBSERVED response, and update ``(theta, U)`` via the exact MIRT Newton/Laplace step.
   Stop when BOTH per-dim SEs < ``--se-target`` (after >= ``--min-items``) or at
   ``--max-items``. Records #items to cross SE<target per dimension and overall, plus
   the SE / ability trajectory.
3. **Metrics** aggregated over all 82 models:
   * **Recovery r** = Pearson corr(CAT ability, full-bank ability), per dimension.
   * **CAT items** = mean/median #items to SE<target (overall + per dimension).
   * **pIRT MAE** = mean |predicted accuracy - observed accuracy|, where predicted
     accuracy = mean over the model's observed items of ``P(pass | ability, item)``.
     Reported from CAT ability (headline) and from full-bank ability (ceiling).

Honesty check (out-of-sample)
-----------------------------
In-sample CAT recovery is somewhat circular (the bank was fit on the same 82 models it
now scores). We ALSO run an OUT-OF-SAMPLE variant: for each model we run the CAT with
its k-fold FOLD-TRAINED item params (``staging/kfold_full/fold_<f>_item_params.csv``;
the model was held out of that fold's fit) and correlate the OOS CAT ability against the
model's FULL-BANK ability. In-sample is the headline (matching the reference); OOS is the
honest generalisation check.

Reuse (no reimplementation of the M2PL)
---------------------------------------
* ``scripts/calibrate_mirt.py`` -- ``build_grid``, ``base_log_weights``,
  ``prior_log_weights`` (the same fixed Gauss-Hermite quadrature + standard-normal prior),
  and its ``cp.load_matrix`` loader.
* ``scripts/kfold_cat_pilot.py`` -- the EAP full-response ability + the max-Fisher-info
  adaptive administration pattern.
* ``tutor_cat.mirt`` -- ``update`` (PRD Eqs 1-3) and ``standard_errors``.

READ-ONLY: nothing is written back to the bank, the rubric bank, or the fitters.

Usage
-----
    python scripts/cat_eval_tutorbench.py
    python scripts/cat_eval_tutorbench.py --se-target 0.3 --max-items 100 --seed 20260730
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
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

DEFAULT_BANK = ROOT / "staging" / "calibration_mirt_full2skill.csv"
DEFAULT_MATRIX = ROOT / "staging" / "response_matrix_full_nonopt.csv"
DEFAULT_KFOLD_DIR = ROOT / "staging" / "kfold_full"
DEFAULT_OUT_DIR = ROOT / "staging" / "cat_eval"

DIMS = ("correctness", "scaffolding")
N_DIMS = 2

# The reference table's fixed structural facts for this bank (from the definitive fit
# manifest ``calibration_mirt_full2skill_manifest.json``).
ITEMS_KEPT = 3497
ITEMS_TOTAL = 6180


# ---------------------------------------------------------------------------
# ability estimation (reused math)
# ---------------------------------------------------------------------------


def eap_full(y_obs: np.ndarray, mask: np.ndarray, A: np.ndarray, b: np.ndarray,
             grid: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Full-response EAP ability over ALL observed items (the recovery reference).

    Identical E-step math to ``calibrate_mirt.fit_m2pl_em`` / ``kfold_cat_pilot.eap_full``:
    per-node log-likelihood from observed cells only -> posterior -> posterior mean.
    """
    ym = np.where(mask, y_obs, 0.0)
    nm = np.where(mask, 1.0 - y_obs, 0.0)
    eta = A @ grid.T - b[:, None]
    ll = ym @ log_expit(eta) + nm @ log_expit(-eta)
    joint = ll + log_prior
    post = np.exp(joint - logsumexp(joint))
    return post @ grid


def run_cat_person(y: np.ndarray, mask: np.ndarray, A: np.ndarray, b: np.ndarray,
                   min_items: int, max_items: int, se_target: float) -> dict:
    """Max-Fisher-info adaptive administration for one person over observed items.

    At each step the not-yet-administered observed item with the largest multidimensional
    Fisher information ``p(1-p) * (m . m)`` at the current ability is administered (its
    OBSERVED response is revealed) and ``(theta, U)`` are updated by the exact MIRT
    Newton/Laplace step (``tutor_cat.mirt.update``). SEs are monotone non-increasing, so
    the first step a dimension drops below ``se_target`` is its convergence point.

    Returns the final ability, per-dim + overall convergence counts, and the full
    SE / ability trajectory (indexed by #items administered).
    """
    obs_idx = np.where(mask)[0]
    theta = np.zeros(N_DIMS)
    U = np.eye(N_DIMS)
    ones_q = np.ones(N_DIMS)
    remaining = set(int(i) for i in obs_idx)

    se_trace: list[list[float]] = []
    theta_trace: list[list[float]] = []
    n_cross = [None, None]  # first #items where each dim SE < se_target

    n_admin = 0
    while remaining and n_admin < max_items:
        rem = np.fromiter(remaining, dtype=int)
        m = A[rem]
        p = expit(m @ theta - b[rem])
        info = p * (1.0 - p) * np.sum(m * m, axis=1)
        pick = int(rem[int(np.argmax(info))])
        theta, U, _ = tc_mirt.update(theta, U, A[pick], ones_q, float(b[pick]),
                                     int(y[pick]))
        remaining.discard(pick)
        n_admin += 1
        se = tc_mirt.standard_errors(U)
        se_trace.append([float(se[0]), float(se[1])])
        theta_trace.append([float(theta[0]), float(theta[1])])
        for d in range(N_DIMS):
            if n_cross[d] is None and float(se[d]) < se_target:
                n_cross[d] = n_admin
        if n_admin >= min_items and float(np.max(se)) < se_target:
            break

    se_final = tc_mirt.standard_errors(U)
    both = (n_cross[0] is not None and n_cross[1] is not None)
    n_overall = max(n_cross[0], n_cross[1]) if both else None
    return {
        "theta_cat": theta,
        "n_items": n_admin,
        "se_final": [float(se_final[0]), float(se_final[1])],
        "n_cross_correctness": n_cross[0],
        "n_cross_scaffolding": n_cross[1],
        "n_cross_overall": n_overall,
        "converged_overall": both,
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


def make_figures(df: pd.DataFrame, agg: dict, A: np.ndarray, b: np.ndarray,
                 fig_dir: Path, se_target: float, max_items: int) -> dict:
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

    # --- recovery scatters (per dimension) ---
    for dim in DIMS:
        full = df[f"theta_full_{dim}"].to_numpy()
        cat = df[f"theta_cat_{dim}"].to_numpy()
        r = agg["in_sample"]["recovery_r"][dim]
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.scatter(full, cat, s=28, alpha=0.75, edgecolor="k", linewidth=0.3)
        lo = min(full.min(), cat.min()) - 0.3
        hi = max(full.max(), cat.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"full-bank ability, {dim} (logits)")
        ax.set_ylabel(f"CAT ability, {dim} (logits)")
        ax.set_title(f"CAT recovers {dim} ability\n"
                     f"r = {r:.3f} (n = {len(df)} models)")
        ax.legend(loc="upper left", fontsize=9)
        _save(fig, f"recovery_scatter_{dim}.png")

    # --- CAT length histogram (correctness convergence; the axis that converges) ---
    lens = df["n_cross_correctness"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if lens.size:
        ax.hist(lens, bins=range(0, int(lens.max()) + 2), color="#4c72b0",
                edgecolor="white")
        mean_v, med_v = float(lens.mean()), float(np.median(lens))
        ax.axvline(mean_v, color="crimson", ls="--", lw=1.5, label=f"mean = {mean_v:.1f}")
        ax.axvline(med_v, color="green", ls=":", lw=1.5, label=f"median = {med_v:.0f}")
        ax.legend()
    ax.set_xlabel(f"items to reach correctness SE < {se_target}")
    ax.set_ylabel("models")
    conv = int(df["n_cross_correctness"].notna().sum())
    ax.set_title(f"Items to reach correctness SE < {se_target}\n"
                 f"{conv}/{len(df)} models converged (cap {max_items} items)")
    _save(fig, "cat_length_hist.png")

    # --- SE reduction curve (mean SE vs #items, both dimensions) ---
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    max_len = int(df["cat_n_items"].max())
    for dim, idx, color in [("correctness", 0, "#4c72b0"), ("scaffolding", 1, "#dd8452")]:
        means = []
        xs = []
        for step in range(max_len):
            vals = [tr[step][idx] for tr in df["se_trace"] if len(tr) > step]
            if len(vals) >= 5:  # only plot where >=5 models still administering
                means.append(float(np.mean(vals)))
                xs.append(step + 1)
        ax.plot(xs, means, label=f"{dim} (mean SE)", color=color, lw=2)
    ax.axhline(se_target, color="gray", ls="--", lw=1, label=f"SE target = {se_target}")
    ax.set_xlabel("items administered")
    ax.set_ylabel("mean posterior SE (logits)")
    ax.set_title("Posterior SE falls as items are administered")
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
    ax.set_xlabel("observed pass-rate")
    ax.set_ylabel("predicted pass-rate (from CAT ability)")
    ax.set_title(f"CAT predicts observed pass-rate\nMAE = {mae:.4f} (n = {len(df)} models)")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "pirt_calibration.png")

    # --- item info by skill (a_correctness vs a_scaffolding loadings) ---
    a_corr = A[:, 0][A[:, 0] > 0]
    a_scaff = A[:, 1][A[:, 1] > 0]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bins = np.linspace(0, max(a_corr.max(), a_scaff.max()) + 0.1, 40)
    ax.hist(a_corr, bins=bins, alpha=0.6, label=f"a_correctness (n={a_corr.size}, "
            f"med={np.median(a_corr):.2f})", color="#4c72b0")
    ax.hist(a_scaff, bins=bins, alpha=0.6, label=f"a_scaffolding (n={a_scaff.size}, "
            f"med={np.median(a_scaff):.2f})", color="#dd8452")
    ax.set_xlabel("discrimination loading a (non-zero loadings only)")
    ax.set_ylabel("items")
    ax.set_title("Item discrimination by skill axis")
    ax.legend()
    _save(fig, "item_info_by_skill.png")

    return paths


# ---------------------------------------------------------------------------
# core run
# ---------------------------------------------------------------------------


def load_bank(path: Path):
    df = pd.read_csv(path)
    items = df["criterion_id"].tolist()
    A = df[["a_correctness", "a_scaffolding"]].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    return items, A, b


def run_cat_over_models(models, Y, M, A, b, grid, log_prior,
                        min_items, max_items, se_target):
    """Run full-bank EAP + adaptive CAT for every model. Returns a per-model list."""
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
        rows.append({
            "model": model,
            "n_obs_items": int(mask.sum()),
            "theta_full_correctness": float(theta_full[0]),
            "theta_full_scaffolding": float(theta_full[1]),
            "theta_cat_correctness": float(theta_cat[0]),
            "theta_cat_scaffolding": float(theta_cat[1]),
            "cat_n_items": res["n_items"],
            "n_cross_correctness": res["n_cross_correctness"],
            "n_cross_scaffolding": res["n_cross_scaffolding"],
            "n_cross_overall": res["n_cross_overall"],
            "converged_overall": res["converged_overall"],
            "final_se_correctness": res["se_final"][0],
            "final_se_scaffolding": res["se_final"][1],
            "obs_acc": obs_acc,
            "pred_acc_cat": pred_acc_cat,
            "pred_acc_full": pred_acc_full,
            "se_trace": res["se_trace"],
            "theta_trace": res["theta_trace"],
        })
    return rows


def aggregate(df: pd.DataFrame, tag: str) -> dict:
    out = {}
    out["recovery_r"] = {
        d: pearson(df[f"theta_full_{d}"], df[f"theta_cat_{d}"]) for d in DIMS
    }
    # overall = correlation on the stacked (both-dim) ability points.
    full_all = np.concatenate([df["theta_full_correctness"], df["theta_full_scaffolding"]])
    cat_all = np.concatenate([df["theta_cat_correctness"], df["theta_cat_scaffolding"]])
    out["recovery_r"]["overall_stacked"] = pearson(full_all, cat_all)

    out["cat_items"] = {
        "correctness": {"mean": _mean(df["n_cross_correctness"]),
                        "median": _median(df["n_cross_correctness"]),
                        "converged": int(df["n_cross_correctness"].notna().sum())},
        "scaffolding": {"mean": _mean(df["n_cross_scaffolding"]),
                        "median": _median(df["n_cross_scaffolding"]),
                        "converged": int(df["n_cross_scaffolding"].notna().sum())},
        "overall": {"mean": _mean(df["n_cross_overall"]),
                    "median": _median(df["n_cross_overall"]),
                    "converged": int(df["n_cross_overall"].notna().sum())},
        "n_models": int(len(df)),
    }
    d_cat = (df["pred_acc_cat"] - df["obs_acc"]).abs()
    d_full = (df["pred_acc_full"] - df["obs_acc"]).abs()
    out["pirt_mae_cat"] = float(d_cat.mean())
    out["pirt_mae_full"] = float(d_full.mean())
    out["pirt_corr_cat"] = pearson(df["pred_acc_cat"], df["obs_acc"])
    out["tag"] = tag
    return out


def run_oos(models, model_to_fold, Y_full_by_item, items_full, item_col_idx,
            kfold_dir, df_full, grid, log_prior, min_items, max_items, se_target):
    """Out-of-sample CAT: for each model run the CAT with its fold's item params (the
    model was held out of that fold), then correlate against the model's full-bank
    ability (already in ``df_full``). Honest generalisation check."""
    full_theta = {r["model"]: (r["theta_full_correctness"], r["theta_full_scaffolding"])
                  for _, r in df_full.iterrows()}
    fold_cache: dict[int, dict] = {}
    rows = []
    for r_idx, model in enumerate(models):
        fold = model_to_fold[model]
        if fold not in fold_cache:
            fp = pd.read_csv(kfold_dir / f"fold_{fold}_item_params.csv")
            fitems = fp["criterion_id"].tolist()
            fA = fp[["a_correctness", "a_scaffolding"]].to_numpy(dtype=float)
            fb = fp["b"].to_numpy(dtype=float)
            fidx = {it: k for k, it in enumerate(fitems)}
            fold_cache[fold] = {"items": fitems, "A": fA, "b": fb, "idx": fidx}
        fc = fold_cache[fold]
        # This model's observed responses restricted to the fold's fitted items.
        cols = [item_col_idx[it] for it in fc["items"]]
        y = Y_full_by_item[r_idx, cols]
        mask = ~np.isnan(y)
        y = np.nan_to_num(y, nan=0.0)
        res = run_cat_person(y, mask, fc["A"], fc["b"], min_items, max_items, se_target)
        tc = res["theta_cat"]
        ft = full_theta[model]
        rows.append({
            "model": model,
            "fold": int(fold),
            "theta_full_correctness": ft[0],
            "theta_full_scaffolding": ft[1],
            "theta_cat_correctness": float(tc[0]),
            "theta_cat_scaffolding": float(tc[1]),
            "cat_n_items": res["n_items"],
            "n_cross_correctness": res["n_cross_correctness"],
            "n_cross_scaffolding": res["n_cross_scaffolding"],
            "n_cross_overall": res["n_cross_overall"],
            "final_se_correctness": res["se_final"][0],
            "final_se_scaffolding": res["se_final"][1],
        })
    oos_df = pd.DataFrame(rows)
    oos = {
        "recovery_r": {d: pearson(oos_df[f"theta_full_{d}"], oos_df[f"theta_cat_{d}"])
                       for d in DIMS},
        "cat_items": {
            "correctness": {"mean": _mean(oos_df["n_cross_correctness"]),
                            "median": _median(oos_df["n_cross_correctness"]),
                            "converged": int(oos_df["n_cross_correctness"].notna().sum())},
            "scaffolding": {"mean": _mean(oos_df["n_cross_scaffolding"]),
                            "median": _median(oos_df["n_cross_scaffolding"]),
                            "converged": int(oos_df["n_cross_scaffolding"].notna().sum())},
        },
        "n_models": int(len(oos_df)),
    }
    return oos_df, oos


def a_medians(A: np.ndarray) -> dict:
    return {
        "a_correctness_median_nonzero": float(np.median(A[:, 0][A[:, 0] > 0])),
        "a_scaffolding_median_nonzero": float(np.median(A[:, 1][A[:, 1] > 0])),
        "a_correctness_n_loading": int((A[:, 0] > 0).sum()),
        "a_scaffolding_n_loading": int((A[:, 1] > 0).sum()),
    }


# ---------------------------------------------------------------------------
# table writers
# ---------------------------------------------------------------------------


def build_summary_table(agg: dict, oos: dict, am: dict, se_target: float) -> pd.DataFrame:
    """Reference-shaped table with a TutorBench overall row + per-skill breakdown."""
    ins = agg["in_sample"]
    bank_health = {"correctness": "healthy", "scaffolding": "needs refit"}

    n_models = ins["cat_items"]["n_models"]

    def fmt_items(dim_key):
        c = ins["cat_items"][dim_key]
        if c["mean"] is None:
            return f"n/a (0/{n_models} conv.)"
        return f"{c['mean']:.1f} (median {c['median']:.0f}; {c['converged']}/{n_models} conv.)"

    rows = []
    # overall row
    rows.append({
        "Benchmark": "TutorBench (2-skill, overall)",
        "Models": agg["n_models"],
        "Items kept (fit_block / total)": f"{ITEMS_KEPT} / {ITEMS_TOTAL}",
        "Bank (healthy / needs refit)": "correctness healthy / scaffolding needs refit",
        "a median": (f"corr {am['a_correctness_median_nonzero']:.2f} / "
                     f"scaff {am['a_scaffolding_median_nonzero']:.2f}"),
        "Recovery r": (f"corr {ins['recovery_r']['correctness']:.3f} / "
                       f"scaff {ins['recovery_r']['scaffolding']:.3f}"),
        "CAT items": fmt_items("overall"),
        "pIRT MAE": f"{ins['pirt_mae_cat']:.4f}",
    })
    # per-skill rows
    for dim in DIMS:
        am_key = f"a_{dim}_median_nonzero"
        rows.append({
            "Benchmark": f"  - {dim}",
            "Models": agg["n_models"],
            "Items kept (fit_block / total)": (
                f"{am[f'a_{dim}_n_loading']} loading"),
            "Bank (healthy / needs refit)": bank_health[dim],
            "a median": f"{am[am_key]:.3f}",
            "Recovery r": f"{ins['recovery_r'][dim]:.3f}",
            "CAT items": fmt_items(dim),
            "pIRT MAE": (f"{ins['pirt_mae_cat']:.4f} (overall)"
                         if dim == "correctness" else "-"),
        })
    return pd.DataFrame(rows)


def table_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--kfold-dir", type=Path, default=DEFAULT_KFOLD_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--grid", type=int, default=7,
                   help="Gauss-Hermite nodes per dim (default 7 -> 49 nodes for 2-d).")
    p.add_argument("--se-target", type=float, default=0.3,
                   help="stop when BOTH per-dim SEs < this (after --min-items).")
    p.add_argument("--min-items", type=int, default=1)
    p.add_argument("--max-items", type=int, default=100,
                   help="hard cap on adaptive items (non-convergence handling).")
    p.add_argument("--seed", type=int, default=20260730)
    p.add_argument("--no-oos", action="store_true", help="skip the OOS k-fold check.")
    args = p.parse_args()

    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"

    print("=" * 72)
    print("FULL CAT evaluation - TutorBench definitive 2-skill M2PL")
    print("=" * 72)

    items, A, b = load_bank(args.bank)
    print(f"bank: {len(items)} fitted items ({(A[:,0]>0).sum()} load correctness, "
          f"{(A[:,1]>0).sum()} load scaffolding)")

    mat = cp.load_matrix(args.matrix)
    models = list(mat.index)
    sub = mat.reindex(columns=items)
    Yraw = sub.to_numpy(dtype=float)
    M = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    item_col_idx = {it: k for k, it in enumerate(items)}
    print(f"matrix: {len(models)} models x {len(items)} bank items "
          f"({int(M.sum())} observed cells, {M.mean()*100:.1f}% filled)")

    grid = cm.build_grid(N_DIMS, args.grid)
    base_logw = cm.base_log_weights(N_DIMS, args.grid)
    log_prior = cm.prior_log_weights(grid, base_logw, np.eye(N_DIMS))

    print(f"\nrunning full-bank EAP + adaptive CAT (SE target {args.se_target}, "
          f"cap {args.max_items}) over {len(models)} models ...")
    rows = run_cat_over_models(models, Y, M, A, b, grid, log_prior,
                               args.min_items, args.max_items, args.se_target)
    df = pd.DataFrame(rows)

    agg = {"n_models": int(len(df))}
    agg["in_sample"] = aggregate(df, "in_sample")
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
            print(f"WARNING: {len(missing)} models absent from fold assignments; "
                  "OOS will skip them.")
        oos_models = [m for m in models if m in model_to_fold]
        print(f"\nrunning OUT-OF-SAMPLE CAT with fold-trained params "
              f"({len(oos_models)} models) ...")
        oos_df, oos = run_oos(oos_models, model_to_fold, Yraw, items, item_col_idx,
                              args.kfold_dir, df, grid, log_prior,
                              args.min_items, args.max_items, args.se_target)
        agg["out_of_sample"] = oos

    # --- write per-model CSV (drop bulky traces) ---
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"bank": str(args.bank), "matrix": str(args.matrix),
                   "kfold_dir": str(args.kfold_dir), "grid": args.grid,
                   "se_target": args.se_target, "min_items": args.min_items,
                   "max_items": args.max_items, "seed": args.seed},
        "bank": {"n_items_fit": ITEMS_KEPT, "n_items_total": ITEMS_TOTAL, **am,
                 "bank_health": {"correctness": "healthy (cross-fold a r=0.80)",
                                 "scaffolding": "needs refit (cross-fold a r=0.48)"}},
        "n_models": agg["n_models"],
        **agg,
        "figures": fig_paths,
    }
    metrics_json = args.out_dir / "cat_metrics.json"
    with metrics_json.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # --- summary table (md + csv) ---
    table_df = build_summary_table(agg, oos, am, args.se_target)
    table_csv = args.out_dir / "cat_summary_table.csv"
    table_df.to_csv(table_csv, index=False)
    md = table_to_markdown(table_df)
    table_md = args.out_dir / "cat_summary_table.md"
    with table_md.open("w", encoding="utf-8") as fh:
        fh.write("# TutorBench 2-skill CAT summary table\n\n")
        fh.write(md + "\n")
        if oos is not None:
            fh.write("\n## Out-of-sample (k-fold fold-trained params) recovery\n\n")
            fh.write(f"- Recovery r (correctness): "
                     f"{oos['recovery_r']['correctness']:.3f}\n")
            fh.write(f"- Recovery r (scaffolding): "
                     f"{oos['recovery_r']['scaffolding']:.3f}\n")

    # --- console summary ---
    ins = agg["in_sample"]
    print("\n" + "=" * 72)
    print("RESULTS (in-sample headline)")
    print("=" * 72)
    print(f"Recovery r  : correctness={ins['recovery_r']['correctness']:.3f}  "
          f"scaffolding={ins['recovery_r']['scaffolding']:.3f}  "
          f"overall(stacked)={ins['recovery_r']['overall_stacked']:.3f}")
    n_models_ci = ins["cat_items"]["n_models"]
    for d in DIMS:
        c = ins["cat_items"][d]
        mv = f"{c['mean']:.1f}" if c["mean"] is not None else "n/a"
        print(f"CAT items ({d:11s}): mean={mv}  median={c['median']}  "
              f"converged={c['converged']}/{n_models_ci}")
    co = ins["cat_items"]["overall"]
    print(f"CAT items (overall both): mean={co['mean']}  median={co['median']}  "
          f"converged={co['converged']}/{n_models_ci}")
    print(f"pIRT MAE    : CAT={ins['pirt_mae_cat']:.4f}  "
          f"full-bank(ceiling)={ins['pirt_mae_full']:.4f}")
    print(f"a median    : correctness={am['a_correctness_median_nonzero']:.3f}  "
          f"scaffolding={am['a_scaffolding_median_nonzero']:.3f}")
    if oos is not None:
        print("\nOUT-OF-SAMPLE (fold-trained params) recovery r:")
        print(f"  correctness={oos['recovery_r']['correctness']:.3f}  "
              f"scaffolding={oos['recovery_r']['scaffolding']:.3f}")

    print("\nwrote:")
    for pth in [per_model_csv, metrics_json, table_csv, table_md]:
        print(f"  {pth}")
    for name, pth in fig_paths.items():
        print(f"  {pth}")
    print("\nSUMMARY TABLE (markdown):\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
