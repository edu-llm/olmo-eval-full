"""Person-level k-fold cross-validation of the 2-skill (collapsed) M2PL calibration.

Why this exists
---------------
The frozen 2-skill M2PL item bank (``--collapse content,diagnosis`` in
``scripts/calibrate_mirt.py``) was calibrated on the SAME 82 TutorBench models that
the adaptive test (CAT) then scores. That is circular: the CAT test-takers ARE the
calibration crowd, so in-sample fit says nothing about how the calibration
generalises to an UNSEEN model. This harness answers the honest question directly:

    Fit the 2-skill M2PL on a TRAIN subset of models, freeze the item params, then
    score the HELD-OUT models from their own responses (EAP) and measure how well
    the frozen items predict the held-out cells.

It is strictly READ-ONLY with respect to the calibration pipeline: it IMPORTS and
reuses the exact fitter internals from ``scripts/calibrate_mirt.py`` (same EM/MML,
same Gauss-Hermite quadrature, same ridge, same zero-variance / all-NaN / Q-less
item dropping, same ``collapse_q_matrix`` mechanism) and NEVER edits the fitter, the
rubric bank, or writes item params back. It only writes new artifacts under
``staging/kfold/``.

What it computes
----------------
* Person folds (default k=5): partition the 82 models by a seeded shuffle.
* Per fold f: fit the 2-skill M2PL on TRAIN = models not in f (same settings as the
  Run-1 collapse), freeze ``(a_correctness, a_scaffolding, b)`` for the items that
  survive the TRAIN-fold variance filter, then EAP-score each TEST person over the
  fixed 2-d quadrature grid using ONLY their observed responses on the fitted items,
  and predict ``P(pass)`` for every observed (test-person, fitted-item) cell.
* Out-of-sample metrics on those held-out cells: log-loss, accuracy@0.5, AUC, Brier.
* In-sample baseline: fit once on all 82, EAP-score every person from their OWN
  responses, predict the observed cells, compute the same metrics -> optimism gap.
* Item-param cross-fold stability: correlate per-item a_correctness, a_scaffolding,
  b across folds (items fit in >= 2 folds) -> a direct read on N=82 identifiability.

The 2-skill structure is built EXACTLY as Run 1 did: the 3-skill confirmatory Q
(content, diagnosis, scaffolding) is collapsed via
``calibrate_mirt.collapse_q_matrix(Q, ["content", "diagnosis"])`` into the 2-d
``[content+diagnosis, scaffolding]`` layout. We label dim 0 "correctness" (the
merged content+diagnosis skill) and dim 1 "scaffolding" in all outputs.

Usage
-----
    # default 5-fold CV, grid 7 (49 nodes for the 2-d model), seed 20260729:
    python scripts/kfold_cv_mirt.py

    # 10-fold, custom seed:
    python scripts/kfold_cv_mirt.py --k 10 --seed 123

    # faster smoke test:
    python scripts/kfold_cv_mirt.py --k 3 --grid 3 --max-iter 20

Outputs (under --out-dir, default staging/kfold/)
-------------------------------------------------
* fold_assignments.json / .csv          -- which model is in which fold.
* fold_<f>_item_params.csv              -- frozen a_correctness/a_scaffolding/b per fold.
* metrics_per_fold.csv                  -- OOS metrics for each fold.
* metrics_aggregate.json                -- pooled OOS + per-fold spread + in-sample gap.
* insample_baseline.csv / .json         -- all-82 in-sample metrics + per-item params.
* item_param_stability.csv / .json      -- cross-fold correlations of a/b.
* kfold_summary.json                    -- everything, plus run provenance.
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
from scipy.stats import rankdata

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


# Reuse the EXACT fitter internals (EM/MML, quadrature, collapse, block prep,
# zero-variance dropping) from the calibration script -- never reimplemented here.
cm = _load_module("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
cp = cm.cp  # calibrate_partial (loader / select_sparse / split_zero_variance)
SKILLS = list(cm.SKILLS)  # (content, diagnosis, scaffolding)

# 2-skill collapse: content+diagnosis -> "correctness"; scaffolding kept.
COLLAPSE_SKILLS = ["content", "diagnosis"]
DIM_LABELS = ["correctness", "scaffolding"]  # order matches collapse_q_matrix output

DEFAULT_MATRIX = ROOT / "staging" / "response_matrix.csv"
DEFAULT_RUBRICS = ROOT / "data" / "TutorBench" / "curated" / "rubrics_qmatrix_curated.jsonl"
DEFAULT_OUT_DIR = ROOT / "staging" / "kfold"

EPS = 1e-12


# ---------------------------------------------------------------------------
# fold construction
# ---------------------------------------------------------------------------


def make_folds(models: list[str], k: int, seed: int) -> list[list[str]]:
    """Partition ``models`` into ``k`` roughly-equal folds by a seeded shuffle."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds: list[list[str]] = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


# ---------------------------------------------------------------------------
# fit / score
# ---------------------------------------------------------------------------


def fit_2skill(
    train_df: pd.DataFrame, q_by: dict, args: argparse.Namespace
) -> dict:
    """Fit the 2-skill collapsed M2PL on the TRAIN models.

    Reuses ``calibrate_mirt.prepare_block`` (select_sparse + split_zero_variance +
    align_q_rows) so the item set surviving the TRAIN-fold variance filter matches
    the calibration pipeline exactly, then collapses the 3-skill Q to 2-skill and
    calls the shared ``fit_m2pl_em``.

    Returns dict with items list, A (n_items, 2), b, R, and the block diagnostics.
    """
    Y, M, Q3, items, block_df, diag = cm.prepare_block(train_df, q_by)
    Q2, labels, info = cm.collapse_q_matrix(Q3, COLLAPSE_SKILLS)
    fit = cm.fit_m2pl_em(
        Y, M, Q2, args.grid,
        estimate_corr=args.estimate_latent_corr,
        ridge=args.ridge, max_iter=args.max_iter, tol=args.tol,
    )
    return {
        "items": items,
        "A": fit["A"],          # (n_items, 2): col0 correctness, col1 scaffolding
        "b": fit["b"],
        "R": fit["R"],
        "collapsed_labels": labels,
        "loglik": fit["loglik"],
        "n_iter": fit["n_iter"],
        "converged": fit["converged"],
        "diag": diag,
    }


def eap_predict(
    score_df: pd.DataFrame,
    items: list[str],
    A: np.ndarray,
    b: np.ndarray,
    R: np.ndarray,
    grid: np.ndarray,
    log_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """EAP-score each person in ``score_df`` on the FIXED item params, then predict.

    Holding (A, b) fixed, each person's 2-d ability is the posterior mean (EAP) over
    the quadrature grid computed from ONLY their observed responses on ``items``
    (holes are marginalised, exactly as the fitter's E-step does). We then predict
    ``P(pass) = sigmoid(a_j . theta_hat - b_j)`` for every observed cell.

    Returns (y_obs, p_obs, theta, obs_mask): flattened observed labels + predicted
    probabilities over all (person, fitted-item) observed cells, the per-person EAP
    thetas, and the (n_persons, n_items) observation mask.
    """
    sub = score_df.reindex(columns=items)
    Yraw = sub.to_numpy(dtype=float)
    Mobs = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    YM = np.where(Mobs, Y, 0.0)
    NM = np.where(Mobs, 1.0 - Y, 0.0)

    eta = A @ grid.T - b[:, None]          # (n_items, n_nodes)
    logP = log_expit(eta)
    log1mP = log_expit(-eta)

    LL = YM @ logP + NM @ log1mP           # (n_persons, n_nodes)
    joint = LL + log_prior[None, :]
    person_ll = logsumexp(joint, axis=1)
    post = np.exp(joint - person_ll[:, None])   # (n_persons, n_nodes)
    theta = post @ grid                    # (n_persons, 2)

    P = expit(theta @ A.T - b[None, :])    # (n_persons, n_items)

    y_obs = Y[Mobs]
    p_obs = P[Mobs]
    return y_obs, p_obs, theta, Mobs


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = float(np.mean((p >= 0.5).astype(float) == y))
    brier = float(np.mean((p - y) ** 2))
    return {
        "n_cells": int(len(y)),
        "pos_rate": float(y.mean()),
        "log_loss": logloss,
        "accuracy": acc,
        "auc": _auc(y, p),
        "brier": brier,
    }


def _spread(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if len(arr) == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# item-param cross-fold stability
# ---------------------------------------------------------------------------


def item_param_tables(fold_fits: list[dict]) -> dict[str, pd.DataFrame]:
    """Wide per-parameter tables: rows = criterion_id, cols = fold_<f>."""
    tables: dict[str, dict] = {"a_correctness": {}, "a_scaffolding": {}, "b": {}}
    for f, fit in enumerate(fold_fits):
        A, b, items = fit["A"], fit["b"], fit["items"]
        tables["a_correctness"][f"fold_{f}"] = pd.Series(A[:, 0], index=items)
        tables["a_scaffolding"][f"fold_{f}"] = pd.Series(A[:, 1], index=items)
        tables["b"][f"fold_{f}"] = pd.Series(b, index=items)
    return {name: pd.DataFrame(cols) for name, cols in tables.items()}


def cross_fold_stability(fold_fits: list[dict]) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Pairwise cross-fold Pearson correlations of each item parameter.

    Only items fit in BOTH folds of a pair contribute to that pair's correlation.
    Returns (summary, wide_tables). ``summary`` reports, per parameter, the pairwise
    correlation matrix, the median off-diagonal correlation, and the median number
    of shared items per pair.
    """
    tables = item_param_tables(fold_fits)
    k = len(fold_fits)
    summary: dict[str, dict] = {}
    for name, df in tables.items():
        pair_corrs: list[float] = []
        pair_ns: list[int] = []
        mat = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(k):
                if i == j:
                    mat[i, j] = 1.0
                    continue
                a = df[f"fold_{i}"]
                b = df[f"fold_{j}"]
                common = a.notna() & b.notna()
                n = int(common.sum())
                if n >= 3:
                    va = a[common].to_numpy()
                    vb = b[common].to_numpy()
                    if va.std() > 0 and vb.std() > 0:
                        c = float(np.corrcoef(va, vb)[0, 1])
                    else:
                        c = float("nan")
                else:
                    c = float("nan")
                mat[i, j] = c
                if i < j:
                    pair_corrs.append(c)
                    pair_ns.append(n)
        valid = [c for c in pair_corrs if c == c]
        summary[name] = {
            "pairwise_matrix": [[None if v != v else round(float(v), 4) for v in row]
                                for row in mat],
            "median_pairwise_corr": (float(np.median(valid)) if valid else None),
            "mean_pairwise_corr": (float(np.mean(valid)) if valid else None),
            "n_fold_pairs": len(pair_corrs),
            "median_shared_items_per_pair": (int(np.median(pair_ns)) if pair_ns else 0),
        }
    return summary, tables


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = cp.load_matrix(args.matrix)
    q_by = cm.load_q_matrix(args.rubrics)
    models = list(mat.index)
    n_models = len(models)
    print(f"loaded matrix: {n_models} models x {mat.shape[1]} criteria")
    print(f"Q-matrix source: {args.rubrics} ({len(q_by)} criteria with q_mapping)")

    # Shared 2-d quadrature grid (same GH nodes/weights as the fitter).
    grid = cm.build_grid(2, args.grid)
    base_logw = cm.base_log_weights(2, args.grid)

    folds = make_folds(models, args.k, args.seed)
    fold_of = {m: f for f, fold in enumerate(folds) for m in fold}
    print(f"\nk={args.k} folds (seed={args.seed}): sizes {[len(f) for f in folds]}")

    # ---- per-fold fit + held-out scoring ----
    fold_fits: list[dict] = []
    per_fold_metrics: list[dict] = []
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []

    for f in range(args.k):
        test_models = folds[f]
        train_models = [m for m in models if m not in set(test_models)]
        print(f"\n=== fold {f}: TRAIN={len(train_models)} TEST={len(test_models)} ===")
        print("  fitting 2-skill M2PL on TRAIN ...", flush=True)
        fit = fit_2skill(mat.loc[train_models], q_by, args)
        fold_fits.append(fit)
        print(f"  fitted {len(fit['items'])} items "
              f"(loglik={fit['loglik']:.1f}, iters={fit['n_iter']}, "
              f"converged={fit['converged']})")

        log_prior = cm.prior_log_weights(grid, base_logw, fit["R"])
        y, p, theta, _ = eap_predict(
            mat.loc[test_models], fit["items"], fit["A"], fit["b"], fit["R"],
            grid, log_prior,
        )
        m = metrics(y, p)
        m["fold"] = f
        m["n_train"] = len(train_models)
        m["n_test"] = len(test_models)
        m["n_items_fit"] = len(fit["items"])
        per_fold_metrics.append(m)
        all_y.append(y)
        all_p.append(p)
        print(f"  OOS: cells={m['n_cells']} logloss={m['log_loss']:.4f} "
              f"acc={m['accuracy']:.4f} auc={m['auc']:.4f} brier={m['brier']:.4f}")

        # per-fold item params
        pd.DataFrame({
            "criterion_id": fit["items"],
            "a_correctness": np.round(fit["A"][:, 0], 6),
            "a_scaffolding": np.round(fit["A"][:, 1], 6),
            "b": np.round(fit["b"], 6),
        }).to_csv(out_dir / f"fold_{f}_item_params.csv", index=False)

    # pooled OOS (all held-out cells across folds)
    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_p)
    pooled = metrics(y_all, p_all)
    print(f"\n=== POOLED OOS: cells={pooled['n_cells']} "
          f"logloss={pooled['log_loss']:.4f} acc={pooled['accuracy']:.4f} "
          f"auc={pooled['auc']:.4f} brier={pooled['brier']:.4f} ===")

    # ---- in-sample baseline (fit on all 82, score everyone from own responses) ----
    print("\n=== in-sample baseline: fit on all 82, score all 82 ===", flush=True)
    full_fit = fit_2skill(mat, q_by, args)
    log_prior_full = cm.prior_log_weights(grid, base_logw, full_fit["R"])
    yi, pi, theta_full, _ = eap_predict(
        mat, full_fit["items"], full_fit["A"], full_fit["b"], full_fit["R"],
        grid, log_prior_full,
    )
    insample = metrics(yi, pi)
    print(f"  in-sample: cells={insample['n_cells']} logloss={insample['log_loss']:.4f} "
          f"acc={insample['accuracy']:.4f} auc={insample['auc']:.4f} "
          f"brier={insample['brier']:.4f}")

    pd.DataFrame({
        "criterion_id": full_fit["items"],
        "a_correctness": np.round(full_fit["A"][:, 0], 6),
        "a_scaffolding": np.round(full_fit["A"][:, 1], 6),
        "b": np.round(full_fit["b"], 6),
    }).to_csv(out_dir / "insample_baseline_item_params.csv", index=False)

    # ---- item-param cross-fold stability ----
    print("\n=== item-param cross-fold stability ===")
    stability, wide_tables = cross_fold_stability(fold_fits)
    for name, s in stability.items():
        print(f"  {name}: median pairwise corr = {s['median_pairwise_corr']} "
              f"(median shared items/pair = {s['median_shared_items_per_pair']})")

    # also correlate each fold vs the all-82 reference params
    full_series = {
        "a_correctness": pd.Series(full_fit["A"][:, 0], index=full_fit["items"]),
        "a_scaffolding": pd.Series(full_fit["A"][:, 1], index=full_fit["items"]),
        "b": pd.Series(full_fit["b"], index=full_fit["items"]),
    }
    fold_vs_full: dict[str, list] = {}
    for name, df in wide_tables.items():
        ref = full_series[name]
        corrs = []
        for f in range(args.k):
            col = df[f"fold_{f}"]
            common = col.notna() & ref.notna()
            if int(common.sum()) >= 3 and col[common].std() > 0 and ref[common].std() > 0:
                corrs.append(float(np.corrcoef(col[common], ref[common])[0, 1]))
            else:
                corrs.append(float("nan"))
        fold_vs_full[name] = corrs
        stability[name]["median_fold_vs_full_corr"] = (
            float(np.nanmedian(corrs)) if any(c == c for c in corrs) else None
        )

    # ---- write outputs ----
    _write_outputs(out_dir, args, folds, fold_of, per_fold_metrics, pooled,
                   insample, stability, wide_tables, fold_vs_full, full_fit,
                   mat, models)

    print(f"\nall outputs written under {out_dir}")
    return 0


def _write_outputs(out_dir, args, folds, fold_of, per_fold_metrics, pooled,
                   insample, stability, wide_tables, fold_vs_full, full_fit,
                   mat, models):
    # fold assignments
    with (out_dir / "fold_assignments.json").open("w", encoding="utf-8") as fh:
        json.dump({"k": args.k, "seed": args.seed,
                   "folds": {str(f): folds[f] for f in range(args.k)},
                   "model_to_fold": fold_of}, fh, indent=2)
    pd.DataFrame({"model": models, "fold": [fold_of[m] for m in models]}) \
        .sort_values(["fold", "model"]).to_csv(out_dir / "fold_assignments.csv", index=False)

    # per-fold metrics csv
    pf_cols = ["fold", "n_train", "n_test", "n_items_fit", "n_cells", "pos_rate",
               "log_loss", "accuracy", "auc", "brier"]
    pd.DataFrame(per_fold_metrics)[pf_cols].to_csv(
        out_dir / "metrics_per_fold.csv", index=False)

    # aggregate metrics json (pooled OOS, per-fold spread, in-sample gap)
    spread = {m: _spread([pf[m] for pf in per_fold_metrics])
              for m in ["log_loss", "accuracy", "auc", "brier"]}
    gap = {m: (pooled[m] - insample[m]) for m in ["log_loss", "accuracy", "auc", "brier"]}
    aggregate = {
        "generated_at": _utcnow(),
        "k": args.k, "grid": args.grid, "seed": args.seed,
        "estimate_latent_corr": args.estimate_latent_corr,
        "pooled_oos": pooled,
        "per_fold_spread": spread,
        "in_sample_baseline": insample,
        "oos_minus_insample_gap": gap,
        "note": ("gap = pooled OOS metric - in-sample metric. For log_loss & brier "
                 "a POSITIVE gap = OOS worse (expected optimism); for accuracy & auc "
                 "a NEGATIVE gap = OOS worse."),
    }
    with (out_dir / "metrics_aggregate.json").open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2)

    # in-sample baseline json
    with (out_dir / "insample_baseline.json").open("w", encoding="utf-8") as fh:
        json.dump({"generated_at": _utcnow(), "metrics": insample,
                   "n_items_fit": len(full_fit["items"]),
                   "loglik": full_fit["loglik"],
                   "latent_correlation": np.round(np.asarray(full_fit["R"]), 6).tolist()},
                  fh, indent=2)

    # stability tables
    with (out_dir / "item_param_stability.json").open("w", encoding="utf-8") as fh:
        json.dump({"generated_at": _utcnow(),
                   "stability": stability,
                   "fold_vs_full_corr": fold_vs_full}, fh, indent=2)
    # long-format stability summary csv
    rows = []
    for name, s in stability.items():
        rows.append({
            "parameter": name,
            "median_pairwise_corr": s["median_pairwise_corr"],
            "mean_pairwise_corr": s["mean_pairwise_corr"],
            "median_fold_vs_full_corr": s.get("median_fold_vs_full_corr"),
            "n_fold_pairs": s["n_fold_pairs"],
            "median_shared_items_per_pair": s["median_shared_items_per_pair"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "item_param_stability.csv", index=False)

    # full summary
    with (out_dir / "kfold_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": _utcnow(),
            "provenance": {"script": "scripts/kfold_cv_mirt.py", "argv": sys.argv[1:],
                           "matrix": str(args.matrix), "rubrics": str(args.rubrics)},
            "config": {"k": args.k, "grid": args.grid, "seed": args.seed,
                       "ridge": args.ridge, "max_iter": args.max_iter, "tol": args.tol,
                       "estimate_latent_corr": args.estimate_latent_corr,
                       "collapse_skills": COLLAPSE_SKILLS, "dim_labels": DIM_LABELS},
            "n_models": len(models),
            "fold_sizes": [len(f) for f in folds],
            "per_fold_metrics": per_fold_metrics,
            "pooled_oos": pooled,
            "per_fold_spread": spread,
            "in_sample_baseline": insample,
            "oos_minus_insample_gap": gap,
            "item_param_stability": stability,
        }, fh, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX,
                   help=f"response matrix CSV (default: {DEFAULT_MATRIX}).")
    p.add_argument("--rubrics", type=Path, default=DEFAULT_RUBRICS,
                   help=f"rubric bank JSONL for the Q-matrix (default: {DEFAULT_RUBRICS}).")
    p.add_argument("--k", type=int, default=5, help="number of person folds (default 5).")
    p.add_argument("--grid", type=int, default=7,
                   help="Gauss-Hermite nodes per latent dim (default 7 -> 49 nodes for 2-d).")
    p.add_argument("--seed", type=int, default=20260729, help="shuffle seed for folds.")
    p.add_argument("--estimate-latent-corr", action="store_true",
                   help="re-estimate the 2x2 latent correlation each EM iter (default off).")
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="L2 ridge on loadings in the M-step (default 1e-3; matches Run 1).")
    p.add_argument("--max-iter", type=int, default=200, help="max EM iterations.")
    p.add_argument("--tol", type=float, default=1e-4, help="EM convergence tol on loglik.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"output directory (default: {DEFAULT_OUT_DIR}).")
    return p


def main() -> int:
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
