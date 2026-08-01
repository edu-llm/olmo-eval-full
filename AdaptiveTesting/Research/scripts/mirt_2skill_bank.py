#!/usr/bin/env python3
"""Fit the DEFINITIVE 2-skill (correctness = content+diagnosis, scaffolding) M2PL on a
TutorBench response matrix and emit a bank CSV in the (criterion_id, a_correctness,
a_scaffolding, b) format that scripts/cat_eval_tutorbench.py consumes.

Reuses scripts/calibrate_mirt.py verbatim (Q-matrix load, block prep, collapse, EM) so
the fit is identical to the collapsed model in the 3-way comparison, but here the
collapsed 2-dim model is the PRIMARY output and its loadings are written to disk.

Also reports approximate asymptotic loading SEs (observed information at the fit) so we
can quantify the "parameter SE shrinks as N grows" limitation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
EDU = REPO / "eduLLM-Evals"
sys.path.insert(0, str(EDU))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", EDU / "scripts" / "calibrate_mirt.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, type=Path)
    ap.add_argument("--rubrics", type=Path,
                    default=EDU / "data" / "TutorBench" / "curated" / "rubrics_qmatrix_curated.jsonl")
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--out-bank", required=True, type=Path)
    ap.add_argument("--out-manifest", required=True, type=Path)
    args = ap.parse_args()

    mat = cm.cp.load_matrix(args.matrix)
    q_by = cm.load_q_matrix(args.rubrics)
    Y, M, Q, items, block_df, diag = cm.prepare_block(mat, q_by)
    print(f"prepared block: {len(items)} items x {diag['n_persons_fit']} persons")

    Q2, labels, info = cm.collapse_q_matrix(Q, ["content", "diagnosis"])
    print(f"collapsed dims: {labels}")
    fit = cm.fit_m2pl_em(Y, M, Q2, args.grid, estimate_corr=True,
                         ridge=args.ridge, max_iter=200, tol=1e-4)
    A, b, R = fit["A"], fit["b"], fit["R"]
    # col0 = correctness (content+diagnosis merged), col1 = scaffolding
    a_corr, a_scaff = A[:, 0], A[:, 1]

    # Approximate asymptotic loading SEs from the observed information at the fit:
    # per item, refit-Hessian on expected counts gives Cov ~= H^{-1}; take sqrt(diag)
    # for the free loadings only. Cheap E-step recompute for the counts.
    grid = cm.build_grid(Q2.shape[1], args.grid)
    base_logw = cm.base_log_weights(Q2.shape[1], args.grid)
    log_prior = cm.prior_log_weights(grid, base_logw, R)
    from scipy.special import log_expit, logsumexp
    YM = np.where(M, Y, 0.0)
    NM = np.where(M, 1.0 - Y, 0.0)
    Mf = M.astype(float)
    eta = A @ grid.T - b[:, None]
    LL = YM @ log_expit(eta) + NM @ log_expit(-eta)
    joint = LL + log_prior[None, :]
    person_ll = logsumexp(joint, axis=1)
    post = np.exp(joint - person_ll[:, None])
    r_jg = YM.T @ post
    N_jg = Mf.T @ post
    free_dims = [np.where(Q2[j] == 1)[0] for j in range(len(items))]
    ones_col = np.ones((grid.shape[0], 1))
    a_se = np.full(len(items), np.nan)
    for j in range(len(items)):
        X = np.hstack([grid[:, free_dims[j]], ones_col])
        beta = np.append(A[j, free_dims[j]], -b[j])
        _, _, H = cm._item_neg_loglik(beta, X, r_jg[j], N_jg[j], args.ridge)
        try:
            cov = np.linalg.inv(H)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
            if len(free_dims[j]) >= 1:
                a_se[j] = float(se[0])
        except np.linalg.LinAlgError:
            pass

    bank = pd.DataFrame({
        "criterion_id": items,
        "a_correctness": np.round(a_corr, 6),
        "a_scaffolding": np.round(a_scaff, 6),
        "b": np.round(b, 6),
        "n_persons": M.sum(axis=0).astype(int),
        "a_se_approx": np.round(a_se, 6),
    })
    args.out_bank.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(args.out_bank, index=False)

    nz_corr = a_corr[a_corr > 0]
    nz_scaff = a_scaff[a_scaff > 0]
    manifest = {
        "matrix": str(args.matrix),
        "n_persons_fit": diag["n_persons_fit"],
        "n_items_fit": len(items),
        "n_observed_cells": int(M.sum()),
        "grid_nodes_per_dim": args.grid,
        "ridge": args.ridge,
        "loglik": fit["loglik"],
        "converged": fit["converged"],
        "n_iter": fit["n_iter"],
        "latent_correlation_correctness_scaffolding": float(R[0, 1]),
        "a_correctness_median_nonzero": float(np.median(nz_corr)) if nz_corr.size else None,
        "a_scaffolding_median_nonzero": float(np.median(nz_scaff)) if nz_scaff.size else None,
        "a_correctness_n_loading": int((a_corr > 0).sum()),
        "a_scaffolding_n_loading": int((a_scaff > 0).sum()),
        "a_se_median": float(np.nanmedian(a_se)),
        "a_se_correctness_median": float(np.nanmedian(a_se[a_corr > 0])) if nz_corr.size else None,
        "a_se_scaffolding_median": float(np.nanmedian(a_se[a_scaff > 0])) if nz_scaff.size else None,
        "b_median": float(np.median(b)),
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
