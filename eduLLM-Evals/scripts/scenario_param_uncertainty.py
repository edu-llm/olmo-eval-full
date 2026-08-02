"""SCENARIO-LEVEL parameter-uncertainty inflation of leaderboard error bars.

Scenario-level replacement for the item-level ``leaderboard_param_uncertainty.py``. The
administered criteria set for each model now comes from the REAL scenario engine (not an
item-level CAT). Everything else matches: per-item parameter covariance from the observed
information at the frozen fit, then a parametric bootstrap of those parameters, re-estimating
each model's ability on its administered set to get the parameter-induced SE, combined in
quadrature with the posterior SE.

The bootstrap is serial over models (vectorised over grid nodes per draw); use a coarse
``--eap-grid`` for the 3-skill bank so the dense product grid stays tractable.

Usage
-----
    python scripts/scenario_param_uncertainty.py \
        --bank data/TutorBench/rubrics_qmatrix_calibrated_2skill_fitted.jsonl \
        --matrix staging/response_matrix_full_nonopt.csv \
        --out-dir regenerated_figures/scenario_level/param_uncertainty/2_skills \
        --eap-grid 61 --n-boot 150 --workers 12
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
from scipy.special import log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.scenario_cat_lib as scat  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")


def item_param_cov(Y, mask, A, Q, b, n_dims, fit_nodes, ridge):
    """Per-item parameter covariance from observed information at the frozen fit.

    One E-step at (A, b) over the fit's GH grid -> posterior expected counts -> invert each
    item's M-step Hessian. Returns per item (free_dims, mean_beta=[a_free..., -b], cov)."""
    grid = cm.build_grid(n_dims, fit_nodes)
    base_logw = cm.base_log_weights(n_dims, fit_nodes)
    log_prior = cm.prior_log_weights(grid, base_logw, np.eye(n_dims))
    YM = np.where(mask, Y, 0.0)
    NM = np.where(mask, 1.0 - Y, 0.0)
    Mf = mask.astype(float)
    eta = A @ grid.T - b[:, None]
    LL = YM @ log_expit(eta) + NM @ log_expit(-eta)
    joint = LL + log_prior[None, :]
    post = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    r_jg = YM.T @ post
    N_jg = Mf.T @ post
    ones_col = np.ones((grid.shape[0], 1))
    out = []
    for j in range(A.shape[0]):
        fd = np.where(Q[j] == 1)[0]
        if fd.size == 0:
            out.append((fd, np.array([-b[j]]), np.zeros((1, 1))))
            continue
        X = np.hstack([grid[:, fd], ones_col])
        beta = np.concatenate([A[j, fd], [-b[j]]])
        _, _, H = cm._item_neg_loglik(beta, X, r_jg[j], N_jg[j], ridge)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)
        cov = (cov + cov.T) / 2.0
        w, V = np.linalg.eigh(cov)
        cov = (V * np.clip(w, 0.0, None)) @ V.T
        out.append((fd, beta, cov))
    return out


def eap_mean(y, idx, A, b, grid, log_prior):
    Ai, bi, yi = A[idx], b[idx], y[idx]
    eta = Ai @ grid.T - bi[:, None]
    ll = yi @ log_expit(eta) + (1.0 - yi) @ log_expit(-eta)
    post = np.exp(ll + log_prior - logsumexp(ll + log_prior))
    return post @ grid


def eap_mean_var(y, idx, A, b, grid, log_prior):
    Ai, bi, yi = A[idx], b[idx], y[idx]
    eta = Ai @ grid.T - bi[:, None]
    ll = yi @ log_expit(eta) + (1.0 - yi) @ log_expit(-eta)
    post = np.exp(ll + log_prior - logsumexp(ll + log_prior))
    mean = post @ grid
    var = post @ (grid**2) - mean**2
    return mean, np.clip(var, 0.0, None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--scenarios", type=Path,
                   default=ROOT / "data" / "TutorBench" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, default=ROOT / "staging" / "engine_runs_pu")
    p.add_argument("--fit-nodes", type=int, default=7)
    p.add_argument("--eap-grid", type=int, default=61)
    p.add_argument("--range", type=float, default=6.0)
    p.add_argument("--max-se", type=float, default=0.30)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--max-scenarios", type=int, default=50)
    p.add_argument("--n-boot", type=int, default=150)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--workers", type=int, default=scat.default_workers())
    args = p.parse_args()

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    ids, A, b = scat.assemble_arrays(records, dims)
    Q = np.array([[int(r["q_modeled"][d]) for d in dims] for r in records])
    matrix = pd.read_csv(args.matrix, index_col=0)
    sub = matrix.reindex(columns=ids)
    Ymat = np.nan_to_num(sub.to_numpy(float), nan=0.0)
    Mmat = ~np.isnan(sub.to_numpy(float))
    col = {c: i for i, c in enumerate(ids)}
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    n_dims = len(dims)
    print(f"bank={args.bank.name} dims={dims} models={len(models)} "
          f"eap_grid={args.eap_grid} n_boot={args.n_boot} workers={args.workers}")

    print("computing item-parameter covariance (observed information) ...", flush=True)
    item_cov = item_param_cov(Ymat, Mmat, A, Q, b, n_dims, args.fit_nodes, args.ridge)
    chol = []
    for fd, beta, cov in item_cov:
        try:
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            L = np.zeros_like(cov)
        chol.append(L)

    se_a = {d: [] for d in dims}
    for (fd, beta, cov), qrow in zip(item_cov, Q):
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        for kk, di in enumerate(fd):
            se_a[dims[di]].append(float(sd[kk]))
    print("  median item-param SE:",
          {d: round(float(np.median(v)), 3) for d, v in se_a.items() if v})

    grid, log_prior = scat.build_grid(n_dims, args.eap_grid, args.range)

    # administered criteria per model from the REAL scenario engine (trace/production)
    print("running the scenario engine to get administered sets ...", flush=True)
    spec = scat.RunSpec(seed=42, max_se=args.max_se,
                        min_evals_per_skill=args.min_evals_per_skill,
                        max_scenarios=args.max_scenarios, selection="trace",
                        runs_dir=str(args.runs_dir))
    results = scat.run_models(models, args.bank, args.matrix, args.scenarios,
                              "clamp", dims, spec, workers=args.workers)
    admin = {r["model"]: [c for c in r["order"] if c in col] for r in results}

    print("parametric bootstrap over administered params ...", flush=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    A_draw = A.copy()
    b_draw = b.copy()
    for n, m in enumerate(models, 1):
        r = row_of[m]
        idx = np.array([col[c] for c in admin[m]], dtype=int)
        if idx.size == 0:
            continue
        y = Ymat[r]
        mean, var = eap_mean_var(y, idx, A, b, grid, log_prior)
        se_post = np.sqrt(var)
        boot = np.empty((args.n_boot, n_dims))
        for t in range(args.n_boot):
            for j in idx:
                fd, beta, cov = item_cov[j]
                if fd.size == 0:
                    continue
                draw = beta + chol[j] @ rng.standard_normal(beta.shape[0])
                A_draw[j] = 0.0
                A_draw[j, fd] = draw[:-1]
                b_draw[j] = -draw[-1]
            boot[t] = eap_mean(y, idx, A_draw, b_draw, grid, log_prior)
        A_draw[idx] = A[idx]
        b_draw[idx] = b[idx]
        se_param = boot.std(axis=0, ddof=1)
        se_total = np.sqrt(se_post**2 + se_param**2)
        rec = {"model": m, "n_admin": int(idx.size)}
        for k, d in enumerate(dims):
            rec[f"theta_{d}"] = float(mean[k])
            rec[f"se_posterior_{d}"] = float(se_post[k])
            rec[f"se_param_{d}"] = float(se_param[k])
            rec[f"se_total_{d}"] = float(se_total[k])
            rec[f"bar_inflation_{d}"] = float(se_total[k] / se_post[k]) if se_post[k] > 0 else float("nan")
        rows.append(rec)
        if n % 20 == 0:
            print(f"  {n}/{len(models)} models", flush=True)

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "leaderboard_se_components.csv", index=False)

    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "purpose": "scenario-level parameter-uncertainty inflation of ability SEs",
               "bank": str(args.bank), "matrix": str(args.matrix), "dims": dims,
               "config": {"fit_nodes": args.fit_nodes, "eap_grid": args.eap_grid,
                          "n_boot": args.n_boot, "max_se": args.max_se},
               "n_models": int(len(df)),
               "median_item_param_se": {d: float(np.median(v)) for d, v in se_a.items() if v},
               "se_components": {}}
    for d in dims:
        summary["se_components"][d] = {
            "median_se_posterior": float(df[f"se_posterior_{d}"].median()),
            "median_se_param": float(df[f"se_param_{d}"].median()),
            "median_se_total": float(df[f"se_total_{d}"].median()),
            "median_bar_inflation": float(df[f"bar_inflation_{d}"].median()),
            "max_bar_inflation": float(df[f"bar_inflation_{d}"].max())}
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _figure(df, dims, args.out_dir / "figures")

    print("\n" + "=" * 84)
    print("SCENARIO-LEVEL LEADERBOARD SE DECOMPOSITION")
    print("=" * 84)
    for d in dims:
        s = summary["se_components"][d]
        print(f"{d:14s}: SE_post={s['median_se_posterior']:.3f}  SE_param={s['median_se_param']:.3f}  "
              f"SE_total={s['median_se_total']:.3f}  bar x{s['median_bar_inflation']:.2f} "
              f"(max x{s['max_bar_inflation']:.2f})")
    print(f"\nwrote -> {args.out_dir}")
    return 0


def _figure(df, dims, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    d = dims[0]
    order = df.sort_values(f"theta_{d}").reset_index(drop=True)
    ypos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, max(9, 0.22 * len(order))))
    ax.errorbar(order[f"theta_{d}"], ypos, xerr=1.96 * order[f"se_total_{d}"], fmt="none",
                ecolor="#c1666b", elinewidth=2.4, alpha=0.6, label="+/-1.96 SE_total (params uncertain)")
    ax.errorbar(order[f"theta_{d}"], ypos, xerr=1.96 * order[f"se_posterior_{d}"], fmt="none",
                ecolor="#4d648d", elinewidth=1.2, label="+/-1.96 SE_posterior (params known)")
    ax.scatter(order[f"theta_{d}"], ypos, s=10, color="k", zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels(order["model"], fontsize=5)
    ax.set_xlabel(f"ability ({d})")
    ax.set_title(f"Scenario-level leaderboard bars: posterior vs parameter-uncertainty ({d})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / f"leaderboard_bars_{d}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
