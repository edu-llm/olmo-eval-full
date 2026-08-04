"""Propagate item-parameter calibration uncertainty into leaderboard ability error bars.

The problem
-----------
The published leaderboard bars are +/-1.96 * SE_theta, where SE_theta = sqrt(U_kk) comes
from the CAT posterior covariance. That treats the calibrated item parameters (a, b) as if
they were known exactly. They are not: they were estimated from only 82 models, so each
carries its own sampling error. Ignoring it makes every model's bar too narrow, and the
CAT-uncertainty audit flagged exactly this.

Two variance components
-----------------------
For a model's ability theta_hat we report

    Var(theta_hat) = Var_posterior(theta | params)          <- current bars (params known)
                   + Var_params(theta_hat)                  <- added here (params uncertain)

* Var_posterior is the usual EAP posterior variance over the administered items.
* Var_params is obtained by a PARAMETRIC BOOTSTRAP: item parameters are resampled from their
  own asymptotic sampling distribution and the ability is re-estimated each draw; the spread
  of theta across draws is the parameter-induced component. The two are combined in
  quadrature for the total SE.

Item-parameter sampling distribution
------------------------------------
The confirmatory M2PL M-step (``calibrate_mirt._item_neg_loglik``) already returns the exact
observed information (Hessian) of each item's weighted-logistic log-likelihood. Evaluated at
the frozen fitted parameters, using the E-step posterior expected counts, its inverse is the
per-item parameter covariance -> the sampling distribution we draw from. This reuses the
calibration math verbatim; nothing is re-derived.

READ-ONLY: recomputes one E-step at the frozen bank in memory; never edits the fitter, bank,
or any frozen artifact. Writes only under --out-dir.

Usage
-----
    python scripts/leaderboard_param_uncertainty.py \
        --bank data/TutorBench/rubrics_qmatrix_calibrated_2skill_fitted.jsonl \
        --matrix staging/response_matrix_full_nonopt.csv \
        --out-dir regenerated_figures/param_uncertainty/2_skills
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load("calibrate_mirt", ROOT / "scripts" / "calibrate_mirt.py")
regen = _load("regen", ROOT / "scripts" / "regen_cat_figures.py")


def load_fitted_bank(path: Path):
    """Parse a ``*_fitted`` bank -> (items, A, Q, b, dims)."""
    recs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    dims = list(recs[0]["discrimination"].keys())
    items = [r["criterion_id"] for r in recs]
    A = np.array([[float(r["discrimination"][d] or 0.0) for d in dims] for r in recs])
    Q = np.array([[int(r["q_modeled"][d]) for d in dims] for r in recs])
    b = np.array([float(r["difficulty"]) for r in recs])
    return items, A, Q, b, dims


def item_param_cov(Y, mask, A, Q, b, dims, fit_nodes, ridge):
    """Per-item parameter covariance from the observed information at the frozen fit.

    Runs ONE E-step at (A, b) over the fit's Gauss-Hermite grid to get the posterior
    expected counts, then inverts each item's M-step Hessian. Returns a list of
    (free_dims, mean_beta, cov) where beta = [a_free..., intercept(=-b)].
    """
    n_dims = A.shape[1]
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
            out.append((fd, np.array([-b[j]]), np.array([[np.inf]])))
            continue
        X = np.hstack([grid[:, fd], ones_col])
        beta = np.concatenate([A[j, fd], [-b[j]]])
        _, _, H = cm._item_neg_loglik(beta, X, r_jg[j], N_jg[j], ridge)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)
        # symmetrise + PSD clip so we can Cholesky-sample later
        cov = (cov + cov.T) / 2.0
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 0.0, None)
        cov = (V * w) @ V.T
        out.append((fd, beta, cov))
    return out


def eap_mean_var(y, idx, A, b, grid, log_prior):
    """EAP posterior mean AND variance over the administered subset."""
    Ai, bi, yi = A[idx], b[idx], y[idx]
    eta = Ai @ grid.T - bi[:, None]
    ll = yi @ log_expit(eta) + (1.0 - yi) @ log_expit(-eta)
    post = np.exp(ll + log_prior - logsumexp(ll + log_prior))
    mean = post @ grid
    var = post @ (grid ** 2) - mean ** 2
    return mean, np.clip(var, 0.0, None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fit-nodes", type=int, default=7,
                   help="GH nodes/dim for the observed-information E-step (match the fit).")
    p.add_argument("--eap-grid", type=int, default=61)
    p.add_argument("--range", type=float, default=6.0)
    p.add_argument("--se-target", type=float, default=0.3)
    p.add_argument("--min-items", type=int, default=1)
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--selection", choices=("trace", "dopt"), default="trace")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=20260801)
    args = p.parse_args()

    items, A, Q, b, dims = load_fitted_bank(args.bank)
    matrix = pd.read_csv(args.matrix, index_col=0)
    sub = matrix.reindex(columns=items)
    Yraw = sub.to_numpy(dtype=float)
    mask = ~np.isnan(Yraw)
    Y = np.nan_to_num(Yraw, nan=0.0)
    models = list(matrix.index)
    n_dims = A.shape[1]
    print(f"bank={args.bank.name} items={len(items)} dims={dims} models={len(models)}")

    print("computing per-item parameter covariance (observed information) ...", flush=True)
    item_cov = item_param_cov(Y, mask, A, Q, b, dims, args.fit_nodes, args.ridge)

    # median item-parameter SEs (diagnostic)
    se_a = {d: [] for d in dims}
    se_b = []
    for (fd, beta, cov), qrow in zip(item_cov, Q):
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        for kk, d_idx in enumerate(fd):
            se_a[dims[d_idx]].append(float(sd[kk]))
        if fd.size:
            se_b.append(float(sd[-1]))
    print("  median item-param SE:",
          {d: round(float(np.median(v)), 3) for d, v in se_a.items() if v},
          "b:", round(float(np.median(se_b)), 3))

    grid, log_prior = regen.build_grid(n_dims, args.eap_grid, "uniform", args.range)

    # Cholesky factors for sampling (precompute)
    chol = []
    for fd, beta, cov in item_cov:
        try:
            L = np.linalg.cholesky(cov + 1e-10 * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            L = np.zeros_like(cov)
        chol.append(L)

    rng = np.random.default_rng(args.seed)
    rows = []
    for r, model in enumerate(models):
        th_on, n_admin, se_on, order = regen_run(A, b, Y[r], mask[r], args)
        idx = np.asarray(order, dtype=int)
        if idx.size == 0:
            continue
        mean, var = eap_mean_var(Y[r], idx, A, b, grid, log_prior)
        se_post = np.sqrt(var)

        # parametric bootstrap over the administered items' parameters
        boot = np.empty((args.n_boot, n_dims))
        A_draw = A.copy()
        b_draw = b.copy()
        for t in range(args.n_boot):
            for j in idx:
                fd, beta, cov = item_cov[j]
                if fd.size == 0:
                    continue
                z = rng.standard_normal(beta.shape[0])
                draw = beta + chol[j] @ z
                A_draw[j] = 0.0
                A_draw[j, fd] = draw[:-1]
                b_draw[j] = -draw[-1]
            m_t, _ = eap_mean_var(Y[r], idx, A_draw, b_draw, grid, log_prior)
            boot[t] = m_t
        # restore perturbed rows
        A_draw[idx] = A[idx]
        b_draw[idx] = b[idx]

        se_param = boot.std(axis=0, ddof=1)
        se_total = np.sqrt(se_post ** 2 + se_param ** 2)

        rec = {"model": model, "n_admin": int(n_admin)}
        for k, d in enumerate(dims):
            rec[f"theta_{d}"] = float(mean[k])
            rec[f"se_posterior_{d}"] = float(se_post[k])
            rec[f"se_param_{d}"] = float(se_param[k])
            rec[f"se_total_{d}"] = float(se_total[k])
            rec[f"bar_inflation_{d}"] = float(se_total[k] / se_post[k]) if se_post[k] > 0 else float("nan")
        rows.append(rec)
        if (r + 1) % 20 == 0:
            print(f"  {r + 1}/{len(models)} models", flush=True)

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "leaderboard_se_components.csv", index=False)

    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "bank": str(args.bank), "matrix": str(args.matrix),
               "config": {"fit_nodes": args.fit_nodes, "eap_grid": args.eap_grid,
                          "se_target": args.se_target, "selection": args.selection,
                          "n_boot": args.n_boot, "ridge": args.ridge},
               "dims": dims, "n_models": int(len(df)),
               "median_item_param_se": {d: float(np.median(v)) for d, v in se_a.items() if v},
               "median_item_b_se": float(np.median(se_b)),
               "se_components": {}}
    for d in dims:
        summary["se_components"][d] = {
            "median_se_posterior": float(df[f"se_posterior_{d}"].median()),
            "median_se_param": float(df[f"se_param_{d}"].median()),
            "median_se_total": float(df[f"se_total_{d}"].median()),
            "median_bar_inflation": float(df[f"bar_inflation_{d}"].median()),
            "max_bar_inflation": float(df[f"bar_inflation_{d}"].max()),
        }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _figure(df, dims, args.out_dir / "figures")

    print("\n" + "=" * 88)
    print("LEADERBOARD SE DECOMPOSITION (params known -> params uncertain)")
    print("=" * 88)
    for d in dims:
        s = summary["se_components"][d]
        print(f"{d:14s}: SE_post={s['median_se_posterior']:.3f}  "
              f"SE_param={s['median_se_param']:.3f}  SE_total={s['median_se_total']:.3f}  "
              f"bar x{s['median_bar_inflation']:.2f} (max x{s['max_bar_inflation']:.2f})")
    print(f"\nwrote -> {args.out_dir}")
    return 0


def regen_run(A, b, y, mask, args):
    """Run the CAT to fix the administered set, using the chosen selection rule."""
    # reuse the selection experiment's runner if present, else regen's trace runner
    sel_path = ROOT / "scripts" / "cat_selection_experiment.py"
    if args.selection == "dopt" and sel_path.exists():
        se = _load("cat_selection_experiment", sel_path)
        th, n, se_v, order = se.run_cat_person(
            y, mask, A, b, args.min_items, args.max_items, args.se_target, "dopt")
        return th, n, se_v, order
    th, n, se_v, _, order = regen.run_cat_person(
        y, mask, A, b, args.min_items, args.max_items, args.se_target)
    return th, n, se_v, order


def _figure(df, dims, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    d = dims[0]  # correctness
    order = df.sort_values(f"theta_{d}").reset_index(drop=True)
    ypos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, max(9, 0.22 * len(order))))
    ax.errorbar(order[f"theta_{d}"], ypos,
                xerr=1.96 * order[f"se_total_{d}"], fmt="none",
                ecolor="#c1666b", elinewidth=2.4, capsize=0, alpha=0.6,
                label="+/-1.96 SE_total (params uncertain)")
    ax.errorbar(order[f"theta_{d}"], ypos,
                xerr=1.96 * order[f"se_posterior_{d}"], fmt="none",
                ecolor="#4d648d", elinewidth=1.2, capsize=0,
                label="+/-1.96 SE_posterior (params known)")
    ax.scatter(order[f"theta_{d}"], ypos, s=10, color="k", zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order["model"], fontsize=5)
    ax.set_xlabel(f"{d} ability (logits)")
    ax.set_title(f"Parameter uncertainty widens {d} ability intervals")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"leaderboard_bars_{d}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
