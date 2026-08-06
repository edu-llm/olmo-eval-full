"""Phase 2a step 2: scenario-level SE_param bootstrap (calibration-noise floor) for WildBench.

Observed-information PARAMETRIC bootstrap (NOT jackknife), mirroring the Bridge scenario
study + the shared reference ``scripts/scenario_param_uncertainty.py``:

  1. Frozen 1D CAT-pool fit (masked bank).
  2. Per-criterion parameter covariance from the observed information at the frozen fit
     (one E-step -> expected counts -> invert each item's M-step Hessian).
  3. Draw B replicates of the item parameters from that sampling distribution and
     re-estimate each model's ability; SE_param = SD across replicates.

The reported FLOOR administers the FULL administrable CAT pool per model (matrix ~100%
filled) -> maximal information -> minimal calibration-noise SE. WildBench scenarios are
lighter testlets (~11 criteria) than Bridge's (~18), so the full-bank floor is expected
to be small. SE_total = sqrt(SE_posterior^2 + SE_param^2).

Exposes ``compute_item_cov`` + ``bootstrap_theta`` for reuse by the operating-point grid
(step 3), which computes op-point-specific SE_param on the engine's administered sets.
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import wildbench_scenario_lib as L  # noqa: E402

ROOT = L.ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.scenario_cat_lib as scat  # noqa: E402

# reuse the shared reference observed-info covariance verbatim
_spec = importlib.util.spec_from_file_location(
    "ref_param_unc", ROOT / "scripts" / "scenario_param_uncertainty.py")
_ref = importlib.util.module_from_spec(_spec)
sys.modules["ref_param_unc"] = _ref
_spec.loader.exec_module(_ref)


def compute_item_cov(bank_path: Path, matrix_path: Path, fit_nodes: int, ridge: float):
    """Per-criterion (beta, chol) from observed information at the frozen fit."""
    records, dims, _ = scat.load_fitted_bank(bank_path, "clamp")
    ids, A, b = scat.assemble_arrays(records, dims)
    Q = np.array([[int(r["q_modeled"][d]) for d in dims] for r in records])
    matrix = pd.read_csv(matrix_path, index_col=0)
    sub = matrix.reindex(columns=ids)
    Ymat = np.nan_to_num(sub.to_numpy(float), nan=0.0)
    Mmat = ~np.isnan(sub.to_numpy(float))
    col = {c: i for i, c in enumerate(ids)}
    n_dims = len(dims)
    item_cov = _ref.item_param_cov(Ymat, Mmat, A, Q, b, n_dims, fit_nodes, ridge)
    beta, chol = [], []
    for fd, bta, cov in item_cov:
        beta.append(bta)
        try:
            Lc = np.linalg.cholesky(cov + 1e-10 * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            Lc = np.zeros_like(cov)
        chol.append(Lc)
    return {"ids": ids, "dims": dims, "A": A, "b": b, "Q": Q, "Ymat": Ymat,
            "Mmat": Mmat, "col": col, "beta": beta, "chol": chol,
            "matrix": matrix, "n_dims": n_dims}


def bootstrap_theta(y, idx, beta, chol, grid, log_prior, n_boot, rng, batch=40):
    """Vectorised 1D parametric bootstrap of theta over administered items ``idx``.

    Returns (theta_mean_point, se_posterior, se_param). 1D only (single loading + intercept).
    """
    idx = np.asarray(idx, dtype=int)
    k = idx.size
    if k == 0:
        return np.nan, np.nan, np.nan
    beta_sub = np.stack([beta[j] for j in idx])            # (k, 2): [a, -b]
    L_sub = np.stack([chol[j] for j in idx])               # (k, 2, 2)
    yk = y[idx]

    a0 = beta_sub[:, 0]
    mb0 = beta_sub[:, 1]
    eta0 = a0[:, None] * grid[None, :] + mb0[:, None]
    ll0 = yk @ log_expit(eta0) + (1.0 - yk) @ log_expit(-eta0)
    post0 = np.exp(ll0 + log_prior - logsumexp(ll0 + log_prior))
    mean0 = float(post0 @ grid)
    var0 = float(post0 @ (grid ** 2) - mean0 ** 2)
    se_post = float(np.sqrt(max(var0, 0.0)))

    thetas = np.empty(n_boot)
    done = 0
    while done < n_boot:
        bsz = min(batch, n_boot - done)
        z = rng.standard_normal((bsz, k, 2))
        draw = beta_sub[None] + np.einsum("kij,bkj->bki", L_sub, z)
        a_d = draw[:, :, 0]
        mb_d = draw[:, :, 1]
        eta = a_d[:, :, None] * grid[None, None, :] + mb_d[:, :, None]
        logP = log_expit(eta)
        log1mP = log_expit(-eta)
        ll = np.einsum("k,bkn->bn", yk, logP) + np.einsum("k,bkn->bn", 1.0 - yk, log1mP)
        post = np.exp(ll + log_prior[None, :]
                      - logsumexp(ll + log_prior[None, :], axis=1)[:, None])
        thetas[done:done + bsz] = post @ grid
        done += bsz
    se_param = float(thetas.std(ddof=1))
    return mean0, se_post, se_param


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "wildbench_calibration"
    p.add_argument("--bank", type=Path,
                   default=base / "wildbench_scenario_fitted_1d_catpool.jsonl")
    p.add_argument("--matrix", type=Path, default=L.DEFAULT_MATRIX)
    p.add_argument("--out-dir", type=Path,
                   default=base / "experiments" / "07_parameter_uncertainty")
    p.add_argument("--fit-nodes", type=int, default=7)
    p.add_argument("--eap-grid", type=int, default=161,
                   help="uniform-EAP nodes for the full-bank bootstrap. SE_param = SD of "
                        "the EAP MEAN across draws, which interpolates smoothly, so a "
                        "moderate grid resolves it; kept moderate for tractability on the "
                        "full 8,345-criterion bank.")
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--boot-batch", type=int, default=10,
                   help="replicate batch size; keeps the (batch x n_items x nodes) "
                        "bootstrap array in memory on the full administrable bank.")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=20260801)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("STEP 2: scenario-level SE_param bootstrap (calibration-noise floor)")
    print("=" * 84)
    cov = compute_item_cov(args.bank, args.matrix, args.fit_nodes, args.ridge)
    dims = cov["dims"]
    assert len(dims) == 1, "Phase 2a is locked to 1D"
    d = dims[0]
    ids = cov["ids"]
    Ymat = cov["Ymat"]
    models = list(cov["matrix"].index)
    print(f"bank={args.bank.name} dims={dims} models={len(models)} "
          f"admin_criteria={len(ids)} n_boot={args.n_boot}")

    item_se = np.array([float(np.sqrt(max(Lc[0, 0] ** 2, 0.0))) for Lc in cov["chol"]])
    print(f"median per-criterion param SE(a): {np.median(item_se):.4f}")

    grid = np.linspace(-args.range, args.range, args.eap_grid)
    lp = -0.5 * grid ** 2
    lp = lp - logsumexp(lp)

    rng = np.random.default_rng(args.seed)
    all_idx = np.arange(len(ids))
    rows = []
    for n, m in enumerate(models, 1):
        y = Ymat[n - 1]
        mean0, se_post, se_param = bootstrap_theta(
            y, all_idx, cov["beta"], cov["chol"], grid, lp, args.n_boot, rng,
            batch=args.boot_batch)
        se_total = float(np.sqrt(se_post ** 2 + se_param ** 2))
        rows.append({"model": m, "n_admin": int(all_idx.size),
                     f"theta_{d}": mean0, f"se_posterior_{d}": se_post,
                     f"se_param_{d}": se_param, f"se_total_{d}": se_total,
                     f"bar_inflation_{d}": (se_total / se_post) if se_post > 0 else np.nan})
        if n % 10 == 0:
            print(f"  {n}/{len(models)} models", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "leaderboard_se_components.csv", index=False)

    floor = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "scenario-level calibration-noise floor (SE_param) via observed-info "
                   "parametric bootstrap over the FULL administrable CAT pool (maximal info).",
        "method": "observed-information parametric bootstrap (NOT jackknife)",
        "bank": str(args.bank), "matrix": str(args.matrix), "dims": dims,
        "config": {"fit_nodes": args.fit_nodes, "eap_grid": args.eap_grid,
                   "range": args.range, "n_boot": args.n_boot, "ridge": args.ridge},
        "n_models": len(df), "n_admin_criteria": len(ids),
        "median_item_param_se_a": float(np.median(item_se)),
        "se_param_floor": {
            "mean": float(df[f"se_param_{d}"].mean()),
            "median": float(df[f"se_param_{d}"].median()),
            "min": float(df[f"se_param_{d}"].min()),
            "max": float(df[f"se_param_{d}"].max()),
        },
        "se_posterior_full_bank": {
            "mean": float(df[f"se_posterior_{d}"].mean()),
            "median": float(df[f"se_posterior_{d}"].median()),
        },
        "se_total_full_bank": {
            "mean": float(df[f"se_total_{d}"].mean()),
            "median": float(df[f"se_total_{d}"].median()),
        },
        "comparison_note": (
            "Bridge scenario-level SE_param floor: mean 0.033 / median 0.027 (~18-criterion "
            "testlets, N=51). TutorBench scenario-level ~0.21-0.25. WildBench uses ~11-criterion "
            "testlets over 983 administrable scenarios at N=52 -> compare the floor here to those."),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(floor, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.hist(df[f"se_param_{d}"], bins=20, color="#c1666b", alpha=0.85)
    ax1.axvline(df[f"se_param_{d}"].median(), ls="--", color="k",
                label=f"median={df[f'se_param_{d}'].median():.3f}")
    ax1.set_xlabel("SE_param (calibration noise)"); ax1.set_ylabel("models")
    ax1.set_title("WildBench SE_param floor (full administrable bank)"); ax1.legend()
    order = df.sort_values(f"theta_{d}").reset_index(drop=True)
    ypos = np.arange(len(order))
    ax2.errorbar(order[f"theta_{d}"], ypos, xerr=1.96 * order[f"se_total_{d}"], fmt="none",
                 ecolor="#c1666b", elinewidth=2.2, alpha=0.6, label="+/-1.96 SE_total")
    ax2.errorbar(order[f"theta_{d}"], ypos, xerr=1.96 * order[f"se_posterior_{d}"], fmt="none",
                 ecolor="#4d648d", elinewidth=1.0, label="+/-1.96 SE_posterior")
    ax2.scatter(order[f"theta_{d}"], ypos, s=8, color="k", zorder=3)
    ax2.set_yticks([]); ax2.set_xlabel(f"ability ({d})")
    ax2.set_title("Full-bank leaderboard bars"); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "se_param_floor.png", dpi=140)
    plt.close(fig)

    print("\n" + "=" * 84)
    print(f"SE_param FLOOR (full bank): mean={floor['se_param_floor']['mean']:.4f} "
          f"median={floor['se_param_floor']['median']:.4f} "
          f"[{floor['se_param_floor']['min']:.4f}, {floor['se_param_floor']['max']:.4f}]")
    print(f"SE_posterior (full bank)  : median={floor['se_posterior_full_bank']['median']:.4f}")
    print(f"SE_total (full bank)      : median={floor['se_total_full_bank']['median']:.4f}")
    print(f"wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
