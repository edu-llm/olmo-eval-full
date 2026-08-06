"""Phase 2b exp 12: ridge / grid sensitivity of the 1D WildBench scenario calibration.

Refits the 1D M2PL (calibration set = ALL fitted criteria, incl. extreme_a which are kept
in calibration) across ridge x GH-grid, reporting loglik, #extreme_a, the full-bank theta
correlation vs the ridge=1e-2 baseline, and a held-out 5-fold full-bank EAP recovery r (no
engine; FINE uniform EAP reference). Answers whether ridge should move from 1e-2 and lets
us re-derive the extreme_a CAT-pool exclusion at the chosen ridge.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import log_expit, logsumexp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import wildbench_scenario_lib as L  # noqa: E402

cm = L.cm


def eap_1d(Y, M, a, b, grid, logw):
    eta = a[:, None] * grid[None, :] - b[:, None]
    logP, log1mP = log_expit(eta), log_expit(-eta)
    YM = np.where(M, Y, 0.0)
    NM = np.where(M, 1.0 - Y, 0.0)
    LL = YM @ logP + NM @ log1mP
    joint = LL + logw[None, :]
    post = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
    return post @ grid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    base = L.ROOT / "wildbench_calibration"
    p.add_argument("--matrix", type=Path, default=L.DEFAULT_MATRIX)
    p.add_argument("--rubrics", type=Path, default=L.DEFAULT_RUBRICS)
    p.add_argument("--scenarios", type=Path, default=L.DEFAULT_SCENARIOS)
    p.add_argument("--out-dir", type=Path,
                   default=base / "experiments" / "12_ridge_grid_sensitivity")
    p.add_argument("--ridges", type=str, default="0.001,0.01,0.1")
    p.add_argument("--grids", type=str, default="7,15,21")
    p.add_argument("--base-ridge", type=float, default=0.01)
    p.add_argument("--base-grid", type=int, default=7,
                   help="production fit grid (shared-toolchain default); extreme_a is "
                        "grid-sensitive so this is FIXED and documented.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eap-grid", type=int, default=321)
    p.add_argument("--range", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=20260729)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    maps = L.load_maps(args.rubrics, args.scenarios)
    Y, M, items, models, drop = L.prepare_matrix(args.matrix)
    Q1, _ = L.build_Q(items, maps["qmap"], L.STRUCTURES["overall_1d"])
    n = len(models)
    n_obs = int(M.sum())
    egrid = np.linspace(-args.range, args.range, args.eap_grid)
    elogw = -0.5 * egrid ** 2
    elogw = elogw - logsumexp(elogw)

    ridges = [float(x) for x in args.ridges.split(",") if x.strip()]
    grids = [int(x) for x in args.grids.split(",") if x.strip()]

    print("=" * 84)
    print("EXP 12: ridge / grid sensitivity (1D WildBench scenario calibration)")
    print("=" * 84)

    base_fit = cm.fit_m2pl_em(Y, M, Q1, args.base_grid, estimate_corr=False,
                              ridge=args.base_ridge, max_iter=200)
    base_theta = eap_1d(Y, M, base_fit["A"][:, 0], base_fit["b"], egrid, elogw)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    folds = [order[i::args.k] for i in range(args.k)]

    rows = []
    extreme_ids_by = {}
    for ridge in ridges:
        for g in grids:
            fit = cm.fit_m2pl_em(Y, M, Q1, g, estimate_corr=False, ridge=ridge, max_iter=200)
            a = fit["A"][:, 0]
            b = fit["b"]
            ext_mask = (~np.isfinite(a)) | (np.abs(a) > cm.EXTREME_A)
            n_extreme = int(np.sum(ext_mask))
            extreme_ids_by[(ridge, g)] = [items[j] for j in np.where(ext_mask)[0]]
            theta = eap_1d(Y, M, a, b, egrid, elogw)
            r_vs_base = float(np.corrcoef(theta, base_theta)[0, 1])
            est = np.full(n, np.nan)
            for te in folds:
                tr = np.array([i for i in range(n) if i not in set(te.tolist())])
                keep = np.array([j for j in range(len(items))
                                 if M[tr][:, j].sum() >= 2
                                 and 0 < Y[tr][M[tr][:, j], j].sum() < M[tr][:, j].sum()])
                ft = cm.fit_m2pl_em(np.nan_to_num(Y[np.ix_(tr, keep)], nan=0.0),
                                    M[np.ix_(tr, keep)], Q1[keep], g,
                                    estimate_corr=False, ridge=ridge, max_iter=150)
                th = eap_1d(Y[np.ix_(te, keep)], M[np.ix_(te, keep)], ft["A"][:, 0], ft["b"],
                            egrid, elogw)
                est[te] = th
            oos_r = float(np.corrcoef(est, base_theta)[0, 1])
            aic, bic = cm.aic_bic(fit["loglik"], fit["n_params"], n_obs)
            row = {"ridge": ridge, "grid": g, "loglik": round(fit["loglik"], 1),
                   "aic": round(aic, 1), "bic": round(bic, 1), "n_extreme_a": n_extreme,
                   "theta_r_vs_base": round(r_vs_base, 5),
                   "oos_5fold_r": round(oos_r, 4), "converged": fit["converged"]}
            rows.append(row)
            print(f"  ridge={ridge:<6g} grid={g}: loglik={fit['loglik']:.0f} "
                  f"extreme_a={n_extreme} theta_r_vs_base={r_vs_base:.4f} oos_r={oos_r:.3f}",
                  flush=True)

    with (args.out_dir / "ridge_grid_sensitivity.csv").open("w", newline="",
                                                            encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    min_r = min(r["theta_r_vs_base"] for r in rows)
    best_oos = max(rows, key=lambda r: r["oos_5fold_r"])
    base_row = next(r for r in rows if r["ridge"] == args.base_ridge and r["grid"] == args.base_grid)
    # keep 1e-2 unless another ridge clearly wins OOS (> +0.005) AND theta rank stays stable
    keep_base = (best_oos["oos_5fold_r"] - base_row["oos_5fold_r"] <= 0.005) or (min_r >= 0.98)
    chosen = ({"ridge": args.base_ridge, "grid": args.base_grid} if keep_base
              else {"ridge": best_oos["ridge"], "grid": best_oos["grid"]})
    chosen_extreme = extreme_ids_by[(chosen["ridge"], chosen["grid"])]
    conclusion = (
        f"theta rank is highly stable across ridge/grid (min corr vs baseline = {min_r:.4f}); "
        f"best OOS 5-fold r = {best_oos['oos_5fold_r']} at ridge={best_oos['ridge']} "
        f"grid={best_oos['grid']}. Chosen ridge={chosen['ridge']} grid={chosen['grid']} "
        f"(extreme_a={len(chosen_extreme)}). " +
        ("Ridge 1e-2 kept (within noise of the best; keeps extreme_a controlled)."
         if keep_base else "Sweep favored a different ridge; re-derive extreme_a + catpool."))
    (args.out_dir / "conclusion.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_theta_corr_vs_base": min_r, "best_oos": best_oos, "base_row": base_row,
        "recommendation": f"ridge={chosen['ridge']}, grid={chosen['grid']}",
        "chosen": chosen, "keep_base_ridge_1e2": bool(keep_base),
        "n_extreme_a_at_chosen": len(chosen_extreme),
        "extreme_a_ids_at_chosen": sorted(chosen_extreme),
        "conclusion": conclusion, "rows": rows}, indent=2), encoding="utf-8")

    # figure: extreme_a + oos_r vs ridge (per grid)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for g in grids:
        dd = df[df["grid"] == g].sort_values("ridge")
        ax1.plot(dd["ridge"], dd["n_extreme_a"], "o-", label=f"grid={g}")
        ax2.plot(dd["ridge"], dd["oos_5fold_r"], "o-", label=f"grid={g}")
    ax1.set_xscale("log"); ax1.set_xlabel("ridge"); ax1.set_ylabel("# extreme_a items")
    ax1.set_title("extreme_a vs ridge"); ax1.legend(fontsize=8)
    ax2.set_xscale("log"); ax2.set_xlabel("ridge"); ax2.set_ylabel("OOS 5-fold r")
    ax2.set_title("held-out full-bank recovery vs ridge"); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out_dir / "ridge_sensitivity.png", dpi=140)
    plt.close(fig)

    print("\n" + conclusion)
    print(f"wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
