"""Phase 1 step 3: scenario-level dimensionality sweep (1..5 skills).

Fits the confirmatory M2PL at 1/2/3/4/5 latent skills over ALL 250 scenarios' criteria
(no source dedup), reports loglik / #params / AIC / BIC / held-out marginal log-loss(+SE)
/ max latent correlation / per-axis discriminations, and applies the selection rule the
item-level study used: lowest held-out log-loss, keep everything within 1 SE, then BIC +
fewest dimensions (parsimony). Held-out log-loss uses person k-folds, mirroring
``bridge_calibration/scripts/bridge_dimensionality.py``.

Writes ``experiments/03_structures/{structure_comparison.csv, selection.json}`` and a
CV figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_scenario_lib as L  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=L.DEFAULT_MATRIX)
    p.add_argument("--rubrics", type=Path, default=L.DEFAULT_RUBRICS)
    p.add_argument("--scenarios", type=Path, default=L.DEFAULT_SCENARIOS)
    p.add_argument("--out-dir", type=Path,
                   default=L.ROOT / "bridge_calibration" / "experiments" / "03_structures")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--grid", type=int, default=5, help="GH nodes/dim (5 -> 5^D nodes).")
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=20260729)
    args = p.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    maps = L.load_maps(args.rubrics, args.scenarios)
    Y, M, items, models, drop_info = L.prepare_matrix(args.matrix)
    n = len(models)
    n_obs = int(M.sum())
    print(f"dimensionality sweep: {len(items)} criteria x {n} models; "
          f"skills={L.BRIDGE_SKILLS}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    folds = [order[i::args.k] for i in range(args.k)]

    rows = []
    per_axis_5d = None
    for name, structure in L.STRUCTURES.items():
        Q, labels = L.build_Q(items, maps["qmap"], structure)
        keepj = np.where(Q.sum(axis=1) > 0)[0]
        Qk = Q[keepj]
        Yk, Mk = Y[:, keepj], M[:, keepj]
        ndim = Qk.shape[1]
        fit = L.cm.fit_m2pl_em(Yk, Mk, Qk, args.grid, estimate_corr=(ndim > 1),
                               ridge=args.ridge, max_iter=200, tol=1e-4)
        aic, bic = L.cm.aic_bic(fit["loglik"], fit["n_params"], n_obs)

        grid = L.cm.build_grid(ndim, args.grid)
        base = L.cm.base_log_weights(ndim, args.grid)
        lp = base - logsumexp(base)
        lls = []
        for te in folds:
            tr = np.array([i for i in range(n) if i not in set(te.tolist())])
            ft = L.cm.fit_m2pl_em(Yk[tr], Mk[tr], Qk, args.grid, estimate_corr=False,
                                  ridge=args.ridge, max_iter=150, tol=1e-4)
            lls.append(L.marginal_logloss(Yk[te], Mk[te], ft["A"], ft["b"], grid, lp))
        ll_mean = float(np.mean(lls))
        ll_se = float(np.std(lls, ddof=1) / np.sqrt(len(lls)))
        R = fit.get("R")
        max_off = (float(np.max(np.abs(R - np.eye(ndim))))
                   if (R is not None and ndim > 1) else 0.0)
        rows.append({"structure": name, "n_dims": ndim, "n_criteria": int(len(keepj)),
                     "loglik": round(fit["loglik"], 1), "n_params": fit["n_params"],
                     "aic": round(aic, 1), "bic": round(bic, 1),
                     "oos_logloss_mean": round(ll_mean, 5),
                     "oos_logloss_se": round(ll_se, 5),
                     "max_latent_corr_offdiag": round(max_off, 3),
                     "converged": fit["converged"]})
        print(f"  {name} ({ndim}d): AIC={aic:.0f} BIC={bic:.0f} "
              f"OOS_ll={ll_mean:.5f}+/-{ll_se:.5f} max|corr|={max_off:.3f}", flush=True)

        if name == "full_5d":
            A5 = fit["A"]
            per_axis_5d = {}
            for di, sk in enumerate(labels):
                load = Qk[:, di] == 1
                av = A5[load, di]
                per_axis_5d[sk] = {
                    "n_anchor_items": int(load.sum()),
                    "median_disc": round(float(np.median(av)), 4) if av.size else None,
                    "mean_disc": round(float(np.mean(av)), 4) if av.size else None,
                    "frac_disc_le_0": round(float(np.mean(av <= 0)), 4) if av.size else None}
            print("    per-axis median disc: "
                  + ", ".join(f"{k}={v['median_disc']}" for k, v in per_axis_5d.items()),
                  flush=True)

    # selection: lowest OOS log-loss, keep within 1 SE, then fewest dims (BIC tiebreak)
    best = min(rows, key=lambda r: r["oos_logloss_mean"])
    thresh = best["oos_logloss_mean"] + best["oos_logloss_se"]
    within = [r for r in rows if r["oos_logloss_mean"] <= thresh]
    selected = min(within, key=lambda r: (r["n_dims"], r["bic"]))
    aic_best = min(rows, key=lambda r: r["aic"])
    bic_best = min(rows, key=lambda r: r["bic"])

    with (args.out_dir / "structure_comparison.csv").open("w", newline="",
                                                          encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (args.out_dir / "selection.json").write_text(json.dumps({
        "rule": "lowest held-out marginal log-loss; keep within 1 SE; then fewest "
                "dimensions (BIC tiebreak) + parsimony (NOT AIC)",
        "selected": selected["structure"], "selected_n_dims": selected["n_dims"],
        "best_ll_structure": best["structure"], "one_se_threshold": round(thresh, 5),
        "n_within_1se": len(within),
        "aic_best_structure": aic_best["structure"],
        "bic_best_structure": bic_best["structure"],
        "criterion_divergence_note": (
            f"AIC favors {aic_best['structure']} (AIC {aic_best['aic']:.0f}); BIC favors "
            f"{bic_best['structure']} (BIC {bic_best['bic']:.0f}). Selection rests on "
            "held-out log-loss + BIC + parsimony, NOT AIC -- AIC's weak per-parameter "
            "penalty prefers the higher-D fit that overfits at N=51."),
        "per_axis_discrimination_full_5d": per_axis_5d,
        "note": ("EXPLORATORY at N=51: higher-D confirmatory M2PL is not identifiable; "
                 "large max|latent corr| indicates collapse toward 1D."),
        "structures": rows}, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    xs = [r["structure"] for r in rows]
    ys = [r["oos_logloss_mean"] for r in rows]
    es = [r["oos_logloss_se"] for r in rows]
    ax.errorbar(range(len(xs)), ys, yerr=es, fmt="o", capsize=4)
    ax.axhline(thresh, ls="--", color="gray", label="1-SE threshold")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("held-out marginal log-loss (lower=better)")
    ax.set_title(f"Bridge scenario-level dimensionality (N={n}); "
                 f"selected: {selected['structure']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "structure_cv.png", dpi=140)

    print(f"\nselected: {selected['structure']} ({selected['n_dims']}d) | "
          f"AIC-best={aic_best['structure']} BIC-best={bic_best['structure']}")
    print(f"wrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
