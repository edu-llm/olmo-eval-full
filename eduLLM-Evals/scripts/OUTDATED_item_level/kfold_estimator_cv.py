"""Out-of-sample k-fold comparison of the three CAT ability estimators.

Why this exists
---------------
``scripts/regen_cat_figures.py`` and ``scripts/offline_engine_driver.py`` show that the
production *online* (sequential Laplace) theta compresses the ability scale (slope well
below 1), and that batch-EAP and especially MWLE largely correct it. But those numbers
are IN SAMPLE: the item parameters were calibrated on the very 82 models that are then
scored. An estimator can look good in-sample by over-fitting the calibration crowd.

This harness answers the honest question: does the MWLE (and batch) advantage over the
online estimator SURVIVE when the item parameters were fit on OTHER people?

Design (person-level k-fold, strictly reusing existing code)
-----------------------------------------------------------
For each fold f:
  1. Fit the 2-skill collapsed M2PL on the TRAIN models (all models not in f) using
     ``kfold_cv_mirt.fit_2skill`` -> the EXACT calibration EM/MML from
     ``scripts/calibrate_mirt.py`` (same collapse, ridge, quadrature, item filtering).
     The held-out (TEST) models NEVER contribute to these item parameters.
  2. Freeze (A, b). For every TEST person, using ONLY their observed responses on the
     fitted items:
       * reference theta  = full-information EAP over the dense uniform grid
         (``regen.eap_all_models``) -- the best estimate the frozen bank can give.
       * run the production max-Fisher-info CAT (``regen.run_cat_person``) to get the
         administered item subset + the ONLINE sequential theta.
       * batch theta = EAP over that administered subset (``regen.eap_subset``).
       * mwle  theta = multidimensional WLE over the subset (``regen.mwle_subset``).
  3. Pool all held-out persons across folds and, per dimension and per estimator,
     measure recovery of the reference theta: slope (regression of estimate on
     reference), Pearson r, and the mean gap on the 12 lowest-ability models.

Reference and CAT estimators all use the SAME standard-normal-prior uniform grid, exactly
as ``regen_cat_figures`` does in sample, so the ONLY thing that differs from the published
in-sample figures is that the item bank came from TRAIN. The delta between this and the
in-sample recovery is the generalisation gap.

This script is READ-ONLY w.r.t. calibration: it re-fits per fold in memory and never edits
the fitter, the rubric bank, or any frozen artifact. It only writes under ``--out-dir``.

Usage
-----
    python scripts/kfold_estimator_cv.py \
        --matrix staging/response_matrix_full_nonopt.csv \
        --rubrics data/TutorBench/curated/rubrics_qmatrix_curated.jsonl \
        --ridge 1e-2 --out-dir regenerated_figures/kfold_estimator/2_skills
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# parents[2] because this file now lives in scripts/OUTDATED_item_level/ (see README there).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Reuse the OOS calibration (fold fit) and the three estimators verbatim.
kf = _load("kfold_cv_mirt", ROOT / "scripts" / "kfold_cv_mirt.py")
regen = _load("regen", ROOT / "scripts" / "regen_cat_figures.py")

DIM_LABELS = ["correctness", "scaffolding"]
ESTIMATORS = {"theta_online": "production online",
              "theta_batch": "batch EAP",
              "theta_mwle": "batch EAP + MWLE"}


def recovery_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Recovery of reference ``x`` by estimate ``y``: slope, r, worst-12 gap."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    w = np.argsort(x)[:12]
    return {
        "slope": float(np.polyfit(x, y, 1)[0]),
        "r": float(np.corrcoef(x, y)[0, 1]),
        "gap_worst12": float(np.mean(y[w] - x[w])),
        "n": int(x.size),
    }


def run(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = kf.cp.load_matrix(args.matrix)
    q_by = kf.cm.load_q_matrix(args.rubrics)
    models = list(mat.index)
    print(f"loaded matrix: {len(models)} models x {mat.shape[1]} criteria")
    print(f"Q-matrix source: {args.rubrics} ({len(q_by)} criteria)")

    folds = kf.make_folds(models, args.k, args.seed)
    print(f"k={args.k} folds (seed={args.seed}): sizes {[len(f) for f in folds]}")

    # namespace fit_2skill expects
    fit_args = SimpleNamespace(
        grid=args.fit_grid, estimate_latent_corr=args.estimate_latent_corr,
        ridge=args.ridge, max_iter=args.max_iter, tol=args.tol,
    )

    grid, log_prior = regen.build_grid(2, args.eap_grid, "uniform", args.range)

    rows: list[dict] = []
    for f in range(args.k):
        test_models = folds[f]
        train_models = [m for m in models if m not in set(test_models)]
        print(f"\n=== fold {f}: TRAIN={len(train_models)} TEST={len(test_models)} ===",
              flush=True)
        fit = kf.fit_2skill(mat.loc[train_models], q_by, fit_args)
        items, A, b = fit["items"], fit["A"], fit["b"]
        print(f"  fitted {len(items)} items "
              f"(loglik={fit['loglik']:.1f}, iters={fit['n_iter']}, "
              f"converged={fit['converged']})")

        sub = mat.loc[test_models].reindex(columns=items)
        Yraw = sub.to_numpy(dtype=float)
        mask = ~np.isnan(Yraw)
        Y = np.nan_to_num(Yraw, nan=0.0)

        theta_ref = regen.eap_all_models(Y, mask, A, b, grid, log_prior, 4096)

        for r, model in enumerate(test_models):
            th_on, n_admin, se, _, order = regen.run_cat_person(
                Y[r], mask[r], A, b, args.min_items, args.max_items, args.se_target)
            idx = np.asarray(order, dtype=int)
            th_ba = regen.eap_subset(Y[r], idx, A, b, grid, log_prior) if idx.size else th_on.copy()
            th_mw, ok = (regen.mwle_subset(Y[r], idx, A, b, th_ba)
                         if idx.size else (th_ba.copy(), True))
            rec = {"model": model, "fold": f, "n_admin": int(n_admin),
                   "mwle_converged": bool(ok)}
            for k, d in enumerate(DIM_LABELS):
                rec[f"theta_ref_{d}"] = float(theta_ref[r, k])
                rec[f"theta_online_{d}"] = float(th_on[k])
                rec[f"theta_batch_{d}"] = float(th_ba[k])
                rec[f"theta_mwle_{d}"] = float(th_mw[k])
                rec[f"se_{d}"] = float(se[k])
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cat_estimator_per_model.csv", index=False)

    # pooled OOS recovery, per estimator per dim
    agg: dict[str, dict] = {}
    for est in ESTIMATORS:
        agg[est] = {}
        for d in DIM_LABELS:
            x = df[f"theta_ref_{d}"].to_numpy()
            y = df[f"{est.replace('theta_', 'theta_')}_{d}"].to_numpy()
            agg[est][d] = recovery_stats(x, y)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("out-of-sample recovery of the full-information reference theta by the "
                    "three CAT final estimators, item params fit on TRAIN folds only"),
        "config": {"k": args.k, "seed": args.seed, "fit_grid": args.fit_grid,
                   "eap_grid": args.eap_grid, "range": args.range, "ridge": args.ridge,
                   "max_iter": args.max_iter, "tol": args.tol,
                   "se_target": args.se_target, "min_items": args.min_items,
                   "max_items": args.max_items,
                   "estimate_latent_corr": args.estimate_latent_corr},
        "provenance": {"matrix": str(args.matrix), "rubrics": str(args.rubrics)},
        "n_models": int(df.model.nunique()),
        "fold_sizes": [len(f) for f in folds],
        "cat_length": {"mean": float(df.n_admin.mean()),
                       "median": float(df.n_admin.median()),
                       "max": int(df.n_admin.max())},
        "mwle_nonconvergence": int((~df.mwle_converged).sum()),
        "oos_recovery": agg,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    _figures(df, out_dir / "figures", agg)

    print("\n" + "=" * 92)
    print("OUT-OF-SAMPLE RECOVERY (reference = full-info EAP under TRAIN-fit params)")
    print("=" * 92)
    hdr = f"{'dim':14s}" + "".join(f"{ESTIMATORS[e]:>24s}" for e in ESTIMATORS)
    print(hdr)
    for d in DIM_LABELS:
        print(f"{d:14s}" + "".join(
            f"  r={agg[e][d]['r']:.3f} m={agg[e][d]['slope']:.3f}" for e in ESTIMATORS))
    print(f"\nCAT length: mean={df.n_admin.mean():.1f} median={df.n_admin.median():.0f} "
          f"max={df.n_admin.max()}")
    print(f"wrote -> {out_dir}")
    return 0


def _figures(df: pd.DataFrame, fig_dir: Path, agg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    for d in DIM_LABELS:
        x = df[f"theta_ref_{d}"].to_numpy()
        fig, axes = plt.subplots(1, len(ESTIMATORS), figsize=(4.4 * len(ESTIMATORS), 4.4),
                                 sharex=True, sharey=True)
        for ax, (est, label) in zip(axes, ESTIMATORS.items()):
            y = df[f"{est}_{d}"].to_numpy()
            s = agg[est][d]
            ax.scatter(x, y, s=24, alpha=0.7, edgecolor="k", linewidth=0.3)
            lo = min(np.nanmin(x), np.nanmin(y)) - 0.3
            hi = max(np.nanmax(x), np.nanmax(y)) + 0.3
            ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="y = x")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(f"reference EAP ({d})")
            ax.set_ylabel(f"CAT estimate ({d})")
            ax.set_title(f"{label}\nr={s['r']:.3f}  slope={s['slope']:.3f}", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(f"Out-of-sample estimator recovery: {d}", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"oos_estimator_recovery_{d}.png", dpi=130)
        plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path,
                   default=ROOT / "staging" / "response_matrix_full_nonopt.csv")
    p.add_argument("--rubrics", type=Path,
                   default=ROOT / "data" / "TutorBench" / "curated" / "rubrics_qmatrix_curated.jsonl")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--fit-grid", type=int, default=7,
                   help="Gauss-Hermite nodes/dim for the per-fold EM fit (default 7).")
    p.add_argument("--eap-grid", type=int, default=61,
                   help="uniform nodes/dim for reference + batch EAP scoring (default 61).")
    p.add_argument("--range", type=float, default=6.0)
    p.add_argument("--ridge", type=float, default=1e-2,
                   help="M-step L2 ridge; default 1e-2 to match the frozen 2-skill pilot.")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--estimate-latent-corr", action="store_true")
    p.add_argument("--se-target", type=float, default=0.3)
    p.add_argument("--min-items", type=int, default=1)
    p.add_argument("--max-items", type=int, default=100)
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "regenerated_figures" / "kfold_estimator" / "2_skills")
    return p


def main() -> int:
    return run(build_argparser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
