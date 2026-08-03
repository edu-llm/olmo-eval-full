"""SCENARIO-LEVEL out-of-sample k-fold comparison of the CAT ability estimators.

Scenario-level replacement for the item-level ``kfold_estimator_cv.py``. For each person
fold we refit the item parameters (a, b) on the TRAIN models, then run the REAL scenario
engine on the held-out TEST models with those params and score recovery of the fold's
full-bank reference by each final estimator (online / batch EAP / MWLE). Item selection is
scenario-level throughout, exactly as an online CAT does.

Fitting: the confirmatory Q (which modeled skill each criterion loads) is taken from the
frozen ``*_fitted`` bank's ``q_modeled`` and held fixed (that is the pre-specified design);
only (a, b) are re-estimated on TRAIN via ``calibrate_mirt.fit_m2pl_em`` after dropping
items with no TRAIN variance. This works uniformly for the 2- and 3-skill banks. Because the
frozen item SET was originally chosen using all 82 models there is a negligible selection
leak, but the estimator-vs-estimator comparison (all sharing the same fold params) is
unaffected.

Parallel across TEST models via ``--workers``.

Usage
-----
    python scripts/scenario_kfold_estimator_cv.py \
        --bank data/TutorBench/rubrics_qmatrix_calibrated_2skill_fitted.jsonl \
        --matrix staging/response_matrix_full_nonopt.csv \
        --out-dir regenerated_figures/scenario_level/kfold/2_skills --workers 12
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
ESTIMATORS = ("online", "batch", "mwle")


def make_folds(models, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(models))
    folds = [[] for _ in range(k)]
    for pos, i in enumerate(idx):
        folds[pos % k].append(models[int(i)])
    return [sorted(f) for f in folds]


def recovery_stats(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    w = np.argsort(x)[:12]
    return {"slope": float(np.polyfit(x, y, 1)[0]), "r": float(np.corrcoef(x, y)[0, 1]),
            "gap_worst12": float(np.mean(y[w] - x[w])), "n": int(x.size)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--scenarios", type=Path,
                   default=ROOT / "data" / "TutorBench" / "scenarios.jsonl")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tmp-dir", type=Path, default=ROOT / "staging" / "_kfold_scen")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--fit-grid", type=int, default=7)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--eap-grid", type=int, default=61)
    p.add_argument("--range", type=float, default=6.0)
    p.add_argument("--max-se", type=float, default=0.30)
    p.add_argument("--min-evals-per-skill", type=int, default=15)
    p.add_argument("--min-scenarios", type=int, default=0,
                   help="minimum scenarios before a precision-based stop (0=off).")
    p.add_argument("--max-scenarios", type=int, default=50)
    p.add_argument("--workers", type=int, default=scat.default_workers())
    args = p.parse_args()

    records, dims, _ = scat.load_fitted_bank(args.bank, "clamp")
    ids = [r["criterion_id"] for r in records]
    scen_of = {r["criterion_id"]: r["scenario_id"] for r in records}
    crit_of = {r["criterion_id"]: r.get("criterion", "") for r in records}
    Q_all = np.array([[int(r["q_modeled"][d]) for d in dims] for r in records])
    matrix = pd.read_csv(args.matrix, index_col=0)
    sub = matrix.reindex(columns=ids)
    Yraw = sub.to_numpy(float)
    Mall = ~np.isnan(Yraw)
    models = list(matrix.index)
    row_of = {m: i for i, m in enumerate(models)}
    n_dims = len(dims)
    print(f"bank={args.bank.name} dims={dims} models={len(models)} items={len(ids)} "
          f"k={args.k} workers={args.workers}")

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    grid, log_prior = scat.build_grid(n_dims, args.eap_grid, args.range)
    folds = make_folds(models, args.k, args.seed)

    pooled = {e: {"x": [], "y": {d: [] for d in dims}} for e in ESTIMATORS}
    fold_len = []
    per_model_rows: list[dict] = []  # additive per-model OOS export
    for f in range(args.k):
        test = folds[f]
        train = [m for m in models if m not in set(test)]
        tr_idx = [row_of[m] for m in train]
        Ytr = Yraw[tr_idx]
        Mtr = Mall[tr_idx]
        # keep items with TRAIN variance (>=1 pass and >=1 fail among observed)
        keep = []
        for j in range(len(ids)):
            obs = Mtr[:, j]
            if obs.sum() >= 2:
                vals = Ytr[obs, j]
                if 0 < vals.sum() < obs.sum():
                    keep.append(j)
        keep = np.array(keep, dtype=int)
        Q_k = Q_all[keep]
        Ytr_k = np.nan_to_num(Ytr[:, keep], nan=0.0)
        Mtr_k = Mtr[:, keep]
        fit = cm.fit_m2pl_em(Ytr_k, Mtr_k, Q_k, args.fit_grid, ridge=args.ridge,
                             max_iter=args.max_iter)
        A_f, b_f = fit["A"], fit["b"]
        kept_ids = [ids[j] for j in keep]
        print(f"  fold {f}: TRAIN={len(train)} TEST={len(test)} fit {len(kept_ids)} items "
              f"(loglik={fit['loglik']:.0f}, iters={fit['n_iter']})", flush=True)

        # write a temp fold bank in the fitted-bank schema
        fold_bank = args.tmp_dir / f"fold{f}_bank.jsonl"
        with fold_bank.open("w", encoding="utf-8") as fh:
            for jj, cid in enumerate(kept_ids):
                rec = {"criterion_id": cid, "scenario_id": scen_of[cid],
                       "criterion": crit_of[cid],
                       "discrimination": {d: float(A_f[jj, kk]) for kk, d in enumerate(dims)},
                       "q_modeled": {d: int(Q_k[jj, kk]) for kk, d in enumerate(dims)},
                       "difficulty": float(b_f[jj])}
                fh.write(json.dumps(rec) + "\n")

        # reference ability for TEST models under fold params
        te_idx = [row_of[m] for m in test]
        colk = {c: i for i, c in enumerate(kept_ids)}
        Ak = A_f
        bk = b_f
        subte = matrix.loc[test].reindex(columns=kept_ids)
        Yte = np.nan_to_num(subte.to_numpy(float), nan=0.0)
        Mte = ~np.isnan(subte.to_numpy(float))
        theta_ref = scat.eap_all_models(Yte, Mte, Ak, bk, grid, log_prior)

        # run the scenario engine on TEST models with the fold bank
        spec = scat.RunSpec(seed=args.seed, max_se=args.max_se,
                            min_evals_per_skill=args.min_evals_per_skill,
                            min_scenarios=args.min_scenarios,
                            max_scenarios=args.max_scenarios, selection="trace",
                            runs_dir=str(args.tmp_dir / f"runs_f{f}"))
        results = scat.run_models(test, fold_bank, args.matrix, args.scenarios,
                                  "clamp", dims, spec, workers=args.workers)
        res_by = {r["model"]: r for r in results}
        for ti, m in enumerate(test):
            rec0 = res_by[m]
            y = Yte[ti]
            idx = np.array([colk[c] for c in rec0["order"] if c in colk], dtype=int)
            th_on = np.asarray(rec0["theta_online"], float)
            th_ba = scat.eap_subset(y, idx, Ak, bk, grid, log_prior) if idx.size else th_on.copy()
            th_mw, _ = (scat.mwle_subset(y, idx, Ak, bk, th_ba) if idx.size else (th_ba.copy(), True))
            ths = {"online": th_on, "batch": th_ba, "mwle": th_mw}
            fold_len.append(rec0["criteria_administered"])
            for e in ESTIMATORS:
                pooled[e]["x"].append(theta_ref[ti])
                for kk, d in enumerate(dims):
                    pooled[e]["y"][d].append(ths[e][kk])
            prow = {"model": m, "fold": f,
                    "criteria_administered": rec0["criteria_administered"],
                    "scenarios_administered": rec0["scenarios_administered"]}
            for kk, d in enumerate(dims):
                prow[f"theta_ref_{d}"] = float(theta_ref[ti][kk])
                prow[f"theta_online_{d}"] = float(th_on[kk])
                prow[f"theta_batch_{d}"] = float(th_ba[kk])
                prow[f"theta_mwle_{d}"] = float(th_mw[kk])
            per_model_rows.append(prow)

    # aggregate pooled OOS recovery
    agg = {}
    for e in ESTIMATORS:
        X = np.array(pooled[e]["x"])
        agg[e] = {}
        for kk, d in enumerate(dims):
            agg[e][d] = recovery_stats(X[:, kk], np.array(pooled[e]["y"][d]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "purpose": "scenario-level OOS k-fold estimator recovery (real engine on held-out models)",
               "bank": str(args.bank), "matrix": str(args.matrix), "dims": dims,
               "k": args.k, "seed": args.seed, "n_models": len(models),
               "config": {"fit_grid": args.fit_grid, "ridge": args.ridge,
                          "eap_grid": args.eap_grid, "max_se": args.max_se,
                          "min_evals_per_skill": args.min_evals_per_skill,
                          "max_scenarios": args.max_scenarios},
               "mean_criteria_administered": float(np.mean(fold_len)),
               "oos_recovery": agg}
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # additive: per-model OOS abilities (held-out reference + each estimator)
    pd.DataFrame(per_model_rows).sort_values("model").to_csv(
        args.out_dir / "oos_per_model.csv", index=False)

    _figures(pooled, dims, agg, args.out_dir / "figures")

    print("\n" + "=" * 90)
    print("SCENARIO-LEVEL OOS RECOVERY (reference = full-info EAP under TRAIN-fit params)")
    print("=" * 90)
    for e in ESTIMATORS:
        print(f"  {e:7s}: " + "  ".join(
            f"{d[:4]} r={agg[e][d]['r']:.3f} m={agg[e][d]['slope']:.3f}" for d in dims))
    print(f"  mean criteria/model administered: {np.mean(fold_len):.1f}")
    print(f"wrote -> {args.out_dir}")
    return 0


def _figures(pooled, dims, agg, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    for kk, d in enumerate(dims):
        fig, axes = plt.subplots(1, len(ESTIMATORS), figsize=(4.4 * len(ESTIMATORS), 4.4),
                                 sharex=True, sharey=True)
        X = np.array(pooled["online"]["x"])[:, kk]
        for ax, e in zip(axes, ESTIMATORS):
            y = np.array(pooled[e]["y"][d])
            s = agg[e][d]
            ax.scatter(X, y, s=20, alpha=0.6, edgecolor="k", linewidth=0.2)
            lo = min(np.nanmin(X), np.nanmin(y)) - 0.3
            hi = max(np.nanmax(X), np.nanmax(y)) + 0.3
            ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(f"reference EAP ({d})"); ax.set_ylabel(f"CAT {e} ({d})")
            ax.set_title(f"{e}: r={s['r']:.3f} slope={s['slope']:.3f}", fontsize=9)
        fig.suptitle(f"Scenario-level OOS estimator recovery: {d}", fontsize=11)
        fig.tight_layout(); fig.savefig(fig_dir / f"oos_recovery_{d}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
