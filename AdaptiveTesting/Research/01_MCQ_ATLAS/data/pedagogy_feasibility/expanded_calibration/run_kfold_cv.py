#!/usr/bin/env python3
"""K-fold cross-validated recovery for the pedagogy CAT diagnostic, OLD vs NEW.

The original feasibility study used a single 12-model held-out split. That is faithful but
noisy: the OLD and NEW seed-7 splits draw *different* held-out models, so their Pearson r
values are not measured on the same targets. This script complements the single split with
model-level K-fold CV, where every model is held out exactly once and predicted from a bank
calibrated on the remaining models. Pooling all out-of-sample predictions gives a lower-
variance, population-level recovery number that is a fairer OLD-vs-NEW comparison.

Same machinery as ``run_expanded_calibration.py`` / ``scripts/mcq_diagnostic.py``: same
``load_benchmark`` / ``filter_items`` / ``fit_2pl``, same EAP-theta Fisher-info CAT with an
8-item floor and mean-probability predictor, same SE targets. For the linear recalibration
we use leave-one-out over the pooled out-of-sample points (honest, no in-sample leakage).

Also reports NEW restricted to the original 52 models ("NEW_on_old52"): the recovery of the
SAME 52 models under the expanded (78-model) calibration regime, so the calibration-set
effect can be read off holding the evaluated models fixed.

CPU-only, torch-free, ``uv run``, local ``.mplcache``.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "eduLLM-Evals"))

from tutor_cat.mcq_irt.calibrate import fit_2pl  # noqa: E402
from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark  # noqa: E402
from run_expanded_calibration import (  # noqa: E402
    BENCH, MCQ_DIR, NEW_STEMS, SE_TARGETS, PRIMARY_SE, lsq_fit, mae, prob, run_cat,
)

SEED = 7
K = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def kfold_preds(full_mat, models):
    mat = full_mat.loc[sorted(models)]
    actual_full = mat.mean(axis=1).to_dict()
    model_list = list(mat.index)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(model_list))
    folds = [[model_list[i] for i in perm[f::K]] for f in range(K)]
    pooled = {mn: {se: {"model": [], "pred": [], "act": [], "nit": []}
                   for se in SE_TARGETS} for mn in ["2PL", "1PL"]}
    kept_sizes = []
    for f, test in enumerate(folds):
        tset = set(test)
        train = [m for m in model_list if m not in tset]
        kept_mat, report = filter_items(mat.loc[train], benchmark=BENCH)
        kept_sizes.append(report.n_items_kept)
        for mn, method in [("2PL", "girth"), ("1PL", "rasch")]:
            cal = fit_2pl(kept_mat, method=method)
            items = cal.items
            a = np.asarray(cal.a, float)
            b = np.asarray(cal.b, float)
            for se in SE_TARGETS:
                for m in test:
                    resp = mat.loc[m, items].to_numpy(float)
                    theta, _se, nit = run_cat(resp, a, b, se, max_items=len(items))
                    pooled[mn][se]["model"].append(m)
                    pooled[mn][se]["pred"].append(float(prob(theta, a, b).mean()))
                    pooled[mn][se]["act"].append(float(actual_full[m]))
                    pooled[mn][se]["nit"].append(nit)
        print(f"  fold {f + 1}/{K}: held_out={len(test)} train={len(train)} kept={report.n_items_kept}",
              flush=True)
    return pooled, float(np.mean(kept_sizes))


def loo_linear_mae(pred, act):
    """Leave-one-out linear recalibration MAE over pooled out-of-sample points."""
    pred = np.asarray(pred)
    act = np.asarray(act)
    out = np.empty_like(pred)
    for i in range(len(pred)):
        msk = np.arange(len(pred)) != i
        a, b = lsq_fit(pred[msk], act[msk])
        out[i] = a * pred[i] + b
    return mae(out, act)


def summarize(pooled, kept_mean, subset=None, label=""):
    rows = {}
    for mn in ["2PL", "1PL"]:
        rows[mn] = {}
        for se in SE_TARGETS:
            d = pooled[mn][se]
            model = np.array(d["model"])
            pred = np.array(d["pred"])
            act = np.array(d["act"])
            nit = np.array(d["nit"])
            if subset is not None:
                keep = np.array([m in subset for m in model])
                pred, act, nit = pred[keep], act[keep], nit[keep]
            r = float(np.corrcoef(pred, act)[0, 1])
            a_all, b_all = lsq_fit(pred, act)
            rows[mn][se] = {
                "n_eval": int(len(pred)), "r": round(r, 4),
                "mae_raw": round(mae(pred, act), 4),
                "mae_loo_cal": round(loo_linear_mae(pred, act), 4),
                "avg_items": round(float(np.mean(nit)), 2),
                "pct_items": round(100.0 * float(np.mean(nit)) / kept_mean, 2),
                "slope_a": round(a_all, 4), "intercept_b": round(b_all, 4),
                "pred_spread": round(float(pred.max() - pred.min()), 4),
                "acts": act, "preds": pred,
            }
    return rows


def main():
    full = load_benchmark(MCQ_DIR, BENCH).dropna(axis=0, how="any")
    all_models = list(full.index)
    new_models = [m for m in all_models if m.replace("/", "__") in NEW_STEMS]
    old_models = [m for m in all_models if m.replace("/", "__") not in NEW_STEMS]
    old_set = set(old_models)
    print(f"K={K}  OLD={len(old_models)}  NEW={len(all_models)}", flush=True)

    print("OLD k-fold:", flush=True)
    old_pooled, old_kept = kfold_preds(full, old_models)
    print("NEW k-fold:", flush=True)
    new_pooled, new_kept = kfold_preds(full, all_models)

    old = summarize(old_pooled, old_kept, label="OLD")
    new = summarize(new_pooled, new_kept, label="NEW")
    new_on_old = summarize(new_pooled, new_kept, subset=old_set, label="NEW_on_old52")

    path = HERE / "kfold_cv_summary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "K", "model", "se_stop", "n_eval", "corr", "mae_raw",
                    "mae_loo_calibrated", "avg_cat_items", "pct_items", "slope_a",
                    "intercept_b", "pred_spread"])
        for tag, res in [("OLD", old), ("NEW", new), ("NEW_on_old52", new_on_old)]:
            for mn in ["2PL", "1PL"]:
                for se in SE_TARGETS:
                    d = res[mn][se]
                    w.writerow([tag, K, mn, se, d["n_eval"], d["r"], d["mae_raw"],
                                d["mae_loo_cal"], d["avg_items"], d["pct_items"],
                                d["slope_a"], d["intercept_b"], d["pred_spread"]])
    print(f"wrote {path}", flush=True)

    snap = {
        "K": K, "old_kept_mean": round(old_kept, 1), "new_kept_mean": round(new_kept, 1),
        "sets": {}
    }
    for tag, res in [("OLD", old), ("NEW", new), ("NEW_on_old52", new_on_old)]:
        snap["sets"][tag] = {mn: {str(se): {k: v for k, v in d.items()
                                            if k not in ("acts", "preds")}
                                  for se, d in res[mn].items()} for mn in ["2PL", "1PL"]}
    (HERE / "kfold_cv_snapshot.json").write_text(json.dumps(snap, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, mn in zip(axes, ["2PL", "1PL"]):
        do = old[mn][PRIMARY_SE]
        dn = new[mn][PRIMARY_SE]
        allv = np.concatenate([do["acts"], do["preds"], dn["acts"], dn["preds"]])
        lo, hi = allv.min() - 0.02, allv.max() + 0.02
        ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect prediction (y=x)")
        ax.scatter(do["acts"], do["preds"], s=42, color="#ff7f0e", edgecolor="black",
                   alpha=0.8, zorder=3, label=f"OLD 52 (r={do['r']:.3f})")
        ax.scatter(dn["acts"], dn["preds"], s=42, color="#1f77b4", marker="D", edgecolor="black",
                   alpha=0.8, zorder=3, label=f"NEW 78 (r={dn['r']:.3f})")
        ax.set_xlabel(f"actual full {BENCH} accuracy")
        ax.set_ylabel(f"CV out-of-sample CAT-predicted (SE<={PRIMARY_SE:g})")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{mn}: {K}-fold CV recovery")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Pedagogy {K}-fold CV recovery: OLD 52 vs NEW 78 (every model held out once)")
    fig.tight_layout()
    fig.savefig(HERE / f"kfold_old_vs_new_se{PRIMARY_SE:g}.png", dpi=130)
    plt.close(fig)

    print("\n== K-FOLD CV (SE<=0.3) ==", flush=True)
    for mn in ["2PL", "1PL"]:
        do = old[mn][0.3]; dn = new[mn][0.3]; dr = new_on_old[mn][0.3]
        print(f"  {mn}: OLD52 r={do['r']:.3f} | NEW78 r={dn['r']:.3f} | "
              f"NEW_on_old52 r={dr['r']:.3f}  (items {do['avg_items']:.1f}->{dn['avg_items']:.1f})",
              flush=True)


if __name__ == "__main__":
    main()
