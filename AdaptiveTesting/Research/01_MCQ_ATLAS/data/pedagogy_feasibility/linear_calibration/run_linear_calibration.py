#!/usr/bin/env python3
"""Linear calibration (y_hat = a*x + b) of the pedagogy MCQ CAT diagnostic.

Maps CAT/p-IRT predicted accuracy (x) -> true pedagogy accuracy (y) to remove the
systematic slope/intercept offset that inflates MAE while leaving Pearson r unchanged.

Reuses the exact bank calibration, EAP theta estimation, and Fisher-info CAT from
``scripts/mcq_diagnostic.py`` (2PL) and ``rasch_1pl/run_pedagogy_1pl.py`` (1PL). To
avoid in-sample circularity the calibration (a, b) is fit by least squares on the 40
TRAIN models' (predicted, actual) pairs and applied to the 12 held-out TEST models.
Also reports leave-one-out (LOO) on the 12 test models and the optimistic naive
in-sample fit for reference.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
REPO = HERE.parents[5]  # .../olmo-eval-full
sys.path.insert(0, str(REPO / "eduLLM-Evals"))

from tutor_cat.mcq_irt.calibrate import fit_2pl  # noqa: E402
from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark  # noqa: E402

MCQ_DIR = REPO / "AdaptiveTesting/Outputs/full200_results/Outputs/mcq"
BENCH = "pedagogy"
N_TEST = 12
SEED = 7
SE_TARGETS = [0.3, 0.15]

NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()


def prob(theta, a, b):
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(a * (theta - b), -30, 30))), 1e-6, 1 - 1e-6)


def eap_se(resp, a, b):
    z = np.clip(a[None, :] * (NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * PRIORW
    w /= w.sum()
    m = float((NODES * w).sum())
    sd = float(np.sqrt(((NODES - m) ** 2 * w).sum()))
    return m, sd


def run_cat(resp_all, a, b, se_stop, max_items, min_items=8):
    n = len(a)
    used = np.zeros(n, bool)
    theta, se, order = 0.0, 1.0, []
    for _ in range(min(max_items, n)):
        p = prob(theta, a, b)
        info = (a**2) * p * (1 - p)
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.array(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx])
        if len(order) >= min_items and se <= se_stop:
            break
    return theta, se, len(order)


def predict_models(mat, model_list, items, a, b, se_stop, actual_full):
    preds, acts = [], []
    for m in model_list:
        resp = mat.loc[m, items].to_numpy(float)
        theta, se, _ = run_cat(resp, a, b, se_stop, max_items=len(items))
        preds.append(float(prob(theta, a, b).mean()))
        acts.append(float(actual_full[m]))
    return np.array(preds), np.array(acts)


def lsq_fit(x, y):
    """Least-squares a, b for y = a*x + b."""
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def mae(pred, act):
    return float(np.mean(np.abs(pred - act)))


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = load_benchmark(MCQ_DIR, BENCH).dropna(axis=0, how="any")
    models = list(mat.index)
    rng = np.random.default_rng(SEED)
    order = [models[i] for i in rng.permutation(len(models))]
    n_test = min(N_TEST, len(models) // 3)
    test = sorted(order[:n_test])
    train = order[n_test:]
    actual_full = mat.mean(axis=1).to_dict()

    kept_mat, report = filter_items(mat.loc[train], benchmark=BENCH)

    rows = []
    for model_name, method in [("2PL", "girth"), ("1PL", "rasch")]:
        cal = fit_2pl(kept_mat, method=method)
        items = cal.items
        a = np.asarray(cal.a, float)
        b = np.asarray(cal.b, float)
        print(f"{model_name}: items_kept={len(items)} method={cal.method}", flush=True)

        for se_stop in SE_TARGETS:
            xtr, ytr = predict_models(mat, train, items, a, b, se_stop, actual_full)
            xte, yte = predict_models(mat, test, items, a, b, se_stop, actual_full)

            r = float(np.corrcoef(xte, yte)[0, 1])
            mae_raw = mae(xte, yte)

            # Honest: fit on 40 train, apply to 12 test.
            a_tr, b_tr = lsq_fit(xtr, ytr)
            pred_cal_train_fit = a_tr * xte + b_tr
            mae_train_fit = mae(pred_cal_train_fit, yte)

            # LOO on the 12 test models.
            loo_pred = np.empty_like(xte)
            for i in range(len(xte)):
                mask = np.arange(len(xte)) != i
                ai, bi = lsq_fit(xte[mask], yte[mask])
                loo_pred[i] = ai * xte[i] + bi
            mae_loo = mae(loo_pred, yte)

            # Naive in-sample (optimistic reference).
            a_in, b_in = lsq_fit(xte, yte)
            pred_cal_in = a_in * xte + b_in
            mae_in = mae(pred_cal_in, yte)

            def rec(method_name, mae_cal, aa, bb, n_train, n_test_):
                red = 100.0 * (mae_raw - mae_cal) / mae_raw if mae_raw > 0 else 0.0
                return {
                    "model": model_name, "se": se_stop, "calibration_method": method_name,
                    "r": round(r, 4), "mae_raw": round(mae_raw, 4),
                    "mae_calibrated": round(mae_cal, 4), "slope_a": round(aa, 4),
                    "intercept_b": round(bb, 4), "pct_mae_reduction": round(red, 1),
                    "n_train": n_train, "n_test": n_test_,
                }

            rows.append(rec("train_fit", mae_train_fit, a_tr, b_tr, len(train), len(test)))
            rows.append(rec("loo_test", mae_loo, float("nan"), float("nan"), len(test) - 1, len(test)))
            rows.append(rec("insample_naive", mae_in, a_in, b_in, len(test), len(test)))

            print(
                f"  {model_name} SE<{se_stop:g}: r={r:.3f} MAE_raw={mae_raw:.4f} "
                f"-> train_fit={mae_train_fit:.4f} ({rows[-3]['pct_mae_reduction']:.1f}%) "
                f"LOO={mae_loo:.4f} insample={mae_in:.4f} | a={a_tr:.3f} b={b_tr:.3f}",
                flush=True,
            )

            # Figure: before vs after (train-fit calibration, honest headline).
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
            lo = min(xte.min(), yte.min(), pred_cal_train_fit.min()) - 0.03
            hi = max(xte.max(), yte.max(), pred_cal_train_fit.max()) + 0.03
            # Left: raw predicted (y) vs actual (x), with fitted line.
            axL = axes[0]
            axL.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect (y=x)")
            axL.scatter(yte, xte, s=70, color="#1f77b4", edgecolor="black", zorder=3,
                        label="test models")
            xs = np.linspace(lo, hi, 100)
            # fitted line in (actual, predicted) space: pred = (act - b)/a inverse; but we
            # fit act = a*pred + b, so draw pred-axis line act_hat = a*pred+b as pred vs act.
            axL.plot(a_tr * xs + b_tr, xs, "-", color="#d62728",
                     label=f"train fit: act={a_tr:.2f}*pred+{b_tr:.2f}")
            axL.set_xlabel(f"actual full {BENCH} accuracy")
            axL.set_ylabel(f"raw CAT-predicted ({model_name}, SE<{se_stop})")
            axL.set_xlim(lo, hi); axL.set_ylim(lo, hi)
            axL.set_title(f"RAW  r={r:.3f}  MAE={mae_raw:.4f}")
            axL.grid(True, alpha=0.3); axL.legend(loc="upper left", fontsize=8)
            # Right: calibrated predicted vs actual.
            axR = axes[1]
            axR.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect (y=x)")
            axR.scatter(yte, pred_cal_train_fit, s=70, color="#2ca02c", edgecolor="black",
                        zorder=3, label="test models (calibrated)")
            axR.set_xlabel(f"actual full {BENCH} accuracy")
            axR.set_ylabel(f"calibrated predicted ({model_name}, SE<{se_stop})")
            axR.set_xlim(lo, hi); axR.set_ylim(lo, hi)
            axR.set_title(f"CALIBRATED (train-fit)  r={r:.3f}  MAE={mae_train_fit:.4f} "
                          f"({rows[-3]['pct_mae_reduction']:.0f}% lower)")
            axR.grid(True, alpha=0.3); axR.legend(loc="upper left", fontsize=8)
            fig.suptitle(f"{BENCH} {model_name} linear calibration, {len(test)} held-out models "
                         f"(fit on {len(train)} train)")
            fig.tight_layout()
            figpath = HERE / f"calib_{BENCH}_{model_name.lower()}_se{se_stop:g}.png"
            fig.savefig(figpath, dpi=130)
            plt.close(fig)
            print(f"  saved {figpath}", flush=True)

    outcsv = HERE / "pedagogy_linear_calibration.csv"
    fields = ["model", "se", "calibration_method", "r", "mae_raw", "mae_calibrated",
              "slope_a", "intercept_b", "pct_mae_reduction", "n_train", "n_test"]
    with open(outcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {outcsv}", flush=True)


if __name__ == "__main__":
    main()
