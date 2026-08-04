#!/usr/bin/env python3
"""Pedagogy 1PL (Rasch) held-out CAT diagnostic, reusing the exact same fitter and
CAT/EAP conventions as ``scripts/mcq_diagnostic.py`` (only difference: Rasch a==1).

Same response matrix, same 40-train/12-test split (seed 7), same item-keep filter,
same SE targets. Writes 1PL diag CSV+PNG per SE and a 1PL-vs-2PL side-by-side table.
"""
from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # .../olmo-eval-full
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplcache"))
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
        info = (a**2) * p * (1 - p)  # Fisher information (a==1 -> 1PL)
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.array(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx])
        if len(order) >= min_items and se <= se_stop:
            break
    return theta, se, len(order)


def main():
    # Same load + split + filter as mcq_diagnostic.py (order-independent of SE)
    mat = load_benchmark(MCQ_DIR, BENCH).dropna(axis=0, how="any")
    models = list(mat.index)
    rng = np.random.default_rng(SEED)
    order = [models[i] for i in rng.permutation(len(models))]
    n_test = min(N_TEST, len(models) // 3)
    test = sorted(order[:n_test])
    train = order[n_test:]
    actual_full = mat.mean(axis=1).to_dict()

    kept_mat, report = filter_items(mat.loc[train], benchmark=BENCH)
    cal = fit_2pl(kept_mat, method="rasch")  # 1PL: discrimination fixed at 1
    items = cal.items
    a = np.asarray(cal.a, float)
    b = np.asarray(cal.b, float)
    assert np.allclose(a, 1.0), "Rasch fit must have a==1"
    print(
        f"{BENCH} 1PL: n_models={len(models)} train={len(train)} test={len(test)} "
        f"items_raw={report.n_items_raw} items_kept={len(items)} method={cal.method}",
        flush=True,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_rows = []
    for se_stop in SE_TARGETS:
        recs = []
        for m in test:
            resp = mat.loc[m, items].to_numpy(float)
            theta, se, n_adm = run_cat(resp, a, b, se_stop, max_items=len(items))
            pred = float(prob(theta, a, b).mean())
            recs.append((m, float(actual_full[m]), pred, theta, se, n_adm))

        acts = np.array([r[1] for r in recs])
        preds = np.array([r[2] for r in recs])
        r = float(np.corrcoef(preds, acts)[0, 1])
        mae = float(np.mean(np.abs(preds - acts)))
        avg_items = float(np.mean([x[5] for x in recs]))
        full_items = len(items)
        pct = 100.0 * avg_items / full_items
        print(
            f"  SE<{se_stop:g}: corr={r:.3f} MAE={mae:.3f} "
            f"avg_items={avg_items:.1f}/{full_items} ({pct:.1f}%)",
            flush=True,
        )

        with open(HERE / f"diag_{BENCH}_1pl_se{se_stop:g}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "actual_full", "diagnostic_pred", "theta", "se", "n_items"])
            w.writerows(recs)

        fig, ax = plt.subplots(figsize=(7.2, 7))
        lo = min(acts.min(), preds.min()) - 0.03
        hi = max(acts.max(), preds.max()) + 0.03
        ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect prediction (y=x)")
        ax.scatter(acts, preds, s=70, color="#1f77b4", edgecolor="black", zorder=3)
        for m, act, pred, *_ in recs:
            ax.annotate(m.split("/")[-1][:16], (act, pred), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel(f"actual full {BENCH} accuracy (fraction correct, {full_items} items)")
        ax.set_ylabel(f"CAT-predicted accuracy (fraction correct, 1PL/Rasch, SE<={se_stop:g})")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(
            f"{BENCH} 1PL: CAT-predicted vs actual full accuracy, {len(test)} held-out models\n"
            f"Pearson r={r:.3f}, MAE={mae:.3f}, {avg_items:.1f} of {full_items} items ({pct:.0f}%)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(HERE / f"diag_{BENCH}_1pl_se{se_stop:g}.png", dpi=130)
        plt.close(fig)
        print(f"saved {HERE / f'diag_{BENCH}_1pl_se{se_stop:g}.png'}", flush=True)

        summary_rows.append({
            "model": "1PL", "se_stop": se_stop, "corr": round(r, 4), "mae": round(mae, 4),
            "avg_cat_items": round(avg_items, 2), "pct_items": round(pct, 2),
            "n_items_kept": len(items),
        })

    # 2PL rows from the existing pedagogy feasibility summary (source of truth).
    twopl = {}
    with open(HERE.parent / "mcq_diagnostic_summary.csv") as fh:
        for row in csv.DictReader(fh):
            if row["benchmark"] == BENCH:
                twopl[float(row["se_stop"])] = row

    combined = HERE / f"{BENCH}_1pl_vs_2pl_summary.csv"
    with open(combined, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "se_stop", "r", "mae", "avg_cat_items", "pct_items", "n_items_kept"])
        for se_stop in SE_TARGETS:
            two = twopl.get(se_stop)
            if two:
                w.writerow(["2PL", se_stop, two["corr"], two["mae"],
                            two["avg_cat_items"], two["pct_items"], two["n_items_kept"]])
            one = next(s for s in summary_rows if s["se_stop"] == se_stop)
            w.writerow(["1PL", se_stop, one["corr"], one["mae"],
                        one["avg_cat_items"], one["pct_items"], one["n_items_kept"]])
    print(f"wrote {combined}", flush=True)


if __name__ == "__main__":
    main()
