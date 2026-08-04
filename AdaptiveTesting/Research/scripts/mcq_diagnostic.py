#!/usr/bin/env python3
"""Held-out CAT diagnostic validation for an arbitrary MCQ benchmark directory.

Generalises AdaptiveTesting/Outputs/diagnostic_validation.py so it can point at any
directory of per-model loglikelihood CSVs (columns: question_id, model, result, ...).
Calibrates a 2PL bank (girth MML, torch-free) on a train split of models, then runs a
Fisher-information CAT that stops at SE < SE_STOP, and correlates each held-out model's
CAT-predicted score against its true full-benchmark accuracy.

Usage:
  PYTHONPATH=$REPO/eduLLM-Evals uv run python mcq_diagnostic.py \
      --mcq-dir <dir-containing-<bench>-subdir> --bench pedagogy \
      --se-stop 0.3 --out-dir <outdir> [--n-test 12] [--seed 7]

Writes: <outdir>/diag_<bench>_se<SE>.csv, .png, and appends a summary row to
        <outdir>/mcq_diagnostic_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "eduLLM-Evals"))
from tutor_cat.mcq_irt.calibrate import fit_2pl  # noqa: E402
from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark  # noqa: E402

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
        info = (a**2) * p * (1 - p)  # 2PL Fisher information
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcq-dir", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--se-stop", type=float, default=0.3)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--method", default="girth")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mat = load_benchmark(args.mcq_dir, args.bench).dropna(axis=0, how="any")
    models = list(mat.index)
    rng = np.random.default_rng(args.seed)
    order = [models[i] for i in rng.permutation(len(models))]
    n_test = min(args.n_test, len(models) // 3)
    test = sorted(order[:n_test])
    train = order[n_test:]
    actual_full = mat.mean(axis=1).to_dict()

    kept_mat, report = filter_items(mat.loc[train], benchmark=args.bench)
    cal = fit_2pl(kept_mat, method=args.method)
    items = cal.items
    a = np.asarray(cal.a, float)
    b = np.asarray(cal.b, float)
    print(
        f"{args.bench}: n_models={len(models)} train={len(train)} test={len(test)} "
        f"items_raw={report.n_items_raw} items_kept={len(items)} extreme_a={cal.n_extreme_a} "
        f"SE_stop={args.se_stop}",
        flush=True,
    )

    recs = []
    for m in test:
        resp = mat.loc[m, items].to_numpy(float)
        theta, se, n_adm = run_cat(resp, a, b, args.se_stop, max_items=len(items))
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
        f"  corr={r:.3f} MAE={mae:.3f} avg_items={avg_items:.1f}/{full_items} ({pct:.1f}%)",
        flush=True,
    )

    tag = f"{args.bench}_se{args.se_stop:g}"
    with open(out / f"diag_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "actual_full", "diagnostic_pred", "theta", "se", "n_items"])
        w.writerows(recs)

    summ = out / "mcq_diagnostic_summary.csv"
    new = not summ.exists()
    with open(summ, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(
                ["benchmark", "se_stop", "n_train", "n_test", "n_items_raw",
                 "n_items_kept", "corr", "mae", "avg_cat_items", "pct_items"]
            )
        w.writerow([args.bench, args.se_stop, len(train), len(test), report.n_items_raw,
                    len(items), round(r, 4), round(mae, 4), round(avg_items, 2), round(pct, 2)])

    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_label = {"girth": "2PL", "rasch": "1PL (Rasch)"}.get(args.method, args.method)
    fig, ax = plt.subplots(figsize=(7.2, 7))
    lo = min(acts.min(), preds.min()) - 0.03
    hi = max(acts.max(), preds.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect prediction (y=x)")
    ax.scatter(acts, preds, s=70, color="#1f77b4", edgecolor="black", zorder=3)
    for m, act, pred, *_ in recs:
        ax.annotate(m.split("/")[-1][:16], (act, pred), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(f"actual full {args.bench} accuracy (fraction correct, {full_items} items)")
    ax.set_ylabel(f"CAT-predicted accuracy (fraction correct, {model_label}, SE<={args.se_stop:g})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(
        f"{args.bench}: CAT-predicted vs actual full accuracy, {len(test)} held-out models\n"
        f"Pearson r={r:.3f}, MAE={mae:.3f}, {avg_items:.1f} of {full_items} items ({pct:.0f}%)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"diag_{tag}.png", dpi=130)
    print(f"saved {out / f'diag_{tag}.png'}", flush=True)


if __name__ == "__main__":
    main()
