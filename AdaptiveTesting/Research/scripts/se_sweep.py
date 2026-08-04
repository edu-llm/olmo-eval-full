#!/usr/bin/env python3
"""Sweep the CAT stopping standard-error (SE) requirement and record, per SE target,
the mean number of items administered and the Pearson correlation (+ MAE) of the
diagnostic-predicted vs. actual full-benchmark accuracy on held-out models.

Because the Fisher-information CAT administers items in the same order regardless of the
stop threshold (SE only decides WHERE to stop), we run one full CAT per held-out model to a
tight floor, store the SE and running prediction after each item, then read off (#items,
prediction) at every SE target. One fit, many thresholds -> a clean, self-consistent curve.

Sources:
  --source local  : fit a 1PL/2PL bank on our in-house matrix (train split), replay the
                    held-out models' responses.  --mcq-dir --bench --method {rasch,girth}
  --source atlas  : use the published ATLAS 3PL ARC bank + ATLAS's own test response matrix
                    (prompt-matched); item-index aligned.
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

NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()
MIN_ITEMS = 8
DEFAULT_SES = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.12]


def prob(theta, a, b, c):
    z = np.clip(a * (theta - b), -30, 30)
    return np.clip(c + (1 - c) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


def eap_se(resp, a, b, c):
    z = np.clip(a[None, :] * (NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(c[None, :] + (1 - c[None, :]) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * PRIORW
    w /= w.sum()
    m = float((NODES * w).sum())
    sd = float(np.sqrt(((NODES - m) ** 2 * w).sum()))
    return m, sd


def full_cat_traces(resp_all, a, b, c, predictor):
    """Run one full CAT (until all items or SE floor); return per-step (n_items, se, pred)."""
    n = len(a)
    used = np.zeros(n, bool)
    theta, order = 0.0, []
    steps = []
    for _ in range(n):
        p = prob(theta, a, b, c)
        info = (a**2) * ((1 - p) / p) * ((p - c) / (1 - c + 1e-9)) ** 2
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.asarray(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx], c[idx])
        steps.append((len(order), se, predictor(resp_all, order, theta, a, b, c)))
        if se <= 0.10 and len(order) >= MIN_ITEMS:
            break
    return steps


def pred_meanprob(resp_all, order, theta, a, b, c):
    return float(prob(theta, a, b, c).mean())


def pred_pirt(resp_all, order, theta, a, b, c):
    n = len(a)
    subset = set(order)
    avg_obs = float(resp_all[np.asarray(order)].mean()) if order else 0.0
    unobs = [i for i in range(n) if i not in subset]
    avg_pred = float(prob(theta, a[unobs], b[unobs], c[unobs]).mean()) if unobs else avg_obs
    w = len(order) / n
    return w * avg_obs + (1 - w) * avg_pred


def load_local(mcq_dir, bench, method, n_test, seed):
    from tutor_cat.mcq_irt.calibrate import fit_2pl
    from tutor_cat.mcq_irt.matrix import filter_items, load_benchmark

    mat = load_benchmark(mcq_dir, bench).dropna(axis=0, how="any")
    models = list(mat.index)
    rng = np.random.default_rng(seed)
    order = [models[i] for i in rng.permutation(len(models))]
    test = sorted(order[:n_test])
    train = order[n_test:]
    actual = mat.mean(axis=1).to_dict()
    kept, _ = filter_items(mat.loc[train], benchmark=bench)
    cal = fit_2pl(kept, method=method)
    items = cal.items
    a = np.asarray(cal.a, float)
    b = np.asarray(cal.b, float)
    c = np.zeros_like(a)
    resp = {m: mat.loc[m, items].to_numpy(float) for m in test}
    return items, a, b, c, resp, {m: float(actual[m]) for m in test}, pred_meanprob


def load_atlas(n_test, seed):
    ATLAS = REPO / "AdaptiveTesting/Inputs/ATLAS"
    bank = {}
    for row in csv.DictReader(open(ATLAS / "arc/irt_item_parameters_combined.csv")):
        k = int(str(row["X"]).lstrip("X"))
        aa = float(row["a1"]); d = float(row["d"]); g = float(row["g"])
        if aa > 0 and np.isfinite(aa):
            bank[k] = (aa, -d / aa, float(np.clip(g, 0, 0.999)))
    with open(ATLAS / "data/gaussian_sampled_arc_response_matrix_test.csv") as fh:
        r = csv.reader(fh); hdr = next(r); cols = hdr[1:]
        rows = {}
        for row in r:
            m = row[0].strip('"')
            rows[m] = {int(c_): int(v) for c_, v in zip(cols, row[1:]) if v.strip() in ("0", "1")}
    items = sorted(k for k in bank if all(k in rows[m] for m in rows))
    a = np.array([bank[k][0] for k in items])
    b = np.array([bank[k][1] for k in items])
    c = np.array([bank[k][2] for k in items])
    models = sorted(rows)
    rng = np.random.default_rng(seed)
    held = sorted(models[i] for i in rng.permutation(len(models))[:n_test])
    resp = {m: np.array([rows[m][k] for k in items], float) for m in held}
    actual = {m: float(resp[m].mean()) for m in held}
    return items, a, b, c, resp, actual, pred_pirt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["local", "atlas"], required=True)
    ap.add_argument("--mcq-dir")
    ap.add_argument("--bench", default="arc_challenge")
    ap.add_argument("--method", default="girth")
    ap.add_argument("--n-test", type=int, default=13,
                    help="number of held-out models sampled by --seed; the local ARC "
                         "banks use 13, the ATLAS own-response baseline uses 60")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--se-list", default=",".join(str(s) for s in DEFAULT_SES))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ses = [float(s) for s in args.se_list.split(",")]
    if args.source == "local":
        items, a, b, c, resp, actual, predictor = load_local(
            args.mcq_dir, args.bench, args.method, args.n_test, args.seed)
    else:
        items, a, b, c, resp, actual, predictor = load_atlas(args.n_test, args.seed)
    n_bank = len(items)
    print(f"[{args.tag}] bank={n_bank} items, held-out={len(resp)} models")

    traces = {m: full_cat_traces(resp[m], a, b, c, predictor) for m in resp}

    rows = []
    for se in ses:
        n_items, preds, acts = [], [], []
        for m, steps in traces.items():
            chosen = None
            for (ni, s, pr) in steps:
                if ni >= MIN_ITEMS and s <= se:
                    chosen = (ni, pr); break
            if chosen is None:
                chosen = (steps[-1][0], steps[-1][2])  # never reached: use full CAT end
            n_items.append(chosen[0]); preds.append(chosen[1]); acts.append(actual[m])
        preds = np.array(preds); acts = np.array(acts)
        r = float(np.corrcoef(preds, acts)[0, 1])
        mae = float(np.mean(np.abs(preds - acts)))
        rows.append({
            "source": args.source, "tag": args.tag, "se_target": se,
            "mean_items": round(float(np.mean(n_items)), 2),
            "median_items": float(np.median(n_items)),
            "pct_items": round(100 * float(np.mean(n_items)) / n_bank, 2),
            "corr": round(r, 4), "mae": round(mae, 4), "n_bank_items": n_bank,
        })
        print(f"  SE<={se:<4} items={rows[-1]['mean_items']:6}  ({rows[-1]['pct_items']:5}%)  "
              f"r={r:.3f}  MAE={mae:.3f}")

    csv_path = args.out_dir / f"se_sweep_{args.tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path}")

    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    se_x = [r["se_target"] for r in rows]
    corr_y = [r["corr"] for r in rows]
    items_y = [r["mean_items"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))
    ax1.plot(se_x, corr_y, "o-", color="#1f77b4", label="Pearson r")
    ax1.set_xlabel("SE stopping target (smaller = stricter)")
    ax1.set_ylabel("Pearson r (pred vs actual)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.invert_xaxis()
    ax1.set_ylim(min(corr_y) - 0.05, 1.01)
    ax2 = ax1.twinx()
    ax2.plot(se_x, items_y, "s--", color="#d62728", label="mean # items")
    ax2.set_ylabel("mean # items administered", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title(f"SE stopping target vs correlation and test length ({args.tag})\n"
                  f"(bank {n_bank} items, {len(resp)} held-out models)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"se_sweep_{args.tag}.png", dpi=140)
    print(f"wrote {args.out_dir / f'se_sweep_{args.tag}.png'}")


if __name__ == "__main__":
    main()
