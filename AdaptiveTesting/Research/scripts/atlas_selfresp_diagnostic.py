#!/usr/bin/env python3
"""ATLAS transfer validation using ATLAS's OWN held-out responses.

Difference vs atlas_transfer_published/atlas_diagnostic_validation.py: the held-out
responses are ATLAS's own ARC test response matrix (gaussian_sampled_arc_response_matrix_
test.csv), aligned to the published 3PL bank DIRECTLY by ATLAS item index (bank "X{k}" <->
matrix column "{k}"). This removes the prompt / few-shot mismatch of replaying our own
0-shot loglikelihood responses through a bank ATLAS fit on 25-shot leaderboard labels.

For each sampled held-out model: run the Fisher-information 3PL CAT to SE<=SE_STOP, then
ATLAS p-IRT reconstruct full accuracy and correlate against the model's actual full-bank
accuracy.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ATLAS = REPO / "AdaptiveTesting/Inputs/ATLAS"
PARAMS = ATLAS / "arc/irt_item_parameters_combined.csv"
TEST = ATLAS / "data/gaussian_sampled_arc_response_matrix_test.csv"

NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()
MIN_ITEMS = 8
MAX_ITEMS = 200


def load_bank():
    bank = {}
    with open(PARAMS) as fh:
        for row in csv.DictReader(fh):
            k = int(str(row["X"]).lstrip("X"))
            a = float(row["a1"]); d = float(row["d"]); g = float(row["g"])
            if a <= 0 or not np.isfinite(a):
                continue
            bank[k] = (a, -d / a, float(np.clip(g, 0.0, 0.999)))
    return bank


def load_test_matrix():
    with open(TEST) as fh:
        r = csv.reader(fh)
        hdr = next(r)
        cols = hdr[1:]
        rows = {}
        for row in r:
            m = row[0].strip('"')
            vals = {}
            for c, v in zip(cols, row[1:]):
                v = v.strip()
                if v in ("0", "1"):
                    vals[int(c)] = int(v)
            rows[m] = vals
    return cols, rows


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


def run_cat(resp_all, a, b, c, se_stop):
    n = len(a)
    used = np.zeros(n, bool)
    theta, se, order = 0.0, 1.0, []
    for _ in range(min(MAX_ITEMS, n)):
        p = prob(theta, a, b, c)
        info = (a**2) * ((1 - p) / p) * ((p - c) / (1 - c + 1e-9)) ** 2
        info[used] = -1.0
        j = int(np.argmax(info))
        used[j] = True
        order.append(j)
        idx = np.asarray(order)
        theta, se = eap_se(resp_all[idx], a[idx], b[idx], c[idx])
        if len(order) >= MIN_ITEMS and se <= se_stop:
            break
    return theta, se, order


def pirt_accuracy(resp_all, order, theta, a, b, c):
    n = len(a)
    subset = set(order)
    avg_obs = float(resp_all[np.asarray(order)].mean()) if order else 0.0
    unobs = [i for i in range(n) if i not in subset]
    avg_pred = float(prob(theta, a[unobs], b[unobs], c[unobs]).mean()) if unobs else avg_obs
    w_obs = len(order) / n
    return w_obs * avg_obs + (1 - w_obs) * avg_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--se-stop", type=float, default=0.3)
    ap.add_argument("--n-held", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)

    bank = load_bank()
    cols, rows = load_test_matrix()
    items = sorted(k for k in bank if all(k in rows[m] for m in rows))
    a = np.array([bank[k][0] for k in items])
    b = np.array([bank[k][1] for k in items])
    c = np.array([bank[k][2] for k in items])

    models = sorted(rows)
    rng = np.random.default_rng(args.seed)
    held = sorted(models[i] for i in rng.permutation(len(models))[: args.n_held])
    print(f"bank items aligned to test matrix: {len(items)} | test models: {len(models)} "
          f"| held-out sampled: {len(held)} | SE_stop={args.se_stop}")

    recs = []
    for m in held:
        resp = np.array([rows[m][k] for k in items], float)
        actual = float(resp.mean())
        theta, se, order = run_cat(resp, a, b, c, args.se_stop)
        pred = pirt_accuracy(resp, order, theta, a, b, c)
        mean_p = float(prob(theta, a, b, c).mean())
        recs.append((m, actual, pred, mean_p, theta, se, len(order)))

    acts = np.array([r[1] for r in recs])
    preds = np.array([r[2] for r in recs])
    r = float(np.corrcoef(preds, acts)[0, 1])
    mae = float(np.mean(np.abs(preds - acts)))
    rmse = float(np.sqrt(np.mean((preds - acts) ** 2)))
    avg_items = float(np.mean([x[6] for x in recs]))
    print(f"p-IRT vs actual (ATLAS own responses): r={r:.3f} MAE={mae:.3f} RMSE={rmse:.3f} "
          f"avg_items={avg_items:.1f}")

    tag = f"atlas_selfresp_arc_se{args.se_stop:g}"
    with open(args.out_dir / f"{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "actual_full", "pirt_pred", "mean_p_pred", "theta", "se", "n_items"])
        w.writerows(recs)

    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    lo = min(acts.min(), preds.min()) - 0.03
    hi = max(acts.max(), preds.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect (y=x)")
    ax.scatter(acts, preds, s=60, color="#2ca02c", edgecolor="black", zorder=3)
    ax.set_xlabel("actual accuracy on ATLAS ARC bank (ATLAS's own 25-shot responses)")
    ax.set_ylabel(f"ATLAS p-IRT diagnostic (3PL CAT, SE<={args.se_stop:g})")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"ATLAS 3PL bank -> {len(held)} ATLAS held-out models (own responses)\n"
                 f"Pearson r={r:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  avg {avg_items:.0f} items")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / f"{tag}.png", dpi=140)
    print(f"saved {args.out_dir / f'{tag}.csv'} and figure")


if __name__ == "__main__":
    main()
