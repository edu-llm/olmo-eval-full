#!/usr/bin/env python3
"""Held-out CAT + p-IRT using ATLAS-linked 3PL params on OpenLM GPQA.

Calibrate on train (90%) via chunked mirt 3PL; this script scores the held-out
10% with Fisher-information CAT and scatters p-IRT diagnostic vs full accuracy.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
DATA = EXP / "data"
CALIB = Path(os.environ.get("CALIB_DIR", str(EXP / "calibration")))
OUT_RES = EXP / "results"
OUT_FIG = EXP / "figures"

SE_STOP = float(os.environ.get("SE_STOP", "0.3"))
MIN_ITEMS = 8
MAX_ITEMS = 200
MODEL_TAG = os.environ.get("MODEL_TAG", "atlas3pl")
TAG = os.environ.get("OUT_TAG", "") or f"openlm_gpqa_{MODEL_TAG}_se{SE_STOP:g}"

NODES = np.linspace(-4.0, 4.0, 81)
PRIORW = np.exp(-0.5 * NODES**2)
PRIORW /= PRIORW.sum()


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # First column is model id (empty header in ATLAS CSV).
    model_col = df.columns[0]
    df = df.rename(columns={model_col: "model"}).set_index("model")
    df.columns = [int(c) for c in df.columns]
    return df.astype(float)


def load_item_bank(mat_cols: list[int]) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Load linked mirt 3PL params; return (atlas_idx, a, b, c) aligned to mat cols."""
    params_path = CALIB / "irt_item_parameters_combined.csv"
    by_idx: dict[int, tuple[float, float, float]] = {}
    with params_path.open() as fh:
        for row in csv.DictReader(fh):
            key = str(row.get("X") or row.get("") or "").lstrip("X")
            if not key.isdigit():
                # row names may be bare integers
                continue
            idx = int(key)
            a = float(row["a1"])
            d = float(row["d"])
            g = float(row.get("g", 0.0) or 0.0)
            if a <= 0 or not np.isfinite(a):
                continue
            by_idx[idx] = (a, -d / a, float(np.clip(g, 0.0, 0.999)))

    # Fallback: if X column missing, try first column name from pandas
    if not by_idx:
        pdf = pd.read_csv(params_path)
        name_col = pdf.columns[0]
        for _, row in pdf.iterrows():
            key = str(row[name_col]).lstrip("X")
            if not key.replace(".", "", 1).isdigit():
                # allow "1" etc
                try:
                    idx = int(float(key))
                except ValueError:
                    continue
            else:
                idx = int(float(key))
            a = float(row["a1"])
            d = float(row["d"])
            g = float(row["g"]) if "g" in pdf.columns else 0.0
            if a <= 0 or not np.isfinite(a):
                continue
            by_idx[idx] = (a, -d / a, float(np.clip(g, 0.0, 0.999)))

    idxs, a_l, b_l, c_l = [], [], [], []
    missing = 0
    for idx in mat_cols:
        if idx not in by_idx:
            missing += 1
            continue
        a, b, c = by_idx[idx]
        idxs.append(idx)
        a_l.append(a)
        b_l.append(b)
        c_l.append(c)
    if missing:
        print(f"warning: {missing} matrix items missing from linked params", flush=True)
    if not idxs:
        raise SystemExit(f"no overlapping items between matrix and {params_path}")
    return idxs, np.asarray(a_l), np.asarray(b_l), np.asarray(c_l)


def prob(theta: float, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    z = np.clip(a * (theta - b), -30, 30)
    return np.clip(c + (1 - c) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


def eap_se(resp: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    z = np.clip(a[None, :] * (NODES[:, None] - b[None, :]), -30, 30)
    p = np.clip(c[None, :] + (1 - c[None, :]) / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
    ll = (resp[None, :] * np.log(p) + (1 - resp)[None, :] * np.log(1 - p)).sum(axis=1)
    w = np.exp(ll - ll.max()) * PRIORW
    w /= w.sum()
    m = float((NODES * w).sum())
    sd = float(np.sqrt(((NODES - m) ** 2 * w).sum()))
    return m, sd


def run_cat(
    resp_all: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[float, float, list[int]]:
    n = len(a)
    used = np.zeros(n, dtype=bool)
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
        if len(order) >= MIN_ITEMS and se <= SE_STOP:
            break
    return theta, se, order


def pirt_accuracy(
    resp_all: np.ndarray,
    order: list[int],
    theta: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    n = len(a)
    subset = set(order)
    avg_obs = float(resp_all[np.asarray(order)].mean()) if order else 0.0
    unobs = [i for i in range(n) if i not in subset]
    avg_pred = float(prob(theta, a[unobs], b[unobs], c[unobs]).mean()) if unobs else avg_obs
    w_obs = len(order) / n
    return w_obs * avg_obs + (1 - w_obs) * avg_pred


def main() -> None:
    test_path = DATA / "gpqa_response_matrix_test.csv"
    if not test_path.is_file():
        raise SystemExit(f"missing {test_path}")
    if not (CALIB / "irt_item_parameters_combined.csv").is_file():
        raise SystemExit("missing linked params — run run_calibration.sh first")

    test = load_matrix(test_path)
    idxs, a, b, c = load_item_bank(list(test.columns))
    print(
        f"OpenLM GPQA held-out diagnostic | bank={len(idxs)} items ({MODEL_TAG})  "
        f"params={CALIB}  SE_stop={SE_STOP}  n_test={len(test)}",
        flush=True,
    )

    recs = []
    for mid in test.index:
        resp = test.loc[mid, idxs].to_numpy(float)
        actual = float(resp.mean())
        theta, se, order = run_cat(resp, a, b, c)
        pred = pirt_accuracy(resp, order, theta, a, b, c)
        mean_p = float(prob(theta, a, b, c).mean())
        recs.append((mid, actual, pred, mean_p, theta, se, len(order)))
        print(
            f"  {str(mid)[:42]:42s} actual={actual:.3f} pirt={pred:.3f} "
            f"meanP={mean_p:.3f} θ={theta:+.2f} SE={se:.3f} items={len(order)}",
            flush=True,
        )

    acts = np.asarray([r[1] for r in recs])
    preds = np.asarray([r[2] for r in recs])
    r = float(np.corrcoef(preds, acts)[0, 1])
    mae = float(np.mean(np.abs(preds - acts)))
    rmse = float(np.sqrt(np.mean((preds - acts) ** 2)))
    avg_items = float(np.mean([r[6] for r in recs]))
    print(
        f"\np-IRT vs actual: r={r:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  "
        f"avg_items={avg_items:.1f}",
        flush=True,
    )

    OUT_RES.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_RES / f"{TAG}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["model", "actual_full", "pirt_pred", "mean_p_pred", "theta", "se", "n_items"]
        )
        w.writerows(recs)

    os.environ.setdefault("MPLCONFIGDIR", str(EXP / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    lo = min(acts.min(), preds.min()) - 0.03
    hi = max(acts.max(), preds.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="perfect (y=x)")
    ax.scatter(acts, preds, s=36, color="#1f77b4", edgecolor="black", linewidths=0.3, zorder=3)
    ax.set_xlabel("actual full GPQA accuracy (all items)")
    ax.set_ylabel(f"ATLAS p-IRT diagnostic ({MODEL_TAG} k-subset CAT, SE≤{SE_STOP:g})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"OpenLM GPQA — ATLAS {MODEL_TAG} k-subset calib (90% train) → {len(recs)} held-out (10%)\n"
        f"Pearson r={r:.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}  avg {avg_items:.0f} items/CAT"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    png = OUT_FIG / f"{TAG}.png"
    fig.savefig(png, dpi=140)
    print(f"saved {csv_path}\nsaved {png}", flush=True)


if __name__ == "__main__":
    main()
