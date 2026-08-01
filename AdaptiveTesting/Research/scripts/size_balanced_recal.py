#!/usr/bin/env python3
"""Size-skew-controlled ARC 3PL recalibration (GAPS #9).

Isolates *range-restriction* from *size-skew* in the 0.5-7B ATLAS recalibration.
The published bank (thousands of models, wide range) hits r=0.83; refitting on the
0.5-7B pool (95% 7B, thin) drops to r=0.59. That drop conflates (a) restricting the
parameter range and (b) a size-skewed pool. Here we hold N fixed and vary ONLY the
size distribution: a size-balanced pool vs a size-skewed (random, ~95% 7B) pool,
both recalibrated with the identical ATLAS chunked-3PL + mean-sigma pipeline and run
through the identical held-out Fisher-info CAT / p-IRT diagnostic on the same 60
held-out models.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = "/Users/arhant/Documents/EDLM/olmo-eval-full"
ATLAS = os.path.join(REPO, "AdaptiveTesting/Inputs/ATLAS")
EXP = os.path.join(REPO, "AdaptiveTesting/Experiments/atlas_recalibrate_0p5_7b")
SCRIPTS = os.path.join(REPO, "AdaptiveTesting/Research/scripts")
OUTDIR = os.path.join(REPO, "AdaptiveTesting/Research/01_MCQ_ATLAS/data/size_balanced_recal")
WORK = os.path.join(OUTDIR, "_work")
CACHE = os.path.join(WORK, "cache")
ROSTER = os.path.join(EXP, "calibration/cal_models_0p5_7b.csv")
TRAIN = os.path.join(
    ATLAS, "data/gaussian_sampled_arc_response_matrix_train_with_scores_0p5_7b.csv"
)
RFIT = os.path.join(SCRIPTS, "recal_fit_link.r")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUTDIR, ".mplcache"))

SE_TARGETS = [0.2, 0.3]
SEEDS = [7, 8, 9, 10, 11]

# --- import the existing ATLAS diagnostic (identical CAT/EAP/p-IRT conventions) ---
_spec = importlib.util.spec_from_file_location(
    "atlas_diag", os.path.join(EXP, "atlas_diagnostic_validation.py")
)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


def band_of(p: float, edges: list[float]) -> int:
    """Return band index for size p given band left-edges (last band open)."""
    for i in range(len(edges) - 1):
        if edges[i] <= p < edges[i + 1]:
            return i
    return len(edges) - 1 if p >= edges[-1] else -1


def load_roster() -> pd.DataFrame:
    df = pd.read_csv(ROSTER)
    df = df.rename(columns={df.columns[0]: "model", df.columns[1]: "params_b"})
    return df


def size_histogram(df: pd.DataFrame) -> dict:
    fine_edges = [0.5, 1, 2, 3, 5, 7]
    labels = ["[0.5,1)", "[1,2)", "[2,3)", "[3,5)", "[5,7]"]
    counts = {lab: 0 for lab in labels}
    for p in df["params_b"]:
        if 0.5 <= p < 1:
            counts["[0.5,1)"] += 1
        elif 1 <= p < 2:
            counts["[1,2)"] += 1
        elif 2 <= p < 3:
            counts["[2,3)"] += 1
        elif 3 <= p < 5:
            counts["[3,5)"] += 1
        else:
            counts["[5,7]"] += 1
    return counts


def build_balanced(df: pd.DataFrame, edges: list[float], labels: list[str], seed: int):
    """Equal-N-per-band, N capped by the scarcest *populated* band."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["band"] = [band_of(p, edges) for p in df["params_b"]]
    groups = {b: df[df["band"] == b]["model"].tolist() for b in range(len(edges))}
    populated = {b: m for b, m in groups.items() if len(m) > 0}
    cap = min(len(m) for m in populated.values())
    picked = []
    for b, models in populated.items():
        idx = rng.choice(len(models), size=cap, replace=False)
        picked.extend([models[i] for i in idx])
    return sorted(picked), cap, {labels[b]: len(groups[b]) for b in range(len(edges))}


def build_skewed(df: pd.DataFrame, n: int, seed: int):
    rng = np.random.default_rng(1000 + seed)
    models = df["model"].tolist()
    idx = rng.choice(len(models), size=n, replace=False)
    return sorted([models[i] for i in idx])


def recalibrate(models: list[str], train: pd.DataFrame) -> str:
    """Write subset train matrix, run R 3PL fit+link, return combined-params path."""
    sig = hashlib.md5(("|".join(models)).encode()).hexdigest()[:12]
    os.makedirs(CACHE, exist_ok=True)
    out_params = os.path.join(CACHE, f"params_{sig}.csv")
    if os.path.isfile(out_params) and os.path.getsize(out_params) > 0:
        return out_params
    sub = train[train.iloc[:, 0].isin(models)].copy()
    subset_csv = os.path.join(CACHE, f"train_{sig}.csv")
    sub.to_csv(subset_csv, index=False)
    log = os.path.join(CACHE, f"fit_{sig}.log")
    with open(log, "w") as lf:
        r = subprocess.run(
            ["Rscript", RFIT, f"--data_file={subset_csv}", f"--out={out_params}"],
            stdout=lf, stderr=subprocess.STDOUT,
        )
    if r.returncode != 0 or not os.path.isfile(out_params):
        raise RuntimeError(f"R fit failed for {sig}; see {log}")
    return out_params


# --- cache held-out responses once (model -> {qid: 0/1}) ---
_RESP_CACHE: dict[str, dict[str, int]] = {}


def all_mcq_responses() -> dict[str, dict[str, int]]:
    import csv
    import glob

    if _RESP_CACHE:
        return _RESP_CACHE
    for path in sorted(glob.glob(os.path.join(diag.MCQ, "*.csv"))):
        slug = os.path.basename(path)[:-4]
        mid = slug.replace("__", "/")
        by_q: dict[str, int] = {}
        with open(path) as fh:
            for row in csv.DictReader(fh):
                by_q[row["question_id"]] = (
                    1 if row["result"].strip().lower() == "correct" else 0
                )
        _RESP_CACHE[mid] = by_q
    return _RESP_CACHE


def evaluate(params_path: str, idx_map, atlas_models, se_stop: float):
    """Run the identical held-out CAT + p-IRT diagnostic; return metrics."""
    diag.PARAMS = params_path
    diag.SE_STOP = se_stop
    qids, a, b, c = diag.load_atlas_item_bank(idx_map)
    if len(qids) < 8:
        return dict(r=float("nan"), mae=float("nan"), items=float("nan"),
                    n_items_bank=len(qids), n_held=0)
    resp_by_model = all_mcq_responses()
    held = sorted(
        m for m in resp_by_model
        if m not in atlas_models and all(q in resp_by_model[m] for q in qids)
    )
    acts, preds, nitems = [], [], []
    for mid in held:
        resp = np.asarray([resp_by_model[mid][q] for q in qids], dtype=float)
        theta, se, order = diag.run_cat(resp, a, b, c)
        pred = diag.pirt_accuracy(resp, order, theta, a, b, c)
        acts.append(float(resp.mean()))
        preds.append(pred)
        nitems.append(len(order))
    acts = np.asarray(acts)
    preds = np.asarray(preds)
    r = float(np.corrcoef(preds, acts)[0, 1])
    mae = float(np.mean(np.abs(preds - acts)))
    return dict(r=r, mae=mae, items=float(np.mean(nitems)),
                n_items_bank=len(qids), n_held=len(held))


def main() -> None:
    df = load_roster()
    train = pd.read_csv(TRAIN)
    idx_map = diag.load_atlas_idx_map()
    atlas_models = diag.load_atlas_models()

    hist = size_histogram(df)
    print(f"Roster: {len(df)} models. Fine-band histogram:")
    for k, v in hist.items():
        print(f"  {k:8s} {v:5d}  ({100*v/len(df):.1f}%)")

    # Balanced pool schemes (coarse enough to keep a usable N). "3band" spreads
    # across the parameter *range*; "2band" only de-skews the 7B dominance but at
    # a larger matched N, to rule out that N=72 was simply too small.
    schemes = [
        ("3band", [0.5, 3, 7], ["<3B", "3-<7B", "7B"]),
        ("2band", [0.5, 7], ["<7B", "7B"]),
    ]

    rows = []
    for seed in SEEDS:
      for scheme, bal_edges, bal_labels in schemes:
        bal_models, cap, band_counts = build_balanced(df, bal_edges, bal_labels, seed)
        n = len(bal_models)
        skew_models = build_skewed(df, n, seed)
        print(f"\nseed {seed} [{scheme}]: balanced N={n} (cap {cap}/band, "
              f"bands {band_counts}); skewed N={n}")
        for base, models in [("balanced", bal_models), ("skewed", skew_models)]:
            tag = f"{base}_{scheme}"
            sizes = df.set_index("model").loc[models, "params_b"]
            pct7 = 100 * float((sizes >= 7).mean())
            print(f"  recalibrating {tag:9s} (N={len(models)}, {pct7:.0f}% 7B)...",
                  flush=True)
            params_path = recalibrate(models, train)
            for se in SE_TARGETS:
                m = evaluate(params_path, idx_map, atlas_models, se)
                rows.append(dict(
                    pool=tag, seed=seed, se_stop=se, n_cal=len(models),
                    pct_7b=round(pct7, 1), bank_items=m["n_items_bank"],
                    n_held=m["n_held"], r=round(m["r"], 4),
                    mae=round(m["mae"], 4), mean_items=round(m["items"], 2),
                ))
                print(f"    SE{se}: r={m['r']:.3f} MAE={m['mae']:.3f} "
                      f"items={m['items']:.1f} bank={m['n_items_bank']}", flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    csv_path = os.path.join(OUTDIR, "results.csv")
    res.to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path}")

    summ = (res.groupby(["pool", "se_stop"])
            .agg(r_mean=("r", "mean"), r_sd=("r", "std"),
                 mae_mean=("mae", "mean"), items_mean=("mean_items", "mean"),
                 n_cal=("n_cal", "mean"), seeds=("seed", "count"))
            .reset_index())
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
