"""Gate the refit: reproduce the published filter_items counts before trusting anything.

`pl_1_2_3_comparison` reports 318 / 831 / 969 kept items for pedagogy / piqa / socialiqa after
`filter_items` on its train slice. If we cannot recover those three numbers we do not
understand the pipeline well enough to ship parameters from it, so this runs first.

`load_benchmark`, `_point_biserial` and `filter_items` are transcribed verbatim from
`tutor_cat/mcq_irt/matrix.py` rather than imported, because that package lives in a separate
worktree with its own dependencies and the point here is to reproduce it exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BANKS = ("pedagogy", "piqa", "socialiqa")
EXPECTED_KEPT = {"pedagogy": 318, "piqa": 831, "socialiqa": 969}
EXPECTED_RAW = {"pedagogy": 920, "piqa": 1838, "socialiqa": 1954}


def load_benchmark(mcq_dir, benchmark):
    bdir = Path(mcq_dir) / benchmark
    files = sorted(bdir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs in {bdir}")
    series = {}
    for f in files:
        df = pd.read_csv(f, usecols=["question_id", "model", "result"])
        if df.empty:
            continue
        model = str(df["model"].iloc[0])
        correct = (df["result"] == "correct").astype("int8")
        s = pd.Series(correct.values, index=df["question_id"].values)
        series[model] = s[~s.index.duplicated(keep="first")]
    mat = pd.DataFrame(series).T
    mat.index.name = "model"
    mat.columns.name = "item"
    return mat.sort_index()


def _point_biserial(X):
    total = X.sum(axis=1, keepdims=True)
    rest = total - X
    xc = X - X.mean(axis=0, keepdims=True)
    rc = rest - rest.mean(axis=0, keepdims=True)
    cov = (xc * rc).sum(axis=0)
    sx = np.sqrt((xc**2).sum(axis=0))
    sr = np.sqrt((rc**2).sum(axis=0))
    denom = sx * sr
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, cov / denom, np.nan)


def filter_items(mat, min_point_biserial=0.05):
    mat = mat.dropna(axis=0, how="any")
    X = mat.to_numpy(dtype=float)
    items = list(mat.columns)
    p = X.mean(axis=0)
    pbis = _point_biserial(X)
    kept, counts = [], {"all_pass": 0, "all_fail": 0, "low_point_biserial": 0}
    for j, it in enumerate(items):
        if p[j] >= 1.0:
            counts["all_pass"] += 1
        elif p[j] <= 0.0:
            counts["all_fail"] += 1
        elif not np.isfinite(pbis[j]) or pbis[j] < min_point_biserial:
            counts["low_point_biserial"] += 1
        else:
            kept.append(it)
    return mat[kept], counts


def split_40_12(models):
    """The study's split, byte-for-byte from scripts/mcq_diagnostic.py."""
    order = np.random.default_rng(7).permutation(len(models))
    n_test = min(12, len(models) // 3)
    return [models[i] for i in order[n_test:]], sorted(models[i] for i in order[:n_test])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--matrices",
        required=True,
        type=Path,
        help="directory holding pedagogy/ piqa/ socialiqa/ subdirs of per-model response CSVs",
    )
    args = ap.parse_args()

    mats = {b: load_benchmark(args.matrices, b) for b in BANKS}
    common = set.intersection(*(set(m.index) for m in mats.values()))

    print(f"models common to all three banks: {len(common)}\n")
    failures = []
    for b, m in mats.items():
        if m.shape[1] != EXPECTED_RAW[b]:
            failures.append(f"{b}: {m.shape[1]} raw items, expected {EXPECTED_RAW[b]}")
        # The published split was over the models common to all three banks, which is why
        # pedagogy's number comes from 52 of its 78 rather than the full roster.
        sub = m.loc[sorted(common)]
        train, _test = split_40_12(list(sub.index))
        kept, counts = filter_items(sub.loc[train])
        ok = kept.shape[1] == EXPECTED_KEPT[b]
        print(
            f"  {b:10} common-52 train-{len(train)}: kept={kept.shape[1]:5} "
            f"expected={EXPECTED_KEPT[b]:5}  {'OK' if ok else 'MISMATCH'}   {counts}"
        )
        if not ok:
            failures.append(f"{b}: kept {kept.shape[1]}, expected {EXPECTED_KEPT[b]}")

    if failures:
        raise SystemExit("\nGATE FAILED:\n  " + "\n  ".join(failures))
    print("\nGate passed: the transcribed pipeline reproduces the published counts.")


if __name__ == "__main__":
    main()
