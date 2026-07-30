#!/usr/bin/env python3
"""Build ATLAS-style GPQA response matrices from OpenLM CSVs (90/10 model split)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[2]
OPENLM_GPQA = REPO / "AdaptiveTesting/Inputs/OpenLM/gpqa"


def load_long() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(OPENLM_GPQA.glob("*.csv")):
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                item = f"{r['subtask']}|{r['question_id']}"
                rows.append(
                    {
                        "model": r["model"],
                        "item": item,
                        "correct": 1 if r["result"].strip().lower() == "correct" else 0,
                    }
                )
    return pd.DataFrame(rows)


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    mat = df.pivot_table(index="model", columns="item", values="correct", aggfunc="max")
    # Require complete rows (all models in OpenLM GPQA are complete, but be safe).
    mat = mat.dropna(axis=0, how="any")
    # Drop constant items / models (same as ATLAS R cleaning).
    nunique = mat.nunique(axis=0)
    mat = mat.loc[:, nunique > 1]
    row_nunique = mat.nunique(axis=1)
    mat = mat.loc[row_nunique > 1]
    return mat.astype(int)


def write_atlas_csv(mat: pd.DataFrame, path: Path, item_ids: list[str]) -> None:
    """ATLAS format: empty first header cell, then 1..N item columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(range(1, len(item_ids) + 1))
    out = mat.loc[:, item_ids].copy()
    out.columns = cols
    # Leading empty column name for model id
    out.insert(0, "", out.index)
    out.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--out-dir", type=Path, default=EXP / "data")
    args = p.parse_args()

    print(f"Loading OpenLM GPQA from {OPENLM_GPQA} …", flush=True)
    long = load_long()
    mat = to_matrix(long)
    item_ids = sorted(mat.columns)
    models = list(mat.index)
    print(f"Matrix: {len(models)} models × {len(item_ids)} items", flush=True)

    rng = np.random.default_rng(args.seed)
    order = [models[i] for i in rng.permutation(len(models))]
    n_test = max(1, int(round(len(order) * args.test_frac)))
    test_models = sorted(order[:n_test])
    train_models = sorted(order[n_test:])
    print(f"Split: train={len(train_models)} test={len(test_models)} "
          f"(seed={args.seed}, test_frac={args.test_frac})", flush=True)

    train = mat.loc[train_models, item_ids]
    test = mat.loc[test_models, item_ids]
    # Re-drop constants on train only (calibration set).
    train = train.loc[:, train.nunique() > 1]
    train = train.loc[train.nunique(axis=1) > 1]
    kept_items = list(train.columns)
    test = test.loc[:, kept_items]
    print(f"After train cleaning: {train.shape[0]} models × {len(kept_items)} items", flush=True)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_atlas_csv(train, out / "gpqa_response_matrix_train.csv", kept_items)
    write_atlas_csv(test, out / "gpqa_response_matrix_test.csv", kept_items)

    # Map ATLAS column index → item id
    with (out / "item_id_map.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["atlas_idx", "item_id", "subtask", "question_id"])
        for i, item in enumerate(kept_items, start=1):
            sub, qid = item.split("|", 1)
            w.writerow([i, item, sub, qid])

    with (out / "split_models.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split"])
        for m in train_models:
            if m in train.index:
                w.writerow([m, "train"])
        for m in test_models:
            w.writerow([m, "test"])

    # Chunk ends (~100 items/chunk), column indices into cleaned matrix
    # (col 1 = model id; items occupy 2..ncol).
    n_cols = 1 + len(kept_items)
    chunk_size = 100
    ends = list(range(1 + chunk_size, n_cols, chunk_size))
    if not ends or ends[-1] != n_cols:
        ends.append(n_cols)
    # Ensure first chunk has enough items
    ends = [e for e in ends if e >= 2 + 20]
    if ends[-1] != n_cols:
        ends.append(n_cols)
    (out / "chunk_ends.txt").write_text(",".join(str(e) for e in ends) + "\n")
    print(f"chunk_ends ({len(ends)}): {ends}", flush=True)
    print(f"Wrote matrices under {out}", flush=True)


if __name__ == "__main__":
    main()
