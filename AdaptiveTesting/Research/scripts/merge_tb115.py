#!/usr/bin/env python3
"""Merge the original 82-model TutorBench response matrix with the recent 33-model
AWS Qwen-judge run into a single 115-model calibration matrix.

The two runs used the same criterion naming (tb_XXXX_cYY). The 33 AWS models were
selected to be NON-overlapping with the original 82, so the union is 115 distinct
models. Columns are the union of criteria; cells graded in only one run are NaN
(the M2PL EM marginalises holes natively).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

M82 = Path("/Users/arhant/Documents/EDLM/olmo-eval-full/eduLLM-Evals/staging/response_matrix_full_nonopt.csv")
M33 = Path("/tmp/qwen_judge/tb33_matrix/TutorBench/response_matrix.csv")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/arhant/Documents/EDLM/olmo-eval-full/AdaptiveTesting/Research/03_FRQ_MIRT/data/tb115/response_matrix_115.csv"
)

d82 = pd.read_csv(M82, index_col=0)
d33 = pd.read_csv(M33, index_col=0)
print(f"82-run: {d82.shape[0]} models x {d82.shape[1]} criteria")
print(f"33-run: {d33.shape[0]} models x {d33.shape[1]} criteria")

overlap_models = sorted(set(d82.index) & set(d33.index))
print(f"overlapping models: {len(overlap_models)} -> {overlap_models}")

# Union of models (rows) and criteria (cols). For any overlapping model, prefer the
# original 82-run grade where present, filling holes from the 33-run.
merged = d82.combine_first(d33)
# combine_first aligns on index+columns union; d82 takes precedence on overlaps.
merged = merged.reindex(index=sorted(set(d82.index) | set(d33.index)),
                        columns=sorted(set(d82.columns) | set(d33.columns)))
merged.index.name = "model"

n_cells = int(merged.notna().sum().sum())
total = merged.shape[0] * merged.shape[1]
print(f"merged: {merged.shape[0]} models x {merged.shape[1]} criteria")
print(f"observed cells: {n_cells}/{total} ({100*n_cells/total:.1f}% fill)")

OUT.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT)
print(f"wrote {OUT}")
