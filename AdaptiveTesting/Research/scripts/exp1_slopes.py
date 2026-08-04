#!/usr/bin/env python3
"""Slope (predicted~actual), Pearson r, MAE, SEM, 95% CI per OpenLM benchmark at SE<=0.3."""
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "AdaptiveTesting/Research/01_MCQ_ATLAS/data/atlas_replication"
benches = ["math", "ifeval", "gpqa", "musr", "bbh"]

print(f"{'bench':8} {'n':>4} {'slope':>7} {'r':>7} {'MAE':>7} {'SEM':>8} {'CI95_half':>10}")
for b in benches:
    pred, act, ae = [], [], []
    with open(BASE / b / "pirt_vs_actual_se_0.3.csv") as fh:
        for row in csv.DictReader(fh):
            pred.append(float(row["pirt_accuracy"]))
            act.append(float(row["actual_accuracy"]))
            ae.append(float(row["abs_error"]))
    pred, act, ae = np.array(pred), np.array(act), np.array(ae)
    n = len(ae)
    slope = float(np.polyfit(act, pred, 1)[0])
    r = float(np.corrcoef(pred, act)[0, 1])
    mae = float(ae.mean())
    sem = float(ae.std(ddof=0) / np.sqrt(n))
    ci = 1.96 * sem
    print(f"{b:8} {n:>4} {slope:7.3f} {r:7.3f} {mae:7.3f} {sem:8.4f} {ci:10.4f}")
