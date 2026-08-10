#!/usr/bin/env python
"""Assemble a models x criteria pass/fail response matrix from graded verdicts.

Reads a verdicts.jsonl (one record per (model, criterion_id)) plus the benchmark's
rubrics.jsonl (for deterministic column order + criterion metadata) and writes:

  - <out>/response_matrix.csv   wide matrix: rows=models, cols=criterion_id, cell=1 (pass) / 0 (fail)
  - <out>/criteria_index.csv    column order with scenario_id / criterion text / criticality
  - prints coverage + pass-rate summary

Verdict -> cell:  "pass" -> 1 ; everything else (fail / auto_fail / error) -> 0.
Missing (model, criterion) cells are written blank and reported (should be zero on a
complete run).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--rubrics", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Column order + metadata from the rubric bank (first occurrence wins).
    criteria: list[dict[str, Any]] = []
    seen_crit: set[str] = set()
    for r in _load_jsonl(args.rubrics):
        cid = str(r["criterion_id"])
        if cid in seen_crit:
            continue
        seen_crit.add(cid)
        criteria.append(
            {
                "criterion_id": cid,
                "scenario_id": str(r.get("scenario_id") or ""),
                "criterion": str(r.get("criterion") or ""),
                "criticality": r.get("criticality") or r.get("weight") or "",
            }
        )
    col_ids = [c["criterion_id"] for c in criteria]

    verdicts = _load_jsonl(args.verdicts)
    cell: dict[tuple[str, str], int] = {}
    models: set[str] = set()
    for v in verdicts:
        m, cid = str(v["model"]), str(v["criterion_id"])
        models.add(m)
        cell[(m, cid)] = 1 if v.get("verdict") == "pass" else 0
    model_rows = sorted(models)

    # Wide matrix
    matrix_path = args.out_dir / "response_matrix.csv"
    missing = 0
    passes = 0
    filled = 0
    with matrix_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", *col_ids])
        for m in model_rows:
            row = [m]
            for cid in col_ids:
                val = cell.get((m, cid))
                if val is None:
                    row.append("")
                    missing += 1
                else:
                    row.append(val)
                    filled += 1
                    passes += val
            w.writerow(row)

    # Criteria index
    idx_path = args.out_dir / "criteria_index.csv"
    with idx_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["col_index", "criterion_id", "scenario_id", "criticality", "criterion"])
        for i, c in enumerate(criteria):
            w.writerow([i, c["criterion_id"], c["scenario_id"], c["criticality"], c["criterion"]])

    n_cells = len(model_rows) * len(col_ids)
    print(f"models={len(model_rows)} criteria={len(col_ids)} cells={n_cells}")
    print(f"filled={filled} missing={missing} coverage={filled / n_cells:.4%}")
    print(f"overall_pass_rate={passes / filled:.4%}" if filled else "overall_pass_rate=n/a")
    print(f"wrote {matrix_path}")
    print(f"wrote {idx_path}")
    if missing:
        print(f"WARNING: {missing} (model,criterion) cells missing from verdicts -> blank in matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
