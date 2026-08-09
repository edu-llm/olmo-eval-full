#!/usr/bin/env python
"""Convert the 261-case tutorbench human set into the score_gold gold/sample format.

Lets the same full-metric gold scorer (score_gold.py) run on tutorbench: joins the blinded
cases, human labels (capability := criticality bucket), and the frozen Qwen v3 verdicts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("eduLLM-Evals")
BLINDED = ROOT / "AdaptiveTesting" / "Inputs"  # placeholder; real path below
BLINDED = Path(
    "AdaptiveTesting/Inputs/Open/LLM-Judge/aws_judge_handoff_extracted/"
    "aws_judge_handoff/inputs/judge_cases.blinded.jsonl"
)
LABELS = ROOT / "api_judge_pilot" / "human_labels.csv"
QWEN = ROOT / "graderValidationStuff" / "judge-v3-results" / "qwen" / "canonical_r1" / "canonical_r1.jsonl"
OUT = ROOT / "api_judge_pilot" / "gold" / "tutorbench_core"


def main() -> int:
    cases = {json.loads(x)["case_id"]: json.loads(x)
             for x in BLINDED.read_text(encoding="utf-8").splitlines() if x.strip()}
    labels = {}
    with LABELS.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            labels[row["case_id"]] = row
    qwen = {}
    if QWEN.is_file():
        for x in QWEN.read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                qwen[r["case_id"]] = 1 if str(r.get("verdict")).lower() == "pass" else 0

    OUT.mkdir(parents=True, exist_ok=True)
    gold_rows, sample_rows = [], []
    for cid, lab in labels.items():
        case = cases.get(cid)
        if case is None:
            continue
        crit = (lab.get("criticality") or "").strip()
        capability = "critical" if crit.startswith("critical") else "not_critical"
        qv = qwen.get(cid, 0)
        gold_rows.append({
            "gold_case_id": cid, "model": "", "scenario_id": lab.get("scenario_id", ""),
            "criterion_id": lab.get("criterion_id", ""), "capability": capability,
            "gold_label": lab["human_label"].strip().lower(), "provenance": "human", "_qwen_verdict": qv,
        })
        srow = dict(case)
        srow["gold_case_id"] = cid
        srow["capability"] = capability
        srow["_qwen_verdict"] = qv
        sample_rows.append(srow)

    (OUT / "gold_labels.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold_rows), encoding="utf-8")
    (OUT / "sample.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample_rows), encoding="utf-8")
    (OUT / "sample_manifest.json").write_text(json.dumps({
        "benchmark": "tutorbench", "population_gradable_cells": None, "sample_size": len(gold_rows),
        "stratify_by": "criticality", "capability_population": {},
        "note": "Full 261-case human set (not a probability sample); no population weighting.",
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(gold_rows)} gold rows, {len(sample_rows)} sample rows to {OUT}; "
          f"qwen_joined={sum(1 for c in gold_rows if c['gold_case_id'] in qwen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
