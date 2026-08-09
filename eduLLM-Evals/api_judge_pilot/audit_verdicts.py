#!/usr/bin/env python
"""Tier-0 acceptance-gate audit of already-collected judge verdicts (zero API calls).

Joins a verdicts.jsonl (case_id, verdict, human_label) to human_labels.csv (which carries
criticality + primary_skill) and reports the same view used for the Qwen v3 gate:
overall agreement, false-pass split by criticality, critical-failure sensitivity, and
per-skill F1. Tells you WHERE the false-pass lives so prompt work can be aimed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_verdicts(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["case_id"]] = r
    return out


def _load_labels(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["case_id"]] = row
    return out


def _f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _confusion(pairs: list[tuple[str, str]]) -> dict[str, int]:
    tp = sum(1 for v, h in pairs if v == "pass" and h == "pass")
    tn = sum(1 for v, h in pairs if v == "fail" and h == "fail")
    fp = sum(1 for v, h in pairs if v == "pass" and h == "fail")
    fn = sum(1 for v, h in pairs if v == "fail" and h == "pass")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": len(pairs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    verdicts = _load_verdicts(args.verdicts)
    labels = _load_labels(args.labels)

    pairs: list[tuple[str, str]] = []
    by_crit: dict[str, list[tuple[str, str]]] = {}
    by_skill: dict[str, list[tuple[str, str]]] = {}
    for cid, lab in labels.items():
        v = verdicts.get(cid)
        if v is None:
            continue
        human = lab["human_label"].strip().lower()
        verdict = v["verdict"].strip().lower()
        if human not in ("pass", "fail") or verdict not in ("pass", "fail"):
            continue
        pair = (verdict, human)
        pairs.append(pair)
        crit = lab.get("criticality") or "unspecified"
        crit_bucket = "critical" if crit.startswith("critical") else "not_critical"
        by_crit.setdefault(crit_bucket, []).append(pair)
        skill = lab.get("primary_skill") or "unmapped"
        by_skill.setdefault(skill or "unmapped", []).append(pair)

    overall = _confusion(pairs)
    fp_rate = overall["fp"] / (overall["fp"] + overall["tn"]) if (overall["fp"] + overall["tn"]) else 0.0
    acc = (overall["tp"] + overall["tn"]) / overall["n"] if overall["n"] else 0.0

    report: dict[str, Any] = {
        "verdicts": str(args.verdicts),
        "overall": {
            **overall,
            "accuracy": round(acc, 4),
            "false_pass_rate": round(fp_rate, 4),
        },
        "by_criticality": {},
        "by_skill": {},
    }

    # false-pass by criticality + critical-failure sensitivity (recall on human fails)
    for bucket, bpairs in sorted(by_crit.items()):
        c = _confusion(bpairs)
        human_fails = c["fp"] + c["tn"]
        sensitivity = c["tn"] / human_fails if human_fails else None  # fraction of human-fails caught
        report["by_criticality"][bucket] = {
            **c,
            "false_pass_rate": round(c["fp"] / human_fails, 4) if human_fails else None,
            "failure_sensitivity": round(sensitivity, 4) if sensitivity is not None else None,
        }

    for skill, spairs in sorted(by_skill.items()):
        c = _confusion(spairs)
        human_fails = c["fp"] + c["tn"]
        report["by_skill"][skill] = {
            "n": c["n"],
            "f1_pass": round(_f1(c["tp"], c["fp"], c["fn"]), 4),
            "false_pass_rate": round(c["fp"] / human_fails, 4) if human_fails else None,
            "accuracy": round((c["tp"] + c["tn"]) / c["n"], 4) if c["n"] else None,
        }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
