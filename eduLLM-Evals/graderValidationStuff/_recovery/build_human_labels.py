"""Reconstruct human_labels.csv from the frq/tutorbench grader packets + v3 judgment input hashes."""
from __future__ import annotations
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKETS = sorted((HERE / "grader_packets").glob("grader_*.csv"))
V3_ROOT = HERE.parent / "judge-v3-results"
QWEN_R1 = V3_ROOT / "qwen" / "canonical_r1" / "canonical_r1.jsonl"
OUT = HERE / "human_labels.csv"

# case_id -> input_hash from a canonical judgment wave (identical across all waves/judges).
input_hash: dict[str, str] = {}
for line in QWEN_R1.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    input_hash[row["case_id"]] = row.get("input_hash", "")

rows: list[dict[str, str]] = []
seen: set[str] = set()
label_counts = {"pass": 0, "fail": 0}
blank_grade = 0
missing_hash: list[str] = []

for packet in PACKETS:
    with packet.open("r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            assignment_id = (r.get("assignment_id") or "").strip()
            criterion_id = (r.get("criterion_id") or "").strip()
            if not assignment_id or not criterion_id:
                continue
            case_id = f"{assignment_id}__{criterion_id}"
            if case_id not in input_hash:
                # Surplus human label for a criterion not in the v3 judged set; skip.
                continue
            if case_id in seen:
                raise SystemExit(f"duplicate case_id {case_id} in {packet.name}")
            seen.add(case_id)
            grade = (r.get("grade") or "").strip().upper()
            if grade == "P":
                human_label = "pass"
            elif grade == "F":
                human_label = "fail"
            else:
                blank_grade += 1
                continue
            label_counts[human_label] += 1
            h = input_hash.get(case_id, "")
            if not h:
                missing_hash.append(case_id)
            rows.append({
                "case_id": case_id,
                "case_input_hash": h,
                "human_label": human_label,
                "scenario_id": (r.get("scenario_id") or "").strip(),
                "criterion_id": criterion_id,
                "primary_skill": (r.get("primary_skill") or "").strip(),
                "criticality": (r.get("criticality") or "").strip(),
            })

fields = ["case_id", "case_input_hash", "human_label", "scenario_id",
          "criterion_id", "primary_skill", "criticality"]
with OUT.open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

judged = set(input_hash)
labeled = set(seen)
print(f"packets: {len(PACKETS)}")
print(f"human-labeled cases written: {len(rows)}  (pass={label_counts['pass']}, fail={label_counts['fail']}, blank_grade={blank_grade})")
print(f"distinct judged case_ids in v3: {len(judged)}")
print(f"labeled not judged: {sorted(labeled - judged)[:10]}  (n={len(labeled - judged)})")
print(f"judged not labeled: {sorted(judged - labeled)[:10]}  (n={len(judged - labeled)})")
print(f"cases missing input_hash: {len(missing_hash)} {missing_hash[:5]}")
print(f"wrote {OUT}")
