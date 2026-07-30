"""Transform the Dr. SCI parquet files into scenario + rubric JSONL.

Reads:
  data/dr-sci/Dr_SCI_verifiable.parquet
  data/dr-sci/Dr_SCI_open-ended.parquet

Writes (into data/dr-sci/):
  scenario_verifiable.jsonl   + rubrics_verifiable.jsonl
  scenario_openended.jsonl    + rubrics_openended.jsonl

Scenario schema (per user spec):
  scenario_id, use_case, subject, grade_band, modality, prompt,
  conversation_context, reference_solution, criterion_ids, source, split, version

Decisions:
  - prompt is kept as the raw chat list [{content, role}, ...] verbatim.
  - scenario_ids use distinct prefixes: dsv_<n> (verifiable), dso_<n> (open-ended), 0-based per file.
  - verifiable items (no native rubric) get ONE synthetic exact-match criterion.
  - open-ended items expose their native rubric criteria (title/description/weight).
"""
import json
import os
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DR_SCI = os.path.normpath(os.path.join(HERE, "..", "data", "dr-sci"))

SPLIT = "calibration"
VERSION = "1.0"


def scenario_base(sid, subject, prompt, reference_solution, criterion_ids, source):
    return {
        "scenario_id": sid,
        "use_case": subject,
        "subject": subject,
        "grade_band": "",
        "modality": "text",
        "prompt": prompt,
        "conversation_context": [],
        "reference_solution": reference_solution,
        "criterion_ids": criterion_ids,
        "source": source,
        "split": SPLIT,
        "version": VERSION,
    }


def dump(obj):
    return json.dumps(obj, ensure_ascii=False)


def build_verifiable():
    src = os.path.join(DR_SCI, "Dr_SCI_verifiable.parquet")
    scen_path = os.path.join(DR_SCI, "scenario_verifiable.jsonl")
    rub_path = os.path.join(DR_SCI, "rubrics_verifiable.jsonl")
    pf = pq.ParquetFile(src)
    n = 0
    with open(scen_path, "w", encoding="utf-8") as fs, open(rub_path, "w", encoding="utf-8") as fr:
        for g in range(pf.metadata.num_row_groups):
            rows = pf.read_row_group(g, columns=["prompt", "reward_model", "extra_info"]).to_pylist()
            for row in rows:
                sid = f"dsv_{n}"
                subject = row["extra_info"].get("subject")
                gt = row["reward_model"].get("ground_truth")
                cid = f"{sid}_c01"
                fs.write(dump(scenario_base(
                    sid, subject, row["prompt"], gt, [cid], "Dr. SCI_verifiable")) + "\n")
                fr.write(dump({
                    "criterion_id": cid,
                    "scenario_id": sid,
                    "title": "Exact Match",
                    "description": ("Rule-based check: the response's final answer exactly matches the "
                                    "reference ground_truth for this item."),
                    "weight": 1,
                }) + "\n")
                n += 1
            print(f"[verifiable] row_group {g}: cumulative {n}")
    print(f"[verifiable] DONE: {n} scenarios -> {scen_path}")
    return n


def build_openended():
    src = os.path.join(DR_SCI, "Dr_SCI_open-ended.parquet")
    scen_path = os.path.join(DR_SCI, "scenario_openended.jsonl")
    rub_path = os.path.join(DR_SCI, "rubrics_openended.jsonl")
    pf = pq.ParquetFile(src)
    n = 0
    n_crit = 0
    with open(scen_path, "w", encoding="utf-8") as fs, open(rub_path, "w", encoding="utf-8") as fr:
        for g in range(pf.metadata.num_row_groups):
            rows = pf.read_row_group(g, columns=["prompt", "reward_model", "extra_info"]).to_pylist()
            for row in rows:
                sid = f"dso_{n}"
                subject = row["extra_info"].get("subject")
                gt = row["reward_model"].get("ground_truth")
                rubric = row["reward_model"].get("rubric") or []
                cids = []
                for j, c in enumerate(rubric, start=1):
                    cid = f"{sid}_c{j:02d}"
                    cids.append(cid)
                    fr.write(dump({
                        "criterion_id": cid,
                        "scenario_id": sid,
                        "title": c.get("title"),
                        "description": c.get("description"),
                        "weight": c.get("weight"),
                    }) + "\n")
                    n_crit += 1
                fs.write(dump(scenario_base(
                    sid, subject, row["prompt"], gt, cids, "Dr. SCI_open_ended")) + "\n")
                n += 1
            print(f"[open-ended] row_group {g}: cumulative {n} scenarios / {n_crit} criteria")
    print(f"[open-ended] DONE: {n} scenarios / {n_crit} criteria -> {scen_path}")
    return n, n_crit


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    v = build_verifiable()
    o, oc = build_openended()
    print("=" * 60)
    print(f"verifiable scenarios : {v}")
    print(f"open-ended scenarios : {o}  (criteria: {oc})")
