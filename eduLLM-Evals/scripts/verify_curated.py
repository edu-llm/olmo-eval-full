"""Integrity checks on the curated rubric bank + scenarios."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

rubrics = [json.loads(l) for l in open("data/TutorBench/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8")]
scenarios = [json.loads(l) for l in open("data/TutorBench/curated/scenarios_curated.jsonl", encoding="utf-8")]

by_id = {}
dupes = []
empty = []
no_rat = []
for r in rubrics:
    cid = r["criterion_id"]
    if cid in by_id:
        dupes.append(cid)
    by_id[cid] = r
    if not (r.get("criterion") or "").strip():
        empty.append(cid)
    if not (r.get("q_rationale") or "").strip():
        no_rat.append(cid)

# scenario.criterion_ids must all resolve, and match the rubric membership
rub_ids = set(by_id)
missing_refs = []
for s in scenarios:
    for cid in s.get("criterion_ids", []):
        if cid not in rub_ids:
            missing_refs.append((s["scenario_id"], cid))

# every rubric's scenario should list it
listed = {cid for s in scenarios for cid in s.get("criterion_ids", [])}
unlisted = [cid for cid in rub_ids if cid not in listed]

print(f"curated criteria: {len(rubrics)}  unique ids: {len(by_id)}")
print(f"duplicate ids: {len(dupes)} {dupes[:5]}")
print(f"empty criterion text: {len(empty)} {empty[:5]}")
print(f"missing q_rationale: {len(no_rat)} {no_rat[:5]}")
print(f"scenario refs to missing criteria: {len(missing_refs)} {missing_refs[:5]}")
print(f"rubrics not listed in any scenario: {len(unlisted)} {unlisted[:5]}")

# spot-check a few known splits (by original_criterion_id in curation provenance)
def children_of(orig):
    return [r["criterion_id"] for r in rubrics
            if (r.get("curation") or {}).get("op") == "split"
            and (r.get("curation") or {}).get("from") == orig]

for orig in ["tb_0271_c03", "tb_0464_c13", "tb_0368_c03", "tb_0276_c05", "tb_0600_c02"]:
    print(f"  split {orig} -> {children_of(orig)}")
