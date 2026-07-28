"""Preview what bank-wide format-orphan consolidation would merge, per scenario.

Shows scenarios that currently have >=2 surface/style orphans among the 308
flagged, the exact criteria that would be replaced, and the single standard
presentation criterion they'd consolidate into.
"""
import csv
import json
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

FORMAT_ORPHAN_RE = re.compile(
    r"\b(second person|first person|perspective of|use of \"?you|markdown|latex|"
    r"headings?|bold|bullet|formatt?ing|demarcate)\b",
    re.I,
)

curated = {j["criterion_id"]: j for j in
           (json.loads(l) for l in open("data/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8"))}
spec = json.load(open("curation/pilot_edits.json", encoding="utf-8"))
std_text = spec["meta"]["standard_format_criterion"]

rows = [r for r in csv.DictReader(open("staging/curated/audit_rubric_quality.csv", encoding="utf-8"))
        if "surface_format_or_persona_only" in r["reasons"].split("|")]

by_scen = defaultdict(list)
for r in rows:
    by_scen[curated[r["criterion_id"]]["scenario_id"]].append(r["criterion_id"])

# how many would actually match apply_curation's FORMAT_ORPHAN_RE (i.e. be merged)
would_merge = {cid for cid in curated
               if FORMAT_ORPHAN_RE.search(curated[cid].get("criterion", "") or "")
               and not (curated[cid].get("q_mapping", {}) or {}).get("content")
               and not (curated[cid].get("q_mapping", {}) or {}).get("diagnosis")
               and not (curated[cid].get("q_mapping", {}) or {}).get("scaffolding")}

multi = {s: ids for s, ids in by_scen.items() if len(ids) >= 2}
print(f"scenarios with >=2 surface orphans: {len(multi)}")
print(f"scenarios with exactly 1 surface orphan: {sum(1 for v in by_scen.values() if len(v)==1)}")
print(f"total surface-flagged: {sum(len(v) for v in by_scen.values())}\n")
print("STANDARD PRESENTATION CRITERION (merge target):")
print("   ", std_text, "\n")
print("="*90)

for sid in list(multi)[:6]:
    ids = multi[sid]
    print(f"\n### {sid}  ({len(ids)} orphans -> 1 presentation criterion)")
    for cid in ids:
        c = curated[cid]
        mark = "" if cid in would_merge else "  [NOTE: audit-flagged but NOT matched by merge regex -> would stay]"
        print(f"  - {cid} (crit={c.get('criticality')}){mark}")
        print(f"      {c['criterion'][:200]}")

# also flag any audit-flagged surface items that the merge regex misses
missed = [r["criterion_id"] for r in rows if r["criterion_id"] not in would_merge]
print("\n" + "="*90)
print(f"audit-flagged surface items NOT caught by the merge regex: {len(missed)}")
for cid in missed[:12]:
    print(f"  - {cid}: {curated[cid]['criterion'][:160]}")
