"""Tabulate all curation changes: source vs curated, by op, with deltas."""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

src = [json.loads(l) for l in open("data/TutorBench/rubrics_qmatrix_final.jsonl", encoding="utf-8")]
cur = [json.loads(l) for l in open("data/TutorBench/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8")]
src_ids = {r["criterion_id"] for r in src}

op = Counter((c.get("curation", {}) or {}).get("op", "unchanged") for c in cur)

# source criteria that were removed (split parents + consolidated orphans)
split_parents = set()
consolidated_orphans = set()
for c in cur:
    cu = c.get("curation", {}) or {}
    if cu.get("op") == "split":
        split_parents.add(cu.get("from"))
    if cu.get("op") == "format_consolidated":
        consolidated_orphans.update(cu.get("from", []))

split_children = sum(1 for c in cur if (c.get("curation", {}) or {}).get("op") == "split")
consolidated = sum(1 for c in cur if (c.get("curation", {}) or {}).get("op") == "format_consolidated")
optional = sum(1 for c in cur if c.get("optional") is True)
style = sum(1 for c in cur if c.get("dimension") == "style_surface")

print(f"SOURCE criteria : {len(src)}")
print(f"CURATED criteria: {len(cur)}   (net {len(cur)-len(src):+d})")
print()
print("Curated criteria by curation.op:")
for k, v in op.most_common():
    print(f"  {k:20s} {v:5d}")
print()
print(f"split parents removed        : {len(split_parents)}  -> {split_children} children (+{split_children-len(split_parents)})")
print(f"orphans consolidated         : {len(consolidated_orphans)} -> {consolidated} presentation criteria ({consolidated-len(consolidated_orphans):+d})")
print(f"optional criteria (total)    : {optional}")
print(f"  of which style_surface     : {style}")
print(f"  other optional (rescope)   : {optional-style}")
