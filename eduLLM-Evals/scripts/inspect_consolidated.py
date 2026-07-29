"""Inspect consolidated presentation criteria: show aspect coverage + a few
single-aspect and multi-aspect examples with the orphans they replaced."""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

src = {j["criterion_id"]: j for j in
       (json.loads(l) for l in open("data/TutorBench/rubrics_qmatrix_final.jsonl", encoding="utf-8"))}
cur = [json.loads(l) for l in open("data/TutorBench/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8")]

cons = [c for c in cur if (c.get("curation", {}) or {}).get("op") == "format_consolidated"]
print(f"consolidated presentation criteria: {len(cons)}")
naspect = Counter(len((c.get("curation", {}) or {}).get("aspects", [])) for c in cons)
print("by #aspects:", dict(sorted(naspect.items())))
asp = Counter(a for c in cons for a in (c.get("curation", {}) or {}).get("aspects", []))
print("aspect frequency:", dict(asp))
# sanity: all optional + tagged
assert all(c.get("optional") is True for c in cons), "some consolidated not optional"
assert all(c.get("dimension") == "style_surface" for c in cons), "some missing tag"
print("OK: all consolidated are optional + dimension=style_surface\n")

def show(pred, label, n=3):
    picks = [c for c in cons if pred(c)][:n]
    print(f"\n===== {label} =====")
    for c in picks:
        ca = c.get("curation", {})
        print(f"--- {c['criterion_id']}  aspects={ca.get('aspects')}  from={ca.get('from')}")
        print("   NOW:", c["criterion"])
        for oid in ca.get("from", []):
            if oid in src:
                print(f"   was {oid}:", src[oid]["criterion"][:160])

show(lambda c: len(c["curation"].get("aspects", [])) == 1 and len(c["curation"]["from"]) == 1,
     "SINGLE orphan, SINGLE aspect (targets once)")
show(lambda c: len(c["curation"]["from"]) >= 2, "MULTI orphan (targets all present aspects)")
show(lambda c: len(c["curation"].get("aspects", [])) == 0, "NO aspect matched (generic fallback)", n=4)
