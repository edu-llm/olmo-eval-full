"""Print exact before->after examples per op for the human-readable CHANGES doc."""
import json
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

src = {j["criterion_id"]: j for j in
       (json.loads(l) for l in open("data/rubrics_qmatrix_final.jsonl", encoding="utf-8"))}
cur = [json.loads(l) for l in open("data/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8")]

# group split children by parent
splits = defaultdict(list)
softens, consols, added, rescopes = [], [], [], []
for c in cur:
    cu = c.get("curation", {}) or {}
    op = cu.get("op")
    if op == "split":
        splits[cu.get("from")].append(c)
    elif op == "soften":
        softens.append(c)
    elif op == "format_consolidated":
        consols.append(c)
    elif op == "presentation_added":
        added.append(c)
    elif op == "rescope_optional":
        rescopes.append(c)

def p(s):
    print(s)

p("\n########## SPLIT (1 example) ##########")
pid = next(iter(splits))
p(f"PARENT {pid}:\n  {src[pid]['criterion']}")
for ch in splits[pid]:
    p(f"CHILD {ch['criterion_id']} [{ch.get('primary_skill')}]:\n  {ch['criterion']}")

p("\n########## SOFTEN (2 examples) ##########")
for c in softens[:2]:
    sid = c['curation']['original_criterion_id']
    p(f"{sid}\n  BEFORE: {src[sid]['criterion']}\n  AFTER : {c['criterion']}")

p("\n########## PRESENTATION CONSOLIDATION (2 examples) ##########")
for c in consols[:2]:
    frm = c['curation']['from']
    p(f"{c['criterion_id']} (from {frm}, aspects={c['curation'].get('aspects')})")
    for oid in frm:
        if oid in src:
            p(f"  BEFORE {oid}: {src[oid]['criterion'][:180]}")
    p(f"  AFTER : {c['criterion']}")

p("\n########## PRESENTATION ADDED (1 example) ##########")
c = added[0]
p(f"{c['criterion_id']} (aspects={c['curation'].get('aspects')})\n  NEW: {c['criterion']}")

p("\n########## RESCOPE_OPTIONAL (all) ##########")
for c in rescopes:
    sid = c['curation']['original_criterion_id']
    p(f"{sid}\n  BEFORE: {src[sid]['criterion'][:180]}\n  AFTER : {c['criterion'][:180]}\n  judge_guidance: {c.get('judge_guidance','')[:160]}")
