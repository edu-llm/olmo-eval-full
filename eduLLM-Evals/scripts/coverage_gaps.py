"""Cross-scenario coverage gaps: where a criterion 'type' is present in most
scenarios of a group but MISSING from some -> candidates to ADD by pattern.

Types (heuristic, from criterion text + fields):
  presentation : dimension==style_surface OR all-zero Q + style regex
  final_answer : content criterion demanding the correct answer/result/value
  withhold     : "must not ... give/reveal the answer" (scaffolding restraint)
  error_ident  : diagnosis criterion identifying the student's error/mistake
"""
import json
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

cur = [json.loads(l) for l in open("data/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8")]
scen = {s["scenario_id"]: s for s in
        (json.loads(l) for l in open("data/curated/scenarios_curated.jsonl", encoding="utf-8"))}

ANSWER = re.compile(r"\b(correct )?(final )?(answer|result|value)\b", re.I)
PROVIDE = re.compile(r"\b(provide|state|give|include|arrive at|correct answer|final answer)\b", re.I)
WITHHOLD = re.compile(r"\b(must not|should not|shouldn'?t|without)\b.*\b(answer|solution|solve it)\b", re.I)
ERROR = re.compile(r"\b(error|mistake|incorrect|wrong|misconception|conflat|flaw)\b", re.I)

def types_of(c):
    t = set()
    txt = c.get("criterion", "") or ""
    q = c.get("q_mapping", {}) or {}
    skill = c.get("primary_skill")
    if c.get("dimension") == "style_surface":
        t.add("presentation")
    if skill == "content" and ANSWER.search(txt) and PROVIDE.search(txt):
        t.add("final_answer")
    if WITHHOLD.search(txt):
        t.add("withhold")
    if skill == "diagnosis" and ERROR.search(txt):
        t.add("error_ident")
    return t

by_scen = defaultdict(set)
subj_of = {}
for c in cur:
    sid = c["scenario_id"]
    by_scen[sid] |= types_of(c)
    s = scen.get(sid, {})
    subj_of[sid] = s.get("subject") or "?"

all_scen = list(by_scen)
N = len(all_scen)
print(f"scenarios: {N}\n")

for typ in ["presentation", "final_answer", "withhold", "error_ident"]:
    have = [s for s in all_scen if typ in by_scen[s]]
    missing = [s for s in all_scen if typ not in by_scen[s]]
    print(f"=== {typ}: present in {len(have)}/{N} ({100*len(have)//N}%), MISSING in {len(missing)} ===")

# The clearest add-by-pattern candidate: scenarios with NO presentation criterion
miss_pres = sorted(s for s in all_scen if "presentation" not in by_scen[s])
print(f"\nScenarios with NO presentation criterion: {len(miss_pres)}")
from collections import Counter
print("  by subject:", dict(Counter(subj_of[s] for s in miss_pres).most_common()))
print("  examples:", miss_pres[:15])
