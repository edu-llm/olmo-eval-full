"""Per-use_case pattern coverage: within each use_case, which criterion PATTERNS
are prevalent (present in most scenarios) but MISSING in some -> candidates to add
by shared pattern. Also prints a prevalence matrix.
"""
import json
import re
import sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")

cur = [json.loads(l) for l in open("data/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8")]
scen = {s["scenario_id"]: s for s in
        (json.loads(l) for l in open("data/curated/scenarios_curated.jsonl", encoding="utf-8"))}

# Tightened detectors: match plurals, answer-specific withhold wording, and
# explaining/error-identifying criteria regardless of exact skill tag.
_NEG = re.compile(r"\b(must not|should not|shouldn'?t|do not|don'?t|never|avoid|without)\b", re.I)
_GIVE = re.compile(r"\b(give|giv\w*|provide|provid\w*|state|reveal\w*|show\w*|tell\w*|"
                   r"calculat\w*|comput\w*|solv\w*|writ\w*|present\w*|report\w*|disclos\w*)\b", re.I)

PAT = {
    "final_answer":  (lambda c, t: c.get("primary_skill") == "content"
                      and re.search(r"\b(correct |final )*(answer|result|value)s?\b", t, re.I)
                      and re.search(r"\b(provide|state|give|include|arrive at|report)\b", t, re.I)
                      and not (_NEG.search(t))),
    "error_ident":   (lambda c, t: re.search(r"\b(error|mistake|incorrect|wrong|misconception|"
                                             r"conflat\w*|flaw\w*|misunderstand\w*)s?\b", t, re.I)
                      and (c.get("primary_skill") == "diagnosis"
                           or re.search(r"\bidentif\w+|fails? to\b", t, re.I))),
    "withhold":      (lambda c, t: bool(_NEG.search(t) and _GIVE.search(t))
                      or (c.get("criticality") == "critical_negative" and _NEG.search(t))),
    "hint_scaffold": (lambda c, t: (c.get("primary_skill") == "scaffolding")
                      and re.search(r"\b(hint|guide\w*|prompt\w*|leading question|nudg\w*|"
                                    r"ask\w*|suggest\w*|encourag\w*|lead\w*|consider)\b", t, re.I)),
    "concept_explain": (lambda c, t: c.get("primary_skill") in ("content", "diagnosis")
                        and re.search(r"\b(explain\w*|clarif\w*|\bwhy\b|because|conceptual|"
                                      r"intuition|the reason|the rule|principle|define\w*)\b", t, re.I)),
    "acknowledge":   (lambda c, t: bool(re.search(r"\b(acknowledg\w*|confusion|struggl\w*|empath\w*|"
                                                  r"validat\w*|reassur\w*|prais\w*|recogniz\w*|effort)\b", t, re.I))),
}

types_by_scen = defaultdict(set)
uc_of = {}
for c in cur:
    sid = c["scenario_id"]
    t = c.get("criterion", "") or ""
    for name, fn in PAT.items():
        try:
            if fn(c, t):
                types_by_scen[sid].add(name)
        except Exception:
            pass
for sid, s in scen.items():
    uc_of[sid] = s.get("use_case") or "?"

by_uc = defaultdict(list)
for sid in scen:
    by_uc[uc_of[sid]].append(sid)

names = list(PAT)
print(f"use_cases: {len(by_uc)}   scenarios: {len(scen)}\n")
print(f"{'use_case':28s} {'n':>4s}  " + "  ".join(f"{n[:11]:>11s}" for n in names))
for uc, sids in sorted(by_uc.items(), key=lambda kv: -len(kv[1])):
    n = len(sids)
    cells = []
    for nm in names:
        have = sum(1 for s in sids if nm in types_by_scen[s])
        cells.append(f"{100*have//n:>9d}%" if n else "   -")
    print(f"{uc[:28]:28s} {n:>4d}  " + "  ".join(cells))

print("\n### Prevalent-but-missing candidates (pattern in 55-95% of a use_case) ###")
for uc, sids in sorted(by_uc.items(), key=lambda kv: -len(kv[1])):
    n = len(sids)
    if n < 8:
        continue
    for nm in names:
        have = [s for s in sids if nm in types_by_scen[s]]
        miss = [s for s in sids if nm not in types_by_scen[s]]
        frac = len(have) / n
        if 0.55 <= frac <= 0.95:
            print(f"- [{uc}] '{nm}' in {len(have)}/{n} ({int(100*frac)}%); "
                  f"missing e.g.: {miss[:6]}")
