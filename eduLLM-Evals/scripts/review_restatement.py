"""Build the restatement-of-formula/answer review set for context judgment.

A restatement criterion is only genuinely MIS-SCOPED if the answer/formula it
demands was already delivered in a PRIOR conversation turn and the current turn
doesn't call for repeating it. So we:
  - re-run the RESTATE detector on the SOURCE bank (source ids for ops),
  - drop withhold ("must not give the answer") criteria (those are KEEP),
  - join each to its scenario's conversation_context,
  - split into: no-context (auto-KEEP, legit content) vs has-context (REVIEW).
Writes curation/tranche/restatement_review.json for triage.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RESTATE = re.compile(
    r"(must|should)\b[^.]*\b(include|provide|state|give)\b[^.]*\b(the formula|the answer|"
    r"the correct answer|the value|the equation)\b",
    re.I,
)
WITHHOLD = re.compile(r"\b(must not|should not|shouldn't|do not|don't|without)\b", re.I)

rubrics = [json.loads(l) for l in open("data/rubrics_qmatrix_final.jsonl", encoding="utf-8")]
scen = {s["scenario_id"]: s for s in
        (json.loads(l) for l in open("data/scenarios.jsonl", encoding="utf-8"))}

hits = [r for r in rubrics if RESTATE.search(r.get("criterion", "") or "")]
withheld = [r for r in hits if WITHHOLD.search(r.get("criterion", "") or "")]
candidates = [r for r in hits if not WITHHOLD.search(r.get("criterion", "") or "")]

def ctx(sid):
    s = scen.get(sid, {})
    return s.get("conversation_context") or []

no_ctx = [r for r in candidates if not ctx(r["scenario_id"])]
has_ctx = [r for r in candidates if ctx(r["scenario_id"])]

print(f"RESTATE hits (source):        {len(hits)}")
print(f"  withhold (auto-KEEP):       {len(withheld)}")
print(f"  candidates:                 {len(candidates)}")
print(f"    no prior context KEEP:    {len(no_ctx)}")
print(f"    HAS context -> REVIEW:    {len(has_ctx)}")

review = []
for r in has_ctx:
    sid = r["scenario_id"]
    s = scen.get(sid, {})
    review.append({
        "criterion_id": r["criterion_id"],
        "criticality": r.get("criticality"),
        "primary_skill": r.get("primary_skill"),
        "criterion": r["criterion"],
        "scenario_prompt": (s.get("prompt") or "")[:600],
        "conversation_context": [
            {"role": t.get("role"), "content": (t.get("content") or "")[:500]}
            for t in ctx(sid)
        ],
    })

out = Path("curation/tranche/restatement_review.json")
out.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {out} ({len(review)} items to review)")
