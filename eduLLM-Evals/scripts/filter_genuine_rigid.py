"""Filter the curated rigid_verbatim set down to GENUINE verbatim demands.

Genuine = a correct-but-differently-worded response could actually fail because
the criterion demands a specific phrase/label/exact restatement. Three buckets:
  A. verbatim_incorrectness_label  (Q5: "identify that X is incorrect")
  B. literal_phrase                (must say/include the phrase "...", verbatim, word-for-word)
  C. over_specified_restatement    (must explicitly restate ... as <formula> and not just ...)

Excludes withhold ("must not ...") criteria (kept per Q6).
Prints each with curated id + provenance (source id + current spec op) so we can
key soften ops correctly.
"""
import csv
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

rows = [r for r in csv.DictReader(open("staging/curated/audit_rubric_quality.csv", encoding="utf-8"))
        if r["rigid_verbatim"] == "1"]
curated = {j["criterion_id"]: j for j in
           (json.loads(l) for l in open("data/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8"))}
spec = json.load(open("curation/pilot_edits.json", encoding="utf-8"))
op_by_id = {o["id"]: o["op"] for o in spec["ops"]}

LITERAL = re.compile(
    r"(include the (?:phrase|sentence|wording)|the exact (?:phrase|wording)|word[- ]for[- ]word|"
    r"\bverbatim\b|must say that|should say that|state exactly|use the (?:exact )?(?:phrase|wording)|"
    r"say (?:something )?(?:like )?[\"\u201c\u2018])",
    re.I,
)
RESTATE_EXACT = re.compile(r"(explicitly restate|must restate[^.]*\bas\b|and not (?:just|only) the (?:final )?simplif)", re.I)
WITHHOLD = re.compile(r"\bmust not\b|\bshould not\b|without (?:actually |explicitly )?(?:saying|revealing|giving|stating)", re.I)


def bucket(text, reasons):
    if WITHHOLD.search(text):
        return None
    if "verbatim_incorrectness_label" in reasons:
        return "A_incorrect_label"
    if LITERAL.search(text):
        return "B_literal_phrase"
    if RESTATE_EXACT.search(text):
        return "C_over_specified_restatement"
    return None


buckets = {"A_incorrect_label": [], "B_literal_phrase": [], "C_over_specified_restatement": []}
for r in rows:
    cid = r["criterion_id"]
    text = curated[cid]["criterion"]
    b = bucket(text, r["reasons"].split("|"))
    if b:
        buckets[b].append(cid)

for b, ids in buckets.items():
    print(f"\n########## {b}: {len(ids)} ##########")
    for cid in ids:
        c = curated[cid]
        prov = c.get("curation", {}) or {}
        src = prov.get("original_criterion_id", cid)
        op = prov.get("op")
        spec_op = op_by_id.get(src, "(none)")
        print(f"--- curated={cid} | src={src} | curatedOp={op} | specOp={spec_op} | crit={r.get('criticality')}")
        print("   ", c["criterion"][:280])

tot = sum(len(v) for v in buckets.values())
print(f"\nGENUINE rigid total: {tot}")
