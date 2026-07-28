"""Sample irrelevant_misscoped (+ numeric_no_tolerance) criteria from the curated
audit, grouped by reason, with q_mapping + skill so we can judge scope."""
import csv
import json
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

CSV = "staging/curated/audit_rubric_quality.csv"
curated = {j["criterion_id"]: j for j in
           (json.loads(l) for l in open("data/TutorBench/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8"))}
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

def qm(cid):
    q = curated[cid].get("q_mapping", {}) or {}
    return f"c{q.get('content',0)}/d{q.get('diagnosis',0)}/s{q.get('scaffolding',0)}"

random.seed(7)
buckets = {
    "surface_format_or_persona_only": [r for r in rows if "surface_format_or_persona_only" in r["reasons"].split("|")],
    "restatement_of_formula_or_answer": [r for r in rows if "restatement_of_formula_or_answer" in r["reasons"].split("|")],
    "numeric_no_tolerance": [r for r in rows if r["numeric_no_tolerance"] == "1"],
}
for name, hits in buckets.items():
    print(f"\n########## {name}: {len(hits)} ##########")
    for r in random.sample(hits, min(9, len(hits))):
        cid = r["criterion_id"]
        print(f"--- {cid} | {r['criticality']} | skill={r['primary_skill']} | q={qm(cid)}")
        print("   ", curated[cid]["criterion"][:280])
