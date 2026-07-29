"""Sample rigid_verbatim criteria from the curated audit, grouped by reason."""
import csv
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")

CSV = "staging/curated/audit_rubric_quality.csv"
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["rigid_verbatim"] == "1"]

reasons = ["demands_specific_wording", "very_long_criterion", "long_ie_anchor", "verbatim_incorrectness_label"]
random.seed(11)
for reason in reasons:
    hits = [r for r in rows if reason in r["reasons"].split("|")]
    print(f"\n########## {reason}: {len(hits)} ##########")
    for r in random.sample(hits, min(8, len(hits))):
        print(f"--- {r['criterion_id']} | {r['criticality']} | {r['primary_skill']}")
        print("   ", r["criterion"][:300])
