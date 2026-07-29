"""Ad-hoc viewer: dump full text of criteria flagged for given issue types.

Usage: python scripts/show_flagged.py absolute_failure reference_leak ...
If no args, dumps the three small mechanical buckets.
"""
import csv
import sys

CSV = "staging/audit_rubric_quality.csv"


def main() -> None:
    types = sys.argv[1:] or ["absolute_failure", "reference_leak", "prescribed_wording"]
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    for t in types:
        hits = [r for r in rows if r.get(t) == "1"]
        print(f"\n########## {t}: {len(hits)} ##########")
        for r in hits:
            print(f"--- {r['criterion_id']} | {r['subject']} | {r['criticality']} | {r['primary_skill']}")
            print(r["criterion"])


if __name__ == "__main__":
    main()
