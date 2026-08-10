import json
import sys
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
issues = [r for r in rows if any(p.get("status") != "ok" for p in r["proposals"].values())]
disc = [r for r in rows if not r["concordant"]]
def _stratum(r):
    return r.get("stratum", r.get("capability"))

print(f"total={len(rows)} concordant={sum(r['concordant'] for r in rows)} discordant={len(disc)}")
print(f"stratum dist: {dict(Counter(_stratum(r) for r in rows))}")
print(f"recommendation: {dict(Counter(r['recommendation'] for r in rows))}")
print(f"\ncases_with_proposer_issue ({len(issues)}):")
for r in issues:
    bad = {n: p.get("status") for n, p in r["proposals"].items() if p.get("status") != "ok"}
    print(f"  {r['gold_case_id']}  {bad}")
print(f"\ndiscordant cases ({len(disc)}):")
for r in disc:
    labels = {n: p.get("label") for n, p in r["proposals"].items()}
    print(f"  {r['gold_case_id']}  stratum={_stratum(r)}  {labels}")
