import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
packets = Path(sys.argv[1])
ids = sys.argv[2:]
rows = {json.loads(l)["gold_case_id"]: json.loads(l)
        for l in packets.read_text(encoding="utf-8").splitlines() if l.strip()}
for i, cid in enumerate(ids, 1):
    r = rows[cid]
    why = (r["proposals"].get("gpt-5.5", {}).get("rationale")
           or r["proposals"].get("opus-5", {}).get("rationale") or "")
    resp = r["candidate_response"].strip().replace("\n", " ")
    stratum = r.get("stratum", r.get("capability"))
    print(f"\n##### {i}. {cid} | {stratum} | crit={r.get('criticality')} | LABEL={str(r['recommendation']).upper()}")
    print("CRITERION: " + r["criterion"])
    print(f"RESPONSE ({len(r['candidate_response'].strip())} chars): " + resp[:460])
    print("WHY: " + str(why)[:220])
