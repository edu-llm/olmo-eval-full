import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
packets = Path(sys.argv[1])
ids = sys.argv[2:]
rows = {json.loads(l)["gold_case_id"]: json.loads(l)
        for l in packets.read_text(encoding="utf-8").splitlines() if l.strip()}
for cid in ids:
    r = rows[cid]
    print("\n\n########## " + cid + " ##########")
    print("STRATUM:", r.get("stratum"), "| CRITICALITY:", r.get("criticality"), "| TASK:", r.get("task"))
    print("\n--SCENARIO_PROMPT--\n" + str(r.get("scenario_prompt", ""))[:1400])
    ctx = r.get("conversation_context") or []
    if ctx:
        print("\n--CONVERSATION--")
        for t in ctx:
            print(f"[{t.get('role','?')}] {str(t.get('content',''))[:500]}")
    ref = str(r.get("reference_solution") or "").strip()
    if ref:
        print("\n--REFERENCE--\n" + ref[:600])
    print("\n--CRITERION--\n" + r["criterion"])
    resp = r["candidate_response"].strip()
    print(f"\n--RESPONSE ({len(resp)} chars)--\n" + resp[:1400])
    if len(resp) > 1400:
        print("...[truncated]")
    for n, p in r["proposals"].items():
        print(f"\n--{n}: {p.get('label')}--")
        print("for:", str(p.get('evidence_for') or '')[:280])
        print("against:", str(p.get('evidence_against') or '')[:280])
        print("why:", str(p.get('rationale') or '')[:280])
