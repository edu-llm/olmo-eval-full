import json
from pathlib import Path

ids = ["biggen__0021","biggen__0048","biggen__0070","biggen__0073","biggen__0085","biggen__0087","biggen__0011"]
rows = {json.loads(l)["gold_case_id"]: json.loads(l)
        for l in Path("api_judge_pilot/gold/biggen_core/packets.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
for cid in ids:
    r = rows[cid]
    print("\n\n########## " + cid + " ##########")
    print("CAPABILITY:", r["capability"], "| TASK:", r["task"])
    print("\n--SCENARIO_PROMPT--\n" + str(r.get("scenario_prompt",""))[:1800])
    ctx = r.get("conversation_context") or []
    if ctx:
        print("\n--CONVERSATION--")
        for t in ctx:
            print(f"[{t.get('role','?')}] {str(t.get('content',''))[:600]}")
    print("\n--CRITERION--\n" + r["criterion"])
    resp = r["candidate_response"].strip()
    print(f"\n--RESPONSE ({len(resp)} chars)--\n" + resp[:1600])
    if len(resp) > 1600:
        print("...[truncated]")
    for n, p in r["proposals"].items():
        print(f"\n--{n}: {p.get('label')}--")
        print("for:", str(p.get('evidence_for') or '')[:300])
        print("against:", str(p.get('evidence_against') or '')[:300])
        print("why:", str(p.get('rationale') or '')[:300])
