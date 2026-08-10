"""Render packets for in-chat adjudication (blinded: no judge/Qwen verdicts).

Usage:
  python review.py PACKETS [--discordant] [--ids id1 id2 ...] [--resp-chars N]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("packets", type=Path)
ap.add_argument("--discordant", action="store_true")
ap.add_argument("--ids", nargs="*", default=[])
ap.add_argument("--resp-chars", type=int, default=700)
args = ap.parse_args()

rows = [json.loads(l) for l in args.packets.read_text(encoding="utf-8").splitlines() if l.strip()]
by_id = {r["gold_case_id"]: r for r in rows}
if args.ids:
    sel = [by_id[i] for i in args.ids]
elif args.discordant:
    sel = [r for r in rows if not r["concordant"]]
else:
    sel = rows

for r in sel:
    print("\n" + "=" * 92)
    print(f"[{r['gold_case_id']}]  stratum={r.get('stratum', r.get('capability'))}  criticality={r.get('criticality')}  task={r.get('task','')}")
    print(f"CRITERION: {r['criterion']}")
    ctx = r.get("conversation_context") or []
    if ctx:
        print(f"\nCONVERSATION ({len(ctx)} turns):")
        for t in ctx:
            print(f"  [{t.get('role','?')}] {str(t.get('content',''))[:300]}")
    resp = r["candidate_response"].strip()
    print(f"\nRESPONSE ({len(resp)} chars; showing {args.resp_chars}):\n  {resp[:args.resp_chars]}")
    if len(resp) > args.resp_chars:
        print("  [...truncated — ask for full response if needed]")
    for n, p in r["proposals"].items():
        print(f"\n  >> {n}: {str(p.get('label')).upper()}  (conf {p.get('confidence')}, {p.get('status')})")
        if p.get("evidence_for"):
            print(f"     FOR    : {str(p.get('evidence_for'))[:260]}")
        if p.get("evidence_against"):
            print(f"     AGAINST: {str(p.get('evidence_against'))[:260]}")
        if p.get("rationale"):
            print(f"     WHY    : {str(p.get('rationale'))[:260]}")
    print(f"\n  RECOMMENDATION: {r['recommendation']}  (concordant={r['concordant']})")
