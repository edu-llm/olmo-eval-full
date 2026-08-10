"""Pretty-print gold packets for human review (blinded: no judge/Qwen verdicts shown)."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
idx = [int(a) for a in sys.argv[2:]] or [0]
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
for i in idx:
    r = rows[i]
    print("=" * 90)
    print(f"[{r['gold_case_id']}]  capability={r['capability']}  task={r['task']}  model={r['model']}")
    print(f"concordant={r['concordant']}  recommendation={r['recommendation']}")
    print(f"\nCRITERION:\n  {r['criterion']}")
    ctx = r.get("conversation_context") or []
    if ctx:
        print(f"\nCONVERSATION ({len(ctx)} turns): shown in full during review")
    print(f"\nRESPONSE (first 400 chars):\n  {r['candidate_response'][:400].strip()}")
    for n, p in r["proposals"].items():
        print(f"\n--- proposer {n}: {p.get('label')}  (confidence {p.get('confidence')}, status {p.get('status')})")
        print(f"    required_parts : {p.get('required_parts')}")
        print(f"    evidence_for   : {str(p.get('evidence_for'))[:220]}")
        print(f"    evidence_against: {str(p.get('evidence_against'))[:220]}")
        print(f"    rationale      : {str(p.get('rationale'))[:220]}")
