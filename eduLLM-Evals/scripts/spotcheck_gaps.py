import json, sys
sys.stdout.reconfigure(encoding="utf-8")
cur = {}
for l in open("data/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8"):
    c = json.loads(l); cur.setdefault(c["scenario_id"], []).append(c)
scen = {s["scenario_id"]: s for s in (json.loads(l) for l in open("data/curated/scenarios_curated.jsonl", encoding="utf-8"))}

checks = {
    "hint_scaffold MISSING (hint_generation)": ["tb_0499", "tb_0518", "tb_0520"],
    "feedback error_ident MISSING": ["tb_0335", "tb_0339"],
    "adaptive acknowledge MISSING": ["tb_0003", "tb_0004"],
}
for label, sids in checks.items():
    print("\n" + "#"*80 + f"\n{label}")
    for sid in sids:
        s = scen.get(sid, {})
        print(f"\n=== {sid} | use_case={s.get('use_case')} | subj={s.get('subject')}")
        print(f"    PROMPT: {(s.get('prompt') or '')[:180]}")
        for c in cur.get(sid, []):
            print(f"    [{c.get('primary_skill')}|{c.get('criticality')}] {c['criterion'][:150]}")
