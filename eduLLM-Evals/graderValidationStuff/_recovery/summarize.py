import json
from pathlib import Path

rep = json.loads((Path(__file__).resolve().parent / "reliability_report.json").read_text(encoding="utf-8"))

def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d

judges = rep.get("judges") or rep.get("per_judge") or {}
if isinstance(judges, list):
    judges = {j.get("judge") or j.get("name"): j for j in judges}

print("TOP-LEVEL KEYS:", list(rep.keys()))
for name, j in judges.items():
    print("\n==============================", name, "==============================")
    can = g(j, "canonical") or {}
    print("  macro_f1            :", g(can, "macro_f1"))
    print("  weighted_f1         :", g(can, "weighted_f1"))
    print("  mcc                 :", g(can, "mcc"))
    print("  coverage            :", g(can, "coverage"))
    # false pass / confusion if present
    for k in ("false_pass_rate", "pass_recall", "fail_recall", "accuracy", "balanced_accuracy", "confusion"):
        if k in can:
            print(f"  {k:20}:", can[k])
    print("  critical_failure_sensitivity:", g(j, "critical_failure_sensitivity") or g(can, "critical_failure_sensitivity"))
    # per skill
    skills = g(j, "primary_skill_macro_f1") or g(j, "skills") or g(can, "primary_skill_macro_f1")
    print("  per-skill macro_f1  :", skills)
    tr = g(j, "test_retest") or {}
    print("  test_retest worst strict:", g(tr, "worst_pairwise_strict_agreement"), " kappa:", g(tr, "worst_pairwise_kappa") or g(tr, "min_kappa"))
    pr = g(j, "prompt") or g(j, "prompt_variants") or {}
    print("  prompt worst flip   :", g(pr, "worst_variant_flip_rate"))
    acc = g(j, "acceptance") or g(j, "acceptance_report") or {}
    if acc:
        print("  ACCEPTANCE:")
        for gate, v in acc.items():
            print("     ", gate, "->", v)
