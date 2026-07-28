"""Scan the whole curated bank for criteria that explicitly FORBID an equivalent
answer or demand a mandatory verbatim phrase (the only genuine rigid targets)."""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

curated = [json.loads(l) for l in open("data/curated/rubrics_qmatrix_curated.jsonl", encoding="utf-8")]

# "forbid an equivalent" / "mandatory verbatim" markers. Exclude withhold (must not / without saying).
FORBID = re.compile(
    r"(and not just\b|not (?:just|only|merely) the\b|not simply\b|"
    r"must use the exact\b|the exact (?:phrase|wording|words)\b|word[- ]for[- ]word|\bverbatim\b|"
    r"say exactly\b|state exactly\b|use the (?:exact )?(?:phrase|wording)\b|"
    r"exactly as (?:written|follows|stated)\b)",
    re.I,
)
WITHHOLD = re.compile(r"\bmust not\b|\bshould not\b|without (?:actually |explicitly )?(?:saying|revealing|giving|stating)", re.I)

hits = []
for c in curated:
    t = c.get("criterion", "") or ""
    if FORBID.search(t) and not WITHHOLD.search(t):
        hits.append(c)

print(f"explicit forbid-equivalent / mandatory-verbatim (non-withhold): {len(hits)}\n")
for c in hits:
    m = FORBID.search(c["criterion"])
    print(f"--- {c['criterion_id']} | {c.get('criticality')} | marker='{m.group(0)}'")
    print("   ", c["criterion"][:320])
    print()
