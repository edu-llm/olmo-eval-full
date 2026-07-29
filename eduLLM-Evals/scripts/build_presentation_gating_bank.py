#!/usr/bin/env python3
"""Build a *gating* presentation rubric bank for a targeted judge run.

The curated bank marks every per-scenario ``style_surface`` presentation
criterion ``optional: true`` (non-gating), so the calibration ``prepare`` step
-- which selects criteria via ``optional is not True`` -- skips them and they
were never graded. To test presentation as a candidate axis we need verdicts,
which means flipping those criteria to gating so ``prepare`` picks them up.

This is fully reversible: it reads the official curated bank read-only and
writes a SEPARATE experimental bank. The official bank is never modified. It
selects ONLY the 662 ``dimension == style_surface`` criteria (one per scenario)
and excludes the 3 ``rescope_optional`` conditional-formula criteria.

Usage:
    python scripts/build_presentation_gating_bank.py            # write the bank
    python scripts/build_presentation_gating_bank.py --check    # counts only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "TutorBench" / "curated" / "rubrics_qmatrix_curated.jsonl"
OUT = ROOT / "data" / "TutorBench" / "experimental" / "rubrics_presentation_gating.jsonl"

GATING_GUIDANCE = (
    "Presentation/style criterion, graded pass/fail (gating for this targeted "
    "run). PASS if the response follows the stated tutoring presentation "
    "conventions for the aspects that apply to it (persona / structure / math / "
    "code); FAIL if it clearly violates them. Judge only the presentation "
    "aspects named in the criterion; do not require aspects that do not apply."
)


def is_presentation(rec: dict) -> bool:
    return rec.get("optional") is True and rec.get("dimension") == "style_surface"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report counts, do not write")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"curated bank not found: {SRC}")

    selected: list[dict] = []
    scenarios: set[str] = set()
    total = 0
    for line in SRC.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        rec = json.loads(line)
        if not is_presentation(rec):
            continue
        rec["optional"] = False
        rec["judge_guidance"] = GATING_GUIDANCE
        rec.setdefault("curation", {})
        # provenance breadcrumb so this is traceable / reversible
        rec["curation"]["presentation_gating"] = {
            "flipped_from_optional": True,
            "source_bank": "data/TutorBench/curated/rubrics_qmatrix_curated.jsonl",
        }
        selected.append(rec)
        scenarios.add(rec.get("scenario_id"))

    n = len(selected)
    dup = n - len(scenarios)
    print(f"curated bank records read:        {total}")
    print(f"style_surface presentation picked: {n}")
    print(f"distinct scenarios covered:        {len(scenarios)}")
    print(f"scenarios with >1 presentation:    {dup}")
    all_gating = all(r.get("optional") is False for r in selected)
    print(f"all optional=false:                {all_gating}")

    if args.check:
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in selected:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {n} records -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
