"""Merge per-batch op files into the master curation spec.

Each file in curation/tranche/*.json is a JSON list of op dicts (same schema as
the master spec's "ops": {"id","op",...}). This merges them into
curation/pilot_edits.json, skipping any id already present (idempotent), so
parallel subagents can each write their own batch file without conflicts.

Usage: python scripts/merge_ops.py
"""
from __future__ import annotations

import json
from pathlib import Path

SPEC = Path("curation/pilot_edits.json")
TRANCHE = Path("curation/tranche")


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    existing = {o["id"] for o in spec["ops"]}
    added, skipped, files = 0, 0, 0
    for f in sorted(TRANCHE.glob("*.json")):
        if f.name.startswith("worklist"):
            continue  # canonical id list, not an op file
        files += 1
        ops = json.loads(f.read_text(encoding="utf-8"))
        new = []
        for o in ops:
            if o["id"] in existing:
                skipped += 1
                continue
            existing.add(o["id"])
            new.append(o)
            added += 1
        spec["ops"].extend(new)
        print(f"  {f.name}: +{len(new)} ops")
    SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {files} file(s): +{added} new ops, {skipped} skipped (already present).")
    print(f"Total ops now: {len(spec['ops'])}")


if __name__ == "__main__":
    main()
