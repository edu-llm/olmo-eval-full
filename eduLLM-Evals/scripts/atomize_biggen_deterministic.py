"""Deterministic atomic-criterion overlay for the 18 controlled-edit BiGGen instances.

The `planning/travel_plan` (8) and `tool_usage/item_recommendation` (10) instances
carry curated, structured requirement lists (from `biggen_edit_v1`), so their single
holistic criterion can be split into atomic binary criteria WITHOUT an LLM -- one
checkable claim per stated requirement, fully auditable and reproducible.

Output overlay (`atoms_deterministic.jsonl`), one line per instance:

    {"id": <source id>, "method": <str>, "atoms": [<criterion text>, ...]}

Atoms are equal-weight (each becomes one binary rubric row at ingest, like
InFoBench/IFEval). Consumed by ingest_biggen.py alongside the LLM overlay.

    uv run python scripts/edit_biggen_travel_plan.py
    uv run python scripts/edit_biggen_item_recommendation.py
    uv run python scripts/atomize_biggen_deterministic.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "BiGGen"
TRAVEL = DATA_DIR / "travel_plan_edited.jsonl"
ITEMREC = DATA_DIR / "item_recommendation_edited.jsonl"
OUT = DATA_DIR / "atoms_deterministic.jsonl"

METHOD_TP = "deterministic_travel_plan_v1"
METHOD_IR = "deterministic_item_recommendation_v1"

# Requirements block: "- <Label>: <value>" lines (Total Duration, Transportation, Must Have).
REQ_LINE = re.compile(r"(?m)^[ \t]*-[ \t]*([^:\n]+?):[ \t]*(.+?)[ \t]*$")
# Trailing special section: "<Header>:\n- <value>" (Food Constraints, Special Requests, ...).
SECTION = re.compile(r"(?m)^([A-Z][A-Za-z /]+):[ \t]*\n[ \t]*-[ \t]*(.+?)[ \t]*$")

FEASIBILITY = (
    "The itinerary is realistic and efficient, without significant backtracking "
    "or timing issues."
)


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}; run the edit scripts first")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_items(value: str) -> list[str]:
    """Comma-separated requirement list -> trimmed items (parentheticals kept intact)."""
    return [p.strip() for p in value.split(",") if p.strip()]


def travel_atoms(rec: dict) -> list[str]:
    text = rec["input"]
    labels = {m.group(1).strip().lower(): m.group(2).strip() for m in REQ_LINE.finditer(text)}
    must = labels.get("must have")
    if not must:
        raise SystemExit(f"{rec['id']}: no 'Must Have' line to atomize")

    atoms = [f"The itinerary includes {item}." for item in split_items(must)]
    if "transportation" in labels:
        atoms.append(
            f"The itinerary uses only the stated transportation ({labels['transportation']})."
        )
    if "total duration" in labels:
        atoms.append(f"The itinerary fits the stated duration ({labels['total duration']}).")

    # single trailing special section (food/eco/photography/etc.), if any
    for header, value in SECTION.findall(text):
        if header.strip().lower() in {"requirements", "destination"}:
            continue
        atoms.append(f"The itinerary satisfies the stated {header.strip().lower()}: {value.strip()}.")

    atoms.append(FEASIBILITY)
    return atoms


def item_atoms(rec: dict) -> list[str]:
    ep = rec["edit_provenance"]
    tool = ep["tool"]
    reqs = ep["required_elements"]
    atoms = [
        f"The response only generates the tool-call argument in the {tool}'s specified "
        f"format, rather than guessing results or answering directly."
    ]
    atoms += [f"The tool-call argument captures {req}." for req in reqs]
    return atoms


def main() -> int:
    out: list[dict] = []
    for rec in load(TRAVEL):
        out.append({"id": rec["id"], "method": METHOD_TP, "atoms": travel_atoms(rec)})
    for rec in load(ITEMREC):
        out.append({"id": rec["id"], "method": METHOD_IR, "atoms": item_atoms(rec)})

    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {OUT.relative_to(ROOT)}  ({len(out)} instances)")
    for r in out:
        print(f"\n--- {r['id']}  [{r['method']}]  ({len(r['atoms'])} atoms) ---")
        for i, a in enumerate(r["atoms"], 1):
            print(f"  c{i:02d}: {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
