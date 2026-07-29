"""Apply controlled BiGGen edits to planning/travel_plan instances.

Rule set ``biggen_edit_v1`` (deterministic, auditable):

  Rule 1 (input):  fold the ``Optional:`` requirement list into ``Must Have:``
      and delete the ``Optional:`` line, so every listed experience is required.
  Rule 2 (rubric): replace the five score anchors with a generic count-based
      template (5 = all required experiences, 4 = missing exactly one, ...) and
      strip the "optional" clause from ``criteria``.

Only ``travel_plan`` instances that actually ship an ``Optional:`` block are
edited; every other instance passes through untouched. Each edited record keeps
its pre-edit input, criteria, and rubric under ``edit_provenance`` so the change
is reviewable and reversible.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "BiGGen" / "biggen_source.json"
OUT_DIR = ROOT / "data" / "BiGGen"
OUT_EDITED = OUT_DIR / "travel_plan_edited.jsonl"

POLICY = "biggen_edit_v1"
RULE = "merge_optional_into_required + count_based_anchor_reframe"

# Generic count-based anchors (same wording for every edited instance).
NEW_ANCHORS = {
    "score1_description": (
        "The response misses more than two of the required experiences, or "
        "fundamentally ignores the stated constraints (e.g., transportation or "
        "duration), showing a lack of detailed planning."
    ),
    "score2_description": (
        "The response includes the required experiences but ignores a stated "
        "constraint, or the itinerary is unrealistic or poorly structured "
        "(e.g., significant backtracking or timing issues)."
    ),
    "score3_description": (
        "The response adheres to the stated constraints with a realistic "
        "itinerary, but is missing two of the required experiences, or includes "
        "all of them with notable feasibility issues."
    ),
    "score4_description": (
        "The response adheres to the stated constraints with a realistic and "
        "efficient itinerary, but is missing exactly one required experience or "
        "has one minor feasibility gap."
    ),
    "score5_description": (
        "The response presents a well-thought-out, efficient, and realistic "
        "itinerary that includes all required experiences within the stated "
        "constraints."
    ),
}

OPT_LINE_RE = re.compile(r"(?im)^[ \t]*-?[ \t]*optional[ \t]*:[ \t]*(.+?)[ \t]*$")
MUST_LINE_RE = re.compile(r"(?im)^([ \t]*-?[ \t]*must[ -]?have[ \t]*:[ \t]*)(.+?)[ \t]*$")
OPT_LINE_STRIP_RE = re.compile(r"(?im)^[ \t]*-?[ \t]*optional[ \t]*:[ \t]*.+?[ \t]*(?:\r?\n|$)")


def merge_optional(input_text: str):
    """Rule 1: append Optional items to Must Have, drop the Optional line."""
    opt = OPT_LINE_RE.search(input_text)
    if not opt:
        return None
    opt_items = opt.group(1).strip()
    must = MUST_LINE_RE.search(input_text)
    if not must:
        return None  # unexpected shape; leave for manual review
    new_must = f"{must.group(1)}{must.group(2).rstrip().rstrip(',')}, {opt_items}"
    text = input_text[: must.start()] + new_must + input_text[must.end():]
    text = OPT_LINE_STRIP_RE.sub("", text, count=1)
    text = text.rstrip("\n") + ("\n" if input_text.endswith("\n") else "")
    return {"input": text, "opt_items": opt_items}


def fix_criteria(criteria: str) -> str:
    """Rule 2: drop every 'optional ...' mention; 'must-have(s)' -> 'required experiences'.

    Handles the several phrasings the 8 instances use for the optional list
    (``optional activities`` / ``experiences`` / ``visits`` / ``items``) and the
    trailing ``while also considering the optional items`` clause, then tidies up
    the connectors and punctuation left behind.
    """
    c = criteria
    # normalize must-have(s): plural noun -> "required experiences";
    # adjective ("must-have activities") -> "required".
    c = re.sub(r"(?i)must[- ]?have experiences", "required experiences", c)
    c = re.sub(r"(?i)must[- ]?haves", "required experiences", c)
    c = re.sub(r"(?i)must[- ]?have", "required", c)
    # trailing "while also considering the optional items" clause
    c = re.sub(r"(?i),?\s*while also considering the optional items", "", c)
    # "..., optional X, ..."  ->  "..., ..."
    c = re.sub(r"(?i),\s*optional\s+\w+\s*,", ",", c)
    # "... and optional X"     ->  "..."
    c = re.sub(r"(?i)\s+and\s+optional\s+\w+", "", c)
    # "..., optional X"        ->  "..."
    c = re.sub(r"(?i),\s*optional\s+\w+", "", c)
    # any remaining "optional X"
    c = re.sub(r"(?i)\boptional\s+\w+\b", "", c)
    # tidy connectors / punctuation
    c = re.sub(r"\s*,\s*,", ",", c)
    c = re.sub(r"\s{2,}", " ", c)
    c = re.sub(r"\s+([,?])", r"\1", c)
    c = re.sub(r",\s*\?", "?", c)
    c = c.strip().rstrip(",").strip()
    if not c.endswith("?"):
        c += "?"
    return c


def main() -> int:
    records = json.loads(SRC.read_text(encoding="utf-8"))
    edited = []
    warnings = []

    for rec in records:
        if rec.get("task") != "travel_plan":
            continue
        merged = merge_optional(rec["input"])
        if merged is None:
            continue  # no Optional block -> untouched (e.g. _1, _2)

        orig_input = rec["input"]
        orig_criteria = rec["score_rubric"]["criteria"]
        orig_rubric = dict(rec["score_rubric"])

        new_rec = json.loads(json.dumps(rec))  # deep copy
        new_rec["input"] = merged["input"]
        new_rec["score_rubric"]["criteria"] = fix_criteria(orig_criteria)
        for key, val in NEW_ANCHORS.items():
            new_rec["score_rubric"][key] = val
        new_rec["edit_provenance"] = {
            "policy": POLICY,
            "rule": RULE,
            "task": "travel_plan",
            "folded_optional_items": merged["opt_items"],
            "original_input": orig_input,
            "original_criteria": orig_criteria,
            "original_score_rubric": orig_rubric,
        }

        # sanity: no residual "optional" in edited surfaces
        blob = (
            new_rec["input"]
            + new_rec["score_rubric"]["criteria"]
            + "".join(
                new_rec["score_rubric"][k] for k in NEW_ANCHORS
            )
        )
        if re.search(r"(?i)optional", blob):
            warnings.append(f"{rec['id']}: residual 'optional' after edit")

        edited.append(new_rec)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_EDITED.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in edited:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"edited {len(edited)} travel_plan instances -> {OUT_EDITED.relative_to(ROOT)}")
    for rec in edited:
        ep = rec["edit_provenance"]
        must_before = MUST_LINE_RE.search(ep["original_input"])
        must_after = MUST_LINE_RE.search(rec["input"])
        print(f"\n--- {rec['id']}  (folded: {ep['folded_optional_items']}) ---")
        print(f"  Must Have (before): {must_before.group(2) if must_before else '?'}")
        print(f"  Must Have (after) : {must_after.group(2) if must_after else '?'}")
        print(f"  Optional line present after edit: {bool(OPT_LINE_RE.search(rec['input']))}")
        print(f"  criteria (after)  : {rec['score_rubric']['criteria']}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")
        return 1
    print("\nno residual 'optional' references in edited instances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
