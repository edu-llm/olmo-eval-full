"""Reframe tool_usage/item_recommendation anchors (levels 3-5) for BiGGen.

Rule set ``biggen_edit_v1`` (deterministic, auditable):

  Rule 3 (rubric, anchors 2-5): replace the quality/expertise gradient with an
      enumerated, count-based checklist over each prompt's *stated* requirements.
      Level 5 = all stated requirements captured with the correct tool format;
      level 4 = exactly one omitted; level 3 = two omitted or a format/parameter
      error; level 2 = most omitted or the required format misused. This removes
      the level-5 "expert extras / extra unrequested rationale" gate and the
      subjective 3<->4 "minor details" line, and keeps the 2<->3<->4 boundaries
      disjoint.

Level 1 is left untouched (it already encodes the wrong-tool / no-tool-format
failure). Inputs and ``criteria`` are unchanged: none of the 10 prompts contain
optional language, and the criteria never gated on unrequested extras.

Requirement checklists are curated per instance from the stated instruction (the
text after "Here is the instruction you should follow:"), not from the old
level-5 anchor, so the "beyond the request" polish is excluded by construction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "BiGGen" / "biggen_source.json"
OUT_DIR = ROOT / "data" / "BiGGen"
OUT_EDITED = OUT_DIR / "item_recommendation_edited.jsonl"

POLICY = "biggen_edit_v1"
RULE = "count_based_anchor_reframe_L2toL5 (enumerated stated requirements)"

# id -> (tool name, [stated requirements]) curated from the instruction text.
REQS = {
    "tool_usage_item_recommendation_0": (
        "Tech Gadgets Tool",
        ["a smartphone (gadget type)", "photography / camera quality", "a price under $500"],
    ),
    "tool_usage_item_recommendation_1": (
        "Travel Destinations Tool",
        ["family-friendly suitability", "a destination located in Europe"],
    ),
    "tool_usage_item_recommendation_2": (
        "Educational Books Tool",
        ["high user ratings", "coverage of advanced Python programming"],
    ),
    "tool_usage_item_recommendation_3": (
        "Cooking Appliances Tool",
        ["healthy meal preparation", "quick / time-efficient operation"],
    ),
    "tool_usage_item_recommendation_4": (
        "Tech Gadgets Tool",
        [
            "portability",
            "large storage for academic books and papers",
            "focus on environmental-science material (climate change, renewable energy)",
            "an eye-friendly display for long reading",
            "budget-friendliness",
        ],
    ),
    "tool_usage_item_recommendation_5": (
        "Travel Destinations Tool",
        ["extreme-sports / adventure opportunities", "historical-exploration opportunities"],
    ),
    "tool_usage_item_recommendation_6": (
        "Tech Gadgets Tool",
        [
            "space efficiency for limited urban space",
            "plant-health optimization",
            "automatic watering",
            "support for a variety of plants",
        ],
    ),
    "tool_usage_item_recommendation_7": (
        "Tech Gadgets Tool",
        [
            "portable, solar-powered operation",
            "charging multiple devices",
            "serving as a light source",
            "suitability for camping",
        ],
    ),
    "tool_usage_item_recommendation_8": (
        "Cooking Appliances Tool",
        [
            "combined cooking, baking, and steaming functions",
            "energy efficiency",
            "user-friendliness",
            "suitability for studio apartments",
        ],
    ),
    "tool_usage_item_recommendation_9": (
        "Educational Books Tool",
        [
            "a focus on language learning",
            "design for children",
            "gamification elements to boost engagement and retention",
        ],
    ),
}


def make_anchors(tool: str, reqs: list[str]) -> dict[str, str]:
    joined = ", ".join(reqs)
    return {
        "score2_description": (
            f"The response uses the {tool} but omits most of the required "
            f"elements ({joined}), or misuses the required format, producing an "
            f"incomplete or largely irrelevant query."
        ),
        "score3_description": (
            f"The response uses the {tool} but its argument omits two of the "
            f"required elements ({joined}), or includes them with a format or "
            f"parameter error that would degrade the tool's results."
        ),
        "score4_description": (
            f"The response uses the {tool}'s required format correctly and "
            f"captures the required elements but omits exactly one of: {joined}."
        ),
        "score5_description": (
            f"The response uses the {tool}'s required format correctly and its "
            f"argument captures all required elements: {joined}."
        ),
    }


def main() -> int:
    records = json.loads(SRC.read_text(encoding="utf-8"))
    edited = []
    seen = set()

    for rec in records:
        if rec.get("task") != "item_recommendation":
            continue
        rid = rec["id"]
        if rid not in REQS:
            print(f"WARNING: no checklist for {rid}; skipped")
            continue
        seen.add(rid)
        tool, reqs = REQS[rid]
        new_anchors = make_anchors(tool, reqs)

        orig_rubric = dict(rec["score_rubric"])
        new_rec = json.loads(json.dumps(rec))
        for key, val in new_anchors.items():
            new_rec["score_rubric"][key] = val
        new_rec["edit_provenance"] = {
            "policy": POLICY,
            "rule": RULE,
            "task": "item_recommendation",
            "tool": tool,
            "required_elements": reqs,
            "levels_changed": [
                "score2_description",
                "score3_description",
                "score4_description",
                "score5_description",
            ],
            "original_score_rubric": orig_rubric,
        }
        edited.append(new_rec)

    missing = set(REQS) - seen
    if missing:
        print(f"WARNING: checklist ids not found in source: {sorted(missing)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_EDITED.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in edited:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"edited {len(edited)} item_recommendation instances -> {OUT_EDITED.relative_to(ROOT)}")
    for rec in edited:
        rb = rec["score_rubric"]
        print(f"\n--- {rec['id']} ({rec['edit_provenance']['tool']}) ---")
        print(f"  reqs: {rec['edit_provenance']['required_elements']}")
        print(f"  L2: {rb['score2_description']}")
        print(f"  L3: {rb['score3_description']}")
        print(f"  L4: {rb['score4_description']}")
        print(f"  L5: {rb['score5_description']}")
    return 0 if len(edited) == len(REQS) else 1


if __name__ == "__main__":
    sys.exit(main())
