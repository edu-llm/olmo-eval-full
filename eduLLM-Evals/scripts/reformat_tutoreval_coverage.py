"""Reformat the TutorEval skeleton into scenario-level coverage items (experimental).

TutorEval's source ``key_points`` is a decomposed *reference answer*, graded
holistically for coverage — not one pass/fail per bullet. The current ingest
(``ingest_tutoreval.py``) splits each bullet into an independent binary criterion,
which produces fragment "criteria" (``yes``, ``MNIST``, ``friction``) and shattered
enumerations that are meaningless to grade standalone.

This script regroups to **one gradeable item per scenario**:
- ``criterion`` becomes a numbered key-point checklist (the judge reads only this
  field — see ``tutor_cat/judge.py``), so bullets are graded for coverage *in the
  context of the question*.
- the raw per-bullet strings are preserved in ``expected_evidence`` (currently
  empty in the skeleton) as structured provenance.

The originals are never modified; output goes to ``*_coverage.{jsonl,json}``.
Skill/IRT fields stay null (skeleton), same as the per-bullet ingest — a q-matrix
pass fills them later, now attaching one label per scenario item.

Run:
    python scripts/reformat_tutoreval_coverage.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "TutorEval"

# Same key sets/order as ingest_tutoreval.py so the outputs load identically.
SCENARIO_KEYS = [
    "scenario_id", "source_id", "use_case", "subject", "subset", "grade_band",
    "modality", "prompt", "conversation_context", "reference_solution",
    "book_condition", "misleading_question", "answer_in_chapter",
    "criterion_ids", "source", "split", "version",
]
RUBRIC_KEYS = [
    "criterion_id", "scenario_id", "criterion", "expected_evidence",
    "scoring_type", "score_anchors", "primary_skill", "q_mapping", "q_rationale",
    "criticality", "objectivity", "explicitness", "source", "status", "version",
]

CHECKLIST_PREAMBLE = (
    "A complete response correctly addresses all of the following key points:"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_checklist(bullets: list[str]) -> str:
    """Render the per-scenario key points as one gradeable coverage criterion.

    A single-bullet scenario is used verbatim (already a standalone criterion);
    multi-bullet scenarios become a numbered checklist under a coverage preamble.
    """
    if len(bullets) == 1:
        return bullets[0]
    lines = [CHECKLIST_PREAMBLE]
    lines += [f"{i}. {b}" for i, b in enumerate(bullets, 1)]
    return "\n".join(lines)


def build() -> tuple[list[dict], list[dict]]:
    scenarios = read_jsonl(OUT_DIR / "scenarios.jsonl")
    rubrics = read_jsonl(OUT_DIR / "rubrics.jsonl")

    # Group the per-bullet criteria back under their scenario, preserving order.
    bullets_by_scenario: dict[str, list[str]] = {}
    for r in rubrics:
        bullets_by_scenario.setdefault(r["scenario_id"], []).append(r["criterion"])

    new_scenarios: list[dict] = []
    new_rubrics: list[dict] = []
    for s in scenarios:
        sid = s["scenario_id"]
        bullets = bullets_by_scenario.get(sid, [])
        if not bullets:
            continue  # scenario with no criteria (shouldn't happen post-ingest)
        cid = f"{sid}_cov"

        new_scenarios.append({**{k: s[k] for k in SCENARIO_KEYS if k != "criterion_ids"},
                              "criterion_ids": [cid]})
        # Reorder to the canonical key order.
        new_scenarios[-1] = {k: new_scenarios[-1][k] for k in SCENARIO_KEYS}

        new_rubrics.append({
            "criterion_id": cid,
            "scenario_id": sid,
            "criterion": build_checklist(bullets),
            "expected_evidence": bullets,          # raw key points preserved (#2)
            "scoring_type": "binary",
            "score_anchors": None,
            "primary_skill": None,
            "q_mapping": None,
            "q_rationale": None,
            "criticality": None,
            "objectivity": None,
            "explicitness": None,
            "source": rubrics[0]["source"],
            "status": "approved",
            "version": rubrics[0]["version"],
        })

    return new_scenarios, new_rubrics


def write(name: str, records: list[dict]) -> None:
    jsonl_path = OUT_DIR / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json_path = OUT_DIR / f"{name}.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"wrote {jsonl_path.relative_to(ROOT)} + .json  ({len(records)} rows)")


def main() -> None:
    scenarios, rubrics = build()
    write("scenarios_coverage", scenarios)
    write("rubrics_coverage", rubrics)

    counts = [len(r["expected_evidence"]) for r in rubrics]
    print(f"\n{len(scenarios)} scenario items (was {sum(counts)} per-bullet criteria)")
    print(f"key points per item: min={min(counts)} max={max(counts)} "
          f"mean={sum(counts) / len(counts):.1f}")
    print("\n--- examples ---")
    for r in rubrics[:2] + [r for r in rubrics if len(r["expected_evidence"]) >= 3][:1]:
        print(f"\n[{r['criterion_id']}]  ({len(r['expected_evidence'])} key points)")
        print(r["criterion"])


if __name__ == "__main__":
    main()
