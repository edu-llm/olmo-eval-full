"""Convert the BiGGen-Bench dataset into Scenario + Rubric Schema JSONL.

Source: https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench
(frozen snapshot at data/BiGGen/biggen_source.json, 765 instances).

BiGGen ships one 5-point (polytomous) instance-level rubric per prompt. This repo
scores binary, so each instance is **threshold-collapsed** into a single binary
criterion: the five level descriptions are preserved in `score_anchors` together
with an explicit `pass_threshold`, and the judge grades on the 1-5 scale and
thresholds to pass/fail at grading time (config `result_pass_threshold`, default
4 -> a response passes iff it reaches "all requirements met" or "missing exactly
one"). One criterion per scenario, so criterion ids are always `<sid>_c01`.

Decisions baked in here (see the handoff discussion):
  - multilingual: BiGGen exposes multilingual as a *capability* (70 instances),
    not a per-row language tag, so it is EXCLUDED at ingest (held for later).
    695 English instances remain.
  - reference answers: HELD. `reference_solution` is null; the native BiGGen
    reference is preserved verbatim under `native_reference_answer` (provenance,
    not yet wired into judging) so the decision stays reversible.
  - skill axis: UNDECIDED. `q_mapping` and `primary_skill` are null; MIRT
    difficulty/discrimination/irt_params are NOT emitted (appended later by
    assign_irt_params.py once the skill axis is chosen), matching InFoBench.
  - controlled edits: the planning/travel_plan and tool_usage/item_recommendation
    anchor/input reframes (biggen_edit_v1) are overlaid from their *_edited.jsonl
    files, and each edited row carries its `edit_provenance` block.

Run:
    uv run python scripts/edit_biggen_travel_plan.py
    uv run python scripts/edit_biggen_item_recommendation.py
    uv run python scripts/ingest_biggen.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "BiGGen"
SRC = DATA_DIR / "biggen_source.json"
EDIT_FILES = [
    DATA_DIR / "travel_plan_edited.jsonl",
    DATA_DIR / "item_recommendation_edited.jsonl",
]

SOURCE_URL = "https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench"
SPLIT = "calibration"   # pipeline-role label, not an upstream split
VERSION = "1.0"
EXCLUDE_CAPABILITY = "multilingual"   # held for later; ingested English-only
PASS_THRESHOLD = 4      # threshold-collapse pass bar over the 1-5 anchor scale
ANCHOR_LEVELS = [1, 2, 3, 4, 5]

SCENARIO_KEYS = [
    "scenario_id", "source_id", "use_case", "subject", "task", "grade_band",
    "modality", "system_prompt", "prompt", "conversation_context",
    "reference_solution", "native_reference_answer", "criterion_ids",
    "source", "split", "version",
]
RUBRIC_KEYS = [
    "criterion_id", "scenario_id", "criterion", "expected_evidence",
    "scoring_type", "score_anchors", "primary_skill", "q_mapping", "q_rationale",
    "capability", "task", "criticality", "objectivity", "explicitness",
    "edit_provenance", "source", "status", "version",
]

# json.dumps(ensure_ascii=False) leaves U+2028/U+2029 raw; BiGGen carries non-Latin
# text, so escape them back (JS JSON.parse / str.splitlines treat them as breaks).
LINE_SEPS = {0x2028: "\\u2028", 0x2029: "\\u2029"}


def scenario_id(index: int) -> str:
    return f"bgb_{index:04d}"


def build_score_anchors(rubric: dict) -> dict:
    """Preserve the 5 polytomous levels plus the binary-collapse pass rule."""
    return {
        "scale_min": ANCHOR_LEVELS[0],
        "scale_max": ANCHOR_LEVELS[-1],
        "pass_threshold": PASS_THRESHOLD,
        "levels": {str(n): rubric[f"score{n}_description"] for n in ANCHOR_LEVELS},
    }


def load_edits() -> dict[str, dict]:
    edits: dict[str, dict] = {}
    for path in EDIT_FILES:
        if not path.exists():
            raise SystemExit(f"missing edit overlay {path}; run the edit scripts first")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                edits[rec["id"]] = rec
    return edits


def build() -> tuple[list[dict], list[dict], dict]:
    source = json.loads(SRC.read_text(encoding="utf-8"))
    edits = load_edits()

    scenarios: list[dict] = []
    rubrics: list[dict] = []
    stats = {"excluded_multilingual": 0, "edits_applied": 0}
    index = 0

    for src in source:
        if src["capability"] == EXCLUDE_CAPABILITY:
            stats["excluded_multilingual"] += 1
            continue

        rec = edits.get(src["id"], src)          # overlay controlled edits
        edited = src["id"] in edits
        if edited:
            stats["edits_applied"] += 1

        sid = scenario_id(index)
        index += 1
        cid = f"{sid}_c01"
        rubric = rec["score_rubric"]

        scenarios.append({
            "scenario_id": sid,
            "source_id": rec["id"],
            "use_case": rec["capability"],
            "subject": rec["capability"],
            "task": rec["task"],
            "grade_band": None,
            "modality": "text",
            "system_prompt": rec.get("system_prompt") or "",
            "prompt": rec["input"],
            "conversation_context": [],
            "reference_solution": None,                       # held
            "native_reference_answer": rec.get("reference_answer") or "",
            "criterion_ids": [cid],
            "source": SOURCE_URL,
            "split": SPLIT,
            "version": VERSION,
        })

        rubrics.append({
            "criterion_id": cid,
            "scenario_id": sid,
            "criterion": rubric["criteria"],
            "expected_evidence": [],                          # references held
            "scoring_type": "binary",
            "score_anchors": build_score_anchors(rubric),
            "primary_skill": None,                            # skill axis TBD
            "q_mapping": None,                                # left null at ingest
            "q_rationale": "",
            "capability": rec["capability"],
            "task": rec["task"],
            "criticality": "standard",
            "objectivity": "",
            "explicitness": "",
            "edit_provenance": rec.get("edit_provenance"),
            "source": SOURCE_URL,
            "status": "approved",
            "version": VERSION,
        })

    return scenarios, rubrics, stats


def validate(scenarios: list[dict], rubrics: list[dict], edits: dict[str, dict]) -> list[str]:
    errs: list[str] = []

    for name, records, keys in (
        ("scenario", scenarios, SCENARIO_KEYS),
        ("rubric", rubrics, RUBRIC_KEYS),
    ):
        for r in records:
            if list(r.keys()) != keys:
                errs.append(f"{name} {list(r.values())[0]}: key set/order mismatch")
                break

    sids = [s["scenario_id"] for s in scenarios]
    cids = [r["criterion_id"] for r in rubrics]
    if len(set(sids)) != len(sids):
        errs.append("duplicate scenario_id")
    if len(set(cids)) != len(cids):
        errs.append("duplicate criterion_id")
    if len(scenarios) != len(rubrics):
        errs.append(f"scenario/rubric count mismatch: {len(scenarios)} vs {len(rubrics)}")

    declared = {c for s in scenarios for c in s["criterion_ids"]}
    if declared != set(cids):
        errs.append("criterion_ids / rubric id set mismatch")

    for s in scenarios:
        if not (s["prompt"] or "").strip():
            errs.append(f"{s['scenario_id']}: empty prompt")
        if len(s["criterion_ids"]) != 1:
            errs.append(f"{s['scenario_id']}: expected exactly one criterion")
    for r in rubrics:
        if not (r["criterion"] or "").strip():
            errs.append(f"{r['criterion_id']}: empty criterion")
        if r["q_mapping"] is not None or r["primary_skill"] is not None:
            errs.append(f"{r['criterion_id']}: q_mapping/primary_skill must be null at ingest")
        sa = r["score_anchors"]
        if not isinstance(sa, dict) or set(sa.get("levels", {})) != {"1", "2", "3", "4", "5"}:
            errs.append(f"{r['criterion_id']}: score_anchors missing 5 levels")
        elif any(not (sa["levels"][k] or "").strip() for k in sa["levels"]):
            errs.append(f"{r['criterion_id']}: blank score anchor level")
        if sa.get("pass_threshold") != PASS_THRESHOLD:
            errs.append(f"{r['criterion_id']}: unexpected pass_threshold")

    # every controlled edit must have landed (and not been dropped by the filter)
    edited_source_ids = {s["source_id"] for s in scenarios} & set(edits)
    if edited_source_ids != set(edits):
        errs.append(f"edits lost at ingest: {sorted(set(edits) - edited_source_ids)}")
    for r in rubrics:
        src_id = next(s["source_id"] for s in scenarios if s["scenario_id"] == r["scenario_id"])
        if (src_id in edits) != (r["edit_provenance"] is not None):
            errs.append(f"{r['criterion_id']}: edit_provenance presence mismatch")

    return errs


def write(name: str, records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_DIR / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False).translate(LINE_SEPS) + "\n")
    json_path = DATA_DIR / f"{name}.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    size = jsonl_path.stat().st_size / 1e6
    print(f"wrote {jsonl_path.relative_to(ROOT)} + .json  ({len(records)} rows, {size:.1f} MB)")


def main() -> None:
    scenarios, rubrics, stats = build()
    edits = load_edits()

    errs = validate(scenarios, rubrics, edits)
    if errs:
        print(f"VALIDATION FAILED ({len(errs)} issues):", file=sys.stderr)
        for e in errs[:20]:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"validation passed: {len(scenarios)} scenarios, {len(rubrics)} criteria")

    write("scenarios", scenarios)
    write("rubrics", rubrics)

    dist = Counter(s["use_case"] for s in scenarios)
    print("\nper capability (scenarios == criteria):")
    for cap, n in dist.most_common():
        print(f"  {cap:<22} {n:>4}")
    print(f"\nexcluded (multilingual, held):  {stats['excluded_multilingual']}")
    print(f"controlled edits applied:       {stats['edits_applied']}")
    print("reference answers: held (native_reference_answer kept; reference_solution null)")
    print("q_mapping/primary_skill: null (skill axis TBD); irt params appended later.")


if __name__ == "__main__":
    main()
