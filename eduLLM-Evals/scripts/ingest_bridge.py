"""Convert the Bridge dataset into Scenario + Rubric Schema JSONL (5-skill artifact).

Source: https://huggingface.co/datasets/rose-e-wang/bridge  (train/validation/test,
700 tutor-student math remediation conversations total).
Paper: *Bridging the Novice-Expert Gap via Models of Decision-Making: A Case Study on
Remediating Math Mistakes* (Wang et al.).

What Bridge is: each row is a real tutoring conversation that ends on the STUDENT's
turn containing a mistake. The task for a tutor model is to produce the next tutor
turn (the remediation). The dataset ships the original novice tutor reply (`c_r`) and
an expert-revised reply (`c_r_`), plus the expert's annotation of the error type (`e`)
and their remediation strategy/intention (`z_what` / `z_why`).

HAND-AUTHORING NOTE (deliberate exception to data/README.md's standing rule that a
benchmark "must ship its own rubrics -- no hand-authoring"): Bridge ships NO per-response
criterion checklist. Its native rubric is a 4-dimension conversation-level scale, not a
set of binary per-response criteria. So this ingester attaches a fixed, hand-authored
15-criterion rubric (the CRITERIA template below) to every scenario. This is a conscious,
user-directed departure from the no-hand-authoring rule, made because Bridge has no native
per-response rubric to derive from. The mapping of criteria -> skills (the q-matrix) is
likewise hand-authored, not dataset-derived. Discrimination is left for calibration.

Skills (5). This is Bridge's own q-matrix axis, NOT the engine's default
content/diagnosis/scaffolding tuple. Like InFoBench's 5-skill artifact, Bridge ships as
an on-disk artifact only and is NOT engine-loadable under the default 3-skill
`tutor_cat.SKILLS`; loading it requires pointing the engine at this 5-skill axis first.
The list order is the fixed q-matrix column order and must not be reordered in place --
downstream MIRT code indexes skills positionally.

    diagnosis      - correctly identifying the student's specific error and its cause
    strategy       - pedagogical decision-making: choosing an effective remediation move
    math           - mathematical correctness of the tutor's own statements
    communication  - clarity, structure, and audience-appropriateness of the explanation
    affective      - supportive stance: tone, encouragement, preserving student agency

Mapping (Bridge field -> schema field):
    c_id                   -> source_id            (join key back to HuggingFace)
    c_h[-1].text           -> prompt               (student's final turn = the mistake)
    c_h[:-1]               -> conversation_context  ({role, content}; role student/tutor)
    c_r_ (joined)          -> reference_solution    (expert revised reply = the gold key)
    c_r  (joined)          -> novice_response        (original tutor reply; provenance)
    e                      -> error_type             (drives exclusion + D3 applicability)
    z_what / z_why         -> expert_strategy / expert_intention (provenance)
    lesson_topic           -> lesson_topic           (curriculum code; provenance)
    <HF split>             -> native_split            (provenance; `split` is the pipeline role)

Applicability (deterministic, no API). Two rules, both keyed on the `e` field:
  1. EXCLUDE rows whose `e` is not one of the six clean error types -- these are the
     free-text "the student did not make a mistake" / "end session" / "unresponsive"
     annotations (56 of 700). They are not mistake-remediation items and are logged to
     data/Bridge/dropped.jsonl. Kept: 644.
  2. Criterion D3 (addresses the underlying misconception) is attached ONLY when `e` is a
     conceptual error (`misinterpret` or `diagnose`); it is skipped for the non-conceptual
     slips (`guess`, `right-idea`, `careless`, `imprecise`). Every other criterion is
     attached to every kept scenario. Response-dependent skips (e.g. M2 when the tutor
     shows no worked steps) are NOT applied here -- they are a judge-time concern, handled
     via JudgeVerdict.unscorable_reason at grading.

Criterion ids use a STABLE suffix per template code (D1=c01 ... A3=c15), so a given
suffix always denotes the same criterion across scenarios; a scenario without D3 simply
omits `_c03` (criterion_ids stay otherwise contiguous).

The synthetic MIRT `difficulty`/`discrimination`/`irt_params` are appended afterwards by
scripts/assign_irt_params.py, so re-running this script strips them -- always re-run the
assign step after a rebuild:

    python scripts/ingest_bridge.py
    python scripts/assign_irt_params.py \
        --input data/Bridge/rubrics.jsonl \
        --skills diagnosis,strategy,math,communication,affective \
        --log-dir data/Bridge/irt_logs --no-backup
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "Bridge"

HF_DATASET = "rose-e-wang/bridge"
SOURCE_URL = "https://huggingface.co/datasets/rose-e-wang/bridge"
HF_SPLITS = ["train", "validation", "test"]   # iterated in order; ids number across all
SPLIT = "calibration"   # pipeline-role label (matches TutorBench/InFoBench), not the HF split
USE_CASE = "mistake_remediation"   # unknown to respgen -> falls back to the adaptive_
                                   # explanation system prompt, which preserves multi-turn
SUBJECT = "mathematics"
VERSION = "1.0"

# The fixed q-matrix column order. Bridge's own 5-skill axis (see module docstring).
SKILLS = ["diagnosis", "strategy", "math", "communication", "affective"]

# The six clean error-type labels in `e`. Anything else is free text ("no mistake" /
# "end session" / "unresponsive") and the row is excluded.
CLEAN_ERROR_TYPES = {"guess", "right-idea", "diagnose", "careless", "imprecise", "misinterpret"}
# Conceptual errors: the only ones for which D3 (misconception) is attached.
CONCEPTUAL_ERROR_TYPES = {"misinterpret", "diagnose"}

# ---------------------------------------------------------------------------
# The hand-authored 15-criterion rubric applied to every kept scenario.
# Each entry: (code, primary_skill, [skills marked 1], criticality, objectivity,
#             criterion text, q_rationale).
# explicitness is uniformly "implicit": Bridge prompts issue no explicit per-response
# instruction, so every criterion is an implicit pedagogical expectation.
# Cross-loaded criteria (q sums to 2): D3, P3, M3, C3, A2. All others are single-skill.
# ---------------------------------------------------------------------------
CRITERIA: list[tuple[str, str, list[str], str, str, str, str]] = [
    ("D1", "diagnosis", ["diagnosis"], "critical", "objective",
     "The response correctly identifies the specific error the student made.",
     "Locating the student's actual error is a pure diagnosis act."),
    ("D2", "diagnosis", ["diagnosis"], "critical", "objective",
     "The response addresses the student's real mistake and does not invent or correct an "
     "error the student did not make.",
     "Distinguishing the real error from a nonexistent one is a diagnosis act."),
    ("D3", "diagnosis", ["diagnosis", "strategy"], "standard", "subjective",
     "The response addresses the underlying misconception or faulty reasoning behind the "
     "error, not just the surface wrong answer.",
     "Recognizing the misconception is diagnostic; choosing to target it rather than only "
     "the surface answer is a remediation-strategy decision."),
    ("P1", "strategy", ["strategy"], "critical", "subjective",
     "The response guides the student toward the correction (e.g., a question, hint, or "
     "prompt to reconsider) rather than simply stating the right answer.",
     "Guiding rather than telling is the core pedagogical-strategy choice; giving the "
     "answer away is a strategy failure."),
    ("P2", "strategy", ["strategy"], "standard", "subjective",
     "The instructional move is appropriate for the situation (e.g., re-teaching for a "
     "misconception versus a brief nudge for a slip).",
     "Selecting a remediation move that fits the situation is a strategy judgment."),
    ("P3", "strategy", ["strategy", "affective"], "standard", "subjective",
     "The response keeps the student actively involved in reaching the answer and does not "
     "take over the entire solution.",
     "Preserving productive struggle is a scaffolding strategy, and sustaining the "
     "student's sense of ownership is an affective goal."),
    ("M1", "math", ["math"], "critical", "objective",
     "All mathematical statements in the response are correct.",
     "Correctness of the tutor's own mathematics is pure content."),
    ("M2", "math", ["math"], "standard", "objective",
     "Any worked steps or calculations shown in the response are correct and complete.",
     "Accuracy of shown work is pure content; unscorable when no steps are shown."),
    ("M3", "math", ["math", "diagnosis"], "standard", "objective",
     "The correction reflects the mathematically correct resolution of the student's "
     "specific error.",
     "The fix must be mathematically sound (content) and must actually resolve the "
     "diagnosed error (diagnosis)."),
    ("C1", "communication", ["communication"], "standard", "subjective",
     "The explanation is clear and well-organized, so the student can follow it.",
     "Clarity and organization are communication qualities."),
    ("C2", "communication", ["communication"], "standard", "subjective",
     "The language and vocabulary are appropriate for the student and not confusing.",
     "Audience-appropriate language is a communication quality."),
    ("C3", "communication", ["communication", "strategy"], "standard", "subjective",
     "The response stays focused on the error and avoids overwhelming the student with "
     "unnecessary content.",
     "Concise, focused delivery is a communication quality, and managing how much to "
     "reveal is a scaffolding-strategy decision."),
    ("A1", "affective", ["affective"], "standard", "subjective",
     "The tone is supportive and encouraging rather than harsh or dismissive.",
     "A supportive tone is an affective quality."),
    ("A2", "affective", ["affective", "communication"], "standard", "subjective",
     "The response frames the mistake constructively (e.g., acknowledges the student's "
     "effort) rather than simply labeling it wrong.",
     "Constructive framing is an affective stance enacted through how the message is "
     "communicated."),
    ("A3", "affective", ["affective"], "critical", "subjective",
     "The response avoids discouraging or judgmental language that could undermine the "
     "student's confidence.",
     "Avoiding harm to the student's confidence is an affective concern; a serious "
     "violation is a critical failure."),
]
EXPLICITNESS = "implicit"

# Stable 1-based suffix per code (template order): D1=01 ... A3=15.
CODE_INDEX = {code: i for i, (code, *_) in enumerate(CRITERIA, start=1)}

SCENARIO_KEYS = [
    "scenario_id", "source_id", "use_case", "subject", "grade_band", "modality",
    "prompt", "conversation_context", "reference_solution", "novice_response",
    "criterion_ids", "error_type", "expert_strategy", "expert_intention",
    "lesson_topic", "native_split", "source", "split", "version",
]
RUBRIC_KEYS = [
    "criterion_id", "scenario_id", "criterion", "expected_evidence", "scoring_type",
    "score_anchors", "criterion_code", "primary_skill", "q_mapping", "q_rationale",
    "criticality", "objectivity", "explicitness", "source", "status", "version",
]


def scenario_id(index: int) -> str:
    """Zero-based, 4-digit so the ids sort lexicographically."""
    return f"bridge_{index:04d}"


def join_turns(turns: list[dict] | None) -> str:
    """Join a list of message bubbles into one text block (one bubble per line)."""
    return "\n".join((t.get("text") or "").strip() for t in (turns or []) if (t.get("text") or "").strip())


def q_mapping(marked: list[str]) -> dict[str, int]:
    """Multi-hot over the five skills: 1 for every skill this criterion loads on."""
    marked_set = set(marked)
    return {skill: int(skill in marked_set) for skill in SKILLS}


def applicable_codes(error_type: str) -> list[str]:
    """Codes attached to a scenario: all 15, minus D3 for non-conceptual errors."""
    return [
        code for (code, *_rest) in CRITERIA
        if code != "D3" or error_type in CONCEPTUAL_ERROR_TYPES
    ]


def build() -> tuple[list[dict], list[dict], list[dict]]:
    ds = load_dataset(HF_DATASET)

    scenarios: list[dict] = []
    rubrics: list[dict] = []
    dropped: list[dict] = []
    index = 0

    for native_split in HF_SPLITS:
        for row in ds[native_split]:
            error_type = (row.get("e") or "").strip()
            c_id = row.get("c_id")
            if error_type not in CLEAN_ERROR_TYPES:
                dropped.append({
                    "source_id": c_id,
                    "native_split": native_split,
                    "error_type": row.get("e"),
                    "reason": "no_clear_mistake",
                })
                continue

            turns = row.get("c_h") or []
            if not turns:
                dropped.append({
                    "source_id": c_id,
                    "native_split": native_split,
                    "error_type": error_type,
                    "reason": "empty_conversation",
                })
                continue

            # The final student turn becomes the prompt; a blank one leaves the tutor
            # nothing to respond to (a handful of rows record no student answer text).
            prompt_text = (turns[-1].get("text") or "").strip()
            if not prompt_text:
                dropped.append({
                    "source_id": c_id,
                    "native_split": native_split,
                    "error_type": error_type,
                    "reason": "empty_student_turn",
                })
                continue

            sid = scenario_id(index)
            index += 1

            context = [
                {"role": t.get("user", "student"), "content": (t.get("text") or "").strip()}
                for t in turns[:-1]
            ]
            codes = applicable_codes(error_type)
            criterion_ids = [f"{sid}_c{CODE_INDEX[code]:02d}" for code in codes]

            scenarios.append({
                "scenario_id": sid,
                "source_id": c_id,
                "use_case": USE_CASE,
                "subject": SUBJECT,
                "grade_band": None,
                "modality": "text",
                "prompt": prompt_text,
                "conversation_context": context,
                "reference_solution": join_turns(row.get("c_r_")),
                "novice_response": join_turns(row.get("c_r")),
                "criterion_ids": criterion_ids,
                "error_type": error_type,
                "expert_strategy": row.get("z_what"),
                "expert_intention": row.get("z_why"),
                "lesson_topic": row.get("lesson_topic"),
                "native_split": native_split,
                "source": SOURCE_URL,
                "split": SPLIT,
                "version": VERSION,
            })

            spec = {c[0]: c for c in CRITERIA}
            for code in codes:
                _code, primary, marked, criticality, objectivity, text, rationale = spec[code]
                rubrics.append({
                    "criterion_id": f"{sid}_c{CODE_INDEX[code]:02d}",
                    "scenario_id": sid,
                    "criterion": text,
                    "expected_evidence": [],
                    "scoring_type": "binary",
                    "score_anchors": None,
                    "criterion_code": code,
                    "primary_skill": primary,
                    "q_mapping": q_mapping(marked),
                    "q_rationale": rationale,
                    "criticality": criticality,
                    "objectivity": objectivity,
                    "explicitness": EXPLICITNESS,
                    "source": SOURCE_URL,
                    "status": "approved",
                    "version": VERSION,
                })

    return scenarios, rubrics, dropped


def validate(scenarios: list[dict], rubrics: list[dict]) -> list[str]:
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

    declared = {c for s in scenarios for c in s["criterion_ids"]}
    actual = set(cids)
    if declared != actual:
        errs.append(
            f"criterion_ids mismatch: {len(declared - actual)} declared-but-missing, "
            f"{len(actual - declared)} present-but-undeclared"
        )
    orphans = {r["scenario_id"] for r in rubrics} - set(sids)
    if orphans:
        errs.append(f"{len(orphans)} rubrics reference unknown scenarios")

    valid_codes = {c[0] for c in CRITERIA}
    for s in scenarios:
        if not (s["prompt"] or "").strip():
            errs.append(f"{s['scenario_id']}: empty prompt (student turn)")
        if not s["criterion_ids"]:
            errs.append(f"{s['scenario_id']}: no criteria")
        if s["error_type"] not in CLEAN_ERROR_TYPES:
            errs.append(f"{s['scenario_id']}: error_type {s['error_type']!r} not a clean type")
        for turn in s["conversation_context"]:
            if turn.get("role") not in {"student", "tutor"}:
                errs.append(f"{s['scenario_id']}: context role {turn.get('role')!r} not student/tutor")
                break

    for r in rubrics:
        if not (r["criterion"] or "").strip():
            errs.append(f"{r['criterion_id']}: empty criterion")
        if r["criterion_code"] not in valid_codes:
            errs.append(f"{r['criterion_id']}: unknown criterion_code {r['criterion_code']!r}")
        q = r["q_mapping"]
        if list(q.keys()) != SKILLS:
            errs.append(f"{r['criterion_id']}: q_mapping keys/order != {SKILLS}")
        if set(q.values()) - {0, 1}:
            errs.append(f"{r['criterion_id']}: q_mapping values must be 0/1")
        if sum(q.values()) < 1:
            errs.append(f"{r['criterion_id']}: q_mapping maps to no skill")
        if r["primary_skill"] not in SKILLS:
            errs.append(f"{r['criterion_id']}: primary_skill {r['primary_skill']!r} not a skill")
        elif q.get(r["primary_skill"]) != 1:
            errs.append(f"{r['criterion_id']}: primary_skill not marked in q_mapping")

    # Applicability consistency: D3 present iff the scenario is a conceptual error.
    by_sid = {s["scenario_id"]: s for s in scenarios}
    for r in rubrics:
        if r["criterion_code"] == "D3":
            et = by_sid[r["scenario_id"]]["error_type"]
            if et not in CONCEPTUAL_ERROR_TYPES:
                errs.append(f"{r['criterion_id']}: D3 attached to non-conceptual error {et!r}")
    for s in scenarios:
        has_d3 = f"{s['scenario_id']}_c{CODE_INDEX['D3']:02d}" in s["criterion_ids"]
        should = s["error_type"] in CONCEPTUAL_ERROR_TYPES
        if has_d3 != should:
            errs.append(f"{s['scenario_id']}: D3 presence {has_d3} != conceptual {should}")

    return errs


def write(name: str, records: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / f"{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json_path = OUT_DIR / f"{name}.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    size = jsonl_path.stat().st_size / 1e6
    print(f"wrote {jsonl_path.relative_to(ROOT)} + .json  ({len(records)} rows, {size:.1f} MB)")


def main() -> None:
    scenarios, rubrics, dropped = build()

    errs = validate(scenarios, rubrics)
    if errs:
        print(f"VALIDATION FAILED ({len(errs)} issues):", file=sys.stderr)
        for e in errs[:20]:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"validation passed: {len(scenarios)} scenarios, {len(rubrics)} criteria")

    write("scenarios", scenarios)
    write("rubrics", rubrics)
    if dropped:
        write("dropped", dropped)

    # q-matrix column loads (criteria marking each skill) + primary distribution.
    loads = Counter(s for r in rubrics for s in SKILLS if r["q_mapping"][s])
    primaries = Counter(r["primary_skill"] for r in rubrics)
    multi = sum(1 for r in rubrics if sum(r["q_mapping"].values()) > 1)
    by_error = Counter(s["error_type"] for s in scenarios)
    with_d3 = sum(1 for s in scenarios if s["error_type"] in CONCEPTUAL_ERROR_TYPES)
    # Bridge annotates some conversations more than once (different experts / error
    # types). source_id (== the Bridge c_id) is the conversation key; report how many
    # scenarios share one so calibration can group or hold them out (see README).
    cid_counts = Counter(s["source_id"] for s in scenarios)
    dup_scenarios = sum(n for n in cid_counts.values() if n > 1)
    dup_groups = sum(1 for n in cid_counts.values() if n > 1)
    print("\nq_mapping loads (criteria marking each skill) / primary_skill:")
    for skill in SKILLS:
        print(f"  {skill:<14} load={loads.get(skill, 0):>5}   primary={primaries.get(skill, 0):>5}")
    print(f"\ncross-loaded criterion instances (q sums >1): {multi}")
    print(f"scenarios by error_type: " + ", ".join(f"{k}={v}" for k, v in by_error.most_common()))
    print(f"scenarios with D3 (conceptual): {with_d3}   without: {len(scenarios) - with_d3}")
    print(f"unique conversations (source_id): {len(cid_counts)}   "
          f"scenarios sharing a conversation: {dup_scenarios} in {dup_groups} groups")
    print(f"dropped (no clear mistake / empty): {len(dropped)}")
    print("difficulty/discrimination/irt_params: appended by assign_irt_params.py (dummy synthetic).")


if __name__ == "__main__":
    main()
