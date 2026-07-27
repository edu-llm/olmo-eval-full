"""Convert the IFEval dataset into Scenario + Rubric Schema JSONL.

Source: https://huggingface.co/datasets/google/IFEval  (one `train` split, 541 prompts,
834 instruction instances). Paper: *Instruction-Following Evaluation for Large Language
Models* (Zhou et al., 2023).

Unlike WildBench/InFoBench (LLM-judge benchmarks), IFEval's criteria are graded by a
**deterministic code verifier** -- Google's `instruction_following_eval` package, vendored
at tutor_cat/verifiers/ifeval/. Each `instruction_id` maps to a checker class exposing
`build_description(**kwargs) -> str` (the human-readable constraint) and
`check_following(response) -> bool` (the pass/fail). This ingester therefore:

  * emits one binary criterion per instruction instance (834 total);
  * sets `criterion` to the checker's own `build_description(**kwargs)` output -- faithful,
    standalone constraint text;
  * preserves the verifier's inputs in a `verifier` provenance dict
    {"instruction_id", "kwargs"} so the IFEvalVerifier can reproduce the check at scoring
    time (Rubric.from_json carries this through as an optional field);
  * builds a single-hot `q_mapping` over IFEval's **9 instruction categories**
    (the `instruction_id` prefix), exactly like WildBench derives one skill from
    `primary_tag`.

The synthetic MIRT `difficulty`/`discrimination`/`irt_params` are appended afterwards by
scripts/assign_irt_params.py over the 9-category axis, so re-running this script strips
them -- always re-run the assign step after a rebuild:

    PYTHONPATH=. python scripts/ingest_ifeval.py
    python scripts/assign_irt_params.py \
        --input data/IFEval/rubrics.jsonl \
        --skills keywords,detectable_format,length_constraints,change_case,startend,punctuation,combination,detectable_content,language \
        --log-dir data/IFEval/irt_logs --no-backup

Mapping (IFEval field -> schema field):
    key                         -> source_id       (join key back to HuggingFace)
    prompt                      -> prompt
    instruction_id_list[i]      -> verifier.instruction_id + q_mapping category
    kwargs[i]                   -> verifier.kwargs (nulls stripped)
    build_description(**kwargs) -> criterion

Placeholders (NOT native to IFEval): every criterion is stamped
`objectivity="objective"` (genuinely true -- the check is deterministic),
`explicitness="explicit"`, `criticality="critical"` (uniform, matching InFoBench). These
feed only the synthetic IRT heuristic downstream.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from tutor_cat.verifiers.ifeval import instructions_registry

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "IFEval"

HF_DATASET = "google/IFEval"
# Pin the revision so ids/counts reproduce even if the dataset is re-uploaded.
REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
SOURCE_URL = "https://huggingface.co/datasets/google/IFEval"
SPLIT = "calibration"   # pipeline-role label (matches the other benchmarks), not the HF split
VERSION = "1.0"

REGISTRY = instructions_registry.INSTRUCTION_DICT

# IFEval's 9 instruction categories = the `instruction_id` prefix (before ':'). This is
# the fixed q-matrix column order; downstream MIRT code indexes skills positionally, so
# it must not be reordered in place. Every id's prefix must be one of these.
SKILLS = [
    "keywords", "detectable_format", "length_constraints", "change_case",
    "startend", "punctuation", "combination", "detectable_content", "language",
]

# Placeholder metadata (IFEval has no native equivalent; feed only synthetic IRT).
CRITICALITY = "critical"
OBJECTIVITY = "objective"     # genuinely true: the verifier is deterministic
EXPLICITNESS = "explicit"

SCENARIO_KEYS = [
    "scenario_id", "source_id", "use_case", "subject", "subset", "grade_band",
    "modality", "prompt", "conversation_context", "reference_solution",
    "criterion_ids", "source", "split", "version",
]
RUBRIC_KEYS = [
    "criterion_id", "scenario_id", "criterion", "expected_evidence",
    "scoring_type", "score_anchors", "verifier", "primary_skill", "q_mapping",
    "q_rationale", "criticality", "objectivity", "explicitness",
    "source", "status", "version",
]


def scenario_id(index: int) -> str:
    """Zero-based, 4-digit so the ids sort lexicographically."""
    return f"ife_{index:04d}"


def category_of(instruction_id: str) -> str:
    """IFEval category = the id prefix before ':'."""
    return instruction_id.split(":", 1)[0]


def q_mapping(category: str) -> dict[str, int]:
    """Single-hot over the 9 categories (each instruction has exactly one)."""
    return {skill: int(skill == category) for skill in SKILLS}


def clean_kwargs(kw: dict) -> dict:
    """HuggingFace pads every kwargs key across the whole dataset with null; keep only
    the args actually set for this instruction."""
    return {k: v for k, v in kw.items() if v is not None}


def describe(instruction_id: str, kwargs: dict) -> str:
    """The checker's own natural-language constraint, used as the criterion text.

    Reproduces the official grading path's `build_description(**kwargs)`. A few checkers
    draw random defaults for *absent* kwargs; IFEval supplies the needed ones, so this is
    deterministic, but we seed anyway for byte-stable reruns.
    """
    checker = REGISTRY[instruction_id](instruction_id)
    random.seed(0)
    desc = checker.build_description(**kwargs)
    return " ".join(desc.split())  # collapse whitespace


def build() -> tuple[list[dict], list[dict]]:
    rows = list(load_dataset(HF_DATASET, revision=REVISION)["train"])
    # Stable ordering so ids don't move if the parquet reorders.
    rows.sort(key=lambda r: r["key"])

    scenarios: list[dict] = []
    rubrics: list[dict] = []

    for index, row in enumerate(rows):
        sid = scenario_id(index)
        ids = list(row["instruction_id_list"])
        kwargs_list = list(row["kwargs"])
        if len(ids) != len(kwargs_list):
            raise SystemExit(
                f"{sid}: instruction_id_list ({len(ids)}) != kwargs ({len(kwargs_list)})"
            )

        criterion_ids = [f"{sid}_c{i:02d}" for i in range(1, len(ids) + 1)]

        scenarios.append({
            "scenario_id": sid,
            "source_id": row["key"],
            "use_case": "instruction_following",
            "subject": None,                 # IFEval has no domain field
            "subset": None,                  # no native difficulty band
            "grade_band": None,
            "modality": "text",
            "prompt": (row["prompt"] or "").strip(),
            "conversation_context": [],      # single-turn
            "reference_solution": None,      # IFEval ships none
            "criterion_ids": criterion_ids,
            "source": SOURCE_URL,
            "split": SPLIT,
            "version": VERSION,
        })

        for cid, iid, kw in zip(criterion_ids, ids, kwargs_list):
            if iid not in REGISTRY:
                raise SystemExit(f"{cid}: instruction_id {iid!r} not in verifier registry")
            category = category_of(iid)
            if category not in SKILLS:
                raise SystemExit(f"{cid}: category {category!r} (from {iid!r}) not in axis")
            kwargs = clean_kwargs(kw)
            rubrics.append({
                "criterion_id": cid,
                "scenario_id": sid,
                "criterion": describe(iid, kwargs),
                "expected_evidence": [],
                "scoring_type": "binary",
                "score_anchors": None,
                # Verifier inputs -- reproduce the deterministic check at scoring time.
                "verifier": {"instruction_id": iid, "kwargs": kwargs},
                "primary_skill": category,
                "q_mapping": q_mapping(category),
                "q_rationale": f"IFEval instruction category '{category}' (from id '{iid}')",
                # Placeholders (uniform); feed only the synthetic IRT step.
                "criticality": CRITICALITY,
                "objectivity": OBJECTIVITY,
                "explicitness": EXPLICITNESS,
                "source": SOURCE_URL,
                "status": "approved",
                "version": VERSION,
            })

    return scenarios, rubrics


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

    for s in scenarios:
        if not (s["prompt"] or "").strip():
            errs.append(f"{s['scenario_id']}: empty prompt")
        if not s["criterion_ids"]:
            errs.append(f"{s['scenario_id']}: no criteria")
    for r in rubrics:
        if not (r["criterion"] or "").strip():
            errs.append(f"{r['criterion_id']}: empty criterion")
        if r["scoring_type"] != "binary":
            errs.append(f"{r['criterion_id']}: scoring_type must be binary")
        # Verifier spec: resolvable id + dict kwargs.
        v = r["verifier"]
        if not isinstance(v, dict) or v.get("instruction_id") not in REGISTRY:
            errs.append(f"{r['criterion_id']}: verifier.instruction_id not in registry")
        elif not isinstance(v.get("kwargs"), dict):
            errs.append(f"{r['criterion_id']}: verifier.kwargs must be a dict")
        # q-matrix: keys == the fixed axis, single-hot, primary marked.
        q = r["q_mapping"]
        if list(q.keys()) != SKILLS:
            errs.append(f"{r['criterion_id']}: q_mapping keys/order != axis")
        if set(q.values()) - {0, 1} or sum(q.values()) != 1:
            errs.append(f"{r['criterion_id']}: q_mapping must be single-hot")
        if r["primary_skill"] not in SKILLS or q.get(r["primary_skill"]) != 1:
            errs.append(f"{r['criterion_id']}: primary_skill not the marked category")

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
    scenarios, rubrics = build()

    errs = validate(scenarios, rubrics)
    if errs:
        print(f"VALIDATION FAILED ({len(errs)} issues):", file=sys.stderr)
        for e in errs[:20]:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"validation passed: {len(scenarios)} scenarios, {len(rubrics)} criteria")

    write("scenarios", scenarios)
    write("rubrics", rubrics)

    # q-matrix column loads (criteria per category) + multiplicity.
    loads = Counter(r["primary_skill"] for r in rubrics)
    mult = Counter(len(s["criterion_ids"]) for s in scenarios)
    print("\nq_mapping loads (criteria per IFEval category):")
    for skill in SKILLS:
        print(f"  {skill:<20} {loads.get(skill, 0):>4}")
    print(f"\ninstructions per prompt: {dict(sorted(mult.items()))}")
    print("difficulty/discrimination/irt_params: appended by assign_irt_params.py (synthetic).")


if __name__ == "__main__":
    main()
