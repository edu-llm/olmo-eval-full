"""Convert the TutorEval dataset into Scenario + Rubric Schema JSONL (offline skeleton).

Source: https://huggingface.co/datasets/princeton-nlp/TutorEval  (one `train` split,
834 undergraduate free-response STEM tutoring questions, 1846 `key_points` bullets).
Paper: *Language Models as Science Tutors* (princeton-nlp/LM-Science-Tutor).

Unlike the WildBench and InFoBench ingesters, this is a **format-only skeleton**:
TutorEval ships **no per-criterion skill labels** (WildBench got `q_mapping` from
`primary_tag`, InFoBench from `question_label` -- TutorEval's bullets carry nothing),
so there is no offline source for a q-matrix. We therefore leave `q_mapping`,
`primary_skill`, and the `criticality`/`objectivity`/`explicitness` triple as **null**,
and write **no** `difficulty`/`discrimination`/`irt_params`. The result is a pure data
artifact -- NOT engine-loadable (Rubric.from_json hard-requires q_mapping/difficulty/
discrimination). Populating those is deferred to the LLM q-matrix pass
(scripts/generate_qmatrix.py, content/diagnosis/scaffolding axis) and then
scripts/assign_irt_params.py.

Two conditions, partitioned (each question asked exactly once, never double-graded):
    closed_book == True  (370) -> prompt is the question verbatim, chapter withheld
    closed_book == False (464) -> prompt embeds the chapter via the authors' template

Mapping (TutorEval field -> schema field):
    (no id field)          -> source_id = None
    question (+ chapter)    -> prompt          (chapter embedded only for open-book)
    domain                  -> subject
    difficulty (easy/hard)  -> subset          (native band; kept out of IRT `difficulty`)
    key_points bullet[i]    -> criterion        (one binary rubric row each)
    closed_book             -> book_condition   (provenance extra)
    misleading_question     -> misleading_question (provenance extra)
    answer_in_chapter       -> answer_in_chapter    (provenance extra)

Bad items are DROPPED at ingest (not flagged, not hand-fixed), each logged with a
reason to data/TutorEval/dropped.jsonl:
    scenario-level  : figure-referencing questions; equation-stripped chapters
    criterion-level : closed-book bullets that defer to the withheld chapter, or that
                      are too terse to grade standalone (< TERSE_MIN_CHARS)
A scenario emptied of all criteria by criterion-level drops is dropped too.

Run:
    python scripts/ingest_tutoreval.py
    # deferred (needs API / calibration):
    #   python scripts/generate_qmatrix.py   ... (fills q_mapping + metadata triple)
    #   python scripts/assign_irt_params.py --input data/TutorEval/rubrics.jsonl \
    #       --skills content,diagnosis,scaffolding --log-dir data/TutorEval/irt_logs
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "TutorEval"

HF_DATASET = "princeton-nlp/TutorEval"
# Pin the revision so ids/counts reproduce even if the dataset is re-uploaded.
REVISION = "aafd8c96c506f11755c005842c893677b76f20ca"
SOURCE_URL = "https://huggingface.co/datasets/princeton-nlp/TutorEval"
SPLIT = "calibration"   # pipeline-role label (matches the other benchmarks), not the HF split
VERSION = "1.0"

# Open-book prompt template, verbatim from the authors' generation_template.txt.
OPEN_BOOK_TEMPLATE = (
    "Here is a passage from a textbook I am trying to understand:\n\n"
    '"""\n{chapter}\n"""\n\n{question}'
)

# --- Drop heuristics (tunable) -------------------------------------------------
# Terse closed-book bullets below this length are dropped as ungradable-standalone.
# 20 reproduces the 34 bullets the integration spec flagged as "very short".
# NOTE: many terse bullets ("Answer is radiation") are in fact gradable given the
# question; raise this to 0 to keep them all. Every drop is logged, so this is
# fully reversible by re-running.
TERSE_MIN_CHARS = 20

# Closed-book bullets that point at the withheld chapter cannot be graded (the tutor
# never sees the chapter in closed-book mode).
DEFER_RE = re.compile(
    r"the (chapter|passage|text)\b|see (the )?(chapter|passage)|"
    r"according to the (chapter|passage)|as (shown|stated|described) in the (chapter|passage|text)",
    re.I,
)

# Questions that ask about an absent figure. Excludes the idiom "figure out".
# Currently matches 0 rows on the pinned revision; kept as a documented safeguard.
FIGURE_RE = re.compile(
    r"\bfigure\s+\d|\bthe figure\b|\bfollowing figure\b|\bthis figure\b|"
    r"\bshown in (the )?fig|\bin (the )?fig(?:ure|\.)\b|\bdiagram\b",
    re.I,
)

# ------------------------------------------------------------------------------

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


def scenario_id(index: int) -> str:
    """Zero-based, 4-digit so the ids sort lexicographically."""
    return f"te_{index:04d}"


def parse_bullets(key_points: str) -> list[str]:
    """Split `key_points` markdown into clean criterion strings.

    Strips the leading ``- ``/``* `` marker and collapses internal whitespace.
    """
    out: list[str] = []
    for line in (key_points or "").split("\n"):
        s = re.sub(r"^\s*[-*]\s+", "", line.strip())
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            out.append(s)
    return out


def build_prompt(closed_book: bool, chapter: str, question: str) -> str:
    """Closed-book: the question verbatim. Open-book: the authors' chapter template."""
    question = (question or "").strip()
    if closed_book:
        return question
    return OPEN_BOOK_TEMPLATE.format(chapter=(chapter or "").strip(), question=question)


def is_equation_stripped(chapter: str) -> bool:
    """Chapter whose equations were reduced to bare ``(2)``/``(3)`` placeholders.

    Heuristic: no real LaTeX anywhere, yet several bare parenthesised integers.
    Matches the 4 items the integration spec identified.
    """
    ch = chapter or ""
    has_latex = any(tok in ch for tok in ("$", "\\(", "\\[", "\\frac", "\\begin"))
    bare = len(re.findall(r"(?<![\w])\(\d{1,3}\)", ch))
    return (not has_latex) and bare >= 3


def build() -> tuple[list[dict], list[dict], list[dict]]:
    """Return (scenarios, rubrics, dropped). Drops are collected, not silently skipped."""
    rows = list(load_dataset(HF_DATASET, revision=REVISION)["train"])
    # Stable ordering so ids don't move if the parquet reorders.
    rows.sort(key=lambda r: (r["path_to_chapter"], r["question"]))

    scenarios: list[dict] = []
    rubrics: list[dict] = []
    dropped: list[dict] = []
    next_index = 0

    for row in rows:
        question = row["question"]
        chapter = row["chapter"]
        closed = bool(row["closed_book"])
        ref = f"{row['path_to_chapter']} :: {question[:80]}"

        # --- scenario-level drops (whole question + all its bullets) ---
        if FIGURE_RE.search(question) and not re.search(r"figure out", question, re.I):
            dropped.append({"level": "scenario", "reason": "figure_reference",
                            "closed_book": closed, "ref": ref})
            continue
        if is_equation_stripped(chapter):
            dropped.append({"level": "scenario", "reason": "equation_stripped_chapter",
                            "closed_book": closed, "ref": ref})
            continue

        # --- criterion-level drops (closed-book only; chapter is present open-book) ---
        kept_criteria: list[str] = []
        for bullet in parse_bullets(row["key_points"]):
            if closed and DEFER_RE.search(bullet):
                dropped.append({"level": "criterion", "reason": "defers_to_chapter",
                                "closed_book": closed, "ref": ref, "criterion": bullet})
                continue
            if closed and len(bullet) < TERSE_MIN_CHARS:
                dropped.append({"level": "criterion", "reason": "too_terse",
                                "closed_book": closed, "ref": ref, "criterion": bullet})
                continue
            kept_criteria.append(bullet)

        if not kept_criteria:
            dropped.append({"level": "scenario", "reason": "no_criteria_left",
                            "closed_book": closed, "ref": ref})
            continue

        sid = scenario_id(next_index)
        next_index += 1
        criterion_ids = [f"{sid}_c{i:02d}" for i in range(1, len(kept_criteria) + 1)]

        scenarios.append({
            "scenario_id": sid,
            "source_id": None,                       # TutorEval ships no id field
            "use_case": "adaptive_explanation",
            "subject": row["domain"],
            "subset": row["difficulty"],             # native easy/hard band
            "grade_band": None,
            "modality": "text",
            "prompt": build_prompt(closed, chapter, question),
            "conversation_context": [],
            "reference_solution": "",
            "book_condition": "closed_book" if closed else "open_book",
            "misleading_question": bool(row["misleading_question"]),
            "answer_in_chapter": bool(row["answer_in_chapter"]),
            "criterion_ids": criterion_ids,
            "source": SOURCE_URL,
            "split": SPLIT,
            "version": VERSION,
        })

        for cid, criterion in zip(criterion_ids, kept_criteria):
            rubrics.append({
                "criterion_id": cid,
                "scenario_id": sid,
                "criterion": criterion,
                "expected_evidence": [],
                "scoring_type": "binary",
                "score_anchors": None,
                # Deferred -- no offline source; filled by generate_qmatrix.py later.
                "primary_skill": None,
                "q_mapping": None,
                "q_rationale": None,
                "criticality": None,
                "objectivity": None,
                "explicitness": None,
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

    header = OPEN_BOOK_TEMPLATE.split("{")[0]  # "Here is a passage from a textbook..."
    for s in scenarios:
        if not (s["prompt"] or "").strip():
            errs.append(f"{s['scenario_id']}: empty prompt")
        if not s["criterion_ids"]:
            errs.append(f"{s['scenario_id']}: no criteria")
        cond = s["book_condition"]
        if cond not in ("closed_book", "open_book"):
            errs.append(f"{s['scenario_id']}: bad book_condition {cond!r}")
        # Partition/prompt-construction check: the chapter template appears iff open-book.
        has_tmpl = header in s["prompt"] and '"""' in s["prompt"]
        if cond == "open_book" and not has_tmpl:
            errs.append(f"{s['scenario_id']}: open_book prompt missing chapter template")
        if cond == "closed_book" and has_tmpl:
            errs.append(f"{s['scenario_id']}: closed_book prompt contains chapter template")

    for r in rubrics:
        if not (r["criterion"] or "").strip():
            errs.append(f"{r['criterion_id']}: empty criterion")
        if r["scoring_type"] != "binary":
            errs.append(f"{r['criterion_id']}: scoring_type must be binary")
        # Skeleton invariants: no q-matrix, no metadata, no IRT.
        for f in ("primary_skill", "q_mapping", "q_rationale",
                  "criticality", "objectivity", "explicitness"):
            if r[f] is not None:
                errs.append(f"{r['criterion_id']}: {f} must be null in the skeleton")
        for f in ("difficulty", "discrimination", "irt_params"):
            if f in r:
                errs.append(f"{r['criterion_id']}: unexpected IRT field {f}")

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


def write_dropped(dropped: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "dropped.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in dropped:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(ROOT)}  ({len(dropped)} dropped rows)")


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
    write_dropped(dropped)

    # --- reconciliation summary (no silent drops) ---
    by_cond = Counter(s["book_condition"] for s in scenarios)
    drop_scn = Counter(d["reason"] for d in dropped if d["level"] == "scenario")
    drop_crit = Counter(d["reason"] for d in dropped if d["level"] == "criterion")
    n_mislead = sum(1 for s in scenarios if s["misleading_question"])
    print("\nkept scenarios by condition:")
    for cond in ("closed_book", "open_book"):
        print(f"  {cond:<12} {by_cond.get(cond, 0):>4}")
    print(f"  misleading_question items kept: {n_mislead}")
    print("\ndropped (scenario-level):")
    for reason, n in drop_scn.most_common():
        print(f"  {reason:<26} {n:>4}")
    print("dropped (criterion-level):")
    for reason, n in drop_crit.most_common():
        print(f"  {reason:<26} {n:>4}")
    print("\nq_mapping / primary_skill / metadata triple / IRT: null (deferred; see README).")


if __name__ == "__main__":
    main()
