"""Assemble a final TutorEval rubric bank from a reviewed edit sheet.

Two flows share one assembler (originals are NEVER modified):

A. Legacy reframe flow (default, no flags -- unchanged behavior):
   1. Start from ``rubrics_reframed.jsonl`` / ``scenarios_reframed.jsonl``.
   2. Overlay the human-reviewed ``after`` text from ``reframe_review.tsv`` (a blank
      or ``DROP`` cell removes that criterion).
   3. Apply the targeted grammar fixes (``FIX``) and compound splits (``SPLIT``).
   4. Renumber ``criterion_id`` per scenario; update each scenario's ``criterion_ids``.
   Output: ``rubrics_final.{jsonl,json}`` + ``scenarios_final.{jsonl,json}``.

B. Reword flow (``--reword-sheet``): overlay the human-edited wording sheet
   (``wording_review.tsv``, which only lists scanner-flagged criteria) onto the CURRENT
   final bank (``rubrics_final.jsonl`` / ``scenarios_final.jsonl``) and write to a SEPARATE
   reworded bank so the tracked ``*_final`` files stay intact as a fallback. Each row is
   governed by its ``after`` cell (plus a ``drop`` decision):
     * DROP the criterion when ``after`` is empty, is a bare ``X``/``x`` marker, or the
       ``decision`` is ``drop``;
     * otherwise ``after`` is the new criterion text, split into multiple criteria when it
       contains ``SPLIT_DELIM``.
   Criteria absent from the sheet keep their original ``rubrics_final`` text. Any scenario
   left with zero surviving criteria is removed ENTIRELY (no rubric rows and dropped from
   the scenarios output). The legacy FIX/SPLIT dicts are NOT applied here (their IDs belong
   to the reframed bank).
   Output (default): ``rubrics_reworded.{jsonl,json}`` + ``scenarios_reworded.{jsonl,json}``.

The output stems and every input path are CLI-overridable, so nothing is hardcoded to
clobber a tracked bank.

Run:
    python scripts/build_final_rubrics.py                 # legacy reframe -> *_final
    python scripts/build_final_rubrics.py --reword-sheet  # wording review -> *_reworded
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "TutorEval"

# Delimiter joining an atomic split into one ``after`` TSV cell. MUST match
# reword_criteria.py's SPLIT_DELIM.
SPLIT_DELIM = " ||| "

# Sentinel: a reviewed row that removes its criterion.
_DROP = object()

# --- 3. grammar fixes: full-text replacements for reviewed criteria (not split) ---
FIX = {
    "te_0611_c01": "The response provides external information to supplement the text, as the student requested.",
    "te_0614_c01": "The response provides external information to supplement the text, as the student requested.",
    "te_0733_c01": "Identifies that the student does not properly understand what the term 'work' refers to.",
    "te_0795_c01": "Identifies that the student provided an incorrect definition for a convolution: it should be addition, not multiplication.",
    "te_0822_c01": "Identifies that the student has not properly understood how multinomial distributions work or are formatted.",
    "te_0742_c01": "Provides the student a starting point: the basic definition of variance Var=E(x - E(x))^2.",
    "te_0233_c01": "Identifies that the student misunderstood an important concept regarding the information plot.",
    "te_0590_c01": "Identifies that the homunculus is relevant to what the student is asking.",
    "te_0680_c02": "Acknowledges that the student is partly correct: the Heisenberg uncertainty principle means we can't know the momentum exactly at the same time as its position.",
}

# --- 4. splits: one compound criterion -> ordered list of atomic criteria ---
SPLIT = {
    "te_0583_c04": [
        "Do not solve it.",
        "Give a hint as short as possible.",
    ],
    "te_0664_c03": [
        r"Explains that there are two solutions for $\psi(x)$: one with $A_1 \exp\left(+ \frac{in\omega}{c}x\right)$ and another one $A_2 \exp\left(- \frac{in\omega}{c}x\right)$.",
        "Explains that the general solution is a linear combination of these two solutions.",
        "Prompts the student to find the coefficients $A_1$ and $A_2$ from the boundary conditions.",
    ],
    "te_0554_c02": [
        "Does not give out a solution to the problem statement.",
        "Only provides clarification on the statement, as the student requested.",
    ],
    "te_0446_c01": [
        "Does not give out the textbook solution (or another complete solution).",
        "Focuses on finding the mistake in the student's argument and describing it to them.",
    ],
    "te_0344_c01": [
        "Targets the first part of the exercise, since the student's question is about that part.",
        "Does not discuss the second part.",
    ],
    "te_0344_c02": [
        "Recognizes that the student is struggling with the algebra, not the geometry.",
        "Explains the algebra involved in the problem.",
    ],
    "te_0104_c01": [
        "Acknowledges the student's confusion about the rationale for explicitly raising an exception.",
        "Addresses the rationale for explicitly raising an exception.",
    ],
}

SCENARIO_KEYS = [
    "scenario_id",
    "source_id",
    "use_case",
    "subject",
    "subset",
    "grade_band",
    "modality",
    "prompt",
    "conversation_context",
    "reference_solution",
    "book_condition",
    "misleading_question",
    "answer_in_chapter",
    "criterion_ids",
    "source",
    "split",
    "version",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_reframe_edits(path: Path) -> dict[str, str]:
    """Legacy loader: ``criterion_id -> after`` (blank/DROP handled downstream)."""
    return {
        r["criterion_id"]: r["after"].strip()
        for r in csv.DictReader(path.open(encoding="utf-8"), delimiter="\t")
    }


def load_reword_edits(path: Path) -> dict[str, object]:
    """Loader for the human-edited wording review sheet.

    Returns ``edits`` mapping ``criterion_id -> edit`` where ``edit`` is ``_DROP`` or a
    list of one-or-more atomic replacement texts. A row is a DROP when ANY hold: ``after``
    is empty, ``after`` is a bare ``X``/``x`` marker, or ``decision`` is ``drop``.
    Otherwise the non-empty ``after`` is used as-is (split into parts on ``SPLIT_DELIM``).
    Criteria absent from the sheet are simply not in ``edits`` and keep their base text
    downstream.
    """
    edits: dict[str, object] = {}
    for r in csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"):
        cid = r["criterion_id"].strip()
        after = (r.get("after") or "").strip()
        decision = (r.get("decision") or "").strip().lower()
        if not after or after.lower() == "x" or decision == "drop":
            edits[cid] = _DROP
        else:
            edits[cid] = [t.strip() for t in after.split(SPLIT_DELIM) if t.strip()]
    return edits


def resolve_texts(cid: str, base_text: str, edits: dict[str, str]) -> list[str] | None:
    """Legacy reframe resolver. ``None`` => drop the criterion entirely."""
    if cid in SPLIT:
        return SPLIT[cid]
    if cid in FIX:
        return [FIX[cid]]
    if cid in edits:
        aft = edits[cid]
        if aft == "" or aft.upper() == "DROP":
            return None
        return [aft]
    return [base_text]


def resolve_texts_reword(cid: str, base_text: str, edits: dict[str, object]) -> list[str] | None:
    """Reword resolver driven purely by the decision-aware sheet."""
    if cid in edits:
        val = edits[cid]
        if val is _DROP:
            return None
        return list(val)  # type: ignore[arg-type]
    return [base_text]


def assemble(
    scenarios: list[dict], rubrics: list[dict], resolve, edits, drop_empty_scenarios: bool = False
) -> tuple[list[dict], list[dict], dict]:
    """Overlay ``resolve`` onto the bank, renumber IDs per scenario, rebuild scenarios.

    When ``drop_empty_scenarios`` is True, a scenario whose criteria are ALL dropped is
    removed entirely (no rubric rows and omitted from the scenarios output); the legacy
    reframe flow leaves it False to keep every scenario.
    """
    by_scenario: dict[str, list[dict]] = {}
    for r in rubrics:
        by_scenario.setdefault(r["scenario_id"], []).append(r)

    new_rubrics: list[dict] = []
    new_scenarios: list[dict] = []
    dropped_scenarios: list[str] = []
    n_changed = n_split = n_added = n_dropped = 0

    for s in scenarios:
        sid = s["scenario_id"]
        expanded: list[dict] = []
        for r in by_scenario.get(sid, []):
            texts = resolve(r["criterion_id"], r["criterion"], edits)
            if not texts:
                n_dropped += 1
                continue
            if texts != [r["criterion"]]:
                n_changed += 1
                if len(texts) > 1:
                    n_split += 1
                    n_added += len(texts) - 1
            for t in texts:
                expanded.append({**r, "criterion": t})

        if drop_empty_scenarios and not expanded:
            dropped_scenarios.append(sid)
            continue

        ids = [f"{sid}_c{i:02d}" for i in range(1, len(expanded) + 1)]
        for cid, r in zip(ids, expanded, strict=False):
            new_rubrics.append({**r, "criterion_id": cid})
        new_scenarios.append({k: (ids if k == "criterion_ids" else s[k]) for k in SCENARIO_KEYS})

    stats = {
        "changed": n_changed,
        "split": n_split,
        "added": n_added,
        "dropped": n_dropped,
        "dropped_scenarios": dropped_scenarios,
    }
    return new_rubrics, new_scenarios, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--reword-sheet",
        action="store_true",
        help="Reword flow: overlay the human-edited wording sheet onto the final "
        "bank and write a SEPARATE reworded bank (defaults switch to "
        "rubrics_final -> rubrics_reworded).",
    )
    ap.add_argument(
        "--rubrics-in",
        type=Path,
        default=None,
        help="Base rubric JSONL (default: reframed, or final under --reword-sheet).",
    )
    ap.add_argument(
        "--scenarios-in",
        type=Path,
        default=None,
        help="Base scenario JSONL (default: reframed, or final under --reword-sheet).",
    )
    ap.add_argument(
        "--review",
        type=Path,
        default=None,
        help="Reviewed edit sheet TSV (default: reframe_review.tsv, or "
        "wording_review.tsv under --reword-sheet).",
    )
    ap.add_argument(
        "--rubrics-out",
        default=None,
        help="Output rubric stem, no extension (default: rubrics_final, or "
        "rubrics_reworded under --reword-sheet).",
    )
    ap.add_argument(
        "--scenarios-out",
        default=None,
        help="Output scenario stem, no extension (default: scenarios_final, or "
        "scenarios_reworded under --reword-sheet).",
    )
    args = ap.parse_args()

    if args.reword_sheet:
        rubrics_in = args.rubrics_in or OUT_DIR / "rubrics_final.jsonl"
        scenarios_in = args.scenarios_in or OUT_DIR / "scenarios_final.jsonl"
        review = args.review or OUT_DIR / "wording_review.tsv"
        rubrics_out = args.rubrics_out or "rubrics_reworded"
        scenarios_out = args.scenarios_out or "scenarios_reworded"
        edits = load_reword_edits(review)
        resolve = resolve_texts_reword
    else:
        rubrics_in = args.rubrics_in or OUT_DIR / "rubrics_reframed.jsonl"
        scenarios_in = args.scenarios_in or OUT_DIR / "scenarios_reframed.jsonl"
        review = args.review or OUT_DIR / "reframe_review.tsv"
        rubrics_out = args.rubrics_out or "rubrics_final"
        scenarios_out = args.scenarios_out or "scenarios_final"
        edits = load_reframe_edits(review)
        resolve = resolve_texts

    # Guard: never silently clobber the tracked final/labeled bank from the reword flow.
    if args.reword_sheet and rubrics_out in ("rubrics_final", "rubrics_qmatrix_final"):
        ap.error(f"--reword-sheet refuses to overwrite {rubrics_out}; pick another --rubrics-out.")

    scenarios = read_jsonl(scenarios_in)
    rubrics = read_jsonl(rubrics_in)
    new_rubrics, new_scenarios, stats = assemble(
        scenarios, rubrics, resolve, edits, drop_empty_scenarios=args.reword_sheet
    )

    def write(name: str, records: list[dict]) -> None:
        (OUT_DIR / f"{name}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8"
        )
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote data/TutorEval/{name}.jsonl + .json  ({len(records)} rows)")

    print(f"base rubrics:   {rubrics_in}")
    print(f"base scenarios: {scenarios_in}")
    print(f"review sheet:   {review}  ({len(edits)} actionable edits)")
    write(rubrics_out, new_rubrics)
    write(scenarios_out, new_scenarios)
    print(
        f"\n{len(scenarios)} -> {len(new_scenarios)} scenarios "
        f"({len(stats['dropped_scenarios'])} dropped)"
    )
    print(
        f"{len(rubrics)} -> {len(new_rubrics)} criteria "
        f"({stats['changed']} changed, {stats['split']} split, "
        f"{stats['added']} added-by-split, {stats['dropped']} dropped)"
    )
    if stats["dropped_scenarios"]:
        print(f"dropped scenario_ids: {', '.join(stats['dropped_scenarios'])}")


if __name__ == "__main__":
    main()
