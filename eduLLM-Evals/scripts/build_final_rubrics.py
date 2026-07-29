"""Assemble the final TutorEval rubrics from the reviewed reframe sheet.

Pipeline (originals never modified):
1. Start from ``rubrics_reframed.jsonl`` / ``scenarios_reframed.jsonl``.
2. Overlay the human-reviewed ``after`` text from ``reframe_review.tsv`` (a blank
   or ``DROP`` cell removes that criterion).
3. Apply targeted grammar fixes (typos in the reviewed text).
4. Split the few genuinely compound criteria -- ones bundling two independent
   tutor actions -- into separate atomic criteria. Code snippets and single
   mathematical arguments that merely contain semicolons are left intact.
5. Renumber ``criterion_id`` contiguously per scenario and update each scenario's
   ``criterion_ids``. Skill/IRT fields stay null (skeleton).

Output: ``rubrics_final.{jsonl,json}`` + ``scenarios_final.{jsonl,json}``.

Run:
    python scripts/build_final_rubrics.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "TutorEval"

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
    "scenario_id", "source_id", "use_case", "subject", "subset", "grade_band",
    "modality", "prompt", "conversation_context", "reference_solution",
    "book_condition", "misleading_question", "answer_in_chapter",
    "criterion_ids", "source", "split", "version",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def resolve_texts(cid: str, base_text: str, edits: dict[str, str]) -> list[str] | None:
    """Return the final atomic criterion text(s) for one source criterion.

    None => drop the criterion entirely.
    """
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


def main() -> None:
    scenarios = read_jsonl(OUT_DIR / "scenarios_reframed.jsonl")
    rubrics = read_jsonl(OUT_DIR / "rubrics_reframed.jsonl")

    # Human-reviewed edits (after column), keyed by criterion_id.
    edits = {r["criterion_id"]: r["after"].strip()
             for r in csv.DictReader((OUT_DIR / "reframe_review.tsv").open(encoding="utf-8"),
                                     delimiter="\t")}

    by_scenario: dict[str, list[dict]] = {}
    for r in rubrics:
        by_scenario.setdefault(r["scenario_id"], []).append(r)

    new_rubrics: list[dict] = []
    new_scenarios: list[dict] = []
    n_split = n_fixed = n_dropped = 0

    for s in scenarios:
        sid = s["scenario_id"]
        expanded: list[dict] = []
        for r in by_scenario.get(sid, []):
            texts = resolve_texts(r["criterion_id"], r["criterion"], edits)
            if texts is None:
                n_dropped += 1
                continue
            if r["criterion_id"] in SPLIT:
                n_split += 1
            elif r["criterion_id"] in FIX:
                n_fixed += 1
            for t in texts:
                expanded.append({**r, "criterion": t})

        ids = [f"{sid}_c{i:02d}" for i in range(1, len(expanded) + 1)]
        for cid, r in zip(ids, expanded):
            new_rubrics.append({**r, "criterion_id": cid})
        new_scenarios.append({k: (ids if k == "criterion_ids" else s[k]) for k in SCENARIO_KEYS})

    def write(name: str, records: list[dict]) -> None:
        (OUT_DIR / f"{name}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote data/TutorEval/{name}.jsonl + .json  ({len(records)} rows)")

    write("rubrics_final", new_rubrics)
    write("scenarios_final", new_scenarios)
    print(f"\n{len(rubrics)} -> {len(new_rubrics)} criteria "
          f"({n_split} split, {n_fixed} grammar-fixed, {n_dropped} dropped)")


if __name__ == "__main__":
    main()
