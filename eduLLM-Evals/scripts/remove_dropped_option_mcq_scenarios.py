"""Remove grading & Q&A dropped-option MCQ scenarios + their rubrics.

Same defect already removed from error_correction: the question stem says "which of the
following/these ..." but no options are shown (fewer than 2 option letters in the stem),
so the item is malformed -- for grading the grader can't see the choices the student's
answer refers to; for Q&A there is nothing to choose among.

Scope: use_case in {grading, answering_questions} (error_correction and the all-MCQ
idea_provision set were removed by their own scripts). Removes matched scenarios from
scenarios.jsonl/.json AND every rubric whose scenario_id is in the removed set from
rubrics.jsonl/.json. Detection-based + idempotent. --write to apply; default dry run.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

U = Path(__file__).resolve().parents[1] / "data" / "EduBench" / "augmented_qmat" / "unmerged"
SCEN_JSONL, RUB_JSONL = U / "scenarios.jsonl", U / "rubrics.jsonl"
TARGET_UC = {"grading", "answering_questions"}
MCQ = re.compile(r"which of the (following|these)", re.IGNORECASE)
OPTION = re.compile(r"\(?([A-D])[\)\.\:]|\b([1-4])\)")


def stem(p: str) -> str:
    return re.split(r"Original Answer|Student'?s Answer", p, maxsplit=1)[0]


def is_dropped_option_mcq(rec: dict) -> bool:
    if rec.get("use_case") not in TARGET_UC:
        return False
    st = stem(rec.get("prompt", ""))
    if not MCQ.search(st):
        return False
    letters = {m for grp in OPTION.findall(st) for m in grp if m}
    return len(letters) < 2


def load(p): return [json.loads(l) for l in p.open(encoding="utf-8")]


def write_pair(jsonl_path: Path, records: list) -> None:
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with jsonl_path.with_suffix(".json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="apply changes (default: dry run)")
    args = ap.parse_args()

    scenarios, rubrics = load(SCEN_JSONL), load(RUB_JSONL)
    remove_ids = {r["scenario_id"] for r in scenarios if is_dropped_option_mcq(r)}
    by_uc = Counter(r.get("use_case") for r in scenarios if r["scenario_id"] in remove_ids)

    new_scen = [r for r in scenarios if r["scenario_id"] not in remove_ids]
    new_rub = [r for r in rubrics if r.get("scenario_id") not in remove_ids]

    print(f"scenarios: {len(scenarios)} -> {len(new_scen)}  (remove {len(remove_ids)} dropped-option MCQ)")
    print(f"  by use_case: {dict(by_uc)}")
    print(f"rubrics:   {len(rubrics)} -> {len(new_rub)}  (remove {len(rubrics) - len(new_rub)})")

    remaining_ids = {r["scenario_id"] for r in new_scen}
    ref = set()
    for s in new_scen:
        ref.update(s.get("criterion_ids", []))
    rub_ids = {r["criterion_id"] for r in new_rub}
    dangling = ref - rub_ids
    orphan = {r.get("scenario_id") for r in new_rub} - remaining_ids
    print("\npost-removal integrity:")
    print(f"  scenario criterion_ids missing from rubrics: {len(dangling)}")
    print(f"  rubrics pointing to a removed scenario: {len(orphan)}")

    if not args.write:
        print("\n--dry-run: nothing written (pass --write to apply)")
        return 0
    if dangling or orphan:
        raise SystemExit("refusing to write: integrity check failed")
    write_pair(SCEN_JSONL, new_scen)
    write_pair(RUB_JSONL, new_rub)
    print(f"\nwrote scenarios.jsonl/.json ({len(new_scen)}) and rubrics.jsonl/.json ({len(new_rub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
