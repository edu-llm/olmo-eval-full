"""Remove easy (Elementary/Middle School) idea_provision MULTIPLE-CHOICE scenarios + rubrics.

Target: use_case == idea_provision AND grade_band in {Elementary School, Middle School}
AND the prompt is multiple-choice (has option markers like "A)"). These are the "provide
reasoning but do not give the answer" items whose answer is a visible option at an easy
grade level, where withholding the answer is near-impossible / low-value.

Removes matched scenarios from scenarios.jsonl/.json AND every rubric whose scenario_id is
in the removed set from rubrics.jsonl/.json, preserving referential integrity.
Detection-based + idempotent. --write to apply; default is a dry run.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

U = Path(__file__).resolve().parents[1] / "data" / "EduBench" / "augmented_qmat" / "unmerged"
SCEN_JSONL, RUB_JSONL = U / "scenarios.jsonl", U / "rubrics.jsonl"
EASY = {"Elementary School", "Middle School"}
OPTION = re.compile(r"\b[A-D]\)")


def is_easy_ip_mcq(rec: dict) -> bool:
    return (
        rec.get("use_case") == "idea_provision"
        and rec.get("grade_band") in EASY
        and bool(OPTION.search(rec.get("prompt", "")))
    )


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
    remove_ids = {r["scenario_id"] for r in scenarios if is_easy_ip_mcq(r)}

    new_scen = [r for r in scenarios if r["scenario_id"] not in remove_ids]
    new_rub = [r for r in rubrics if r.get("scenario_id") not in remove_ids]

    from collections import Counter
    by_grade = Counter(r.get("grade_band") for r in scenarios if r["scenario_id"] in remove_ids)
    print(f"scenarios: {len(scenarios)} -> {len(new_scen)}  (remove {len(remove_ids)} easy IP MCQ)")
    print(f"  by grade_band: {dict(by_grade)}")
    print(f"rubrics:   {len(rubrics)} -> {len(new_rub)}  (remove {len(rubrics) - len(new_rub)})")
    print("sample removed ids:", sorted(remove_ids)[:12])

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
