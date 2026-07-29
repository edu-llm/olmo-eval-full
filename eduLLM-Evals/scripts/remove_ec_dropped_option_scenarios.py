"""Remove error-correction scenarios whose MCQ option list was dropped, plus their rubrics.

Target: EC (error_correction) scenarios whose question stem says "which of the following/
these ..." but lists NO options (fewer than 2 option letters in the stem, in any format),
so the item is unanswerable -- the model can't see the choices it's asked to pick among.
The student's "Original Answer" often references a letter (e.g. "A) ...") whose options
were never shown.

Removes the matched scenarios from scenarios.jsonl/.json AND every rubric criterion whose
scenario_id is in the removed set from rubrics.jsonl/.json, preserving referential
integrity. Detection-based (not a fixed id list), so it stays correct if the bank changed
(e.g. after deduplication). Idempotent. --write to apply; default is a dry run.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

U = Path(__file__).resolve().parents[1] / "data" / "EduBench" / "augmented_qmat" / "unmerged"
SCEN_JSONL, RUB_JSONL = U / "scenarios.jsonl", U / "rubrics.jsonl"

MCQ_PHRASING = re.compile(r"which of the (following|these)", re.IGNORECASE)
# option markers in any common format: "A)", "(A)", "A.", "A:", "1)"
OPTION = re.compile(r"\(?([A-D])[\)\.\:]|\b([1-4])\)")


def is_dropped_option_ec(rec: dict) -> bool:
    if rec.get("use_case") != "error_correction":
        return False
    stem = re.split(r"Original Answer", rec.get("prompt", ""), maxsplit=1)[0]
    if not MCQ_PHRASING.search(stem):
        return False
    letters = {m for grp in OPTION.findall(stem) for m in grp if m}
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
    remove_ids = {r["scenario_id"] for r in scenarios if is_dropped_option_ec(r)}

    new_scen = [r for r in scenarios if r["scenario_id"] not in remove_ids]
    new_rub = [r for r in rubrics if r.get("scenario_id") not in remove_ids]

    print(f"scenarios: {len(scenarios)} -> {len(new_scen)}  (remove {len(remove_ids)} EC dropped-option)")
    print(f"rubrics:   {len(rubrics)} -> {len(new_rub)}  (remove {len(rubrics) - len(new_rub)})")
    print("\nsample removed ids:", sorted(remove_ids)[:12])
    ex = next((r for r in scenarios if r["scenario_id"] in remove_ids), None)
    if ex:
        print("example:", " ".join(ex["prompt"].split())[:220])

    # integrity of the REMAINING bank
    remaining_ids = {r["scenario_id"] for r in new_scen}
    ref = set()
    for s in new_scen:
        ref.update(s.get("criterion_ids", []))
    rub_ids = {r["criterion_id"] for r in new_rub}
    dangling = ref - rub_ids
    orphan_scn = {r.get("scenario_id") for r in new_rub} - remaining_ids
    print("\npost-removal integrity:")
    print(f"  scenario criterion_ids missing from rubrics: {len(dangling)}")
    print(f"  rubrics pointing to a removed scenario: {len(orphan_scn)}")

    if not args.write:
        print("\n--dry-run: nothing written (pass --write to apply)")
        return 0
    if dangling or orphan_scn:
        raise SystemExit("refusing to write: integrity check failed")
    write_pair(SCEN_JSONL, new_scen)
    write_pair(RUB_JSONL, new_rub)
    print(f"\nwrote scenarios.jsonl/.json ({len(new_scen)}) and rubrics.jsonl/.json ({len(new_rub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
