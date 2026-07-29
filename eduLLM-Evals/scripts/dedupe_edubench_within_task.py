"""Deduplicate the unmerged EduBench bank WITHIN each task type.

A scenario is a duplicate of another only if it has the same use_case AND the same
full model input -- (prompt, conversation_context). Cross-task-type duplicates are
kept (the same question re-templated as Q&A / EC / IP / ... stays). The first
occurrence in file order is kept; later identical ones are dropped.

Rubrics follow their scenario: every criterion whose scenario_id was dropped is
removed (criterion_id = <scenario_id>_cNN, a strict 1:1 scenario->criteria
partition, so this is exact). Both the .jsonl (canonical) and the pretty .json twin
are rewritten for scenarios and rubrics.

Expected result (must match, or the script aborts before writing):
  scenarios 9163 -> 8081   (remove 1082)
  rubrics   67823 -> 59700 (remove 8123)

Default is a dry run; pass --write to apply.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

U = Path(__file__).resolve().parents[1] / "data" / "EduBench" / "augmented_qmat" / "unmerged"
SCEN_JSONL = U / "scenarios.jsonl"
SCEN_JSON = SCEN_JSONL.with_suffix(".json")
RUB_JSONL = U / "rubrics.jsonl"
RUB_JSON = RUB_JSONL.with_suffix(".json")

EXPECT_SCEN_KEPT = 8081
EXPECT_RUB_KEPT = 59700


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def write_jsonl(p: Path, records: list[dict]) -> None:
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(p: Path, records: list[dict]) -> None:
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    args = ap.parse_args()

    scenarios = load_jsonl(SCEN_JSONL)
    rubrics = load_jsonl(RUB_JSONL)

    # --- dedup scenarios within use_case by (prompt, conversation_context) ---
    seen: dict[str, set] = defaultdict(set)
    kept_scen: list[dict] = []
    kept_ids: set[str] = set()
    removed_ids: set[str] = set()
    for r in scenarios:
        uc = r.get("use_case", "")
        key = (r.get("prompt", ""), json.dumps(r.get("conversation_context", []), ensure_ascii=False))
        if key in seen[uc]:
            removed_ids.add(r["scenario_id"])
        else:
            seen[uc].add(key)
            kept_scen.append(r)
            kept_ids.add(r["scenario_id"])

    # --- rubrics follow their scenario_id ---
    kept_rub = [c for c in rubrics if c["scenario_id"] in kept_ids]

    # --- report ---
    per_uc = defaultdict(lambda: [0, 0])
    for r in scenarios:
        per_uc[r["use_case"]][0] += 1
    for r in kept_scen:
        per_uc[r["use_case"]][1] += 1
    print(f"{'task_type':<32}{'orig':>7}{'kept':>7}{'removed':>9}")
    for uc in sorted(per_uc):
        o, k = per_uc[uc]
        print(f"{uc:<32}{o:>7}{k:>7}{o - k:>9}")
    print("-" * 55)
    print(f"scenarios: {len(scenarios)} -> {len(kept_scen)}  (removed {len(scenarios) - len(kept_scen)})")
    print(f"rubrics:   {len(rubrics)} -> {len(kept_rub)}  (removed {len(rubrics) - len(kept_rub)})")

    # --- integrity + count guards (abort before any write on mismatch) ---
    kept_crit_ids = {c["criterion_id"] for c in kept_rub}
    referenced = {cid for r in kept_scen for cid in r["criterion_ids"]}
    problems = []
    if len(kept_scen) != EXPECT_SCEN_KEPT:
        problems.append(f"scenario count {len(kept_scen)} != expected {EXPECT_SCEN_KEPT}")
    if len(kept_rub) != EXPECT_RUB_KEPT:
        problems.append(f"rubric count {len(kept_rub)} != expected {EXPECT_RUB_KEPT}")
    if referenced - kept_crit_ids:
        problems.append(f"{len(referenced - kept_crit_ids)} dangling criterion refs after dedup")
    if kept_crit_ids - referenced:
        problems.append(f"{len(kept_crit_ids - referenced)} orphan rubrics after dedup")
    if {c['scenario_id'] for c in kept_rub} - kept_ids:
        problems.append("a kept rubric points to a removed scenario")

    print("\nintegrity:", "OK" if not problems else "FAILED")
    for p in problems:
        print("  -", p)

    if not args.write:
        print("\n--dry-run: nothing written (pass --write to apply)")
        return 0
    if problems:
        raise SystemExit("refusing to write: guards failed (see above)")

    write_jsonl(SCEN_JSONL, kept_scen)
    write_json(SCEN_JSON, kept_scen)
    write_jsonl(RUB_JSONL, kept_rub)
    write_json(RUB_JSON, kept_rub)
    print(f"\nwrote {SCEN_JSONL.name}/.json ({len(kept_scen)}) and {RUB_JSONL.name}/.json ({len(kept_rub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
