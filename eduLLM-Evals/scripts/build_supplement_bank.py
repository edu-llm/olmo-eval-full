"""Build the SUBSET curated bank for the supplemental calibration judging run.

The big calibration run graded every NONOPTIONAL curated criterion for each
selected scenario and hard-excluded every OPTIONAL criterion. It also failed to
emit verdicts for a handful of scenarios. This helper computes the criteria the
big run skipped and emits subset scenario/rubric banks that present exactly those
criteria as gradable (nonoptional) binary criteria, so the frozen
``run_calibration_judging.py prepare`` path grades them without code changes.

Targets = {all optional criteria} UNION {nonoptional criteria never graded}.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _graded_criteria_and_models(path: Path) -> tuple[set[str], set[str]]:
    graded: set[str] = set()
    models: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            criterion = row.get("criterion_id")
            if criterion:
                graded.add(str(criterion))
            model = row.get("tutor_model")
            if model:
                models.add(str(model))
    return graded, models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_bank = (
        Path(__file__).resolve().parents[1]
        / "supplement_handoff_build"
        / "_src"
        / "qwen_calibration_handoff_32k_20260728"
        / "data"
        / "curated"
    )
    parser.add_argument("--scenarios", type=Path, default=default_bank / "scenarios_curated.jsonl")
    parser.add_argument("--rubrics", type=Path, default=default_bank / "rubrics_qmatrix_curated.jsonl")
    parser.add_argument(
        "--run-data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "run_data.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "supplement_handoff_build" / "subset_data",
    )
    args = parser.parse_args()

    scenarios = _load_jsonl(args.scenarios)
    rubrics = _load_jsonl(args.rubrics)
    scenarios_by_id = {str(s["scenario_id"]): s for s in scenarios}
    rubrics_by_id = {str(r["criterion_id"]): r for r in rubrics}
    if len(rubrics_by_id) != len(rubrics):
        raise SystemExit("duplicate criterion_id in rubric bank")
    if len(scenarios_by_id) != len(scenarios):
        raise SystemExit("duplicate scenario_id in scenario bank")

    optional_ids = {rid for rid, r in rubrics_by_id.items() if r.get("optional") is True}
    nonoptional_ids = {rid for rid, r in rubrics_by_id.items() if r.get("optional") is not True}

    graded, run_models = _graded_criteria_and_models(args.run_data)

    recovered_nonoptional = nonoptional_ids - graded
    target_ids = optional_ids | recovered_nonoptional

    # Sanity: no optional criterion should already have verdicts.
    optional_graded = optional_ids & graded
    if optional_graded:
        raise SystemExit(f"unexpected: {len(optional_graded)} optional criteria already graded")

    # Group targets by scenario.
    targets_by_scenario: dict[str, list[str]] = defaultdict(list)
    for cid in target_ids:
        rub = rubrics_by_id[cid]
        targets_by_scenario[str(rub["scenario_id"])].append(cid)

    # Emit subset scenario records: only scenarios with >=1 target, criterion_ids
    # restricted to that scenario's targets (sorted).
    out_scenarios: list[dict] = []
    out_rubrics: list[dict] = []
    for sid in sorted(targets_by_scenario):
        scenario = dict(scenarios_by_id[sid])
        target_cids = sorted(targets_by_scenario[sid])
        scenario["criterion_ids"] = target_cids
        out_scenarios.append(scenario)
        for cid in target_cids:
            rub = dict(rubrics_by_id[cid])
            # Make gradable: it must not be optional so _selected_bank retains it.
            rub["optional"] = False
            # scoring_type is already binary for the whole bank; assert to be safe.
            if rub.get("scoring_type") not in {None, "binary"}:
                raise SystemExit(f"{cid}: non-binary scoring_type cannot be graded")
            if str(rub.get("scenario_id")) != sid:
                raise SystemExit(f"{cid}: rubric scenario mismatch")
            if not str(rub.get("criterion") or "").strip():
                raise SystemExit(f"{cid}: blank criterion text")
            out_rubrics.append(rub)

    # Validate: every subset scenario has >=1 retained (nonoptional) criterion.
    for scenario in out_scenarios:
        retained = [
            cid
            for cid in scenario["criterion_ids"]
            if rubrics_by_id[cid] is not None
        ]
        if not retained:
            raise SystemExit(f"{scenario['scenario_id']}: no retained criteria after subsetting")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios_out = args.out_dir / "scenarios_supplement.jsonl"
    rubrics_out = args.out_dir / "rubrics_supplement.jsonl"
    with scenarios_out.open("w", encoding="utf-8") as handle:
        for row in out_scenarios:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with rubrics_out.open("w", encoding="utf-8") as handle:
        for row in out_rubrics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    optional_now_gradable = len(optional_ids)
    recovered = len(recovered_nonoptional)
    total = len(target_ids)
    n_models = len(run_models)

    # Per-scenario grouping summary.
    per_scenario_counts = {sid: len(cids) for sid, cids in targets_by_scenario.items()}
    scenarios_with_recovered = sorted(
        {str(rubrics_by_id[c]["scenario_id"]) for c in recovered_nonoptional}
    )

    print("=== supplement bank build ===")
    print(f"bank scenarios: {len(scenarios)}  bank rubrics: {len(rubrics)}")
    print(f"nonoptional in bank: {len(nonoptional_ids)}  optional in bank: {len(optional_ids)}")
    print(f"graded criteria in run_data: {len(graded)}")
    print(f"optional-now-gradable: {optional_now_gradable}")
    print(f"recovered-nonoptional (ungraded): {recovered}")
    print(f"  across {len(scenarios_with_recovered)} scenarios: {scenarios_with_recovered}")
    print(f"TOTAL target criteria: {total}")
    print(f"subset scenarios (>=1 target): {len(out_scenarios)}")
    print(f"models (from run_data tutor_model): {n_models}")
    print(f"cells = {n_models} x {total} = {n_models * total:,}")
    print(f"targets-per-scenario histogram: {dict(sorted(Counter(per_scenario_counts.values()).items()))}")
    print(f"wrote: {scenarios_out}")
    print(f"wrote: {rubrics_out}")

    if total != 909:
        print(f"WARNING: expected 909 total targets, got {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
