"""Stratified subsample of any scenario bank (+ its rubric bank), to JSONL.

Dataset-agnostic by design: the bank paths and the stratification field are all
command-line arguments, so one tool serves EduBench (stratify by ``use_case``),
Dr. SCI (by ``subject``), or any future bank, instead of a per-dataset script.

Two jobs in one pass:

  * **Subsample.** An equal quota per group, uniformly at random within each
    group under a fixed seed. Response generation runs every model against every
    scenario, so a full bank is often unaffordable while a balanced slice is not.
  * **Materialize as JSONL.** The pipeline's loader reads ``.jsonl`` only, but
    some banks ship as pretty-printed ``.json``. With ``--per-group`` omitted this
    is a pure format conversion of the whole bank.

Scenario ids are never rewritten, so ``criterion_ids`` links into the rubric bank
stay intact and provenance back to the source bank is preserved.

Examples:
    # EduBench: 252 scenarios per task type (2,268 total)
    python scripts/stratified_sample_bank.py \\
        --scenarios data/EduBench/augmented_qmat/unmerged/scenarios.json \\
        --rubrics   data/EduBench/augmented_qmat/unmerged/rubrics.json \\
        --stratify-by use_case --per-group 252 \\
        --out-scenarios data/EduBench/augmented_qmat/unmerged/scenarios_252.jsonl \\
        --out-rubrics   data/EduBench/augmented_qmat/unmerged/rubrics_252.jsonl

    # Whole bank, .json -> .jsonl, no subsampling
    python scripts/stratified_sample_bank.py \\
        --scenarios data/EduBench/augmented_qmat/unmerged/scenarios.json \\
        --out-scenarios data/EduBench/augmented_qmat/unmerged/scenarios.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ID_FIELD = "scenario_id"


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def read_records(path: Path) -> list[tuple[dict, str]]:
    """Load a bank as [(record, source_line)]. Accepts a JSON array or JSONL.

    The source line is kept so sampled rows can be written back byte-identical
    when the input was already JSONL.
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return [(obj, json.dumps(obj, ensure_ascii=False)) for obj in json.loads(text)]
    out: list[tuple[dict, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append((json.loads(line), line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: malformed JSON line: {exc}") from exc
    return out


def write_jsonl(path: Path, rows: list[str], force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite {path} (pass --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in rows:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def select_ids(
    records: list[tuple[dict, str]],
    field: str | None,
    per_group: int | None,
    seed: int,
    allow_short: bool,
) -> tuple[set[str], dict[str, int]]:
    """(selected ids, per-group selected count). No field or no quota => take all."""
    if field is None or per_group is None:
        return {r[ID_FIELD] for r, _ in records}, {"(all)": len(records)}

    by_group: dict[str, list[str]] = collections.defaultdict(list)
    for rec, _ in records:
        if field not in rec:
            raise SystemExit(
                f"scenario {rec.get(ID_FIELD, '?')} has no field {field!r}; "
                f"available: {', '.join(sorted(rec))}"
            )
        by_group[str(rec[field])].append(rec[ID_FIELD])

    print(f"\navailable per {field}:")
    for g in sorted(by_group):
        print(f"  {g:<34}{len(by_group[g]):>7}")

    short = {g: len(ids) for g, ids in by_group.items() if len(ids) < per_group}
    if short and not allow_short:
        detail = ", ".join(f"{g}={n}" for g, n in sorted(short.items()))
        raise SystemExit(
            f"\n{len(short)} group(s) have fewer than --per-group {per_group}: {detail}\n"
            f"  Lower --per-group to {min(short.values())} for a balanced sample, "
            f"or pass --allow-short to take whatever each group has."
        )

    rng = random.Random(seed)
    selected: set[str] = set()
    counts: dict[str, int] = {}
    for g in sorted(by_group):  # sorted => the seed alone determines the sample
        take = min(per_group, len(by_group[g]))
        selected.update(rng.sample(by_group[g], take))
        counts[g] = take
    return selected, counts


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def skill_coverage(rubrics: list[tuple[dict, str]]) -> dict[str, int]:
    """Criteria loading each skill, from every row's ``q_mapping``. Skills whose
    value is 0 are not counted - the Q-matrix row is what defines coverage."""
    counts: collections.Counter[str] = collections.Counter()
    for rec, _ in rubrics:
        for skill, loaded in (rec.get("q_mapping") or {}).items():
            if loaded:
                counts[skill] += 1
    return dict(counts)


def report(
    field: str | None,
    counts: dict[str, int],
    n_scen: int,
    rubrics: list[tuple[dict, str]] | None,
) -> None:
    print(f"\nselected {n_scen} scenarios", end="")
    if field:
        print(f" ({len(counts)} {field} groups)")
        for g in sorted(counts):
            print(f"  {g:<34}{counts[g]:>7}")
    else:
        print()

    if not rubrics:
        return
    print(f"\nselected {len(rubrics)} rubric criteria")
    per_scenario = collections.Counter(r[ID_FIELD] for r, _ in rubrics)
    if per_scenario:
        vals = sorted(per_scenario.values())
        mean = sum(vals) / len(vals)
        print(f"  criteria per scenario: min {vals[0]}, mean {mean:.2f}, max {vals[-1]}")

    skills = skill_coverage(rubrics)
    if skills:
        total = len(rubrics)
        print(f"\nskill coverage ({len(skills)} skills, criteria loading each):")
        for skill, n in sorted(skills.items(), key=lambda kv: -kv[1]):
            print(f"  {skill:<34}{n:>7}  {100.0 * n / total:5.1f}%")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenarios", required=True, type=Path, help="input bank (.json or .jsonl)")
    ap.add_argument("--out-scenarios", required=True, type=Path, help="output .jsonl")
    ap.add_argument("--rubrics", type=Path, help="optional rubric bank to filter alongside")
    ap.add_argument("--out-rubrics", type=Path, help="output .jsonl for the filtered rubrics")
    ap.add_argument(
        "--stratify-by",
        metavar="FIELD",
        help="scenario field to balance across (e.g. use_case, subject); omit for no stratification",
    )
    ap.add_argument(
        "--per-group",
        type=int,
        help="scenarios to draw per group; omit to take the whole bank",
    )
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    ap.add_argument(
        "--allow-short",
        action="store_true",
        help="take all of a group that has fewer than --per-group instead of failing",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rubrics and not args.out_rubrics:
        raise SystemExit("--rubrics requires --out-rubrics")
    if args.per_group is not None and not args.stratify_by:
        raise SystemExit("--per-group requires --stratify-by")

    scenarios = read_records(args.scenarios)
    print(f"read {len(scenarios)} scenarios from {args.scenarios}")
    ids = [r[ID_FIELD] for r, _ in scenarios]
    if len(set(ids)) != len(ids):
        dupes = [k for k, n in collections.Counter(ids).items() if n > 1]
        raise SystemExit(f"duplicate {ID_FIELD} in source bank: {dupes[:5]} ({len(dupes)} total)")

    selected, counts = select_ids(
        scenarios, args.stratify_by, args.per_group, args.seed, args.allow_short
    )
    scen_rows = [line for rec, line in scenarios if rec[ID_FIELD] in selected]

    kept_rubrics: list[tuple[dict, str]] | None = None
    if args.rubrics:
        rubrics = read_records(args.rubrics)
        print(f"read {len(rubrics)} rubric criteria from {args.rubrics}")
        kept_rubrics = [(rec, line) for rec, line in rubrics if rec[ID_FIELD] in selected]
        orphans = selected - {rec[ID_FIELD] for rec, _ in kept_rubrics}
        if orphans:
            print(
                f"warning: {len(orphans)} sampled scenario(s) have no rubric criteria, "
                f"e.g. {sorted(orphans)[:3]}"
            )

    report(args.stratify_by, counts, len(scen_rows), kept_rubrics)

    write_jsonl(args.out_scenarios, scen_rows, args.force)
    print(f"\nwrote {len(scen_rows)} scenarios -> {args.out_scenarios}")
    if kept_rubrics is not None and args.out_rubrics:
        write_jsonl(args.out_rubrics, [line for _, line in kept_rubrics], args.force)
        print(f"wrote {len(kept_rubrics)} rubric criteria -> {args.out_rubrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
