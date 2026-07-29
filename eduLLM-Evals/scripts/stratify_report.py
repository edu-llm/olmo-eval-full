"""Post-hoc reporting lens: stratify a completed run's per-criterion pass/fail
results by TutorEval's paper-derived scenario flags.

This is a READ-ONLY reporting tool. It reads a run's ``judge_results.jsonl`` and
the scenarios file, then writes its own report next to the run. It does not touch
run outputs, calibration artifacts, the Q-matrix, or any IRT quantity.

It computes PASS RATES ONLY. It deliberately does NOT read or stratify theta/SE
or any IRT parameter. Ability (theta) is estimated once per run across the whole
administered set, so it is a per-run quantity, not a per-scenario one -- slicing
theta by a scenario flag would be statistically meaningless. Difficulty and
discrimination are per-criterion calibration inputs, not run results, so they
have no place in a run-outcome report either. Pass rate is the only per-criterion
outcome that is well defined to stratify.

Flags stratified (each independently):
    book_condition      open_book / closed_book
    answer_in_chapter   true / false
    misleading_question true / false
    subset              easy / hard

Scenarios missing a flag (e.g. pooled TutorBench items) bucket as "n/a". A 2x2
cross-tab of book_condition x misleading_question is also emitted.

Usage
-----
.. code-block:: bash

    uv run python scripts/stratify_report.py --run-dir runs/<run_id>
    uv run python scripts/stratify_report.py --run-dir runs/<run_id> \\
        --scenarios data/TutorEval/scenarios_final.jsonl \\
        --out runs/<run_id>/stratified_report.json --min-n 10 --csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Flags stratified independently, in display order.
FLAGS: tuple[str, ...] = (
    "book_condition",
    "answer_in_chapter",
    "misleading_question",
    "subset",
)
# Fallback scenarios file when the run manifest doesn't record one.
DEFAULT_SCENARIOS = "data/TutorEval/scenarios_final.jsonl"
NA = "n/a"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_scenarios(path: Path) -> dict[str, dict[str, Any]]:
    """Map scenario_id -> scenario record from a JSONL scenarios file."""
    return {s["scenario_id"]: s for s in read_jsonl(path) if "scenario_id" in s}


def resolve_scenarios_path(run_dir: Path, override: str | None) -> Path:
    """Pick the scenarios file: explicit override, else the run manifest's
    ``data_scenarios``, else the repo default."""
    if override:
        return Path(override)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = manifest.get("data_scenarios")
        if recorded:
            return Path(recorded)
    return Path(DEFAULT_SCENARIOS)


def is_pass(row: dict[str, Any]) -> bool:
    """Decode a judge-result row's pass/fail robustly across field names.

    Prefers the integer ``score``/``y`` (1 = pass, 0 = fail); falls back to the
    string ``verdict`` ("pass"/"fail")."""
    for key in ("score", "y"):
        if key in row and row[key] is not None:
            return int(row[key]) == 1
    verdict = str(row.get("verdict", "")).strip().lower()
    return verdict == "pass"


def flag_value(scenario: dict[str, Any] | None, flag: str) -> str:
    """Stringified flag value for a scenario, or "n/a" if absent/unknown."""
    if scenario is None or scenario.get(flag) is None:
        return NA
    value = scenario[flag]
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _stratum(n_criteria: int, n_pass: int, min_n: int) -> dict[str, Any]:
    return {
        "n_criteria": n_criteria,
        "n_pass": n_pass,
        "pass_rate": (n_pass / n_criteria) if n_criteria else None,
        "low_n": n_criteria < min_n,
    }


def stratify(
    rows: list[dict[str, Any]],
    scenarios: dict[str, dict[str, Any]],
    min_n: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Compute per-flag strata and the book_condition x misleading_question
    cross-tab. Returns (strata, crosstab)."""
    # totals[flag][value] = [n_criteria, n_pass]
    totals: dict[str, dict[str, list[int]]] = {f: defaultdict(lambda: [0, 0]) for f in FLAGS}
    cross: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for row in rows:
        scenario = scenarios.get(row.get("scenario_id"))
        passed = 1 if is_pass(row) else 0
        for flag in FLAGS:
            cell = totals[flag][flag_value(scenario, flag)]
            cell[0] += 1
            cell[1] += passed
        book = flag_value(scenario, "book_condition")
        misleading = flag_value(scenario, "misleading_question")
        key = f"{book} x {misleading}"
        cross[key][0] += 1
        cross[key][1] += passed

    strata = {
        flag: {value: _stratum(n, p, min_n) for value, (n, p) in sorted(totals[flag].items())}
        for flag in FLAGS
    }
    crosstab = {key: _stratum(n, p, min_n) for key, (n, p) in sorted(cross.items())}
    return strata, crosstab


def _fmt_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.3f}"


def print_table(
    strata: dict[str, dict[str, dict[str, Any]]],
    crosstab: dict[str, dict[str, Any]],
) -> None:
    """Pretty-print an aligned table of every stratum to stdout."""
    header = ("flag", "value", "n_criteria", "n_pass", "pass_rate", "low_n")
    rows: list[tuple[str, str, str, str, str, str]] = []
    for flag in FLAGS:
        for value, m in strata[flag].items():
            rows.append(
                (
                    flag,
                    value,
                    str(m["n_criteria"]),
                    str(m["n_pass"]),
                    _fmt_rate(m["pass_rate"]),
                    "yes" if m["low_n"] else "",
                )
            )
    for key, m in crosstab.items():
        rows.append(
            (
                "book x misleading",
                key,
                str(m["n_criteria"]),
                str(m["n_pass"]),
                _fmt_rate(m["pass_rate"]),
                "yes" if m["low_n"] else "",
            )
        )

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(header))))
    for r in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)))


def write_csv(
    path: Path,
    strata: dict[str, dict[str, dict[str, Any]]],
    crosstab: dict[str, dict[str, Any]],
) -> None:
    """Write a flat CSV twin of the report table."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["flag", "value", "n_criteria", "n_pass", "pass_rate", "low_n"])
        for flag in FLAGS:
            for value, m in strata[flag].items():
                writer.writerow(
                    [
                        flag,
                        value,
                        m["n_criteria"],
                        m["n_pass"],
                        _fmt_rate(m["pass_rate"]),
                        m["low_n"],
                    ]
                )
        for key, m in crosstab.items():
            writer.writerow(
                [
                    "book x misleading",
                    key,
                    m["n_criteria"],
                    m["n_pass"],
                    _fmt_rate(m["pass_rate"]),
                    m["low_n"],
                ]
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path, help="a runs/<id>/ directory")
    parser.add_argument(
        "--scenarios",
        default=None,
        help="scenarios JSONL (default: auto-detect from manifest, else repo default)",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help="JSON report path (default: <run-dir>/stratified_report.json)",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=10,
        help="strata with fewer graded criteria are flagged low_n (default: 10)",
    )
    parser.add_argument("--csv", action="store_true", help="also write a flat .csv twin")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    run_dir: Path = args.run_dir
    judge_path = run_dir / "judge_results.jsonl"
    if not judge_path.exists():
        print(f"error: no judge_results.jsonl in {run_dir}", file=sys.stderr)
        return 1
    rows = read_jsonl(judge_path)
    if not rows:
        print(f"error: {judge_path} is empty (no graded criteria)", file=sys.stderr)
        return 1

    scenarios_path = resolve_scenarios_path(run_dir, args.scenarios)
    if not scenarios_path.exists():
        print(f"error: scenarios file not found: {scenarios_path}", file=sys.stderr)
        return 1
    scenarios = load_scenarios(scenarios_path)

    models = sorted({str(r.get("candidate_model")) for r in rows if r.get("candidate_model")})
    run_ids = sorted({str(r.get("run_id")) for r in rows if r.get("run_id")})
    if len(models) > 1:
        print(
            f"note: {len(models)} candidate_models present ({', '.join(models)}); "
            "reporting pooled pass rates across all of them.",
            file=sys.stderr,
        )

    strata, crosstab = stratify(rows, scenarios, args.min_n)

    print(f"run_dir       : {run_dir}")
    print(f"scenarios     : {scenarios_path}")
    print(f"candidate(s)  : {', '.join(models) or 'unknown'}")
    print(f"graded rows   : {len(rows)}    min_n: {args.min_n}")
    print()
    print_table(strata, crosstab)

    report = {
        "run_id": run_ids[0] if len(run_ids) == 1 else run_ids,
        "candidate_models": models,
        "scenarios_file": str(scenarios_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "min_n": args.min_n,
        "n_graded_criteria": len(rows),
        "strata": strata,
        "crosstab": crosstab,
    }
    out_path: Path = args.out or (run_dir / "stratified_report.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")

    if args.csv:
        csv_path = out_path.with_suffix(".csv")
        write_csv(csv_path, strata, crosstab)
        print(f"wrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
