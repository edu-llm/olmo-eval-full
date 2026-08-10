"""CLI: MRBench Milestone-0 gold-DAMR sanity check (no judge, no network).

Computes gold-label DAMR from MRBench V1 and diffs it, cell by cell, against the
paper's Table 3, printing a clear PASS/FAIL verdict.

    uv run python -m diagnostics.mrbench.gold_damr_check
    uv run python -m diagnostics.mrbench.gold_damr_check --tolerance 1.0
"""

from __future__ import annotations

import argparse
import sys

from .gold_damr import DamrTable, compute_damr, load_conversations
from .reference import (
    DEFAULT_TOLERANCE_PP,
    DIMENSION_HEADERS,
    DIMENSIONS,
    PAPER_TUTOR_ORDER,
    TABLE3_REFERENCE,
)

_COL_W = 8
_NAME_W = 16


def _fmt_row(name: str, values: list[str]) -> str:
    return f"{name:<{_NAME_W}}" + "".join(f"{v:>{_COL_W}}" for v in values)


def _header_row() -> str:
    return _fmt_row("tutor / n", [DIMENSION_HEADERS[d] for d in DIMENSIONS])


def print_computed_table(table: DamrTable) -> None:
    print("Computed gold DAMR (%)  —  MRBench V1")
    print(_header_row())
    for tutor in PAPER_TUTOR_ORDER:
        cells = table.cells[tutor]
        n = table.response_totals[tutor]
        values = [f"{cells[d].rate:.2f}" for d in DIMENSIONS]
        print(_fmt_row(f"{tutor} ({n})", values))


def print_diff_table(table: DamrTable, tolerance: float) -> tuple[int, int, float]:
    """Print per-cell diffs vs Table 3. Return (n_pass, n_total, max_abs_diff)."""
    print(f"\nPer-cell diff vs paper Table 3 (computed - paper), tol = +/-{tolerance:g} pp")
    print(_header_row())
    n_pass = 0
    n_total = 0
    max_abs = 0.0
    for tutor in PAPER_TUTOR_ORDER:
        cells = table.cells[tutor]
        ref = TABLE3_REFERENCE[tutor]
        diff_cells: list[str] = []
        for i, dim in enumerate(DIMENSIONS):
            diff = cells[dim].rate - ref[i]
            max_abs = max(max_abs, abs(diff))
            n_total += 1
            ok = abs(diff) <= tolerance
            n_pass += ok
            mark = "" if ok else "*"
            diff_cells.append(f"{diff:+.2f}{mark}")
        print(_fmt_row(tutor, diff_cells))
    print("(* = cell exceeds tolerance)")
    return n_pass, n_total, max_abs


def print_integrity(table: DamrTable) -> list[str]:
    """Print dataset-integrity facts. Return a list of human-readable warnings."""
    warnings: list[str] = []
    print("\nDataset integrity")
    print(f"  conversations parsed : {table.n_conversations}")
    print(f"  responses scored     : {table.n_responses}")
    print(f"  data source counts   : {dict(sorted(table.data_source_counts.items()))}")

    if table.n_conversations != 192:
        warnings.append(f"expected 192 conversations, found {table.n_conversations}")
    if table.n_responses != 1596:
        warnings.append(f"paper reports 1,596 responses; this file yields {table.n_responses}")
    bridge = table.data_source_counts.get("Bridge")
    if bridge is not None and bridge != 60:
        warnings.append(
            f"Novice/Bridge denominator is {bridge}, not the paper's 60 Bridge dialogues"
        )
    if table.missing_tutors:
        warnings.append(f"tutors with no responses (mapping gap?): {table.missing_tutors}")
    return warnings


def run(data_path: str | None, tolerance: float) -> int:
    conversations = load_conversations(data_path)
    table = compute_damr(conversations)

    print_computed_table(table)
    n_pass, n_total, max_abs = print_diff_table(table, tolerance)
    warnings = print_integrity(table)

    print("\nVerdict")
    print(f"  cells within tolerance : {n_pass}/{n_total}")
    print(f"  max absolute diff      : {max_abs:.2f} pp")

    passed = n_pass == n_total and not warnings
    if warnings:
        print("  warnings:")
        for w in warnings:
            print(f"    - {w}")

    verdict = "PASS" if passed else "FAIL"
    print(
        f"\n{verdict}: gold DAMR "
        f"{'reproduces' if passed else 'does NOT reproduce'} paper Table 3 "
        f"within +/-{tolerance:g} pp."
    )
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MRBench Milestone-0 gold-DAMR sanity check (no judge, no network).",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to MRBench_V1.json (defaults to the copy bundled under data/).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_PP,
        help="Per-cell PASS tolerance in percentage points (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    return run(args.data_path, args.tolerance)


if __name__ == "__main__":
    sys.exit(main())
