#!/usr/bin/env python3
"""Score what a crashed run managed to finish.

    python scripts/salvage_partial.py --runs-dir ./downloaded-prefix

A run that dies partway leaves two kinds of prediction file behind. Benchmarks that
completed have their ordinary ``*-predictions.jsonl``, written whole and already
represented in ``metrics.json``. A benchmark that was still going has a
``*-predictions.partial.jsonl`` instead: the rows scored before the process died,
appended as they were produced, and deliberately absent from ``metrics.json`` because a
mean over whichever instances happened to be reached is not that benchmark's accuracy.

This reads both and reports them, so the interrupted benchmark's number can be looked at
deliberately rather than mistaken for a result.

THE LAST LINE OF A PARTIAL FILE MAY BE TRUNCATED, and that is expected rather than
corruption: the uploader copies the file while the evaluator is still appending to it, so
whatever was mid-write at that moment arrives cut off. A truncated final line is dropped
with a note. A malformed line anywhere else is real damage and is reported as an error,
because nothing in the write path should produce one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PARTIAL_SUFFIX = "-predictions.partial.jsonl"
COMPLETE_SUFFIX = "-predictions.jsonl"


def read_rows(path: Path) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Rows from a predictions jsonl, tolerating a truncated tail.

    Returns the rows, a note if the final line was dropped as truncated, and an error if
    the file is damaged in a way upload truncation cannot explain. Damage is reported
    rather than raised so that one bad file does not hide every other file's results.
    """
    # utf-8-sig rather than utf-8: the writer never emits a BOM, but a file that has been
    # round-tripped through a Windows editor may carry one, and refusing to read a
    # salvaged file over a three-byte prefix would be a poor trade.
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.split("\n") if line.strip()]

    rows: list[dict[str, Any]] = []
    note: str | None = None
    for position, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            if position == len(lines) - 1:
                note = f"dropped a truncated final line ({len(line)} bytes)"
            else:
                return (
                    rows,
                    note,
                    f"line {position + 1} of {len(lines)} is malformed and is not the last "
                    f"one, so upload truncation does not explain it: {error}",
                )

    if lines and not rows:
        # Only the tail is ever expected to be unreadable. A file that yields nothing has
        # not been truncated, it has been mangled, and saying "0 instances" would let that
        # pass for an interruption that happened to arrive early.
        return rows, None, f"none of its {len(lines)} line(s) parsed; this file is not salvageable"

    return rows, note, None


def score(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of every instance metric present, keyed ``metric.scorer``.

    Mirrors how a benchmark's headline number is produced: ``instance_metrics`` already
    carries each instance's own score, so accuracy is their mean and needs no model.
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        instance_metrics = row.get("instance_metrics") or {}
        if not isinstance(instance_metrics, dict):
            continue
        for metric, scorers in instance_metrics.items():
            if not isinstance(scorers, dict):
                continue
            for scorer, value in scorers.items():
                if isinstance(value, (int, float)):
                    key = metric if metric == scorer else f"{metric}.{scorer}"
                    totals[key] += float(value)
                    counts[key] += 1
    return {key: totals[key] / counts[key] for key in totals if counts[key]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="A downloaded run prefix, or any directory above one. Searched recursively.",
    )
    parser.add_argument(
        "--partial-only",
        action="store_true",
        help="Report only interrupted benchmarks, skipping those metrics.json already covers.",
    )
    args = parser.parse_args(argv)

    root = Path(args.runs_dir)
    if not root.is_dir():
        print(f"no directory at {root}", file=sys.stderr)
        return 2

    partials = sorted(root.rglob(f"*{PARTIAL_SUFFIX}"))
    complete = (
        []
        if args.partial_only
        else sorted(p for p in root.rglob(f"*{COMPLETE_SUFFIX}") if not p.name.endswith(PARTIAL_SUFFIX))
    )

    if not partials and not complete:
        print(f"no prediction files under {root}")
        return 0

    damaged = 0
    for path in partials + complete:
        interrupted = path.name.endswith(PARTIAL_SUFFIX)
        name = path.name.removesuffix(PARTIAL_SUFFIX if interrupted else COMPLETE_SUFFIX)
        rows, note, error = read_rows(path)

        if error:
            damaged += 1
            print(f"DAMAGED      {name}", file=sys.stderr)
            print(f"              {error}", file=sys.stderr)
            if not rows:
                continue

        label = "INTERRUPTED" if interrupted else "complete   "
        print(f"{label}  {name}  {len(rows)} instance(s)")
        if note:
            print(f"              {note}")
        for key, value in sorted(score(rows).items()):
            print(f"              {key} = {value:.4f}")
        if interrupted:
            print("              not a measurement: these are whichever instances were reached")

    if partials:
        print(
            f"\n{len(partials)} benchmark(s) were interrupted. Their numbers above are over a "
            "subset chosen by processing order, not a sample, so they are not comparable to "
            "a published figure or to a complete run."
        )
    return 1 if damaged else 0


if __name__ == "__main__":
    raise SystemExit(main())
