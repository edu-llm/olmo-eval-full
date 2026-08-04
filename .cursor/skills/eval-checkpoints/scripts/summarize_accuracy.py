#!/usr/bin/env python3
"""Aggregate a checkpoint sweep into an accuracy table.

Reads the per-checkpoint output tree written by ``run_eval_sweep.sh``::

    <runs-dir>/<run_id>/
      metrics.json           olmo-eval's standard output, one entry per task
      run_provenance.json    checkpoint URI, requested benchmarks, status

Emits a long CSV (one row per checkpoint/benchmark/metric) and a wide CSV (one
row per checkpoint, one column per score) since the wide shape is what plots as
a training curve.

Scores come from ``tasks[].metrics`` rather than the top-level ``summary``
block. ``summary`` carries only each task's *primary* metric, which would drop
the exact-match score that ``naturalqs`` and ``jeopardy`` report alongside F1.

Rows are ordered by the trailing integer in the run id (training step) so the
CSV plots directly against training progress.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

FIELDS = [
    "run_id",
    "step",
    "checkpoint",
    "benchmark",
    "metric",
    "scorer",
    "score",
    "is_primary",
    "num_instances",
    "status",
]


def step_of(run_id: str) -> int | None:
    """Training step from a run id, or None when it does not end in one.

    Anchored at the end so a run id like ``exp1-latest`` reports no step instead
    of picking the ``1`` out of the experiment name.
    """
    m = re.search(r"(\d+)(?:-hf)?$", run_id)
    return int(m.group(1)) if m else None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_task_rows(run_dir: Path) -> list[dict[str, Any]]:
    """One row per (task, metric, scorer) in a checkpoint's metrics.json."""
    payload = _load_json(run_dir / "metrics.json")
    rows: list[dict[str, Any]] = []
    for task in payload.get("tasks") or []:
        name = task.get("task") or ""
        primary = task.get("primary_metric")
        metrics = task.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for metric_name, scorers in metrics.items():
            # Shape is {metric: {scorer: value}}, but tolerate a bare scalar in
            # case a task ever reports one directly.
            if not isinstance(scorers, dict):
                if isinstance(scorers, (int, float)):
                    scorers = {"": float(scorers)}
                else:
                    continue
            for scorer_name, value in scorers.items():
                if not isinstance(value, (int, float)):
                    continue
                key = f"{metric_name}:{scorer_name}" if scorer_name else metric_name
                rows.append(
                    {
                        "benchmark": name,
                        "metric": metric_name,
                        "scorer": scorer_name,
                        "score": float(value),
                        "is_primary": key == primary,
                        "num_instances": task.get("num_instances"),
                    }
                )
    return rows


def collect(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        run_id = run_dir.name
        prov = _load_json(run_dir / "run_provenance.json")
        checkpoint = prov.get("checkpoint", "")
        status = prov.get("status", "unknown")

        found = read_task_rows(run_dir)
        if not found:
            # A checkpoint whose eval failed writes provenance but no metrics.
            # Emit one placeholder per requested benchmark so the failure stays
            # visible instead of the checkpoint vanishing from the curve.
            requested = prov.get("benchmarks") or [""]
            found = [{"benchmark": b, "status": status} for b in requested]

        for row in found:
            row.setdefault("status", status)
            row.update(run_id=run_id, step=step_of(run_id), checkpoint=checkpoint)
            rows.append({k: row.get(k) for k in FIELDS})

    rows.sort(
        key=lambda r: (
            r["step"] if r["step"] is not None else 1 << 62,
            str(r["run_id"]),
            str(r["benchmark"]),
            str(r["metric"]),
        )
    )
    return rows


def column_for(benchmark: str, metric: str | None, multi: set[str]) -> str:
    """Wide-CSV column name.

    Bare benchmark name when it reports a single metric, ``benchmark.metric``
    when it reports several -- which is how the F1 and exact-match scores for
    ``naturalqs`` and ``jeopardy`` stay distinguishable.
    """
    if benchmark in multi and metric:
        return f"{benchmark}.{metric}"
    return benchmark


def build_wide(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Pivot to one row per checkpoint, one column per score."""
    metrics_per_bench: dict[str, set[str]] = {}
    for r in rows:
        if r.get("metric"):
            metrics_per_bench.setdefault(str(r["benchmark"]), set()).add(str(r["metric"]))
    multi = {b for b, m in metrics_per_bench.items() if len(m) > 1}

    order: list[str] = []
    by_run: dict[str, dict[str, Any]] = {}
    for r in rows:
        run_id = str(r["run_id"])
        entry = by_run.setdefault(
            run_id,
            {"run_id": run_id, "step": r["step"], "checkpoint": r["checkpoint"]},
        )
        if r.get("score") is None:
            # Preserve the failure signal rather than leaving a blank that reads
            # as "not requested".
            entry.setdefault("status", r.get("status") or "no_results")
            continue
        col = column_for(str(r["benchmark"]), r.get("metric"), multi)
        entry[col] = r["score"]
        if col not in order:
            order.append(col)

    header = ["run_id", "step", "checkpoint", *sorted(order), "status"]
    wide = [by_run[k] for k in sorted(by_run, key=lambda k: (by_run[k]["step"] is None, by_run[k]["step"] or 0, k))]
    for entry in wide:
        entry.setdefault("status", "ok")
    return header, wide


def render(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no results found"
    cols = ["run_id", "benchmark", "metric", "scorer", "score", "num_instances", "status"]
    table = [[c.upper() for c in cols]]
    for r in rows:
        table.append(
            [
                "-"
                if r.get(c) is None
                else (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))
                for c in cols
            ]
        )
    widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
    out = ["  ".join(table[0][i].ljust(widths[i]) for i in range(len(cols))).rstrip()]
    out.append("-" * min(sum(widths) + 2 * len(cols), 118))
    for row in table[1:]:
        out.append("  ".join(row[i].ljust(widths[i]) for i in range(len(cols))).rstrip())
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-csv")
    ap.add_argument("--out-wide-csv")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"no runs directory at {runs_dir}", file=sys.stderr)
        return 1

    rows = collect(runs_dir)
    print(render(rows))

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    if args.out_wide_csv:
        header, wide = build_wide(rows)
        with open(args.out_wide_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(wide)

    if args.out_json:
        header, wide = build_wide(rows)
        Path(args.out_json).write_text(
            json.dumps({"rows": rows, "wide": wide}, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
