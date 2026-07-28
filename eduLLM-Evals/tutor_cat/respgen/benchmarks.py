"""Parse benchmarks.yaml (the benchmark registry) into BenchmarkSpec objects and
apply run-time subset selection.

The registry decouples *which* benchmarks the fleet answers from the pipeline
logic: adding or removing a benchmark is a data change here, not a code change.
Each spec carries the canonical `name` (used as both the output shard directory
and the "Benchmark" row label) and the path to its scenarios.jsonl.

Pure module (yaml only). BenchmarkSpec is a plain dataclass so it pickles cleanly
across the orchestrator's spawn processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class BenchmarkSpec:
    name: str
    scenarios: str
    enabled: bool = True


def load_benchmarks(path: str | Path) -> list[BenchmarkSpec]:
    """Parse the benchmark registry. Raises on a missing/duplicate name or a
    malformed entry so a typo fails fast instead of silently skipping data."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    specs: list[BenchmarkSpec] = []
    seen: set[str] = set()
    for entry in raw.get("benchmarks") or []:
        if not isinstance(entry, dict) or "name" not in entry or "scenarios" not in entry:
            raise ValueError(
                f"benchmarks.yaml: each entry needs 'name' and 'scenarios': {entry!r}"
            )
        name = str(entry["name"])
        if name in seen:
            raise ValueError(f"benchmarks.yaml: duplicate benchmark name {name!r}")
        seen.add(name)
        specs.append(
            BenchmarkSpec(
                name=name,
                scenarios=str(entry["scenarios"]),
                enabled=bool(entry.get("enabled", True)),
            )
        )
    if not specs:
        raise ValueError(f"benchmarks.yaml: no benchmarks listed in {path}")
    return specs


def select_benchmarks(
    specs: list[BenchmarkSpec], only: list[str] | None = None
) -> list[BenchmarkSpec]:
    """Resolve the benchmarks to actually run.

    `only` (e.g. from `--only IFEval,Bridge`) wins: run exactly those, in the
    order given, regardless of each spec's `enabled`. An unknown name fails fast
    with the available list. With no `only`, run every spec whose `enabled` is
    true (the persistent registry default)."""
    by_name = {s.name: s for s in specs}
    if only:
        chosen: list[BenchmarkSpec] = []
        for name in only:
            if name not in by_name:
                available = ", ".join(sorted(by_name))
                raise ValueError(
                    f"unknown benchmark {name!r}; available: {available}"
                )
            chosen.append(by_name[name])
        return chosen
    enabled = [s for s in specs if s.enabled]
    if not enabled:
        raise ValueError(
            "no benchmarks are enabled; enable at least one in benchmarks.yaml "
            "or pass --only <names>"
        )
    return enabled
