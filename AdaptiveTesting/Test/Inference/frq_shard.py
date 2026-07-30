"""FRQ checkpointing: validity-aware resume + shard compaction.

Ported from eduLLM-Evals ``respgen/shard.py``. This is strictly stronger than the
generic ``results_writer.existing_ids`` resume, and the difference matters:

  * ``existing_ids`` treats ANY row bearing the id as done - so a model whose
    load failed (every row Issue=1) would be permanently "complete" and never
    retried.
  * here a scenario counts as done only if its row is *valid*: generation
    succeeded (``Issue == 0``) AND a genuine prompt reached the model
    (``Prompt Tokens > 1``). Failure cells and truncation-bug rows regenerate.

Writing still goes through ``results_writer.JsonlResultWriter`` so we keep the
fsync-batched durability; this module only owns the resume/compaction view.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ID_KEY = "Scenario"


def _iter_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield each JSON row, tolerating a torn final line from a killed run."""
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line from an interrupted append


def completed_ids(path: str | Path, key: str = ID_KEY) -> set[str]:
    """Every Scenario id present, regardless of Issue/Truncated state. Low-level
    primitive; resume uses ``scan_shard`` for the *valid*-only view."""
    return {sid for obj in _iter_rows(path) if isinstance(sid := obj.get(key), str)}


def _is_valid_row(obj: dict[str, Any]) -> bool:
    """A row is a real, reusable result when generation succeeded (Issue==0) AND a
    genuine prompt reached the model (Prompt Tokens > 1). Prompt Tokens <= 1 is the
    fingerprint of a truncation bug that discarded the whole prompt; no real
    scenario prompt (system prompt + student turn) is ever that short. A row
    missing the field (legacy/minimal) is assumed complete."""
    if obj.get("Issue", 0) == 1:
        return False
    pt = obj.get("Prompt Tokens")
    return pt is None or (isinstance(pt, int) and pt > 1)


def scan_shard(
    path: str | Path, key: str = ID_KEY
) -> tuple[set[str], list[dict[str, Any]], bool]:
    """Resume view of a shard. Returns (done_ids, valid_rows, had_invalid):

      * done_ids    - scenarios with a valid row (skip these on resume)
      * valid_rows  - one row per done scenario, in file order (for compaction)
      * had_invalid - True if ANY Issue / truncation-corrupted / duplicate row was
                      seen, i.e. the shard should be rewritten to valid_rows so
                      failed + corrupted cells regenerate into a clean shard.
    """
    done: set[str] = set()
    valid_rows: list[dict[str, Any]] = []
    had_invalid = False
    for obj in _iter_rows(path):
        sid = obj.get(key)
        if not isinstance(sid, str):
            had_invalid = True
            continue
        if _is_valid_row(obj) and sid not in done:
            done.add(sid)
            valid_rows.append(obj)
        else:
            had_invalid = True  # Issue row, 1-token-bug row, or duplicate
    return done, valid_rows, had_invalid


def rewrite_shard(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Atomically replace a shard with `rows` (temp file + os.replace), compacting
    out Issue/corrupted/duplicate rows so the resumed shard ends with exactly one
    clean row per scenario."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, p)
