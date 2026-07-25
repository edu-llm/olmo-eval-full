"""Durable, append-per-item result writers (see README 4.4).

Guarantees:
  * one file per (benchmark, model) pair;
  * every row flushed to the OS immediately (line-buffered);
  * ``os.fsync`` batched every N rows / T seconds for durability without
    throttling high-QPS small models;
  * header written once; append-only thereafter;
  * question-level resume via :func:`existing_ids`.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from common import ensure_dirs


def existing_ids(path: Path, id_field: str = "question_id") -> set[str]:
    """Return the set of already-persisted ids in a CSV or JSONL result file."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, newline="") as f:
        if path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(str(json.loads(line)[id_field]))
                except (json.JSONDecodeError, KeyError):
                    continue
        else:
            reader = csv.DictReader(f)
            for row in reader:
                if id_field in row:
                    ids.add(row[id_field])
    return ids


class _BaseWriter:
    def __init__(self, path: Path, fsync_every_rows: int = 50, fsync_every_seconds: float = 5.0):
        self.path = path
        self.fsync_every_rows = max(1, fsync_every_rows)
        self.fsync_every_seconds = fsync_every_seconds
        ensure_dirs(path.parent)
        self._rows_since_sync = 0
        self._last_sync = time.monotonic()
        self._fh = None

    def _maybe_fsync(self) -> None:
        self._rows_since_sync += 1
        now = time.monotonic()
        if (
            self._rows_since_sync >= self.fsync_every_rows
            or (now - self._last_sync) >= self.fsync_every_seconds
        ):
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._rows_since_sync = 0
            self._last_sync = now

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class CsvResultWriter(_BaseWriter):
    """Append-only CSV writer with a header written exactly once."""

    def __init__(self, path: Path, fieldnames: list[str], **kw):
        super().__init__(path, **kw)
        self.fieldnames = fieldnames
        is_new = not path.exists() or path.stat().st_size == 0
        self._fh = open(path, "a", newline="", buffering=1)  # noqa: SIM115 (long-lived handle)
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write_row(self, row: dict) -> None:
        self._writer.writerow(row)
        self._maybe_fsync()


class JsonlResultWriter(_BaseWriter):
    """Append-only JSONL writer."""

    def __init__(self, path: Path, **kw):
        super().__init__(path, **kw)
        self._fh = open(path, "a", buffering=1)  # noqa: SIM115 (long-lived handle)

    def write_row(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._maybe_fsync()
