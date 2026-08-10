"""Idempotent append-only cache for judge results, keyed per judge call.

The key is ``(conversation_id, tutor, dimension)`` — one MRBench judge call. A
completed key is skipped on rerun (skip-if-done), so an interrupted run resumes
without repeating (or re-paying for) work. Storage is JSONL: one record per line,
appended, never rewritten.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CacheKey = tuple[str, str, str]  # (conversation_id, tutor, dimension)


@dataclass
class JudgeRecord:
    conversation_id: str
    tutor: str
    dimension: str
    score: int | None
    label: str | None
    feedback: str | None
    ok: bool
    used_fallback: bool
    raw: str
    model: str
    # Per-sample raw outputs when self-consistency (k>1) is used; None for k=1.
    samples: list[str] | None = None

    @property
    def key(self) -> CacheKey:
        return (self.conversation_id, self.tutor, self.dimension)


class JudgeCache:
    """Append-only JSONL cache with in-memory index for skip-if-done."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._done: dict[CacheKey, JudgeRecord] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rec = JudgeRecord(**data)
                self._done[rec.key] = rec

    def is_done(self, key: CacheKey) -> bool:
        """A key counts as done only if a successfully-parsed result is stored."""
        rec = self._done.get(key)
        return rec is not None and rec.ok

    def get(self, key: CacheKey) -> JudgeRecord | None:
        return self._done.get(key)

    def put(self, record: JudgeRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._done[record.key] = record

    def records(self) -> list[JudgeRecord]:
        return list(self._done.values())

    def __len__(self) -> int:
        return len(self._done)
