"""Regression tests for the multi-benchmark judge ingest / verdict-writer path.

These pin the crash-safety and memory/resume fixes in
``scripts/run_judge_grading.py`` (reused by ``scripts/run_all_judge_grading.py``):

  * ingest indexing PROJECTS teammate verdict rows to the small set of fields
    used downstream, dropping the large ``raw_output`` blob;
  * the reused in-memory verdict set matches a fresh disk read (done-keys are
    loaded once, not twice);
  * a truncated trailing line in ``verdicts.jsonl`` is tolerated on load while
    genuine mid-file corruption still raises;
  * voided (``y=None``) rows count as NOT done for ``--resume`` (they get
    retried) but do not duplicate on rewrite;
  * the atomic rewrite yields no duplicate rows across a simulated ``--no-resume``
    re-run.

All fixtures are tiny and offline.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _load_module(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rjg = _load_module("run_judge_grading", "scripts/run_judge_grading.py")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# S1: projection drops raw_output, keeps the downstream fields
# ---------------------------------------------------------------------------
def test_load_ingest_index_projects_out_raw_output(tmp_path: Path) -> None:
    src = tmp_path / "verdicts_in.jsonl"
    _write_jsonl(src, [
        {
            "case_id": "resp_abc__c1",
            "verdict": "pass",
            "status": "ok",
            "rationale": "looks good",
            "evidence": "quote",
            "native_score": 4,
            "judge_model": "Qwen/Qwen3.5-9B",
            "prompt_version": "v3",
            # The big field the frozen runner writes per cell -- must be dropped.
            "raw_output": "x" * 10_000,
            "some_other_noise": {"a": 1},
        }
    ])

    by_case, prov, conflicts = rjg.load_ingest_index([src])
    assert conflicts == 0
    row = by_case["resp_abc__c1"]

    # raw_output (and any unlisted field) is gone; needed fields survive.
    assert "raw_output" not in row
    assert "some_other_noise" not in row
    for keep in ("case_id", "verdict", "status", "rationale", "evidence", "native_score"):
        assert keep in row
    assert row["native_score"] == 4
    assert set(row).issubset(set(rjg.INGEST_ROW_FIELDS))
    # Provenance is still harvested for cross-checking.
    assert prov["judge_model"] == {"Qwen/Qwen3.5-9B"}


# ---------------------------------------------------------------------------
# S1(b): the reused in-memory set equals a fresh load (loaded once, same set)
# ---------------------------------------------------------------------------
def test_writer_rows_match_disk_load(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    writer = rjg.VerdictWriter(path)
    rows = [
        rjg.make_verdict_row(("m", "s1", "c1"), y=1, source="ingest", verdict="pass"),
        rjg.make_verdict_row(("m", "s1", "c2"), y=0, source="ingest", verdict="fail"),
        rjg.make_verdict_row(("m", "s2", "c3"), y=None, source="ingest_no_decision",
                             verdict="no_decision"),
    ]
    for r in rows:
        writer.write(r)
    writer.close()

    reused = writer.rows()
    from_disk = rjg.load_done_keys(path)
    assert set(reused) == set(from_disk)
    assert {k: v.get("y") for k, v in reused.items()} == \
           {k: v.get("y") for k, v in from_disk.items()}


# ---------------------------------------------------------------------------
# S5: truncated trailing line is tolerated; real mid-file corruption raises
# ---------------------------------------------------------------------------
def test_truncated_trailing_line_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    good = json.dumps({"model": "m", "scenario": "s", "criterion_id": "c1", "y": 1})
    # Second line is a spot-kill truncation: invalid JSON, no trailing newline.
    path.write_text(good + "\n" + '{"model": "m", "scenario": "s", "criter',
                    encoding="utf-8")

    done = rjg.load_done_keys(path)
    assert set(done) == {("m", "s", "c1")}


def test_midfile_corruption_still_raises(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    # A malformed line that DOES end with a newline is not a recoverable tail.
    path.write_text('{"broken": true' + "\n" + '{"ok": 1}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(rjg.iter_jsonl(path, tolerate_truncated_tail=True))


# ---------------------------------------------------------------------------
# S4: y=None rows are NOT done for resume; retrying them does not duplicate
# ---------------------------------------------------------------------------
def test_is_resume_done_treats_void_as_not_done() -> None:
    assert rjg.is_resume_done({"y": 1}) is True
    assert rjg.is_resume_done({"y": 0}) is True
    assert rjg.is_resume_done({"y": None}) is False
    assert rjg.is_resume_done(None) is False


def test_resume_retries_void_without_duplicating(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    # Prior run left c1 as a void (y=None) and c2 as a real pass.
    _write_jsonl(path, [
        rjg.make_verdict_row(("m", "s", "c1"), y=None, source="ingest_no_decision",
                             verdict="no_decision"),
        rjg.make_verdict_row(("m", "s", "c2"), y=1, source="ingest", verdict="pass"),
    ])

    existing = rjg.load_done_keys(path)
    writer = rjg.VerdictWriter(path)
    writer.seed(existing.values())

    staged = [("m", "s", "c1"), ("m", "s", "c2")]
    reprocessed = []
    for key in staged:
        if rjg.is_resume_done(existing.get(key)):
            continue  # c2 already decided -> skipped
        reprocessed.append(key)
        # A later judge wave now resolves the void to a real fail.
        writer.write(rjg.make_verdict_row(key, y=0, source="ingest", verdict="fail"))
    writer.close()

    assert reprocessed == [("m", "s", "c1")]  # only the void was retried
    lines = _read_lines(path)
    assert len(lines) == 2  # no duplicate c1 row
    done = rjg.load_done_keys(path)
    assert done[("m", "s", "c1")]["y"] == 0  # void replaced by the new verdict
    assert done[("m", "s", "c2")]["y"] == 1  # skipped cell preserved


# ---------------------------------------------------------------------------
# S4: atomic rewrite -> no duplicate rows on a simulated --no-resume re-run
# ---------------------------------------------------------------------------
def test_no_resume_rerun_has_no_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    cells = [("m", "s", "c1"), ("m", "s", "c2"), ("m", "s", "c3")]

    def run(y: int) -> None:
        # --no-resume => no seed; every staged cell is (re)written.
        writer = rjg.VerdictWriter(path)
        for key in cells:
            writer.write(rjg.make_verdict_row(key, y=y, source="ingest",
                                              verdict="pass" if y else "fail"))
        writer.close()

    run(1)
    assert len(_read_lines(path)) == 3
    run(0)  # blind re-run must not accumulate duplicates
    lines = _read_lines(path)
    assert len(lines) == 3
    keys = [(r["model"], r["scenario"], r["criterion_id"]) for r in map(json.loads, lines)]
    assert len(keys) == len(set(keys)) == 3
    assert all(r["y"] == 0 for r in map(json.loads, lines))  # latest run wins


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "verdicts.jsonl"
    writer = rjg.VerdictWriter(path)
    writer.write(rjg.make_verdict_row(("m", "s", "c1"), y=1, source="ingest", verdict="pass"))
    writer.close()
    assert path.is_file()
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != path.name]
    assert leftovers == []
