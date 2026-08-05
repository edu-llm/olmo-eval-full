from __future__ import annotations

import json
from pathlib import Path

from tutor_cat.dataio import load_bank, summarize


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_bank_uses_explicit_five_skill_order(tmp_path: Path) -> None:
    skills = ("content", "format", "number", "style", "linguistic")
    scenarios = tmp_path / "scenarios.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    _write_jsonl(
        scenarios,
        [{"scenario_id": "s1", "prompt": "p", "criterion_ids": ["c1"]}],
    )
    _write_jsonl(
        rubrics,
        [
            {
                "criterion_id": "c1",
                "scenario_id": "s1",
                "criterion": "criterion",
                "q_mapping": {
                    "content": 0,
                    "format": 1,
                    "number": 0,
                    "style": 0,
                    "linguistic": 1,
                },
                "discrimination": {
                    "content": 0.0,
                    "format": 1.2,
                    "number": 0.0,
                    "style": 0.0,
                    "linguistic": 0.8,
                },
                "difficulty": -0.4,
            }
        ],
    )

    bank, report = load_bank(scenarios, rubrics, skills=skills)

    assert report.ok
    assert bank.skills == skills
    assert bank.rubrics["c1"].q.tolist() == [0, 1, 0, 0, 1]
    assert bank.rubrics["c1"].a.tolist() == [0.0, 1.2, 0.0, 0.0, 0.8]
    assert "format=1" in summarize(bank)
    assert "linguistic=1" in summarize(bank)
