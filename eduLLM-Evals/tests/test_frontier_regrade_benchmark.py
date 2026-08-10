from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "api_judge_pilot"
    / "regrade_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("frontier_regrade_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
regrade = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regrade)


def _write_jsonl(path: Path, rows: list[dict], *, bom: bool = False) -> None:
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8-sig" if bom else "utf-8")


def test_load_jsonl_accepts_utf8_bom(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    _write_jsonl(source, [{"value": 1}], bom=True)
    assert regrade._load_jsonl(source) == [{"value": 1}]


def test_build_cases_counts_auto_fails_per_criterion(tmp_path: Path) -> None:
    scenarios = tmp_path / "scenarios.jsonl"
    rubrics = tmp_path / "rubrics.jsonl"
    responses = tmp_path / "responses"
    responses.mkdir()
    _write_jsonl(
        scenarios,
        [{"scenario_id": "s1", "prompt": "question", "conversation_context": []}],
    )
    _write_jsonl(
        rubrics,
        [
            {"scenario_id": "s1", "criterion_id": "c1", "criterion": "one"},
            {"scenario_id": "s1", "criterion_id": "c2", "criterion": "two"},
        ],
    )
    _write_jsonl(
        responses / "model.responses.jsonl",
        [{"Model": "org/model", "Scenario": "s1", "Output": "   ", "Finish Reason": "stop"}],
    )

    cases, counts = regrade.build_cases(responses, rubrics, scenarios, model_limit=None)

    assert len(cases) == 2
    assert counts["models"] == 1
    assert counts["criteria"] == 2
    assert counts["auto_fail_empty"] == 2
    assert counts["gradable"] == 0
    assert {case["auto_fail_reason"] for case in cases} == {"empty_output"}


def test_gateway_credentials_accept_existing_tfy_names() -> None:
    resolved = regrade._resolve_gateway(
        {"TFY_BASE_URL": "https://gateway.example/openai/", "TFY_API_KEY": "secret"}
    )
    assert resolved == (
        "https://gateway.example/openai",
        "secret",
        "TFY_BASE_URL",
        "TFY_API_KEY",
    )


def test_terminal_verdict_file_rejects_error_rows(tmp_path: Path) -> None:
    verdicts = tmp_path / "verdicts.jsonl"
    _write_jsonl(
        verdicts,
        [{"model": "m", "scenario_id": "s", "criterion_id": "c", "status": "error"}],
    )
    with pytest.raises(ValueError, match="Non-terminal status"):
        regrade._load_terminal_records(verdicts)


def test_summary_covers_full_resumed_file(tmp_path: Path) -> None:
    verdicts = tmp_path / "verdicts.jsonl"
    _write_jsonl(
        verdicts,
        [
            {
                "model": "m",
                "scenario_id": "s",
                "criterion_id": "c1",
                "status": "ok",
                "verdict": "pass",
                "reason": "",
                "prompt_tokens": 10,
                "completion_tokens": 2,
            },
            {
                "model": "m",
                "scenario_id": "s",
                "criterion_id": "c2",
                "status": "auto_fail",
                "verdict": "fail",
                "reason": "empty_output",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        ],
    )

    summary, _ = regrade._summarize_verdicts(
        verdicts,
        {("m", "s", "c1"), ("m", "s", "c2")},
        wall_s=2.0,
        new_cells=1,
        new_judge_calls=1,
        errors_this_invocation=0,
    )

    assert summary["cells_total"] == 2
    assert summary["missing_cells"] == 0
    assert summary["auto_failed_total"] == 1
    assert summary["completed"] is True
    assert summary["tokens_total"]["prompt"] == 10


def test_frozen_strict_v1_prompt_hash() -> None:
    assert regrade._prompt_template_hash("generic-binary-strict") == (
        "d126853bc98a49ee151a8c235961f4340a95441937512c5738cdb85f24cd6d11"
    )
