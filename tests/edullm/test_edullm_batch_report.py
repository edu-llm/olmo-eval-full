"""Focused tests for deterministic EduLLM batch reporting."""

from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
from typing import Any

import pytest

from olmo_eval.edullm.batch_report import (
    BATCH_REPORT_MARKDOWN,
    BATCH_REPORT_SCHEMA_VERSION,
    BATCH_RESULTS_CSV,
    BATCH_RESULTS_JSON,
    CSV_COLUMNS,
    build_batch_csv,
    build_batch_markdown,
    build_batch_report,
    write_batch_reports,
)


def _inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = {
        "status": "failed",
        "run_id": "run-123",
        "checkpoint_generation": 3,
        "resume_contract_fingerprint": "d" * 64,
        "bank": {
            "benchmark_id": "infobench",
            "calibration_version": "calibration-v2",
            "policy_id": "cat-policy-v3",
            "scientific_status": "experimental",
            "skills_order": ["content", "scaffolding"],
        },
    }
    summary = {
        "status": "failed",
        "checkpoint_generation": 3,
        "resume_contract_fingerprint": "d" * 64,
        "models_total": 2,
        "models_completed": 2,
        "models_succeeded": 1,
        "models_failed": 1,
        "models_cancelled": 0,
        "models_pending": 0,
        "response_batch_sha256": "b" * 64,
        "model_results_path": "model_results.jsonl",
        "attempt_history_path": "attempt_history.jsonl",
        "attempts_committed": 2,
    }
    model_rows = [
        {
            "model_index": 0,
            "candidate_id": "candidate-0000-safe",
            "attempt_number": 1,
            "committed_at": "2026-08-08T12:00:00+00:00",
            "model_id": '=HYPERLINK("https://invalid")|<script>\nnext',
            "model_family": "family-a",
            "model_revision": "revision-a",
            "output_dir": "models/candidate-0000-safe",
            "status": "succeeded",
            "completed_units": 2,
            "metrics": {
                "stop_reason": "max_scenarios_reached",
                "stop_se_method": "eap",
                "precision_reached": False,
                "scenarios_administered": 2,
                "criteria_observed": 3,
                "criteria_no_decision": 1,
                "mwle_converged": True,
                "mwle_message": "converged",
                "counts": {"content": 2, "scaffolding": 1},
                "theta_eap": {"content": 0.2, "scaffolding": -0.3},
                "se_eap": {"content": 0.4, "scaffolding": 0.5},
                "theta_mwle": {"content": 0.1, "scaffolding": -0.2},
                "se_mwle": {"content": 0.6, "scaffolding": 0.7},
                "critical_failures": ["criterion-1"],
            },
            "warnings": ["+untrusted warning"],
            "error": None,
            "manifest_sha256": "a" * 64,
        },
        {
            "model_index": 1,
            "candidate_id": "candidate-0001-safe",
            "attempt_number": 1,
            "committed_at": "2026-08-08T12:01:00+00:00",
            "model_id": "failed-model",
            "model_family": "family-b",
            "model_revision": "revision-b",
            "output_dir": "models/candidate-0001-safe",
            "status": "failed",
            "completed_units": 1,
            # Partial progress is intentionally not promoted to comparable metrics.
            "metrics": {"scenarios_administered": 1},
            "warnings": [],
            "error": "@untrusted error",
            "manifest_sha256": "c" * 64,
        },
    ]
    return manifest, summary, model_rows


def test_build_report_preserves_nulls_and_labels_metric_denominators() -> None:
    manifest, summary, model_rows = _inputs()

    report = build_batch_report(manifest, summary, model_rows)

    assert report["schema_version"] == BATCH_REPORT_SCHEMA_VERSION
    assert report["scientific_status"] == "experimental"
    assert report["skills_order"] == ["content", "scaffolding"]
    assert report["completion"] == {
        "models_total": 2,
        "models_completed": 2,
        "models_succeeded": 1,
        "models_failed": 1,
        "models_cancelled": 0,
        "models_pending": 0,
        "attempts_committed": 2,
        "models_with_metrics": 1,
        "metrics_denominator_label": "models with metrics",
        "scenarios_administered_for_models_with_metrics": 2,
        "criteria_observed_for_models_with_metrics": 3,
        "criteria_no_decision_for_models_with_metrics": 1,
        "no_decision_rate_for_models_with_metrics": 0.25,
    }
    succeeded, failed = report["models"]
    assert succeeded["no_decision_rate"] == 0.25
    assert succeeded["skills"][0] == {
        "skill": "content",
        "criteria_count": 2,
        "eap_theta": 0.2,
        "eap_se": 0.4,
        "mwle_theta": 0.1,
        "mwle_se": 0.6,
    }
    assert failed["has_metrics"] is False
    assert failed["scenarios_administered"] is None
    assert failed["critical_failures"] is None
    assert all(skill["eap_theta"] is None for skill in failed["skills"])


def test_csv_is_long_form_and_neutralizes_formula_injection() -> None:
    manifest, summary, model_rows = _inputs()
    report = build_batch_report(manifest, summary, model_rows)

    payload = build_batch_csv(report)
    rows = list(csv.DictReader(io.StringIO(payload)))

    assert tuple(rows[0]) == CSV_COLUMNS
    assert len(rows) == 4
    assert {(row["model_id"], row["skill"]) for row in rows[2:]} == {
        ("failed-model", "content"),
        ("failed-model", "scaffolding"),
    }
    assert rows[0]["model_id"].startswith("'=")
    assert rows[0]["scientific_status"] == "experimental"
    assert rows[0]["models_with_metrics"] == "1"
    assert rows[2]["has_metrics"] == "false"
    assert rows[2]["scenarios_administered"] == ""
    assert rows[2]["eap_theta"] == ""
    assert rows[2]["error"].startswith("'@")


def test_markdown_marks_experimental_scope_and_escapes_untrusted_text() -> None:
    manifest, summary, model_rows = _inputs()
    report = build_batch_report(manifest, summary, model_rows)

    payload = build_batch_markdown(report)

    assert "**SCIENTIFIC STATUS: EXPERIMENTAL**" in payload
    assert "Only adaptively selected tutor responses were judged" in payload
    assert "does not compute model rankings, a global score, or cross-skill averages" in payload
    assert "<span>&#60;</span>script<span>&#62;</span>" in payload
    assert "<span>&#124;</span>" in payload
    assert "<br>next" in payload
    assert "| failed<span>&#45;</span>model |" in payload


def test_markdown_encodes_active_gfm_as_literal_table_text() -> None:
    manifest, summary, model_rows = _inputs()
    report = build_batch_report(manifest, summary, model_rows)
    attack = (
        "![pixel](https://attacker.invalid/pixel.png) "
        "[click](javascript:alert(1)) <https://attacker.invalid/autolink> "
        "www.attacker.invalid user@attacker.invalid <img src=x>"
        "\n# injected heading\n```html\n<script>alert(1)</script>\n```"
        "\nleft|right `code` **bold** &#33;"
    )
    report["models"][0]["model_id"] = attack

    payload = build_batch_markdown(report)
    model_id_row = next(line for line in payload.splitlines() if line.startswith("| Model ID |"))
    encoded_value = model_id_row.removeprefix("| Model ID | ").removesuffix(" |")

    assert model_id_row.count("|") == 3
    assert encoded_value.count("<br>") == attack.count("\n")
    rendered_text = html.unescape(encoded_value).replace("<span>", "").replace("</span>", "")
    assert rendered_text.replace("<br>", "\n") == attack
    for active_gfm in (
        "![pixel](",
        "[click](",
        "https://",
        "www.attacker.invalid",
        "user@attacker.invalid",
        "<img",
        "# injected heading",
        "```html",
        "<script>",
        "`code`",
        "**bold**",
        "left|right",
    ):
        assert active_gfm not in encoded_value
    assert "<span>&#64;</span>" in encoded_value
    assert "<span>&#38;</span><span>&#35;</span>33<span>&#59;</span>" in encoded_value


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda metrics: metrics["theta_eap"].pop("scaffolding"),
            "theta_eap keys differ from skills_order",
        ),
        (
            lambda metrics: metrics.update(
                {"mwle_converged": False, "mwle_message": "did not converge"}
            ),
            "non-converged MWLE must have null theta_mwle and se_mwle",
        ),
        (
            lambda metrics: metrics.update({"theta_mwle": None}),
            "converged MWLE requires theta_mwle",
        ),
    ],
)
def test_report_rejects_skill_or_mwle_inconsistency(mutation: Any, match: str) -> None:
    manifest, summary, model_rows = _inputs()
    mutation(model_rows[0]["metrics"])

    with pytest.raises(ValueError, match=match):
        build_batch_report(manifest, summary, model_rows)


def test_report_rejects_summary_counts_that_hide_models() -> None:
    manifest, summary, model_rows = _inputs()
    summary["models_completed"] = 1

    with pytest.raises(ValueError, match="models_completed does not match model rows"):
        build_batch_report(manifest, summary, model_rows)


def test_atomic_writers_replace_all_three_reports(tmp_path: Path) -> None:
    manifest, summary, model_rows = _inputs()
    report = build_batch_report(manifest, summary, model_rows)
    for filename in (BATCH_RESULTS_JSON, BATCH_RESULTS_CSV, BATCH_REPORT_MARKDOWN):
        (tmp_path / filename).write_text("stale", encoding="utf-8")

    paths = write_batch_reports(tmp_path, report)

    assert set(paths) == {BATCH_RESULTS_JSON, BATCH_RESULTS_CSV, BATCH_REPORT_MARKDOWN}
    assert json.loads((tmp_path / BATCH_RESULTS_JSON).read_text()) == report
    assert (tmp_path / BATCH_RESULTS_CSV).read_text().startswith("report_schema_version,")
    assert (
        (tmp_path / BATCH_REPORT_MARKDOWN)
        .read_text()
        .startswith("# EduLLM Adaptive Batch Report\n")
    )
    assert not list(tmp_path.glob(".*.tmp"))


def test_running_empty_report_has_csv_header_and_no_model_message() -> None:
    manifest, summary, _ = _inputs()
    manifest["status"] = "running"
    summary.update(
        {
            "status": "running",
            "models_completed": 0,
            "models_succeeded": 0,
            "models_failed": 0,
            "models_pending": 2,
        }
    )

    report = build_batch_report(manifest, summary, [])

    assert build_batch_csv(report).count("\n") == 1
    assert "No model results have been recorded yet." in build_batch_markdown(report)
    assert report["completion"]["no_decision_rate_for_models_with_metrics"] is None
