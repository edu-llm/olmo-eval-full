"""Derived, human-readable reporting for EduLLM adaptive batch runs.

The batch manifest, summary, and model-results JSONL remain the authoritative
artifacts.  This module validates their reporting projection and produces
deterministic JSON, long-form CSV, and Markdown views without calculating a
global score, averaging skill estimates, or ranking tutor models.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import string
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

BATCH_REPORT_SCHEMA_VERSION = "edullm-adaptive-batch-report-v1"
BATCH_RESULTS_JSON = "batch_results.json"
BATCH_RESULTS_CSV = "batch_results.csv"
BATCH_REPORT_MARKDOWN = "BATCH_REPORT.md"
EVALUATION_SCOPE = "adaptive_selected_responses_only"

_TERMINAL_MODEL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
_SCIENCE_STATUSES = frozenset({"validated", "experimental"})
_HEX_SHA256 = frozenset("0123456789abcdef")
_MARKDOWN_PUNCTUATION = frozenset(string.punctuation)

CSV_COLUMNS = (
    "report_schema_version",
    "batch_status",
    "run_id",
    "scientific_status",
    "benchmark_id",
    "calibration_version",
    "policy_id",
    "evaluation_scope",
    "checkpoint_generation",
    "resume_contract_fingerprint",
    "attempts_committed",
    "models_with_metrics",
    "model_index",
    "candidate_id",
    "attempt_number",
    "committed_at",
    "model_id",
    "model_family",
    "model_revision",
    "model_status",
    "output_dir",
    "manifest_sha256",
    "has_metrics",
    "completed_units",
    "error",
    "warnings",
    "precision_reached",
    "stop_reason",
    "stop_se_method",
    "scenarios_administered",
    "criteria_observed",
    "criteria_no_decision",
    "no_decision_rate",
    "mwle_converged",
    "mwle_message",
    "critical_failures",
    "skill",
    "criteria_count",
    "eap_theta",
    "eap_se",
    "mwle_theta",
    "mwle_se",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{label} must be a {qualifier}string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _string_sequence(value: Any, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_string(item, f"{label}[{index}]"))
    return result


def _optional_sha256(value: Any, label: str) -> str | None:
    if value is None:
        return None
    digest = _string(value, label)
    if len(digest) != 64 or any(character not in _HEX_SHA256 for character in digest):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256 digest")
    return digest


def _skill_values(
    value: Any,
    skills: Sequence[str],
    label: str,
    *,
    nonnegative: bool = False,
) -> dict[str, float]:
    values = _mapping(value, label)
    missing = [skill for skill in skills if skill not in values]
    extra = sorted(set(values) - set(skills))
    if missing or extra:
        raise ValueError(f"{label} keys differ from skills_order; missing={missing}, extra={extra}")
    return {
        skill: _number(values[skill], f"{label}.{skill}", nonnegative=nonnegative)
        for skill in skills
    }


def _optional_error(value: Any, label: str) -> str | Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        _json_safe(value, path=label)
        return dict(value)
    raise ValueError(f"{label} must be null, a string, or an object")


def _skill_counts(value: Any, skills: Sequence[str], label: str) -> dict[str, int]:
    values = _mapping(value, label)
    missing = [skill for skill in skills if skill not in values]
    extra = sorted(set(values) - set(skills))
    if missing or extra:
        raise ValueError(f"{label} keys differ from skills_order; missing={missing}, extra={extra}")
    return {skill: _integer(values[skill], f"{label}.{skill}") for skill in skills}


def _json_safe(value: Any, *, path: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _json_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _json_safe(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported value {type(value).__name__}")


def _empty_skill_rows(skills: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "skill": skill,
            "criteria_count": None,
            "eap_theta": None,
            "eap_se": None,
            "mwle_theta": None,
            "mwle_se": None,
        }
        for skill in skills
    ]


def _model_report(row: Mapping[str, Any], skills: Sequence[str]) -> dict[str, Any]:
    model_index = _integer(row.get("model_index"), "model row model_index")
    prefix = f"model row {model_index}"
    status = _string(row.get("status"), f"{prefix}.status")
    if status not in _TERMINAL_MODEL_STATUSES:
        raise ValueError(f"{prefix}.status is not a terminal model status: {status!r}")

    warnings = _string_sequence(row.get("warnings", []), f"{prefix}.warnings")
    error = _optional_error(row.get("error"), f"{prefix}.error")
    if status == "succeeded" and error is not None:
        raise ValueError(f"{prefix}: a succeeded model cannot carry an error")

    model: dict[str, Any] = {
        "model_index": model_index,
        "candidate_id": _string(row.get("candidate_id"), f"{prefix}.candidate_id"),
        "attempt_number": _integer(row.get("attempt_number"), f"{prefix}.attempt_number"),
        "committed_at": _string(row.get("committed_at"), f"{prefix}.committed_at"),
        "model_id": _string(row.get("model_id"), f"{prefix}.model_id"),
        "model_family": _string(row.get("model_family"), f"{prefix}.model_family"),
        "model_revision": _string(row.get("model_revision"), f"{prefix}.model_revision"),
        "status": status,
        "output_dir": _string(row.get("output_dir"), f"{prefix}.output_dir"),
        "manifest_sha256": _optional_sha256(
            row.get("manifest_sha256"), f"{prefix}.manifest_sha256"
        ),
        "warnings": warnings,
        "error": error,
        "has_metrics": status == "succeeded",
        "completed_units": _integer(row.get("completed_units"), f"{prefix}.completed_units"),
    }
    if model["attempt_number"] < 1:
        raise ValueError(f"{prefix}.attempt_number must be positive")

    metrics = _mapping(row.get("metrics", {}), f"{prefix}.metrics")
    if status != "succeeded":
        model.update(
            {
                "precision_reached": None,
                "stop_reason": None,
                "stop_se_method": None,
                "scenarios_administered": None,
                "criteria_observed": None,
                "criteria_no_decision": None,
                "no_decision_rate": None,
                "mwle_converged": None,
                "mwle_message": None,
                "critical_failures": None,
                "skills": _empty_skill_rows(skills),
            }
        )
        return model

    precision_reached = metrics.get("precision_reached")
    if not isinstance(precision_reached, bool):
        raise ValueError(f"{prefix}.metrics.precision_reached must be boolean")
    stop_reason = _string(metrics.get("stop_reason"), f"{prefix}.metrics.stop_reason")
    stop_se_method = _string(metrics.get("stop_se_method"), f"{prefix}.metrics.stop_se_method")
    if stop_se_method not in {"eap", "online"}:
        raise ValueError(f"{prefix}.metrics.stop_se_method must be 'eap' or 'online'")
    scenarios_administered = _integer(
        metrics.get("scenarios_administered"),
        f"{prefix}.metrics.scenarios_administered",
    )
    criteria_observed = _integer(
        metrics.get("criteria_observed"), f"{prefix}.metrics.criteria_observed"
    )
    criteria_no_decision = _integer(
        metrics.get("criteria_no_decision"), f"{prefix}.metrics.criteria_no_decision"
    )
    decision_total = criteria_observed + criteria_no_decision
    no_decision_rate = criteria_no_decision / decision_total if decision_total else None

    mwle_converged = metrics.get("mwle_converged")
    if not isinstance(mwle_converged, bool):
        raise ValueError(f"{prefix}.metrics.mwle_converged must be boolean")
    mwle_message = _string(
        metrics.get("mwle_message"), f"{prefix}.metrics.mwle_message", nonempty=False
    )
    critical_failures = _string_sequence(
        metrics.get("critical_failures"), f"{prefix}.metrics.critical_failures"
    )
    criterion_counts = _skill_counts(metrics.get("counts"), skills, f"{prefix}.metrics.counts")

    theta_eap = _skill_values(metrics.get("theta_eap"), skills, f"{prefix}.metrics.theta_eap")
    se_eap = _skill_values(
        metrics.get("se_eap"), skills, f"{prefix}.metrics.se_eap", nonnegative=True
    )

    raw_theta_mwle = metrics.get("theta_mwle")
    raw_se_mwle = metrics.get("se_mwle")
    theta_mwle: dict[str, float] | None
    se_mwle: dict[str, float] | None
    if not mwle_converged:
        if raw_theta_mwle is not None or raw_se_mwle is not None:
            raise ValueError(
                f"{prefix}.metrics: non-converged MWLE must have null theta_mwle and se_mwle"
            )
        theta_mwle = None
        se_mwle = None
    else:
        if raw_theta_mwle is None:
            raise ValueError(f"{prefix}.metrics: converged MWLE requires theta_mwle")
        theta_mwle = _skill_values(raw_theta_mwle, skills, f"{prefix}.metrics.theta_mwle")
        se_mwle = (
            None
            if raw_se_mwle is None
            else _skill_values(
                raw_se_mwle,
                skills,
                f"{prefix}.metrics.se_mwle",
                nonnegative=True,
            )
        )

    model.update(
        {
            "precision_reached": precision_reached,
            "stop_reason": stop_reason,
            "stop_se_method": stop_se_method,
            "scenarios_administered": scenarios_administered,
            "criteria_observed": criteria_observed,
            "criteria_no_decision": criteria_no_decision,
            "no_decision_rate": no_decision_rate,
            "mwle_converged": mwle_converged,
            "mwle_message": mwle_message,
            "critical_failures": critical_failures,
            "skills": [
                {
                    "skill": skill,
                    "criteria_count": criterion_counts[skill],
                    "eap_theta": theta_eap[skill],
                    "eap_se": se_eap[skill],
                    "mwle_theta": None if theta_mwle is None else theta_mwle[skill],
                    "mwle_se": None if se_mwle is None else se_mwle[skill],
                }
                for skill in skills
            ],
        }
    )
    return model


def build_batch_report(
    batch_manifest: Mapping[str, Any],
    batch_summary: Mapping[str, Any],
    model_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and validate the deterministic derived JSON report."""

    manifest = _mapping(batch_manifest, "batch manifest")
    summary = _mapping(batch_summary, "batch summary")
    bank = _mapping(manifest.get("bank"), "batch manifest.bank")

    manifest_status = _string(manifest.get("status"), "batch manifest.status")
    summary_status = _string(summary.get("status"), "batch summary.status")
    if manifest_status != summary_status:
        raise ValueError(
            "batch manifest.status and batch summary.status differ: "
            f"{manifest_status!r} != {summary_status!r}"
        )

    skills = _string_sequence(bank.get("skills_order"), "batch manifest.bank.skills_order")
    if not skills or len(skills) != len(set(skills)):
        raise ValueError("batch manifest.bank.skills_order must be non-empty and unique")
    scientific_status = _string(
        bank.get("scientific_status"), "batch manifest.bank.scientific_status"
    )
    if scientific_status not in _SCIENCE_STATUSES:
        raise ValueError(
            "batch manifest.bank.scientific_status must be 'validated' or 'experimental'"
        )

    models = [
        _model_report(_mapping(row, f"model_rows[{index}]"), skills)
        for index, row in enumerate(model_rows)
    ]
    for field in ("model_index", "candidate_id", "model_id", "output_dir"):
        values = [model[field] for model in models]
        if len(values) != len(set(values)):
            raise ValueError(f"model rows contain duplicate {field} values")

    count_names = (
        "models_total",
        "models_completed",
        "models_succeeded",
        "models_failed",
        "models_cancelled",
        "models_pending",
    )
    counts = {name: _integer(summary.get(name), f"batch summary.{name}") for name in count_names}
    checkpoint_generation = _integer(
        summary.get("checkpoint_generation"), "batch summary.checkpoint_generation"
    )
    attempts_committed = _integer(
        summary.get("attempts_committed"), "batch summary.attempts_committed"
    )
    resume_fingerprint = _optional_sha256(
        summary.get("resume_contract_fingerprint"),
        "batch summary.resume_contract_fingerprint",
    )
    manifest_generation = _integer(
        manifest.get("checkpoint_generation"), "batch manifest.checkpoint_generation"
    )
    manifest_fingerprint = _optional_sha256(
        manifest.get("resume_contract_fingerprint"),
        "batch manifest.resume_contract_fingerprint",
    )
    if manifest_generation != checkpoint_generation:
        raise ValueError("batch manifest and summary checkpoint generations differ")
    if manifest_fingerprint != resume_fingerprint:
        raise ValueError("batch manifest and summary resume fingerprints differ")
    observed_counts = {
        "models_completed": len(models),
        "models_succeeded": sum(model["status"] == "succeeded" for model in models),
        "models_failed": sum(model["status"] == "failed" for model in models),
        "models_cancelled": sum(model["status"] == "cancelled" for model in models),
    }
    for name, observed in observed_counts.items():
        if counts[name] != observed:
            raise ValueError(
                f"batch summary.{name} does not match model rows: {counts[name]} != {observed}"
            )
    if counts["models_total"] != counts["models_completed"] + counts["models_pending"]:
        raise ValueError("batch summary model totals do not reconcile")
    if counts["models_completed"] != (
        counts["models_succeeded"] + counts["models_failed"] + counts["models_cancelled"]
    ):
        raise ValueError("batch summary completed model statuses do not reconcile")
    if attempts_committed < counts["models_completed"]:
        raise ValueError("batch summary attempts_committed cannot be below models_completed")

    models_with_metrics = [model for model in models if model["has_metrics"]]
    scenarios_total = sum(model["scenarios_administered"] for model in models_with_metrics)
    criteria_observed_total = sum(model["criteria_observed"] for model in models_with_metrics)
    no_decision_total = sum(model["criteria_no_decision"] for model in models_with_metrics)
    decision_total = criteria_observed_total + no_decision_total

    report = {
        "schema_version": BATCH_REPORT_SCHEMA_VERSION,
        "status": manifest_status,
        "run_id": _string(manifest.get("run_id"), "batch manifest.run_id"),
        "benchmark_id": _string(bank.get("benchmark_id"), "batch manifest.bank.benchmark_id"),
        "calibration_version": _string(
            bank.get("calibration_version"), "batch manifest.bank.calibration_version"
        ),
        "policy_id": _string(bank.get("policy_id"), "batch manifest.bank.policy_id"),
        "scientific_status": scientific_status,
        "checkpoint_generation": checkpoint_generation,
        "resume_contract_fingerprint": resume_fingerprint,
        "skills_order": skills,
        "evaluation_scope": EVALUATION_SCOPE,
        "response_batch_sha256": _optional_sha256(
            summary.get("response_batch_sha256"), "batch summary.response_batch_sha256"
        ),
        "authoritative_artifacts": {
            "manifest": "manifest.json",
            "batch_summary": "batch_summary.json",
            "model_results": _string(
                summary.get("model_results_path"), "batch summary.model_results_path"
            ),
            "attempt_history": _string(
                summary.get("attempt_history_path"), "batch summary.attempt_history_path"
            ),
            "checkpoint_pointer": "checkpoints/latest.json",
        },
        "completion": {
            **counts,
            "attempts_committed": attempts_committed,
            "models_with_metrics": len(models_with_metrics),
            "metrics_denominator_label": "models with metrics",
            "scenarios_administered_for_models_with_metrics": scenarios_total,
            "criteria_observed_for_models_with_metrics": criteria_observed_total,
            "criteria_no_decision_for_models_with_metrics": no_decision_total,
            "no_decision_rate_for_models_with_metrics": (
                no_decision_total / decision_total if decision_total else None
            ),
        },
        "models": models,
    }
    _json_safe(report, path="batch report")
    return report


def _csv_text(value: str) -> str:
    """Neutralize spreadsheet formulas while preserving the original text."""

    significant = value.lstrip(" \t\r\n")
    if significant.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _csv_text(value)
    return value


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return _csv_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def build_batch_csv(report: Mapping[str, Any]) -> str:
    """Render one detached CSV row for every completed model and skill."""

    value = _mapping(report, "batch report")
    skills = _string_sequence(value.get("skills_order"), "batch report.skills_order")
    models = value.get("models")
    if isinstance(models, (str, bytes)) or not isinstance(models, Sequence):
        raise ValueError("batch report.models must be an array")
    completion = _mapping(value.get("completion"), "batch report.completion")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()

    run_values = {
        "report_schema_version": value.get("schema_version"),
        "batch_status": value.get("status"),
        "run_id": value.get("run_id"),
        "scientific_status": value.get("scientific_status"),
        "benchmark_id": value.get("benchmark_id"),
        "calibration_version": value.get("calibration_version"),
        "policy_id": value.get("policy_id"),
        "evaluation_scope": value.get("evaluation_scope"),
        "checkpoint_generation": value.get("checkpoint_generation"),
        "resume_contract_fingerprint": value.get("resume_contract_fingerprint"),
        "attempts_committed": completion.get("attempts_committed"),
        "models_with_metrics": completion.get("models_with_metrics"),
    }
    for model_index, raw_model in enumerate(models):
        model = _mapping(raw_model, f"batch report.models[{model_index}]")
        raw_skill_rows = model.get("skills")
        if isinstance(raw_skill_rows, (str, bytes)) or not isinstance(raw_skill_rows, Sequence):
            raise ValueError(f"batch report.models[{model_index}].skills must be an array")
        skill_rows = [_mapping(item, "skill row") for item in raw_skill_rows]
        if [item.get("skill") for item in skill_rows] != skills:
            raise ValueError(f"batch report.models[{model_index}].skills do not match skills_order")
        common = {
            **run_values,
            "model_index": model.get("model_index"),
            "candidate_id": model.get("candidate_id"),
            "attempt_number": model.get("attempt_number"),
            "committed_at": model.get("committed_at"),
            "model_id": model.get("model_id"),
            "model_family": model.get("model_family"),
            "model_revision": model.get("model_revision"),
            "model_status": model.get("status"),
            "output_dir": model.get("output_dir"),
            "manifest_sha256": model.get("manifest_sha256"),
            "has_metrics": model.get("has_metrics"),
            "completed_units": model.get("completed_units"),
            "error": _json_cell(model.get("error"))
            if isinstance(model.get("error"), Mapping)
            else model.get("error"),
            "warnings": _json_cell(model.get("warnings")),
            "precision_reached": model.get("precision_reached"),
            "stop_reason": model.get("stop_reason"),
            "stop_se_method": model.get("stop_se_method"),
            "scenarios_administered": model.get("scenarios_administered"),
            "criteria_observed": model.get("criteria_observed"),
            "criteria_no_decision": model.get("criteria_no_decision"),
            "no_decision_rate": model.get("no_decision_rate"),
            "mwle_converged": model.get("mwle_converged"),
            "mwle_message": model.get("mwle_message"),
            "critical_failures": _json_cell(model.get("critical_failures")),
        }
        for skill_row in skill_rows:
            row = {
                **common,
                "skill": skill_row.get("skill"),
                "criteria_count": skill_row.get("criteria_count"),
                "eap_theta": skill_row.get("eap_theta"),
                "eap_se": skill_row.get("eap_se"),
                "mwle_theta": skill_row.get("mwle_theta"),
                "mwle_se": skill_row.get("mwle_se"),
            }
            writer.writerow({column: _csv_cell(row.get(column)) for column in CSV_COLUMNS})
    return output.getvalue()


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    # GFM parses character references as literal text rather than feeding the
    # decoded punctuation back through its inline parser. Keep each reference
    # in a trusted span too: GitHub's email autolinker can otherwise recombine
    # adjacent text and character-reference tokens into an active mailto link.
    text = "".join(
        f"<span>&#{ord(character)};</span>" if character in _MARKDOWN_PUNCTUATION else character
        for character in text
    )
    return text.replace("\n", "<br>")


def _markdown_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return format(value, ".10g")
    return _markdown_cell(value)


def _markdown_table(rows: Sequence[tuple[Any, Any]]) -> list[str]:
    result = ["| Field | Value |", "|---|---|"]
    result.extend(f"| {_markdown_cell(name)} | {_markdown_cell(value)} |" for name, value in rows)
    return result


def build_batch_markdown(report: Mapping[str, Any]) -> str:
    """Render a readable scientific report without rankings or aggregate scores."""

    value = _mapping(report, "batch report")
    completion = _mapping(value.get("completion"), "batch report.completion")
    models = value.get("models")
    if isinstance(models, (str, bytes)) or not isinstance(models, Sequence):
        raise ValueError("batch report.models must be an array")

    scientific_status = _string(value.get("scientific_status"), "batch report.scientific_status")
    lines = ["# EduLLM Adaptive Batch Report", ""]
    if scientific_status == "experimental":
        lines.extend(
            [
                "> [!WARNING]",
                "> **SCIENTIFIC STATUS: EXPERIMENTAL**",
                "> These estimates are experimental and must not be presented as "
                "validated results.",
            ]
        )
    else:
        lines.extend(["> **SCIENTIFIC STATUS: VALIDATED**"])
    lines.extend(
        [
            "",
            "> Only adaptively selected tutor responses were judged. Uploaded but unselected "
            "responses were not judged.",
            "> This report does not compute model rankings, a global score, or cross-skill "
            "averages.",
            "",
            "## Run context",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            (
                ("Run ID", value.get("run_id")),
                ("Batch status", value.get("status")),
                ("Benchmark", value.get("benchmark_id")),
                ("Calibration version", value.get("calibration_version")),
                ("Policy ID", value.get("policy_id")),
                ("Scientific status", scientific_status),
                ("Evaluation scope", value.get("evaluation_scope")),
                ("Checkpoint generation", value.get("checkpoint_generation")),
                ("Resume fingerprint", value.get("resume_contract_fingerprint")),
            )
        )
    )
    lines.extend(["", "## Completion", ""])
    lines.extend(
        _markdown_table(
            (
                ("Models total", completion.get("models_total")),
                ("Models completed", completion.get("models_completed")),
                ("Models succeeded", completion.get("models_succeeded")),
                ("Models failed", completion.get("models_failed")),
                ("Models cancelled", completion.get("models_cancelled")),
                ("Models pending", completion.get("models_pending")),
                ("Committed attempts", completion.get("attempts_committed")),
                ("Models with metrics", completion.get("models_with_metrics")),
                (
                    "Scenarios administered (models with metrics)",
                    completion.get("scenarios_administered_for_models_with_metrics"),
                ),
                (
                    "Criteria observed (models with metrics)",
                    completion.get("criteria_observed_for_models_with_metrics"),
                ),
                (
                    "No decisions (models with metrics)",
                    completion.get("criteria_no_decision_for_models_with_metrics"),
                ),
                (
                    "No-decision rate (models with metrics)",
                    completion.get("no_decision_rate_for_models_with_metrics"),
                ),
            )
        )
    )

    if not models:
        lines.extend(["", "No model results have been recorded yet."])
    for position, raw_model in enumerate(models, start=1):
        model = _mapping(raw_model, f"batch report.models[{position - 1}]")
        lines.extend(["", f"## Model {position}", ""])
        lines.extend(
            _markdown_table(
                (
                    ("Model ID", model.get("model_id")),
                    ("Model family", model.get("model_family")),
                    ("Model revision", model.get("model_revision")),
                    ("Attempt number", model.get("attempt_number")),
                    ("Attempt committed at", model.get("committed_at")),
                    ("Status", model.get("status")),
                    ("Output directory", model.get("output_dir")),
                    ("Metrics available", model.get("has_metrics")),
                    ("Completed units", model.get("completed_units")),
                    ("Error", model.get("error")),
                    ("Warnings", json.dumps(model.get("warnings"), ensure_ascii=False)),
                    ("Precision reached", model.get("precision_reached")),
                    ("Stop reason", model.get("stop_reason")),
                    ("Stopping SE method", model.get("stop_se_method")),
                    ("Scenarios administered", model.get("scenarios_administered")),
                    ("Criteria observed", model.get("criteria_observed")),
                    ("No decisions", model.get("criteria_no_decision")),
                    ("No-decision rate", model.get("no_decision_rate")),
                    ("MWLE converged", model.get("mwle_converged")),
                    ("MWLE message", model.get("mwle_message")),
                    (
                        "Critical failures",
                        (
                            None
                            if model.get("critical_failures") is None
                            else json.dumps(model.get("critical_failures"), ensure_ascii=False)
                        ),
                    ),
                )
            )
        )
        lines.extend(
            [
                "",
                "### Skill estimates",
                "",
                "| Skill | Criteria observed | EAP theta | EAP SE | MWLE theta | MWLE SE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        raw_skills = model.get("skills")
        if isinstance(raw_skills, (str, bytes)) or not isinstance(raw_skills, Sequence):
            raise ValueError(f"batch report.models[{position - 1}].skills must be an array")
        for raw_skill in raw_skills:
            skill = _mapping(raw_skill, "skill row")
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_cell(skill.get("skill")),
                        _markdown_number(skill.get("criteria_count")),
                        _markdown_number(skill.get("eap_theta")),
                        _markdown_number(skill.get("eap_se")),
                        _markdown_number(skill.get("mwle_theta")),
                        _markdown_number(skill.get("mwle_se")),
                    )
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: str | Path, payload: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8", newline="")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_batch_report_json(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Atomically write the derived JSON report."""

    _json_safe(report, path="batch report")
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    return _atomic_write_text(path, payload)


def write_batch_report_csv(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Atomically write the derived long-form CSV report."""

    return _atomic_write_text(path, build_batch_csv(report))


def write_batch_report_markdown(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Atomically write the derived Markdown report."""

    return _atomic_write_text(path, build_batch_markdown(report))


def write_batch_reports(output_dir: str | Path, report: Mapping[str, Any]) -> dict[str, Path]:
    """Atomically write all three derived views into ``output_dir``."""

    root = Path(output_dir)
    paths = {
        BATCH_RESULTS_JSON: root / BATCH_RESULTS_JSON,
        BATCH_RESULTS_CSV: root / BATCH_RESULTS_CSV,
        BATCH_REPORT_MARKDOWN: root / BATCH_REPORT_MARKDOWN,
    }
    write_batch_report_json(paths[BATCH_RESULTS_JSON], report)
    write_batch_report_csv(paths[BATCH_RESULTS_CSV], report)
    write_batch_report_markdown(paths[BATCH_REPORT_MARKDOWN], report)
    return paths


__all__ = [
    "BATCH_REPORT_MARKDOWN",
    "BATCH_REPORT_SCHEMA_VERSION",
    "BATCH_RESULTS_CSV",
    "BATCH_RESULTS_JSON",
    "CSV_COLUMNS",
    "EVALUATION_SCOPE",
    "build_batch_csv",
    "build_batch_markdown",
    "build_batch_report",
    "write_batch_report_csv",
    "write_batch_report_json",
    "write_batch_report_markdown",
    "write_batch_reports",
]
