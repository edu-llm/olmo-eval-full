#!/usr/bin/env python3
"""Reproduce the frozen InFoBench baseline from stored artifacts only.

This checker is deliberately read-only with respect to calibration: it hashes and
parses the frozen inputs, reconciles criterion identities through the fitted-bank
export, and recomputes headline summary statistics from stored CSV/JSON outputs.
It never imports or invokes a fitter, CAT runner, judge, or model endpoint.

Use ``--write`` to create the deterministic reproduction report and criterion-flow
table declared by the config. Use ``--check`` to prove that checked-in copies are
byte-for-byte equal to a fresh reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tarfile
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "infobench-baseline-reproduction-v1"
DEFAULT_CONFIG = "configs/infobench_baseline_reproduction.json"


class BaselineVerificationError(RuntimeError):
    """Raised when a frozen artifact or reproduced metric does not match."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BaselineVerificationError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineVerificationError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BaselineVerificationError(f"{label} must contain one JSON object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BaselineVerificationError(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BaselineVerificationError(
                    f"invalid JSON in {label} {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise BaselineVerificationError(
                    f"{label} {path}:{line_number} must contain a JSON object"
                )
            rows.append(row)
    return rows


def _read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise BaselineVerificationError(f"missing {label}: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise BaselineVerificationError(f"{label} has no CSV header: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _require_unique(values: Sequence[str], label: str) -> None:
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise BaselineVerificationError(f"duplicate {label}: {preview}")


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise BaselineVerificationError(
            f"{label} mismatch: expected {expected!r}, observed {actual!r}"
        )


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise BaselineVerificationError(
            f"{label} mismatch: expected {expected:.17g}, observed {actual:.17g}"
        )


def _as_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise BaselineVerificationError(f"{label} must be True or False, observed {value!r}")


def _as_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise BaselineVerificationError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise BaselineVerificationError(f"{label} is not finite: {value!r}")
    return parsed


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise BaselineVerificationError("cannot calculate a mean from no values")
    return math.fsum(values) / len(values)


def _pearson(reference: Sequence[float], estimate: Sequence[float]) -> float:
    if len(reference) != len(estimate) or len(reference) < 2:
        raise BaselineVerificationError("correlation requires aligned vectors of length >= 2")
    ref_mean = _mean(reference)
    est_mean = _mean(estimate)
    covariance = math.fsum(
        (left - ref_mean) * (right - est_mean)
        for left, right in zip(reference, estimate)
    )
    ref_ss = math.fsum((value - ref_mean) ** 2 for value in reference)
    est_ss = math.fsum((value - est_mean) ** 2 for value in estimate)
    denominator = math.sqrt(ref_ss * est_ss)
    if denominator == 0.0:
        raise BaselineVerificationError("correlation is undefined for a constant vector")
    return covariance / denominator


def _slope(reference: Sequence[float], estimate: Sequence[float]) -> float:
    ref_mean = _mean(reference)
    est_mean = _mean(estimate)
    denominator = math.fsum((value - ref_mean) ** 2 for value in reference)
    if denominator == 0.0:
        raise BaselineVerificationError("slope is undefined for a constant reference")
    return math.fsum(
        (left - ref_mean) * (right - est_mean)
        for left, right in zip(reference, estimate)
    ) / denominator


def _recovery(reference: Sequence[float], estimate: Sequence[float]) -> dict[str, float]:
    return {
        "r": _pearson(reference, estimate),
        "slope": _slope(reference, estimate),
        "mae": _mean([abs(left - right) for left, right in zip(reference, estimate)]),
    }


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise BaselineVerificationError("cannot calculate a quantile from no values")
    if not 0.0 <= probability <= 1.0:
        raise BaselineVerificationError("quantile probability must be between zero and one")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def _assert_expected_subset(actual: Any, expected: Any, label: str = "expected") -> None:
    """Validate a declarative expected-value subset with strict numeric tolerance."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise BaselineVerificationError(f"{label} must be an object")
        for key, expected_value in expected.items():
            if key not in actual:
                raise BaselineVerificationError(f"{label}.{key} is missing")
            _assert_expected_subset(actual[key], expected_value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise BaselineVerificationError(f"{label} must be a list")
        _require_equal(len(actual), len(expected), f"{label} length")
        for index, (actual_value, expected_value) in enumerate(
            zip(actual, expected)
        ):
            _assert_expected_subset(actual_value, expected_value, f"{label}[{index}]")
        return
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise BaselineVerificationError(f"{label} must be numeric")
        _require_close(float(actual), expected, label)
        return
    _require_equal(actual, expected, label)


def _verify_frozen_artifacts(
    repo_root: Path, config: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    artifacts = config.get("frozen_inputs")
    if not isinstance(artifacts, dict) or not artifacts:
        raise BaselineVerificationError("config frozen_inputs must be a non-empty object")
    report: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            raise BaselineVerificationError(f"frozen_inputs.{name} must be an object")
        relative_path = str(artifact.get("path") or "")
        expected_hash = str(artifact.get("sha256") or "")
        if not relative_path or len(expected_hash) != 64:
            raise BaselineVerificationError(
                f"frozen_inputs.{name} must declare path and SHA-256"
            )
        path = _resolve(repo_root, relative_path)
        if not path.is_file():
            raise BaselineVerificationError(f"missing frozen input {name}: {path}")
        observed_hash = sha256_file(path)
        _require_equal(observed_hash, expected_hash, f"{name} SHA-256")
        report[name] = {
            "path": relative_path,
            "sha256": observed_hash,
            "bytes": path.stat().st_size,
        }
        paths[name] = path
    return report, paths


def _verify_matrix(
    matrix_path: Path,
    rubrics_path: Path,
    scenarios_path: Path,
) -> tuple[dict[str, Any], set[str], set[str], list[str]]:
    header, matrix_rows = _read_csv(matrix_path, "response matrix")
    if header[0] != "model":
        raise BaselineVerificationError("response matrix first column must be 'model'")
    criterion_ids = header[1:]
    model_ids = [row["model"] for row in matrix_rows]
    _require_unique(criterion_ids, "response-matrix criterion IDs")
    _require_unique(model_ids, "response-matrix model IDs")
    if any(not value for value in criterion_ids + model_ids):
        raise BaselineVerificationError("response matrix contains a blank model or criterion ID")

    counts: Counter[str] = Counter()
    for row_number, row in enumerate(matrix_rows, 2):
        for criterion_id in criterion_ids:
            value = row.get(criterion_id)
            if value not in {"0", "1", ""}:
                raise BaselineVerificationError(
                    f"response matrix {matrix_path}:{row_number} has invalid value "
                    f"{value!r} for {criterion_id}"
                )
            counts[value] += 1

    rubric_rows = _read_jsonl(rubrics_path, "source rubrics")
    source_ids = [str(row.get("criterion_id") or "") for row in rubric_rows]
    _require_unique(source_ids, "source-rubric criterion IDs")
    _require_equal(set(criterion_ids), set(source_ids), "matrix/source-rubric criterion IDs")

    scenario_rows = _read_jsonl(scenarios_path, "source scenarios")
    scenario_ids = [str(row.get("scenario_id") or "") for row in scenario_rows]
    _require_unique(scenario_ids, "source scenario IDs")
    scenario_criterion_ids: list[str] = []
    criterion_to_scenario: dict[str, str] = {}
    for row in scenario_rows:
        scenario_id = str(row.get("scenario_id") or "")
        values = row.get("criterion_ids")
        if not isinstance(values, list):
            raise BaselineVerificationError(
                f"source scenario {scenario_id!r} must contain criterion_ids"
            )
        for criterion_id in values:
            criterion_text = str(criterion_id)
            scenario_criterion_ids.append(criterion_text)
            if criterion_text in criterion_to_scenario:
                raise BaselineVerificationError(
                    f"criterion {criterion_text} occurs in more than one scenario"
                )
            criterion_to_scenario[criterion_text] = scenario_id
    _require_equal(
        set(scenario_criterion_ids), set(source_ids), "scenario/source-rubric criterion IDs"
    )
    for rubric in rubric_rows:
        criterion_id = str(rubric["criterion_id"])
        _require_equal(
            str(rubric.get("scenario_id") or ""),
            criterion_to_scenario[criterion_id],
            f"scenario assignment for {criterion_id}",
        )

    total_cells = len(model_ids) * len(criterion_ids)
    report = {
        "models": len(model_ids),
        "criteria": len(criterion_ids),
        "scenarios": len(scenario_ids),
        "total_cells": total_cells,
        "observed_cells": counts["0"] + counts["1"],
        "missing_cells": counts[""],
        "pass_cells": counts["1"],
        "fail_cells": counts["0"],
        "binary_or_missing_only": True,
        "bank_alignment_complete": True,
    }
    _require_equal(sum(counts.values()), total_cells, "response-matrix cell count")
    return report, set(source_ids), set(scenario_ids), model_ids


def _verify_judge_manifest(
    manifest_path: Path,
    matrix: dict[str, Any],
    model_ids: Sequence[str],
) -> dict[str, Any]:
    manifest = _read_json(manifest_path, "judge manifest")
    expected = manifest.get("judge_expected")
    observed = manifest.get("judge_observed")
    counts = manifest.get("counts")
    coverage = manifest.get("coverage")
    if not all(isinstance(value, dict) for value in (expected, observed, counts, coverage)):
        raise BaselineVerificationError("judge manifest is missing provenance/count objects")

    provenance_keys = (
        "judge_name",
        "judge_model",
        "judge_revision",
        "adapter",
        "prompt_version",
        "normalization_version",
        "evidence_policy_version",
        "prompt_variant",
        "replicate_id",
    )
    provenance: dict[str, str] = {}
    for key in provenance_keys:
        expected_value = str(expected.get(key) or "")
        observed_values = observed.get(key)
        if not isinstance(observed_values, list) or len(observed_values) != 1:
            raise BaselineVerificationError(
                f"judge manifest judge_observed.{key} must contain exactly one value"
            )
        _require_equal(str(observed_values[0]), expected_value, f"judge provenance {key}")
        provenance[key] = expected_value

    _require_equal(manifest.get("n_criteria"), matrix["criteria"], "judge criterion count")
    manifest_models = manifest.get("models")
    if not isinstance(manifest_models, list):
        raise BaselineVerificationError("judge manifest models must be a list")
    _require_equal(set(map(str, manifest_models)), set(model_ids), "judge/matrix models")
    _require_equal(counts.get("total_cells"), matrix["total_cells"], "judge total cells")
    _require_equal(coverage.get("n_holes"), matrix["missing_cells"], "judge holes")
    _require_equal(coverage.get("n_filled"), matrix["observed_cells"], "judge filled cells")
    this_run = counts.get("this_run")
    if not isinstance(this_run, dict):
        raise BaselineVerificationError("judge manifest counts.this_run must be an object")
    _require_equal(this_run.get("no_decision_policy"), "missing", "no-decision policy")

    return {
        **provenance,
        "configuration_hash": observed.get("configuration_hash", [None])[0],
        "frozen_configuration_hash": observed.get("frozen_configuration_hash", [None])[0],
        "total_cells": counts["total_cells"],
        "filled_cells": coverage["n_filled"],
        "no_decision_cells": coverage["n_holes"],
        "no_decision_policy": this_run["no_decision_policy"],
    }


def _verify_criterion_flow(
    repo_root: Path,
    config: dict[str, Any],
    source_ids: set[str],
    frozen_inputs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = config.get("stored_artifacts")
    if not isinstance(paths, dict):
        raise BaselineVerificationError("config stored_artifacts must be an object")
    export_manifest_path = _resolve(repo_root, str(paths["export_manifest"]))
    export_manifest = _read_json(export_manifest_path, "fitted-bank export manifest")
    counts = export_manifest.get("counts")
    reasons = export_manifest.get("exclusion_reason_counts")
    exclusions = export_manifest.get("excluded_criteria")
    if not isinstance(counts, dict) or not isinstance(reasons, dict) or not isinstance(
        exclusions, list
    ):
        raise BaselineVerificationError("export manifest is missing count/exclusion fields")

    source_count = len(source_ids)
    fitted_count = int(counts.get("calibration_rows", -1))
    exported_count = int(counts.get("exported_criteria", -1))
    _require_equal(counts.get("source_criteria"), source_count, "source criterion count")
    _require_equal(
        counts.get("excluded_criteria"), source_count - exported_count, "total exclusions"
    )

    excluded_by_reason: dict[str, set[str]] = {}
    for record in exclusions:
        if not isinstance(record, dict):
            raise BaselineVerificationError("excluded_criteria entries must be objects")
        criterion_id = str(record.get("criterion_id") or "")
        record_reasons = record.get("reasons")
        if criterion_id not in source_ids or not isinstance(record_reasons, list):
            raise BaselineVerificationError(
                f"invalid exclusion entry for criterion {criterion_id!r}"
            )
        if len(record_reasons) != 1:
            raise BaselineVerificationError(
                f"criterion {criterion_id} must have exactly one exclusion reason"
            )
        excluded_by_reason.setdefault(str(record_reasons[0]), set()).add(criterion_id)

    actual_reason_counts = {
        reason: len(ids) for reason, ids in sorted(excluded_by_reason.items())
    }
    _require_equal(actual_reason_counts, reasons, "exclusion reason counts")
    unfitted_ids = excluded_by_reason.get("unfitted", set())
    nonpositive_ids = excluded_by_reason.get("nonpositive_a", set())
    _require_equal(source_count - len(unfitted_ids), fitted_count, "fitted criterion flow")
    _require_equal(fitted_count - len(nonpositive_ids), exported_count, "export flow")
    if unfitted_ids & nonpositive_ids:
        raise BaselineVerificationError("unfitted and nonpositive exclusions overlap")

    calibration_csv_path = _resolve(repo_root, str(paths["selected_calibration_csv"]))
    calibration_header, calibration_rows = _read_csv(
        calibration_csv_path, "selected calibration CSV"
    )
    if "criterion_id" not in calibration_header:
        raise BaselineVerificationError("selected calibration CSV lacks criterion_id")
    discrimination_columns = [name for name in calibration_header if name.startswith("a_")]
    if not discrimination_columns:
        raise BaselineVerificationError("selected calibration CSV lacks discrimination columns")
    fitted_ids = [row["criterion_id"] for row in calibration_rows]
    _require_unique(fitted_ids, "fitted criterion IDs")
    _require_equal(set(fitted_ids), source_ids - unfitted_ids, "fitted criterion identities")
    observed_nonpositive = {
        row["criterion_id"]
        for row in calibration_rows
        if not any(
            _as_float(row[column], f"{row['criterion_id']} {column}") > 0.0
            for column in discrimination_columns
        )
    }
    _require_equal(observed_nonpositive, nonpositive_ids, "nonpositive criterion identities")

    manifest_inputs = export_manifest.get("inputs")
    if not isinstance(manifest_inputs, dict) or not isinstance(
        manifest_inputs.get("sha256"), dict
    ):
        raise BaselineVerificationError("export manifest inputs.sha256 is missing")
    input_hashes = manifest_inputs["sha256"]
    _require_equal(
        sha256_file(calibration_csv_path),
        input_hashes.get("calibration_csv"),
        "selected calibration CSV SHA-256",
    )
    _require_equal(
        frozen_inputs["rubrics"]["sha256"],
        input_hashes.get("rubrics"),
        "export/source-rubrics SHA-256",
    )

    exported_rubrics_path = _resolve(repo_root, str(paths["exported_rubrics"]))
    exported_hash = sha256_file(exported_rubrics_path)
    manifest_outputs = export_manifest.get("outputs")
    if not isinstance(manifest_outputs, dict) or not isinstance(
        manifest_outputs.get("sha256"), dict
    ):
        raise BaselineVerificationError("export manifest outputs.sha256 is missing")
    _require_equal(
        exported_hash,
        manifest_outputs["sha256"].get("rubrics"),
        "exported rubrics SHA-256",
    )
    exported_rows = _read_jsonl(exported_rubrics_path, "exported rubrics")
    exported_ids = [str(row.get("criterion_id") or "") for row in exported_rows]
    _require_unique(exported_ids, "exported criterion IDs")
    _require_equal(
        set(exported_ids), set(fitted_ids) - nonpositive_ids, "exported criterion identities"
    )

    flow = {
        "source_criteria": source_count,
        "fitted_criteria": fitted_count,
        "exported_criteria": exported_count,
        "total_excluded": source_count - exported_count,
        "exclusions": {
            "unfitted": len(unfitted_ids),
            "nonpositive_discrimination": len(nonpositive_ids),
        },
        "identities_reconciled": True,
        "arithmetic": {
            "source_minus_unfitted_equals_fitted": source_count - len(unfitted_ids),
            "fitted_minus_nonpositive_equals_exported": fitted_count
            - len(nonpositive_ids),
        },
    }
    rows = [
        {
            "stage": "source",
            "criteria": source_count,
            "removed_from_prior_stage": 0,
            "exclusion_reason": "",
        },
        {
            "stage": "fitted",
            "criteria": fitted_count,
            "removed_from_prior_stage": len(unfitted_ids),
            "exclusion_reason": "unfitted (all-fail/zero-variance)",
        },
        {
            "stage": "exported",
            "criteria": exported_count,
            "removed_from_prior_stage": len(nonpositive_ids),
            "exclusion_reason": "nonpositive discrimination",
        },
    ]
    return flow, rows


def _verify_phase0_bundle(
    repo_root: Path,
    paths: dict[str, Any],
    frozen_inputs: dict[str, dict[str, Any]],
    matrix: dict[str, Any],
    judge: dict[str, Any],
    criterion_flow: dict[str, Any],
    model_ids: Sequence[str],
) -> dict[str, Any] | None:
    """Cross-check the immutable Phase-0 archive when the config declares it."""
    report_value = paths.get("phase0_input_artifact")
    archive_value = paths.get("phase0_input_archive")
    if report_value is None and archive_value is None:
        return None
    if not report_value or not archive_value:
        raise BaselineVerificationError(
            "phase0_input_artifact and phase0_input_archive must be declared together"
        )
    report_path = _resolve(repo_root, str(report_value))
    archive_path = _resolve(repo_root, str(archive_value))
    artifact = _read_json(report_path, "Phase-0 input artifact manifest")
    _require_equal(
        artifact.get("schema_version"),
        "infobench-phase0-input-bundle-v1",
        "Phase-0 bundle schema",
    )
    archive = artifact.get("archive")
    payload = artifact.get("payload")
    cohort = artifact.get("cohort")
    artifact_judge = artifact.get("judge_provenance")
    artifact_flow = artifact.get("criterion_count_flow")
    storage = artifact.get("storage")
    if not all(
        isinstance(value, dict)
        for value in (archive, payload, cohort, artifact_judge, artifact_flow, storage)
    ):
        raise BaselineVerificationError("Phase-0 artifact manifest is missing required objects")
    if not archive_path.is_file():
        raise BaselineVerificationError(f"missing Phase-0 input archive: {archive_path}")
    observed_archive_hash = sha256_file(archive_path)
    _require_equal(observed_archive_hash, archive.get("sha256"), "Phase-0 archive SHA-256")
    _require_equal(archive_path.stat().st_size, archive.get("bytes"), "Phase-0 archive bytes")

    required_payload = {
        "inputs/response_matrix.csv": frozen_inputs["response_matrix"]["sha256"],
        "inputs/judge_manifest.json": frozen_inputs["judge_manifest"]["sha256"],
    }
    for member_name, expected_hash in required_payload.items():
        member = payload.get(member_name)
        if not isinstance(member, dict):
            raise BaselineVerificationError(f"Phase-0 payload lacks {member_name}")
        _require_equal(member.get("sha256"), expected_hash, f"Phase-0 {member_name} hash")
    if "inputs/normalized_verdicts.jsonl" not in payload:
        raise BaselineVerificationError("Phase-0 payload lacks normalized verdicts")
    response_payload_names = sorted(
        name
        for name in payload
        if name.startswith("inputs/responses/") and name.endswith(".responses.jsonl")
    )
    _require_equal(len(response_payload_names), matrix["models"], "Phase-0 response shard count")

    try:
        with tarfile.open(archive_path, "r:gz") as handle:
            members = handle.getmembers()
            names = [member.name for member in members]
            if any(
                name.startswith("/") or ".." in Path(name).parts or not member.isfile()
                for name, member in zip(names, members)
            ):
                raise BaselineVerificationError(
                    "Phase-0 archive contains an unsafe path or non-file member"
                )
            _require_unique(names, "Phase-0 archive member names")
            _require_equal(len(names), archive.get("members"), "Phase-0 archive member count")
            roots = {Path(name).parts[0] for name in names if Path(name).parts}
            _require_equal(len(roots), 1, "Phase-0 archive root count")
            archive_root = next(iter(roots))
            logical_members = {
                "/".join(Path(member.name).parts[1:]): member for member in members
            }
            _require_unique(list(logical_members), "Phase-0 logical member names")
            if not set(payload).issubset(logical_members):
                missing = sorted(set(payload) - set(logical_members))
                raise BaselineVerificationError(
                    f"Phase-0 archive lacks payload members: {', '.join(missing[:5])}"
                )
            # Verify the complete declared payload, not only its outer archive hash.
            for name, record in payload.items():
                if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
                    raise BaselineVerificationError(f"invalid Phase-0 payload record: {name}")
                extracted = handle.extractfile(logical_members[name])
                if extracted is None:
                    raise BaselineVerificationError(f"cannot read Phase-0 payload member: {name}")
                digest = hashlib.sha256()
                for block in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(block)
                _require_equal(digest.hexdigest(), record["sha256"], f"Phase-0 member {name}")
    except (OSError, tarfile.TarError) as exc:
        raise BaselineVerificationError(f"invalid Phase-0 archive {archive_path}: {exc}") from exc

    _require_equal(cohort.get("models"), matrix["models"], "Phase-0 cohort models")
    _require_equal(cohort.get("criteria"), matrix["criteria"], "Phase-0 cohort criteria")
    _require_equal(cohort.get("matrix_cells"), matrix["total_cells"], "Phase-0 matrix cells")
    _require_equal(
        cohort.get("matrix_value_counts"),
        {
            "0": matrix["fail_cells"],
            "1": matrix["pass_cells"],
            "missing": matrix["missing_cells"],
        },
        "Phase-0 matrix value counts",
    )
    response_shards = cohort.get("response_shards")
    if not isinstance(response_shards, list):
        raise BaselineVerificationError("Phase-0 response_shards must be a list")
    shard_models = [str(record.get("model") or "") for record in response_shards]
    _require_unique(shard_models, "Phase-0 response-shard models")
    _require_equal(set(shard_models), set(model_ids), "Phase-0 response-shard cohort")
    if any(int(record.get("rows", -1)) != 500 for record in response_shards):
        raise BaselineVerificationError("each Phase-0 response shard must contain 500 scenarios")

    for key in (
        "judge_name",
        "judge_model",
        "judge_revision",
        "adapter",
        "prompt_version",
        "normalization_version",
        "evidence_policy_version",
        "prompt_variant",
        "replicate_id",
    ):
        _require_equal(artifact_judge.get(key), judge[key], f"Phase-0 judge {key}")
    _require_equal(
        artifact_flow.get("source_criteria"),
        criterion_flow["source_criteria"],
        "Phase-0 source criteria",
    )
    _require_equal(
        artifact_flow.get("fitted_criteria"),
        criterion_flow["fitted_criteria"],
        "Phase-0 fitted criteria",
    )
    _require_equal(
        artifact_flow.get("exported_criteria"),
        criterion_flow["exported_criteria"],
        "Phase-0 exported criteria",
    )
    source_to_fit = artifact_flow.get("source_to_fit")
    fit_to_export = artifact_flow.get("fit_to_export")
    if not isinstance(source_to_fit, dict) or not isinstance(fit_to_export, dict):
        raise BaselineVerificationError("Phase-0 criterion flow lacks exclusion blocks")
    _require_equal(
        source_to_fit.get("excluded_all_fail"),
        criterion_flow["exclusions"]["unfitted"],
        "Phase-0 unfitted exclusions",
    )
    _require_equal(
        fit_to_export.get("excluded_nonpositive_discrimination"),
        criterion_flow["exclusions"]["nonpositive_discrimination"],
        "Phase-0 nonpositive exclusions",
    )

    return {
        "schema_version": artifact["schema_version"],
        "manifest_path": str(report_value),
        "archive_path": str(archive_value),
        "archive_sha256": observed_archive_hash,
        "archive_bytes": archive_path.stat().st_size,
        "archive_members": len(members),
        "archive_root": archive_root,
        "payload_members_verified": len(payload),
        "response_shards": len(response_shards),
        "normalized_verdicts_present": True,
        "storage_status": storage.get("status"),
        "storage_uri": storage.get("uri"),
    }


def _reproduce_structure(repo_root: Path, paths: dict[str, Any]) -> dict[str, Any]:
    _, rows = _read_csv(
        _resolve(repo_root, str(paths["structure_comparison"])), "structure comparison"
    )
    selection = _read_json(
        _resolve(repo_root, str(paths["structure_selection"])), "structure selection"
    )
    eligible_fit = [
        row
        for row in rows
        if _as_bool(row["core_converged"], "core_converged")
        and _as_bool(row["all_fold_fits_converged"], "all_fold_fits_converged")
    ]
    if not eligible_fit:
        raise BaselineVerificationError("no converged structures are available")
    if {row["primary_metric"] for row in eligible_fit} != {"log_loss"}:
        raise BaselineVerificationError("frozen structure reproduction expects log_loss")
    best = min(eligible_fit, key=lambda row: _as_float(row["cv_mean"], "cv_mean"))
    best_mean = _as_float(best["cv_mean"], "best cv_mean")
    best_se = _as_float(best["cv_se"], "best cv_se")
    boundary = best_mean + best_se
    one_se = [
        row
        for row in eligible_fit
        if _as_float(row["cv_mean"], "cv_mean") <= boundary + 1e-15
    ]
    selected = min(
        one_se,
        key=lambda row: (
            int(row["n_dims"]),
            _as_float(row["cv_mean"], "cv_mean"),
            row["structure"],
        ),
    )
    _require_equal(
        selection.get("selected_structure"), selected["structure"], "selected structure"
    )
    _require_close(
        float(selection.get("one_standard_error_threshold")), boundary, "one-SE boundary"
    )
    reported_selected = [
        row for row in rows if _as_bool(row["selected"], f"{row['structure']} selected")
    ]
    _require_equal(len(reported_selected), 1, "number of selected structures")
    _require_equal(reported_selected[0]["structure"], selected["structure"], "selected row")
    return {
        "selected_structure": selected["structure"],
        "selected_dimensions": int(selected["n_dims"]),
        "best_mean_structure": best["structure"],
        "best_cv_mean_log_loss": best_mean,
        "best_cv_standard_error": best_se,
        "one_standard_error_boundary": boundary,
        "eligible_structures": sorted(row["structure"] for row in one_se),
    }


def _cat_rows(
    repo_root: Path, relative_path: str, label: str
) -> tuple[list[dict[str, str]], list[str]]:
    _, rows = _read_csv(_resolve(repo_root, relative_path), label)
    model_ids = [row["model"] for row in rows]
    _require_unique(model_ids, f"{label} model IDs")
    return rows, model_ids


def _reproduce_cat(repo_root: Path, paths: dict[str, Any]) -> dict[str, Any]:
    selected_rows, selected_models = _cat_rows(
        repo_root, str(paths["selected_cat_per_model"]), "selected CAT per-model results"
    )
    random_rows, random_models = _cat_rows(
        repo_root, str(paths["random_per_model"]), "random per-model results"
    )
    _require_equal(set(selected_models), set(random_models), "CAT/random model IDs")
    random_by_model = {row["model"]: row for row in random_rows}

    def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
        reference = [
            _as_float(
                row["theta_full_eap_instruction_following"],
                f"{row['model']} full-EAP theta",
            )
            for row in rows
        ]
        estimate = [
            _as_float(row["theta_mwle_instruction_following"], f"{row['model']} MWLE")
            for row in rows
        ]
        return {
            "models": len(rows),
            "mean_scenarios": _mean(
                [_as_float(row["scenarios_administered"], "scenarios") for row in rows]
            ),
            "mean_criteria": _mean(
                [_as_float(row["criteria_administered"], "criteria") for row in rows]
            ),
            "precision_reached_rate": _mean(
                [float(_as_bool(row["precision_reached"], "precision_reached")) for row in rows]
            ),
            "mwle_convergence_rate": _mean(
                [float(_as_bool(row["mwle_converged"], "mwle_converged")) for row in rows]
            ),
            "mwle_recovery": _recovery(reference, estimate),
        }

    selected_summary = summarize(selected_rows)
    random_summary = summarize(random_rows)
    paired_scenario_difference = _mean(
        [
            _as_float(row["scenarios_administered"], "CAT scenarios")
            - _as_float(
                random_by_model[row["model"]]["scenarios_administered"], "random scenarios"
            )
            for row in selected_rows
        ]
    )

    configuration = _read_json(
        _resolve(repo_root, str(paths["cat_configuration"])), "CAT configuration selection"
    )
    validation = _read_json(
        _resolve(repo_root, str(paths["cat_validation"])), "CAT configuration validation"
    )
    validation_metrics = validation.get("metrics")
    if not isinstance(validation_metrics, dict):
        raise BaselineVerificationError("CAT validation metrics are missing")
    _require_close(
        selected_summary["mean_scenarios"],
        float(validation_metrics["mean_scenarios"]),
        "validated CAT mean scenarios",
    )
    _require_close(
        selected_summary["mwle_recovery"]["r"],
        float(validation_metrics["mwle_recovery_r"]),
        "validated CAT MWLE recovery r",
    )
    _require_close(
        selected_summary["mwle_recovery"]["slope"],
        float(validation_metrics["mwle_recovery_slope"]),
        "validated CAT MWLE recovery slope",
    )
    _require_close(
        selected_summary["precision_reached_rate"],
        float(validation_metrics["precision_rate"]),
        "validated CAT precision rate",
    )

    return {
        "configuration": {
            "minimum_scenarios": int(configuration["selected_min_scenarios"]),
            "nominal_se_target": float(configuration["selected_se_target"]),
            "selection_rule": str(configuration["selected_selection_rule"]),
        },
        "selected_cat": selected_summary,
        "random_baseline": random_summary,
        "paired_mean_scenario_difference": paired_scenario_difference,
    }


def _reproduce_oos(repo_root: Path, paths: dict[str, Any]) -> dict[str, Any]:
    _, rows = _read_csv(_resolve(repo_root, str(paths["oos_per_model"])), "OOS per-model")
    _, metric_rows = _read_csv(
        _resolve(repo_root, str(paths["oos_metrics"])), "OOS estimator metrics"
    )
    result: dict[str, Any] = {}
    for estimator in ("eap", "mwle", "online"):
        usable = [
            row
            for row in rows
            if row["status"] == "ok"
            and (
                estimator != "mwle"
                or _as_bool(row["mwle_converged"], f"{row['model']} mwle_converged")
            )
        ]
        model_ids = [row["model"] for row in usable]
        _require_unique(model_ids, f"OOS {estimator} model IDs")
        reference = [
            _as_float(row["theta_ref_instruction_following"], f"{row['model']} reference")
            for row in usable
        ]
        estimates = [
            _as_float(
                row[f"theta_{estimator}_instruction_following"],
                f"{row['model']} {estimator}",
            )
            for row in usable
        ]
        recovery = _recovery(reference, estimates)
        reported = {
            row["metric"]: row
            for row in metric_rows
            if row["estimator"] == estimator
            and row["skill"] == "instruction_following"
        }
        _require_equal(set(reported), {"r", "slope", "mae"}, f"reported {estimator} metrics")
        for metric, value in recovery.items():
            _require_close(
                value,
                _as_float(reported[metric]["estimate"], f"reported {estimator} {metric}"),
                f"OOS {estimator} {metric}",
            )
            _require_equal(
                int(reported[metric]["n_models"]), len(usable), f"OOS {estimator} n_models"
            )
        result[estimator] = {"models": len(usable), **recovery}
    return result


def _reproduce_uncertainty(repo_root: Path, paths: dict[str, Any]) -> dict[str, Any]:
    _, rows = _read_csv(
        _resolve(repo_root, str(paths["uncertainty_components"])),
        "parameter-uncertainty components",
    )
    _, summary_rows = _read_csv(
        _resolve(repo_root, str(paths["uncertainty_summary"])),
        "parameter-uncertainty summary",
    )
    target = 0.25
    usable = [
        row
        for row in rows
        if row["estimator"] == "mwle"
        and row["dimension"] == "instruction_following"
        and _as_bool(row["reliable"], f"{row['model']} reliable")
        and math.isclose(_as_float(row["se_target"], "se_target"), target)
    ]
    _require_unique([row["model"] for row in usable], "uncertainty model IDs")
    values = sorted(_as_float(row["se_total"], f"{row['model']} se_total") for row in usable)
    result = {
        "nominal_se_target": target,
        "reliable_models": len(values),
        "median_total_se": statistics.median(values),
        "p90_total_se": _linear_quantile(values, 0.90),
        "models_at_or_below_nominal_target": sum(value <= target for value in values),
    }
    matching_summary = [
        row
        for row in summary_rows
        if row["estimator"] == "mwle"
        and row["dimension"] == "instruction_following"
        and math.isclose(_as_float(row["se_target"], "summary se_target"), target)
    ]
    _require_equal(len(matching_summary), 1, "uncertainty summary row count")
    summary = matching_summary[0]
    _require_equal(int(summary["n_reliable_models"]), len(values), "reliable model count")
    _require_close(
        result["median_total_se"],
        _as_float(summary["median_se_total"], "summary median_se_total"),
        "median total SE",
    )
    return result


def reproduce_baseline(repo_root: Path, config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reproduce and validate the frozen baseline, returning report and flow rows."""
    config = _read_json(config_path, "baseline reproduction config")
    _require_equal(config.get("schema_version"), SCHEMA_VERSION, "config schema_version")
    frozen_inputs, frozen_paths = _verify_frozen_artifacts(repo_root, config)
    required_frozen = {"response_matrix", "judge_manifest", "rubrics", "scenarios"}
    if set(frozen_paths) != required_frozen:
        raise BaselineVerificationError(
            f"frozen_inputs must contain exactly {sorted(required_frozen)}"
        )

    matrix, source_ids, _, model_ids = _verify_matrix(
        frozen_paths["response_matrix"],
        frozen_paths["rubrics"],
        frozen_paths["scenarios"],
    )
    judge = _verify_judge_manifest(frozen_paths["judge_manifest"], matrix, model_ids)
    criterion_flow, flow_rows = _verify_criterion_flow(
        repo_root, config, source_ids, frozen_inputs
    )
    stored = config.get("stored_artifacts")
    if not isinstance(stored, dict):
        raise BaselineVerificationError("config stored_artifacts must be an object")
    phase0_bundle = _verify_phase0_bundle(
        repo_root,
        stored,
        frozen_inputs,
        matrix,
        judge,
        criterion_flow,
        model_ids,
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "method": "read-only reproduction from stored artifacts; no refitting",
        "frozen_inputs": frozen_inputs,
        "matrix": matrix,
        "judge_provenance": judge,
        "criterion_flow": criterion_flow,
        "headline_metrics": {
            "structure_selection": _reproduce_structure(repo_root, stored),
            "cat_vs_random": _reproduce_cat(repo_root, stored),
            "oos_estimator_recovery": _reproduce_oos(repo_root, stored),
            "mwle_uncertainty": _reproduce_uncertainty(repo_root, stored),
        },
    }
    if phase0_bundle is not None:
        report["phase0_input_bundle"] = phase0_bundle
    expected = config.get("expected")
    if not isinstance(expected, dict):
        raise BaselineVerificationError("config expected must be an object")
    _assert_expected_subset(report, expected)
    return report, flow_rows


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _flow_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    fields = ("stage", "criteria", "removed_from_prior_stage", "exclusion_reason")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def output_paths(repo_root: Path, config_path: Path) -> tuple[Path, Path]:
    config = _read_json(config_path, "baseline reproduction config")
    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise BaselineVerificationError("config outputs must be an object")
    return (
        _resolve(repo_root, str(outputs["report"])),
        _resolve(repo_root, str(outputs["criterion_flow_csv"])),
    )


def write_outputs(
    report_path: Path,
    flow_path: Path,
    report: dict[str, Any],
    flow_rows: list[dict[str, Any]],
) -> None:
    _atomic_write(report_path, _json_bytes(report))
    _atomic_write(flow_path, _flow_bytes(flow_rows))


def check_outputs(
    report_path: Path,
    flow_path: Path,
    report: dict[str, Any],
    flow_rows: list[dict[str, Any]],
) -> None:
    expected_outputs = {
        report_path: _json_bytes(report),
        flow_path: _flow_bytes(flow_rows),
    }
    for path, expected in expected_outputs.items():
        if not path.is_file():
            raise BaselineVerificationError(f"missing stored reproduction output: {path}")
        observed = path.read_bytes()
        if observed != expected:
            raise BaselineVerificationError(
                f"stored reproduction output is stale or modified: {path}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=f"baseline specification (default: {DEFAULT_CONFIG})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write deterministic report files")
    mode.add_argument("--check", action="store_true", help="verify stored report files bytewise")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config or repo_root / DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    try:
        report, flow_rows = reproduce_baseline(repo_root, config_path)
        report_path, flow_path = output_paths(repo_root, config_path)
        if args.write:
            write_outputs(report_path, flow_path, report, flow_rows)
            action = "wrote"
        elif args.check:
            check_outputs(report_path, flow_path, report, flow_rows)
            action = "checked"
        else:
            action = "verified"
        print(
            f"InFoBench baseline {action}: "
            f"{report['criterion_flow']['source_criteria']} -> "
            f"{report['criterion_flow']['fitted_criteria']} -> "
            f"{report['criterion_flow']['exported_criteria']} criteria; "
            f"inputs and stored metrics match."
        )
        return 0
    except BaselineVerificationError as exc:
        print(f"baseline verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
