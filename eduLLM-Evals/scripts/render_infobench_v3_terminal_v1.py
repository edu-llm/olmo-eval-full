#!/usr/bin/env python3
# ruff: noqa: E501
"""Render a terminal, provenance-linked report for the InFoBench V3 study.

The renderer has two deliberately separate input paths:

* a terminally blocked V4 numerical study produces a numerical-only report and
  never opens, lists, or hashes a Phase-3/CAT artifact; and
* a passing V4 numerical lock requires a complete V3 Phase-3 artifact set and
  produces numerical plus CAT-validation diagnostics.

The historical August 4/5 images are layout references only.  Every plotted
value comes from the supplied V4/V3 artifacts.  Phase 4 and final-fit outputs
are outside this renderer's current evidence contract, so a Phase-3 pass is
reported as ``phase3_pass_phase4_pending`` rather than as a final success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "infobench_calibration_cat_v3.json"
DEFAULT_V4_DIR = ROOT / "runs" / "calibration" / "InFoBench_v2_numerical_followup_v4"
DEFAULT_PHASE3_DIR = ROOT / "runs" / "calibration" / "InFoBench_v3" / "phase3"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "infobench_calibration_cat_v3"

REPORT_SCHEMA = "infobench-v3-terminal-report-v1"
FIGURE_MANIFEST_SCHEMA = "infobench-v3-terminal-figure-manifest-v1"
V4_RUN_SCHEMA = "infobench-v2-numerical-followup-run-v4"
V4_LOCK_SCHEMA = "infobench-v2-numerical-followup-lock-v4"
V4_CONFIG_SCHEMA = "infobench-v2-numerical-followup-v4"
V4_CHECKPOINT_SCHEMA = "infobench-v2-numerical-followup-checkpoint-v4"
V3_CONFIG_SCHEMA = "infobench-calibration-cat-v3-v1"
V3_RUN_SCHEMA = "infobench-nested-scenario-cat-cv-v3-v1"
V3_DECISION_SCHEMA = "infobench-v3-phase3-decision-v1"
V3_SELECTED_SPECS_SCHEMA = "infobench-v3-selected-calibration-specs-v1"
V3_STRUCTURE_NAME = "infobench_overall_1d_v3"

V3_CODE_DEPENDENCIES = (
    "scripts/nested_scenario_cat_cv_v3.py",
    "scripts/nested_scenario_cat_cv.py",
    "scripts/calibrate_mirt.py",
    "scripts/kfold_cv_mirt.py",
    "scripts/scenario_cat_lib.py",
    "scripts/scenario_kfold_estimator_cv.py",
    "tutor_cat/dataio.py",
    "tutor_cat/engine.py",
    "tutor_cat/mirt.py",
    "tutor_cat/schemas.py",
    "tutor_cat/selector.py",
)

SPEC_IDS = (
    "1pl_fixed_a1",
    "log_shrinkage_2pl_lambda16",
    "log_shrinkage_2pl_lambda4",
)
SPEC_LABELS = {
    "1pl_fixed_a1": "1PL (a=1)",
    "log_shrinkage_2pl_lambda16": "shrinkage 2PL (lambda=16)",
    "log_shrinkage_2pl_lambda4": "shrinkage 2PL (lambda=4)",
}
POLICY_LABELS = {
    "primary_floor15_se0p20_trace": "primary: floor 15 / SE .20 / trace",
    "sensitivity_floor12_se0p20_trace": "sensitivity: floor 12",
    "sensitivity_floor15_se0p20_dopt": "sensitivity: D-opt",
}
PRIMARY_POLICY = "primary_floor15_se0p20_trace"
SCOPE_IDS = ("full", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4")
FIT_VARIANTS = ((401, "cold"), (801, "cold"), (801, "continuation"))

EXPECTED_FITS = 54
EXPECTED_STARTS = 18
EXPECTED_INNER_ROWS = 75
EXPECTED_OUTER_ROWS = 780
EXPECTED_GATE_ROWS = 90
EXPECTED_PREDICTION_ROWS = 150
EXPECTED_CROSS_MODEL_ROWS = 156
EXPECTED_PRIMARY_MODEL_REPEAT_ROWS = 260
EXPECTED_PRIMARY_PANELS_PER_ARM = 25

ABSOLUTE_GATE_NAMES = (
    "replay_success_rate",
    "mwle_convergence_rate",
    "nominal_precision_rate",
    "nominal_precision_lower_95_ci",
    "recovery_correlation_lower_95_ci",
    "recovery_slope",
    "disjoint_pass_rate_mae",
    "absolute_disjoint_pass_rate_bias",
    "scenario_reduction_vs_random",
    "paired_ci_favors_cat",
)
PANEL_APPLIED_GATE_NAMES = (
    "replay_success_rate",
    "mwle_convergence_rate",
    "nominal_precision_rate",
    "recovery_slope",
    "disjoint_pass_rate_mae",
    "absolute_disjoint_pass_rate_bias",
    "scenario_reduction_vs_random",
)
PANEL_DIAGNOSTIC_GATE_NAMES = tuple(
    name for name in ABSOLUTE_GATE_NAMES if name not in PANEL_APPLIED_GATE_NAMES
)
REPETITION_APPLIED_GATE_NAMES = ABSOLUTE_GATE_NAMES
INNER_COMBINATION_CHECK_NAMES = (
    "rows_present",
    "validation_model_coverage_exact",
    "status_ok",
    "fit_eligible",
    "all_specs_fitted",
    "common_support_verified",
    "fit_cache_key_present",
    "common_support_hash_present",
    "positive_administration_support",
    "positive_evaluation_support",
    "positive_observed_cells",
    "finite_model_log_loss",
    "observed_cell_hash_present",
)

PHASE3_SELECTION_OUTPUTS = {
    "fold_assignments.json",
    "pre_outer_selection_lock.json",
    "inner_calibration_model_results.csv",
    "calibration_model_choices.json",
    "selected_calibration_specs.json",
    "phase3_decision.json",
    "v3_decision.json",
}
PHASE3_OUTCOME_OUTPUTS = {
    "outer_oof_per_model.csv",
    "disjoint_prediction_metrics.csv",
    "repeat_fold_gate_results.csv",
    "repeat_metrics.csv",
    "calibration_spec_selection_frequency.csv",
    "duplicate_cat_paths.csv",
    "cross_repeat_per_model.csv",
    "cross_repeat_metrics.csv",
}

COLORS = {
    "blue": "#2F80B7",
    "orange": "#D95F02",
    "green": "#2A9D6F",
    "red": "#C84630",
    "purple": "#7251A6",
    "gray": "#72777D",
    "light_gray": "#D9DDE1",
    "dark": "#28323C",
}


class TerminalReportError(RuntimeError):
    """A terminal artifact, schema, provenance link, or plotting input is invalid."""


@dataclass
class SourceTracker:
    paths: set[Path] = field(default_factory=set)

    def add(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise TerminalReportError(f"required source artifact is missing: {resolved}")
        self.paths.add(resolved)
        return resolved


@dataclass
class TerminalInputs:
    terminal_state: str
    config_path: Path
    v4_dir: Path
    phase3_dir: Path | None
    config: dict[str, Any]
    v4_decision: dict[str, Any]
    v4_manifest: dict[str, Any]
    v4_lock: dict[str, Any] | None
    phase3_decision: dict[str, Any] | None
    phase3_manifest: dict[str, Any] | None
    numerical: dict[str, Any]
    phase3: dict[str, Any] | None
    tracker: SourceTracker


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _resolve_recorded(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_json(path: Path, tracker: SourceTracker) -> dict[str, Any]:
    source = tracker.add(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TerminalReportError(f"could not read JSON {source}: {error}") from error
    if not isinstance(value, dict):
        raise TerminalReportError(f"JSON artifact must contain an object: {source}")
    return value


def _read_csv(
    path: Path,
    tracker: SourceTracker,
    *,
    required: Sequence[str] = (),
    expected_rows: int | None = None,
) -> pd.DataFrame:
    source = tracker.add(path)
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise TerminalReportError(f"could not read CSV {source}: {error}") from error
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise TerminalReportError(f"{source} is missing columns: {', '.join(missing)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise TerminalReportError(f"{source} has {len(frame)} rows; expected {expected_rows}")
    return frame


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalReportError(message)


def _strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TerminalReportError(f"{label} is not boolean")


def _validate_recorded_file(
    record: Mapping[str, Any], expected_path: Path, tracker: SourceTracker, *, label: str
) -> Path:
    _require(set(record) >= {"path", "sha256"}, f"{label} provenance is incomplete")
    recorded = _resolve_recorded(str(record["path"]))
    expected = expected_path.resolve()
    _require(recorded == expected, f"{label} path is not canonical")
    source = tracker.add(expected)
    _require(_sha256(source) == str(record["sha256"]), f"{label} hash mismatch")
    return source


def _validate_v4_study_signature(
    v4_dir: Path,
    decision: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tracker: SourceTracker,
) -> tuple[Path, dict[str, Any]]:
    signature = manifest.get("study_signature")
    _require(isinstance(signature, Mapping), "V4 manifest study signature is missing")
    payload_keys = {
        "schema_version",
        "config_sha256",
        "frozen_contract_sha256",
        "input_sha256",
        "code_sha256",
        "historical_evidence_sha256",
        "environment_sha256",
        "spec_ids",
        "scope_ids",
        "fit_nodes",
        "required_total_new_fits",
        "historical_fit_or_checkpoint_artifacts_reused",
        "adaptive_runs",
    }
    _require(
        set(signature) == payload_keys | {"environment", "canonical_sha256"},
        "V4 study-signature fields changed",
    )
    payload = {key: signature[key] for key in payload_keys}
    canonical = _canonical_hash(payload)
    _require(
        signature.get("canonical_sha256")
        == manifest.get("study_signature_sha256")
        == decision.get("study_signature_sha256")
        == canonical,
        "V4 study-signature hash chain is invalid",
    )
    environment = signature.get("environment")
    _require(isinstance(environment, Mapping), "V4 signature environment is missing")
    environment_payload = {
        key: value for key, value in environment.items() if key != "canonical_sha256"
    }
    _require(
        environment.get("canonical_sha256")
        == signature.get("environment_sha256")
        == _canonical_hash(environment_payload),
        "V4 environment signature is invalid",
    )
    _require(list(signature.get("spec_ids") or []) == list(SPEC_IDS), "V4 signature specs changed")
    _require(
        list(signature.get("scope_ids") or []) == list(SCOPE_IDS), "V4 signature scopes changed"
    )
    _require(signature.get("fit_nodes") == [401, 801], "V4 signature fit nodes changed")
    _require(signature.get("required_total_new_fits") == 54, "V4 signature fit count changed")

    config_path = _resolve_recorded(str(manifest.get("config") or ""))
    config_source = tracker.add(config_path)
    config_hash = _sha256(config_source)
    _require(
        config_hash
        == manifest.get("config_sha256")
        == signature.get("config_sha256")
        == decision.get("config_sha256"),
        "V4 config hash chain is invalid",
    )
    config = _read_json(config_source, tracker)
    _require(config.get("schema_version") == V4_CONFIG_SCHEMA, "unexpected V4 config schema")
    _require(
        _resolve_recorded(str(config.get("output_dir") or "")) == v4_dir.resolve(),
        "V4 config output path differs from the rendered run",
    )
    _require(
        _canonical_hash(config.get("frozen_contract") or {})
        == signature.get("frozen_contract_sha256")
        == decision.get("frozen_contract_sha256"),
        "V4 frozen-contract hash chain is invalid",
    )
    for field_name in ("input_sha256", "code_sha256"):
        recorded = signature.get(field_name)
        _require(
            isinstance(recorded, Mapping) and bool(recorded),
            f"V4 {field_name} is missing",
        )
        for path_text, expected_hash in recorded.items():
            path = tracker.add(_resolve_recorded(str(path_text)))
            _require(
                _sha256(path) == str(expected_hash),
                f"V4 {field_name} hash mismatch: {path_text}",
            )
    _require(
        decision.get("code_sha256_reverified_before_finalization") == signature.get("code_sha256"),
        "V4 final code rehash differs from the study signature",
    )
    return config_source, dict(signature)


def _v4_stage_roster(
    v4_dir: Path, *, include_fixed: bool
) -> tuple[dict[str, Path], dict[str, tuple[str, list[Path]]]]:
    evidence: dict[str, Path] = {}
    stages: dict[str, tuple[str, list[Path]]] = {}
    for spec_id in SPEC_IDS:
        for scope in SCOPE_IDS:
            for nodes, start in FIT_VARIANTS:
                directory = v4_dir / "fits" / spec_id / f"nodes_{nodes:04d}" / scope / start
                stage_id = f"fit/{spec_id}/nodes_{nodes:04d}/{scope}/{start}"
                outputs = [
                    directory / "fit.npz",
                    directory / "item_params.csv",
                    directory / "convergence_trace.json",
                    directory / "fit_manifest.json",
                ]
                evidence[f"fit/{spec_id}/{scope}/{nodes}/{start}"] = outputs[-1]
                stages[stage_id] = ("fit", outputs)
            selection = v4_dir / "start_selection" / spec_id / f"{scope}.json"
            stage_id = f"start_selection/{spec_id}/{scope}"
            evidence[f"start/{spec_id}/{scope}"] = selection
            stages[stage_id] = ("start", [selection])
        comparison = v4_dir / "fit_comparisons" / "401_vs_801" / spec_id
        stage_id = f"fit_comparison/{spec_id}/401_vs_801"
        comparison_outputs = [
            comparison / "parameter_scope_gates.csv",
            comparison / "common_support.csv",
            comparison / "theta_common_support.csv",
            comparison / "heldout_cells.csv",
            comparison / "fit_pair_gate.json",
        ]
        evidence[f"refit/{spec_id}"] = comparison_outputs[-1]
        stages[stage_id] = ("refit", comparison_outputs)
        if include_fixed:
            fixed = v4_dir / "fixed_bank" / spec_id
            stage_id = f"fixed_bank/{spec_id}"
            fixed_outputs = [
                fixed / "theta_profiles.csv",
                fixed / "support.csv",
                fixed / "fixed_bank_gate.json",
            ]
            evidence[f"fixed_bank/{spec_id}"] = fixed_outputs[-1]
            stages[stage_id] = ("fixed", fixed_outputs)
    return evidence, stages


def _validate_v4_checkpoint(
    v4_dir: Path,
    stage_id: str,
    stage_kind: str,
    outputs: Sequence[Path],
    signature_sha256: str,
    tracker: SourceTracker,
) -> dict[str, Any]:
    checkpoint_path = v4_dir / "checkpoints" / f"{stage_id.replace('/', '__')}.json"
    checkpoint = _read_json(checkpoint_path, tracker)
    _require(
        checkpoint.get("schema_version") == V4_CHECKPOINT_SCHEMA,
        f"invalid checkpoint schema: {stage_id}",
    )
    _require(checkpoint.get("status") == "completed", f"checkpoint is not complete: {stage_id}")
    _require(checkpoint.get("stage_id") == stage_id, f"checkpoint stage differs: {stage_id}")
    _require(
        checkpoint.get("study_signature_sha256") == signature_sha256,
        f"checkpoint signature differs: {stage_id}",
    )
    stage_inputs = checkpoint.get("stage_inputs")
    _require(isinstance(stage_inputs, Mapping), f"checkpoint inputs missing: {stage_id}")
    _require(
        checkpoint.get("stage_inputs_sha256") == _canonical_hash(stage_inputs),
        f"checkpoint input hash mismatch: {stage_id}",
    )
    recorded_outputs = checkpoint.get("output_sha256")
    _require(isinstance(recorded_outputs, Mapping), f"checkpoint outputs missing: {stage_id}")
    resolved_outputs = {
        _resolve_recorded(str(path)): str(digest) for path, digest in recorded_outputs.items()
    }
    expected_outputs = {path.resolve() for path in outputs}
    _require(
        set(resolved_outputs) == expected_outputs, f"checkpoint output roster differs: {stage_id}"
    )
    for path in outputs:
        source = tracker.add(path)
        _require(
            _sha256(source) == resolved_outputs[source],
            f"checkpoint output hash mismatch: {stage_id}/{source.name}",
        )
    if stage_kind == "fit":
        fit_manifest = _read_json(outputs[-1], tracker)
        _require(
            fit_manifest.get("stage_inputs") == stage_inputs
            and fit_manifest.get("stage_inputs_sha256") == checkpoint.get("stage_inputs_sha256"),
            f"fit manifest/checkpoint inputs differ: {stage_id}",
        )
        _require(
            _sha256(outputs[0]) == fit_manifest.get("fit_sha256"), f"fit hash mismatch: {stage_id}"
        )
        _require(
            _sha256(outputs[1]) == fit_manifest.get("parameter_csv_sha256"),
            f"parameter CSV hash mismatch: {stage_id}",
        )
        _require(
            _sha256(outputs[2]) == fit_manifest.get("trace_sha256"),
            f"trace hash mismatch: {stage_id}",
        )
    return checkpoint


def _validate_v4_evidence_and_checkpoints(
    v4_dir: Path,
    decision: Mapping[str, Any],
    signature_sha256: str,
    tracker: SourceTracker,
) -> None:
    include_fixed = decision.get("all_three_refit_bank_gates_passed") is True
    fixed_component_pass: dict[str, bool] = {}
    expected_evidence, stages = _v4_stage_roster(v4_dir, include_fixed=include_fixed)
    paths = decision.get("evidence_paths")
    hashes = decision.get("evidence_sha256")
    _require(isinstance(paths, Mapping), "V4 decision has no evidence paths")
    _require(isinstance(hashes, Mapping), "V4 decision has no evidence hashes")
    _require(set(paths) == set(hashes) == set(expected_evidence), "V4 evidence roster differs")
    resolved_evidence: set[Path] = set()
    for key, expected_path in expected_evidence.items():
        recorded_path = _resolve_recorded(str(paths[key]))
        expected = expected_path.resolve()
        _require(recorded_path == expected, f"V4 evidence path is not canonical: {key}")
        _require(expected not in resolved_evidence, "V4 evidence paths are not one-to-one")
        resolved_evidence.add(expected)
        source = tracker.add(expected)
        _require(_sha256(source) == str(hashes[key]), f"V4 evidence hash mismatch: {key}")

    expected_checkpoint_files = {
        (v4_dir / "checkpoints" / f"{stage_id.replace('/', '__')}.json").resolve()
        for stage_id in stages
    }
    checkpoint_dir = v4_dir / "checkpoints"
    _require(checkpoint_dir.is_dir(), "V4 checkpoint directory is missing")
    actual_checkpoint_files = {
        path.resolve() for path in checkpoint_dir.iterdir() if path.is_file()
    }
    _require(
        actual_checkpoint_files == expected_checkpoint_files,
        "V4 checkpoint inventory differs from the exact stage roster",
    )
    _require(
        all(path.suffix == ".json" for path in checkpoint_dir.iterdir()),
        "unexpected V4 checkpoint entry",
    )

    fit_keys = {
        f"fit/{spec_id}/{scope}/{nodes}/{start}"
        for spec_id in SPEC_IDS
        for scope in SCOPE_IDS
        for nodes, start in FIT_VARIANTS
    }
    start_keys = {f"start/{spec_id}/{scope}" for spec_id in SPEC_IDS for scope in SCOPE_IDS}
    _require(set(decision.get("fit_validity") or {}) == fit_keys, "V4 fit-validity roster differs")
    _require(
        set(decision.get("start_validity") or {}) == start_keys, "V4 start-validity roster differs"
    )
    _require(
        set(decision.get("refit_gate_passed") or {}) == set(SPEC_IDS),
        "V4 refit gate roster differs",
    )
    _require(
        set(decision.get("fixed_bank_gate_passed") or {}) == set(SPEC_IDS),
        "V4 fixed-bank gate roster differs",
    )

    for stage_id, (stage_kind, outputs) in stages.items():
        _validate_v4_checkpoint(v4_dir, stage_id, stage_kind, outputs, signature_sha256, tracker)
        if stage_kind == "fit":
            fit_manifest = _read_json(outputs[-1], tracker)
            parts = stage_id.split("/")
            _require(
                fit_manifest.get("schema_version") == V4_RUN_SCHEMA,
                f"fit manifest schema differs: {stage_id}",
            )
            _require(
                fit_manifest.get("spec_id") == parts[1]
                and fit_manifest.get("nodes") == int(parts[2].removeprefix("nodes_"))
                and fit_manifest.get("scope") == parts[3]
                and fit_manifest.get("start") == parts[4],
                f"fit manifest coordinates differ from its stage: {stage_id}",
            )
            spec_id = str(fit_manifest.get("spec_id"))
            scope = str(fit_manifest.get("scope"))
            nodes = int(fit_manifest.get("nodes", -1))
            start = str(fit_manifest.get("start"))
            key = f"fit/{spec_id}/{scope}/{nodes}/{start}"
            _require(key in fit_keys, f"fit manifest coordinates are invalid: {stage_id}")
            observed_fit_valid = _strict_bool(
                (fit_manifest.get("fit_validity") or {}).get("fit_valid"),
                label=f"fit validity for {stage_id}",
            )
            decision_fit_valid = _strict_bool(
                (decision.get("fit_validity") or {})[key],
                label=f"decision fit validity for {stage_id}",
            )
            _require(
                observed_fit_valid == decision_fit_valid,
                f"fit validity differs from decision: {stage_id}",
            )
        elif stage_kind == "start":
            record = _read_json(outputs[0], tracker)
            parts = stage_id.split("/")
            _require(
                record.get("schema_version") == V4_RUN_SCHEMA,
                f"start record schema differs: {stage_id}",
            )
            _require(
                record.get("spec_id") == parts[1] and record.get("scope") == parts[2],
                f"start record coordinates differ from its stage: {stage_id}",
            )
            key = f"start/{record.get('spec_id')}/{record.get('scope')}"
            _require(key in start_keys, f"start record coordinates are invalid: {stage_id}")
            observed_start_valid = _strict_bool(
                record.get("selection_valid"), label=f"start validity for {stage_id}"
            )
            decision_start_valid = _strict_bool(
                (decision.get("start_validity") or {})[key],
                label=f"decision start validity for {stage_id}",
            )
            _require(
                observed_start_valid == decision_start_valid,
                f"start validity differs from decision: {stage_id}",
            )
        elif stage_kind == "refit":
            record = _read_json(outputs[-1], tracker)
            parts = stage_id.split("/")
            _require(
                record.get("schema_version") == V4_RUN_SCHEMA,
                f"refit record schema differs: {stage_id}",
            )
            _require(
                record.get("spec_id") == parts[1], f"refit record coordinates differ: {stage_id}"
            )
            spec_id = str(record.get("spec_id"))
            _require(spec_id in SPEC_IDS, f"refit record spec is invalid: {stage_id}")
            parameter_rows = _read_csv(
                outputs[0], tracker, required=("scope", "passed"), expected_rows=6
            )
            _require(
                set(map(str, parameter_rows["scope"])) == set(SCOPE_IDS),
                f"refit parameter scope roster differs: {stage_id}",
            )
            parameter_passed = all(
                _strict_bool(value, label=f"refit parameter gate for {stage_id}")
                for value in parameter_rows["passed"]
            )
            start_passed = all(
                _strict_bool(
                    (decision.get("start_validity") or {})[f"start/{spec_id}/{scope}"],
                    label=f"decision start validity for {spec_id}/{scope}",
                )
                for scope in SCOPE_IDS
            )
            heldout = record.get("heldout_common_cell_gate")
            theta = record.get("common_support_refit_theta_gate")
            _require(
                isinstance(heldout, Mapping) and isinstance(theta, Mapping),
                f"refit component-gate records differ: {stage_id}",
            )
            record_start_passed = _strict_bool(
                record.get("all_scope_start_agreement_gates_passed"),
                label=f"refit start aggregate for {stage_id}",
            )
            record_parameter_passed = _strict_bool(
                record.get("all_scope_parameter_gates_passed"),
                label=f"refit parameter aggregate for {stage_id}",
            )
            heldout_passed = _strict_bool(
                heldout.get("passed"), label=f"refit held-out gate for {stage_id}"
            )
            theta_passed = _strict_bool(
                theta.get("passed"), label=f"refit theta gate for {stage_id}"
            )
            record_passed = _strict_bool(
                record.get("passed"), label=f"refit aggregate for {stage_id}"
            )
            _require(
                record.get("comparison") == "normal_trapezoid_bound8_401_vs_801"
                and record_start_passed == start_passed
                and record_parameter_passed == parameter_passed
                and record_passed
                == (start_passed and parameter_passed and heldout_passed and theta_passed),
                f"refit component gates disagree: {stage_id}",
            )
            _require(
                record_passed
                == _strict_bool(
                    (decision.get("refit_gate_passed") or {})[spec_id],
                    label=f"decision refit gate for {stage_id}",
                ),
                f"refit gate differs from decision: {stage_id}",
            )
        else:
            record = _read_json(outputs[-1], tracker)
            parts = stage_id.split("/")
            _require(
                record.get("schema_version") == V4_RUN_SCHEMA,
                f"fixed-bank record schema differs: {stage_id}",
            )
            _require(
                record.get("spec_id") == parts[1],
                f"fixed-bank record coordinates differ: {stage_id}",
            )
            spec_id = str(record.get("spec_id"))
            _require(spec_id in SPEC_IDS, f"fixed-bank record spec is invalid: {stage_id}")
            all_banks_valid = _strict_bool(
                record.get("all_six_fixed_banks_valid"),
                label=f"fixed-bank fit aggregate for {stage_id}",
            )
            numerical_gate = record.get("numerical_scoring_gate")
            _require(
                isinstance(numerical_gate, Mapping),
                f"fixed-bank numerical gate differs: {stage_id}",
            )
            numerical_passed = _strict_bool(
                numerical_gate.get("passed"),
                label=f"fixed-bank numerical aggregate for {stage_id}",
            )
            all_values_finite = _strict_bool(
                numerical_gate.get("all_values_finite"),
                label=f"fixed-bank finite-values gate for {stage_id}",
            )
            comparisons = numerical_gate.get("comparisons")
            expected_comparisons = {
                "bound8_801_vs_bound8_1601",
                "bound8_801_vs_bound10_1001",
            }
            _require(
                isinstance(comparisons, Mapping)
                and set(comparisons) == expected_comparisons
                and all(isinstance(comparisons[name], Mapping) for name in expected_comparisons),
                f"fixed-bank comparison roster differs: {stage_id}",
            )
            comparison_passed = all(
                _strict_bool(
                    (comparisons[name] or {}).get("passed"),
                    label=f"fixed-bank comparison {name} for {stage_id}",
                )
                for name in sorted(expected_comparisons)
            )
            tail_gate = numerical_gate.get("tail_mass")
            _require(isinstance(tail_gate, Mapping), f"fixed-bank tail gate differs: {stage_id}")
            tail_passed = _strict_bool(
                tail_gate.get("passed"), label=f"fixed-bank tail gate for {stage_id}"
            )
            record_passed = _strict_bool(
                record.get("passed"), label=f"fixed-bank aggregate for {stage_id}"
            )
            _require(
                numerical_passed == (all_values_finite and comparison_passed and tail_passed)
                and record_passed == (all_banks_valid and numerical_passed),
                f"fixed-bank component gates disagree: {stage_id}",
            )
            _require(
                record_passed
                == _strict_bool(
                    (decision.get("fixed_bank_gate_passed") or {})[spec_id],
                    label=f"decision fixed-bank gate for {stage_id}",
                ),
                f"fixed-bank gate differs from decision: {stage_id}",
            )
            fixed_component_pass[spec_id] = record_passed and numerical_passed and tail_passed

    if include_fixed:
        _require(
            set(fixed_component_pass) == set(SPEC_IDS)
            and decision.get("all_three_fixed_bank_scoring_and_tail_gates_passed")
            is all(fixed_component_pass.values()),
            "V4 aggregate fixed-bank scoring/tail gate disagrees with component evidence",
        )
    else:
        _require(
            decision.get("all_three_fixed_bank_scoring_and_tail_gates_passed") is False
            and all(
                value is False for value in (decision.get("fixed_bank_gate_passed") or {}).values()
            ),
            "V4 blocked refit stage has inconsistent fixed-bank flags",
        )


def _load_v4_terminal(
    v4_dir: Path, tracker: SourceTracker
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision_path = v4_dir / "numerical_followup_decision.json"
    manifest_path = v4_dir / "study_manifest.json"
    decision = _read_json(decision_path, tracker)
    manifest = _read_json(manifest_path, tracker)
    _require(decision.get("schema_version") == V4_RUN_SCHEMA, "unexpected V4 decision schema")
    _require(manifest.get("schema_version") == V4_RUN_SCHEMA, "unexpected V4 manifest schema")
    _require(
        manifest.get("decision_sha256") == _sha256(decision_path),
        "V4 manifest does not hash-link its decision",
    )
    _require(
        decision.get("status") in {"complete_pass", "blocked_numerical_followup"},
        "V4 study is not terminal",
    )
    _require(manifest.get("status") == decision.get("status"), "V4 terminal statuses differ")
    _require(
        (decision.get("status") == "complete_pass") == (decision.get("passed") is True),
        "V4 status/pass fields disagree",
    )
    _require(
        list(map(str, decision.get("eligible_spec_ids") or [])) == list(SPEC_IDS),
        "V4 eligible specification roster changed",
    )
    _require(
        list(map(str, decision.get("scope_ids") or [])) == list(SCOPE_IDS),
        "V4 scope roster changed",
    )
    _require(
        len(decision.get("fit_validity") or {}) == EXPECTED_FITS,
        "V4 decision does not cover all 54 fits",
    )
    _require(
        len(decision.get("start_validity") or {}) == EXPECTED_STARTS,
        "V4 decision does not cover all 18 start comparisons",
    )
    fit_flags = decision.get("fit_validity") or {}
    start_flags = decision.get("start_validity") or {}
    _require(
        all(type(value) is bool for value in fit_flags.values()),
        "V4 fit-validity values are not boolean",
    )
    _require(
        all(type(value) is bool for value in start_flags.values()),
        "V4 start-validity values are not boolean",
    )
    _require(
        decision.get("all_54_fits_valid") is all(fit_flags.values()),
        "V4 aggregate fit-validity field disagrees",
    )
    _require(
        decision.get("all_18_grid801_start_gates_passed") is all(start_flags.values()),
        "V4 aggregate start-validity field disagrees",
    )
    refit_flags = decision.get("refit_gate_passed") or {}
    fixed_flags = decision.get("fixed_bank_gate_passed") or {}
    _require(
        set(refit_flags) == set(SPEC_IDS)
        and all(type(value) is bool for value in refit_flags.values())
        and decision.get("all_three_refit_bank_gates_passed") is all(refit_flags.values()),
        "V4 aggregate refit-bank gate disagrees with its component roster",
    )
    _require(
        set(fixed_flags) == set(SPEC_IDS)
        and all(type(value) is bool for value in fixed_flags.values()),
        "V4 fixed-bank component roster is invalid",
    )
    aggregate_pass = all(
        decision.get(field) is True
        for field in (
            "all_54_fits_valid",
            "all_18_grid801_start_gates_passed",
            "all_three_refit_bank_gates_passed",
            "all_three_fixed_bank_scoring_and_tail_gates_passed",
        )
    )
    _require(decision.get("passed") is aggregate_pass, "V4 aggregate pass flag disagrees")
    if decision.get("passed") is True:
        _require(
            isinstance(manifest.get("lock_sha256"), str)
            and len(str(manifest.get("lock_sha256"))) == 64,
            "passing V4 manifest has no lock hash",
        )
    else:
        _require(manifest.get("lock_sha256") is None, "blocked V4 manifest claims a lock hash")
    _config_source, signature = _validate_v4_study_signature(v4_dir, decision, manifest, tracker)
    _validate_v4_evidence_and_checkpoints(
        v4_dir, decision, str(signature["canonical_sha256"]), tracker
    )
    return decision, manifest


def _load_v4_lock(
    v4_dir: Path,
    decision: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tracker: SourceTracker,
) -> dict[str, Any]:
    lock_path = v4_dir / "numerical_followup_lock.json"
    companion_path = v4_dir / "numerical_followup_lock.sha256"
    lock = _read_json(lock_path, tracker)
    companion = tracker.add(companion_path).read_text(encoding="utf-8").strip()
    _require(lock.get("schema_version") == V4_LOCK_SCHEMA, "unexpected V4 lock schema")
    _require(
        lock.get("status") == "complete_pass" and lock.get("passed") is True, "invalid V4 lock"
    )
    _require(list(lock.get("eligible_spec_ids") or []) == list(SPEC_IDS), "V4 lock specs changed")
    _require(list(lock.get("scope_ids") or []) == list(SCOPE_IDS), "V4 lock scopes changed")
    _require(
        lock.get("followup_decision_sha256")
        == _sha256(v4_dir / "numerical_followup_decision.json"),
        "V4 lock does not hash-link its decision",
    )
    _require(manifest.get("lock_sha256") == _sha256(lock_path), "V4 manifest lock hash differs")
    _require(
        companion == f"{_sha256(lock_path)}  numerical_followup_lock.json",
        "V4 lock companion hash is invalid",
    )
    _require(
        lock.get("evidence_paths") == decision.get("evidence_paths"),
        "V4 lock evidence paths differ",
    )
    _require(
        lock.get("evidence_sha256") == decision.get("evidence_sha256"),
        "V4 lock evidence hashes differ",
    )
    for field_name in ("config_sha256", "frozen_contract_sha256", "study_signature_sha256"):
        _require(
            lock.get(field_name) == decision.get(field_name),
            f"V4 lock/decision {field_name} differs",
        )
    return lock


def _load_numerical_tables(
    v4_dir: Path,
    decision: Mapping[str, Any],
    tracker: SourceTracker,
) -> dict[str, Any]:
    item_frames: dict[str, pd.DataFrame] = {}
    fit_manifests: list[dict[str, Any]] = []
    start_records: list[dict[str, Any]] = []
    refit_parameters: dict[str, pd.DataFrame] = {}
    refit_theta: dict[str, pd.DataFrame] = {}
    refit_gates: dict[str, dict[str, Any]] = {}
    fixed_theta: dict[str, pd.DataFrame] = {}
    fixed_gates: dict[str, dict[str, Any]] = {}

    for spec_id in SPEC_IDS:
        item_frames[spec_id] = _read_csv(
            v4_dir / "fits" / spec_id / "nodes_0401" / "full" / "cold" / "item_params.csv",
            tracker,
            required=("criterion_id", "a_instruction_following", "b", "exportable"),
        )
        for nodes, start in ((401, "cold"), (801, "cold"), (801, "continuation")):
            for scope in ("full", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4"):
                manifest = _read_json(
                    v4_dir
                    / "fits"
                    / spec_id
                    / f"nodes_{nodes:04d}"
                    / scope
                    / start
                    / "fit_manifest.json",
                    tracker,
                )
                fit_manifests.append(manifest)
        for scope in ("full", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4"):
            start_records.append(
                _read_json(v4_dir / "start_selection" / spec_id / f"{scope}.json", tracker)
            )
        comparison_dir = v4_dir / "fit_comparisons" / "401_vs_801" / spec_id
        refit_parameters[spec_id] = _read_csv(
            comparison_dir / "parameter_scope_gates.csv",
            tracker,
            required=(
                "scope",
                "b_spearman",
                "exportability_agreement",
                "minimum_parameter_spearman",
                "minimum_exportability_agreement",
                "passed",
            ),
            expected_rows=6,
        )
        refit_theta[spec_id] = _read_csv(
            comparison_dir / "theta_common_support.csv",
            tracker,
            required=("scope", "model", "support", "theta_401", "theta_801"),
        )
        # The held-out cell table is a plotted source through its gate summary and
        # is still tracked/validated so the report cannot hide its denominator.
        _read_csv(
            comparison_dir / "heldout_cells.csv",
            tracker,
            required=(
                "outer_fold",
                "model",
                "criterion_id",
                "log_loss_delta_801_minus_401",
                "brier_delta_801_minus_401",
            ),
        )
        refit_gates[spec_id] = _read_json(comparison_dir / "fit_pair_gate.json", tracker)

    _require(len(fit_manifests) == EXPECTED_FITS, "renderer did not load 54 fit manifests")
    _require(len(start_records) == EXPECTED_STARTS, "renderer did not load 18 start records")

    # Fixed-bank scoring is intentionally conditional.  The V4 runner does not
    # create it when a refit-bank gate fails.
    if decision.get("all_three_refit_bank_gates_passed") is True:
        for spec_id in SPEC_IDS:
            fixed_dir = v4_dir / "fixed_bank" / spec_id
            fixed_theta[spec_id] = _read_csv(
                fixed_dir / "theta_profiles.csv",
                tracker,
                required=(
                    "scope",
                    "model",
                    "support",
                    "theta_bound8_801",
                    "theta_bound8_1601",
                    "theta_bound10_1001",
                    "tail_bound8_801",
                    "tail_bound8_1601",
                    "tail_bound10_1001",
                ),
            )
            _read_csv(fixed_dir / "support.csv", tracker, required=("scope", "n_score_models"))
            fixed_gates[spec_id] = _read_json(fixed_dir / "fixed_bank_gate.json", tracker)

    return {
        "item_frames": item_frames,
        "fit_manifests": fit_manifests,
        "start_records": start_records,
        "refit_parameters": refit_parameters,
        "refit_theta": refit_theta,
        "refit_gates": refit_gates,
        "fixed_theta": fixed_theta,
        "fixed_gates": fixed_gates,
    }


def _manifest_output_hash(manifest: Mapping[str, Any], name: str) -> str | None:
    raw = (manifest.get("outputs") or {}).get(name)
    return str(raw.get("sha256")) if isinstance(raw, Mapping) and raw.get("sha256") else None


def _validate_phase3_provenance(
    *,
    phase3_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    v4_dir: Path,
    v4_decision: Mapping[str, Any],
    v4_manifest: Mapping[str, Any],
    v4_lock: Mapping[str, Any],
    decision: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selected: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    tracker: SourceTracker,
) -> None:
    inputs = manifest.get("inputs")
    _require(isinstance(inputs, Mapping), "Phase-3 manifest input provenance is missing")
    expected_input_names = {
        "config",
        "split_manifest",
        "response_matrix",
        "rubrics",
        "scenarios",
        "judge_manifest",
    }
    _require(set(inputs) == expected_input_names, "Phase-3 manifest input roster differs")
    config_source = _validate_recorded_file(
        inputs["config"], config_path, tracker, label="Phase-3 config"
    )
    _require(config_source == config_path.resolve(), "Phase-3 config path changed")
    _require(
        list(map(str, config.get("code_dependencies") or [])) == list(V3_CODE_DEPENDENCIES),
        "Phase-3 config code-dependency roster differs",
    )
    _require(
        manifest.get("script") == "scripts/nested_scenario_cat_cv_v3.py",
        "Phase-3 producer script differs",
    )
    configured_cv = config.get("cross_validation") or {}
    _require(
        configured_cv.get("repetitions") == 5
        and configured_cv.get("outer_folds_per_repetition") == 5
        and configured_cv.get("inner_folds_per_outer_panel") == 4
        and configured_cv.get("group_related_model_families") is True
        and configured_cv.get("aggregate_repeated_predictions_per_model") is True
        and configured_cv.get("family_cluster_bootstrap") is True
        and configured_cv.get("treat_model_repeat_rows_as_independent") is False,
        "Phase-3 frozen cross-validation contract differs",
    )
    _require(
        manifest.get("cross_validation")
        == {
            "repetitions": 5,
            "outer_folds_per_repeat": 5,
            "inner_folds_per_outer": 4,
            "bootstrap_unit": "tutor_family",
            "repeated_rows_treated_as_independent": False,
        },
        "Phase-3 manifest cross-validation contract differs",
    )
    _require(
        manifest.get("inner_selection_fail_closed")
        == {
            "require_all_specs_every_inner_fold": True,
            "expected_spec_fold_combinations_per_panel": 12,
            "survivor_selection_allowed": False,
            "outer_outcomes_opened_only_after_all_panel_selections_locked": True,
        }
        and manifest.get("primary_policy_only_can_pass_phase3") is True
        and manifest.get("sensitivity_promotion_allowed") is False,
        "Phase-3 manifest selection policy contract differs",
    )
    configured_outputs = config.get("outputs") or {}
    _require(
        configured_outputs
        == {
            "numerical_checks": "runs/calibration/InFoBench_v2_numerical_followup_v4",
            "phase3": "runs/calibration/InFoBench_v3/phase3",
            "phase4": "runs/calibration/InFoBench_v3/phase4",
            "final_fit": "runs/calibration/InFoBench_v3/final_fit",
            "report": "reports/infobench_calibration_cat_v3",
            "never_overwrite_v1_or_v2": True,
        },
        "Phase-3 configured output directories differ",
    )

    baseline = config.get("baseline") or {}
    cross_validation = config.get("cross_validation") or {}
    configured_sources = {
        "response_matrix": (
            _resolve_recorded(str(baseline.get("response_matrix") or "")),
            str(baseline.get("response_matrix_sha256") or ""),
        ),
        "rubrics": (
            _resolve_recorded(str(baseline.get("rubrics") or "")),
            str(baseline.get("rubrics_sha256") or ""),
        ),
        "scenarios": (
            _resolve_recorded(str(baseline.get("scenarios") or "")),
            str(baseline.get("scenarios_sha256") or ""),
        ),
        "judge_manifest": (
            _resolve_recorded(str(baseline.get("judge_manifest") or "")),
            str(baseline.get("judge_manifest_sha256") or ""),
        ),
        "split_manifest": (
            _resolve_recorded(str(cross_validation.get("split_manifest") or "")),
            str(cross_validation.get("split_manifest_sha256") or ""),
        ),
    }
    input_hashes: dict[str, str] = {}
    for name, (path, configured_hash) in configured_sources.items():
        source = _validate_recorded_file(inputs[name], path, tracker, label=f"Phase-3 {name}")
        observed = _sha256(source)
        _require(observed == configured_hash, f"Phase-3 config hash differs for {name}")
        if name != "split_manifest":
            input_hashes[name] = observed
    split_path = configured_sources["split_manifest"][0]
    split_hash = configured_sources["split_manifest"][1]

    code = manifest.get("code_provenance")
    _require(isinstance(code, Mapping), "Phase-3 code provenance is missing")
    files = code.get("files")
    inventory = code.get("dependency_inventory")
    _require(isinstance(files, Mapping) and bool(files), "Phase-3 code file hashes are missing")
    _require(isinstance(inventory, list), "Phase-3 dependency inventory is missing")
    _require(
        list(map(str, inventory)) == list(V3_CODE_DEPENDENCIES)
        and set(map(str, files)) == set(V3_CODE_DEPENDENCIES),
        "Phase-3 code inventory differs",
    )
    for path_text, expected_hash in files.items():
        source = tracker.add(_resolve_recorded(str(path_text)))
        _require(_sha256(source) == str(expected_hash), f"Phase-3 code hash mismatch: {path_text}")
    code_hash = _canonical_hash(files)
    _require(code.get("canonical_sha256") == code_hash, "Phase-3 code hash panel is invalid")

    environment = manifest.get("environment")
    _require(isinstance(environment, Mapping), "Phase-3 environment provenance is missing")
    environment_payload = {
        key: value for key, value in environment.items() if key != "canonical_sha256"
    }
    environment_hash = _canonical_hash(environment_payload)
    _require(
        environment.get("canonical_sha256") == environment_hash,
        "Phase-3 environment hash is invalid",
    )

    study_signature = str(manifest.get("study_signature") or "")
    _require(len(study_signature) == 64, "Phase-3 study signature is invalid")
    config_hash = _sha256(config_path)
    for artifact_name, artifact in (
        ("decision", decision),
        ("selected-spec handoff", selected),
        ("pre-outer selection lock", selection_lock),
    ):
        _require(
            artifact.get("study_signature") == study_signature,
            f"Phase-3 {artifact_name} study signature differs",
        )
    expected_common = {
        "config_sha256": config_hash,
        "split_sha256": split_hash,
        "input_hashes": input_hashes,
        "code_sha256": code_hash,
        "environment_sha256": environment_hash,
    }
    for field_name, expected in expected_common.items():
        _require(
            decision.get(field_name) == expected,
            f"Phase-3 decision {field_name} differs from its manifest",
        )
    for field_name in ("input_hashes", "code_sha256", "environment_sha256"):
        _require(
            selected.get(field_name) == expected_common[field_name],
            f"Phase-3 selected-spec handoff {field_name} differs from its manifest",
        )
    selected_config = selected.get("config") or {}
    selected_split = selected.get("split_manifest") or {}
    _validate_recorded_file(selected_config, config_path, tracker, label="selected-spec config")
    _validate_recorded_file(selected_split, split_path, tracker, label="selected-spec split")

    numerical = manifest.get("numerical_lock")
    _require(isinstance(numerical, Mapping), "Phase-3 numerical-lock provenance is missing")
    lock_path = v4_dir / "numerical_followup_lock.json"
    v4_config_path = _resolve_recorded(str(v4_manifest.get("config") or ""))
    _require(numerical.get("status") == "passed", "Phase-3 did not consume a passing V4 lock")
    _require(
        _resolve_recorded(str(numerical.get("followup_config_path") or ""))
        == v4_config_path.resolve()
        and numerical.get("followup_config_sha256") == _sha256(v4_config_path),
        "Phase-3 V4 config path/hash differs",
    )
    _require(
        _resolve_recorded(str(numerical.get("verification_path") or "")) == lock_path.resolve()
        and numerical.get("verification_sha256") == _sha256(lock_path),
        "Phase-3 V4 lock path/hash differs",
    )
    _require(
        numerical.get("followup_decision_sha256")
        == _sha256(v4_dir / "numerical_followup_decision.json")
        and numerical.get("followup_manifest_sha256") == _sha256(v4_dir / "study_manifest.json"),
        "Phase-3 V4 decision/manifest chain differs",
    )
    _require(
        numerical.get("study_signature_sha256")
        == v4_lock.get("study_signature_sha256")
        == v4_decision.get("study_signature_sha256")
        == v4_manifest.get("study_signature_sha256"),
        "Phase-3 V4 study-signature chain differs",
    )
    _require(
        numerical.get("evidence_sha256") == v4_lock.get("evidence_sha256"),
        "Phase-3 V4 evidence chain differs",
    )
    _require(
        numerical.get("frozen_contract_sha256") == v4_lock.get("frozen_contract_sha256"),
        "Phase-3 V4 frozen-contract chain differs",
    )
    expected_numerical_keys = {
        "fit_grid",
        "fit_quadrature_method",
        "fit_linear_bound",
        "fit_convergence_mode",
        "fit_max_iter",
        "fit_objective_tolerance",
        "fit_parameter_tolerance",
        "fit_consecutive_convergence_passes",
        "eap_grid",
        "quadrature_method",
        "linear_bound",
        "tail_region",
        "verification_path",
        "verification_sha256",
        "followup_config_path",
        "followup_config_sha256",
        "followup_decision_sha256",
        "followup_manifest_sha256",
        "study_signature_sha256",
        "frozen_contract_sha256",
        "evidence_sha256",
        "status",
        "passed_spec_ids",
        "eligible_spec_ids",
        "historical_fit_or_checkpoint_artifacts_reused",
    }
    _require(
        set(numerical) == expected_numerical_keys,
        "Phase-3 numerical-lock field inventory differs",
    )
    expected_numerical_profile = {
        "fit_grid": 401,
        "fit_quadrature_method": "normal_trapezoid",
        "fit_linear_bound": 8.0,
        "fit_convergence_mode": "returned_iterate",
        "fit_max_iter": 1500,
        "fit_objective_tolerance": 1e-4,
        "fit_parameter_tolerance": 5e-5,
        "fit_consecutive_convergence_passes": 2,
        "eap_grid": 801,
        "quadrature_method": "normal_trapezoid",
        "linear_bound": 8.0,
        "tail_region": 7.5,
    }
    _require(
        all(numerical.get(key) == value for key, value in expected_numerical_profile.items()),
        "Phase-3 numerical profile differs from the frozen V4 profile",
    )
    _require(
        list(map(str, numerical.get("passed_spec_ids") or [])) == list(SPEC_IDS)
        and list(map(str, numerical.get("eligible_spec_ids") or [])) == list(SPEC_IDS)
        and numerical.get("historical_fit_or_checkpoint_artifacts_reused") == 0,
        "Phase-3 numerical specification/reuse contract differs",
    )
    config_numerical = config.get("numerical_lock") or {}
    expected_config_numerical = {
        "fit_quadrature_method": "normal_trapezoid",
        "fit_linear_bound": 8.0,
        "fit_grid": 401,
        "convergence_mode": "returned_iterate",
        "objective_tolerance": 1e-4,
        "parameter_tolerance": 5e-5,
        "consecutive_convergence_passes": 2,
        "eap_quadrature_method": "normal_trapezoid",
        "eap_linear_bound": 8.0,
        "eap_grid": 801,
        "tail_region": 7.5,
    }
    _require(
        all(config_numerical.get(key) == value for key, value in expected_config_numerical.items())
        and list(map(str, config_numerical.get("eligible_spec_ids") or [])) == list(SPEC_IDS),
        "Phase-3 config numerical lock differs",
    )

    latent = config.get("latent_structure") or {}
    _require(
        latent.get("name") == "overall_1d"
        and latent.get("source_skills") == ["content", "format", "number", "style", "linguistic"]
        and latent.get("dimensions")
        == [
            {
                "label": "instruction_following",
                "members": ["content", "format", "number", "style", "linguistic"],
            }
        ],
        "Phase-3 latent-structure configuration differs",
    )
    # The V3 producer hashes the normalized CLI-built structure, whose frozen
    # name differs from the shorter descriptive name in the config.
    structure = {
        "name": V3_STRUCTURE_NAME,
        "source_skills": latent.get("source_skills"),
        "dimensions": latent.get("dimensions"),
    }
    configured_spec_records = config.get("calibration_specifications")
    _require(
        isinstance(configured_spec_records, list)
        and all(isinstance(record, Mapping) for record in configured_spec_records)
        and [str(record.get("spec_id")) for record in configured_spec_records] == list(SPEC_IDS),
        "Phase-3 config calibration-specification roster differs",
    )
    configured_specs = {str(record.get("spec_id")): record for record in configured_spec_records}
    manifest_specs = manifest.get("calibration_specifications")
    _require(
        isinstance(manifest_specs, list)
        and all(isinstance(record, Mapping) for record in manifest_specs)
        and [str(record.get("spec_id")) for record in manifest_specs] == list(SPEC_IDS),
        "Phase-3 manifest calibration-specification roster differs",
    )
    signature_specs: list[dict[str, Any]] = []
    for index, record in enumerate(manifest_specs):
        spec_id = str(record.get("spec_id"))
        configured = configured_specs.get(spec_id)
        _require(configured is not None, f"Phase-3 config is missing specification {spec_id}")
        raw_ridge = configured.get("ridge")
        raw_shrinkage = configured.get("log_a_shrinkage")
        simplicity_rank = int(record.get("simplicity_rank", -1))
        _require(
            simplicity_rank == int(configured.get("simplicity_rank", -1)) == index,
            f"Phase-3 calibration simplicity rank differs for {spec_id}",
        )
        canonical = record.get("complete_specification")
        _require(
            isinstance(canonical, Mapping),
            f"Phase-3 complete calibration specification is missing for {spec_id}",
        )
        semantic = {key: value for key, value in canonical.items() if key != "cache_key"}
        expected_cache_key = f"calibration-spec-v1-{_canonical_hash(semantic)}"
        _require(
            canonical.get("cache_key")
            == configured.get("canonical_cache_key")
            == expected_cache_key
            and canonical.get("family") == configured.get("family"),
            f"Phase-3 canonical calibration specification differs for {spec_id}",
        )
        signature_specs.append(
            {
                "spec_id": spec_id,
                "family": configured.get("family"),
                "ridge": None if raw_ridge is None else float(raw_ridge),
                "log_a_shrinkage": (None if raw_shrinkage is None else float(raw_shrinkage)),
                "simplicity_rank": simplicity_rank,
                "canonical": dict(canonical),
            }
        )

    expected_policies = [
        {
            "policy_id": PRIMARY_POLICY,
            "role": "primary",
            "minimum_scenarios": 15,
            "conditional_se_target": 0.2,
            "selector": "trace",
        },
        {
            "policy_id": "sensitivity_floor12_se0p20_trace",
            "role": "sensitivity",
            "minimum_scenarios": 12,
            "conditional_se_target": 0.2,
            "selector": "trace",
        },
        {
            "policy_id": "sensitivity_floor15_se0p20_dopt",
            "role": "sensitivity",
            "minimum_scenarios": 15,
            "conditional_se_target": 0.2,
            "selector": "dopt",
        },
    ]
    configured_policy_panel = config.get("cat_policies") or {}
    configured_primary = configured_policy_panel.get("primary") or {}
    configured_sensitivities = configured_policy_panel.get("sensitivities") or []
    _require(
        isinstance(configured_primary, Mapping)
        and isinstance(configured_sensitivities, list)
        and len(configured_sensitivities) == 2
        and all(isinstance(record, Mapping) for record in configured_sensitivities),
        "Phase-3 config CAT policy panel is invalid",
    )
    configured_policy_records = [configured_primary, *configured_sensitivities]
    normalized_config_policies = [
        {
            "policy_id": str(record.get("policy_id") or ""),
            "role": "primary" if index == 0 else "sensitivity",
            "minimum_scenarios": int(record.get("minimum_scenarios", -1)),
            "conditional_se_target": float(record.get("conditional_se_target", math.nan)),
            "selector": str(record.get("selector") or ""),
        }
        for index, record in enumerate(configured_policy_records)
    ]
    manifest_policies = manifest.get("cat_policies")
    _require(
        normalized_config_policies == expected_policies
        and manifest_policies == expected_policies
        and configured_policy_panel.get("sensitivities_can_replace_primary") is False,
        "Phase-3 frozen CAT policy contract differs",
    )
    _require(
        selected.get("primary_policy") == expected_policies[0]
        and selected.get("sensitivities") == expected_policies[1:]
        and selected.get("sensitivity_promotion_allowed") is False
        and selected.get("require_all_specs_every_inner_fold") is True
        and selected.get("survivor_selection_allowed") is False,
        "Phase-3 selected-spec policy contract differs",
    )

    runtime = manifest.get("runtime")
    _require(isinstance(runtime, Mapping), "Phase-3 runtime provenance is missing")
    runtime_keys = (
        "seed",
        "top_n",
        "max_scenarios",
        "minimum_scored_criteria",
        "max_iter",
        "tol",
        "fit_parameter_tolerance",
        "fit_consecutive_convergence_passes",
        "fit_convergence_mode",
        "fit_quadrature_method",
        "fit_linear_bound",
        "mwle_ridge",
        "metric_bootstrap_replicates",
        "negative_policy",
    )
    _require(
        set(runtime) == set(runtime_keys) | {"max_grid_nodes"},
        "Phase-3 runtime field inventory differs",
    )
    configured_runtime = config.get("runtime") or {}
    expected_runtime = {
        "seed": configured_runtime.get("master_seed"),
        "top_n": configured_policy_panel.get("top_n"),
        "max_scenarios": configured_policy_panel.get("maximum_adaptive_scenarios"),
        "minimum_scored_criteria": configured_policy_panel.get("minimum_scored_criteria"),
        "max_iter": configured_runtime.get("fit_max_iter"),
        "tol": configured_runtime.get("fit_tolerance"),
        "fit_parameter_tolerance": numerical.get("fit_parameter_tolerance"),
        "fit_consecutive_convergence_passes": numerical.get("fit_consecutive_convergence_passes"),
        "fit_convergence_mode": numerical.get("fit_convergence_mode"),
        "fit_quadrature_method": numerical.get("fit_quadrature_method"),
        "fit_linear_bound": numerical.get("fit_linear_bound"),
        "mwle_ridge": configured_runtime.get("mwle_ridge"),
        "metric_bootstrap_replicates": configured_runtime.get("metric_family_bootstrap_replicates"),
        "negative_policy": configured_runtime.get("negative_loading_policy"),
    }
    _require(
        expected_runtime
        == {
            "seed": 20260805,
            "top_n": 5,
            "max_scenarios": 50,
            "minimum_scored_criteria": 15,
            "max_iter": 1500,
            "tol": 1e-4,
            "fit_parameter_tolerance": 5e-5,
            "fit_consecutive_convergence_passes": 2,
            "fit_convergence_mode": "returned_iterate",
            "fit_quadrature_method": "normal_trapezoid",
            "fit_linear_bound": 8.0,
            "mwle_ridge": 1e-6,
            "metric_bootstrap_replicates": 2000,
            "negative_policy": "drop",
        }
        and {key: runtime[key] for key in runtime_keys} == expected_runtime
        and isinstance(runtime.get("max_grid_nodes"), int)
        and int(runtime["max_grid_nodes"]) >= int(numerical["eap_grid"]),
        "Phase-3 frozen runtime contract differs",
    )
    configured_gates = config.get("selection_gates") or {}
    _require(
        configured_gates.get("apply_to_primary_only") is True
        and configured_gates.get("allow_fallback_if_none_pass", False) is False
        and configured_gates.get("allow_fallback_if_primary_fails", False) is False
        and configured_gates.get("require_every_outer_panel") is True
        and configured_gates.get("require_every_repetition_pooled_gate") is True
        and configured_gates.get("minimum_nominal_precision_lower_95_ci") == 0.90
        and configured_gates.get("minimum_scenario_reduction_vs_random") == 0.50
        and tuple(configured_gates.get("outer_panel_applied_gate_names") or ())
        == PANEL_APPLIED_GATE_NAMES
        and tuple(configured_gates.get("outer_panel_diagnostic_gate_names") or ())
        == PANEL_DIAGNOSTIC_GATE_NAMES
        and tuple(configured_gates.get("repetition_applied_gate_names") or ())
        == REPETITION_APPLIED_GATE_NAMES
        and configured_gates.get("outer_panel_inference_unit")
        == "unique_outer_test_tutor_within_panel; confidence intervals diagnostic only"
        and configured_gates.get("repetition_inference_unit")
        == "52_unique_out_of_fold_tutors_once_per_repetition"
        and configured_gates.get("repetition_expected_unique_oof_tutors") == 52
        and configured_gates.get("cross_repeat_gate_role")
        == "diagnostic_only_after_per_model_aggregation"
        and configured_gates.get("treat_260_model_repeat_rows_as_independent") is False,
        "Phase-3 configured gate-scope contract differs",
    )
    gate_scope_contract = {
        "outer_panel": {
            "decision_role": "applied_point_estimate_gates",
            "inference_unit": configured_gates["outer_panel_inference_unit"],
            "applied_gate_names": list(PANEL_APPLIED_GATE_NAMES),
            "diagnostic_only_gate_names": list(PANEL_DIAGNOSTIC_GATE_NAMES),
            "confidence_intervals_affect_decision": False,
        },
        "repetition": {
            "decision_role": "applied_all_absolute_gates",
            "inference_unit": configured_gates["repetition_inference_unit"],
            "expected_unique_oof_tutors": configured_gates["repetition_expected_unique_oof_tutors"],
            "applied_gate_names": list(REPETITION_APPLIED_GATE_NAMES),
            "diagnostic_only_gate_names": [],
            "nominal_precision_lower_95_ci_threshold": configured_gates[
                "minimum_nominal_precision_lower_95_ci"
            ],
        },
        "cross_repeat": {
            "decision_role": configured_gates["cross_repeat_gate_role"],
            "inference_unit": "52_unique_tutors_after_within_tutor_mean_across_5_repetitions",
            "raw_model_repeat_rows": 260,
            "raw_model_repeat_rows_treated_as_independent": False,
        },
    }
    _require(
        manifest.get("gate_scope_contract")
        == decision.get("gate_scope_contract")
        == gate_scope_contract,
        "Phase-3 gate-scope provenance differs",
    )
    signature_payload = {
        "schema": V3_RUN_SCHEMA,
        "config_sha256": config_hash,
        "split_sha256": split_hash,
        "inputs": input_hashes,
        "code": dict(files),
        "environment": dict(environment),
        "structure": structure,
        "numerical": dict(numerical),
        "specifications": signature_specs,
        "policies": manifest_policies,
        "runtime": {key: runtime[key] for key in runtime_keys},
    }
    _require(
        _canonical_hash(signature_payload) == study_signature,
        "Phase-3 study signature cannot be reproduced from frozen provenance",
    )


def _phase3_panel_models(
    fold_assignments: Mapping[str, Any],
) -> tuple[set[str], dict[tuple[int, int], set[str]]]:
    model_to_family = fold_assignments.get("model_to_family")
    repetitions = fold_assignments.get("repetitions")
    _require(isinstance(model_to_family, Mapping), "Phase-3 model-family roster is missing")
    _require(isinstance(repetitions, list), "Phase-3 repeated-fold roster is missing")
    models = set(map(str, model_to_family))
    _require(len(models) == 52, "Phase-3 fold assignment does not contain 52 models")
    panel_models: dict[tuple[int, int], set[str]] = {}
    for repetition in repetitions:
        _require(isinstance(repetition, Mapping), "invalid Phase-3 repetition record")
        repeat = int(repetition.get("repeat", -1))
        outer_folds = repetition.get("outer_folds")
        _require(isinstance(outer_folds, list), f"Phase-3 repeat {repeat} has no outer folds")
        observed_in_repeat: list[str] = []
        for outer in outer_folds:
            _require(isinstance(outer, Mapping), "invalid Phase-3 outer-fold record")
            outer_fold = int(outer.get("outer_fold", -1))
            key = (repeat, outer_fold)
            _require(key not in panel_models, f"duplicate Phase-3 panel {key}")
            test_models = list(map(str, outer.get("test_model_ids") or []))
            _require(
                len(test_models) == len(set(test_models)), f"duplicate test model in panel {key}"
            )
            _require(set(test_models) <= models, f"unknown test model in panel {key}")
            panel_models[key] = set(test_models)
            observed_in_repeat.extend(test_models)
        _require(
            len(observed_in_repeat) == 52 and set(observed_in_repeat) == models,
            f"Phase-3 repeat {repeat} is not an exact 52-model partition",
        )
    expected_panels = {(repeat, fold) for repeat in range(5) for fold in range(5)}
    _require(set(panel_models) == expected_panels, "Phase-3 repeated panel roster differs")
    return models, panel_models


def _panel_payload_map(
    payload: Mapping[str, Any], *, label: str
) -> dict[tuple[int, int], Mapping[str, Any]]:
    rows = payload.get("panels")
    _require(isinstance(rows, list) and len(rows) == 25, f"{label} does not contain 25 panels")
    output: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), f"{label} contains an invalid panel")
        key = (int(row.get("repeat", -1)), int(row.get("outer_fold", -1)))
        _require(key not in output, f"{label} contains duplicate panel {key}")
        _require(
            row.get("panel_id") == f"repeat_{key[0]:02d}_outer_{key[1]:02d}",
            f"{label} panel ID differs for {key}",
        )
        output[key] = row
    _require(
        set(output) == {(repeat, fold) for repeat in range(5) for fold in range(5)},
        f"{label} panel roster differs",
    )
    return output


def _panel_inner_audit(panel: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    inner = panel.get("inner_selection")
    _require(isinstance(inner, Mapping), f"{label} has no inner-selection audit")
    audit = inner.get("complete_inner_evidence_audit")
    _require(isinstance(audit, Mapping), f"{label} has no complete-inner-evidence audit")
    expected = int(audit.get("expected_spec_fold_combinations", -1))
    valid = int(audit.get("valid_spec_fold_combinations", -1))
    failures = audit.get("failed_checks")
    expected_folds = audit.get("expected_inner_folds")
    expected_specs = audit.get("expected_spec_ids")
    validation_by_fold = audit.get("validation_models_by_inner_fold")
    combinations = audit.get("combination_audits")
    _require(
        expected == 12
        and 0 <= valid <= expected
        and isinstance(failures, list)
        and all(isinstance(value, str) for value in failures),
        f"{label} inner-evidence audit fields differ",
    )
    _require(
        audit.get("require_all_specs_every_inner_fold") is True
        and audit.get("survivor_selection_allowed") is False
        and expected_folds == list(range(4))
        and expected_specs == list(SPEC_IDS)
        and isinstance(validation_by_fold, Mapping)
        and set(map(str, validation_by_fold)) == {str(index) for index in range(4)}
        and all(
            isinstance(values, list)
            and bool(values)
            and list(map(str, values)) == sorted(set(map(str, values)))
            for values in validation_by_fold.values()
        )
        and isinstance(combinations, list)
        and len(combinations) == 12,
        f"{label} inner-evidence audit roster differs",
    )
    observed_keys: set[tuple[int, str]] = set()
    derived_valid = 0
    direct_failures: set[str] = set()
    for combination in combinations:
        _require(isinstance(combination, Mapping), f"{label} has an invalid combination audit")
        inner_fold = int(combination.get("inner_fold", -1))
        spec_id = str(combination.get("spec_id") or "")
        key = (inner_fold, spec_id)
        _require(
            key not in observed_keys and inner_fold in range(4) and spec_id in SPEC_IDS,
            f"{label} combination-audit roster differs",
        )
        observed_keys.add(key)
        expected_models = list(map(str, combination.get("expected_validation_models") or []))
        observed_models = list(map(str, combination.get("observed_validation_models") or []))
        checks = combination.get("checks")
        _require(
            expected_models == list(map(str, validation_by_fold[str(inner_fold)]))
            and isinstance(checks, Mapping)
            and set(map(str, checks)) == set(INNER_COMBINATION_CHECK_NAMES)
            and all(type(value) is bool for value in checks.values())
            and checks.get("rows_present") is bool(observed_models)
            and checks.get("validation_model_coverage_exact")
            is (
                len(observed_models) == len(expected_models)
                and len(set(observed_models)) == len(observed_models)
                and set(observed_models) == set(expected_models)
            ),
            f"{label} combination-audit fields differ for {key}",
        )
        combination_passed = all(checks.values())
        _require(
            combination.get("passed") is combination_passed,
            f"{label} combination-audit pass flag disagrees for {key}",
        )
        if combination_passed:
            derived_valid += 1
        else:
            direct_failures.update(
                f"inner_{inner_fold}_{spec_id}_{name}"
                for name, check_passed in checks.items()
                if not check_passed
            )
    cross_support_failures = {
        f"inner_{inner_fold}_common_support_hash_not_identical" for inner_fold in range(4)
    } | {
        f"inner_{inner_fold}_{model}_observed_cell_support_not_identical"
        for inner_fold in range(4)
        for model in map(str, validation_by_fold[str(inner_fold)])
    }

    _require(
        observed_keys == {(inner_fold, spec_id) for inner_fold in range(4) for spec_id in SPEC_IDS}
        and valid == derived_valid
        and failures == sorted(set(failures))
        and direct_failures <= set(failures) <= direct_failures | cross_support_failures,
        f"{label} combination-audit totals differ",
    )
    passed = valid == expected and not failures
    _require(
        audit.get("passed") is passed,
        f"{label} inner-evidence audit pass flag disagrees",
    )
    return dict(audit)


def _strict_gate_checks(value: Any, *, scope: str, primary: bool, label: str) -> dict[str, bool]:
    if isinstance(value, Mapping):
        parsed: Any = dict(value)
        _require(
            len(set(map(str, parsed))) == len(parsed),
            f"{label} gate_checks contains duplicate JSON keys",
        )
    else:

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            parsed_object: dict[str, Any] = {}
            for key, item in pairs:
                if key in parsed_object:
                    raise ValueError(f"duplicate JSON key: {key}")
                parsed_object[key] = item
            return parsed_object

        try:
            parsed = json.loads(str(value), object_pairs_hook=reject_duplicate_keys)
        except (TypeError, json.JSONDecodeError) as error:
            raise TerminalReportError(f"{label} gate_checks is malformed JSON") from error
        except ValueError as error:
            raise TerminalReportError(
                f"{label} gate_checks contains duplicate JSON keys"
            ) from error
    _require(isinstance(parsed, Mapping), f"{label} gate_checks is not a JSON object")
    expected = (
        set(PANEL_APPLIED_GATE_NAMES if scope == "outer_panel" else REPETITION_APPLIED_GATE_NAMES)
        | {"inference_scope_complete"}
        if primary
        else set()
    )
    _require(set(map(str, parsed)) == expected, f"{label} gate_checks key roster differs")
    _require(
        all(type(value) is bool for value in parsed.values()),
        f"{label} gate_checks values are not boolean",
    )
    return {str(key): bool(item) for key, item in parsed.items()}


def _validate_phase3_rosters(
    decision: Mapping[str, Any], tables: Mapping[str, Any], *, selection_only: bool
) -> dict[str, Any]:
    models, panel_models = _phase3_panel_models(tables["fold_assignments"])
    model_to_family = {
        str(model): str(family)
        for model, family in (tables["fold_assignments"].get("model_to_family") or {}).items()
    }
    selected_panels = _panel_payload_map(tables["selected"], label="selected-spec handoff")
    choice_panels = _panel_payload_map(tables["choices"], label="calibration choices")
    lock_panels = _panel_payload_map(tables["selection_lock"], label="pre-outer selection lock")
    for key in selected_panels:
        selected_id = selected_panels[key].get("selected_spec_id")
        _require(
            selected_id
            == choice_panels[key].get("selected_spec_id")
            == lock_panels[key].get("selected_spec_id"),
            f"Phase-3 panel selection differs across artifacts: {key}",
        )
        _require(
            selected_id is None or str(selected_id) in SPEC_IDS,
            f"Phase-3 panel selected an ineligible specification: {key}",
        )
        selected_audit = _panel_inner_audit(
            selected_panels[key], label=f"selected-spec panel {key}"
        )
        choice_audit = _panel_inner_audit(
            choice_panels[key], label=f"calibration-choice panel {key}"
        )
        lock_audit = _panel_inner_audit(
            lock_panels[key], label=f"pre-outer selection-lock panel {key}"
        )
        _require(
            selected_audit == choice_audit == lock_audit,
            f"Phase-3 panel inner-evidence audit differs across artifacts: {key}",
        )
        if "outer_evaluation" in choice_panels[key]:
            _require(
                choice_panels[key].get("outer_evaluation")
                == selected_panels[key].get("outer_evaluation"),
                f"Phase-3 choice outer-evaluation record differs: {key}",
            )
        if selection_only:
            _require(
                selected_panels[key].get("status")
                == choice_panels[key].get("status")
                == lock_panels[key].get("status"),
                f"selection-only Phase-3 panel status differs across artifacts: {key}",
            )
        else:
            _require(
                lock_panels[key].get("status") == "selection_complete"
                and selected_id is not None
                and selected_audit.get("passed") is True,
                f"Phase-3 pre-outer selection lock is incomplete: {key}",
            )
            _require(
                selected_panels[key].get("status") == choice_panels[key].get("status"),
                f"Phase-3 outcome panel status differs across artifacts: {key}",
            )

    inner = tables["inner"]
    inner_keys = {
        (int(row.repeat), int(row.outer_fold), str(row.spec_id))
        for row in inner.itertuples(index=False)
    }
    expected_inner = {
        (repeat, fold, spec_id) for repeat in range(5) for fold in range(5) for spec_id in SPEC_IDS
    }
    _require(len(inner_keys) == len(inner), "Phase-3 inner selection contains duplicate keys")
    _require(inner_keys <= expected_inner, "Phase-3 inner selection contains impossible keys")
    if not selection_only:
        _require(inner_keys == expected_inner, "Phase-3 inner selection roster is incomplete")
    if selection_only:
        selection = decision.get("calibration_selection") or {}
        complete_keys: list[tuple[int, int]] = []
        failed_keys: list[tuple[int, int]] = []
        valid_spec_fold_combinations = 0
        status_counts: Counter[str] = Counter()
        for key, panel in selected_panels.items():
            status = str(panel.get("status") or "missing")
            selected_id = panel.get("selected_spec_id")
            audit = _panel_inner_audit(panel, label=f"selection-only panel {key}")
            valid_spec_fold_combinations += int(audit["valid_spec_fold_combinations"])
            status_counts[status] += 1
            if status == "selection_complete":
                _require(
                    selected_id is not None and audit["passed"] is True,
                    f"selection-only completed panel is inconsistent: {key}",
                )
                complete_keys.append(key)
            elif status == "selection_blocked_incomplete_inner_evidence":
                _require(
                    selected_id is None and audit["passed"] is False,
                    f"selection-only audit-blocked panel is inconsistent: {key}",
                )
                failed_keys.append(key)
            elif status == "selection_failed":
                _require(
                    selected_id is None and audit["passed"] is True,
                    f"selection-only failed panel is inconsistent: {key}",
                )
                failed_keys.append(key)
            else:
                raise TerminalReportError(f"unexpected selection-only panel status: {key}/{status}")
        failed_ids = [f"repeat_{repeat:02d}_outer_{fold:02d}" for repeat, fold in failed_keys]
        failed_statuses = {
            f"repeat_{repeat:02d}_outer_{fold:02d}": str(selected_panels[(repeat, fold)]["status"])
            for repeat, fold in failed_keys
        }
        _require(
            selection.get("status") == "incomplete_fail_closed"
            and selection.get("expected_panels_denominator") == 25
            and selection.get("attempted_panels_numerator") == 25
            and selection.get("complete_panels_numerator") == len(complete_keys)
            and selection.get("failed_panels_numerator") == len(failed_keys)
            and selection.get("complete_panel_fraction") == len(complete_keys) / 25
            and selection.get("status_counts") == dict(sorted(status_counts.items()))
            and selection.get("failed_panel_ids") == failed_ids
            and selection.get("failed_panel_statuses") == failed_statuses
            and selection.get("require_all_specs_every_inner_fold") is True
            and selection.get("survivor_selection_allowed") is False
            and selection.get("expected_spec_fold_combinations_denominator") == 300
            and selection.get("valid_spec_fold_combinations_numerator")
            == valid_spec_fold_combinations
            and selection.get("valid_spec_fold_fraction") == valid_spec_fold_combinations / 300,
            "selection-only Phase-3 panel denominators differ from artifacts",
        )
        outer = decision.get("outer_evaluation") or {}
        _require(
            outer.get("status") == "not_run_preregistered_selection_gate"
            and outer.get("expected_outer_panels_denominator") == 25
            and outer.get("evaluated_outer_panels_numerator") == 0
            and outer.get("expected_unique_oof_tutors_per_repetition_denominator") == 52
            and outer.get("evaluated_unique_oof_tutors_numerator") == 0
            and outer.get("expected_model_repeat_rows_per_policy_denominator") == 260
            and outer.get("evaluated_model_repeat_rows_per_policy_numerator") == 0
            and outer.get("cat_or_baseline_metrics_estimable") is False,
            "selection-only Phase 3 claims outer outcomes",
        )
        _require(
            decision.get("primary_policy")
            == {
                "policy": tables["selected"].get("primary_policy"),
                "status": "not_evaluated_selection_gate_failed",
                "absolute_gates_applied": False,
            }
            and decision.get("sensitivities")
            == {
                "policies": tables["selected"].get("sensitivities"),
                "status": "not_evaluated_selection_gate_failed",
                "diagnostic_only": True,
                "can_promote": False,
            },
            "selection-only Phase-3 policy disposition differs",
        )
        _require(
            decision.get("failed_conditions") == ["incomplete_calibration_spec_selection"],
            "selection-only Phase-3 failed conditions differ",
        )
        return {
            "selection_panels_expected": 25,
            "selection_panels_complete": len(complete_keys),
            "spec_fold_blocks_expected": 300,
            "spec_fold_blocks_valid": valid_spec_fold_combinations,
            "evaluated_panels": set(),
            "primary_rows": 0,
            "coverage_by_repeat": {repeat: {"rows": 0, "unique": 0} for repeat in range(5)},
        }

    policies = set(POLICY_LABELS)
    outer = tables["outer"]
    observed_outer: set[tuple[int, int, str, str]] = set()
    for row in outer.itertuples(index=False):
        key = (int(row.repeat), int(row.outer_fold), str(row.policy_id), str(row.model))
        _require(key not in observed_outer, f"duplicate Phase-3 outer key: {key}")
        _require((key[0], key[1]) in panel_models, f"impossible Phase-3 outer panel: {key}")
        _require(key[2] in policies, f"unknown Phase-3 policy: {key[2]}")
        _require(
            key[3] in panel_models[(key[0], key[1])], f"model assigned to wrong outer panel: {key}"
        )
        observed_outer.add(key)
    expected_outer = {
        (repeat, fold, policy, model)
        for (repeat, fold), panel_roster in panel_models.items()
        for policy in policies
        for model in panel_roster
    }
    _require(observed_outer <= expected_outer, "Phase-3 outer rows contain impossible keys")
    evaluated_panels: set[tuple[int, int]] = set()
    for panel, panel_roster in panel_models.items():
        expected_block = {
            (panel[0], panel[1], policy, model) for policy in policies for model in panel_roster
        }
        observed_block = observed_outer & expected_block
        _require(
            not observed_block or observed_block == expected_block,
            f"Phase-3 outer panel is only partially represented: {panel}",
        )
        if observed_block:
            evaluated_panels.add(panel)
        status = selected_panels[panel].get("status")
        outer_evaluation = selected_panels[panel].get("outer_evaluation")
        _require(
            isinstance(outer_evaluation, Mapping),
            f"Phase-3 selected panel has no outer-evaluation record: {panel}",
        )
        if panel in evaluated_panels:
            _require(
                status == "evaluation_complete"
                and outer_evaluation.get("status") == "complete"
                and outer_evaluation.get("primary_rows_complete") is True
                and int(outer_evaluation.get("n_rows", -1)) == 3 * len(panel_roster),
                f"Phase-3 evaluated panel status differs from outer rows: {panel}",
            )
        else:
            _require(
                status == "outer_fit_failed"
                and outer_evaluation.get("status") == "not_run_outer_fit_failed"
                and outer_evaluation.get("primary_rows_complete") is False,
                f"Phase-3 missing outer panel status differs from outer rows: {panel}",
            )

    prediction = tables["prediction"]
    prediction_keys = [
        (int(row.repeat), int(row.outer_fold), str(row.candidate_id), str(row.mode))
        for row in prediction.itertuples(index=False)
    ]
    expected_prediction = {
        (repeat, fold, policy, mode)
        for repeat in range(5)
        for fold in range(5)
        for policy in policies
        for mode in ("cat", "baseline")
    }
    _require(
        len(prediction_keys) == len(set(prediction_keys))
        and set(prediction_keys) == expected_prediction,
        "Phase-3 prediction composite-key roster differs",
    )
    for row in prediction.itertuples(index=False):
        panel = (int(row.repeat), int(row.outer_fold))
        if panel not in evaluated_panels:
            _require(
                int(row.n_models) == 0
                and int(row.n_cells) == 0
                and _finite_float(row.log_loss) is None,
                f"Phase-3 absent outer panel claims prediction support: {panel}",
            )

    gates = tables["gates"]
    gate_keys: list[tuple[str, int, int | None, str]] = []
    gate_scope_contract = decision.get("gate_scope_contract") or {}
    gate_scope_contract_sha256 = _canonical_hash(gate_scope_contract)
    for row in gates.itertuples(index=False):
        scope = str(row.scope)
        _require(scope in {"outer_panel", "repetition"}, f"unknown Phase-3 gate scope: {scope}")
        outer_fold = None if pd.isna(row.outer_fold) else int(row.outer_fold)
        repeat = int(row.repeat)
        policy = str(row.policy_id)
        gate_keys.append((scope, repeat, outer_fold, policy))
        primary = policy == PRIMARY_POLICY
        _require(
            policy in policies
            and str(row.policy_role) == ("primary" if primary else "sensitivity")
            and str(row.inference_unit)
            == str((gate_scope_contract.get(scope) or {}).get("inference_unit"))
            and _strict_bool(
                row.model_repeat_rows_treated_as_independent,
                label=f"Phase-3 gate independence flag {scope}/{repeat}/{outer_fold}/{policy}",
            )
            is False
            and str(row.gate_scope_contract_sha256) == gate_scope_contract_sha256,
            f"Phase-3 gate-scope provenance differs: {scope}/{repeat}/{outer_fold}/{policy}",
        )
        checks = _strict_gate_checks(
            row.gate_checks,
            scope=scope,
            primary=primary,
            label=f"Phase-3 {scope} gate {repeat}/{outer_fold}/{policy}",
        )
        if scope == "outer_panel":
            _require(outer_fold is not None, "outer-panel gate has no outer fold")
            panel = (repeat, outer_fold)
            observed_models = {
                model
                for observed_repeat, observed_fold, observed_policy, model in observed_outer
                if observed_repeat == repeat
                and observed_fold == outer_fold
                and observed_policy == policy
            }
            expected_unique = len(panel_models[panel])
            inference_complete = panel in evaluated_panels
        else:
            observed_models = {
                model
                for observed_repeat, _observed_fold, observed_policy, model in observed_outer
                if observed_repeat == repeat and observed_policy == policy
            }
            expected_unique = 52
            inference_complete = len(observed_models) == 52 and observed_models == models
        recorded_inference_complete = _strict_bool(
            row.inference_scope_complete,
            label=f"Phase-3 gate inference flag {scope}/{repeat}/{outer_fold}/{policy}",
        )
        _require(
            int(row.n_inference_rows) == len(observed_models)
            and int(row.n_unique_tutor_models) == len(observed_models)
            and int(row.expected_unique_tutor_models) == expected_unique
            and recorded_inference_complete is inference_complete,
            f"Phase-3 gate inference coverage differs: {scope}/{repeat}/{outer_fold}/{policy}",
        )
        if primary:
            recorded_gate_pass = _strict_bool(
                row.all_gates_pass,
                label=f"Phase-3 primary gate pass flag {scope}/{repeat}/{outer_fold}",
            )
            _require(
                checks["inference_scope_complete"] is inference_complete
                and recorded_gate_pass is all(checks.values())
                and str(row.gate_status) == ("pass" if all(checks.values()) else "fail"),
                f"Phase-3 primary gate status differs: {scope}/{repeat}/{outer_fold}",
            )
        else:
            _require(
                pd.isna(row.all_gates_pass)
                and str(row.gate_status) == "diagnostic_only_non_promotable",
                f"Phase-3 sensitivity gate status differs: {scope}/{repeat}/{outer_fold}/{policy}",
            )
    expected_gates = {
        ("outer_panel", repeat, fold, policy)
        for repeat in range(5)
        for fold in range(5)
        for policy in policies
    } | {("repetition", repeat, None, policy) for repeat in range(5) for policy in policies}
    _require(
        len(gate_keys) == len(set(gate_keys)) and set(gate_keys) == expected_gates,
        "Phase-3 gate composite-key roster differs",
    )

    repeat_metrics = tables["repeat_metrics"]
    repeat_keys = {
        (str(row.scope), int(row.repeat), str(row.policy_id))
        for row in repeat_metrics.itertuples(index=False)
    }
    expected_repeat_keys = {
        ("repetition", repeat, policy) for repeat in range(5) for policy in policies
    }
    _require(
        len(repeat_metrics) == 15 and repeat_keys == expected_repeat_keys,
        "Phase-3 repetition-metric roster differs",
    )
    gate_repeat_rows = gates[gates["scope"].astype(str) == "repetition"].copy()
    _require(
        list(repeat_metrics.columns) == list(gate_repeat_rows.columns),
        "Phase-3 repeat_metrics columns differ from the gate mirror",
    )
    sort_columns = ["repeat", "policy_id"]
    _require(
        repeat_metrics.sort_values(sort_columns, kind="stable")
        .reset_index(drop=True)
        .equals(gate_repeat_rows.sort_values(sort_columns, kind="stable").reset_index(drop=True)),
        "Phase-3 repeat_metrics is not an exact repetition-gate mirror",
    )

    frequency = tables["frequency"]
    _require(
        set(map(str, frequency["spec_id"])) == set(SPEC_IDS),
        "Phase-3 frequency spec roster differs",
    )
    selected_counts = Counter(
        str(panel.get("selected_spec_id"))
        for panel in selected_panels.values()
        if panel.get("selected_spec_id") is not None
    )
    _require(sum(selected_counts.values()) == 25, "Phase-3 did not lock all 25 specifications")
    for row in frequency.itertuples(index=False):
        _require(
            int(row.selected_outer_panels) == selected_counts[str(row.spec_id)]
            and float(row.selection_fraction_all_25_panels)
            == selected_counts[str(row.spec_id)] / 25,
            f"Phase-3 selection frequency differs for {row.spec_id}",
        )
    maximum = max(selected_counts.values(), default=0)
    modes = sorted(spec_id for spec_id, count in selected_counts.items() if count == maximum)
    unique_modal = modes[0] if len(modes) == 1 else None
    modal_fraction = maximum / 25
    stability_pass = unique_modal is not None and modal_fraction >= 0.80
    _require(
        decision.get("calibration_stability")
        == {
            "unique_modal_spec_id": unique_modal,
            "tied_modal_spec_ids": modes if len(modes) != 1 else [],
            "modal_count": maximum,
            "modal_fraction_all_25_panels": modal_fraction,
            "required_fraction": 0.80,
            "passed": stability_pass,
        },
        "Phase-3 calibration stability differs from selection frequencies",
    )

    coverage_by_repeat: dict[str, dict[str, Any]] = {}
    repeat_coverage_pass = True
    for repeat in range(5):
        primary_models = [
            model
            for observed_repeat, _fold, policy, model in observed_outer
            if observed_repeat == repeat and policy == PRIMARY_POLICY
        ]
        exact = (
            len(primary_models) == 52
            and len(set(primary_models)) == 52
            and set(primary_models) == models
        )
        repeat_coverage_pass = repeat_coverage_pass and exact
        coverage_by_repeat[str(repeat)] = {
            "n_primary_rows": len(primary_models),
            "n_unique_models": len(set(primary_models)),
            "all_52_models_exactly_once": exact,
        }
    all_panels_evaluated = len(evaluated_panels) == 25
    coverage_pass = repeat_coverage_pass and all_panels_evaluated
    _require(
        decision.get("coverage")
        == {
            "all_25_panels_evaluated": all_panels_evaluated,
            "all_52_models_each_repetition": coverage_pass,
            "by_repetition": coverage_by_repeat,
            "model_repeat_rows_treated_as_independent": False,
        },
        "Phase-3 decision coverage differs from outer rows",
    )
    cross_metrics = tables["cross_metrics"]
    _require(
        set(map(str, cross_metrics["policy_id"])) == policies,
        "Phase-3 cross-metric policy roster differs",
    )

    cross_model = tables["cross_model"]
    observed_cross: dict[tuple[str, str], int] = {}
    for row in cross_model.itertuples(index=False):
        key = (str(row.policy_id), str(row.model))
        _require(key not in observed_cross, "Phase-3 cross-model rows contain duplicates")
        _require(
            key[0] in policies
            and key[1] in models
            and str(row.model_family) == model_to_family[key[1]]
            and int(row.expected_repetitions) == 5
            and _strict_bool(
                row.repeat_rows_treated_as_independent,
                label=f"Phase-3 cross-model independence flag for {key}",
            )
            is False,
            f"Phase-3 cross-model provenance differs for {key}",
        )
        observed_cross[key] = int(row.n_repetitions)
    _require(len(observed_cross) == len(cross_model), "Phase-3 cross-model rows contain duplicates")
    expected_cross_keys = {(policy, model) for _, _, policy, model in observed_outer}
    _require(
        set(observed_cross) == expected_cross_keys,
        "Phase-3 cross-model roster differs from outer rows",
    )
    for key, n_repetitions in observed_cross.items():
        observed_repeats = {
            repeat for repeat, _fold, policy, model in observed_outer if (policy, model) == key
        }
        _require(
            n_repetitions == len(observed_repeats),
            f"Phase-3 cross-model repetition count differs for {key}",
        )
    for policy in policies:
        policy_cross = cross_model[cross_model["policy_id"].astype(str) == policy]
        complete = policy_cross[pd.to_numeric(policy_cross["n_repetitions"], errors="coerce").eq(5)]
        finite_recovery = np.isfinite(
            pd.to_numeric(complete["mean_theta_reference"], errors="coerce")
        ) & np.isfinite(pd.to_numeric(complete["mean_theta_cat_mwle"], errors="coerce"))
        metric = cross_metrics[cross_metrics["policy_id"].astype(str) == policy].iloc[0]
        expected_role = "primary" if policy == PRIMARY_POLICY else "sensitivity"
        complete_families = {model_to_family[str(model)] for model in complete["model"].astype(str)}
        _require(
            str(metric["policy_role"]) == expected_role
            and int(metric["n_models"]) == len(policy_cross)
            and int(metric["n_models_with_all_repetitions"]) == len(complete)
            and int(metric["n_families"]) == len(complete_families)
            and str(metric["aggregation_unit"]) == "model_after_mean_across_repetitions"
            and str(metric["bootstrap_unit"]) == "tutor_family"
            and _strict_bool(
                metric["model_repeat_rows_treated_as_independent"],
                label=f"Phase-3 cross-repeat independence flag for {policy}",
            )
            is False
            and int(metric["recovery_n"]) == int(finite_recovery.sum()),
            f"Phase-3 cross-repeat denominators differ for {policy}",
        )
        _require(
            str(metric["gate_role"]) == "diagnostic_only_cross_repeat_summary",
            f"Phase-3 cross-repeat gate role differs for {policy}",
        )
        reduction = _finite_float(metric["scenario_reduction_vs_random"])
        reduction_lower = _finite_float(metric["scenario_reduction_lower_95_ci"])
        reduction_upper = _finite_float(metric["scenario_reduction_upper_95_ci"])
        _require(
            (reduction is None and reduction_lower is None and reduction_upper is None)
            or (
                reduction is not None
                and reduction_lower is not None
                and reduction_upper is not None
                and reduction_lower <= reduction_upper
            ),
            f"Phase-3 scenario-reduction family-bootstrap interval differs for {policy}",
        )
        recovery = _finite_float(metric["recovery_correlation"])
        recovery_lower = _finite_float(metric["recovery_correlation_lower_95_ci"])
        _require(
            (recovery is None and recovery_lower is None)
            or (
                recovery is not None
                and recovery_lower is not None
                and -1.0 <= recovery <= 1.0
                and -1.0 <= recovery_lower <= 1.0
            ),
            f"Phase-3 recovery family-bootstrap interval differs for {policy}",
        )

    primary_panel_gates = gates[
        (gates["scope"].astype(str) == "outer_panel")
        & (gates["policy_id"].astype(str) == PRIMARY_POLICY)
    ].sort_values(["repeat", "outer_fold"], kind="stable")
    primary_repeat_gates = gates[
        (gates["scope"].astype(str) == "repetition")
        & (gates["policy_id"].astype(str) == PRIMARY_POLICY)
    ].sort_values(["repeat"], kind="stable")
    panel_passes = [
        _strict_bool(value, label="Phase-3 outer-panel primary gate pass flag")
        for value in primary_panel_gates["all_gates_pass"]
    ]
    repeat_passes = [
        _strict_bool(value, label="Phase-3 repetition primary gate pass flag")
        for value in primary_repeat_gates["all_gates_pass"]
    ]
    all_panel_gates = len(panel_passes) == 25 and all(panel_passes)
    all_repeat_gates = len(repeat_passes) == 5 and all(repeat_passes)
    all_inner_evidence_complete = all(
        _panel_inner_audit(panel, label=f"Phase-3 selected panel {key}")["passed"] is True
        for key, panel in selected_panels.items()
    )
    failures: list[str] = []
    if not coverage_pass:
        failures.append("incomplete_outer_coverage")
    if not all_inner_evidence_complete:
        failures.append("incomplete_inner_spec_fold_evidence")
    if not stability_pass:
        failures.append("exact_calibration_spec_modal_stability_below_80_percent")
    if not all_panel_gates:
        failures.append("primary_policy_failed_one_or_more_outer_panels")
    if not all_repeat_gates:
        failures.append("primary_policy_failed_one_or_more_pooled_repetitions")
    failed_panel_ids = [
        f"repeat_{int(row.repeat):02d}_outer_{int(row.outer_fold):02d}"
        for row in primary_panel_gates.itertuples(index=False)
        if not _strict_bool(row.all_gates_pass, label="Phase-3 outer-panel primary gate pass flag")
    ]
    failed_repetitions = [
        int(row.repeat)
        for row in primary_repeat_gates.itertuples(index=False)
        if not _strict_bool(row.all_gates_pass, label="Phase-3 repetition primary gate pass flag")
    ]
    _require(
        decision.get("failed_conditions") == failures
        and decision.get("phase3_pass") is (not failures)
        and decision.get("phase4_authorized") is (not failures)
        and decision.get("status") == ("pass" if not failures else "fail"),
        "Phase-3 decision status/failed conditions differ from evidence",
    )
    _require(
        decision.get("primary_policy")
        == {
            "policy": tables["selected"].get("primary_policy"),
            "all_outer_panels_pass": all_panel_gates,
            "all_repetitions_pass": all_repeat_gates,
            "failed_panel_ids": failed_panel_ids,
            "failed_repetitions": failed_repetitions,
        },
        "Phase-3 primary-policy decision differs from gate evidence",
    )

    if decision.get("phase3_pass") is True:
        _require(
            all(panel.get("selected_spec_id") is not None for panel in selected_panels.values()),
            "passing Phase 3 has an incomplete panel selection",
        )
        _require(
            observed_outer == expected_outer and len(outer) == EXPECTED_OUTER_ROWS,
            "passing Phase 3 lacks 780 exact outer rows",
        )
        _require(
            set(observed_cross) == {(policy, model) for policy in policies for model in models}
            and len(cross_model) == EXPECTED_CROSS_MODEL_ROWS
            and all(value == 5 for value in observed_cross.values()),
            "passing Phase 3 lacks 156 complete cross-model rows",
        )
        valid_prediction = (
            pd.to_numeric(prediction["n_cells"], errors="coerce").gt(0)
            & pd.to_numeric(prediction["log_loss"], errors="coerce").gt(0)
            & np.isfinite(pd.to_numeric(prediction["log_loss"], errors="coerce"))
        )
        _require(valid_prediction.all(), "passing Phase 3 contains invalid prediction panels")
        primary_gates = gates[gates["policy_id"].astype(str) == PRIMARY_POLICY]
        _require(
            len(primary_gates) == 30 and primary_gates["gate_status"].astype(str).eq("pass").all(),
            "passing Phase 3 contains a failed primary-policy gate",
        )
        coverage = decision.get("coverage") or {}
        _require(
            coverage.get("all_25_panels_evaluated") is True
            and coverage.get("all_52_models_each_repetition") is True,
            "passing Phase 3 coverage fields are incomplete",
        )
        for field in (
            "recovery_correlation_lower_95_ci",
            "scenario_reduction_lower_95_ci",
            "scenario_reduction_upper_95_ci",
        ):
            _require(
                np.isfinite(pd.to_numeric(cross_metrics[field], errors="coerce")).all(),
                f"passing Phase 3 lacks family-bootstrap {field}",
            )
    return {
        "selection_panels_expected": 25,
        "selection_panels_complete": 25,
        "spec_fold_blocks_expected": 300,
        "spec_fold_blocks_valid": 300 if all_inner_evidence_complete else 0,
        "evaluated_panels": evaluated_panels,
        "primary_rows": sum(
            1 for _repeat, _fold, policy, _model in observed_outer if policy == PRIMARY_POLICY
        ),
        "coverage_by_repeat": {
            int(repeat): {
                "rows": int(values["n_primary_rows"]),
                "unique": int(values["n_unique_models"]),
            }
            for repeat, values in coverage_by_repeat.items()
        },
    }


def _load_phase3_terminal(
    phase3_dir: Path,
    tracker: SourceTracker,
    *,
    config_path: Path,
    config: Mapping[str, Any],
    v4_dir: Path,
    v4_decision: Mapping[str, Any],
    v4_manifest: Mapping[str, Any],
    v4_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    decision_path = phase3_dir / "phase3_decision.json"
    manifest_path = phase3_dir / "manifest.json"
    decision = _read_json(decision_path, tracker)
    manifest = _read_json(manifest_path, tracker)
    _require(
        decision.get("schema_version") == V3_DECISION_SCHEMA, "unexpected Phase-3 decision schema"
    )
    _require(manifest.get("schema_version") == V3_RUN_SCHEMA, "unexpected Phase-3 manifest schema")
    manifest_status = str(manifest.get("status") or "")
    selection_only = manifest_status == "phase3_terminal_selection_failed"
    _require(selection_only or manifest_status == "phase3_complete", "Phase 3 is not terminal")
    if selection_only:
        _require(
            decision.get("status") == "fail_selection_incomplete"
            and decision.get("decision_scope") == "terminal_selection_only",
            "selection-only Phase-3 decision is not terminal",
        )
        _require(decision.get("phase3_pass") is False, "selection-only Phase 3 cannot pass")
        _require(
            decision.get("phase4_authorized") is False,
            "selection-only Phase 3 cannot authorize Phase 4",
        )
        _require(
            decision.get("terminal_stop")
            == {
                "stage": "pre_outer_selection_gate",
                "reason": "one_or_more_panel_selections_incomplete",
                "preregistered_fail_closed_stop": True,
                "outer_outcomes_opened": False,
                "outer_policy_evaluation_called": False,
            },
            "selection-only Phase-3 terminal-stop contract differs",
        )
        expected_outputs = PHASE3_SELECTION_OUTPUTS
    else:
        _require(decision.get("status") in {"pass", "fail"}, "Phase-3 decision is not terminal")
        expected_outputs = PHASE3_SELECTION_OUTPUTS | PHASE3_OUTCOME_OUTPUTS
    _require(
        manifest.get("phase3_decision")
        == {
            "status": decision.get("status"),
            "phase3_pass": decision.get("phase3_pass"),
            "phase4_authorized": decision.get("phase4_authorized"),
        },
        "Phase-3 manifest decision summary differs",
    )
    if selection_only:
        _require(
            manifest.get("terminal_stop") == decision.get("terminal_stop"),
            "selection-only Phase-3 terminal-stop records differ",
        )

    raw_outputs = manifest.get("outputs")
    _require(isinstance(raw_outputs, Mapping), "Phase-3 manifest outputs are missing")
    actual_outputs = set(map(str, raw_outputs))
    _require(
        actual_outputs == expected_outputs,
        "Phase-3 manifest output inventory mismatch: "
        f"missing={sorted(expected_outputs - actual_outputs)}, "
        f"unexpected={sorted(actual_outputs - expected_outputs)}",
    )
    for filename in sorted(expected_outputs):
        path = phase3_dir / filename
        record = raw_outputs[filename]
        _require(isinstance(record, Mapping), f"Phase-3 output provenance is invalid: {filename}")
        _validate_recorded_file(record, path, tracker, label=f"Phase-3 manifest output {filename}")

    inner = _read_csv(
        phase3_dir / "inner_calibration_model_results.csv",
        tracker,
        required=("repeat", "outer_fold", "spec_id")
        if selection_only
        else (
            "repeat",
            "outer_fold",
            "spec_id",
            "mean_log_loss",
            "family_cluster_se",
            "selected",
        ),
        expected_rows=None if selection_only else EXPECTED_INNER_ROWS,
    )
    if selection_only:
        _require(
            len(inner) <= EXPECTED_INNER_ROWS,
            "selection-only inner_calibration_model_results.csv has "
            f"{len(inner)} rows; maximum is {EXPECTED_INNER_ROWS}",
        )
    _require(
        not inner.duplicated(["repeat", "outer_fold", "spec_id"]).any(),
        "inner_calibration_model_results.csv contains duplicate panel/spec rows",
    )
    selected = _read_json(phase3_dir / "selected_calibration_specs.json", tracker)
    choices = _read_json(phase3_dir / "calibration_model_choices.json", tracker)
    selection_lock = _read_json(phase3_dir / "pre_outer_selection_lock.json", tracker)
    fold_assignments = _read_json(phase3_dir / "fold_assignments.json", tracker)
    mirrored_decision = _read_json(phase3_dir / "v3_decision.json", tracker)
    _require(
        selected.get("schema_version") == V3_SELECTED_SPECS_SCHEMA,
        "unexpected selected-spec handoff schema",
    )
    for artifact_name, artifact in (
        ("calibration choices", choices),
        ("pre-outer selection lock", selection_lock),
        ("fold assignments", fold_assignments),
    ):
        _require(
            artifact.get("schema_version") == V3_RUN_SCHEMA,
            f"unexpected Phase-3 {artifact_name} schema",
        )
    _require(mirrored_decision == decision, "Phase-3 decision mirrors disagree")
    _require(
        len(selected.get("panels") or []) == 25, "selected-spec handoff does not contain 25 panels"
    )
    _require(len(choices.get("panels") or []) == 25, "calibration choices do not contain 25 panels")
    _require(
        len(selection_lock.get("panels") or []) == 25,
        "pre-outer selection lock does not contain 25 panels",
    )
    split_path = _resolve_recorded(
        str(((config.get("cross_validation") or {}).get("split_manifest")) or "")
    )
    _require(
        _resolve_recorded(str(fold_assignments.get("source_split_manifest") or ""))
        == split_path.resolve(),
        "Phase-3 fold assignments point to the wrong split manifest",
    )
    _require(
        fold_assignments.get("source_split_sha256") == _sha256(split_path),
        "Phase-3 fold-assignment split hash differs",
    )
    _validate_phase3_provenance(
        phase3_dir=phase3_dir,
        config_path=config_path,
        config=config,
        v4_dir=v4_dir,
        v4_decision=v4_decision,
        v4_manifest=v4_manifest,
        v4_lock=v4_lock,
        decision=decision,
        manifest=manifest,
        selected=selected,
        selection_lock=selection_lock,
        tracker=tracker,
    )
    tables: dict[str, Any] = {
        "inner": inner,
        "selected": selected,
        "choices": choices,
        "selection_lock": selection_lock,
        "fold_assignments": fold_assignments,
        "outcomes_available": not selection_only,
    }
    if selection_only:
        tables["derived_denominators"] = _validate_phase3_rosters(
            decision, tables, selection_only=True
        )
        return decision, manifest, tables

    tables.update(
        {
            "outer": _read_csv(
                phase3_dir / "outer_oof_per_model.csv",
                tracker,
                required=(
                    "repeat",
                    "outer_fold",
                    "policy_id",
                    "model",
                    "theta_reference",
                    "theta_cat_mwle",
                    "cat_scenarios_administered",
                    "baseline_scenarios_administered",
                    "cat_eval_observed_pass_rate",
                    "cat_eval_predicted_pass_rate",
                    "baseline_eval_observed_pass_rate",
                    "baseline_eval_predicted_pass_rate",
                ),
            ),
            "prediction": _read_csv(
                phase3_dir / "disjoint_prediction_metrics.csv",
                tracker,
                required=(
                    "repeat",
                    "outer_fold",
                    "candidate_id",
                    "mode",
                    "n_models",
                    "n_cells",
                    "log_loss",
                ),
                expected_rows=EXPECTED_PREDICTION_ROWS,
            ),
            "gates": _read_csv(
                phase3_dir / "repeat_fold_gate_results.csv",
                tracker,
                required=(
                    "scope",
                    "repeat",
                    "policy_id",
                    "policy_role",
                    "recovery_slope",
                    "disjoint_pass_rate_mae",
                    "mean_scenario_count",
                    "mean_random_scenario_count",
                    "scenario_reduction_vs_random",
                    "gate_status",
                    "all_gates_pass",
                    "gate_checks",
                    "n_inference_rows",
                    "n_unique_tutor_models",
                    "expected_unique_tutor_models",
                    "inference_scope_complete",
                    "inference_unit",
                    "model_repeat_rows_treated_as_independent",
                    "gate_scope_contract_sha256",
                ),
                expected_rows=EXPECTED_GATE_ROWS,
            ),
            "repeat_metrics": _read_csv(
                phase3_dir / "repeat_metrics.csv",
                tracker,
                required=("scope", "repeat", "policy_id"),
                expected_rows=15,
            ),
            "frequency": _read_csv(
                phase3_dir / "calibration_spec_selection_frequency.csv",
                tracker,
                required=(
                    "spec_id",
                    "selected_outer_panels",
                    "selection_fraction_all_25_panels",
                ),
                expected_rows=3,
            ),
            "cross_model": _read_csv(
                phase3_dir / "cross_repeat_per_model.csv",
                tracker,
                required=(
                    "policy_id",
                    "model",
                    "model_family",
                    "n_repetitions",
                    "expected_repetitions",
                    "repeat_rows_treated_as_independent",
                    "mean_theta_reference",
                    "mean_theta_cat_mwle",
                ),
            ),
            "cross_metrics": _read_csv(
                phase3_dir / "cross_repeat_metrics.csv",
                tracker,
                required=(
                    "policy_id",
                    "policy_role",
                    "n_models",
                    "n_models_with_all_repetitions",
                    "n_families",
                    "aggregation_unit",
                    "bootstrap_unit",
                    "model_repeat_rows_treated_as_independent",
                    "recovery_n",
                    "recovery_correlation",
                    "recovery_correlation_lower_95_ci",
                    "recovery_slope",
                    "scenario_reduction_vs_random",
                    "scenario_reduction_lower_95_ci",
                    "scenario_reduction_upper_95_ci",
                    "gate_role",
                ),
                expected_rows=3,
            ),
        }
    )
    outer = tables["outer"]
    cross_model = tables["cross_model"]
    _require(
        len(outer) <= EXPECTED_OUTER_ROWS,
        f"outer_oof_per_model.csv has {len(outer)} rows; maximum is {EXPECTED_OUTER_ROWS}",
    )
    _require(
        not outer.duplicated(["repeat", "policy_id", "model"]).any(),
        "outer_oof_per_model.csv contains duplicate repeat/policy/model rows",
    )
    _require(
        len(cross_model) <= EXPECTED_CROSS_MODEL_ROWS,
        "cross_repeat_per_model.csv has "
        f"{len(cross_model)} rows; maximum is {EXPECTED_CROSS_MODEL_ROWS}",
    )
    _require(
        not cross_model.duplicated(["policy_id", "model"]).any(),
        "cross_repeat_per_model.csv contains duplicate policy/model rows",
    )
    tables["derived_denominators"] = _validate_phase3_rosters(
        decision, tables, selection_only=False
    )
    return decision, manifest, tables


def load_terminal_inputs(*, config_path: Path, v4_dir: Path, phase3_dir: Path) -> TerminalInputs:
    tracker = SourceTracker()
    config_path = config_path.resolve()
    v4_dir = v4_dir.resolve()
    config = _read_json(config_path, tracker)
    _require(config.get("schema_version") == V3_CONFIG_SCHEMA, "unexpected V3 config schema")
    decision, manifest = _load_v4_terminal(v4_dir, tracker)
    numerical = _load_numerical_tables(v4_dir, decision, tracker)

    if decision.get("passed") is not True:
        _require(
            decision.get("status") == "blocked_numerical_followup", "invalid blocked V4 decision"
        )
        _require(
            not (v4_dir / "numerical_followup_lock.json").exists(), "blocked V4 retained a lock"
        )
        _require(
            not (v4_dir / "numerical_followup_lock.sha256").exists(),
            "blocked V4 retained a lock hash",
        )
        # Critical isolation rule: do not stat, list, parse, or hash phase3_dir.
        return TerminalInputs(
            terminal_state="blocked_after_v4_numerical_followup",
            config_path=config_path,
            v4_dir=v4_dir,
            phase3_dir=None,
            config=config,
            v4_decision=decision,
            v4_manifest=manifest,
            v4_lock=None,
            phase3_decision=None,
            phase3_manifest=None,
            numerical=numerical,
            phase3=None,
            tracker=tracker,
        )

    _require(decision.get("status") == "complete_pass", "invalid passing V4 decision")
    lock = _load_v4_lock(v4_dir, decision, manifest, tracker)
    # Resolving the CAT directory is deliberately deferred until the V4 lock
    # authorizes Phase 3.  A blocked V4 report must not touch the CAT path.
    phase3_dir = phase3_dir.resolve()
    phase3_decision, phase3_manifest, phase3 = _load_phase3_terminal(
        phase3_dir,
        tracker,
        config_path=config_path,
        config=config,
        v4_dir=v4_dir,
        v4_decision=decision,
        v4_manifest=manifest,
        v4_lock=lock,
    )
    passed = phase3_decision.get("phase3_pass") is True
    _require(passed == (phase3_decision.get("status") == "pass"), "Phase-3 pass fields disagree")
    _require(
        bool(phase3_decision.get("phase4_authorized")) == passed,
        "Phase-3 authorization field disagrees with its decision",
    )
    if not phase3["outcomes_available"]:
        state = "blocked_after_phase3_selection"
    else:
        state = "phase3_pass_phase4_pending" if passed else "blocked_after_phase3_validation"
    return TerminalInputs(
        terminal_state=state,
        config_path=config_path,
        v4_dir=v4_dir,
        phase3_dir=phase3_dir,
        config=config,
        v4_decision=decision,
        v4_manifest=manifest,
        v4_lock=lock,
        phase3_decision=phase3_decision,
        phase3_manifest=phase3_manifest,
        numerical=numerical,
        phase3=phase3,
        tracker=tracker,
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fig.savefig(temporary, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    temporary.replace(path)
    _require(path.stat().st_size > 2_000, f"rendered figure is unexpectedly small: {path}")


def _terminal_banner(inputs: TerminalInputs) -> str:
    if inputs.terminal_state == "blocked_after_v4_numerical_followup":
        return "V4 BLOCKED — CAT NOT RUN"
    if inputs.terminal_state == "blocked_after_phase3_selection":
        return "PHASE 3 SELECTION FAILED — CAT NOT RUN"
    if inputs.terminal_state == "blocked_after_phase3_validation":
        return "PHASE 3 FAILED VALIDATION — INCOMPLETE"
    return "PHASE 3 PASSED — PHASE 4 STILL PENDING"


def _plot_item_parameters(inputs: TerminalInputs, path: Path) -> None:
    frames = inputs.numerical["item_frames"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    for column, spec_id in enumerate(SPEC_IDS):
        frame = frames[spec_id]
        axes[0, column].hist(frame["b"].astype(float), bins=30, color=COLORS["orange"], alpha=0.85)
        axes[0, column].set(title=SPEC_LABELS[spec_id], xlabel="difficulty (b)", ylabel="criteria")
        axes[1, column].hist(
            frame["a_instruction_following"].astype(float),
            bins=30,
            color=COLORS["blue"],
            alpha=0.85,
        )
        axes[1, column].set(xlabel="discrimination (a)", ylabel="criteria")
        axes[0, column].text(
            0.98,
            0.95,
            f"N={len(frame):,}",
            transform=axes[0, column].transAxes,
            ha="right",
            va="top",
        )
    fig.suptitle(
        "V4 candidate item-parameter distributions\n"
        "Numerical candidates only; no calibration specification selected",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, path)


def _plot_fit_validity(inputs: TerminalInputs, path: Path) -> None:
    manifests = inputs.numerical["fit_manifests"]
    starts = inputs.numerical["start_records"]
    fit_valid = Counter()
    fit_total = Counter()
    iterations: dict[str, list[float]] = {spec: [] for spec in SPEC_IDS}
    for row in manifests:
        spec = str(row.get("spec_id"))
        fit_total[spec] += 1
        fit_valid[spec] += int(bool((row.get("fit_validity") or {}).get("fit_valid")))
        if row.get("n_iter") is not None:
            iterations[spec].append(float(row["n_iter"]))
    start_valid = Counter()
    start_total = Counter()
    for row in starts:
        spec = str(row.get("spec_id"))
        start_total[spec] += 1
        start_valid[spec] += int(row.get("selection_valid") is True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    x = np.arange(len(SPEC_IDS))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        [fit_valid[s] for s in SPEC_IDS],
        width,
        label="valid fits",
        color=COLORS["blue"],
    )
    axes[0].bar(
        x + width / 2,
        [start_valid[s] for s in SPEC_IDS],
        width,
        label="valid 801-start gates",
        color=COLORS["green"],
    )
    for idx, spec in enumerate(SPEC_IDS):
        axes[0].text(
            idx - width / 2,
            fit_valid[spec] + 0.2,
            f"{fit_valid[spec]}/{fit_total[spec]}",
            ha="center",
        )
        axes[0].text(
            idx + width / 2,
            start_valid[spec] + 0.2,
            f"{start_valid[spec]}/{start_total[spec]}",
            ha="center",
        )
    axes[0].set_xticks(x, [SPEC_LABELS[s] for s in SPEC_IDS], rotation=15, ha="right")
    axes[0].set(ylabel="count", title="Returned-iterate and start validity")
    axes[0].legend()
    axes[1].boxplot(
        [iterations[s] for s in SPEC_IDS], tick_labels=[SPEC_LABELS[s] for s in SPEC_IDS]
    )
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set(ylabel="M-steps", title="Fit convergence iterations (18 fits/spec)")
    fig.suptitle(
        _terminal_banner(inputs),
        color=COLORS["red"] if "BLOCKED" in _terminal_banner(inputs) else COLORS["dark"],
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, path)


def _plot_parameter_equivalence(inputs: TerminalInputs, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    markers = ["o", "s", "^"]
    parameter_values: list[float] = []
    export_values: list[float] = []
    for marker, spec_id in zip(markers, SPEC_IDS, strict=True):
        frame = inputs.numerical["refit_parameters"][spec_id]
        x = np.arange(len(frame))
        b_values = frame["b_spearman"].astype(float)
        axes[0].plot(x, b_values, marker=marker, label=SPEC_LABELS[spec_id])
        parameter_values.extend(b_values.tolist())
        a_values = pd.to_numeric(frame.get("a_spearman"), errors="coerce")
        if a_values.notna().any():
            axes[0].plot(x, a_values, marker=marker, linestyle="--", alpha=0.75)
            parameter_values.extend(a_values.dropna().astype(float).tolist())
        agreement = frame["exportability_agreement"].astype(float)
        axes[1].plot(
            x,
            agreement,
            marker=marker,
            label=SPEC_LABELS[spec_id],
        )
        export_values.extend(agreement.tolist())
    first = inputs.numerical["refit_parameters"][SPEC_IDS[0]]
    parameter_gate = float(first["minimum_parameter_spearman"].iloc[0])
    export_gate = float(first["minimum_exportability_agreement"].iloc[0])
    parameter_values.append(parameter_gate)
    export_values.append(export_gate)
    axes[0].axhline(
        parameter_gate,
        color=COLORS["red"],
        linestyle=":",
        label="gate",
    )
    axes[1].axhline(
        export_gate,
        color=COLORS["red"],
        linestyle=":",
        label="gate",
    )
    for ax in axes:
        ax.set_xticks(np.arange(6), ["full", "o0", "o1", "o2", "o3", "o4"])
        ax.grid(alpha=0.2)

    def limits(values: Sequence[float], *, hard_lower: float) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        minimum = float(finite.min()) if finite.size else hard_lower
        padding = max(0.01, 0.05 * max(1.0 - minimum, 0.05))
        return max(hard_lower, min(0.95, minimum - padding)), 1.005

    axes[0].set_ylim(*limits(parameter_values, hard_lower=-1.0))
    axes[1].set_ylim(*limits(export_values, hard_lower=0.0))
    axes[0].set(title="401 vs 801 parameter rank agreement", ylabel="Spearman r")
    axes[1].set(title="401 vs 801 exportability agreement", ylabel="fraction equal")
    axes[1].legend(loc="lower left")
    fig.suptitle("Refitted-bank parameter equivalence (six scopes per specification)", fontsize=13)
    fig.tight_layout()
    _save(fig, path)


def _plot_prediction_equivalence(inputs: TerminalInputs, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for axis, metric in zip(axes, ("log_loss", "brier"), strict=True):
        for index, spec_id in enumerate(SPEC_IDS):
            gate = inputs.numerical["refit_gates"][spec_id]["heldout_common_cell_gate"]["metrics"][
                metric
            ]
            point = float(gate["pooled_shift_801_minus_401"])
            low, high = map(float, gate["family_cluster_bootstrap_ci_95"])
            axis.errorbar(
                index,
                point,
                yerr=[[point - low], [high - point]],
                fmt="o",
                capsize=4,
                color=COLORS["blue"],
            )
            axis.text(
                index,
                high,
                f" N={inputs.numerical['refit_gates'][spec_id]['heldout_common_cell_gate']['n_common_cells']:,}",
                rotation=90,
                va="bottom",
                fontsize=7,
            )
        margin = float(
            inputs.numerical["refit_gates"][SPEC_IDS[0]]["heldout_common_cell_gate"]["metrics"][
                metric
            ]["equivalence_margin"]
        )
        axis.axhspan(-margin, margin, color=COLORS["green"], alpha=0.12, label="equivalence region")
        axis.axhline(0, color=COLORS["gray"], linewidth=1)
        axis.set_xticks(range(3), [SPEC_LABELS[s] for s in SPEC_IDS], rotation=15, ha="right")
        axis.set(title=f"Held-out {metric.replace('_', ' ')} shift", ylabel="801 minus 401")
        axis.legend(loc="lower left")
    fig.suptitle(
        "Common-cell held-out prediction equivalence (family-clustered 95% CIs)", fontsize=13
    )
    fig.tight_layout()
    _save(fig, path)


def _plot_refit_theta(inputs: TerminalInputs, path: Path) -> None:
    x = np.arange(len(SPEC_IDS))
    median: list[float] = []
    p95: list[float] = []
    maximum: list[float] = []
    median_limit = None
    p95_limit = None
    for spec_id in SPEC_IDS:
        gate = inputs.numerical["refit_gates"][spec_id]["common_support_refit_theta_gate"]
        median.append(float(gate["median_absolute_theta_shift"]))
        p95.append(float(gate["p95_absolute_theta_shift"]))
        maximum.append(float(gate["maximum_absolute_theta_shift_diagnostic_only"]))
        median_limit = float(gate["thresholds"]["maximum_median_absolute_theta_shift"])
        p95_limit = float(gate["thresholds"]["maximum_p95_absolute_theta_shift"])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(x, median, "o-", label="median |delta theta|", color=COLORS["blue"])
    ax.plot(x, p95, "s-", label="p95 |delta theta|", color=COLORS["orange"])
    ax.scatter(
        x, maximum, marker="x", s=70, label="maximum (diagnostic only)", color=COLORS["purple"]
    )
    ax.axhline(
        median_limit, linestyle=":", color=COLORS["blue"], label=f"median gate={median_limit:g}"
    )
    ax.axhline(p95_limit, linestyle=":", color=COLORS["orange"], label=f"p95 gate={p95_limit:g}")
    ax.set_xticks(x, [SPEC_LABELS[s] for s in SPEC_IDS])
    positive = [value for value in [*median, *p95, *maximum] if value > 0]
    if positive and max(positive) / min(positive) > 50:
        ax.set_yscale("log")
    ax.set(ylabel="absolute theta shift", title="401-vs-801 refitted-bank theta stability")
    ax.legend(ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, path)


def _plot_fixed_bank(inputs: TerminalInputs, path: Path) -> None:
    _require(bool(inputs.numerical["fixed_gates"]), "fixed-bank plot requested without evidence")
    shifts: dict[str, list[float]] = {"density": [], "bound": []}
    tails: list[float] = []
    shift_limit = None
    tail_limit = None
    for spec_id in SPEC_IDS:
        gate = inputs.numerical["fixed_gates"][spec_id]["numerical_scoring_gate"]
        comparisons = gate["comparisons"]
        density = comparisons["bound8_801_vs_bound8_1601"]
        bound = comparisons["bound8_801_vs_bound10_1001"]
        shifts["density"].append(float(density["maximum_absolute_theta_shift"]))
        shifts["bound"].append(float(bound["maximum_absolute_theta_shift"]))
        tails.append(float(gate["tail_mass"]["maximum_posterior_tail_mass"]))
        shift_limit = float(density["threshold"])
        tail_limit = float(gate["tail_mass"]["threshold"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(3)
    axes[0].plot(x, shifts["density"], "o-", label="801 vs 1601 nodes")
    axes[0].plot(x, shifts["bound"], "s-", label="bound 8 vs bound 10")
    axes[0].axhline(shift_limit, linestyle=":", color=COLORS["red"], label="gate")
    axes[0].set(ylabel="maximum |delta theta|", title="Same-bank EAP resolution/bound checks")
    axes[0].legend()
    axes[1].plot(x, tails, "o-", color=COLORS["purple"])
    axes[1].axhline(tail_limit, linestyle=":", color=COLORS["red"], label="gate")
    axes[1].set(ylabel="maximum posterior tail mass", title="Posterior tail check")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(x, [SPEC_LABELS[s] for s in SPEC_IDS], rotation=15, ha="right")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    fig.suptitle("Fixed fitted-bank numerical scoring stability", fontsize=13)
    fig.tight_layout()
    _save(fig, path)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _plot_calibration_selection(inputs: TerminalInputs, path: Path) -> None:
    assert inputs.phase3 is not None
    inner = inputs.phase3["inner"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    finite_counts = pd.Series(0, index=list(SPEC_IDS), dtype=int)
    if "mean_log_loss" in inner:
        loss = pd.to_numeric(inner["mean_log_loss"], errors="coerce")
        means = loss.groupby(inner["spec_id"], sort=False).mean().reindex(SPEC_IDS)
        errors = loss.groupby(inner["spec_id"], sort=False).sem().reindex(SPEC_IDS).fillna(0)
        finite_counts = (
            loss.groupby(inner["spec_id"], sort=False)
            .count()
            .reindex(SPEC_IDS)
            .fillna(0)
            .astype(int)
        )
        axes[0].errorbar(range(3), means, yerr=errors, fmt="o", capsize=4, color=COLORS["blue"])
    else:
        axes[0].text(
            0.5,
            0.5,
            "Aggregate inner loss unavailable\n(selection evidence incomplete)",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            color=COLORS["red"],
        )
    axes[0].set_xticks(range(3), [SPEC_LABELS[s] for s in SPEC_IDS], rotation=15, ha="right")
    axes[0].set(
        ylabel="mean inner disjoint log loss",
        title=(
            "Inner-only calibration evidence\n"
            f"finite panels {list(finite_counts)}/25; error bars are SE"
        ),
    )
    panels = inputs.phase3["selected"].get("panels") or []
    counts = Counter(
        str(panel.get("selected_spec_id"))
        for panel in panels
        if panel.get("selected_spec_id") is not None
    )
    selected = pd.Series([counts[spec_id] for spec_id in SPEC_IDS], index=SPEC_IDS, dtype=float)
    axes[1].bar(range(3), selected, color=[COLORS["blue"], COLORS["green"], COLORS["purple"]])
    axes[1].axhline(20, color=COLORS["red"], linestyle=":", label="80% modal lock (20/25)")
    axes[1].set_xticks(range(3), [SPEC_LABELS[s] for s in SPEC_IDS], rotation=15, ha="right")
    axes[1].set(
        ylabel="selected outer panels (of 25)",
        title=f"One-SE simplicity-rule selections ({sum(counts.values())}/25 locked)",
    )
    axes[1].legend()
    fig.suptitle(
        "Calibration specification selection — outer outcomes never used\n"
        f"{_terminal_banner(inputs)}",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, path)


def _parse_json_cell(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_text(value: Any, *, digits: int = 3) -> str:
    result = _finite_float(value)
    return f"{result:.{digits}f}" if result is not None else "N/A"


def _plot_cat_gates(inputs: TerminalInputs, path: Path) -> None:
    assert inputs.phase3 is not None
    gates = inputs.phase3["gates"]
    primary = gates[(gates["policy_id"] == PRIMARY_POLICY) & (gates["scope"] == "outer_panel")]
    counts: Counter[str] = Counter()
    for value in primary["gate_checks"]:
        for name, passed in _parse_json_cell(value).items():
            counts[name] += int(bool(passed))
    names = sorted(counts)
    fig, ax = plt.subplots(figsize=(12.5, max(4.8, 0.35 * len(names) + 1.8)))
    values = [counts[name] for name in names]
    ax.barh(
        range(len(names)),
        values,
        color=[COLORS["green"] if value == 25 else COLORS["orange"] for value in values],
    )
    ax.axvline(25, color=COLORS["dark"], linestyle=":")
    ax.set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
    ax.set(
        xlabel="outer panels passing gate (of 25)",
        title=f"Frozen primary-policy gate outcomes\n{_terminal_banner(inputs)}",
    )
    for index, value in enumerate(values):
        ax.text(value + 0.15, index, f"{value}/25", va="center")
    ax.set_xlim(0, 27)
    fig.tight_layout()
    _save(fig, path)


def _plot_policy_sensitivities(inputs: TerminalInputs, path: Path) -> None:
    assert inputs.phase3 is not None
    gates = inputs.phase3["gates"]
    repeat = gates[gates["scope"] == "repetition"]
    cross = inputs.phase3["cross_metrics"].set_index("policy_id")
    metrics = (
        ("scenario_reduction_vs_random", "scenario reduction", 0.50),
        ("recovery_slope", "theta recovery slope", 1.0),
        ("disjoint_pass_rate_mae", "pass-rate MAE", 0.07),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    policies = list(POLICY_LABELS)
    for axis, (column, label, reference) in zip(axes, metrics, strict=True):
        if column == "scenario_reduction_vs_random":
            ordered = cross.reindex(policies)
            means = pd.to_numeric(ordered[column], errors="coerce")
            lower = pd.to_numeric(ordered["scenario_reduction_lower_95_ci"], errors="coerce")
            upper = pd.to_numeric(ordered["scenario_reduction_upper_95_ci"], errors="coerce")
            errors: Any = np.vstack(
                [
                    (means - lower).clip(lower=0).fillna(0).to_numpy(),
                    (upper - means).clip(lower=0).fillna(0).to_numpy(),
                ]
            )
            counts = (
                pd.to_numeric(ordered["n_models_with_all_repetitions"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            title = f"{label}\nfamily-bootstrap 95% CI; complete models {list(counts)}/52"
        else:
            numeric = pd.to_numeric(repeat[column], errors="coerce")
            grouped = numeric.groupby(repeat["policy_id"])
            means = grouped.mean().reindex(policies)
            errors = grouped.sem().reindex(policies).fillna(0)
            counts = grouped.count().reindex(policies).fillna(0).astype(int)
            title = f"{label}\nmean +/- SE; finite repetitions {list(counts)}/5"
        axis.bar(
            range(3),
            means,
            yerr=errors,
            capsize=3,
            color=[COLORS["blue"], COLORS["green"], COLORS["purple"]],
        )
        axis.axhline(reference, color=COLORS["red"], linestyle=":")
        axis.set_xticks(range(3), ["primary", "floor 12", "D-opt"], rotation=15)
        axis.set(
            ylabel=label,
            title=title,
        )
    fig.suptitle(
        "Prespecified one-change sensitivity panel (sensitivities cannot replace primary)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, path)


def _plot_recovery(inputs: TerminalInputs, path: Path) -> None:
    assert inputs.phase3 is not None
    frame = inputs.phase3["cross_model"]
    frame = frame[frame["policy_id"] == PRIMARY_POLICY]
    complete = pd.to_numeric(frame["n_repetitions"], errors="coerce").eq(5)
    frame = frame[complete]
    metrics = inputs.phase3["cross_metrics"]
    metric = metrics[metrics["policy_id"] == PRIMARY_POLICY].iloc[0]
    x = pd.to_numeric(frame["mean_theta_reference"], errors="coerce").to_numpy()
    y = pd.to_numeric(frame["mean_theta_cat_mwle"], errors="coerce").to_numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    complete_recovery = int(finite.sum()) == 52
    recovery_r = (
        float(np.corrcoef(x[finite], y[finite])[0, 1])
        if complete_recovery and float(np.std(x[finite])) > 0 and float(np.std(y[finite])) > 0
        else math.nan
    )
    recovery_slope = (
        float(np.polyfit(x[finite], y[finite], 1)[0])
        if complete_recovery and float(np.std(x[finite])) > 0
        else math.nan
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(x[finite], y[finite], color=COLORS["blue"], alpha=0.75)
    if finite.any():
        low = float(min(x[finite].min(), y[finite].min()))
        high = float(max(x[finite].max(), y[finite].max()))
        ax.plot([low, high], [low, high], color=COLORS["gray"], linestyle="--", label="ideal")
    ax.set(
        xlabel="full-bank reference theta",
        ylabel="CAT MWLE theta",
        title=(
            "Held-out ability recovery "
            f"({int(finite.sum())}/52 models with all 5 repetitions)\n"
            f"{_terminal_banner(inputs)}"
        ),
    )
    ax.text(
        0.03,
        0.97,
        (
            f"r={_metric_text(recovery_r)}\nslope={_metric_text(recovery_slope)}\n"
            "family-bootstrap lower 95% r="
            f"{_metric_text(metric['recovery_correlation_lower_95_ci']) if complete_recovery else 'N/A'}"
        ),
        transform=ax.transAxes,
        va="top",
    )
    if finite.any():
        ax.legend()
    fig.tight_layout()
    _save(fig, path)


def _plot_pass_rate(inputs: TerminalInputs, path: Path) -> None:
    assert inputs.phase3 is not None
    frame = inputs.phase3["outer"]
    frame = frame[frame["policy_id"] == PRIMARY_POLICY]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharex=True, sharey=True)
    for axis, prefix, title in (
        (axes[0], "cat", "CAT"),
        (axes[1], "baseline", "paired random baseline"),
    ):
        x = pd.to_numeric(frame[f"{prefix}_eval_observed_pass_rate"], errors="coerce")
        y = pd.to_numeric(frame[f"{prefix}_eval_predicted_pass_rate"], errors="coerce")
        mask = (
            x.notna()
            & y.notna()
            & np.isfinite(x)
            & np.isfinite(y)
            & x.between(0, 1)
            & y.between(0, 1)
        )
        axis.scatter(
            x[mask],
            y[mask],
            alpha=0.35,
            s=20,
            color=COLORS["blue"] if prefix == "cat" else COLORS["orange"],
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"])
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="observed pass rate",
            ylabel="predicted pass rate",
            title=f"{title} (valid rows={int(mask.sum())}/{EXPECTED_PRIMARY_MODEL_REPEAT_ROWS})",
        )
    fig.suptitle(
        f"Disjoint held-out pass-rate calibration\n{_terminal_banner(inputs)}", fontsize=13
    )
    fig.tight_layout()
    _save(fig, path)


def _plot_cat_vs_random(inputs: TerminalInputs, path: Path) -> None:
    assert inputs.phase3 is not None
    outer = inputs.phase3["outer"]
    outer = outer[outer["policy_id"] == PRIMARY_POLICY]
    prediction = inputs.phase3["prediction"]
    prediction = prediction[prediction["candidate_id"] == PRIMARY_POLICY]
    cat_length = pd.to_numeric(outer["cat_scenarios_administered"], errors="coerce")
    baseline_length = pd.to_numeric(outer["baseline_scenarios_administered"], errors="coerce")
    paired = (
        np.isfinite(cat_length)
        & np.isfinite(baseline_length)
        & cat_length.gt(0)
        & baseline_length.gt(0)
    )
    lengths = [
        cat_length[paired].mean(),
        baseline_length[paired].mean(),
    ]
    prediction = prediction.assign(
        _numeric_log_loss=pd.to_numeric(prediction["log_loss"], errors="coerce"),
        _numeric_n_cells=pd.to_numeric(prediction["n_cells"], errors="coerce"),
    )
    valid_prediction = (
        np.isfinite(prediction["_numeric_log_loss"])
        & np.isfinite(prediction["_numeric_n_cells"])
        & prediction["_numeric_n_cells"].gt(0)
        & prediction["_numeric_log_loss"].gt(0)
    )
    valid_panels = prediction[valid_prediction]
    loss = valid_panels.groupby("mode")["_numeric_log_loss"].mean()
    losses = [float(loss.get("cat", np.nan)), float(loss.get("baseline", np.nan))]
    panel_counts = valid_panels.groupby("mode").size()
    cat_panels = int(panel_counts.get("cat", 0))
    baseline_panels = int(panel_counts.get("baseline", 0))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    labels = ["CAT", "random"]
    colors = [COLORS["blue"], COLORS["orange"]]
    cross_metrics = inputs.phase3["cross_metrics"]
    primary_metric = cross_metrics[cross_metrics["policy_id"] == PRIMARY_POLICY].iloc[0]
    complete_models = int(primary_metric["n_models_with_all_repetitions"])
    axes[0].bar(labels, lengths, color=colors)
    axes[0].set(
        ylabel="mean scenarios",
        title=(
            "Test length "
            f"({int(paired.sum())}/{EXPECTED_PRIMARY_MODEL_REPEAT_ROWS} paired model-repeats)"
        ),
    )
    axes[0].text(
        0.5,
        0.97,
        (
            f"reduction={_metric_text(primary_metric['scenario_reduction_vs_random'])}\n"
            f"family-bootstrap 95% CI (complete models {complete_models}/52) "
            f"[{_metric_text(primary_metric['scenario_reduction_lower_95_ci'])}, "
            f"{_metric_text(primary_metric['scenario_reduction_upper_95_ci'])}]"
        ),
        transform=axes[0].transAxes,
        ha="center",
        va="top",
    )
    axes[1].bar(labels, losses, color=colors)
    axes[1].set(
        ylabel="mean disjoint log loss",
        title=(
            "Held-out prediction performance\n"
            f"(CAT {cat_panels}/{EXPECTED_PRIMARY_PANELS_PER_ARM}; "
            f"random {baseline_panels}/{EXPECTED_PRIMARY_PANELS_PER_ARM} panels)"
        ),
    )
    fig.suptitle(f"CAT versus paired random baseline\n{_terminal_banner(inputs)}", fontsize=13)
    fig.tight_layout()
    _save(fig, path)


def _plot_coverage(inputs: TerminalInputs, path: Path) -> None:
    assert inputs.phase3 is not None
    derived = inputs.phase3.get("derived_denominators") or {}
    by_repeat = derived.get("coverage_by_repeat") or {}
    values = [int((by_repeat.get(index) or {}).get("unique") or 0) for index in range(5)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].bar(
        range(5),
        values,
        color=[COLORS["green"] if value == 52 else COLORS["orange"] for value in values],
    )
    axes[0].axhline(52, color=COLORS["dark"], linestyle=":")
    axes[0].set_xticks(range(5), [f"repeat {index}" for index in range(5)])
    axes[0].set(ylabel="unique primary-policy models (of 52)", title="Outer model coverage")
    phase3 = inputs.phase3 or {}
    choices = phase3.get("choices") or {}
    statuses = Counter(str(panel.get("status")) for panel in choices.get("panels") or [])
    names = sorted(statuses)
    axes[1].bar(names, [statuses[name] for name in names], color=COLORS["blue"])
    axes[1].axhline(25, color=COLORS["dark"], linestyle=":")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(ylabel="outer panels (of 25)", title="Calibration-selection panel status")
    fig.suptitle(f"Complete-versus-partial coverage\n{_terminal_banner(inputs)}", fontsize=13)
    fig.tight_layout()
    _save(fig, path)


def _figure_plan(inputs: TerminalInputs) -> list[tuple[str, str, str]]:
    plan = [
        (
            "01_candidate_item_parameter_distributions.png",
            "Candidate item-parameter distributions",
            "Full-cohort 401-node numerical-candidate banks; no calibration family was selected in V4.",
        ),
        (
            "02_fit_convergence_and_start_agreement.png",
            "Fit convergence and start agreement",
            "Validity counts cover all 54 fresh fits and all 18 cold-versus-continuation start comparisons.",
        ),
        (
            "03_refit_parameter_equivalence.png",
            "Refitted-bank parameter equivalence",
            "Difficulty/discrimination rank and exportability agreement for 401 versus selected 801 fits across six scopes.",
        ),
        (
            "04_heldout_prediction_equivalence.png",
            "Held-out common-cell prediction equivalence",
            "Family-clustered 95% intervals compare held-out log loss and Brier score on identical cells.",
        ),
        (
            "05_refit_theta_stability.png",
            "Refitted-bank theta stability",
            "Median and p95 shifts are gates; maximum shift is visibly diagnostic only.",
        ),
    ]
    if inputs.numerical["fixed_gates"]:
        plan.append(
            (
                "06_fixed_bank_eap_stability.png",
                "Fixed-bank EAP and tail stability",
                "Pure scoring-resolution and bound checks hold each 401-node fitted bank fixed.",
            )
        )
    if inputs.phase3 is not None:
        plan.append(
            (
                "07_calibration_spec_inner_selection.png",
                "Inner-only calibration specification selection",
                "Disjoint inner log loss and the one-SE simplicity rule; outer CAT outcomes were not used.",
            )
        )
        if inputs.phase3["outcomes_available"]:
            incomplete = inputs.terminal_state == "blocked_after_phase3_validation"
            suffix = "_INCOMPLETE" if incomplete else ""
            plan.extend(
                [
                    (
                        "08_cat_gate_outcomes.png",
                        "Frozen primary CAT gate outcomes",
                        "Panel counts use all 25 outer panels; sensitivities never affect the decision.",
                    ),
                    (
                        "09_prespecified_policy_sensitivities.png",
                        "Prespecified floor and selector sensitivities",
                        "The two one-change sensitivities are diagnostic and cannot replace the primary policy.",
                    ),
                    (
                        f"10_oos_ability_recovery{suffix}.png",
                        "Held-out ability recovery",
                        "Five repeated OOF estimates are averaged per model before the 52-model recovery summary.",
                    ),
                    (
                        f"11_oos_pass_rate_calibration{suffix}.png",
                        "Held-out predicted-versus-observed pass rates",
                        "Disjoint evaluation outcomes for 260 primary-policy model-repeat rows.",
                    ),
                    (
                        f"12_cat_vs_random{suffix}.png",
                        "CAT versus paired random",
                        "Primary-policy test length and held-out prediction performance, with visible denominators.",
                    ),
                    (
                        f"13_outer_coverage{suffix}.png",
                        "Outer coverage",
                        "Shows all five repetitions and all 25 calibration-selection panels.",
                    ),
                ]
            )
    return plan


def _render_figures(inputs: TerminalInputs, figures_dir: Path) -> list[Path]:
    plan = _figure_plan(inputs)
    by_name = {name: figures_dir / name for name, _title, _description in plan}
    _plot_item_parameters(inputs, by_name["01_candidate_item_parameter_distributions.png"])
    _plot_fit_validity(inputs, by_name["02_fit_convergence_and_start_agreement.png"])
    _plot_parameter_equivalence(inputs, by_name["03_refit_parameter_equivalence.png"])
    _plot_prediction_equivalence(inputs, by_name["04_heldout_prediction_equivalence.png"])
    _plot_refit_theta(inputs, by_name["05_refit_theta_stability.png"])
    if "06_fixed_bank_eap_stability.png" in by_name:
        _plot_fixed_bank(inputs, by_name["06_fixed_bank_eap_stability.png"])
    if inputs.phase3 is not None:
        _plot_calibration_selection(inputs, by_name["07_calibration_spec_inner_selection.png"])
    if inputs.phase3 is not None and inputs.phase3["outcomes_available"]:
        _plot_cat_gates(inputs, by_name["08_cat_gate_outcomes.png"])
        _plot_policy_sensitivities(inputs, by_name["09_prespecified_policy_sensitivities.png"])
        names = sorted(name for name in by_name if name.startswith("10_oos_ability_recovery"))
        _plot_recovery(inputs, by_name[names[0]])
        names = sorted(name for name in by_name if name.startswith("11_oos_pass_rate_calibration"))
        _plot_pass_rate(inputs, by_name[names[0]])
        names = sorted(name for name in by_name if name.startswith("12_cat_vs_random"))
        _plot_cat_vs_random(inputs, by_name[names[0]])
        names = sorted(name for name in by_name if name.startswith("13_outer_coverage"))
        _plot_coverage(inputs, by_name[names[0]])
    paths = [by_name[name] for name, _title, _description in plan]
    _require(all(path.is_file() for path in paths), "not every planned figure was rendered")
    return paths


def _gate_status(inputs: TerminalInputs) -> dict[str, Any]:
    v4_pass = inputs.v4_decision.get("passed") is True
    phase3_pass = bool(inputs.phase3_decision and inputs.phase3_decision.get("phase3_pass") is True)
    outcomes_available = bool(inputs.phase3 and inputs.phase3.get("outcomes_available"))
    if not v4_pass:
        phase3_status = "not_run_v4_no_lock"
    elif not outcomes_available:
        phase3_status = "failed_selection_incomplete_cat_not_run"
    else:
        phase3_status = "passed" if phase3_pass else "failed_validation"
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": _utcnow(),
        "overall_status": inputs.terminal_state,
        "release_claims_authorized": False,
        "final_policy_valid": False,
        "gates": {
            "v4_numerical_verification": {
                "status": "passed" if v4_pass else "blocked",
                "decision": _display(inputs.v4_dir / "numerical_followup_decision.json"),
                "lock_present": inputs.v4_lock is not None,
                "fixed_bank_scoring_status": (
                    "not_applicable_refit_gate_failed"
                    if inputs.v4_decision.get("all_three_refit_bank_gates_passed") is not True
                    else (
                        "passed"
                        if inputs.v4_decision.get(
                            "all_three_fixed_bank_scoring_and_tail_gates_passed"
                        )
                        is True
                        else "failed"
                    )
                ),
            },
            "phase3_repeated_nested_cat": {
                "status": phase3_status,
                "outcome_artifacts_available": outcomes_available,
                "phase4_authorized": phase3_pass,
                "failed_conditions": (
                    list(inputs.phase3_decision.get("failed_conditions") or [])
                    if inputs.phase3_decision
                    else []
                ),
            },
            "phase4_total_uncertainty_and_order": {
                "status": "pending_not_run" if phase3_pass else "not_run_not_authorized"
            },
            "final_fit_export_and_replay": {"status": "not_run_not_authorized"},
        },
        "limitations": [
            "same 52-tutor, 22-family cohort reused for internal development",
            "results are conditional on frozen Qwen labels not human-validated on InFoBench",
            "no unseen-family or independent-confirmation claim is authorized",
        ],
    }


def _summary_markdown(inputs: TerminalInputs, figure_plan: Sequence[tuple[str, str, str]]) -> str:
    fixed_bank_status = (
        str(inputs.v4_decision.get("all_three_fixed_bank_scoring_and_tail_gates_passed"))
        if inputs.v4_decision.get("all_three_refit_bank_gates_passed") is True
        else "N/A (not run because a refitted-bank gate failed)"
    )
    lines = [
        "# InFoBench V3 calibration/CAT execution summary",
        "",
        f"**Status:** `{inputs.terminal_state}`  ",
        "**Interpretation:** Same-cohort internal-development evidence conditional on frozen Qwen labels",
        "",
        "## Bottom line",
        "",
    ]
    if inputs.terminal_state == "blocked_after_v4_numerical_followup":
        lines.extend(
            [
                "V4 did not pass every frozen numerical gate. It correctly wrote no numerical lock,",
                "so Phase 3/CAT was not run and no CAT-efficiency or final-policy claim is available.",
            ]
        )
    elif inputs.terminal_state == "blocked_after_phase3_selection":
        selection = (inputs.phase3_decision or {}).get("calibration_selection") or {}
        lines.extend(
            [
                "V4 passed, but at least one preregistered inner calibration selection was incomplete.",
                "The fail-closed gate stopped before any outer response, CAT path, or random baseline was",
                "opened. Only selection evidence is reportable; CAT accuracy and efficiency are not estimable.",
                "Selection completeness: "
                f"{selection.get('complete_panels_numerator', 0)}/"
                f"{selection.get('expected_panels_denominator', 25)} panels and "
                f"{selection.get('valid_spec_fold_combinations_numerator', 0)}/"
                f"{selection.get('expected_spec_fold_combinations_denominator', 300)} "
                "specification-fold evidence blocks.",
            ]
        )
    elif inputs.terminal_state == "blocked_after_phase3_validation":
        failures = ", ".join(inputs.phase3_decision.get("failed_conditions") or [])  # type: ignore[union-attr]
        lines.extend(
            [
                "V4 passed and authorized the dense numerical profile, but the repeated nested Phase-3",
                f"CAT study failed its frozen validation conditions: {failures or 'unspecified failure'}.",
                "Phase 4 and final fitting are not authorized.",
            ]
        )
    else:
        lines.extend(
            [
                "V4 passed and Phase 3 passed its repeated nested-CV gates. This is not yet a final",
                "deployment result: Phase 4 total-uncertainty/order validation remains pending, and no",
                "final all-52 bank, deployment replay, or leaderboard has been authorized.",
            ]
        )
    lines.extend(
        [
            "",
            "## V4 numerical verification",
            "",
            f"- Fresh fits covered: {len(inputs.v4_decision.get('fit_validity') or {})}/54.",
            f"- Cold-versus-continuation start gates covered: {len(inputs.v4_decision.get('start_validity') or {})}/18.",
            f"- All three refitted-bank gates passed: `{inputs.v4_decision.get('all_three_refit_bank_gates_passed')}`.",
            f"- All three fixed-bank scoring/tail gates passed: `{fixed_bank_status}`.",
            f"- Numerical lock present: `{inputs.v4_lock is not None}`.",
            "- V4 selected no calibration family and ran no CAT; those decisions belong to later stages.",
            "",
            "## Phase 3",
            "",
        ]
    )
    if inputs.phase3_decision is None:
        lines.append("Not run because V4 did not produce a valid numerical lock.")
    else:
        decision = inputs.phase3_decision
        stability = decision.get("calibration_stability") or {}
        coverage = decision.get("coverage") or {}
        lines.append(
            f"- Decision: `{decision.get('status')}`; Phase 4 authorized: `{decision.get('phase4_authorized')}`."
        )
        if inputs.phase3 and not inputs.phase3["outcomes_available"]:
            selection = decision.get("calibration_selection") or {}
            outer = decision.get("outer_evaluation") or {}
            lines.extend(
                [
                    "- Inner selections complete: "
                    f"`{selection.get('complete_panels_numerator')}/"
                    f"{selection.get('expected_panels_denominator')}` panels.",
                    "- Valid specification-fold evidence: "
                    f"`{selection.get('valid_spec_fold_combinations_numerator')}/"
                    f"{selection.get('expected_spec_fold_combinations_denominator')}`.",
                    "- Outer CAT panels evaluated: "
                    f"`{outer.get('evaluated_outer_panels_numerator')}/"
                    f"{outer.get('expected_outer_panels_denominator')}`; CAT metrics are not estimable.",
                ]
            )
        else:
            lines.extend(
                [
                    f"- Outer panels evaluated: `{coverage.get('all_25_panels_evaluated')}` (denominator 25).",
                    f"- All 52 models present in every repetition: `{coverage.get('all_52_models_each_repetition')}`.",
                    f"- Unique modal calibration specification: `{stability.get('unique_modal_spec_id')}`.",
                    f"- Modal selection fraction: `{stability.get('modal_fraction_all_25_panels')}` (required 0.80).",
                    "- The floor-12 and D-opt policies are diagnostic only and cannot replace the frozen primary policy.",
                ]
            )
    lines.extend(
        [
            "",
            "## Figures",
            "",
        ]
    )
    for filename, title, _description in figure_plan:
        lines.append(f"- `{filename}` — {title}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The same 52 tutors from 22 model families are reused; this is not independent confirmation.",
            "- InFoBench labels come from the frozen Qwen judge and were not independently human-validated on InFoBench.",
            "- No claim of generalization to unseen model families is authorized.",
            "- A Phase-3 pass is only authorization to run Phase 4, not a valid final CAT policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _figure_guide(inputs: TerminalInputs, plan: Sequence[tuple[str, str, str]]) -> str:
    lines = [
        "# InFoBench V3 figure guide",
        "",
        f"**Terminal state:** `{inputs.terminal_state}`",
        f"**Visible status:** {_terminal_banner(inputs)}",
        "",
        "Every image was recomputed from V4/V3 artifacts. Historical images were used only as style references.",
        "",
        "## Figures",
        "",
    ]
    for filename, title, description in plan:
        lines.extend([f"### `{filename}` — {title}", "", description, ""])
    lines.extend(["## Deliberately not applicable or not run", ""])
    if not inputs.numerical["fixed_gates"]:
        lines.append(
            "- Fixed-bank scoring plot: `not run` because a refitted-bank gate failed first."
        )
    if inputs.phase3 is None:
        lines.extend(
            [
                "- All CAT, sensitivity, recovery, CAT-versus-random, uncertainty, order, and leaderboard figures: `not run` because V4 produced no lock.",
            ]
        )
    elif not inputs.phase3["outcomes_available"]:
        lines.extend(
            [
                "- CAT gate, policy-sensitivity, recovery, pass-rate, CAT-versus-random, and outer-coverage outcome figures: `not run / not estimable` because the preregistered selection gate stopped before outer outcomes were opened.",
                "- Phase-4 conditional/parameter/total-uncertainty and order figures: `not run`; Phase 3 did not authorize them.",
                "- Final-policy leaderboard: `not run`; no CAT policy outcome was evaluated.",
            ]
        )
    else:
        lines.extend(
            [
                "- Old 48-way floor/SE sweep and inner-CV CAT tradeoff: `not applicable`; V3 froze one primary and two non-promotable sensitivities instead of rerunning the search.",
                "- Phase-4 conditional/parameter/total-uncertainty and order figures: `not run` by this Phase-3 renderer.",
                "- Final-policy leaderboard: `not run`; it requires Phase 4 plus an authorized all-52 final fit/replay.",
            ]
        )
    if inputs.terminal_state == "blocked_after_phase3_selection":
        lines.append(
            "- Figure 07 is selection evidence only and visibly states that CAT was not run."
        )
    elif inputs.terminal_state == "blocked_after_phase3_validation":
        lines.append(
            "- Outcome figures carry `INCOMPLETE` and are failed-validation diagnostics, not deployment evidence."
        )
    elif inputs.terminal_state == "phase3_pass_phase4_pending":
        lines.append(
            "- Phase-3 figures are internal validation evidence only; Phase 4 remains pending."
        )
    lines.extend(
        [
            "",
            "## Counterpart index",
            "",
            "- Figures 01–05 correspond to the August 5 item-parameter and numerical-stability plots.",
            "- Figure 07 replaces the old ridge-only selection plot with exact calibration-specification selection.",
            "- Figures 08–13 correspond to gate, sensitivity, recovery, pass-rate, CAT-versus-random, and coverage diagnostics.",
            "- No old image is copied or presented as new evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _reproduction_commands(inputs: TerminalInputs) -> str:
    lines = [
        "# Run from the eduLLM-Evals repository root.",
        "",
        "# V4 fresh run (a partially completed identical run must use --resume instead).",
        "uv run --frozen --extra irt python scripts/run_infobench_v2_numerical_followup_v4.py \\",
        "  --config configs/infobench_v2_numerical_followup_v4.json \\",
        "  --out-dir runs/calibration/InFoBench_v2_numerical_followup_v4",
        "",
        "# Resume an interrupted, hash-identical V4 run.",
        "uv run --frozen --extra irt python scripts/run_infobench_v2_numerical_followup_v4.py \\",
        "  --config configs/infobench_v2_numerical_followup_v4.json \\",
        "  --out-dir runs/calibration/InFoBench_v2_numerical_followup_v4 --resume",
        "",
    ]
    if inputs.v4_lock is None:
        lines.extend(["# Phase 3 is not authorized because V4 produced no lock.", ""])
    else:
        lines.extend(
            [
                "# Phase 3 (only after the valid V4 lock).",
                "uv run --frozen --extra irt python scripts/nested_scenario_cat_cv_v3.py \\",
                "  --config configs/infobench_calibration_cat_v3.json \\",
                "  --split-manifest configs/infobench_v2_splits.manifest.json \\",
                "  --out-dir runs/calibration/InFoBench_v3/phase3 \\",
                "  --numerical-followup-config configs/infobench_v2_numerical_followup_v4.json \\",
                "  --numerical-lock runs/calibration/InFoBench_v2_numerical_followup_v4/numerical_followup_lock.json",
                "",
            ]
        )
    if inputs.terminal_state == "phase3_pass_phase4_pending":
        lines.extend(
            [
                "# Phase 4 is authorized by the complete passing Phase-3 decision.",
                "uv run --frozen --extra irt python scripts/nested_cat_total_uncertainty_v3.py \\",
                "  --config configs/infobench_calibration_cat_v3.json \\",
                "  --phase3-dir runs/calibration/InFoBench_v3/phase3 \\",
                "  --out-dir runs/calibration/InFoBench_v3/phase4 \\",
                "  --numerical-followup-config configs/infobench_v2_numerical_followup_v4.json \\",
                "  --numerical-lock runs/calibration/InFoBench_v2_numerical_followup_v4/numerical_followup_lock.json",
                "",
                "# Final fit/export/replay remains unavailable until Phase 4 completes and passes.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "# Phase 4 is not authorized by this terminal state.",
                "# Final fit/export/replay is therefore unavailable.",
                "",
            ]
        )
    lines.extend(
        [
            "# Terminal renderer (run once after V4 blocks or after Phase 3 completes).",
            "uv run --frozen --extra irt python scripts/render_infobench_v3_terminal_v1.py \\",
            "  --config configs/infobench_calibration_cat_v3.json \\",
            "  --v4-dir runs/calibration/InFoBench_v2_numerical_followup_v4 \\",
            "  --phase3-dir runs/calibration/InFoBench_v3/phase3 \\",
            "  --out-dir reports/infobench_calibration_cat_v3",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def render_terminal_report(
    *, config_path: Path, v4_dir: Path, phase3_dir: Path, output_dir: Path
) -> list[Path]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        _require(output_dir.is_dir(), f"renderer output path is not a directory: {output_dir}")
        _require(
            not any(output_dir.iterdir()),
            f"renderer output directory must be absent or empty; stale content found: {output_dir}",
        )
    inputs = load_terminal_inputs(
        config_path=config_path.resolve(),
        v4_dir=v4_dir.resolve(),
        # Do not resolve the CAT path until load_terminal_inputs has verified a
        # passing V4 lock; blocked V4 rendering is strictly CAT-isolated.
        phase3_dir=phase3_dir,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.rendering-", dir=output_dir.parent))
    try:
        figures_dir = staging / "figures"
        _style()
        figures = _render_figures(inputs, figures_dir)
        plan = _figure_plan(inputs)
        _write_text(staging / "EXECUTION_SUMMARY.md", _summary_markdown(inputs, plan))
        _write_text(
            staging / "gate_status.json",
            json.dumps(_gate_status(inputs), indent=2, sort_keys=True) + "\n",
        )
        _write_text(staging / "reproduction_commands.txt", _reproduction_commands(inputs))
        _write_text(figures_dir / "FIGURE_GUIDE.md", _figure_guide(inputs, plan))

        source_hashes = {_display(path): _sha256(path) for path in sorted(inputs.tracker.paths)}
        report_files = {
            str(path.relative_to(staging)): _sha256(path)
            for path in (
                staging / "EXECUTION_SUMMARY.md",
                staging / "gate_status.json",
                staging / "reproduction_commands.txt",
                figures_dir / "FIGURE_GUIDE.md",
            )
        }
        outcomes_available = bool(inputs.phase3 and inputs.phase3.get("outcomes_available"))
        derived = (inputs.phase3 or {}).get("derived_denominators") or {}
        selection_panels_expected = int(derived.get("selection_panels_expected", 0))
        selection_panels_complete = int(derived.get("selection_panels_complete", 0))
        spec_fold_blocks_expected = int(derived.get("spec_fold_blocks_expected", 0))
        spec_fold_blocks_valid = int(derived.get("spec_fold_blocks_valid", 0))
        completed_panels = len(derived.get("evaluated_panels") or set())
        primary_rows = int(derived.get("primary_rows", 0))
        valid_prediction_counts = {"cat": 0, "baseline": 0}
        complete_recovery_models = 0
        valid_pass_rate_counts = {"cat": 0, "baseline": 0}
        if outcomes_available:
            prediction = inputs.phase3["prediction"]
            prediction = prediction[prediction["candidate_id"] == PRIMARY_POLICY]
            log_loss = pd.to_numeric(prediction["log_loss"], errors="coerce")
            n_cells = pd.to_numeric(prediction["n_cells"], errors="coerce")
            valid_prediction = np.isfinite(log_loss) & log_loss.gt(0) & n_cells.gt(0)
            valid_prediction_counts = {
                mode: int((valid_prediction & prediction["mode"].astype(str).eq(mode)).sum())
                for mode in ("cat", "baseline")
            }
            cross = inputs.phase3["cross_model"]
            cross = cross[
                (cross["policy_id"] == PRIMARY_POLICY)
                & pd.to_numeric(cross["n_repetitions"], errors="coerce").eq(5)
            ]
            complete_recovery_models = int(
                (
                    np.isfinite(pd.to_numeric(cross["mean_theta_reference"], errors="coerce"))
                    & np.isfinite(pd.to_numeric(cross["mean_theta_cat_mwle"], errors="coerce"))
                ).sum()
            )
            primary = inputs.phase3["outer"]
            primary = primary[primary["policy_id"] == PRIMARY_POLICY]
            for mode in ("cat", "baseline"):
                observed = pd.to_numeric(
                    primary[f"{mode}_eval_observed_pass_rate"], errors="coerce"
                )
                predicted = pd.to_numeric(
                    primary[f"{mode}_eval_predicted_pass_rate"], errors="coerce"
                )
                valid_pass_rate_counts[mode] = int(
                    (
                        np.isfinite(observed)
                        & np.isfinite(predicted)
                        & observed.between(0, 1)
                        & predicted.between(0, 1)
                    ).sum()
                )
        manifest = {
            "schema_version": FIGURE_MANIFEST_SCHEMA,
            "generated_at_utc": _utcnow(),
            "terminal_state": inputs.terminal_state,
            "renderer": _display(Path(__file__)),
            "renderer_sha256": _sha256(Path(__file__)),
            "source_files": source_hashes,
            "figure_files": {path.name: _sha256(path) for path in figures},
            "report_files": report_files,
            "phase3_artifacts_opened": inputs.phase3 is not None,
            "phase3_outcome_artifacts_opened": outcomes_available,
            "old_images_used_as_data": False,
            "denominators": {
                "v4_fits": EXPECTED_FITS,
                "v4_start_comparisons": EXPECTED_STARTS,
                "phase3_selection_panels_expected": (
                    selection_panels_expected if inputs.phase3 is not None else 0
                ),
                "phase3_selection_panels_complete": (
                    selection_panels_complete if inputs.phase3 is not None else 0
                ),
                "phase3_spec_fold_blocks_expected": (
                    spec_fold_blocks_expected if inputs.phase3 is not None else 0
                ),
                "phase3_spec_fold_blocks_valid": (
                    spec_fold_blocks_valid if inputs.phase3 is not None else 0
                ),
                "phase3_outer_panels_expected": 25 if inputs.phase3 is not None else 0,
                "phase3_outer_panels_evaluated": completed_panels,
                "phase3_model_repetitions_expected_per_policy": (
                    EXPECTED_PRIMARY_MODEL_REPEAT_ROWS if inputs.phase3 is not None else 0
                ),
                "phase3_model_repetitions_evaluated_primary": primary_rows,
                "phase3_recovery_models_with_all_5_repetitions": complete_recovery_models,
                "phase3_valid_cat_prediction_panels": valid_prediction_counts["cat"],
                "phase3_valid_random_prediction_panels": valid_prediction_counts["baseline"],
                "phase3_valid_cat_pass_rate_rows": valid_pass_rate_counts["cat"],
                "phase3_valid_random_pass_rate_rows": valid_pass_rate_counts["baseline"],
                "phase3_distinct_models_per_repetition_expected": (
                    52 if inputs.phase3 is not None else 0
                ),
            },
        }
        manifest_path = figures_dir / "figure_source_manifest.json"
        _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        targets = [
            output_dir / "EXECUTION_SUMMARY.md",
            output_dir / "gate_status.json",
            output_dir / "reproduction_commands.txt",
            output_dir / "figures" / "FIGURE_GUIDE.md",
            output_dir / "figures" / "figure_source_manifest.json",
            *[output_dir / "figures" / path.name for path in figures],
        ]
        existing = [path for path in targets if path.exists()]
        if existing:
            raise TerminalReportError(
                "refusing to overwrite existing derived report files: "
                + ", ".join(map(str, existing[:5]))
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "figures").mkdir(parents=True, exist_ok=True)
        for relative in (
            Path("EXECUTION_SUMMARY.md"),
            Path("gate_status.json"),
            Path("reproduction_commands.txt"),
            Path("figures/FIGURE_GUIDE.md"),
            Path("figures/figure_source_manifest.json"),
            *[Path("figures") / path.name for path in figures],
        ):
            os.replace(staging / relative, output_dir / relative)
        expected_relative_files = {path.relative_to(output_dir) for path in targets}
        actual_relative_files = {
            path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file()
        }
        _require(
            actual_relative_files == expected_relative_files,
            "renderer output inventory differs after publication",
        )
        return targets
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v4-dir", type=Path, default=DEFAULT_V4_DIR)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        outputs = render_terminal_report(
            config_path=args.config,
            v4_dir=args.v4_dir,
            phase3_dir=args.phase3_dir,
            output_dir=args.out_dir,
        )
        print(f"Wrote {len(outputs)} terminal report artifacts to {args.out_dir.resolve()}")
        return 0
    except (TerminalReportError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
