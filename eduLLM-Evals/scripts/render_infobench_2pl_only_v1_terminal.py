#!/usr/bin/env python3
"""Render the terminal InFoBench 2PL-only-v1 numerical/CAT report.

The unchanged V4 numerical study is validated against its full three-spec
roster.  The versioned Phase-3 evidence is independently validated against the
two preregistered 2PL candidates and exactly 200 inner spec-fold blocks.  This
renderer never opens a completed V3 Phase-3, Phase-4, final-fit, or report leaf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import nested_scenario_cat_cv_2pl_only_v1 as phase3  # noqa: E402
from scripts import render_infobench_v3_terminal_v1 as base  # noqa: E402

DEFAULT_CONFIG = phase3.DEFAULT_CONFIG
DEFAULT_V4_DIR = ROOT / "runs" / "calibration" / "InFoBench_v2_numerical_followup_v4"
DEFAULT_PHASE3_DIR = ROOT / "runs" / "calibration" / "InFoBench_2pl_only_v1" / "phase3"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "infobench_calibration_cat_2pl_only_v1"

REPORT_SCHEMA = "infobench-2pl-only-v1-terminal-report-v1"
FIGURE_MANIFEST_SCHEMA = "infobench-2pl-only-v1-terminal-figure-manifest-v1"
PHASE3_SPEC_IDS = phase3.EXPECTED_PHASE3_SPEC_IDS
V4_SPEC_IDS = phase3.EXPECTED_V4_SPEC_IDS
SPEC_LABELS = {
    "log_shrinkage_2pl_lambda16": "shrinkage 2PL (lambda=16)",
    "log_shrinkage_2pl_lambda4": "shrinkage 2PL (lambda=4)",
}
EXPECTED_INNER_SUMMARY_ROWS = phase3.EXPECTED_PANELS * len(PHASE3_SPEC_IDS)
EXPECTED_OUTER_ROWS = 780
EXPECTED_PREDICTION_ROWS = 150
EXPECTED_GATE_ROWS = 90
EXPECTED_CROSS_MODEL_ROWS = 156
CANONICAL_PHASE3_DECISION_NAME = "phase3_decision.json"
# The inherited V3 Phase-4 inventory still consumes this byte-identical alias.
COMPATIBILITY_PHASE3_DECISION_MIRROR_NAME = "v3_decision.json"


class TerminalReportError(base.TerminalReportError):
    """A versioned source, schema, provenance, or reporting check failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalReportError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _assert_no_symlink_components(path: Path, *, root: Path = ROOT) -> None:
    lexical = _lexical_absolute(path)
    lexical_root = _lexical_absolute(root)
    try:
        relative = lexical.relative_to(lexical_root)
    except ValueError as error:
        raise TerminalReportError("renderer path is outside the repository root") from error
    current = lexical_root
    if current.exists() and current.is_symlink():
        raise TerminalReportError("repository root is a symlink")
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise TerminalReportError(f"renderer path traverses a symlink: {current}")


def _pin_canonical_path(
    requested: Path,
    canonical: Path,
    *,
    label: str,
    inspect_filesystem: bool = True,
    root: Path = ROOT,
) -> Path:
    """Reject aliases and CLI overrides outside one frozen versioned leaf."""

    requested_path = requested if requested.is_absolute() else Path.cwd() / requested
    requested_lexical = _lexical_absolute(requested_path)
    canonical_lexical = _lexical_absolute(canonical)
    _require(
        requested_lexical == canonical_lexical,
        f"{label} is not the canonical 2PL-only-v1 path",
    )
    if inspect_filesystem:
        _require(
            requested_path.resolve(strict=False) == canonical_lexical,
            f"{label} resolves through a noncanonical alias",
        )
        _assert_no_symlink_components(requested_lexical, root=root)
    return canonical_lexical


@contextmanager
def _phase3_plot_bindings() -> Iterator[None]:
    names = {
        "SPEC_IDS": PHASE3_SPEC_IDS,
        "SPEC_LABELS": SPEC_LABELS,
        "V3_CONFIG_SCHEMA": phase3.CONFIG_SCHEMA,
        "V3_RUN_SCHEMA": phase3.SCRIPT_SCHEMA,
        "V3_DECISION_SCHEMA": phase3.DECISION_SCHEMA,
        "V3_SELECTED_SPECS_SCHEMA": phase3.SELECTED_SPECS_SCHEMA,
        "V3_CODE_DEPENDENCIES": phase3.CODE_DEPENDENCY_RELATIVE_PATHS,
        "EXPECTED_INNER_ROWS": EXPECTED_INNER_SUMMARY_ROWS,
    }
    previous = {name: getattr(base, name) for name in names}
    for name, value in names.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def _validate_config(config_path: Path, tracker: base.SourceTracker) -> dict[str, Any]:
    config = base._read_json(config_path, tracker)
    try:
        phase3._check_frozen_config(config)
    except phase3.V3Phase3Error as error:
        raise TerminalReportError(str(error)) from error
    outputs = config.get("outputs") or {}
    expected = {
        "phase3": "runs/calibration/InFoBench_2pl_only_v1/phase3",
        "phase4": "runs/calibration/InFoBench_2pl_only_v1/phase4",
        "final_fit": "runs/calibration/InFoBench_2pl_only_v1/final_fit",
        "report": "reports/infobench_calibration_cat_2pl_only_v1",
    }
    _require(
        all(outputs.get(key) == value for key, value in expected.items()),
        "2PL output paths changed",
    )
    _require(
        outputs.get("never_read_or_write_parent_v3_artifacts") is True,
        "2PL config does not protect the completed V3 study",
    )
    _require(
        [str(row.get("spec_id")) for row in config.get("calibration_specifications") or []]
        == list(PHASE3_SPEC_IDS),
        "Phase-3 2PL candidate roster changed",
    )
    _require(
        list((config.get("numerical_lock") or {}).get("eligible_spec_ids") or [])
        == list(V4_SPEC_IDS),
        "full V4 numerical roster changed",
    )
    return config


def _recorded_output_inventory(
    directory: Path,
    manifest: Mapping[str, Any],
    expected_names: set[str],
    tracker: base.SourceTracker,
) -> None:
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, Mapping), "Phase-3 output inventory is missing")
    _require(set(map(str, outputs)) == expected_names, "Phase-3 output inventory differs")
    for name in sorted(expected_names):
        raw = outputs[name]
        _require(isinstance(raw, Mapping), f"Phase-3 output provenance is invalid: {name}")
        base._validate_recorded_file(raw, directory / name, tracker, label=f"Phase-3 {name}")


def _validate_panel_audits(
    selected: Mapping[str, Any], *, require_complete: bool
) -> dict[str, Any]:
    panels = selected.get("panels")
    _require(
        isinstance(panels, list) and len(panels) == phase3.EXPECTED_PANELS,
        "expected 25 Phase-3 panels",
    )
    keys: set[tuple[int, int]] = set()
    valid_blocks = 0
    complete_panels = 0
    for raw in panels:
        _require(isinstance(raw, Mapping), "Phase-3 panel record is invalid")
        key = (int(raw.get("repeat", -1)), int(raw.get("outer_fold", -1)))
        _require(
            key not in keys and key[0] in range(5) and key[1] in range(5),
            "Phase-3 panel coordinates are not the exact 5x5 grid",
        )
        keys.add(key)
        audit = (raw.get("inner_selection") or {}).get("complete_inner_evidence_audit") or {}
        expected_ids = list(audit.get("expected_spec_ids") or [])
        expected_blocks = audit.get("expected_spec_fold_combinations")
        valid = int(audit.get("valid_spec_fold_combinations", 0))
        combinations = audit.get("combination_audits") or []
        _require(expected_ids == list(PHASE3_SPEC_IDS), f"panel {key} changed its two-spec roster")
        _require(
            expected_blocks == phase3.EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL
            and len(combinations) == phase3.EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL,
            f"panel {key} does not contain exactly 8 spec-fold blocks",
        )
        _require(
            audit.get("require_all_specs_every_inner_fold") is True
            and audit.get("survivor_selection_allowed") is False
            and audit.get("expected_inner_folds") == list(range(4)),
            f"panel {key} changed its fail-closed inner-fold contract",
        )
        _require(
            all(isinstance(row, Mapping) for row in combinations),
            f"panel {key} contains an invalid spec-fold block",
        )
        _require(
            {(int(row.get("inner_fold", -1)), str(row.get("spec_id"))) for row in combinations}
            == {(fold, spec) for fold in range(4) for spec in PHASE3_SPEC_IDS},
            f"panel {key} spec-fold block coordinates differ",
        )
        _require(
            all(type(row.get("passed")) is bool for row in combinations),
            f"panel {key} has a non-boolean spec-fold pass flag",
        )
        derived_valid = sum(row.get("passed") is True for row in combinations)
        failures = audit.get("failed_checks")
        _require(
            isinstance(failures, list)
            and failures == sorted(set(map(str, failures)))
            and 0 <= valid <= phase3.EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL
            and valid == derived_valid,
            f"panel {key} spec-fold totals differ from its block audits",
        )
        expected_pass = valid == phase3.EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL and not failures
        _require(
            audit.get("passed") is expected_pass,
            f"panel {key} inner-evidence pass flag disagrees",
        )
        valid_blocks += valid
        complete = (
            audit.get("passed") is True and valid == phase3.EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL
        )
        complete_panels += int(complete)
        if require_complete:
            _require(complete, f"terminal Phase-3 panel {key} has incomplete inner evidence")
    if require_complete:
        _require(
            valid_blocks == phase3.EXPECTED_TOTAL_SPEC_FOLD_BLOCKS,
            "Phase-3 does not contain 200 valid blocks",
        )
    return {
        "selection_panels_expected": phase3.EXPECTED_PANELS,
        "selection_panels_complete": complete_panels,
        "spec_fold_blocks_expected": phase3.EXPECTED_TOTAL_SPEC_FOLD_BLOCKS,
        "spec_fold_blocks_valid": valid_blocks,
    }


def _validate_panel_mirrors(
    selected: Mapping[str, Any],
    choices: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate the same frozen selection and inner evidence in all handoffs."""

    derived = _validate_panel_audits(selected, require_complete=require_complete)
    for label, artifact in (
        ("calibration choices", choices),
        ("pre-outer selection lock", selection_lock),
    ):
        mirror_derived = _validate_panel_audits(artifact, require_complete=require_complete)
        _require(mirror_derived == derived, f"{label} panel denominators differ")

    def panel_map(value: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
        return {
            (int(row["repeat"]), int(row["outer_fold"])): row for row in value.get("panels") or []
        }

    selected_panels = panel_map(selected)
    choice_panels = panel_map(choices)
    lock_panels = panel_map(selection_lock)
    for key, panel in selected_panels.items():
        choice = choice_panels[key]
        locked = lock_panels[key]
        _require(
            panel.get("selected_spec_id")
            == choice.get("selected_spec_id")
            == locked.get("selected_spec_id"),
            f"Phase-3 selected specification differs across panel mirrors: {key}",
        )
        selected_audit = (panel.get("inner_selection") or {}).get("complete_inner_evidence_audit")
        _require(
            selected_audit
            == (choice.get("inner_selection") or {}).get("complete_inner_evidence_audit")
            == (locked.get("inner_selection") or {}).get("complete_inner_evidence_audit"),
            f"Phase-3 inner evidence differs across panel mirrors: {key}",
        )
    return derived


def _load_phase3_terminal(
    phase3_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    v4_dir: Path,
    v4_lock: Mapping[str, Any],
    tracker: base.SourceTracker,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # phase3_decision.json is canonical. v3_decision.json is retained only as
    # the byte-equivalent compatibility mirror expected by inherited Phase 4.
    decision = base._read_json(phase3_dir / CANONICAL_PHASE3_DECISION_NAME, tracker)
    mirror = base._read_json(phase3_dir / COMPATIBILITY_PHASE3_DECISION_MIRROR_NAME, tracker)
    manifest = base._read_json(phase3_dir / "manifest.json", tracker)
    _require(decision == mirror, "Phase-3 decision mirrors differ")
    _require(
        decision.get("schema_version") == phase3.DECISION_SCHEMA,
        "unexpected 2PL Phase-3 decision schema",
    )
    _require(
        manifest.get("schema_version") == phase3.SCRIPT_SCHEMA,
        "unexpected 2PL Phase-3 manifest schema",
    )
    status = str(manifest.get("status") or "")
    selection_only = status == "phase3_terminal_selection_failed"
    _require(selection_only or status == "phase3_complete", "2PL Phase 3 is not terminal")
    if selection_only:
        _require(
            decision.get("status") == "fail_selection_incomplete"
            and decision.get("phase3_pass") is False
            and decision.get("phase4_authorized") is False,
            "selection-only Phase-3 decision is inconsistent",
        )
    else:
        phase3_pass = decision.get("phase3_pass") is True
        _require(
            decision.get("status") == ("pass" if phase3_pass else "fail")
            and decision.get("phase4_authorized") is phase3_pass,
            "Phase-3 status and downstream authorization disagree",
        )
    _require(
        manifest.get("phase3_decision")
        == {
            "status": decision.get("status"),
            "phase3_pass": decision.get("phase3_pass"),
            "phase4_authorized": decision.get("phase4_authorized"),
        },
        "Phase-3 manifest decision summary differs",
    )
    expected_outputs = set(phase3.TERMINAL_MANIFEST_OUTPUTS[status])
    _recorded_output_inventory(phase3_dir, manifest, expected_outputs, tracker)
    _require(
        manifest.get("script") == "scripts/nested_scenario_cat_cv_2pl_only_v1.py",
        "Phase-3 producer script differs",
    )
    inputs = manifest.get("inputs") or {}
    base._validate_recorded_file(
        inputs.get("config") or {}, config_path, tracker, label="Phase-3 config"
    )
    code = manifest.get("code_provenance") or {}
    files = code.get("files") or {}
    _require(
        list(code.get("dependency_inventory") or []) == list(phase3.CODE_DEPENDENCY_RELATIVE_PATHS)
        and set(files) == set(phase3.CODE_DEPENDENCY_RELATIVE_PATHS),
        "Phase-3 code inventory differs",
    )
    for name, digest in files.items():
        path = ROOT / name
        _require(path.is_file() and _sha256(path) == digest, f"Phase-3 code hash differs: {name}")
    _require(
        code.get("canonical_sha256") == phase3._canonical_hash(files),
        "Phase-3 code hash panel differs",
    )
    numerical = manifest.get("numerical_lock") or {}
    _require(
        numerical.get("status") == "passed"
        and list(numerical.get("eligible_spec_ids") or []) == list(V4_SPEC_IDS)
        and list(numerical.get("passed_spec_ids") or []) == list(V4_SPEC_IDS),
        "Phase-3 did not preserve the full passing V4 roster",
    )
    _require(
        numerical.get("verification_sha256") == _sha256(v4_dir / "numerical_followup_lock.json")
        and numerical.get("study_signature_sha256") == v4_lock.get("study_signature_sha256"),
        "Phase-3 V4 lock provenance differs",
    )
    _require(
        [str(row.get("spec_id")) for row in manifest.get("calibration_specifications") or []]
        == list(PHASE3_SPEC_IDS),
        "Phase-3 manifest 2PL roster differs",
    )

    selected = base._read_json(phase3_dir / "selected_calibration_specs.json", tracker)
    choices = base._read_json(phase3_dir / "calibration_model_choices.json", tracker)
    selection_lock = base._read_json(phase3_dir / "pre_outer_selection_lock.json", tracker)
    fold_assignments = base._read_json(phase3_dir / "fold_assignments.json", tracker)
    _require(
        selected.get("schema_version") == phase3.SELECTED_SPECS_SCHEMA,
        "selected-spec schema differs",
    )
    for value in (choices, selection_lock, fold_assignments):
        _require(
            value.get("schema_version") == phase3.SCRIPT_SCHEMA, "Phase-3 auxiliary schema differs"
        )
    _require(
        decision.get("study_signature")
        == selected.get("study_signature")
        == manifest.get("study_signature"),
        "Phase-3 study signatures differ",
    )
    derived = _validate_panel_mirrors(
        selected,
        choices,
        selection_lock,
        require_complete=not selection_only,
    )
    inner = base._read_csv(
        phase3_dir / "inner_calibration_model_results.csv",
        tracker,
        required=("repeat", "outer_fold", "spec_id"),
        expected_rows=None if selection_only else EXPECTED_INNER_SUMMARY_ROWS,
    )
    _require(
        set(inner["spec_id"].astype(str)) <= set(PHASE3_SPEC_IDS),
        "inner summary contains a non-2PL candidate",
    )
    if not selection_only:
        inner_contract = decision.get("inner_selection_fail_closed") or {}
        _require(
            inner_contract.get("expected_spec_fold_combinations_per_panel")
            == phase3.EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL,
            "Phase-3 decision must record 8 blocks per panel",
        )

    tables: dict[str, Any] = {
        "inner": inner,
        "selected": selected,
        "choices": choices,
        "selection_lock": selection_lock,
        "fold_assignments": fold_assignments,
        "outcomes_available": not selection_only,
        "derived_denominators": derived,
    }
    if selection_only:
        _require(
            decision.get("phase3_pass") is False and decision.get("phase4_authorized") is False,
            "selection failure authorized downstream work",
        )
        return decision, manifest, tables

    tables.update(
        {
            "outer": base._read_csv(
                phase3_dir / "outer_oof_per_model.csv", tracker, expected_rows=EXPECTED_OUTER_ROWS
            ),
            "prediction": base._read_csv(
                phase3_dir / "disjoint_prediction_metrics.csv",
                tracker,
                expected_rows=EXPECTED_PREDICTION_ROWS,
            ),
            "gates": base._read_csv(
                phase3_dir / "repeat_fold_gate_results.csv",
                tracker,
                expected_rows=EXPECTED_GATE_ROWS,
            ),
            "repeat_metrics": base._read_csv(
                phase3_dir / "repeat_metrics.csv", tracker, expected_rows=15
            ),
            "frequency": base._read_csv(
                phase3_dir / "calibration_spec_selection_frequency.csv", tracker, expected_rows=2
            ),
            "cross_model": base._read_csv(
                phase3_dir / "cross_repeat_per_model.csv",
                tracker,
                expected_rows=EXPECTED_CROSS_MODEL_ROWS,
            ),
            "cross_metrics": base._read_csv(
                phase3_dir / "cross_repeat_metrics.csv", tracker, expected_rows=3
            ),
        }
    )
    _require(
        set(tables["frequency"]["spec_id"].astype(str)) == set(PHASE3_SPEC_IDS),
        "selection frequency roster differs",
    )
    return decision, manifest, tables


def load_terminal_inputs(
    *, config_path: Path, v4_dir: Path, phase3_dir: Path
) -> base.TerminalInputs:
    config_path = _pin_canonical_path(config_path, DEFAULT_CONFIG, label="renderer config")
    v4_dir = _pin_canonical_path(v4_dir, DEFAULT_V4_DIR, label="V4 input leaf")
    # Preserve blocked-V4 CAT isolation: reject lexical CLI overrides now, but
    # do not stat or resolve the Phase-3 tree until a valid V4 lock exists.
    phase3_dir = _pin_canonical_path(
        phase3_dir,
        DEFAULT_PHASE3_DIR,
        label="Phase-3 input leaf",
        inspect_filesystem=False,
    )
    tracker = base.SourceTracker()
    config = _validate_config(config_path, tracker)
    v4_decision, v4_manifest = base._load_v4_terminal(v4_dir, tracker)
    numerical = base._load_numerical_tables(v4_dir, v4_decision, tracker)
    if v4_decision.get("passed") is not True:
        return base.TerminalInputs(
            terminal_state="blocked_after_v4_numerical_followup",
            config_path=config_path,
            v4_dir=v4_dir,
            phase3_dir=phase3_dir,
            config=config,
            v4_decision=v4_decision,
            v4_manifest=v4_manifest,
            v4_lock=None,
            phase3_decision=None,
            phase3_manifest=None,
            numerical=numerical,
            phase3=None,
            tracker=tracker,
        )
    phase3_dir = _pin_canonical_path(phase3_dir, DEFAULT_PHASE3_DIR, label="Phase-3 input leaf")
    lock = base._load_v4_lock(v4_dir, v4_decision, v4_manifest, tracker)
    decision, manifest, tables = _load_phase3_terminal(
        phase3_dir, config_path, config, v4_dir, lock, tracker
    )
    selection_only = not tables["outcomes_available"]
    passed = decision.get("phase3_pass") is True
    state = (
        "blocked_after_phase3_selection"
        if selection_only
        else "phase3_pass_phase4_pending"
        if passed
        else "blocked_after_phase3_validation"
    )
    return base.TerminalInputs(
        terminal_state=state,
        config_path=config_path,
        v4_dir=v4_dir,
        phase3_dir=phase3_dir,
        config=config,
        v4_decision=v4_decision,
        v4_manifest=v4_manifest,
        v4_lock=lock,
        phase3_decision=decision,
        phase3_manifest=manifest,
        numerical=numerical,
        phase3=tables,
        tracker=tracker,
    )


def _figure_plan(inputs: base.TerminalInputs) -> list[tuple[str, str, str]]:
    return base._figure_plan(inputs)


def _plot_calibration_selection_2pl(inputs: base.TerminalInputs, path: Path) -> None:
    """Render the inherited selection panel with a dynamic two-spec x-axis."""

    assert inputs.phase3 is not None
    inner = inputs.phase3["inner"]
    spec_count = len(PHASE3_SPEC_IDS)
    x = range(spec_count)
    figure, axes = base.plt.subplots(1, 2, figsize=(12.5, 4.8))
    finite_counts = pd.Series(0, index=list(PHASE3_SPEC_IDS), dtype=int)
    if "mean_log_loss" in inner:
        loss = pd.to_numeric(inner["mean_log_loss"], errors="coerce")
        means = loss.groupby(inner["spec_id"], sort=False).mean().reindex(PHASE3_SPEC_IDS)
        errors = loss.groupby(inner["spec_id"], sort=False).sem().reindex(PHASE3_SPEC_IDS).fillna(0)
        finite_counts = (
            loss.groupby(inner["spec_id"], sort=False)
            .count()
            .reindex(PHASE3_SPEC_IDS)
            .fillna(0)
            .astype(int)
        )
        axes[0].errorbar(x, means, yerr=errors, fmt="o", capsize=4, color=base.COLORS["blue"])
    else:
        axes[0].text(
            0.5,
            0.5,
            "Aggregate inner loss unavailable\n(selection evidence incomplete)",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            color=base.COLORS["red"],
        )
    axes[0].set_xticks(
        x,
        [SPEC_LABELS[spec_id] for spec_id in PHASE3_SPEC_IDS],
        rotation=15,
        ha="right",
    )
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
    selected_counts = [counts[spec_id] for spec_id in PHASE3_SPEC_IDS]
    axes[1].bar(x, selected_counts, color=[base.COLORS["green"], base.COLORS["purple"]])
    axes[1].axhline(20, color=base.COLORS["red"], linestyle=":", label="80% modal lock (20/25)")
    axes[1].set_xticks(
        x,
        [SPEC_LABELS[spec_id] for spec_id in PHASE3_SPEC_IDS],
        rotation=15,
        ha="right",
    )
    axes[1].set(
        ylabel="selected outer panels (of 25)",
        title=f"One-SE simplicity-rule selections ({sum(counts.values())}/25 locked)",
    )
    axes[1].legend()
    figure.suptitle(
        "Calibration specification selection — outer outcomes never used\n"
        f"{base._terminal_banner(inputs)}",
        fontsize=13,
    )
    figure.tight_layout()
    base._save(figure, path)


def _render_figures(inputs: base.TerminalInputs, figures_dir: Path) -> list[Path]:
    plan = _figure_plan(inputs)
    by_name = {name: figures_dir / name for name, _title, _description in plan}
    # Numerical evidence always retains the full three-spec V4 roster.
    base._plot_item_parameters(inputs, by_name["01_candidate_item_parameter_distributions.png"])
    base._plot_fit_validity(inputs, by_name["02_fit_convergence_and_start_agreement.png"])
    base._plot_parameter_equivalence(inputs, by_name["03_refit_parameter_equivalence.png"])
    base._plot_prediction_equivalence(inputs, by_name["04_heldout_prediction_equivalence.png"])
    base._plot_refit_theta(inputs, by_name["05_refit_theta_stability.png"])
    if "06_fixed_bank_eap_stability.png" in by_name:
        base._plot_fixed_bank(inputs, by_name["06_fixed_bank_eap_stability.png"])
    if inputs.phase3 is not None:
        _plot_calibration_selection_2pl(inputs, by_name["07_calibration_spec_inner_selection.png"])
        with _phase3_plot_bindings():
            if inputs.phase3["outcomes_available"]:
                base._plot_cat_gates(inputs, by_name["08_cat_gate_outcomes.png"])
                base._plot_policy_sensitivities(
                    inputs, by_name["09_prespecified_policy_sensitivities.png"]
                )
                base._plot_recovery(
                    inputs,
                    by_name[
                        next(name for name in by_name if name.startswith("10_oos_ability_recovery"))
                    ],
                )
                base._plot_pass_rate(
                    inputs,
                    by_name[next(name for name in by_name if name.startswith("11_oos_pass_rate"))],
                )
                base._plot_cat_vs_random(
                    inputs,
                    by_name[next(name for name in by_name if name.startswith("12_cat_vs_random"))],
                )
                base._plot_coverage(
                    inputs,
                    by_name[next(name for name in by_name if name.startswith("13_outer_coverage"))],
                )
    paths = [by_name[name] for name, _title, _description in plan]
    _require(all(path.is_file() for path in paths), "not every planned figure was rendered")
    return paths


def _gate_status(inputs: base.TerminalInputs) -> dict[str, Any]:
    phase3_pass = bool(inputs.phase3_decision and inputs.phase3_decision.get("phase3_pass") is True)
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": _utcnow(),
        "overall_status": inputs.terminal_state,
        "release_claims_authorized": False,
        "final_policy_valid": False,
        "gates": {
            "v4_numerical_verification": {
                "status": "passed" if inputs.v4_lock is not None else "blocked",
                "spec_ids": list(V4_SPEC_IDS),
            },
            "phase3_repeated_nested_cat": {
                "status": "passed" if phase3_pass else "failed_or_not_run",
                "candidate_spec_ids": list(PHASE3_SPEC_IDS),
                "inner_spec_fold_blocks_expected": phase3.EXPECTED_TOTAL_SPEC_FOLD_BLOCKS,
                "phase4_authorized": phase3_pass,
            },
            "phase4_total_uncertainty_and_order": {
                "status": "pending_not_run" if phase3_pass else "not_run_not_authorized"
            },
            "final_fit_export_and_replay": {"status": "not_run_not_authorized"},
        },
        "limitations": [
            "secondary same-cohort internal-development follow-up after V3",
            "conditional on frozen Qwen labels not human-validated on InFoBench",
            "no unseen-family or independent-confirmation claim is authorized",
        ],
    }


def _commands(inputs: base.TerminalInputs) -> str:
    lines = [
        "# Run from the eduLLM-Evals repository root.",
        "",
        "# V4 fresh run (use --resume for a partially completed identical run).",
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
    if inputs.terminal_state == "blocked_after_v4_numerical_followup":
        lines.extend(
            [
                "# Phase 3 is not authorized because V4 produced no valid lock.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "# Phase 3 (only after the valid V4 lock).",
                "uv run --frozen --extra irt python "
                "scripts/nested_scenario_cat_cv_2pl_only_v1.py \\",
                "  --config configs/infobench_calibration_cat_2pl_only_v1.json \\",
                "  --split-manifest configs/infobench_v2_splits.manifest.json \\",
                "  --out-dir runs/calibration/InFoBench_2pl_only_v1/phase3 \\",
                "  --numerical-followup-config configs/infobench_v2_numerical_followup_v4.json \\",
                "  --numerical-lock runs/calibration/InFoBench_v2_numerical_followup_v4/"
                "numerical_followup_lock.json",
                "",
            ]
        )
    if inputs.terminal_state == "phase3_pass_phase4_pending":
        lines.extend(
            [
                "# Authorized only by the complete passing 2PL-only Phase-3 decision.",
                "uv run --frozen --extra irt python "
                "scripts/nested_cat_total_uncertainty_2pl_only_v1.py \\",
                "  --config configs/infobench_calibration_cat_2pl_only_v1.json \\",
                "  --phase3-dir runs/calibration/InFoBench_2pl_only_v1/phase3 \\",
                "  --out-dir runs/calibration/InFoBench_2pl_only_v1/phase4 \\",
                "  --numerical-followup-config configs/infobench_v2_numerical_followup_v4.json \\",
                "  --numerical-lock runs/calibration/InFoBench_v2_numerical_followup_v4/"
                "numerical_followup_lock.json",
                "",
                "# Final fit remains unavailable until this Phase 4 completes and passes.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "# Phase 4 and final fit are not authorized by this terminal state.",
                "",
            ]
        )
    lines.extend(
        [
            "# Render the terminal state after V4 blocks or Phase 3 completes.",
            "uv run --frozen --extra irt python "
            "scripts/render_infobench_2pl_only_v1_terminal.py \\",
            "  --config configs/infobench_calibration_cat_2pl_only_v1.json \\",
            "  --v4-dir runs/calibration/InFoBench_v2_numerical_followup_v4 \\",
            "  --phase3-dir runs/calibration/InFoBench_2pl_only_v1/phase3 \\",
            "  --out-dir reports/infobench_calibration_cat_2pl_only_v1",
        ]
    )
    return "\n".join(lines) + "\n"


def _summary(inputs: base.TerminalInputs) -> str:
    derived = (inputs.phase3 or {}).get("derived_denominators") or {}
    phase4_authorized = bool(
        inputs.phase3_decision and inputs.phase3_decision.get("phase4_authorized") is True
    )
    return f"""# InFoBench 2PL-only-v1 terminal report

**Status:** `{inputs.terminal_state}`

This append-only follow-up compares only `log_shrinkage_2pl_lambda16` and
`log_shrinkage_2pl_lambda4`. The inherited numerical evidence retains the full
three-spec V4 roster, including 1PL, but 1PL is not a Phase-3 candidate here.

- Phase-3 panels complete: {derived.get("selection_panels_complete", 0)}/25.
- Valid inner spec-fold blocks: {derived.get("spec_fold_blocks_valid", 0)}/200.
- Phase 4 authorized: {phase4_authorized}.

This is same-cohort internal-development evidence after V3, conditional on the
frozen Qwen labels. It is not independent confirmation.
"""


def _guide(plan: Sequence[tuple[str, str, str]]) -> str:
    lines = ["# Figure guide", ""]
    lines.extend(f"- `{name}` — {title}: {description}" for name, title, description in plan)
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_terminal_report(
    *, config_path: Path, v4_dir: Path, phase3_dir: Path, output_dir: Path
) -> list[Path]:
    config_path = _pin_canonical_path(config_path, DEFAULT_CONFIG, label="renderer config")
    v4_dir = _pin_canonical_path(v4_dir, DEFAULT_V4_DIR, label="V4 input leaf")
    phase3_dir = _pin_canonical_path(
        phase3_dir,
        DEFAULT_PHASE3_DIR,
        label="Phase-3 input leaf",
        inspect_filesystem=False,
    )
    output_dir = _pin_canonical_path(output_dir, DEFAULT_OUTPUT_DIR, label="renderer output leaf")
    _require(
        not output_dir.exists() or (output_dir.is_dir() and not any(output_dir.iterdir())),
        "renderer output must be absent or empty",
    )
    inputs = load_terminal_inputs(config_path=config_path, v4_dir=v4_dir, phase3_dir=phase3_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.rendering-", dir=output_dir.parent))
    try:
        figures_dir = staging / "figures"
        base._style()
        figures = _render_figures(inputs, figures_dir)
        plan = _figure_plan(inputs)
        _write(staging / "EXECUTION_SUMMARY.md", _summary(inputs))
        _write(
            staging / "gate_status.json",
            json.dumps(_gate_status(inputs), indent=2, sort_keys=True) + "\n",
        )
        _write(staging / "reproduction_commands.txt", _commands(inputs))
        _write(figures_dir / "FIGURE_GUIDE.md", _guide(plan))
        source_hashes = {_display(path): _sha256(path) for path in sorted(inputs.tracker.paths)}
        derived = (inputs.phase3 or {}).get("derived_denominators") or {}
        manifest = {
            "schema_version": FIGURE_MANIFEST_SCHEMA,
            "generated_at_utc": _utcnow(),
            "terminal_state": inputs.terminal_state,
            "renderer": _display(Path(__file__)),
            "renderer_sha256": _sha256(Path(__file__)),
            "source_files": source_hashes,
            "figure_files": {path.name: _sha256(path) for path in figures},
            "v4_numerical_spec_ids": list(V4_SPEC_IDS),
            "phase3_candidate_spec_ids": list(PHASE3_SPEC_IDS),
            "denominators": {
                "v4_fits": base.EXPECTED_FITS,
                "v4_start_comparisons": base.EXPECTED_STARTS,
                "phase3_selection_panels_expected": phase3.EXPECTED_PANELS,
                "phase3_selection_panels_complete": derived.get("selection_panels_complete", 0),
                "phase3_spec_fold_blocks_expected": phase3.EXPECTED_TOTAL_SPEC_FOLD_BLOCKS,
                "phase3_spec_fold_blocks_valid": derived.get("spec_fold_blocks_valid", 0),
            },
            "parent_v3_artifacts_opened": False,
            "old_images_used_as_data": False,
        }
        _write(
            figures_dir / "figure_source_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        relative = [
            Path("EXECUTION_SUMMARY.md"),
            Path("gate_status.json"),
            Path("reproduction_commands.txt"),
            Path("figures/FIGURE_GUIDE.md"),
            Path("figures/figure_source_manifest.json"),
            *[Path("figures") / path.name for path in figures],
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "figures").mkdir(parents=True, exist_ok=True)
        for item in relative:
            os.replace(staging / item, output_dir / item)
        actual = {path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file()}
        _require(actual == set(relative), "renderer output inventory differs after publication")
        return [output_dir / item for item in relative]
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
    except (
        TerminalReportError,
        base.TerminalReportError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
