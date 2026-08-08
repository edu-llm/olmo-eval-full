"""Phase-4 uncertainty and order validation for the frozen InFoBench V3 study.

This is a versioned V3 adapter around the previously tested Phase-4
calculation engine.  It accepts only a complete, passing V3 Phase-3 handoff
whose V4 dense-fitter lock, evidence, configuration, and provenance chain are
still valid.  It then evaluates only the preregistered primary CAT policy.

The design remains frozen at 100 outer-training-family bootstrap refits per
panel and 20 order seeds.  Sensitivity policies cannot enter Phase 4, failed
bootstrap refits are retained in the valid-draw denominator, and no fallback
policy or calibration specification is permitted.  Existing output is never
overwritten: an interrupted byte-identical study requires ``--resume`` and a
new attempt requires a new versioned output leaf.

``--plan-only`` is read-only.  This module does not launch Phase 4 as part of
tests or import-time validation.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import nested_cat_total_uncertainty_v2 as engine  # noqa: E402
from scripts import nested_scenario_cat_cv as phase3_v1  # noqa: E402
from scripts import nested_scenario_cat_cv_v3 as phase3  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402
from scripts import scenario_kfold_estimator_cv as scenario_cv  # noqa: E402

SCRIPT_SCHEMA = "infobench-nested-cat-total-uncertainty-v3-v1"
CHECKPOINT_SCHEMA = "infobench-v3-phase4-checkpoint-v1"
BOOTSTRAP_CACHE_SCHEMA = "infobench-v3-phase4-bootstrap-fit-v1"
DECISION_SCHEMA = "infobench-v3-phase4-decision-v1"

DEFAULT_CONFIG = ROOT / "configs" / "infobench_calibration_cat_v3.json"
DEFAULT_PHASE3 = ROOT / "runs" / "calibration" / "InFoBench_v3" / "phase3"
DEFAULT_OUTPUT = ROOT / "runs" / "calibration" / "InFoBench_v3" / "phase4"
DEFAULT_NUMERICAL_FOLLOWUP_CONFIG = phase3.DEFAULT_NUMERICAL_FOLLOWUP_CONFIG
DEFAULT_NUMERICAL_LOCK = phase3.DEFAULT_NUMERICAL_LOCK

EXPECTED_REPEATS = 5
EXPECTED_OUTER_FOLDS = 5
EXPECTED_MODELS = 52
EXPECTED_PANELS = EXPECTED_REPEATS * EXPECTED_OUTER_FOLDS
PRIMARY_POLICY_ID = phase3.PRIMARY_POLICY_ID
MAXIMUM_INNER_GRADIENT = 1e-6
INFOBENCH_SKILLS = ("content", "format", "number", "style", "linguistic")

CODE_DEPENDENCIES = (
    ROOT / "scripts" / "nested_cat_total_uncertainty_v3.py",
    ROOT / "scripts" / "nested_cat_total_uncertainty_v2.py",
    ROOT / "scripts" / "nested_scenario_cat_cv_v3.py",
    ROOT / "scripts" / "nested_scenario_cat_cv.py",
    ROOT / "scripts" / "calibrate_partial.py",
    ROOT / "scripts" / "calibrate_mirt.py",
    ROOT / "scripts" / "kfold_cv_mirt.py",
    ROOT / "scripts" / "scenario_cat_lib.py",
    ROOT / "scripts" / "scenario_kfold_estimator_cv.py",
    ROOT / "scripts" / "run_infobench_v2_numerical_followup_v4.py",
    ROOT / "tutor_cat" / "mirt.py",
    ROOT / "tutor_cat" / "engine.py",
    ROOT / "tutor_cat" / "selector.py",
    ROOT / "tutor_cat" / "dataio.py",
    ROOT / "tutor_cat" / "schemas.py",
    ROOT / "tutor_cat" / "skill_structure.py",
    ROOT / "tutor_cat" / "__init__.py",
)


class V3Phase4Error(engine.V2Phase4Error):
    """A V3 frozen-design, provenance, leakage, or completeness check failed."""


# The numerical calculations and immutable record types are shared with the
# tested V2 engine.  V3-specific authorization and fit provenance are below.
FamilyBootstrapSample = engine.FamilyBootstrapSample
Panel = engine.Panel
BankBundle = engine.BankBundle
law_total_variance = engine.law_total_variance
combine_cat_and_full_total_se = engine.combine_cat_and_full_total_se
order_stability_frame = engine.order_stability_frame
repetition_gate_results = engine.repetition_gate_results
aggregate_repeated_models = engine.aggregate_repeated_models
family_cluster_bootstrap_summary = engine.family_cluster_bootstrap_summary
draw_outer_training_family_sample = engine.draw_outer_training_family_sample
assert_fit_boundary = engine.assert_fit_boundary
validate_order_seeds = engine.validate_order_seeds
replay_primary_model = engine.replay_primary_model

_ENGINE_AUTHORIZATION = engine.validate_phase3_authorization
_ENGINE_NUMERICAL_PROFILE = engine.require_phase3_numerical_profile
_REAL_SHA256 = engine._sha256
_BINDING_LOCK = threading.RLock()


@contextmanager
def _v3_engine_bindings(*, config_path: Path | None = None) -> Iterator[None]:
    """Bind the generic V2 engine to V3 schemas for one scoped operation.

    Bindings are restored even on error, so importing this module cannot alter
    historical V2 behavior.  The one in-memory output alias lets the V2 engine
    recognize V3's stronger ``never_overwrite_v1_or_v2`` protection; the file
    on disk and its hash are never modified.
    """

    names = {
        "phase3": phase3,
        "SCRIPT_SCHEMA": SCRIPT_SCHEMA,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
        "BOOTSTRAP_CACHE_SCHEMA": BOOTSTRAP_CACHE_SCHEMA,
        "DECISION_SCHEMA": DECISION_SCHEMA,
        "PRIMARY_POLICY_ID": PRIMARY_POLICY_ID,
        "CODE_DEPENDENCIES": CODE_DEPENDENCIES,
        "V2Phase4Error": V3Phase4Error,
    }
    with _BINDING_LOCK:
        previous = {name: getattr(engine, name) for name in names}
        previous_reader = engine._read_json
        previous_skills = cm.SKILLS
        previous_n_skills = cm.N_SKILLS
        for name, value in names.items():
            setattr(engine, name, value)
        cm.SKILLS = INFOBENCH_SKILLS
        cm.N_SKILLS = len(INFOBENCH_SKILLS)

        if config_path is not None:
            resolved_config = config_path.resolve()

            def read_json_with_v3_output_guard(path: Path) -> dict[str, Any]:
                payload = previous_reader(path)
                if path.resolve() == resolved_config:
                    outputs = dict(payload.get("outputs") or {})
                    if outputs.get("never_overwrite_v1_or_v2") is True:
                        outputs["never_overwrite_v1"] = True
                        payload = {**payload, "outputs": outputs}
                return payload

            engine._read_json = read_json_with_v3_output_guard
        try:
            yield
        finally:
            engine._read_json = previous_reader
            cm.SKILLS = previous_skills
            cm.N_SKILLS = previous_n_skills
            for name, value in previous.items():
                setattr(engine, name, value)


def validate_phase3_authorization(
    decision: Mapping[str, Any],
    selected: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Fail closed unless the V3 Phase-3 decision explicitly authorizes Phase 4."""

    with _v3_engine_bindings():
        _ENGINE_AUTHORIZATION(decision, selected, manifest)


def require_phase3_numerical_profile(
    manifest: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    """Require Phase 4 to consume the byte-semantic V4 profile used in Phase 3."""

    with _v3_engine_bindings():
        _ENGINE_NUMERICAL_PROFILE(manifest, profile)


def validate_phase3_artifact_inventory(
    phase3_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    """Hash-check every canonical V3 Phase-3 artifact and return its inventory."""

    directory = phase3_dir.resolve()
    if manifest.get("schema_version") != phase3.SCRIPT_SCHEMA:
        raise V3Phase4Error("Phase-3 manifest is not the V3 driver schema")
    if manifest.get("status") != "phase3_complete":
        raise V3Phase4Error("Phase 3 is not complete")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise V3Phase4Error("Phase-3 manifest lacks its output inventory")

    inventory: dict[str, str] = {}
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise V3Phase4Error("Phase-3 manifest file is missing")
    if engine._read_json(manifest_path) != dict(manifest):
        raise V3Phase4Error("Phase-3 manifest changed after it was loaded")
    inventory["manifest.json"] = _REAL_SHA256(manifest_path)
    for name in phase3.REQUIRED_OUTPUTS:
        if name == "manifest.json":
            continue
        path = directory / name
        raw = outputs.get(name)
        if not path.is_file() or not isinstance(raw, Mapping):
            raise V3Phase4Error(f"required V3 Phase-3 artifact is missing: {name}")
        recorded_path = engine._repo_path(str(raw.get("path") or ""))
        recorded_hash = str(raw.get("sha256") or "")
        observed_hash = _REAL_SHA256(path)
        if recorded_path != path.resolve() or recorded_hash != observed_hash:
            raise V3Phase4Error(f"V3 Phase-3 artifact failed provenance: {name}")
        inventory[name] = observed_hash

    decision = directory / "phase3_decision.json"
    mirror = directory / "v3_decision.json"
    if decision.read_bytes() != mirror.read_bytes():
        raise V3Phase4Error("V3 decision mirror differs from phase3_decision.json")
    return dict(sorted(inventory.items()))


def validate_captured_path_hashes(captured: Mapping[str, str]) -> None:
    """Fail when any frozen file differs from its initialization-time hash."""

    for raw_path, expected_hash in sorted(captured.items()):
        path = Path(raw_path)
        if not path.is_file() or _REAL_SHA256(path) != str(expected_hash):
            raise V3Phase4Error(f"frozen provenance changed after initialization: {path}")


def validate_frozen_phase4_design(config: Mapping[str, Any]) -> None:
    """Assert that no V3 policy, bootstrap, order, or gate setting drifted."""

    try:
        phase3._check_frozen_config(config)
    except phase3.V3Phase3Error as error:
        raise V3Phase4Error(str(error)) from error
    primary = (config.get("cat_policies") or {}).get("primary") or {}
    expected_primary = {
        "policy_id": PRIMARY_POLICY_ID,
        "role": "primary",
        "minimum_scenarios": 15,
        "conditional_se_target": 0.2,
        "selector": "trace",
    }
    if primary != expected_primary:
        raise V3Phase4Error("the frozen primary CAT policy changed")
    cat = config.get("cat_policies") or {}
    if cat.get("sensitivities_can_replace_primary") is not False:
        raise V3Phase4Error("a sensitivity policy may replace the primary policy")
    selection = config.get("calibration_selection") or {}
    gates = config.get("selection_gates") or {}
    if selection.get("allow_fallback") is not False:
        raise V3Phase4Error("calibration selection permits a fallback")
    if gates.get("allow_fallback_if_primary_fails") is not False:
        raise V3Phase4Error("CAT selection permits a fallback")

    uncertainty = config.get("uncertainty") or {}
    expected = {
        "phase4_runs_only_after_complete_phase3_pass": True,
        "parameter_bootstrap_replicates": 100,
        "parameter_bootstrap_unit": "outer_training_model_family",
        "minimum_valid_parameter_bootstrap_rate": 0.9,
        "total_variance_method": "law_of_total_variance",
        "absolute_p90_total_se_maximum": 0.5,
        "full_administration_total_se_ratio_is_diagnostic_only": True,
        "maximum_median_order_path_sd": 0.2,
        "require_every_seed_present_and_valid": True,
    }
    for field, value in expected.items():
        if uncertainty.get(field) != value:
            raise V3Phase4Error(f"frozen uncertainty.{field} changed")
    with _v3_engine_bindings():
        order_seeds = validate_order_seeds(uncertainty.get("order_seeds") or [])
    if order_seeds != tuple(range(1000, 1020)):
        raise V3Phase4Error("the exact frozen order seeds changed")

    runtime = config.get("runtime") or {}
    exact_runtime = {
        "master_seed": 20260805,
        "mwle_ridge": 1e-6,
        "negative_loading_policy": "drop",
        "metric_family_bootstrap_replicates": 2000,
    }
    for field, value in exact_runtime.items():
        if runtime.get(field) != value:
            raise V3Phase4Error(f"frozen runtime.{field} changed")
    exact_cat = {
        "estimator": "mwle",
        "top_n": 5,
        "maximum_adaptive_scenarios": 50,
        "minimum_scored_criteria": 15,
    }
    for field, value in exact_cat.items():
        if cat.get(field) != value:
            raise V3Phase4Error(f"frozen cat_policies.{field} changed")
    outputs = config.get("outputs") or {}
    if outputs.get("never_overwrite_v1_or_v2") is not True:
        raise V3Phase4Error("V3 output does not protect historical V1/V2 evidence")
    official_output = engine._repo_path(str(outputs.get("phase4") or ""))
    if official_output != DEFAULT_OUTPUT.resolve():
        raise V3Phase4Error("the official V3 Phase-4 output path changed")


def validate_phase3_cross_links(
    *,
    manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
    selected: Mapping[str, Any],
    selected_path: Path,
    primary_policy: Mapping[str, Any],
) -> None:
    """Validate decision summaries and selected-handoff links in both directions."""

    embedded = manifest.get("phase3_decision") or {}
    if embedded != {
        "status": "pass",
        "phase3_pass": True,
        "phase4_authorized": True,
    }:
        raise V3Phase4Error("Phase-3 manifest does not embed a passing authorization")
    selected_primary = selected.get("primary_policy")
    decision_primary = (decision.get("primary_policy") or {}).get("policy")
    expected_primary = dict(primary_policy)
    if selected_primary != expected_primary or decision_primary != expected_primary:
        raise V3Phase4Error("Phase-3 primary-policy cross-link changed")
    selected_ref = decision.get("selected_calibration_specs")
    if not isinstance(selected_ref, Mapping):
        raise V3Phase4Error("Phase-3 decision lacks its selected-spec handoff")
    referenced_path = engine._repo_path(str(selected_ref.get("path") or ""))
    if referenced_path != selected_path.resolve():
        raise V3Phase4Error("Phase-3 decision selected-spec path changed")
    if selected_ref.get("sha256") != _REAL_SHA256(selected_path):
        raise V3Phase4Error("Phase-3 decision selected-spec hash changed")


def dense_fit_validity(
    fit: Mapping[str, Any], spec: phase3.CalibrationSpec
) -> dict[str, Any]:
    """Audit one fit against the complete V4 dense-fitter contract."""

    arrays = [
        np.asarray(fit.get("A"), dtype=float),
        np.asarray(fit.get("b"), dtype=float),
        np.asarray(fit.get("R"), dtype=float),
    ]
    arrays_and_objective_finite = bool(
        all(np.all(np.isfinite(value)) for value in arrays)
        and math.isfinite(float(fit.get("penalized_objective", math.nan)))
    )
    items = list(map(str, fit.get("items") or []))
    dim_labels = list(map(str, fit.get("dim_labels") or []))
    raw_n_iter = fit.get("n_iter")
    valid_iteration_count = bool(
        isinstance(raw_n_iter, (int, np.integer))
        and not isinstance(raw_n_iter, (bool, np.bool_))
        and 1 <= int(raw_n_iter) <= 1500
    )
    array_shapes_and_identification_valid = bool(
        items
        and len(items) == len(set(items))
        and dim_labels == ["instruction_following"]
        and arrays[0].shape == (len(items), 1)
        and arrays[1].shape == (len(items),)
        and arrays[2].shape == (1, 1)
        and np.array_equal(arrays[2], np.ones((1, 1)))
        and valid_iteration_count
    )
    active = arrays[0][np.abs(arrays[0]) > 0]
    loadings_valid = bool(
        active.size > 0
        and (
            np.allclose(active, 1.0, rtol=0, atol=1e-12)
            if spec.family == cm.ONE_PL
            else np.all(active > 0)
        )
    )
    diagnostics = fit.get("convergence_diagnostics") or {}
    trace = diagnostics.get("trace") or []
    optimizers = [row.get("mstep_optimizer") or {} for row in trace]
    all_item_optimizers_finished = bool(
        optimizers
        and all(
            int(optimizer.get("n_failed", 1)) == 0
            and int(optimizer.get("n_items", 0)) > 0
            and int(optimizer.get("n_converged", -1))
            == int(optimizer.get("n_items", 0))
            for optimizer in optimizers
        )
    )
    gradients = [
        float(optimizer.get("max_abs_gradient", math.nan))
        for optimizer in optimizers
    ]
    gradients_finite = bool(gradients and all(math.isfinite(value) for value in gradients))
    maximum_gradient = max(gradients) if gradients_finite else math.inf
    profile_valid = bool(
        fit.get("grid_nodes") == 401
        and fit.get("quadrature_method") == cm.NORMAL_TRAPEZOID_QUADRATURE
        and math.isclose(
            float(fit.get("quadrature_linear_bound", math.nan)),
            8.0,
            rel_tol=0,
            abs_tol=0,
        )
        and fit.get("convergence_mode") == cm.RETURNED_ITERATE_CONVERGENCE
        and math.isclose(
            float(diagnostics.get("objective_tolerance", math.nan)),
            1e-4,
            rel_tol=0,
            abs_tol=0,
        )
        and math.isclose(
            float(diagnostics.get("parameter_tolerance", math.nan)),
            5e-5,
            rel_tol=0,
            abs_tol=0,
        )
        and diagnostics.get("required_consecutive_passes") == 2
        and int(diagnostics.get("final_consecutive_passes", 0)) >= 2
    )
    returned_iterate_valid = bool(
        diagnostics.get("returned_iterate_matches_last_trace") is True
        and diagnostics.get("final_exact_recomputation") is True
        and diagnostics.get("stopped_before_extra_mstep") is True
        and diagnostics.get("all_objective_changes_monotone_within_tolerance") is True
    )
    exact_specification = fit.get("calibration_specification") == spec.canonical
    valid = bool(
        fit.get("converged") is True
        and arrays_and_objective_finite
        and array_shapes_and_identification_valid
        and loadings_valid
        and all_item_optimizers_finished
        and gradients_finite
        and maximum_gradient <= MAXIMUM_INNER_GRADIENT
        and profile_valid
        and returned_iterate_valid
        and exact_specification
    )
    return {
        "fit_valid": valid,
        "converged": fit.get("converged") is True,
        "arrays_and_objective_finite": arrays_and_objective_finite,
        "array_shapes_and_identification_valid": (
            array_shapes_and_identification_valid
        ),
        "valid_iteration_count": valid_iteration_count,
        "loadings_valid": loadings_valid,
        "all_item_optimizers_finished": all_item_optimizers_finished,
        "maximum_observed_inner_gradient": maximum_gradient,
        "maximum_allowed_inner_gradient": MAXIMUM_INNER_GRADIENT,
        "inner_gradient_within_tolerance": bool(
            gradients_finite and maximum_gradient <= MAXIMUM_INNER_GRADIENT
        ),
        "profile_valid": profile_valid,
        "returned_iterate_valid": returned_iterate_valid,
        "exact_specification": exact_specification,
    }


def _enrich_fit_from_manifest(
    fit: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **dict(fit),
        "grid_nodes": manifest.get("grid_nodes"),
        "quadrature_method": manifest.get("quadrature_method"),
        "quadrature_linear_bound": manifest.get("quadrature_linear_bound"),
        "convergence_mode": manifest.get("convergence_mode"),
        "penalized_objective": manifest.get("penalized_objective"),
        "convergence_diagnostics": manifest.get("convergence_diagnostics") or {},
    }


def _compact_convergence_contract(fit: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = fit.get("convergence_diagnostics") or {}
    return {
        "objective_tolerance": diagnostics.get("objective_tolerance"),
        "parameter_tolerance": diagnostics.get("parameter_tolerance"),
        "required_consecutive_passes": diagnostics.get(
            "required_consecutive_passes"
        ),
        "final_consecutive_passes": diagnostics.get("final_consecutive_passes"),
        "returned_iterate_matches_last_trace": diagnostics.get(
            "returned_iterate_matches_last_trace"
        ),
        "final_exact_recomputation": diagnostics.get("final_exact_recomputation"),
        "stopped_before_extra_mstep": diagnostics.get("stopped_before_extra_mstep"),
        "all_objective_changes_monotone_within_tolerance": diagnostics.get(
            "all_objective_changes_monotone_within_tolerance"
        ),
        "trace_iterations": len(diagnostics.get("trace") or []),
    }


def _stored_dense_audit_is_valid(manifest: Mapping[str, Any]) -> bool:
    validity = manifest.get("dense_fit_validity") or {}
    convergence = manifest.get("convergence_contract") or {}
    required_true = (
        "fit_valid",
        "converged",
        "arrays_and_objective_finite",
        "array_shapes_and_identification_valid",
        "valid_iteration_count",
        "loadings_valid",
        "all_item_optimizers_finished",
        "inner_gradient_within_tolerance",
        "profile_valid",
        "returned_iterate_valid",
        "exact_specification",
    )
    return bool(
        all(validity.get(field) is True for field in required_true)
        and math.isfinite(
            float(validity.get("maximum_observed_inner_gradient", math.nan))
        )
        and float(validity["maximum_observed_inner_gradient"])
        <= MAXIMUM_INNER_GRADIENT
        and validity.get("maximum_allowed_inner_gradient")
        == MAXIMUM_INNER_GRADIENT
        and manifest.get("grid_nodes") == 401
        and manifest.get("quadrature_method")
        == cm.NORMAL_TRAPEZOID_QUADRATURE
        and manifest.get("quadrature_linear_bound") == 8.0
        and manifest.get("convergence_mode")
        == cm.RETURNED_ITERATE_CONVERGENCE
        and math.isfinite(float(manifest.get("penalized_objective", math.nan)))
        and convergence.get("objective_tolerance") == 1e-4
        and convergence.get("parameter_tolerance") == 5e-5
        and convergence.get("required_consecutive_passes") == 2
        and int(convergence.get("final_consecutive_passes", 0)) >= 2
        and convergence.get("returned_iterate_matches_last_trace") is True
        and convergence.get("final_exact_recomputation") is True
        and convergence.get("stopped_before_extra_mstep") is True
        and convergence.get("all_objective_changes_monotone_within_tolerance")
        is True
        and int(convergence.get("trace_iterations", 0)) > 0
    )


@contextmanager
def _captured_runner_bindings(runner: V3Phase4Runner) -> Iterator[None]:
    """Use initialization-time hashes in every cache key and checkpoint context."""

    with _v3_engine_bindings():
        previous_sha256 = engine._sha256

        def captured_sha256(path: Path) -> str:
            resolved = str(path.resolve())
            captured = getattr(runner, "_captured_path_hashes", {})
            if resolved in captured:
                return str(captured[resolved])
            return previous_sha256(path)

        engine._sha256 = captured_sha256
        try:
            yield
        finally:
            engine._sha256 = previous_sha256


class V3Phase4Runner(engine.V2Phase4Runner):
    """V3 provenance adapter plus the frozen Phase-4 calculation engine."""

    def __init__(self, args: argparse.Namespace):
        with _v3_engine_bindings(config_path=args.config.resolve()):
            super().__init__(args)

    def _capture_provenance_snapshot(self) -> None:
        captured: dict[str, str] = {}

        def add(path: Path, expected: str | None = None) -> None:
            resolved = path.resolve()
            if not resolved.is_file():
                raise V3Phase4Error(f"frozen provenance file is missing: {resolved}")
            observed = _REAL_SHA256(resolved)
            if expected is not None and observed != expected:
                raise V3Phase4Error(f"frozen provenance hash mismatch: {resolved}")
            prior = captured.get(str(resolved))
            if prior is not None and prior != observed:
                raise V3Phase4Error(f"conflicting frozen hashes for: {resolved}")
            captured[str(resolved)] = observed

        add(self.config_path)
        add(
            self.split_path,
            str(
                (self.config.get("cross_validation") or {}).get(
                    "split_manifest_sha256"
                )
                or ""
            ),
        )
        for name, path in self.input_paths.items():
            add(path, self.input_hashes[name])
        for name, expected in self.phase3_artifact_hashes.items():
            add(self.phase3_dir / name, expected)
        for path in CODE_DEPENDENCIES:
            expected = self.code_hashes.get(engine._display_path(path))
            if not expected:
                raise V3Phase4Error(f"code dependency lacks captured provenance: {path}")
            add(path, expected)

        followup_path = self.args.numerical_followup_config.resolve()
        lock_path = self.args.numerical_lock.resolve()
        add(followup_path, str(self.numerical["followup_config_sha256"]))
        add(lock_path, str(self.numerical["verification_sha256"]))
        add(
            lock_path.parent / "numerical_followup_decision.json",
            str(self.numerical["followup_decision_sha256"]),
        )
        add(
            lock_path.parent / "study_manifest.json",
            str(self.numerical["followup_manifest_sha256"]),
        )
        lock = engine._read_json(lock_path)
        evidence_paths = lock.get("evidence_paths") or {}
        evidence_hashes = lock.get("evidence_sha256") or {}
        for key, expected in evidence_hashes.items():
            raw = str(evidence_paths.get(key) or "")
            candidate = Path(raw)
            if candidate.is_absolute():
                path = candidate.resolve()
            elif raw.startswith("runs/"):
                path = engine._repo_path(raw)
            else:
                path = (lock_path.parent / candidate).resolve()
            add(path, str(expected))

        for panel in self.panels:
            add(panel.outer_fit_manifest_path, panel.outer_fit_manifest_sha256)
            fit_manifest = engine._read_json(panel.outer_fit_manifest_path)
            add(
                panel.outer_fit_cache_dir / "fit_arrays.npz",
                str(fit_manifest.get("arrays_sha256") or ""),
            )

        self._captured_path_hashes = dict(sorted(captured.items()))
        self._captured_semantic_hashes = {
            "config": engine._canonical_hash(self.config),
            "phase3_manifest": engine._canonical_hash(self.phase3_manifest),
            "phase3_decision": engine._canonical_hash(self.phase3_decision),
            "selected": engine._canonical_hash(self.selected),
            "numerical": engine._canonical_hash(self.numerical),
            "code_hashes": engine._canonical_hash(self.code_hashes),
            "input_hashes": engine._canonical_hash(self.input_hashes),
            "primary_policy": engine._canonical_hash(asdict(self.primary_policy)),
        }
        self._captured_provenance_sha256 = engine._canonical_hash(
            {
                "paths": self._captured_path_hashes,
                "semantics": self._captured_semantic_hashes,
            }
        )

    def _revalidate_frozen_provenance(self) -> None:
        validate_captured_path_hashes(self._captured_path_hashes)
        semantics = {
            "config": engine._canonical_hash(self.config),
            "phase3_manifest": engine._canonical_hash(self.phase3_manifest),
            "phase3_decision": engine._canonical_hash(self.phase3_decision),
            "selected": engine._canonical_hash(self.selected),
            "numerical": engine._canonical_hash(self.numerical),
            "code_hashes": engine._canonical_hash(self.code_hashes),
            "input_hashes": engine._canonical_hash(self.input_hashes),
            "primary_policy": engine._canonical_hash(asdict(self.primary_policy)),
        }
        if semantics != self._captured_semantic_hashes:
            raise V3Phase4Error("frozen in-memory provenance changed after initialization")
        inventory = validate_phase3_artifact_inventory(
            self.phase3_dir, self.phase3_manifest
        )
        if inventory != self.phase3_artifact_hashes:
            raise V3Phase4Error("Phase-3 artifact inventory changed after initialization")
        validate_frozen_phase4_design(self.config)
        validate_phase3_cross_links(
            manifest=self.phase3_manifest,
            decision=self.phase3_decision,
            selected=self.selected,
            selected_path=self.selected_path,
            primary_policy=asdict(self.primary_policy),
        )
        strict_numerical = phase3._validate_numerical_lock(
            self.config,
            list(self.specs.values()),
            base_config_path=self.config_path,
            lock_path=self.args.numerical_lock,
            followup_config_path=self.args.numerical_followup_config,
            allow_pending=False,
            enforce_configured_paths=True,
        )
        if strict_numerical != self.numerical:
            raise V3Phase4Error("V4 provenance chain changed after initialization")
        actual_code_hashes = {
            engine._display_path(path): _REAL_SHA256(path)
            for path in CODE_DEPENDENCIES
        }
        if actual_code_hashes != self.code_hashes:
            raise V3Phase4Error("Phase-4 code changed after initialization")

    def _validate_phase3_runtime(self) -> None:
        with _v3_engine_bindings():
            super()._validate_phase3_runtime()
        validate_frozen_phase4_design(self.config)
        self.phase3_artifact_hashes = validate_phase3_artifact_inventory(
            self.phase3_dir, self.phase3_manifest
        )
        validate_phase3_cross_links(
            manifest=self.phase3_manifest,
            decision=self.phase3_decision,
            selected=self.selected,
            selected_path=self.selected_path,
            primary_policy=asdict(self.primary_policy),
        )
        strict_numerical = phase3._validate_numerical_lock(
            self.config,
            list(self.specs.values()),
            base_config_path=self.config_path,
            lock_path=self.args.numerical_lock,
            followup_config_path=self.args.numerical_followup_config,
            allow_pending=False,
            enforce_configured_paths=True,
        )
        if strict_numerical != self.numerical:
            raise V3Phase4Error("V4 numerical authorization changed after preflight")
        runtime = self.phase3_manifest.get("runtime") or {}
        expected = {
            "fit_parameter_tolerance": 5e-5,
            "fit_consecutive_convergence_passes": 2,
            "fit_convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
            "fit_quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
            "fit_linear_bound": 8.0,
        }
        for field, value in expected.items():
            observed = runtime.get(field)
            if isinstance(value, float):
                matches = math.isclose(
                    float(observed), value, rel_tol=0, abs_tol=0
                )
            else:
                matches = observed == value
            if not matches:
                raise V3Phase4Error(f"Phase-3 dense runtime changed {field}")

    def _preflight_base_fits(self) -> None:
        observed = validate_phase3_artifact_inventory(
            self.phase3_dir, self.phase3_manifest
        )
        if observed != self.phase3_artifact_hashes:
            raise V3Phase4Error("Phase-3 artifact inventory changed during preflight")
        with _captured_runner_bindings(self):
            super()._preflight_base_fits()

    def _load_phase3_fit(self, panel: Panel) -> dict[str, Any]:
        with _captured_runner_bindings(self):
            fit = super()._load_phase3_fit(panel)
        manifest = engine._read_json(panel.outer_fit_manifest_path)
        enriched = _enrich_fit_from_manifest(fit, manifest)
        validity = dense_fit_validity(enriched, panel.selected_spec)
        if validity["fit_valid"] is not True:
            raise V3Phase4Error(
                f"outer fit violates the V4 dense-fitter contract: {panel.panel_id}"
            )
        return enriched

    def _signature(self) -> str:
        if not hasattr(self, "_captured_path_hashes"):
            self._capture_provenance_snapshot()
        with _captured_runner_bindings(self):
            inherited = super()._signature()
        return engine._canonical_hash(
            {
                "schema_version": SCRIPT_SCHEMA,
                "inherited_calculation_signature": inherited,
                "phase3_artifact_hashes": getattr(
                    self, "phase3_artifact_hashes", {}
                ),
                "v4_numerical_authorization": self.numerical,
                "captured_provenance_sha256": self._captured_provenance_sha256,
                "dense_fit_contract": {
                    "fit_grid": 401,
                    "fit_quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
                    "fit_linear_bound": 8.0,
                    "fit_convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
                    "fit_max_iter": 1500,
                    "fit_objective_tolerance": 1e-4,
                    "fit_parameter_tolerance": 5e-5,
                    "fit_consecutive_convergence_passes": 2,
                    "maximum_inner_gradient": MAXIMUM_INNER_GRADIENT,
                },
            }
        )

    def _bootstrap_cache_key(
        self, panel: Panel, replicate: int, sample: FamilyBootstrapSample
    ) -> str:
        with _captured_runner_bindings(self):
            return super()._bootstrap_cache_key(panel, replicate, sample)

    def _checkpoint_context(
        self,
        kind: str,
        panel: Panel,
        index: int,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        with _captured_runner_bindings(self):
            context = super()._checkpoint_context(kind, panel, index, extra)
        context["captured_provenance_sha256"] = self._captured_provenance_sha256
        return context

    def _prepare_output(self) -> None:
        self._revalidate_frozen_provenance()
        with _captured_runner_bindings(self):
            super()._prepare_output()

    def _load_bootstrap_fit(
        self,
        panel: Panel,
        replicate: int,
        sample: FamilyBootstrapSample,
        cache_key: str,
    ) -> dict[str, Any] | None:
        with _captured_runner_bindings(self):
            fit = super()._load_bootstrap_fit(panel, replicate, sample, cache_key)
        if fit is None:
            return None
        manifest_path = self._bootstrap_cache_dir(panel, replicate) / "fit_manifest.json"
        manifest = engine._read_json(manifest_path)
        if not _stored_dense_audit_is_valid(manifest):
            raise V3Phase4Error(f"cached bootstrap fit violates V4: {manifest_path}")
        return {
            **fit,
            "grid_nodes": manifest.get("grid_nodes"),
            "quadrature_method": manifest.get("quadrature_method"),
            "quadrature_linear_bound": manifest.get("quadrature_linear_bound"),
            "convergence_mode": manifest.get("convergence_mode"),
            "penalized_objective": manifest.get("penalized_objective"),
        }

    def _save_bootstrap_fit(
        self,
        panel: Panel,
        replicate: int,
        sample: FamilyBootstrapSample,
        cache_key: str,
        fit: Mapping[str, Any],
        bank_policy: Mapping[str, Any],
    ) -> None:
        validity = dense_fit_validity(fit, panel.selected_spec)
        if validity["fit_valid"] is not True:
            raise V3Phase4Error("refusing to cache an invalid dense bootstrap fit")
        with _captured_runner_bindings(self):
            super()._save_bootstrap_fit(
                panel, replicate, sample, cache_key, fit, bank_policy
            )
        manifest_path = self._bootstrap_cache_dir(panel, replicate) / "fit_manifest.json"
        payload = engine._read_json(manifest_path)
        payload.pop("manifest_content_sha256", None)
        payload.update(
            {
                "grid_nodes": fit.get("grid_nodes"),
                "quadrature_method": fit.get("quadrature_method"),
                "quadrature_linear_bound": fit.get("quadrature_linear_bound"),
                "convergence_mode": fit.get("convergence_mode"),
                "penalized_objective": fit.get("penalized_objective"),
                "convergence_contract": _compact_convergence_contract(fit),
                "dense_fit_validity": validity,
            }
        )
        payload["manifest_content_sha256"] = engine._canonical_hash(payload)
        engine._atomic_json(manifest_path, payload)

    def _fit_bootstrap(
        self, panel: Panel, replicate: int, sample: FamilyBootstrapSample
    ) -> BankBundle:
        with _captured_runner_bindings(self):
            assert_fit_boundary(
                panel.training_model_ids, panel.test_model_ids, sample.model_ids
            )
            cache_key = self._bootstrap_cache_key(panel, replicate, sample)
            fit = self._load_bootstrap_fit(panel, replicate, sample, cache_key)
            if fit is None:
                fit_args = argparse.Namespace(
                    grid=int(self.numerical["fit_grid"]),
                    ridge=(
                        0.0
                        if panel.selected_spec.ridge is None
                        else float(panel.selected_spec.ridge)
                    ),
                    log_a_shrinkage=(
                        cm.DEFAULT_LOG_A_SHRINKAGE
                        if panel.selected_spec.log_a_shrinkage is None
                        else float(panel.selected_spec.log_a_shrinkage)
                    ),
                    calibration_model=panel.selected_spec.family,
                    max_iter=int(self.numerical["fit_max_iter"]),
                    tol=float(self.numerical["fit_objective_tolerance"]),
                    quadrature_method=str(
                        self.numerical["fit_quadrature_method"]
                    ),
                    linear_bound=float(self.numerical["fit_linear_bound"]),
                    convergence_mode=str(
                        self.numerical["fit_convergence_mode"]
                    ),
                    parameter_tol=float(
                        self.numerical["fit_parameter_tolerance"]
                    ),
                    consecutive_convergence_passes=int(
                        self.numerical["fit_consecutive_convergence_passes"]
                    ),
                )
                sampled_matrix = self.matrix.loc[list(sample.model_ids)]
                if set(map(str, sampled_matrix.index)) & set(panel.test_model_ids):
                    raise V3Phase4Error(
                        "outer-test model reached bootstrap calibration matrix"
                    )
                fit = phase3._fit_structure_dense(
                    sampled_matrix, self.q_by, fit_args, self.structure
                )
                validity = dense_fit_validity(fit, panel.selected_spec)
                if validity["fit_valid"] is not True:
                    raise engine.RecoverableNumericalFailure(
                        "bootstrap fit failed the frozen dense-fitter contract"
                    )
                _bank, policy = scenario_cv.build_fold_bank(
                    fit,
                    self.structure,
                    self.source_records,
                    negative_policy=self.negative_policy,
                )
                self._save_bootstrap_fit(
                    panel, replicate, sample, cache_key, fit, policy
                )
            return self._bundle_from_fit(fit, cache_key)

    def _base_manifest(self, status: str) -> dict[str, Any]:
        with _captured_runner_bindings(self):
            payload = super()._base_manifest(status)
        payload.update(
            {
                "script": "scripts/nested_cat_total_uncertainty_v3.py",
                "calculation_engine": "scripts/nested_cat_total_uncertainty_v2.py",
                "phase3_artifact_hashes": self.phase3_artifact_hashes,
                "v4_numerical_authorization": self.numerical,
                "frozen_primary_policy_only": True,
                "fallback_allowed": False,
                "dense_bootstrap_fit_contract_enforced": True,
                "captured_provenance_sha256": self._captured_provenance_sha256,
            }
        )
        return payload

    def run(self) -> int:
        self._revalidate_frozen_provenance()
        with _captured_runner_bindings(self):
            original_atomic_json = engine._atomic_json

            def guarded_atomic_json(path: Path, value: Any) -> None:
                status = value.get("status") if isinstance(value, Mapping) else None
                final_manifest = (
                    path.resolve() == (self.output_dir / "manifest.json").resolve()
                    and status in {"phase4_complete_pass", "phase4_complete_fail"}
                )
                if final_manifest:
                    self._revalidate_frozen_provenance()
                original_atomic_json(path, value)

            engine._atomic_json = guarded_atomic_json
            try:
                return super().run()
            finally:
                engine._atomic_json = original_atomic_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--numerical-followup-config",
        type=Path,
        default=DEFAULT_NUMERICAL_FOLLOWUP_CONFIG,
    )
    parser.add_argument("--numerical-lock", type=Path, default=DEFAULT_NUMERICAL_LOCK)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        return V3Phase4Runner(args).run()
    except (
        V3Phase4Error,
        engine.V2Phase4Error,
        phase3.V3Phase3Error,
        phase3_v1.NestedCVError,
        scat.OfflineStudyError,
        cm.CalibrationError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
