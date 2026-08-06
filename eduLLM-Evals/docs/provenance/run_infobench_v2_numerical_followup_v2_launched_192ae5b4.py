#!/usr/bin/env python3
"""Corrected append-only numerical follow-up for InFoBench v2.

This runner preserves the aborted v1 attempt and the original blocked parent
run.  It recomputes every grid-101 and grid-121 fit in a new output leaf,
compares 81→101 and 101→121 with truthfully named evidence, and applies one
all-six fit-grid and EAP-profile contract.  It performs no CAT or calibration
model selection.

The large numerical kernels are imported from the hash-frozen v1 runner.  The
audited orchestration points are replaced here: configuration loading, exact
specification validation, common-cell labeling, append-only preparation,
checkpoint writing, and finalization.  The v1 partial output is never read as
fit evidence or reused as a cache.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import run_infobench_v2_numerical_followup as v1  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "infobench_v2_numerical_followup_v2.json"
CONFIG_SCHEMA = "infobench-v2-numerical-followup-v2"
RUNNER_SCHEMA = "infobench-v2-numerical-followup-run-v2"
CHECKPOINT_SCHEMA = "infobench-v2-numerical-followup-checkpoint-v2"
LOCK_SCHEMA = "infobench-v2-numerical-followup-lock-v2"
OUTPUT_LEAF = "InFoBench_v2_numerical_followup_v2"

FIT_GRIDS = (81, 101, 121)
COMPARISON_SCHEDULE = ((81, 101), (101, 121))
FROZEN_CONTRACT_SHA256 = "25e6db82f534492dfdb82448ea5bb6fb00fed5237ae9e7b607a9b05e89b9a809"
V1_CONFIG_SCHEMA = "infobench-v2-numerical-followup-v1"
V1_RUNNER_SCHEMA = "infobench-v2-numerical-followup-run-v1"
EXPECTED_CORRECTIONS = {
    "dynamic_common_cell_labels": True,
    "full_parent_exact_spec_contract_match": True,
    "append_only_no_fresh_mode": True,
    "terminal_evidence_is_immutable": True,
    "resume_requires_identical_signature_and_checkpoints": True,
    "all_material_preregistration_fields_fail_closed": True,
}

CODE_DEPENDENCIES = (
    Path(__file__).resolve(),
    ROOT / "scripts" / "run_infobench_v2_numerical_followup.py",
    ROOT / "scripts" / "run_infobench_v2_numerical_checks.py",
    ROOT / "scripts" / "calibrate_mirt.py",
    ROOT / "scripts" / "kfold_cv_mirt.py",
    ROOT / "scripts" / "scenario_cat_lib.py",
    ROOT / "tutor_cat" / "mirt.py",
)


class FollowupV2Error(v1.FollowupError):
    """The corrected frozen design, provenance, or append-only contract failed."""


def _resolve(value: str | Path) -> Path:
    return v1._resolve(value)


def _display(path: Path) -> str:
    return v1._display(path)


def _sha256(path: Path) -> str:
    return v1._sha256(path)


def _read_json(path: Path) -> dict[str, Any]:
    return v1._read_json(path)


def _atomic_json(path: Path, value: Any) -> None:
    v1._atomic_json(path, value)


def _canonical_sha256(value: Any) -> str:
    return v1._canonical_sha256(value)


def _require_hash(path: Path, expected: str, label: str) -> None:
    v1._require_hash(path, expected, label)


def _tree_sha256(root: Path) -> tuple[int, str]:
    if not root.is_dir():
        raise FollowupV2Error(f"preserved predecessor directory is missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return len(files), digest.hexdigest()


def _validate_overlay(raw: Mapping[str, Any]) -> None:
    expected_scalars = {
        "schema_version": CONFIG_SCHEMA,
        "status": "preregistered_before_followup_results",
        "benchmark": "InFoBench",
        "output_dir": f"runs/calibration/{OUTPUT_LEAF}",
        "lock_path": f"runs/calibration/{OUTPUT_LEAF}/numerical_followup_lock.json",
    }
    for key, expected in expected_scalars.items():
        if raw.get(key) != expected:
            raise FollowupV2Error(f"v2 config field changed: {key}")
    if raw.get("implementation_corrections") != EXPECTED_CORRECTIONS:
        raise FollowupV2Error("audited implementation-correction contract changed")
    contract = raw.get("frozen_contract")
    if not isinstance(contract, Mapping):
        raise FollowupV2Error("frozen_contract is missing")
    if _canonical_sha256(contract) != FROZEN_CONTRACT_SHA256:
        raise FollowupV2Error("material preregistration fields differ from the frozen contract")
    source = raw.get("design_source") or {}
    superseded = raw.get("superseded_attempt") or {}
    if source.get("config") != "configs/infobench_v2_numerical_followup_v1.json":
        raise FollowupV2Error("v1 design-source path changed")
    if source.get("config_sha256") != superseded.get("config_sha256"):
        raise FollowupV2Error("design-source and superseded config hashes disagree")
    required_predecessor = {
        "config": "configs/infobench_v2_numerical_followup_v1.json",
        "runner": "scripts/run_infobench_v2_numerical_followup.py",
        "run_dir": "runs/calibration/InFoBench_v2_numerical_followup_v1",
        "manifest": (
            "runs/calibration/InFoBench_v2_numerical_followup_v1/study_manifest.json"
        ),
        "aborted_marker": "runs/calibration/InFoBench_v2_numerical_followup_v1/ABORTED.json",
        "required_aborted_status": "aborted_before_fit_comparison",
        "output_file_count": 295,
    }
    for key, expected in required_predecessor.items():
        if superseded.get(key) != expected:
            raise FollowupV2Error(f"superseded-attempt field changed: {key}")
    for key in (
        "config_sha256",
        "runner_sha256",
        "manifest_sha256",
        "aborted_marker_sha256",
        "output_tree_sha256",
    ):
        value = superseded.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise FollowupV2Error(f"superseded-attempt digest is invalid: {key}")
    if not str(raw.get("threshold_change_policy", "")).startswith("No threshold"):
        raise FollowupV2Error("threshold-relaxation prohibition is missing")


def _verify_superseded_attempt(raw: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = raw["superseded_attempt"]
    paths = {
        "config": _resolve(predecessor["config"]),
        "runner": _resolve(predecessor["runner"]),
        "manifest": _resolve(predecessor["manifest"]),
        "aborted_marker": _resolve(predecessor["aborted_marker"]),
    }
    for name, path in paths.items():
        _require_hash(path, str(predecessor[f"{name}_sha256"]), f"aborted v1 {name}")
    run_dir = _resolve(predecessor["run_dir"])
    count, tree_hash = _tree_sha256(run_dir)
    if count != int(predecessor["output_file_count"]):
        raise FollowupV2Error("aborted v1 output file count changed")
    if tree_hash != predecessor["output_tree_sha256"]:
        raise FollowupV2Error("aborted v1 output tree changed")
    marker = _read_json(paths["aborted_marker"])
    if marker.get("status") != predecessor["required_aborted_status"]:
        raise FollowupV2Error("aborted v1 marker status changed")
    manifest = _read_json(paths["manifest"])
    if (
        manifest.get("schema_version") != V1_RUNNER_SCHEMA
        or manifest.get("status") != "running"
    ):
        raise FollowupV2Error("aborted v1 manifest provenance changed")
    forbidden = (
        run_dir / "common_fit_grid_decision.json",
        run_dir / "common_eap_profile_decision.json",
        run_dir / "numerical_followup_decision.json",
        run_dir / "numerical_followup_lock.json",
        run_dir / "numerical_followup_lock.sha256",
    )
    if any(path.exists() for path in forbidden):
        raise FollowupV2Error("aborted v1 output unexpectedly contains a decision or lock")
    if (run_dir / "fit_comparisons").exists() or (run_dir / "eap_bound").exists():
        raise FollowupV2Error("aborted v1 output unexpectedly contains comparison evidence")
    return {
        "config_sha256": predecessor["config_sha256"],
        "runner_sha256": predecessor["runner_sha256"],
        "manifest_sha256": predecessor["manifest_sha256"],
        "aborted_marker_sha256": predecessor["aborted_marker_sha256"],
        "output_tree_sha256": tree_hash,
        "output_file_count": count,
    }


def _materialize_effective_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    source = raw["design_source"]
    source_path = _resolve(source["config"])
    _require_hash(source_path, str(source["config_sha256"]), "v1 frozen design source")
    effective = copy.deepcopy(_read_json(source_path))
    if effective.get("schema_version") != V1_CONFIG_SCHEMA:
        raise FollowupV2Error("v1 design source schema changed")
    effective["schema_version"] = CONFIG_SCHEMA
    effective["status"] = raw["status"]
    effective["purpose"] = raw["purpose"]
    effective["design_source"] = copy.deepcopy(raw["design_source"])
    effective["superseded_attempt"] = copy.deepcopy(raw["superseded_attempt"])
    effective["implementation_corrections"] = copy.deepcopy(
        raw["implementation_corrections"]
    )
    effective["frozen_contract"] = copy.deepcopy(raw["frozen_contract"])
    effective["output_dir"] = raw["output_dir"]
    effective["lock_path"] = raw["lock_path"]
    effective["threshold_change_policy"] = raw["threshold_change_policy"]
    effective["selection_policy"] = raw["selection_policy"]
    fit = effective["fit_grid_followup"]
    fit["heldout_dynamic_common_cell_equivalence"] = fit.pop(
        "heldout_common_cell_equivalence"
    )
    effective["eap_bound_followup"]["lock_rule"] = copy.deepcopy(
        raw["frozen_contract"]["eap_bound_followup"]["lock_rule"]
    )
    return effective


def _profile_tuple(profile: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        profile.get("method"),
        int(profile.get("nodes", -1)),
        float(profile.get("bound", -1)),
        float(profile.get("step", -1)),
        float(profile.get("tail_region", -1)),
    )


def _validate_effective_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise FollowupV2Error("effective config schema changed")
    contract = config["frozen_contract"]
    frozen_fit = contract["fit_grid_followup"]
    fit = config.get("fit_grid_followup") or {}
    if tuple(map(int, fit.get("grid_candidates") or [])) != FIT_GRIDS:
        raise FollowupV2Error("fit-grid candidates changed")
    observed_schedule = (fit.get("first_comparison"), fit.get("second_comparison"))
    if observed_schedule != tuple(frozen_fit["comparison_schedule"]):
        raise FollowupV2Error("fit-comparison schedule changed")
    if tuple(fit.get("required_comparisons") or []) != tuple(
        frozen_fit["comparison_schedule"]
    ):
        raise FollowupV2Error("required fit-comparison schedule changed")
    fit_fields = (
        "always_run_both_fit_comparisons",
        "common_grid_required_for_all_exact_specifications",
        "require_all_fits_converged",
        "minimum_item_parameter_spearman",
        "minimum_exportability_agreement",
    )
    for key in fit_fields:
        if fit.get(key) != frozen_fit[key]:
            raise FollowupV2Error(f"material fit field changed: {key}")
    if fit.get("lock_rule") != frozen_fit["lock_rule"]:
        raise FollowupV2Error("common fit-grid lock rule changed")
    if "heldout_common_cell_equivalence" in fit:
        raise FollowupV2Error("mislabeled inherited common-cell field remains active")
    cells = fit.get("heldout_dynamic_common_cell_equivalence") or {}
    expected_cells = {
        "metrics": frozen_fit["heldout_metrics"],
        "equivalence_margin": frozen_fit["equivalence_margin"],
        "require_absolute_pooled_shift_within_margin": True,
        "require_family_cluster_bootstrap_ci_within_margin": True,
        "bootstrap_replicates": frozen_fit["bootstrap_replicates"],
        "bootstrap_seed": frozen_fit["bootstrap_seed"],
    }
    if cells != expected_cells:
        raise FollowupV2Error("held-out equivalence contract changed")
    theta = fit.get("common_support_theta_stability") or {}
    expected_theta = {
        "quadrature_method": frozen_fit["theta_quadrature_method"],
        "quadrature_nodes": frozen_fit["theta_quadrature_nodes"],
        "linear_bound": frozen_fit["theta_linear_bound"],
        "supports": frozen_fit["theta_supports"],
        "scopes": frozen_fit["theta_scopes"],
        "maximum_median_absolute_theta_shift": frozen_fit[
            "maximum_median_absolute_theta_shift"
        ],
        "maximum_p95_absolute_theta_shift": frozen_fit[
            "maximum_p95_absolute_theta_shift"
        ],
        "maximum_absolute_theta_shift": frozen_fit["maximum_absolute_theta_shift"],
        "require_identical_model_scope_support_keys": True,
    }
    if theta != expected_theta:
        raise FollowupV2Error("common-support theta contract changed")

    frozen_eap = contract["eap_bound_followup"]
    eap = config.get("eap_bound_followup") or {}
    for key in (
        "run_only_after_common_fit_grid_lock",
        "run_for_every_exact_specification",
        "maximum_posterior_tail_mass",
        "lock_rule",
    ):
        if eap.get(key) != frozen_eap[key]:
            raise FollowupV2Error(f"material EAP field changed: {key}")
    for comparison in ("bound8_density", "equal_step_cross_bound", "bound10_density"):
        for side in ("lower", "upper"):
            if _profile_tuple((eap.get(comparison) or {}).get(side) or {}) != tuple(
                frozen_eap[comparison][side]
            ):
                raise FollowupV2Error(f"EAP profile changed: {comparison}/{side}")
    thresholds = eap.get("theta_stability_thresholds") or {}
    for key in (
        "maximum_median_absolute_theta_shift",
        "maximum_p95_absolute_theta_shift",
        "maximum_absolute_theta_shift",
    ):
        if thresholds.get(key) != frozen_eap[key]:
            raise FollowupV2Error(f"EAP theta threshold changed: {key}")
    if config.get("runtime") != contract["runtime"]:
        raise FollowupV2Error("runtime max_iter/tolerance/seed changed")


def _spec_contract(spec: Any) -> dict[str, Any]:
    return {
        "spec_id": spec.spec_id,
        "family": spec.family,
        "ridge": spec.ridge,
        "log_a_shrinkage": spec.log_a_shrinkage,
        "canonical_specification": copy.deepcopy(spec.canonical),
    }


def _require_full_parent_spec_match(
    followup_specs: Sequence[Any], parent_specs: Sequence[Any]
) -> list[dict[str, Any]]:
    observed = [_spec_contract(spec) for spec in followup_specs]
    expected = [_spec_contract(spec) for spec in parent_specs]
    if observed != expected:
        raise FollowupV2Error(
            "follow-up exact specifications differ from parent "
            "family/regularization/canonical panel"
        )
    return observed


def load_context(
    config_path: Path = DEFAULT_CONFIG, out_dir: Path | None = None
) -> v1.Context:
    config_path = config_path.resolve()
    raw = _read_json(config_path)
    _validate_overlay(raw)
    predecessor_hashes = _verify_superseded_attempt(raw)
    config = _materialize_effective_config(raw)
    _validate_effective_config(config)

    parent = config["parent_study"]
    parent_config_path = _resolve(parent["config"])
    parent_runner_path = _resolve(parent["runner"])
    parent_run = _resolve(parent["run_dir"])
    parent_manifest_path = _resolve(parent["study_manifest"])
    parent_decision_path = _resolve(parent["decision"])
    for path, expected, label in (
        (parent_config_path, parent["config_sha256"], "parent config"),
        (parent_runner_path, parent["runner_sha256"], "parent runner"),
        (parent_manifest_path, parent["study_manifest_sha256"], "parent manifest"),
        (parent_decision_path, parent["decision_sha256"], "parent decision"),
    ):
        _require_hash(path, str(expected), label)
    parent_config = _read_json(parent_config_path)
    parent_manifest = _read_json(parent_manifest_path)
    parent_decision = _read_json(parent_decision_path)
    if parent_manifest.get("status") != parent["required_status"]:
        raise FollowupV2Error("parent study is not the frozen blocked result")
    if parent_decision.get("passed") is not parent["required_decision_passed"]:
        raise FollowupV2Error("parent decision no longer records a failure")
    if parent_manifest.get("study_signature_sha256") != parent[
        "required_study_signature_sha256"
    ]:
        raise FollowupV2Error("parent study signature changed")

    specs = v1._parse_specs(config)
    parent_specs = v1.parent_runner._parse_specs(parent_config)
    exact_spec_contracts = _require_full_parent_spec_match(specs, parent_specs)

    frozen = config["frozen_inputs"]
    matrix_path = _resolve(frozen["response_matrix"])
    rubrics_path = _resolve(frozen["rubrics"])
    scenarios_path = _resolve(frozen["scenarios"])
    split_path = _resolve(frozen["split_manifest"])
    for path, expected, label in (
        (matrix_path, frozen["response_matrix_sha256"], "response matrix"),
        (rubrics_path, frozen["rubrics_sha256"], "rubrics"),
        (scenarios_path, frozen["scenarios_sha256"], "scenarios"),
        (split_path, frozen["split_manifest_sha256"], "split manifest"),
    ):
        _require_hash(path, str(expected), label)
    matrix = cm.load_matrix_strict(matrix_path)
    splits = _read_json(split_path)
    parent_scopes, families, administration, evaluation = v1.parent_runner._parse_scopes(
        splits, list(map(str, matrix.index))
    )
    scopes = tuple(
        v1.Scope(scope.scope_id, scope.train_models, scope.score_models, scope.outer_fold)
        for scope in parent_scopes
    )
    configured_out = _resolve(config["output_dir"])
    chosen_out = configured_out if out_dir is None else out_dir.resolve()
    if chosen_out != configured_out:
        raise FollowupV2Error("--out-dir must equal the preregistered v2 output leaf")
    if _resolve(config["lock_path"]) != chosen_out / "numerical_followup_lock.json":
        raise FollowupV2Error("lock path is outside the v2 follow-up output")

    provisional = v1.Context(
        config_path=config_path,
        config=config,
        parent_config=parent_config,
        parent_run=parent_run,
        parent_manifest=parent_manifest,
        matrix_path=matrix_path,
        rubrics_path=rubrics_path,
        scenarios_path=scenarios_path,
        split_path=split_path,
        splits=splits,
        specs=specs,
        scopes=scopes,
        model_to_family=families,
        administration_scenarios=administration,
        evaluation_scenarios=evaluation,
        out_dir=chosen_out,
        study_signature={},
    )
    parent_checkpoint_hashes: dict[str, str] = {}
    for spec in specs:
        for scope in scopes:
            v1._verify_parent_checkpoint(provisional, spec, scope)
            key = f"{spec.spec_id}/{scope.scope_id}"
            parent_checkpoint_hashes[key] = str(
                config["parent_study"]["grid81_checkpoint_sha256"][key]
            )
    environment = v1.parent_runner._environment()
    signature_payload = {
        "schema_version": RUNNER_SCHEMA,
        "config_sha256": _sha256(config_path),
        "effective_config_sha256": _canonical_sha256(config),
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "superseded_attempt_sha256": predecessor_hashes,
        "parent_manifest_sha256": _sha256(parent_manifest_path),
        "parent_decision_sha256": _sha256(parent_decision_path),
        "parent_checkpoint_sha256": parent_checkpoint_hashes,
        "input_sha256": {
            _display(path): _sha256(path)
            for path in (matrix_path, rubrics_path, scenarios_path, split_path)
        },
        "code_sha256": {_display(path): _sha256(path) for path in CODE_DEPENDENCIES},
        "environment_sha256": environment["canonical_sha256"],
        "exact_specifications": exact_spec_contracts,
        "fit_grids": list(FIT_GRIDS),
        "comparison_schedule": [list(pair) for pair in COMPARISON_SCHEDULE],
        "eap_profiles": v1.EAP_PROFILES,
        "aborted_v1_artifacts_reused": False,
    }
    signature = {
        **signature_payload,
        "environment": environment,
        "canonical_sha256": _canonical_sha256(signature_payload),
    }
    return v1.Context(**{**provisional.__dict__, "study_signature": signature})


def runtime_schedule(context: v1.Context) -> dict[str, Any]:
    schedule = v1.runtime_schedule(context)
    schedule.update(
        {
            "comparison_schedule": ["81_to_101", "101_to_121"],
            "aborted_v1_fit_artifacts_reused": 0,
            "append_only_output_leaf": _display(context.out_dir),
            "fresh_mode_available": False,
        }
    )
    return schedule


def _assert_safe_output(context: v1.Context) -> None:
    calibration_root = (ROOT / "runs" / "calibration").resolve()
    output = context.out_dir.resolve()
    if output == calibration_root or calibration_root not in output.parents:
        raise FollowupV2Error("v2 output must be a leaf below runs/calibration")
    if output.name != OUTPUT_LEAF:
        raise FollowupV2Error("refusing a non-preregistered v2 output leaf")
    protected = {
        context.parent_run.resolve(),
        _resolve(context.config["superseded_attempt"]["run_dir"]),
    }
    if output in protected:
        raise FollowupV2Error("parent and aborted-v1 outputs are immutable")


def _allowed_stage_ids(context: v1.Context) -> set[str]:
    stages: set[str] = set()
    for spec in context.specs:
        for scope in context.scopes:
            stages.add(f"import_grid081/{spec.namespace}/{scope.scope_id}")
            for grid in (101, 121):
                stages.add(f"fit/{spec.namespace}/grid_{grid:03d}/{scope.scope_id}")
        for lower, upper in COMPARISON_SCHEDULE:
            stages.add(f"fit_pair/{lower}_vs_{upper}/{spec.namespace}")
    fit_decision_path = context.out_dir / "common_fit_grid_decision.json"
    if fit_decision_path.is_file():
        fit_decision = _read_json(fit_decision_path)
        locked = fit_decision.get("locked_common_fit_grid")
        if fit_decision.get("passed") is True and locked in FIT_GRIDS:
            for spec in context.specs:
                stages.add(f"eap_bound/{spec.namespace}/fit_grid_{int(locked):03d}")
        elif locked is not None:
            raise FollowupV2Error("stored common fit-grid decision is malformed")
    return stages


def _validate_existing_checkpoints(context: v1.Context) -> None:
    checkpoint_dir = context.out_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return
    allowed = _allowed_stage_ids(context)
    for path in sorted(checkpoint_dir.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            raise FollowupV2Error(f"unexpected checkpoint entry: {path}")
        marker = _read_json(path)
        stage_id = str(marker.get("stage_id") or "")
        expected_name = f"{stage_id.replace('/', '__')}.json"
        if stage_id not in allowed or path.name != expected_name:
            raise FollowupV2Error(f"unexpected checkpoint stage: {path}")
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "status": "completed",
            "study_signature_sha256": context.study_signature["canonical_sha256"],
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            raise FollowupV2Error(f"checkpoint provenance mismatch: {stage_id}")
        inputs = marker.get("stage_inputs")
        if not isinstance(inputs, Mapping) or marker.get(
            "stage_inputs_sha256"
        ) != _canonical_sha256(inputs):
            raise FollowupV2Error(f"checkpoint input digest mismatch: {stage_id}")
        outputs = marker.get("output_sha256")
        if not isinstance(outputs, Mapping) or not outputs:
            raise FollowupV2Error(f"checkpoint has no outputs: {stage_id}")
        for recorded, digest in outputs.items():
            output = _resolve(recorded)
            if context.out_dir.resolve() not in output.resolve().parents:
                raise FollowupV2Error(f"checkpoint output escapes v2 leaf: {stage_id}")
            _require_hash(output, str(digest), f"checkpoint output {stage_id}")


def _prepare(context: v1.Context, *, resume: bool, fresh: bool = False) -> None:
    _assert_safe_output(context)
    _verify_superseded_attempt(context.config)
    if fresh:
        raise FollowupV2Error("destructive --fresh mode does not exist in append-only v2")
    manifest_path = context.out_dir / "study_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise FollowupV2Error("--resume requires an existing v2 manifest")
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != RUNNER_SCHEMA:
            raise FollowupV2Error("resume manifest schema differs")
        if manifest.get("study_signature_sha256") != context.study_signature[
            "canonical_sha256"
        ]:
            raise FollowupV2Error("resume signature differs from inputs/code/environment")
        if manifest.get("status") != "running":
            raise FollowupV2Error(
                f"terminal evidence is immutable; cannot resume {manifest.get('status')!r}"
            )
        if (context.out_dir / "ABORTED.json").exists():
            raise FollowupV2Error("aborted v2 evidence is immutable and cannot be resumed")
        _validate_existing_checkpoints(context)
        return
    if context.out_dir.exists():
        raise FollowupV2Error("v2 output already exists; use --resume only for a running study")
    context.out_dir.mkdir(parents=True)
    _atomic_json(
        manifest_path,
        {
            "schema_version": RUNNER_SCHEMA,
            "status": "running",
            "started_at": v1.parent_runner._utcnow(),
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "study_signature": context.study_signature,
            "runtime_schedule": runtime_schedule(context),
            "audit_disclosure": context.config["audit_disclosure"],
            "superseded_attempt": context.config["superseded_attempt"],
            "parent_failure_preserved": True,
            "aborted_v1_preserved": True,
            "aborted_v1_artifacts_reused": False,
            "parent_run": _display(context.parent_run),
            "cat_results_inspected": False,
            "selection_performed": False,
        },
    )


def _checkpoint_reusable(
    context: v1.Context, stage_id: str, inputs: Mapping[str, Any]
) -> bool:
    path = v1._checkpoint(context, stage_id)
    if not path.exists():
        return False
    if not path.is_file():
        raise FollowupV2Error(f"checkpoint path is not a file: {stage_id}")
    marker = _read_json(path)
    expected = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "completed",
        "stage_id": stage_id,
        "study_signature_sha256": context.study_signature["canonical_sha256"],
        "stage_inputs_sha256": _canonical_sha256(inputs),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise FollowupV2Error(f"checkpoint provenance mismatch: {stage_id}")
    if marker.get("stage_inputs") != dict(inputs):
        raise FollowupV2Error(f"checkpoint inputs are not identical: {stage_id}")
    outputs = marker.get("output_sha256") or {}
    if not outputs:
        raise FollowupV2Error(f"checkpoint has no outputs: {stage_id}")
    for recorded, expected_hash in outputs.items():
        output = _resolve(recorded)
        if context.out_dir.resolve() not in output.resolve().parents:
            raise FollowupV2Error(f"checkpoint output escapes v2 leaf: {stage_id}")
        _require_hash(output, str(expected_hash), f"checkpoint output {stage_id}")
    return True


def _write_checkpoint(
    context: v1.Context,
    stage_id: str,
    inputs: Mapping[str, Any],
    outputs: Sequence[Path],
    elapsed: float,
) -> None:
    marker = v1._checkpoint(context, stage_id)
    if marker.exists():
        raise FollowupV2Error(f"checkpoint evidence already exists: {stage_id}")
    if any(not path.is_file() for path in outputs):
        raise FollowupV2Error(f"cannot checkpoint incomplete stage: {stage_id}")
    for output in outputs:
        if context.out_dir.resolve() not in output.resolve().parents:
            raise FollowupV2Error(f"stage output escapes v2 leaf: {stage_id}")
    _atomic_json(
        marker,
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "status": "completed",
            "stage_id": stage_id,
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "stage_inputs": dict(inputs),
            "stage_inputs_sha256": _canonical_sha256(inputs),
            "output_sha256": {_display(path): _sha256(path) for path in outputs},
            "elapsed_seconds": float(elapsed),
        },
    )


def dynamic_common_cell_equivalence(
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    *,
    lower_grid: int,
    upper_grid: int,
    log_loss_margin: float,
    brier_margin: float,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if lower_grid >= upper_grid or (lower_grid, upper_grid) not in COMPARISON_SCHEDULE:
        raise FollowupV2Error("common-cell helper received an unfrozen grid comparison")
    keys = ["outer_fold", "model", "criterion_id"]
    required = set(keys + ["family", "label", "probability"])
    for grid, frame in ((lower_grid, lower), (upper_grid, upper)):
        if required - set(frame.columns) or frame.duplicated(keys).any():
            raise FollowupV2Error(f"grid {grid} held-out cell table is malformed")
    left_keys = lower[keys].sort_values(keys).reset_index(drop=True)
    right_keys = upper[keys].sort_values(keys).reset_index(drop=True)
    keys_identical = left_keys.equals(right_keys)
    if not keys_identical:
        raise FollowupV2Error("fit grids did not score identical common held-out cells")
    lower_suffix = f"_grid_{lower_grid}"
    upper_suffix = f"_grid_{upper_grid}"
    merged = lower.merge(
        upper, on=keys, suffixes=(lower_suffix, upper_suffix), validate="one_to_one"
    )
    lower_label = f"label_grid_{lower_grid}"
    upper_label = f"label_grid_{upper_grid}"
    lower_family = f"family_grid_{lower_grid}"
    upper_family = f"family_grid_{upper_grid}"
    labels_identical = bool((merged[lower_label] == merged[upper_label]).all())
    families_identical = bool((merged[lower_family] == merged[upper_family]).all())
    if not labels_identical or not families_identical or merged.empty:
        raise FollowupV2Error("common held-out labels/families differ or are empty")
    merged["label"] = merged[lower_label].astype(int)
    merged["family"] = merged[lower_family]
    y = merged["label"].to_numpy(float)
    for grid in (lower_grid, upper_grid):
        probability_column = f"probability_grid_{grid}"
        probability = np.clip(
            merged[probability_column].to_numpy(float),
            v1.parent_runner.EPSILON,
            1 - v1.parent_runner.EPSILON,
        )
        merged[f"log_loss_grid_{grid}"] = -(
            y * np.log(probability) + (1 - y) * np.log(1 - probability)
        )
        merged[f"brier_grid_{grid}"] = (probability - y) ** 2
    metric_details: dict[str, Any] = {}
    delta_columns: list[str] = []
    overall = True
    for offset, (metric, margin) in enumerate(
        (("log_loss", log_loss_margin), ("brier", brier_margin))
    ):
        delta = f"{metric}_delta_grid_{upper_grid}_minus_grid_{lower_grid}"
        delta_columns.append(delta)
        merged[delta] = (
            merged[f"{metric}_grid_{upper_grid}"]
            - merged[f"{metric}_grid_{lower_grid}"]
        )
        pooled = float(merged[delta].mean())
        low, high = v1.parent_runner._cluster_interval(
            merged, delta, replicates=replicates, seed=seed + offset
        )
        passed = abs(pooled) <= margin and low >= -margin and high <= margin
        directional_key = f"pooled_shift_grid_{upper_grid}_minus_grid_{lower_grid}"
        metric_details[metric] = {
            directional_key: pooled,
            "absolute_pooled_shift": abs(pooled),
            "family_cluster_bootstrap_ci_95": [low, high],
            "equivalence_margin": margin,
            "pooled_shift_within_margin": abs(pooled) <= margin,
            "ci_wholly_within_equivalence_bounds": low >= -margin and high <= margin,
            "passed": bool(passed),
        }
        overall = overall and passed
    probability_columns = [
        f"probability_grid_{lower_grid}",
        f"probability_grid_{upper_grid}",
    ]
    columns = keys + ["family", "label", *probability_columns, *delta_columns]
    comparison = f"grid_{lower_grid}_to_grid_{upper_grid}"
    return (
        {
            "comparison": comparison,
            "lower_grid": lower_grid,
            "upper_grid": upper_grid,
            "n_common_cells": len(merged),
            "keys_identical": keys_identical,
            "labels_identical": labels_identical,
            "families_identical": families_identical,
            "cluster_unit": "tutor_model_family",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "csv_probability_columns": probability_columns,
            "csv_delta_columns": delta_columns,
            "metrics": metric_details,
            "passed": bool(overall),
        },
        merged[columns].sort_values(keys),
    )


def _run_pair_gate(
    context: v1.Context,
    spec: v1.ExactSpec,
    lower_grid: int,
    upper_grid: int,
    matrix: pd.DataFrame,
    item_to_scenario: Mapping[str, str],
    *,
    resume: bool,
) -> dict[str, Any]:
    comparison = f"{lower_grid}_vs_{upper_grid}"
    stage_id = f"fit_pair/{comparison}/{spec.namespace}"
    inputs = {
        "comparison": comparison,
        "comparison_direction": f"grid_{lower_grid}_to_grid_{upper_grid}",
        "spec_id": spec.spec_id,
        "fit_sha256": {
            f"{grid}/{scope.scope_id}": _sha256(
                v1._fit_dir(context, spec, grid, scope) / "fit.npz"
            )
            for grid in (lower_grid, upper_grid)
            for scope in context.scopes
        },
        "fit_grid_followup": context.config["fit_grid_followup"],
    }
    out = context.out_dir / "fit_comparisons" / comparison / spec.namespace
    outputs = [
        out / "parameter_scope_gates.csv",
        out / "common_support.csv",
        out / "theta_common_support.csv",
        out / "heldout_cells.csv",
        out / "fit_pair_gate.json",
    ]
    if resume and _checkpoint_reusable(context, stage_id, inputs):
        return _read_json(outputs[-1])
    if any(path.exists() for path in outputs) or v1._checkpoint(context, stage_id).exists():
        raise FollowupV2Error(f"partial fit-pair stage exists: {stage_id}")
    started = time.monotonic()
    fit_config = context.config["fit_grid_followup"]
    parameter_rows: list[dict[str, Any]] = []
    for scope in context.scopes:
        dirs = {
            grid: v1._fit_dir(context, spec, grid, scope)
            for grid in (lower_grid, upper_grid)
        }
        manifests = {
            grid: _read_json(dirs[grid] / "fit_manifest.json")
            for grid in (lower_grid, upper_grid)
        }
        gate = v1.parent_runner.parameter_scope_gate(
            pd.read_csv(dirs[lower_grid] / "item_params.csv"),
            pd.read_csv(dirs[upper_grid] / "item_params.csv"),
            family=spec.family,
            source_ids=list(map(str, matrix.columns)),
            minimum_spearman=float(fit_config["minimum_item_parameter_spearman"]),
            minimum_exportability_agreement=float(
                fit_config["minimum_exportability_agreement"]
            ),
            lower_converged=bool(manifests[lower_grid]["converged"]),
            upper_converged=bool(manifests[upper_grid]["converged"]),
        )
        parameter_rows.append({"scope": scope.scope_id, **gate})
    theta_rows, lower_cells, upper_cells, support_rows = v1._fit_pair_evidence(
        context, spec, lower_grid, upper_grid, matrix, item_to_scenario
    )
    cell_config = fit_config["heldout_dynamic_common_cell_equivalence"]
    seed = int(cell_config["bootstrap_seed"]) + int(spec.canonical["cache_key"][-8:], 16)
    cell_gate, cell_details = dynamic_common_cell_equivalence(
        lower_cells,
        upper_cells,
        lower_grid=lower_grid,
        upper_grid=upper_grid,
        log_loss_margin=float(cell_config["equivalence_margin"]),
        brier_margin=float(cell_config["equivalence_margin"]),
        replicates=int(cell_config["bootstrap_replicates"]),
        seed=seed,
    )
    theta_frame = pd.DataFrame(theta_rows)
    theta_gate = v1.theta_shift_gate(
        theta_frame,
        lower_column=f"theta_grid_{lower_grid}",
        upper_column=f"theta_grid_{upper_grid}",
        thresholds=fit_config["common_support_theta_stability"],
    )
    parameter_passed = all(bool(row["passed"]) for row in parameter_rows)
    result = {
        "schema_version": RUNNER_SCHEMA,
        "spec_id": spec.spec_id,
        "canonical_specification": spec.canonical,
        "comparison": comparison,
        "comparison_direction": f"grid_{lower_grid}_to_grid_{upper_grid}",
        "all_scope_parameter_gate_passed": parameter_passed,
        "heldout_common_cell_gate": cell_gate,
        "common_support_theta_gate": theta_gate,
        "theta_gate_was_omitted_from_parent": True,
        "cat_results_inspected": False,
        "selection_performed": False,
        "passed": bool(parameter_passed and cell_gate["passed"] and theta_gate["passed"]),
    }
    v1._atomic_csv(outputs[0], pd.DataFrame(parameter_rows))
    v1._atomic_csv(outputs[1], pd.DataFrame(support_rows))
    v1._atomic_csv(outputs[2], theta_frame)
    v1._atomic_csv(outputs[3], cell_details)
    _atomic_json(outputs[4], result)
    _write_checkpoint(context, stage_id, inputs, outputs, time.monotonic() - started)
    return result


def _write_json_once(path: Path, value: Any) -> None:
    if path.exists():
        if not path.is_file() or _read_json(path) != value:
            raise FollowupV2Error(f"append-only JSON evidence differs: {path}")
        return
    _atomic_json(path, value)


def _write_text_once(path: Path, value: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != value:
            raise FollowupV2Error(f"append-only text evidence differs: {path}")
        return
    path.write_text(value, encoding="utf-8")


def _finalize(
    context: v1.Context,
    fit_decision: Mapping[str, Any],
    first_gates: Mapping[str, Mapping[str, Any]],
    second_gates: Mapping[str, Mapping[str, Any]],
    eap_gates: Mapping[str, Mapping[str, Any]],
    eap_decision: Mapping[str, Any],
) -> dict[str, Any]:
    passed = fit_decision.get("passed") is True and eap_decision.get("passed") is True
    effective_profile = (
        {
            "fit_grid": int(fit_decision["locked_common_fit_grid"]),
            **dict(eap_decision["locked_common_eap_profile"]),
        }
        if passed
        else None
    )
    evidence_hashes: dict[str, str] = {}
    for comparison, gates in (
        ("81_vs_101", first_gates),
        ("101_vs_121", second_gates),
    ):
        for spec_id in gates:
            path = (
                context.out_dir
                / "fit_comparisons"
                / comparison
                / spec_id
                / "fit_pair_gate.json"
            )
            evidence_hashes[f"fit/{comparison}/{spec_id}"] = _sha256(path)
    for spec_id in eap_gates:
        path = context.out_dir / "eap_bound" / spec_id / "eap_bound_gate.json"
        evidence_hashes[f"eap/{spec_id}"] = _sha256(path)
    predecessor = context.config["superseded_attempt"]
    decision = {
        "schema_version": RUNNER_SCHEMA,
        "status": "complete_pass" if passed else "blocked_numerical_followup",
        "passed": passed,
        "parent_failure_preserved": True,
        "aborted_v1_preserved": True,
        "aborted_v1_artifacts_reused": False,
        "aborted_v1_marker_sha256": predecessor["aborted_marker_sha256"],
        "aborted_v1_output_tree_sha256": predecessor["output_tree_sha256"],
        "audit_disclosure": context.config["audit_disclosure"],
        "parent_manifest_sha256": context.config["parent_study"][
            "study_manifest_sha256"
        ],
        "parent_decision_sha256": context.config["parent_study"]["decision_sha256"],
        "fit_grid_decision": dict(fit_decision),
        "eap_profile_decision": dict(eap_decision),
        "fit_comparison_all_six_passed": dict(
            fit_decision.get("fit_comparison_all_six_passed") or {}
        ),
        "effective_global_profile": effective_profile,
        "evidence_sha256": evidence_hashes,
        "passed_spec_ids": [spec.spec_id for spec in context.specs] if passed else [],
        "calibration_specification_selected": None,
        "selection_performed": False,
        "cat_results_inspected": False,
        "phase3_ready": False,
        "phase3_blocker": (
            "Phase 3 must explicitly consume and hash this v2 lock/profile; this runner "
            "does not alter the Phase-3 consumer."
            if passed
            else "Numerical follow-up did not produce a common locked profile."
        ),
    }
    decision_path = context.out_dir / "numerical_followup_decision.json"
    _write_json_once(decision_path, decision)
    lock_path = context.out_dir / "numerical_followup_lock.json"
    sha_path = context.out_dir / "numerical_followup_lock.sha256"
    if passed:
        lock = {
            "schema_version": LOCK_SCHEMA,
            "status": "complete_pass",
            "passed_spec_ids": [spec.spec_id for spec in context.specs],
            "common_fit_grid": fit_decision["locked_common_fit_grid"],
            **dict(eap_decision["locked_common_eap_profile"]),
            "fit_comparison_all_six_passed": dict(
                fit_decision["fit_comparison_all_six_passed"]
            ),
            "effective_global_profile": effective_profile,
            "all_six_exact_specifications_share_profile": True,
            "config_sha256": _sha256(context.config_path),
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "parent_manifest_sha256": context.config["parent_study"][
                "study_manifest_sha256"
            ],
            "parent_decision_sha256": context.config["parent_study"][
                "decision_sha256"
            ],
            "aborted_v1_marker_sha256": predecessor["aborted_marker_sha256"],
            "aborted_v1_output_tree_sha256": predecessor["output_tree_sha256"],
            "aborted_v1_artifacts_reused": False,
            "followup_decision_sha256": _sha256(decision_path),
            "evidence_sha256": evidence_hashes,
            "calibration_specification_selected": None,
            "selection_performed": False,
            "cat_results_inspected": False,
            "historical_parent_result": "preserved_blocked_numerical_equivalence",
            "historical_v1_attempt": "preserved_aborted_before_fit_comparison",
            "audit_disclosure": context.config["audit_disclosure"],
        }
        _write_json_once(lock_path, lock)
        _write_text_once(
            sha_path, f"{_sha256(lock_path)}  numerical_followup_lock.json\n"
        )
    elif lock_path.exists() or sha_path.exists():
        raise FollowupV2Error("failed follow-up must not retain a lock")
    manifest_path = context.out_dir / "study_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "running":
        raise FollowupV2Error("terminal manifest evidence cannot be replaced")
    manifest.update(
        {
            "status": decision["status"],
            "finished_at": v1.parent_runner._utcnow(),
            "decision_sha256": _sha256(decision_path),
            "lock_sha256": _sha256(lock_path) if lock_path.is_file() else None,
            "parent_failure_preserved": True,
            "aborted_v1_preserved": True,
            "aborted_v1_artifacts_reused": False,
            "selection_performed": False,
            "cat_results_inspected": False,
        }
    )
    _atomic_json(manifest_path, manifest)
    return decision


def _install_runtime_contract() -> None:
    v1.CONFIG_SCHEMA = CONFIG_SCHEMA
    v1.RUNNER_SCHEMA = RUNNER_SCHEMA
    v1.CHECKPOINT_SCHEMA = CHECKPOINT_SCHEMA
    v1.LOCK_SCHEMA = LOCK_SCHEMA
    v1.DEFAULT_CONFIG = DEFAULT_CONFIG
    v1.CODE_DEPENDENCIES = CODE_DEPENDENCIES
    v1._assert_safe_output = _assert_safe_output
    v1._prepare = _prepare
    v1._checkpoint_reusable = _checkpoint_reusable
    v1._write_checkpoint = _write_checkpoint
    v1._run_pair_gate = _run_pair_gate


def run(context: v1.Context, *, resume: bool) -> dict[str, Any]:
    _install_runtime_contract()
    _prepare(context, resume=resume)
    matrix, q_by, structure, item_to_scenario = v1._runtime_objects(context)
    for spec in context.specs:
        for scope in context.scopes:
            v1._import_grid81(context, spec, scope, resume=resume)
            v1._run_fit(
                context,
                spec,
                101,
                scope,
                matrix,
                q_by,
                structure,
                resume=resume,
            )
    first = {
        spec.spec_id: _run_pair_gate(
            context, spec, 81, 101, matrix, item_to_scenario, resume=resume
        )
        for spec in context.specs
    }
    for spec in context.specs:
        for scope in context.scopes:
            v1._run_fit(
                context,
                spec,
                121,
                scope,
                matrix,
                q_by,
                structure,
                resume=resume,
            )
    second = {
        spec.spec_id: _run_pair_gate(
            context, spec, 101, 121, matrix, item_to_scenario, resume=resume
        )
        for spec in context.specs
    }
    fit_decision = v1.select_common_fit_grid(
        [spec.spec_id for spec in context.specs], first, second
    )
    _write_json_once(context.out_dir / "common_fit_grid_decision.json", fit_decision)
    eap_gates: dict[str, dict[str, Any]] = {}
    if fit_decision["passed"]:
        locked_grid = int(fit_decision["locked_common_fit_grid"])
        eap_gates = {
            spec.spec_id: v1._run_eap_gate(
                context,
                spec,
                locked_grid,
                matrix,
                item_to_scenario,
                resume=resume,
            )
            for spec in context.specs
        }
    eap_decision = v1.select_common_eap_profile(
        [spec.spec_id for spec in context.specs], eap_gates
    )
    _write_json_once(context.out_dir / "common_eap_profile_decision.json", eap_decision)
    return _finalize(context, fit_decision, first, second, eap_gates, eap_decision)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _install_runtime_contract()
        context = load_context(args.config, args.out_dir)
        if args.plan_only:
            if args.resume:
                raise FollowupV2Error("--plan-only cannot combine with --resume")
            print(json.dumps(runtime_schedule(context), indent=2, sort_keys=True))
            return 0
        decision = run(context, resume=args.resume)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["passed"] else 2
    except (
        FollowupV2Error,
        v1.FollowupError,
        v1.parent_runner.NumericalCheckError,
        cm.CalibrationError,
        ValueError,
        OSError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
