#!/usr/bin/env python3
"""Append-only all-spec numerical follow-up for InFoBench v2.

The parent preflight remains an immutable blocked result.  This runner imports
and hash-verifies its grid-81 fit bundles, adds the theta-stability check omitted
from the parent fit-grid gate, and tests a prospectively frozen 81/101/121
successive panel for all six exact calibration specifications.  One common fit
grid is locked only when every specification passes unchanged gates.

After a common fit-grid lock, fixed banks are scored under bound-8 density,
equal-step bound-8 versus bound-10, and bound-10 density comparisons.  One
common EAP profile is locked only when every specification passes.  No CAT is
run and no calibration specification is selected.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import kfold_cv_mirt as cell_cv  # noqa: E402
from scripts import run_infobench_v2_numerical_checks as parent_runner  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "infobench_v2_numerical_followup_v1.json"
CONFIG_SCHEMA = "infobench-v2-numerical-followup-v1"
RUNNER_SCHEMA = "infobench-v2-numerical-followup-run-v1"
CHECKPOINT_SCHEMA = "infobench-v2-numerical-followup-checkpoint-v1"
LOCK_SCHEMA = "infobench-v2-numerical-followup-lock-v1"

FIT_GRIDS = (81, 101, 121)
FIRST_PAIR = (81, 101)
SECOND_PAIR = (101, 121)
SOURCE_SKILLS = ("content", "format", "number", "style", "linguistic")
DIMENSIONS = "instruction_following=content+format+number+style+linguistic"
TAIL_REGIONS = {8.0: 7.5, 10.0: 9.5}
EAP_PROFILES = {
    "bound8_401": (401, 8.0),
    "bound8_801": (801, 8.0),
    "bound10_501": (501, 10.0),
    "bound10_1001": (1001, 10.0),
}

CODE_DEPENDENCIES = (
    Path(__file__).resolve(),
    ROOT / "scripts" / "run_infobench_v2_numerical_checks.py",
    ROOT / "scripts" / "calibrate_mirt.py",
    ROOT / "scripts" / "kfold_cv_mirt.py",
    ROOT / "scripts" / "scenario_cat_lib.py",
    ROOT / "tutor_cat" / "mirt.py",
)


class FollowupError(RuntimeError):
    """Frozen follow-up, provenance, cache, or numerical invariant failed."""


@dataclass(frozen=True)
class ExactSpec:
    spec_id: str
    family: str
    ridge: float | None
    log_a_shrinkage: float | None
    canonical: dict[str, Any]

    @property
    def namespace(self) -> str:
        return self.spec_id


@dataclass(frozen=True)
class Scope:
    scope_id: str
    train_models: tuple[str, ...]
    score_models: tuple[str, ...]
    outer_fold: int | None


@dataclass(frozen=True)
class Context:
    config_path: Path
    config: dict[str, Any]
    parent_config: dict[str, Any]
    parent_run: Path
    parent_manifest: dict[str, Any]
    matrix_path: Path
    rubrics_path: Path
    scenarios_path: Path
    split_path: Path
    splits: dict[str, Any]
    specs: tuple[ExactSpec, ...]
    scopes: tuple[Scope, ...]
    model_to_family: dict[str, str]
    administration_scenarios: frozenset[str]
    evaluation_scenarios: frozenset[str]
    out_dir: Path
    study_signature: dict[str, Any]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    return parent_runner._sha256(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return parent_runner._read_json(path)
    except parent_runner.NumericalCheckError as error:
        raise FollowupError(str(error)) from error


def _atomic_json(path: Path, value: Any) -> None:
    parent_runner._atomic_json(path, value)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    parent_runner._atomic_csv(path, frame)


def _canonical_sha256(value: Any) -> str:
    return parent_runner._canonical_sha256(value)


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FollowupError(f"{label} is missing: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise FollowupError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}: {path}"
        )


def _validate_exact_config(raw: Mapping[str, Any]) -> None:
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise FollowupError("unexpected follow-up config schema")
    if raw.get("status") != "preregistered_before_followup_results":
        raise FollowupError("follow-up config is not preregistered")
    if raw.get("benchmark") != "InFoBench":
        raise FollowupError("follow-up requires benchmark=InFoBench")
    if (raw.get("audit_disclosure") or {}).get("no_inherited_passes") is not True:
        raise FollowupError("the follow-up may not inherit any parent pass")
    fit = raw.get("fit_grid_followup") or {}
    if tuple(map(int, fit.get("grid_candidates") or [])) != FIT_GRIDS:
        raise FollowupError("fit-grid candidates changed from 81/101/121")
    if fit.get("first_comparison") != "81_vs_101":
        raise FollowupError("first fit-grid comparison changed")
    if fit.get("second_comparison") != "101_vs_121":
        raise FollowupError("second fit-grid comparison changed")
    if fit.get("always_run_both_fit_comparisons") is not True:
        raise FollowupError("both successive fit comparisons must always run")
    if tuple(fit.get("required_comparisons") or []) != (
        "81_vs_101",
        "101_vs_121",
    ):
        raise FollowupError("required successive comparisons changed")
    if fit.get("lock_rule") != {
        "both_comparisons_pass_all_six": 81,
        "first_fails_second_passes_all_six": 101,
        "otherwise": None,
    }:
        raise FollowupError("successive common-grid lock rule changed")
    if fit.get("common_grid_required_for_all_exact_specifications") is not True:
        raise FollowupError("all specifications must share one fit grid")
    expected_fit = {
        "require_all_fits_converged": True,
        "minimum_item_parameter_spearman": 0.99,
        "minimum_exportability_agreement": 0.995,
    }
    for key, expected in expected_fit.items():
        if fit.get(key) != expected:
            raise FollowupError(f"fit gate {key} changed")
    cells = fit.get("heldout_common_cell_equivalence") or {}
    if (
        tuple(cells.get("metrics") or []) != ("log_loss", "brier")
        or float(cells.get("equivalence_margin", -1)) != 0.005
        or cells.get("require_absolute_pooled_shift_within_margin") is not True
        or cells.get("require_family_cluster_bootstrap_ci_within_margin") is not True
        or int(cells.get("bootstrap_replicates", -1)) != 2000
    ):
        raise FollowupError("held-out equivalence gate changed")
    theta = fit.get("common_support_theta_stability") or {}
    expected_theta = {
        "quadrature_method": "normal_trapezoid",
        "quadrature_nodes": 801,
        "linear_bound": 8.0,
        "maximum_median_absolute_theta_shift": 0.02,
        "maximum_p95_absolute_theta_shift": 0.05,
        "maximum_absolute_theta_shift": 0.005,
        "require_identical_model_scope_support_keys": True,
    }
    for key, expected in expected_theta.items():
        if theta.get(key) != expected:
            raise FollowupError(f"theta gate {key} changed")
    if tuple(theta.get("supports") or []) != ("full_bank", "administration_only"):
        raise FollowupError("theta supports changed")
    eap = raw.get("eap_bound_followup") or {}
    expected_profiles = {
        "bound8_density": ((401, 8.0, 0.04, 7.5), (801, 8.0, 0.02, 7.5)),
        "equal_step_cross_bound": ((401, 8.0, 0.04, 7.5), (501, 10.0, 0.04, 9.5)),
        "bound10_density": ((501, 10.0, 0.04, 9.5), (1001, 10.0, 0.02, 9.5)),
    }
    for comparison, expected_pair in expected_profiles.items():
        configured = eap.get(comparison) or {}
        for side, expected in zip(("lower", "upper"), expected_pair, strict=True):
            profile = configured.get(side) or {}
            observed = (
                int(profile.get("nodes", -1)),
                float(profile.get("bound", -1)),
                float(profile.get("step", -1)),
                float(profile.get("tail_region", -1)),
            )
            if profile.get("method") != "normal_trapezoid" or observed != expected:
                raise FollowupError(f"EAP profile changed: {comparison}/{side}")
    thresholds = eap.get("theta_stability_thresholds") or {}
    if thresholds != {
        "maximum_median_absolute_theta_shift": 0.02,
        "maximum_p95_absolute_theta_shift": 0.05,
        "maximum_absolute_theta_shift": 0.005,
    }:
        raise FollowupError("EAP theta thresholds changed")
    if float(eap.get("maximum_posterior_tail_mass", -1)) != 0.00001:
        raise FollowupError("posterior-tail threshold changed")
    if not str(raw.get("threshold_change_policy", "")).startswith("No threshold"):
        raise FollowupError("threshold-relaxation prohibition is missing")


def _parse_specs(raw: Mapping[str, Any]) -> tuple[ExactSpec, ...]:
    records = raw.get("exact_specifications")
    if not isinstance(records, list) or len(records) != 6:
        raise FollowupError("follow-up must retain all six exact specifications")
    output: list[ExactSpec] = []
    for record in records:
        family = str(record.get("family") or "")
        ridge = None if record.get("ridge") is None else float(record["ridge"])
        shrinkage = (
            None if record.get("log_a_shrinkage") is None else float(record["log_a_shrinkage"])
        )
        canonical = cm.calibration_specification(
            family,
            ridge=0.0 if ridge is None else ridge,
            log_a_shrinkage=(cm.DEFAULT_LOG_A_SHRINKAGE if shrinkage is None else shrinkage),
        )
        if canonical["cache_key"] != record.get("canonical_cache_key"):
            raise FollowupError(f"canonical cache key mismatch: {record.get('spec_id')}")
        output.append(
            ExactSpec(
                spec_id=str(record["spec_id"]),
                family=family,
                ridge=ridge,
                log_a_shrinkage=shrinkage,
                canonical=canonical,
            )
        )
    expected_ids = (
        "1pl_fixed_a1",
        "log_shrinkage_2pl_lambda16",
        "log_shrinkage_2pl_lambda4",
        "free_2pl_ridge_0p1",
        "free_2pl_ridge_0p01",
        "free_2pl_ridge_0p001",
    )
    if tuple(spec.spec_id for spec in output) != expected_ids:
        raise FollowupError("exact specification IDs/order changed")
    return tuple(output)


def _parent_checkpoint_path(parent_run: Path, spec: ExactSpec, scope: Scope) -> Path:
    return parent_run / "checkpoints" / f"fit__{spec.namespace}__grid_081__{scope.scope_id}.json"


def _verify_parent_checkpoint(context: Context, spec: ExactSpec, scope: Scope) -> dict[str, Any]:
    key = f"{spec.spec_id}/{scope.scope_id}"
    expected_hash = context.config["parent_study"]["grid81_checkpoint_sha256"].get(key)
    if not expected_hash:
        raise FollowupError(f"parent checkpoint hash is not frozen: {key}")
    marker_path = _parent_checkpoint_path(context.parent_run, spec, scope)
    _require_hash(marker_path, str(expected_hash), f"parent grid81 checkpoint {key}")
    marker = _read_json(marker_path)
    expected_stage = f"fit/{spec.namespace}/grid_081/{scope.scope_id}"
    if (
        marker.get("schema_version") != parent_runner.CHECKPOINT_SCHEMA
        or marker.get("status") != "completed"
        or marker.get("stage_id") != expected_stage
        or marker.get("study_signature_sha256")
        != context.config["parent_study"]["required_study_signature_sha256"]
    ):
        raise FollowupError(f"parent checkpoint provenance differs: {key}")
    outputs = marker.get("output_sha256") or {}
    required_names = {"fit.npz", "item_params.csv", "fit_manifest.json"}
    by_name: dict[str, tuple[Path, str]] = {}
    for recorded, expected in outputs.items():
        path = _resolve(recorded)
        _require_hash(path, str(expected), f"parent checkpoint output {key}")
        if path.name in required_names:
            by_name[path.name] = (path, str(expected))
    if set(by_name) != required_names:
        raise FollowupError(f"parent checkpoint lacks exact fit bundle: {key}")
    manifest = _read_json(by_name["fit_manifest.json"][0])
    inputs = manifest.get("stage_inputs") or {}
    if (
        int(inputs.get("fit_grid", -1)) != 81
        or inputs.get("scope") != scope.scope_id
        or inputs.get("spec_id") != spec.spec_id
        or list(map(str, inputs.get("train_model_ids") or [])) != list(scope.train_models)
        or (inputs.get("canonical_specification") or {}).get("cache_key")
        != spec.canonical["cache_key"]
    ):
        raise FollowupError(f"parent fit manifest differs from frozen scope/spec: {key}")
    return {name: {"path": path, "sha256": digest} for name, (path, digest) in by_name.items()}


def load_context(config_path: Path = DEFAULT_CONFIG, out_dir: Path | None = None) -> Context:
    config_path = config_path.resolve()
    raw = _read_json(config_path)
    _validate_exact_config(raw)
    parent = raw["parent_study"]
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
        raise FollowupError("parent study is not the frozen blocked result")
    if parent_decision.get("passed") is not parent["required_decision_passed"]:
        raise FollowupError("parent decision no longer records a failure")
    if parent_manifest.get("study_signature_sha256") != parent["required_study_signature_sha256"]:
        raise FollowupError("parent study signature changed")
    specs = _parse_specs(raw)
    parent_specs = parent_runner._parse_specs(parent_config)
    if [spec.spec_id for spec in parent_specs] != [spec.spec_id for spec in specs]:
        raise FollowupError("follow-up candidate panel differs from parent")

    frozen = raw["frozen_inputs"]
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
    parent_scopes, family, administration, evaluation = parent_runner._parse_scopes(
        splits, list(map(str, matrix.index))
    )
    scopes = tuple(
        Scope(scope.scope_id, scope.train_models, scope.score_models, scope.outer_fold)
        for scope in parent_scopes
    )
    configured_out = _resolve(raw["output_dir"])
    chosen_out = configured_out if out_dir is None else out_dir.resolve()
    if chosen_out != configured_out:
        raise FollowupError("--out-dir must equal the preregistered output leaf")
    if _resolve(raw["lock_path"]) != chosen_out / "numerical_followup_lock.json":
        raise FollowupError("lock path is outside the follow-up output")

    provisional = Context(
        config_path=config_path,
        config=raw,
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
        model_to_family=family,
        administration_scenarios=administration,
        evaluation_scenarios=evaluation,
        out_dir=chosen_out,
        study_signature={},
    )
    parent_checkpoint_hashes: dict[str, str] = {}
    for spec in specs:
        for scope in scopes:
            _verify_parent_checkpoint(provisional, spec, scope)
            key = f"{spec.spec_id}/{scope.scope_id}"
            parent_checkpoint_hashes[key] = str(
                raw["parent_study"]["grid81_checkpoint_sha256"][key]
            )
    code_hashes = {_display(path): _sha256(path) for path in CODE_DEPENDENCIES}
    environment = parent_runner._environment()
    signature_payload = {
        "schema_version": RUNNER_SCHEMA,
        "config_sha256": _sha256(config_path),
        "parent_manifest_sha256": _sha256(parent_manifest_path),
        "parent_decision_sha256": _sha256(parent_decision_path),
        "parent_checkpoint_sha256": parent_checkpoint_hashes,
        "input_sha256": {
            _display(path): _sha256(path)
            for path in (matrix_path, rubrics_path, scenarios_path, split_path)
        },
        "code_sha256": code_hashes,
        "environment_sha256": environment["canonical_sha256"],
        "spec_cache_keys": [spec.canonical["cache_key"] for spec in specs],
        "fit_grids": list(FIT_GRIDS),
        "eap_profiles": EAP_PROFILES,
    }
    signature = {
        **signature_payload,
        "environment": environment,
        "canonical_sha256": _canonical_sha256(signature_payload),
    }
    return Context(**{**provisional.__dict__, "study_signature": signature})


def runtime_schedule(context: Context) -> dict[str, Any]:
    per_grid = len(context.specs) * len(context.scopes)
    return {
        "cat_runs": 0,
        "calibration_model_selection_runs": 0,
        "imported_grid81_fit_bundles": per_grid,
        "required_new_grid101_fits": per_grid,
        "required_new_grid121_fits": per_grid,
        "minimum_new_fits": per_grid * 2,
        "maximum_new_fits": per_grid * 2,
        "fit_pair_gates": ["81_vs_101", "101_vs_121"],
        "eap_profiles_after_fit_lock": list(EAP_PROFILES),
        "selection_performed": False,
        "blockers": [
            "any parent provenance/hash mismatch",
            "any all-six common fit-grid gate failure",
            "any all-six EAP/bound theta gate failure",
            "failure of both common boundary-tail profiles",
        ],
    }


def _assert_safe_output(context: Context) -> None:
    root = (ROOT / "runs" / "calibration").resolve()
    output = context.out_dir.resolve()
    if output == root or root not in output.parents:
        raise FollowupError("follow-up output must be a leaf below runs/calibration")
    if output.name != "InFoBench_v2_numerical_followup_v1":
        raise FollowupError("refusing a non-preregistered output leaf")
    if output == context.parent_run.resolve():
        raise FollowupError("parent failed run is immutable")


def _prepare(context: Context, *, resume: bool, fresh: bool) -> None:
    _assert_safe_output(context)
    if resume and fresh:
        raise FollowupError("--resume and --fresh are mutually exclusive")
    manifest_path = context.out_dir / "study_manifest.json"
    if fresh and context.out_dir.exists():
        shutil.rmtree(context.out_dir)
    if resume:
        if not manifest_path.is_file():
            raise FollowupError("--resume requires an existing follow-up manifest")
        manifest = _read_json(manifest_path)
        if manifest.get("study_signature_sha256") != context.study_signature["canonical_sha256"]:
            raise FollowupError("resume signature differs from current inputs/code/environment")
        if manifest.get("status") not in {"running", "blocked_numerical_followup"}:
            raise FollowupError(f"cannot resume status {manifest.get('status')!r}")
        return
    if context.out_dir.exists():
        raise FollowupError("follow-up output exists; use --resume or scoped --fresh")
    context.out_dir.mkdir(parents=True)
    _atomic_json(
        manifest_path,
        {
            "schema_version": RUNNER_SCHEMA,
            "status": "running",
            "started_at": parent_runner._utcnow(),
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "study_signature": context.study_signature,
            "runtime_schedule": runtime_schedule(context),
            "audit_disclosure": context.config["audit_disclosure"],
            "parent_failure_preserved": True,
            "parent_run": _display(context.parent_run),
            "cat_results_inspected": False,
            "selection_performed": False,
        },
    )


def _fit_dir(context: Context, spec: ExactSpec, grid: int, scope: Scope) -> Path:
    return context.out_dir / "fits" / spec.namespace / f"grid_{grid:03d}" / scope.scope_id


def _checkpoint(context: Context, stage_id: str) -> Path:
    return context.out_dir / "checkpoints" / f"{stage_id.replace('/', '__')}.json"


def _checkpoint_reusable(context: Context, stage_id: str, inputs: Mapping[str, Any]) -> bool:
    path = _checkpoint(context, stage_id)
    if not path.is_file():
        return False
    marker = _read_json(path)
    if (
        marker.get("schema_version") != CHECKPOINT_SCHEMA
        or marker.get("status") != "completed"
        or marker.get("stage_id") != stage_id
        or marker.get("study_signature_sha256") != context.study_signature["canonical_sha256"]
        or marker.get("stage_inputs_sha256") != _canonical_sha256(inputs)
    ):
        raise FollowupError(f"checkpoint provenance mismatch: {stage_id}")
    outputs = marker.get("output_sha256") or {}
    if not outputs:
        raise FollowupError(f"checkpoint has no outputs: {stage_id}")
    for recorded, expected in outputs.items():
        _require_hash(_resolve(recorded), str(expected), f"checkpoint output {stage_id}")
    return True


def _write_checkpoint(
    context: Context,
    stage_id: str,
    inputs: Mapping[str, Any],
    outputs: Sequence[Path],
    elapsed: float,
) -> None:
    if any(not path.is_file() for path in outputs):
        raise FollowupError(f"cannot checkpoint incomplete stage: {stage_id}")
    _atomic_json(
        _checkpoint(context, stage_id),
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "status": "completed",
            "stage_id": stage_id,
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "stage_inputs": inputs,
            "stage_inputs_sha256": _canonical_sha256(inputs),
            "output_sha256": {_display(path): _sha256(path) for path in outputs},
            "elapsed_seconds": float(elapsed),
        },
    )


def _import_grid81(context: Context, spec: ExactSpec, scope: Scope, *, resume: bool) -> None:
    stage_id = f"import_grid081/{spec.namespace}/{scope.scope_id}"
    source = _verify_parent_checkpoint(context, spec, scope)
    inputs = {
        "parent_checkpoint_sha256": _sha256(
            _parent_checkpoint_path(context.parent_run, spec, scope)
        ),
        "source_outputs": source,
        "spec_cache_key": spec.canonical["cache_key"],
        "scope": scope.scope_id,
    }
    if resume and _checkpoint_reusable(context, stage_id, inputs):
        return
    destination = _fit_dir(context, spec, 81, scope)
    output_files = [
        destination / name for name in ("fit.npz", "item_params.csv", "fit_manifest.json")
    ]
    import_manifest = destination / "import_manifest.json"
    outputs = [*output_files, import_manifest]
    if any(path.exists() for path in outputs) or _checkpoint(context, stage_id).exists():
        raise FollowupError(f"partial import stage exists: {stage_id}")
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)
    for output in output_files:
        shutil.copy2(source[output.name]["path"], output)
        if _sha256(output) != source[output.name]["sha256"]:
            raise FollowupError(f"copied parent fit failed hash verification: {stage_id}")
    _atomic_json(
        import_manifest,
        {
            "schema_version": RUNNER_SCHEMA,
            "reuse_without_refit": True,
            "parent_checkpoint": _display(_parent_checkpoint_path(context.parent_run, spec, scope)),
            "parent_checkpoint_sha256": inputs["parent_checkpoint_sha256"],
            "source_outputs": source,
            "copied_output_sha256": {output.name: _sha256(output) for output in output_files},
        },
    )
    _write_checkpoint(context, stage_id, inputs, outputs, time.monotonic() - started)


def _runtime_objects(context: Context):
    matrix = cm.load_matrix_strict(context.matrix_path)
    cm.configure_skills(",".join(SOURCE_SKILLS))
    q_by = cm.load_q_matrix(context.rubrics_path)
    cm.validate_matrix_bank_alignment(matrix, q_by, True)
    structure = cell_cv.build_structure(
        SOURCE_SKILLS, DIMENSIONS, "overall_1d", historical_default=False
    )
    item_to_scenario = cell_cv.load_item_scenarios(context.rubrics_path)
    return matrix, q_by, structure, item_to_scenario


def _run_fit(
    context: Context,
    spec: ExactSpec,
    grid: int,
    scope: Scope,
    matrix: pd.DataFrame,
    q_by: dict[str, Any],
    structure: Any,
    *,
    resume: bool,
) -> None:
    if grid not in {101, 121}:
        raise FollowupError("follow-up may fit only grids 101/121")
    stage_id = f"fit/{spec.namespace}/grid_{grid:03d}/{scope.scope_id}"
    inputs = {
        "spec_id": spec.spec_id,
        "canonical_specification": spec.canonical,
        "fit_grid": grid,
        "scope": scope.scope_id,
        "train_model_ids": list(scope.train_models),
        "matrix_sha256": _sha256(context.matrix_path),
        "rubrics_sha256": _sha256(context.rubrics_path),
        "split_manifest_sha256": _sha256(context.split_path),
        "max_iter": int(context.config["runtime"]["fit_max_iter"]),
        "tolerance": float(context.config["runtime"]["fit_tolerance"]),
    }
    if resume and _checkpoint_reusable(context, stage_id, inputs):
        return
    destination = _fit_dir(context, spec, grid, scope)
    outputs = [
        destination / "fit.npz",
        destination / "item_params.csv",
        destination / "fit_manifest.json",
    ]
    if any(path.exists() for path in outputs) or _checkpoint(context, stage_id).exists():
        raise FollowupError(f"partial fit stage exists: {stage_id}")
    started = time.monotonic()
    args = argparse.Namespace(
        grid=grid,
        ridge=0.0 if spec.ridge is None else spec.ridge,
        max_iter=inputs["max_iter"],
        tol=inputs["tolerance"],
        estimate_latent_corr=False,
        calibration_model=spec.family,
        log_a_shrinkage=(
            cm.DEFAULT_LOG_A_SHRINKAGE if spec.log_a_shrinkage is None else spec.log_a_shrinkage
        ),
    )
    fit = cell_cv.fit_structure(matrix.loc[list(scope.train_models)], q_by, args, structure)
    if fit.get("calibration_specification", {}).get("cache_key") != spec.canonical["cache_key"]:
        raise FollowupError(f"fitter returned wrong exact spec: {spec.spec_id}")
    destination.mkdir(parents=True, exist_ok=True)
    parent_runner._atomic_fit(outputs[0], fit)
    parameters = parent_runner._parameter_frame(fit, spec)
    _atomic_csv(outputs[1], parameters)
    _atomic_json(
        outputs[2],
        {
            "schema_version": RUNNER_SCHEMA,
            "stage_inputs": inputs,
            "stage_inputs_sha256": _canonical_sha256(inputs),
            "converged": bool(fit["converged"]),
            "n_iter": int(fit["n_iter"]),
            "n_items": len(fit["items"]),
            "n_exportable": int(parameters["exportable"].sum()),
            "n_params": int(fit["n_params"]),
            "loglik": float(fit["loglik"]),
            "fit_sha256": _sha256(outputs[0]),
            "parameter_csv_sha256": _sha256(outputs[1]),
            "diagnostics": fit.get("diag") or {},
        },
    )
    _write_checkpoint(context, stage_id, inputs, outputs, time.monotonic() - started)


def _load_fit(context: Context, spec: ExactSpec, grid: int, scope: Scope):
    return parent_runner._load_fit(_fit_dir(context, spec, grid, scope) / "fit.npz")


def _quadrature(nodes: int, bound: float):
    return scat.build_quadrature(
        1,
        nodes,
        np.eye(1),
        max_nodes=max(5000, nodes),
        method="normal_trapezoid",
        linear_bound=bound,
    )


def _posterior(
    raw: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    indices: np.ndarray,
    *,
    nodes: int,
    bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quadrature = _quadrature(nodes, bound)
    selected = np.asarray(indices, dtype=int)
    values = np.asarray(raw[:, selected], dtype=float)
    observed = np.isfinite(values)
    y = np.nan_to_num(values, nan=0.0)
    successes = np.where(observed, y, 0.0)
    failures = np.where(observed, 1.0 - y, 0.0)
    eta = a[selected] @ quadrature.grid.T - b[selected, None]
    likelihood = successes @ log_expit(eta) + failures @ log_expit(-eta)
    joint = likelihood + quadrature.log_prior[None, :]
    posterior = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    theta = (posterior @ quadrature.grid)[:, 0]
    centered = quadrature.grid[:, 0][None, :] - theta[:, None]
    se = np.sqrt(np.clip((posterior * centered**2).sum(axis=1), 0.0, None))
    tail = posterior[:, np.abs(quadrature.grid[:, 0]) >= TAIL_REGIONS[bound]].sum(axis=1)
    return theta, se, tail


def theta_shift_gate(
    rows: pd.DataFrame,
    *,
    lower_column: str,
    upper_column: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    keys = ["scope", "model", "support"]
    required = set(keys + [lower_column, upper_column])
    if required - set(rows.columns) or rows.empty or rows.duplicated(keys).any():
        raise FollowupError("theta comparison rows are incomplete or duplicated")
    shift = np.abs(rows[upper_column].to_numpy(float) - rows[lower_column].to_numpy(float))
    finite = bool(np.isfinite(shift).all())
    median = float(np.median(shift)) if finite else math.inf
    p95 = float(np.quantile(shift, 0.95)) if finite else math.inf
    maximum = float(np.max(shift)) if finite else math.inf
    passed = bool(
        finite
        and median <= float(thresholds["maximum_median_absolute_theta_shift"])
        and p95 <= float(thresholds["maximum_p95_absolute_theta_shift"])
        and maximum <= float(thresholds["maximum_absolute_theta_shift"])
    )
    worst_index = int(np.argmax(shift)) if len(shift) else 0
    worst = rows.iloc[worst_index]
    return {
        "n_rows": len(rows),
        "keys_identical": True,
        "median_absolute_theta_shift": median,
        "p95_absolute_theta_shift": p95,
        "maximum_absolute_theta_shift": maximum,
        "worst_case": {
            "scope": str(worst["scope"]),
            "model": str(worst["model"]),
            "support": str(worst["support"]),
            "absolute_theta_shift": float(shift[worst_index]),
        },
        "thresholds": {
            key: float(thresholds[key])
            for key in (
                "maximum_median_absolute_theta_shift",
                "maximum_p95_absolute_theta_shift",
                "maximum_absolute_theta_shift",
            )
        },
        "passed": passed,
    }


def _common_fit_support(
    context: Context,
    spec: ExactSpec,
    lower_grid: int,
    upper_grid: int,
    scope: Scope,
    matrix: pd.DataFrame,
    item_to_scenario: Mapping[str, str],
):
    fits = {grid: _load_fit(context, spec, grid, scope) for grid in (lower_grid, upper_grid)}
    parameters = {
        grid: pd.read_csv(_fit_dir(context, spec, grid, scope) / "item_params.csv").set_index(
            "criterion_id"
        )
        for grid in (lower_grid, upper_grid)
    }
    if any(not frame.index.is_unique for frame in parameters.values()):
        raise FollowupError("fit parameter table contains duplicate criteria")
    common = sorted(
        set(parameters[lower_grid].index[parameters[lower_grid]["exportable"].astype(bool)])
        & set(parameters[upper_grid].index[parameters[upper_grid]["exportable"].astype(bool)])
    )
    admin = [
        item
        for item in common
        if item_to_scenario.get(item, item) in context.administration_scenarios
    ]
    evaluation = [
        item for item in common if item_to_scenario.get(item, item) in context.evaluation_scenarios
    ]
    if not common or not admin or (scope.outer_fold is not None and not evaluation):
        raise FollowupError(f"common support is incomplete: {spec.spec_id}/{scope.scope_id}")
    raw = matrix.loc[list(scope.score_models)].reindex(columns=common).to_numpy(float)
    positions = {item: index for index, item in enumerate(common)}
    admin_index = np.asarray([positions[item] for item in admin], dtype=int)
    evaluation_index = np.asarray([positions[item] for item in evaluation], dtype=int)
    arrays = {
        grid: parent_runner._fit_arrays_on_ids(fits[grid], common)
        for grid in (lower_grid, upper_grid)
    }
    return common, admin, evaluation, raw, admin_index, evaluation_index, arrays


def _fit_pair_evidence(
    context: Context,
    spec: ExactSpec,
    lower_grid: int,
    upper_grid: int,
    matrix: pd.DataFrame,
    item_to_scenario: Mapping[str, str],
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    theta_rows: list[dict[str, Any]] = []
    prediction_rows: dict[int, list[dict[str, Any]]] = {
        lower_grid: [],
        upper_grid: [],
    }
    support_rows: list[dict[str, Any]] = []
    for scope in context.scopes:
        common, admin, evaluation, raw, admin_idx, eval_idx, arrays = _common_fit_support(
            context,
            spec,
            lower_grid,
            upper_grid,
            scope,
            matrix,
            item_to_scenario,
        )
        support_rows.append(
            {
                "scope": scope.scope_id,
                "n_common_exportable_items": len(common),
                "n_common_administration_items": len(admin),
                "n_common_evaluation_items": len(evaluation),
                "n_score_models": len(scope.score_models),
            }
        )
        theta_by_grid: dict[int, dict[str, np.ndarray]] = {}
        for grid in (lower_grid, upper_grid):
            a, b = arrays[grid]
            theta_by_grid[grid] = {}
            for support, indices in (
                ("full_bank", np.arange(len(common), dtype=int)),
                ("administration_only", admin_idx),
            ):
                theta, _se, _tail = _posterior(raw, a, b, indices, nodes=801, bound=8.0)
                theta_by_grid[grid][support] = theta
        for model_index, model in enumerate(scope.score_models):
            for support in ("full_bank", "administration_only"):
                theta_rows.append(
                    {
                        "scope": scope.scope_id,
                        "outer_fold": scope.outer_fold,
                        "model": model,
                        "family": context.model_to_family[model],
                        "support": support,
                        f"theta_grid_{lower_grid}": float(
                            theta_by_grid[lower_grid][support][model_index]
                        ),
                        f"theta_grid_{upper_grid}": float(
                            theta_by_grid[upper_grid][support][model_index]
                        ),
                    }
                )
        if scope.outer_fold is None:
            continue
        observed = np.isfinite(raw[:, eval_idx])
        for grid in (lower_grid, upper_grid):
            a, b = arrays[grid]
            theta = theta_by_grid[grid]["administration_only"]
            probability = expit(theta[:, None] * a[eval_idx, 0][None, :] - b[eval_idx][None, :])
            for model_index, criterion_index in np.argwhere(observed):
                model = scope.score_models[int(model_index)]
                prediction_rows[grid].append(
                    {
                        "outer_fold": scope.outer_fold,
                        "model": model,
                        "family": context.model_to_family[model],
                        "criterion_id": evaluation[int(criterion_index)],
                        "label": int(raw[model_index, eval_idx[criterion_index]]),
                        "probability": float(probability[model_index, criterion_index]),
                    }
                )
    return (
        theta_rows,
        pd.DataFrame(prediction_rows[lower_grid]),
        pd.DataFrame(prediction_rows[upper_grid]),
        support_rows,
    )


def _run_pair_gate(
    context: Context,
    spec: ExactSpec,
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
        "spec_id": spec.spec_id,
        "fit_sha256": {
            f"{grid}/{scope.scope_id}": _sha256(_fit_dir(context, spec, grid, scope) / "fit.npz")
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
    if any(path.exists() for path in outputs) or _checkpoint(context, stage_id).exists():
        raise FollowupError(f"partial fit-pair stage exists: {stage_id}")
    started = time.monotonic()
    fit_config = context.config["fit_grid_followup"]
    parameter_rows: list[dict[str, Any]] = []
    for scope in context.scopes:
        dirs = {grid: _fit_dir(context, spec, grid, scope) for grid in (lower_grid, upper_grid)}
        manifests = {
            grid: _read_json(dirs[grid] / "fit_manifest.json") for grid in (lower_grid, upper_grid)
        }
        gate = parent_runner.parameter_scope_gate(
            pd.read_csv(dirs[lower_grid] / "item_params.csv"),
            pd.read_csv(dirs[upper_grid] / "item_params.csv"),
            family=spec.family,
            source_ids=list(map(str, matrix.columns)),
            minimum_spearman=float(fit_config["minimum_item_parameter_spearman"]),
            minimum_exportability_agreement=float(fit_config["minimum_exportability_agreement"]),
            lower_converged=bool(manifests[lower_grid]["converged"]),
            upper_converged=bool(manifests[upper_grid]["converged"]),
        )
        parameter_rows.append({"scope": scope.scope_id, **gate})
    theta_rows, lower_cells, upper_cells, support_rows = _fit_pair_evidence(
        context, spec, lower_grid, upper_grid, matrix, item_to_scenario
    )
    cell_config = fit_config["heldout_common_cell_equivalence"]
    seed = int(cell_config["bootstrap_seed"]) + int(spec.canonical["cache_key"][-8:], 16)
    cell_gate, cell_details = parent_runner.common_cell_equivalence(
        lower_cells,
        upper_cells,
        log_loss_margin=float(cell_config["equivalence_margin"]),
        brier_margin=float(cell_config["equivalence_margin"]),
        replicates=int(cell_config["bootstrap_replicates"]),
        seed=seed,
    )
    theta_frame = pd.DataFrame(theta_rows)
    theta_gate = theta_shift_gate(
        theta_frame,
        lower_column=f"theta_grid_{lower_grid}",
        upper_column=f"theta_grid_{upper_grid}",
        thresholds=fit_config["common_support_theta_stability"],
    )
    parameter_passed = all(bool(row["passed"]) for row in parameter_rows)
    result = {
        "schema_version": RUNNER_SCHEMA,
        "spec_id": spec.spec_id,
        "canonical_cache_key": spec.canonical["cache_key"],
        "comparison": comparison,
        "all_scope_parameter_gate_passed": parameter_passed,
        "heldout_common_cell_gate": cell_gate,
        "common_support_theta_gate": theta_gate,
        "theta_gate_was_omitted_from_parent": True,
        "cat_results_inspected": False,
        "selection_performed": False,
        "passed": bool(parameter_passed and cell_gate["passed"] and theta_gate["passed"]),
    }
    _atomic_csv(outputs[0], pd.DataFrame(parameter_rows))
    _atomic_csv(outputs[1], pd.DataFrame(support_rows))
    _atomic_csv(outputs[2], theta_frame)
    _atomic_csv(outputs[3], cell_details)
    _atomic_json(outputs[4], result)
    _write_checkpoint(context, stage_id, inputs, outputs, time.monotonic() - started)
    return result


def select_common_fit_grid(
    spec_ids: Sequence[str],
    first_gates: Mapping[str, Mapping[str, Any]],
    second_gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required = set(map(str, spec_ids))
    first_complete = set(first_gates) == required
    second_complete = set(second_gates) == required
    first_passed = first_complete and all(
        gate.get("passed") is True for gate in first_gates.values()
    )
    second_passed = second_complete and all(
        gate.get("passed") is True for gate in second_gates.values()
    )
    if first_passed and second_passed:
        locked = 81
        status = "locked_grid81_both_successive_panels_passed_all_six"
    elif not first_passed and second_passed:
        locked = 101
        status = "locked_grid101_second_panel_passed_all_six"
    else:
        locked = None
        status = "blocked_no_common_fit_grid"
    return {
        "status": status,
        "required_spec_ids": list(spec_ids),
        "required_comparisons": ["81_vs_101", "101_vs_121"],
        "always_ran_both_fit_comparisons": True,
        "first_comparison_complete": first_complete,
        "first_comparison_passed_all_six": bool(first_passed),
        "second_comparison_complete": second_complete,
        "second_comparison_passed_all_six": bool(second_passed),
        "fit_comparison_all_six_passed": {
            "81_vs_101": bool(first_passed),
            "101_vs_121": bool(second_passed),
        },
        "locked_common_fit_grid": locked,
        "selection_performed": False,
        "passed": locked is not None,
    }


def _run_eap_gate(
    context: Context,
    spec: ExactSpec,
    fit_grid: int,
    matrix: pd.DataFrame,
    item_to_scenario: Mapping[str, str],
    *,
    resume: bool,
) -> dict[str, Any]:
    stage_id = f"eap_bound/{spec.namespace}/fit_grid_{fit_grid:03d}"
    inputs = {
        "spec_id": spec.spec_id,
        "fit_grid": fit_grid,
        "fit_sha256": {
            scope.scope_id: _sha256(_fit_dir(context, spec, fit_grid, scope) / "fit.npz")
            for scope in context.scopes
        },
        "eap_bound_followup": context.config["eap_bound_followup"],
    }
    out = context.out_dir / "eap_bound" / spec.namespace
    outputs = [out / "per_model_scope.csv", out / "eap_bound_gate.json"]
    if resume and _checkpoint_reusable(context, stage_id, inputs):
        return _read_json(outputs[-1])
    if any(path.exists() for path in outputs) or _checkpoint(context, stage_id).exists():
        raise FollowupError(f"partial EAP-bound stage exists: {stage_id}")
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for scope in context.scopes:
        fit = _load_fit(context, spec, fit_grid, scope)
        params = pd.read_csv(
            _fit_dir(context, spec, fit_grid, scope) / "item_params.csv"
        ).set_index("criterion_id")
        exportable = list(params.index[params["exportable"].astype(bool)])
        if not exportable:
            raise FollowupError(f"no exportable bank: {spec.spec_id}/{scope.scope_id}")
        a, b = parent_runner._fit_arrays_on_ids(fit, exportable)
        raw = matrix.loc[list(scope.score_models)].reindex(columns=exportable).to_numpy(float)
        admin_indices = np.asarray(
            [
                index
                for index, item in enumerate(exportable)
                if item_to_scenario.get(item, item) in context.administration_scenarios
            ],
            dtype=int,
        )
        for support, indices in (
            ("full_bank", np.arange(len(exportable), dtype=int)),
            ("administration_only", admin_indices),
        ):
            results = {
                name: _posterior(raw, a, b, indices, nodes=nodes, bound=bound)
                for name, (nodes, bound) in EAP_PROFILES.items()
            }
            for model_index, model in enumerate(scope.score_models):
                row: dict[str, Any] = {
                    "scope": scope.scope_id,
                    "outer_fold": scope.outer_fold,
                    "model": model,
                    "family": context.model_to_family[model],
                    "support": support,
                    "fit_grid": fit_grid,
                    "n_bank_items": len(exportable),
                    "n_scoring_items": len(indices),
                }
                for name, (nodes, bound) in EAP_PROFILES.items():
                    theta, se, tail = results[name]
                    row[f"theta_{name}"] = float(theta[model_index])
                    row[f"se_{name}"] = float(se[model_index])
                    row[f"tail_mass_{name}"] = float(tail[model_index])
                    row[f"nodes_{name}"] = nodes
                    row[f"bound_{name}"] = bound
                rows.append(row)
    frame = pd.DataFrame(rows)
    threshold = context.config["eap_bound_followup"]["theta_stability_thresholds"]
    comparisons = {
        "bound8_density": theta_shift_gate(
            frame,
            lower_column="theta_bound8_401",
            upper_column="theta_bound8_801",
            thresholds=threshold,
        ),
        "equal_step_cross_bound": theta_shift_gate(
            frame,
            lower_column="theta_bound8_401",
            upper_column="theta_bound10_501",
            thresholds=threshold,
        ),
        "bound10_density": theta_shift_gate(
            frame,
            lower_column="theta_bound10_501",
            upper_column="theta_bound10_1001",
            thresholds=threshold,
        ),
    }
    tail_threshold = float(context.config["eap_bound_followup"]["maximum_posterior_tail_mass"])
    tail: dict[str, Any] = {}
    for bound_name, columns in (
        ("bound8", ("tail_mass_bound8_401", "tail_mass_bound8_801")),
        ("bound10", ("tail_mass_bound10_501", "tail_mass_bound10_1001")),
    ):
        values = frame[list(columns)].to_numpy(float)
        location = np.unravel_index(int(np.argmax(values)), values.shape)
        worst = frame.iloc[int(location[0])]
        maximum = float(values[location])
        tail[bound_name] = {
            "maximum_posterior_tail_mass": maximum,
            "maximum_allowed": tail_threshold,
            "tail_region": TAIL_REGIONS[8.0 if bound_name == "bound8" else 10.0],
            "worst_case": {
                "scope": str(worst["scope"]),
                "model": str(worst["model"]),
                "support": str(worst["support"]),
                "profile": columns[int(location[1])],
            },
            "passed": maximum <= tail_threshold,
        }
    result = {
        "schema_version": RUNNER_SCHEMA,
        "spec_id": spec.spec_id,
        "fit_grid": fit_grid,
        "fixed_bank_for_all_eap_profiles": True,
        "theta_comparisons": comparisons,
        "tail_gates": tail,
        "all_theta_comparisons_passed": all(gate["passed"] for gate in comparisons.values()),
        "selection_performed": False,
        "cat_results_inspected": False,
    }
    _atomic_csv(outputs[0], frame)
    _atomic_json(outputs[1], result)
    _write_checkpoint(context, stage_id, inputs, outputs, time.monotonic() - started)
    return result


def select_common_eap_profile(
    spec_ids: Sequence[str], gates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    required = set(map(str, spec_ids))
    complete = set(gates) == required
    theta_passed = complete and all(
        gate.get("all_theta_comparisons_passed") is True for gate in gates.values()
    )
    bound8_tail = complete and all(
        (gate.get("tail_gates") or {}).get("bound8", {}).get("passed") is True
        for gate in gates.values()
    )
    bound10_tail = complete and all(
        (gate.get("tail_gates") or {}).get("bound10", {}).get("passed") is True
        for gate in gates.values()
    )
    if theta_passed and bound8_tail:
        profile = {
            "quadrature_method": "normal_trapezoid",
            "linear_bound": 8.0,
            "eap_grid": 401,
            "tail_region": 7.5,
        }
        status = "retained_bound8_eap401"
    elif theta_passed and bound10_tail:
        profile = {
            "quadrature_method": "normal_trapezoid",
            "linear_bound": 10.0,
            "eap_grid": 501,
            "tail_region": 9.5,
        }
        status = "locked_bound10_eap501"
    else:
        profile = None
        status = "blocked_no_common_eap_profile"
    return {
        "status": status,
        "complete_all_six_panel": complete,
        "all_theta_comparisons_passed": bool(theta_passed),
        "all_bound8_tail_gates_passed": bool(bound8_tail),
        "all_bound10_tail_gates_passed": bool(bound10_tail),
        "locked_common_eap_profile": profile,
        "selection_performed": False,
        "passed": profile is not None,
    }


def _finalize(
    context: Context,
    fit_decision: Mapping[str, Any],
    first_gates: Mapping[str, Mapping[str, Any]],
    second_gates: Mapping[str, Mapping[str, Any]] | None,
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
    for comparison, gates in (("81_vs_101", first_gates), ("101_vs_121", second_gates or {})):
        for spec_id in gates:
            path = context.out_dir / "fit_comparisons" / comparison / spec_id / "fit_pair_gate.json"
            evidence_hashes[f"fit/{comparison}/{spec_id}"] = _sha256(path)
    for spec_id in eap_gates:
        path = context.out_dir / "eap_bound" / spec_id / "eap_bound_gate.json"
        evidence_hashes[f"eap/{spec_id}"] = _sha256(path)
    decision = {
        "schema_version": RUNNER_SCHEMA,
        "status": "complete_pass" if passed else "blocked_numerical_followup",
        "passed": passed,
        "parent_failure_preserved": True,
        "audit_disclosure": context.config["audit_disclosure"],
        "parent_manifest_sha256": context.config["parent_study"]["study_manifest_sha256"],
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
            "Phase 3 must explicitly consume and hash this follow-up lock/profile; "
            "this runner does not alter the Phase-3 consumer."
            if passed
            else "Numerical follow-up did not produce a common locked profile."
        ),
    }
    decision_path = context.out_dir / "numerical_followup_decision.json"
    _atomic_json(decision_path, decision)
    lock_path = context.out_dir / "numerical_followup_lock.json"
    sha_path = context.out_dir / "numerical_followup_lock.sha256"
    if passed:
        lock = {
            "schema_version": LOCK_SCHEMA,
            "status": "complete_pass",
            "passed_spec_ids": [spec.spec_id for spec in context.specs],
            "common_fit_grid": fit_decision["locked_common_fit_grid"],
            **dict(eap_decision["locked_common_eap_profile"]),
            "fit_comparison_all_six_passed": dict(fit_decision["fit_comparison_all_six_passed"]),
            "effective_global_profile": effective_profile,
            "all_six_exact_specifications_share_profile": True,
            "config_sha256": _sha256(context.config_path),
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "parent_manifest_sha256": context.config["parent_study"]["study_manifest_sha256"],
            "parent_decision_sha256": context.config["parent_study"]["decision_sha256"],
            "followup_decision_sha256": _sha256(decision_path),
            "evidence_sha256": evidence_hashes,
            "calibration_specification_selected": None,
            "selection_performed": False,
            "cat_results_inspected": False,
            "historical_parent_result": "preserved_blocked_numerical_equivalence",
            "audit_disclosure": context.config["audit_disclosure"],
        }
        _atomic_json(lock_path, lock)
        sha_path.write_text(f"{_sha256(lock_path)}  numerical_followup_lock.json\n")
    elif lock_path.exists() or sha_path.exists():
        raise FollowupError("failed follow-up must not retain a lock")
    manifest_path = context.out_dir / "study_manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "status": decision["status"],
            "finished_at": parent_runner._utcnow(),
            "decision_sha256": _sha256(decision_path),
            "lock_sha256": _sha256(lock_path) if lock_path.is_file() else None,
            "parent_failure_preserved": True,
            "selection_performed": False,
            "cat_results_inspected": False,
        }
    )
    _atomic_json(manifest_path, manifest)
    return decision


def run(context: Context, *, resume: bool, fresh: bool) -> dict[str, Any]:
    _prepare(context, resume=resume, fresh=fresh)
    matrix, q_by, structure, item_to_scenario = _runtime_objects(context)
    for spec in context.specs:
        for scope in context.scopes:
            _import_grid81(context, spec, scope, resume=resume)
            _run_fit(
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
            _run_fit(
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
    fit_decision = select_common_fit_grid([spec.spec_id for spec in context.specs], first, second)
    fit_decision_path = context.out_dir / "common_fit_grid_decision.json"
    _atomic_json(fit_decision_path, fit_decision)
    eap_gates: dict[str, dict[str, Any]] = {}
    if fit_decision["passed"]:
        locked_grid = int(fit_decision["locked_common_fit_grid"])
        eap_gates = {
            spec.spec_id: _run_eap_gate(
                context,
                spec,
                locked_grid,
                matrix,
                item_to_scenario,
                resume=resume,
            )
            for spec in context.specs
        }
    eap_decision = select_common_eap_profile([spec.spec_id for spec in context.specs], eap_gates)
    _atomic_json(context.out_dir / "common_eap_profile_decision.json", eap_decision)
    return _finalize(context, fit_decision, first, second, eap_gates, eap_decision)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--fresh", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = load_context(args.config, args.out_dir)
        if args.plan_only:
            if args.resume or args.fresh:
                raise FollowupError("--plan-only cannot combine with --resume/--fresh")
            print(json.dumps(runtime_schedule(context), indent=2, sort_keys=True))
            return 0
        decision = run(context, resume=args.resume, fresh=args.fresh)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["passed"] else 2
    except (
        FollowupError,
        parent_runner.NumericalCheckError,
        cm.CalibrationError,
        ValueError,
        OSError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
