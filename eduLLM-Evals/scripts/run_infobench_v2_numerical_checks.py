#!/usr/bin/env python3
"""Numerical-equivalence preflight for the frozen InFoBench v2 study.

This runner is deliberately separate from the historical v1 numerical studies.
It does not select a calibration model and it never runs CAT.  For each of the
six exact, prospectively frozen calibration specifications it verifies that:

* fitting with 61 versus 81 quadrature nodes converges on the full cohort and
  repeat-0's five outer-training cohorts;
* common-item parameters and exportability are stable;
* predictions are made on exactly the same disjoint held-out response cells and
  their log-loss/Brier shifts (including family-cluster bootstrap intervals) are
  inside the frozen v1 equivalence margin; and
* holding every grid-61 bank fixed, 401- versus 801-point normal-trapezoid EAP
  scoring has stable theta estimates and negligible posterior boundary mass.

``numerical_lock.json`` is created only when every exact specification passes.
The companion ``numerical_lock.sha256`` lets Phase 3 require the exact lock
bytes.  A failed check produces a decision artifact but no lock and performs no
fallback selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
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
from scripts import scenario_cat_lib as scat  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "infobench_calibration_cat_v2.json"
CONFIG_SCHEMA = "infobench-calibration-cat-v2-v1"
SPLIT_SCHEMA = "infobench-v2-repeated-splits-v1"
RUNNER_SCHEMA = "infobench-v2-numerical-checks-v1"
CHECKPOINT_SCHEMA = "infobench-v2-numerical-checkpoint-v1"
DECISION_SCHEMA = "infobench-v2-numerical-decision-v1"
LOCK_SCHEMA = "infobench-v2-numerical-lock-v1"

EXPECTED_SPEC_SIGNATURE = (
    ("1pl", None, None),
    ("log-shrinkage-2pl", None, 16.0),
    ("log-shrinkage-2pl", None, 4.0),
    ("free-2pl", 0.1, None),
    ("free-2pl", 0.01, None),
    ("free-2pl", 0.001, None),
)
EXPECTED_SOURCE_SKILLS = ("content", "format", "number", "style", "linguistic")
EXPECTED_DIMENSIONS = "instruction_following=content+format+number+style+linguistic"
EXPECTED_REPEATS = 5
EXPECTED_OUTER_FOLDS = 5
EXPECTED_MODELS = 52
EXPECTED_FAMILIES = 22
FIT_GRIDS = (61, 81)
EAP_GRIDS = (401, 801)
LINEAR_BOUND = 8.0
TAIL_REGION = 7.5
EXTREME_A = 6.0
EPSILON = 1e-12

CODE_DEPENDENCIES = (
    Path(__file__).resolve(),
    ROOT / "scripts" / "calibrate_mirt.py",
    ROOT / "scripts" / "kfold_cv_mirt.py",
    ROOT / "scripts" / "scenario_cat_lib.py",
    ROOT / "tutor_cat" / "mirt.py",
)


class NumericalCheckError(RuntimeError):
    """A frozen-design, provenance, cache, or numerical invariant failed."""


@dataclass(frozen=True)
class ExactSpec:
    spec_id: str
    family: str
    ridge: float | None
    log_a_shrinkage: float | None
    canonical: dict[str, Any]

    @property
    def namespace(self) -> str:
        return self.spec_id.replace("/", "_").replace(" ", "_")


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
    split_path: Path
    splits: dict[str, Any]
    matrix_path: Path
    rubrics_path: Path
    scenarios_path: Path
    out_dir: Path
    specs: tuple[ExactSpec, ...]
    scopes: tuple[Scope, ...]
    model_to_family: dict[str, str]
    administration_scenarios: frozenset[str]
    evaluation_scenarios: frozenset[str]
    study_signature: dict[str, Any]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NumericalCheckError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise NumericalCheckError(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise NumericalCheckError(f"{label} is missing: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise NumericalCheckError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}: {path}"
        )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "packages": versions,
    }
    return {**payload, "canonical_sha256": _canonical_sha256(payload)}


def _code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in CODE_DEPENDENCIES:
        if not path.is_file():
            raise NumericalCheckError(f"required code dependency is missing: {path}")
        hashes[_display(path)] = _sha256(path)
    return hashes


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise NumericalCheckError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise NumericalCheckError(f"{field} must be finite")
    return result


def _parse_specs(config: Mapping[str, Any]) -> tuple[ExactSpec, ...]:
    raw_specs = config.get("calibration_specifications")
    if not isinstance(raw_specs, list) or len(raw_specs) != len(EXPECTED_SPEC_SIGNATURE):
        raise NumericalCheckError("the frozen configuration must contain exactly six specs")
    observed: list[tuple[str, float | None, float | None]] = []
    output: list[ExactSpec] = []
    seen_ids: set[str] = set()
    for rank, raw in enumerate(raw_specs):
        if not isinstance(raw, Mapping):
            raise NumericalCheckError("each calibration specification must be an object")
        spec_id = str(raw.get("spec_id") or "")
        if not spec_id or spec_id in seen_ids:
            raise NumericalCheckError("calibration spec IDs must be non-empty and unique")
        seen_ids.add(spec_id)
        if int(raw.get("simplicity_rank", -1)) != rank:
            raise NumericalCheckError("calibration simplicity ranks/order are not frozen")
        family = str(raw.get("family") or "")
        ridge = None if raw.get("ridge") is None else _number(raw["ridge"], "ridge")
        shrinkage = (
            None
            if raw.get("log_a_shrinkage") is None
            else _number(raw["log_a_shrinkage"], "log_a_shrinkage")
        )
        canonical = cm.calibration_specification(
            family,
            ridge=0.0 if ridge is None else ridge,
            log_a_shrinkage=(cm.DEFAULT_LOG_A_SHRINKAGE if shrinkage is None else shrinkage),
        )
        if str(raw.get("canonical_cache_key")) != canonical["cache_key"]:
            raise NumericalCheckError(f"canonical cache key mismatch for {spec_id}")
        if family == cm.ONE_PL and _number(raw.get("fixed_discrimination"), "fixed_a") != 1.0:
            raise NumericalCheckError("1PL must keep fixed discrimination a=1")
        observed.append((family, ridge, shrinkage))
        output.append(ExactSpec(spec_id, family, ridge, shrinkage, canonical))
    if tuple(observed) != EXPECTED_SPEC_SIGNATURE:
        raise NumericalCheckError(
            "candidate panel changed; expected 1PL, shrinkage lambda 16/4, and "
            "free-2PL ridge 0.1/0.01/0.001"
        )
    return tuple(output)


def _validate_numerical_lock(raw: Mapping[str, Any]) -> None:
    expected = {
        "fit_grid": 61,
        "fit_grid_comparator": 81,
        "eap_method": "normal_trapezoid",
        "eap_grid": 401,
        "eap_grid_comparator": 801,
        "linear_bound": 8.0,
        "verification_required_for_every_exact_specification": True,
        "fit_parameter_minimum_spearman": 0.99,
        "minimum_exportability_agreement": 0.995,
        "maximum_absolute_heldout_log_loss_shift": 0.005,
        "maximum_absolute_heldout_brier_shift": 0.005,
        "maximum_median_absolute_theta_shift": 0.02,
        "maximum_p95_absolute_theta_shift": 0.05,
        "maximum_absolute_theta_shift": 0.005,
        "maximum_posterior_tail_mass": 0.00001,
        "require_all_fits_converged": True,
    }
    for key, value in expected.items():
        observed = raw.get(key)
        if isinstance(value, float):
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), value, rel_tol=0.0, abs_tol=1e-15
            ):
                raise NumericalCheckError(f"numerical_lock.{key} changed from v1 margin")
        elif observed != value:
            raise NumericalCheckError(f"numerical_lock.{key} changed from frozen value")


def _parse_scopes(
    splits: Mapping[str, Any], matrix_models: Sequence[str]
) -> tuple[tuple[Scope, ...], dict[str, str], frozenset[str], frozenset[str]]:
    if splits.get("schema_version") != SPLIT_SCHEMA:
        raise NumericalCheckError("unexpected repeated split-manifest schema")
    repetitions = splits.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != EXPECTED_REPEATS:
        raise NumericalCheckError("numerical checks require the frozen five repetitions")
    repeat0 = next((record for record in repetitions if int(record.get("repeat", -1)) == 0), None)
    if repeat0 is None:
        raise NumericalCheckError("split manifest has no repeat 0")
    outer = repeat0.get("outer_folds")
    if not isinstance(outer, list) or len(outer) != EXPECTED_OUTER_FOLDS:
        raise NumericalCheckError("repeat 0 must contain exactly five outer folds")
    all_models = set(map(str, matrix_models))
    model_to_family = {
        str(model): str(family) for model, family in (splits.get("model_to_family") or {}).items()
    }
    if len(all_models) != EXPECTED_MODELS or set(model_to_family) != all_models:
        raise NumericalCheckError("matrix and frozen 52-model family map differ")
    if len(set(model_to_family.values())) != EXPECTED_FAMILIES:
        raise NumericalCheckError("frozen cohort must contain 22 tutor families")
    scopes: list[Scope] = [
        Scope("full", tuple(map(str, matrix_models)), tuple(map(str, matrix_models)), None)
    ]
    seen_test: set[str] = set()
    for record in sorted(outer, key=lambda value: int(value["outer_fold"])):
        fold = int(record["outer_fold"])
        train = tuple(map(str, record.get("train_model_ids") or []))
        test = tuple(map(str, record.get("test_model_ids") or []))
        if set(train) & set(test) or set(train) | set(test) != all_models:
            raise NumericalCheckError(f"repeat-0 outer fold {fold} is not a partition")
        if {model_to_family[m] for m in train} & {model_to_family[m] for m in test}:
            raise NumericalCheckError(f"repeat-0 outer fold {fold} leaks a model family")
        if seen_test & set(test):
            raise NumericalCheckError("a repeat-0 model appears in multiple outer tests")
        seen_test.update(test)
        scopes.append(Scope(f"outer_{fold}", train, test, fold))
    if seen_test != all_models:
        raise NumericalCheckError("repeat 0 does not outer-test every model exactly once")
    scenario_split = splits.get("scenario_split") or {}
    administration = frozenset(map(str, scenario_split.get("administration_scenario_ids") or []))
    evaluation = frozenset(map(str, scenario_split.get("evaluation_scenario_ids") or []))
    if not administration or not evaluation or administration & evaluation:
        raise NumericalCheckError("frozen administration/evaluation scenarios are invalid")
    return tuple(scopes), model_to_family, administration, evaluation


def load_context(config_path: Path = DEFAULT_CONFIG, out_dir: Path | None = None) -> Context:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise NumericalCheckError("unexpected v2 configuration schema")
    if config.get("status") != "frozen_pre_run" or config.get("benchmark") != "InFoBench":
        raise NumericalCheckError("v2 configuration is not the frozen InFoBench design")
    specs = _parse_specs(config)
    numerical = config.get("numerical_lock") or {}
    _validate_numerical_lock(numerical)

    baseline = config.get("baseline") or {}
    matrix_path = _resolve(baseline["response_matrix"])
    rubrics_path = _resolve(baseline["rubrics"])
    scenarios_path = _resolve(baseline["scenarios"])
    for path, expected, label in (
        (matrix_path, baseline["response_matrix_sha256"], "response matrix"),
        (rubrics_path, baseline["rubrics_sha256"], "rubrics"),
        (scenarios_path, baseline["scenarios_sha256"], "scenarios"),
        (_resolve(baseline["judge_manifest"]), baseline["judge_manifest_sha256"], "judge manifest"),
    ):
        _require_file_hash(path, str(expected), label)

    cross = config.get("cross_validation") or {}
    split_path = _resolve(cross["split_manifest"])
    _require_file_hash(split_path, str(cross["split_manifest_sha256"]), "split manifest")
    _require_file_hash(
        _resolve(cross["split_config"]), str(cross["split_config_sha256"]), "split config"
    )
    _require_file_hash(
        _resolve(cross["model_assignments"]),
        str(cross["model_assignments_sha256"]),
        "model assignments",
    )
    _require_file_hash(
        _resolve(cross["scenario_assignments"]),
        str(cross["scenario_assignments_sha256"]),
        "scenario assignments",
    )
    matrix = cm.load_matrix_strict(matrix_path)
    splits = _read_json(split_path)
    scopes, model_to_family, administration, evaluation = _parse_scopes(
        splits, list(map(str, matrix.index))
    )

    latent = config.get("latent_structure") or {}
    if tuple(map(str, latent.get("source_skills") or [])) != EXPECTED_SOURCE_SKILLS:
        raise NumericalCheckError("source skill order differs from frozen v2 structure")
    dimensions = latent.get("dimensions") or []
    if len(dimensions) != 1 or str(dimensions[0].get("label")) != "instruction_following":
        raise NumericalCheckError("numerical checks require frozen one-dimensional structure")
    if tuple(map(str, dimensions[0].get("members") or [])) != EXPECTED_SOURCE_SKILLS:
        raise NumericalCheckError("instruction_following members changed")

    configured_output = _resolve((config.get("outputs") or {})["numerical_checks"])
    expected_lock = _resolve(numerical["expected_lock"])
    if expected_lock != configured_output / "numerical_lock.json":
        raise NumericalCheckError("expected numerical lock is outside configured output")
    chosen_output = configured_output if out_dir is None else out_dir.resolve()
    if chosen_output != configured_output:
        raise NumericalCheckError("--out-dir must equal the prospectively frozen output path")

    code_hashes = _code_hashes()
    environment = _environment()
    input_hashes = {
        _display(config_path): _sha256(config_path),
        _display(matrix_path): _sha256(matrix_path),
        _display(rubrics_path): _sha256(rubrics_path),
        _display(scenarios_path): _sha256(scenarios_path),
        _display(split_path): _sha256(split_path),
        _display(_resolve(baseline["judge_manifest"])): str(baseline["judge_manifest_sha256"]),
        _display(_resolve(cross["split_config"])): str(cross["split_config_sha256"]),
        _display(_resolve(cross["model_assignments"])): str(cross["model_assignments_sha256"]),
        _display(_resolve(cross["scenario_assignments"])): str(
            cross["scenario_assignments_sha256"]
        ),
    }
    signature_payload = {
        "schema_version": RUNNER_SCHEMA,
        "config_sha256": _sha256(config_path),
        "input_hashes": input_hashes,
        "code_hashes": code_hashes,
        "environment_sha256": environment["canonical_sha256"],
        "fit_grids": list(FIT_GRIDS),
        "eap_grids": list(EAP_GRIDS),
        "scope_ids": [scope.scope_id for scope in scopes],
        "exact_spec_cache_keys": [spec.canonical["cache_key"] for spec in specs],
    }
    study_signature = {
        **signature_payload,
        "canonical_sha256": _canonical_sha256(signature_payload),
        "git_commit": _git_commit(),
        "environment": environment,
    }
    return Context(
        config_path=config_path,
        config=config,
        split_path=split_path,
        splits=splits,
        matrix_path=matrix_path,
        rubrics_path=rubrics_path,
        scenarios_path=scenarios_path,
        out_dir=chosen_output,
        specs=specs,
        scopes=scopes,
        model_to_family=model_to_family,
        administration_scenarios=administration,
        evaluation_scenarios=evaluation,
        study_signature=study_signature,
    )


def runtime_schedule(context: Context) -> dict[str, Any]:
    fit_count = len(context.specs) * len(FIT_GRIDS) * len(context.scopes)
    return {
        "response_dependent_results_generated": False,
        "cat_runs": 0,
        "calibration_specifications": len(context.specs),
        "fit_grids": list(FIT_GRIDS),
        "fit_scopes_per_spec_grid": len(context.scopes),
        "fit_invocations": fit_count,
        "fit_comparison_stages": len(context.specs),
        "fixed_bank_eap_comparison_stages": len(context.specs),
        "eap_grid_evaluations": len(context.specs) * len(context.scopes) * len(EAP_GRIDS),
        "execution_order": [
            "72 calibration fits (spec -> grid -> full/repeat0 outer scopes)",
            "6 common-support held-out fit-grid comparisons",
            "6 fixed-grid61-bank EAP 401-vs-801 comparisons",
            "one all-specification blocking decision",
        ],
        "runtime_estimate": (
            "Hardware-dependent and intentionally not extrapolated from CAT outcomes; "
            "the 72 EM fits dominate. Measured elapsed seconds are recorded per checkpoint."
        ),
        "blockers": [
            "any provenance/hash mismatch",
            "any non-converged fit",
            "any exact-spec numerical gate failure",
            "partial or unverifiable resume cache",
        ],
        "selection_performed": False,
    }


def _assert_safe_output(context: Context) -> None:
    calibration_root = (ROOT / "runs" / "calibration").resolve()
    output = context.out_dir.resolve()
    if output == calibration_root or calibration_root not in output.parents:
        raise NumericalCheckError("numerical output must be a leaf below runs/calibration")
    if output.name != "InFoBench_v2_numerical_checks":
        raise NumericalCheckError("refusing a non-frozen numerical output leaf")
    protected = {
        (ROOT / "runs" / "calibration" / "InFoBench_remediation_v1").resolve(),
        (ROOT / "runs" / "calibration" / "InFoBench_playbook_full").resolve(),
    }
    if output in protected:
        raise NumericalCheckError("v1 output paths are immutable")


def _prepare_output(context: Context, *, resume: bool, fresh: bool) -> None:
    _assert_safe_output(context)
    if resume and fresh:
        raise NumericalCheckError("--resume and --fresh are mutually exclusive")
    if fresh and context.out_dir.exists():
        shutil.rmtree(context.out_dir)
    manifest_path = context.out_dir / "study_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise NumericalCheckError("--resume requires an existing study manifest")
        manifest = _read_json(manifest_path)
        if manifest.get("study_signature_sha256") != context.study_signature["canonical_sha256"]:
            raise NumericalCheckError("resume study signature differs from current inputs/code/env")
        if manifest.get("status") not in {"running", "blocked_numerical_equivalence"}:
            raise NumericalCheckError(f"cannot resume terminal status {manifest.get('status')!r}")
        return
    if context.out_dir.exists():
        raise NumericalCheckError(
            f"output already exists; use --resume or the scoped --fresh option: {context.out_dir}"
        )
    context.out_dir.mkdir(parents=True)
    _atomic_json(
        manifest_path,
        {
            "schema_version": RUNNER_SCHEMA,
            "status": "running",
            "started_at": _utcnow(),
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "study_signature": context.study_signature,
            "runtime_schedule": runtime_schedule(context),
            "calibration_model_selection_performed": False,
            "cat_results_inspected": False,
        },
    )


def _fit_dir(context: Context, spec: ExactSpec, grid: int, scope: Scope) -> Path:
    return context.out_dir / "fits" / spec.namespace / f"grid_{grid:03d}" / scope.scope_id


def _checkpoint_path(context: Context, stage_id: str) -> Path:
    safe = stage_id.replace("/", "__")
    return context.out_dir / "checkpoints" / f"{safe}.json"


def _verified_checkpoint(context: Context, stage_id: str, expected_inputs_sha256: str) -> bool:
    marker = _checkpoint_path(context, stage_id)
    if not marker.is_file():
        return False
    payload = _read_json(marker)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise NumericalCheckError(f"checkpoint schema mismatch: {marker}")
    expected = {
        "stage_id": stage_id,
        "study_signature_sha256": context.study_signature["canonical_sha256"],
        "stage_inputs_sha256": expected_inputs_sha256,
        "status": "completed",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise NumericalCheckError(f"checkpoint provenance mismatch: {marker}")
    outputs = payload.get("output_sha256") or {}
    if not outputs:
        raise NumericalCheckError(f"checkpoint has no output hashes: {marker}")
    for recorded, expected_hash in outputs.items():
        _require_file_hash(_resolve(recorded), str(expected_hash), f"checkpoint output {stage_id}")
    return True


def _write_checkpoint(
    context: Context,
    stage_id: str,
    stage_inputs: Mapping[str, Any],
    outputs: Sequence[Path],
    elapsed_seconds: float,
) -> None:
    if not outputs or any(not path.is_file() for path in outputs):
        raise NumericalCheckError(f"cannot checkpoint incomplete stage {stage_id}")
    _atomic_json(
        _checkpoint_path(context, stage_id),
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "status": "completed",
            "stage_id": stage_id,
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "stage_inputs": stage_inputs,
            "stage_inputs_sha256": _canonical_sha256(stage_inputs),
            "output_sha256": {_display(path): _sha256(path) for path in outputs},
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def _atomic_fit(path: Path, fit: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            A=np.asarray(fit["A"], dtype=float),
            b=np.asarray(fit["b"], dtype=float),
            R=np.asarray(fit["R"], dtype=float),
            items=np.asarray(list(map(str, fit["items"])), dtype=str),
            dim_labels=np.asarray(list(map(str, fit["dim_labels"])), dtype=str),
            loglik=np.asarray(float(fit["loglik"])),
            n_params=np.asarray(int(fit["n_params"])),
            n_iter=np.asarray(int(fit["n_iter"])),
            converged=np.asarray(bool(fit["converged"])),
        )
    temporary.replace(path)


def _load_fit(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as data:
            fit = {
                "A": np.asarray(data["A"], dtype=float),
                "b": np.asarray(data["b"], dtype=float),
                "R": np.asarray(data["R"], dtype=float),
                "items": list(map(str, data["items"].tolist())),
                "dim_labels": list(map(str, data["dim_labels"].tolist())),
                "loglik": float(data["loglik"]),
                "n_params": int(data["n_params"]),
                "n_iter": int(data["n_iter"]),
                "converged": bool(data["converged"]),
            }
    except (OSError, KeyError, ValueError) as error:
        raise NumericalCheckError(f"could not load fit cache {path}: {error}") from error
    if fit["A"].shape != (len(fit["items"]), 1) or fit["b"].shape != (len(fit["items"]),):
        raise NumericalCheckError(f"invalid one-dimensional fit arrays: {path}")
    if fit["dim_labels"] != ["instruction_following"]:
        raise NumericalCheckError(f"fit has the wrong latent axis: {path}")
    return fit


def _parameter_frame(fit: Mapping[str, Any], spec: ExactSpec) -> pd.DataFrame:
    a = np.asarray(fit["A"], dtype=float)[:, 0]
    b = np.asarray(fit["b"], dtype=float)
    exportable = np.isfinite(a) & np.isfinite(b) & (a > 0) & (a <= EXTREME_A)
    return pd.DataFrame(
        {
            "criterion_id": list(map(str, fit["items"])),
            "a_instruction_following": a,
            "b": b,
            "exportable": exportable,
            "fixed_a_expected": spec.family == cm.ONE_PL,
            "fixed_a_exact": np.isclose(a, 1.0, rtol=0.0, atol=0.0),
        }
    )


def _runtime_objects(context: Context):
    matrix = cm.load_matrix_strict(context.matrix_path)
    cm.configure_skills(",".join(EXPECTED_SOURCE_SKILLS))
    q_by = cm.load_q_matrix(context.rubrics_path)
    cm.validate_matrix_bank_alignment(matrix, q_by, True)
    structure = cell_cv.build_structure(
        EXPECTED_SOURCE_SKILLS,
        EXPECTED_DIMENSIONS,
        "overall_1d",
        historical_default=False,
    )
    item_to_scenario = cell_cv.load_item_scenarios(context.rubrics_path)
    return matrix, q_by, structure, item_to_scenario


def _run_fit_stage(
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
    stage_id = f"fit/{spec.namespace}/grid_{grid:03d}/{scope.scope_id}"
    stage_inputs = {
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
    inputs_hash = _canonical_sha256(stage_inputs)
    if resume and _verified_checkpoint(context, stage_id, inputs_hash):
        return
    destination = _fit_dir(context, spec, grid, scope)
    outputs = (
        destination / "fit.npz",
        destination / "item_params.csv",
        destination / "fit_manifest.json",
    )
    if any(path.exists() for path in outputs) or _checkpoint_path(context, stage_id).exists():
        raise NumericalCheckError(f"partial/uncheckpointed fit stage exists: {stage_id}")
    started = time.monotonic()
    args = argparse.Namespace(
        grid=grid,
        ridge=0.0 if spec.ridge is None else spec.ridge,
        max_iter=int(context.config["runtime"]["fit_max_iter"]),
        tol=float(context.config["runtime"]["fit_tolerance"]),
        estimate_latent_corr=False,
        calibration_model=spec.family,
        log_a_shrinkage=(
            cm.DEFAULT_LOG_A_SHRINKAGE if spec.log_a_shrinkage is None else spec.log_a_shrinkage
        ),
    )
    fit = cell_cv.fit_structure(matrix.loc[list(scope.train_models)], q_by, args, structure)
    if fit.get("calibration_specification", {}).get("cache_key") != spec.canonical["cache_key"]:
        raise NumericalCheckError(f"fitter returned wrong exact specification for {spec.spec_id}")
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_fit(outputs[0], fit)
    parameters = _parameter_frame(fit, spec)
    _atomic_csv(outputs[1], parameters)
    _atomic_json(
        outputs[2],
        {
            "schema_version": RUNNER_SCHEMA,
            "stage_inputs": stage_inputs,
            "stage_inputs_sha256": inputs_hash,
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
    _write_checkpoint(context, stage_id, stage_inputs, outputs, time.monotonic() - started)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    value = frame["left"].corr(frame["right"], method="spearman")
    return float(value) if value == value else None


def parameter_scope_gate(
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    *,
    family: str,
    source_ids: Sequence[str],
    minimum_spearman: float,
    minimum_exportability_agreement: float,
    lower_converged: bool,
    upper_converged: bool,
) -> dict[str, Any]:
    """Pure parameter gate; constant-a 1PL uses exactness, not undefined rank r."""

    left = lower.set_index("criterion_id")
    right = upper.set_index("criterion_id")
    if not left.index.is_unique or not right.index.is_unique:
        raise NumericalCheckError("parameter tables contain duplicate criterion IDs")
    common = sorted(set(left.index) & set(right.index))
    if not common:
        raise NumericalCheckError("fit grids have no common fitted items")
    source = list(map(str, source_ids))
    left_export = left["exportable"].astype(bool).reindex(source, fill_value=False)
    right_export = right["exportable"].astype(bool).reindex(source, fill_value=False)
    agreement = float((left_export == right_export).mean())
    b_spearman = _spearman(left.loc[common, "b"], right.loc[common, "b"])
    if family == cm.ONE_PL:
        left_a = left.loc[common, "a_instruction_following"].to_numpy(float)
        right_a = right.loc[common, "a_instruction_following"].to_numpy(float)
        fixed_exact = bool(
            np.allclose(left_a, 1.0, rtol=0.0, atol=0.0)
            and np.allclose(right_a, 1.0, rtol=0.0, atol=0.0)
        )
        a_spearman = None
        a_passed = fixed_exact
        a_method = "fixed-a-exact-equality (Spearman undefined by design)"
    else:
        a_spearman = _spearman(
            left.loc[common, "a_instruction_following"],
            right.loc[common, "a_instruction_following"],
        )
        fixed_exact = None
        a_passed = a_spearman is not None and a_spearman >= minimum_spearman
        a_method = "common-item Spearman"
    b_passed = b_spearman is not None and b_spearman >= minimum_spearman
    return {
        "n_lower_items": len(left),
        "n_upper_items": len(right),
        "n_common_items": len(common),
        "a_stability_method": a_method,
        "a_spearman": a_spearman,
        "fixed_a_exact": fixed_exact,
        "a_stability_passed": bool(a_passed),
        "b_spearman": b_spearman,
        "b_stability_passed": bool(b_passed),
        "minimum_parameter_spearman": minimum_spearman,
        "exportability_agreement": agreement,
        "minimum_exportability_agreement": minimum_exportability_agreement,
        "exportability_passed": agreement >= minimum_exportability_agreement,
        "lower_exportable": int(left_export.sum()),
        "upper_exportable": int(right_export.sum()),
        "lower_converged": bool(lower_converged),
        "upper_converged": bool(upper_converged),
        "passed": bool(
            lower_converged
            and upper_converged
            and a_passed
            and b_passed
            and agreement >= minimum_exportability_agreement
        ),
    }


def _quadrature(nodes: int):
    return scat.build_quadrature(
        1,
        nodes,
        np.eye(1),
        max_nodes=max(5000, nodes),
        method="normal_trapezoid",
        linear_bound=LINEAR_BOUND,
    )


def _posterior(
    raw: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    item_indices: np.ndarray,
    nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quadrature = _quadrature(nodes)
    selected = np.asarray(item_indices, dtype=int)
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
    tail = posterior[:, np.abs(quadrature.grid[:, 0]) >= TAIL_REGION].sum(axis=1)
    return theta, se, tail


def _fit_arrays_on_ids(fit: Mapping[str, Any], criterion_ids: Sequence[str]):
    position = {str(item): index for index, item in enumerate(fit["items"])}
    indices = np.asarray([position[str(item)] for item in criterion_ids], dtype=int)
    return np.asarray(fit["A"], dtype=float)[indices], np.asarray(fit["b"], dtype=float)[indices]


def _heldout_predictions(
    context: Context,
    spec: ExactSpec,
    matrix: pd.DataFrame,
    item_to_scenario: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    by_grid: dict[int, list[dict[str, Any]]] = {61: [], 81: []}
    support_rows: list[dict[str, Any]] = []
    for scope in context.scopes:
        if scope.outer_fold is None:
            continue
        fits: dict[int, dict[str, Any]] = {}
        params: dict[int, pd.DataFrame] = {}
        for grid in FIT_GRIDS:
            directory = _fit_dir(context, spec, grid, scope)
            fits[grid] = _load_fit(directory / "fit.npz")
            params[grid] = pd.read_csv(directory / "item_params.csv").set_index("criterion_id")
            if not params[grid].index.is_unique:
                raise NumericalCheckError(
                    f"{spec.spec_id}/{scope.scope_id}/grid{grid} has duplicate items"
                )
        common = sorted(
            set(params[61].index[params[61]["exportable"].astype(bool)])
            & set(params[81].index[params[81]["exportable"].astype(bool)])
        )
        scoring_ids = [
            item
            for item in common
            if item_to_scenario.get(item, item) in context.administration_scenarios
        ]
        evaluation_ids = [
            item
            for item in common
            if item_to_scenario.get(item, item) in context.evaluation_scenarios
        ]
        if not scoring_ids or not evaluation_ids:
            raise NumericalCheckError(f"{scope.scope_id} common support lacks scenario roles")
        raw = matrix.loc[list(scope.score_models)].reindex(columns=common).to_numpy(float)
        common_position = {item: index for index, item in enumerate(common)}
        scoring_index = np.asarray([common_position[item] for item in scoring_ids], dtype=int)
        evaluation_index = np.asarray([common_position[item] for item in evaluation_ids], dtype=int)
        observed = np.isfinite(raw[:, evaluation_index])
        support_rows.append(
            {
                "outer_fold": scope.outer_fold,
                "n_common_exportable_items": len(common),
                "n_common_administration_items": len(scoring_ids),
                "n_common_evaluation_items": len(evaluation_ids),
                "n_common_heldout_cells": int(observed.sum()),
            }
        )
        for grid in FIT_GRIDS:
            a, b = _fit_arrays_on_ids(fits[grid], common)
            theta, _se, _tail = _posterior(raw, a, b, scoring_index, EAP_GRIDS[0])
            probability = expit(
                theta[:, None] * a[evaluation_index, 0][None, :] - b[evaluation_index][None, :]
            )
            for model_index, item_index in np.argwhere(observed):
                model = scope.score_models[int(model_index)]
                criterion = evaluation_ids[int(item_index)]
                by_grid[grid].append(
                    {
                        "outer_fold": scope.outer_fold,
                        "model": model,
                        "family": context.model_to_family[model],
                        "criterion_id": criterion,
                        "label": int(raw[model_index, evaluation_index[item_index]]),
                        "probability": float(probability[model_index, item_index]),
                    }
                )
    return pd.DataFrame(by_grid[61]), pd.DataFrame(by_grid[81]), support_rows


def _cluster_interval(
    frame: pd.DataFrame, column: str, *, replicates: int, seed: int
) -> tuple[float, float]:
    grouped = frame.groupby("family", sort=True)[column].agg(["sum", "count"])
    if len(grouped) < 2:
        raise NumericalCheckError("family bootstrap requires at least two families")
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        values[index] = sums[selected].sum() / counts[selected].sum()
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def common_cell_equivalence(
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    *,
    log_loss_margin: float,
    brier_margin: float,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    keys = ["outer_fold", "model", "criterion_id"]
    required = set(keys + ["family", "label", "probability"])
    for name, frame in (("grid61", lower), ("grid81", upper)):
        if required - set(frame.columns) or frame.duplicated(keys).any():
            raise NumericalCheckError(f"{name} held-out cell table is malformed")
    left_keys = lower[keys].sort_values(keys).reset_index(drop=True)
    right_keys = upper[keys].sort_values(keys).reset_index(drop=True)
    keys_identical = left_keys.equals(right_keys)
    if not keys_identical:
        raise NumericalCheckError("fit grids did not score identical common held-out cells")
    merged = lower.merge(upper, on=keys, suffixes=("_61", "_81"), validate="one_to_one")
    labels_identical = bool((merged["label_61"] == merged["label_81"]).all())
    families_identical = bool((merged["family_61"] == merged["family_81"]).all())
    if not labels_identical or not families_identical or merged.empty:
        raise NumericalCheckError("common held-out labels/families differ or are empty")
    merged["label"] = merged["label_61"].astype(int)
    merged["family"] = merged["family_61"]
    y = merged["label"].to_numpy(float)
    for suffix in ("61", "81"):
        probability = np.clip(merged[f"probability_{suffix}"].to_numpy(float), EPSILON, 1 - EPSILON)
        merged[f"log_loss_{suffix}"] = -(
            y * np.log(probability) + (1 - y) * np.log(1 - probability)
        )
        merged[f"brier_{suffix}"] = (probability - y) ** 2
    details: dict[str, Any] = {}
    overall = True
    for offset, (metric, margin) in enumerate(
        (("log_loss", log_loss_margin), ("brier", brier_margin))
    ):
        column = f"{metric}_delta_81_minus_61"
        merged[column] = merged[f"{metric}_81"] - merged[f"{metric}_61"]
        pooled = float(merged[column].mean())
        low, high = _cluster_interval(merged, column, replicates=replicates, seed=seed + offset)
        passed = abs(pooled) <= margin and low >= -margin and high <= margin
        details[metric] = {
            "pooled_shift_grid81_minus_grid61": pooled,
            "absolute_pooled_shift": abs(pooled),
            "family_cluster_bootstrap_ci_95": [low, high],
            "equivalence_margin": margin,
            "pooled_shift_within_margin": abs(pooled) <= margin,
            "ci_wholly_within_equivalence_bounds": low >= -margin and high <= margin,
            "passed": bool(passed),
        }
        overall = overall and passed
    columns = keys + [
        "family",
        "label",
        "probability_61",
        "probability_81",
        "log_loss_delta_81_minus_61",
        "brier_delta_81_minus_61",
    ]
    return (
        {
            "n_common_cells": len(merged),
            "keys_identical": keys_identical,
            "labels_identical": labels_identical,
            "families_identical": families_identical,
            "cluster_unit": "tutor_model_family",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "metrics": details,
            "passed": bool(overall),
        },
        merged[columns].sort_values(keys),
    )


def _fit_comparison_stage(
    context: Context,
    spec: ExactSpec,
    matrix: pd.DataFrame,
    item_to_scenario: Mapping[str, str],
    *,
    resume: bool,
) -> dict[str, Any]:
    stage_id = f"fit_comparison/{spec.namespace}"
    stage_inputs = {
        "spec_id": spec.spec_id,
        "fit_grids": list(FIT_GRIDS),
        "scope_ids": [scope.scope_id for scope in context.scopes],
        "thresholds": context.config["numerical_lock"],
        "fit_hashes": {
            f"{grid}/{scope.scope_id}": _sha256(_fit_dir(context, spec, grid, scope) / "fit.npz")
            for grid in FIT_GRIDS
            for scope in context.scopes
        },
    }
    inputs_hash = _canonical_sha256(stage_inputs)
    output_dir = context.out_dir / "fit_comparisons" / spec.namespace
    outputs = (
        output_dir / "parameter_scope_gates.csv",
        output_dir / "common_support.csv",
        output_dir / "heldout_cells.csv",
        output_dir / "fit_grid_gate.json",
    )
    if resume and _verified_checkpoint(context, stage_id, inputs_hash):
        return _read_json(outputs[-1])
    if any(path.exists() for path in outputs) or _checkpoint_path(context, stage_id).exists():
        raise NumericalCheckError(f"partial fit-comparison stage exists: {stage_id}")
    started = time.monotonic()
    numerical = context.config["numerical_lock"]
    source_ids = list(map(str, matrix.columns))
    parameter_rows: list[dict[str, Any]] = []
    for scope in context.scopes:
        directories = {grid: _fit_dir(context, spec, grid, scope) for grid in FIT_GRIDS}
        manifests = {
            grid: _read_json(directories[grid] / "fit_manifest.json") for grid in FIT_GRIDS
        }
        gate = parameter_scope_gate(
            pd.read_csv(directories[61] / "item_params.csv"),
            pd.read_csv(directories[81] / "item_params.csv"),
            family=spec.family,
            source_ids=source_ids,
            minimum_spearman=float(numerical["fit_parameter_minimum_spearman"]),
            minimum_exportability_agreement=float(numerical["minimum_exportability_agreement"]),
            lower_converged=bool(manifests[61]["converged"]),
            upper_converged=bool(manifests[81]["converged"]),
        )
        parameter_rows.append({"scope": scope.scope_id, **gate})
    lower, upper, support_rows = _heldout_predictions(context, spec, matrix, item_to_scenario)
    bootstrap_seed = int(context.config["runtime"]["master_seed"]) + int(
        spec.canonical["cache_key"][-8:], 16
    )
    cell_gate, cell_details = common_cell_equivalence(
        lower,
        upper,
        log_loss_margin=float(numerical["maximum_absolute_heldout_log_loss_shift"]),
        brier_margin=float(numerical["maximum_absolute_heldout_brier_shift"]),
        replicates=int(context.config["runtime"]["metric_family_bootstrap_replicates"]),
        seed=bootstrap_seed,
    )
    parameters_passed = all(bool(row["passed"]) for row in parameter_rows)
    result = {
        "schema_version": RUNNER_SCHEMA,
        "spec_id": spec.spec_id,
        "canonical_cache_key": spec.canonical["cache_key"],
        "comparison": "fit_grid_61_vs_81",
        "all_six_scopes_parameter_stable": parameters_passed,
        "heldout_common_cell_gate": cell_gate,
        "calibration_model_selected": False,
        "cat_results_inspected": False,
        "passed": bool(parameters_passed and cell_gate["passed"]),
    }
    _atomic_csv(outputs[0], pd.DataFrame(parameter_rows))
    _atomic_csv(outputs[1], pd.DataFrame(support_rows))
    _atomic_csv(outputs[2], cell_details)
    _atomic_json(outputs[3], result)
    _write_checkpoint(context, stage_id, stage_inputs, outputs, time.monotonic() - started)
    return result


def eap_rows_gate(rows: pd.DataFrame, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "theta_401",
        "theta_801",
        "tail_mass_401",
        "tail_mass_801",
        "model",
        "scope",
        "support",
    }
    if required - set(rows.columns) or rows.empty:
        raise NumericalCheckError("EAP comparison rows are incomplete")
    shift = np.abs(rows["theta_801"].to_numpy(float) - rows["theta_401"].to_numpy(float))
    tail = np.concatenate(
        [rows["tail_mass_401"].to_numpy(float), rows["tail_mass_801"].to_numpy(float)]
    )
    finite = bool(np.isfinite(shift).all() and np.isfinite(tail).all())
    median = float(np.median(shift)) if finite else math.inf
    p95 = float(np.quantile(shift, 0.95)) if finite else math.inf
    maximum = float(np.max(shift)) if finite else math.inf
    maximum_tail = float(np.max(tail)) if finite else math.inf
    passed = bool(
        finite
        and median <= float(thresholds["maximum_median_absolute_theta_shift"])
        and p95 <= float(thresholds["maximum_p95_absolute_theta_shift"])
        and maximum <= float(thresholds["maximum_absolute_theta_shift"])
        and maximum_tail <= float(thresholds["maximum_posterior_tail_mass"])
    )
    return {
        "n_model_scope_support_rows": len(rows),
        "median_absolute_theta_shift": median,
        "p95_absolute_theta_shift": p95,
        "maximum_absolute_theta_shift": maximum,
        "maximum_posterior_tail_mass": maximum_tail,
        "tail_region_absolute_theta_at_least": TAIL_REGION,
        "thresholds": {
            key: thresholds[key]
            for key in (
                "maximum_median_absolute_theta_shift",
                "maximum_p95_absolute_theta_shift",
                "maximum_absolute_theta_shift",
                "maximum_posterior_tail_mass",
            )
        },
        "all_values_finite": finite,
        "passed": passed,
    }


def _eap_comparison_stage(
    context: Context,
    spec: ExactSpec,
    matrix: pd.DataFrame,
    item_to_scenario: Mapping[str, str],
    *,
    resume: bool,
) -> dict[str, Any]:
    stage_id = f"eap_comparison/{spec.namespace}"
    fit_hashes = {
        scope.scope_id: _sha256(_fit_dir(context, spec, 61, scope) / "fit.npz")
        for scope in context.scopes
    }
    stage_inputs = {
        "spec_id": spec.spec_id,
        "fixed_fit_grid": 61,
        "fit_hashes": fit_hashes,
        "method": "normal_trapezoid",
        "eap_grids": list(EAP_GRIDS),
        "linear_bound": LINEAR_BOUND,
        "tail_region": TAIL_REGION,
        "thresholds": {
            key: context.config["numerical_lock"][key]
            for key in (
                "maximum_median_absolute_theta_shift",
                "maximum_p95_absolute_theta_shift",
                "maximum_absolute_theta_shift",
                "maximum_posterior_tail_mass",
            )
        },
    }
    inputs_hash = _canonical_sha256(stage_inputs)
    output_dir = context.out_dir / "eap_comparisons" / spec.namespace
    outputs = (output_dir / "per_model_scope.csv", output_dir / "eap_grid_gate.json")
    if resume and _verified_checkpoint(context, stage_id, inputs_hash):
        return _read_json(outputs[-1])
    if any(path.exists() for path in outputs) or _checkpoint_path(context, stage_id).exists():
        raise NumericalCheckError(f"partial EAP-comparison stage exists: {stage_id}")
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for scope in context.scopes:
        directory = _fit_dir(context, spec, 61, scope)
        fit = _load_fit(directory / "fit.npz")
        params = pd.read_csv(directory / "item_params.csv").set_index("criterion_id")
        if not params.index.is_unique:
            raise NumericalCheckError(
                f"{spec.spec_id}/{scope.scope_id} has duplicate parameter rows"
            )
        exportable = list(params.index[params["exportable"].astype(bool)])
        if not exportable:
            raise NumericalCheckError(f"{spec.spec_id}/{scope.scope_id} has no exportable bank")
        a, b = _fit_arrays_on_ids(fit, exportable)
        raw = matrix.loc[list(scope.score_models)].reindex(columns=exportable).to_numpy(float)
        all_index = np.arange(len(exportable), dtype=int)
        admin_index = np.asarray(
            [
                index
                for index, item in enumerate(exportable)
                if item_to_scenario.get(item, item) in context.administration_scenarios
            ],
            dtype=int,
        )
        if not len(admin_index):
            raise NumericalCheckError(f"{spec.spec_id}/{scope.scope_id} has no admin items")
        for support, indices in (("full_bank", all_index), ("administration_only", admin_index)):
            results = {nodes: _posterior(raw, a, b, indices, nodes) for nodes in EAP_GRIDS}
            for model_index, model in enumerate(scope.score_models):
                rows.append(
                    {
                        "scope": scope.scope_id,
                        "outer_fold": scope.outer_fold,
                        "support": support,
                        "model": model,
                        "family": context.model_to_family[model],
                        "n_bank_items": len(exportable),
                        "n_scoring_items": len(indices),
                        "theta_401": float(results[401][0][model_index]),
                        "se_401": float(results[401][1][model_index]),
                        "tail_mass_401": float(results[401][2][model_index]),
                        "theta_801": float(results[801][0][model_index]),
                        "se_801": float(results[801][1][model_index]),
                        "tail_mass_801": float(results[801][2][model_index]),
                    }
                )
    frame = pd.DataFrame(rows)
    gate = eap_rows_gate(frame, context.config["numerical_lock"])
    result = {
        "schema_version": RUNNER_SCHEMA,
        "spec_id": spec.spec_id,
        "canonical_cache_key": spec.canonical["cache_key"],
        "comparison": "normal_trapezoid_eap_401_vs_801",
        "fit_grid_61_banks_held_fixed": True,
        "fit_refit_during_eap_comparison": False,
        "eap_gate": gate,
        "calibration_model_selected": False,
        "cat_results_inspected": False,
        "passed": bool(gate["passed"]),
    }
    _atomic_csv(outputs[0], frame)
    _atomic_json(outputs[1], result)
    _write_checkpoint(context, stage_id, stage_inputs, outputs, time.monotonic() - started)
    return result


def build_numerical_decision(
    spec_ids: Sequence[str],
    fit_gates: Mapping[str, Mapping[str, Any]],
    eap_gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Blocking all-spec decision; deliberately has no selection/fallback branch."""

    ids = list(map(str, spec_ids))
    complete = set(fit_gates) == set(ids) and set(eap_gates) == set(ids)
    per_spec: list[dict[str, Any]] = []
    for spec_id in ids:
        fit = fit_gates.get(spec_id) or {}
        eap = eap_gates.get(spec_id) or {}
        passed = fit.get("passed") is True and eap.get("passed") is True
        per_spec.append(
            {
                "spec_id": spec_id,
                "fit_grid_gate_passed": fit.get("passed") is True,
                "eap_grid_gate_passed": eap.get("passed") is True,
                "passed": passed,
            }
        )
    all_passed = bool(complete and per_spec and all(row["passed"] for row in per_spec))
    return {
        "schema_version": DECISION_SCHEMA,
        "complete_exact_specification_panel": complete,
        "required_exact_specifications": ids,
        "per_specification": per_spec,
        "all_exact_specifications_passed": all_passed,
        "passed": all_passed,
        "calibration_specification_selected": None,
        "selection_performed": False,
        "fallback_allowed": False,
        "cat_results_inspected": False,
        "failure_action": None if all_passed else "block_phase3_and_investigate",
    }


def emit_numerical_lock(
    out_dir: Path,
    decision: Mapping[str, Any],
    lock_payload: Mapping[str, Any],
) -> Path | None:
    """Write the lock and its hash iff the complete no-selection gate passed."""

    lock_path = out_dir / "numerical_lock.json"
    sha_path = out_dir / "numerical_lock.sha256"
    if decision.get("passed") is not True:
        if lock_path.exists() or sha_path.exists():
            raise NumericalCheckError("a failed decision must not retain a numerical lock")
        return None
    payload = {
        "schema_version": LOCK_SCHEMA,
        # ``complete_pass`` is the frozen Phase-3 consumer's accepted terminal
        # status.  The more explicit schema and decision below prevent it from
        # being confused with a selected-specification artifact.
        "status": "complete_pass",
        **dict(lock_payload),
        "decision": dict(decision),
        "passed_spec_ids": list(decision["required_exact_specifications"]),
        "specifications": list(decision["per_specification"]),
        "calibration_specification_selected": None,
        "selection_performed": False,
        "cat_results_inspected": False,
    }
    _atomic_json(lock_path, payload)
    digest = _sha256(lock_path)
    sha_path.write_text(f"{digest}  numerical_lock.json\n", encoding="utf-8")
    return lock_path


def _finalize(
    context: Context,
    fit_gates: Mapping[str, Mapping[str, Any]],
    eap_gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    decision = build_numerical_decision(
        [spec.spec_id for spec in context.specs], fit_gates, eap_gates
    )
    decision.update(
        {
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "config_sha256": _sha256(context.config_path),
            "split_manifest_sha256": _sha256(context.split_path),
            "fit_gate_sha256": {
                spec.spec_id: _sha256(
                    context.out_dir / "fit_comparisons" / spec.namespace / "fit_grid_gate.json"
                )
                for spec in context.specs
            },
            "eap_gate_sha256": {
                spec.spec_id: _sha256(
                    context.out_dir / "eap_comparisons" / spec.namespace / "eap_grid_gate.json"
                )
                for spec in context.specs
            },
        }
    )
    decision_path = context.out_dir / "numerical_decision.json"
    _atomic_json(decision_path, decision)
    lock_path = emit_numerical_lock(
        context.out_dir,
        decision,
        {
            "locked_fit_grid": 61,
            "verified_fit_grid_comparator": 81,
            "locked_eap_method": "normal_trapezoid",
            "locked_eap_grid": 401,
            "verified_eap_grid_comparator": 801,
            "linear_bound": LINEAR_BOUND,
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "config_sha256": _sha256(context.config_path),
            "split_manifest_sha256": _sha256(context.split_path),
            "decision_sha256": _sha256(decision_path),
            "exact_specification_cache_keys": {
                spec.spec_id: spec.canonical["cache_key"] for spec in context.specs
            },
            "phase3_requirement": (
                "Phase 3 must hash-verify this exact numerical_lock.json and record "
                "that SHA-256 in its manifest before fitting or scoring."
            ),
        },
    )
    manifest_path = context.out_dir / "study_manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update(
        {
            "status": (
                "passed_numerical_equivalence" if lock_path else "blocked_numerical_equivalence"
            ),
            "finished_at": _utcnow(),
            "decision_sha256": _sha256(decision_path),
            "numerical_lock": _display(lock_path) if lock_path else None,
            "numerical_lock_sha256": _sha256(lock_path) if lock_path else None,
            "calibration_model_selection_performed": False,
            "cat_results_inspected": False,
        }
    )
    _atomic_json(manifest_path, manifest)
    return decision


def run(context: Context, *, resume: bool = False, fresh: bool = False) -> dict[str, Any]:
    _prepare_output(context, resume=resume, fresh=fresh)
    matrix, q_by, structure, item_to_scenario = _runtime_objects(context)
    for spec in context.specs:
        for grid in FIT_GRIDS:
            for scope in context.scopes:
                _run_fit_stage(
                    context,
                    spec,
                    grid,
                    scope,
                    matrix,
                    q_by,
                    structure,
                    resume=resume,
                )
    fit_gates = {
        spec.spec_id: _fit_comparison_stage(context, spec, matrix, item_to_scenario, resume=resume)
        for spec in context.specs
    }
    eap_gates = {
        spec.spec_id: _eap_comparison_stage(context, spec, matrix, item_to_scenario, resume=resume)
        for spec in context.specs
    }
    return _finalize(context, fit_gates, eap_gates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate all provenance/design inputs and print the schedule without writing",
    )
    parser.add_argument("--resume", action="store_true", help="resume only verified checkpoints")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete only the frozen v2 numerical-check output leaf before starting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = load_context(args.config, args.out_dir)
        if args.plan_only:
            if args.resume or args.fresh:
                raise NumericalCheckError("--plan-only cannot be combined with --resume/--fresh")
            print(json.dumps(runtime_schedule(context), indent=2, sort_keys=True))
            return 0
        decision = run(context, resume=args.resume, fresh=args.fresh)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["passed"] else 2
    except (NumericalCheckError, cm.CalibrationError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
