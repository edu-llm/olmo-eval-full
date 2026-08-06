#!/usr/bin/env python3
"""Append-only V4 numerical verification for the InFoBench calibration fitter.

V4 fits exactly three production-eligible one-dimensional calibration
specifications under bounded normal-trapezoid integration. For every full/outer
scope it runs a cold 401-node fit, a cold 801-node fit, and an 801-node
continuation fit initialized from the 401-node parameters.

This module has no adaptive-testing import or execution path. It cannot select a
calibration family, administer an adaptive test, or inspect adaptive-test
results. A passing run emits only a numerical-profile lock for a later,
separately versioned study.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
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

DEFAULT_CONFIG = ROOT / "configs" / "infobench_v2_numerical_followup_v4.json"
CONFIG_SCHEMA = "infobench-v2-numerical-followup-v4"
RUNNER_SCHEMA = "infobench-v2-numerical-followup-run-v4"
CHECKPOINT_SCHEMA = "infobench-v2-numerical-followup-checkpoint-v4"
LOCK_SCHEMA = "infobench-v2-numerical-followup-lock-v4"
OUTPUT_LEAF = "InFoBench_v2_numerical_followup_v4"
FROZEN_CONTRACT_SHA256 = "0044a89b24aa0415d16abe123d7d9928c2d30b75a4c947e5d84656a879e7d555"

SOURCE_SKILLS = ("content", "format", "number", "style", "linguistic")
DIMENSION_LABEL = "instruction_following"
FIT_NODES = (401, 801)
PRIMARY_SPEC_IDS = (
    "1pl_fixed_a1",
    "log_shrinkage_2pl_lambda16",
    "log_shrinkage_2pl_lambda4",
)
SCOPE_IDS = ("full", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4")
EPSILON = 1e-12


class V4Error(RuntimeError):
    """Frozen-design, provenance, append-only, or numerical invariant failed."""


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


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


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
    encoded = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        raise V4Error(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise V4Error(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_once(path: Path, value: Any) -> None:
    if path.exists():
        if not path.is_file() or _read_json(path) != _json_ready(value):
            raise V4Error(f"append-only JSON evidence differs: {path}")
        return
    _atomic_json(path, value)


def _write_text_once(path: Path, value: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != value:
            raise V4Error(f"append-only text evidence differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise V4Error(f"{label} is missing: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise V4Error(f"{label} hash changed: expected {expected}, observed {observed}")


def _tree_sha256(root: Path) -> tuple[int, str]:
    if not root.is_dir():
        raise V4Error(f"historical run directory is missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return len(files), digest.hexdigest()


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    return {**payload, "canonical_sha256": _canonical_sha256(payload)}


def _validate_config(raw: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "status": "preregistered_before_v4_results",
        "benchmark": "InFoBench",
        "output_dir": f"runs/calibration/{OUTPUT_LEAF}",
        "lock_path": (f"runs/calibration/{OUTPUT_LEAF}/numerical_followup_lock.json"),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise V4Error(f"V4 config field changed: {key}")
    contract = raw.get("frozen_contract")
    if not isinstance(contract, Mapping):
        raise V4Error("V4 frozen_contract is missing")
    if _canonical_sha256(contract) != FROZEN_CONTRACT_SHA256:
        raise V4Error("V4 scientific contract differs from the frozen runner contract")
    freeze = raw.get("code_freeze") or {}
    if freeze.get("status") != "finalized_before_official_run":
        raise V4Error("V4 code freeze is not finalized")
    dependencies = freeze.get("dependencies")
    if not isinstance(dependencies, Mapping) or not dependencies:
        raise V4Error("V4 dependency freeze is missing")
    for path, digest in dependencies.items():
        if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
            raise V4Error(f"V4 dependency digest is not finalized: {path}")


def _verify_historical_evidence(raw: Mapping[str, Any]) -> dict[str, Any]:
    history = raw.get("historical_evidence") or {}
    observed: dict[str, Any] = {}
    keys = ("v1_aborted", "v2_terminal_nonpromotable", "v3_terminal_blocked")
    for key in keys:
        record = history.get(key) or {}
        run_dir = _resolve(str(record.get("run_dir") or ""))
        count, tree_hash = _tree_sha256(run_dir)
        if count != int(record.get("output_file_count", -1)):
            raise V4Error(f"{key} historical output count changed")
        if tree_hash != record.get("output_tree_sha256"):
            raise V4Error(f"{key} historical output tree changed")
        observed[key] = {
            "output_file_count": count,
            "output_tree_sha256": tree_hash,
            "promotable": False,
        }
        if record.get("required_lock_absent") is not True:
            raise V4Error(f"{key} lock-absence contract changed")
        if (run_dir / "numerical_followup_lock.json").exists() or (
            run_dir / "numerical_followup_lock.sha256"
        ).exists():
            raise V4Error(f"{key} unexpectedly contains a numerical lock")

    v1 = history["v1_aborted"]
    marker_path = _resolve(v1["marker"])
    _require_hash(marker_path, v1["marker_sha256"], "V1 abort marker")
    if _read_json(marker_path).get("status") != v1["required_status"]:
        raise V4Error("V1 abort status changed")

    for key in ("v2_terminal_nonpromotable", "v3_terminal_blocked"):
        record = history[key]
        manifest_path = _resolve(record["manifest"])
        decision_path = _resolve(record["decision"])
        _require_hash(manifest_path, record["manifest_sha256"], f"{key} manifest")
        _require_hash(decision_path, record["decision_sha256"], f"{key} decision")
        manifest = _read_json(manifest_path)
        decision = _read_json(decision_path)
        if manifest.get("status") != record["required_status"]:
            raise V4Error(f"{key} terminal manifest status changed")
        if decision.get("status") != record["required_status"]:
            raise V4Error(f"{key} terminal decision status changed")
        if decision.get("passed") is not record["required_passed"]:
            raise V4Error(f"{key} terminal decision changed")
    v3 = history["v3_terminal_blocked"]
    _require_hash(_resolve(v3["config"]), v3["config_sha256"], "V3 config")
    _require_hash(_resolve(v3["runner"]), v3["runner_sha256"], "V3 runner")

    reuse = history.get("reuse_policy") or {}
    if reuse != {
        "v1_fit_or_checkpoint_artifacts_reused": 0,
        "v2_fit_or_checkpoint_artifacts_reused": 0,
        "v3_fit_or_checkpoint_artifacts_reused": 0,
    }:
        raise V4Error("V4 historical zero-reuse policy changed")
    return observed


def _verify_code_freeze(raw: Mapping[str, Any]) -> dict[str, str]:
    dependencies = raw["code_freeze"]["dependencies"]
    observed: dict[str, str] = {}
    for relative, expected in dependencies.items():
        path = _resolve(relative)
        _require_hash(path, expected, f"frozen dependency {relative}")
        observed[relative] = expected
    return observed


def _parse_specs(contract: Mapping[str, Any]) -> tuple[ExactSpec, ...]:
    rows = contract.get("eligible_specifications") or []
    if [row.get("spec_id") for row in rows] != list(PRIMARY_SPEC_IDS):
        raise V4Error("V4 eligible specification panel changed")
    specs: list[ExactSpec] = []
    for row in rows:
        family = str(row["family"])
        ridge = None if row.get("ridge") is None else float(row["ridge"])
        shrinkage = None if row.get("log_a_shrinkage") is None else float(row["log_a_shrinkage"])
        canonical = cm.calibration_specification(
            family,
            ridge=0.0 if ridge is None else ridge,
            log_a_shrinkage=(cm.DEFAULT_LOG_A_SHRINKAGE if shrinkage is None else shrinkage),
        )
        if canonical["cache_key"] != row.get("canonical_cache_key"):
            raise V4Error(f"canonical specification changed: {row['spec_id']}")
        specs.append(
            ExactSpec(
                spec_id=str(row["spec_id"]),
                family=family,
                ridge=ridge,
                log_a_shrinkage=shrinkage,
                canonical=canonical,
            )
        )
    return tuple(specs)


def _parse_scopes(
    splits: Mapping[str, Any], models: Sequence[str]
) -> tuple[
    tuple[Scope, ...],
    dict[str, str],
    frozenset[str],
    frozenset[str],
]:
    model_ids = tuple(map(str, models))
    model_set = set(model_ids)
    model_to_family = {
        str(model): str(family) for model, family in (splits.get("model_to_family") or {}).items()
    }
    if set(model_to_family) != model_set:
        raise V4Error("split manifest model roster differs from the response matrix")
    if len(set(model_to_family.values())) != 22:
        raise V4Error("split manifest must retain exactly 22 tutor-model families")
    repetitions = splits.get("repetitions") or []
    if len(repetitions) < 1 or int(repetitions[0].get("repeat", -1)) != 0:
        raise V4Error("repeat-0 split is missing")
    outer = repetitions[0].get("outer_folds") or []
    if len(outer) != 5:
        raise V4Error("repeat-0 must contain exactly five outer folds")
    scopes: list[Scope] = [Scope("full", model_ids, model_ids, None)]
    seen_test: set[str] = set()
    for expected, fold in enumerate(sorted(outer, key=lambda row: int(row["outer_fold"]))):
        if int(fold["outer_fold"]) != expected:
            raise V4Error("repeat-0 outer fold indices changed")
        train = tuple(map(str, fold["train_model_ids"]))
        test = tuple(map(str, fold["test_model_ids"]))
        if set(train) & set(test) or set(train) | set(test) != model_set:
            raise V4Error(f"outer_{expected} train/test partition is invalid")
        if seen_test & set(test):
            raise V4Error("repeat-0 outer test folds overlap")
        seen_test.update(test)
        scopes.append(Scope(f"outer_{expected}", train, test, expected))
    if seen_test != model_set:
        raise V4Error("repeat-0 outer test folds do not cover the cohort")
    if tuple(scope.scope_id for scope in scopes) != SCOPE_IDS:
        raise V4Error("V4 scope roster changed")
    scenario = splits.get("scenario_split") or {}
    administration = frozenset(map(str, scenario.get("administration_scenario_ids") or []))
    evaluation = frozenset(map(str, scenario.get("evaluation_scenario_ids") or []))
    if not administration or not evaluation or administration & evaluation:
        raise V4Error("administration/evaluation scenario split is invalid")
    return tuple(scopes), model_to_family, administration, evaluation


def load_context(config_path: Path = DEFAULT_CONFIG, out_dir: Path | None = None) -> Context:
    config_path = config_path.resolve()
    raw = _read_json(config_path)
    _validate_config(raw)
    code_hashes = _verify_code_freeze(raw)
    history = _verify_historical_evidence(raw)
    design = raw["design_source"]
    _require_hash(_resolve(design["path"]), design["sha256"], "V4 design source")
    frozen = raw["frozen_inputs"]
    paths = {
        "response_matrix": _resolve(frozen["response_matrix"]),
        "rubrics": _resolve(frozen["rubrics"]),
        "scenarios": _resolve(frozen["scenarios"]),
        "split_manifest": _resolve(frozen["split_manifest"]),
    }
    for key, path in paths.items():
        _require_hash(path, frozen[f"{key}_sha256"], f"frozen {key}")
    matrix = cm.load_matrix_strict(paths["response_matrix"])
    if len(matrix) != 52:
        raise V4Error("V4 response matrix must contain exactly 52 tutor models")
    splits = _read_json(paths["split_manifest"])
    scopes, families, administration, evaluation = _parse_scopes(
        splits, list(map(str, matrix.index))
    )
    specs = _parse_specs(raw["frozen_contract"])
    configured_out = _resolve(raw["output_dir"])
    chosen_out = configured_out if out_dir is None else out_dir.resolve()
    if chosen_out != configured_out:
        raise V4Error("--out-dir must equal the preregistered V4 output leaf")
    if _resolve(raw["lock_path"]) != (chosen_out / "numerical_followup_lock.json"):
        raise V4Error("V4 lock path escapes the output leaf")
    environment = _environment()
    signature_payload = {
        "schema_version": RUNNER_SCHEMA,
        "config_sha256": _sha256(config_path),
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "input_sha256": {_display(path): _sha256(path) for path in paths.values()},
        "code_sha256": code_hashes,
        "historical_evidence_sha256": history,
        "environment_sha256": environment["canonical_sha256"],
        "spec_ids": list(PRIMARY_SPEC_IDS),
        "scope_ids": list(SCOPE_IDS),
        "fit_nodes": list(FIT_NODES),
        "required_total_new_fits": 54,
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "adaptive_runs": 0,
    }
    signature = {
        **signature_payload,
        "environment": environment,
        "canonical_sha256": _canonical_sha256(signature_payload),
    }
    return Context(
        config_path=config_path,
        config=raw,
        matrix_path=paths["response_matrix"],
        rubrics_path=paths["rubrics"],
        scenarios_path=paths["scenarios"],
        split_path=paths["split_manifest"],
        splits=splits,
        specs=specs,
        scopes=scopes,
        model_to_family=families,
        administration_scenarios=administration,
        evaluation_scenarios=evaluation,
        out_dir=chosen_out,
        study_signature=signature,
    )


def runtime_schedule(context: Context) -> dict[str, Any]:
    return {
        "schema_version": RUNNER_SCHEMA,
        "status": "plan_only_no_response_dependent_work_performed",
        "output_dir": _display(context.out_dir),
        "eligible_spec_ids": [spec.spec_id for spec in context.specs],
        "scope_ids": [scope.scope_id for scope in context.scopes],
        "fit_schedule": {
            "grid401_cold": 18,
            "grid801_cold": 18,
            "grid801_continuation_from_grid401": 18,
            "minimum_new_fits": 54,
            "maximum_new_fits": 54,
        },
        "fit_comparison": "normal_trapezoid_bound8_401_vs_801",
        "fixed_bank_scoring_profiles": context.config["frozen_contract"]["fixed_bank_scoring_gate"][
            "profiles"
        ],
        "historical_v1_fit_or_checkpoint_artifacts_reused": 0,
        "historical_v2_fit_or_checkpoint_artifacts_reused": 0,
        "historical_v3_fit_or_checkpoint_artifacts_reused": 0,
        "adaptive_runs": 0,
        "calibration_model_selection_runs": 0,
        "append_only": True,
        "resume_requires_identical_signature": True,
        "terminal_evidence_is_immutable": True,
    }


def _checkpoint_path(context: Context, stage_id: str) -> Path:
    return context.out_dir / "checkpoints" / f"{stage_id.replace('/', '__')}.json"


def _assert_safe_output(context: Context) -> None:
    calibration_root = (ROOT / "runs" / "calibration").resolve()
    output = context.out_dir.resolve()
    if output == calibration_root or calibration_root not in output.parents:
        raise V4Error("V4 output must be a leaf below runs/calibration")
    if output.name != OUTPUT_LEAF:
        raise V4Error("refusing a non-preregistered V4 output leaf")
    historical = {
        _resolve(record["run_dir"])
        for key, record in context.config["historical_evidence"].items()
        if key.startswith("v") and isinstance(record, Mapping) and "run_dir" in record
    }
    if output in historical:
        raise V4Error("V4 output may not overwrite a historical run")


def _validate_existing_checkpoints(context: Context) -> None:
    directory = context.out_dir / "checkpoints"
    if not directory.exists():
        return
    allowed_prefixes = (
        "fit/",
        "start_selection/",
        "fit_comparison/",
        "fixed_bank/",
    )
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            raise V4Error(f"unexpected checkpoint entry: {path}")
        marker = _read_json(path)
        stage_id = str(marker.get("stage_id") or "")
        if not stage_id.startswith(allowed_prefixes):
            raise V4Error(f"unexpected V4 checkpoint stage: {stage_id}")
        if path.name != f"{stage_id.replace('/', '__')}.json":
            raise V4Error(f"checkpoint filename does not match stage: {stage_id}")
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "status": "completed",
            "study_signature_sha256": context.study_signature["canonical_sha256"],
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            raise V4Error(f"checkpoint provenance mismatch: {stage_id}")
        inputs = marker.get("stage_inputs")
        if not isinstance(inputs, Mapping) or marker.get(
            "stage_inputs_sha256"
        ) != _canonical_sha256(inputs):
            raise V4Error(f"checkpoint input digest mismatch: {stage_id}")
        outputs = marker.get("output_sha256")
        if not isinstance(outputs, Mapping) or not outputs:
            raise V4Error(f"checkpoint has no outputs: {stage_id}")
        for recorded, digest in outputs.items():
            output = _resolve(recorded)
            if context.out_dir.resolve() not in output.resolve().parents:
                raise V4Error(f"checkpoint output escapes V4 leaf: {stage_id}")
            _require_hash(output, str(digest), f"checkpoint output {stage_id}")


def _prepare(context: Context, *, resume: bool) -> None:
    _assert_safe_output(context)
    _verify_code_freeze(context.config)
    manifest_path = context.out_dir / "study_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise V4Error("--resume requires an existing V4 manifest")
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != RUNNER_SCHEMA:
            raise V4Error("resume manifest schema differs")
        if manifest.get("study_signature_sha256") != context.study_signature["canonical_sha256"]:
            raise V4Error("resume signature differs from frozen inputs/code/environment")
        if manifest.get("status") != "running":
            raise V4Error(f"terminal V4 evidence is immutable: {manifest.get('status')!r}")
        _validate_existing_checkpoints(context)
        return
    if context.out_dir.exists():
        raise V4Error("V4 output already exists; only an exact --resume may continue it")
    context.out_dir.mkdir(parents=True)
    _atomic_json(
        manifest_path,
        {
            "schema_version": RUNNER_SCHEMA,
            "status": "running",
            "started_at": _utcnow(),
            "config": _display(context.config_path),
            "config_sha256": _sha256(context.config_path),
            "study_signature": context.study_signature,
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "required_total_new_fits": 54,
            "historical_fit_or_checkpoint_artifacts_reused": 0,
            "calibration_model_selection_performed": False,
            "adaptive_results_inspected": False,
            "adaptive_runs": 0,
        },
    )


def _checkpoint_reusable(context: Context, stage_id: str, inputs: Mapping[str, Any]) -> bool:
    path = _checkpoint_path(context, stage_id)
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
        raise V4Error(f"checkpoint provenance mismatch: {stage_id}")
    outputs = marker.get("output_sha256") or {}
    if not outputs:
        raise V4Error(f"checkpoint has no outputs: {stage_id}")
    for recorded, digest in outputs.items():
        _require_hash(_resolve(recorded), str(digest), f"checkpoint output {stage_id}")
    return True


def _write_checkpoint(
    context: Context,
    stage_id: str,
    inputs: Mapping[str, Any],
    outputs: Sequence[Path],
    elapsed_seconds: float,
) -> None:
    if not outputs or any(not path.is_file() for path in outputs):
        raise V4Error(f"cannot checkpoint incomplete stage: {stage_id}")
    _atomic_json(
        _checkpoint_path(context, stage_id),
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "status": "completed",
            "stage_id": stage_id,
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "stage_inputs": dict(inputs),
            "stage_inputs_sha256": _canonical_sha256(inputs),
            "output_sha256": {_display(path): _sha256(path) for path in outputs},
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def _fit_dir(context: Context, spec: ExactSpec, nodes: int, scope: Scope, start: str) -> Path:
    return context.out_dir / "fits" / spec.namespace / f"nodes_{nodes:04d}" / scope.scope_id / start


def _selection_path(context: Context, spec: ExactSpec, scope: Scope) -> Path:
    return context.out_dir / "start_selection" / spec.namespace / f"{scope.scope_id}.json"


def _fit_comparison_dir(context: Context, spec: ExactSpec) -> Path:
    """Canonical producer/consumer path for the 401-versus-801 refit gate."""

    return context.out_dir / "fit_comparisons" / "401_vs_801" / spec.namespace


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
            dim_labels=np.asarray([DIMENSION_LABEL], dtype=str),
            loglik=np.asarray(float(fit["loglik"])),
            penalized_objective=np.asarray(float(fit["penalized_objective"])),
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
                "penalized_objective": float(data["penalized_objective"]),
                "n_params": int(data["n_params"]),
                "n_iter": int(data["n_iter"]),
                "converged": bool(data["converged"]),
            }
    except (OSError, KeyError, ValueError) as error:
        raise V4Error(f"could not load fit cache {path}: {error}") from error
    if fit["A"].shape != (len(fit["items"]), 1):
        raise V4Error(f"invalid one-dimensional fit loading array: {path}")
    if fit["b"].shape != (len(fit["items"]),):
        raise V4Error(f"invalid fit difficulty array: {path}")
    if fit["dim_labels"] != [DIMENSION_LABEL]:
        raise V4Error(f"fit has the wrong latent axis: {path}")
    return fit


def _parameter_frame(fit: Mapping[str, Any], *, family: str) -> pd.DataFrame:
    a = np.asarray(fit["A"], dtype=float)[:, 0]
    b = np.asarray(fit["b"], dtype=float)
    exportable = np.isfinite(a) & np.isfinite(b) & (a > 0) & (a <= cm.EXTREME_A)
    return pd.DataFrame(
        {
            "criterion_id": list(map(str, fit["items"])),
            "a_instruction_following": a,
            "b": b,
            "exportable": exportable,
            "fixed_a_expected": family == cm.ONE_PL,
            "fixed_a_exact": np.isclose(a, 1.0, rtol=0.0, atol=0.0),
        }
    )


def _load_item_to_scenario(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise V4Error(f"invalid rubric JSON at line {line_number}: {error}") from error
            criterion = str(row.get("criterion_id") or "")
            scenario = str(row.get("scenario_id") or "")
            if not criterion or not scenario or criterion in mapping:
                raise V4Error(f"invalid/duplicate rubric mapping at line {line_number}")
            mapping[criterion] = scenario
    return mapping


def _runtime_objects(context: Context):
    matrix = cm.load_matrix_strict(context.matrix_path)
    cm.configure_skills(",".join(SOURCE_SKILLS))
    q_by = cm.load_q_matrix(context.rubrics_path)
    cm.validate_matrix_bank_alignment(matrix, q_by, True)
    return matrix, q_by, _load_item_to_scenario(context.rubrics_path)


def _fit_structure(
    train: pd.DataFrame,
    q_by: dict[str, Any],
    spec: ExactSpec,
    *,
    nodes: int,
    runtime: Mapping[str, Any],
    initial_A: np.ndarray | None,
    initial_b: np.ndarray | None,
) -> dict[str, Any]:
    Y, M, q_source, items, _block, diagnostics = cm.prepare_block(train, q_by)
    q_modeled = (np.asarray(q_source, dtype=int).sum(axis=1) > 0).astype(int)[:, None]
    fit = cm.fit_m2pl_em(
        Y,
        M,
        q_modeled,
        nodes,
        estimate_corr=False,
        ridge=0.0 if spec.ridge is None else spec.ridge,
        max_iter=int(runtime["fit_max_iter"]),
        tol=float(runtime["objective_tolerance"]),
        calibration_model=spec.family,
        log_a_shrinkage=(
            cm.DEFAULT_LOG_A_SHRINKAGE if spec.log_a_shrinkage is None else spec.log_a_shrinkage
        ),
        quadrature_method=cm.NORMAL_TRAPEZOID_QUADRATURE,
        linear_bound=8.0,
        convergence_mode=cm.RETURNED_ITERATE_CONVERGENCE,
        parameter_tol=float(runtime["parameter_tolerance"]),
        consecutive_convergence_passes=int(runtime["consecutive_convergence_passes"]),
        initial_A=initial_A,
        initial_b=initial_b,
    )
    return {
        **fit,
        "items": list(map(str, items)),
        "dim_labels": [DIMENSION_LABEL],
        "diag": diagnostics,
    }


def _fit_validity(
    fit: Mapping[str, Any],
    spec: ExactSpec,
    *,
    maximum_inner_gradient: float,
) -> dict[str, Any]:
    arrays_finite = bool(
        np.all(np.isfinite(np.asarray(fit["A"], dtype=float)))
        and np.all(np.isfinite(np.asarray(fit["b"], dtype=float)))
        and np.all(np.isfinite(np.asarray(fit["R"], dtype=float)))
        and math.isfinite(float(fit["penalized_objective"]))
    )
    active = np.asarray(fit["A"], dtype=float)[:, 0]
    loadings_valid = bool(
        np.all(active == 1.0) if spec.family == cm.ONE_PL else np.all(active > 0.0)
    )
    diagnostics = fit.get("convergence_diagnostics") or {}
    trace = diagnostics.get("trace") or []
    optimizer_rows = [row.get("mstep_optimizer") or {} for row in trace]
    optimizers_finished = bool(
        optimizer_rows
        and all(
            int(optimizer.get("n_failed", 1)) == 0
            and int(optimizer.get("n_items", 0)) > 0
            and int(optimizer.get("n_converged", -1)) == int(optimizer.get("n_items", 0))
            for optimizer in optimizer_rows
        )
    )
    inner_gradients_finite = bool(
        optimizer_rows
        and all(
            math.isfinite(float(optimizer.get("max_abs_gradient", math.nan)))
            for optimizer in optimizer_rows
        )
    )
    maximum_observed_inner_gradient = (
        max(float(optimizer["max_abs_gradient"]) for optimizer in optimizer_rows)
        if inner_gradients_finite
        else math.inf
    )
    inner_gradient_within_tolerance = bool(
        inner_gradients_finite and maximum_observed_inner_gradient <= maximum_inner_gradient
    )
    returned_matches = diagnostics.get("returned_iterate_matches_last_trace") is True
    exact_recomputation = diagnostics.get("final_exact_recomputation") is True
    stopped_cleanly = diagnostics.get("stopped_before_extra_mstep") is True
    monotone = diagnostics.get("all_objective_changes_monotone_within_tolerance") is True
    valid = bool(
        fit.get("converged") is True
        and arrays_finite
        and loadings_valid
        and optimizers_finished
        and inner_gradient_within_tolerance
        and returned_matches
        and exact_recomputation
        and stopped_cleanly
        and monotone
    )
    return {
        "converged": bool(fit.get("converged")),
        "arrays_and_objective_finite": arrays_finite,
        "loadings_valid": loadings_valid,
        "all_item_optimizers_finished": optimizers_finished,
        "maximum_observed_inner_gradient": maximum_observed_inner_gradient,
        "maximum_allowed_inner_gradient": maximum_inner_gradient,
        "inner_gradient_within_tolerance": inner_gradient_within_tolerance,
        "returned_iterate_matches_last_trace": returned_matches,
        "final_exact_recomputation": exact_recomputation,
        "stopped_before_extra_mstep": stopped_cleanly,
        "penalized_objective_monotone": monotone,
        "fit_valid": valid,
    }


def _run_fit_stage(
    context: Context,
    spec: ExactSpec,
    scope: Scope,
    matrix: pd.DataFrame,
    q_by: dict[str, Any],
    *,
    nodes: int,
    start: str,
    continuation_source: Path | None,
    resume: bool,
) -> None:
    if (nodes, start) not in {
        (401, "cold"),
        (801, "cold"),
        (801, "continuation"),
    }:
        raise V4Error("attempted a fit outside the frozen 54-fit schedule")
    initial_A = None
    initial_b = None
    continuation_sha = None
    if start == "continuation":
        if continuation_source is None or not continuation_source.is_file():
            raise V4Error("801 continuation requires its completed 401 cold fit")
        source_fit = _load_fit(continuation_source)
        initial_A = source_fit["A"]
        initial_b = source_fit["b"]
        continuation_sha = _sha256(continuation_source)
    stage_id = f"fit/{spec.namespace}/nodes_{nodes:04d}/{scope.scope_id}/{start}"
    contract = context.config["frozen_contract"]
    stage_inputs = {
        "spec_id": spec.spec_id,
        "canonical_specification": spec.canonical,
        "nodes": nodes,
        "start": start,
        "continuation_source_sha256": continuation_sha,
        "scope": scope.scope_id,
        "train_model_ids": list(scope.train_models),
        "response_matrix_sha256": _sha256(context.matrix_path),
        "rubrics_sha256": _sha256(context.rubrics_path),
        "split_manifest_sha256": _sha256(context.split_path),
        "fit_integration": contract["fit_integration"],
        "convergence": contract["convergence"],
    }
    destination = _fit_dir(context, spec, nodes, scope, start)
    outputs = [
        destination / "fit.npz",
        destination / "item_params.csv",
        destination / "convergence_trace.json",
        destination / "fit_manifest.json",
    ]
    if resume and _checkpoint_reusable(context, stage_id, stage_inputs):
        return
    if any(path.exists() for path in outputs) or _checkpoint_path(context, stage_id).exists():
        raise V4Error(f"partial/uncheckpointed fit stage exists: {stage_id}")
    started = time.monotonic()
    fit = _fit_structure(
        matrix.loc[list(scope.train_models)],
        q_by,
        spec,
        nodes=nodes,
        runtime=contract["convergence"],
        initial_A=initial_A,
        initial_b=initial_b,
    )
    if fit.get("calibration_specification", {}).get("cache_key") != spec.canonical["cache_key"]:
        raise V4Error(f"fitter returned wrong specification: {spec.spec_id}")
    if fit.get("quadrature_method") != cm.NORMAL_TRAPEZOID_QUADRATURE:
        raise V4Error("fitter did not use normal-trapezoid integration")
    if float(fit.get("quadrature_linear_bound")) != 8.0:
        raise V4Error("fitter did not use the frozen linear bound")
    validity = _fit_validity(
        fit,
        spec,
        maximum_inner_gradient=float(contract["convergence"]["maximum_inner_gradient"]),
    )
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_fit(outputs[0], fit)
    parameters = _parameter_frame(fit, family=spec.family)
    _atomic_csv(outputs[1], parameters)
    diagnostics = fit.get("convergence_diagnostics") or {}
    _atomic_json(
        outputs[2],
        {
            "schema_version": RUNNER_SCHEMA,
            "spec_id": spec.spec_id,
            "scope": scope.scope_id,
            "nodes": nodes,
            "start": start,
            "initial_evaluation": diagnostics.get("initial_evaluation"),
            "trace": diagnostics.get("trace") or [],
        },
    )
    fitted_items = list(map(str, fit["items"]))
    missing_fitted_items = sorted(set(fitted_items) - set(matrix.columns))
    if missing_fitted_items:
        raise V4Error(
            "fitted items are absent from the response matrix: "
            + ", ".join(missing_fitted_items[:5])
        )
    n_observed = int(
        matrix.loc[list(scope.train_models), fitted_items].notna().to_numpy(dtype=bool).sum()
    )
    _atomic_json(
        outputs[3],
        {
            "schema_version": RUNNER_SCHEMA,
            "stage_inputs": stage_inputs,
            "stage_inputs_sha256": _canonical_sha256(stage_inputs),
            "spec_id": spec.spec_id,
            "family": spec.family,
            "scope": scope.scope_id,
            "nodes": nodes,
            "start": start,
            "n_observed_train_cells": n_observed,
            "n_items": len(fit["items"]),
            "n_exportable": int(parameters["exportable"].sum()),
            "n_iter": int(fit["n_iter"]),
            "loglik": float(fit["loglik"]),
            "penalized_objective": float(fit["penalized_objective"]),
            "fit_validity": validity,
            "fit_sha256": _sha256(outputs[0]),
            "parameter_csv_sha256": _sha256(outputs[1]),
            "trace_sha256": _sha256(outputs[2]),
            "continuation_source_sha256": continuation_sha,
            "historical_fit_or_checkpoint_artifacts_reused": 0,
        },
    )
    _write_checkpoint(
        context,
        stage_id,
        stage_inputs,
        outputs,
        time.monotonic() - started,
    )


def _normal_rule(nodes: int, bound: float) -> tuple[np.ndarray, np.ndarray]:
    grid, log_prior = cm.build_calibration_quadrature(
        1,
        nodes,
        method=cm.NORMAL_TRAPEZOID_QUADRATURE,
        linear_bound=bound,
    )
    return grid[:, 0], log_prior


def _posterior(
    raw: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    indices: np.ndarray,
    *,
    nodes: int,
    bound: float,
    tail_region: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis, log_prior = _normal_rule(nodes, bound)
    selected = np.asarray(indices, dtype=int)
    values = np.asarray(raw[:, selected], dtype=float)
    observed = np.isfinite(values)
    y = np.nan_to_num(values, nan=0.0)
    successes = np.where(observed, y, 0.0)
    failures = np.where(observed, 1.0 - y, 0.0)
    eta = a[selected, 0][:, None] * axis[None, :] - b[selected, None]
    likelihood = successes @ log_expit(eta) + failures @ log_expit(-eta)
    joint = likelihood + log_prior[None, :]
    posterior = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    theta = posterior @ axis
    centered = axis[None, :] - theta[:, None]
    se = np.sqrt(np.clip((posterior * centered**2).sum(axis=1), 0.0, None))
    tail = posterior[:, np.abs(axis) >= tail_region].sum(axis=1)
    return theta, se, tail


def _fit_arrays_on_ids(
    fit: Mapping[str, Any], criterion_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    position = {str(item): index for index, item in enumerate(fit["items"])}
    try:
        indices = np.asarray([position[str(item)] for item in criterion_ids], dtype=int)
    except KeyError as error:
        raise V4Error(f"fit is missing common criterion {error}") from error
    return (
        np.asarray(fit["A"], dtype=float)[indices],
        np.asarray(fit["b"], dtype=float)[indices],
    )


def _start_agreement_gate(
    *,
    cold_fit: Mapping[str, Any],
    continuation_fit: Mapping[str, Any],
    cold_manifest: Mapping[str, Any],
    continuation_manifest: Mapping[str, Any],
    raw: np.ndarray,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if list(cold_fit["items"]) != list(continuation_fit["items"]):
        raise V4Error("cold/continuation item orders differ")
    parameter_shift = float(
        max(
            np.max(np.abs(cold_fit["A"] - continuation_fit["A"])),
            np.max(np.abs(cold_fit["b"] - continuation_fit["b"])),
            np.max(np.abs(cold_fit["R"] - continuation_fit["R"])),
        )
    )
    n_observed = int(cold_manifest.get("n_observed_train_cells", 0))
    if n_observed <= 0 or n_observed != int(
        continuation_manifest.get("n_observed_train_cells", -1)
    ):
        raise V4Error("cold/continuation observed-cell denominators differ")
    objective_gap = abs(
        float(cold_fit["penalized_objective"]) - float(continuation_fit["penalized_objective"])
    )
    objective_gap_per_cell = objective_gap / n_observed
    cold_parameters = _parameter_frame(cold_fit, family=str(cold_manifest.get("family") or ""))
    continuation_parameters = _parameter_frame(
        continuation_fit,
        family=str(continuation_manifest.get("family") or ""),
    )
    exportability_identical = bool(
        cold_parameters["exportable"].equals(continuation_parameters["exportable"])
    )
    indices = np.arange(len(cold_fit["items"]), dtype=int)
    cold_theta, _cold_se, _cold_tail = _posterior(
        raw,
        cold_fit["A"],
        cold_fit["b"],
        indices,
        nodes=int(thresholds["theta_scoring_nodes"]),
        bound=float(thresholds["theta_scoring_bound"]),
        tail_region=7.5,
    )
    continuation_theta, _continuation_se, _continuation_tail = _posterior(
        raw,
        continuation_fit["A"],
        continuation_fit["b"],
        indices,
        nodes=int(thresholds["theta_scoring_nodes"]),
        bound=float(thresholds["theta_scoring_bound"]),
        tail_region=7.5,
    )
    theta_shift = np.abs(cold_theta - continuation_theta)
    median = float(np.median(theta_shift))
    p95 = float(np.quantile(theta_shift, 0.95))
    cold_valid = bool((cold_manifest.get("fit_validity") or {}).get("fit_valid"))
    continuation_valid = bool((continuation_manifest.get("fit_validity") or {}).get("fit_valid"))
    passed = bool(
        cold_valid
        and continuation_valid
        and objective_gap_per_cell
        <= float(thresholds["maximum_penalized_objective_gap_per_observed_cell"])
        and parameter_shift <= float(thresholds["maximum_parameter_coordinate_shift"])
        and median <= float(thresholds["maximum_median_absolute_theta_shift"])
        and p95 <= float(thresholds["maximum_p95_absolute_theta_shift"])
        and (exportability_identical if thresholds["require_identical_exportability"] else True)
    )
    return {
        "cold_fit_valid": cold_valid,
        "continuation_fit_valid": continuation_valid,
        "penalized_objective_gap": objective_gap,
        "penalized_objective_gap_per_observed_cell": objective_gap_per_cell,
        "maximum_parameter_coordinate_shift": parameter_shift,
        "exportability_identical": exportability_identical,
        "median_absolute_theta_shift": median,
        "p95_absolute_theta_shift": p95,
        "maximum_absolute_theta_shift_diagnostic": float(theta_shift.max()),
        "thresholds": dict(thresholds),
        "passed": passed,
    }


def _run_start_selection(
    context: Context,
    spec: ExactSpec,
    scope: Scope,
    matrix: pd.DataFrame,
    *,
    resume: bool,
) -> dict[str, Any]:
    cold_dir = _fit_dir(context, spec, 801, scope, "cold")
    continuation_dir = _fit_dir(context, spec, 801, scope, "continuation")
    stage_id = f"start_selection/{spec.namespace}/{scope.scope_id}"
    inputs = {
        "spec_id": spec.spec_id,
        "scope": scope.scope_id,
        "cold_fit_sha256": _sha256(cold_dir / "fit.npz"),
        "continuation_fit_sha256": _sha256(continuation_dir / "fit.npz"),
        "start_policy": context.config["frozen_contract"]["start_policy"],
        "start_agreement_gate": context.config["frozen_contract"]["start_agreement_gate"],
    }
    output = _selection_path(context, spec, scope)
    if resume and _checkpoint_reusable(context, stage_id, inputs):
        return _read_json(output)
    if output.exists() or _checkpoint_path(context, stage_id).exists():
        raise V4Error(f"partial start-selection stage exists: {stage_id}")
    started = time.monotonic()
    cold_fit = _load_fit(cold_dir / "fit.npz")
    continuation_fit = _load_fit(continuation_dir / "fit.npz")
    cold_manifest = _read_json(cold_dir / "fit_manifest.json")
    continuation_manifest = _read_json(continuation_dir / "fit_manifest.json")
    raw = matrix.loc[list(scope.score_models)].reindex(columns=cold_fit["items"]).to_numpy(float)
    gate = _start_agreement_gate(
        cold_fit=cold_fit,
        continuation_fit=continuation_fit,
        cold_manifest=cold_manifest,
        continuation_manifest=continuation_manifest,
        raw=raw,
        thresholds=context.config["frozen_contract"]["start_agreement_gate"],
    )
    cold_objective = float(cold_fit["penalized_objective"])
    continuation_objective = float(continuation_fit["penalized_objective"])
    selected = "cold" if cold_objective >= continuation_objective else "continuation"
    selected_dir = cold_dir if selected == "cold" else continuation_dir
    result = {
        "schema_version": RUNNER_SCHEMA,
        "spec_id": spec.spec_id,
        "scope": scope.scope_id,
        "nodes": 801,
        "agreement_gate": gate,
        "selection_rule": ("higher_penalized_objective_exact_tie_to_cold"),
        "cold_penalized_objective": cold_objective,
        "continuation_penalized_objective": continuation_objective,
        "selected_start": selected,
        "selected_fit": _display(selected_dir / "fit.npz"),
        "selected_fit_sha256": _sha256(selected_dir / "fit.npz"),
        "selection_valid": gate["passed"],
        "continuation_cannot_rescue_cold": True,
        "calibration_model_selected": False,
    }
    _atomic_json(output, result)
    _write_checkpoint(
        context,
        stage_id,
        inputs,
        [output],
        time.monotonic() - started,
    )
    return result


def _selected_801_dir(context: Context, spec: ExactSpec, scope: Scope) -> Path:
    selection = _read_json(_selection_path(context, spec, scope))
    selected = selection.get("selected_start")
    if selected not in {"cold", "continuation"}:
        raise V4Error(f"invalid stored start selection: {spec.spec_id}/{scope.scope_id}")
    path = _fit_dir(context, spec, 801, scope, str(selected))
    _require_hash(
        path / "fit.npz",
        selection["selected_fit_sha256"],
        "selected 801 fit",
    )
    return path


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    value = frame["left"].corr(frame["right"], method="spearman")
    return float(value) if value == value else None


def _parameter_scope_gate(
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    *,
    family: str,
    source_ids: Sequence[str],
    minimum_spearman: float,
    minimum_exportability_agreement: float,
    lower_valid: bool,
    upper_valid: bool,
) -> dict[str, Any]:
    left = lower.set_index("criterion_id")
    right = upper.set_index("criterion_id")
    if not left.index.is_unique or not right.index.is_unique:
        raise V4Error("parameter tables contain duplicate criterion IDs")
    common = sorted(set(left.index) & set(right.index))
    if not common:
        raise V4Error("fit resolutions have no common fitted items")
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
        a_method = "fixed-a-exact-equality"
    else:
        a_spearman = _spearman(
            left.loc[common, "a_instruction_following"],
            right.loc[common, "a_instruction_following"],
        )
        fixed_exact = None
        a_passed = a_spearman is not None and a_spearman >= minimum_spearman
        a_method = "common-item-spearman"
    b_passed = b_spearman is not None and b_spearman >= minimum_spearman
    passed = bool(
        lower_valid
        and upper_valid
        and a_passed
        and b_passed
        and agreement >= minimum_exportability_agreement
    )
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
        "minimum_exportability_agreement": (minimum_exportability_agreement),
        "lower_fit_valid": lower_valid,
        "upper_fit_valid": upper_valid,
        "passed": passed,
    }


def _refit_theta_gate(rows: pd.DataFrame, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "scope",
        "model",
        "support",
        "theta_401",
        "theta_801",
    }
    keys = ["scope", "model", "support"]
    if required - set(rows.columns) or rows.empty or rows.duplicated(keys).any():
        raise V4Error("refit-bank theta rows are incomplete or duplicated")
    shift = np.abs(rows["theta_801"].to_numpy(float) - rows["theta_401"].to_numpy(float))
    finite = bool(np.isfinite(shift).all())
    median = float(np.median(shift)) if finite else math.inf
    p95 = float(np.quantile(shift, 0.95)) if finite else math.inf
    maximum = float(np.max(shift)) if finite else math.inf
    passed = bool(
        finite
        and median <= float(thresholds["maximum_median_absolute_theta_shift"])
        and p95 <= float(thresholds["maximum_p95_absolute_theta_shift"])
    )
    return {
        "n_rows": len(rows),
        "median_absolute_theta_shift": median,
        "p95_absolute_theta_shift": p95,
        "maximum_absolute_theta_shift_diagnostic_only": maximum,
        "maximum_is_a_gate": False,
        "thresholds": {
            "maximum_median_absolute_theta_shift": float(
                thresholds["maximum_median_absolute_theta_shift"]
            ),
            "maximum_p95_absolute_theta_shift": float(
                thresholds["maximum_p95_absolute_theta_shift"]
            ),
        },
        "passed": passed,
    }


def _cluster_interval(
    frame: pd.DataFrame,
    column: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    grouped = frame.groupby("family", sort=True)[column].agg(["sum", "count"])
    if len(grouped) < 2:
        raise V4Error("family bootstrap requires at least two tutor-model families")
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        values[index] = sums[selected].sum() / counts[selected].sum()
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def _common_cell_equivalence(
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    *,
    margin: float,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    keys = ["outer_fold", "model", "criterion_id"]
    required = set(keys + ["family", "label", "probability"])
    for nodes, frame in ((401, lower), (801, upper)):
        if required - set(frame.columns) or frame.duplicated(keys).any():
            raise V4Error(f"nodes-{nodes} held-out cell table is malformed")
    left_keys = lower[keys].sort_values(keys).reset_index(drop=True)
    right_keys = upper[keys].sort_values(keys).reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise V4Error("fit resolutions did not score identical held-out cells")
    merged = lower.merge(
        upper,
        on=keys,
        suffixes=("_401", "_801"),
        validate="one_to_one",
    )
    if merged.empty or not (merged["label_401"] == merged["label_801"]).all():
        raise V4Error("common held-out labels differ or are empty")
    if not (merged["family_401"] == merged["family_801"]).all():
        raise V4Error("common held-out model families differ")
    merged["label"] = merged["label_401"].astype(int)
    merged["family"] = merged["family_401"]
    y = merged["label"].to_numpy(float)
    for nodes in FIT_NODES:
        probability = np.clip(
            merged[f"probability_{nodes}"].to_numpy(float),
            EPSILON,
            1 - EPSILON,
        )
        merged[f"log_loss_{nodes}"] = -(y * np.log(probability) + (1 - y) * np.log(1 - probability))
        merged[f"brier_{nodes}"] = (probability - y) ** 2
    details: dict[str, Any] = {}
    overall = True
    for offset, metric in enumerate(("log_loss", "brier")):
        column = f"{metric}_delta_801_minus_401"
        merged[column] = merged[f"{metric}_801"] - merged[f"{metric}_401"]
        pooled = float(merged[column].mean())
        low, high = _cluster_interval(
            merged,
            column,
            replicates=replicates,
            seed=seed + offset,
        )
        passed = abs(pooled) <= margin and low >= -margin and high <= margin
        details[metric] = {
            "pooled_shift_801_minus_401": pooled,
            "absolute_pooled_shift": abs(pooled),
            "family_cluster_bootstrap_ci_95": [low, high],
            "equivalence_margin": margin,
            "passed": bool(passed),
        }
        overall = overall and passed
    columns = keys + [
        "family",
        "label",
        "probability_401",
        "probability_801",
        "log_loss_delta_801_minus_401",
        "brier_delta_801_minus_401",
    ]
    return (
        {
            "n_common_cells": len(merged),
            "cluster_unit": "tutor_model_family",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "metrics": details,
            "passed": bool(overall),
        },
        merged[columns].sort_values(keys),
    )


def _fit_comparison_evidence(
    context: Context,
    spec: ExactSpec,
    matrix: pd.DataFrame,
    item_to_scenario: Mapping[str, str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    gate = context.config["frozen_contract"]["refit_bank_gate"]
    theta_rows: list[dict[str, Any]] = []
    predictions: dict[int, list[dict[str, Any]]] = {
        401: [],
        801: [],
    }
    support_rows: list[dict[str, Any]] = []
    for scope in context.scopes:
        directories = {
            401: _fit_dir(context, spec, 401, scope, "cold"),
            801: _selected_801_dir(context, spec, scope),
        }
        fits = {nodes: _load_fit(path / "fit.npz") for nodes, path in directories.items()}
        parameters = {
            nodes: pd.read_csv(path / "item_params.csv").set_index("criterion_id")
            for nodes, path in directories.items()
        }
        common = sorted(
            set(parameters[401].index[parameters[401]["exportable"].astype(bool)])
            & set(parameters[801].index[parameters[801]["exportable"].astype(bool)])
        )
        administration = [
            item
            for item in common
            if item_to_scenario.get(item) in context.administration_scenarios
        ]
        evaluation = [
            item for item in common if item_to_scenario.get(item) in context.evaluation_scenarios
        ]
        if not common or not administration or (scope.outer_fold is not None and not evaluation):
            raise V4Error(f"common support is incomplete: {spec.spec_id}/{scope.scope_id}")
        raw = matrix.loc[list(scope.score_models)].reindex(columns=common).to_numpy(float)
        positions = {item: index for index, item in enumerate(common)}
        admin_index = np.asarray(
            [positions[item] for item in administration],
            dtype=int,
        )
        eval_index = np.asarray(
            [positions[item] for item in evaluation],
            dtype=int,
        )
        support_rows.append(
            {
                "scope": scope.scope_id,
                "n_common_exportable_items": len(common),
                "n_common_administration_items": len(administration),
                "n_common_evaluation_items": len(evaluation),
                "n_score_models": len(scope.score_models),
            }
        )
        theta_by_nodes: dict[int, dict[str, np.ndarray]] = {401: {}, 801: {}}
        arrays = {nodes: _fit_arrays_on_ids(fits[nodes], common) for nodes in FIT_NODES}
        for nodes in FIT_NODES:
            a, b = arrays[nodes]
            for support, indices in (
                (
                    "full_bank",
                    np.arange(len(common), dtype=int),
                ),
                (
                    "administration_only",
                    admin_index,
                ),
            ):
                theta, _se, _tail = _posterior(
                    raw,
                    a,
                    b,
                    indices,
                    nodes=int(gate["theta_scoring_nodes"]),
                    bound=float(gate["theta_scoring_bound"]),
                    tail_region=7.5,
                )
                theta_by_nodes[nodes][support] = theta
        for model_index, model in enumerate(scope.score_models):
            for support in gate["theta_supports"]:
                theta_rows.append(
                    {
                        "scope": scope.scope_id,
                        "outer_fold": (scope.outer_fold),
                        "model": model,
                        "family": (context.model_to_family[model]),
                        "support": support,
                        "theta_401": float(theta_by_nodes[401][support][model_index]),
                        "theta_801": float(theta_by_nodes[801][support][model_index]),
                    }
                )
        if scope.outer_fold is None:
            continue
        observed = np.isfinite(raw[:, eval_index])
        for nodes in FIT_NODES:
            a, b = arrays[nodes]
            theta = theta_by_nodes[nodes]["administration_only"]
            probability = expit(theta[:, None] * a[eval_index, 0][None, :] - b[eval_index][None, :])
            for model_index, criterion_index in np.argwhere(observed):
                model = scope.score_models[int(model_index)]
                predictions[nodes].append(
                    {
                        "outer_fold": (scope.outer_fold),
                        "model": model,
                        "family": (context.model_to_family[model]),
                        "criterion_id": evaluation[int(criterion_index)],
                        "label": int(
                            raw[
                                model_index,
                                eval_index[criterion_index],
                            ]
                        ),
                        "probability": float(
                            probability[
                                model_index,
                                criterion_index,
                            ]
                        ),
                    }
                )
    return (
        pd.DataFrame(theta_rows),
        pd.DataFrame(predictions[401]),
        pd.DataFrame(predictions[801]),
        pd.DataFrame(support_rows),
    )


def _run_fit_comparison(
    context: Context,
    spec: ExactSpec,
    matrix: pd.DataFrame,
    item_to_scenario: Mapping[str, str],
    *,
    resume: bool,
) -> dict[str, Any]:
    stage_id = f"fit_comparison/{spec.namespace}/401_vs_801"
    inputs = {
        "spec_id": spec.spec_id,
        "comparison": "401_vs_801",
        "grid401_fit_sha256": {
            scope.scope_id: _sha256(
                _fit_dir(
                    context,
                    spec,
                    401,
                    scope,
                    "cold",
                )
                / "fit.npz"
            )
            for scope in context.scopes
        },
        "selected_grid801_fit_sha256": {
            scope.scope_id: _sha256(_selected_801_dir(context, spec, scope) / "fit.npz")
            for scope in context.scopes
        },
        "start_selection_sha256": {
            scope.scope_id: _sha256(_selection_path(context, spec, scope))
            for scope in context.scopes
        },
        "refit_bank_gate": context.config["frozen_contract"]["refit_bank_gate"],
    }
    output_dir = _fit_comparison_dir(context, spec)
    outputs = [
        output_dir / "parameter_scope_gates.csv",
        output_dir / "common_support.csv",
        output_dir / "theta_common_support.csv",
        output_dir / "heldout_cells.csv",
        output_dir / "fit_pair_gate.json",
    ]
    if resume and _checkpoint_reusable(context, stage_id, inputs):
        return _read_json(outputs[-1])
    if any(path.exists() for path in outputs) or _checkpoint_path(context, stage_id).exists():
        raise V4Error(f"partial fit-comparison stage exists: {stage_id}")
    started = time.monotonic()
    gate = context.config["frozen_contract"]["refit_bank_gate"]
    parameter_rows: list[dict[str, Any]] = []
    start_gates_passed = True
    for scope in context.scopes:
        lower_dir = _fit_dir(context, spec, 401, scope, "cold")
        upper_dir = _selected_801_dir(context, spec, scope)
        lower_manifest = _read_json(lower_dir / "fit_manifest.json")
        upper_manifest = _read_json(upper_dir / "fit_manifest.json")
        selection = _read_json(_selection_path(context, spec, scope))
        start_gates_passed = start_gates_passed and selection["selection_valid"] is True
        result = _parameter_scope_gate(
            pd.read_csv(lower_dir / "item_params.csv"),
            pd.read_csv(upper_dir / "item_params.csv"),
            family=spec.family,
            source_ids=list(map(str, matrix.columns)),
            minimum_spearman=float(gate["minimum_item_parameter_spearman"]),
            minimum_exportability_agreement=float(gate["minimum_exportability_agreement"]),
            lower_valid=bool((lower_manifest.get("fit_validity") or {}).get("fit_valid")),
            upper_valid=bool((upper_manifest.get("fit_validity") or {}).get("fit_valid")),
        )
        parameter_rows.append(
            {
                "scope": scope.scope_id,
                **result,
            }
        )
    (
        theta_rows,
        lower_cells,
        upper_cells,
        support_rows,
    ) = _fit_comparison_evidence(
        context,
        spec,
        matrix,
        item_to_scenario,
    )
    theta_gate = _refit_theta_gate(theta_rows, gate)
    seed = int(gate["family_bootstrap_seed"]) + int(
        spec.canonical["cache_key"][-8:],
        16,
    )
    cell_gate, cell_details = _common_cell_equivalence(
        lower_cells,
        upper_cells,
        margin=float(gate["equivalence_margin"]),
        replicates=int(gate["family_bootstrap_replicates"]),
        seed=seed,
    )
    parameters_passed = all(bool(row["passed"]) for row in parameter_rows)
    result = {
        "schema_version": RUNNER_SCHEMA,
        "spec_id": spec.spec_id,
        "comparison": ("normal_trapezoid_bound8_401_vs_801"),
        "all_scope_start_agreement_gates_passed": bool(start_gates_passed),
        "all_scope_parameter_gates_passed": bool(parameters_passed),
        "heldout_common_cell_gate": cell_gate,
        "common_support_refit_theta_gate": (theta_gate),
        "maximum_theta_shift_is_diagnostic_only": True,
        "posthoc_affine_linking_applied": False,
        "calibration_model_selected": False,
        "adaptive_results_inspected": False,
        "passed": bool(
            start_gates_passed
            and parameters_passed
            and cell_gate["passed"]
            and theta_gate["passed"]
        ),
    }
    _atomic_csv(
        outputs[0],
        pd.DataFrame(parameter_rows),
    )
    _atomic_csv(outputs[1], support_rows)
    _atomic_csv(outputs[2], theta_rows)
    _atomic_csv(outputs[3], cell_details)
    _atomic_json(outputs[4], result)
    _write_checkpoint(
        context,
        stage_id,
        inputs,
        outputs,
        time.monotonic() - started,
    )
    return result


def _fixed_bank_scoring_gate(
    theta_rows: pd.DataFrame,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply pure-integration gates while holding every fitted bank fixed."""

    profiles = thresholds.get("profiles") or {}
    comparisons = list(thresholds.get("comparisons") or [])
    expected_profiles = {
        "bound8_801",
        "bound8_1601",
        "bound10_1001",
    }
    expected_comparisons = {
        "bound8_801_vs_bound8_1601": (
            "bound8_801",
            "bound8_1601",
        ),
        "bound8_801_vs_bound10_1001": (
            "bound8_801",
            "bound10_1001",
        ),
    }
    if set(profiles) != expected_profiles:
        raise V4Error("fixed-bank scoring profile roster changed")
    if comparisons != list(expected_comparisons):
        raise V4Error("fixed-bank scoring comparison roster changed")
    if theta_rows.empty:
        return {
            "n_scored_model_support_rows": 0,
            "comparisons": {},
            "tail_mass": {
                "maximum_posterior_tail_mass": None,
                "threshold": float(thresholds["maximum_posterior_tail_mass"]),
                "passed": False,
            },
            "all_values_finite": False,
            "passed": False,
        }

    required = {
        *(f"theta_{profile}" for profile in profiles),
        *(f"tail_{profile}" for profile in profiles),
    }
    missing = sorted(required - set(theta_rows.columns))
    if missing:
        raise V4Error("fixed-bank evidence is missing columns: " + ", ".join(missing))
    values = theta_rows.loc[:, sorted(required)].to_numpy(float)
    finite = bool(np.all(np.isfinite(values)))
    maximum_shift = float(thresholds["maximum_absolute_theta_shift"])
    comparison_results: dict[str, Any] = {}
    comparisons_pass = finite
    for name, (reference, candidate) in expected_comparisons.items():
        shift = np.abs(
            theta_rows[f"theta_{reference}"].to_numpy(float)
            - theta_rows[f"theta_{candidate}"].to_numpy(float)
        )
        observed = float(np.max(shift)) if finite else math.inf
        passed = bool(finite and observed <= maximum_shift)
        comparison_results[name] = {
            "reference_profile": reference,
            "candidate_profile": candidate,
            "maximum_absolute_theta_shift": observed,
            "threshold": maximum_shift,
            "passed": passed,
        }
        comparisons_pass = comparisons_pass and passed

    tail_columns = [f"tail_{profile}" for profile in profiles]
    tail_values = theta_rows.loc[:, tail_columns].to_numpy(float)
    observed_tail = float(np.max(tail_values)) if finite else math.inf
    tail_threshold = float(thresholds["maximum_posterior_tail_mass"])
    tail_passed = bool(finite and observed_tail <= tail_threshold)
    return {
        "n_scored_model_support_rows": int(len(theta_rows)),
        "comparisons": comparison_results,
        "tail_mass": {
            "maximum_posterior_tail_mass": observed_tail,
            "threshold": tail_threshold,
            "passed": tail_passed,
        },
        "all_values_finite": finite,
        "passed": bool(comparisons_pass and tail_passed),
    }


def _run_fixed_bank_scoring(
    context: Context,
    spec: ExactSpec,
    matrix: pd.DataFrame,
    item_to_scenario: Mapping[str, str],
    *,
    resume: bool,
) -> dict[str, Any]:
    """Score each frozen 401-node bank on denser/broader EAP rules."""

    thresholds = context.config["frozen_contract"]["fixed_bank_scoring_gate"]
    fit_hashes = {
        scope.scope_id: {
            "fit": _sha256(_fit_dir(context, spec, 401, scope, "cold") / "fit.npz"),
            "manifest": _sha256(_fit_dir(context, spec, 401, scope, "cold") / "fit_manifest.json"),
        }
        for scope in context.scopes
    }
    refit_gate_path = _fit_comparison_dir(context, spec) / "fit_pair_gate.json"
    stage_id = f"fixed_bank/{spec.namespace}"
    stage_inputs = {
        "spec_id": spec.spec_id,
        "fixed_bank_nodes": 401,
        "fit_sha256": fit_hashes,
        "refit_gate_sha256": _sha256(refit_gate_path),
        "profiles": thresholds["profiles"],
        "comparisons": thresholds["comparisons"],
        "maximum_absolute_theta_shift": thresholds["maximum_absolute_theta_shift"],
        "maximum_posterior_tail_mass": thresholds["maximum_posterior_tail_mass"],
        "theta_supports": thresholds["theta_supports"],
    }
    output_dir = context.out_dir / "fixed_bank" / spec.namespace
    outputs = [
        output_dir / "theta_profiles.csv",
        output_dir / "support.csv",
        output_dir / "fixed_bank_gate.json",
    ]
    if resume and _checkpoint_reusable(context, stage_id, stage_inputs):
        return _read_json(outputs[-1])
    if any(path.exists() for path in outputs) or _checkpoint_path(context, stage_id).exists():
        raise V4Error(f"partial fixed-bank stage exists: {stage_id}")

    started = time.monotonic()
    profiles = thresholds["profiles"]
    theta_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    all_banks_valid = True
    for scope in context.scopes:
        fit_dir = _fit_dir(context, spec, 401, scope, "cold")
        fit = _load_fit(fit_dir / "fit.npz")
        manifest = _read_json(fit_dir / "fit_manifest.json")
        all_banks_valid = all_banks_valid and bool(
            (manifest.get("fit_validity") or {}).get("fit_valid")
        )
        parameters = pd.read_csv(fit_dir / "item_params.csv")
        parameters["criterion_id"] = parameters["criterion_id"].astype(str)
        exportable = sorted(
            parameters.loc[parameters["exportable"].astype(bool), "criterion_id"].tolist()
        )
        administration = [
            item
            for item in exportable
            if item_to_scenario.get(item) in context.administration_scenarios
        ]
        if not exportable or not administration:
            raise V4Error(f"fixed-bank support is incomplete: {spec.spec_id}/{scope.scope_id}")
        raw = matrix.loc[list(scope.score_models)].reindex(columns=exportable).to_numpy(float)
        a, b = _fit_arrays_on_ids(fit, exportable)
        positions = {item: index for index, item in enumerate(exportable)}
        supports = {
            "full_bank": np.arange(len(exportable), dtype=int),
            "administration_only": np.asarray(
                [positions[item] for item in administration], dtype=int
            ),
        }
        if set(thresholds["theta_supports"]) != set(supports):
            raise V4Error("fixed-bank theta support roster changed")
        support_rows.append(
            {
                "scope": scope.scope_id,
                "n_exportable_items": len(exportable),
                "n_administration_items": len(administration),
                "n_score_models": len(scope.score_models),
                "fixed_bank_fit_valid": bool((manifest.get("fit_validity") or {}).get("fit_valid")),
            }
        )
        profile_values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for profile_name, profile in profiles.items():
            if (
                not isinstance(profile, list)
                or len(profile) != 4
                or profile[0] != cm.NORMAL_TRAPEZOID_QUADRATURE
            ):
                raise V4Error(f"invalid fixed-bank profile: {profile_name}")
            profile_values[profile_name] = _posterior(
                raw,
                a,
                b,
                supports["full_bank"],
                nodes=int(profile[1]),
                bound=float(profile[2]),
                tail_region=float(profile[3]),
            )
        administration_positions = supports["administration_only"]
        administration_values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for profile_name, profile in profiles.items():
            administration_values[profile_name] = _posterior(
                raw,
                a,
                b,
                administration_positions,
                nodes=int(profile[1]),
                bound=float(profile[2]),
                tail_region=float(profile[3]),
            )
        for support_name, values_by_profile in (
            ("full_bank", profile_values),
            ("administration_only", administration_values),
        ):
            for model_index, model in enumerate(scope.score_models):
                row: dict[str, Any] = {
                    "scope": scope.scope_id,
                    "outer_fold": scope.outer_fold,
                    "model": model,
                    "family": context.model_to_family[model],
                    "support": support_name,
                }
                for profile_name, (theta, se, tail) in values_by_profile.items():
                    row[f"theta_{profile_name}"] = float(theta[model_index])
                    row[f"se_{profile_name}"] = float(se[model_index])
                    row[f"tail_{profile_name}"] = float(tail[model_index])
                theta_rows.append(row)

    theta_frame = pd.DataFrame(theta_rows).sort_values(["scope", "model", "support"])
    numerical_gate = _fixed_bank_scoring_gate(theta_frame, thresholds)
    result = {
        "schema_version": RUNNER_SCHEMA,
        "spec_id": spec.spec_id,
        "fixed_fitted_bank_profile": [
            cm.NORMAL_TRAPEZOID_QUADRATURE,
            401,
            8.0,
        ],
        "profiles": profiles,
        "all_six_fixed_banks_valid": bool(all_banks_valid),
        "numerical_scoring_gate": numerical_gate,
        "calibration_model_selected": False,
        "adaptive_results_inspected": False,
        "passed": bool(all_banks_valid and numerical_gate["passed"]),
    }
    _atomic_csv(outputs[0], theta_frame)
    _atomic_csv(outputs[1], pd.DataFrame(support_rows))
    _atomic_json(outputs[2], result)
    _write_checkpoint(
        context,
        stage_id,
        stage_inputs,
        outputs,
        time.monotonic() - started,
    )
    return result


def _rehash_before_finalization(context: Context) -> dict[str, str]:
    """Fail closed if any preregistered source changed after launch."""

    raw = _read_json(context.config_path)
    _validate_config(raw)
    if _sha256(context.config_path) != context.study_signature["config_sha256"]:
        raise V4Error("V4 config changed after launch")
    code_hashes = _verify_code_freeze(raw)
    design = raw["design_source"]
    _require_hash(_resolve(design["path"]), design["sha256"], "V4 design source")
    frozen = raw["frozen_inputs"]
    for key in ("response_matrix", "rubrics", "scenarios", "split_manifest"):
        _require_hash(
            _resolve(frozen[key]),
            frozen[f"{key}_sha256"],
            f"frozen {key}",
        )
    _verify_historical_evidence(raw)
    return code_hashes


def _finalize(
    context: Context,
    refit_gates: Mapping[str, Mapping[str, Any]],
    fixed_bank_gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Write a terminal decision and, only on complete success, a lock."""

    observed_code_hashes = _rehash_before_finalization(context)
    expected_spec_ids = [spec.spec_id for spec in context.specs]
    fit_validity: dict[str, bool] = {}
    start_validity: dict[str, bool] = {}
    evidence_paths: dict[str, str] = {}
    evidence_hashes: dict[str, str] = {}

    def add_evidence(key: str, path: Path) -> None:
        if key in evidence_paths:
            raise V4Error(f"duplicate evidence key: {key}")
        if not path.is_file():
            raise V4Error(f"required V4 evidence is missing: {path}")
        evidence_paths[key] = _display(path)
        evidence_hashes[key] = _sha256(path)

    for spec in context.specs:
        for scope in context.scopes:
            for nodes, start in (
                (401, "cold"),
                (801, "cold"),
                (801, "continuation"),
            ):
                manifest_path = _fit_dir(context, spec, nodes, scope, start) / "fit_manifest.json"
                manifest = _read_json(manifest_path)
                key = f"fit/{spec.spec_id}/{scope.scope_id}/{nodes}/{start}"
                fit_validity[key] = bool((manifest.get("fit_validity") or {}).get("fit_valid"))
                add_evidence(key, manifest_path)
            selection_path = _selection_path(context, spec, scope)
            selection = _read_json(selection_path)
            selection_key = f"start/{spec.spec_id}/{scope.scope_id}"
            start_validity[selection_key] = selection.get("selection_valid") is True
            add_evidence(selection_key, selection_path)

        refit_path = _fit_comparison_dir(context, spec) / "fit_pair_gate.json"
        add_evidence(f"refit/{spec.spec_id}", refit_path)
        if spec.spec_id in fixed_bank_gates:
            add_evidence(
                f"fixed_bank/{spec.spec_id}",
                context.out_dir / "fixed_bank" / spec.namespace / "fixed_bank_gate.json",
            )

    if set(evidence_paths) != set(evidence_hashes):
        raise V4Error("evidence path/hash keys differ")
    all_fits_valid = bool(fit_validity) and all(fit_validity.values())
    all_starts_valid = bool(start_validity) and all(start_validity.values())
    all_refits_passed = set(refit_gates) == set(expected_spec_ids) and all(
        refit_gates[spec_id].get("passed") is True for spec_id in expected_spec_ids
    )
    all_fixed_bank_passed = set(fixed_bank_gates) == set(expected_spec_ids) and all(
        fixed_bank_gates[spec_id].get("passed") is True
        and (fixed_bank_gates[spec_id].get("numerical_scoring_gate") or {}).get("passed") is True
        and (
            (fixed_bank_gates[spec_id].get("numerical_scoring_gate") or {}).get("tail_mass") or {}
        ).get("passed")
        is True
        for spec_id in expected_spec_ids
    )
    passed = bool(
        all_fits_valid and all_starts_valid and all_refits_passed and all_fixed_bank_passed
    )
    convergence = context.config["frozen_contract"]["convergence"]
    common_fields = {
        "fit_convergence_mode": "returned_iterate",
        "fit_max_iter": 1500,
        "fit_objective_tolerance": 1e-4,
        "fit_parameter_tolerance": 5e-5,
        "fit_consecutive_convergence_passes": 2,
    }
    if common_fields != {
        "fit_convergence_mode": convergence["mode"],
        "fit_max_iter": int(convergence["fit_max_iter"]),
        "fit_objective_tolerance": float(convergence["objective_tolerance"]),
        "fit_parameter_tolerance": float(convergence["parameter_tolerance"]),
        "fit_consecutive_convergence_passes": int(convergence["consecutive_convergence_passes"]),
    }:
        raise V4Error("frozen convergence fields changed")
    decision = {
        "schema_version": RUNNER_SCHEMA,
        "status": "complete_pass" if passed else "blocked_numerical_followup",
        "passed": passed,
        "eligible_spec_ids": expected_spec_ids,
        "scope_ids": [scope.scope_id for scope in context.scopes],
        "all_54_fits_valid": all_fits_valid,
        "all_18_grid801_start_gates_passed": all_starts_valid,
        "all_three_refit_bank_gates_passed": all_refits_passed,
        "all_three_fixed_bank_scoring_and_tail_gates_passed": (all_fixed_bank_passed),
        "fit_validity": fit_validity,
        "start_validity": start_validity,
        "refit_gate_passed": {
            spec_id: refit_gates.get(spec_id, {}).get("passed") is True
            for spec_id in expected_spec_ids
        },
        "fixed_bank_gate_passed": {
            spec_id: fixed_bank_gates.get(spec_id, {}).get("passed") is True
            for spec_id in expected_spec_ids
        },
        "effective_numerical_profile": (
            {
                "calibration_integration": [
                    cm.NORMAL_TRAPEZOID_QUADRATURE,
                    401,
                    8.0,
                ],
                "reference_eap_scoring": [
                    cm.NORMAL_TRAPEZOID_QUADRATURE,
                    801,
                    8.0,
                ],
            }
            if passed
            else None
        ),
        **common_fields,
        "maximum_inner_gradient": float(convergence["maximum_inner_gradient"]),
        "config_sha256": _sha256(context.config_path),
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "study_signature_sha256": context.study_signature["canonical_sha256"],
        "code_sha256_reverified_before_finalization": observed_code_hashes,
        "evidence_paths": evidence_paths,
        "evidence_sha256": evidence_hashes,
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "calibration_specification_selected": None,
        "calibration_model_selection_performed": False,
        "adaptive_results_inspected": False,
        "adaptive_runs": 0,
    }
    decision_path = context.out_dir / "numerical_followup_decision.json"
    _write_json_once(decision_path, decision)
    lock_path = context.out_dir / "numerical_followup_lock.json"
    lock_hash_path = context.out_dir / "numerical_followup_lock.sha256"
    if passed:
        lock = {
            "schema_version": LOCK_SCHEMA,
            "status": "complete_pass",
            "passed": True,
            "eligible_spec_ids": expected_spec_ids,
            "scope_ids": [scope.scope_id for scope in context.scopes],
            "calibration_integration": [
                cm.NORMAL_TRAPEZOID_QUADRATURE,
                401,
                8.0,
            ],
            "reference_eap_scoring": [
                cm.NORMAL_TRAPEZOID_QUADRATURE,
                801,
                8.0,
            ],
            **common_fields,
            "maximum_inner_gradient": float(convergence["maximum_inner_gradient"]),
            "config_sha256": _sha256(context.config_path),
            "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
            "study_signature_sha256": context.study_signature["canonical_sha256"],
            "followup_decision_sha256": _sha256(decision_path),
            "evidence_paths": evidence_paths,
            "evidence_sha256": evidence_hashes,
            "historical_fit_or_checkpoint_artifacts_reused": 0,
            "calibration_specification_selected": None,
            "calibration_model_selection_performed": False,
            "adaptive_results_inspected": False,
            "adaptive_runs": 0,
        }
        _write_json_once(lock_path, lock)
        _write_text_once(
            lock_hash_path,
            f"{_sha256(lock_path)}  numerical_followup_lock.json\n",
        )
    elif lock_path.exists() or lock_hash_path.exists():
        raise V4Error("failed V4 study must not retain a numerical lock")

    manifest_path = context.out_dir / "study_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "running":
        raise V4Error("terminal V4 manifest evidence cannot be replaced")
    manifest.update(
        {
            "status": decision["status"],
            "finished_at": _utcnow(),
            "decision_sha256": _sha256(decision_path),
            "lock_sha256": _sha256(lock_path) if lock_path.is_file() else None,
            "historical_fit_or_checkpoint_artifacts_reused": 0,
            "calibration_model_selection_performed": False,
            "adaptive_results_inspected": False,
            "adaptive_runs": 0,
        }
    )
    _atomic_json(manifest_path, manifest)
    return decision


def run(context: Context, *, resume: bool) -> dict[str, Any]:
    """Execute the exact 54-fit numerical schedule and no adaptive work."""

    _prepare(context, resume=resume)
    matrix, q_by, item_to_scenario = _runtime_objects(context)
    for spec in context.specs:
        for scope in context.scopes:
            fit401 = _fit_dir(context, spec, 401, scope, "cold") / "fit.npz"
            _run_fit_stage(
                context,
                spec,
                scope,
                matrix,
                q_by,
                nodes=401,
                start="cold",
                continuation_source=None,
                resume=resume,
            )
            _run_fit_stage(
                context,
                spec,
                scope,
                matrix,
                q_by,
                nodes=801,
                start="cold",
                continuation_source=None,
                resume=resume,
            )
            _run_fit_stage(
                context,
                spec,
                scope,
                matrix,
                q_by,
                nodes=801,
                start="continuation",
                continuation_source=fit401,
                resume=resume,
            )
            _run_start_selection(
                context,
                spec,
                scope,
                matrix,
                resume=resume,
            )

    refit_gates = {
        spec.spec_id: _run_fit_comparison(
            context,
            spec,
            matrix,
            item_to_scenario,
            resume=resume,
        )
        for spec in context.specs
    }
    all_refit_gates_passed = all(gate.get("passed") is True for gate in refit_gates.values())
    fixed_bank_gates: dict[str, Mapping[str, Any]] = {}
    if all_refit_gates_passed:
        fixed_bank_gates = {
            spec.spec_id: _run_fixed_bank_scoring(
                context,
                spec,
                matrix,
                item_to_scenario,
                resume=resume,
            )
            for spec in context.specs
        }
    return _finalize(context, refit_gates, fixed_bank_gates)


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
        context = load_context(args.config, args.out_dir)
        if args.plan_only:
            if args.resume:
                raise V4Error("--plan-only cannot combine with --resume")
            print(json.dumps(runtime_schedule(context), indent=2, sort_keys=True))
            return 0
        decision = run(context, resume=args.resume)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["passed"] else 2
    except (V4Error, cm.CalibrationError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
