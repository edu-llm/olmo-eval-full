"""Repeated, family-grouped nested CV for the frozen InFoBench V3 2PL study.

This is a new Phase-3 driver.  It deliberately does not share checkpoint or
cache schemas with :mod:`scripts.nested_scenario_cat_cv`, the historical v1
driver.  The command consumes a prospectively frozen V3 configuration and the
five-repeat split manifest, selects an *exact* calibration specification using
inner folds only, and evaluates the one frozen primary CAT policy plus two
non-promotable sensitivity policies on every outer-test model.

The Phase-3 candidates are exactly the two preregistered log-shrinkage 2PL
specifications, in simple order: lambda 16 followed by lambda 4.  Before that
subset is accepted, this driver revalidates the unchanged V4 provenance chain
for all three V4 specifications (1PL, lambda 16, and lambda 4).  Every fold fit
uses the numerical profile authorized by that V4 lock: normal-trapezoid
integration on [-8, 8] with 401 nodes and returned-iterate convergence. EAP
scoring uses the locked 801-node normal-trapezoid profile.

The driver is offline: it makes no tutor- or judge-model calls. ``--plan-only``
performs all input, split, decision, numerical-verification and provenance
checks without fitting. ``--precompute-panel REPEAT FOLD`` performs inner
selection for exactly one panel and can never open outer outcomes. ``--resume``
accepts only hash-identical artifacts from this versioned study.
Completed, failed, or otherwise non-resumable output is never deleted in place;
a new attempt must use a new versioned output leaf.

The official InFoBench run is intentionally not started by this module's tests.
Synthetic fixtures can exercise the orchestration by monkeypatching
``fit_or_load`` and ``evaluate_policy_panel``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import kfold_cv_mirt as cell_cv  # noqa: E402
from scripts import nested_scenario_cat_cv as v1  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402
from scripts import scenario_kfold_estimator_cv as scenario_cv  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "infobench_calibration_cat_2pl_only_v1.json"
DEFAULT_PARENT_CONFIG = ROOT / "configs" / "infobench_calibration_cat_v3.json"
DEFAULT_SPLITS = ROOT / "configs" / "infobench_v2_splits.manifest.json"
DEFAULT_OUTPUT = ROOT / "runs" / "calibration" / "InFoBench_2pl_only_v1" / "phase3"
DEFAULT_SYNTHETIC_OUTPUT = (
    ROOT / "reports" / "infobench_calibration_cat_2pl_only_v1" / "synthetic_preflight"
)
DEFAULT_NUMERICAL_FOLLOWUP_CONFIG = ROOT / "configs" / "infobench_v2_numerical_followup_v4.json"
DEFAULT_NUMERICAL_LOCK = (
    ROOT
    / "runs"
    / "calibration"
    / "InFoBench_v2_numerical_followup_v4"
    / "numerical_followup_lock.json"
)
DEFAULT_SKILLS = ("content", "format", "number", "style", "linguistic")
DEFAULT_DIMENSIONS = "instruction_following=content+format+number+style+linguistic"

CONFIG_SCHEMA = "infobench-calibration-cat-2pl-only-v1"
SPLIT_SCHEMA = "infobench-v2-repeated-splits-v1"
SCRIPT_SCHEMA = "infobench-nested-scenario-cat-cv-2pl-only-v1"
CACHE_SCHEMA = "infobench-2pl-only-v1-calibration-fit-cache-v1"
CHECKPOINT_SCHEMA = "infobench-2pl-only-v1-phase3-checkpoint-v1"
PANEL_CHECKPOINT_SCHEMA = "infobench-2pl-only-v1-panel-selection-checkpoint-v1"
PANEL_LOCK_SCHEMA = "infobench-2pl-only-v1-panel-owner-lock-v1"
DECISION_SCHEMA = "infobench-2pl-only-v1-phase3-decision-v1"
SELECTED_SPECS_SCHEMA = "infobench-2pl-only-v1-selected-calibration-specs-v1"
FOLLOWUP_CONFIG_SCHEMA = "infobench-v2-numerical-followup-v4"
FOLLOWUP_RUN_SCHEMA = "infobench-v2-numerical-followup-run-v4"
FOLLOWUP_LOCK_SCHEMA = "infobench-v2-numerical-followup-lock-v4"

EXPECTED_REPEATS = 5
EXPECTED_OUTER_FOLDS = 5
EXPECTED_INNER_FOLDS = 4
EXPECTED_MODELS = 52
EXPECTED_FAMILIES = 22
EXPECTED_PHASE3_SPEC_IDS = (
    "log_shrinkage_2pl_lambda16",
    "log_shrinkage_2pl_lambda4",
)
EXPECTED_V4_SPEC_IDS = (
    "1pl_fixed_a1",
    *EXPECTED_PHASE3_SPEC_IDS,
)
EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL = EXPECTED_INNER_FOLDS * len(EXPECTED_PHASE3_SPEC_IDS)
EXPECTED_PANELS = EXPECTED_REPEATS * EXPECTED_OUTER_FOLDS
EXPECTED_TOTAL_SPEC_FOLD_BLOCKS = EXPECTED_PANELS * EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL
PARENT_CONFIG_SHA256 = "3af71bd01ac40ad6fadc3c71bcb3170acdf2055e3176c28c290e09d72ecc90f4"
PARENT_ALLOWED_DIFF_ROOTS = (
    "schema_version",
    "frozen_date",
    "purpose",
    "parent_config_provenance",
    "code_dependencies",
    "calibration_specifications",
    "calibration_selection",
    "outputs",
)

PRIMARY_POLICY_ID = "primary_floor15_se0p20_trace"
SENSITIVITY_POLICY_IDS = (
    "sensitivity_floor12_se0p20_trace",
    "sensitivity_floor15_se0p20_dopt",
)

REQUIRED_OUTPUTS = (
    "fold_assignments.json",
    "pre_outer_selection_lock.json",
    "inner_calibration_model_results.csv",
    "calibration_model_choices.json",
    "selected_calibration_specs.json",
    "outer_oof_per_model.csv",
    "disjoint_prediction_metrics.csv",
    "repeat_fold_gate_results.csv",
    "repeat_metrics.csv",
    "calibration_spec_selection_frequency.csv",
    "duplicate_cat_paths.csv",
    "cross_repeat_per_model.csv",
    "cross_repeat_metrics.csv",
    "phase3_decision.json",
    "v3_decision.json",
    "manifest.json",
)

SELECTION_ONLY_REQUIRED_OUTPUTS = (
    "fold_assignments.json",
    "pre_outer_selection_lock.json",
    "inner_calibration_model_results.csv",
    "calibration_model_choices.json",
    "selected_calibration_specs.json",
    "phase3_decision.json",
    "v3_decision.json",
    "manifest.json",
)
TERMINAL_MANIFEST_OUTPUTS = {
    "phase3_complete": frozenset(REQUIRED_OUTPUTS) - {"manifest.json"},
    "phase3_terminal_selection_failed": frozenset(SELECTION_ONLY_REQUIRED_OUTPUTS)
    - {"manifest.json"},
}

CODE_DEPENDENCY_RELATIVE_PATHS = (
    "scripts/nested_scenario_cat_cv_2pl_only_v1.py",
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
CODE_DEPENDENCY_PATHS = tuple(ROOT / path for path in CODE_DEPENDENCY_RELATIVE_PATHS)

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


class V3Phase3Error(RuntimeError):
    """A frozen-design, leakage, provenance, or completeness invariant failed."""


@dataclass(frozen=True)
class CalibrationSpec:
    """An exact item-calibration candidate, including regularization."""

    spec_id: str
    family: str
    ridge: float | None
    log_a_shrinkage: float | None
    simplicity_rank: int
    canonical: dict[str, Any]

    def fit_namespace(self) -> str:
        return self.spec_id.replace("/", "_").replace(" ", "_")


@dataclass(frozen=True)
class Policy:
    policy_id: str
    role: str
    minimum_scenarios: int
    conditional_se_target: float
    selector: str

    def as_candidate(self) -> v1.Candidate:
        return v1.Candidate(
            minimum_scenarios=self.minimum_scenarios,
            conditional_se_target=self.conditional_se_target,
            selector=self.selector,
        )


@dataclass
class FitBundle:
    fit: dict[str, Any]
    bank: scat.FittedBank
    quadrature: scat.Quadrature
    cache_key: str
    cache_dir: Path
    training_model_ids: tuple[str, ...]
    exact_spec: CalibrationSpec
    bank_policy: dict[str, Any]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
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


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V3Phase3Error(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise V3Phase3Error(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_ready(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _panel_id(repeat: int, outer_fold: int) -> str:
    return f"repeat_{int(repeat):02d}_outer_{int(outer_fold):02d}"


def _panel_lock_path(output_dir: Path, repeat: int, outer_fold: int) -> Path:
    return output_dir / "two_pl_only_v1_panel_locks" / f"{_panel_id(repeat, outer_fold)}.json"


@contextmanager
def _exclusive_panel_lock(
    output_dir: Path,
    *,
    study_signature: str,
    repeat: int,
    outer_fold: int,
) -> Iterable[dict[str, Any]]:
    """Own one panel exclusively; existing locks are never removed as stale."""

    path = _panel_lock_path(output_dir, repeat, outer_fold)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner_token = uuid.uuid4().hex
    payload = {
        "schema_version": PANEL_LOCK_SCHEMA,
        "study_signature": study_signature,
        "panel_id": _panel_id(repeat, outer_fold),
        "repeat": int(repeat),
        "outer_fold": int(outer_fold),
        "owner_token": owner_token,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": _utcnow(),
        "automatic_stale_lock_removal_allowed": False,
    }
    encoded = (
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise V3Phase3Error(
            f"panel {_panel_id(repeat, outer_fold)} is already owned; "
            f"inspect without deleting {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        yield payload
    finally:
        if path.is_file():
            try:
                observed = _read_json(path)
            except V3Phase3Error:
                observed = {}
            if observed.get("owner_token") == owner_token:
                path.unlink()


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


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


def _environment_provenance() -> dict[str, Any]:
    versions = v1._dependency_versions()
    value = {
        **versions,
        "platform": platform.platform(),
        "executable": sys.executable,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    return {**value, "canonical_sha256": _canonical_hash(value)}


def _code_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for path in CODE_DEPENDENCY_PATHS:
        if not path.is_file():
            raise V3Phase3Error(f"required code dependency is missing: {path}")
        output[_display_path(path)] = _sha256(path)
    return output


def _number(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise V3Phase3Error(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise V3Phase3Error(f"{field} must be finite")
    return result


def _spec_id(family: str, ridge: float | None, shrinkage: float | None) -> str:
    if family == cm.ONE_PL:
        return "1pl_fixed_a1"
    if family == cm.LOG_SHRINKAGE_2PL:
        return f"log_shrinkage_2pl_lambda{float(shrinkage):g}"
    ridge_slug = f"{float(ridge):g}".replace(".", "p")
    return f"free_2pl_ridge_{ridge_slug}"


def _build_calibration_specs(
    raw: Sequence[tuple[str, float | None, float | None]],
) -> list[CalibrationSpec]:
    """Build canonical specifications while assigning study-local simple ranks."""

    specs: list[CalibrationSpec] = []
    for rank, (family, ridge, shrinkage) in enumerate(raw):
        canonical = cm.calibration_specification(
            family,
            ridge=0.0 if ridge is None else ridge,
            log_a_shrinkage=cm.DEFAULT_LOG_A_SHRINKAGE if shrinkage is None else shrinkage,
        )
        specs.append(
            CalibrationSpec(
                spec_id=_spec_id(family, ridge, shrinkage),
                family=family,
                ridge=ridge,
                log_a_shrinkage=shrinkage,
                simplicity_rank=rank,
                canonical=canonical,
            )
        )
    return specs


def expected_v4_calibration_specs() -> list[CalibrationSpec]:
    """Return all three specifications required by the unchanged V4 lock."""

    return _build_calibration_specs(
        [
            (cm.ONE_PL, None, None),
            (cm.LOG_SHRINKAGE_2PL, None, 16.0),
            (cm.LOG_SHRINKAGE_2PL, None, 4.0),
        ]
    )


def expected_calibration_specs() -> list[CalibrationSpec]:
    """Return the exact ordered 2PL-only Phase-3 candidate subset."""

    return _build_calibration_specs(
        [
            (cm.LOG_SHRINKAGE_2PL, None, 16.0),
            (cm.LOG_SHRINKAGE_2PL, None, 4.0),
        ]
    )


def _normalize_config_spec(raw: Mapping[str, Any], rank: int) -> CalibrationSpec:
    family = str(raw.get("family") or raw.get("calibration_model") or "")
    ridge_raw = raw.get("ridge")
    shrink_raw = raw.get("log_a_shrinkage")
    regularization = raw.get("regularization") or {}
    if ridge_raw is None:
        ridge_raw = regularization.get("ridge")
    if shrink_raw is None:
        shrink_raw = regularization.get("log_a_shrinkage")
    ridge = None if ridge_raw is None else _number(ridge_raw, field="ridge")
    shrinkage = None if shrink_raw is None else _number(shrink_raw, field="log_a_shrinkage")
    canonical = cm.calibration_specification(
        family,
        ridge=0.0 if ridge is None else ridge,
        log_a_shrinkage=(cm.DEFAULT_LOG_A_SHRINKAGE if shrinkage is None else shrinkage),
    )
    configured_cache_key = str(raw.get("canonical_cache_key") or "")
    if configured_cache_key and configured_cache_key != canonical["cache_key"]:
        raise V3Phase3Error(
            f"{raw.get('spec_id')}: configured calibration cache key does not reproduce"
        )
    if int(raw.get("simplicity_rank", rank)) != rank:
        raise V3Phase3Error(
            "calibration simplicity ranks must be ordered consecutive integers from zero"
        )
    spec = CalibrationSpec(
        spec_id=str(raw.get("spec_id") or raw.get("id") or _spec_id(family, ridge, shrinkage)),
        family=family,
        ridge=ridge if family == cm.FREE_2PL else None,
        log_a_shrinkage=shrinkage if family == cm.LOG_SHRINKAGE_2PL else None,
        simplicity_rank=rank,
        canonical=canonical,
    )
    return spec


def load_calibration_specs(config: Mapping[str, Any]) -> list[CalibrationSpec]:
    raw_specs = config.get("calibration_specifications")
    if isinstance(raw_specs, Mapping):
        raw_specs = raw_specs.get("candidates") or raw_specs.get("specifications")
    if not isinstance(raw_specs, list):
        raise V3Phase3Error("calibration_specifications must be a frozen ordered list")
    specs = [_normalize_config_spec(raw, rank) for rank, raw in enumerate(raw_specs)]
    expected = expected_calibration_specs()
    observed_signature = [(spec.family, spec.ridge, spec.log_a_shrinkage) for spec in specs]
    expected_signature = [(spec.family, spec.ridge, spec.log_a_shrinkage) for spec in expected]
    if observed_signature != expected_signature:
        raise V3Phase3Error(
            "Phase-3 calibration specifications/order must be exactly shrinkage "
            "lambda 16 followed by shrinkage lambda 4"
        )
    if [spec.spec_id for spec in specs] != [spec.spec_id for spec in expected]:
        raise V3Phase3Error("calibration specification IDs are not canonical")
    return specs


def _normalize_policy(raw: Mapping[str, Any], *, policy_id: str, role: str) -> Policy:
    policy = Policy(
        policy_id=str(raw.get("policy_id") or raw.get("id") or policy_id),
        role=role,
        minimum_scenarios=int(raw.get("minimum_scenarios")),
        conditional_se_target=_number(
            raw.get("conditional_se_target", raw.get("se_target")),
            field=f"{policy_id}.conditional_se_target",
        ),
        selector=str(raw.get("selector") or ""),
    )
    return policy


def load_policies(config: Mapping[str, Any]) -> list[Policy]:
    panel = config.get("cat_policies") or {}
    if not isinstance(panel, Mapping):
        raise V3Phase3Error("cat_policies must be an object")
    primary_raw = panel.get("primary")
    sensitivities_raw = panel.get("sensitivities")
    if not isinstance(primary_raw, Mapping) or not isinstance(sensitivities_raw, list):
        raise V3Phase3Error("cat_policies requires primary and two sensitivities")
    policies = [_normalize_policy(primary_raw, policy_id=PRIMARY_POLICY_ID, role="primary")]
    policies.extend(
        _normalize_policy(raw, policy_id=policy_id, role="sensitivity")
        for raw, policy_id in zip(sensitivities_raw, SENSITIVITY_POLICY_IDS, strict=False)
    )
    expected = [
        (PRIMARY_POLICY_ID, "primary", 15, 0.2, "trace"),
        (SENSITIVITY_POLICY_IDS[0], "sensitivity", 12, 0.2, "trace"),
        (SENSITIVITY_POLICY_IDS[1], "sensitivity", 15, 0.2, "dopt"),
    ]
    observed = [
        (
            item.policy_id,
            item.role,
            item.minimum_scenarios,
            item.conditional_se_target,
            item.selector,
        )
        for item in policies
    ]
    if len(sensitivities_raw) != 2 or observed != expected:
        raise V3Phase3Error(
            "CAT policies must exactly match the frozen primary floor15/SE.20/trace "
            "and floor12/trace plus floor15/D-opt sensitivities"
        )
    return policies


def audit_repeated_split_manifest(
    split: Mapping[str, Any],
    *,
    expected_models: Iterable[str] | None = None,
    expected_scenarios: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate all five frozen repeated family-grouped nested partitions."""

    if split.get("schema_version") != SPLIT_SCHEMA:
        raise V3Phase3Error(f"unexpected repeated split schema: {split.get('schema_version')}")
    model_to_family = {
        str(model): str(family) for model, family in (split.get("model_to_family") or {}).items()
    }
    all_models = set(model_to_family)
    all_families = set(model_to_family.values())
    if len(all_models) != EXPECTED_MODELS or len(all_families) != EXPECTED_FAMILIES:
        raise V3Phase3Error(
            f"split must contain {EXPECTED_MODELS} models in {EXPECTED_FAMILIES} families"
        )
    if expected_models is not None and set(map(str, expected_models)) != all_models:
        raise V3Phase3Error("response-matrix models do not match the frozen split")

    scenario = split.get("scenario_split") or {}
    administration = set(map(str, scenario.get("administration_scenario_ids") or []))
    evaluation = set(map(str, scenario.get("evaluation_scenario_ids") or []))
    if administration & evaluation or len(administration) != 400 or len(evaluation) != 100:
        raise V3Phase3Error("scenario split must be disjoint with exactly 400/100 scenarios")
    if expected_scenarios is not None and administration | evaluation != set(
        map(str, expected_scenarios)
    ):
        raise V3Phase3Error("scenario records do not match the frozen scenario split")

    repetitions = split.get("repetitions") or []
    if len(repetitions) != EXPECTED_REPEATS:
        raise V3Phase3Error(f"expected {EXPECTED_REPEATS} frozen repetitions")
    runtime_repeats: list[dict[str, Any]] = []
    partition_hashes: list[str] = []
    for expected_repeat, repetition in enumerate(repetitions):
        repeat = int(repetition.get("repeat", -1))
        if repeat != expected_repeat:
            raise V3Phase3Error("repetitions must be ordered and numbered 0..4")
        outer_folds = repetition.get("outer_folds") or []
        if len(outer_folds) != EXPECTED_OUTER_FOLDS:
            raise V3Phase3Error(f"repeat {repeat} must have five outer folds")
        seen_test: set[str] = set()
        model_to_outer: dict[str, int] = {}
        runtime_outer: list[dict[str, Any]] = []
        for expected_fold, outer in enumerate(outer_folds):
            fold = int(outer.get("outer_fold", -1))
            if fold != expected_fold:
                raise V3Phase3Error(f"repeat {repeat} outer folds must be numbered 0..4")
            train = set(map(str, outer.get("train_model_ids") or []))
            test = set(map(str, outer.get("test_model_ids") or []))
            if train & test or train | test != all_models:
                raise V3Phase3Error(f"repeat/fold {repeat}/{fold} does not partition models")
            if seen_test & test:
                raise V3Phase3Error(f"repeat {repeat} repeats an outer-test model")
            seen_test |= test
            for model in test:
                model_to_outer[model] = fold
            for family in all_families:
                members = {model for model, item in model_to_family.items() if item == family}
                if members & train and members & test:
                    raise V3Phase3Error(f"family {family} crosses repeat/fold {repeat}/{fold}")

            inner_folds = outer.get("inner_folds") or []
            if len(inner_folds) != EXPECTED_INNER_FOLDS:
                raise V3Phase3Error(f"repeat/fold {repeat}/{fold} needs four inner folds")
            inner_seen: list[str] = []
            runtime_inner: list[dict[str, Any]] = []
            for expected_inner, inner in enumerate(inner_folds):
                inner_fold = int(inner.get("inner_fold", -1))
                if inner_fold != expected_inner:
                    raise V3Phase3Error(
                        f"repeat/fold {repeat}/{fold} inner folds must be numbered 0..3"
                    )
                fit_ids = set(map(str, inner.get("fit_model_ids") or []))
                validation_ids = set(map(str, inner.get("validation_model_ids") or []))
                if test & (fit_ids | validation_ids):
                    raise V3Phase3Error(
                        f"outer-test leakage in repeat/fold/inner {repeat}/{fold}/{inner_fold}"
                    )
                if fit_ids & validation_ids or fit_ids | validation_ids != train:
                    raise V3Phase3Error(
                        f"inner {repeat}/{fold}/{inner_fold} does not partition outer train"
                    )
                for family in all_families:
                    members = {model for model, item in model_to_family.items() if item == family}
                    if members & fit_ids and members & validation_ids:
                        raise V3Phase3Error(
                            f"family {family} crosses inner {repeat}/{fold}/{inner_fold}"
                        )
                inner_seen.extend(sorted(validation_ids))
                runtime_inner.append(
                    {
                        "inner_fold": inner_fold,
                        "fit_model_ids": sorted(fit_ids),
                        "validation_model_ids": sorted(validation_ids),
                    }
                )
            if sorted(inner_seen) != sorted(train):
                raise V3Phase3Error(
                    f"each train model must validate once in repeat/fold {repeat}/{fold}"
                )
            runtime_outer.append(
                {
                    "outer_fold": fold,
                    "train_model_ids": sorted(train),
                    "test_model_ids": sorted(test),
                    "inner_folds": runtime_inner,
                }
            )
        if seen_test != all_models:
            raise V3Phase3Error(f"repeat {repeat} does not outer-test all models")
        assignment_hash = _canonical_hash(model_to_outer)
        partition_hashes.append(assignment_hash)
        runtime_repeats.append(
            {
                "repeat": repeat,
                "derived_repeat_seed": repetition.get("derived_repeat_seed"),
                "partition_sha256": assignment_hash,
                "outer_folds": runtime_outer,
                "model_to_outer_fold": model_to_outer,
            }
        )
    if len(set(partition_hashes)) != EXPECTED_REPEATS:
        raise V3Phase3Error("the five frozen repetitions are not distinct")
    return {
        "status": "passed",
        "repetitions": runtime_repeats,
        "model_to_family": model_to_family,
        "administration_scenario_ids": sorted(administration),
        "evaluation_scenario_ids": sorted(evaluation),
    }


def _family_cluster_se(
    rows: Sequence[Mapping[str, Any]], model_to_family: Mapping[str, str]
) -> float | None:
    """SE over family-mean losses; cells and sibling models are not independent."""

    model_losses: list[tuple[str, float]] = []
    for row in rows:
        if row.get("status") != "ok" or row.get("model_log_loss") is None:
            continue
        model_losses.append((model_to_family[str(row["model"])], float(row["model_log_loss"])))
    families = sorted({family for family, _loss in model_losses})
    if len(families) < 2:
        return None
    # Deterministic leave-one-family-out jackknife SE of the equal-model-weighted
    # mean.  This retains the configured point estimand while respecting sibling
    # dependence in its uncertainty.
    leave_one_out: list[float] = []
    for held_out in families:
        values = [loss for family, loss in model_losses if family != held_out]
        if not values:
            return None
        leave_one_out.append(float(np.mean(values)))
    estimates = np.asarray(leave_one_out, dtype=float)
    center = float(estimates.mean())
    count = len(families)
    return float(math.sqrt((count - 1) / count * np.sum((estimates - center) ** 2)))


def aggregate_calibration_evidence(
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[CalibrationSpec],
    model_to_family: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Aggregate inner evidence on exact common model/item/cell support."""

    output: list[dict[str, Any]] = []
    for spec in specs:
        subset = [row for row in rows if str(row.get("spec_id")) == spec.spec_id]
        valid = [row for row in subset if row.get("status") == "ok"]
        expected_models = {str(row.get("model")) for row in subset}
        valid_models = {str(row.get("model")) for row in valid}
        n_cells = sum(int(row.get("n_cells") or 0) for row in valid)
        losses = [float(row["model_log_loss"]) for row in valid]
        family_se = _family_cluster_se(valid, model_to_family)
        fit_eligible = all(bool(row.get("fit_eligible", True)) for row in subset)
        common_support = all(bool(row.get("common_support_verified")) for row in valid)
        eligible = (
            bool(subset)
            and valid_models == expected_models
            and n_cells > 0
            and family_se is not None
            and fit_eligible
            and common_support
        )
        output.append(
            {
                "spec_id": spec.spec_id,
                "family": spec.family,
                "ridge": spec.ridge,
                "log_a_shrinkage": spec.log_a_shrinkage,
                "simplicity_rank": spec.simplicity_rank,
                "calibration_specification": spec.canonical,
                "n_models": len(expected_models),
                "n_models_scored": len(valid_models),
                "n_families_scored": len({model_to_family[model] for model in valid_models}),
                "n_cells": n_cells,
                "mean_log_loss": float(np.mean(losses)) if losses else None,
                "pooled_log_loss": (
                    sum(float(row["log_loss_sum"]) for row in valid) / n_cells if n_cells else None
                ),
                "family_cluster_se": family_se,
                "coverage": len(valid_models) / len(expected_models) if expected_models else 0.0,
                "common_support_verified": common_support,
                "fit_eligible": fit_eligible,
                "eligible": eligible,
                "fit_cache_keys": sorted({str(row.get("fit_cache_key")) for row in subset}),
            }
        )
    return output


def audit_complete_inner_evidence(
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[CalibrationSpec],
    expected_validation_by_fold: Mapping[int, Sequence[str]],
) -> dict[str, Any]:
    """Fail-closed audit of the complete four-fold by two-spec evidence panel."""

    expected_folds = sorted(map(int, expected_validation_by_fold))
    expected_spec_ids = [spec.spec_id for spec in specs]
    expected_keys = {
        (inner_fold, spec_id) for inner_fold in expected_folds for spec_id in expected_spec_ids
    }
    failures: list[str] = []
    if len(expected_folds) != EXPECTED_INNER_FOLDS:
        failures.append(f"expected_{EXPECTED_INNER_FOLDS}_inner_folds_got_{len(expected_folds)}")
    if expected_spec_ids != list(EXPECTED_PHASE3_SPEC_IDS):
        failures.append("expected_exact_ordered_two_spec_subset")

    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        try:
            key = (int(row.get("inner_fold")), str(row.get("spec_id")))
        except (TypeError, ValueError):
            failures.append(f"row_{row_index}_invalid_inner_fold_or_spec")
            continue
        if key not in expected_keys:
            failures.append(f"unexpected_spec_fold_{key[0]}_{key[1]}")
        grouped[key].append(row)

    def _positive_integer(row: Mapping[str, Any], field: str) -> bool:
        try:
            return int(row.get(field)) > 0
        except (TypeError, ValueError):
            return False

    def _valid_hash(value: Any) -> bool:
        text = str(value or "")
        return len(text) == 64 and all(character in "0123456789abcdef" for character in text)

    def _finite_number(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    combination_audits: list[dict[str, Any]] = []
    valid_combinations = 0
    for inner_fold, spec_id in sorted(expected_keys):
        subset = grouped.get((inner_fold, spec_id), [])
        expected_models = set(map(str, expected_validation_by_fold.get(inner_fold, ())))
        observed_models = [str(row.get("model")) for row in subset]
        checks = {
            "rows_present": bool(subset),
            "validation_model_coverage_exact": (
                len(observed_models) == len(expected_models)
                and len(set(observed_models)) == len(observed_models)
                and set(observed_models) == expected_models
            ),
            "status_ok": bool(subset) and all(row.get("status") == "ok" for row in subset),
            "fit_eligible": bool(subset) and all(row.get("fit_eligible") is True for row in subset),
            "all_specs_fitted": bool(subset)
            and all(row.get("all_specs_fitted") is True for row in subset),
            "common_support_verified": bool(subset)
            and all(row.get("common_support_verified") is True for row in subset),
            "fit_cache_key_present": bool(subset)
            and all(bool(str(row.get("fit_cache_key") or "")) for row in subset),
            "common_support_hash_present": bool(subset)
            and all(_valid_hash(row.get("common_support_sha256")) for row in subset),
            "positive_administration_support": bool(subset)
            and all(_positive_integer(row, "n_common_admin_items") for row in subset),
            "positive_evaluation_support": bool(subset)
            and all(_positive_integer(row, "n_common_evaluation_items") for row in subset),
            "positive_observed_cells": bool(subset)
            and all(_positive_integer(row, "n_cells") for row in subset),
            "finite_model_log_loss": bool(subset)
            and all(_finite_number(row.get("model_log_loss")) for row in subset),
            "observed_cell_hash_present": bool(subset)
            and all(_valid_hash(row.get("observed_common_cells_sha256")) for row in subset),
        }
        passed = all(checks.values())
        if passed:
            valid_combinations += 1
        else:
            failures.extend(
                f"inner_{inner_fold}_{spec_id}_{name}"
                for name, value in checks.items()
                if not value
            )
        combination_audits.append(
            {
                "inner_fold": inner_fold,
                "spec_id": spec_id,
                "expected_validation_models": sorted(expected_models),
                "observed_validation_models": observed_models,
                "checks": checks,
                "passed": passed,
            }
        )

    for inner_fold in expected_folds:
        fold_rows = [
            row
            for row in rows
            if str(row.get("spec_id")) in expected_spec_ids
            and str(row.get("inner_fold")) == str(inner_fold)
        ]
        support_hashes = {
            str(row.get("common_support_sha256"))
            for row in fold_rows
            if _valid_hash(row.get("common_support_sha256"))
        }
        if len(support_hashes) != 1:
            failures.append(f"inner_{inner_fold}_common_support_hash_not_identical")
        for model in sorted(set(map(str, expected_validation_by_fold.get(inner_fold, ())))):
            cell_hashes = {
                str(row.get("observed_common_cells_sha256"))
                for row in fold_rows
                if str(row.get("model")) == model
                and _valid_hash(row.get("observed_common_cells_sha256"))
            }
            if len(cell_hashes) != 1:
                failures.append(f"inner_{inner_fold}_{model}_observed_cell_support_not_identical")

    failures = sorted(set(failures))
    return {
        "passed": (
            not failures
            and len(expected_keys) == EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL
            and valid_combinations == EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL
        ),
        "require_all_specs_every_inner_fold": True,
        "survivor_selection_allowed": False,
        "expected_inner_folds": expected_folds,
        "expected_spec_ids": expected_spec_ids,
        "expected_spec_fold_combinations": len(expected_keys),
        "valid_spec_fold_combinations": valid_combinations,
        "validation_models_by_inner_fold": {
            str(inner_fold): sorted(set(map(str, expected_validation_by_fold.get(inner_fold, ()))))
            for inner_fold in expected_folds
        },
        "failed_checks": failures,
        "combination_audits": combination_audits,
    }


def select_calibration_spec_one_se(
    evidence: Sequence[Mapping[str, Any]], specs: Sequence[CalibrationSpec]
) -> tuple[CalibrationSpec, list[dict[str, Any]], dict[str, Any]]:
    """Use only inner family-clustered loss and the frozen simplicity order."""

    rows = [dict(row) for row in evidence]
    simplicity = [spec.canonical["cache_key"] for spec in specs]
    try:
        selected = cell_cv.select_calibration_spec_one_se(rows, simplicity)
    except ValueError as error:
        raise V3Phase3Error(str(error)) from error
    selected_key = str(selected["selected_cache_key"])
    by_key = {spec.canonical["cache_key"]: spec for spec in specs}
    if selected_key not in by_key:
        raise V3Phase3Error("one-SE helper returned a non-frozen calibration spec")
    within = set(map(str, selected["within_one_se_cache_keys"]))
    for row in rows:
        key = str((row.get("calibration_specification") or {}).get("cache_key"))
        row["empirical_best"] = key == selected["empirical_best_cache_key"]
        row["one_se_cutoff"] = float(selected["one_se_cutoff"])
        row["within_one_se"] = key in within
        row["selected"] = key == selected_key
        row["selection_used_outer_outcomes"] = False
    return by_key[selected_key], rows, dict(selected)


def _bootstrap_family_indices(
    rows: Sequence[Mapping[str, Any]],
    model_to_family: Mapping[str, str],
    *,
    rng: np.random.Generator,
) -> list[int]:
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_family[model_to_family[str(row["model"])]].append(index)
    families = sorted(by_family)
    sampled = rng.integers(0, len(families), size=len(families))
    return [index for choice in sampled for index in by_family[families[int(choice)]]]


def _clustered_metric_intervals(
    rows: Sequence[Mapping[str, Any]],
    model_to_family: Mapping[str, str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | None]:
    """Family-cluster bootstrap for recovery correlation and CAT reduction."""

    valid = [
        row
        for row in rows
        if row.get("cat_replay_success")
        and row.get("baseline_replay_success")
        and row.get("cat_mwle_converged")
        and row.get("theta_reference") is not None
        and row.get("theta_cat_mwle") is not None
        and row.get("cat_scenarios_administered") is not None
        and row.get("baseline_scenarios_administered") is not None
    ]
    if not valid:
        return {
            "recovery_correlation_lower_95_ci": None,
            "scenario_reduction_lower_95_ci": None,
            "scenario_reduction_upper_95_ci": None,
        }
    rng = np.random.default_rng(seed)
    correlations: list[float] = []
    reductions: list[float] = []
    for _ in range(replicates):
        indices = _bootstrap_family_indices(valid, model_to_family, rng=rng)
        sampled = [valid[index] for index in indices]
        x = np.asarray([float(row["theta_reference"]) for row in sampled])
        y = np.asarray([float(row["theta_cat_mwle"]) for row in sampled])
        if x.size >= 3 and float(x.std()) > 0 and float(y.std()) > 0:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
        cat = np.asarray([float(row["cat_scenarios_administered"]) for row in sampled])
        baseline = np.asarray([float(row["baseline_scenarios_administered"]) for row in sampled])
        if baseline.size and float(baseline.mean()) > 0:
            reductions.append(1.0 - float(cat.mean()) / float(baseline.mean()))

    def percentile(values: Sequence[float], q: float) -> float | None:
        return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None

    return {
        "recovery_correlation_lower_95_ci": percentile(correlations, 2.5),
        "scenario_reduction_lower_95_ci": percentile(reductions, 2.5),
        "scenario_reduction_upper_95_ci": percentile(reductions, 97.5),
    }


def aggregate_policy_rows_clustered(
    rows: Sequence[Mapping[str, Any]],
    model_to_family: Mapping[str, str],
    *,
    seed: int,
    replicates: int,
    label: str,
) -> dict[str, Any]:
    """V1 point metrics plus family-clustered, rather than person-row, intervals."""

    metrics = v1.aggregate_candidate_rows(
        rows, seed=seed, bootstrap_replicates=replicates, label=label
    )
    clustered = _clustered_metric_intervals(
        rows,
        model_to_family,
        seed=v1._bootstrap_seed(seed, f"v3-family:{label}"),
        replicates=replicates,
    )
    metrics.update(clustered)
    metrics["paired_ci_favors_cat"] = (
        clustered["scenario_reduction_lower_95_ci"] is not None
        and float(clustered["scenario_reduction_lower_95_ci"]) > 0
    )
    metrics["bootstrap_unit"] = "tutor_family"
    metrics["n_families"] = len({model_to_family[str(row["model"])] for row in rows})
    return metrics


def apply_scoped_absolute_gates(
    metrics: Mapping[str, Any], gates: Mapping[str, Any], *, scope: str
) -> dict[str, Any]:
    """Apply only gates authorized for an inferential scope."""

    _all_passed, _all_failures, all_checks = v1.apply_absolute_gates(metrics, gates)
    if scope == "outer_panel":
        applied_names = PANEL_APPLIED_GATE_NAMES
        diagnostic_names = PANEL_DIAGNOSTIC_GATE_NAMES
    elif scope == "repetition":
        applied_names = REPETITION_APPLIED_GATE_NAMES
        diagnostic_names = ()
    else:
        raise V3Phase3Error(f"unsupported gate scope: {scope}")
    if set(all_checks) != set(ABSOLUTE_GATE_NAMES):
        raise V3Phase3Error("absolute-gate helper returned an unexpected gate inventory")
    applied_checks = {name: bool(all_checks[name]) for name in applied_names}
    diagnostic_checks = {name: bool(all_checks[name]) for name in diagnostic_names}
    failures = [name for name, passed in applied_checks.items() if not passed]
    diagnostic_failures = [name for name, passed in diagnostic_checks.items() if not passed]
    return {
        "passed": not failures,
        "failures": failures,
        "diagnostic_failures": diagnostic_failures,
        "applied_gate_names": list(applied_names),
        "diagnostic_gate_names": list(diagnostic_names),
        "applied_checks": applied_checks,
        "diagnostic_checks": diagnostic_checks,
    }


def detect_duplicate_cat_paths(
    rows: Sequence[Mapping[str, Any]], policies: Sequence[Policy]
) -> list[dict[str, Any]]:
    """Flag nominal policies whose exact CAT paths coincide within a panel."""

    by_policy_model = {
        (str(row.get("policy_id")), str(row.get("model"))): str(
            row.get("cat_scenario_order") or "[]"
        )
        for row in rows
    }
    models = sorted({str(row.get("model")) for row in rows})
    output: list[dict[str, Any]] = []
    for left_index, left in enumerate(policies):
        for right in policies[left_index + 1 :]:
            common = [
                model
                for model in models
                if (left.policy_id, model) in by_policy_model
                and (right.policy_id, model) in by_policy_model
            ]
            same = sum(
                by_policy_model[(left.policy_id, model)]
                == by_policy_model[(right.policy_id, model)]
                for model in common
            )
            output.append(
                {
                    "left_policy_id": left.policy_id,
                    "right_policy_id": right.policy_id,
                    "n_models_compared": len(common),
                    "n_identical_paths": same,
                    "identical_path_fraction": same / len(common) if common else None,
                    "all_paths_identical": bool(common) and same == len(common),
                    "counted_as_independent_support": False,
                }
            )
    return output


def _input_entry(config: Mapping[str, Any], name: str) -> tuple[Path, str]:
    baseline = config.get("baseline") or {}
    raw = baseline.get(name)
    if isinstance(raw, Mapping):
        path_value = raw.get("path") or raw.get("configured_path")
        expected_hash = str(raw.get("sha256") or "")
    else:
        path_value = raw
        expected_hash = str(baseline.get(f"{name}_sha256") or "")
    if not path_value or not expected_hash:
        raise V3Phase3Error(f"baseline.{name} requires a path and sha256")
    return _resolve_repo_path(str(path_value)), expected_hash


def _resolve_phase3_output_dir(
    config: Mapping[str, Any], requested: Path | None, *, plan_only: bool
) -> Path:
    """Resolve the output leaf and forbid unofficial destinations for real runs."""

    configured_value = (config.get("outputs") or {}).get("phase3")
    if not configured_value:
        raise V3Phase3Error("outputs.phase3 is required")
    configured = _resolve_repo_path(str(configured_value))
    if requested is None:
        resolved = configured
    else:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve()
    if not plan_only and resolved != configured:
        raise V3Phase3Error(
            "official 2PL-only Phase-3 output must equal the frozen "
            f"outputs.phase3 leaf: {configured}"
        )
    return resolved


def _config_diff_paths(parent: Any, child: Any, prefix: str = "") -> list[str]:
    """Return deterministic leaf-level differences for provenance and allowlisting."""

    if isinstance(parent, Mapping) and isinstance(child, Mapping):
        output: list[str] = []
        for key in sorted(set(map(str, parent)) | set(map(str, child))):
            path = f"{prefix}.{key}" if prefix else key
            if key not in parent or key not in child:
                output.append(path)
            else:
                output.extend(_config_diff_paths(parent[key], child[key], path))
        return output
    if isinstance(parent, list) and isinstance(child, list):
        if parent == child:
            return []
        output = []
        for index in range(max(len(parent), len(child))):
            path = f"{prefix}[{index}]"
            if index >= len(parent) or index >= len(child):
                output.append(path)
            else:
                output.extend(_config_diff_paths(parent[index], child[index], path))
        return output or [prefix]
    return [] if parent == child else [prefix]


def _validate_parent_config_provenance(
    config: Mapping[str, Any], *, parent_path: Path = DEFAULT_PARENT_CONFIG
) -> dict[str, Any]:
    """Require every non-allowlisted field to equal the hash-frozen parent."""

    resolved_parent = parent_path.resolve()
    if not resolved_parent.is_file() or _sha256(resolved_parent) != PARENT_CONFIG_SHA256:
        raise V3Phase3Error("the hash-frozen parent V3 config is missing or changed")
    parent = _read_json(resolved_parent)
    declared = config.get("parent_config_provenance") or {}
    expected_declared_path = _display_path(resolved_parent)
    if (
        not isinstance(declared, Mapping)
        or declared.get("path") != expected_declared_path
        or declared.get("sha256") != PARENT_CONFIG_SHA256
        or tuple(map(str, declared.get("allowed_diff_roots") or ())) != PARENT_ALLOWED_DIFF_ROOTS
        or declared.get("all_other_fields_must_match_parent_exactly") is not True
    ):
        raise V3Phase3Error("parent-config provenance or diff allowlist changed")

    diff_paths = _config_diff_paths(parent, config)
    disallowed = sorted(
        path
        for path in diff_paths
        if path.split(".", 1)[0].split("[", 1)[0] not in PARENT_ALLOWED_DIFF_ROOTS
    )
    if disallowed:
        raise V3Phase3Error(
            "2PL-only config changed non-allowlisted parent fields: " + ", ".join(disallowed)
        )

    parent_dependencies = list(map(str, parent.get("code_dependencies") or ()))
    expected_dependencies = list(parent_dependencies)
    if not expected_dependencies:
        raise V3Phase3Error("parent code dependency inventory is empty")
    expected_dependencies[0] = "scripts/nested_scenario_cat_cv_2pl_only_v1.py"
    if list(map(str, config.get("code_dependencies") or ())) != expected_dependencies:
        raise V3Phase3Error("2PL-only code dependencies differ beyond the versioned driver")

    parent_selection = parent.get("calibration_selection") or {}
    selection = config.get("calibration_selection") or {}
    if any(selection.get(key) != value for key, value in parent_selection.items()):
        raise V3Phase3Error("frozen parent calibration-selection behavior changed")
    expected_selection_additions = {
        "candidate_subset_policy": (
            "exact ordered subset of the fully validated V4 panel: "
            "log_shrinkage_2pl_lambda16, then log_shrinkage_2pl_lambda4"
        ),
        "excluded_parent_candidate": "1pl_fixed_a1",
        "excluded_parent_candidate_role": (
            "retained in and required for unchanged V4 provenance, but not eligible "
            "for Phase-3 selection"
        ),
    }
    if set(selection) != set(parent_selection) | set(expected_selection_additions) or any(
        selection.get(key) != value for key, value in expected_selection_additions.items()
    ):
        raise V3Phase3Error("2PL-only calibration-selection provenance fields changed")

    parent_outputs = parent.get("outputs") or {}
    outputs = config.get("outputs") or {}
    expected_outputs = {
        **parent_outputs,
        "phase3": "runs/calibration/InFoBench_2pl_only_v1/phase3",
        "phase4": "runs/calibration/InFoBench_2pl_only_v1/phase4",
        "final_fit": "runs/calibration/InFoBench_2pl_only_v1/final_fit",
        "report": "reports/infobench_calibration_cat_2pl_only_v1",
        "never_read_or_write_parent_v3_artifacts": True,
    }
    if outputs != expected_outputs:
        raise V3Phase3Error("2PL-only versioned output contract changed")

    return {
        "path": expected_declared_path,
        "sha256": PARENT_CONFIG_SHA256,
        "allowed_diff_roots": list(PARENT_ALLOWED_DIFF_ROOTS),
        "observed_diff_paths": diff_paths,
        "observed_diff_paths_sha256": _canonical_hash(diff_paths),
        "all_non_allowlisted_fields_equal": True,
    }


def _check_frozen_config(config: Mapping[str, Any]) -> dict[str, Any]:
    parent_provenance = _validate_parent_config_provenance(config)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise V3Phase3Error(f"unexpected 2PL-only config schema: {config.get('schema_version')}")
    status = str(config.get("status") or "")
    if status not in {
        "preregistered",
        "frozen",
        "frozen_pre_run",
    }:
        raise V3Phase3Error("2PL-only config must be prospectively frozen, not draft/post-hoc")
    if tuple(map(str, config.get("code_dependencies") or ())) != (CODE_DEPENDENCY_RELATIVE_PATHS):
        raise V3Phase3Error("V3 code dependency inventory changed or is incomplete")
    cv = config.get("cross_validation") or {}
    if (
        int(cv.get("repetitions", 0)) != EXPECTED_REPEATS
        or int(cv.get("outer_folds", cv.get("outer_folds_per_repetition", 0)))
        != EXPECTED_OUTER_FOLDS
        or int(cv.get("inner_folds", cv.get("inner_folds_per_outer_panel", 0)))
        != EXPECTED_INNER_FOLDS
    ):
        raise V3Phase3Error("V3 config must freeze 5 repeats x 5 outer x 4 inner folds")
    if cv.get("group_related_model_families", True) is not True:
        raise V3Phase3Error("family-grouped cross-validation cannot be disabled")
    if (
        cv.get("aggregate_repeated_predictions_per_model") is not True
        or cv.get("family_cluster_bootstrap") is not True
        or cv.get("treat_model_repeat_rows_as_independent") is not False
    ):
        raise V3Phase3Error("repeated-CV aggregation must remain per-model/family-clustered")
    selection = config.get("calibration_selection") or {}
    if selection.get("outer_outcomes_used", False) is not False:
        raise V3Phase3Error("outer outcomes may not select calibration specifications")
    if selection.get("cat_outcomes_used", False) is not False:
        raise V3Phase3Error("CAT outcomes may not select calibration specifications")
    if selection.get("allow_fallback", False) is not False:
        raise V3Phase3Error("calibration specification fallback is forbidden")
    if selection.get("require_all_specs_every_inner_fold") is not True:
        raise V3Phase3Error("every exact specification must succeed in every inner fold")
    if selection.get("survivor_selection_allowed") is not False:
        raise V3Phase3Error("survivor-only calibration selection is forbidden")
    rule = str(selection.get("rule") or selection.get("method") or "")
    if rule and "one" not in rule.lower():
        raise V3Phase3Error("calibration selection must use the frozen one-SE rule")
    if selection.get("common_item_and_response_cell_support_required") is not True:
        raise V3Phase3Error("common held-out item/cell support must remain required")
    if selection.get("uncertainty_unit") != "tutor_model_family":
        raise V3Phase3Error("calibration uncertainty must be clustered by tutor family")
    if selection.get("uncertainty_method") != "deterministic_leave_one_family_out_jackknife_se":
        raise V3Phase3Error("calibration selection must retain family jackknife SE")
    if not math.isclose(
        float(selection.get("minimum_exact_spec_modal_fraction_across_outer_panels", -1)),
        0.80,
        rel_tol=0,
        abs_tol=0,
    ):
        raise V3Phase3Error("exact-spec modal stability threshold must remain 80%")
    gates = config.get("selection_gates") or {}
    if (
        gates.get("allow_fallback_if_none_pass", False) is not False
        or gates.get("allow_fallback_if_primary_fails", False) is not False
    ):
        raise V3Phase3Error("fallback after a failed primary policy is forbidden")
    if gates.get("apply_to_primary_only") is not True:
        raise V3Phase3Error("scientific gates must apply to the primary policy only")
    if (
        gates.get("require_every_outer_panel") is not True
        or gates.get("require_every_repetition_pooled_gate") is not True
    ):
        raise V3Phase3Error("the primary must pass every panel and every repetition")
    if not math.isclose(
        _number(
            gates.get("minimum_nominal_precision_lower_95_ci"),
            field="minimum_nominal_precision_lower_95_ci",
        ),
        0.90,
        rel_tol=0,
        abs_tol=0,
    ):
        raise V3Phase3Error("the pooled nominal-precision lower-CI gate must remain 0.90")
    if tuple(map(str, gates.get("outer_panel_applied_gate_names") or ())) != (
        PANEL_APPLIED_GATE_NAMES
    ):
        raise V3Phase3Error("outer-panel applied gates changed")
    if tuple(map(str, gates.get("outer_panel_diagnostic_gate_names") or ())) != (
        PANEL_DIAGNOSTIC_GATE_NAMES
    ):
        raise V3Phase3Error("outer-panel diagnostic CI gates changed")
    if tuple(map(str, gates.get("repetition_applied_gate_names") or ())) != (
        REPETITION_APPLIED_GATE_NAMES
    ):
        raise V3Phase3Error("pooled-repetition applied gates changed")
    if (
        gates.get("outer_panel_inference_unit")
        != "unique_outer_test_tutor_within_panel; confidence intervals diagnostic only"
        or gates.get("repetition_inference_unit")
        != "52_unique_out_of_fold_tutors_once_per_repetition"
        or int(gates.get("repetition_expected_unique_oof_tutors", 0)) != EXPECTED_MODELS
        or gates.get("cross_repeat_gate_role") != "diagnostic_only_after_per_model_aggregation"
        or gates.get("treat_260_model_repeat_rows_as_independent") is not False
    ):
        raise V3Phase3Error("scope-specific gate inference units changed")
    if not math.isclose(
        _number(
            gates.get("minimum_scenario_reduction_vs_random"),
            field="minimum_scenario_reduction_vs_random",
        ),
        0.50,
        rel_tol=0,
        abs_tol=0,
    ):
        raise V3Phase3Error("the frozen 50% scenario-reduction gate changed")
    policies = config.get("cat_policies") or {}
    if policies.get("sensitivities_can_replace_primary") is not False:
        raise V3Phase3Error("sensitivity policies must remain non-promotable")
    numerical = config.get("numerical_lock") or {}
    required_v4_ids = [spec.spec_id for spec in expected_v4_calibration_specs()]
    if numerical.get("eligible_spec_ids") != required_v4_ids:
        raise V3Phase3Error("config must retain exactly the full three-spec V4 lock panel")
    if [spec.spec_id for spec in load_calibration_specs(config)] != list(EXPECTED_PHASE3_SPEC_IDS):
        raise V3Phase3Error("Phase-3 candidate subset or order changed")
    expected_strings = {
        "fit_quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
        "convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
        "eap_quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
    }
    for field, expected in expected_strings.items():
        if numerical.get(field) != expected:
            raise V3Phase3Error(f"V3 numerical_lock.{field} changed")
    for field, expected in (
        ("fit_grid", 401),
        ("fit_linear_bound", 8.0),
        ("objective_tolerance", 1e-4),
        ("parameter_tolerance", 5e-5),
        ("consecutive_convergence_passes", 2),
        ("eap_grid", 801),
        ("eap_linear_bound", 8.0),
        ("tail_region", 7.5),
    ):
        _exact_number(numerical.get(field), expected, field=f"numerical_lock.{field}")
    runtime = config.get("runtime") or {}
    expected_runtime = {
        "fit_max_iter": 1500,
        "fit_tolerance": 1e-4,
        "fit_parameter_tolerance": 5e-5,
        "fit_consecutive_convergence_passes": 2,
        "fit_convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
        "fit_quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
        "fit_linear_bound": 8.0,
    }
    for field, expected in expected_runtime.items():
        value = runtime.get(field)
        if isinstance(expected, str):
            if value != expected:
                raise V3Phase3Error(f"V3 runtime.{field} changed")
        else:
            _exact_number(value, expected, field=f"runtime.{field}")
    outputs = config.get("outputs") or {}
    if outputs.get("phase3") != "runs/calibration/InFoBench_2pl_only_v1/phase3":
        raise V3Phase3Error("2PL-only Phase-3 output leaf changed")
    if outputs.get("never_overwrite_v1_or_v2") is not True:
        raise V3Phase3Error("2PL-only study must preserve every V1/V2 artifact")
    if outputs.get("never_read_or_write_parent_v3_artifacts") is not True:
        raise V3Phase3Error("2PL-only study must isolate every parent V3 artifact")
    return parent_provenance


def _sha_companion_matches(path: Path) -> bool:
    companion = path.with_suffix(".sha256")
    if not companion.is_file():
        return False
    parts = companion.read_text(encoding="utf-8").strip().split()
    return (
        len(parts) == 2
        and parts[1] == path.name
        and len(parts[0]) == 64
        and all(character in "0123456789abcdef" for character in parts[0])
        and path.is_file()
        and parts[0] == _sha256(path)
    )


def _eligible_spec_ids(specs: Sequence[CalibrationSpec]) -> list[str]:
    return [spec.spec_id for spec in specs]


def _exact_number(value: Any, expected: float, *, field: str) -> None:
    if not math.isclose(_number(value, field=field), expected, rel_tol=0, abs_tol=0):
        raise V3Phase3Error(f"{field} must remain exactly {expected:g}")


def _extract_v4_profile(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the V4 lock's fit/EAP profile without weakening its contract."""

    profile = raw.get("effective_global_profile")
    profile = profile if isinstance(profile, Mapping) else {}
    fit = profile.get("fit") or profile.get("calibration") or {}
    eap = profile.get("eap") or profile.get("scoring") or {}
    if not isinstance(fit, Mapping) or not isinstance(eap, Mapping):
        raise V3Phase3Error("V4 effective profile must contain fit and EAP mappings")
    calibration_integration = raw.get("calibration_integration")
    reference_eap = raw.get("reference_eap_scoring")
    if calibration_integration is not None and calibration_integration != [
        cm.NORMAL_TRAPEZOID_QUADRATURE,
        401,
        8.0,
    ]:
        raise V3Phase3Error("V4 calibration_integration lock changed")
    if reference_eap is not None and reference_eap != [
        cm.NORMAL_TRAPEZOID_QUADRATURE,
        801,
        8.0,
    ]:
        raise V3Phase3Error("V4 reference_eap_scoring lock changed")

    normalized = {
        "fit_grid": profile.get(
            "fit_grid",
            fit.get(
                "grid",
                fit.get(
                    "nodes",
                    raw.get(
                        "common_fit_grid",
                        calibration_integration[1] if calibration_integration is not None else None,
                    ),
                ),
            ),
        ),
        "fit_quadrature_method": profile.get(
            "fit_quadrature_method",
            fit.get(
                "quadrature_method",
                raw.get(
                    "fit_quadrature_method",
                    calibration_integration[0] if calibration_integration is not None else None,
                ),
            ),
        ),
        "fit_linear_bound": profile.get(
            "fit_linear_bound",
            fit.get(
                "linear_bound",
                raw.get(
                    "fit_linear_bound",
                    calibration_integration[2] if calibration_integration is not None else None,
                ),
            ),
        ),
        "fit_convergence_mode": profile.get(
            "fit_convergence_mode",
            fit.get("convergence_mode", raw.get("fit_convergence_mode")),
        ),
        "fit_max_iter": profile.get("fit_max_iter", fit.get("max_iter", raw.get("fit_max_iter"))),
        "fit_objective_tolerance": profile.get(
            "fit_objective_tolerance",
            fit.get("objective_tolerance", raw.get("fit_objective_tolerance")),
        ),
        "fit_parameter_tolerance": profile.get(
            "fit_parameter_tolerance",
            fit.get("parameter_tolerance", raw.get("fit_parameter_tolerance")),
        ),
        "fit_consecutive_convergence_passes": profile.get(
            "fit_consecutive_convergence_passes",
            fit.get(
                "consecutive_convergence_passes",
                raw.get("fit_consecutive_convergence_passes"),
            ),
        ),
        "eap_grid": profile.get(
            "eap_grid",
            eap.get(
                "grid",
                eap.get(
                    "nodes",
                    raw.get(
                        "eap_grid",
                        reference_eap[1] if reference_eap is not None else None,
                    ),
                ),
            ),
        ),
        "quadrature_method": profile.get(
            "quadrature_method",
            eap.get(
                "quadrature_method",
                raw.get(
                    "quadrature_method",
                    reference_eap[0] if reference_eap is not None else None,
                ),
            ),
        ),
        "linear_bound": profile.get(
            "linear_bound",
            eap.get(
                "linear_bound",
                raw.get(
                    "linear_bound",
                    reference_eap[2] if reference_eap is not None else None,
                ),
            ),
        ),
        "tail_region": profile.get(
            "tail_region", eap.get("tail_region", raw.get("tail_region", 7.5))
        ),
    }
    expected_strings = {
        "fit_quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
        "fit_convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
        "quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
    }
    for field, expected in expected_strings.items():
        if normalized[field] != expected:
            raise V3Phase3Error(
                f"V4 effective profile {field} must be {expected!r}, got {normalized[field]!r}"
            )
    for field, expected in (
        ("fit_grid", 401),
        ("fit_linear_bound", 8.0),
        ("fit_max_iter", 1500),
        ("fit_objective_tolerance", 1e-4),
        ("fit_parameter_tolerance", 5e-5),
        ("fit_consecutive_convergence_passes", 2),
        ("eap_grid", 801),
        ("linear_bound", 8.0),
        ("tail_region", 7.5),
    ):
        _exact_number(normalized[field], expected, field=f"effective_global_profile.{field}")
    return {
        "fit_grid": 401,
        "fit_quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
        "fit_linear_bound": 8.0,
        "fit_convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
        "fit_max_iter": 1500,
        "fit_objective_tolerance": 1e-4,
        "fit_parameter_tolerance": 5e-5,
        "fit_consecutive_convergence_passes": 2,
        "eap_grid": 801,
        "quadrature_method": cm.NORMAL_TRAPEZOID_QUADRATURE,
        "linear_bound": 8.0,
        "tail_region": 7.5,
    }


def _v4_config_spec_ids(raw: Mapping[str, Any]) -> list[str]:
    candidates = (
        raw.get("eligible_specifications")
        or raw.get("calibration_specifications")
        or (raw.get("production_panel") or {}).get("eligible_specifications")
        or (raw.get("frozen_contract") or {}).get("eligible_specifications")
    )
    if not isinstance(candidates, list):
        raise V3Phase3Error("V4 config must freeze an ordered eligible-specification list")
    ids: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            ids.append(str(candidate.get("spec_id") or candidate.get("id") or ""))
        else:
            ids.append(str(candidate))
    if not all(ids):
        raise V3Phase3Error("V4 eligible specification has no spec_id")
    return ids


def _validate_v4_evidence(
    lock_dir: Path,
    lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    hashes = lock.get("evidence_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise V3Phase3Error("V4 lock must contain non-empty evidence_sha256")
    paths = lock.get("evidence_paths")
    if not isinstance(paths, Mapping):
        paths = manifest.get("evidence_paths")
    paths = paths if isinstance(paths, Mapping) else {}

    verified: dict[str, str] = {}
    for key, expected_hash in hashes.items():
        key_text = str(key)
        relative = paths.get(key_text, key_text)
        if not isinstance(relative, str) or not relative:
            raise V3Phase3Error(f"V4 evidence path is missing for {key_text}")
        candidate = Path(relative)
        if candidate.is_absolute():
            path = candidate.resolve()
        elif relative.startswith("runs/"):
            path = _resolve_repo_path(relative)
        else:
            path = (lock_dir / candidate).resolve()
        try:
            path.relative_to(lock_dir.resolve())
        except ValueError as error:
            raise V3Phase3Error(
                f"V4 evidence path escapes the versioned output leaf: {relative}"
            ) from error
        if not path.is_file() or _sha256(path) != str(expected_hash):
            raise V3Phase3Error(f"V4 evidence hash mismatch: {key_text}")
        verified[key_text] = str(expected_hash)
    return verified


def _validate_v4_study_signature(
    followup: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    followup_path: Path,
) -> str:
    """Independently reproduce the V4 producer's frozen study signature."""

    signature = manifest.get("study_signature")
    if not isinstance(signature, Mapping):
        raise V3Phase3Error("V4 manifest study_signature is missing")
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
    if set(signature) != payload_keys | {"environment", "canonical_sha256"}:
        raise V3Phase3Error("V4 study-signature fields changed")
    payload = {key: signature.get(key) for key in payload_keys}
    canonical = _canonical_hash(payload)
    if not (
        signature.get("canonical_sha256") == manifest.get("study_signature_sha256") == canonical
    ):
        raise V3Phase3Error("V4 study-signature hash is invalid")

    environment = signature.get("environment")
    if not isinstance(environment, Mapping):
        raise V3Phase3Error("V4 signature environment is missing")
    environment_payload = {
        str(key): value for key, value in environment.items() if str(key) != "canonical_sha256"
    }
    if not (
        environment.get("canonical_sha256")
        == signature.get("environment_sha256")
        == _canonical_hash(environment_payload)
    ):
        raise V3Phase3Error("V4 signature environment hash is invalid")

    contract = followup.get("frozen_contract") or {}
    contract_hash = _canonical_hash(contract)
    dependencies = (followup.get("code_freeze") or {}).get("dependencies") or {}
    if not isinstance(dependencies, Mapping) or not dependencies:
        raise V3Phase3Error("V4 frozen dependency map is missing")
    observed_code: dict[str, str] = {}
    for relative, expected_hash in dependencies.items():
        path = _resolve_repo_path(str(relative))
        if (
            not path.is_file()
            or len(str(expected_hash)) != 64
            or _sha256(path) != str(expected_hash)
        ):
            raise V3Phase3Error(f"V4 frozen code dependency changed: {relative}")
        observed_code[str(relative)] = str(expected_hash)

    frozen = followup.get("frozen_inputs") or {}
    expected_inputs: dict[str, str] = {}
    for name in ("response_matrix", "rubrics", "scenarios", "split_manifest"):
        path = _resolve_repo_path(str(frozen.get(name) or ""))
        expected_hash = str(frozen.get(f"{name}_sha256") or "")
        if not path.is_file() or _sha256(path) != expected_hash:
            raise V3Phase3Error(f"V4 frozen input changed: {name}")
        expected_inputs[_display_path(path)] = expected_hash

    history = followup.get("historical_evidence") or {}
    expected_history = {
        key: {
            "output_file_count": int((history.get(key) or {}).get("output_file_count", -1)),
            "output_tree_sha256": (history.get(key) or {}).get("output_tree_sha256"),
            "promotable": False,
        }
        for key in ("v1_aborted", "v2_terminal_nonpromotable", "v3_terminal_blocked")
    }
    expected = {
        "schema_version": FOLLOWUP_RUN_SCHEMA,
        "config_sha256": _sha256(followup_path),
        "frozen_contract_sha256": contract_hash,
        "input_sha256": expected_inputs,
        "code_sha256": observed_code,
        "historical_evidence_sha256": expected_history,
        "spec_ids": [
            "1pl_fixed_a1",
            "log_shrinkage_2pl_lambda16",
            "log_shrinkage_2pl_lambda4",
        ],
        "scope_ids": ["full", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4"],
        "fit_nodes": [401, 801],
        "required_total_new_fits": 54,
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "adaptive_runs": 0,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise V3Phase3Error("V4 study signature differs for " + ", ".join(sorted(mismatches)))
    return canonical


def _validate_numerical_lock(
    config: Mapping[str, Any],
    specs: Sequence[CalibrationSpec],
    *,
    base_config_path: Path | None = None,
    lock_path: Path | None = None,
    followup_config_path: Path | None = None,
    allow_pending: bool = False,
    enforce_configured_paths: bool = False,
) -> dict[str, Any]:
    """Hash-validate the append-only V4 dense-fitter lock.

    This follow-up never infers a lock from numerical outputs. It accepts only
    the exact unchanged V4 config/decision/manifest/lock chain, and only when
    all three V4 specifications passed without inspecting CAT results. The
    two-spec Phase-3 subset is validated separately and may not redefine V4.
    """

    del base_config_path
    path = (lock_path or DEFAULT_NUMERICAL_LOCK).resolve()
    followup_path = (followup_config_path or DEFAULT_NUMERICAL_FOLLOWUP_CONFIG).resolve()
    if enforce_configured_paths:
        configured = config.get("numerical_lock") or {}
        expected_followup = _resolve_repo_path(
            str(configured.get("expected_followup_config") or "")
        )
        expected_lock = _resolve_repo_path(str(configured.get("expected_lock") or ""))
        if followup_path != expected_followup or path != expected_lock:
            raise V3Phase3Error("runtime override changed the frozen V4 lock path")
    pending = {
        "fit_grid": None,
        "fit_quadrature_method": None,
        "fit_linear_bound": None,
        "fit_convergence_mode": None,
        "fit_max_iter": None,
        "fit_objective_tolerance": None,
        "fit_parameter_tolerance": None,
        "fit_consecutive_convergence_passes": None,
        "eap_grid": None,
        "quadrature_method": None,
        "linear_bound": None,
        "tail_region": None,
        "verification_path": _display_path(path),
        "verification_sha256": None,
        "followup_config_path": _display_path(followup_path),
        "followup_config_sha256": (_sha256(followup_path) if followup_path.is_file() else None),
        "status": "pending_blocker",
        "passed_spec_ids": [],
        "phase3_candidate_spec_ids": list(EXPECTED_PHASE3_SPEC_IDS),
    }
    if not followup_path.is_file():
        if allow_pending:
            return pending
        raise V3Phase3Error("V4 dense-fitter follow-up config is missing")

    followup = _read_json(followup_path)
    if followup.get("schema_version") != FOLLOWUP_CONFIG_SCHEMA:
        raise V3Phase3Error("unexpected V4 dense-fitter config schema")
    if followup.get("status") != "preregistered_before_v4_results":
        raise V3Phase3Error("V4 numerical config is not prospectively frozen")
    configured_output = _resolve_repo_path(str(followup.get("output_dir") or ""))
    configured_lock = _resolve_repo_path(str(followup.get("lock_path") or ""))
    if configured_output != path.parent or configured_lock != path:
        raise V3Phase3Error("V4 config output/lock paths do not match this consumer")

    phase3_ids = _eligible_spec_ids(specs)
    if phase3_ids != list(EXPECTED_PHASE3_SPEC_IDS):
        raise V3Phase3Error("Phase-3 2PL candidate subset or order changed")
    required_ids = _eligible_spec_ids(expected_v4_calibration_specs())
    if required_ids != list(EXPECTED_V4_SPEC_IDS):
        raise V3Phase3Error("internal V4 specification contract changed")
    if _v4_config_spec_ids(followup) != required_ids:
        raise V3Phase3Error("V4 config does not freeze the exact full three-spec panel")
    convergence = (followup.get("frozen_contract") or {}).get("convergence") or {}
    expected_convergence = {
        "mode": cm.RETURNED_ITERATE_CONVERGENCE,
        "fit_max_iter": 1500,
        "objective_tolerance": 1e-4,
        "parameter_tolerance": 5e-5,
        "consecutive_convergence_passes": 2,
    }
    for field, expected in expected_convergence.items():
        value = convergence.get(field)
        if isinstance(expected, str):
            if value != expected:
                raise V3Phase3Error(f"V4 frozen convergence.{field} changed")
        else:
            _exact_number(value, expected, field=f"V4 convergence.{field}")
    _exact_number(
        convergence.get("maximum_inner_gradient"),
        1e-6,
        field="V4 convergence.maximum_inner_gradient",
    )
    integration = (followup.get("frozen_contract") or {}).get("fit_integration") or {}
    if integration.get("method") != cm.NORMAL_TRAPEZOID_QUADRATURE:
        raise V3Phase3Error("V4 fit integration method changed")
    _exact_number(integration.get("linear_bound"), 8.0, field="V4 fit linear_bound")
    if integration.get("fit_nodes") != [401, 801]:
        raise V3Phase3Error("V4 fit-node comparison changed")
    scoring = (followup.get("frozen_contract") or {}).get("fixed_bank_scoring_gate") or {}
    expected_profiles = {
        "bound8_801": [cm.NORMAL_TRAPEZOID_QUADRATURE, 801, 8.0, 7.5],
        "bound8_1601": [cm.NORMAL_TRAPEZOID_QUADRATURE, 1601, 8.0, 7.5],
        "bound10_1001": [cm.NORMAL_TRAPEZOID_QUADRATURE, 1001, 10.0, 9.5],
    }
    if scoring.get("profiles") != expected_profiles:
        raise V3Phase3Error("V4 fixed-bank scoring profiles changed")

    baseline = config.get("baseline") or {}
    frozen = followup.get("frozen_inputs") or {}
    expected_hashes = {
        "response_matrix_sha256": baseline.get("response_matrix_sha256"),
        "rubrics_sha256": baseline.get("rubrics_sha256"),
        "scenarios_sha256": baseline.get("scenarios_sha256"),
        "split_manifest_sha256": (config.get("cross_validation") or {}).get(
            "split_manifest_sha256"
        ),
    }
    if any(frozen.get(key) != value for key, value in expected_hashes.items()):
        raise V3Phase3Error("V4 numerical inputs differ from frozen Phase 3 inputs")

    decision_path = path.parent / "numerical_followup_decision.json"
    manifest_path = path.parent / "study_manifest.json"
    if not path.is_file():
        terminal_failure = False
        if decision_path.is_file() and manifest_path.is_file():
            decision = _read_json(decision_path)
            manifest = _read_json(manifest_path)
            terminal_failure = (
                decision.get("schema_version") == FOLLOWUP_RUN_SCHEMA
                and decision.get("passed") is False
                and manifest.get("schema_version") == FOLLOWUP_RUN_SCHEMA
                and manifest.get("decision_sha256") == _sha256(decision_path)
                and manifest.get("lock_sha256") is None
            )
        if allow_pending:
            if terminal_failure:
                pending["status"] = "failed_blocker"
                pending["followup_decision_sha256"] = _sha256(decision_path)
                pending["followup_manifest_sha256"] = _sha256(manifest_path)
            return pending
        if terminal_failure:
            raise V3Phase3Error("V4 numerical study completed without a promotable lock")
        raise V3Phase3Error("V4 dense-fitter numerical lock is missing")

    if not _sha_companion_matches(path):
        raise V3Phase3Error("V4 numerical lock companion hash is invalid")
    if not decision_path.is_file() or not manifest_path.is_file():
        raise V3Phase3Error("V4 numerical decision or manifest is missing")

    lock = _read_json(path)
    decision = _read_json(decision_path)
    manifest = _read_json(manifest_path)
    eligible_ids = list(map(str, lock.get("eligible_spec_ids") or []))
    expected_scopes = ["full", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4"]
    profile = _extract_v4_profile(lock)
    contract_hash = _canonical_hash(followup.get("frozen_contract") or {})
    convergence = (followup.get("frozen_contract") or {}).get("convergence") or {}

    expected_calibration_profile = [
        cm.NORMAL_TRAPEZOID_QUADRATURE,
        401,
        8.0,
    ]
    expected_eap_profile = [
        cm.NORMAL_TRAPEZOID_QUADRATURE,
        801,
        8.0,
    ]
    common_fields = {
        "fit_convergence_mode": cm.RETURNED_ITERATE_CONVERGENCE,
        "fit_max_iter": 1500,
        "fit_objective_tolerance": 1e-4,
        "fit_parameter_tolerance": 5e-5,
        "fit_consecutive_convergence_passes": 2,
    }
    for field, expected in common_fields.items():
        value = lock.get(field)
        if isinstance(expected, str):
            if value != expected:
                raise V3Phase3Error(f"V4 lock {field} changed")
        else:
            _exact_number(value, expected, field=field)
    _exact_number(
        lock.get("maximum_inner_gradient"),
        float(convergence["maximum_inner_gradient"]),
        field="maximum_inner_gradient",
    )

    if (
        lock.get("schema_version") != FOLLOWUP_LOCK_SCHEMA
        or lock.get("status") != "complete_pass"
        or lock.get("passed") is not True
        or eligible_ids != required_ids
        or list(map(str, lock.get("scope_ids") or [])) != expected_scopes
        or lock.get("calibration_integration") != expected_calibration_profile
        or lock.get("reference_eap_scoring") != expected_eap_profile
        or lock.get("calibration_specification_selected") is not None
        or lock.get("calibration_model_selection_performed") is not False
        or lock.get("adaptive_results_inspected") is not False
        or lock.get("adaptive_runs") != 0
        or lock.get("historical_fit_or_checkpoint_artifacts_reused") != 0
        or lock.get("config_sha256") != _sha256(followup_path)
        or lock.get("frozen_contract_sha256") != contract_hash
    ):
        raise V3Phase3Error("V4 numerical lock is incomplete or mismatched")

    expected_evidence_keys = {
        *(
            f"fit/{spec_id}/{scope}/{nodes}/{start}"
            for spec_id in required_ids
            for scope in expected_scopes
            for nodes, start in (
                (401, "cold"),
                (801, "cold"),
                (801, "continuation"),
            )
        ),
        *(f"start/{spec_id}/{scope}" for spec_id in required_ids for scope in expected_scopes),
        *(f"refit/{spec_id}" for spec_id in required_ids),
        *(f"fixed_bank/{spec_id}" for spec_id in required_ids),
    }
    if set(map(str, (lock.get("evidence_sha256") or {}))) != expected_evidence_keys:
        raise V3Phase3Error("V4 lock evidence panel is incomplete")
    if set(map(str, (lock.get("evidence_paths") or {}))) != expected_evidence_keys:
        raise V3Phase3Error("V4 lock evidence-path panel is incomplete")
    if len(set(map(str, (lock.get("evidence_paths") or {}).values()))) != len(
        expected_evidence_keys
    ):
        raise V3Phase3Error("V4 evidence paths must be one-to-one with evidence keys")
    evidence = _validate_v4_evidence(path.parent, lock, manifest)
    signature_hash = _validate_v4_study_signature(
        followup,
        manifest,
        followup_path=followup_path,
    )

    expected_effective = {
        "calibration_integration": expected_calibration_profile,
        "reference_eap_scoring": expected_eap_profile,
    }
    decision_fit_validity = decision.get("fit_validity") or {}
    decision_start_validity = decision.get("start_validity") or {}
    if (
        lock.get("study_signature_sha256") != signature_hash
        or lock.get("followup_decision_sha256") != _sha256(decision_path)
        or decision.get("schema_version") != FOLLOWUP_RUN_SCHEMA
        or decision.get("status") != "complete_pass"
        or decision.get("passed") is not True
        or list(map(str, decision.get("eligible_spec_ids") or [])) != required_ids
        or list(map(str, decision.get("scope_ids") or [])) != expected_scopes
        or decision.get("all_54_fits_valid") is not True
        or decision.get("all_18_grid801_start_gates_passed") is not True
        or decision.get("all_three_refit_bank_gates_passed") is not True
        or decision.get("all_three_fixed_bank_scoring_and_tail_gates_passed") is not True
        or len(decision_fit_validity) != 54
        or not all(value is True for value in decision_fit_validity.values())
        or len(decision_start_validity) != 18
        or not all(value is True for value in decision_start_validity.values())
        or set(decision.get("refit_gate_passed") or {}) != set(required_ids)
        or not all(value is True for value in (decision.get("refit_gate_passed") or {}).values())
        or set(decision.get("fixed_bank_gate_passed") or {}) != set(required_ids)
        or not all(
            value is True for value in (decision.get("fixed_bank_gate_passed") or {}).values()
        )
        or decision.get("effective_numerical_profile") != expected_effective
        or any(decision.get(field) != lock.get(field) for field in common_fields)
        or decision.get("maximum_inner_gradient") != lock.get("maximum_inner_gradient")
        or decision.get("config_sha256") != lock.get("config_sha256")
        or decision.get("frozen_contract_sha256") != lock.get("frozen_contract_sha256")
        or decision.get("study_signature_sha256") != signature_hash
        or decision.get("code_sha256_reverified_before_finalization")
        != (followup.get("code_freeze") or {}).get("dependencies")
        or decision.get("evidence_paths") != lock.get("evidence_paths")
        or decision.get("evidence_sha256") != lock.get("evidence_sha256")
        or decision.get("historical_fit_or_checkpoint_artifacts_reused") != 0
        or decision.get("calibration_specification_selected") is not None
        or decision.get("calibration_model_selection_performed") is not False
        or decision.get("adaptive_results_inspected") is not False
        or decision.get("adaptive_runs") != 0
        or manifest.get("schema_version") != FOLLOWUP_RUN_SCHEMA
        or manifest.get("status") != "complete_pass"
        or manifest.get("config_sha256") != _sha256(followup_path)
        or manifest.get("decision_sha256") != _sha256(decision_path)
        or manifest.get("lock_sha256") != _sha256(path)
        or manifest.get("study_signature_sha256") != signature_hash
        or manifest.get("historical_fit_or_checkpoint_artifacts_reused") != 0
        or manifest.get("calibration_model_selection_performed") is not False
        or manifest.get("adaptive_results_inspected") is not False
        or manifest.get("adaptive_runs") != 0
    ):
        raise V3Phase3Error("V4 lock/decision/manifest provenance chain is invalid")

    return {
        **profile,
        "verification_path": _display_path(path),
        "verification_sha256": _sha256(path),
        "followup_config_path": _display_path(followup_path),
        "followup_config_sha256": _sha256(followup_path),
        "followup_decision_sha256": _sha256(decision_path),
        "followup_manifest_sha256": _sha256(manifest_path),
        "study_signature_sha256": signature_hash,
        "frozen_contract_sha256": contract_hash,
        "evidence_sha256": evidence,
        "status": "passed",
        "passed_spec_ids": eligible_ids,
        "eligible_spec_ids": eligible_ids,
        "phase3_candidate_spec_ids": phase3_ids,
        "phase3_subset_of_v4_exact": all(spec_id in eligible_ids for spec_id in phase3_ids)
        and phase3_ids == list(EXPECTED_PHASE3_SPEC_IDS),
        "historical_fit_or_checkpoint_artifacts_reused": 0,
    }


def _subset_bank_by_criteria(
    bank: scat.FittedBank, criterion_ids: Iterable[str]
) -> scat.FittedBank:
    allowed = set(map(str, criterion_ids))
    indices = np.asarray(
        [index for index, item in enumerate(bank.criterion_ids) if item in allowed],
        dtype=int,
    )
    if indices.size != len(allowed):
        raise V3Phase3Error("fitted bank is missing requested common criteria")
    return scat.FittedBank(
        records=[bank.records[index] for index in indices],
        dims=bank.dims,
        criterion_ids=tuple(bank.criterion_ids[index] for index in indices),
        scenario_ids=tuple(bank.scenario_ids[index] for index in indices),
        Q=bank.Q[indices].copy(),
        A=bank.A[indices].copy(),
        b=bank.b[indices].copy(),
        latent_correlation=bank.latent_correlation.copy(),
        source_path=bank.source_path,
        dropped_negative_items=bank.dropped_negative_items,
    )


def _fit_structure_dense(
    train_df: pd.DataFrame,
    q_by: dict[str, Any],
    args: argparse.Namespace,
    structure: cell_cv.SkillStructure,
) -> dict[str, Any]:
    """Fit one V3 fold under the complete V4 dense-fitter contract."""

    Y, M, q_source, items, _block_df, diag = cm.prepare_block(train_df, q_by)
    q_modeled = structure.transform_q(q_source)
    fit = cm.fit_m2pl_em(
        Y,
        M,
        q_modeled,
        args.grid,
        estimate_corr=False,
        ridge=args.ridge,
        max_iter=args.max_iter,
        tol=args.tol,
        calibration_model=args.calibration_model,
        log_a_shrinkage=args.log_a_shrinkage,
        quadrature_method=args.quadrature_method,
        linear_bound=args.linear_bound,
        convergence_mode=args.convergence_mode,
        parameter_tol=args.parameter_tol,
        consecutive_convergence_passes=args.consecutive_convergence_passes,
    )
    modeled_pattern_counts: dict[str, int] = {}
    for row in q_modeled:
        key = "".join(str(int(value)) for value in row)
        modeled_pattern_counts[key] = modeled_pattern_counts.get(key, 0) + 1
    diagnostics = {
        **diag,
        "modeled_structure": structure.as_dict(),
        "modeled_q_pattern_counts": modeled_pattern_counts,
        "modeled_items_per_dimension": {
            label: int(q_modeled[:, index].sum()) for index, label in enumerate(structure.labels)
        },
    }
    result: dict[str, Any] = {
        "items": items,
        "A": fit["A"],
        "b": fit["b"],
        "R": fit["R"],
        "dim_labels": list(structure.labels),
        "collapsed_labels": list(structure.labels),
        "loglik": fit["loglik"],
        "n_params": fit["n_params"],
        "n_iter": fit["n_iter"],
        "converged": fit["converged"],
        "calibration_specification": fit["calibration_specification"],
        "diag": diagnostics,
    }
    for key in (
        "grid_nodes",
        "quadrature_method",
        "quadrature_linear_bound",
        "convergence_mode",
        "penalized_objective",
        "convergence_diagnostics",
    ):
        if key in fit:
            result[key] = fit[key]
    return result


class V3Phase3Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_path = args.config.resolve()
        self.split_path = args.split_manifest.resolve()
        self.config = _read_json(self.config_path)
        self.split = _read_json(self.split_path)
        self.parent_config_provenance = _check_frozen_config(self.config)
        self.output_dir = _resolve_phase3_output_dir(
            self.config,
            getattr(args, "out_dir", None),
            plan_only=bool(args.plan_only),
        )
        self.output_requested = self.output_dir
        self.specs = load_calibration_specs(self.config)
        self.policies = load_policies(self.config)
        self.numerical = _validate_numerical_lock(
            self.config,
            self.specs,
            base_config_path=self.config_path,
            lock_path=getattr(args, "numerical_lock", DEFAULT_NUMERICAL_LOCK),
            followup_config_path=getattr(
                args,
                "numerical_followup_config",
                DEFAULT_NUMERICAL_FOLLOWUP_CONFIG,
            ),
            allow_pending=args.plan_only,
            enforce_configured_paths=True,
        )
        if self.numerical.get("status") == "passed" and self.args.max_grid_nodes < int(
            self.numerical["eap_grid"]
        ):
            raise V3Phase3Error("max-grid-nodes cannot hold the verified EAP grid")

        cv = self.config.get("cross_validation") or {}
        expected_split_hash = str(
            cv.get("split_manifest_sha256") or (cv.get("split_manifest") or {}).get("sha256")
            if isinstance(cv.get("split_manifest"), Mapping)
            else cv.get("split_manifest_sha256") or ""
        )
        # The conditional expression above is intentionally normalized again for
        # configs that store the path as a plain string.
        expected_split_hash = str(cv.get("split_manifest_sha256") or expected_split_hash)
        if not expected_split_hash or _sha256(self.split_path) != expected_split_hash:
            raise V3Phase3Error("repeated split manifest hash does not match V3 config")

        baseline = self.config.get("baseline") or {}
        if int(baseline.get("models", 0)) != EXPECTED_MODELS:
            raise V3Phase3Error("V3 baseline must freeze the 52-model cohort")
        self.matrix_path, matrix_hash = _input_entry(self.config, "response_matrix")
        self.rubrics_path, rubrics_hash = _input_entry(self.config, "rubrics")
        self.scenarios_path, scenarios_hash = _input_entry(self.config, "scenarios")
        self.judge_manifest_path, judge_hash = _input_entry(self.config, "judge_manifest")
        self.input_paths = {
            "response_matrix": self.matrix_path,
            "rubrics": self.rubrics_path,
            "scenarios": self.scenarios_path,
            "judge_manifest": self.judge_manifest_path,
        }
        expected_hashes = {
            "response_matrix": matrix_hash,
            "rubrics": rubrics_hash,
            "scenarios": scenarios_hash,
            "judge_manifest": judge_hash,
        }
        for name, path in self.input_paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            if _sha256(path) != expected_hashes[name]:
                raise V3Phase3Error(f"{name} does not match the frozen config hash")
        self.input_hashes = expected_hashes

        self.matrix = cm.load_matrix_strict(self.matrix_path)
        self.scenario_records = scat.load_scenario_records(self.scenarios_path)
        self.split_audit = audit_repeated_split_manifest(
            self.split,
            expected_models=self.matrix.index,
            expected_scenarios=self.scenario_records,
        )
        split_input = (self.split.get("inputs") or {}).get("response_matrix") or {}
        split_matrix_hash = str(split_input.get("sha256") or "")
        if split_matrix_hash and split_matrix_hash != matrix_hash:
            raise V3Phase3Error("config and split manifest disagree on response matrix")

        source_skills = cm.configure_skills(args.skills)
        self.structure = cell_cv.build_structure(
            tuple(source_skills), args.dimensions, args.structure_name
        )
        if self.structure.n_dims != 1 or tuple(self.structure.labels) != ("instruction_following",):
            raise V3Phase3Error("V3 remains frozen to one instruction-following dimension")
        self.q_by = cm.load_q_matrix(self.rubrics_path)
        cm.validate_matrix_bank_alignment(self.matrix, self.q_by, self.args.require_complete_bank)
        self.source_records = scenario_cv.source_records_by_id(self.rubrics_path)

        self.code_hashes = _code_hashes()
        self.environment = _environment_provenance()
        self.study_signature = self._study_signature()
        self._already_complete = False
        self._validate_frozen_runtime()

    def _validate_frozen_runtime(self) -> None:
        cat = self.config.get("cat_policies") or {}
        runtime = self.config.get("runtime") or {}
        expected = {
            "top_n": int(cat.get("top_n", -1)),
            "max_scenarios": int(cat.get("maximum_adaptive_scenarios", -1)),
            "minimum_scored_criteria": int(cat.get("minimum_scored_criteria", -1)),
            "seed": int(runtime.get("master_seed", -1)),
            "max_iter": int(runtime.get("fit_max_iter", -1)),
            "tol": float(runtime.get("fit_tolerance", math.nan)),
            "mwle_ridge": float(runtime.get("mwle_ridge", math.nan)),
            "metric_bootstrap_replicates": int(
                runtime.get("metric_family_bootstrap_replicates", -1)
            ),
            "negative_policy": str(runtime.get("negative_loading_policy") or ""),
        }
        observed = {
            "top_n": self.args.top_n,
            "max_scenarios": self.args.max_scenarios,
            "minimum_scored_criteria": self.args.minimum_scored_criteria,
            "seed": self.args.seed,
            "max_iter": self.args.max_iter,
            "tol": self.args.tol,
            "mwle_ridge": self.args.mwle_ridge,
            "metric_bootstrap_replicates": self.args.metric_bootstrap_replicates,
            "negative_policy": self.args.negative_policy,
        }
        for key, expected_value in expected.items():
            value = observed[key]
            if isinstance(expected_value, float):
                equal = math.isfinite(expected_value) and math.isclose(
                    float(value), expected_value, rel_tol=0, abs_tol=0
                )
            else:
                equal = value == expected_value
            if not equal:
                raise V3Phase3Error(
                    f"runtime override changed frozen {key}: expected {expected_value}, got {value}"
                )

    def _study_signature(self) -> str:
        return _canonical_hash(
            {
                "schema": SCRIPT_SCHEMA,
                "study_family": "InFoBench_2pl_only_v1",
                "config_sha256": _sha256(self.config_path),
                "parent_config_provenance": self.parent_config_provenance,
                "split_sha256": _sha256(self.split_path),
                "inputs": self.input_hashes,
                "code": self.code_hashes,
                "environment": self.environment,
                "structure": self.structure.as_dict(),
                "numerical": self.numerical,
                "specifications": [
                    {**asdict(spec), "canonical": spec.canonical} for spec in self.specs
                ],
                "policies": [asdict(policy) for policy in self.policies],
                "runtime": {
                    "seed": self.args.seed,
                    "top_n": self.args.top_n,
                    "max_scenarios": self.args.max_scenarios,
                    "minimum_scored_criteria": self.args.minimum_scored_criteria,
                    "max_iter": self.args.max_iter,
                    "tol": self.args.tol,
                    "fit_parameter_tolerance": self.numerical["fit_parameter_tolerance"],
                    "fit_consecutive_convergence_passes": self.numerical[
                        "fit_consecutive_convergence_passes"
                    ],
                    "fit_convergence_mode": self.numerical["fit_convergence_mode"],
                    "fit_quadrature_method": self.numerical["fit_quadrature_method"],
                    "fit_linear_bound": self.numerical["fit_linear_bound"],
                    "mwle_ridge": self.args.mwle_ridge,
                    "metric_bootstrap_replicates": self.args.metric_bootstrap_replicates,
                    "negative_policy": self.args.negative_policy,
                },
            }
        )

    def _base_manifest(self, status: str, *, created_at: str | None = None) -> dict[str, Any]:
        return {
            "schema_version": SCRIPT_SCHEMA,
            "status": status,
            "created_at": created_at or _utcnow(),
            "study_signature": self.study_signature,
            "script": "scripts/nested_scenario_cat_cv_2pl_only_v1.py",
            "command": sys.argv,
            "git_commit": _git_commit(),
            "inputs": {
                "config": {
                    "path": _display_path(self.config_path),
                    "sha256": _sha256(self.config_path),
                },
                "split_manifest": {
                    "path": _display_path(self.split_path),
                    "sha256": _sha256(self.split_path),
                },
                **{
                    name: {"path": _display_path(path), "sha256": self.input_hashes[name]}
                    for name, path in self.input_paths.items()
                },
            },
            "code_provenance": {
                "dependency_inventory": list(CODE_DEPENDENCY_RELATIVE_PATHS),
                "files": self.code_hashes,
                "canonical_sha256": _canonical_hash(self.code_hashes),
            },
            "parent_config_provenance": self.parent_config_provenance,
            "environment": self.environment,
            "numerical_lock": self.numerical,
            "cross_validation": {
                "repetitions": EXPECTED_REPEATS,
                "outer_folds_per_repeat": EXPECTED_OUTER_FOLDS,
                "inner_folds_per_outer": EXPECTED_INNER_FOLDS,
                "bootstrap_unit": "tutor_family",
                "repeated_rows_treated_as_independent": False,
            },
            "inner_selection_fail_closed": {
                "require_all_specs_every_inner_fold": True,
                "expected_spec_fold_combinations_per_panel": (EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL),
                "survivor_selection_allowed": False,
                "outer_outcomes_opened_only_after_all_panel_selections_locked": True,
            },
            "gate_scope_contract": self._gate_scope_contract(),
            "calibration_specifications": [
                {
                    "spec_id": spec.spec_id,
                    "simplicity_rank": spec.simplicity_rank,
                    "complete_specification": spec.canonical,
                }
                for spec in self.specs
            ],
            "cat_policies": [asdict(policy) for policy in self.policies],
            "primary_policy_only_can_pass_phase3": True,
            "sensitivity_promotion_allowed": False,
            "limitations": [
                "same 52-model cohort is internal development, not independent confirmation",
                "results are conditional on frozen Qwen labels not human-validated on InFoBench",
            ],
        }

    def _prepare_output(self) -> None:
        manifest_path = self.output_dir / "manifest.json"
        if self.args.resume:
            if manifest_path.is_file():
                manifest = _read_json(manifest_path)
                if (
                    manifest.get("schema_version") != SCRIPT_SCHEMA
                    or manifest.get("study_signature") != self.study_signature
                ):
                    raise V3Phase3Error("resume target is not this exact 2PL-only study")
                terminal_status = str(manifest.get("status") or "")
                if terminal_status in TERMINAL_MANIFEST_OUTPUTS:
                    inventory = manifest.get("outputs")
                    if not isinstance(inventory, Mapping):
                        raise V3Phase3Error("completed resume manifest has no output inventory")
                    expected = set(TERMINAL_MANIFEST_OUTPUTS[terminal_status])
                    observed = set(map(str, inventory))
                    if observed != expected:
                        missing = sorted(expected - observed)
                        extra = sorted(observed - expected)
                        raise V3Phase3Error(
                            "completed resume output inventory is not exact: "
                            f"missing={missing}, extra={extra}"
                        )
                    for name in sorted(expected):
                        provenance = inventory[name]
                        if not isinstance(provenance, Mapping):
                            raise V3Phase3Error(
                                f"completed resume output provenance is invalid: {name}"
                            )
                        path = self.output_dir / name
                        if (
                            provenance.get("path") != _display_path(path)
                            or not path.is_file()
                            or _sha256(path) != provenance.get("sha256")
                        ):
                            raise V3Phase3Error(f"completed resume output failed hash: {name}")
                    self._already_complete = True
                    return
            else:
                self._validate_precompute_only_tree()
        elif self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise V3Phase3Error(
                f"output directory is not empty: {self.output_dir}; use --resume for "
                "an interrupted identical study or choose a new versioned output leaf"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_precompute_only_tree(self) -> None:
        """Allow resume from isolated panel work, never from parent/unknown artifacts."""

        if not self.output_dir.is_dir():
            raise V3Phase3Error(
                "--resume without a manifest requires an existing precompute output leaf"
            )
        allowed_roots = {
            "two_pl_only_v1_fit_cache",
            "two_pl_only_v1_checkpoints",
            "two_pl_only_v1_precomputed_panels",
            "two_pl_only_v1_panel_locks",
        }
        observed_roots = {path.name for path in self.output_dir.iterdir()}
        unexpected = sorted(observed_roots - allowed_roots)
        if unexpected:
            raise V3Phase3Error(
                "manifest-free resume contains non-precompute artifacts: " + ", ".join(unexpected)
            )
        lock_root = self.output_dir / "two_pl_only_v1_panel_locks"
        if lock_root.is_dir() and any(path.is_file() for path in lock_root.rglob("*")):
            raise V3Phase3Error("cannot resume while one or more panel ownership locks exist")
        if not observed_roots:
            raise V3Phase3Error("manifest-free resume found no precompute artifacts")
        panel_root = self.output_dir / "two_pl_only_v1_precomputed_panels"
        panel_paths = sorted(panel_root.rglob("*.json")) if panel_root.is_dir() else []
        if not panel_paths:
            raise V3Phase3Error(
                "manifest-free resume requires at least one completed panel checkpoint"
            )
        expected_outer = {
            (int(repetition["repeat"]), int(outer["outer_fold"])): outer
            for repetition in self.split_audit["repetitions"]
            for outer in repetition["outer_folds"]
        }
        for path in panel_paths:
            raw = _read_json(path)
            context = raw.get("context") or {}
            try:
                key = (int(context.get("repeat")), int(context.get("outer_fold")))
            except (TypeError, ValueError) as error:
                raise V3Phase3Error(f"invalid panel checkpoint indices: {path}") from error
            outer = expected_outer.get(key)
            if (
                outer is None
                or path.resolve() != self._panel_selection_checkpoint_path(*key).resolve()
            ):
                raise V3Phase3Error(f"unexpected panel checkpoint path/context: {path}")
            self._load_panel_selection_checkpoint(repeat=key[0], outer=outer)

    def _fit_cache_key(
        self,
        *,
        repeat: int,
        outer_fold: int,
        inner_fold: int | None,
        role: str,
        training_model_ids: Sequence[str],
        spec: CalibrationSpec,
    ) -> str:
        """Key includes every semantic, data, code, environment, and split input."""

        return _canonical_hash(
            {
                "cache_schema": CACHE_SCHEMA,
                "v1_cache_compatible": False,
                "study_signature": self.study_signature,
                "repeat": int(repeat),
                "outer_fold": int(outer_fold),
                "inner_fold": inner_fold,
                "role": role,
                "training_model_ids": sorted(map(str, training_model_ids)),
                "exact_calibration_specification": spec.canonical,
                "exact_spec_id": spec.spec_id,
                "inputs": self.input_hashes,
                "config_sha256": _sha256(self.config_path),
                "split_sha256": _sha256(self.split_path),
                "code_sha256": _canonical_hash(self.code_hashes),
                "environment_sha256": self.environment["canonical_sha256"],
                "numerical": self.numerical,
                "optimizer": {
                    "max_iter": self.args.max_iter,
                    "tol": self.args.tol,
                    "parameter_tol": self.numerical["fit_parameter_tolerance"],
                    "consecutive_convergence_passes": self.numerical[
                        "fit_consecutive_convergence_passes"
                    ],
                    "convergence_mode": self.numerical["fit_convergence_mode"],
                    "quadrature_method": self.numerical["fit_quadrature_method"],
                    "linear_bound": self.numerical["fit_linear_bound"],
                    "estimate_latent_corr": False,
                    "negative_policy": self.args.negative_policy,
                },
            }
        )

    def _cache_dir(
        self,
        *,
        repeat: int,
        outer_fold: int,
        inner_fold: int | None,
        spec: CalibrationSpec,
    ) -> Path:
        stage = f"inner_{inner_fold:02d}" if inner_fold is not None else "outer_fit"
        return (
            self.output_dir
            / "two_pl_only_v1_fit_cache"
            / f"repeat_{repeat:02d}"
            / f"outer_{outer_fold:02d}"
            / stage
            / spec.fit_namespace()
        )

    def _write_fit_cache(
        self,
        *,
        cache_dir: Path,
        fit: Mapping[str, Any],
        cache_key: str,
        repeat: int,
        outer_fold: int,
        inner_fold: int | None,
        role: str,
        train_ids: Sequence[str],
        spec: CalibrationSpec,
        bank_policy: Mapping[str, Any],
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        array_path = cache_dir / "fit_arrays.npz"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=cache_dir, prefix=".fit_arrays.npz.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(
                    handle,
                    A=np.asarray(fit["A"], dtype=float),
                    b=np.asarray(fit["b"], dtype=float),
                    R=np.asarray(fit["R"], dtype=float),
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(array_path)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "v1_cache_compatible": False,
            "study_signature": self.study_signature,
            "cache_key": cache_key,
            "repeat": repeat,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "role": role,
            "training_model_ids": sorted(map(str, train_ids)),
            "spec_id": spec.spec_id,
            "calibration_specification": spec.canonical,
            "config_sha256": _sha256(self.config_path),
            "split_sha256": _sha256(self.split_path),
            "input_hashes": self.input_hashes,
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
            "numerical_lock": self.numerical,
            "items": list(map(str, fit["items"])),
            "dim_labels": list(map(str, fit["dim_labels"])),
            "loglik": fit["loglik"],
            "n_params": fit["n_params"],
            "n_iter": fit["n_iter"],
            "converged": fit["converged"],
            "grid_nodes": fit.get("grid_nodes"),
            "quadrature_method": fit.get("quadrature_method"),
            "quadrature_linear_bound": fit.get("quadrature_linear_bound"),
            "convergence_mode": fit.get("convergence_mode"),
            "penalized_objective": fit.get("penalized_objective"),
            "convergence_diagnostics": fit.get("convergence_diagnostics"),
            "diagnostics": fit.get("diag") or {},
            "bank_policy": dict(bank_policy),
            "arrays_sha256": _sha256(array_path),
        }
        manifest["manifest_content_sha256"] = _canonical_hash(manifest)
        _atomic_json(cache_dir / "fit_manifest.json", manifest)

    def _load_fit_cache(
        self,
        *,
        cache_dir: Path,
        expected_key: str,
        repeat: int,
        outer_fold: int,
        inner_fold: int | None,
        role: str,
        train_ids: Sequence[str],
        spec: CalibrationSpec,
    ) -> dict[str, Any] | None:
        manifest_path = cache_dir / "fit_manifest.json"
        array_path = cache_dir / "fit_arrays.npz"
        if not (manifest_path.is_file() and array_path.is_file()):
            return None
        manifest = _read_json(manifest_path)
        content_hash = str(manifest.pop("manifest_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(manifest):
            raise V3Phase3Error(f"fit cache content hash mismatch: {cache_dir}")
        expected = {
            "schema_version": CACHE_SCHEMA,
            "v1_cache_compatible": False,
            "study_signature": self.study_signature,
            "cache_key": expected_key,
            "repeat": repeat,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "role": role,
            "training_model_ids": sorted(map(str, train_ids)),
            "spec_id": spec.spec_id,
            "calibration_specification": spec.canonical,
            "config_sha256": _sha256(self.config_path),
            "split_sha256": _sha256(self.split_path),
            "input_hashes": self.input_hashes,
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
            "numerical_lock": self.numerical,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise V3Phase3Error(f"fit cache mismatch for {key}: {cache_dir}")
        if manifest.get("arrays_sha256") != _sha256(array_path):
            raise V3Phase3Error(f"fit cache array hash mismatch: {cache_dir}")
        with np.load(array_path, allow_pickle=False) as arrays:
            return {
                "items": list(map(str, manifest["items"])),
                "A": np.asarray(arrays["A"], dtype=float),
                "b": np.asarray(arrays["b"], dtype=float),
                "R": np.asarray(arrays["R"], dtype=float),
                "dim_labels": list(map(str, manifest["dim_labels"])),
                "collapsed_labels": list(map(str, manifest["dim_labels"])),
                "loglik": float(manifest["loglik"]),
                "n_params": int(manifest["n_params"]),
                "n_iter": int(manifest["n_iter"]),
                "converged": bool(manifest["converged"]),
                "grid_nodes": int(manifest["grid_nodes"]),
                "quadrature_method": str(manifest["quadrature_method"]),
                "quadrature_linear_bound": float(manifest["quadrature_linear_bound"]),
                "convergence_mode": str(manifest["convergence_mode"]),
                "penalized_objective": float(manifest["penalized_objective"]),
                "convergence_diagnostics": dict(manifest["convergence_diagnostics"]),
                "calibration_specification": dict(manifest["calibration_specification"]),
                "diag": manifest.get("diagnostics") or {},
            }

    def fit_or_load(
        self,
        *,
        repeat: int,
        outer_fold: int,
        inner_fold: int | None,
        role: str,
        training_model_ids: Sequence[str],
        forbidden_model_ids: Iterable[str],
        spec: CalibrationSpec,
    ) -> FitBundle:
        train_ids = tuple(sorted(map(str, training_model_ids)))
        forbidden = set(map(str, forbidden_model_ids))
        if set(train_ids) & forbidden:
            raise V3Phase3Error(f"{role}: held-out model leaked into calibration")
        if not set(train_ids) <= set(map(str, self.matrix.index)):
            raise V3Phase3Error(f"{role}: unknown item-fit model ID")
        cache_key = self._fit_cache_key(
            repeat=repeat,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            role=role,
            training_model_ids=train_ids,
            spec=spec,
        )
        cache_dir = self._cache_dir(
            repeat=repeat,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            spec=spec,
        )
        fit = self._load_fit_cache(
            cache_dir=cache_dir,
            expected_key=cache_key,
            repeat=repeat,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            role=role,
            train_ids=train_ids,
            spec=spec,
        )
        was_cached = fit is not None
        if fit is None:
            fit_args = argparse.Namespace(
                grid=int(self.numerical["fit_grid"]),
                estimate_latent_corr=False,
                ridge=0.0 if spec.ridge is None else float(spec.ridge),
                log_a_shrinkage=(
                    cm.DEFAULT_LOG_A_SHRINKAGE
                    if spec.log_a_shrinkage is None
                    else float(spec.log_a_shrinkage)
                ),
                calibration_model=spec.family,
                max_iter=int(self.numerical["fit_max_iter"]),
                tol=float(self.numerical["fit_objective_tolerance"]),
                quadrature_method=str(self.numerical["fit_quadrature_method"]),
                linear_bound=float(self.numerical["fit_linear_bound"]),
                convergence_mode=str(self.numerical["fit_convergence_mode"]),
                parameter_tol=float(self.numerical["fit_parameter_tolerance"]),
                consecutive_convergence_passes=int(
                    self.numerical["fit_consecutive_convergence_passes"]
                ),
            )
            fit = _fit_structure_dense(
                self.matrix.loc[list(train_ids)], self.q_by, fit_args, self.structure
            )
        if fit.get("calibration_specification") != spec.canonical:
            raise V3Phase3Error(f"{role}/{spec.spec_id}: fitter changed exact specification")
        if not bool(fit.get("converged")):
            raise V3Phase3Error(f"{role}/{spec.spec_id}: calibration did not converge")
        diagnostics = fit.get("convergence_diagnostics") or {}
        if (
            fit.get("grid_nodes") != 401
            or fit.get("quadrature_method") != cm.NORMAL_TRAPEZOID_QUADRATURE
            or not math.isclose(
                float(fit.get("quadrature_linear_bound", math.nan)),
                8.0,
                rel_tol=0,
                abs_tol=0,
            )
            or fit.get("convergence_mode") != cm.RETURNED_ITERATE_CONVERGENCE
            or diagnostics.get("returned_iterate_matches_last_trace") is not True
            or diagnostics.get("stopped_before_extra_mstep") is not True
            or diagnostics.get("all_objective_changes_monotone_within_tolerance") is not True
            or diagnostics.get("required_consecutive_passes") != 2
            or diagnostics.get("final_consecutive_passes", 0) < 2
        ):
            raise V3Phase3Error(f"{role}/{spec.spec_id}: fit violates the V4 dense-fitter contract")
        A = np.asarray(fit["A"], dtype=float)
        b = np.asarray(fit["b"], dtype=float)
        R = np.asarray(fit["R"], dtype=float)
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b)) and np.all(np.isfinite(R))):
            raise V3Phase3Error(f"{role}/{spec.spec_id}: non-finite fitted parameters")
        active = A[np.abs(A) > 0]
        if spec.family == cm.ONE_PL and (
            active.size == 0 or not np.allclose(active, 1.0, rtol=0, atol=1e-12)
        ):
            raise V3Phase3Error(f"{role}/{spec.spec_id}: 1PL discrimination is not fixed at 1")
        if spec.family == cm.LOG_SHRINKAGE_2PL and (active.size == 0 or np.any(active <= 0)):
            raise V3Phase3Error(f"{role}/{spec.spec_id}: shrinkage discrimination is invalid")
        bank, policy = scenario_cv.build_fold_bank(
            fit,
            self.structure,
            self.source_records,
            negative_policy=self.args.negative_policy,
        )
        admin = set(self.split_audit["administration_scenario_ids"])
        evaluation = set(self.split_audit["evaluation_scenario_ids"])
        if not (set(bank.scenario_ids) & admin and set(bank.scenario_ids) & evaluation):
            raise V3Phase3Error(
                f"{role}/{spec.spec_id}: fitted bank lacks required scenario support"
            )
        if not was_cached:
            self._write_fit_cache(
                cache_dir=cache_dir,
                fit=fit,
                cache_key=cache_key,
                repeat=repeat,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                role=role,
                train_ids=train_ids,
                spec=spec,
                bank_policy=policy,
            )
        quadrature = scat.build_quadrature(
            self.structure.n_dims,
            int(self.numerical["eap_grid"]),
            bank.latent_correlation,
            max_nodes=max(self.args.max_grid_nodes, int(self.numerical["eap_grid"])),
            method=str(self.numerical["quadrature_method"]),
            linear_bound=float(self.numerical["linear_bound"]),
        )
        return FitBundle(
            fit=dict(fit),
            bank=bank,
            quadrature=quadrature,
            cache_key=cache_key,
            cache_dir=cache_dir,
            training_model_ids=train_ids,
            exact_spec=spec,
            bank_policy=dict(policy),
        )

    def _checkpoint_path(self, stage: str, repeat: int, outer_fold: int, suffix: str) -> Path:
        return (
            self.output_dir
            / "two_pl_only_v1_checkpoints"
            / f"repeat_{repeat:02d}"
            / f"outer_{outer_fold:02d}"
            / f"{stage}__{suffix}.json"
        )

    def _load_rows_checkpoint(
        self, path: Path, *, stage: str, context: Mapping[str, Any]
    ) -> list[dict[str, Any]] | None:
        if not (self.args.resume and path.is_file()):
            return None
        value = _read_json(path)
        content_hash = str(value.pop("checkpoint_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(value):
            raise V3Phase3Error(f"checkpoint content hash mismatch: {path}")
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "study_signature": self.study_signature,
            "stage": stage,
            "context": dict(context),
        }
        for key, item in expected.items():
            if value.get(key) != item:
                raise V3Phase3Error(f"checkpoint mismatch for {key}: {path}")
        rows = value.get("rows")
        if not isinstance(rows, list) or value.get("rows_sha256") != _canonical_hash(rows):
            raise V3Phase3Error(f"checkpoint rows failed integrity: {path}")
        return [dict(row) for row in rows]

    def _write_rows_checkpoint(
        self,
        path: Path,
        *,
        stage: str,
        context: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "study_signature": self.study_signature,
            "stage": stage,
            "context": dict(context),
            "rows_sha256": _canonical_hash(rows),
            "rows": list(rows),
        }
        payload["checkpoint_content_sha256"] = _canonical_hash(payload)
        _atomic_json(path, payload)

    def _panel_selection_checkpoint_path(self, repeat: int, outer_fold: int) -> Path:
        return (
            self.output_dir
            / "two_pl_only_v1_precomputed_panels"
            / f"repeat_{repeat:02d}"
            / f"outer_{outer_fold:02d}.json"
        )

    def _panel_checkpoint_context(self, repeat: int, outer: Mapping[str, Any]) -> dict[str, Any]:
        outer_fold = int(outer["outer_fold"])
        return {
            "panel_id": _panel_id(repeat, outer_fold),
            "repeat": int(repeat),
            "outer_fold": outer_fold,
            "outer_split_sha256": _canonical_hash(outer),
            "phase3_candidate_spec_ids": [spec.spec_id for spec in self.specs],
            "expected_spec_fold_combinations": EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL,
            "config_sha256": _sha256(self.config_path),
            "split_sha256": _sha256(self.split_path),
        }

    def _write_panel_selection_checkpoint(
        self,
        *,
        repeat: int,
        outer: Mapping[str, Any],
        panel: Mapping[str, Any],
        evidence: Sequence[Mapping[str, Any]],
    ) -> None:
        if panel.get("status") != "selection_complete" or panel.get("selected_spec_id") not in {
            spec.spec_id for spec in self.specs
        }:
            raise V3Phase3Error(
                f"{_panel_id(repeat, int(outer['outer_fold']))}: incomplete selection "
                "cannot become a reusable completed checkpoint"
            )
        context = self._panel_checkpoint_context(repeat, outer)
        payload = {
            "schema_version": PANEL_CHECKPOINT_SCHEMA,
            "status": "selection_complete",
            "study_signature": self.study_signature,
            "context": context,
            "panel_sha256": _canonical_hash(panel),
            "evidence_sha256": _canonical_hash(evidence),
            "panel": dict(panel),
            "evidence": [dict(row) for row in evidence],
            "outer_outcomes_opened": False,
            "outer_policy_evaluation_called": False,
        }
        payload["checkpoint_content_sha256"] = _canonical_hash(payload)
        _atomic_json(
            self._panel_selection_checkpoint_path(repeat, int(outer["outer_fold"])),
            payload,
        )

    def _load_panel_selection_checkpoint(
        self, *, repeat: int, outer: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        path = self._panel_selection_checkpoint_path(repeat, int(outer["outer_fold"]))
        if not path.is_file():
            return None
        payload = _read_json(path)
        content_hash = str(payload.pop("checkpoint_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(payload):
            raise V3Phase3Error(f"panel checkpoint content hash mismatch: {path}")
        expected_context = self._panel_checkpoint_context(repeat, outer)
        if (
            payload.get("schema_version") != PANEL_CHECKPOINT_SCHEMA
            or payload.get("status") != "selection_complete"
            or payload.get("study_signature") != self.study_signature
            or payload.get("context") != expected_context
            or payload.get("outer_outcomes_opened") is not False
            or payload.get("outer_policy_evaluation_called") is not False
        ):
            raise V3Phase3Error(f"panel checkpoint provenance mismatch: {path}")
        panel = payload.get("panel")
        evidence = payload.get("evidence")
        if not isinstance(panel, Mapping) or not isinstance(evidence, list):
            raise V3Phase3Error(f"panel checkpoint payload is malformed: {path}")
        if (
            payload.get("panel_sha256") != _canonical_hash(panel)
            or payload.get("evidence_sha256") != _canonical_hash(evidence)
            or panel.get("panel_id") != expected_context["panel_id"]
            or panel.get("repeat") != repeat
            or panel.get("outer_fold") != int(outer["outer_fold"])
            or panel.get("status") != "selection_complete"
            or panel.get("selected_spec_id") not in {spec.spec_id for spec in self.specs}
        ):
            raise V3Phase3Error(f"panel checkpoint rows or selection failed integrity: {path}")
        audit = (panel.get("inner_selection") or {}).get("complete_inner_evidence_audit") or {}
        if (
            audit.get("passed") is not True
            or audit.get("expected_spec_fold_combinations") != EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL
            or audit.get("valid_spec_fold_combinations") != EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL
        ):
            raise V3Phase3Error(f"panel checkpoint lacks complete 8-block evidence: {path}")
        return dict(panel), [dict(row) for row in evidence]

    def _select_or_load_panel(
        self,
        repeat: int,
        outer: Mapping[str, Any],
        *,
        permit_precomputed: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        outer_fold = int(outer["outer_fold"])
        with _exclusive_panel_lock(
            self.output_dir,
            study_signature=self.study_signature,
            repeat=repeat,
            outer_fold=outer_fold,
        ):
            if permit_precomputed:
                completed = self._load_panel_selection_checkpoint(repeat=repeat, outer=outer)
                if completed is not None:
                    return completed
            panel, evidence = self._select_panel(repeat, outer)
            if panel.get("status") == "selection_complete":
                self._write_panel_selection_checkpoint(
                    repeat=repeat,
                    outer=outer,
                    panel=panel,
                    evidence=evidence,
                )
            return panel, evidence

    def _score_inner_common_support(
        self,
        *,
        repeat: int,
        outer_fold: int,
        inner_fold: int,
        validation_model_ids: Sequence[str],
        bundles: Mapping[str, FitBundle],
        fit_errors: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        """Score all successful specs on identical common administration/eval items."""

        required_spec_ids = {spec.spec_id for spec in self.specs}
        all_specs_fitted = set(bundles) == required_spec_ids and not fit_errors
        if not bundles:
            common_ids: set[str] = set()
        else:
            common_ids = set.intersection(
                *(set(bundle.bank.criterion_ids) for bundle in bundles.values())
            )
        administration_scenarios = set(self.split_audit["administration_scenario_ids"])
        evaluation_scenarios = set(self.split_audit["evaluation_scenario_ids"])
        if bundles:
            exemplar = next(iter(bundles.values())).bank
            scenario_by_item = dict(zip(exemplar.criterion_ids, exemplar.scenario_ids, strict=True))
            common_admin_ids = {
                item
                for item in common_ids
                if scenario_by_item.get(item) in administration_scenarios
            }
            common_eval_ids = {
                item for item in common_ids if scenario_by_item.get(item) in evaluation_scenarios
            }
        else:
            common_admin_ids = set()
            common_eval_ids = set()
        support_hash = _canonical_hash(
            {
                "administration": sorted(common_admin_ids),
                "evaluation": sorted(common_eval_ids),
            }
        )
        context = {
            "repeat": repeat,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "validation_model_ids": sorted(map(str, validation_model_ids)),
            "fit_cache_keys": {
                spec_id: bundle.cache_key for spec_id, bundle in sorted(bundles.items())
            },
            "fit_errors": dict(sorted(fit_errors.items())),
            "all_specs_fitted": all_specs_fitted,
            "common_support_sha256": support_hash,
        }
        checkpoint = self._checkpoint_path(
            "inner_calibration_evidence", repeat, outer_fold, f"inner_{inner_fold:02d}"
        )
        cached = self._load_rows_checkpoint(
            checkpoint, stage="inner_calibration_evidence", context=context
        )
        if cached is not None:
            return cached

        rows: list[dict[str, Any]] = []
        for spec in self.specs:
            bundle = bundles.get(spec.spec_id)
            if bundle is None:
                for model in sorted(map(str, validation_model_ids)):
                    rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "model": model,
                            "model_family": self.split_audit["model_to_family"][model],
                            "spec_id": spec.spec_id,
                            "family": spec.family,
                            "ridge": spec.ridge,
                            "log_a_shrinkage": spec.log_a_shrinkage,
                            "status": "fit_error",
                            "error": fit_errors.get(spec.spec_id, "fit unavailable"),
                            "fit_eligible": False,
                            "fit_cache_key": None,
                            "all_specs_fitted": all_specs_fitted,
                            "common_support_verified": False,
                            "common_support_sha256": support_hash,
                            "n_common_admin_items": len(common_admin_ids),
                            "n_common_evaluation_items": len(common_eval_ids),
                            "n_cells": 0,
                            "log_loss_sum": 0.0,
                            "brier_sum": 0.0,
                            "model_log_loss": None,
                        }
                    )
                continue
            if not common_admin_ids or not common_eval_ids:
                for model in sorted(map(str, validation_model_ids)):
                    rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "model": model,
                            "model_family": self.split_audit["model_to_family"][model],
                            "spec_id": spec.spec_id,
                            "family": spec.family,
                            "ridge": spec.ridge,
                            "log_a_shrinkage": spec.log_a_shrinkage,
                            "status": "common_support_error",
                            "error": "no common administration/evaluation support",
                            "fit_eligible": True,
                            "fit_cache_key": bundle.cache_key,
                            "all_specs_fitted": all_specs_fitted,
                            "common_support_verified": False,
                            "common_support_sha256": support_hash,
                            "n_common_admin_items": len(common_admin_ids),
                            "n_common_evaluation_items": len(common_eval_ids),
                            "n_cells": 0,
                            "log_loss_sum": 0.0,
                            "brier_sum": 0.0,
                            "model_log_loss": None,
                            "observed_common_cells_sha256": None,
                        }
                    )
                continue
            common_bank = _subset_bank_by_criteria(bundle.bank, common_ids)
            admin_indices = np.asarray(
                [
                    index
                    for index, item in enumerate(common_bank.criterion_ids)
                    if item in common_admin_ids
                ],
                dtype=int,
            )
            eval_indices = np.asarray(
                [
                    index
                    for index, item in enumerate(common_bank.criterion_ids)
                    if item in common_eval_ids
                ],
                dtype=int,
            )
            for model in sorted(map(str, validation_model_ids)):
                base = {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "model": model,
                    "model_family": self.split_audit["model_to_family"][model],
                    "spec_id": spec.spec_id,
                    "family": spec.family,
                    "ridge": spec.ridge,
                    "log_a_shrinkage": spec.log_a_shrinkage,
                    "fit_eligible": True,
                    "fit_cache_key": bundle.cache_key,
                    "all_specs_fitted": all_specs_fitted,
                    "fit_bank_items": len(bundle.bank.criterion_ids),
                    "fit_dropped_negative_items": len(bundle.bank.dropped_negative_items),
                    "common_support_verified": all_specs_fitted,
                    "common_support_sha256": support_hash,
                    "n_common_admin_items": len(common_admin_ids),
                    "n_common_evaluation_items": len(common_eval_ids),
                }
                try:
                    response = scat.responses_for_bank(self.matrix.loc[model], common_bank)
                    estimate = scat.batch_eap(
                        response,
                        common_bank.A,
                        common_bank.b,
                        bundle.quadrature,
                        item_indices=admin_indices,
                    )
                    stats = v1.prediction_sufficient_statistics(
                        self.matrix.loc[model], common_bank, eval_indices, estimate.theta
                    )
                    if int(stats["n_cells"]) <= 0:
                        raise V3Phase3Error("no observed common disjoint evaluation cells")
                    observed_eval = [
                        common_bank.criterion_ids[index]
                        for index in eval_indices
                        if math.isfinite(float(response[index]))
                    ]
                    rows.append(
                        {
                            **base,
                            "status": "ok",
                            "error": "",
                            **stats,
                            "model_log_loss": float(stats["log_loss_sum"]) / int(stats["n_cells"]),
                            "observed_common_cells_sha256": _canonical_hash(sorted(observed_eval)),
                        }
                    )
                except (
                    V3Phase3Error,
                    scat.OfflineStudyError,
                    ValueError,
                    np.linalg.LinAlgError,
                ) as error:
                    rows.append(
                        {
                            **base,
                            "status": "prediction_error",
                            "error": f"{type(error).__name__}: {error}",
                            "n_cells": 0,
                            "log_loss_sum": 0.0,
                            "brier_sum": 0.0,
                            "model_log_loss": None,
                            "observed_common_cells_sha256": None,
                        }
                    )
        # Every successful spec/model pair must carry the exact same held-out cell
        # hash; this catches accidental support differences despite item intersection.
        for model in sorted(map(str, validation_model_ids)):
            hashes = {
                str(row["observed_common_cells_sha256"])
                for row in rows
                if row["model"] == model and row["status"] == "ok"
            }
            if len(hashes) > 1:
                raise V3Phase3Error(
                    f"inner {repeat}/{outer_fold}/{inner_fold}/{model}: cell support differs"
                )
        self._write_rows_checkpoint(
            checkpoint,
            stage="inner_calibration_evidence",
            context=context,
            rows=rows,
        )
        return rows

    def _select_panel(
        self, repeat: int, outer: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        outer_fold = int(outer["outer_fold"])
        outer_test = set(map(str, outer["test_model_ids"]))
        evidence_rows: list[dict[str, Any]] = []
        expected_validation_by_fold: dict[int, list[str]] = {}
        for inner in outer["inner_folds"]:
            inner_fold = int(inner["inner_fold"])
            expected_validation_by_fold[inner_fold] = sorted(
                map(str, inner["validation_model_ids"])
            )
            bundles: dict[str, FitBundle] = {}
            fit_errors: dict[str, str] = {}
            for spec in self.specs:
                try:
                    bundles[spec.spec_id] = self.fit_or_load(
                        repeat=repeat,
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        role=(
                            f"repeat_{repeat:02d}/outer_{outer_fold:02d}/"
                            f"inner_{inner_fold:02d}/{spec.spec_id}"
                        ),
                        training_model_ids=inner["fit_model_ids"],
                        forbidden_model_ids=(
                            outer_test | set(map(str, inner["validation_model_ids"]))
                        ),
                        spec=spec,
                    )
                except (
                    V3Phase3Error,
                    v1.NestedCVError,
                    scat.OfflineStudyError,
                    cm.CalibrationError,
                    ValueError,
                    np.linalg.LinAlgError,
                ) as error:
                    fit_errors[spec.spec_id] = f"{type(error).__name__}: {error}"
            evidence_rows.extend(
                self._score_inner_common_support(
                    repeat=repeat,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    validation_model_ids=inner["validation_model_ids"],
                    bundles=bundles,
                    fit_errors=fit_errors,
                )
            )
        inner_evidence_audit = audit_complete_inner_evidence(
            evidence_rows,
            self.specs,
            expected_validation_by_fold,
        )
        aggregate = aggregate_calibration_evidence(
            evidence_rows, self.specs, self.split_audit["model_to_family"]
        )
        for row in aggregate:
            row.update(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "complete_inner_evidence": inner_evidence_audit["passed"],
                    "survivor_selection_used": False,
                }
            )
        if not inner_evidence_audit["passed"]:
            selected_spec = None
            annotated = aggregate
            selection = {}
            status = "selection_blocked_incomplete_inner_evidence"
            error = "; ".join(inner_evidence_audit["failed_checks"])
        else:
            try:
                selected_spec, annotated, selection = select_calibration_spec_one_se(
                    aggregate, self.specs
                )
                status = "selection_complete"
                error = ""
            except V3Phase3Error as selection_error:
                selected_spec = None
                annotated = aggregate
                selection = {}
                status = "selection_failed"
                error = str(selection_error)
        evidence_sha = _canonical_hash(evidence_rows)
        panel_id = f"repeat_{repeat:02d}_outer_{outer_fold:02d}"
        model_to_family = self.split_audit["model_to_family"]
        panel = {
            "panel_id": panel_id,
            "repeat": repeat,
            "outer_fold": outer_fold,
            "status": status,
            "error": error,
            "train_model_ids": list(outer["train_model_ids"]),
            "test_model_ids": list(outer["test_model_ids"]),
            "train_family_ids": sorted(
                {model_to_family[model] for model in outer["train_model_ids"]}
            ),
            "test_family_ids": sorted(
                {model_to_family[model] for model in outer["test_model_ids"]}
            ),
            "selected_spec_id": selected_spec.spec_id if selected_spec else None,
            "selected_calibration_specification": (
                selected_spec.canonical if selected_spec else None
            ),
            "inner_selection": {
                "evidence_sha256": evidence_sha,
                "aggregate_evidence_sha256": _canonical_hash(annotated),
                "empirical_best_cache_key": selection.get("empirical_best_cache_key"),
                "one_se_cutoff": selection.get("one_se_cutoff"),
                "within_one_se_cache_keys": selection.get("within_one_se_cache_keys", []),
                "selection_basis": selection.get("selection_basis"),
                "complete_inner_evidence_audit": inner_evidence_audit,
                "require_all_specs_every_inner_fold": True,
                "survivor_selection_allowed": False,
                "survivor_selection_used": False,
                "used_outer_outcomes": False,
                "used_cat_outcomes": False,
            },
            "outer_fit": None,
            "outer_evaluation": {"status": "not_run"},
        }
        return panel, [{**row, "panel_id": panel_id} for row in annotated]

    def evaluate_policy_panel(
        self,
        *,
        repeat: int,
        outer_fold: int,
        test_model_ids: Sequence[str],
        bundle: FitBundle,
    ) -> list[dict[str, Any]]:
        """Evaluate every frozen policy; sensitivities never affect selection."""

        administration_bank = v1.subset_fitted_bank(
            bundle.bank, self.split_audit["administration_scenario_ids"]
        )
        evaluation_indices = v1.evaluation_item_indices(
            bundle.bank, self.split_audit["evaluation_scenario_ids"]
        )
        rows: list[dict[str, Any]] = []
        for policy in self.policies:
            for model in sorted(map(str, test_model_ids)):
                row = v1.evaluate_model_pair(
                    model=model,
                    row=self.matrix.loc[model],
                    full_bank=bundle.bank,
                    administration_bank=administration_bank,
                    evaluation_indices=evaluation_indices,
                    scenario_records=self.scenario_records,
                    quadrature=bundle.quadrature,
                    candidate=policy.as_candidate(),
                    seed=self.args.seed,
                    top_n=self.args.top_n,
                    maximum_scenarios=self.args.max_scenarios,
                    minimum_scored_criteria=self.args.minimum_scored_criteria,
                    mwle_ridge=self.args.mwle_ridge,
                )
                row.update(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "panel_id": f"repeat_{repeat:02d}_outer_{outer_fold:02d}",
                        "model_family": self.split_audit["model_to_family"][model],
                        "candidate_id": policy.policy_id,
                        "policy_id": policy.policy_id,
                        "policy_role": policy.role,
                        "fit_cache_key": bundle.cache_key,
                        "selected_spec_id": bundle.exact_spec.spec_id,
                        "selected_spec_cache_key": bundle.exact_spec.canonical["cache_key"],
                    }
                )
                rows.append(row)
        expected = {
            (policy.policy_id, model)
            for policy in self.policies
            for model in sorted(map(str, test_model_ids))
        }
        observed = {(str(row["policy_id"]), str(row["model"])) for row in rows}
        if observed != expected or len(rows) != len(expected):
            raise V3Phase3Error(f"outer {repeat}/{outer_fold}: policy/model coverage is incomplete")
        return rows

    def _evaluate_outer_panel(
        self, panel: dict[str, Any], outer: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        if panel["selected_spec_id"] is None:
            panel["outer_evaluation"] = {
                "status": "not_run_selection_failed",
                "primary_rows_complete": False,
            }
            return []
        repeat = int(panel["repeat"])
        outer_fold = int(panel["outer_fold"])
        spec = next(item for item in self.specs if item.spec_id == panel["selected_spec_id"])
        try:
            bundle = self.fit_or_load(
                repeat=repeat,
                outer_fold=outer_fold,
                inner_fold=None,
                role=f"repeat_{repeat:02d}/outer_{outer_fold:02d}/selected_outer_fit",
                training_model_ids=outer["train_model_ids"],
                forbidden_model_ids=outer["test_model_ids"],
                spec=spec,
            )
        except (
            V3Phase3Error,
            v1.NestedCVError,
            scat.OfflineStudyError,
            cm.CalibrationError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            panel["status"] = "outer_fit_failed"
            panel["error"] = f"{type(error).__name__}: {error}"
            panel["outer_evaluation"] = {
                "status": "not_run_outer_fit_failed",
                "primary_rows_complete": False,
            }
            return []
        fit_manifest = bundle.cache_dir / "fit_manifest.json"
        panel["outer_fit"] = {
            "cache_key": bundle.cache_key,
            "cache_dir": _display_path(bundle.cache_dir),
            "manifest_path": _display_path(fit_manifest),
            "manifest_sha256": _sha256(fit_manifest),
            "training_model_ids": list(bundle.training_model_ids),
            "spec_id": spec.spec_id,
            "calibration_specification": spec.canonical,
        }
        context = {
            "repeat": repeat,
            "outer_fold": outer_fold,
            "test_model_ids": sorted(map(str, outer["test_model_ids"])),
            "fit_cache_key": bundle.cache_key,
            "selected_spec_id": spec.spec_id,
            "policy_ids": [policy.policy_id for policy in self.policies],
        }
        checkpoint = self._checkpoint_path(
            "outer_policy_evaluation", repeat, outer_fold, "all_policies"
        )
        rows = self._load_rows_checkpoint(
            checkpoint, stage="outer_policy_evaluation", context=context
        )
        if rows is None:
            rows = self.evaluate_policy_panel(
                repeat=repeat,
                outer_fold=outer_fold,
                test_model_ids=outer["test_model_ids"],
                bundle=bundle,
            )
            self._write_rows_checkpoint(
                checkpoint,
                stage="outer_policy_evaluation",
                context=context,
                rows=rows,
            )
        expected_primary = set(map(str, outer["test_model_ids"]))
        observed_primary = {
            str(row["model"]) for row in rows if row.get("policy_id") == PRIMARY_POLICY_ID
        }
        if observed_primary != expected_primary:
            raise V3Phase3Error(f"outer {repeat}/{outer_fold}: primary coverage incomplete")
        panel["status"] = "evaluation_complete"
        panel["outer_evaluation"] = {
            "status": "complete",
            "checkpoint_path": _display_path(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "n_rows": len(rows),
            "primary_rows_complete": True,
        }
        return rows

    def _gates(self) -> dict[str, Any]:
        gates = dict(self.config.get("selection_gates") or {})
        gates["require_paired_ci_favors_cat"] = bool(
            gates.get("require_paired_family_bootstrap_ci_favors_cat", True)
        )
        return gates

    def _gate_scope_contract(self) -> dict[str, Any]:
        gates = self._gates()
        return {
            "outer_panel": {
                "decision_role": "applied_point_estimate_gates",
                "inference_unit": gates["outer_panel_inference_unit"],
                "applied_gate_names": list(PANEL_APPLIED_GATE_NAMES),
                "diagnostic_only_gate_names": list(PANEL_DIAGNOSTIC_GATE_NAMES),
                "confidence_intervals_affect_decision": False,
            },
            "repetition": {
                "decision_role": "applied_all_absolute_gates",
                "inference_unit": gates["repetition_inference_unit"],
                "expected_unique_oof_tutors": EXPECTED_MODELS,
                "applied_gate_names": list(REPETITION_APPLIED_GATE_NAMES),
                "diagnostic_only_gate_names": [],
                "nominal_precision_lower_95_ci_threshold": float(
                    gates["minimum_nominal_precision_lower_95_ci"]
                ),
            },
            "cross_repeat": {
                "decision_role": gates["cross_repeat_gate_role"],
                "inference_unit": ("52_unique_tutors_after_within_tutor_mean_across_5_repetitions"),
                "raw_model_repeat_rows": EXPECTED_MODELS * EXPECTED_REPEATS,
                "raw_model_repeat_rows_treated_as_independent": False,
            },
        }

    def _metric_seed(self, label: str) -> int:
        return v1._bootstrap_seed(self.args.seed, f"phase3-v3:{label}")

    def _gate_rows(
        self, outer_rows: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        gate_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        gates = self._gates()
        gate_scope_contract = self._gate_scope_contract()
        gate_scope_contract_sha256 = _canonical_hash(gate_scope_contract)
        expected_panel_models = {
            (int(repetition["repeat"]), int(outer["outer_fold"])): set(
                map(str, outer["test_model_ids"])
            )
            for repetition in self.split_audit["repetitions"]
            for outer in repetition["outer_folds"]
        }
        expected_repetition_models = set(map(str, self.matrix.index))

        def scoped_fields(
            *,
            metrics: Mapping[str, Any],
            scope: str,
            policy: Policy,
            inference_complete: bool,
            n_rows: int,
            n_unique_models: int,
            expected_unique_models: int,
        ) -> dict[str, Any]:
            evaluation = apply_scoped_absolute_gates(metrics, gates, scope=scope)
            if policy.role == "primary":
                applied_checks = dict(evaluation["applied_checks"])
                applied_checks["inference_scope_complete"] = inference_complete
                failures = list(evaluation["failures"])
                if not inference_complete:
                    failures.append("inference_scope_complete")
                diagnostic_checks = dict(evaluation["diagnostic_checks"])
                diagnostic_failures = list(evaluation["diagnostic_failures"])
                passed: bool | None = not failures
                gate_status = "pass" if passed else "fail"
                applied_names = [
                    *evaluation["applied_gate_names"],
                    "inference_scope_complete",
                ]
                diagnostic_names = list(evaluation["diagnostic_gate_names"])
            else:
                applied_checks = {}
                failures = []
                diagnostic_checks = {
                    **evaluation["applied_checks"],
                    **evaluation["diagnostic_checks"],
                    "inference_scope_complete": inference_complete,
                }
                diagnostic_failures = [
                    name for name, value in diagnostic_checks.items() if not value
                ]
                passed = None
                gate_status = "diagnostic_only_non_promotable"
                applied_names = []
                diagnostic_names = [*ABSOLUTE_GATE_NAMES, "inference_scope_complete"]
            return {
                "all_gates_pass": passed,
                "gate_status": gate_status,
                "failed_gates": json.dumps(failures),
                "gate_checks": json.dumps(applied_checks, sort_keys=True),
                "applied_gate_names": json.dumps(applied_names),
                "applied_gate_checks": json.dumps(applied_checks, sort_keys=True),
                "diagnostic_only_gate_names": json.dumps(diagnostic_names),
                "diagnostic_gate_checks": json.dumps(diagnostic_checks, sort_keys=True),
                "diagnostic_failures": json.dumps(diagnostic_failures),
                "inference_unit": gate_scope_contract[scope]["inference_unit"],
                "n_inference_rows": n_rows,
                "n_unique_tutor_models": n_unique_models,
                "expected_unique_tutor_models": expected_unique_models,
                "inference_scope_complete": inference_complete,
                "model_repeat_rows_treated_as_independent": False,
                "gate_scope_contract_sha256": gate_scope_contract_sha256,
            }

        for repeat in range(EXPECTED_REPEATS):
            repeat_rows = [row for row in outer_rows if int(row["repeat"]) == repeat]
            for outer_fold in range(EXPECTED_OUTER_FOLDS):
                panel_rows = [row for row in repeat_rows if int(row["outer_fold"]) == outer_fold]
                for policy in self.policies:
                    subset = [row for row in panel_rows if row.get("policy_id") == policy.policy_id]
                    metrics = aggregate_policy_rows_clustered(
                        subset,
                        self.split_audit["model_to_family"],
                        seed=self._metric_seed(f"panel:{repeat}:{outer_fold}:{policy.policy_id}"),
                        replicates=self.args.metric_bootstrap_replicates,
                        label=f"panel:{repeat}:{outer_fold}:{policy.policy_id}",
                    )
                    observed_models = [str(row["model"]) for row in subset]
                    expected_models = expected_panel_models[(repeat, outer_fold)]
                    inference_complete = (
                        len(observed_models) == len(expected_models)
                        and len(set(observed_models)) == len(observed_models)
                        and set(observed_models) == expected_models
                    )
                    gate_rows.append(
                        {
                            "scope": "outer_panel",
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "policy_id": policy.policy_id,
                            "policy_role": policy.role,
                            **metrics,
                            **scoped_fields(
                                metrics=metrics,
                                scope="outer_panel",
                                policy=policy,
                                inference_complete=inference_complete,
                                n_rows=len(subset),
                                n_unique_models=len(set(observed_models)),
                                expected_unique_models=len(expected_models),
                            ),
                            "sensitivity_can_promote": False,
                        }
                    )
                    for mode in ("cat", "baseline"):
                        prediction_rows.append(
                            {
                                "repeat": repeat,
                                "outer_fold": outer_fold,
                                "policy_role": policy.role,
                                **v1.aggregate_disjoint_predictions(
                                    subset,
                                    fold=f"repeat_{repeat:02d}_outer_{outer_fold:02d}",
                                    candidate_id=policy.policy_id,
                                    mode=mode,
                                ),
                            }
                        )
            for policy in self.policies:
                subset = [row for row in repeat_rows if row.get("policy_id") == policy.policy_id]
                metrics = aggregate_policy_rows_clustered(
                    subset,
                    self.split_audit["model_to_family"],
                    seed=self._metric_seed(f"repeat:{repeat}:{policy.policy_id}"),
                    replicates=self.args.metric_bootstrap_replicates,
                    label=f"repeat:{repeat}:{policy.policy_id}",
                )
                observed_models = [str(row["model"]) for row in subset]
                inference_complete = (
                    len(observed_models) == EXPECTED_MODELS
                    and len(set(observed_models)) == EXPECTED_MODELS
                    and set(observed_models) == expected_repetition_models
                )
                gate_rows.append(
                    {
                        "scope": "repetition",
                        "repeat": repeat,
                        "outer_fold": None,
                        "policy_id": policy.policy_id,
                        "policy_role": policy.role,
                        **metrics,
                        **scoped_fields(
                            metrics=metrics,
                            scope="repetition",
                            policy=policy,
                            inference_complete=inference_complete,
                            n_rows=len(subset),
                            n_unique_models=len(set(observed_models)),
                            expected_unique_models=EXPECTED_MODELS,
                        ),
                        "sensitivity_can_promote": False,
                    }
                )
        return gate_rows, prediction_rows

    def _aggregate_repeated_models(
        self, outer_rows: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collapse the five OOF estimates per model before cross-repeat inference."""

        numeric_fields = (
            "theta_reference",
            "theta_cat_mwle",
            "theta_baseline_mwle",
            "cat_scenarios_administered",
            "baseline_scenarios_administered",
            "cat_eval_log_loss_sum",
            "cat_eval_brier_sum",
            "cat_eval_correct_count",
            "cat_eval_n_cells",
            "cat_eval_observed_pass_rate",
            "cat_eval_predicted_pass_rate",
            "baseline_eval_log_loss_sum",
            "baseline_eval_brier_sum",
            "baseline_eval_correct_count",
            "baseline_eval_n_cells",
            "baseline_eval_observed_pass_rate",
            "baseline_eval_predicted_pass_rate",
        )
        boolean_fields = (
            "cat_replay_success",
            "baseline_replay_success",
            "cat_mwle_converged",
            "baseline_mwle_converged",
            "cat_precision_reached",
            "baseline_precision_reached",
        )
        per_model: list[dict[str, Any]] = []
        for policy in self.policies:
            policy_rows = [row for row in outer_rows if row.get("policy_id") == policy.policy_id]
            for model in sorted({str(row["model"]) for row in policy_rows}):
                rows = [row for row in policy_rows if str(row["model"]) == model]
                record: dict[str, Any] = {
                    "policy_id": policy.policy_id,
                    "policy_role": policy.role,
                    "model": model,
                    "model_family": self.split_audit["model_to_family"][model],
                    "n_repetitions": len({int(row["repeat"]) for row in rows}),
                    "expected_repetitions": EXPECTED_REPEATS,
                    "repeat_rows_treated_as_independent": False,
                }
                for field in numeric_fields:
                    values = [
                        float(row[field])
                        for row in rows
                        if row.get(field) is not None and math.isfinite(float(row[field]))
                    ]
                    record[f"mean_{field}"] = float(np.mean(values)) if values else None
                for field in boolean_fields:
                    record[f"rate_{field}"] = (
                        float(np.mean([bool(row.get(field)) for row in rows])) if rows else None
                    )
                per_model.append(record)

        cross_metrics: list[dict[str, Any]] = []
        for policy in self.policies:
            rows = [row for row in per_model if row["policy_id"] == policy.policy_id]
            complete = [row for row in rows if int(row["n_repetitions"]) == EXPECTED_REPEATS]
            reference = np.asarray(
                [
                    float(row["mean_theta_reference"])
                    if row.get("mean_theta_reference") is not None
                    else np.nan
                    for row in complete
                ]
            )
            estimate = np.asarray(
                [
                    float(row["mean_theta_cat_mwle"])
                    if row.get("mean_theta_cat_mwle") is not None
                    else np.nan
                    for row in complete
                ]
            )
            recovery = scenario_cv.recovery_stats(reference, estimate)
            pseudo_rows: list[dict[str, Any]] = []
            for row in complete:
                pseudo_rows.append(
                    {
                        "model": row["model"],
                        "cat_replay_success": row.get("mean_cat_scenarios_administered")
                        is not None,
                        "baseline_replay_success": row.get("mean_baseline_scenarios_administered")
                        is not None,
                        "cat_mwle_converged": row.get("mean_theta_cat_mwle") is not None,
                        "theta_reference": row.get("mean_theta_reference"),
                        "theta_cat_mwle": row.get("mean_theta_cat_mwle"),
                        "cat_scenarios_administered": row.get("mean_cat_scenarios_administered"),
                        "baseline_scenarios_administered": row.get(
                            "mean_baseline_scenarios_administered"
                        ),
                    }
                )
            clustered = _clustered_metric_intervals(
                pseudo_rows,
                self.split_audit["model_to_family"],
                seed=self._metric_seed(f"cross-repeat:{policy.policy_id}"),
                replicates=self.args.metric_bootstrap_replicates,
            )
            cat_lengths = np.asarray(
                [
                    float(row["mean_cat_scenarios_administered"])
                    for row in complete
                    if row.get("mean_cat_scenarios_administered") is not None
                    and row.get("mean_baseline_scenarios_administered") is not None
                ]
            )
            baseline_lengths = np.asarray(
                [
                    float(row["mean_baseline_scenarios_administered"])
                    for row in complete
                    if row.get("mean_cat_scenarios_administered") is not None
                    and row.get("mean_baseline_scenarios_administered") is not None
                ]
            )
            reduction = (
                1.0 - float(cat_lengths.mean()) / float(baseline_lengths.mean())
                if cat_lengths.size and float(baseline_lengths.mean()) > 0
                else None
            )
            cross_metrics.append(
                {
                    "policy_id": policy.policy_id,
                    "policy_role": policy.role,
                    "n_models": len(rows),
                    "n_models_with_all_repetitions": len(complete),
                    "n_families": len({str(row["model_family"]) for row in complete}),
                    "aggregation_unit": "model_after_mean_across_repetitions",
                    "bootstrap_unit": "tutor_family",
                    "model_repeat_rows_treated_as_independent": False,
                    "recovery_n": recovery["n"],
                    "recovery_correlation": recovery["r"],
                    "recovery_slope": recovery["slope"],
                    "recovery_mae": recovery["mae"],
                    "recovery_bias": recovery["bias"],
                    "scenario_reduction_vs_random": reduction,
                    **clustered,
                    "gate_role": "diagnostic_only_cross_repeat_summary",
                }
            )
        return per_model, cross_metrics

    def _runtime_manifest_payload(self) -> dict[str, Any]:
        return {
            "seed": self.args.seed,
            "top_n": self.args.top_n,
            "max_scenarios": self.args.max_scenarios,
            "minimum_scored_criteria": self.args.minimum_scored_criteria,
            "max_iter": self.args.max_iter,
            "tol": self.args.tol,
            "fit_parameter_tolerance": self.numerical["fit_parameter_tolerance"],
            "fit_consecutive_convergence_passes": self.numerical[
                "fit_consecutive_convergence_passes"
            ],
            "fit_convergence_mode": self.numerical["fit_convergence_mode"],
            "fit_quadrature_method": self.numerical["fit_quadrature_method"],
            "fit_linear_bound": self.numerical["fit_linear_bound"],
            "mwle_ridge": self.args.mwle_ridge,
            "max_grid_nodes": self.args.max_grid_nodes,
            "negative_policy": self.args.negative_policy,
            "metric_bootstrap_replicates": self.args.metric_bootstrap_replicates,
        }

    def _output_inventory(self, required_outputs: Sequence[str]) -> dict[str, dict[str, str]]:
        inventory: dict[str, dict[str, str]] = {}
        for name in required_outputs:
            if name == "manifest.json":
                continue
            path = self.output_dir / name
            if not path.is_file():
                raise V3Phase3Error(f"required terminal output was not written: {name}")
            inventory[name] = {
                "path": _display_path(path),
                "sha256": _sha256(path),
            }
        return inventory

    def _write_terminal_selection_failure_outputs(
        self,
        *,
        panels: Sequence[Mapping[str, Any]],
        inner_results: Sequence[Mapping[str, Any]],
        started_at: str,
    ) -> None:
        """Finalize an honest selection-only failure without opening outer outcomes."""

        panel_list = [dict(panel) for panel in panels]
        selected_payload = {
            "schema_version": SELECTED_SPECS_SCHEMA,
            "study_signature": self.study_signature,
            "config": {
                "path": _display_path(self.config_path),
                "sha256": _sha256(self.config_path),
            },
            "split_manifest": {
                "path": _display_path(self.split_path),
                "sha256": _sha256(self.split_path),
            },
            "input_hashes": self.input_hashes,
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
            "primary_policy": asdict(self.policies[0]),
            "sensitivities": [asdict(policy) for policy in self.policies[1:]],
            "sensitivity_promotion_allowed": False,
            "require_all_specs_every_inner_fold": True,
            "survivor_selection_allowed": False,
            "terminal_selection_only": True,
            "outer_outcomes_opened": False,
            "panels": panel_list,
        }
        selected_path = self.output_dir / "selected_calibration_specs.json"
        _atomic_json(selected_path, selected_payload)

        inner_frame = pd.DataFrame(inner_results)
        if not inner_frame.empty:
            inner_frame = inner_frame.sort_values(
                ["repeat", "outer_fold", "simplicity_rank"], kind="stable"
            )
        _atomic_csv(self.output_dir / "inner_calibration_model_results.csv", inner_frame)
        choices = {
            "schema_version": SCRIPT_SCHEMA,
            "status": "terminal_selection_incomplete",
            "selection_source": "inner-only common-item/common-cell disjoint log loss",
            "uncertainty_unit": "tutor_family_leave_one_out_jackknife",
            "outer_outcomes_used": False,
            "outer_outcomes_opened": False,
            "require_all_specs_every_inner_fold": True,
            "survivor_selection_allowed": False,
            "panels": [
                {
                    key: panel.get(key)
                    for key in (
                        "panel_id",
                        "repeat",
                        "outer_fold",
                        "status",
                        "error",
                        "selected_spec_id",
                        "selected_calibration_specification",
                        "inner_selection",
                    )
                }
                for panel in panel_list
            ],
        }
        _atomic_json(self.output_dir / "calibration_model_choices.json", choices)

        expected_panels = EXPECTED_REPEATS * EXPECTED_OUTER_FOLDS
        complete_panels = [
            panel
            for panel in panel_list
            if panel.get("status") == "selection_complete"
            and panel.get("selected_spec_id") is not None
        ]
        failed_panels = [
            panel
            for panel in panel_list
            if not (
                panel.get("status") == "selection_complete"
                and panel.get("selected_spec_id") is not None
            )
        ]
        status_counts = Counter(str(panel.get("status") or "missing") for panel in panel_list)
        valid_spec_fold_combinations = sum(
            int(
                (
                    (panel.get("inner_selection") or {}).get("complete_inner_evidence_audit") or {}
                ).get("valid_spec_fold_combinations", 0)
            )
            for panel in panel_list
        )
        expected_spec_fold_combinations = EXPECTED_TOTAL_SPEC_FOLD_BLOCKS
        decision = {
            "schema_version": DECISION_SCHEMA,
            "status": "fail_selection_incomplete",
            "decision_scope": "terminal_selection_only",
            "phase3_pass": False,
            "phase4_authorized": False,
            "study_signature": self.study_signature,
            "config_sha256": _sha256(self.config_path),
            "split_sha256": _sha256(self.split_path),
            "input_hashes": self.input_hashes,
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
            "gate_scope_contract": self._gate_scope_contract(),
            "terminal_stop": {
                "stage": "pre_outer_selection_gate",
                "reason": "one_or_more_panel_selections_incomplete",
                "preregistered_fail_closed_stop": True,
                "outer_outcomes_opened": False,
                "outer_policy_evaluation_called": False,
            },
            "calibration_selection": {
                "status": "incomplete_fail_closed",
                "expected_panels_denominator": expected_panels,
                "attempted_panels_numerator": len(panel_list),
                "complete_panels_numerator": len(complete_panels),
                "failed_panels_numerator": len(failed_panels),
                "complete_panel_fraction": (
                    len(complete_panels) / expected_panels if expected_panels else None
                ),
                "status_counts": dict(sorted(status_counts.items())),
                "failed_panel_ids": [str(panel.get("panel_id")) for panel in failed_panels],
                "failed_panel_statuses": {
                    str(panel.get("panel_id")): str(panel.get("status")) for panel in failed_panels
                },
                "require_all_specs_every_inner_fold": True,
                "survivor_selection_allowed": False,
                "expected_spec_fold_combinations_denominator": (expected_spec_fold_combinations),
                "valid_spec_fold_combinations_numerator": valid_spec_fold_combinations,
                "valid_spec_fold_fraction": (
                    valid_spec_fold_combinations / expected_spec_fold_combinations
                ),
            },
            "selected_calibration_specs": {
                "path": _display_path(selected_path),
                "sha256": _sha256(selected_path),
            },
            "outer_evaluation": {
                "status": "not_run_preregistered_selection_gate",
                "expected_outer_panels_denominator": expected_panels,
                "evaluated_outer_panels_numerator": 0,
                "expected_unique_oof_tutors_per_repetition_denominator": EXPECTED_MODELS,
                "evaluated_unique_oof_tutors_numerator": 0,
                "expected_model_repeat_rows_per_policy_denominator": (
                    EXPECTED_MODELS * EXPECTED_REPEATS
                ),
                "evaluated_model_repeat_rows_per_policy_numerator": 0,
                "cat_or_baseline_metrics_estimable": False,
            },
            "primary_policy": {
                "policy": asdict(self.policies[0]),
                "status": "not_evaluated_selection_gate_failed",
                "absolute_gates_applied": False,
            },
            "sensitivities": {
                "policies": [asdict(policy) for policy in self.policies[1:]],
                "status": "not_evaluated_selection_gate_failed",
                "diagnostic_only": True,
                "can_promote": False,
            },
            "failed_conditions": ["incomplete_calibration_spec_selection"],
            "phase4_not_run_by_this_driver": True,
            "limitations": [
                "no outer CAT, random-baseline, gate, or cross-repeat outcome was opened",
                "same-cohort internal development, not independent confirmation",
                "conditional on frozen Qwen labels not human-validated on InFoBench",
            ],
        }
        _atomic_json(self.output_dir / "phase3_decision.json", decision)
        _atomic_json(self.output_dir / "v3_decision.json", decision)

        manifest = self._base_manifest("phase3_terminal_selection_failed", created_at=started_at)
        manifest["completed_at"] = _utcnow()
        manifest["terminal_stop"] = decision["terminal_stop"]
        manifest["phase3_decision"] = {
            "status": decision["status"],
            "phase3_pass": False,
            "phase4_authorized": False,
        }
        manifest["runtime"] = self._runtime_manifest_payload()
        manifest["outputs"] = self._output_inventory(SELECTION_ONLY_REQUIRED_OUTPUTS)
        _atomic_json(self.output_dir / "manifest.json", manifest)

    def _write_phase3_outputs(
        self,
        *,
        panels: Sequence[Mapping[str, Any]],
        inner_results: Sequence[Mapping[str, Any]],
        outer_rows: Sequence[Mapping[str, Any]],
        started_at: str,
    ) -> None:
        panel_list = [dict(panel) for panel in panels]
        selected_payload = {
            "schema_version": SELECTED_SPECS_SCHEMA,
            "study_signature": self.study_signature,
            "config": {
                "path": _display_path(self.config_path),
                "sha256": _sha256(self.config_path),
            },
            "split_manifest": {
                "path": _display_path(self.split_path),
                "sha256": _sha256(self.split_path),
            },
            "input_hashes": self.input_hashes,
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
            "primary_policy": asdict(self.policies[0]),
            "sensitivities": [asdict(policy) for policy in self.policies[1:]],
            "sensitivity_promotion_allowed": False,
            "require_all_specs_every_inner_fold": True,
            "survivor_selection_allowed": False,
            "panels": panel_list,
        }
        selected_path = self.output_dir / "selected_calibration_specs.json"
        _atomic_json(selected_path, selected_payload)

        inner_frame = pd.DataFrame(inner_results)
        if not inner_frame.empty:
            inner_frame = inner_frame.sort_values(
                ["repeat", "outer_fold", "simplicity_rank"], kind="stable"
            )
        _atomic_csv(self.output_dir / "inner_calibration_model_results.csv", inner_frame)
        outer_frame = pd.DataFrame(outer_rows)
        if not outer_frame.empty:
            outer_frame = outer_frame.sort_values(
                ["repeat", "outer_fold", "policy_id", "model"], kind="stable"
            )
        _atomic_csv(self.output_dir / "outer_oof_per_model.csv", outer_frame)

        choices = {
            "schema_version": SCRIPT_SCHEMA,
            "selection_source": "inner-only common-item/common-cell disjoint log loss",
            "uncertainty_unit": "tutor_family_leave_one_out_jackknife",
            "outer_outcomes_used": False,
            "require_all_specs_every_inner_fold": True,
            "survivor_selection_allowed": False,
            "panels": [
                {
                    key: panel.get(key)
                    for key in (
                        "panel_id",
                        "repeat",
                        "outer_fold",
                        "status",
                        "error",
                        "selected_spec_id",
                        "selected_calibration_specification",
                        "inner_selection",
                    )
                }
                for panel in panel_list
            ],
        }
        _atomic_json(self.output_dir / "calibration_model_choices.json", choices)

        gate_rows, prediction_rows = self._gate_rows(outer_rows)
        gate_frame = pd.DataFrame(gate_rows)
        _atomic_csv(self.output_dir / "repeat_fold_gate_results.csv", gate_frame)
        repeat_frame = gate_frame[gate_frame["scope"] == "repetition"].copy()
        _atomic_csv(self.output_dir / "repeat_metrics.csv", repeat_frame)
        _atomic_csv(
            self.output_dir / "disjoint_prediction_metrics.csv",
            pd.DataFrame(prediction_rows),
        )

        duplicate_rows: list[dict[str, Any]] = []
        for panel in panel_list:
            subset = [row for row in outer_rows if row.get("panel_id") == panel["panel_id"]]
            for row in detect_duplicate_cat_paths(subset, self.policies):
                duplicate_rows.append(
                    {
                        "panel_id": panel["panel_id"],
                        "repeat": panel["repeat"],
                        "outer_fold": panel["outer_fold"],
                        **row,
                    }
                )
        _atomic_csv(self.output_dir / "duplicate_cat_paths.csv", pd.DataFrame(duplicate_rows))

        per_model, cross_metrics = self._aggregate_repeated_models(outer_rows)
        _atomic_csv(self.output_dir / "cross_repeat_per_model.csv", pd.DataFrame(per_model))
        _atomic_csv(self.output_dir / "cross_repeat_metrics.csv", pd.DataFrame(cross_metrics))

        selected_ids = [
            str(panel["selected_spec_id"])
            for panel in panel_list
            if panel.get("selected_spec_id") is not None
        ]
        counts = Counter(selected_ids)
        frequency_rows = [
            {
                "spec_id": spec.spec_id,
                "family": spec.family,
                "ridge": spec.ridge,
                "log_a_shrinkage": spec.log_a_shrinkage,
                "simplicity_rank": spec.simplicity_rank,
                "selected_outer_panels": counts[spec.spec_id],
                "selection_fraction_all_25_panels": counts[spec.spec_id]
                / (EXPECTED_REPEATS * EXPECTED_OUTER_FOLDS),
            }
            for spec in self.specs
        ]
        _atomic_csv(
            self.output_dir / "calibration_spec_selection_frequency.csv",
            pd.DataFrame(frequency_rows),
        )
        maximum = max(counts.values(), default=0)
        modes = sorted(spec_id for spec_id, count in counts.items() if count == maximum)
        unique_modal = modes[0] if len(modes) == 1 else None
        modal_fraction = maximum / (EXPECTED_REPEATS * EXPECTED_OUTER_FOLDS)
        stability_pass = unique_modal is not None and modal_fraction >= 0.80

        expected_models = set(map(str, self.matrix.index))
        coverage_by_repeat: dict[str, Any] = {}
        coverage_pass = True
        for repeat in range(EXPECTED_REPEATS):
            primary = [
                row
                for row in outer_rows
                if int(row["repeat"]) == repeat and row.get("policy_id") == PRIMARY_POLICY_ID
            ]
            models = [str(row["model"]) for row in primary]
            valid = len(models) == EXPECTED_MODELS and set(models) == expected_models
            coverage_pass = coverage_pass and valid
            coverage_by_repeat[str(repeat)] = {
                "n_primary_rows": len(models),
                "n_unique_models": len(set(models)),
                "all_52_models_exactly_once": valid,
            }
        all_panels_evaluated = len(panel_list) == 25 and all(
            panel.get("status") == "evaluation_complete" for panel in panel_list
        )
        coverage_pass = coverage_pass and all_panels_evaluated

        primary_panel_gates = [
            row
            for row in gate_rows
            if row["scope"] == "outer_panel" and row["policy_id"] == PRIMARY_POLICY_ID
        ]
        primary_repeat_gates = [
            row
            for row in gate_rows
            if row["scope"] == "repetition" and row["policy_id"] == PRIMARY_POLICY_ID
        ]
        all_panel_gates = len(primary_panel_gates) == 25 and all(
            row.get("all_gates_pass") is True for row in primary_panel_gates
        )
        all_repeat_gates = len(primary_repeat_gates) == 5 and all(
            row.get("all_gates_pass") is True for row in primary_repeat_gates
        )
        all_inner_evidence_complete = len(panel_list) == 25 and all(
            ((panel.get("inner_selection") or {}).get("complete_inner_evidence_audit") or {}).get(
                "passed"
            )
            is True
            for panel in panel_list
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
        phase3_pass = not failures
        decision = {
            "schema_version": DECISION_SCHEMA,
            "status": "pass" if phase3_pass else "fail",
            "phase3_pass": phase3_pass,
            "phase4_authorized": phase3_pass,
            "study_signature": self.study_signature,
            "config_sha256": _sha256(self.config_path),
            "split_sha256": _sha256(self.split_path),
            "input_hashes": self.input_hashes,
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
            "gate_scope_contract": self._gate_scope_contract(),
            "inner_selection_fail_closed": {
                "require_all_specs_every_inner_fold": True,
                "expected_spec_fold_combinations_per_panel": (EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL),
                "survivor_selection_allowed": False,
                "all_panel_selections_locked_before_outer_outcomes": True,
                "all_25_panels_complete": all_inner_evidence_complete,
            },
            "selected_calibration_specs": {
                "path": _display_path(selected_path),
                "sha256": _sha256(selected_path),
            },
            "coverage": {
                "all_25_panels_evaluated": all_panels_evaluated,
                "all_52_models_each_repetition": coverage_pass,
                "by_repetition": coverage_by_repeat,
                "model_repeat_rows_treated_as_independent": False,
            },
            "calibration_stability": {
                "unique_modal_spec_id": unique_modal,
                "tied_modal_spec_ids": modes if len(modes) != 1 else [],
                "modal_count": maximum,
                "modal_fraction_all_25_panels": modal_fraction,
                "required_fraction": 0.80,
                "passed": stability_pass,
            },
            "primary_policy": {
                "policy": asdict(self.policies[0]),
                "all_outer_panels_pass": all_panel_gates,
                "all_repetitions_pass": all_repeat_gates,
                "failed_panel_ids": [
                    f"repeat_{int(row['repeat']):02d}_outer_{int(row['outer_fold']):02d}"
                    for row in primary_panel_gates
                    if row.get("all_gates_pass") is not True
                ],
                "failed_repetitions": [
                    int(row["repeat"])
                    for row in primary_repeat_gates
                    if row.get("all_gates_pass") is not True
                ],
            },
            "sensitivities": {
                "policies": [asdict(policy) for policy in self.policies[1:]],
                "diagnostic_only": True,
                "can_promote": False,
                "affected_phase3_decision": False,
            },
            "failed_conditions": failures,
            "phase4_not_run_by_this_driver": True,
            "limitations": [
                "same-cohort internal development, not independent confirmation",
                "conditional on frozen Qwen labels not human-validated on InFoBench",
            ],
        }
        _atomic_json(self.output_dir / "phase3_decision.json", decision)
        _atomic_json(self.output_dir / "v3_decision.json", decision)

        # Repeat-local mirrors make each repetition independently auditable while
        # the combined files remain the canonical cross-repeat evidence.
        for repeat in range(EXPECTED_REPEATS):
            repeat_dir = self.output_dir / f"repeat_{repeat:02d}"
            repeat_inner = inner_frame[inner_frame["repeat"] == repeat]
            repeat_outer = (
                outer_frame[outer_frame["repeat"] == repeat]
                if "repeat" in outer_frame.columns
                else outer_frame.copy()
            )
            _atomic_csv(repeat_dir / "inner_calibration_model_results.csv", repeat_inner)
            _atomic_csv(repeat_dir / "outer_oof_per_model.csv", repeat_outer)
            _atomic_json(
                repeat_dir / "calibration_model_choices.json",
                {
                    **{key: value for key, value in choices.items() if key != "panels"},
                    "repeat": repeat,
                    "panels": [row for row in choices["panels"] if row["repeat"] == repeat],
                },
            )

        manifest = self._base_manifest("phase3_complete", created_at=started_at)
        manifest["completed_at"] = _utcnow()
        manifest["phase3_decision"] = {
            "status": decision["status"],
            "phase3_pass": phase3_pass,
            "phase4_authorized": phase3_pass,
        }
        manifest["runtime"] = self._runtime_manifest_payload()
        manifest["outputs"] = self._output_inventory(REQUIRED_OUTPUTS)
        _atomic_json(self.output_dir / "manifest.json", manifest)

    def _fold_assignment_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCRIPT_SCHEMA,
            "source_split_manifest": _display_path(self.split_path),
            "source_split_sha256": _sha256(self.split_path),
            "leakage_audit": "passed",
            "repetitions": self.split_audit["repetitions"],
            "model_to_family": self.split_audit["model_to_family"],
            "scenario_split": {
                "administration_scenario_ids": self.split_audit["administration_scenario_ids"],
                "evaluation_scenario_ids": self.split_audit["evaluation_scenario_ids"],
            },
        }

    def precompute_panel(self, repeat: int, outer_fold: int) -> int:
        """Compute one inner-selection panel without ever opening outer outcomes."""

        if self.numerical.get("status") != "passed":
            raise V3Phase3Error("panel precompute is blocked by numerical verification")
        if not (0 <= repeat < EXPECTED_REPEATS and 0 <= outer_fold < EXPECTED_OUTER_FOLDS):
            raise V3Phase3Error("--precompute-panel indices must both be in 0..4")
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.exists():
            raise V3Phase3Error(
                "panel precompute cannot run after the official root manifest exists"
            )
        forbidden_outer_outputs = {
            "pre_outer_selection_lock.json",
            "outer_oof_per_model.csv",
            "phase3_decision.json",
            "v3_decision.json",
        }
        if any((self.output_dir / name).exists() for name in forbidden_outer_outputs):
            raise V3Phase3Error("panel precompute found forbidden official/outer artifacts")
        repetition = next(
            (row for row in self.split_audit["repetitions"] if int(row["repeat"]) == repeat),
            None,
        )
        if repetition is None:
            raise V3Phase3Error(f"frozen split has no repetition {repeat}")
        outer = next(
            (row for row in repetition["outer_folds"] if int(row["outer_fold"]) == outer_fold),
            None,
        )
        if outer is None:
            raise V3Phase3Error(f"frozen split has no panel {repeat}/{outer_fold}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        panel, _evidence = self._select_or_load_panel(
            repeat,
            outer,
            permit_precomputed=True,
        )
        if panel.get("status") != "selection_complete":
            raise V3Phase3Error(
                f"panel {_panel_id(repeat, outer_fold)} failed inner selection: "
                f"{panel.get('error')}"
            )
        print(
            f"precomputed {_panel_id(repeat, outer_fold)} -> "
            f"{panel.get('selected_spec_id')}; outer outcomes were not opened"
        )
        return 0

    def run(self) -> int:
        if self.args.plan_only:
            ready = self.numerical.get("status") == "passed"
            print(
                "validated 2PL-only Phase-3 plan "
                f"({'ready' if ready else self.numerical.get('status', 'blocked')}); "
                "read-only planning wrote no output artifacts"
            )
            return 0
        if self.numerical.get("status") != "passed":
            raise V3Phase3Error("official Phase 3 is blocked by numerical verification")
        self._prepare_output()
        if self._already_complete:
            print(f"verified completed 2PL-only Phase-3 run -> {self.output_dir}")
            return 0
        started_at = _utcnow()
        _atomic_json(self.output_dir / "fold_assignments.json", self._fold_assignment_payload())
        _atomic_json(self.output_dir / "manifest.json", self._base_manifest("running"))
        panels: list[dict[str, Any]] = []
        inner_results: list[dict[str, Any]] = []
        outer_lookup: dict[tuple[int, int], Mapping[str, Any]] = {}
        # Freeze every panel's exact calibration specification before opening any
        # outer-test response for CAT evaluation.
        for repetition in self.split_audit["repetitions"]:
            repeat = int(repetition["repeat"])
            for outer in repetition["outer_folds"]:
                outer_fold = int(outer["outer_fold"])
                panel, evidence = self._select_or_load_panel(
                    repeat,
                    outer,
                    permit_precomputed=bool(self.args.resume),
                )
                panels.append(panel)
                inner_results.extend(evidence)
                outer_lookup[(repeat, outer_fold)] = outer
                print(
                    f"selected {panel.get('selected_spec_id')} for "
                    f"repeat {repeat}/outer {outer_fold} ({panel['status']})"
                )
        if len(panels) != EXPECTED_REPEATS * EXPECTED_OUTER_FOLDS:
            raise V3Phase3Error("selection phase did not produce all 25 panel decisions")
        all_panel_selections_complete = all(
            panel.get("status") == "selection_complete"
            and panel.get("selected_spec_id") is not None
            for panel in panels
        )
        selection_lock = {
            "schema_version": SCRIPT_SCHEMA,
            "study_signature": self.study_signature,
            "status": "locked_before_any_outer_cat_evaluation",
            "outer_outcomes_used": False,
            "require_all_specs_every_inner_fold": True,
            "expected_spec_fold_combinations_per_panel": (EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL),
            "survivor_selection_allowed": False,
            "all_panel_selections_complete": all_panel_selections_complete,
            "outer_evaluation_authorized": all_panel_selections_complete,
            "panels": [
                {
                    key: panel.get(key)
                    for key in (
                        "panel_id",
                        "repeat",
                        "outer_fold",
                        "status",
                        "selected_spec_id",
                        "selected_calibration_specification",
                        "inner_selection",
                    )
                }
                for panel in panels
            ],
        }
        selection_lock_path = self.output_dir / "pre_outer_selection_lock.json"
        if self.args.resume and selection_lock_path.is_file():
            if _read_json(selection_lock_path) != _json_ready(selection_lock):
                raise V3Phase3Error("resume selection lock differs from recomputed inner evidence")
        else:
            _atomic_json(selection_lock_path, selection_lock)

        if not all_panel_selections_complete:
            self._write_terminal_selection_failure_outputs(
                panels=panels,
                inner_results=inner_results,
                started_at=started_at,
            )
            print(
                "wrote terminal 2PL-only selection failure; no outer outcome was opened "
                f"-> {self.output_dir}"
            )
            return 0

        outer_rows: list[dict[str, Any]] = []
        for panel in panels:
            key = (int(panel["repeat"]), int(panel["outer_fold"]))
            rows = self._evaluate_outer_panel(panel, outer_lookup[key])
            outer_rows.extend(rows)
            print(f"evaluated repeat {key[0]}/outer {key[1]}: {len(rows)} frozen policy/model rows")
        self._write_phase3_outputs(
            panels=panels,
            inner_results=inner_results,
            outer_rows=outer_rows,
            started_at=started_at,
        )
        print(f"wrote complete 2PL-only Phase-3 decision -> {self.output_dir}")
        return 0


def _run_synthetic_preflight(out_dir: Path) -> int:
    """Exercise selection/cluster/duplicate logic without InFoBench or fitting.

    This CLI path is deliberately small and cannot produce a Phase-3 decision. It
    gives cluster schedulers a dependency-free smoke check before a real plan.
    Full synthetic end-to-end orchestration is covered through the runner's
    monkeypatchable ``fit_or_load``/``evaluate_policy_panel`` API in tests.
    """

    specs = expected_calibration_specs()
    families = {f"model_{index}": f"family_{index // 2}" for index in range(8)}
    evidence: list[dict[str, Any]] = []
    for spec in specs:
        for index, model in enumerate(sorted(families)):
            loss = 0.40 + 0.002 * spec.simplicity_rank + 0.001 * index
            evidence.append(
                {
                    "spec_id": spec.spec_id,
                    "model": model,
                    "status": "ok",
                    "model_log_loss": loss,
                    "log_loss_sum": loss * 10,
                    "n_cells": 10,
                    "fit_eligible": True,
                    "common_support_verified": True,
                    "fit_cache_key": f"synthetic-{spec.spec_id}",
                }
            )
    aggregate = aggregate_calibration_evidence(evidence, specs, families)
    selected, annotated, detail = select_calibration_spec_one_se(aggregate, specs)
    payload = {
        "schema_version": "infobench-v3-phase3-synthetic-preflight-v1",
        "status": "passed",
        "official_result": False,
        "selected_spec_id": selected.spec_id,
        "selection": detail,
        "evidence": annotated,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir / "synthetic_phase3_preflight.json", payload)
    print(f"synthetic Phase-3 preflight passed -> {out_dir}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--numerical-followup-config",
        type=Path,
        default=DEFAULT_NUMERICAL_FOLLOWUP_CONFIG,
    )
    parser.add_argument("--numerical-lock", type=Path, default=DEFAULT_NUMERICAL_LOCK)
    parser.add_argument("--skills", default=",".join(DEFAULT_SKILLS))
    parser.add_argument("--dimensions", default=DEFAULT_DIMENSIONS)
    parser.add_argument("--structure-name", default="infobench_overall_1d_2pl_only_v1")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-scenarios", type=int, default=50)
    parser.add_argument("--minimum-scored-criteria", type=int, default=15)
    parser.add_argument("--mwle-ridge", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-grid-nodes", type=int, default=50_000)
    parser.add_argument("--metric-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--negative-policy", choices=("error", "drop", "keep"), default="drop")
    parser.add_argument("--require-complete-bank", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--synthetic-preflight",
        action="store_true",
        help="run a no-data orchestration smoke check; never an official result",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument(
        "--precompute-panel",
        nargs=2,
        type=int,
        metavar=("REPEAT", "FOLD"),
        help=("compute inner selection for one frozen panel only; never opens outer outcomes"),
    )
    return parser


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.synthetic_preflight:
        if args.resume or args.plan_only or args.precompute_panel is not None:
            raise V3Phase3Error("--synthetic-preflight cannot combine with run modes")
        return
    if args.precompute_panel is not None and args.plan_only:
        raise V3Phase3Error("--precompute-panel cannot combine with --plan-only")
    if args.top_n < 1 or args.max_scenarios < 1 or args.minimum_scored_criteria < 0:
        raise V3Phase3Error("invalid CAT runtime limits")
    if args.max_iter < 1 or args.tol <= 0 or args.mwle_ridge <= 0:
        raise V3Phase3Error("invalid optimizer settings")
    if args.max_grid_nodes < 1:
        raise V3Phase3Error("max-grid-nodes must be positive")
    if args.metric_bootstrap_replicates < 100:
        raise V3Phase3Error("family bootstrap requires at least 100 replicates")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        if args.out_dir is None:
            args.out_dir = DEFAULT_SYNTHETIC_OUTPUT if args.synthetic_preflight else DEFAULT_OUTPUT
        _validate_runtime_args(args)
        if args.synthetic_preflight:
            return _run_synthetic_preflight(args.out_dir.resolve())
        runner = V3Phase3Runner(args)
        if args.precompute_panel is not None:
            repeat, outer_fold = map(int, args.precompute_panel)
            return runner.precompute_panel(repeat, outer_fold)
        return runner.run()
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
        V3Phase3Error,
        v1.NestedCVError,
        scat.OfflineStudyError,
        cm.CalibrationError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    finally:
        cm.configure_skills(None)


if __name__ == "__main__":
    raise SystemExit(main())
