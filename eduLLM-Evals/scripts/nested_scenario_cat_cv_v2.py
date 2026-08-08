"""Repeated, family-grouped nested CV for the frozen InFoBench v2 CAT study.

This is a new Phase-3 driver.  It deliberately does not share checkpoint or
cache schemas with :mod:`scripts.nested_scenario_cat_cv`, the historical v1
driver.  The command consumes a prospectively frozen v2 configuration and the
five-repeat split manifest, selects an *exact* calibration specification using
inner folds only, and evaluates the one frozen primary CAT policy plus two
non-promotable sensitivity policies on every outer-test model.

The driver is offline: it makes no tutor- or judge-model calls.  ``--plan-only``
performs all input, split, decision, numerical-verification and provenance
checks without fitting.  ``--resume`` accepts only byte-verified v2 artifacts.
Completed, failed, or otherwise non-resumable output is never deleted in place;
a new attempt must use a new versioned output leaf.

The official InFoBench run is intentionally not started by this module's tests.
Synthetic fixtures can exercise the orchestration by monkeypatching
``fit_or_load`` and ``evaluate_policy_panel``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
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
from scripts import run_infobench_v2_numerical_followup_v2 as numerical_followup_v2  # noqa: E402
from scripts import run_infobench_v2_numerical_followup_v3 as numerical_followup  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402
from scripts import scenario_kfold_estimator_cv as scenario_cv  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "infobench_calibration_cat_v2.json"
DEFAULT_SPLITS = ROOT / "configs" / "infobench_v2_splits.manifest.json"
DEFAULT_OUTPUT = ROOT / "runs" / "calibration" / "InFoBench_v2" / "phase3"
DEFAULT_NUMERICAL_FOLLOWUP_CONFIG = (
    ROOT / "configs" / "infobench_v2_numerical_followup_v3.json"
)
DEFAULT_NUMERICAL_LOCK = (
    ROOT
    / "runs"
    / "calibration"
    / "InFoBench_v2_numerical_followup_v3"
    / "numerical_followup_lock.json"
)
DEFAULT_SKILLS = ("content", "format", "number", "style", "linguistic")
DEFAULT_DIMENSIONS = "instruction_following=content+format+number+style+linguistic"

CONFIG_SCHEMA = "infobench-calibration-cat-v2-v1"
SPLIT_SCHEMA = "infobench-v2-repeated-splits-v1"
SCRIPT_SCHEMA = "infobench-nested-scenario-cat-cv-v2-v1"
CACHE_SCHEMA = "infobench-v2-calibration-fit-cache-v1"
CHECKPOINT_SCHEMA = "infobench-v2-phase3-checkpoint-v1"
DECISION_SCHEMA = "infobench-v2-phase3-decision-v1"
SELECTED_SPECS_SCHEMA = "infobench-v2-selected-calibration-specs-v1"
FOLLOWUP_CONFIG_SCHEMA = "infobench-v2-numerical-followup-v3"
FOLLOWUP_RUN_SCHEMA = "infobench-v2-numerical-followup-run-v3"
FOLLOWUP_LOCK_SCHEMA = "infobench-v2-numerical-followup-lock-v3"
FOLLOWUP_FROZEN_CONTRACT_SHA256 = (
    "25e6db82f534492dfdb82448ea5bb6fb00fed5237ae9e7b607a9b05e89b9a809"
)

EXPECTED_REPEATS = 5
EXPECTED_OUTER_FOLDS = 5
EXPECTED_INNER_FOLDS = 4
EXPECTED_MODELS = 52
EXPECTED_FAMILIES = 22

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
    "v2_decision.json",
    "manifest.json",
)

CODE_DEPENDENCY_PATHS = (
    ROOT / "scripts" / "nested_scenario_cat_cv_v2.py",
    ROOT / "scripts" / "nested_scenario_cat_cv.py",
    ROOT / "scripts" / "calibrate_mirt.py",
    ROOT / "scripts" / "kfold_cv_mirt.py",
    ROOT / "scripts" / "scenario_cat_lib.py",
    ROOT / "scripts" / "scenario_kfold_estimator_cv.py",
    ROOT / "scripts" / "run_infobench_v2_numerical_followup_v3.py",
    ROOT / "scripts" / "run_infobench_v2_numerical_followup_v2.py",
    ROOT / "scripts" / "run_infobench_v2_numerical_followup.py",
    ROOT / "tutor_cat" / "mirt.py",
)


class V2Phase3Error(RuntimeError):
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
        raise V2Phase3Error(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise V2Phase3Error(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


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
            raise V2Phase3Error(f"required code dependency is missing: {path}")
        output[_display_path(path)] = _sha256(path)
    return output


def _number(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise V2Phase3Error(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise V2Phase3Error(f"{field} must be finite")
    return result


def _spec_id(family: str, ridge: float | None, shrinkage: float | None) -> str:
    if family == cm.ONE_PL:
        return "1pl_fixed_a1"
    if family == cm.LOG_SHRINKAGE_2PL:
        return f"log_shrinkage_2pl_lambda{float(shrinkage):g}"
    ridge_slug = f"{float(ridge):g}".replace(".", "p")
    return f"free_2pl_ridge_{ridge_slug}"


def expected_calibration_specs() -> list[CalibrationSpec]:
    """Return the six prospectively frozen exact specifications in simple order."""

    raw = [
        (cm.ONE_PL, None, None),
        (cm.LOG_SHRINKAGE_2PL, None, 16.0),
        (cm.LOG_SHRINKAGE_2PL, None, 4.0),
        (cm.FREE_2PL, 0.1, None),
        (cm.FREE_2PL, 0.01, None),
        (cm.FREE_2PL, 0.001, None),
    ]
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
    shrinkage = (
        None
        if shrink_raw is None
        else _number(shrink_raw, field="log_a_shrinkage")
    )
    canonical = cm.calibration_specification(
        family,
        ridge=0.0 if ridge is None else ridge,
        log_a_shrinkage=(
            cm.DEFAULT_LOG_A_SHRINKAGE if shrinkage is None else shrinkage
        ),
    )
    configured_cache_key = str(raw.get("canonical_cache_key") or "")
    if configured_cache_key and configured_cache_key != canonical["cache_key"]:
        raise V2Phase3Error(
            f"{raw.get('spec_id')}: configured calibration cache key does not reproduce"
        )
    if int(raw.get("simplicity_rank", rank)) != rank:
        raise V2Phase3Error("calibration simplicity ranks must be the ordered integers 0..5")
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
        raise V2Phase3Error("calibration_specifications must be a frozen ordered list")
    specs = [_normalize_config_spec(raw, rank) for rank, raw in enumerate(raw_specs)]
    expected = expected_calibration_specs()
    observed_signature = [
        (spec.family, spec.ridge, spec.log_a_shrinkage) for spec in specs
    ]
    expected_signature = [
        (spec.family, spec.ridge, spec.log_a_shrinkage) for spec in expected
    ]
    if observed_signature != expected_signature:
        raise V2Phase3Error(
            "calibration specifications/order must be exactly 1PL, shrinkage "
            "lambda 16, shrinkage lambda 4, and free-2PL ridge 0.1/0.01/0.001"
        )
    if [spec.spec_id for spec in specs] != [spec.spec_id for spec in expected]:
        raise V2Phase3Error("calibration specification IDs are not canonical")
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
        raise V2Phase3Error("cat_policies must be an object")
    primary_raw = panel.get("primary")
    sensitivities_raw = panel.get("sensitivities")
    if not isinstance(primary_raw, Mapping) or not isinstance(sensitivities_raw, list):
        raise V2Phase3Error("cat_policies requires primary and two sensitivities")
    policies = [
        _normalize_policy(primary_raw, policy_id=PRIMARY_POLICY_ID, role="primary")
    ]
    policies.extend(
        _normalize_policy(raw, policy_id=policy_id, role="sensitivity")
        for raw, policy_id in zip(
            sensitivities_raw, SENSITIVITY_POLICY_IDS, strict=False
        )
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
        raise V2Phase3Error(
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
        raise V2Phase3Error(f"unexpected repeated split schema: {split.get('schema_version')}")
    model_to_family = {
        str(model): str(family)
        for model, family in (split.get("model_to_family") or {}).items()
    }
    all_models = set(model_to_family)
    all_families = set(model_to_family.values())
    if len(all_models) != EXPECTED_MODELS or len(all_families) != EXPECTED_FAMILIES:
        raise V2Phase3Error(
            f"split must contain {EXPECTED_MODELS} models in {EXPECTED_FAMILIES} families"
        )
    if expected_models is not None and set(map(str, expected_models)) != all_models:
        raise V2Phase3Error("response-matrix models do not match the frozen split")

    scenario = split.get("scenario_split") or {}
    administration = set(map(str, scenario.get("administration_scenario_ids") or []))
    evaluation = set(map(str, scenario.get("evaluation_scenario_ids") or []))
    if administration & evaluation or len(administration) != 400 or len(evaluation) != 100:
        raise V2Phase3Error("scenario split must be disjoint with exactly 400/100 scenarios")
    if expected_scenarios is not None and administration | evaluation != set(
        map(str, expected_scenarios)
    ):
        raise V2Phase3Error("scenario records do not match the frozen scenario split")

    repetitions = split.get("repetitions") or []
    if len(repetitions) != EXPECTED_REPEATS:
        raise V2Phase3Error(f"expected {EXPECTED_REPEATS} frozen repetitions")
    runtime_repeats: list[dict[str, Any]] = []
    partition_hashes: list[str] = []
    for expected_repeat, repetition in enumerate(repetitions):
        repeat = int(repetition.get("repeat", -1))
        if repeat != expected_repeat:
            raise V2Phase3Error("repetitions must be ordered and numbered 0..4")
        outer_folds = repetition.get("outer_folds") or []
        if len(outer_folds) != EXPECTED_OUTER_FOLDS:
            raise V2Phase3Error(f"repeat {repeat} must have five outer folds")
        seen_test: set[str] = set()
        model_to_outer: dict[str, int] = {}
        runtime_outer: list[dict[str, Any]] = []
        for expected_fold, outer in enumerate(outer_folds):
            fold = int(outer.get("outer_fold", -1))
            if fold != expected_fold:
                raise V2Phase3Error(f"repeat {repeat} outer folds must be numbered 0..4")
            train = set(map(str, outer.get("train_model_ids") or []))
            test = set(map(str, outer.get("test_model_ids") or []))
            if train & test or train | test != all_models:
                raise V2Phase3Error(f"repeat/fold {repeat}/{fold} does not partition models")
            if seen_test & test:
                raise V2Phase3Error(f"repeat {repeat} repeats an outer-test model")
            seen_test |= test
            for model in test:
                model_to_outer[model] = fold
            for family in all_families:
                members = {model for model, item in model_to_family.items() if item == family}
                if members & train and members & test:
                    raise V2Phase3Error(
                        f"family {family} crosses repeat/fold {repeat}/{fold}"
                    )

            inner_folds = outer.get("inner_folds") or []
            if len(inner_folds) != EXPECTED_INNER_FOLDS:
                raise V2Phase3Error(f"repeat/fold {repeat}/{fold} needs four inner folds")
            inner_seen: list[str] = []
            runtime_inner: list[dict[str, Any]] = []
            for expected_inner, inner in enumerate(inner_folds):
                inner_fold = int(inner.get("inner_fold", -1))
                if inner_fold != expected_inner:
                    raise V2Phase3Error(
                        f"repeat/fold {repeat}/{fold} inner folds must be numbered 0..3"
                    )
                fit_ids = set(map(str, inner.get("fit_model_ids") or []))
                validation_ids = set(map(str, inner.get("validation_model_ids") or []))
                if test & (fit_ids | validation_ids):
                    raise V2Phase3Error(
                        f"outer-test leakage in repeat/fold/inner {repeat}/{fold}/{inner_fold}"
                    )
                if fit_ids & validation_ids or fit_ids | validation_ids != train:
                    raise V2Phase3Error(
                        f"inner {repeat}/{fold}/{inner_fold} does not partition outer train"
                    )
                for family in all_families:
                    members = {
                        model for model, item in model_to_family.items() if item == family
                    }
                    if members & fit_ids and members & validation_ids:
                        raise V2Phase3Error(
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
                raise V2Phase3Error(
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
            raise V2Phase3Error(f"repeat {repeat} does not outer-test all models")
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
        raise V2Phase3Error("the five frozen repetitions are not distinct")
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
        model_losses.append(
            (model_to_family[str(row["model"])], float(row["model_log_loss"]))
        )
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
                "n_families_scored": len(
                    {model_to_family[model] for model in valid_models}
                ),
                "n_cells": n_cells,
                "mean_log_loss": float(np.mean(losses)) if losses else None,
                "pooled_log_loss": (
                    sum(float(row["log_loss_sum"]) for row in valid) / n_cells
                    if n_cells
                    else None
                ),
                "family_cluster_se": family_se,
                "coverage": len(valid_models) / len(expected_models)
                if expected_models
                else 0.0,
                "common_support_verified": common_support,
                "fit_eligible": fit_eligible,
                "eligible": eligible,
                "fit_cache_keys": sorted(
                    {str(row.get("fit_cache_key")) for row in subset}
                ),
            }
        )
    return output


def select_calibration_spec_one_se(
    evidence: Sequence[Mapping[str, Any]], specs: Sequence[CalibrationSpec]
) -> tuple[CalibrationSpec, list[dict[str, Any]], dict[str, Any]]:
    """Use only inner family-clustered loss and the frozen simplicity order."""

    rows = [dict(row) for row in evidence]
    simplicity = [spec.canonical["cache_key"] for spec in specs]
    try:
        selected = cell_cv.select_calibration_spec_one_se(rows, simplicity)
    except ValueError as error:
        raise V2Phase3Error(str(error)) from error
    selected_key = str(selected["selected_cache_key"])
    by_key = {spec.canonical["cache_key"]: spec for spec in specs}
    if selected_key not in by_key:
        raise V2Phase3Error("one-SE helper returned a non-frozen calibration spec")
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
        baseline = np.asarray(
            [float(row["baseline_scenarios_administered"]) for row in sampled]
        )
        if baseline.size and float(baseline.mean()) > 0:
            reductions.append(1.0 - float(cat.mean()) / float(baseline.mean()))
    def percentile(values: Sequence[float], q: float) -> float | None:
        return (
            float(np.percentile(np.asarray(values, dtype=float), q))
            if values
            else None
        )
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
        seed=v1._bootstrap_seed(seed, f"v2-family:{label}"),
        replicates=replicates,
    )
    metrics.update(clustered)
    metrics["paired_ci_favors_cat"] = (
        clustered["scenario_reduction_lower_95_ci"] is not None
        and float(clustered["scenario_reduction_lower_95_ci"]) > 0
    )
    metrics["bootstrap_unit"] = "tutor_family"
    metrics["n_families"] = len(
        {model_to_family[str(row["model"])] for row in rows}
    )
    return metrics


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
        raise V2Phase3Error(f"baseline.{name} requires a path and sha256")
    return _resolve_repo_path(str(path_value)), expected_hash


def _check_frozen_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise V2Phase3Error(f"unexpected v2 config schema: {config.get('schema_version')}")
    status = str(config.get("status") or "")
    if status not in {
        "preregistered",
        "frozen",
        "frozen_before_v2_fitting",
        "frozen_before_response_dependent_v2_results",
        "frozen_pre_run",
    }:
        raise V2Phase3Error("v2 config must be prospectively frozen, not draft/post-hoc")
    cv = config.get("cross_validation") or {}
    if (
        int(cv.get("repetitions", 0)) != EXPECTED_REPEATS
        or int(cv.get("outer_folds", cv.get("outer_folds_per_repetition", 0)))
        != EXPECTED_OUTER_FOLDS
        or int(cv.get("inner_folds", cv.get("inner_folds_per_outer_panel", 0)))
        != EXPECTED_INNER_FOLDS
    ):
        raise V2Phase3Error("v2 config must freeze 5 repeats x 5 outer x 4 inner folds")
    if cv.get("group_related_model_families", True) is not True:
        raise V2Phase3Error("family-grouped cross-validation cannot be disabled")
    if (
        cv.get("aggregate_repeated_predictions_per_model") is not True
        or cv.get("family_cluster_bootstrap") is not True
        or cv.get("treat_model_repeat_rows_as_independent") is not False
    ):
        raise V2Phase3Error("repeated-CV aggregation must remain per-model/family-clustered")
    selection = config.get("calibration_selection") or {}
    if selection.get("outer_outcomes_used", False) is not False:
        raise V2Phase3Error("outer outcomes may not select calibration specifications")
    if selection.get("cat_outcomes_used", False) is not False:
        raise V2Phase3Error("CAT outcomes may not select calibration specifications")
    if selection.get("allow_fallback", False) is not False:
        raise V2Phase3Error("calibration specification fallback is forbidden")
    rule = str(selection.get("rule") or selection.get("method") or "")
    if rule and "one" not in rule.lower():
        raise V2Phase3Error("calibration selection must use the frozen one-SE rule")
    if selection.get("common_item_and_response_cell_support_required") is not True:
        raise V2Phase3Error("common held-out item/cell support must remain required")
    if selection.get("uncertainty_unit") != "tutor_model_family":
        raise V2Phase3Error("calibration uncertainty must be clustered by tutor family")
    if (
        selection.get("uncertainty_method")
        != "deterministic_leave_one_family_out_jackknife_se"
    ):
        raise V2Phase3Error("calibration selection must retain family jackknife SE")
    if not math.isclose(
        float(selection.get("minimum_exact_spec_modal_fraction_across_outer_panels", -1)),
        0.80,
        rel_tol=0,
        abs_tol=0,
    ):
        raise V2Phase3Error("exact-spec modal stability threshold must remain 80%")
    gates = config.get("selection_gates") or {}
    if (
        gates.get("allow_fallback_if_none_pass", False) is not False
        or gates.get("allow_fallback_if_primary_fails", False) is not False
    ):
        raise V2Phase3Error("fallback after a failed primary policy is forbidden")
    if gates.get("apply_to_primary_only") is not True:
        raise V2Phase3Error("scientific gates must apply to the primary policy only")
    if (
        gates.get("require_every_outer_panel") is not True
        or gates.get("require_every_repetition_pooled_gate") is not True
    ):
        raise V2Phase3Error("the primary must pass every panel and every repetition")
    if not math.isclose(
        _number(
            gates.get("minimum_scenario_reduction_vs_random"),
            field="minimum_scenario_reduction_vs_random",
        ),
        0.50,
        rel_tol=0,
        abs_tol=0,
    ):
        raise V2Phase3Error("the frozen 50% scenario-reduction gate changed")
    policies = config.get("cat_policies") or {}
    if policies.get("sensitivities_can_replace_primary") is not False:
        raise V2Phase3Error("sensitivity policies must remain non-promotable")


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


def _same_optional_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=0)
    except (TypeError, ValueError):
        return False


def _materialize_followup_config(
    overlay: Mapping[str, Any], *, followup_path: Path, lock_path: Path
) -> dict[str, Any]:
    """Rebuild and validate the exact config used by the frozen v3 runner."""

    if (
        overlay.get("schema_version") != FOLLOWUP_CONFIG_SCHEMA
        or overlay.get("status") != "preregistered_before_followup_results"
    ):
        raise V2Phase3Error("numerical follow-up config is not the frozen v3 schema")

    configured_output = _resolve_repo_path(str(overlay.get("output_dir") or ""))
    configured_lock = _resolve_repo_path(str(overlay.get("lock_path") or ""))
    if configured_lock != lock_path or configured_output != lock_path.parent:
        raise V2Phase3Error("numerical follow-up output/lock path differs from its config")

    validation_overlay = copy.deepcopy(dict(overlay))
    validation_overlay["output_dir"] = (
        f"runs/calibration/{numerical_followup.OUTPUT_LEAF}"
    )
    validation_overlay["lock_path"] = (
        f"runs/calibration/{numerical_followup.OUTPUT_LEAF}/"
        "numerical_followup_lock.json"
    )
    try:
        numerical_followup._validate_config(validation_overlay)
    except (numerical_followup.FollowupV3Error, ValueError) as error:
        raise V2Phase3Error(
            f"numerical follow-up v3 config contract is invalid: {error}"
        ) from error

    design = overlay.get("scientific_design_source") or {}
    orchestration = overlay.get("orchestration_source") or {}
    if (
        design.get("frozen_contract_sha256") != FOLLOWUP_FROZEN_CONTRACT_SHA256
        or orchestration.get("required_corrections")
        != numerical_followup.EXPECTED_ORCHESTRATION_CORRECTIONS
    ):
        raise V2Phase3Error("numerical follow-up scientific/orchestration contract changed")
    source_path = _resolve_repo_path(str(design.get("config") or ""))
    if (
        not source_path.is_file()
        or _sha256(source_path) != design.get("config_sha256")
    ):
        raise V2Phase3Error("numerical follow-up design source is missing or changed")
    orchestration_path = _resolve_repo_path(str(orchestration.get("runner") or ""))
    freeze = overlay.get("code_freeze") or {}
    runner_path = _resolve_repo_path(str(freeze.get("runner") or ""))
    try:
        if (
            not orchestration_path.is_file()
            or _sha256(orchestration_path) != orchestration.get("runner_sha256")
            or not runner_path.is_file()
            or _sha256(runner_path) != freeze.get("runner_sha256")
        ):
            raise V2Phase3Error("numerical follow-up code freeze is invalid")
        source_raw = _read_json(source_path)
        numerical_followup_v2._validate_overlay(source_raw)
        source_effective = numerical_followup_v2._materialize_effective_config(
            source_raw
        )
        numerical_followup_v2._validate_effective_config(source_effective)
        if (
            _canonical_hash(source_raw.get("frozen_contract") or {})
            != FOLLOWUP_FROZEN_CONTRACT_SHA256
        ):
            raise V2Phase3Error("numerical follow-up source contract hash changed")
        numerical_followup._verify_v1_history(overlay, source_effective)
        numerical_followup._verify_v2_history(overlay)
    except (
        numerical_followup.FollowupV3Error,
        numerical_followup_v2.FollowupV2Error,
        OSError,
        ValueError,
    ) as error:
        raise V2Phase3Error(
            f"numerical follow-up overlay/provenance is invalid: {error}"
        ) from error

    effective = copy.deepcopy(source_effective)
    effective.update(
        {
            "schema_version": FOLLOWUP_CONFIG_SCHEMA,
            "status": overlay["status"],
            "purpose": overlay["purpose"],
            "scientific_design_source": copy.deepcopy(design),
            "orchestration_source": copy.deepcopy(orchestration),
            "historical_evidence": copy.deepcopy(overlay["historical_evidence"]),
            "code_freeze": copy.deepcopy(freeze),
            "output_dir": overlay["output_dir"],
            "lock_path": overlay["lock_path"],
            "threshold_change_policy": overlay["threshold_change_policy"],
            "selection_policy": overlay["selection_policy"],
        }
    )
    return effective


def _validate_followup_specifications(
    followup: Mapping[str, Any], specs: Sequence[CalibrationSpec]
) -> None:
    rows = followup.get("exact_specifications")
    if not isinstance(rows, list) or len(rows) != len(specs):
        raise V2Phase3Error("numerical follow-up changed the exact specification panel")
    by_id = {
        str(row.get("spec_id")): row for row in rows if isinstance(row, Mapping)
    }
    if set(by_id) != {spec.spec_id for spec in specs}:
        raise V2Phase3Error("numerical follow-up specification IDs differ from Phase 3")
    for spec in specs:
        row = by_id[spec.spec_id]
        if (
            row.get("family") != spec.family
            or not _same_optional_number(row.get("ridge"), spec.ridge)
            or not _same_optional_number(
                row.get("log_a_shrinkage"), spec.log_a_shrinkage
            )
            or row.get("canonical_cache_key") != spec.canonical.get("cache_key")
        ):
            raise V2Phase3Error(
                f"numerical follow-up changed exact specification {spec.spec_id}"
            )


def _validate_followup_study_signature(
    manifest: Mapping[str, Any],
    *,
    overlay: Mapping[str, Any],
    effective: Mapping[str, Any],
    followup_path: Path,
    specs: Sequence[CalibrationSpec],
) -> None:
    """Independently verify the producer's frozen study-signature bundle."""

    signature = manifest.get("study_signature")
    if not isinstance(signature, Mapping):
        raise V2Phase3Error("numerical follow-up study signature is missing")
    payload_keys = {
        "schema_version",
        "config_sha256",
        "effective_config_sha256",
        "frozen_contract_sha256",
        "historical_v1_sha256",
        "historical_v2_sha256",
        "parent_manifest_sha256",
        "parent_decision_sha256",
        "parent_checkpoint_sha256",
        "input_sha256",
        "code_sha256",
        "environment_sha256",
        "exact_specifications",
        "fit_grids",
        "comparison_schedule",
        "eap_profiles",
        "v1_fit_or_checkpoint_artifacts_reused",
        "v2_fit_or_checkpoint_artifacts_reused",
        "required_total_new_fits",
    }
    if set(signature) != payload_keys | {"environment", "canonical_sha256"}:
        raise V2Phase3Error("numerical follow-up study-signature fields changed")
    payload = {key: signature.get(key) for key in payload_keys}
    canonical = _canonical_hash(payload)
    if not (
        signature.get("canonical_sha256")
        == manifest.get("study_signature_sha256")
        == canonical
    ):
        raise V2Phase3Error("numerical follow-up study signature hash is invalid")

    environment = signature.get("environment")
    if not isinstance(environment, Mapping):
        raise V2Phase3Error("numerical follow-up environment signature is missing")
    environment_payload = {
        str(key): value
        for key, value in environment.items()
        if str(key) != "canonical_sha256"
    }
    if not (
        environment.get("canonical_sha256")
        == signature.get("environment_sha256")
        == _canonical_hash(environment_payload)
    ):
        raise V2Phase3Error("numerical follow-up environment signature is invalid")

    expected_specs = [
        {
            "spec_id": spec.spec_id,
            "family": spec.family,
            "ridge": spec.ridge,
            "log_a_shrinkage": spec.log_a_shrinkage,
            "canonical_specification": spec.canonical,
        }
        for spec in specs
    ]
    frozen = effective.get("frozen_inputs") or {}
    expected_inputs = {
        numerical_followup._display(_resolve_repo_path(str(frozen[name]))): frozen[
            f"{name}_sha256"
        ]
        for name in ("response_matrix", "rubrics", "scenarios", "split_manifest")
    }
    expected_code = {
        numerical_followup._display(path): _sha256(path)
        for path in numerical_followup.CODE_DEPENDENCIES
    }
    expected_parent = effective.get("parent_study") or {}
    expected_fields = {
        "schema_version": FOLLOWUP_RUN_SCHEMA,
        "config_sha256": _sha256(followup_path),
        "effective_config_sha256": _canonical_hash(effective),
        "frozen_contract_sha256": FOLLOWUP_FROZEN_CONTRACT_SHA256,
        "historical_v1_sha256": {
            "aborted_marker_sha256": overlay["historical_evidence"]["v1_aborted"][
                "aborted_marker_sha256"
            ],
            "output_tree_sha256": overlay["historical_evidence"]["v1_aborted"][
                "output_tree_sha256"
            ],
        },
        "historical_v2_sha256": numerical_followup._verify_v2_history(overlay),
        "parent_manifest_sha256": expected_parent.get("study_manifest_sha256"),
        "parent_decision_sha256": expected_parent.get("decision_sha256"),
        "parent_checkpoint_sha256": expected_parent.get("grid81_checkpoint_sha256"),
        "input_sha256": expected_inputs,
        "code_sha256": expected_code,
        "exact_specifications": expected_specs,
        "fit_grids": [81, 101, 121],
        "comparison_schedule": [[81, 101], [101, 121]],
        "eap_profiles": _json_ready(numerical_followup.v2.v1.EAP_PROFILES),
        "v1_fit_or_checkpoint_artifacts_reused": 0,
        "v2_fit_or_checkpoint_artifacts_reused": 0,
        "required_total_new_fits": 72,
    }
    mismatches = [
        key for key, value in expected_fields.items() if signature.get(key) != value
    ]
    if mismatches:
        raise V2Phase3Error(
            "numerical follow-up study signature differs for "
            + ", ".join(sorted(mismatches))
        )


def _followup_evidence_path(lock_dir: Path, key: str) -> Path:
    parts = key.split("/")
    if len(parts) == 3 and parts[0] == "fit":
        return (
            lock_dir
            / "fit_comparisons"
            / parts[1]
            / parts[2]
            / "fit_pair_gate.json"
        )
    if len(parts) == 2 and parts[0] == "eap":
        return lock_dir / "eap_bound" / parts[1] / "eap_bound_gate.json"
    raise V2Phase3Error(f"unexpected numerical evidence key: {key}")


def _validate_followup_evidence(
    lock_dir: Path,
    evidence_hashes: Any,
    specifications: Sequence[CalibrationSpec],
) -> tuple[dict[str, bool], dict[str, Any]]:
    specs_by_id = {spec.spec_id: spec for spec in specifications}
    required_spec_ids = set(specs_by_id)
    expected_keys = {
        *(f"fit/81_vs_101/{spec_id}" for spec_id in required_spec_ids),
        *(f"fit/101_vs_121/{spec_id}" for spec_id in required_spec_ids),
        *(f"eap/{spec_id}" for spec_id in required_spec_ids),
    }
    hashes = evidence_hashes if isinstance(evidence_hashes, Mapping) else {}
    if set(map(str, hashes)) != expected_keys:
        raise V2Phase3Error("numerical follow-up evidence panel is incomplete")
    fit_pass: dict[str, dict[str, bool]] = {
        "81_vs_101": {},
        "101_vs_121": {},
    }
    eap_rows: dict[str, Mapping[str, Any]] = {}
    for key in sorted(expected_keys):
        path = _followup_evidence_path(lock_dir, key)
        expected_hash = str(hashes.get(key) or "")
        if not path.is_file() or not expected_hash or _sha256(path) != expected_hash:
            raise V2Phase3Error(f"numerical evidence hash mismatch: {key}")
        payload = _read_json(path)
        parts = key.split("/")
        spec_id = parts[-1]
        if (
            payload.get("schema_version") != FOLLOWUP_RUN_SCHEMA
            or payload.get("spec_id") != spec_id
            or payload.get("selection_performed") is not False
            or payload.get("cat_results_inspected") is not False
        ):
            raise V2Phase3Error(f"numerical evidence provenance is invalid: {key}")
        if parts[0] == "fit":
            comparison = parts[1]
            if (
                payload.get("comparison") != comparison
                or payload.get("canonical_specification")
                != specs_by_id[spec_id].canonical
            ):
                raise V2Phase3Error(f"numerical comparison label is invalid: {key}")
            fit_pass[comparison][spec_id] = payload.get("passed") is True
        else:
            eap_rows[spec_id] = payload
    if any(set(panel) != required_spec_ids for panel in fit_pass.values()):
        raise V2Phase3Error("numerical fit-comparison panel is incomplete")
    if set(eap_rows) != required_spec_ids:
        raise V2Phase3Error("numerical EAP panel is incomplete")
    comparison_summary = {
        comparison: all(panel.values()) for comparison, panel in fit_pass.items()
    }
    theta_passed = all(
        row.get("all_theta_comparisons_passed") is True
        for row in eap_rows.values()
    )
    bound8_passed = all(
        ((row.get("tail_gates") or {}).get("bound8") or {}).get("passed") is True
        for row in eap_rows.values()
    )
    bound10_passed = all(
        ((row.get("tail_gates") or {}).get("bound10") or {}).get("passed") is True
        for row in eap_rows.values()
    )
    if theta_passed and bound8_passed:
        eap_profile: dict[str, Any] | None = {
            "quadrature_method": "normal_trapezoid",
            "linear_bound": 8.0,
            "eap_grid": 401,
            "tail_region": 7.5,
        }
    elif theta_passed and bound10_passed:
        eap_profile = {
            "quadrature_method": "normal_trapezoid",
            "linear_bound": 10.0,
            "eap_grid": 501,
            "tail_region": 9.5,
        }
    else:
        eap_profile = None
    return comparison_summary, {
        "theta_passed": theta_passed,
        "bound8_passed": bound8_passed,
        "bound10_passed": bound10_passed,
        "fit_grids": sorted(
            {
                int(row.get("fit_grid"))
                for row in eap_rows.values()
                if isinstance(row.get("fit_grid"), int)
            }
        ),
        "profile": eap_profile,
    }


def _validate_numerical_lock(
    config: Mapping[str, Any],
    specs: Sequence[CalibrationSpec],
    *,
    base_config_path: Path | None = None,
    lock_path: Path | None = None,
    followup_config_path: Path | None = None,
    allow_pending: bool = False,
) -> dict[str, Any]:
    """Validate and return the one global numerical profile for Phase 3/4.

    The failed 61/401 parent preflight remains immutable.  Only the separately
    versioned all-six-specification follow-up lock may authorize fitting.
    """

    base_path = (base_config_path or DEFAULT_CONFIG).resolve()
    path = (lock_path or DEFAULT_NUMERICAL_LOCK).resolve()
    followup_path = (
        followup_config_path or DEFAULT_NUMERICAL_FOLLOWUP_CONFIG
    ).resolve()
    pending = {
        "fit_grid": None,
        "eap_grid": None,
        "quadrature_method": None,
        "linear_bound": None,
        "tail_region": None,
        "verification_path": _display_path(path),
        "verification_sha256": None,
        "followup_config_path": _display_path(followup_path),
        "followup_config_sha256": (
            _sha256(followup_path) if followup_path.is_file() else None
        ),
        "effective_followup_config_sha256": None,
        "frozen_contract_sha256": FOLLOWUP_FROZEN_CONTRACT_SHA256,
        "status": "pending_blocker",
        "passed_spec_ids": [],
    }
    if not followup_path.is_file():
        if allow_pending:
            return pending
        raise V2Phase3Error("versioned all-spec numerical follow-up lock is missing")
    overlay = _read_json(followup_path)
    followup = _materialize_followup_config(
        overlay,
        followup_path=followup_path,
        lock_path=path,
    )
    _validate_followup_specifications(followup, specs)
    pending["effective_followup_config_sha256"] = _canonical_hash(followup)
    if not path.is_file():
        decision_path = path.parent / "numerical_followup_decision.json"
        manifest_path = path.parent / "study_manifest.json"
        terminal_failure = False
        if decision_path.is_file() and manifest_path.is_file():
            decision = _read_json(decision_path)
            manifest = _read_json(manifest_path)
            terminal_failure = (
                decision.get("schema_version") == FOLLOWUP_RUN_SCHEMA
                and decision.get("passed") is False
                and decision.get("status") == "blocked_numerical_followup"
                and manifest.get("schema_version") == FOLLOWUP_RUN_SCHEMA
                and manifest.get("status") == "blocked_numerical_followup"
                and manifest.get("decision_sha256") == _sha256(decision_path)
                and manifest.get("lock_sha256") is None
            )
            if terminal_failure:
                pending.update(
                    {
                        "status": "failed_blocker",
                        "terminal_followup_status": decision["status"],
                        "followup_decision_sha256": _sha256(decision_path),
                        "followup_manifest_sha256": _sha256(manifest_path),
                    }
                )
        if allow_pending:
            return pending
        if terminal_failure:
            raise V2Phase3Error(
                "numerical follow-up completed without a promotable lock"
            )
        raise V2Phase3Error("versioned all-spec numerical follow-up lock is missing")
    if not _sha_companion_matches(path):
        raise V2Phase3Error("numerical follow-up lock companion hash is invalid")
    parent = followup.get("parent_study") or {}
    if (
        not base_path.is_file()
        or parent.get("config_sha256") != _sha256(base_path)
        or _resolve_repo_path(str(parent.get("config") or "")) != base_path
    ):
        raise V2Phase3Error("numerical follow-up does not descend from this v2 config")

    baseline = config.get("baseline") or {}
    frozen = followup.get("frozen_inputs") or {}
    expected_inputs = {
        "response_matrix_sha256": baseline.get("response_matrix_sha256"),
        "rubrics_sha256": baseline.get("rubrics_sha256"),
        "scenarios_sha256": baseline.get("scenarios_sha256"),
        "split_manifest_sha256": (config.get("cross_validation") or {}).get(
            "split_manifest_sha256"
        ),
    }
    if any(frozen.get(key) != value for key, value in expected_inputs.items()):
        raise V2Phase3Error("numerical follow-up inputs differ from Phase 3")

    artifact = _read_json(path)
    required_ids = {spec.spec_id for spec in specs}
    passed_ids = set(map(str, artifact.get("passed_spec_ids") or []))
    history = overlay.get("historical_evidence") or {}
    v1_history = history.get("v1_aborted") or {}
    v2_history = history.get("v2_terminal") or {}
    if (
        artifact.get("schema_version") != FOLLOWUP_LOCK_SCHEMA
        or artifact.get("status") != "complete_pass"
        or artifact.get("all_six_exact_specifications_share_profile") is not True
        or artifact.get("selection_performed") is not False
        or artifact.get("cat_results_inspected") is not False
        or passed_ids != required_ids
        or artifact.get("config_sha256") != _sha256(followup_path)
        or artifact.get("parent_decision_sha256") != parent.get("decision_sha256")
        or artifact.get("parent_manifest_sha256")
        != parent.get("study_manifest_sha256")
        or artifact.get("historical_v1_aborted_marker_sha256")
        != v1_history.get("aborted_marker_sha256")
        or artifact.get("historical_v1_output_tree_sha256")
        != v1_history.get("output_tree_sha256")
        or artifact.get("historical_v2_manifest_sha256")
        != v2_history.get("manifest_sha256")
        or artifact.get("historical_v2_decision_sha256")
        != v2_history.get("decision_sha256")
        or artifact.get("historical_v2_output_tree_sha256")
        != v2_history.get("output_tree_sha256")
        or artifact.get("historical_fit_or_checkpoint_artifacts_reused") != 0
    ):
        raise V2Phase3Error("numerical follow-up lock is incomplete or mismatched")

    comparison_summary, eap = _validate_followup_evidence(
        path.parent, artifact.get("evidence_sha256"), specs
    )
    if comparison_summary["81_vs_101"] and comparison_summary["101_vs_121"]:
        expected_fit_grid = 81
    elif (
        not comparison_summary["81_vs_101"]
        and comparison_summary["101_vs_121"]
    ):
        expected_fit_grid = 101
    else:
        raise V2Phase3Error("numerical follow-up did not establish a common fit grid")
    profile = artifact.get("effective_global_profile")
    expected_profile = {"fit_grid": expected_fit_grid, **(eap["profile"] or {})}
    if (
        eap["profile"] is None
        or eap["fit_grids"] != [expected_fit_grid]
        or not isinstance(profile, Mapping)
        or dict(profile) != expected_profile
        or artifact.get("fit_comparison_all_six_passed") != comparison_summary
    ):
        raise V2Phase3Error("numerical follow-up global profile is inconsistent")

    decision_path = path.parent / "numerical_followup_decision.json"
    manifest_path = path.parent / "study_manifest.json"
    if not decision_path.is_file() or not manifest_path.is_file():
        raise V2Phase3Error("numerical follow-up decision or manifest is missing")
    decision = _read_json(decision_path)
    manifest = _read_json(manifest_path)
    _validate_followup_study_signature(
        manifest,
        overlay=overlay,
        effective=followup,
        followup_path=followup_path,
        specs=specs,
    )
    if (
        artifact.get("followup_decision_sha256") != _sha256(decision_path)
        or decision.get("schema_version") != FOLLOWUP_RUN_SCHEMA
        or decision.get("status") != "complete_pass"
        or decision.get("passed") is not True
        or decision.get("v1_abort_preserved") is not True
        or decision.get("v2_terminal_preserved") is not True
        or decision.get("historical_fit_or_checkpoint_artifacts_reused") != 0
        or decision.get("historical_v1_sha256")
        != {
            "aborted_marker_sha256": v1_history.get("aborted_marker_sha256"),
            "output_tree_sha256": v1_history.get("output_tree_sha256"),
        }
        or decision.get("historical_v2_sha256")
        != {
            "manifest_sha256": v2_history.get("manifest_sha256"),
            "decision_sha256": v2_history.get("decision_sha256"),
            "launched_runner_sha256": v2_history.get("launched_runner_sha256"),
            "output_tree_sha256": v2_history.get("output_tree_sha256"),
        }
        or decision.get("effective_global_profile") != profile
        or decision.get("evidence_sha256") != artifact.get("evidence_sha256")
        or manifest.get("schema_version") != FOLLOWUP_RUN_SCHEMA
        or manifest.get("status") != "complete_pass"
        or manifest.get("v1_abort_preserved") is not True
        or manifest.get("v2_terminal_preserved") is not True
        or manifest.get("historical_fit_or_checkpoint_artifacts_reused") != 0
        or manifest.get("decision_sha256") != _sha256(decision_path)
        or manifest.get("lock_sha256") != _sha256(path)
        or manifest.get("study_signature_sha256")
        != artifact.get("study_signature_sha256")
    ):
        raise V2Phase3Error("numerical follow-up lock/decision/manifest chain is invalid")

    return {
        "fit_grid": expected_fit_grid,
        "eap_grid": int(profile["eap_grid"]),
        "quadrature_method": str(profile["quadrature_method"]),
        "linear_bound": float(profile["linear_bound"]),
        "tail_region": float(profile["tail_region"]),
        "verification_path": _display_path(path),
        "verification_sha256": _sha256(path),
        "followup_config_path": _display_path(followup_path),
        "followup_config_sha256": _sha256(followup_path),
        "effective_followup_config_sha256": _canonical_hash(followup),
        "frozen_contract_sha256": FOLLOWUP_FROZEN_CONTRACT_SHA256,
        "followup_decision_sha256": _sha256(decision_path),
        "followup_manifest_sha256": _sha256(manifest_path),
        "study_signature_sha256": artifact.get("study_signature_sha256"),
        "parent_decision_sha256": artifact.get("parent_decision_sha256"),
        "historical_v1_output_tree_sha256": v1_history.get("output_tree_sha256"),
        "historical_v2_output_tree_sha256": v2_history.get("output_tree_sha256"),
        "historical_v2_manifest_sha256": v2_history.get("manifest_sha256"),
        "evidence_sha256": dict(artifact["evidence_sha256"]),
        "fit_comparison_all_six_passed": comparison_summary,
        "all_six_exact_specifications_share_profile": True,
        "effective_global_profile": dict(profile),
        "status": "passed",
        "passed_spec_ids": sorted(passed_ids),
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
        raise V2Phase3Error("fitted bank is missing requested common criteria")
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


class V2Phase3Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_path = args.config.resolve()
        self.split_path = args.split_manifest.resolve()
        self.config = _read_json(self.config_path)
        self.split = _read_json(self.split_path)
        _check_frozen_config(self.config)
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
        )
        if (
            self.numerical.get("status") == "passed"
            and self.args.max_grid_nodes < int(self.numerical["eap_grid"])
        ):
            raise V2Phase3Error("max-grid-nodes cannot hold the verified EAP grid")

        cv = self.config.get("cross_validation") or {}
        expected_split_hash = str(
            cv.get("split_manifest_sha256")
            or (cv.get("split_manifest") or {}).get("sha256")
            if isinstance(cv.get("split_manifest"), Mapping)
            else cv.get("split_manifest_sha256")
            or ""
        )
        # The conditional expression above is intentionally normalized again for
        # configs that store the path as a plain string.
        expected_split_hash = str(cv.get("split_manifest_sha256") or expected_split_hash)
        if not expected_split_hash or _sha256(self.split_path) != expected_split_hash:
            raise V2Phase3Error("repeated split manifest hash does not match v2 config")

        baseline = self.config.get("baseline") or {}
        if int(baseline.get("models", 0)) != EXPECTED_MODELS:
            raise V2Phase3Error("v2 baseline must freeze the 52-model cohort")
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
                raise V2Phase3Error(f"{name} does not match the frozen config hash")
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
            raise V2Phase3Error("config and split manifest disagree on response matrix")

        source_skills = cm.configure_skills(args.skills)
        self.structure = cell_cv.build_structure(
            tuple(source_skills), args.dimensions, args.structure_name
        )
        if self.structure.n_dims != 1 or tuple(self.structure.labels) != (
            "instruction_following",
        ):
            raise V2Phase3Error("v2 remains frozen to one instruction-following dimension")
        self.q_by = cm.load_q_matrix(self.rubrics_path)
        cm.validate_matrix_bank_alignment(
            self.matrix, self.q_by, self.args.require_complete_bank
        )
        self.source_records = scenario_cv.source_records_by_id(self.rubrics_path)

        requested = args.out_dir
        if not requested.is_absolute():
            requested = Path.cwd() / requested
        self.output_requested = requested.absolute()
        self.output_dir = self.output_requested.resolve()
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
                raise V2Phase3Error(
                    f"runtime override changed frozen {key}: expected {expected_value}, got {value}"
                )

    def _study_signature(self) -> str:
        return _canonical_hash(
            {
                "schema": SCRIPT_SCHEMA,
                "config_sha256": _sha256(self.config_path),
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
            "script": "scripts/nested_scenario_cat_cv_v2.py",
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
                "files": self.code_hashes,
                "canonical_sha256": _canonical_hash(self.code_hashes),
            },
            "environment": self.environment,
            "numerical_lock": self.numerical,
            "cross_validation": {
                "repetitions": EXPECTED_REPEATS,
                "outer_folds_per_repeat": EXPECTED_OUTER_FOLDS,
                "inner_folds_per_outer": EXPECTED_INNER_FOLDS,
                "bootstrap_unit": "tutor_family",
                "repeated_rows_treated_as_independent": False,
            },
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
            if not manifest_path.is_file():
                raise V2Phase3Error("--resume requires an existing v2 manifest")
            manifest = _read_json(manifest_path)
            if (
                manifest.get("schema_version") != SCRIPT_SCHEMA
                or manifest.get("study_signature") != self.study_signature
            ):
                raise V2Phase3Error("resume target is not this exact v2 study")
            if manifest.get("status") == "phase3_complete":
                for name, provenance in (manifest.get("outputs") or {}).items():
                    path = self.output_dir / name
                    if not path.is_file() or _sha256(path) != provenance.get("sha256"):
                        raise V2Phase3Error(f"completed resume output failed hash: {name}")
                self._already_complete = True
                return
        elif self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise V2Phase3Error(
                f"output directory is not empty: {self.output_dir}; use --resume for "
                "an interrupted identical study or choose a new versioned output leaf"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            / "v2_fit_cache"
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
        temporary = cache_dir / "fit_arrays.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                A=np.asarray(fit["A"], dtype=float),
                b=np.asarray(fit["b"], dtype=float),
                R=np.asarray(fit["R"], dtype=float),
            )
        temporary.replace(array_path)
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
            raise V2Phase3Error(f"fit cache content hash mismatch: {cache_dir}")
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
                raise V2Phase3Error(f"fit cache mismatch for {key}: {cache_dir}")
        if manifest.get("arrays_sha256") != _sha256(array_path):
            raise V2Phase3Error(f"fit cache array hash mismatch: {cache_dir}")
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
                "calibration_specification": dict(
                    manifest["calibration_specification"]
                ),
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
            raise V2Phase3Error(f"{role}: held-out model leaked into calibration")
        if not set(train_ids) <= set(map(str, self.matrix.index)):
            raise V2Phase3Error(f"{role}: unknown item-fit model ID")
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
                max_iter=self.args.max_iter,
                tol=self.args.tol,
            )
            fit = cell_cv.fit_structure(
                self.matrix.loc[list(train_ids)], self.q_by, fit_args, self.structure
            )
        if fit.get("calibration_specification") != spec.canonical:
            raise V2Phase3Error(f"{role}/{spec.spec_id}: fitter changed exact specification")
        if not bool(fit.get("converged")):
            raise V2Phase3Error(f"{role}/{spec.spec_id}: calibration did not converge")
        A = np.asarray(fit["A"], dtype=float)
        b = np.asarray(fit["b"], dtype=float)
        R = np.asarray(fit["R"], dtype=float)
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b)) and np.all(np.isfinite(R))):
            raise V2Phase3Error(f"{role}/{spec.spec_id}: non-finite fitted parameters")
        active = A[np.abs(A) > 0]
        if spec.family == cm.ONE_PL and (
            active.size == 0 or not np.allclose(active, 1.0, rtol=0, atol=1e-12)
        ):
            raise V2Phase3Error(f"{role}/{spec.spec_id}: 1PL discrimination is not fixed at 1")
        if spec.family == cm.LOG_SHRINKAGE_2PL and (
            active.size == 0 or np.any(active <= 0)
        ):
            raise V2Phase3Error(f"{role}/{spec.spec_id}: shrinkage discrimination is invalid")
        bank, policy = scenario_cv.build_fold_bank(
            fit,
            self.structure,
            self.source_records,
            negative_policy=self.args.negative_policy,
        )
        admin = set(self.split_audit["administration_scenario_ids"])
        evaluation = set(self.split_audit["evaluation_scenario_ids"])
        if not (set(bank.scenario_ids) & admin and set(bank.scenario_ids) & evaluation):
            raise V2Phase3Error(
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
            / "v2_checkpoints"
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
            raise V2Phase3Error(f"checkpoint content hash mismatch: {path}")
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "study_signature": self.study_signature,
            "stage": stage,
            "context": dict(context),
        }
        for key, item in expected.items():
            if value.get(key) != item:
                raise V2Phase3Error(f"checkpoint mismatch for {key}: {path}")
        rows = value.get("rows")
        if not isinstance(rows, list) or value.get("rows_sha256") != _canonical_hash(rows):
            raise V2Phase3Error(f"checkpoint rows failed integrity: {path}")
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

        if not bundles:
            common_ids: set[str] = set()
        else:
            common_ids = set.intersection(
                *(set(bundle.bank.criterion_ids) for bundle in bundles.values())
            )
        administration_scenarios = set(
            self.split_audit["administration_scenario_ids"]
        )
        evaluation_scenarios = set(self.split_audit["evaluation_scenario_ids"])
        if bundles:
            exemplar = next(iter(bundles.values())).bank
            scenario_by_item = dict(
                zip(exemplar.criterion_ids, exemplar.scenario_ids, strict=True)
            )
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
                raise V2Phase3Error(
                    f"inner {repeat}/{outer_fold}/{inner_fold}: no common admin/eval support"
                )
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
                    "fit_bank_items": len(bundle.bank.criterion_ids),
                    "fit_dropped_negative_items": len(bundle.bank.dropped_negative_items),
                    "common_support_verified": True,
                    "common_support_sha256": support_hash,
                    "n_common_admin_items": len(common_admin_ids),
                    "n_common_evaluation_items": len(common_eval_ids),
                }
                try:
                    response = scat.responses_for_bank(
                        self.matrix.loc[model], common_bank
                    )
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
                        raise V2Phase3Error("no observed common disjoint evaluation cells")
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
                            "model_log_loss": float(stats["log_loss_sum"])
                            / int(stats["n_cells"]),
                            "observed_common_cells_sha256": _canonical_hash(
                                sorted(observed_eval)
                            ),
                        }
                    )
                except (
                    V2Phase3Error,
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
                raise V2Phase3Error(
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
        for inner in outer["inner_folds"]:
            inner_fold = int(inner["inner_fold"])
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
                    V2Phase3Error,
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
        aggregate = aggregate_calibration_evidence(
            evidence_rows, self.specs, self.split_audit["model_to_family"]
        )
        for row in aggregate:
            row.update({"repeat": repeat, "outer_fold": outer_fold})
        try:
            selected_spec, annotated, selection = select_calibration_spec_one_se(
                aggregate, self.specs
            )
            status = "selection_complete"
            error = ""
        except V2Phase3Error as selection_error:
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
                "within_one_se_cache_keys": selection.get(
                    "within_one_se_cache_keys", []
                ),
                "selection_basis": selection.get("selection_basis"),
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
            raise V2Phase3Error(
                f"outer {repeat}/{outer_fold}: policy/model coverage is incomplete"
            )
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
        spec = next(
            item for item in self.specs if item.spec_id == panel["selected_spec_id"]
        )
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
            V2Phase3Error,
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
            str(row["model"])
            for row in rows
            if row.get("policy_id") == PRIMARY_POLICY_ID
        }
        if observed_primary != expected_primary:
            raise V2Phase3Error(f"outer {repeat}/{outer_fold}: primary coverage incomplete")
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

    def _metric_seed(self, label: str) -> int:
        return v1._bootstrap_seed(self.args.seed, f"phase3-v2:{label}")

    def _gate_rows(
        self, outer_rows: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        gate_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        gates = self._gates()
        for repeat in range(EXPECTED_REPEATS):
            repeat_rows = [row for row in outer_rows if int(row["repeat"]) == repeat]
            for outer_fold in range(EXPECTED_OUTER_FOLDS):
                panel_rows = [
                    row for row in repeat_rows if int(row["outer_fold"]) == outer_fold
                ]
                for policy in self.policies:
                    subset = [
                        row for row in panel_rows if row.get("policy_id") == policy.policy_id
                    ]
                    metrics = aggregate_policy_rows_clustered(
                        subset,
                        self.split_audit["model_to_family"],
                        seed=self._metric_seed(
                            f"panel:{repeat}:{outer_fold}:{policy.policy_id}"
                        ),
                        replicates=self.args.metric_bootstrap_replicates,
                        label=f"panel:{repeat}:{outer_fold}:{policy.policy_id}",
                    )
                    if policy.role == "primary":
                        passed, failures, checks = v1.apply_absolute_gates(metrics, gates)
                        gate_status = "pass" if passed else "fail"
                    else:
                        passed, failures, checks = None, [], {}
                        gate_status = "diagnostic_only_non_promotable"
                    gate_rows.append(
                        {
                            "scope": "outer_panel",
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "policy_id": policy.policy_id,
                            "policy_role": policy.role,
                            **metrics,
                            "all_gates_pass": passed,
                            "gate_status": gate_status,
                            "failed_gates": json.dumps(failures),
                            "gate_checks": json.dumps(checks, sort_keys=True),
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
                subset = [
                    row for row in repeat_rows if row.get("policy_id") == policy.policy_id
                ]
                metrics = aggregate_policy_rows_clustered(
                    subset,
                    self.split_audit["model_to_family"],
                    seed=self._metric_seed(f"repeat:{repeat}:{policy.policy_id}"),
                    replicates=self.args.metric_bootstrap_replicates,
                    label=f"repeat:{repeat}:{policy.policy_id}",
                )
                if policy.role == "primary":
                    passed, failures, checks = v1.apply_absolute_gates(metrics, gates)
                    gate_status = "pass" if passed else "fail"
                else:
                    passed, failures, checks = None, [], {}
                    gate_status = "diagnostic_only_non_promotable"
                gate_rows.append(
                    {
                        "scope": "repetition",
                        "repeat": repeat,
                        "outer_fold": None,
                        "policy_id": policy.policy_id,
                        "policy_role": policy.role,
                        **metrics,
                        "all_gates_pass": passed,
                        "gate_status": gate_status,
                        "failed_gates": json.dumps(failures),
                        "gate_checks": json.dumps(checks, sort_keys=True),
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
            policy_rows = [
                row for row in outer_rows if row.get("policy_id") == policy.policy_id
            ]
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
                        if row.get(field) is not None
                        and math.isfinite(float(row[field]))
                    ]
                    record[f"mean_{field}"] = float(np.mean(values)) if values else None
                for field in boolean_fields:
                    record[f"rate_{field}"] = (
                        float(np.mean([bool(row.get(field)) for row in rows]))
                        if rows
                        else None
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
                        "baseline_replay_success": row.get(
                            "mean_baseline_scenarios_administered"
                        )
                        is not None,
                        "cat_mwle_converged": row.get("mean_theta_cat_mwle") is not None,
                        "theta_reference": row.get("mean_theta_reference"),
                        "theta_cat_mwle": row.get("mean_theta_cat_mwle"),
                        "cat_scenarios_administered": row.get(
                            "mean_cat_scenarios_administered"
                        ),
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
                    "n_families": len(
                        {str(row["model_family"]) for row in complete}
                    ),
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
            subset = [
                row for row in outer_rows if row.get("panel_id") == panel["panel_id"]
            ]
            for row in detect_duplicate_cat_paths(subset, self.policies):
                duplicate_rows.append(
                    {
                        "panel_id": panel["panel_id"],
                        "repeat": panel["repeat"],
                        "outer_fold": panel["outer_fold"],
                        **row,
                    }
                )
        _atomic_csv(
            self.output_dir / "duplicate_cat_paths.csv", pd.DataFrame(duplicate_rows)
        )

        per_model, cross_metrics = self._aggregate_repeated_models(outer_rows)
        _atomic_csv(
            self.output_dir / "cross_repeat_per_model.csv", pd.DataFrame(per_model)
        )
        _atomic_csv(
            self.output_dir / "cross_repeat_metrics.csv", pd.DataFrame(cross_metrics)
        )

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
                if int(row["repeat"]) == repeat
                and row.get("policy_id") == PRIMARY_POLICY_ID
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
        failures: list[str] = []
        if not coverage_pass:
            failures.append("incomplete_outer_coverage")
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
        _atomic_json(self.output_dir / "v2_decision.json", decision)

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
            _atomic_csv(
                repeat_dir / "inner_calibration_model_results.csv", repeat_inner
            )
            _atomic_csv(repeat_dir / "outer_oof_per_model.csv", repeat_outer)
            _atomic_json(
                repeat_dir / "calibration_model_choices.json",
                {
                    **{key: value for key, value in choices.items() if key != "panels"},
                    "repeat": repeat,
                    "panels": [
                        row for row in choices["panels"] if row["repeat"] == repeat
                    ],
                },
            )

        manifest = self._base_manifest("phase3_complete", created_at=started_at)
        manifest["completed_at"] = _utcnow()
        manifest["phase3_decision"] = {
            "status": decision["status"],
            "phase3_pass": phase3_pass,
            "phase4_authorized": phase3_pass,
        }
        manifest["runtime"] = {
            "seed": self.args.seed,
            "top_n": self.args.top_n,
            "max_scenarios": self.args.max_scenarios,
            "minimum_scored_criteria": self.args.minimum_scored_criteria,
            "max_iter": self.args.max_iter,
            "tol": self.args.tol,
            "mwle_ridge": self.args.mwle_ridge,
            "max_grid_nodes": self.args.max_grid_nodes,
            "negative_policy": self.args.negative_policy,
            "metric_bootstrap_replicates": self.args.metric_bootstrap_replicates,
        }
        manifest["outputs"] = {}
        for name in REQUIRED_OUTPUTS:
            if name == "manifest.json":
                continue
            path = self.output_dir / name
            if not path.is_file():
                raise V2Phase3Error(f"required Phase-3 output was not written: {name}")
            manifest["outputs"][name] = {
                "path": _display_path(path),
                "sha256": _sha256(path),
            }
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
                "administration_scenario_ids": self.split_audit[
                    "administration_scenario_ids"
                ],
                "evaluation_scenario_ids": self.split_audit[
                    "evaluation_scenario_ids"
                ],
            },
        }

    def run(self) -> int:
        self._prepare_output()
        if self._already_complete:
            print(f"verified completed v2 Phase-3 run -> {self.output_dir}")
            return 0
        started_at = _utcnow()
        _atomic_json(
            self.output_dir / "fold_assignments.json", self._fold_assignment_payload()
        )
        if self.args.plan_only:
            ready = self.numerical.get("status") == "passed"
            manifest = self._base_manifest(
                "plan_only_ready" if ready else "plan_only_blocked"
            )
            manifest["official_run_ready"] = ready
            manifest["blockers"] = (
                []
                if ready
                else [
                    (
                        "numerical follow-up completed without a promotable lock"
                        if self.numerical.get("status") == "failed_blocker"
                        else "new-family numerical equivalence lock is pending"
                    )
                ]
            )
            manifest["produced_outputs"] = ["fold_assignments.json", "manifest.json"]
            manifest["not_run"] = {
                "item_fits": True,
                "inner_calibration_selection": True,
                "outer_cat": True,
                "phase3_decision": True,
            }
            _atomic_json(self.output_dir / "manifest.json", manifest)
            print(
                f"validated v2 Phase-3 plan ({'ready' if ready else 'blocked'}) "
                f"-> {self.output_dir}"
            )
            return 0
        if self.numerical.get("status") != "passed":
            raise V2Phase3Error("official Phase 3 is blocked by numerical verification")

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
                panel, evidence = self._select_panel(repeat, outer)
                panels.append(panel)
                inner_results.extend(evidence)
                outer_lookup[(repeat, outer_fold)] = outer
                print(
                    f"selected {panel.get('selected_spec_id')} for "
                    f"repeat {repeat}/outer {outer_fold} ({panel['status']})"
                )
        if len(panels) != EXPECTED_REPEATS * EXPECTED_OUTER_FOLDS:
            raise V2Phase3Error("selection phase did not produce all 25 panel decisions")
        selection_lock = {
            "schema_version": SCRIPT_SCHEMA,
            "study_signature": self.study_signature,
            "status": "locked_before_any_outer_cat_evaluation",
            "outer_outcomes_used": False,
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
                raise V2Phase3Error("resume selection lock differs from recomputed inner evidence")
        else:
            _atomic_json(selection_lock_path, selection_lock)

        outer_rows: list[dict[str, Any]] = []
        for panel in panels:
            key = (int(panel["repeat"]), int(panel["outer_fold"]))
            rows = self._evaluate_outer_panel(panel, outer_lookup[key])
            outer_rows.extend(rows)
            print(
                f"evaluated repeat {key[0]}/outer {key[1]}: "
                f"{len(rows)} frozen policy/model rows"
            )
        self._write_phase3_outputs(
            panels=panels,
            inner_results=inner_results,
            outer_rows=outer_rows,
            started_at=started_at,
        )
        print(f"wrote complete v2 Phase-3 decision -> {self.output_dir}")
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
        "schema_version": "infobench-v2-phase3-synthetic-preflight-v1",
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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--numerical-followup-config",
        type=Path,
        default=DEFAULT_NUMERICAL_FOLLOWUP_CONFIG,
    )
    parser.add_argument("--numerical-lock", type=Path, default=DEFAULT_NUMERICAL_LOCK)
    parser.add_argument("--skills", default=",".join(DEFAULT_SKILLS))
    parser.add_argument("--dimensions", default=DEFAULT_DIMENSIONS)
    parser.add_argument("--structure-name", default="infobench_overall_1d_v2")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-scenarios", type=int, default=50)
    parser.add_argument("--minimum-scored-criteria", type=int, default=15)
    parser.add_argument("--mwle-ridge", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=200)
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
    return parser


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.synthetic_preflight:
        if args.resume or args.plan_only:
            raise V2Phase3Error("--synthetic-preflight cannot combine with run modes")
        return
    if args.top_n < 1 or args.max_scenarios < 1 or args.minimum_scored_criteria < 0:
        raise V2Phase3Error("invalid CAT runtime limits")
    if args.max_iter < 1 or args.tol <= 0 or args.mwle_ridge <= 0:
        raise V2Phase3Error("invalid optimizer settings")
    if args.max_grid_nodes < 1:
        raise V2Phase3Error("max-grid-nodes must be positive")
    if args.metric_bootstrap_replicates < 100:
        raise V2Phase3Error("family bootstrap requires at least 100 replicates")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        _validate_runtime_args(args)
        if args.synthetic_preflight:
            return _run_synthetic_preflight(args.out_dir.resolve())
        return V2Phase3Runner(args).run()
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
        V2Phase3Error,
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
