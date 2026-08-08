"""Leakage-free nested person-CV selection of scenario-CAT settings.

This is the Phase-3 driver for the InFoBench calibration remediation study.  It
uses a *pre-frozen* grouped model-fold manifest and scenario split.  Item
parameters are fit only on the applicable training model IDs, cached once per
fit, and reused across the full floor x SE-target x selector factorial.  For a
held-out model, only administration-pool scenarios can affect the CAT path;
responses from the disjoint evaluation pool are opened afterward solely to
score predictions.

The command makes no tutor-model or judge-model calls.  It consumes the frozen
binary response matrix and reuses the existing calibration, CAT replay, EAP,
MWLE, and prediction-metric implementations.

Numerical quadrature settings are deliberately not selected here.  Pass
``--fit-grid`` and ``--eap-grid`` explicitly, or pass a separately locked Phase-2
dense-grid manifest.  Ridge is statistical rather than merely numerical: each
outer fold selects it from the configured candidates using only its four inner
folds, before any outer-test response is opened.  Phase 3 retains every
configuration tied after the preregistered p90- then mean-length filters.  Total
uncertainty and the trace tie-break are deliberately deferred to Phase 4;
Phase 3 never calls one of those finalists the winner.  Outer-fold results can
never choose the grids, ridge, or finalist set.

Examples
--------
Validate the complete plan without fitting or replaying CAT::

    python scripts/nested_scenario_cat_cv.py \
      --fit-grid 25 --eap-grid 41 --plan-only

Start a new run (the output directory must not already contain a study)::

    python scripts/nested_scenario_cat_cv.py \
      --fit-grid 25 --eap-grid 41

Resume from completed inner/outer checkpoints::

    python scripts/nested_scenario_cat_cv.py \
      --fit-grid 25 --eap-grid 41 --resume

``--fresh`` is an explicit request to replace only this driver's output
directory.  Historical calibration outputs are never read as a resume target
or modified by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import Counter
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

from tutor_cat.mirt import pass_probability  # noqa: E402

from scripts import kfold_cv_mirt as cell_cv  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402
from scripts import scenario_kfold_estimator_cv as scenario_cv  # noqa: E402

cm = cell_cv.cm

DEFAULT_CONFIG = ROOT / "configs" / "infobench_calibration_remediation_v1.json"
DEFAULT_SPLITS = ROOT / "configs" / "infobench_remediation_splits_v1.manifest.json"
DEFAULT_OUTPUT = (
    ROOT
    / "runs"
    / "calibration"
    / "InFoBench_remediation_v1_quadrature_resolution"
    / "nested_cat_cv"
)
DEFAULT_SKILLS = ("content", "format", "number", "style", "linguistic")
DEFAULT_DIMENSIONS = "instruction_following=content+format+number+style+linguistic"
SCRIPT_SCHEMA = "infobench-nested-scenario-cat-cv-v3"
FINALIST_SCHEMA = "infobench-phase3-finalist-shortlist-v2"
CHECKPOINT_SCHEMA = "infobench-nested-scenario-cat-checkpoint-v2"
EXPECTED_SELECTION_ORDER = (
    "lowest_p90_scenario_count",
    "lowest_mean_scenario_count",
    "lowest_total_uncertainty",
    "prefer_trace",
)
CODE_DEPENDENCY_PATHS = (
    ROOT / "scripts" / "nested_scenario_cat_cv.py",
    ROOT / "scripts" / "kfold_cv_mirt.py",
    ROOT / "scripts" / "calibrate_mirt.py",
    ROOT / "scripts" / "calibrate_partial.py",
    ROOT / "scripts" / "scenario_cat_lib.py",
    ROOT / "scripts" / "scenario_kfold_estimator_cv.py",
    ROOT / "tutor_cat" / "__init__.py",
    ROOT / "tutor_cat" / "dataio.py",
    ROOT / "tutor_cat" / "engine.py",
    ROOT / "tutor_cat" / "mirt.py",
    ROOT / "tutor_cat" / "mcq_irt" / "__init__.py",
    ROOT / "tutor_cat" / "mcq_irt" / "calibrate.py",
    ROOT / "tutor_cat" / "schemas.py",
    ROOT / "tutor_cat" / "selector.py",
    ROOT / "tutor_cat" / "skill_structure.py",
)

REQUIRED_OUTPUTS = (
    "fold_assignments.json",
    "inner_ridge_results.csv",
    "inner_candidate_results.csv",
    "outer_fold_finalists.json",
    "outer_oof_per_model.csv",
    "disjoint_prediction_metrics.csv",
    "outer_metrics.json",
    "config_finalist_frequency.csv",
    "manifest.json",
)
DEFERRED_PHASE4_OUTPUTS = (
    "outer_total_se.csv",
    "outer_order_stability.csv",
)


class NestedCVError(RuntimeError):
    """Raised when a frozen-design or leakage invariant is violated."""


@dataclass(frozen=True, order=True)
class Candidate:
    minimum_scenarios: int
    conditional_se_target: float
    selector: str

    @property
    def candidate_id(self) -> str:
        se = f"{self.conditional_se_target:.2f}".replace(".", "p")
        return (
            f"floor_{self.minimum_scenarios:02d}__se_{se}__"
            f"selector_{self.selector}"
        )


@dataclass(frozen=True)
class DenseSettings:
    fit_grid: int
    eap_grid: int
    source: str
    source_sha256: str | None = None
    manifest_ridge_field_ignored: float | None = None
    quadrature_method: str = "gauss_hermite"
    linear_bound: float = 8.0
    quadrature_axis_sha256: str | None = None
    quadrature_log_prior_sha256: str | None = None
    effective_node_count: int | None = None
    native_lock_verified: bool = False


@dataclass
class FitBundle:
    fit: dict[str, Any]
    bank: scat.FittedBank
    quadrature: scat.Quadrature
    cache_key: str
    training_model_ids: tuple[str, ...]
    cache_dir: Path
    policy_diagnostics: dict[str, Any]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
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
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NestedCVError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise NestedCVError(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def _code_dependency_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in CODE_DEPENDENCY_PATHS:
        if not path.is_file():
            raise NestedCVError(f"required code dependency is missing: {path}")
        hashes[_display_path(path)] = _sha256(path)
    return hashes


def _dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "scipy": importlib.metadata.version("scipy"),
    }


def load_candidates(config: Mapping[str, Any]) -> list[Candidate]:
    raw = config.get("cat_candidates") or {}
    if raw.get("run_full_factorial") is not True:
        raise NestedCVError("cat_candidates.run_full_factorial must be true")
    try:
        candidates = [
            Candidate(int(floor), float(target), str(selector))
            for floor, target, selector in itertools.product(
                raw["minimum_scenarios"],
                raw["conditional_se_targets"],
                raw["selectors"],
            )
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise NestedCVError(f"invalid CAT candidate grid: {error}") from error
    if len(candidates) != 48 or len({item.candidate_id for item in candidates}) != 48:
        raise NestedCVError(
            f"the preregistered CAT grid must contain 48 unique candidates, got {len(candidates)}"
        )
    for item in candidates:
        if item.minimum_scenarios < 0 or item.conditional_se_target <= 0:
            raise NestedCVError(f"invalid CAT candidate: {item}")
        if item.selector not in {"trace", "dopt"}:
            raise NestedCVError(f"unsupported selector in CAT grid: {item.selector}")
    return candidates


def resolve_dense_settings(args: argparse.Namespace) -> DenseSettings:
    explicit = (args.fit_grid, args.eap_grid)
    if args.dense_grid_manifest is not None:
        if any(value is not None for value in explicit):
            raise NestedCVError(
                "use either --dense-grid-manifest or explicit fit/EAP grids; "
                "do not combine them"
            )
        path = args.dense_grid_manifest.resolve()
        value = _read_json(path)
        if value.get("schema_version") == "infobench-dense-grid-lock-v1":
            raise NestedCVError(
                "legacy dense-grid locks are not valid for Phase 3; pass the native "
                "quadrature-resolution grid_lock.json with every numerical gate"
            )
        else:
            # Native lock emitted only after common-support, primary-EAP, bound,
            # and cross-family checks.  A partial two-gate lock is not enough.
            fit_gate = value.get("fit_gate") or {}
            eap_gate = value.get("eap_gate") or {}
            bound_gate = value.get("bound_gate") or {}
            cross_family_gate = value.get("cross_family_gate") or {}
            fit_comparisons = list((fit_gate.get("comparisons") or {}).values())
            eap_comparisons = list((eap_gate.get("comparisons") or {}).values())
            common_comparisons_pass = bool(fit_comparisons) and all(
                comparison.get("passed") is True
                and comparison.get(
                    "common_cell_keys_identical_after_explicit_intersection"
                )
                is True
                and comparison.get("common_cell_labels_identical") is True
                for comparison in fit_comparisons
            )
            if not (
                fit_gate.get("passed") is True
                and fit_gate.get("require_every_comparison") is True
                and common_comparisons_pass
                and eap_gate.get("passed") is True
                and eap_comparisons
                and all(comparison.get("passed") is True for comparison in eap_comparisons)
                and bound_gate.get("passed") is True
                and bound_gate.get("model_keys_identical") is True
                and bound_gate.get("posterior_tail_mass_passed") is True
                and cross_family_gate.get("passed") is True
                and cross_family_gate.get("model_keys_identical") is True
                and value.get("ridge_scheduled_only_after_this_lock") is True
                and value.get("ridge_scheduled_only_after_every_numerical_gate") is True
                and value.get("production_ridge_selection") == "deferred_to_nested_cv"
            ):
                raise NestedCVError(
                    "native Phase-2 grid lock has not passed every common/EAP/bound/"
                    "cross-family gate and both post-lock ridge flags"
                )
            study_path = path.parent.parent / "study_manifest.json"
            study = _read_json(study_path)
            if study.get("status") != "phase2_complete":
                raise NestedCVError("parent dense-grid study is not phase2_complete")
            selected = value
            study_result = study.get("result") or {}
            if (
                int(fit_gate.get("locked_fit_grid", -1)) != int(selected.get("fit_grid", -2))
                or int(eap_gate.get("locked_eap_grid", -1))
                != int(selected.get("eap_grid", -2))
                or int(study_result.get("fit_grid", -3))
                != int(selected.get("fit_grid", -4))
                or int(study_result.get("eap_grid", -3))
                != int(selected.get("eap_grid", -4))
                or study_result.get("grid_lock") != selected
            ):
                raise NestedCVError("native grid lock disagrees with its gates or study manifest")
        try:
            settings = DenseSettings(
                fit_grid=int(selected["fit_grid"]),
                eap_grid=int(selected["eap_grid"]),
                source=_display_path(path),
                source_sha256=_sha256(path),
                manifest_ridge_field_ignored=(
                    float(selected["ridge"]) if selected.get("ridge") is not None else None
                ),
                quadrature_method=str(
                    selected.get("quadrature_method", "gauss_hermite")
                ),
                linear_bound=float(selected.get("linear_bound", 8.0)),
                quadrature_axis_sha256=(
                    str(selected["quadrature_axis_sha256"])
                    if selected.get("quadrature_axis_sha256")
                    else None
                ),
                quadrature_log_prior_sha256=(
                    str(selected["quadrature_log_prior_sha256"])
                    if selected.get("quadrature_log_prior_sha256")
                    else None
                ),
                effective_node_count=(
                    int(selected["effective_node_count"])
                    if selected.get("effective_node_count") is not None
                    else None
                ),
                native_lock_verified=True,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise NestedCVError(f"invalid dense-grid lock: {error}") from error
    else:
        if any(value is None for value in explicit):
            raise NestedCVError(
                "pass --fit-grid and --eap-grid explicitly, or pass a locked "
                "--dense-grid-manifest"
            )
        settings = DenseSettings(
            fit_grid=int(args.fit_grid),
            eap_grid=int(args.eap_grid),
            source="explicit_cli_pre_nested_lock",
            quadrature_method=str(args.eap_quadrature_method),
            linear_bound=float(args.eap_linear_bound),
        )
    if settings.fit_grid < 2 or settings.eap_grid < 2:
        raise NestedCVError(f"invalid dense numerical settings: {settings}")
    if settings.quadrature_method not in {"gauss_hermite", "normal_trapezoid"}:
        raise NestedCVError(
            f"invalid EAP quadrature method: {settings.quadrature_method!r}"
        )
    if not math.isfinite(settings.linear_bound) or settings.linear_bound <= 0:
        raise NestedCVError("EAP linear bound must be finite and positive")
    if settings.quadrature_method == "normal_trapezoid" and settings.eap_grid % 2 == 0:
        raise NestedCVError("normal_trapezoid EAP grid must be odd and include theta=0")
    if args.dense_grid_manifest is not None and settings.quadrature_method != "gauss_hermite":
        if not settings.quadrature_axis_sha256 or not settings.quadrature_log_prior_sha256:
            raise NestedCVError("nonhistorical quadrature lock lacks axis/prior hashes")
        quadrature = scat.build_quadrature(
            1,
            settings.eap_grid,
            np.eye(1),
            max_nodes=max(settings.eap_grid, 50_000),
            method=settings.quadrature_method,
            linear_bound=settings.linear_bound,
        )
        if _array_sha256(quadrature.grid[:, 0]) != settings.quadrature_axis_sha256:
            raise NestedCVError("locked EAP quadrature axis hash does not reproduce")
        if _array_sha256(quadrature.log_prior) != settings.quadrature_log_prior_sha256:
            raise NestedCVError("locked EAP quadrature prior hash does not reproduce")
        if (
            settings.effective_node_count is not None
            and len(quadrature.grid) != settings.effective_node_count
        ):
            raise NestedCVError("locked EAP effective node count does not reproduce")
    return settings


def audit_split_manifest(
    split: Mapping[str, Any],
    *,
    expected_models: Iterable[str] | None = None,
    expected_scenarios: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fail closed on model/family/scenario leakage in the frozen design."""

    if split.get("schema_version") != "infobench-remediation-nested-splits-v1":
        raise NestedCVError("unexpected frozen split schema_version")
    outer_folds = split.get("outer_folds") or []
    if len(outer_folds) != 5:
        raise NestedCVError(f"expected five outer folds, found {len(outer_folds)}")
    model_to_family = {
        str(model): str(family)
        for model, family in (split.get("model_to_family") or {}).items()
    }
    all_models = set(model_to_family)
    if len(all_models) != 52:
        raise NestedCVError(f"expected 52 frozen models, found {len(all_models)}")
    if expected_models is not None and set(map(str, expected_models)) != all_models:
        raise NestedCVError("response-matrix model IDs do not equal the frozen split IDs")

    seen_test: set[str] = set()
    runtime_folds: list[dict[str, Any]] = []
    model_to_outer: dict[str, int] = {}
    for expected_index, outer in enumerate(outer_folds):
        fold = int(outer.get("outer_fold", -1))
        if fold != expected_index:
            raise NestedCVError("outer folds must be numbered deterministically 0..4")
        test = set(map(str, outer.get("test_model_ids") or []))
        train = set(map(str, outer.get("train_model_ids") or []))
        if test & train or test | train != all_models:
            raise NestedCVError(f"outer fold {fold} has a train/test leakage or omission")
        if seen_test & test:
            raise NestedCVError(f"outer fold {fold} repeats an outer-test model")
        seen_test |= test
        for model in test:
            model_to_outer[model] = fold

        inner_folds = outer.get("inner_folds") or []
        if len(inner_folds) != 4:
            raise NestedCVError(f"outer fold {fold} must have four inner folds")
        validation_seen: list[str] = []
        runtime_inner: list[dict[str, Any]] = []
        for expected_inner, inner in enumerate(inner_folds):
            inner_index = int(inner.get("inner_fold", -1))
            if inner_index != expected_inner:
                raise NestedCVError(
                    f"outer fold {fold} inner folds must be numbered 0..3"
                )
            fit_ids = set(map(str, inner.get("fit_model_ids") or []))
            validation = set(map(str, inner.get("validation_model_ids") or []))
            if test & (fit_ids | validation):
                raise NestedCVError(
                    f"outer-test model leaked into outer fold {fold}, inner fold {inner_index}"
                )
            if fit_ids & validation or fit_ids | validation != train:
                raise NestedCVError(
                    f"inner fold {fold}/{inner_index} does not partition outer training IDs"
                )
            validation_seen.extend(sorted(validation))
            runtime_inner.append(
                {
                    "inner_fold": inner_index,
                    "fit_model_ids": sorted(fit_ids),
                    "validation_model_ids": sorted(validation),
                }
            )
        if sorted(validation_seen) != sorted(train):
            raise NestedCVError(
                f"outer fold {fold}: each outer-training model must validate exactly once"
            )
        runtime_folds.append(
            {
                "outer_fold": fold,
                "train_model_ids": sorted(train),
                "test_model_ids": sorted(test),
                "inner_folds": runtime_inner,
            }
        )
    if seen_test != all_models:
        raise NestedCVError("the five outer-test folds do not cover all frozen models")

    for family, members_raw in (split.get("model_families") or {}).items():
        members = set(map(str, members_raw))
        if not members or not members <= all_models:
            raise NestedCVError(f"family {family!r} has invalid members")
        folds = {model_to_outer[model] for model in members}
        if len(folds) != 1:
            raise NestedCVError(f"model family {family!r} crosses outer-fold boundaries")

    scenario_split = split.get("scenario_split") or {}
    administration = set(map(str, scenario_split.get("administration_scenario_ids") or []))
    evaluation = set(map(str, scenario_split.get("evaluation_scenario_ids") or []))
    if administration & evaluation:
        raise NestedCVError("administration and evaluation scenarios overlap")
    if len(administration) != 400 or len(evaluation) != 100:
        raise NestedCVError(
            "frozen scenario split must contain exactly 400 administration and 100 "
            "evaluation scenarios"
        )
    if expected_scenarios is not None:
        expected = set(map(str, expected_scenarios))
        if administration | evaluation != expected:
            raise NestedCVError("scenario records do not equal the frozen scenario split")
    return {
        "status": "passed",
        "outer_folds": runtime_folds,
        "administration_scenario_ids": sorted(administration),
        "evaluation_scenario_ids": sorted(evaluation),
        "model_to_family": model_to_family,
        "model_to_outer_fold": model_to_outer,
    }


def subset_fitted_bank(
    bank: scat.FittedBank, allowed_scenario_ids: Iterable[str]
) -> scat.FittedBank:
    """Return a structural bank subset; no item parameters are refit or altered."""

    allowed = set(map(str, allowed_scenario_ids))
    indices = np.asarray(
        [index for index, scenario in enumerate(bank.scenario_ids) if scenario in allowed],
        dtype=int,
    )
    if indices.size == 0:
        raise NestedCVError("scenario subset contains no fitted criteria")
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


def evaluation_item_indices(
    bank: scat.FittedBank, evaluation_scenario_ids: Iterable[str]
) -> np.ndarray:
    allowed = set(map(str, evaluation_scenario_ids))
    indices = np.asarray(
        [index for index, scenario in enumerate(bank.scenario_ids) if scenario in allowed],
        dtype=int,
    )
    if indices.size == 0:
        raise NestedCVError("evaluation pool contains no fitted criteria")
    return indices


def prediction_sufficient_statistics(
    row: pd.Series,
    bank: scat.FittedBank,
    item_indices: Sequence[int] | np.ndarray,
    theta: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Score disjoint cells using shared IRT probability and metric functions."""

    responses = scat.responses_for_bank(row, bank)
    indices = np.asarray(item_indices, dtype=int)
    indices = indices[np.isfinite(responses[indices])]
    if indices.size == 0:
        return {
            "n_cells": 0,
            "log_loss_sum": 0.0,
            "brier_sum": 0.0,
            "correct_count": 0,
            "observed_pass_rate": None,
            "predicted_pass_rate": None,
        }
    ability = np.asarray(theta, dtype=float)
    if ability.shape != (bank.n_dims,) or not np.all(np.isfinite(ability)):
        raise ValueError("theta is non-finite or does not match the fitted dimensions")
    labels = responses[indices]
    probabilities = np.asarray(
        [
            pass_probability(ability, bank.A[index], bank.Q[index], bank.b[index])
            for index in indices
        ],
        dtype=float,
    )
    metrics = cell_cv.metrics(labels, probabilities)
    return {
        "n_cells": int(indices.size),
        "log_loss_sum": float(metrics["log_loss"] * indices.size),
        "brier_sum": float(metrics["brier"] * indices.size),
        "correct_count": int(round(float(metrics["accuracy"]) * indices.size)),
        "observed_pass_rate": float(labels.mean()),
        "predicted_pass_rate": float(probabilities.mean()),
    }


def _empty_prediction_stats() -> dict[str, Any]:
    return {
        "n_cells": 0,
        "log_loss_sum": 0.0,
        "brier_sum": 0.0,
        "correct_count": 0,
        "observed_pass_rate": None,
        "predicted_pass_rate": None,
    }


def _ridge_slug(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def evaluate_ridge_model(
    *,
    model: str,
    row: pd.Series,
    bank: scat.FittedBank,
    administration_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    quadrature: scat.Quadrature,
    ridge: float,
) -> dict[str, Any]:
    """Inner-only EAP prediction evidence for selecting calibration ridge.

    Ability is estimated from the frozen administration scenario pool and the
    disjoint evaluation cells are scored afterward.  No CAT path or outer-test
    response participates in this statistical-hyperparameter choice.
    """

    try:
        responses = scat.responses_for_bank(row, bank)
        estimate = scat.batch_eap(
            responses,
            bank.A,
            bank.b,
            quadrature,
            item_indices=administration_indices,
        )
        stats = prediction_sufficient_statistics(
            row, bank, evaluation_indices, estimate.theta
        )
        if int(stats["n_cells"]) == 0:
            raise NestedCVError(f"{model}: no observed disjoint evaluation cells")
    except (NestedCVError, scat.OfflineStudyError, ValueError, np.linalg.LinAlgError) as error:
        return {
            "model": str(model),
            "ridge": float(ridge),
            "status": "prediction_error",
            "error": f"{type(error).__name__}: {error}",
            "n_cells": 0,
            "log_loss_sum": 0.0,
            "brier_sum": 0.0,
            "model_log_loss": None,
        }
    return {
        "model": str(model),
        "ridge": float(ridge),
        "status": "ok",
        "error": "",
        **stats,
        "model_log_loss": float(stats["log_loss_sum"]) / int(stats["n_cells"]),
    }


def aggregate_ridge_evidence(
    rows: Sequence[Mapping[str, Any]], ridge: float
) -> dict[str, Any]:
    subset = [row for row in rows if float(row["ridge"]) == float(ridge)]
    valid = [row for row in subset if row.get("status") == "ok"]
    model_losses = np.asarray(
        [float(row["model_log_loss"]) for row in valid], dtype=float
    )
    fit_cache_keys = sorted({str(row.get("fit_cache_key")) for row in subset})
    n_cells = sum(int(row.get("n_cells") or 0) for row in valid)
    mean = float(model_losses.mean()) if model_losses.size else None
    se = (
        float(model_losses.std(ddof=1) / math.sqrt(model_losses.size))
        if model_losses.size >= 2
        else None
    )
    return {
        "ridge": float(ridge),
        "n_models": len(subset),
        "n_inner_folds": len({row.get("inner_fold") for row in subset}),
        "fit_cache_keys": json.dumps(fit_cache_keys, ensure_ascii=False),
        "n_models_scored": len(valid),
        "scoring_coverage": len(valid) / len(subset) if subset else 0.0,
        "n_cells": n_cells,
        "pooled_disjoint_log_loss": (
            sum(float(row["log_loss_sum"]) for row in valid) / n_cells
            if n_cells
            else None
        ),
        "pooled_disjoint_brier": (
            sum(float(row["brier_sum"]) for row in valid) / n_cells
            if n_cells
            else None
        ),
        "mean_model_disjoint_log_loss": mean,
        "se_model_disjoint_log_loss": se,
    }


def select_ridge_one_se(
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """Choose the strongest ridge within one SE of the inner-CV minimum."""

    rows = [dict(row) for row in evidence]
    eligible = [
        row
        for row in rows
        if float(row.get("scoring_coverage") or 0.0) == 1.0
        and row.get("mean_model_disjoint_log_loss") is not None
        and row.get("se_model_disjoint_log_loss") is not None
    ]
    if not eligible:
        raise NestedCVError("no ridge candidate has complete inner prediction evidence")
    best = min(
        eligible,
        key=lambda row: (
            float(row["mean_model_disjoint_log_loss"]),
            -float(row["ridge"]),
        ),
    )
    threshold = float(best["mean_model_disjoint_log_loss"]) + float(
        best["se_model_disjoint_log_loss"]
    )
    within = [
        row
        for row in eligible
        if float(row["mean_model_disjoint_log_loss"]) <= threshold + 1e-15
    ]
    selected = max(float(row["ridge"]) for row in within)
    for row in rows:
        row["minimum_loss_ridge"] = float(best["ridge"])
        row["one_se_threshold"] = threshold
        row["within_one_se"] = (
            row.get("mean_model_disjoint_log_loss") is not None
            and float(row["scoring_coverage"]) == 1.0
            and float(row["mean_model_disjoint_log_loss"]) <= threshold + 1e-15
        )
        row["selected"] = float(row["ridge"]) == selected
        row["selection_rule"] = (
            "lowest mean per-model disjoint log loss; strongest ridge within one SE"
        )
    return selected, rows


def evaluate_model_pair(
    *,
    model: str,
    row: pd.Series,
    full_bank: scat.FittedBank,
    administration_bank: scat.FittedBank,
    evaluation_indices: np.ndarray,
    scenario_records: Mapping[str, dict[str, Any]],
    quadrature: scat.Quadrature,
    candidate: Candidate,
    seed: int,
    top_n: int,
    maximum_scenarios: int,
    minimum_scored_criteria: int,
    mwle_ridge: float,
) -> dict[str, Any]:
    """Run CAT and paired random replay, then open the evaluation-pool cells."""

    common = dict(
        seed=seed,
        top_n=top_n,
        max_se=candidate.conditional_se_target,
        min_evals_per_skill=minimum_scored_criteria,
        min_scenarios=candidate.minimum_scenarios,
        max_scenarios=maximum_scenarios,
        selection=candidate.selector,
    )
    output: dict[str, Any] = {
        "model": str(model),
        "candidate_id": candidate.candidate_id,
        "minimum_scenarios": candidate.minimum_scenarios,
        "conditional_se_target": candidate.conditional_se_target,
        "selector": candidate.selector,
        "status": "ok",
        "error": "",
    }
    results: dict[str, dict[str, Any]] = {}
    for mode in ("cat", "baseline"):
        try:
            spec = scat.RunSpec(mode=mode, **common)
            result = scat.run_recorded_model(
                str(model),
                row,
                administration_bank,
                scenario_records,
                quadrature,
                spec,
                mwle_ridge=mwle_ridge,
            )
            if not set(result["scenario_order"]) <= set(administration_bank.scenario_ids):
                raise NestedCVError(
                    f"{model}/{candidate.candidate_id}/{mode}: evaluation scenario entered path"
                )
            results[mode] = result
            output[f"{mode}_replay_success"] = True
            output[f"{mode}_stop_reason"] = result["stop_reason"]
            output[f"{mode}_precision_reached"] = bool(result["precision_reached"])
            output[f"{mode}_scenarios_administered"] = int(
                result["scenarios_administered"]
            )
            output[f"{mode}_criteria_administered"] = int(
                result["criteria_administered"]
            )
            output[f"{mode}_mwle_converged"] = bool(result["mwle_converged"])
            output[f"{mode}_scenario_order"] = json.dumps(
                result["scenario_order"], ensure_ascii=False
            )
        except (scat.OfflineStudyError, ValueError, np.linalg.LinAlgError) as error:
            output["status"] = "replay_error"
            output["error"] = f"{type(error).__name__}: {error}"
            output[f"{mode}_replay_success"] = False
            output[f"{mode}_precision_reached"] = False
            output[f"{mode}_mwle_converged"] = False
            output[f"{mode}_scenarios_administered"] = None
            output[f"{mode}_criteria_administered"] = None
            output[f"{mode}_scenario_order"] = "[]"
            # A paired comparison is invalid if either arm failed.  Do not open
            # evaluation responses or run the other arm after a failure.
            break

    if len(results) != 2:
        for mode in ("cat", "baseline"):
            output.setdefault(f"{mode}_replay_success", False)
            output.setdefault(f"{mode}_precision_reached", False)
            output.setdefault(f"{mode}_mwle_converged", False)
            output.setdefault(f"{mode}_scenarios_administered", None)
            output.setdefault(f"{mode}_criteria_administered", None)
            output.setdefault(f"{mode}_scenario_order", "[]")
            for key, value in _empty_prediction_stats().items():
                output[f"{mode}_eval_{key}"] = value
        output["theta_reference"] = None
        output["theta_cat_mwle"] = None
        output["theta_baseline_mwle"] = None
        return output

    # The recovery reference is administration-pool EAP only.  Evaluation-pool
    # outcomes are reserved exclusively for the disjoint prediction metrics
    # below; they can never alter either path or the recovery target.
    responses = scat.responses_for_bank(row, full_bank)
    administration_scenarios = set(administration_bank.scenario_ids)
    administration_indices = np.asarray(
        [
            index
            for index, scenario in enumerate(full_bank.scenario_ids)
            if scenario in administration_scenarios
        ],
        dtype=int,
    )
    reference = scat.batch_eap(
        responses,
        full_bank.A,
        full_bank.b,
        quadrature,
        item_indices=administration_indices,
    )
    if full_bank.n_dims != 1:
        raise NestedCVError("Phase-3 remediation is locked to one latent dimension")
    output["theta_reference"] = float(reference.theta[0])
    output["theta_reference_scope"] = "all_observed_administration_pool_items"
    for mode in ("cat", "baseline"):
        result = results[mode]
        theta = np.asarray([result["theta_mwle"][full_bank.dims[0]]], dtype=float)
        output[f"theta_{mode}_mwle"] = (
            float(theta[0]) if result["mwle_converged"] else None
        )
        stats = (
            prediction_sufficient_statistics(row, full_bank, evaluation_indices, theta)
            if result["mwle_converged"]
            else _empty_prediction_stats()
        )
        for key, value in stats.items():
            output[f"{mode}_eval_{key}"] = value
    return output


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4 * total**2))
    return float((center - radius) / denominator)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = np.asarray([value for value in values if value is not None and math.isfinite(value)])
    return float(np.percentile(finite, percentile)) if finite.size else None


def _bootstrap_seed(master_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _bootstrap_recovery_lower(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> float | None:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    x = reference[valid]
    y = estimate[valid]
    if x.size < 3 or float(x.std()) == 0 or float(y.std()) == 0:
        return None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, x.size, size=x.size)
        xb = x[indices]
        yb = y[indices]
        if float(xb.std()) == 0 or float(yb.std()) == 0:
            continue
        values.append(float(np.corrcoef(xb, yb)[0, 1]))
    return _percentile(values, 2.5)


def _bootstrap_reduction_interval(
    cat_lengths: np.ndarray,
    random_lengths: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> tuple[float | None, float | None]:
    valid = np.isfinite(cat_lengths) & np.isfinite(random_lengths) & (random_lengths > 0)
    cat = cat_lengths[valid]
    baseline = random_lengths[valid]
    if cat.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.integers(0, cat.size, size=cat.size)
        denominator = float(baseline[sample].mean())
        values[index] = (
            1.0 - float(cat[sample].mean()) / denominator if denominator > 0 else np.nan
        )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def aggregate_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int,
    label: str,
) -> dict[str, Any]:
    """Aggregate replay sufficient statistics without reopening response cells."""

    total = len(rows)
    successful = [
        row
        for row in rows
        if row.get("cat_replay_success") and row.get("baseline_replay_success")
    ]
    cat_converged = [row for row in successful if row.get("cat_mwle_converged")]
    precision_count = sum(bool(row.get("cat_precision_reached")) for row in rows)
    cat_lengths = np.asarray(
        [float(row["cat_scenarios_administered"]) for row in successful], dtype=float
    )
    random_lengths = np.asarray(
        [float(row["baseline_scenarios_administered"]) for row in successful], dtype=float
    )
    reference = np.asarray(
        [
            float(row["theta_reference"])
            if row.get("theta_reference") is not None
            else np.nan
            for row in cat_converged
        ],
        dtype=float,
    )
    estimate = np.asarray(
        [
            float(row["theta_cat_mwle"])
            if row.get("theta_cat_mwle") is not None
            else np.nan
            for row in cat_converged
        ],
        dtype=float,
    )
    recovery = scenario_cv.recovery_stats(reference, estimate)
    recovery_lower = _bootstrap_recovery_lower(
        reference,
        estimate,
        seed=_bootstrap_seed(seed, label + ":recovery"),
        replicates=bootstrap_replicates,
    )

    n_cells = sum(int(row.get("cat_eval_n_cells") or 0) for row in cat_converged)
    log_loss_sum = sum(float(row.get("cat_eval_log_loss_sum") or 0.0) for row in cat_converged)
    brier_sum = sum(float(row.get("cat_eval_brier_sum") or 0.0) for row in cat_converged)
    correct = sum(int(row.get("cat_eval_correct_count") or 0) for row in cat_converged)
    pass_errors = [
        float(row["cat_eval_predicted_pass_rate"])
        - float(row["cat_eval_observed_pass_rate"])
        for row in cat_converged
        if row.get("cat_eval_predicted_pass_rate") is not None
        and row.get("cat_eval_observed_pass_rate") is not None
    ]
    reduction = (
        1.0 - float(cat_lengths.mean()) / float(random_lengths.mean())
        if cat_lengths.size and float(random_lengths.mean()) > 0
        else None
    )
    reduction_low, reduction_high = _bootstrap_reduction_interval(
        cat_lengths,
        random_lengths,
        seed=_bootstrap_seed(seed, label + ":paired-reduction"),
        replicates=bootstrap_replicates,
    )
    fit_keys = sorted({str(row.get("fit_cache_key")) for row in rows})
    return {
        "n_models": total,
        "n_inner_folds": len({row.get("inner_fold") for row in rows}),
        "replay_success_rate": len(successful) / total if total else None,
        "mwle_convergence_rate": len(cat_converged) / total if total else None,
        "nominal_precision_rate": precision_count / total if total else None,
        "nominal_precision_lower_95_ci": _wilson_lower(precision_count, total),
        "recovery_n": int(recovery["n"]),
        "recovery_correlation": recovery["r"],
        "recovery_correlation_lower_95_ci": recovery_lower,
        "recovery_slope": recovery["slope"],
        "recovery_mae": recovery["mae"],
        "recovery_bias": recovery["bias"],
        "disjoint_n_cells": n_cells,
        "disjoint_log_loss": log_loss_sum / n_cells if n_cells else None,
        "disjoint_brier": brier_sum / n_cells if n_cells else None,
        "disjoint_accuracy": correct / n_cells if n_cells else None,
        "disjoint_pass_rate_mae": (
            float(np.mean(np.abs(pass_errors))) if pass_errors else None
        ),
        "disjoint_pass_rate_bias": float(np.mean(pass_errors)) if pass_errors else None,
        "mean_scenario_count": float(cat_lengths.mean()) if cat_lengths.size else None,
        "p90_scenario_count": _percentile(cat_lengths.tolist(), 90),
        "mean_random_scenario_count": (
            float(random_lengths.mean()) if random_lengths.size else None
        ),
        "scenario_reduction_vs_random": reduction,
        "scenario_reduction_lower_95_ci": reduction_low,
        "scenario_reduction_upper_95_ci": reduction_high,
        "paired_ci_favors_cat": (
            reduction_low is not None and reduction_low > 0.0
        ),
        "fit_cache_keys": json.dumps(fit_keys, ensure_ascii=False),
    }


def apply_absolute_gates(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> tuple[bool, list[str], dict[str, bool]]:
    """Apply all Phase-3 gates; missing metrics fail closed and no fallback exists."""

    checks = {
        "replay_success_rate": (
            metrics.get("replay_success_rate"),
            lambda value: value >= float(gates["minimum_replay_success_rate"]),
        ),
        "mwle_convergence_rate": (
            metrics.get("mwle_convergence_rate"),
            lambda value: value >= float(gates["minimum_mwle_convergence_rate"]),
        ),
        "nominal_precision_rate": (
            metrics.get("nominal_precision_rate"),
            lambda value: value >= float(gates["minimum_nominal_precision_rate"]),
        ),
        "nominal_precision_lower_95_ci": (
            metrics.get("nominal_precision_lower_95_ci"),
            lambda value: value >= float(gates["minimum_nominal_precision_lower_95_ci"]),
        ),
        "recovery_correlation_lower_95_ci": (
            metrics.get("recovery_correlation_lower_95_ci"),
            lambda value: value
            >= float(gates["minimum_recovery_correlation_lower_95_ci"]),
        ),
        "recovery_slope": (
            metrics.get("recovery_slope"),
            lambda value: float(gates["minimum_recovery_slope"])
            <= value
            <= float(gates["maximum_recovery_slope"]),
        ),
        "disjoint_pass_rate_mae": (
            metrics.get("disjoint_pass_rate_mae"),
            lambda value: value <= float(gates["maximum_disjoint_pass_rate_mae"]),
        ),
        "absolute_disjoint_pass_rate_bias": (
            metrics.get("disjoint_pass_rate_bias"),
            lambda value: abs(value)
            <= float(gates["maximum_absolute_disjoint_pass_rate_bias"]),
        ),
        "scenario_reduction_vs_random": (
            metrics.get("scenario_reduction_vs_random"),
            lambda value: value >= float(gates["minimum_scenario_reduction_vs_random"]),
        ),
        "paired_ci_favors_cat": (
            metrics.get("paired_ci_favors_cat"),
            lambda value: bool(value)
            if gates.get("require_paired_ci_favors_cat")
            else True,
        ),
    }
    outcomes: dict[str, bool] = {}
    failures: list[str] = []
    for name, (value, predicate) in checks.items():
        passed = value is not None and bool(predicate(value))
        outcomes[name] = passed
        if not passed:
            failures.append(name)
    return not failures, failures, outcomes


def shortlist_candidates(
    aggregate_rows: Sequence[Mapping[str, Any]],
    *,
    selection_order: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply only the Phase-3 portion of the preregistered selection order.

    Absolute-gate passers are filtered lexicographically to the exact minimum
    p90 scenario count and then the exact minimum mean scenario count.  All
    configurations still tied at that point are returned in lexical ID order.
    Phase 3 intentionally has no total-SE input and does not prefer trace.
    """

    normalized_order = tuple(map(str, selection_order))
    if normalized_order != EXPECTED_SELECTION_ORDER:
        raise NestedCVError(
            "selection_gates.selection_order must remain exactly "
            f"{list(EXPECTED_SELECTION_ORDER)}, got {list(normalized_order)}"
        )
    eligible = [dict(row) for row in aggregate_rows if bool(row.get("all_gates_pass"))]
    eligible.sort(key=lambda row: str(row.get("candidate_id") or ""))
    candidate_ids = [str(row.get("candidate_id") or "") for row in eligible]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise NestedCVError("an absolute-gate passer lacks candidate_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise NestedCVError("absolute-gate passers contain duplicate candidate IDs")
    for row in eligible:
        for field in ("p90_scenario_count", "mean_scenario_count"):
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise NestedCVError(
                    f"eligible candidate {row['candidate_id']} lacks finite {field}"
                ) from error
            if not math.isfinite(value):
                raise NestedCVError(
                    f"eligible candidate {row['candidate_id']} lacks finite {field}"
                )
            row[field] = value

    if not eligible:
        return [], {
            "n_candidates_passing_all_absolute_gates": 0,
            "minimum_eligible_p90_scenario_count": None,
            "n_after_p90_length_filter": 0,
            "minimum_p90_tied_mean_scenario_count": None,
            "n_after_mean_length_filter": 0,
            "length_tie_policy": "exact_numeric_equality",
        }

    minimum_p90 = min(float(row["p90_scenario_count"]) for row in eligible)
    after_p90 = [
        row for row in eligible if float(row["p90_scenario_count"]) == minimum_p90
    ]
    minimum_mean = min(float(row["mean_scenario_count"]) for row in after_p90)
    finalists = [
        row for row in after_p90 if float(row["mean_scenario_count"]) == minimum_mean
    ]
    finalists.sort(key=lambda row: str(row["candidate_id"]))
    for row in finalists:
        row["phase3_length_filter_status"] = "retained_through_p90_and_mean"
        row["deferred_selection_fields"] = [
            "lowest_total_uncertainty",
            "prefer_trace",
        ]
    return finalists, {
        "n_candidates_passing_all_absolute_gates": len(eligible),
        "minimum_eligible_p90_scenario_count": minimum_p90,
        "n_after_p90_length_filter": len(after_p90),
        "minimum_p90_tied_mean_scenario_count": minimum_mean,
        "n_after_mean_length_filter": len(finalists),
        "length_tie_policy": "exact_numeric_equality",
    }


def aggregate_disjoint_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: int | str,
    candidate_id: str,
    mode: str,
) -> dict[str, Any]:
    prefix = "cat" if mode == "cat" else "baseline"
    eligible = [
        row
        for row in rows
        if row.get(f"{prefix}_replay_success") and row.get(f"{prefix}_mwle_converged")
    ]
    n_cells = sum(int(row.get(f"{prefix}_eval_n_cells") or 0) for row in eligible)
    pass_errors = [
        float(row[f"{prefix}_eval_predicted_pass_rate"])
        - float(row[f"{prefix}_eval_observed_pass_rate"])
        for row in eligible
        if row.get(f"{prefix}_eval_predicted_pass_rate") is not None
        and row.get(f"{prefix}_eval_observed_pass_rate") is not None
    ]
    return {
        "fold": fold,
        "candidate_id": candidate_id,
        "mode": mode,
        "n_models": len(eligible),
        "n_cells": n_cells,
        "log_loss": (
            sum(float(row.get(f"{prefix}_eval_log_loss_sum") or 0) for row in eligible)
            / n_cells
            if n_cells
            else None
        ),
        "brier": (
            sum(float(row.get(f"{prefix}_eval_brier_sum") or 0) for row in eligible)
            / n_cells
            if n_cells
            else None
        ),
        "accuracy": (
            sum(int(row.get(f"{prefix}_eval_correct_count") or 0) for row in eligible)
            / n_cells
            if n_cells
            else None
        ),
        "pass_rate_mae": float(np.mean(np.abs(pass_errors))) if pass_errors else None,
        "pass_rate_bias": float(np.mean(pass_errors)) if pass_errors else None,
    }


class NestedCVRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_path = args.config.resolve()
        self.split_path = args.split_manifest.resolve()
        self.config = _read_json(self.config_path)
        self.split = _read_json(self.split_path)
        if self.config.get("schema_version") != "infobench-calibration-remediation-v1":
            raise NestedCVError("unexpected remediation config schema_version")
        self.candidates = load_candidates(self.config)
        self.dense = resolve_dense_settings(args)
        self.debug_explicit_numerical_settings = not self.dense.native_lock_verified
        if (
            self.debug_explicit_numerical_settings
            and not args.plan_only
            and not args.debug_allow_explicit_numerical_settings
        ):
            raise NestedCVError(
                "a non-plan Phase-3 run requires the verified native numerical lock; "
                "--debug-allow-explicit-numerical-settings is non-headline only"
            )
        configured_ridges = [
            float(value)
            for value in (self.config.get("dense_grid") or {}).get(
                "ridge_candidates_after_grid_lock", []
            )
        ]
        if configured_ridges != [0.001, 0.01, 0.1]:
            raise NestedCVError(
                "remediation config must retain ridge candidates [0.001, 0.01, 0.1]"
            )
        self.debug_ridge_override = args.ridge is not None
        self.ridge_candidates = (
            [float(args.ridge)] if self.debug_ridge_override else configured_ridges
        )
        self._validate_config_links()

        baseline = self.config.get("baseline") or {}
        self.matrix_path = (
            args.matrix.resolve()
            if args.matrix is not None
            else _resolve_repo_path(str(baseline["response_matrix"])).resolve()
        )
        self.rubrics_path = (
            args.rubrics.resolve()
            if args.rubrics is not None
            else _resolve_repo_path(str(baseline["rubrics"])).resolve()
        )
        self.scenarios_path = (
            args.scenarios.resolve()
            if args.scenarios is not None
            else _resolve_repo_path(str(baseline["scenarios"])).resolve()
        )
        requested_output = args.out_dir
        if not requested_output.is_absolute():
            requested_output = Path.cwd() / requested_output
        self.output_dir_requested = requested_output.absolute()
        self.output_dir = self.output_dir_requested.resolve()
        self.input_hashes = self._validate_inputs()
        self.matrix = cm.load_matrix_strict(self.matrix_path)
        self.scenario_records = scat.load_scenario_records(self.scenarios_path)
        self.split_audit = audit_split_manifest(
            self.split,
            expected_models=self.matrix.index,
            expected_scenarios=self.scenario_records,
        )
        self.source_skills = cm.configure_skills(args.skills)
        self.structure = cell_cv.build_structure(
            tuple(self.source_skills), args.dimensions, args.structure_name
        )
        expected_dims = tuple(
            self.config.get("immutable_decisions", {}).get("latent_dimensions") or []
        )
        if tuple(self.structure.labels) != expected_dims or self.structure.n_dims != 1:
            raise NestedCVError(
                "modeled structure must equal the frozen one-dimensional latent decision: "
                f"expected {expected_dims}, got {self.structure.labels}"
            )
        self.q_by = cm.load_q_matrix(self.rubrics_path)
        cm.validate_matrix_bank_alignment(
            self.matrix, self.q_by, self.args.require_complete_bank
        )
        self.source_records = scenario_cv.source_records_by_id(self.rubrics_path)
        self.code_hashes = _code_dependency_hashes()
        self.dependency_versions = _dependency_versions()
        self.study_signature = self._study_signature()

    def _validate_config_links(self) -> None:
        cv = self.config.get("cross_validation") or {}
        if int(cv.get("outer_folds", 0)) != 5 or int(cv.get("inner_folds", 0)) != 4:
            raise NestedCVError("remediation config must lock five outer/four inner folds")
        if cv.get("group_related_model_families") is not True:
            raise NestedCVError("grouped model-family folds must remain enabled")
        expected_hash = str(cv.get("fold_manifest_sha256") or "")
        actual_hash = _sha256(self.split_path)
        if actual_hash != expected_hash:
            raise NestedCVError(
                f"frozen split hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        gates = self.config.get("selection_gates") or {}
        if gates.get("allow_fallback_if_none_pass") is not False:
            raise NestedCVError("selection fallback must remain disabled")
        if (self.config.get("cat_candidates") or {}).get("estimator") != "mwle":
            raise NestedCVError("this Phase-3 driver is preregistered for MWLE")
        selection_order = tuple(
            map(str, (self.config.get("selection_gates") or {}).get("selection_order") or [])
        )
        if selection_order != EXPECTED_SELECTION_ORDER:
            raise NestedCVError(
                "selection_gates.selection_order must remain exactly "
                f"{list(EXPECTED_SELECTION_ORDER)}, got {list(selection_order)}"
            )
        self.selection_order = selection_order

    def _validate_inputs(self) -> dict[str, str]:
        paths = {
            "response_matrix": self.matrix_path,
            "rubrics": self.rubrics_path,
            "scenarios": self.scenarios_path,
        }
        for _name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
        hashes = {name: _sha256(path) for name, path in paths.items()}
        split_inputs = self.split.get("inputs") or {}
        for name, digest in hashes.items():
            expected = str((split_inputs.get(name) or {}).get("sha256") or "")
            if digest != expected:
                raise NestedCVError(
                    f"{name} hash does not match frozen split manifest: "
                    f"expected {expected}, got {digest}"
                )
        baseline_expected = str(
            (self.config.get("baseline") or {}).get("response_matrix_sha256") or ""
        )
        if hashes["response_matrix"] != baseline_expected:
            raise NestedCVError("response matrix does not match the immutable calibration boundary")
        return hashes

    def _study_signature(self) -> str:
        return _canonical_hash(
            {
                "schema": SCRIPT_SCHEMA,
                "config_sha256": _sha256(self.config_path),
                "split_sha256": _sha256(self.split_path),
                "inputs": self.input_hashes,
                "code_dependency_hashes": self.code_hashes,
                "dependency_versions": self.dependency_versions,
                "dense": asdict(self.dense),
                "debug_explicit_numerical_settings": self.debug_explicit_numerical_settings,
                "ridge_candidates": self.ridge_candidates,
                "debug_ridge_override": self.debug_ridge_override,
                "structure": self.structure.as_dict(),
                "candidates": [asdict(candidate) for candidate in self.candidates],
                "runtime": {
                    "seed": self.args.seed,
                    "top_n": self.args.top_n,
                    "max_scenarios": self.args.max_scenarios,
                    "minimum_scored_criteria": self.args.minimum_scored_criteria,
                    "max_iter": self.args.max_iter,
                    "tol": self.args.tol,
                    "mwle_ridge": self.args.mwle_ridge,
                    "negative_policy": self.args.negative_policy,
                    "metric_bootstrap_replicates": self.args.metric_bootstrap_replicates,
                    "estimate_latent_corr": self.args.estimate_latent_corr,
                    "max_grid_nodes": self.args.max_grid_nodes,
                    "allow_unconverged_fit": self.args.allow_unconverged_fit,
                    "require_complete_bank": self.args.require_complete_bank,
                    "debug_allow_explicit_numerical_settings": (
                        self.args.debug_allow_explicit_numerical_settings
                    ),
                },
            }
        )

    def _prepare_output(self) -> None:
        if self.args.fresh and self.output_dir.exists():
            requested = self.output_dir_requested
            dense_manifest = getattr(self.args, "dense_grid_manifest", None)
            expected = (
                dense_manifest.resolve().parent.parent / "nested_cat_cv"
                if dense_manifest is not None
                else DEFAULT_OUTPUT.absolute()
            )
            expected_parent = expected.parent
            relevant_paths = [requested, *requested.parents]
            if ROOT.parent in relevant_paths:
                relevant_paths = relevant_paths[: relevant_paths.index(ROOT.parent)]
            if (
                requested != expected
                or requested.name != "nested_cat_cv"
                or requested.parent != expected_parent
                or self.output_dir.parent != expected_parent.resolve()
                or self.output_dir.name != "nested_cat_cv"
                or any(path.is_symlink() for path in relevant_paths)
            ):
                raise NestedCVError(f"refusing unsafe --fresh target: {requested}")
            shutil.rmtree(requested)
        manifest_path = self.output_dir / "manifest.json"
        if self.args.resume:
            if not manifest_path.is_file():
                raise NestedCVError("--resume requires an existing manifest.json")
            existing = _read_json(manifest_path)
            if existing.get("study_signature") != self.study_signature:
                raise NestedCVError("resume manifest does not match this study signature")
        elif self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise NestedCVError(
                f"output directory is not empty: {self.output_dir}; use --resume or --fresh"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fold_assignment_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCRIPT_SCHEMA,
            "source_split_manifest": _display_path(self.split_path),
            "source_split_sha256": _sha256(self.split_path),
            "leakage_audit": "passed",
            "outer_folds": self.split_audit["outer_folds"],
            "model_to_family": self.split_audit["model_to_family"],
            "model_to_outer_fold": self.split_audit["model_to_outer_fold"],
            "scenario_split": {
                "administration_scenario_ids": self.split_audit[
                    "administration_scenario_ids"
                ],
                "evaluation_scenario_ids": self.split_audit[
                    "evaluation_scenario_ids"
                ],
            },
        }

    def _base_manifest(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": SCRIPT_SCHEMA,
            "status": status,
            "study_signature": self.study_signature,
            "created_at": _utcnow(),
            "script": "scripts/nested_scenario_cat_cv.py",
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
                    for name, path in (
                        ("response_matrix", self.matrix_path),
                        ("rubrics", self.rubrics_path),
                        ("scenarios", self.scenarios_path),
                    )
                },
            },
            "dense_grid_lock": asdict(self.dense),
            "dense_settings_selection_policy": (
                "fit/EAP grids are pre-nested explicit/locked inputs; never inferred "
                "from outer results"
            ),
            "phase3_finalist_handoff": {
                "artifact": "outer_fold_finalists.json",
                "artifact_schema_version": FINALIST_SCHEMA,
                "configured_selection_order": list(self.selection_order),
                "applied_in_phase3": [
                    "absolute_gates",
                    "lowest_p90_scenario_count",
                    "lowest_mean_scenario_count",
                ],
                "deferred_to_phase4": [
                    "lowest_total_uncertainty",
                    "prefer_trace",
                ],
                "length_tie_policy": "exact_numeric_equality",
                "outer_evaluation_panel": (
                    "each fold-local exact-tie finalist set, evaluated only on that "
                    "fold's outer-test models"
                ),
                "cross_fold_candidate_id_comparison_authorized": False,
                "phase3_declares_winner": False,
                "final_policy_freeze_authorized": False,
            },
            "ridge_selection": {
                "candidates": self.ridge_candidates,
                "scope": "four inner folds inside each outer-training set",
                "metric": "per-model disjoint-scenario EAP prediction log loss",
                "rule": "strongest regularization within one SE of minimum mean loss",
                "outer_outcomes_used": False,
                "debug_override": self.debug_ridge_override,
                "headline_eligible": not (
                    self.debug_ridge_override
                    or self.debug_explicit_numerical_settings
                ),
                "warning": (
                    "--ridge bypasses nested ridge selection and is debug-only"
                    if self.debug_ridge_override
                    else None
                ),
            },
            "structure": self.structure.as_dict(),
            "candidate_count": len(self.candidates),
            "candidate_ids": [candidate.candidate_id for candidate in self.candidates],
            "cross_validation": {
                "outer_folds": 5,
                "inner_folds": 4,
                "grouped_model_families": True,
                "administration_scenarios": 400,
                "evaluation_scenarios": 100,
                "evaluation_outcome_usage": "disjoint prediction metrics only",
                "recovery_reference": "administration-pool full EAP",
                "outer_test_used_for_selection": False,
            },
            "runtime": {
                "seed": self.args.seed,
                "top_n": self.args.top_n,
                "maximum_adaptive_scenarios": self.args.max_scenarios,
                "minimum_scored_criteria": self.args.minimum_scored_criteria,
                "max_iter": self.args.max_iter,
                "tol": self.args.tol,
                "mwle_ridge": self.args.mwle_ridge,
                "negative_policy": self.args.negative_policy,
                "metric_bootstrap_replicates": self.args.metric_bootstrap_replicates,
                "estimate_latent_corr": self.args.estimate_latent_corr,
                "max_grid_nodes": self.args.max_grid_nodes,
                "allow_unconverged_fit": self.args.allow_unconverged_fit,
                "require_complete_bank": self.args.require_complete_bank,
                "debug_allow_explicit_numerical_settings": (
                    self.args.debug_allow_explicit_numerical_settings
                ),
            },
            "code_provenance": {
                "files": self.code_hashes,
                "canonical_sha256": _canonical_hash(self.code_hashes),
            },
            "environment": {
                **self.dependency_versions,
                "platform": platform.platform(),
                "canonical_dependency_versions_sha256": _canonical_hash(
                    self.dependency_versions
                ),
            },
            "deferred_outputs": {
                name: "not produced in Phase 3; requires Phase-4 parameter/order study"
                for name in DEFERRED_PHASE4_OUTPUTS
            },
        }

    def _fit_cache_key(
        self, training_model_ids: Sequence[str], *, ridge: float
    ) -> str:
        return _canonical_hash(
            {
                "response_matrix_sha256": self.input_hashes["response_matrix"],
                "rubrics_sha256": self.input_hashes["rubrics"],
                "training_model_ids": sorted(map(str, training_model_ids)),
                "structure": self.structure.as_dict(),
                "fit_grid": self.dense.fit_grid,
                "ridge": float(ridge),
                "estimate_latent_corr": self.args.estimate_latent_corr,
                "max_iter": self.args.max_iter,
                "tol": self.args.tol,
                "negative_policy": self.args.negative_policy,
            }
        )

    def _write_fit_cache(
        self,
        cache_dir: Path,
        fit: Mapping[str, Any],
        cache_key: str,
        training_model_ids: Sequence[str],
        policy: Mapping[str, Any],
        ridge: float,
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
            "schema_version": SCRIPT_SCHEMA,
            "study_signature": self.study_signature,
            "code_provenance_sha256": _canonical_hash(self.code_hashes),
            "dependency_versions_sha256": _canonical_hash(self.dependency_versions),
            "cache_key": cache_key,
            "training_model_ids": sorted(map(str, training_model_ids)),
            "fit_grid": self.dense.fit_grid,
            "ridge": float(ridge),
            "items": list(fit["items"]),
            "dim_labels": list(fit["dim_labels"]),
            "loglik": fit["loglik"],
            "n_params": fit["n_params"],
            "n_iter": fit["n_iter"],
            "converged": fit["converged"],
            "diagnostics": fit.get("diag") or {},
            "bank_policy": policy,
            "arrays_sha256": _sha256(array_path),
        }
        manifest["manifest_content_sha256"] = _canonical_hash(manifest)
        _atomic_json(cache_dir / "fit_manifest.json", manifest)

    def _load_fit_cache(
        self,
        cache_dir: Path,
        expected_key: str,
        training_model_ids: Sequence[str],
        *,
        ridge: float,
    ) -> dict[str, Any] | None:
        manifest_path = cache_dir / "fit_manifest.json"
        array_path = cache_dir / "fit_arrays.npz"
        if not (manifest_path.is_file() and array_path.is_file()):
            return None
        manifest = _read_json(manifest_path)
        content_hash = str(manifest.pop("manifest_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(manifest):
            raise NestedCVError(f"fit-cache manifest content hash mismatch in {cache_dir}")
        if manifest.get("schema_version") != SCRIPT_SCHEMA:
            raise NestedCVError(f"fit-cache schema mismatch in {cache_dir}")
        if manifest.get("study_signature") != self.study_signature:
            raise NestedCVError(f"fit-cache study signature mismatch in {cache_dir}")
        if manifest.get("code_provenance_sha256") != _canonical_hash(self.code_hashes):
            raise NestedCVError(f"fit-cache code provenance mismatch in {cache_dir}")
        if manifest.get("dependency_versions_sha256") != _canonical_hash(
            self.dependency_versions
        ):
            raise NestedCVError(f"fit-cache dependency provenance mismatch in {cache_dir}")
        if manifest.get("cache_key") != expected_key:
            raise NestedCVError(f"fit-cache key mismatch in {cache_dir}")
        if manifest.get("training_model_ids") != sorted(map(str, training_model_ids)):
            raise NestedCVError(f"fit-cache training IDs mismatch in {cache_dir}")
        if manifest.get("arrays_sha256") != _sha256(array_path):
            raise NestedCVError(f"fit-cache array hash mismatch in {cache_dir}")
        if int(manifest.get("fit_grid", -1)) != self.dense.fit_grid or not math.isclose(
            float(manifest.get("ridge", math.nan)), float(ridge), rel_tol=0.0, abs_tol=0.0
        ):
            raise NestedCVError(f"fit-cache numerical settings mismatch in {cache_dir}")
        with np.load(array_path, allow_pickle=False) as arrays:
            fit = {
                "items": list(map(str, manifest["items"])),
                "A": np.asarray(arrays["A"], dtype=float),
                "b": np.asarray(arrays["b"], dtype=float),
                "R": np.asarray(arrays["R"], dtype=float),
                "dim_labels": list(map(str, manifest["dim_labels"])),
                "loglik": float(manifest["loglik"]),
                "n_params": int(manifest["n_params"]),
                "n_iter": int(manifest["n_iter"]),
                "converged": bool(manifest["converged"]),
                "diag": manifest.get("diagnostics") or {},
            }
        return fit

    def fit_or_load(
        self,
        *,
        role: str,
        training_model_ids: Sequence[str],
        forbidden_model_ids: Iterable[str],
        ridge: float,
    ) -> FitBundle:
        train_ids = tuple(sorted(map(str, training_model_ids)))
        forbidden = set(map(str, forbidden_model_ids))
        if set(train_ids) & forbidden:
            raise NestedCVError(f"{role}: forbidden held-out model entered item fitting")
        if not set(train_ids) <= set(self.matrix.index):
            raise NestedCVError(f"{role}: item-fit IDs are absent from the response matrix")
        cache_key = self._fit_cache_key(train_ids, ridge=ridge)
        cache_dir = self.output_dir / "fit_cache" / role
        fit = self._load_fit_cache(cache_dir, cache_key, train_ids, ridge=float(ridge))
        fit_was_cached = fit is not None
        if not fit_was_cached:
            fit_args = argparse.Namespace(
                grid=self.dense.fit_grid,
                estimate_latent_corr=self.args.estimate_latent_corr,
                ridge=float(ridge),
                max_iter=self.args.max_iter,
                tol=self.args.tol,
            )
            fit = cell_cv.fit_structure(
                self.matrix.loc[list(train_ids)], self.q_by, fit_args, self.structure
            )
        if not bool(fit["converged"]) and not self.args.allow_unconverged_fit:
            source = "cached" if fit_was_cached else "fresh"
            raise NestedCVError(f"{role}: {source} item fit did not converge")
        temporary_bank, policy = scenario_cv.build_fold_bank(
            fit,
            self.structure,
            self.source_records,
            negative_policy=self.args.negative_policy,
        )
        if not fit_was_cached:
            self._write_fit_cache(
                cache_dir, fit, cache_key, train_ids, policy, ridge=float(ridge)
            )
        quadrature = scat.build_quadrature(
            self.structure.n_dims,
            self.dense.eap_grid,
            temporary_bank.latent_correlation,
            max_nodes=max(self.args.max_grid_nodes, self.dense.eap_grid),
            method=self.dense.quadrature_method,
            linear_bound=self.dense.linear_bound,
        )
        return FitBundle(
            fit=fit,
            bank=temporary_bank,
            quadrature=quadrature,
            cache_key=cache_key,
            training_model_ids=train_ids,
            cache_dir=cache_dir,
            policy_diagnostics=policy,
        )

    def _checkpoint_path(self, outer_fold: int, inner_fold: int | None) -> Path:
        name = (
            f"outer_{outer_fold}_inner_{inner_fold}.json"
            if inner_fold is not None
            else f"outer_{outer_fold}_test.json"
        )
        return self.output_dir / "checkpoints" / name

    def _ridge_checkpoint_path(
        self, outer_fold: int, inner_fold: int, ridge: float
    ) -> Path:
        return (
            self.output_dir
            / "checkpoints"
            / f"outer_{outer_fold}_inner_{inner_fold}_ridge_{_ridge_slug(ridge)}.json"
        )

    def _validate_checkpoint_rows(
        self,
        *,
        stage: str,
        outer_fold: int,
        inner_fold: int | None,
        ridge: float | None,
        fit_cache_key: str,
        candidate_ids: Sequence[str],
        model_ids: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        source: Path,
    ) -> None:
        candidates = list(map(str, candidate_ids))
        models = sorted(map(str, model_ids))
        if not candidates or len(candidates) != len(set(candidates)):
            raise NestedCVError(f"checkpoint candidate IDs are empty/duplicated: {source}")
        if not models or len(models) != len(set(models)):
            raise NestedCVError(f"checkpoint model IDs are empty/duplicated: {source}")
        normalized_rows = [dict(row) for row in rows]
        for row in normalized_rows:
            try:
                row_outer = int(row["outer_fold"])
            except (KeyError, TypeError, ValueError) as error:
                raise NestedCVError(
                    f"checkpoint row lacks outer-fold provenance: {source}"
                ) from error
            row_inner_raw = row.get("inner_fold")
            row_inner = None if row_inner_raw is None else int(row_inner_raw)
            if row_outer != int(outer_fold) or row_inner != inner_fold:
                raise NestedCVError(f"checkpoint row fold provenance mismatch: {source}")
            if str(row.get("fit_cache_key") or "") != fit_cache_key:
                raise NestedCVError(f"checkpoint row fit provenance mismatch: {source}")

        if stage == "ridge_validation":
            if inner_fold is None or ridge is None or len(candidates) != 1:
                raise NestedCVError(f"invalid ridge checkpoint provenance: {source}")
            actual_models = [str(row.get("model") or "") for row in normalized_rows]
            if actual_models != models:
                raise NestedCVError(
                    f"ridge checkpoint rows are not exactly one sorted row per model: {source}"
                )
            for row in normalized_rows:
                try:
                    row_ridge = float(row["ridge"])
                except (KeyError, TypeError, ValueError) as error:
                    raise NestedCVError(
                        f"ridge checkpoint row lacks ridge provenance: {source}"
                    ) from error
                if not math.isclose(row_ridge, ridge, rel_tol=0.0, abs_tol=0.0):
                    raise NestedCVError(f"ridge checkpoint row uses wrong ridge: {source}")
            return

        if stage not in {"inner_cat", "outer_cat"}:
            raise NestedCVError(f"unknown checkpoint stage {stage!r}: {source}")
        expected_inner = stage == "inner_cat"
        if (inner_fold is not None) != expected_inner or ridge is not None:
            raise NestedCVError(f"CAT checkpoint stage/fold provenance mismatch: {source}")
        expected_pairs = [
            (candidate_id, model) for candidate_id in candidates for model in models
        ]
        actual_pairs = [
            (str(row.get("candidate_id") or ""), str(row.get("model") or ""))
            for row in normalized_rows
        ]
        if actual_pairs != expected_pairs:
            raise NestedCVError(
                f"{stage} checkpoint rows do not exactly match ordered candidate×model support: "
                f"{source}"
            )

    def _load_checkpoint(
        self,
        path: Path,
        *,
        stage: str,
        outer_fold: int,
        inner_fold: int | None,
        ridge: float | None,
        fit_cache_key: str,
        candidate_ids: Sequence[str],
        model_ids: Sequence[str],
    ) -> list[dict[str, Any]] | None:
        if not (self.args.resume and path.is_file()):
            return None
        value = _read_json(path)
        content_hash = str(value.pop("checkpoint_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(value):
            raise NestedCVError(f"checkpoint content hash mismatch: {path}")
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "study_signature": self.study_signature,
            "code_provenance_sha256": _canonical_hash(self.code_hashes),
            "dependency_versions_sha256": _canonical_hash(self.dependency_versions),
            "checkpoint_stage": stage,
            "outer_fold": int(outer_fold),
            "inner_fold": inner_fold,
            "ridge": ridge,
            "fit_cache_key": fit_cache_key,
            "candidate_ids": list(map(str, candidate_ids)),
            "model_ids": sorted(map(str, model_ids)),
        }
        for key, item in expected.items():
            if value.get(key) != item:
                raise NestedCVError(f"checkpoint mismatch for {key}: {path}")
        rows = value.get("rows")
        if not isinstance(rows, list):
            raise NestedCVError(f"checkpoint has no row list: {path}")
        if int(value.get("row_count", -1)) != len(rows):
            raise NestedCVError(f"checkpoint row count mismatch: {path}")
        if value.get("rows_sha256") != _canonical_hash(rows):
            raise NestedCVError(f"checkpoint row hash mismatch: {path}")
        self._validate_checkpoint_rows(
            stage=stage,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            ridge=ridge,
            fit_cache_key=fit_cache_key,
            candidate_ids=candidate_ids,
            model_ids=model_ids,
            rows=rows,
            source=path,
        )
        return [dict(row) for row in rows]

    def _write_checkpoint(
        self,
        path: Path,
        *,
        stage: str,
        outer_fold: int,
        inner_fold: int | None,
        ridge: float | None,
        fit_cache_key: str,
        candidate_ids: Sequence[str],
        model_ids: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self._validate_checkpoint_rows(
            stage=stage,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            ridge=ridge,
            fit_cache_key=fit_cache_key,
            candidate_ids=candidate_ids,
            model_ids=model_ids,
            rows=rows,
            source=path,
        )
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "study_signature": self.study_signature,
            "code_provenance_sha256": _canonical_hash(self.code_hashes),
            "dependency_versions_sha256": _canonical_hash(self.dependency_versions),
            "checkpoint_stage": stage,
            "outer_fold": int(outer_fold),
            "inner_fold": inner_fold,
            "ridge": ridge,
            "fit_cache_key": fit_cache_key,
            "candidate_ids": list(map(str, candidate_ids)),
            "model_ids": sorted(map(str, model_ids)),
            "row_count": len(rows),
            "rows_sha256": _canonical_hash(rows),
            "rows": list(rows),
        }
        payload["checkpoint_content_sha256"] = _canonical_hash(payload)
        _atomic_json(path, payload)

    def _evaluate_panel(
        self,
        *,
        bundle: FitBundle,
        model_ids: Sequence[str],
        candidates: Sequence[Candidate],
        outer_fold: int,
        inner_fold: int | None,
    ) -> list[dict[str, Any]]:
        administration_bank = subset_fitted_bank(
            bundle.bank, self.split_audit["administration_scenario_ids"]
        )
        evaluation_indices = evaluation_item_indices(
            bundle.bank, self.split_audit["evaluation_scenario_ids"]
        )
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            for model in sorted(map(str, model_ids)):
                row = evaluate_model_pair(
                    model=model,
                    row=self.matrix.loc[model],
                    full_bank=bundle.bank,
                    administration_bank=administration_bank,
                    evaluation_indices=evaluation_indices,
                    scenario_records=self.scenario_records,
                    quadrature=bundle.quadrature,
                    candidate=candidate,
                    seed=self.args.seed,
                    top_n=self.args.top_n,
                    maximum_scenarios=self.args.max_scenarios,
                    minimum_scored_criteria=self.args.minimum_scored_criteria,
                    mwle_ridge=self.args.mwle_ridge,
                )
                row.update(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "fit_cache_key": bundle.cache_key,
                    }
                )
                rows.append(row)
        return rows

    def _evaluate_ridge_panel(
        self,
        *,
        bundle: FitBundle,
        model_ids: Sequence[str],
        ridge: float,
        outer_fold: int,
        inner_fold: int,
    ) -> list[dict[str, Any]]:
        administration_ids = set(self.split_audit["administration_scenario_ids"])
        administration_indices = np.asarray(
            [
                index
                for index, scenario in enumerate(bundle.bank.scenario_ids)
                if scenario in administration_ids
            ],
            dtype=int,
        )
        if administration_indices.size == 0:
            raise NestedCVError("ridge validation has no fitted administration items")
        evaluation_indices = evaluation_item_indices(
            bundle.bank, self.split_audit["evaluation_scenario_ids"]
        )
        rows: list[dict[str, Any]] = []
        for model in sorted(map(str, model_ids)):
            row = evaluate_ridge_model(
                model=model,
                row=self.matrix.loc[model],
                bank=bundle.bank,
                administration_indices=administration_indices,
                evaluation_indices=evaluation_indices,
                quadrature=bundle.quadrature,
                ridge=ridge,
            )
            row.update(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "fit_cache_key": bundle.cache_key,
                }
            )
            rows.append(row)
        return rows

    def _select_ridge_for_outer(
        self, outer: Mapping[str, Any]
    ) -> tuple[float, list[dict[str, Any]], dict[tuple[int, float], FitBundle]]:
        """Select ridge from inner-only disjoint predictions and retain fit cache."""

        outer_fold = int(outer["outer_fold"])
        outer_test = set(map(str, outer["test_model_ids"]))
        raw_rows: list[dict[str, Any]] = []
        bundles: dict[tuple[int, float], FitBundle] = {}
        for inner in outer["inner_folds"]:
            inner_fold = int(inner["inner_fold"])
            fit_ids = list(map(str, inner["fit_model_ids"]))
            validation_ids = list(map(str, inner["validation_model_ids"]))
            if outer_test & (set(fit_ids) | set(validation_ids)):
                raise NestedCVError("outer-test IDs entered an inner stage")
            for ridge in self.ridge_candidates:
                bundle = self.fit_or_load(
                    role=(
                        f"outer_{outer_fold}/inner_{inner_fold}/"
                        f"ridge_{_ridge_slug(ridge)}"
                    ),
                    training_model_ids=fit_ids,
                    forbidden_model_ids=outer_test | set(validation_ids),
                    ridge=ridge,
                )
                bundles[(inner_fold, ridge)] = bundle
                checkpoint = self._ridge_checkpoint_path(
                    outer_fold, inner_fold, ridge
                )
                rows = self._load_checkpoint(
                    checkpoint,
                    stage="ridge_validation",
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    ridge=float(ridge),
                    fit_cache_key=bundle.cache_key,
                    candidate_ids=[f"ridge_{_ridge_slug(ridge)}_eap_disjoint"],
                    model_ids=validation_ids,
                )
                if rows is None:
                    rows = self._evaluate_ridge_panel(
                        bundle=bundle,
                        model_ids=validation_ids,
                        ridge=ridge,
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                    )
                    self._write_checkpoint(
                        checkpoint,
                        stage="ridge_validation",
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        ridge=float(ridge),
                        fit_cache_key=bundle.cache_key,
                        candidate_ids=[f"ridge_{_ridge_slug(ridge)}_eap_disjoint"],
                        model_ids=validation_ids,
                        rows=rows,
                    )
                raw_rows.extend(rows)

        evidence = [
            {"outer_fold": outer_fold, **aggregate_ridge_evidence(raw_rows, ridge)}
            for ridge in self.ridge_candidates
        ]
        selected, annotated = select_ridge_one_se(evidence)
        if self.debug_ridge_override:
            for row in annotated:
                row["selection_rule"] = "debug --ridge override; non-headline"
                row["debug_override"] = True
        else:
            for row in annotated:
                row["debug_override"] = False
        return selected, annotated, bundles

    def _inner_rows_for_outer(
        self,
        outer: Mapping[str, Any],
        *,
        selected_ridge: float,
        bundles: Mapping[tuple[int, float], FitBundle],
    ) -> list[dict[str, Any]]:
        outer_fold = int(outer["outer_fold"])
        outer_test = set(map(str, outer["test_model_ids"]))
        all_rows: list[dict[str, Any]] = []
        for inner in outer["inner_folds"]:
            inner_fold = int(inner["inner_fold"])
            validation_ids = list(map(str, inner["validation_model_ids"]))
            if outer_test & set(validation_ids):
                raise NestedCVError("outer-test IDs entered inner CAT validation")
            bundle = bundles[(inner_fold, selected_ridge)]
            checkpoint = self._checkpoint_path(outer_fold, inner_fold)
            rows = self._load_checkpoint(
                checkpoint,
                stage="inner_cat",
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                ridge=None,
                fit_cache_key=bundle.cache_key,
                candidate_ids=[candidate.candidate_id for candidate in self.candidates],
                model_ids=validation_ids,
            )
            if rows is None:
                rows = self._evaluate_panel(
                    bundle=bundle,
                    model_ids=validation_ids,
                    candidates=self.candidates,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                )
                for row in rows:
                    row["selected_ridge"] = selected_ridge
                self._write_checkpoint(
                    checkpoint,
                    stage="inner_cat",
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    ridge=None,
                    fit_cache_key=bundle.cache_key,
                    candidate_ids=[candidate.candidate_id for candidate in self.candidates],
                    model_ids=validation_ids,
                    rows=rows,
                )
            elif any(float(row.get("selected_ridge", math.nan)) != selected_ridge for row in rows):
                raise NestedCVError(
                    f"inner CAT checkpoint has wrong selected ridge: {checkpoint}"
                )
            all_rows.extend(rows)
        return all_rows

    def _aggregate_inner(
        self,
        outer_fold: int,
        rows: Sequence[Mapping[str, Any]],
        *,
        selected_ridge: float,
    ) -> list[dict[str, Any]]:
        gates = self.config["selection_gates"]
        aggregate: list[dict[str, Any]] = []
        for candidate in self.candidates:
            subset = [row for row in rows if row["candidate_id"] == candidate.candidate_id]
            metrics = aggregate_candidate_rows(
                subset,
                seed=self.args.seed,
                bootstrap_replicates=self.args.metric_bootstrap_replicates,
                label=f"outer-{outer_fold}:{candidate.candidate_id}",
            )
            passed, failures, outcomes = apply_absolute_gates(metrics, gates)
            aggregate.append(
                {
                    "outer_fold": outer_fold,
                    "selected_ridge": selected_ridge,
                    "candidate_id": candidate.candidate_id,
                    **asdict(candidate),
                    **metrics,
                    "all_gates_pass": passed,
                    "failed_gates": "|".join(failures),
                    "gate_results_json": json.dumps(outcomes, sort_keys=True),
                    "parameter_bootstrap_gate": "deferred_to_phase4_not_applied",
                    "total_uncertainty_tiebreak": "deferred_to_phase4_not_available",
                }
            )
        if len(aggregate) != 48:
            raise NestedCVError("inner aggregation did not preserve the 48-way factorial")
        return aggregate

    def _outer_rows(
        self,
        outer: Mapping[str, Any],
        finalists: Sequence[Mapping[str, Any]],
        *,
        selected_ridge: float,
    ) -> list[dict[str, Any]]:
        outer_fold = int(outer["outer_fold"])
        test_ids = list(map(str, outer["test_model_ids"]))
        if not finalists:
            return [
                {
                    "outer_fold": outer_fold,
                    "model": model,
                    "candidate_id": None,
                    "selected_ridge": selected_ridge,
                    "status": "not_run_no_phase3_finalist",
                    "cat_replay_success": False,
                    "baseline_replay_success": False,
                    "cat_mwle_converged": False,
                    "baseline_mwle_converged": False,
                }
                for model in sorted(test_ids)
            ]
        finalist_ids = [str(row["candidate_id"]) for row in finalists]
        if finalist_ids != sorted(finalist_ids) or len(finalist_ids) != len(set(finalist_ids)):
            raise NestedCVError("Phase-3 finalists must have unique lexical candidate IDs")
        candidates_by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        try:
            candidates = [candidates_by_id[candidate_id] for candidate_id in finalist_ids]
        except KeyError as error:
            raise NestedCVError(f"unknown Phase-3 finalist candidate: {error}") from error
        train_ids = list(map(str, outer["train_model_ids"]))
        bundle = self.fit_or_load(
            role=(
                f"outer_{outer_fold}/outer_train/"
                f"ridge_{_ridge_slug(selected_ridge)}"
            ),
            training_model_ids=train_ids,
            forbidden_model_ids=test_ids,
            ridge=selected_ridge,
        )
        checkpoint = self._checkpoint_path(outer_fold, None)
        rows = self._load_checkpoint(
            checkpoint,
            stage="outer_cat",
            outer_fold=outer_fold,
            inner_fold=None,
            ridge=None,
            fit_cache_key=bundle.cache_key,
            candidate_ids=finalist_ids,
            model_ids=test_ids,
        )
        if rows is None:
            rows = self._evaluate_panel(
                bundle=bundle,
                model_ids=test_ids,
                candidates=candidates,
                outer_fold=outer_fold,
                inner_fold=None,
            )
            for row in rows:
                row["selected_ridge"] = selected_ridge
            self._write_checkpoint(
                checkpoint,
                stage="outer_cat",
                outer_fold=outer_fold,
                inner_fold=None,
                ridge=None,
                fit_cache_key=bundle.cache_key,
                candidate_ids=finalist_ids,
                model_ids=test_ids,
                rows=rows,
            )
        elif any(float(row.get("selected_ridge", math.nan)) != selected_ridge for row in rows):
            raise NestedCVError(f"outer checkpoint has wrong selected ridge: {checkpoint}")
        return rows

    def _write_final_outputs(
        self,
        ridge_evidence: Sequence[Mapping[str, Any]],
        inner_aggregate: Sequence[Mapping[str, Any]],
        fold_finalists: Sequence[Mapping[str, Any]],
        outer_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        _atomic_csv(
            self.output_dir / "inner_ridge_results.csv", pd.DataFrame(ridge_evidence)
        )
        inner_frame = pd.DataFrame(inner_aggregate)
        _atomic_csv(self.output_dir / "inner_candidate_results.csv", inner_frame)
        _atomic_json(
            self.output_dir / "outer_fold_finalists.json",
            {
                "schema_version": SCRIPT_SCHEMA,
                "artifact_schema_version": FINALIST_SCHEMA,
                "selection_source": "inner-fold metrics only",
                "outer_outcomes_used_for_shortlisting": False,
                "allow_fallback_if_none_pass": False,
                "configured_selection_order": list(self.selection_order),
                "applied_in_phase3": [
                    "absolute_gates",
                    "lowest_p90_scenario_count",
                    "lowest_mean_scenario_count",
                ],
                "deferred_to_phase4": [
                    "lowest_total_uncertainty",
                    "prefer_trace",
                ],
                "length_tie_policy": "exact_numeric_equality",
                "phase3_declares_winner": False,
                "final_policy_freeze_authorized": False,
                "outer_evaluation_support": (
                    "each fold-local finalist set on only that fold's outer-test models"
                ),
                "cross_fold_candidate_id_comparison_authorized": False,
                "folds": list(fold_finalists),
            },
        )
        outer_frame = pd.DataFrame(outer_rows).sort_values(
            ["outer_fold", "candidate_id", "model"],
            kind="stable",
            na_position="last",
        )
        _atomic_csv(self.output_dir / "outer_oof_per_model.csv", outer_frame)

        prediction_rows: list[dict[str, Any]] = []
        outer_diagnostics: list[dict[str, Any]] = []
        for fold in range(5):
            fold_rows = [row for row in outer_rows if row.get("outer_fold") == fold]
            candidate_ids = sorted(
                {
                    str(row["candidate_id"])
                    for row in fold_rows
                    if row.get("candidate_id") is not None
                }
            )
            for candidate_id in candidate_ids:
                candidate_rows = [
                    row
                    for row in fold_rows
                    if str(row.get("candidate_id")) == candidate_id
                ]
                for mode in ("cat", "baseline"):
                    prediction_rows.append(
                        aggregate_disjoint_predictions(
                            candidate_rows,
                            fold=fold,
                            candidate_id=candidate_id,
                            mode=mode,
                        )
                    )
                outer_diagnostics.append(
                    {
                        "outer_fold": fold,
                        "candidate_id": candidate_id,
                        **aggregate_candidate_rows(
                            candidate_rows,
                            seed=self.args.seed,
                            bootstrap_replicates=self.args.metric_bootstrap_replicates,
                            label=f"outer-{fold}:{candidate_id}",
                        ),
                    }
                )
        _atomic_csv(
            self.output_dir / "disjoint_prediction_metrics.csv",
            pd.DataFrame(prediction_rows),
        )

        _atomic_json(
            self.output_dir / "outer_metrics.json",
            {
                "schema_version": SCRIPT_SCHEMA,
                "metric_kind": (
                    "per-fold Phase-3-finalist OOS diagnostics with disjoint "
                    "scenario prediction"
                ),
                "recovery_reference": "administration-pool full EAP",
                "evaluation_outcome_usage": "disjoint prediction metrics only",
                "finalist_shortlist_locked_before_outer_scoring": True,
                "outer_outcomes_used_for_shortlisting": False,
                "folds_with_finalists": sum(
                    int(row.get("n_finalists") or 0) > 0 for row in fold_finalists
                ),
                "candidate_support_policy": (
                    "identical outer-test-model support among tied finalists within "
                    "each fold; no cross-fold candidate-ID comparison"
                ),
                "per_fold_finalist_diagnostics": outer_diagnostics,
                "pooled_final_winner_metrics": None,
                "interpretation": (
                    "Outer metrics evaluate every already-shortlisted finalist. They "
                    "never alter the Phase-3 shortlist, numerical grid, ridge, or item "
                    "fit. A pooled final-winner estimate is unavailable until Phase 4 "
                    "applies total uncertainty and the deferred trace tie-break."
                ),
            },
        )

        finalist_counts = Counter(
            str(candidate_id)
            for fold in fold_finalists
            for candidate_id in fold.get("finalist_candidate_ids") or []
        )
        frequency = pd.DataFrame(
            [
                {
                    "candidate_id": candidate.candidate_id,
                    **asdict(candidate),
                    "outer_folds_shortlisted": finalist_counts[candidate.candidate_id],
                    "shortlist_fraction_all_folds": finalist_counts[candidate.candidate_id] / 5,
                }
                for candidate in self.candidates
            ]
        )
        _atomic_csv(self.output_dir / "config_finalist_frequency.csv", frequency)

        manifest = self._base_manifest("phase3_finalists_complete")
        manifest["completed_at"] = _utcnow()
        manifest["shortlist_summary"] = {
            "folds_with_finalists": sum(
                int(row.get("n_finalists") or 0) > 0 for row in fold_finalists
            ),
            "folds_without_finalists": sum(
                int(row.get("n_finalists") or 0) == 0 for row in fold_finalists
            ),
            "no_fallback_used": True,
            "phase3_declared_winner": False,
            "final_policy_freeze_authorized": False,
            "finalist_counts": dict(finalist_counts),
            "cross_fold_candidate_id_comparison_authorized": False,
            "selected_ridges_by_outer_fold": {
                str(row["outer_fold"]): row["selected_ridge"] for row in fold_finalists
            },
            "ridge_choices_used_outer_outcomes": False,
            "deferred_selection_fields": [
                "lowest_total_uncertainty",
                "prefer_trace",
            ],
            "headline_eligible": not (
                self.debug_ridge_override
                or self.debug_explicit_numerical_settings
            ),
        }
        manifest["outputs"] = {}
        for name in REQUIRED_OUTPUTS:
            if name == "manifest.json":
                continue
            path = self.output_dir / name
            manifest["outputs"][name] = {
                "path": _display_path(path),
                "sha256": _sha256(path),
            }
        _atomic_json(self.output_dir / "manifest.json", manifest)

    def run(self) -> int:
        self._prepare_output()
        _atomic_json(self.output_dir / "fold_assignments.json", self._fold_assignment_payload())
        if self.args.plan_only:
            manifest = self._base_manifest("plan_only")
            manifest["produced_outputs"] = ["fold_assignments.json", "manifest.json"]
            manifest["not_run"] = {
                "item_fits": True,
                "inner_ridge_selection": True,
                "cat_replays": True,
                "outer_scoring": True,
            }
            _atomic_json(self.output_dir / "manifest.json", manifest)
            print(f"validated nested-CV plan -> {self.output_dir}")
            return 0

        running = self._base_manifest("running")
        _atomic_json(self.output_dir / "manifest.json", running)
        all_ridge_evidence: list[dict[str, Any]] = []
        all_inner_aggregate: list[dict[str, Any]] = []
        fold_finalists: list[dict[str, Any]] = []
        all_outer_rows: list[dict[str, Any]] = []
        outer_contexts: list[tuple[Mapping[str, Any], float, dict[str, Any]]] = []
        for outer in self.split_audit["outer_folds"]:
            outer_fold = int(outer["outer_fold"])
            selected_ridge, ridge_evidence, bundles = self._select_ridge_for_outer(outer)
            inner_rows = self._inner_rows_for_outer(
                outer, selected_ridge=selected_ridge, bundles=bundles
            )
            aggregate = self._aggregate_inner(
                outer_fold, inner_rows, selected_ridge=selected_ridge
            )
            finalists, length_stage = shortlist_candidates(
                aggregate, selection_order=self.selection_order
            )
            finalist_ids = [str(row["candidate_id"]) for row in finalists]
            finalist_records = [
                {
                    "candidate_id": str(row["candidate_id"]),
                    "minimum_scenarios": int(row["minimum_scenarios"]),
                    "conditional_se_target": float(row["conditional_se_target"]),
                    "selector": str(row["selector"]),
                    "p90_scenario_count": float(row["p90_scenario_count"]),
                    "mean_scenario_count": float(row["mean_scenario_count"]),
                    "all_gates_pass": True,
                    "phase3_length_filter_status": str(
                        row["phase3_length_filter_status"]
                    ),
                    "deferred_selection_fields": list(
                        row["deferred_selection_fields"]
                    ),
                }
                for row in finalists
            ]
            fold_handoff = {
                "outer_fold": outer_fold,
                "selected_ridge": selected_ridge,
                "ridge_selection_rule": (
                    "debug --ridge override; non-headline"
                    if self.debug_ridge_override
                    else "strongest ridge within one SE of minimum inner disjoint log loss"
                ),
                "ridge_evidence_sha256": _canonical_hash(ridge_evidence),
                "ridge_selected_before_outer_scoring": True,
                "shortlist_status": (
                    "finalists_ready"
                    if finalists
                    else "no_candidate_passed_absolute_gates"
                ),
                "n_candidates_evaluated": len(aggregate),
                **length_stage,
                "n_finalists": len(finalists),
                "configured_selection_order": list(self.selection_order),
                "applied_in_phase3": [
                    "absolute_gates",
                    "lowest_p90_scenario_count",
                    "lowest_mean_scenario_count",
                ],
                "deferred_to_phase4": [
                    "lowest_total_uncertainty",
                    "prefer_trace",
                ],
                "finalist_candidate_ids": finalist_ids,
                "finalists": finalist_records,
                "inner_metrics_sha256": _canonical_hash(aggregate),
                "outer_outcomes_available_at_shortlisting": False,
                "phase3_declares_winner": False,
                "final_policy_freeze_authorized": False,
            }
            all_ridge_evidence.extend(ridge_evidence)
            all_inner_aggregate.extend(aggregate)
            fold_finalists.append(fold_handoff)
            outer_contexts.append((outer, selected_ridge, fold_handoff))
            print(
                f"outer fold {outer_fold}: "
                f"ridge={selected_ridge:g}; {fold_handoff['shortlist_status']} "
                f"({length_stage['n_candidates_passing_all_absolute_gates']}/48 "
                f"pass gates; {len(finalists)} local finalists)"
            )

        # No outer-test response is opened until all five fold-local shortlists
        # are frozen.  Each fold then evaluates only its own tied finalists on
        # the identical test-model support within that fold.  Candidate IDs are
        # never unioned or compared as if they had equal support across folds.
        for outer, selected_ridge, fold_handoff in outer_contexts:
            outer_fold = int(outer["outer_fold"])
            local_candidate_ids = list(fold_handoff["finalist_candidate_ids"])
            local_candidates = [
                {"candidate_id": candidate_id} for candidate_id in local_candidate_ids
            ]
            fold_handoff["outer_evaluation_candidate_ids"] = local_candidate_ids
            fold_handoff["outer_evaluation_support"] = (
                "fold_local_finalists_on_corresponding_outer_test_models_only"
            )
            outer_rows = self._outer_rows(
                outer, local_candidates, selected_ridge=selected_ridge
            )
            if local_candidate_ids:
                expected_pairs = {
                    (candidate_id, str(model))
                    for candidate_id in local_candidate_ids
                    for model in outer["test_model_ids"]
                }
                observed_pairs = {
                    (str(row.get("candidate_id")), str(row.get("model")))
                    for row in outer_rows
                }
                if observed_pairs != expected_pairs:
                    raise NestedCVError(
                        f"outer fold {outer_fold} OOF rows do not cover every local "
                        "finalist/model pair"
                    )
                fit_keys = sorted(
                    {str(row.get("fit_cache_key") or "") for row in outer_rows}
                )
                if len(fit_keys) != 1 or not fit_keys[0]:
                    raise NestedCVError(
                        f"outer fold {outer_fold} local finalists do not share one fit cache"
                    )
                fold_handoff["outer_training_fit_cache_key"] = fit_keys[0]
                for finalist in fold_handoff["finalists"]:
                    finalist["outer_training_fit_cache_key"] = fit_keys[0]
            else:
                fold_handoff["outer_training_fit_cache_key"] = None
            all_outer_rows.extend(outer_rows)
            print(
                f"outer fold {outer_fold}: evaluated {len(local_candidate_ids)} "
                "fold-local finalists on its held-out models"
            )

        self._write_final_outputs(
            all_ridge_evidence, all_inner_aggregate, fold_finalists, all_outer_rows
        )
        print(f"wrote nested OOS CAT finalist study -> {self.output_dir}")
        return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--matrix", type=Path, default=None)
    parser.add_argument("--rubrics", type=Path, default=None)
    parser.add_argument("--scenarios", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skills", default=",".join(DEFAULT_SKILLS))
    parser.add_argument("--dimensions", default=DEFAULT_DIMENSIONS)
    parser.add_argument("--structure-name", default="infobench_overall_1d")

    numerical = parser.add_argument_group("pre-locked numerical settings")
    numerical.add_argument("--fit-grid", type=int, default=None)
    numerical.add_argument("--eap-grid", type=int, default=None)
    numerical.add_argument("--dense-grid-manifest", type=Path, default=None)
    numerical.add_argument(
        "--eap-quadrature-method",
        choices=("gauss_hermite", "normal_trapezoid"),
        default="gauss_hermite",
        help="used only with explicit --fit-grid/--eap-grid; locked manifests carry it",
    )
    numerical.add_argument(
        "--eap-linear-bound",
        type=float,
        default=8.0,
        help="symmetric bound for explicit normal_trapezoid EAP scoring",
    )
    numerical.add_argument(
        "--debug-allow-explicit-numerical-settings",
        action="store_true",
        help=(
            "DEBUG ONLY: permit a non-plan run without the verified native "
            "quadrature-resolution lock; outputs are non-headline"
        ),
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=None,
        help=(
            "DEBUG ONLY: bypass inner ridge selection; outputs are marked non-headline"
        ),
    )

    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-scenarios", type=int, default=50)
    parser.add_argument("--minimum-scored-criteria", type=int, default=15)
    parser.add_argument("--mwle-ridge", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-grid-nodes", type=int, default=50_000)
    parser.add_argument("--metric-bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--estimate-latent-corr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--negative-policy", choices=("error", "drop", "keep"), default="drop"
    )
    parser.add_argument("--allow-unconverged-fit", action="store_true")
    parser.add_argument("--require-complete-bank", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--fresh", action="store_true")
    return parser


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.top_n < 1 or args.max_scenarios < 1 or args.minimum_scored_criteria < 0:
        raise NestedCVError("invalid CAT runtime limits")
    if args.max_iter < 1 or args.tol <= 0 or args.mwle_ridge <= 0:
        raise NestedCVError("invalid calibration/MWLE optimizer settings")
    if args.ridge is not None and args.ridge <= 0:
        raise NestedCVError("debug --ridge must be positive")
    if args.metric_bootstrap_replicates < 100:
        raise NestedCVError("metric bootstrap needs at least 100 replicates")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        _validate_runtime_args(args)
        return NestedCVRunner(args).run()
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        ValueError,
        NestedCVError,
        scat.OfflineStudyError,
        cm.CalibrationError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    finally:
        # calibrate_mirt stores its configured source axis process-locally.
        cm.configure_skills(None)


if __name__ == "__main__":
    raise SystemExit(main())
