#!/usr/bin/env python3
"""Aggregate calibration structures, k-fold validation, and CAT sweeps.

Structure selection
-------------------
The default selection rule is intentionally simple and pre-declared:

1. Find the structure with the best mean held-out k-fold metric.
2. Define the one-standard-error boundary using the *standard error of that best
   structure's fold values* (sample SD / sqrt(k)). For a loss, candidates at or
   below ``best_mean + best_se`` qualify; for a score, candidates at or above
   ``best_mean - best_se`` qualify.
3. Among qualifying structures, choose the fewest latent dimensions. Break a
   remaining tie by mean held-out performance and then stable structure name.

This favors a simpler structure when its held-out performance is statistically
indistinguishable at fold resolution. A named override is supported, but the
automatic recommendation and whether the override qualified remain in the output.
AIC/BIC and item stability are reported as supporting diagnostics; they do not
replace the held-out selection rule.

CAT bootstrap
-------------
CAT CSVs are grouped into paired comparison groups. Within each group, this module
intersects model IDs and draws one shared set of model-bootstrap samples. It uses
those same samples for every run and therefore for every run-minus-baseline
difference. The resulting percentile intervals are paired model-bootstrap 95% CIs,
not item- or criterion-bootstrap intervals.

The module exposes callable functions for an orchestrator and a JSON-spec CLI. A
minimal spec is::

    {
      "structures": [
        {"name": "five_dim", "core_manifest": "...json",
         "kfold_dir": ".../kfold"}
      ],
      "cat_sweeps": [
        {"name": "adaptive", "csv": ".../cat_per_model.csv",
         "group": "chosen_structure", "baseline": false},
        {"name": "random", "csv": ".../cat_per_model.csv",
         "group": "chosen_structure", "baseline": true}
      ],
      "estimator_oos_kfold": {"dir": ".../estimator_oos_kfold"},
      "order_stability": {"dir": ".../order_stability"},
      "parameter_uncertainty": [
        {"name": "se_0p30", "dir": ".../parameter_uncertainty", "se_target": 0.30}
      ],
      "sensitivity": [
        {"name": "grid3_ridge0p01", "dir": ".../sensitivity/grid3_ridge0p01"}
      ]
    }

Every section after ``cat_sweeps`` is optional. Missing optional studies are labeled
as not supplied; the aggregator never synthesizes substitute results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MINIMIZE_METRICS = {"log_loss", "brier", "mae", "theta_mae", "pass_rate_mae"}
MAXIMIZE_METRICS = {"accuracy", "auc", "r", "correlation", "precision_reached"}
RECOVERY_METRICS = ("r", "slope", "mae")
PASS_RATE_METRICS = ("mae", "bias", "r")
LENGTH_COLUMNS = ("scenarios_administered", "criteria_administered")
ESTIMATE_PREFIX_CANDIDATES = (
    "theta_mwle_",
    "theta_eap_",
    "theta_batch_",
    "theta_online_",
    "theta_cat_",
)
REFERENCE_PREFIX_CANDIDATES = ("theta_ref_", "theta_full_eap_", "theta_full_")


class StudyError(RuntimeError):
    """Raised when study artifacts cannot be compared safely."""


@dataclass(frozen=True)
class StructureSpec:
    name: str
    core_manifest: Path
    kfold_dir: Path
    n_dims: int | None = None


@dataclass(frozen=True)
class CatSweepSpec:
    name: str
    csv_path: Path
    group: str = "default"
    baseline: bool = False
    reference_prefix: str | None = None
    estimate_prefixes: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EstimatorOOSSpec:
    dir_path: Path
    name: str = "estimator_oos_kfold"


@dataclass(frozen=True)
class OrderStabilitySpec:
    dir_path: Path


@dataclass(frozen=True)
class ParameterUncertaintySpec:
    name: str
    dir_path: Path
    se_target: float | None = None


@dataclass(frozen=True)
class SensitivitySpec:
    name: str
    core_manifest: Path
    kfold_dir: Path
    grid: int | None = None
    ridge: float | None = None


@dataclass(frozen=True)
class StudyInputs:
    structures: tuple[StructureSpec, ...]
    cat_sweeps: tuple[CatSweepSpec, ...] = ()
    estimator_oos: EstimatorOOSSpec | None = None
    order_stability: OrderStabilitySpec | None = None
    parameter_uncertainty: tuple[ParameterUncertaintySpec, ...] = ()
    sensitivity: tuple[SensitivitySpec, ...] = ()


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise StudyError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudyError(f"{label} must contain one JSON object: {path}")
    return value


def _json_ready(value: Any) -> Any:
    """Recursively convert numpy/pandas values and non-finite floats for strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temp_name)
        raise


def metric_direction(metric: str, explicit: str | None = None) -> str:
    if explicit is not None:
        if explicit not in {"minimize", "maximize"}:
            raise StudyError("metric direction must be minimize or maximize")
        return explicit
    if metric in MINIMIZE_METRICS:
        return "minimize"
    if metric in MAXIMIZE_METRICS:
        return "maximize"
    raise StudyError(
        f"unknown direction for metric {metric!r}; pass direction='minimize' or 'maximize'"
    )


def _kfold_paths(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        return path / "kfold_summary.json", path / "metrics_per_fold.csv"
    if path.name != "kfold_summary.json":
        raise StudyError(
            f"kfold path must be a directory or kfold_summary.json, got {path}"
        )
    return path, path.parent / "metrics_per_fold.csv"


def _core_fit_block(manifest: dict) -> dict:
    comparison = manifest.get("comparison") or {}
    for key in ("modeled", "multi", "fit"):
        value = comparison.get(key)
        if isinstance(value, dict):
            return value
    if all(key in manifest for key in ("loglik", "n_params")):
        return manifest
    raise StudyError("core manifest has no modeled/multi fit block")


def _max_abs_offdiag(value: object) -> float | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        return None
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    vals = np.abs(matrix[mask])
    return float(vals.max()) if vals.size and np.isfinite(vals).any() else None


def _median_stability(summary: dict) -> float | None:
    values = []
    for block in (summary.get("item_param_stability") or {}).values():
        if not isinstance(block, dict):
            continue
        value = block.get("median_pairwise_corr")
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.median(values)) if values else None


def load_structure_result(spec: StructureSpec, primary_metric: str) -> dict:
    """Load one core fit + k-fold result into a flat comparison record."""
    if not spec.name.strip():
        raise StudyError("structure name cannot be blank")
    core = _read_json(spec.core_manifest, "core calibration manifest")
    kfold_json_path, fold_csv_path = _kfold_paths(spec.kfold_dir)
    kfold = _read_json(kfold_json_path, "k-fold summary")
    if not fold_csv_path.is_file():
        raise StudyError(f"k-fold metrics CSV not found: {fold_csv_path}")
    folds = pd.read_csv(fold_csv_path)
    if primary_metric not in folds:
        raise StudyError(
            f"k-fold metrics for {spec.name} have no primary metric {primary_metric!r}"
        )
    values = pd.to_numeric(folds[primary_metric], errors="coerce").dropna().to_numpy(float)
    if len(values) < 2:
        raise StudyError(f"{spec.name} needs at least two finite fold metrics")

    structure = ((kfold.get("config") or {}).get("structure") or {})
    dimensions = structure.get("dimensions") or []
    derived_dims = len(dimensions) if dimensions else None
    n_dims = spec.n_dims if spec.n_dims is not None else derived_dims
    if n_dims is None:
        skills = core.get("skills_order")
        n_dims = len(skills) if isinstance(skills, list) else None
    if n_dims is None or int(n_dims) < 1:
        raise StudyError(f"could not determine positive dimension count for {spec.name}")
    if spec.n_dims is not None and derived_dims is not None and spec.n_dims != derived_dims:
        raise StudyError(
            f"{spec.name}: n_dims={spec.n_dims} disagrees with k-fold structure={derived_dims}"
        )

    fit = _core_fit_block(core)
    pooled = kfold.get("pooled_oos") or {}
    em = core.get("em") or {}
    fit_converged = fit.get("converged")
    if fit_converged is None:
        fit_converged = em.get("multi_converged")
    fold_converged = (
        bool(folds["fit_converged"].astype(bool).all())
        if "fit_converged" in folds
        else None
    )
    return {
        "structure": spec.name,
        "n_dims": int(n_dims),
        "primary_metric": primary_metric,
        "cv_mean": float(np.mean(values)),
        "cv_sd": float(np.std(values, ddof=1)),
        "cv_se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "n_folds": int(len(values)),
        "pooled_oos_log_loss": pooled.get("log_loss"),
        "pooled_oos_accuracy": pooled.get("accuracy"),
        "pooled_oos_auc": pooled.get("auc"),
        "pooled_oos_brier": pooled.get("brier"),
        "metric_kind": kfold.get("metric_kind"),
        "core_loglik": fit.get("loglik"),
        "core_n_params": fit.get("n_params"),
        "core_aic": fit.get("aic"),
        "core_bic": fit.get("bic"),
        "core_converged": fit_converged,
        "all_fold_fits_converged": fold_converged,
        "n_items_fit": core.get("n_items_fit") or fit.get("n_items"),
        "max_abs_latent_correlation": _max_abs_offdiag(core.get("latent_correlation")),
        "median_item_parameter_stability": _median_stability(kfold),
        "core_manifest": str(spec.core_manifest),
        "core_manifest_sha256": _sha256(spec.core_manifest),
        "kfold_summary": str(kfold_json_path),
        "kfold_summary_sha256": _sha256(kfold_json_path),
        "metrics_per_fold": str(fold_csv_path),
        "metrics_per_fold_sha256": _sha256(fold_csv_path),
    }


def compare_structures(
    specs: Iterable[StructureSpec],
    *,
    primary_metric: str = "log_loss",
    direction: str | None = None,
    override: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply the one-SE/fewest-dimensions selection rule."""
    specs = list(specs)
    if not specs:
        raise StudyError("at least one structure is required")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise StudyError(f"duplicate structure names: {names}")
    direction = metric_direction(primary_metric, direction)
    rows = [load_structure_result(spec, primary_metric) for spec in specs]
    frame = pd.DataFrame(rows)
    frame["fit_eligible"] = (
        frame["core_converged"].eq(True)  # noqa: E712 - elementwise pandas comparison
        & frame["all_fold_fits_converged"].eq(True)  # noqa: E712
    )
    valid = frame.loc[frame["fit_eligible"]]
    if valid.empty:
        raise StudyError("no structure has both a converged core fit and all converged folds")

    best_index = (
        valid["cv_mean"].idxmin() if direction == "minimize" else valid["cv_mean"].idxmax()
    )
    best = frame.loc[best_index]
    boundary = (
        float(best["cv_mean"] + best["cv_se"])
        if direction == "minimize"
        else float(best["cv_mean"] - best["cv_se"])
    )
    tolerance = 1e-12
    if direction == "minimize":
        eligible = frame["fit_eligible"] & (frame["cv_mean"] <= boundary + tolerance)
        performance_sort = frame["cv_mean"]
    else:
        eligible = frame["fit_eligible"] & (frame["cv_mean"] >= boundary - tolerance)
        performance_sort = -frame["cv_mean"]
    frame["one_se_eligible"] = eligible
    frame["one_se_boundary"] = boundary
    frame["performance_sort"] = performance_sort

    automatic = (
        frame.loc[eligible]
        .sort_values(["n_dims", "performance_sort", "structure"], kind="stable")
        .iloc[0]
    )
    automatic_name = str(automatic["structure"])
    if override is not None:
        if override not in set(frame["structure"]):
            raise StudyError(f"override structure {override!r} is not in {names}")
        override_row = frame.loc[frame["structure"] == override].iloc[0]
        if not bool(override_row["fit_eligible"]):
            raise StudyError(
                f"override structure {override!r} does not have converged core and fold fits"
            )
        selected_name = override
        override_used = True
    else:
        selected_name = automatic_name
        override_used = False
    frame["automatic_recommendation"] = frame["structure"] == automatic_name
    frame["selected"] = frame["structure"] == selected_name
    frame = frame.drop(columns=["performance_sort"]).sort_values(
        ["n_dims", "structure"], kind="stable"
    )

    selected = frame.loc[frame["selected"]].iloc[0]
    selection = {
        "generated_at": _utcnow(),
        "rule": {
            "name": "one-standard-error-then-fewest-dimensions",
            "primary_metric": primary_metric,
            "direction": direction,
            "standard_error": "sample SD of fold metrics divided by sqrt(number of folds)",
            "boundary": boundary,
            "best_mean_structure": str(best["structure"]),
            "best_mean": float(best["cv_mean"]),
            "best_standard_error": float(best["cv_se"]),
            "tie_breaks": ["fewest_dimensions", "better_cv_mean", "structure_name"],
            "fit_gate": "core fit and every fold fit must have converged",
            "note": (
                "A loss qualifies when mean <= best mean + SE(best); a score qualifies "
                "when mean >= best mean - SE(best). AIC/BIC are supporting diagnostics."
            ),
        },
        "automatic_recommendation": automatic_name,
        "override_requested": override,
        "override_used": override_used,
        "selected_structure": selected_name,
        "selected_n_dims": int(selected["n_dims"]),
        "selected_cv_mean": float(selected["cv_mean"]),
        "selected_was_one_se_eligible": bool(selected["one_se_eligible"]),
        "eligible_structures": frame.loc[frame["one_se_eligible"], "structure"].tolist(),
    }
    return frame, selection


def _resolve_reference_prefix(frame: pd.DataFrame, explicit: str | None) -> str:
    candidates = (explicit,) if explicit else REFERENCE_PREFIX_CANDIDATES
    for prefix in candidates:
        if prefix and any(column.startswith(prefix) for column in frame.columns):
            return prefix
    raise StudyError(
        f"CAT CSV has no reference theta columns; tried {list(candidates)}"
    )


def _resolve_estimate_prefixes(
    frame: pd.DataFrame, explicit: tuple[str, ...]
) -> tuple[str, ...]:
    candidates = explicit or ESTIMATE_PREFIX_CANDIDATES
    present = tuple(
        prefix
        for prefix in candidates
        if any(column.startswith(prefix) for column in frame.columns)
    )
    if not present:
        raise StudyError(f"CAT CSV has no estimate theta columns; tried {list(candidates)}")
    return present


def _load_cat_frame(
    spec: CatSweepSpec,
) -> tuple[pd.DataFrame, str, tuple[str, ...], tuple[str, ...]]:
    if not spec.csv_path.is_file():
        raise StudyError(f"CAT sweep CSV not found: {spec.csv_path}")
    frame = pd.read_csv(spec.csv_path)
    if "model" not in frame:
        raise StudyError(f"CAT sweep {spec.name} has no model column")
    frame["model"] = frame["model"].astype(str)
    if frame["model"].duplicated().any():
        raise StudyError(f"CAT sweep {spec.name} has duplicate model rows")
    frame = frame.set_index("model", drop=False)
    reference = _resolve_reference_prefix(frame, spec.reference_prefix)
    estimates = _resolve_estimate_prefixes(frame, spec.estimate_prefixes)
    reference_skills = {
        column[len(reference) :] for column in frame.columns if column.startswith(reference)
    }
    skills = tuple(spec.skills) if spec.skills else tuple(sorted(reference_skills))
    if not skills:
        raise StudyError(f"CAT sweep {spec.name} has no recoverable skills")
    missing_reference = [skill for skill in skills if f"{reference}{skill}" not in frame]
    if missing_reference:
        raise StudyError(
            f"CAT sweep {spec.name} reference prefix {reference!r} misses skills "
            f"{missing_reference}"
        )
    for prefix in estimates:
        missing = [skill for skill in skills if f"{prefix}{skill}" not in frame]
        if missing:
            raise StudyError(
                f"CAT sweep {spec.name} estimate prefix {prefix!r} misses skills {missing}"
            )
    return frame, reference, estimates, skills


def _recovery(metric: str, reference: np.ndarray, estimate: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    x, y = reference[valid], estimate[valid]
    if len(x) < 2:
        return float("nan")
    if metric == "mae":
        return float(np.mean(np.abs(y - x)))
    if float(np.std(x)) == 0:
        return float("nan")
    if metric == "slope":
        return float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    if metric == "r":
        if float(np.std(y)) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    raise StudyError(f"unknown recovery metric {metric}")


def _pass_rate_metric(metric: str, observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    actual, estimate = observed[valid], predicted[valid]
    if actual.size == 0:
        return float("nan")
    if metric == "mae":
        return float(np.mean(np.abs(estimate - actual)))
    if metric == "bias":
        return float(np.mean(estimate - actual))
    if metric == "r":
        if actual.size < 3 or float(np.std(actual)) == 0 or float(np.std(estimate)) == 0:
            return float("nan")
        return float(np.corrcoef(actual, estimate)[0, 1])
    raise StudyError(f"unknown pass-rate metric {metric}")


def _percentile_ci(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_stat(
    func: Callable[[np.ndarray], float], draws: np.ndarray
) -> np.ndarray:
    return np.array([func(index) for index in draws], dtype=float)


def _metric_row(
    *,
    group: str,
    run: str,
    estimator: str,
    skill: str,
    metric: str,
    estimate: float,
    bootstrap: np.ndarray,
    n_models: int,
    b: int,
    seed: int,
) -> dict:
    low, high = _percentile_ci(bootstrap)
    return {
        "group": group,
        "run": run,
        "estimator": estimator,
        "skill": skill,
        "metric": metric,
        "estimate": estimate,
        "ci_lower": low,
        "ci_upper": high,
        "n_models": n_models,
        "bootstrap_B": b,
        "bootstrap_seed": seed,
    }


def aggregate_cat_sweeps(
    specs: Iterable[CatSweepSpec], *, b: int = 2000, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate CAT sweeps with paired model-bootstrap confidence intervals."""
    specs = list(specs)
    if b < 1:
        raise StudyError("bootstrap B must be positive")
    if not specs:
        empty = pd.DataFrame()
        return empty, empty, {"groups": {}, "bootstrap_B": b, "bootstrap_seed": seed}
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise StudyError(f"duplicate CAT sweep names: {names}")

    loaded = {spec.name: _load_cat_frame(spec) for spec in specs}
    specs_by_group: dict[str, list[CatSweepSpec]] = defaultdict(list)
    for spec in specs:
        specs_by_group[spec.group].append(spec)

    metric_rows: list[dict] = []
    bootstrap_by_key: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    group_meta: dict[str, dict] = {}
    for group_index, (group, members) in enumerate(sorted(specs_by_group.items())):
        baseline = [spec for spec in members if spec.baseline]
        if len(baseline) > 1:
            raise StudyError(f"CAT group {group!r} has multiple baselines")
        common_models = set(loaded[members[0].name][0].index)
        for spec in members[1:]:
            common_models &= set(loaded[spec.name][0].index)
        model_order = sorted(common_models)
        if len(model_order) < 3:
            raise StudyError(f"CAT group {group!r} has fewer than 3 common models")
        group_seed = seed + group_index
        rng = np.random.default_rng(group_seed)
        draws = rng.integers(0, len(model_order), size=(b, len(model_order)))
        group_meta[group] = {
            "runs": [spec.name for spec in members],
            "baseline": baseline[0].name if baseline else None,
            "n_common_models": len(model_order),
            "common_models": model_order,
            "excluded_models_by_run": {
                spec.name: sorted(set(loaded[spec.name][0].index) - set(model_order))
                for spec in members
            },
            "paired_draw_seed": group_seed,
        }

        for spec in members:
            frame, reference_prefix, estimate_prefixes, skills = loaded[spec.name]
            frame = frame.loc[model_order]
            for estimate_prefix in estimate_prefixes:
                estimator = estimate_prefix.removeprefix("theta_").removesuffix("_")
                per_skill_boot: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
                per_skill_point: dict[str, dict[str, float]] = defaultdict(dict)
                for skill in skills:
                    reference = pd.to_numeric(
                        frame[f"{reference_prefix}{skill}"], errors="coerce"
                    ).to_numpy(float)
                    estimate_values = pd.to_numeric(
                        frame[f"{estimate_prefix}{skill}"], errors="coerce"
                    ).to_numpy(float)
                    for metric in RECOVERY_METRICS:
                        point = _recovery(metric, reference, estimate_values)
                        boots = _bootstrap_stat(
                            lambda index, m=metric, x=reference, y=estimate_values: _recovery(
                                m, x[index], y[index]
                            ),
                            draws,
                        )
                        per_skill_point[metric][skill] = point
                        per_skill_boot[metric][skill] = boots
                        key = (group, spec.name, estimator, skill, metric)
                        bootstrap_by_key[key] = boots
                        metric_rows.append(
                            _metric_row(
                                group=group, run=spec.name, estimator=estimator,
                                skill=skill, metric=metric, estimate=point,
                                bootstrap=boots,
                                n_models=len(model_order),
                                b=b,
                                seed=group_seed,
                            )
                        )
                for metric in RECOVERY_METRICS:
                    macro_point = float(np.nanmean(list(per_skill_point[metric].values())))
                    macro_boot = np.nanmean(
                        np.stack(list(per_skill_boot[metric].values()), axis=1), axis=1
                    )
                    key = (group, spec.name, estimator, "__macro__", metric)
                    bootstrap_by_key[key] = macro_boot
                    metric_rows.append(
                        _metric_row(
                            group=group, run=spec.name, estimator=estimator,
                            skill="__macro__", metric=metric, estimate=macro_point,
                            bootstrap=macro_boot,
                            n_models=len(model_order),
                            b=b,
                            seed=group_seed,
                        )
                    )

            for column in LENGTH_COLUMNS:
                if column not in frame:
                    continue
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                point = float(np.nanmean(values))
                boots = _bootstrap_stat(lambda index, v=values: float(np.nanmean(v[index])), draws)
                key = (group, spec.name, "__run__", "__all__", f"mean_{column}")
                bootstrap_by_key[key] = boots
                metric_rows.append(
                    _metric_row(
                        group=group, run=spec.name, estimator="__run__", skill="__all__",
                        metric=f"mean_{column}", estimate=point, bootstrap=boots,
                        n_models=len(model_order),
                        b=b,
                        seed=group_seed,
                    )
                )
            if "precision_reached" in frame:
                raw = frame["precision_reached"]
                if raw.dtype == object:
                    values = raw.astype(str).str.lower().map(
                        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
                    ).to_numpy(float)
                else:
                    values = pd.to_numeric(raw, errors="coerce").to_numpy(float)
                point = float(np.nanmean(values))
                boots = _bootstrap_stat(lambda index, v=values: float(np.nanmean(v[index])), draws)
                key = (group, spec.name, "__run__", "__all__", "precision_reached_rate")
                bootstrap_by_key[key] = boots
                metric_rows.append(
                    _metric_row(
                        group=group, run=spec.name, estimator="__run__", skill="__all__",
                        metric="precision_reached_rate", estimate=point, bootstrap=boots,
                        n_models=len(model_order),
                        b=b,
                        seed=group_seed,
                    )
                )

    metrics_frame = pd.DataFrame(metric_rows)
    differences: list[dict] = []
    for group, members in sorted(specs_by_group.items()):
        baselines = [spec for spec in members if spec.baseline]
        if not baselines:
            continue
        baseline_name = baselines[0].name
        baseline_rows = metrics_frame[
            (metrics_frame["group"] == group) & (metrics_frame["run"] == baseline_name)
        ]
        baseline_lookup = {
            (row.estimator, row.skill, row.metric): row for row in baseline_rows.itertuples()
        }
        for spec in members:
            if spec.name == baseline_name:
                continue
            run_rows = metrics_frame[
                (metrics_frame["group"] == group) & (metrics_frame["run"] == spec.name)
            ]
            for row in run_rows.itertuples():
                metric_key = (row.estimator, row.skill, row.metric)
                baseline_row = baseline_lookup.get(metric_key)
                if baseline_row is None:
                    continue
                run_key = (group, spec.name, row.estimator, row.skill, row.metric)
                base_key = (group, baseline_name, row.estimator, row.skill, row.metric)
                boot = bootstrap_by_key[run_key] - bootstrap_by_key[base_key]
                low, high = _percentile_ci(boot)
                differences.append(
                    {
                        "group": group,
                        "run": spec.name,
                        "baseline": baseline_name,
                        "estimator": row.estimator,
                        "skill": row.skill,
                        "metric": row.metric,
                        "run_minus_baseline": float(row.estimate - baseline_row.estimate),
                        "ci_lower": low,
                        "ci_upper": high,
                        "n_models": int(row.n_models),
                        "bootstrap_B": b,
                        "bootstrap_seed": group_meta[group]["paired_draw_seed"],
                    }
                )
    differences_frame = pd.DataFrame(differences)
    metadata = {
        "generated_at": _utcnow(),
        "method": "paired model bootstrap percentile 95% CI",
        "bootstrap_B": b,
        "bootstrap_seed": seed,
        "pairing": (
            "Within each comparison group, model IDs are intersected and one shared "
            "bootstrap index matrix is used for every run and paired difference."
        ),
        "groups": group_meta,
        "inputs": {
            spec.name: {
                "csv": str(spec.csv_path),
                "sha256": _sha256(spec.csv_path),
                "metadata": spec.metadata,
            }
            for spec in specs
        },
    }
    return metrics_frame, differences_frame, metadata


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame]
    if missing:
        raise StudyError(f"{label} is missing required columns {missing}")


def aggregate_estimator_oos(
    spec: EstimatorOOSSpec | None, *, b: int = 2000, seed: int = 0
) -> tuple[pd.DataFrame, dict]:
    """Aggregate held-out-person estimator recovery with model-bootstrap CIs."""
    if spec is None:
        return pd.DataFrame(), {"status": "not_supplied"}
    if b < 1:
        raise StudyError("estimator OOS bootstrap B must be positive")
    per_model_path = spec.dir_path / "oos_per_model.csv"
    metrics_path = spec.dir_path / "metrics_aggregate.json"
    manifest_path = spec.dir_path / "manifest.json"
    pass_rate_path = spec.dir_path / "pass_rate_calibration.csv"
    for path, label in (
        (per_model_path, "estimator OOS per-model CSV"),
        (metrics_path, "estimator OOS metrics"),
        (manifest_path, "estimator OOS manifest"),
    ):
        if not path.is_file():
            raise StudyError(f"{label} not found: {path}")
    frame = pd.read_csv(per_model_path)
    _require_columns(frame, ("model", "status"), "estimator OOS per-model CSV")
    frame["model"] = frame["model"].astype(str)
    if frame["model"].duplicated().any():
        raise StudyError("estimator OOS per-model CSV has duplicate model rows")
    successful = frame.loc[frame["status"].astype(str).str.lower() == "ok"].copy()
    if len(successful) < 3:
        raise StudyError("estimator OOS study has fewer than 3 successful held-out models")
    reference_prefix = _resolve_reference_prefix(successful, "theta_ref_")
    skills = tuple(
        sorted(
            column.removeprefix(reference_prefix)
            for column in successful
            if column.startswith(reference_prefix)
        )
    )
    estimate_prefixes = _resolve_estimate_prefixes(successful, ())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(successful), size=(b, len(successful)))
    rows: list[dict[str, Any]] = []
    for estimate_prefix in estimate_prefixes:
        estimator = estimate_prefix.removeprefix("theta_").removesuffix("_")
        per_skill_points: dict[str, dict[str, float]] = defaultdict(dict)
        per_skill_boots: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for skill in skills:
            reference = pd.to_numeric(
                successful[f"{reference_prefix}{skill}"], errors="coerce"
            ).to_numpy(float)
            estimate = pd.to_numeric(
                successful[f"{estimate_prefix}{skill}"], errors="coerce"
            ).to_numpy(float)
            if estimator == "mwle" and "mwle_converged" in successful:
                converged = successful["mwle_converged"]
                if converged.dtype == object:
                    valid_mwle = converged.astype(str).str.lower().isin(("true", "1"))
                else:
                    valid_mwle = converged.fillna(False).astype(bool)
                estimate = estimate.copy()
                estimate[~valid_mwle.to_numpy(bool)] = np.nan
            for metric in RECOVERY_METRICS:
                point = _recovery(metric, reference, estimate)
                boots = _bootstrap_stat(
                    lambda index, m=metric, x=reference, y=estimate: _recovery(
                        m, x[index], y[index]
                    ),
                    draws,
                )
                per_skill_points[metric][skill] = point
                per_skill_boots[metric][skill] = boots
                low, high = _percentile_ci(boots)
                rows.append(
                    {
                        "study": spec.name,
                        "estimator": estimator,
                        "skill": skill,
                        "metric": metric,
                        "estimate": point,
                        "ci_lower": low,
                        "ci_upper": high,
                        "n_models": int(
                            np.sum(np.isfinite(reference) & np.isfinite(estimate))
                        ),
                        "bootstrap_B": b,
                        "bootstrap_seed": seed,
                    }
                )
        for metric in RECOVERY_METRICS:
            point = float(np.nanmean(list(per_skill_points[metric].values())))
            boots = np.nanmean(
                np.stack(list(per_skill_boots[metric].values()), axis=1), axis=1
            )
            low, high = _percentile_ci(boots)
            rows.append(
                {
                    "study": spec.name,
                    "estimator": estimator,
                    "skill": "__macro__",
                    "metric": metric,
                    "estimate": point,
                    "ci_lower": low,
                    "ci_upper": high,
                    "n_models": int(len(successful)),
                    "bootstrap_B": b,
                    "bootstrap_seed": seed,
                }
            )
    pooled_pass_rate: list[dict[str, Any]] = []
    if pass_rate_path.is_file():
        pass_rate = pd.read_csv(pass_rate_path)
        _require_columns(
            pass_rate,
            ("fold", "estimator", "n", "mae", "bias", "r"),
            "pass-rate calibration",
        )
        pooled = pass_rate.loc[pass_rate["fold"].astype(str) == "pooled"].copy()
        pooled_pass_rate = _json_ready(pooled.to_dict(orient="records"))
        observed = (
            pd.to_numeric(successful["observed_fitted_pass_rate"], errors="coerce")
            .to_numpy(float)
            if "observed_fitted_pass_rate" in successful
            else None
        )
        for record in pooled.to_dict(orient="records"):
            estimator = str(record["estimator"])
            column = f"pirt_predicted_pass_rate_{estimator}"
            if observed is None or column not in successful:
                continue
            predicted = pd.to_numeric(successful[column], errors="coerce").to_numpy(float)
            if estimator == "mwle" and "mwle_converged" in successful:
                predicted = predicted.copy()
                predicted[~_boolean_series(successful["mwle_converged"]).to_numpy(bool)] = np.nan
            for metric in PASS_RATE_METRICS:
                point = _pass_rate_metric(metric, observed, predicted)
                boots = _bootstrap_stat(
                    lambda index, m=metric, x=observed, y=predicted: _pass_rate_metric(
                        m, x[index], y[index]
                    ),
                    draws,
                )
                low, high = _percentile_ci(boots)
                rows.append(
                    {
                        "study": spec.name,
                        "estimator": estimator,
                        "skill": "__pass_rate__",
                        "metric": f"pass_rate_{metric}",
                        "estimate": point,
                        "ci_lower": low,
                        "ci_upper": high,
                        "n_models": int(
                            np.sum(np.isfinite(observed) & np.isfinite(predicted))
                        ),
                        "bootstrap_B": b,
                        "bootstrap_seed": seed,
                    }
                )
    metadata = {
        "status": "available",
        "study": spec.name,
        "method": "held-out-person paired model bootstrap percentile 95% CI",
        "bootstrap_B": b,
        "bootstrap_seed": seed,
        "n_models_total": int(len(frame)),
        "n_successful_replays": int(len(successful)),
        "n_replay_errors": int(len(frame) - len(successful)),
        "skills": list(skills),
        "estimators": [
            prefix.removeprefix("theta_").removesuffix("_")
            for prefix in estimate_prefixes
        ],
        "pass_rate_calibration": {
            "status": "available" if pass_rate_path.is_file() else "not_supplied",
            "pooled_rows": pooled_pass_rate,
        },
        "inputs": {
            "oos_per_model": {
                "path": str(per_model_path),
                "sha256": _sha256(per_model_path),
            },
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        },
    }
    if pass_rate_path.is_file():
        metadata["inputs"]["pass_rate_calibration"] = {
            "path": str(pass_rate_path),
            "sha256": _sha256(pass_rate_path),
        }
    return pd.DataFrame(rows), metadata


def aggregate_sensitivity(
    specs: Iterable[SensitivitySpec], *, primary_metric: str = "log_loss"
) -> tuple[pd.DataFrame, dict]:
    """Collect core-fit and held-out metrics for grid/ridge sensitivity runs."""
    specs = list(specs)
    if not specs:
        return pd.DataFrame(), {"status": "not_supplied", "runs": []}
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise StudyError(f"duplicate sensitivity names: {names}")
    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = load_structure_result(
            StructureSpec(
                name=spec.name,
                core_manifest=spec.core_manifest,
                kfold_dir=spec.kfold_dir,
            ),
            primary_metric,
        )
        core = _read_json(spec.core_manifest, "sensitivity core manifest")
        kfold_path, _ = _kfold_paths(spec.kfold_dir)
        kfold = _read_json(kfold_path, "sensitivity k-fold summary")
        config = kfold.get("config") or {}
        row["grid"] = (
            spec.grid
            if spec.grid is not None
            else core.get("grid_nodes_per_dim", config.get("grid"))
        )
        row["ridge"] = (
            spec.ridge if spec.ridge is not None else core.get("ridge", config.get("ridge"))
        )
        rows.append(row)
    return pd.DataFrame(rows), {
        "status": "available",
        "primary_metric": primary_metric,
        "runs": names,
    }


def load_order_stability(
    spec: OrderStabilitySpec | None,
) -> tuple[pd.DataFrame, dict]:
    """Load the deterministic order/seed-dependence summary."""
    if spec is None:
        return pd.DataFrame(), {"status": "not_supplied"}
    metrics_path = spec.dir_path / "metrics.json"
    spread_path = spec.dir_path / "spread_summary.csv"
    metrics = _read_json(metrics_path, "order-stability metrics")
    if not spread_path.is_file():
        raise StudyError(f"order-stability summary CSV not found: {spread_path}")
    spread = pd.read_csv(spread_path)
    _require_columns(
        spread,
        ("estimator", "dimension", "n_models_with_spread", "median_sd", "max_sd"),
        "order-stability summary",
    )
    metadata = {
        "status": "available",
        "n_models": metrics.get("n_models"),
        "seeds": metrics.get("seeds"),
        "n_run_failures": metrics.get("n_run_failures"),
        "n_mwle_failures": metrics.get("n_mwle_failures"),
        "config": metrics.get("config"),
        "inputs": {
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
            "spread_summary": {
                "path": str(spread_path),
                "sha256": _sha256(spread_path),
            },
        },
    }
    return spread, metadata


def _boolean_series(values: pd.Series) -> pd.Series:
    if values.dtype == object:
        return values.astype(str).str.lower().isin(("true", "1"))
    return values.fillna(False).astype(bool)


def aggregate_parameter_uncertainty(
    specs: Iterable[ParameterUncertaintySpec],
) -> tuple[pd.DataFrame, dict]:
    """Collect reliable SE components, optionally across multiple requested SE targets."""
    specs = list(specs)
    if not specs:
        return pd.DataFrame(), {"status": "not_supplied", "runs": {}}
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise StudyError(f"duplicate parameter-uncertainty names: {names}")
    rows: list[dict[str, Any]] = []
    input_meta: dict[str, Any] = {}
    for spec in specs:
        metrics_path = spec.dir_path / "metrics.json"
        components_path = spec.dir_path / "ability_se_components.csv"
        aggregate_path = spec.dir_path / "total_se_vs_target.csv"
        metrics = _read_json(metrics_path, "parameter-uncertainty metrics")
        if not components_path.is_file():
            raise StudyError(f"parameter-uncertainty components not found: {components_path}")
        components = pd.read_csv(components_path)
        _require_columns(
            components,
            (
                "model",
                "estimator",
                "dimension",
                "se_ability",
                "se_param",
                "se_total",
                "reliable",
            ),
            "parameter-uncertainty components",
        )
        components["reliable"] = _boolean_series(components["reliable"])
        inferred_target = (metrics.get("config") or {}).get("max_se")
        fallback_target = spec.se_target if spec.se_target is not None else inferred_target
        if "se_target" in components:
            components["se_target"] = pd.to_numeric(components["se_target"], errors="coerce")
            observed_targets = sorted(components["se_target"].dropna().unique().tolist())
            if spec.se_target is not None and any(
                not math.isclose(float(value), float(spec.se_target))
                for value in observed_targets
            ):
                raise StudyError(
                    f"parameter-uncertainty run {spec.name!r} has component SE targets "
                    f"{observed_targets}, inconsistent with spec se_target={spec.se_target}"
                )
        else:
            target = float(fallback_target) if fallback_target is not None else np.nan
            components["se_target"] = target
            observed_targets = [target] if math.isfinite(target) else []
        aggregate_lookup: dict[tuple[float, str, str], dict[str, Any]] = {}
        if aggregate_path.is_file():
            supplied_aggregate = pd.read_csv(aggregate_path)
            _require_columns(
                supplied_aggregate,
                ("se_target", "estimator", "dimension"),
                "total-SE-vs-target summary",
            )
            for supplied in supplied_aggregate.to_dict(orient="records"):
                key = (
                    float(supplied["se_target"]),
                    str(supplied["estimator"]),
                    str(supplied["dimension"]),
                )
                aggregate_lookup[key] = supplied
        for (target, estimator, dimension), subset in components.groupby(
            ["se_target", "estimator", "dimension"], sort=True, dropna=False
        ):
            reliable = subset.loc[subset["reliable"]].copy()
            for column in ("se_ability", "se_param", "se_total"):
                reliable[column] = pd.to_numeric(reliable[column], errors="coerce")
            supplied = aggregate_lookup.get(
                (float(target), str(estimator), str(dimension)), {}
            )
            rows.append(
                {
                    "run": spec.name,
                    "se_target": float(target) if pd.notna(target) else None,
                    "estimator": str(estimator),
                    "dimension": str(dimension),
                    "n_models": int(len(subset)),
                    "n_reliable_models": int(len(reliable)),
                    "median_se_ability": (
                        float(reliable["se_ability"].median()) if len(reliable) else None
                    ),
                    "median_se_param": (
                        float(reliable["se_param"].median()) if len(reliable) else None
                    ),
                    "median_se_total": (
                        float(reliable["se_total"].median()) if len(reliable) else None
                    ),
                    "mean_se_total": supplied.get(
                        "mean_se_total",
                        float(reliable["se_total"].mean()) if len(reliable) else None,
                    ),
                    "mean_scenarios": supplied.get("mean_scenarios"),
                    "precision_reached_rate": supplied.get("precision_reached_rate"),
                }
            )
        input_meta[spec.name] = {
            "se_targets": observed_targets,
            "n_boot_requested": metrics.get("n_boot_requested"),
            "n_boot_fit_success": metrics.get("n_boot_fit_success"),
            "n_boot_fit_failures": metrics.get("n_boot_fit_failures"),
            "n_base_replay_failures": metrics.get("n_base_replay_failures"),
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
            "components": {
                "path": str(components_path),
                "sha256": _sha256(components_path),
            },
        }
        if aggregate_path.is_file():
            input_meta[spec.name]["total_se_vs_target"] = {
                "path": str(aggregate_path),
                "sha256": _sha256(aggregate_path),
            }
    return pd.DataFrame(rows), {"status": "available", "runs": input_meta}


def render_plain_language_summary(
    comparison: pd.DataFrame,
    selection: dict,
    cat_metrics: pd.DataFrame | None = None,
    cat_differences: pd.DataFrame | None = None,
    *,
    cat_metadata: dict | None = None,
    estimator_oos: pd.DataFrame | None = None,
    sensitivity: pd.DataFrame | None = None,
    order_spread: pd.DataFrame | None = None,
    order_metadata: dict | None = None,
    parameter_uncertainty: pd.DataFrame | None = None,
    figures: dict | None = None,
    unavailable: Iterable[str] = (),
) -> str:
    selected = comparison.loc[comparison["selected"]].iloc[0]
    automatic = selection["automatic_recommendation"]
    lines = [
        "# Calibration study summary",
        "",
        "## Skill-structure decision",
        "",
        (
            f"The automatic one-standard-error rule recommends **{automatic}**. "
            f"The reported selection is **{selection['selected_structure']}** "
            f"({int(selected['n_dims'])} latent dimension(s))."
        ),
        "",
        (
            f"The best mean held-out {selection['rule']['primary_metric']} came from "
            f"**{selection['rule']['best_mean_structure']}** at "
            f"{selection['rule']['best_mean']:.4f}. Its fold standard error was "
            f"{selection['rule']['best_standard_error']:.4f}, giving a one-SE boundary "
            f"of {selection['rule']['boundary']:.4f}. Structures inside that boundary: "
            f"{', '.join(selection['eligible_structures'])}."
        ),
        "",
        (
            "The rule first keeps structures whose held-out performance is within one "
            "standard error of the best, then chooses the one with the fewest dimensions. "
            "AIC/BIC and parameter stability remain supporting diagnostics."
        ),
    ]
    if selection["override_used"]:
        lines.extend(
            [
                "",
                (
                    f"A manual override selected **{selection['selected_structure']}**. "
                    "It was "
                    f"{'inside' if selection['selected_was_one_se_eligible'] else 'outside'} "
                    "the automatic one-SE set; the automatic recommendation is preserved above."
                ),
            ]
        )
    if cat_metrics is not None and not cat_metrics.empty:
        lines.extend(
            [
                "",
                "## CAT sweep aggregation",
                "",
                (
                    f"Aggregated {cat_metrics['run'].nunique()} CAT run(s) using paired "
                    "model-bootstrap 95% confidence intervals. Recovery is reported as "
                    "correlation, slope, and MAE; test length is reported as mean scenarios "
                    "and criteria where available."
                ),
            ]
        )
        selected_cat = (
            (((cat_metadata or {}).get("inputs") or {}).get("selected_configuration") or {})
            .get("metadata", {})
        )
        if {
            "min_scenarios",
            "max_se",
            "selection",
        }.issubset(selected_cat):
            lines.extend(
                [
                    "",
                    (
                        "The study-selected provisional CAT configuration used a minimum of "
                        f"**{selected_cat['min_scenarios']} scenarios**, an SE target "
                        f"of **{selected_cat['max_se']}**, and the "
                        f"**{selected_cat['selection']}** selector."
                    ),
                ]
            )
        if cat_differences is not None and not cat_differences.empty:
            length = cat_differences[
                cat_differences["metric"] == "mean_scenarios_administered"
            ]
            for row in length.itertuples():
                lines.append(
                    f"- **{row.run}** used {row.run_minus_baseline:+.2f} scenarios versus "
                    f"**{row.baseline}** (95% CI {row.ci_lower:+.2f} to {row.ci_upper:+.2f})."
                )
    if estimator_oos is not None and not estimator_oos.empty:
        lines.extend(["", "## Held-out estimator recovery", ""])
        macro = estimator_oos[
            (estimator_oos["skill"] == "__macro__")
            & (estimator_oos["metric"].isin(RECOVERY_METRICS))
        ]
        for estimator, group in macro.groupby("estimator", sort=True):
            values = {row.metric: row for row in group.itertuples()}
            parts = []
            for metric in RECOVERY_METRICS:
                row = values.get(metric)
                if row is None:
                    continue
                numeric = (row.estimate, row.ci_lower, row.ci_upper)
                if not all(value is not None and math.isfinite(float(value)) for value in numeric):
                    parts.append(f"{metric} not estimable")
                    continue
                parts.append(
                    f"{metric} {row.estimate:.3f} "
                    f"(95% CI {row.ci_lower:.3f} to {row.ci_upper:.3f})"
                )
            if parts:
                lines.append(f"- **{estimator}**: " + "; ".join(parts) + ".")
        pass_rate = estimator_oos[estimator_oos["skill"] == "__pass_rate__"]
        if len(pass_rate):
            lines.extend(["", "### p-IRT pass-rate calibration", ""])
            for estimator, group in pass_rate.groupby("estimator", sort=True):
                values = {row.metric: row for row in group.itertuples()}
                parts = []
                for metric in ("pass_rate_mae", "pass_rate_bias", "pass_rate_r"):
                    row = values.get(metric)
                    if row is None or not math.isfinite(float(row.estimate)):
                        continue
                    parts.append(f"{metric.removeprefix('pass_rate_')} {row.estimate:.3f}")
                if parts:
                    lines.append(f"- **{estimator}**: " + "; ".join(parts) + ".")
            lines.extend(
                [
                    "",
                    (
                        "MAE is absolute predicted-vs-observed pass-rate error; bias is "
                        "predicted minus observed; r measures whether models are ranked "
                        "similarly by predicted and observed pass rate."
                    ),
                ]
            )
        lines.extend(
            [
                "",
                (
                    "These are honest held-out-person results: each model was excluded "
                    "from the item-parameter fit used to score it. Confidence intervals "
                    "resample tutor models."
                ),
            ]
        )
    if sensitivity is not None and not sensitivity.empty:
        finite = sensitivity.dropna(subset=["cv_mean"])
        lines.extend(["", "## Grid and ridge sensitivity", ""])
        if len(finite):
            sensitivity_direction = metric_direction(str(finite.iloc[0]["primary_metric"]))
            best_index = (
                finite["cv_mean"].idxmin()
                if sensitivity_direction == "minimize"
                else finite["cv_mean"].idxmax()
            )
            best = finite.loc[best_index]
            lines.append(
                f"Compared {len(finite)} supplied sensitivity runs. The lowest held-out "
                f"fold mean was **{best['structure']}** at {best['cv_mean']:.4f} "
                f"(grid={best['grid']}, ridge={best['ridge']})."
            )
        lines.append(
            "Sensitivity results are diagnostics; this report does not silently replace "
            "the predeclared selected structure or settings."
        )
    if order_spread is not None and not order_spread.empty:
        lines.extend(["", "## Scenario-order stability", ""])
        mwle = order_spread[order_spread["estimator"].astype(str) == "mwle"]
        if len(mwle):
            median_sd = pd.to_numeric(mwle["median_sd"], errors="coerce").median()
            max_sd = pd.to_numeric(mwle["max_sd"], errors="coerce").max()
            if math.isfinite(float(median_sd)) and math.isfinite(float(max_sd)):
                lines.append(
                    f"Across the supplied seeds, MWLE's median across-model theta SD was "
                    f"{median_sd:.3f} across dimensions; the largest reported model SD was "
                    f"{max_sd:.3f}."
                )
        if order_metadata and order_metadata.get("seeds"):
            lines.append(
                f"Seeds tested: {', '.join(map(str, order_metadata['seeds']))}; "
                f"run failures: {order_metadata.get('n_run_failures', 'unknown')}."
            )
    if parameter_uncertainty is not None and not parameter_uncertainty.empty:
        lines.extend(["", "## Item-parameter uncertainty and total SE", ""])
        mwle = parameter_uncertainty[
            parameter_uncertainty["estimator"].astype(str) == "mwle"
        ]
        for (run, target), group in mwle.groupby(
            ["run", "se_target"], sort=True, dropna=False
        ):
            total = pd.to_numeric(group["median_se_total"], errors="coerce").dropna()
            reliable_values = pd.to_numeric(
                group["n_reliable_models"], errors="coerce"
            ).dropna()
            target_text = f" at requested SE {float(target):.2f}" if pd.notna(target) else ""
            if len(total):
                reliable_text = (
                    f"; at least {int(reliable_values.min())} reliable models per dimension"
                    if len(reliable_values)
                    else ""
                )
                tradeoff = ""
                scenarios = pd.to_numeric(group["mean_scenarios"], errors="coerce").dropna()
                precision = pd.to_numeric(
                    group["precision_reached_rate"], errors="coerce"
                ).dropna()
                if len(scenarios) and len(precision):
                    tradeoff = (
                        f"; mean scenarios {scenarios.iloc[0]:.1f}; precision reached "
                        f"{precision.iloc[0]:.1%}"
                    )
                lines.append(
                    f"- **{run}**{target_text}: median MWLE total SE across dimensions "
                    f"{total.median():.3f}{reliable_text}{tradeoff}."
                )
        lines.extend(
            [
                "",
                (
                    "Total SE combines conditional ability uncertainty and item-parameter "
                    "uncertainty in quadrature; it is not the same as the CAT stopping SE."
                ),
            ]
        )
    generated_figures = {
        key: value
        for key, value in (figures or {}).items()
        if isinstance(value, dict) and value.get("status") == "generated"
    }
    if generated_figures:
        lines.extend(["", "## Figures", ""])
        for key, value in generated_figures.items():
            lines.append(f"- {key.replace('_', ' ').title()}: `{value['path']}`")
    unavailable = list(unavailable)
    if unavailable:
        lines.extend(
            [
                "",
                "## Optional studies not supplied",
                "",
                (
                    "No values were inferred for these sections: "
                    + ", ".join(sorted(unavailable))
                    + "."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            (
                "This report compares the supplied candidate structures; it does not prove "
                "that an untested structure is worse. Fold uncertainty is based on a small "
                "number of folds, while CAT confidence intervals resample tutor models. "
                "When the EAP reference uses a coarse quadrature grid, recovery against that "
                "reference and CAT-setting choices based on it remain provisional. The CAT "
                "replay also reuses the calibration cohort unless an independent cohort is "
                "explicitly supplied."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def generate_key_figures(
    out_dir: Path,
    comparison: pd.DataFrame,
    cat_metrics: pd.DataFrame,
    estimator_oos: pd.DataFrame,
    parameter_uncertainty: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Write compact evidence figures when both data and matplotlib are available."""
    keys = (
        "structure_cv",
        "cat_length_precision_tradeoff",
        "estimator_recovery",
        "total_se_vs_target",
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {
            key: {"status": "unavailable", "reason": f"matplotlib unavailable: {exc}"}
            for key in keys
        }

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}

    def save(fig: Any, key: str) -> None:
        path = figure_dir / f"{key}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        results[key] = {"status": "generated", "path": str(path.relative_to(out_dir))}

    structure = comparison.dropna(subset=["cv_mean", "cv_se"])
    if len(structure):
        fig, axis = plt.subplots(figsize=(max(6.0, 0.8 * len(structure)), 3.8))
        positions = np.arange(len(structure))
        axis.errorbar(
            positions,
            structure["cv_mean"].to_numpy(float),
            yerr=structure["cv_se"].to_numpy(float),
            fmt="o",
            capsize=4,
        )
        axis.set_xticks(positions, structure["structure"], rotation=35, ha="right")
        axis.set_ylabel(f"Held-out {structure.iloc[0]['primary_metric']} (mean ± SE)")
        axis.set_title("Skill-structure cross-validation")
        save(fig, "structure_cv")
    else:
        results["structure_cv"] = {"status": "skipped", "reason": "no finite CV rows"}

    length = cat_metrics[cat_metrics.get("metric", pd.Series(dtype=str)).eq(
        "mean_scenarios_administered"
    )]
    precision = cat_metrics[cat_metrics.get("metric", pd.Series(dtype=str)).eq(
        "precision_reached_rate"
    )]
    if len(length) and len(precision):
        joined = length[["group", "run", "estimate"]].merge(
            precision[["group", "run", "estimate"]],
            on=["group", "run"],
            suffixes=("_length", "_precision"),
        )
    else:
        joined = pd.DataFrame()
    if len(joined):
        fig, axis = plt.subplots(figsize=(6.0, 4.0))
        axis.scatter(joined["estimate_length"], joined["estimate_precision"])
        for row in joined.itertuples():
            axis.annotate(row.run, (row.estimate_length, row.estimate_precision), fontsize=7)
        axis.set_xlabel("Mean scenarios administered")
        axis.set_ylabel("Precision reached rate")
        axis.set_ylim(-0.03, 1.03)
        axis.set_title("CAT length–precision tradeoff")
        save(fig, "cat_length_precision_tradeoff")
    else:
        results["cat_length_precision_tradeoff"] = {
            "status": "skipped",
            "reason": "CAT length and precision metrics were not both supplied",
        }

    recovery_columns = {
        "skill",
        "metric",
        "estimate",
        "ci_lower",
        "ci_upper",
        "estimator",
    }
    if recovery_columns <= set(estimator_oos):
        recovery = estimator_oos[
            (estimator_oos["skill"] == "__macro__")
            & (estimator_oos["metric"] == "r")
        ].dropna(subset=["estimate", "ci_lower", "ci_upper"])
    else:
        recovery = pd.DataFrame()
    if len(recovery):
        fig, axis = plt.subplots(figsize=(6.0, 3.8))
        positions = np.arange(len(recovery))
        estimate = recovery["estimate"].to_numpy(float)
        low = recovery["ci_lower"].to_numpy(float)
        high = recovery["ci_upper"].to_numpy(float)
        axis.errorbar(
            positions,
            estimate,
            yerr=np.vstack((estimate - low, high - estimate)),
            fmt="o",
            capsize=4,
        )
        axis.set_xticks(positions, recovery["estimator"])
        axis.set_ylabel("Held-out recovery correlation r")
        axis.set_ylim(-1.05, 1.05)
        axis.set_title("Estimator recovery with 95% model-bootstrap CIs")
        save(fig, "estimator_recovery")
    else:
        results["estimator_recovery"] = {
            "status": "skipped",
            "reason": "held-out estimator recovery was not supplied",
        }

    if {"se_target", "median_se_total", "estimator"} <= set(parameter_uncertainty):
        total = parameter_uncertainty.dropna(subset=["se_target", "median_se_total"])
    else:
        total = pd.DataFrame()
    if len(total):
        collapsed = (
            total.groupby(["se_target", "estimator"], as_index=False)["median_se_total"]
            .median()
            .sort_values("se_target")
        )
        fig, axis = plt.subplots(figsize=(6.0, 3.8))
        for estimator, group in collapsed.groupby("estimator", sort=True):
            axis.plot(
                group["se_target"],
                group["median_se_total"],
                marker="o",
                label=str(estimator),
            )
        axis.set_xlabel("Requested CAT SE target")
        axis.set_ylabel("Median total SE")
        axis.set_title("Total uncertainty versus stopping target")
        axis.legend()
        save(fig, "total_se_vs_target")
    else:
        results["total_se_vs_target"] = {
            "status": "skipped",
            "reason": "no parameter-uncertainty rows with an SE target were supplied",
        }
    return results


def write_study_outputs(
    out_dir: Path,
    comparison: pd.DataFrame,
    selection: dict,
    cat_metrics: pd.DataFrame,
    cat_differences: pd.DataFrame,
    cat_metadata: dict,
    *,
    estimator_oos: pd.DataFrame | None = None,
    estimator_oos_metadata: dict | None = None,
    sensitivity: pd.DataFrame | None = None,
    sensitivity_metadata: dict | None = None,
    order_spread: pd.DataFrame | None = None,
    order_metadata: dict | None = None,
    parameter_uncertainty: pd.DataFrame | None = None,
    parameter_uncertainty_metadata: dict | None = None,
    make_figures: bool = True,
) -> dict[str, Path]:
    """Write deterministic study tables plus JSON/Markdown reports."""
    estimator_oos = estimator_oos if estimator_oos is not None else pd.DataFrame()
    estimator_oos_metadata = estimator_oos_metadata or {"status": "not_supplied"}
    sensitivity = sensitivity if sensitivity is not None else pd.DataFrame()
    sensitivity_metadata = sensitivity_metadata or {"status": "not_supplied"}
    order_spread = order_spread if order_spread is not None else pd.DataFrame()
    order_metadata = order_metadata or {"status": "not_supplied"}
    parameter_uncertainty = (
        parameter_uncertainty if parameter_uncertainty is not None else pd.DataFrame()
    )
    parameter_uncertainty_metadata = parameter_uncertainty_metadata or {
        "status": "not_supplied"
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "structure_comparison": out_dir / "structure_comparison.csv",
        "selection": out_dir / "selection.json",
        "summary": out_dir / "CALIBRATION_STUDY_SUMMARY.md",
        "study_results": out_dir / "study_results.json",
        "cat_metrics": out_dir / "cat_sweep_metrics.csv",
        "cat_differences": out_dir / "cat_sweep_paired_differences.csv",
        "cat_bootstrap": out_dir / "cat_sweep_bootstrap.json",
        "estimator_oos_metrics": out_dir / "estimator_oos_metrics.csv",
        "sensitivity_summary": out_dir / "sensitivity_summary.csv",
        "order_stability_summary": out_dir / "order_stability_summary.csv",
        "parameter_uncertainty_summary": out_dir / "parameter_uncertainty_summary.csv",
    }
    comparison.to_csv(outputs["structure_comparison"], index=False, quoting=csv.QUOTE_MINIMAL)
    _atomic_write(
        outputs["selection"],
        (json.dumps(selection, indent=2, ensure_ascii=False) + "\n").encode(),
    )
    cat_metrics.to_csv(outputs["cat_metrics"], index=False)
    cat_differences.to_csv(outputs["cat_differences"], index=False)
    estimator_oos.to_csv(outputs["estimator_oos_metrics"], index=False)
    sensitivity.to_csv(outputs["sensitivity_summary"], index=False)
    order_spread.to_csv(outputs["order_stability_summary"], index=False)
    parameter_uncertainty.to_csv(outputs["parameter_uncertainty_summary"], index=False)
    _atomic_write(
        outputs["cat_bootstrap"],
        (json.dumps(cat_metadata, indent=2, ensure_ascii=False) + "\n").encode(),
    )
    if make_figures:
        figures = generate_key_figures(
            out_dir,
            comparison,
            cat_metrics,
            estimator_oos,
            parameter_uncertainty,
        )
    else:
        figures = {
            key: {"status": "disabled", "reason": "figure generation disabled"}
            for key in (
                "structure_cv",
                "cat_length_precision_tradeoff",
                "estimator_recovery",
                "total_se_vs_target",
            )
        }
    optional_statuses = {
        "estimator_oos_kfold": estimator_oos_metadata.get("status"),
        "sensitivity": sensitivity_metadata.get("status"),
        "order_stability": order_metadata.get("status"),
        "parameter_uncertainty": parameter_uncertainty_metadata.get("status"),
    }
    if estimator_oos_metadata.get("status") == "available":
        optional_statuses["estimator_oos_pass_rate_calibration"] = (
            (estimator_oos_metadata.get("pass_rate_calibration") or {}).get("status")
        )
    unavailable = [name for name, status in optional_statuses.items() if status != "available"]
    consolidated = {
        "schema_version": "calibration-study-results-v2",
        "generated_at": _utcnow(),
        "selection": selection,
        "structures": _json_ready(comparison.to_dict(orient="records")),
        "sensitivity": {
            "metadata": sensitivity_metadata,
            "rows": _json_ready(sensitivity.to_dict(orient="records")),
        },
        "estimator_oos_kfold": {
            "metadata": estimator_oos_metadata,
            "metrics": _json_ready(estimator_oos.to_dict(orient="records")),
        },
        "cat_sweeps": {
            "metadata": cat_metadata,
            "metrics": _json_ready(cat_metrics.to_dict(orient="records")),
            "paired_differences": _json_ready(cat_differences.to_dict(orient="records")),
        },
        "order_stability": {
            "metadata": order_metadata,
            "rows": _json_ready(order_spread.to_dict(orient="records")),
        },
        "parameter_uncertainty": {
            "metadata": parameter_uncertainty_metadata,
            "rows": _json_ready(parameter_uncertainty.to_dict(orient="records")),
        },
        "figures": figures,
        "optional_studies_not_supplied": unavailable,
    }
    _atomic_write(
        outputs["study_results"],
        (json.dumps(_json_ready(consolidated), indent=2, ensure_ascii=False) + "\n").encode(),
    )
    summary = render_plain_language_summary(
        comparison,
        selection,
        cat_metrics,
        cat_differences,
        cat_metadata=cat_metadata,
        estimator_oos=estimator_oos,
        sensitivity=sensitivity,
        order_spread=order_spread,
        order_metadata=order_metadata,
        parameter_uncertainty=parameter_uncertainty,
        figures=figures,
        unavailable=unavailable,
    )
    _atomic_write(outputs["summary"], summary.encode("utf-8"))
    return outputs


def run_study(
    structure_specs: Iterable[StructureSpec],
    cat_specs: Iterable[CatSweepSpec],
    out_dir: Path,
    *,
    primary_metric: str = "log_loss",
    direction: str | None = None,
    override: str | None = None,
    bootstrap_b: int = 2000,
    bootstrap_seed: int = 0,
    estimator_oos_spec: EstimatorOOSSpec | None = None,
    order_stability_spec: OrderStabilitySpec | None = None,
    parameter_uncertainty_specs: Iterable[ParameterUncertaintySpec] = (),
    sensitivity_specs: Iterable[SensitivitySpec] = (),
    make_figures: bool = True,
) -> dict:
    comparison, selection = compare_structures(
        structure_specs,
        primary_metric=primary_metric,
        direction=direction,
        override=override,
    )
    cat_metrics, cat_differences, cat_metadata = aggregate_cat_sweeps(
        cat_specs, b=bootstrap_b, seed=bootstrap_seed
    )
    estimator_oos, estimator_oos_metadata = aggregate_estimator_oos(
        estimator_oos_spec, b=bootstrap_b, seed=bootstrap_seed
    )
    sensitivity, sensitivity_metadata = aggregate_sensitivity(
        sensitivity_specs, primary_metric=primary_metric
    )
    order_spread, order_metadata = load_order_stability(order_stability_spec)
    parameter_uncertainty, parameter_uncertainty_metadata = (
        aggregate_parameter_uncertainty(parameter_uncertainty_specs)
    )
    outputs = write_study_outputs(
        out_dir,
        comparison,
        selection,
        cat_metrics,
        cat_differences,
        cat_metadata,
        estimator_oos=estimator_oos,
        estimator_oos_metadata=estimator_oos_metadata,
        sensitivity=sensitivity,
        sensitivity_metadata=sensitivity_metadata,
        order_spread=order_spread,
        order_metadata=order_metadata,
        parameter_uncertainty=parameter_uncertainty,
        parameter_uncertainty_metadata=parameter_uncertainty_metadata,
        make_figures=make_figures,
    )
    return {
        "comparison": comparison,
        "selection": selection,
        "cat_metrics": cat_metrics,
        "cat_differences": cat_differences,
        "cat_metadata": cat_metadata,
        "estimator_oos": estimator_oos,
        "estimator_oos_metadata": estimator_oos_metadata,
        "sensitivity": sensitivity,
        "sensitivity_metadata": sensitivity_metadata,
        "order_spread": order_spread,
        "order_metadata": order_metadata,
        "parameter_uncertainty": parameter_uncertainty,
        "parameter_uncertainty_metadata": parameter_uncertainty_metadata,
        "outputs": outputs,
    }


def _resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else base / path


def load_study_spec(path: Path) -> tuple[list[StructureSpec], list[CatSweepSpec]]:
    """Load the original required structure/CAT sections (backward-compatible API)."""
    payload = _read_json(path, "study spec")
    base = path.parent
    structures = []
    for row in payload.get("structures") or []:
        structures.append(
            StructureSpec(
                name=str(row["name"]),
                core_manifest=_resolve_path(base, str(row["core_manifest"])),
                kfold_dir=_resolve_path(base, str(row["kfold_dir"])),
                n_dims=(int(row["n_dims"]) if row.get("n_dims") is not None else None),
            )
        )
    cats = []
    for row in payload.get("cat_sweeps") or []:
        prefixes = row.get("estimate_prefixes") or []
        skills = row.get("skills") or []
        cats.append(
            CatSweepSpec(
                name=str(row["name"]),
                csv_path=_resolve_path(base, str(row["csv"])),
                group=str(row.get("group") or "default"),
                baseline=bool(row.get("baseline", False)),
                reference_prefix=row.get("reference_prefix"),
                estimate_prefixes=tuple(str(value) for value in prefixes),
                skills=tuple(str(value) for value in skills),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return structures, cats


def load_extended_study_spec(path: Path) -> StudyInputs:
    """Load required inputs plus every optional consolidated-report artifact."""
    structures, cats = load_study_spec(path)
    payload = _read_json(path, "study spec")
    base = path.parent

    estimator_raw = payload.get("estimator_oos_kfold")
    estimator = None
    if estimator_raw:
        if not isinstance(estimator_raw, dict) or "dir" not in estimator_raw:
            raise StudyError("estimator_oos_kfold must be an object containing dir")
        estimator = EstimatorOOSSpec(
            dir_path=_resolve_path(base, str(estimator_raw["dir"])),
            name=str(estimator_raw.get("name") or "estimator_oos_kfold"),
        )

    order_raw = payload.get("order_stability")
    order = None
    if order_raw:
        if not isinstance(order_raw, dict) or "dir" not in order_raw:
            raise StudyError("order_stability must be an object containing dir")
        order = OrderStabilitySpec(dir_path=_resolve_path(base, str(order_raw["dir"])))

    uncertainty_raw = payload.get("parameter_uncertainty") or []
    if isinstance(uncertainty_raw, dict):
        uncertainty_raw = [uncertainty_raw]
    if not isinstance(uncertainty_raw, list):
        raise StudyError("parameter_uncertainty must be an object or list of objects")
    uncertainty: list[ParameterUncertaintySpec] = []
    for row in uncertainty_raw:
        if not isinstance(row, dict) or "dir" not in row:
            raise StudyError("each parameter_uncertainty entry must contain dir")
        uncertainty.append(
            ParameterUncertaintySpec(
                name=str(row.get("name") or Path(str(row["dir"])).name),
                dir_path=_resolve_path(base, str(row["dir"])),
                se_target=(
                    float(row["se_target"]) if row.get("se_target") is not None else None
                ),
            )
        )

    sensitivity_raw = payload.get("sensitivity") or []
    if not isinstance(sensitivity_raw, list):
        raise StudyError("sensitivity must be a list of objects")
    sensitivity: list[SensitivitySpec] = []
    for row in sensitivity_raw:
        if not isinstance(row, dict):
            raise StudyError("each sensitivity entry must be an object")
        name = str(row.get("name") or "").strip()
        if not name:
            raise StudyError("each sensitivity entry needs a nonblank name")
        if row.get("dir") is not None:
            run_dir = _resolve_path(base, str(row["dir"]))
            core_manifest = run_dir / "fit" / "calibration_mirt_manifest.json"
            kfold_dir = run_dir / "kfold"
        elif row.get("core_manifest") is not None and row.get("kfold_dir") is not None:
            core_manifest = _resolve_path(base, str(row["core_manifest"]))
            kfold_dir = _resolve_path(base, str(row["kfold_dir"]))
        else:
            raise StudyError(
                "each sensitivity entry needs dir or both core_manifest and kfold_dir"
            )
        sensitivity.append(
            SensitivitySpec(
                name=name,
                core_manifest=core_manifest,
                kfold_dir=kfold_dir,
                grid=int(row["grid"]) if row.get("grid") is not None else None,
                ridge=float(row["ridge"]) if row.get("ridge") is not None else None,
            )
        )
    return StudyInputs(
        structures=tuple(structures),
        cat_sweeps=tuple(cats),
        estimator_oos=estimator,
        order_stability=order,
        parameter_uncertainty=tuple(uncertainty),
        sensitivity=tuple(sensitivity),
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--primary-metric", default="log_loss")
    parser.add_argument("--direction", choices=("minimize", "maximize"), default=None)
    parser.add_argument("--override-structure", default=None)
    parser.add_argument("--bootstrap-b", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        study = load_extended_study_spec(args.spec)
        result = run_study(
            study.structures,
            study.cat_sweeps,
            args.out_dir,
            primary_metric=args.primary_metric,
            direction=args.direction,
            override=args.override_structure,
            bootstrap_b=args.bootstrap_b,
            bootstrap_seed=args.bootstrap_seed,
            estimator_oos_spec=study.estimator_oos,
            order_stability_spec=study.order_stability,
            parameter_uncertainty_specs=study.parameter_uncertainty,
            sensitivity_specs=study.sensitivity,
            make_figures=not args.no_figures,
        )
    except (StudyError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    selection = result["selection"]
    print(f"selected structure: {selection['selected_structure']}")
    print(f"automatic recommendation: {selection['automatic_recommendation']}")
    for name, path in result["outputs"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
