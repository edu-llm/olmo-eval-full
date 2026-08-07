#!/usr/bin/env python3
"""Resolve InFoBench post-fit quadrature and lock a numerical scoring rule.

This is an append-only follow-up to the failed dense-EAP study.  It never
refits the already locked grid-61/ridge-0.1 bank.  Instead, it hash-verifies
and imports the historical fits, scores them with explicitly named numerical
integration methods, and opens the ridge-0.001/0.01 stages only after every
frozen numerical gate passes.

The expensive study is opt-in.  ``--plan-only`` validates provenance and
prints the schedule without creating a run directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs" / "infobench_quadrature_resolution_v1.json"
RUNNER = Path(__file__).resolve()
SCENARIO_LIB = ROOT / "scripts" / "scenario_cat_lib.py"
NESTED_RUNNER = ROOT / "scripts" / "nested_scenario_cat_cv.py"
DIMENSION = "instruction_following"
DIAGNOSTIC_CAT = {
    "seed": 42,
    "top_n": 5,
    "max_se": 0.25,
    "min_evals_per_skill": 15,
    "min_scenarios": 0,
    "max_scenarios": 50,
    "selection": "trace",
    "mwle_ridge": 1e-6,
}


class QuadratureResolutionError(RuntimeError):
    """Fail-closed error for a quadrature-resolution study."""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash float64 C-order raw bytes, as required by the downstream lock."""

    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QuadratureResolutionError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QuadratureResolutionError(f"expected one JSON object in {path}")
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise QuadratureResolutionError(f"{label} is missing: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise QuadratureResolutionError(
            f"{label} hash mismatch: expected {expected}, observed {observed}: {path}"
        )


def _safe_relative(path: Path, parent: Path) -> Path | None:
    try:
        return path.resolve().relative_to(parent.resolve())
    except ValueError:
        return None


def portable_artifact_relative(recorded: str | Path, parent_run: Path) -> Path:
    """Recover a run-relative artifact path even after a repository move."""

    recorded_path = Path(recorded)
    direct = _safe_relative(recorded_path, parent_run)
    if direct is not None:
        return direct
    parts = recorded_path.parts
    matches = [index for index, part in enumerate(parts) if part == parent_run.name]
    if matches:
        candidate = Path(*parts[matches[-1] + 1 :])
    elif not recorded_path.is_absolute():
        candidate = recorded_path
        configured = Path(*parent_run.parts[-3:])
        with suppress(ValueError):
            candidate = candidate.relative_to(configured)
    else:
        raise QuadratureResolutionError(
            f"cannot recover a portable path for {recorded_path} under {parent_run}"
        )
    if not candidate.parts or any(part == ".." for part in candidate.parts):
        raise QuadratureResolutionError(f"unsafe portable artifact path: {candidate}")
    return candidate


def _load_verified_module(name: str, path: Path, expected_hash: str):
    _verify_hash(path, expected_hash, f"verified runner {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise QuadratureResolutionError(f"could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True)
class ResolutionContext:
    config_path: Path
    raw: dict[str, Any]
    base: Any
    parent_module: Any
    dense: Any
    parent_run: Path
    dense_parent_run: Path
    out_dir: Path
    protected_paths: tuple[Path, ...]

    @property
    def split_path(self) -> Path:
        return self.base.split_path

    @property
    def matrix(self) -> Path:
        return self.base.matrix

    @property
    def rubrics(self) -> Path:
        return self.base.rubrics

    @property
    def scenarios(self) -> Path:
        return self.base.scenarios


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise QuadratureResolutionError(f"{label} differs from the frozen design")


def _discover_historical_paths(config_path: Path) -> set[Path]:
    """Find every run/report directory in the immutable parent-config chain."""

    discovered: set[Path] = set()
    seen: set[Path] = set()
    current: Path | None = config_path
    while current is not None:
        current = current.resolve()
        if current in seen or not current.is_file():
            break
        seen.add(current)
        payload = _read_json(current)
        for key in ("output_dir",):
            if payload.get(key):
                discovered.add(_resolve(payload[key]))
        for section_name in ("parent_study", "baseline"):
            section = payload.get(section_name) or {}
            for key in ("run_dir", "report_dir"):
                if section.get(key):
                    discovered.add(_resolve(section[key]))
        parent = payload.get("parent_study") or {}
        current = _resolve(parent["config"]) if parent.get("config") else None
    return discovered


def _assert_safe_output(context: ResolutionContext, out_dir: Path) -> None:
    resolved = out_dir.resolve()
    for protected in context.protected_paths:
        historical = protected.resolve()
        if (
            resolved == historical
            or _safe_relative(resolved, historical) is not None
            or _safe_relative(historical, resolved) is not None
        ):
            raise QuadratureResolutionError(
                f"output {resolved} overlaps historical parent path {historical}"
            )


def _validate_config(raw: Mapping[str, Any]) -> None:
    _require_exact(raw.get("schema_version"), "infobench-quadrature-resolution-v1", "schema")
    _require_exact(
        raw.get("status"),
        "frozen_before_official_run_design_informed_by_diagnostics",
        "design status",
    )
    _require_exact(raw.get("benchmark"), "InFoBench", "benchmark")
    _require_exact(int(raw.get("fit_grid", -1)), 61, "fit grid")
    if not math.isclose(float(raw.get("initial_ridge", -1)), 0.1):
        raise QuadratureResolutionError("initial ridge must remain 0.1")
    support = raw.get("fit_grid_common_support_recheck") or {}
    _require_exact(support.get("comparisons"), ["41_vs_61", "61_vs_81"], "fit comparisons")
    _require_exact(support.get("locked_comparison"), "61_vs_81", "locked comparison")
    _require_exact(
        support.get("evaluation_quadrature_method"), "normal_trapezoid", "support method"
    )
    _require_exact(int(support.get("evaluation_nodes", -1)), 401, "support nodes")
    if not math.isclose(float(support.get("linear_bound", -1)), 8.0):
        raise QuadratureResolutionError("support bound must remain 8")
    _require_exact(
        support.get("common_cell_key"),
        ["outer_fold", "model", "criterion_id"],
        "common-cell key",
    )
    if not (
        support.get("require_identical_keys_and_labels") is True
        and support.get("require_absolute_pooled_shift_within_margin") is True
        and support.get("require_family_cluster_bootstrap_ci_within_margin") is True
        and math.isclose(float(support.get("equivalence_margin", -1)), 0.005)
        and int(support.get("bootstrap_replicates", 0)) == 5000
    ):
        raise QuadratureResolutionError("common-support equivalence policy changed")
    primary = raw.get("primary_eap") or {}
    _require_exact(primary.get("quadrature_method"), "normal_trapezoid", "primary method")
    _require_exact(primary.get("grid_candidates"), [401, 801, 1601], "primary nodes")
    _require_exact(
        primary.get("successive_comparisons"),
        ["401_vs_801", "801_vs_1601"],
        "primary comparisons",
    )
    _require_exact(primary.get("endpoint_comparison"), "401_vs_1601", "endpoint")
    if not math.isclose(float(primary.get("linear_bound", -1)), 8.0):
        raise QuadratureResolutionError("primary bound must remain 8")
    thresholds = primary.get("stability_thresholds") or {}
    expected_thresholds = {
        "maximum_median_absolute_theta_shift": 0.02,
        "maximum_p95_absolute_theta_shift": 0.05,
        "maximum_absolute_theta_shift": 0.005,
        "maximum_recovery_correlation_shift": 0.01,
        "maximum_recovery_slope_shift": 0.02,
        "maximum_theta_mae_shift": 0.02,
        "maximum_pass_rate_mae_shift": 0.005,
        "require_identical_fixed_bank_orders": True,
    }
    _require_exact(thresholds, expected_thresholds, "primary stability thresholds")
    bound = raw.get("bound_check") or {}
    required_bound = {
        "reference_method": "normal_trapezoid",
        "reference_nodes": 1601,
        "reference_bound": 8.0,
        "comparison_nodes": 2001,
        "comparison_bound": 10.0,
        "shared_step": 0.01,
        "maximum_full_theta_shift": 0.005,
        "maximum_heldout_theta_shift": 0.005,
        "tail_region_absolute_theta_at_least": 7.5,
        "maximum_posterior_tail_mass": 0.00001,
    }
    _require_exact(bound, required_bound, "bound check")
    cross = raw.get("cross_family_check") or {}
    required_cross = {
        "reference_method": "normal_trapezoid",
        "reference_nodes": 1601,
        "comparison_method": "gauss_hermite_scipy",
        "comparison_nodes": 1601,
        "maximum_full_theta_shift": 0.005,
        "maximum_heldout_theta_shift": 0.005,
    }
    _require_exact(cross, required_cross, "cross-family check")
    _require_exact(
        raw.get("ridge_candidates_after_numerical_lock"),
        [0.001, 0.01, 0.1],
        "ridge candidates",
    )
    if not str(raw.get("threshold_change_policy", "")).startswith("No gate may be relaxed"):
        raise QuadratureResolutionError("threshold-relaxation prohibition is missing")


def load_context(config_path: Path = DEFAULT_CONFIG) -> ResolutionContext:
    config_path = config_path.resolve()
    raw = _read_json(config_path)
    _validate_config(raw)

    parent = raw.get("parent_study") or {}
    parent_config = _resolve(parent["config"])
    parent_runner = _resolve(parent["runner"])
    _verify_hash(parent_config, str(parent["config_sha256"]), "parent config")
    parent_module = _load_verified_module(
        "infobench_verified_eap_parent", parent_runner, str(parent["runner_sha256"])
    )
    base, _parent_raw, parent_run, fit_gate = parent_module.load_context(parent_config)
    parent_run = parent_run.resolve()
    configured_parent_run = _resolve(parent["run_dir"])
    if (
        configured_parent_run
        != (ROOT / "runs/calibration/InFoBench_remediation_v1_eap_followup").resolve()
    ):
        raise QuadratureResolutionError("unexpected quadrature parent run")
    _verify_hash(
        _resolve(parent["study_manifest"]),
        str(parent["study_manifest_sha256"]),
        "parent study manifest",
    )
    parent_manifest = _read_json(_resolve(parent["study_manifest"]))
    _require_exact(parent_manifest.get("status"), parent.get("required_status"), "parent status")
    _verify_hash(
        _resolve(parent["imported_fit_manifest"]),
        str(parent["imported_fit_manifest_sha256"]),
        "parent imported-fit manifest",
    )
    _verify_hash(_resolve(parent["fit_gate"]), str(parent["fit_gate_sha256"]), "fit gate")
    if not (fit_gate.get("passed") is True and int(fit_gate.get("locked_fit_grid", -1)) == 61):
        raise QuadratureResolutionError("parent fit gate does not lock grid 61")
    _verify_hash(
        _resolve(parent["fit_stage_marker"]),
        str(parent["fit_stage_marker_sha256"]),
        "grid-61 fit stage marker",
    )
    stage_marker = _read_json(_resolve(parent["fit_stage_marker"]))
    if stage_marker.get("stage") != "fit_grid_061_ridge_0p1" or not stage_marker.get(
        "output_sha256"
    ):
        raise QuadratureResolutionError("grid-61 parent marker is incomplete")
    for path_key, hash_key, label in (
        (
            "grid_41_import_manifest",
            "grid_41_import_manifest_sha256",
            "grid-41 import manifest",
        ),
        (
            "grid_41_source_stage_marker",
            "grid_41_source_stage_marker_sha256",
            "grid-41 source stage marker",
        ),
        (
            "grid_81_stage_marker",
            "grid_81_stage_marker_sha256",
            "grid-81 stage marker",
        ),
    ):
        _verify_hash(_resolve(parent[path_key]), str(parent[hash_key]), label)

    frozen = raw.get("frozen_inputs") or {}
    for key in ("response_matrix", "rubrics", "scenarios", "split_manifest"):
        path = _resolve(frozen[key])
        _verify_hash(path, str(frozen[f"{key}_sha256"]), f"frozen {key}")
    if base.matrix.resolve() != _resolve(frozen["response_matrix"]):
        raise QuadratureResolutionError("parent context matrix differs from frozen input")
    if base.rubrics.resolve() != _resolve(frozen["rubrics"]):
        raise QuadratureResolutionError("parent context rubrics differ from frozen input")
    if base.scenarios.resolve() != _resolve(frozen["scenarios"]):
        raise QuadratureResolutionError("parent context scenarios differ from frozen input")
    if base.split_path.resolve() != _resolve(frozen["split_manifest"]):
        raise QuadratureResolutionError("parent context splits differ from frozen input")

    dense = parent_module.dense
    merged = copy.deepcopy(base.config)
    merged["quadrature_resolution"] = copy.deepcopy(raw)
    merged["immutable_decisions"]["new_output_dir"] = str(raw["output_dir"])
    base = replace(
        base,
        config_path=config_path,
        config=merged,
        fit_grids=(41, 61, 81),
        eap_grids=(401, 801, 1601, 2001),
        ridge_candidates=(0.001, 0.01, 0.1),
        initial_ridge=0.1,
        default_out_dir=_resolve(raw["output_dir"]),
    )
    dense_parent_run = _resolve(parent["fit_gate"]).parent.parent
    protected = _discover_historical_paths(parent_config)
    protected.update({configured_parent_run, dense_parent_run})
    context = ResolutionContext(
        config_path=config_path,
        raw=raw,
        base=base,
        parent_module=parent_module,
        dense=dense,
        parent_run=configured_parent_run,
        dense_parent_run=dense_parent_run,
        out_dir=_resolve(raw["output_dir"]),
        protected_paths=tuple(sorted(protected)),
    )
    _assert_safe_output(context, context.out_dir)
    return context


def _fit_files(grid: int, folds: Sequence[Mapping[str, Any]]) -> list[Path]:
    root = Path("fit_cache") / f"grid_{grid:03d}" / "ridge_0p1"
    paths = [root / "fit_grid_manifest.json"]
    scopes = ["full"] + [f"outer_{int(record['outer_fold'])}" for record in folds]
    for scope in scopes:
        paths.extend(
            [
                root / scope / "fit.npz",
                root / scope / "fit_manifest.json",
                root / scope / "item_params.csv",
            ]
        )
    return paths


def _hashes_from_marker(marker_path: Path, parent_run: Path) -> dict[Path, str]:
    marker = _read_json(marker_path)
    hashes = marker.get("output_sha256") or {}
    if not hashes:
        raise QuadratureResolutionError(f"stage marker has no output hashes: {marker_path}")
    result: dict[Path, str] = {}
    for recorded, expected in hashes.items():
        relative = portable_artifact_relative(recorded, parent_run)
        source = parent_run / relative
        _verify_hash(source, str(expected), "parent stage output")
        result[relative] = str(expected)
    return result


def _hashes_from_import_manifest(manifest_path: Path, parent_run: Path) -> dict[Path, str]:
    manifest = _read_json(manifest_path)
    result: dict[Path, str] = {}
    records = manifest.get("artifacts") or []
    if not records:
        raise QuadratureResolutionError(f"import manifest is empty: {manifest_path}")
    for record in records:
        destination = record.get("destination")
        if destination == "provenance_only_not_copied":
            source_text = record.get("source")
            if source_text:
                source = Path(source_text)
                if not source.is_file():
                    # Recover through the historical run name when the repo moved.
                    candidates = list(ROOT.glob(f"runs/calibration/**/{source.name}"))
                    source = candidates[0] if len(candidates) == 1 else source
                _verify_hash(source, str(record["source_sha256"]), "provenance marker")
            continue
        relative = portable_artifact_relative(destination, parent_run)
        path = parent_run / relative
        expected = str(record.get("destination_sha256"))
        _verify_hash(path, expected, "imported parent artifact")
        result[relative] = expected
    return result


def _validated_parent_fit_hashes(context: ResolutionContext, grid: int) -> dict[Path, str]:
    expected_paths = set(_fit_files(grid, context.base.splits["outer_folds"]))
    if grid == 61:
        imported_path = _resolve(context.raw["parent_study"]["imported_fit_manifest"])
        imported = _hashes_from_import_manifest(imported_path, context.parent_run)
        marker_path = _resolve(context.raw["parent_study"]["fit_stage_marker"])
        marker_hashes = _hashes_from_marker(marker_path, context.dense_parent_run)
        hashes = {path: imported[path] for path in expected_paths if path in imported}
        for relative, observed in hashes.items():
            if marker_hashes.get(relative) != observed:
                raise QuadratureResolutionError(
                    f"grid-61 imported hash disagrees with frozen stage marker: {relative}"
                )
    elif grid == 41:
        parent = context.raw["parent_study"]
        manifest_path = _resolve(parent["grid_41_import_manifest"])
        hashes = _hashes_from_import_manifest(manifest_path, context.dense_parent_run)
        hashes = {path: hashes[path] for path in expected_paths if path in hashes}
        source_marker = _resolve(parent["grid_41_source_stage_marker"])
        source_run = source_marker.parent.parent
        source_hashes = _hashes_from_marker(source_marker, source_run)
        for relative, observed in hashes.items():
            if source_hashes.get(relative) != observed:
                raise QuadratureResolutionError(
                    f"grid-41 imported hash disagrees with frozen source marker: {relative}"
                )
    elif grid == 81:
        marker_path = _resolve(context.raw["parent_study"]["grid_81_stage_marker"])
        hashes = _hashes_from_marker(marker_path, context.dense_parent_run)
        hashes = {path: hashes[path] for path in expected_paths if path in hashes}
    else:
        raise QuadratureResolutionError(f"unexpected imported fit grid {grid}")
    missing = sorted(map(str, expected_paths - set(hashes)))
    if missing:
        raise QuadratureResolutionError(
            f"parent provenance is missing grid-{grid} fit files: {missing}"
        )
    return hashes


def _copy_parent_fits(context: ResolutionContext, out_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for grid in (41, 61, 81):
        source_run = context.parent_run if grid == 61 else context.dense_parent_run
        hashes = _validated_parent_fit_hashes(context, grid)
        for relative in sorted(hashes, key=str):
            source = source_run / relative
            destination = out_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.importing")
            shutil.copy2(source, temporary)
            if _sha256(temporary) != hashes[relative]:
                raise QuadratureResolutionError(f"copied artifact hash mismatch: {relative}")
            temporary.replace(destination)
            records.append(
                {
                    "fit_grid": grid,
                    "source_run_relative_to_repo": str(source_run.relative_to(ROOT)),
                    "source_path_relative_to_run": str(relative),
                    "destination_path_relative_to_study": str(relative),
                    "sha256": hashes[relative],
                }
            )
    payload = {
        "imported_at": _utcnow(),
        "policy": "portable run-relative imports; grid-61/ridge-0.1 was not refitted",
        "artifacts": records,
    }
    _write_json(out_dir / "parent_fit_imports.json", payload)
    return payload


def _verify_imported_fits(out_dir: Path) -> None:
    manifest_path = out_dir / "parent_fit_imports.json"
    manifest = _read_json(manifest_path)
    for record in manifest.get("artifacts") or []:
        relative = Path(record["destination_path_relative_to_study"])
        if relative.is_absolute() or any(part == ".." for part in relative.parts):
            raise QuadratureResolutionError(f"unsafe imported destination: {relative}")
        _verify_hash(out_dir / relative, str(record["sha256"]), "resumed imported fit")


def _slug_number(value: float | int | None) -> str:
    if value is None:
        return "none"
    return str(value).replace("-", "m").replace(".", "p")


def score_dir(
    out_dir: Path,
    *,
    method: str,
    nodes: int,
    bound: float | None,
    fit_grid: int,
    ridge: float,
) -> Path:
    return (
        out_dir
        / "scores"
        / f"method_{method}"
        / f"bound_{_slug_number(bound)}"
        / f"fit_grid_{fit_grid:03d}"
        / f"nodes_{nodes:04d}"
        / f"ridge_{_slug_number(ridge)}"
    )


def _score_outputs(path: Path, include_cat: bool) -> list[Path]:
    outputs = [
        path / "per_model.csv",
        path / "heldout_cells.csv",
        path / "oos_metrics.csv",
        path / "score_manifest.json",
    ]
    if include_cat:
        outputs.extend([path / "recovery.csv", path / "pass_rate.csv"])
    return outputs


def _quadrature(context: ResolutionContext, method: str, nodes: int, bound: float | None):
    if method == "normal_trapezoid" and bound is None:
        raise QuadratureResolutionError("normal_trapezoid requires an explicit bound")
    return context.dense.scat.build_quadrature(
        1,
        nodes,
        np.eye(1),
        max_nodes=max(5000, nodes),
        method=method,
        linear_bound=8.0 if bound is None else float(bound),
    )


def _posterior_batch(
    raw: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    quadrature: Any,
    item_indices: np.ndarray,
    tail_region: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(item_indices, dtype=int)
    values = np.asarray(raw[:, selected], dtype=float)
    observed = np.isfinite(values)
    successes = np.where(observed, np.nan_to_num(values, nan=0.0), 0.0)
    failures = np.where(observed, 1.0 - np.nan_to_num(values, nan=0.0), 0.0)
    eta = A[selected] @ quadrature.grid.T - b[selected, None]
    likelihood = successes @ log_expit(eta) + failures @ log_expit(-eta)
    joint = likelihood + quadrature.log_prior[None, :]
    posterior = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    theta = posterior @ quadrature.grid
    centered = quadrature.grid[None, :, :] - theta[:, None, :]
    variance = np.sum(posterior[:, :, None] * centered * centered, axis=1)
    tail_nodes = np.abs(quadrature.grid[:, 0]) >= tail_region
    tail_mass = posterior[:, tail_nodes].sum(axis=1)
    return theta[:, 0], np.sqrt(np.clip(variance[:, 0], 0.0, None)), tail_mass


def _model_family_map(context: ResolutionContext) -> dict[str, str]:
    mapping = {
        str(model): str(family)
        for model, family in (context.base.splits.get("model_to_family") or {}).items()
    }
    models = set(map(str, context.dense.cm.load_matrix_strict(context.matrix).index))
    if set(mapping) != models:
        raise QuadratureResolutionError("frozen split manifest lacks an exact model-family map")
    return mapping


def _score_worker(
    context: ResolutionContext,
    out_dir: Path,
    *,
    method: str,
    nodes: int,
    bound: float | None,
    fit_grid: int,
    ridge: float,
    include_cat: bool,
) -> int:
    _validate_running_study(context, out_dir)
    if method not in {"normal_trapezoid", "gauss_hermite_scipy"}:
        raise QuadratureResolutionError(f"unsupported scoring method: {method}")
    if fit_grid not in {41, 61, 81}:
        raise QuadratureResolutionError("score worker fit grid is outside the frozen panel")
    allowed = {
        ("normal_trapezoid", 401, 8.0),
        ("normal_trapezoid", 801, 8.0),
        ("normal_trapezoid", 1601, 8.0),
        ("normal_trapezoid", 2001, 10.0),
        ("gauss_hermite_scipy", 1601, None),
    }
    normalized_bound = None if bound is None else float(bound)
    if (method, int(nodes), normalized_bound) not in allowed:
        raise QuadratureResolutionError("score worker numerical rule is not frozen")
    if ridge not in {0.001, 0.01, 0.1}:
        raise QuadratureResolutionError("score worker ridge is not frozen")
    if ridge != 0.1 and fit_grid != 61:
        raise QuadratureResolutionError("post-lock ridge scores must use fit grid 61")
    if ridge != 0.1:
        _verify_passed_grid_lock(context, out_dir)

    destination = score_dir(
        out_dir,
        method=method,
        nodes=nodes,
        bound=bound,
        fit_grid=fit_grid,
        ridge=ridge,
    )
    expected = _score_outputs(destination, include_cat)
    if destination.exists() or any(path.exists() for path in expected):
        raise QuadratureResolutionError(
            f"score output already exists without an orchestrator resume skip: {destination}"
        )

    matrix = context.dense.cm.load_matrix_strict(context.matrix)
    context.dense.cm.configure_skills(",".join(context.dense.SOURCE_SKILLS))
    structure = context.dense.cell_cv.build_structure(
        context.dense.SOURCE_SKILLS, context.dense.DIMENSIONS, context.dense.STRUCTURE_NAME
    )
    source_records = context.dense.estimator_cv.source_records_by_id(context.rubrics)
    scenario_records = context.dense.scat.load_scenario_records(context.scenarios)
    item_to_scenario = context.dense.cell_cv.load_item_scenarios(context.rubrics)
    scenario_split = context.base.splits["scenario_split"]
    administration = set(map(str, scenario_split["administration_scenario_ids"]))
    evaluation = set(map(str, scenario_split["evaluation_scenario_ids"]))
    admin_records = {key: value for key, value in scenario_records.items() if key in administration}
    families = _model_family_map(context)
    quadrature = _quadrature(context, method, nodes, bound)
    tail_region = float(context.raw["bound_check"]["tail_region_absolute_theta_at_least"])

    model_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    for outer in context.base.splits["outer_folds"]:
        fold = int(outer["outer_fold"])
        test_models = list(map(str, outer["test_model_ids"]))
        fit_cache = context.dense._fit_cache_dir(out_dir, fit_grid, ridge)
        fit_path = fit_cache / f"outer_{fold}" / "fit.npz"
        fit_manifest_path = fit_cache / f"outer_{fold}" / "fit_manifest.json"
        fit_manifest = _read_json(fit_manifest_path)
        expected_fit = {
            "scope": f"outer_{fold}",
            "fit_grid": fit_grid,
            "ridge": ridge,
            "train_model_ids": list(map(str, outer["train_model_ids"])),
            "matrix_sha256": _sha256(context.matrix),
            "rubrics_sha256": _sha256(context.rubrics),
            "fold_manifest_sha256": _sha256(context.split_path),
        }
        if any(fit_manifest.get(key) != value for key, value in expected_fit.items()):
            raise QuadratureResolutionError(f"outer_{fold} fit provenance differs")
        _verify_hash(fit_path, str(fit_manifest["fit_sha256"]), f"outer_{fold} fit")
        fit = context.dense.load_cached_fit(fit_path)
        if not fit["converged"]:
            raise QuadratureResolutionError(f"outer_{fold} fit did not converge")
        bank, policy = context.dense.estimator_cv.build_fold_bank(
            fit, structure, source_records, negative_policy="drop"
        )
        raw = matrix.loc[test_models].reindex(columns=bank.criterion_ids).to_numpy(dtype=float)
        scoring_index = np.asarray(
            [
                index
                for index, criterion in enumerate(bank.criterion_ids)
                if item_to_scenario.get(criterion, criterion) not in evaluation
            ],
            dtype=int,
        )
        evaluation_index = np.asarray(
            [
                index
                for index, criterion in enumerate(bank.criterion_ids)
                if item_to_scenario.get(criterion, criterion) in evaluation
            ],
            dtype=int,
        )
        if not len(scoring_index) or not len(evaluation_index):
            raise QuadratureResolutionError(f"outer_{fold} lacks scoring/evaluation items")
        theta_full, se_full, tail_full = _posterior_batch(
            raw,
            bank.A,
            bank.b,
            quadrature,
            np.arange(len(bank.criterion_ids), dtype=int),
            tail_region,
        )
        theta_heldout, se_heldout, tail_heldout = _posterior_batch(
            raw, bank.A, bank.b, quadrature, scoring_index, tail_region
        )
        evaluation_raw = raw[:, evaluation_index]
        evaluation_observed = np.isfinite(evaluation_raw)
        probabilities = expit(
            theta_heldout[:, None] * bank.A[evaluation_index, 0][None, :]
            - bank.b[evaluation_index][None, :]
        )
        fold_labels = evaluation_raw[evaluation_observed]
        fold_probabilities = probabilities[evaluation_observed]
        fold_metric_rows.append(
            {
                "outer_fold": fold,
                **context.dense.cell_cv.metrics(fold_labels, fold_probabilities),
                "n_models": len(test_models),
                "n_cells": int(evaluation_observed.sum()),
            }
        )
        for model_index, criterion_index in np.argwhere(evaluation_observed):
            model = test_models[int(model_index)]
            criterion = bank.criterion_ids[int(evaluation_index[int(criterion_index)])]
            cell_rows.append(
                {
                    "outer_fold": fold,
                    "model": model,
                    "family": families[model],
                    "criterion_id": criterion,
                    "label": int(evaluation_raw[model_index, criterion_index]),
                    "probability": float(probabilities[model_index, criterion_index]),
                }
            )

        admin_bank = context.dense._subset_bank(bank, administration)
        spec = context.dense.scat.RunSpec(
            seed=DIAGNOSTIC_CAT["seed"],
            top_n=DIAGNOSTIC_CAT["top_n"],
            max_se=DIAGNOSTIC_CAT["max_se"],
            min_evals_per_skill=DIAGNOSTIC_CAT["min_evals_per_skill"],
            min_scenarios=DIAGNOSTIC_CAT["min_scenarios"],
            max_scenarios=DIAGNOSTIC_CAT["max_scenarios"],
            selection=DIAGNOSTIC_CAT["selection"],
            mode="cat",
        )
        for model_index, model in enumerate(test_models):
            row: dict[str, Any] = {
                "model": model,
                "family": families[model],
                "fold": fold,
                "fit_grid": fit_grid,
                "ridge": ridge,
                "quadrature_method": method,
                "quadrature_nodes_requested": nodes,
                "quadrature_nodes_effective": int(quadrature.grid.shape[0]),
                "linear_bound": bound,
                "fit_cache_sha256": _sha256(fit_path),
                "n_items_before_policy": policy["n_items_before_policy"],
                "n_items_after_policy": policy["n_items_after_policy"],
                "n_nonpositive_items": policy["n_nonpositive_items"],
                "theta_full": float(theta_full[model_index]),
                "theta_reference": float(theta_full[model_index]),
                "se_full": float(se_full[model_index]),
                "se_reference": float(se_full[model_index]),
                "theta_heldout": float(theta_heldout[model_index]),
                "se_heldout": float(se_heldout[model_index]),
                "posterior_tail_mass_full": float(tail_full[model_index]),
                "posterior_tail_mass_heldout": float(tail_heldout[model_index]),
            }
            responses = context.dense.scat.responses_for_bank(matrix.loc[model], bank)
            reference_rate = context.dense.scat.pass_rate_check(
                responses, bank.A, bank.b, np.asarray([theta_full[model_index]])
            )
            row.update(
                {
                    "observed_pass_rate": reference_rate["observed_pass_rate"],
                    "predicted_pass_rate_reference": reference_rate["predicted_pass_rate"],
                }
            )
            if include_cat:
                replay = context.dense.scat.run_recorded_model(
                    model,
                    matrix.loc[model],
                    admin_bank,
                    admin_records,
                    quadrature,
                    spec,
                    mwle_ridge=DIAGNOSTIC_CAT["mwle_ridge"],
                )
                estimates = {
                    name: np.asarray([replay[f"theta_{name}"][DIMENSION]], dtype=float)
                    for name in ("online", "eap", "mwle")
                }
                rates = {
                    name: context.dense.scat.pass_rate_check(responses, bank.A, bank.b, estimate)
                    for name, estimate in estimates.items()
                }
                row.update(
                    {
                        "theta_online": float(estimates["online"][0]),
                        "theta_eap": float(estimates["eap"][0]),
                        "theta_mwle": float(estimates["mwle"][0]),
                        "mwle_converged": bool(replay["mwle_converged"]),
                        "predicted_pass_rate_online": rates["online"]["predicted_pass_rate"],
                        "predicted_pass_rate_eap": rates["eap"]["predicted_pass_rate"],
                        "predicted_pass_rate_mwle": (
                            rates["mwle"]["predicted_pass_rate"]
                            if replay["mwle_converged"]
                            else float("nan")
                        ),
                        "scenario_order": json.dumps(replay["scenario_order"], ensure_ascii=False),
                        "criterion_order": json.dumps(
                            replay["criterion_order"], ensure_ascii=False
                        ),
                        "scenarios_administered": replay["scenarios_administered"],
                        "criteria_administered": replay["criteria_administered"],
                    }
                )
            model_rows.append(row)

    per_model = pd.DataFrame(model_rows).sort_values(["fold", "model"])
    cells = pd.DataFrame(cell_rows).sort_values(["outer_fold", "model", "criterion_id"])
    if cells.duplicated(["outer_fold", "model", "criterion_id"]).any():
        raise QuadratureResolutionError("held-out score contains duplicate cell keys")
    pooled = context.dense.cell_cv.metrics(
        cells["label"].to_numpy(dtype=float), cells["probability"].to_numpy(dtype=float)
    )
    oos = pd.DataFrame(
        [
            {
                "outer_fold": "pooled",
                **pooled,
                "n_models": len(per_model),
                "n_cells": len(cells),
            },
            *fold_metric_rows,
        ]
    )
    quantization = context.dense._quantization_diagnostics(
        per_model["theta_full"], quadrature.grid[:, 0]
    )
    _write_csv(destination / "per_model.csv", per_model)
    _write_csv(destination / "heldout_cells.csv", cells)
    _write_csv(destination / "oos_metrics.csv", oos)
    if include_cat:
        _write_csv(destination / "recovery.csv", context.dense._pooled_recovery(per_model))
        _write_csv(destination / "pass_rate.csv", context.dense._pooled_pass_rate(per_model))
    _write_json(
        destination / "score_manifest.json",
        {
            "generated_at": _utcnow(),
            "worker": "score-quadrature",
            "fit_grid": fit_grid,
            "ridge": ridge,
            "quadrature_method": method,
            "linear_bound": bound,
            "requested_node_count": nodes,
            "effective_node_count": int(quadrature.grid.shape[0]),
            "quadrature_axis_sha256": array_sha256(quadrature.grid[:, 0]),
            "quadrature_log_prior_sha256": array_sha256(quadrature.log_prior),
            "fit_cache_dir_relative_to_study": str(
                context.dense._fit_cache_dir(out_dir, fit_grid, ridge).relative_to(out_dir)
            ),
            "fit_cache_reused": True,
            "no_refit_during_scoring": True,
            "include_diagnostic_cat": include_cat,
            "tail_region_absolute_theta_at_least": tail_region,
            "heldout_fold_manifest_relative_to_repo": str(context.split_path.relative_to(ROOT)),
            "heldout_fold_manifest_sha256": _sha256(context.split_path),
            "scenario_split": {
                "administration": len(administration),
                "evaluation": len(evaluation),
                "evaluation_never_enters_theta_or_cat_path": True,
            },
            "diagnostic_cat_configuration": DIAGNOSTIC_CAT if include_cat else None,
            "pooled_oos": pooled,
            "quantization": quantization,
            "n_models": len(per_model),
            "n_heldout_cells": len(cells),
        },
    )
    print(
        f"scored fit={fit_grid} ridge={ridge} with {method} nodes={nodes} "
        f"bound={bound} -> {destination}"
    )
    return 0


def _pooled_row(path: Path, key: str, value: str) -> dict[str, Any]:
    frame = pd.read_csv(path)
    subset = frame[(frame["fold"].astype(str) == "pooled") & (frame[key] == value)]
    if len(subset) != 1:
        raise QuadratureResolutionError(f"expected one pooled {value} row in {path}")
    return subset.iloc[0].to_dict()


def select_primary_nodes(gates: Mapping[str, Mapping[str, Any]]) -> int | None:
    first = bool(gates["401_vs_801"]["passed"])
    second = bool(gates["801_vs_1601"]["passed"])
    endpoint = bool(gates["401_vs_1601"]["passed"])
    if first and second and endpoint:
        return 401
    if not first and second:
        return 801
    return None


def _theta_shift_diagnostics(merged: pd.DataFrame) -> dict[str, Any]:
    """Summarize grid-to-grid theta shifts on every downstream-relevant support."""

    by_support: dict[str, dict[str, float]] = {}
    for support, column in (
        ("full", "theta_full"),
        ("administration_only", "theta_heldout"),
    ):
        lower = merged[f"{column}_lower"].to_numpy(float)
        upper = merged[f"{column}_upper"].to_numpy(float)
        shifts = np.abs(upper - lower)
        by_support[support] = {
            "median_absolute_theta_shift": float(np.median(shifts)),
            "p95_absolute_theta_shift": float(np.quantile(shifts, 0.95)),
            "maximum_absolute_theta_shift": float(np.max(shifts)),
        }
    result: dict[str, Any] = {"theta_shift_by_support": by_support}
    # Apply each frozen threshold to the worst support rather than allowing
    # full-bank stability to hide instability on the administration-only
    # support used by downstream CAT.
    for key in (
        "median_absolute_theta_shift",
        "p95_absolute_theta_shift",
        "maximum_absolute_theta_shift",
    ):
        result[key] = max(record[key] for record in by_support.values())
    return result


def _node_pileup_pass(quantization: Mapping[str, Any], minimum_unique: int) -> bool:
    return bool(
        int(quantization["unique_theta_rounded_6"]) > int(minimum_unique)
        and float(quantization["largest_rounded_value_fraction"]) < 0.5
    )


def _primary_pair_gate(
    context: ResolutionContext,
    out_dir: Path,
    lower: int,
    upper: int,
) -> dict[str, Any]:
    kwargs = {
        "method": "normal_trapezoid",
        "bound": 8.0,
        "fit_grid": 61,
        "ridge": 0.1,
    }
    lower_dir = score_dir(out_dir, nodes=lower, **kwargs)
    upper_dir = score_dir(out_dir, nodes=upper, **kwargs)
    left = pd.read_csv(lower_dir / "per_model.csv")
    right = pd.read_csv(upper_dir / "per_model.csv")
    merged = left.merge(
        right, on=["model", "fold"], suffixes=("_lower", "_upper"), validate="one_to_one"
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise QuadratureResolutionError("primary score panels contain different model keys")
    shift_diagnostics = _theta_shift_diagnostics(merged)
    orders_identical = bool(
        (merged["scenario_order_lower"] == merged["scenario_order_upper"]).all()
        and (merged["criterion_order_lower"] == merged["criterion_order_upper"]).all()
    )
    left_recovery = _pooled_row(lower_dir / "recovery.csv", "estimator", "mwle")
    right_recovery = _pooled_row(upper_dir / "recovery.csv", "estimator", "mwle")
    left_pass = _pooled_row(lower_dir / "pass_rate.csv", "estimator", "mwle")
    right_pass = _pooled_row(upper_dir / "pass_rate.csv", "estimator", "mwle")
    lower_manifest = _read_json(lower_dir / "score_manifest.json")
    upper_manifest = _read_json(upper_dir / "score_manifest.json")
    lower_quantization = lower_manifest["quantization"]
    upper_quantization = upper_manifest["quantization"]
    minimum_unique = int(context.base.config["dense_grid"]["structure_screen_fit_grid"])
    lower_node_pileup_pass = _node_pileup_pass(lower_quantization, minimum_unique)
    upper_node_pileup_pass = _node_pileup_pass(upper_quantization, minimum_unique)
    node_pileup_pass = bool(lower_node_pileup_pass and upper_node_pileup_pass)
    thresholds = context.raw["primary_eap"]["stability_thresholds"]
    gate = {
        "comparison": f"{lower}_vs_{upper}",
        "lower_nodes": lower,
        "upper_nodes": upper,
        "quadrature_method": "normal_trapezoid",
        "linear_bound": 8.0,
        **shift_diagnostics,
        "recovery_correlation_shift": abs(float(right_recovery["r"]) - float(left_recovery["r"])),
        "recovery_slope_shift": abs(float(right_recovery["slope"]) - float(left_recovery["slope"])),
        "theta_mae_shift": abs(float(right_recovery["mae"]) - float(left_recovery["mae"])),
        "pass_rate_mae_shift": abs(float(right_pass["mae"]) - float(left_pass["mae"])),
        "fixed_bank_orders_identical": orders_identical,
        "node_pileup_pass": node_pileup_pass,
        "lower_grid_node_pileup_pass": lower_node_pileup_pass,
        "upper_grid_node_pileup_pass": upper_node_pileup_pass,
        "lower_grid_quantization": lower_quantization,
        "upper_grid_quantization": upper_quantization,
        "thresholds": thresholds,
    }
    finite = all(
        math.isfinite(float(gate[key]))
        for key in (
            "median_absolute_theta_shift",
            "p95_absolute_theta_shift",
            "maximum_absolute_theta_shift",
            "recovery_correlation_shift",
            "recovery_slope_shift",
            "theta_mae_shift",
            "pass_rate_mae_shift",
        )
    )
    gate["passed"] = bool(
        finite
        and gate["median_absolute_theta_shift"]
        <= float(thresholds["maximum_median_absolute_theta_shift"])
        and gate["p95_absolute_theta_shift"]
        <= float(thresholds["maximum_p95_absolute_theta_shift"])
        and gate["maximum_absolute_theta_shift"]
        <= float(thresholds["maximum_absolute_theta_shift"])
        and gate["recovery_correlation_shift"]
        <= float(thresholds["maximum_recovery_correlation_shift"])
        and gate["recovery_slope_shift"] <= float(thresholds["maximum_recovery_slope_shift"])
        and gate["theta_mae_shift"] <= float(thresholds["maximum_theta_mae_shift"])
        and gate["pass_rate_mae_shift"] <= float(thresholds["maximum_pass_rate_mae_shift"])
        and (orders_identical or not thresholds["require_identical_fixed_bank_orders"])
        and node_pileup_pass
    )
    return gate


def summarize_primary(context: ResolutionContext, out_dir: Path) -> dict[str, Any]:
    gates = {
        name: _primary_pair_gate(context, out_dir, *map(int, name.split("_vs_")))
        for name in ("401_vs_801", "801_vs_1601", "401_vs_1601")
    }
    locked = select_primary_nodes(gates)
    rows = []
    for nodes in (401, 801, 1601):
        directory = score_dir(
            out_dir,
            method="normal_trapezoid",
            nodes=nodes,
            bound=8.0,
            fit_grid=61,
            ridge=0.1,
        )
        manifest = _read_json(directory / "score_manifest.json")
        recovery = _pooled_row(directory / "recovery.csv", "estimator", "mwle")
        rows.append(
            {
                "nodes": nodes,
                "heldout_log_loss": manifest["pooled_oos"]["log_loss"],
                "heldout_brier": manifest["pooled_oos"]["brier"],
                "mwle_recovery_r": recovery["r"],
                "mwle_recovery_slope": recovery["slope"],
                "mwle_recovery_mae": recovery["mae"],
                **manifest["quantization"],
            }
        )
    _write_csv(out_dir / "numerical_gates" / "primary_grid_summary.csv", pd.DataFrame(rows))
    decision = {
        "quadrature_method": "normal_trapezoid",
        "linear_bound": 8.0,
        "comparisons": gates,
        "selection_rule": context.raw["primary_eap"]["lock_rule"],
        "passed": locked is not None,
        "locked_eap_grid": locked,
    }
    _write_json(out_dir / "numerical_gates" / "primary_grid_gate.json", decision)
    return decision


def _cluster_bootstrap_interval(
    frame: pd.DataFrame,
    value_column: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    grouped = frame.groupby("family", sort=True)[value_column].agg(["sum", "count"])
    if len(grouped) < 2:
        raise QuadratureResolutionError("family-cluster bootstrap needs at least two families")
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        estimates[index] = sums[sampled].sum() / counts[sampled].sum()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def common_cell_equivalence(
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    *,
    margin: float,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    keys = ["outer_fold", "model", "criterion_id"]
    for name, frame in (("lower", lower), ("upper", upper)):
        missing = set(keys + ["family", "label", "probability"]) - set(frame.columns)
        if missing or frame.duplicated(keys).any():
            raise QuadratureResolutionError(f"{name} held-out cells are invalid: {missing}")
    merged = lower.merge(upper, on=keys, suffixes=("_lower", "_upper"), validate="one_to_one")
    if merged.empty:
        raise QuadratureResolutionError("fit grids have no common held-out cells")
    if not (merged["label_lower"] == merged["label_upper"]).all():
        raise QuadratureResolutionError("common-cell labels differ between fit grids")
    if not (merged["family_lower"] == merged["family_upper"]).all():
        raise QuadratureResolutionError("common-cell family assignments differ")
    merged["label"] = merged["label_lower"].astype(int)
    merged["family"] = merged["family_lower"]
    epsilon = 1e-12
    for suffix in ("lower", "upper"):
        probability = np.clip(merged[f"probability_{suffix}"].to_numpy(float), epsilon, 1 - epsilon)
        label = merged["label"].to_numpy(float)
        merged[f"log_loss_{suffix}"] = -(
            label * np.log(probability) + (1.0 - label) * np.log(1.0 - probability)
        )
        merged[f"brier_{suffix}"] = (probability - label) ** 2
    details: dict[str, Any] = {}
    passed = True
    for metric in ("log_loss", "brier"):
        delta_column = f"{metric}_delta_upper_minus_lower"
        merged[delta_column] = merged[f"{metric}_upper"] - merged[f"{metric}_lower"]
        pooled = float(merged[delta_column].mean())
        low, high = _cluster_bootstrap_interval(
            merged, delta_column, replicates=replicates, seed=seed
        )
        metric_pass = bool(abs(pooled) <= margin and low >= -margin and high <= margin)
        details[metric] = {
            "pooled_shift_upper_minus_lower": pooled,
            "absolute_pooled_shift": abs(pooled),
            "family_cluster_bootstrap_ci_95": [low, high],
            "equivalence_margin": margin,
            "pooled_shift_within_margin": abs(pooled) <= margin,
            "ci_wholly_within_equivalence_bounds": low >= -margin and high <= margin,
            "passed": metric_pass,
        }
        passed = passed and metric_pass
    result = {
        "n_lower_cells": len(lower),
        "n_upper_cells": len(upper),
        "n_common_cells": len(merged),
        "common_cell_keys_identical_after_explicit_intersection": True,
        "common_cell_labels_identical": True,
        "cluster_unit": "model_family",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "metrics": details,
        "passed": bool(passed),
    }
    columns = keys + [
        "family",
        "label",
        "probability_lower",
        "probability_upper",
        "log_loss_delta_upper_minus_lower",
        "brier_delta_upper_minus_lower",
    ]
    return result, merged[columns].sort_values(keys)


def _common_support_predictions(
    context: ResolutionContext,
    out_dir: Path,
    lower_grid: int,
    upper_grid: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Predict identical cells after estimating theta from identical admin items."""

    matrix = context.dense.cm.load_matrix_strict(context.matrix)
    context.dense.cm.configure_skills(",".join(context.dense.SOURCE_SKILLS))
    structure = context.dense.cell_cv.build_structure(
        context.dense.SOURCE_SKILLS,
        context.dense.DIMENSIONS,
        context.dense.STRUCTURE_NAME,
    )
    source_records = context.dense.estimator_cv.source_records_by_id(context.rubrics)
    item_to_scenario = context.dense.cell_cv.load_item_scenarios(context.rubrics)
    evaluation = set(
        map(
            str,
            context.base.splits["scenario_split"]["evaluation_scenario_ids"],
        )
    )
    quadrature = _quadrature(context, "normal_trapezoid", 401, 8.0)
    families = _model_family_map(context)
    rows: dict[str, list[dict[str, Any]]] = {"lower": [], "upper": []}
    diagnostics: list[dict[str, Any]] = []

    for outer in context.base.splits["outer_folds"]:
        fold = int(outer["outer_fold"])
        test_models = list(map(str, outer["test_model_ids"]))
        banks: dict[str, Any] = {}
        for side, grid in (("lower", lower_grid), ("upper", upper_grid)):
            fit_path = (
                context.dense._fit_cache_dir(out_dir, grid, 0.1)
                / f"outer_{fold}"
                / "fit.npz"
            )
            fit = context.dense.load_cached_fit(fit_path)
            bank, _policy = context.dense.estimator_cv.build_fold_bank(
                fit,
                structure,
                source_records,
                negative_policy="drop",
            )
            banks[side] = bank

        common_ids = sorted(
            set(map(str, banks["lower"].criterion_ids))
            & set(map(str, banks["upper"].criterion_ids))
        )
        scoring_ids = [
            criterion
            for criterion in common_ids
            if item_to_scenario.get(criterion, criterion) not in evaluation
        ]
        evaluation_ids = [
            criterion
            for criterion in common_ids
            if item_to_scenario.get(criterion, criterion) in evaluation
        ]
        if not scoring_ids or not evaluation_ids:
            raise QuadratureResolutionError(
                f"common support for fold {fold} lacks administration/evaluation items"
            )
        raw = matrix.loc[test_models].reindex(columns=common_ids).to_numpy(dtype=float)
        common_position = {criterion: index for index, criterion in enumerate(common_ids)}
        scoring_index = np.asarray([common_position[item] for item in scoring_ids], dtype=int)
        evaluation_index = np.asarray(
            [common_position[item] for item in evaluation_ids], dtype=int
        )
        evaluation_raw = raw[:, evaluation_index]
        evaluation_observed = np.isfinite(evaluation_raw)
        diagnostics.append(
            {
                "comparison": f"{lower_grid}_vs_{upper_grid}",
                "outer_fold": fold,
                "n_common_items": len(common_ids),
                "n_common_administration_items": len(scoring_ids),
                "n_common_evaluation_items": len(evaluation_ids),
                "n_common_observed_evaluation_cells": int(evaluation_observed.sum()),
            }
        )

        for side in ("lower", "upper"):
            bank = banks[side]
            bank_position = {
                str(criterion): index for index, criterion in enumerate(bank.criterion_ids)
            }
            bank_indices = np.asarray(
                [bank_position[criterion] for criterion in common_ids], dtype=int
            )
            common_a = bank.A[bank_indices]
            common_b = bank.b[bank_indices]
            theta, _se, _tail = _posterior_batch(
                raw,
                common_a,
                common_b,
                quadrature,
                scoring_index,
                float(
                    context.raw["bound_check"][
                        "tail_region_absolute_theta_at_least"
                    ]
                ),
            )
            probabilities = expit(
                theta[:, None] * common_a[evaluation_index, 0][None, :]
                - common_b[evaluation_index][None, :]
            )
            for model_index, criterion_index in np.argwhere(evaluation_observed):
                model = test_models[int(model_index)]
                rows[side].append(
                    {
                        "outer_fold": fold,
                        "model": model,
                        "family": families[model],
                        "criterion_id": evaluation_ids[int(criterion_index)],
                        "label": int(evaluation_raw[model_index, criterion_index]),
                        "probability": float(probabilities[model_index, criterion_index]),
                    }
                )

    frames = {
        side: pd.DataFrame(side_rows).sort_values(
            ["outer_fold", "model", "criterion_id"]
        )
        for side, side_rows in rows.items()
    }
    keys = ["outer_fold", "model", "criterion_id"]
    if not frames["lower"][keys].reset_index(drop=True).equals(
        frames["upper"][keys].reset_index(drop=True)
    ):
        raise QuadratureResolutionError(
            "common-support construction did not produce identical evaluation keys"
        )
    return frames["lower"], frames["upper"], diagnostics


def summarize_common_support(context: ResolutionContext, out_dir: Path) -> dict[str, Any]:
    policy = context.raw["fit_grid_common_support_recheck"]
    comparisons: dict[str, Any] = {}
    for name in policy["comparisons"]:
        lower, upper = map(int, name.split("_vs_"))
        lower_cells, upper_cells, diagnostics = _common_support_predictions(
            context,
            out_dir,
            lower,
            upper,
        )
        result, common = common_cell_equivalence(
            lower_cells,
            upper_cells,
            margin=float(policy["equivalence_margin"]),
            replicates=int(policy["bootstrap_replicates"]),
            seed=int(policy["bootstrap_seed"]),
        )
        result["comparison"] = name
        result["theta_support"] = "identical common administration criteria"
        result["fold_support"] = diagnostics
        comparisons[name] = result
        _write_csv(out_dir / "common_support" / f"common_cells_{name}.csv", common)
        _write_json(out_dir / "common_support" / f"equivalence_{name}.json", result)
    locked_name = str(policy["locked_comparison"])
    decision = {
        "comparisons": comparisons,
        "require_every_comparison": True,
        "locked_comparison": locked_name,
        "locked_fit_grid": 61 if all(record["passed"] for record in comparisons.values()) else None,
        "passed": bool(all(record["passed"] for record in comparisons.values())),
    }
    _write_json(out_dir / "common_support" / "common_cell_gate.json", decision)
    return decision


def _theta_comparison(
    lower_dir: Path,
    upper_dir: Path,
    *,
    maximum_full: float,
    maximum_heldout: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    lower = pd.read_csv(lower_dir / "per_model.csv")
    upper = pd.read_csv(upper_dir / "per_model.csv")
    merged = lower.merge(
        upper, on=["model", "fold"], suffixes=("_lower", "_upper"), validate="one_to_one"
    )
    if len(merged) != len(lower) or len(merged) != len(upper):
        raise QuadratureResolutionError("numerical methods contain different model keys")
    merged["absolute_theta_full_shift"] = np.abs(
        merged["theta_full_upper"] - merged["theta_full_lower"]
    )
    merged["absolute_theta_heldout_shift"] = np.abs(
        merged["theta_heldout_upper"] - merged["theta_heldout_lower"]
    )
    max_full = float(merged["absolute_theta_full_shift"].max())
    max_heldout = float(merged["absolute_theta_heldout_shift"].max())
    result = {
        "n_models": len(merged),
        "model_keys_identical": True,
        "maximum_full_theta_shift": max_full,
        "maximum_heldout_theta_shift": max_heldout,
        "maximum_full_theta_shift_allowed": maximum_full,
        "maximum_heldout_theta_shift_allowed": maximum_heldout,
        "passed": bool(max_full <= maximum_full and max_heldout <= maximum_heldout),
    }
    return result, merged


def summarize_bound_check(context: ResolutionContext, out_dir: Path) -> dict[str, Any]:
    config = context.raw["bound_check"]
    reference = score_dir(
        out_dir,
        method="normal_trapezoid",
        nodes=1601,
        bound=8.0,
        fit_grid=61,
        ridge=0.1,
    )
    comparison = score_dir(
        out_dir,
        method="normal_trapezoid",
        nodes=2001,
        bound=10.0,
        fit_grid=61,
        ridge=0.1,
    )
    result, merged = _theta_comparison(
        reference,
        comparison,
        maximum_full=float(config["maximum_full_theta_shift"]),
        maximum_heldout=float(config["maximum_heldout_theta_shift"]),
    )
    tail_columns = [
        "posterior_tail_mass_full_lower",
        "posterior_tail_mass_heldout_lower",
        "posterior_tail_mass_full_upper",
        "posterior_tail_mass_heldout_upper",
    ]
    tail_maxima = {column: float(merged[column].max()) for column in tail_columns}
    maximum_tail = max(tail_maxima.values())
    tail_pass = maximum_tail <= float(config["maximum_posterior_tail_mass"])
    result.update(
        {
            "reference": {"method": "normal_trapezoid", "nodes": 1601, "bound": 8.0},
            "comparison": {"method": "normal_trapezoid", "nodes": 2001, "bound": 10.0},
            "shared_step": config["shared_step"],
            "tail_region_absolute_theta_at_least": config["tail_region_absolute_theta_at_least"],
            "posterior_tail_mass_maxima": tail_maxima,
            "maximum_posterior_tail_mass_observed": maximum_tail,
            "maximum_posterior_tail_mass_allowed": config["maximum_posterior_tail_mass"],
            "posterior_tail_mass_passed": tail_pass,
            "passed": bool(result["passed"] and tail_pass),
        }
    )
    _write_csv(
        out_dir / "numerical_gates" / "bound_check_per_model.csv",
        merged[
            [
                "model",
                "fold",
                "absolute_theta_full_shift",
                "absolute_theta_heldout_shift",
                *tail_columns,
            ]
        ],
    )
    _write_json(out_dir / "numerical_gates" / "bound_check.json", result)
    return result


def summarize_cross_family(context: ResolutionContext, out_dir: Path) -> dict[str, Any]:
    config = context.raw["cross_family_check"]
    reference = score_dir(
        out_dir,
        method="normal_trapezoid",
        nodes=1601,
        bound=8.0,
        fit_grid=61,
        ridge=0.1,
    )
    comparison = score_dir(
        out_dir,
        method="gauss_hermite_scipy",
        nodes=1601,
        bound=None,
        fit_grid=61,
        ridge=0.1,
    )
    result, merged = _theta_comparison(
        reference,
        comparison,
        maximum_full=float(config["maximum_full_theta_shift"]),
        maximum_heldout=float(config["maximum_heldout_theta_shift"]),
    )
    reference_manifest = _read_json(reference / "score_manifest.json")
    comparison_manifest = _read_json(comparison / "score_manifest.json")
    result.update(
        {
            "reference_method": "normal_trapezoid",
            "comparison_method": "gauss_hermite_scipy",
            "requested_node_count": 1601,
            "reference_effective_node_count": reference_manifest["effective_node_count"],
            "comparison_effective_node_count": comparison_manifest["effective_node_count"],
        }
    )
    _write_csv(
        out_dir / "numerical_gates" / "cross_family_per_model.csv",
        merged[
            [
                "model",
                "fold",
                "absolute_theta_full_shift",
                "absolute_theta_heldout_shift",
            ]
        ],
    )
    _write_json(out_dir / "numerical_gates" / "cross_family_check.json", result)
    return result


def _ridge_summary(context: ResolutionContext, out_dir: Path, nodes: int) -> None:
    rows: list[dict[str, Any]] = []
    for ridge in (0.001, 0.01, 0.1):
        fit_manifest = _read_json(
            context.dense._fit_cache_dir(out_dir, 61, ridge) / "fit_grid_manifest.json"
        )
        directory = score_dir(
            out_dir,
            method="normal_trapezoid",
            nodes=nodes,
            bound=8.0,
            fit_grid=61,
            ridge=ridge,
        )
        score = _read_json(directory / "score_manifest.json")
        recovery = _pooled_row(directory / "recovery.csv", "estimator", "mwle")
        pass_rate = _pooled_row(directory / "pass_rate.csv", "estimator", "mwle")
        rows.append(
            {
                "fit_grid": 61,
                "quadrature_method": "normal_trapezoid",
                "eap_grid": nodes,
                "linear_bound": 8.0,
                "ridge": ridge,
                "all_fits_converged": all(
                    bool(scope["converged"]) for scope in fit_manifest["scopes"]
                ),
                "heldout_log_loss": score["pooled_oos"]["log_loss"],
                "heldout_brier": score["pooled_oos"]["brier"],
                "mwle_recovery_r": recovery["r"],
                "mwle_recovery_slope": recovery["slope"],
                "mwle_recovery_mae": recovery["mae"],
                "mwle_pass_rate_mae": pass_rate["mae"],
            }
        )
    _write_csv(out_dir / "ridge" / "ridge_sensitivity_summary.csv", pd.DataFrame(rows))
    _write_json(
        out_dir / "ridge" / "ridge_sensitivity_manifest.json",
        {
            "generated_at": _utcnow(),
            "fit_grid": 61,
            "quadrature_method": "normal_trapezoid",
            "eap_grid": nodes,
            "linear_bound": 8.0,
            "candidates": [0.001, 0.01, 0.1],
            "production_selection_deferred_to_nested_cv": True,
        },
    )


def build_grid_lock_payload(
    *,
    locked_nodes: int,
    score_manifest: Mapping[str, Any],
    common_gate: Mapping[str, Any],
    primary_gate: Mapping[str, Any],
    bound_gate: Mapping[str, Any],
    cross_family_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the native-compatible lock consumed by downstream CAT runners."""

    for name, gate in (
        ("common", common_gate),
        ("primary", primary_gate),
        ("bound", bound_gate),
        ("cross_family", cross_family_gate),
    ):
        if gate.get("passed") is not True:
            raise QuadratureResolutionError(f"cannot emit grid lock: {name} gate did not pass")
    required = (
        "effective_node_count",
        "quadrature_axis_sha256",
        "quadrature_log_prior_sha256",
    )
    if any(not score_manifest.get(key) for key in required):
        raise QuadratureResolutionError("locked score manifest lacks quadrature identity hashes")
    if score_manifest.get("quadrature_method") != "normal_trapezoid" or not math.isclose(
        float(score_manifest.get("linear_bound", -1)), 8.0
    ):
        raise QuadratureResolutionError("locked score does not use the frozen primary method")
    if int(score_manifest.get("requested_node_count", -1)) != int(locked_nodes):
        raise QuadratureResolutionError("locked score node count differs from primary gate")
    return {
        "locked_at": _utcnow(),
        "fit_grid": 61,
        "eap_grid": int(locked_nodes),
        "initial_ridge": 0.1,
        "quadrature_method": "normal_trapezoid",
        "linear_bound": 8.0,
        "effective_node_count": int(score_manifest["effective_node_count"]),
        "quadrature_axis_sha256": str(score_manifest["quadrature_axis_sha256"]),
        "quadrature_log_prior_sha256": str(score_manifest["quadrature_log_prior_sha256"]),
        "fit_gate": common_gate,
        "eap_gate": primary_gate,
        "bound_gate": bound_gate,
        "cross_family_gate": cross_family_gate,
        "ridge_scheduled_only_after_this_lock": True,
        "ridge_scheduled_only_after_every_numerical_gate": True,
        "production_ridge_selection": "deferred_to_nested_cv",
    }


def _validate_running_study(context: ResolutionContext, out_dir: Path) -> dict[str, Any]:
    _assert_safe_output(context, out_dir)
    manifest = _read_json(out_dir / "study_manifest.json")
    if manifest.get("status") != "running":
        raise QuadratureResolutionError(
            f"worker/resume refuses terminal study status {manifest.get('status')!r}"
        )
    required = {
        "runner_sha256": _sha256(RUNNER),
        "config_sha256": _sha256(context.config_path),
        "split_manifest_sha256": _sha256(context.split_path),
        "scenario_cat_lib_sha256": _sha256(SCENARIO_LIB),
        "nested_runner_sha256": _sha256(NESTED_RUNNER),
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise QuadratureResolutionError(f"study {key} changed after run creation")
    for key, path in (
        ("matrix", context.matrix),
        ("rubrics", context.rubrics),
        ("scenarios", context.scenarios),
    ):
        record = (manifest.get("inputs") or {}).get(key) or {}
        if record.get("sha256") != _sha256(path):
            raise QuadratureResolutionError(f"study input changed after run creation: {key}")
    imports_path = out_dir / "parent_fit_imports.json"
    if manifest.get("parent_fit_imports_sha256") != _sha256(imports_path):
        raise QuadratureResolutionError("parent-fit import manifest changed after run creation")
    _verify_imported_fits(out_dir)
    return manifest


def _verify_passed_grid_lock(context: ResolutionContext, out_dir: Path) -> dict[str, Any]:
    """Fail closed unless the orchestrator emitted a hash-verified numerical lock."""

    lock_path = out_dir / "dense_grid" / "grid_lock.json"
    marker_path = out_dir / "stage_markers" / "emit_native_grid_lock.json"
    if not lock_path.is_file() or not marker_path.is_file():
        raise QuadratureResolutionError(
            "post-lock ridge worker requires a verified passed grid_lock"
        )
    marker = _read_json(marker_path)
    relative = str(lock_path.relative_to(out_dir))
    if (
        marker.get("stage") != "emit_native_grid_lock"
        or marker.get("kind") != "deterministic_analysis"
        or marker.get("command") != ["internal-analysis", "emit_native_grid_lock"]
        or marker.get("outputs") != [relative]
        or (marker.get("output_sha256") or {}).get(relative) != _sha256(lock_path)
    ):
        raise QuadratureResolutionError(
            "post-lock ridge worker grid_lock stage marker is invalid"
        )
    lock = _read_json(lock_path)
    fit_gate = lock.get("fit_gate") or {}
    eap_gate = lock.get("eap_gate") or {}
    locked_nodes = int(lock.get("eap_grid", -1))
    allowed_nodes = set(map(int, context.raw["primary_eap"]["grid_candidates"]))
    if not (
        int(lock.get("fit_grid", -1)) == 61
        and locked_nodes in allowed_nodes
        and lock.get("quadrature_method") == "normal_trapezoid"
        and math.isclose(float(lock.get("linear_bound", -1)), 8.0)
        and fit_gate.get("passed") is True
        and int(fit_gate.get("locked_fit_grid", -1)) == 61
        and eap_gate.get("passed") is True
        and int(eap_gate.get("locked_eap_grid", -1)) == locked_nodes
        and (lock.get("bound_gate") or {}).get("passed") is True
        and (lock.get("cross_family_gate") or {}).get("passed") is True
        and lock.get("ridge_scheduled_only_after_this_lock") is True
        and lock.get("ridge_scheduled_only_after_every_numerical_gate") is True
    ):
        raise QuadratureResolutionError(
            "post-lock ridge worker grid_lock has not passed every frozen gate"
        )
    return lock


class ResolutionOrchestrator:
    def __init__(
        self,
        context: ResolutionContext,
        out_dir: Path,
        *,
        resume: bool,
        plan_only: bool,
    ) -> None:
        self.context = context
        self.out_dir = out_dir.resolve()
        self.resume = resume
        self.plan_only = plan_only
        self.commands: list[dict[str, Any]] = []

    def prepare(self) -> None:
        _assert_safe_output(self.context, self.out_dir)
        if self.plan_only:
            return
        if self.resume:
            if not self.out_dir.is_dir():
                raise QuadratureResolutionError("--resume output directory does not exist")
            _validate_running_study(self.context, self.out_dir)
            return
        if self.out_dir.exists():
            raise QuadratureResolutionError(
                f"output already exists: {self.out_dir}; preserve it and use "
                "--resume or a fresh path"
            )
        self.out_dir.mkdir(parents=True)
        shutil.copy2(self.context.config_path, self.out_dir / "quadrature_resolution_config.json")
        shutil.copy2(self.context.split_path, self.out_dir / "frozen_splits.json")
        versions: dict[str, str | None] = {}
        for package in ("numpy", "pandas", "scipy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        _write_json(
            self.out_dir / "study_manifest.json",
            {
                "generated_at": _utcnow(),
                "status": "running",
                "runner_relative_to_repo": str(RUNNER.relative_to(ROOT)),
                "runner_sha256": _sha256(RUNNER),
                "scenario_cat_lib_relative_to_repo": str(SCENARIO_LIB.relative_to(ROOT)),
                "scenario_cat_lib_sha256": _sha256(SCENARIO_LIB),
                "nested_runner_relative_to_repo": str(NESTED_RUNNER.relative_to(ROOT)),
                "nested_runner_sha256": _sha256(NESTED_RUNNER),
                "config_relative_to_repo": str(self.context.config_path.relative_to(ROOT)),
                "config_sha256": _sha256(self.context.config_path),
                "split_manifest_relative_to_repo": str(self.context.split_path.relative_to(ROOT)),
                "split_manifest_sha256": _sha256(self.context.split_path),
                "inputs": {
                    key: {
                        "path_relative_to_repo": str(path.relative_to(ROOT)),
                        "sha256": _sha256(path),
                    }
                    for key, path in (
                        ("matrix", self.context.matrix),
                        ("rubrics", self.context.rubrics),
                        ("scenarios", self.context.scenarios),
                    )
                },
                "parent": {
                    key: self.context.raw["parent_study"][key]
                    for key in (
                        "config_sha256",
                        "runner_sha256",
                        "study_manifest_sha256",
                        "imported_fit_manifest_sha256",
                        "fit_gate_sha256",
                        "fit_stage_marker_sha256",
                        "grid_41_import_manifest_sha256",
                        "grid_41_source_stage_marker_sha256",
                        "grid_81_stage_marker_sha256",
                    )
                },
                "git_commit": _git_value("rev-parse", "HEAD"),
                "git_branch": _git_value("branch", "--show-current"),
                "git_dirty": bool(_git_value("status", "--porcelain")),
                "python": sys.version,
                "platform": platform.platform(),
                "packages": versions,
                "fit_grid_61_ridge_0p1_refit": False,
                "judge_calls": False,
                "cat_configuration_selection": False,
                "historical_outputs_modified": False,
            },
        )
        imports = _copy_parent_fits(self.context, self.out_dir)
        manifest = _read_json(self.out_dir / "study_manifest.json")
        manifest["parent_fit_imports_sha256"] = _sha256(self.out_dir / "parent_fit_imports.json")
        manifest["parent_fit_artifact_count"] = len(imports["artifacts"])
        _write_json(self.out_dir / "study_manifest.json", manifest)

    def _relative_output(self, path: Path) -> str:
        relative = _safe_relative(path, self.out_dir)
        if relative is None:
            raise QuadratureResolutionError(f"stage output lies outside study: {path}")
        return str(relative)

    def _record_plan(self, entry: dict[str, Any]) -> None:
        self.commands.append(entry)
        if self.plan_only:
            print(json.dumps({**entry, "status": "planned"}, ensure_ascii=False))
        else:
            _write_json(self.out_dir / "commands.json", self.commands)

    def _resume_or_reject_partial(
        self, stage: str, command: list[str], outputs: list[Path]
    ) -> bool:
        marker = self.out_dir / "stage_markers" / f"{stage}.json"
        relative_outputs = [self._relative_output(path) for path in outputs]
        any_outputs = any(path.exists() for path in outputs)
        all_outputs = all(path.is_file() for path in outputs)
        if not self.resume:
            if marker.exists() or any_outputs:
                raise QuadratureResolutionError(f"unexpected pre-existing stage output: {stage}")
            return False
        if marker.is_file():
            if not all_outputs:
                raise QuadratureResolutionError(f"resume stage marker has missing outputs: {stage}")
            previous = _read_json(marker)
            if previous.get("command") != command or previous.get("outputs") != relative_outputs:
                raise QuadratureResolutionError(f"resume stage definition changed: {stage}")
            hashes = previous.get("output_sha256") or {}
            if set(hashes) != set(relative_outputs):
                raise QuadratureResolutionError(
                    f"resume marker does not hash every output: {stage}"
                )
            for relative, path in zip(relative_outputs, outputs, strict=True):
                if hashes[relative] != _sha256(path):
                    raise QuadratureResolutionError(f"resume output hash changed: {relative}")
            print(f"[resume] {stage}")
            return True
        if any_outputs:
            raise QuadratureResolutionError(
                f"unmarked partial stage output found: {stage}; preserve and start fresh"
            )
        return False

    def run_subprocess_stage(self, stage: str, command: list[str], outputs: Iterable[Path]) -> None:
        expected = [Path(path) for path in outputs]
        entry = {
            "stage": stage,
            "kind": "subprocess",
            "command": command,
            "outputs": [self._relative_output(path) for path in expected],
        }
        self._record_plan(entry)
        if self.plan_only or self._resume_or_reject_partial(stage, command, expected):
            return
        marker_dir = self.out_dir / "stage_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)
        log_path = marker_dir / f"{stage}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND\n" + json.dumps(command) + "\n\nOUTPUT\n")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return_code = process.wait()
        if return_code:
            raise QuadratureResolutionError(
                f"stage {stage} failed with exit code {return_code}; see {log_path}"
            )
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise QuadratureResolutionError(f"stage {stage} omitted outputs: {missing}")
        relative = [self._relative_output(path) for path in expected]
        _write_json(
            marker_dir / f"{stage}.json",
            {
                **entry,
                "completed_at": _utcnow(),
                "output_sha256": {
                    name: _sha256(path) for name, path in zip(relative, expected, strict=True)
                },
            },
        )

    def run_analysis_stage(
        self,
        stage: str,
        function: Callable[[], Any],
        outputs: Iterable[Path],
    ) -> Any:
        expected = [Path(path) for path in outputs]
        command = ["internal-analysis", stage]
        entry = {
            "stage": stage,
            "kind": "deterministic_analysis",
            "command": command,
            "outputs": [self._relative_output(path) for path in expected],
        }
        self._record_plan(entry)
        if self.plan_only:
            return None
        if self._resume_or_reject_partial(stage, command, expected):
            return _read_json(expected[-1]) if expected[-1].suffix == ".json" else None
        result = function()
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise QuadratureResolutionError(f"analysis {stage} omitted outputs: {missing}")
        relative = [self._relative_output(path) for path in expected]
        marker_dir = self.out_dir / "stage_markers"
        marker_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            marker_dir / f"{stage}.json",
            {
                **entry,
                "completed_at": _utcnow(),
                "output_sha256": {
                    name: _sha256(path) for name, path in zip(relative, expected, strict=True)
                },
            },
        )
        return result

    def _worker_command(self, worker: str, **kwargs: Any) -> list[str]:
        command = [
            sys.executable,
            str(RUNNER.relative_to(ROOT)),
            "--config",
            str(self.context.config_path.relative_to(ROOT)),
            "--out-dir",
            str(self.out_dir.relative_to(ROOT)),
            "--worker",
            worker,
        ]
        for key, value in kwargs.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif value is not None:
                command.extend([flag, str(value)])
        return command

    def score_stage(
        self,
        *,
        method: str,
        nodes: int,
        bound: float | None,
        fit_grid: int,
        ridge: float,
        include_cat: bool,
    ) -> None:
        destination = score_dir(
            self.out_dir,
            method=method,
            nodes=nodes,
            bound=bound,
            fit_grid=fit_grid,
            ridge=ridge,
        )
        stage = (
            f"score_{method}_bound_{_slug_number(bound)}_fit_{fit_grid:03d}_"
            f"nodes_{nodes:04d}_ridge_{_slug_number(ridge)}"
        )
        self.run_subprocess_stage(
            stage,
            self._worker_command(
                "score",
                method=method,
                nodes=nodes,
                bound=bound,
                fit_grid=fit_grid,
                ridge=ridge,
                include_cat=include_cat,
            ),
            _score_outputs(destination, include_cat),
        )

    def fit_ridge_stage(self, ridge: float) -> None:
        if math.isclose(ridge, 0.1):
            raise QuadratureResolutionError("grid-61/ridge-0.1 refit is prohibited")
        cache = self.context.dense._fit_cache_dir(self.out_dir, 61, ridge)
        outputs = [cache / "fit_grid_manifest.json"]
        for scope in ["full"] + [
            f"outer_{int(record['outer_fold'])}"
            for record in self.context.base.splits["outer_folds"]
        ]:
            outputs.extend(
                [
                    cache / scope / "fit.npz",
                    cache / scope / "fit_manifest.json",
                    cache / scope / "item_params.csv",
                ]
            )
        self.run_subprocess_stage(
            f"fit_grid_061_ridge_{_slug_number(ridge)}",
            self._worker_command("fit-ridge", fit_grid=61, ridge=ridge),
            outputs,
        )

    def _finish(self, status: str, result: Mapping[str, Any]) -> None:
        manifest = _read_json(self.out_dir / "study_manifest.json")
        if manifest.get("status") != "running":
            raise QuadratureResolutionError("cannot rewrite a terminal study manifest")
        manifest.update(
            {
                "completed_at": _utcnow(),
                "status": status,
                "result": result,
                "historical_outputs_modified": False,
            }
        )
        _write_json(self.out_dir / "study_manifest.json", manifest)

    def _stop_if_failed(self, status: str, label: str, result: Mapping[str, Any]) -> None:
        if result.get("passed") is True:
            return
        self._finish(status, {label: result})
        raise QuadratureResolutionError(f"{label} failed; later stages were not scheduled")

    def run(self) -> int:
        self.prepare()

        # The grid-61 score is also the first primary EAP panel. The common-
        # support analysis below independently loads grids 41/61/81 and forces
        # both theta estimation and evaluation onto identical item sets.
        self.score_stage(
            method="normal_trapezoid",
            nodes=401,
            bound=8.0,
            fit_grid=61,
            ridge=0.1,
            include_cat=True,
        )
        common_outputs = (
            [
                self.out_dir / "common_support" / f"common_cells_{name}.csv"
                for name in ("41_vs_61", "61_vs_81")
            ]
            + [
                self.out_dir / "common_support" / f"equivalence_{name}.json"
                for name in ("41_vs_61", "61_vs_81")
            ]
            + [self.out_dir / "common_support" / "common_cell_gate.json"]
        )
        common = self.run_analysis_stage(
            "common_cell_fit_grid_gate",
            lambda: summarize_common_support(self.context, self.out_dir),
            common_outputs,
        )
        if self.plan_only:
            self._print_conditional_plan()
            return 0
        assert isinstance(common, Mapping)
        self._stop_if_failed("blocked_common_cell_fit_grid_equivalence", "common_support", common)

        for nodes in (801, 1601):
            self.score_stage(
                method="normal_trapezoid",
                nodes=nodes,
                bound=8.0,
                fit_grid=61,
                ridge=0.1,
                include_cat=True,
            )
        primary = self.run_analysis_stage(
            "primary_quadrature_gate",
            lambda: summarize_primary(self.context, self.out_dir),
            [
                self.out_dir / "numerical_gates" / "primary_grid_summary.csv",
                self.out_dir / "numerical_gates" / "primary_grid_gate.json",
            ],
        )
        assert isinstance(primary, Mapping)
        self._stop_if_failed("blocked_primary_quadrature_stability", "primary_eap", primary)
        locked_nodes = int(primary["locked_eap_grid"])

        self.score_stage(
            method="normal_trapezoid",
            nodes=2001,
            bound=10.0,
            fit_grid=61,
            ridge=0.1,
            include_cat=False,
        )
        bound = self.run_analysis_stage(
            "bound_sensitivity_gate",
            lambda: summarize_bound_check(self.context, self.out_dir),
            [
                self.out_dir / "numerical_gates" / "bound_check_per_model.csv",
                self.out_dir / "numerical_gates" / "bound_check.json",
            ],
        )
        assert isinstance(bound, Mapping)
        self._stop_if_failed("blocked_bound_sensitivity", "bound_check", bound)

        self.score_stage(
            method="gauss_hermite_scipy",
            nodes=1601,
            bound=None,
            fit_grid=61,
            ridge=0.1,
            include_cat=False,
        )
        cross = self.run_analysis_stage(
            "cross_family_gate",
            lambda: summarize_cross_family(self.context, self.out_dir),
            [
                self.out_dir / "numerical_gates" / "cross_family_per_model.csv",
                self.out_dir / "numerical_gates" / "cross_family_check.json",
            ],
        )
        assert isinstance(cross, Mapping)
        self._stop_if_failed("blocked_cross_family_stability", "cross_family", cross)

        locked_score = score_dir(
            self.out_dir,
            method="normal_trapezoid",
            nodes=locked_nodes,
            bound=8.0,
            fit_grid=61,
            ridge=0.1,
        )
        score_manifest = _read_json(locked_score / "score_manifest.json")

        def write_lock() -> dict[str, Any]:
            payload = build_grid_lock_payload(
                locked_nodes=locked_nodes,
                score_manifest=score_manifest,
                common_gate=common,
                primary_gate=primary,
                bound_gate=bound,
                cross_family_gate=cross,
            )
            _write_json(self.out_dir / "dense_grid" / "grid_lock.json", payload)
            return payload

        grid_lock = self.run_analysis_stage(
            "emit_native_grid_lock",
            write_lock,
            [self.out_dir / "dense_grid" / "grid_lock.json"],
        )
        assert isinstance(grid_lock, Mapping)

        for ridge in (0.001, 0.01):
            self.fit_ridge_stage(ridge)
            self.score_stage(
                method="normal_trapezoid",
                nodes=locked_nodes,
                bound=8.0,
                fit_grid=61,
                ridge=ridge,
                include_cat=True,
            )
        self.run_analysis_stage(
            "ridge_sensitivity_summary",
            lambda: _ridge_summary(self.context, self.out_dir, locked_nodes),
            [
                self.out_dir / "ridge" / "ridge_sensitivity_summary.csv",
                self.out_dir / "ridge" / "ridge_sensitivity_manifest.json",
            ],
        )
        self._finish(
            "phase2_complete",
            {
                "fit_grid": 61,
                "eap_grid": locked_nodes,
                "quadrature_method": "normal_trapezoid",
                "linear_bound": 8.0,
                "grid_lock": grid_lock,
                "ridge_candidates_completed": [0.001, 0.01, 0.1],
                "grid_61_ridge_0p1_refit": False,
            },
        )
        return 0

    def _print_conditional_plan(self) -> None:
        print(
            json.dumps(
                {
                    "conditional_after_common_support": [
                        "normal_trapezoid primary 801/1601 at bound 8",
                        "direct 401-vs-1601 and both successive gates",
                        "bound-10 T2001 and SciPy GH1601 cross-checks",
                    ],
                    "only_after_every_gate": [
                        "emit grid_lock with method/bound/axis/weight hashes",
                        "fit and score ridge 0.001 and 0.01",
                    ],
                    "explicitly_prohibited": "refit grid 61 / ridge 0.1",
                }
            )
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=("score", "fit-ridge"), default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--bound", type=float, default=None)
    parser.add_argument("--fit-grid", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=None)
    parser.add_argument("--include-cat", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.plan_only and args.resume:
        raise QuadratureResolutionError("--plan-only and --resume are mutually exclusive")
    context = load_context(args.config)
    out_dir = args.out_dir.resolve() if args.out_dir else context.out_dir
    _assert_safe_output(context, out_dir)
    if args.worker:
        if args.plan_only or args.resume:
            raise QuadratureResolutionError("workers do not accept --plan-only/--resume")
        if args.fit_grid is None or args.ridge is None:
            raise QuadratureResolutionError("worker requires --fit-grid and --ridge")
        if args.worker == "fit-ridge":
            if args.method is not None or args.nodes is not None or args.bound is not None:
                raise QuadratureResolutionError("fit-ridge worker received scoring arguments")
            if args.fit_grid != 61 or math.isclose(args.ridge, 0.1):
                raise QuadratureResolutionError(
                    "fit-ridge permits only grid 61 and never ridge 0.1"
                )
            _validate_running_study(context, out_dir)
            _verify_passed_grid_lock(context, out_dir)
            return context.dense.worker_fit_grid(context.base, out_dir, 61, args.ridge)
        if args.method is None or args.nodes is None:
            raise QuadratureResolutionError("score worker requires method and nodes")
        return _score_worker(
            context,
            out_dir,
            method=args.method,
            nodes=args.nodes,
            bound=args.bound,
            fit_grid=args.fit_grid,
            ridge=args.ridge,
            include_cat=args.include_cat,
        )
    if (
        any(
            value is not None
            for value in (args.method, args.nodes, args.bound, args.fit_grid, args.ridge)
        )
        or args.include_cat
    ):
        raise QuadratureResolutionError("worker-only arguments require --worker")
    return ResolutionOrchestrator(
        context,
        out_dir,
        resume=args.resume,
        plan_only=args.plan_only,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
