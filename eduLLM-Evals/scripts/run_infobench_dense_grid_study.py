#!/usr/bin/env python3
"""Run the isolated InFoBench dense-grid calibration study (remediation Phase 2).

This runner deliberately does *not* modify or resume the historical InFoBench
calibration run.  It consumes the preregistered remediation configuration and
the frozen family-grouped outer folds, then:

1. fits the selected ``overall_1d`` structure at each configured fit grid;
2. caches one full-data and five outer-training-only fits per setting;
3. scores held-out models at an explicitly separate EAP grid without refitting;
4. enforces the configured 25-vs-41 fit and 41-vs-81 EAP stability gates; and
5. schedules ridge candidates only after both numerical grids are locked.

The IRT implementation is not duplicated here.  Fits use
``kfold_cv_mirt.fit_structure`` / ``calibrate_mirt.fit_m2pl_em``; held-out cell
prediction uses ``kfold_cv_mirt.eap_predict_disjoint``; reference scoring and CAT
replay use ``scenario_cat_lib``; recovery summaries use
``scenario_kfold_estimator_cv``.

Examples::

    # Read-only command/dependency preview. Creates no output directory.
    python scripts/run_infobench_dense_grid_study.py --plan-only

    # Expensive full Phase-2 study (do not use for a smoke test).
    python scripts/run_infobench_dense_grid_study.py

    # Continue a previously interrupted run, accepting only matching markers.
    python scripts/run_infobench_dense_grid_study.py --resume
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs" / "infobench_calibration_remediation_v1.json"
DEFAULT_SPLITS = ROOT / "configs" / "infobench_remediation_splits_v1.manifest.json"

SOURCE_SKILLS = ("content", "format", "number", "style", "linguistic")
STRUCTURE_NAME = "overall_1d"
DIMENSIONS = "instruction_following=content+format+number+style+linguistic"
DIMENSION = "instruction_following"
MAX_ITER = 200
FIT_TOL = 1e-4
EXTREME_DISCRIMINATION = 6.0

# This fixed path is a numerical diagnostic, not Phase-3 CAT-setting selection.
# It reproduces the provisional configuration solely so a changed EAP grid can be
# compared while all non-grid choices and scenario paths remain fixed.
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


class DenseGridError(RuntimeError):
    """Fail-closed error for an invalid or incomplete dense-grid study."""


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DenseGridError(f"could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


# Loading the scenario-CV module also loads the shared calibrate_mirt,
# kfold_cv_mirt, and scenario_cat_lib modules that it audits/reuses.
estimator_cv = _load_module(
    "infobench_dense_grid_estimator_cv",
    ROOT / "scripts" / "scenario_kfold_estimator_cv.py",
)
cm = estimator_cv.cm
cell_cv = estimator_cv.cell_cv
scat = estimator_cv.scat


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseGridError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DenseGridError(f"expected one JSON object in {path}")
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _slug_number(value: float | int) -> str:
    return str(value).replace("-", "m").replace(".", "p")


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


def _safe_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _assert_not_historical(context: StudyContext, out_dir: Path) -> None:
    """Refuse every public or internal write that overlaps historical artifacts."""

    resolved = out_dir.resolve()
    protected_paths = (
        _resolve(context.config["baseline"]["run_dir"]).resolve(),
        _resolve(context.config["baseline"]["report_dir"]).resolve(),
    )
    for protected in protected_paths:
        if (
            resolved == protected
            or _safe_relative_to(resolved, protected)
            or _safe_relative_to(protected, resolved)
        ):
            raise DenseGridError(
                f"output {resolved} overlaps protected historical path {protected}"
            )


@dataclass(frozen=True)
class StudyContext:
    config_path: Path
    split_path: Path
    config: dict[str, Any]
    splits: dict[str, Any]
    matrix: Path
    rubrics: Path
    scenarios: Path
    judge_manifest: Path
    fit_grids: tuple[int, ...]
    eap_grids: tuple[int, ...]
    ridge_candidates: tuple[float, ...]
    initial_ridge: float
    default_out_dir: Path


def _validate_fold_contract(splits: Mapping[str, Any], models: set[str]) -> None:
    outer = splits.get("outer_folds") or []
    if len(outer) < 2:
        raise DenseGridError("frozen split manifest has fewer than two outer folds")
    seen: set[str] = set()
    for record in outer:
        test = set(map(str, record.get("test_model_ids") or []))
        train = set(map(str, record.get("train_model_ids") or []))
        if not test or test & train or test | train != models:
            raise DenseGridError(
                f"outer fold {record.get('outer_fold')} violates the frozen fit/test contract"
            )
        if seen & test:
            raise DenseGridError("a model occurs in multiple frozen outer-test folds")
        seen |= test
        for inner in record.get("inner_folds") or []:
            inner_fit = set(map(str, inner.get("fit_model_ids") or []))
            validation = set(map(str, inner.get("validation_model_ids") or []))
            if test & (inner_fit | validation):
                raise DenseGridError(
                    "outer-test model leaked into inner fold under outer "
                    f"{record.get('outer_fold')}"
                )
    if seen != models:
        raise DenseGridError("frozen outer-test folds do not cover every matrix model once")


def load_context(config_path: Path, split_path: Path | None = None) -> StudyContext:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != "infobench-calibration-remediation-v1":
        raise DenseGridError("configuration is not infobench-calibration-remediation-v1")
    if config.get("benchmark") != "InFoBench":
        raise DenseGridError("dense-grid runner requires benchmark=InFoBench")
    immutable = config.get("immutable_decisions") or {}
    if immutable.get("latent_structure") != STRUCTURE_NAME:
        raise DenseGridError("Phase 2 is frozen to the selected overall_1d structure")
    if immutable.get("latent_dimensions") != [DIMENSION]:
        raise DenseGridError("Phase 2 expects the instruction_following latent dimension")
    if immutable.get("preserve_historical_run") is not True:
        raise DenseGridError("configuration must require preservation of the historical run")

    dense = config.get("dense_grid") or {}
    fit_grids = tuple(int(value) for value in dense.get("fit_grid_candidates") or [])
    eap_grids = tuple(int(value) for value in dense.get("eap_grid_candidates") or [])
    ridges = tuple(float(value) for value in dense.get("ridge_candidates_after_grid_lock") or [])
    if fit_grids != (5, 7, 15, 25, 41):
        raise DenseGridError("fit-grid candidates differ from the preregistered 5,7,15,25,41")
    if eap_grids != (21, 41, 81):
        raise DenseGridError("EAP-grid candidates differ from the preregistered 21,41,81")
    if not ridges or any(value < 0 for value in ridges):
        raise DenseGridError("ridge candidate list is blank or invalid")
    initial_ridge = float(dense.get("initial_ridge", -1))
    if initial_ridge < 0:
        raise DenseGridError("dense_grid.initial_ridge must be nonnegative")
    if not any(math.isclose(initial_ridge, value) for value in ridges):
        raise DenseGridError("initial ridge must also appear in the post-lock ridge candidates")
    fit_gate = dense.get("fit_grid_stability") or {}
    if fit_gate.get("comparison") != "25_vs_41":
        raise DenseGridError("fit-grid stability comparison must remain 25_vs_41")
    if not 0 <= float(fit_gate.get("minimum_item_parameter_spearman", -1)) <= 1:
        raise DenseGridError("fit-grid Spearman threshold must be in [0,1]")
    if not 0 <= float(fit_gate.get("minimum_exportability_agreement", -1)) <= 1:
        raise DenseGridError("fit-grid exportability threshold must be in [0,1]")
    heldout_policy = fit_gate.get("heldout_metric_stability") or {}
    if (
        heldout_policy.get("rule") != "absolute_pooled_shift_or_paired_fold_se"
        or heldout_policy.get("metrics") != ["log_loss", "brier"]
        or float(heldout_policy.get("maximum_absolute_pooled_shift", -1)) < 0
        or float(heldout_policy.get("paired_fold_se_multiplier", -1)) < 0
        or heldout_policy.get("require_every_metric") is not True
    ):
        raise DenseGridError("fit-grid held-out metric stability policy is invalid")
    eap_gate = dense.get("eap_grid_stability") or {}
    if eap_gate.get("comparison") != "41_vs_81":
        raise DenseGridError("EAP-grid stability comparison must remain 41_vs_81")
    numeric_eap_thresholds = (
        "maximum_median_absolute_theta_shift",
        "maximum_p95_absolute_theta_shift",
        "maximum_recovery_correlation_shift",
        "maximum_recovery_slope_shift",
        "maximum_theta_mae_shift",
        "maximum_pass_rate_mae_shift",
    )
    if any(float(eap_gate.get(key, -1)) < 0 for key in numeric_eap_thresholds):
        raise DenseGridError("EAP-grid stability thresholds must be nonnegative")

    cross = config.get("cross_validation") or {}
    if int(cross.get("outer_folds", -1)) != 5 or int(cross.get("inner_folds", -1)) != 4:
        raise DenseGridError("remediation study requires the frozen five-outer/four-inner design")
    configured_split = _resolve(cross.get("fold_manifest") or DEFAULT_SPLITS)
    split_path = (split_path or configured_split).resolve()
    if split_path != configured_split.resolve():
        raise DenseGridError(
            "--split-manifest must match the frozen manifest named by the remediation config"
        )
    if not split_path.is_file():
        raise DenseGridError(f"frozen split manifest does not exist: {split_path}")
    expected_split_hash = cross.get("fold_manifest_sha256")
    if expected_split_hash and _sha256(split_path) != expected_split_hash:
        raise DenseGridError("frozen split-manifest hash differs from the remediation config")
    splits = _read_json(split_path)
    if splits.get("schema_version") != "infobench-remediation-nested-splits-v1":
        raise DenseGridError("unexpected frozen split-manifest schema")

    split_inputs = splits.get("inputs") or {}
    paths: dict[str, Path] = {}
    for key in ("response_matrix", "rubrics", "scenarios"):
        record = split_inputs.get(key) or {}
        path = _resolve(record.get("configured_path") or "")
        if not path.is_file():
            raise DenseGridError(f"frozen {key} input does not exist: {path}")
        if _sha256(path) != record.get("sha256"):
            raise DenseGridError(f"frozen {key} hash mismatch: {path}")
        paths[key] = path

    baseline = config.get("baseline") or {}
    matrix = paths["response_matrix"]
    if _sha256(matrix) != baseline.get("response_matrix_sha256"):
        raise DenseGridError("response matrix does not match the preregistered baseline hash")
    judge_manifest = _resolve(baseline.get("judge_manifest") or "")
    if not judge_manifest.is_file() or _sha256(judge_manifest) != baseline.get(
        "judge_manifest_sha256"
    ):
        raise DenseGridError("judge manifest does not match the preregistered baseline hash")

    matrix_frame = cm.load_matrix_strict(matrix)
    expected_models = int(baseline.get("models", -1))
    expected_items = int(baseline.get("source_criteria", -1))
    if matrix_frame.shape != (expected_models, expected_items):
        raise DenseGridError(
            f"matrix shape {matrix_frame.shape} != configured {(expected_models, expected_items)}"
        )
    _validate_fold_contract(splits, set(map(str, matrix_frame.index)))

    scenario_split = splits.get("scenario_split") or {}
    administration = set(map(str, scenario_split.get("administration_scenario_ids") or []))
    evaluation = set(map(str, scenario_split.get("evaluation_scenario_ids") or []))
    if not administration or not evaluation or administration & evaluation:
        raise DenseGridError("frozen administration/evaluation scenarios are blank or overlap")

    default_out = _resolve(
        immutable.get("new_output_dir") or "runs/calibration/InFoBench_remediation_v1"
    )
    return StudyContext(
        config_path=config_path,
        split_path=split_path,
        config=config,
        splits=splits,
        matrix=matrix,
        rubrics=paths["rubrics"],
        scenarios=paths["scenarios"],
        judge_manifest=judge_manifest,
        fit_grids=fit_grids,
        eap_grids=eap_grids,
        ridge_candidates=ridges,
        initial_ridge=initial_ridge,
        default_out_dir=default_out,
    )


def _fit_cache_dir(out_dir: Path, fit_grid: int, ridge: float) -> Path:
    return out_dir / "fit_cache" / f"grid_{fit_grid:03d}" / f"ridge_{_slug_number(ridge)}"


def _score_dir(out_dir: Path, fit_grid: int, eap_grid: int, ridge: float) -> Path:
    return (
        out_dir
        / "scores"
        / f"fit_grid_{fit_grid:03d}"
        / f"eap_grid_{eap_grid:03d}"
        / f"ridge_{_slug_number(ridge)}"
    )


def _stage_command(
    context: StudyContext,
    out_dir: Path,
    worker: str,
    *,
    fit_grid: int,
    ridge: float,
    eap_grid: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(context.config_path),
        "--split-manifest",
        str(context.split_path),
        "--out-dir",
        str(out_dir),
        "--worker",
        worker,
        "--fit-grid",
        str(fit_grid),
        "--ridge",
        str(ridge),
    ]
    if eap_grid is not None:
        command.extend(
            [
                "--eap-grid",
                str(eap_grid),
                "--fit-cache-dir",
                str(_fit_cache_dir(out_dir, fit_grid, ridge)),
            ]
        )
    return command


def build_fit_command(
    context: StudyContext, out_dir: Path, fit_grid: int, ridge: float
) -> list[str]:
    """Public/testable command builder with an explicit fit grid only."""

    return _stage_command(context, out_dir, "fit-grid", fit_grid=fit_grid, ridge=ridge)


def build_score_command(
    context: StudyContext,
    out_dir: Path,
    fit_grid: int,
    eap_grid: int,
    ridge: float,
) -> list[str]:
    """Public/testable command builder that keeps fit/EAP grids independent."""

    return _stage_command(
        context,
        out_dir,
        "score-grid",
        fit_grid=fit_grid,
        eap_grid=eap_grid,
        ridge=ridge,
    )


def _save_fit(path: Path, fit: Mapping[str, Any]) -> None:
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


def load_cached_fit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DenseGridError(f"cached fit is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            fit = {
                "A": np.asarray(data["A"], dtype=float),
                "b": np.asarray(data["b"], dtype=float),
                "R": np.asarray(data["R"], dtype=float),
                "items": [str(value) for value in data["items"].tolist()],
                "dim_labels": [str(value) for value in data["dim_labels"].tolist()],
                "loglik": float(data["loglik"]),
                "n_params": int(data["n_params"]),
                "n_iter": int(data["n_iter"]),
                "converged": bool(data["converged"]),
            }
    except (OSError, KeyError, ValueError) as exc:
        raise DenseGridError(f"could not load cached fit {path}: {exc}") from exc
    if fit["A"].shape != (len(fit["items"]), 1) or fit["b"].shape != (len(fit["items"]),):
        raise DenseGridError(f"cached fit has invalid one-dimensional arrays: {path}")
    if fit["dim_labels"] != [DIMENSION]:
        raise DenseGridError(f"cached fit has wrong dimension labels: {path}")
    return fit


def _fit_parameter_frame(fit: Mapping[str, Any], source_criteria: Sequence[str]) -> pd.DataFrame:
    A = np.asarray(fit["A"], dtype=float)[:, 0]
    b = np.asarray(fit["b"], dtype=float)
    exportable = np.isfinite(A) & np.isfinite(b) & (A > 0) & (A <= EXTREME_DISCRIMINATION)
    frame = pd.DataFrame(
        {
            "criterion_id": list(map(str, fit["items"])),
            f"a_{DIMENSION}": A,
            "b": b,
            "exportable": exportable,
        }
    )
    frame["source_criterion_count"] = len(source_criteria)
    return frame


def _scope_records(context: StudyContext, matrix: pd.DataFrame) -> list[tuple[str, list[str]]]:
    records: list[tuple[str, list[str]]] = [("full", list(map(str, matrix.index)))]
    for outer in context.splits["outer_folds"]:
        fold = int(outer["outer_fold"])
        records.append((f"outer_{fold}", list(map(str, outer["train_model_ids"]))))
    return records


def worker_fit_grid(context: StudyContext, out_dir: Path, fit_grid: int, ridge: float) -> int:
    if fit_grid not in context.fit_grids:
        raise DenseGridError(f"fit grid {fit_grid} is not preregistered")
    if ridge != context.initial_ridge and ridge not in context.ridge_candidates:
        raise DenseGridError(f"ridge {ridge} is not preregistered")
    cache_dir = _fit_cache_dir(out_dir, fit_grid, ridge)
    matrix = cm.load_matrix_strict(context.matrix)
    cm.configure_skills(",".join(SOURCE_SKILLS))
    q_by = cm.load_q_matrix(context.rubrics)
    cm.validate_matrix_bank_alignment(matrix, q_by, True)
    structure = cell_cv.build_structure(SOURCE_SKILLS, DIMENSIONS, STRUCTURE_NAME)
    fit_args = argparse.Namespace(
        grid=fit_grid,
        ridge=ridge,
        max_iter=MAX_ITER,
        tol=FIT_TOL,
        estimate_latent_corr=False,
    )
    scope_summaries: list[dict[str, Any]] = []
    for scope, train_models in _scope_records(context, matrix):
        scope_dir = cache_dir / scope
        fit_path = scope_dir / "fit.npz"
        manifest_path = scope_dir / "fit_manifest.json"
        params_path = scope_dir / "item_params.csv"
        if fit_path.is_file() and manifest_path.is_file() and params_path.is_file():
            previous = _read_json(manifest_path)
            expected = {
                "fit_grid": fit_grid,
                "ridge": ridge,
                "train_model_ids": train_models,
                "matrix_sha256": _sha256(context.matrix),
                "rubrics_sha256": _sha256(context.rubrics),
                "fold_manifest_sha256": _sha256(context.split_path),
            }
            if any(previous.get(key) != value for key, value in expected.items()):
                raise DenseGridError(
                    f"cached fit metadata differs for {scope}; use a fresh output directory"
                )
            fit = load_cached_fit(fit_path)
            reused = True
        else:
            if any(path.exists() for path in (fit_path, manifest_path, params_path)):
                raise DenseGridError(
                    f"partial fit cache found under {scope_dir}; preserve it for diagnosis "
                    "and restart in a fresh output directory"
                )
            fit = cell_cv.fit_structure(matrix.loc[train_models], q_by, fit_args, structure)
            scope_dir.mkdir(parents=True, exist_ok=True)
            _save_fit(fit_path, fit)
            _write_csv(params_path, _fit_parameter_frame(fit, matrix.columns))
            _write_json(
                manifest_path,
                {
                    "generated_at": _utcnow(),
                    "method": "kfold_cv_mirt.fit_structure -> calibrate_mirt.fit_m2pl_em",
                    "scope": scope,
                    "structure": structure.as_dict(),
                    "fit_grid": fit_grid,
                    "eap_grid": None,
                    "ridge": ridge,
                    "max_iter": MAX_ITER,
                    "tol": FIT_TOL,
                    "train_model_ids": train_models,
                    "matrix_sha256": _sha256(context.matrix),
                    "rubrics_sha256": _sha256(context.rubrics),
                    "fold_manifest_sha256": _sha256(context.split_path),
                    "converged": bool(fit["converged"]),
                    "n_iter": int(fit["n_iter"]),
                    "loglik": float(fit["loglik"]),
                    "n_params": int(fit["n_params"]),
                    "n_items": len(fit["items"]),
                    "fit_sha256": _sha256(fit_path),
                    "diagnostics": fit.get("diag") or {},
                },
            )
            reused = False
        params = pd.read_csv(params_path)
        scope_summaries.append(
            {
                "scope": scope,
                "n_train": len(train_models),
                "converged": bool(fit["converged"]),
                "n_iter": int(fit["n_iter"]),
                "loglik": float(fit["loglik"]),
                "n_items": len(fit["items"]),
                "n_exportable": int(params["exportable"].astype(bool).sum()),
                "cache_reused": reused,
                "fit_path": str(fit_path),
                "fit_sha256": _sha256(fit_path),
            }
        )
    _write_json(
        cache_dir / "fit_grid_manifest.json",
        {
            "generated_at": _utcnow(),
            "worker": "fit-grid",
            "fit_grid": fit_grid,
            "eap_grid": None,
            "ridge": ridge,
            "structure": STRUCTURE_NAME,
            "fold_manifest": str(context.split_path),
            "fold_manifest_sha256": _sha256(context.split_path),
            "scopes": scope_summaries,
        },
    )
    print(f"cached fit grid {fit_grid} ridge {ridge} -> {cache_dir}")
    return 0


def _subset_bank(bank: Any, scenarios: set[str]) -> Any:
    selected = np.asarray(
        [index for index, sid in enumerate(bank.scenario_ids) if sid in scenarios],
        dtype=int,
    )
    if selected.size == 0:
        raise DenseGridError("scenario restriction left no fitted items")
    return scat.FittedBank(
        records=[bank.records[index] for index in selected],
        dims=bank.dims,
        criterion_ids=tuple(bank.criterion_ids[index] for index in selected),
        scenario_ids=tuple(bank.scenario_ids[index] for index in selected),
        Q=bank.Q[selected].copy(),
        A=bank.A[selected].copy(),
        b=bank.b[selected].copy(),
        latent_correlation=bank.latent_correlation.copy(),
        source_path=f"{bank.source_path} [frozen administration scenarios]",
        dropped_negative_items=bank.dropped_negative_items,
    )


def _pooled_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    folds: list[str | int] = ["pooled"] + sorted(frame["fold"].unique().tolist())
    for fold in folds:
        subset = frame if fold == "pooled" else frame[frame["fold"] == fold]
        for estimator in ("online", "eap", "mwle"):
            eligible = subset
            if estimator == "mwle":
                eligible = eligible[eligible["mwle_converged"] == True]  # noqa: E712
            stats = estimator_cv.recovery_stats(
                eligible["theta_reference"], eligible[f"theta_{estimator}"]
            )
            rows.append({"fold": fold, "estimator": estimator, **stats})
    return pd.DataFrame(rows)


def _pooled_pass_rate(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    folds: list[str | int] = ["pooled"] + sorted(frame["fold"].unique().tolist())
    for fold in folds:
        subset = frame if fold == "pooled" else frame[frame["fold"] == fold]
        for estimator in ("online", "eap", "mwle", "reference"):
            eligible = subset
            if estimator == "mwle":
                eligible = eligible[eligible["mwle_converged"] == True]  # noqa: E712
            stats = estimator_cv.pass_rate_stats(
                eligible["observed_pass_rate"], eligible[f"predicted_pass_rate_{estimator}"]
            )
            rows.append({"fold": fold, "estimator": estimator, **stats})
    return pd.DataFrame(rows)


def _quantization_diagnostics(values: Sequence[float], nodes: np.ndarray) -> dict[str, Any]:
    theta = np.asarray(values, dtype=float)
    theta = theta[np.isfinite(theta)]
    if theta.size == 0:
        return {
            "n_models": 0,
            "unique_theta_rounded_6": 0,
            "largest_rounded_value_count": 0,
            "largest_rounded_value_fraction": None,
            "fraction_within_1e_6_of_node": None,
            "largest_nearest_node_fraction": None,
        }
    rounded = np.round(theta, 6)
    _, counts = np.unique(rounded, return_counts=True)
    axis = np.asarray(nodes, dtype=float).reshape(-1)
    distance = np.abs(theta[:, None] - axis[None, :])
    nearest = np.argmin(distance, axis=1)
    nearest_counts = np.bincount(nearest, minlength=len(axis))
    return {
        "n_models": int(theta.size),
        "unique_theta_rounded_6": int(np.unique(rounded).size),
        "largest_rounded_value_count": int(counts.max()),
        "largest_rounded_value_fraction": float(counts.max() / theta.size),
        "fraction_within_1e_6_of_node": float((distance.min(axis=1) <= 1e-6).mean()),
        "largest_nearest_node_fraction": float(nearest_counts.max() / theta.size),
    }


def worker_score_grid(
    context: StudyContext,
    out_dir: Path,
    fit_cache_dir: Path,
    fit_grid: int,
    eap_grid: int,
    ridge: float,
) -> int:
    if fit_grid not in context.fit_grids or eap_grid not in context.eap_grids:
        raise DenseGridError("fit/EAP grid is not preregistered")
    expected_cache = _fit_cache_dir(out_dir, fit_grid, ridge).resolve()
    if fit_cache_dir.resolve() != expected_cache:
        raise DenseGridError("score worker fit-cache path does not match fit grid/ridge")
    score_dir = _score_dir(out_dir, fit_grid, eap_grid, ridge)
    expected_outputs = (
        score_dir / "per_model.csv",
        score_dir / "oos_metrics.csv",
        score_dir / "recovery.csv",
        score_dir / "pass_rate.csv",
        score_dir / "score_manifest.json",
    )
    if all(path.is_file() for path in expected_outputs):
        previous = _read_json(score_dir / "score_manifest.json")
        if (
            previous.get("fit_grid") != fit_grid
            or previous.get("eap_grid") != eap_grid
            or previous.get("ridge") != ridge
            or previous.get("fit_cache_dir") != str(fit_cache_dir)
        ):
            raise DenseGridError("existing score cache metadata differs; use a fresh output")
        print(f"reused score grid fit={fit_grid} eap={eap_grid} ridge={ridge}")
        return 0
    if any(path.exists() for path in expected_outputs):
        raise DenseGridError(
            f"partial score cache found under {score_dir}; use a fresh output directory"
        )

    matrix = cm.load_matrix_strict(context.matrix)
    cm.configure_skills(",".join(SOURCE_SKILLS))
    structure = cell_cv.build_structure(SOURCE_SKILLS, DIMENSIONS, STRUCTURE_NAME)
    source_records = estimator_cv.source_records_by_id(context.rubrics)
    scenario_records = scat.load_scenario_records(context.scenarios)
    item_to_scenario = cell_cv.load_item_scenarios(context.rubrics)
    scenario_split = context.splits["scenario_split"]
    administration = set(map(str, scenario_split["administration_scenario_ids"]))
    evaluation = set(map(str, scenario_split["evaluation_scenario_ids"]))
    admin_records = {key: value for key, value in scenario_records.items() if key in administration}

    model_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    pooled_y: list[np.ndarray] = []
    pooled_p: list[np.ndarray] = []
    for outer in context.splits["outer_folds"]:
        fold = int(outer["outer_fold"])
        test_models = list(map(str, outer["test_model_ids"]))
        fit_path = fit_cache_dir / f"outer_{fold}" / "fit.npz"
        fit_manifest_path = fit_cache_dir / f"outer_{fold}" / "fit_manifest.json"
        cached_manifest = _read_json(fit_manifest_path)
        expected_train = list(map(str, outer["train_model_ids"]))
        expected_cache_fields = {
            "scope": f"outer_{fold}",
            "fit_grid": fit_grid,
            "eap_grid": None,
            "ridge": ridge,
            "train_model_ids": expected_train,
            "matrix_sha256": _sha256(context.matrix),
            "rubrics_sha256": _sha256(context.rubrics),
            "fold_manifest_sha256": _sha256(context.split_path),
        }
        if any(
            cached_manifest.get(key) != expected for key, expected in expected_cache_fields.items()
        ):
            raise DenseGridError(
                f"outer_{fold} cached fit provenance differs from the frozen study"
            )
        if cached_manifest.get("fit_sha256") != _sha256(fit_path):
            raise DenseGridError(f"outer_{fold} cached fit hash differs from its manifest")
        fit = load_cached_fit(fit_path)
        if not fit["converged"]:
            raise DenseGridError(
                f"outer_{fold} fit did not converge at fit grid {fit_grid}; refusing scoring"
            )
        bank, policy = estimator_cv.build_fold_bank(
            fit, structure, source_records, negative_policy="drop"
        )
        admin_bank = _subset_bank(bank, administration)
        quadrature = scat.build_quadrature(
            structure.n_dims,
            eap_grid,
            bank.latent_correlation,
            max_nodes=max(5000, eap_grid**structure.n_dims),
        )
        spec = scat.RunSpec(
            seed=DIAGNOSTIC_CAT["seed"],
            top_n=DIAGNOSTIC_CAT["top_n"],
            max_se=DIAGNOSTIC_CAT["max_se"],
            min_evals_per_skill=DIAGNOSTIC_CAT["min_evals_per_skill"],
            min_scenarios=DIAGNOSTIC_CAT["min_scenarios"],
            max_scenarios=DIAGNOSTIC_CAT["max_scenarios"],
            selection=DIAGNOSTIC_CAT["selection"],
            mode="cat",
        )

        y_fold, p_fold, _theta_fold, _mask_fold = cell_cv.eap_predict_disjoint(
            matrix.loc[test_models],
            list(bank.criterion_ids),
            bank.A,
            bank.b,
            quadrature.grid,
            quadrature.log_prior,
            item_to_scenario,
            evaluation,
        )
        pooled_y.append(y_fold)
        pooled_p.append(p_fold)
        fold_metric_rows.append(
            {"fold": fold, **cell_cv.metrics(y_fold, p_fold), "n_models": len(test_models)}
        )

        for model in test_models:
            replay = scat.run_recorded_model(
                model,
                matrix.loc[model],
                admin_bank,
                admin_records,
                quadrature,
                spec,
                mwle_ridge=DIAGNOSTIC_CAT["mwle_ridge"],
            )
            responses = scat.responses_for_bank(matrix.loc[model], bank)
            reference = scat.batch_eap(responses, bank.A, bank.b, quadrature)
            reference_rate = scat.pass_rate_check(responses, bank.A, bank.b, reference.theta)
            estimates = {
                name: np.asarray([replay[f"theta_{name}"][DIMENSION]], dtype=float)
                for name in ("online", "eap", "mwle")
            }
            rates = {
                name: scat.pass_rate_check(responses, bank.A, bank.b, theta)
                for name, theta in estimates.items()
            }
            model_rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "fit_grid": fit_grid,
                    "eap_grid": eap_grid,
                    "ridge": ridge,
                    "fit_cache_sha256": _sha256(fit_path),
                    "n_items_before_policy": policy["n_items_before_policy"],
                    "n_items_after_policy": policy["n_items_after_policy"],
                    "n_nonpositive_items": policy["n_nonpositive_items"],
                    "theta_reference": float(reference.theta[0]),
                    "se_reference": float(reference.se[0]),
                    "theta_online": float(estimates["online"][0]),
                    "theta_eap": float(estimates["eap"][0]),
                    "theta_mwle": float(estimates["mwle"][0]),
                    "mwle_converged": bool(replay["mwle_converged"]),
                    "observed_pass_rate": reference_rate["observed_pass_rate"],
                    "predicted_pass_rate_reference": reference_rate["predicted_pass_rate"],
                    "predicted_pass_rate_online": rates["online"]["predicted_pass_rate"],
                    "predicted_pass_rate_eap": rates["eap"]["predicted_pass_rate"],
                    "predicted_pass_rate_mwle": (
                        rates["mwle"]["predicted_pass_rate"]
                        if replay["mwle_converged"]
                        else float("nan")
                    ),
                    "scenario_order": json.dumps(replay["scenario_order"], ensure_ascii=False),
                    "criterion_order": json.dumps(replay["criterion_order"], ensure_ascii=False),
                    "scenarios_administered": replay["scenarios_administered"],
                    "criteria_administered": replay["criteria_administered"],
                }
            )

    per_model = pd.DataFrame(model_rows).sort_values(["fold", "model"])
    recovery = _pooled_recovery(per_model)
    pass_rate = _pooled_pass_rate(per_model)
    pooled_metrics = cell_cv.metrics(np.concatenate(pooled_y), np.concatenate(pooled_p))
    oos_metrics = pd.DataFrame(
        [{"fold": "pooled", **pooled_metrics, "n_models": len(per_model)}] + fold_metric_rows
    )
    quantization = _quantization_diagnostics(
        per_model["theta_reference"],
        scat.build_quadrature(1, eap_grid, np.eye(1), max_nodes=eap_grid).grid[:, 0],
    )
    _write_csv(score_dir / "per_model.csv", per_model)
    _write_csv(score_dir / "oos_metrics.csv", oos_metrics)
    _write_csv(score_dir / "recovery.csv", recovery)
    _write_csv(score_dir / "pass_rate.csv", pass_rate)
    _write_json(
        score_dir / "score_manifest.json",
        {
            "generated_at": _utcnow(),
            "worker": "score-grid",
            "fit_grid": fit_grid,
            "eap_grid": eap_grid,
            "ridge": ridge,
            "fit_cache_dir": str(fit_cache_dir),
            "fit_cache_reused": True,
            "no_refit_during_scoring": True,
            "heldout_fold_manifest": str(context.split_path),
            "heldout_fold_manifest_sha256": _sha256(context.split_path),
            "scenario_split": {
                "administration": len(administration),
                "evaluation": len(evaluation),
                "evaluation_never_enters_cat_path": True,
            },
            "diagnostic_cat_configuration": DIAGNOSTIC_CAT,
            "pooled_oos": pooled_metrics,
            "quantization": quantization,
            "n_models": len(per_model),
        },
    )
    print(f"scored cached fit={fit_grid} with EAP={eap_grid} ridge={ridge} -> {score_dir}")
    return 0


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 3 or aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return None
    value = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
    return float(value) if value == value else None


def _parameter_comparison(
    context: StudyContext,
    out_dir: Path,
    lower_grid: int,
    upper_grid: int,
    ridge: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_ids = list(cm.load_matrix_strict(context.matrix).columns)
    scopes = ["full"] + [
        f"outer_{int(record['outer_fold'])}" for record in context.splits["outer_folds"]
    ]
    for scope in scopes:
        lower_dir = _fit_cache_dir(out_dir, lower_grid, ridge) / scope
        upper_dir = _fit_cache_dir(out_dir, upper_grid, ridge) / scope
        lower = pd.read_csv(lower_dir / "item_params.csv").set_index("criterion_id")
        upper = pd.read_csv(upper_dir / "item_params.csv").set_index("criterion_id")
        lower_export = lower["exportable"].astype(bool).reindex(source_ids, fill_value=False)
        upper_export = upper["exportable"].astype(bool).reindex(source_ids, fill_value=False)
        lower_manifest = _read_json(lower_dir / "fit_manifest.json")
        upper_manifest = _read_json(upper_dir / "fit_manifest.json")
        rows.append(
            {
                "scope": scope,
                "lower_grid": lower_grid,
                "upper_grid": upper_grid,
                "a_spearman": _spearman(lower[f"a_{DIMENSION}"], upper[f"a_{DIMENSION}"]),
                "b_spearman": _spearman(lower["b"], upper["b"]),
                "exportability_agreement": float((lower_export == upper_export).mean()),
                "lower_converged": bool(lower_manifest["converged"]),
                "upper_converged": bool(upper_manifest["converged"]),
                "lower_exportable": int(lower_export.sum()),
                "upper_exportable": int(upper_export.sum()),
            }
        )
    return pd.DataFrame(rows)


def _pooled_row(path: Path, key: str, value: str) -> dict[str, Any]:
    frame = pd.read_csv(path)
    subset = frame[(frame["fold"].astype(str) == "pooled") & (frame[key] == value)]
    if len(subset) != 1:
        raise DenseGridError(f"expected one pooled {value} row in {path}")
    return subset.iloc[0].to_dict()


def paired_fold_metric_stability(
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered pooled-shift OR paired-fold-uncertainty rule."""

    if policy.get("rule") != "absolute_pooled_shift_or_paired_fold_se":
        raise DenseGridError("unknown held-out metric stability rule")
    metrics = list(map(str, policy.get("metrics") or []))
    if metrics != ["log_loss", "brier"]:
        raise DenseGridError("held-out stability must preregister log_loss and brier")
    absolute_limit = float(policy["maximum_absolute_pooled_shift"])
    se_multiplier = float(policy["paired_fold_se_multiplier"])
    if absolute_limit < 0 or se_multiplier < 0:
        raise DenseGridError("held-out stability thresholds must be nonnegative")

    def split(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        fold_text = frame["fold"].astype(str)
        pooled = frame[fold_text == "pooled"]
        folds = frame[fold_text != "pooled"].copy()
        if len(pooled) != 1 or folds.empty:
            raise DenseGridError("held-out metrics require one pooled row and fold rows")
        folds["fold"] = folds["fold"].astype(str)
        if folds["fold"].duplicated().any():
            raise DenseGridError("held-out metrics contain duplicate fold rows")
        return pooled.iloc[0], folds.set_index("fold")

    lower_pooled, lower_folds = split(lower)
    upper_pooled, upper_folds = split(upper)
    if set(lower_folds.index) != set(upper_folds.index):
        raise DenseGridError("fit grids do not contain the same held-out folds")
    details: dict[str, Any] = {}
    for metric in metrics:
        if metric not in lower or metric not in upper:
            raise DenseGridError(f"held-out metric {metric!r} is missing")
        paired = upper_folds.loc[sorted(upper_folds.index), metric].to_numpy(
            dtype=float
        ) - lower_folds.loc[sorted(lower_folds.index), metric].to_numpy(dtype=float)
        paired_se = (
            float(np.std(paired, ddof=1) / math.sqrt(len(paired))) if len(paired) > 1 else 0.0
        )
        pooled_shift = float(upper_pooled[metric]) - float(lower_pooled[metric])
        uncertainty_limit = se_multiplier * paired_se
        passed = bool(abs(pooled_shift) <= absolute_limit or abs(pooled_shift) <= uncertainty_limit)
        details[metric] = {
            "lower_pooled": float(lower_pooled[metric]),
            "upper_pooled": float(upper_pooled[metric]),
            "pooled_shift_upper_minus_lower": pooled_shift,
            "absolute_pooled_shift": abs(pooled_shift),
            "paired_fold_differences": paired.tolist(),
            "paired_fold_standard_error": paired_se,
            "absolute_limit": absolute_limit,
            "uncertainty_multiplier": se_multiplier,
            "uncertainty_limit": uncertainty_limit,
            "passed_absolute_limit": abs(pooled_shift) <= absolute_limit,
            "passed_uncertainty_limit": abs(pooled_shift) <= uncertainty_limit,
            "passed": passed,
        }
    require_every = bool(policy.get("require_every_metric", True))
    overall = (
        all(detail["passed"] for detail in details.values())
        if require_every
        else any(detail["passed"] for detail in details.values())
    )
    return {
        "rule": policy["rule"],
        "require_every_metric": require_every,
        "metrics": details,
        "passed": bool(overall),
    }


def summarize_fit_grids(context: StudyContext, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    common_eap = max(context.eap_grids)
    for grid in context.fit_grids:
        fit_manifest = _read_json(
            _fit_cache_dir(out_dir, grid, context.initial_ridge) / "fit_grid_manifest.json"
        )
        score_dir = _score_dir(out_dir, grid, common_eap, context.initial_ridge)
        score = _read_json(score_dir / "score_manifest.json")
        recovery = _pooled_row(score_dir / "recovery.csv", "estimator", "mwle")
        pass_rate = _pooled_row(score_dir / "pass_rate.csv", "estimator", "mwle")
        rows.append(
            {
                "fit_grid": grid,
                "evaluation_eap_grid": common_eap,
                "ridge": context.initial_ridge,
                "all_fits_converged": all(
                    bool(scope["converged"]) for scope in fit_manifest["scopes"]
                ),
                "max_fit_iterations": max(int(scope["n_iter"]) for scope in fit_manifest["scopes"]),
                "full_fit_exportable": next(
                    int(scope["n_exportable"])
                    for scope in fit_manifest["scopes"]
                    if scope["scope"] == "full"
                ),
                "heldout_log_loss": score["pooled_oos"]["log_loss"],
                "heldout_brier": score["pooled_oos"]["brier"],
                "mwle_recovery_r": recovery["r"],
                "mwle_recovery_slope": recovery["slope"],
                "mwle_recovery_mae": recovery["mae"],
                "mwle_pass_rate_mae": pass_rate["mae"],
            }
        )
    summary = pd.DataFrame(rows)
    _write_csv(out_dir / "dense_grid" / "fit_grid_summary.csv", summary)

    gate_config = context.config["dense_grid"]["fit_grid_stability"]
    comparison_name = str(gate_config["comparison"])
    try:
        lower, upper = [int(value) for value in comparison_name.split("_vs_")]
    except ValueError as exc:
        raise DenseGridError(f"invalid fit-grid comparison {comparison_name!r}") from exc
    pairwise = _parameter_comparison(context, out_dir, lower, upper, context.initial_ridge)
    _write_csv(out_dir / "dense_grid" / "fit_grid_pairwise.csv", pairwise)
    lower_oos = pd.read_csv(
        _score_dir(out_dir, lower, common_eap, context.initial_ridge) / "oos_metrics.csv"
    )
    upper_oos = pd.read_csv(
        _score_dir(out_dir, upper, common_eap, context.initial_ridge) / "oos_metrics.csv"
    )
    heldout_stability = paired_fold_metric_stability(
        lower_oos,
        upper_oos,
        gate_config.get("heldout_metric_stability") or {},
    )
    all_converged = bool(
        pairwise[["lower_converged", "upper_converged"]].to_numpy(dtype=bool).all()
    )
    min_a = float(pairwise["a_spearman"].min())
    min_b = float(pairwise["b_spearman"].min())
    min_export = float(pairwise["exportability_agreement"].min())
    gate = {
        "comparison": comparison_name,
        "lower_grid": lower,
        "upper_grid": upper,
        "all_full_and_fold_fits_converged": all_converged,
        "minimum_a_spearman_observed": min_a,
        "minimum_b_spearman_observed": min_b,
        "minimum_item_parameter_spearman_required": float(
            gate_config["minimum_item_parameter_spearman"]
        ),
        "minimum_exportability_agreement_observed": min_export,
        "minimum_exportability_agreement_required": float(
            gate_config["minimum_exportability_agreement"]
        ),
        "heldout_log_loss_lower": float(
            summary.loc[summary["fit_grid"] == lower, "heldout_log_loss"].iloc[0]
        ),
        "heldout_log_loss_upper": float(
            summary.loc[summary["fit_grid"] == upper, "heldout_log_loss"].iloc[0]
        ),
        "heldout_brier_lower": float(
            summary.loc[summary["fit_grid"] == lower, "heldout_brier"].iloc[0]
        ),
        "heldout_brier_upper": float(
            summary.loc[summary["fit_grid"] == upper, "heldout_brier"].iloc[0]
        ),
        "heldout_metric_stability": heldout_stability,
    }
    gate["passed"] = bool(
        (all_converged or not gate_config.get("require_all_fits_converged", True))
        and min_a >= gate["minimum_item_parameter_spearman_required"]
        and min_b >= gate["minimum_item_parameter_spearman_required"]
        and min_export >= gate["minimum_exportability_agreement_required"]
        and heldout_stability["passed"]
    )
    gate["locked_fit_grid"] = lower if gate["passed"] else None
    _write_json(out_dir / "dense_grid" / "fit_grid_stability.json", gate)
    return gate


def summarize_eap_grids(
    context: StudyContext, out_dir: Path, locked_fit_grid: int
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for eap_grid in context.eap_grids:
        score_dir = _score_dir(out_dir, locked_fit_grid, eap_grid, context.initial_ridge)
        score = _read_json(score_dir / "score_manifest.json")
        recovery = _pooled_row(score_dir / "recovery.csv", "estimator", "mwle")
        pass_rate = _pooled_row(score_dir / "pass_rate.csv", "estimator", "mwle")
        rows.append(
            {
                "fit_grid": locked_fit_grid,
                "eap_grid": eap_grid,
                "ridge": context.initial_ridge,
                "heldout_log_loss": score["pooled_oos"]["log_loss"],
                "heldout_brier": score["pooled_oos"]["brier"],
                "mwle_recovery_r": recovery["r"],
                "mwle_recovery_slope": recovery["slope"],
                "mwle_recovery_mae": recovery["mae"],
                "mwle_pass_rate_mae": pass_rate["mae"],
                **score["quantization"],
            }
        )
    _write_csv(out_dir / "dense_grid" / "eap_grid_summary.csv", pd.DataFrame(rows))

    gate_config = context.config["dense_grid"]["eap_grid_stability"]
    comparison_name = str(gate_config["comparison"])
    try:
        lower, upper = [int(value) for value in comparison_name.split("_vs_")]
    except ValueError as exc:
        raise DenseGridError(f"invalid EAP-grid comparison {comparison_name!r}") from exc
    lower_dir = _score_dir(out_dir, locked_fit_grid, lower, context.initial_ridge)
    upper_dir = _score_dir(out_dir, locked_fit_grid, upper, context.initial_ridge)
    lower_frame = pd.read_csv(lower_dir / "per_model.csv")
    upper_frame = pd.read_csv(upper_dir / "per_model.csv")
    merged = lower_frame.merge(
        upper_frame,
        on=["model", "fold"],
        suffixes=("_lower", "_upper"),
        validate="one_to_one",
    )
    shifts = np.abs(
        merged["theta_reference_upper"].to_numpy(dtype=float)
        - merged["theta_reference_lower"].to_numpy(dtype=float)
    )
    orders_identical = bool(
        (merged["scenario_order_lower"] == merged["scenario_order_upper"]).all()
        and (merged["criterion_order_lower"] == merged["criterion_order_upper"]).all()
    )
    lower_recovery = _pooled_row(lower_dir / "recovery.csv", "estimator", "mwle")
    upper_recovery = _pooled_row(upper_dir / "recovery.csv", "estimator", "mwle")
    lower_pass = _pooled_row(lower_dir / "pass_rate.csv", "estimator", "mwle")
    upper_pass = _pooled_row(upper_dir / "pass_rate.csv", "estimator", "mwle")
    upper_quant = _read_json(upper_dir / "score_manifest.json")["quantization"]
    node_pileup_pass = bool(
        int(upper_quant["unique_theta_rounded_6"])
        > int(context.config["dense_grid"]["structure_screen_fit_grid"])
        and float(upper_quant["largest_rounded_value_fraction"]) < 0.5
    )
    gate = {
        "comparison": comparison_name,
        "fit_grid": locked_fit_grid,
        "lower_eap_grid": lower,
        "upper_eap_grid": upper,
        "median_absolute_theta_shift": float(np.median(shifts)),
        "p95_absolute_theta_shift": float(np.quantile(shifts, 0.95)),
        "recovery_correlation_shift": abs(float(upper_recovery["r"]) - float(lower_recovery["r"])),
        "recovery_slope_shift": abs(
            float(upper_recovery["slope"]) - float(lower_recovery["slope"])
        ),
        "theta_mae_shift": abs(float(upper_recovery["mae"]) - float(lower_recovery["mae"])),
        "pass_rate_mae_shift": abs(float(upper_pass["mae"]) - float(lower_pass["mae"])),
        "fixed_bank_orders_identical": orders_identical,
        "node_pileup_pass": node_pileup_pass,
        "upper_grid_quantization": upper_quant,
        "thresholds": gate_config,
    }
    gate["passed"] = bool(
        gate["median_absolute_theta_shift"]
        <= float(gate_config["maximum_median_absolute_theta_shift"])
        and gate["p95_absolute_theta_shift"]
        <= float(gate_config["maximum_p95_absolute_theta_shift"])
        and gate["recovery_correlation_shift"]
        <= float(gate_config["maximum_recovery_correlation_shift"])
        and gate["recovery_slope_shift"] <= float(gate_config["maximum_recovery_slope_shift"])
        and gate["theta_mae_shift"] <= float(gate_config["maximum_theta_mae_shift"])
        and gate["pass_rate_mae_shift"] <= float(gate_config["maximum_pass_rate_mae_shift"])
        and (orders_identical or not gate_config.get("require_identical_fixed_bank_orders", True))
        and node_pileup_pass
    )
    gate["locked_eap_grid"] = lower if gate["passed"] else None
    _write_json(out_dir / "dense_grid" / "eap_grid_stability.json", gate)
    return gate


def summarize_ridges(context: StudyContext, out_dir: Path, fit_grid: int, eap_grid: int) -> None:
    rows: list[dict[str, Any]] = []
    for ridge in context.ridge_candidates:
        fit_manifest = _read_json(
            _fit_cache_dir(out_dir, fit_grid, ridge) / "fit_grid_manifest.json"
        )
        score_dir = _score_dir(out_dir, fit_grid, eap_grid, ridge)
        score = _read_json(score_dir / "score_manifest.json")
        recovery = _pooled_row(score_dir / "recovery.csv", "estimator", "mwle")
        pass_rate = _pooled_row(score_dir / "pass_rate.csv", "estimator", "mwle")
        rows.append(
            {
                "fit_grid": fit_grid,
                "eap_grid": eap_grid,
                "ridge": ridge,
                "all_fits_converged": all(
                    bool(scope["converged"]) for scope in fit_manifest["scopes"]
                ),
                "full_fit_exportable": next(
                    int(scope["n_exportable"])
                    for scope in fit_manifest["scopes"]
                    if scope["scope"] == "full"
                ),
                "heldout_log_loss": score["pooled_oos"]["log_loss"],
                "heldout_brier": score["pooled_oos"]["brier"],
                "mwle_recovery_r": recovery["r"],
                "mwle_recovery_slope": recovery["slope"],
                "mwle_recovery_mae": recovery["mae"],
                "mwle_pass_rate_mae": pass_rate["mae"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("ridge")
    _write_csv(out_dir / "ridge" / "ridge_sensitivity_summary.csv", frame)
    valid = frame[frame["all_fits_converged"] == True]  # noqa: E712
    recommendation = (
        None
        if valid.empty
        else float(
            valid.sort_values(["heldout_log_loss", "heldout_brier", "ridge"]).iloc[0]["ridge"]
        )
    )
    _write_json(
        out_dir / "ridge" / "ridge_sensitivity_manifest.json",
        {
            "generated_at": _utcnow(),
            "fit_grid": fit_grid,
            "eap_grid": eap_grid,
            "candidates": list(context.ridge_candidates),
            "engineering_recommendation_by_heldout_log_loss": recommendation,
            "production_selection_deferred_to_nested_cv": True,
            "note": (
                "This Phase-2 table is a sensitivity screen. Phase 3 must select ridge "
                "inside each outer-training fold, without using outer-test outcomes."
            ),
        },
    )


class Orchestrator:
    def __init__(
        self,
        context: StudyContext,
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

    def _guard_output(self) -> None:
        _assert_not_historical(self.context, self.out_dir)
        if self.out_dir.exists() and not self.resume and not self.plan_only:
            raise DenseGridError(
                f"output directory already exists: {self.out_dir}; pass --resume only for "
                "this exact study, or choose a fresh --out-dir"
            )
        if self.resume and not self.out_dir.is_dir():
            raise DenseGridError(
                f"--resume requested but output directory is absent: {self.out_dir}"
            )

    def prepare(self) -> None:
        self._guard_output()
        if self.plan_only:
            return
        if not self.resume:
            self.out_dir.mkdir(parents=True)
            shutil.copy2(self.context.config_path, self.out_dir / "remediation_config.json")
            shutil.copy2(self.context.split_path, self.out_dir / "frozen_splits.json")
            versions = {}
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
                    "runner": str(Path(__file__).resolve()),
                    "runner_sha256": _sha256(Path(__file__).resolve()),
                    "config": str(self.context.config_path),
                    "config_sha256": _sha256(self.context.config_path),
                    "split_manifest": str(self.context.split_path),
                    "split_manifest_sha256": _sha256(self.context.split_path),
                    "inputs": {
                        "matrix": {
                            "path": str(self.context.matrix),
                            "sha256": _sha256(self.context.matrix),
                        },
                        "rubrics": {
                            "path": str(self.context.rubrics),
                            "sha256": _sha256(self.context.rubrics),
                        },
                        "scenarios": {
                            "path": str(self.context.scenarios),
                            "sha256": _sha256(self.context.scenarios),
                        },
                        "judge_manifest": {
                            "path": str(self.context.judge_manifest),
                            "sha256": _sha256(self.context.judge_manifest),
                            "used_for_new_calls": False,
                        },
                    },
                    "git_commit": _git_value("rev-parse", "HEAD"),
                    "git_branch": _git_value("branch", "--show-current"),
                    "git_dirty": bool(_git_value("status", "--porcelain")),
                    "python": sys.version,
                    "platform": platform.platform(),
                    "packages": versions,
                    "fit_grid_candidates": list(self.context.fit_grids),
                    "eap_grid_candidates": list(self.context.eap_grids),
                    "initial_ridge": self.context.initial_ridge,
                    "ridge_candidates_deferred_until_grid_lock": list(
                        self.context.ridge_candidates
                    ),
                    "historical_outputs_modified": False,
                    "judge_calls": False,
                    "cat_configuration_selection": False,
                },
            )
        else:
            manifest = _read_json(self.out_dir / "study_manifest.json")
            for key, expected in (
                ("config_sha256", _sha256(self.context.config_path)),
                ("split_manifest_sha256", _sha256(self.context.split_path)),
                ("runner_sha256", _sha256(Path(__file__).resolve())),
            ):
                if manifest.get(key) != expected:
                    raise DenseGridError(
                        f"resume {key} differs; preserve this run and start a fresh output"
                    )

    def _persist_commands(self) -> None:
        if not self.plan_only:
            _write_json(self.out_dir / "commands.json", self.commands)

    def run_stage(self, stage: str, command: list[str], outputs: Iterable[Path]) -> None:
        expected = [Path(path) for path in outputs]
        entry = {
            "stage": stage,
            "command": command,
            "outputs": [str(path) for path in expected],
        }
        self.commands.append(entry)
        self._persist_commands()
        if self.plan_only:
            print(json.dumps({**entry, "status": "planned"}, ensure_ascii=False))
            return
        marker_dir = self.out_dir / "stage_markers"
        marker = marker_dir / f"{stage}.json"
        if self.resume and marker.is_file() and all(path.is_file() for path in expected):
            previous = _read_json(marker)
            if previous.get("command") != command:
                raise DenseGridError(
                    f"resume command differs for stage {stage}; use a fresh output directory"
                )
            expected_hashes = previous.get("output_sha256") or {}
            if any(expected_hashes.get(str(path)) != _sha256(path) for path in expected):
                raise DenseGridError(f"resume output hash differs for stage {stage}")
            print(f"[resume] {stage}")
            return
        marker_dir.mkdir(parents=True, exist_ok=True)
        log_path = marker_dir / f"{stage}.log"
        print(f"\n[{stage}]", flush=True)
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
            returncode = process.wait()
        if returncode != 0:
            raise DenseGridError(
                f"stage {stage} failed with exit code {returncode}; see {log_path}"
            )
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise DenseGridError(f"stage {stage} did not produce expected outputs: {missing}")
        _write_json(
            marker,
            {
                **entry,
                "completed_at": _utcnow(),
                "output_sha256": {str(path): _sha256(path) for path in expected},
            },
        )

    def _fit_stage(self, grid: int, ridge: float) -> None:
        cache = _fit_cache_dir(self.out_dir, grid, ridge)
        scopes = ["full"] + [
            f"outer_{int(record['outer_fold'])}" for record in self.context.splits["outer_folds"]
        ]
        outputs = [cache / "fit_grid_manifest.json"]
        for scope in scopes:
            outputs.extend(
                [
                    cache / scope / "fit.npz",
                    cache / scope / "fit_manifest.json",
                    cache / scope / "item_params.csv",
                ]
            )
        self.run_stage(
            f"fit_grid_{grid:03d}_ridge_{_slug_number(ridge)}",
            build_fit_command(self.context, self.out_dir, grid, ridge),
            outputs,
        )

    def _score_stage(self, fit_grid: int, eap_grid: int, ridge: float) -> None:
        score = _score_dir(self.out_dir, fit_grid, eap_grid, ridge)
        self.run_stage(
            f"score_fit_{fit_grid:03d}_eap_{eap_grid:03d}_ridge_{_slug_number(ridge)}",
            build_score_command(self.context, self.out_dir, fit_grid, eap_grid, ridge),
            [
                score / "per_model.csv",
                score / "oos_metrics.csv",
                score / "recovery.csv",
                score / "pass_rate.csv",
                score / "score_manifest.json",
            ],
        )

    def print_plan_footer(self) -> None:
        fit_comparison = str(self.context.config["dense_grid"]["fit_grid_stability"]["comparison"])
        conditional_fit_grid = int(fit_comparison.split("_vs_")[0])
        conditional_eap_commands = [
            {
                "eap_grid": grid,
                "command": build_score_command(
                    self.context,
                    self.out_dir,
                    conditional_fit_grid,
                    grid,
                    self.context.initial_ridge,
                ),
                "cache_policy": "load the locked fit cache; never refit",
            }
            for grid in self.context.eap_grids
        ]
        print(
            json.dumps(
                {
                    "stage": "fit_grid_gate",
                    "comparison": fit_comparison,
                    "on_pass": "lock smaller stable fit grid and run EAP 21/41/81",
                    "on_fail": "stop; do not schedule ridge sweep",
                    "conditional_eap_commands": conditional_eap_commands,
                }
            )
        )
        print(
            json.dumps(
                {
                    "stage": "eap_grid_gate",
                    "comparison": self.context.config["dense_grid"]["eap_grid_stability"][
                        "comparison"
                    ],
                    "on_pass": "lock smaller stable EAP grid",
                    "on_fail": "stop; do not schedule ridge sweep",
                }
            )
        )
        print(
            json.dumps(
                {
                    "stage": "ridge_candidates",
                    "status": "deferred_until_both_grid_locks_exist",
                    "candidates": list(self.context.ridge_candidates),
                    "commands": None,
                }
            )
        )

    def run(self) -> int:
        self.prepare()
        common_eap = max(self.context.eap_grids)
        for fit_grid in self.context.fit_grids:
            self._fit_stage(fit_grid, self.context.initial_ridge)
        for fit_grid in self.context.fit_grids:
            self._score_stage(fit_grid, common_eap, self.context.initial_ridge)
        if self.plan_only:
            self.print_plan_footer()
            return 0

        fit_gate = summarize_fit_grids(self.context, self.out_dir)
        if not fit_gate["passed"]:
            self._finish("blocked_fit_grid_stability", {"fit_gate": fit_gate})
            raise DenseGridError(
                "25-vs-41 fit-grid stability gate failed; no EAP lock or ridge run was scheduled"
            )
        locked_fit = int(fit_gate["locked_fit_grid"])
        for eap_grid in self.context.eap_grids:
            self._score_stage(locked_fit, eap_grid, self.context.initial_ridge)
        eap_gate = summarize_eap_grids(self.context, self.out_dir, locked_fit)
        if not eap_gate["passed"]:
            self._finish(
                "blocked_eap_grid_stability",
                {"fit_gate": fit_gate, "eap_gate": eap_gate},
            )
            raise DenseGridError(
                "41-vs-81 EAP-grid stability gate failed; no ridge run was scheduled"
            )
        locked_eap = int(eap_gate["locked_eap_grid"])
        _write_json(
            self.out_dir / "dense_grid" / "grid_lock.json",
            {
                "locked_at": _utcnow(),
                "fit_grid": locked_fit,
                "eap_grid": locked_eap,
                "initial_ridge": self.context.initial_ridge,
                "fit_gate": fit_gate,
                "eap_gate": eap_gate,
                "ridge_scheduled_only_after_this_lock": True,
            },
        )

        # This block is deliberately unreachable until both lock files pass.
        for ridge in self.context.ridge_candidates:
            if math.isclose(ridge, self.context.initial_ridge):
                # The initial-ridge fit and locked-grid score are already cached.
                continue
            self._fit_stage(locked_fit, ridge)
            self._score_stage(locked_fit, locked_eap, ridge)
        summarize_ridges(self.context, self.out_dir, locked_fit, locked_eap)
        self._finish(
            "phase2_complete",
            {
                "fit_grid": locked_fit,
                "eap_grid": locked_eap,
                "ridge_candidates_completed": list(self.context.ridge_candidates),
                "production_ridge_selection": "deferred_to_nested_cv",
            },
        )
        return 0

    def _finish(self, status: str, result: Mapping[str, Any]) -> None:
        manifest_path = self.out_dir / "study_manifest.json"
        manifest = _read_json(manifest_path)
        manifest.update(
            {
                "completed_at": _utcnow(),
                "status": status,
                "result": result,
                "historical_outputs_modified": False,
            }
        )
        _write_json(manifest_path, manifest)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--worker",
        choices=("fit-grid", "score-grid"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fit-grid", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eap-grid", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ridge", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fit-cache-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.plan_only and args.resume:
        raise DenseGridError("--plan-only and --resume are mutually exclusive")
    context = load_context(args.config, args.split_manifest)
    out_dir = (args.out_dir or context.default_out_dir).resolve()
    if args.worker:
        if args.plan_only or args.resume:
            raise DenseGridError("internal workers do not accept --plan-only/--resume")
        if args.fit_grid is None or args.ridge is None:
            raise DenseGridError("internal worker requires --fit-grid and --ridge")
        _assert_not_historical(context, out_dir)
        if args.worker == "fit-grid":
            if args.eap_grid is not None or args.fit_cache_dir is not None:
                raise DenseGridError("fit-grid worker cannot receive EAP/cache arguments")
            return worker_fit_grid(context, out_dir, args.fit_grid, args.ridge)
        if args.eap_grid is None or args.fit_cache_dir is None:
            raise DenseGridError("score-grid worker requires --eap-grid and --fit-cache-dir")
        return worker_score_grid(
            context,
            out_dir,
            args.fit_cache_dir,
            args.fit_grid,
            args.eap_grid,
            args.ridge,
        )
    if any(
        value is not None
        for value in (args.fit_grid, args.eap_grid, args.ridge, args.fit_cache_dir)
    ):
        raise DenseGridError("worker-only arguments were supplied without --worker")
    return Orchestrator(
        context,
        out_dir,
        resume=args.resume,
        plan_only=args.plan_only,
    ).run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DenseGridError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from None
