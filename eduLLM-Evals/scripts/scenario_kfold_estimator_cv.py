"""Person-level k-fold validation of scenario CAT ability estimators.

For each fold this script:

1. refits the selected confirmatory M2PL structure on TRAIN tutor models only;
2. freezes those fold-specific item parameters and latent correlation;
3. computes a full-bank EAP reference for each held-out tutor model;
4. replays the real scenario-level CAT engine on that held-out model; and
5. compares online-Gaussian, administered-set batch EAP, and MWLE estimates with
   the fold-specific full-bank EAP reference.

The response matrix already contains binary judge decisions, so this command makes no
tutor-model or judge-model calls.  It supports any validated ``SkillStructure`` used by
``scripts/kfold_cv_mirt.py``.  The headline outputs are recovery correlation, slope, MAE,
and signed bias.  A good estimator needs high correlation *and* slope near one; correlation
alone can hide severe shrinkage.

Example (native five-dimensional InFoBench structure)::

    python scripts/scenario_kfold_estimator_cv.py \
      --matrix runs/calibration/.../response_matrix.csv \
      --rubrics data/InFoBench/rubrics.jsonl \
      --scenarios data/InFoBench/scenarios.jsonl \
      --skills content,format,number,style,linguistic \
      --fit-grid 5 --eap-grid 5 --negative-policy drop \
      --out-dir runs/calibration/.../scenario_kfold_native5

This study is intentionally distinct from the cell-level k-fold script.  Both refit item
parameters on training people, but this script asks whether a shortened, scenario-level
adaptive test recovers the held-out person's full-bank ability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kfold_cv_mirt as cell_cv  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402
from tutor_cat.skill_structure import SkillStructure  # noqa: E402

cm = cell_cv.cm

DEFAULT_MAX_GRID_NODES = 5_000
ESTIMATORS = ("online", "eap", "mwle")
PASS_RATE_ESTIMATORS = ("online", "eap", "mwle", "full_eap")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def source_records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in cm.cp.read_jsonl(path):
        criterion_id = str(record.get("criterion_id") or "")
        if not criterion_id or criterion_id in records:
            raise scat.OfflineStudyError(
                f"blank or duplicate criterion_id in source rubrics: {criterion_id!r}"
            )
        records[criterion_id] = record
    return records


def build_fold_bank(
    fit: Mapping[str, Any],
    structure: SkillStructure,
    source_records: Mapping[str, dict[str, Any]],
    *,
    negative_policy: str,
) -> tuple[scat.FittedBank, dict[str, Any]]:
    """Convert one training-only MIRT fit into the generic fitted-bank schema."""

    if negative_policy not in {"error", "drop", "keep"}:
        raise ValueError("negative_policy must be 'error', 'drop', or 'keep'")
    items = [str(item) for item in fit["items"]]
    A = np.asarray(fit["A"], dtype=float)
    b = np.asarray(fit["b"], dtype=float)
    R = np.asarray(fit["R"], dtype=float)
    if A.shape != (len(items), structure.n_dims) or b.shape != (len(items),):
        raise scat.OfflineStudyError("fold fit item arrays do not match its skill structure")
    if list(fit.get("dim_labels") or structure.labels) != list(structure.labels):
        raise scat.OfflineStudyError("fold fit dimension labels do not match SkillStructure")

    missing = [criterion_id for criterion_id in items if criterion_id not in source_records]
    if missing:
        raise scat.OfflineStudyError(
            f"fold fit contains {len(missing)} item(s) absent from source rubrics; "
            f"first: {missing[:10]}"
        )
    q_source = np.asarray(
        [
            [int((source_records[cid].get("q_mapping") or {})[s]) for s in structure.source_skills]
            for cid in items
        ],
        dtype=int,
    )
    Q = structure.transform_q(q_source)
    bad_parameter = ~np.isfinite(b) | ~np.all(np.isfinite(A), axis=1)
    nonpositive = np.asarray(
        [bool(np.any(A[j, Q[j] == 1] <= 0)) for j in range(len(items))], dtype=bool
    )
    affected = [items[j] for j in np.where(bad_parameter | nonpositive)[0]]
    if bool(bad_parameter.any()):
        bad = [items[j] for j in np.where(bad_parameter)[0]]
        raise scat.OfflineStudyError(
            f"fold fit has non-finite parameters for {len(bad)} item(s); first: {bad[:10]}"
        )
    if affected and negative_policy == "error":
        raise scat.OfflineStudyError(
            f"fold fit has {len(affected)} item(s) with nonpositive modeled loadings; "
            f"first: {affected[:10]}. Choose an explicit policy."
        )
    keep = ~nonpositive if negative_policy == "drop" else np.ones(len(items), dtype=bool)
    if not bool(keep.any()):
        raise scat.OfflineStudyError("no fold-fitted items remain after loading policy")
    if np.any(Q[keep].sum(axis=0) == 0):
        empty = [structure.labels[k] for k in np.where(Q[keep].sum(axis=0) == 0)[0]]
        raise scat.OfflineStudyError(
            f"negative-loading policy leaves no item measuring dimension(s) {empty}"
        )

    selected = np.where(keep)[0]
    records: list[dict[str, Any]] = []
    for j in selected:
        source = dict(source_records[items[j]])
        source["q_modeled"] = {
            dim: int(Q[j, k]) for k, dim in enumerate(structure.labels)
        }
        source["discrimination"] = {
            dim: float(A[j, k]) for k, dim in enumerate(structure.labels)
        }
        source["difficulty"] = float(b[j])
        source["irt_params"] = {
            "source": "calibrated-m2pl-kfold-train-only",
            "calibrated": True,
            "skills_order": list(structure.labels),
            "skill_structure": structure.as_dict(),
            "latent_correlation": R.tolist(),
        }
        records.append(source)

    Q_kept = Q[selected]
    A_kept = Q_kept * A[selected]
    bank = scat.FittedBank(
        records=records,
        dims=structure.labels,
        criterion_ids=tuple(items[j] for j in selected),
        scenario_ids=tuple(str(source_records[items[j]]["scenario_id"]) for j in selected),
        Q=Q_kept,
        A=A_kept,
        b=b[selected],
        latent_correlation=scat._validate_correlation(R, structure.n_dims),
        source_path="training-fold in-memory fit",
        dropped_negative_items=tuple(affected if negative_policy == "drop" else ()),
    )
    diagnostics = {
        "n_items_before_policy": len(items),
        "n_items_after_policy": bank.n_items,
        "n_nonpositive_items": len(affected),
        "nonpositive_items": affected,
        "negative_policy": negative_policy,
        "items_per_dimension_after_policy": {
            dim: int(bank.Q[:, k].sum()) for k, dim in enumerate(bank.dims)
        },
    }
    return bank, diagnostics


def fold_item_frame(bank: scat.FittedBank) -> pd.DataFrame:
    data: dict[str, Any] = {
        "criterion_id": list(bank.criterion_ids),
        "scenario_id": list(bank.scenario_ids),
        "b": bank.b,
    }
    for k, dim in enumerate(bank.dims):
        data[f"q_{dim}"] = bank.Q[:, k]
        data[f"a_{dim}"] = bank.A[:, k]
    return pd.DataFrame(data)


def recovery_stats(
    reference: Sequence[float], estimate: Sequence[float]
) -> dict[str, float | int | None]:
    x = np.asarray(reference, dtype=float)
    y = np.asarray(estimate, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    result: dict[str, float | int | None] = {
        "n": int(x.size),
        "r": None,
        "slope": None,
        "mae": None,
        "bias": None,
    }
    if x.size == 0:
        return result
    result["mae"] = float(np.mean(np.abs(y - x)))
    result["bias"] = float(np.mean(y - x))
    if x.size >= 3 and float(x.std()) > 0:
        result["slope"] = float(np.polyfit(x, y, 1)[0])
        if float(y.std()) > 0:
            result["r"] = float(np.corrcoef(x, y)[0, 1])
    return result


def recovery_table(frame: pd.DataFrame, dims: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    successful = frame[frame["status"] == "ok"]
    folds: list[int | str] = ["pooled"] + sorted(successful["fold"].unique().tolist())
    for fold in folds:
        subset = successful if fold == "pooled" else successful[successful["fold"] == fold]
        for estimator in ESTIMATORS:
            eligible = subset
            if estimator == "mwle":
                eligible = eligible[eligible["mwle_converged"] == True]  # noqa: E712
            for dim in dims:
                ref_col = f"theta_ref_{dim}"
                est_col = f"theta_{estimator}_{dim}"
                metrics = (
                    recovery_stats(eligible[ref_col], eligible[est_col])
                    if ref_col in eligible and est_col in eligible
                    else recovery_stats([], [])
                )
                rows.append(
                    {
                        "fold": fold,
                        "estimator": estimator,
                        "dimension": dim,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def pass_rate_stats(
    observed: Sequence[float], predicted: Sequence[float]
) -> dict[str, float | int | None]:
    actual = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(estimate)
    actual = actual[valid]
    estimate = estimate[valid]
    result: dict[str, float | int | None] = {
        "n": int(actual.size),
        "mae": None,
        "bias": None,
        "r": None,
    }
    if actual.size == 0:
        return result
    result["mae"] = float(np.mean(np.abs(estimate - actual)))
    result["bias"] = float(np.mean(estimate - actual))
    if actual.size >= 3 and float(actual.std()) > 0 and float(estimate.std()) > 0:
        result["r"] = float(np.corrcoef(actual, estimate)[0, 1])
    return result


def pass_rate_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate p-IRT predicted-vs-observed pass rates, pooled and by fold."""

    rows: list[dict[str, Any]] = []
    successful = frame[frame["status"] == "ok"]
    folds: list[int | str] = ["pooled"] + sorted(successful["fold"].unique().tolist())
    for fold in folds:
        subset = successful if fold == "pooled" else successful[successful["fold"] == fold]
        for estimator in PASS_RATE_ESTIMATORS:
            eligible = subset
            if estimator == "mwle":
                eligible = eligible[eligible["mwle_converged"] == True]  # noqa: E712
            observed_col = "observed_fitted_pass_rate"
            predicted_col = f"pirt_predicted_pass_rate_{estimator}"
            metrics = (
                pass_rate_stats(eligible[observed_col], eligible[predicted_col])
                if observed_col in eligible and predicted_col in eligible
                else pass_rate_stats([], [])
            )
            rows.append({"fold": fold, "estimator": estimator, **metrics})
    return pd.DataFrame(rows)


def _base_failure_row(
    model: str,
    fold: int,
    train_count: int,
    test_count: int,
    fit: Mapping[str, Any],
    policy_diag: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "model": model,
        "fold": fold,
        "status": "replay_error",
        "error": f"{type(error).__name__}: {error}",
        "n_train": train_count,
        "n_test": test_count,
        "fit_converged": bool(fit["converged"]),
        "n_items_fit_before_policy": int(policy_diag["n_items_before_policy"]),
        "n_items_fit_after_policy": int(policy_diag["n_items_after_policy"]),
        "n_nonpositive_items": int(policy_diag["n_nonpositive_items"]),
        "mwle_converged": False,
    }


def run_fold(
    fold_index: int,
    train_models: Sequence[str],
    test_models: Sequence[str],
    matrix: pd.DataFrame,
    q_by: Mapping[str, np.ndarray],
    structure: SkillStructure,
    source_records: Mapping[str, dict[str, Any]],
    scenarios: Mapping[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], scat.FittedBank]:
    """Fit on TRAIN and replay every held-out model in one fold."""

    fit = cell_cv.fit_structure(matrix.loc[list(train_models)], q_by, args, structure)
    if not fit["converged"] and not args.allow_unconverged_fit:
        raise scat.OfflineStudyError(
            f"fold {fold_index} MIRT fit did not converge; refusing held-out scoring"
        )
    fold_bank, policy_diag = build_fold_bank(
        fit, structure, source_records, negative_policy=args.negative_policy
    )
    eap_grid = args.eap_grid if args.eap_grid is not None else args.grid
    quadrature_cap = (
        max(args.max_grid_nodes, eap_grid**structure.n_dims)
        if args.allow_large_grid
        else args.max_grid_nodes
    )
    quadrature = scat.build_quadrature(
        structure.n_dims,
        eap_grid,
        fold_bank.latent_correlation,
        max_nodes=quadrature_cap,
    )
    spec = scat.RunSpec(
        seed=args.seed,
        top_n=args.top_n,
        max_se=args.max_se,
        min_evals_per_skill=args.min_evals_per_skill,
        min_scenarios=args.min_scenarios,
        max_scenarios=args.max_scenarios,
        selection=args.selection,
        mode="cat",
    )
    rows: list[dict[str, Any]] = []
    for model in test_models:
        try:
            result = scat.run_recorded_model(
                str(model),
                matrix.loc[model],
                fold_bank,
                scenarios,
                quadrature,
                spec,
                mwle_ridge=args.mwle_ridge,
            )
        except (scat.OfflineStudyError, ValueError, np.linalg.LinAlgError) as error:
            rows.append(
                _base_failure_row(
                    str(model), fold_index, len(train_models), len(test_models),
                    fit, policy_diag, error,
                )
            )
            continue
        row: dict[str, Any] = {
            "model": str(model),
            "fold": fold_index,
            "status": "ok",
            "error": "",
            "n_train": len(train_models),
            "n_test": len(test_models),
            "fit_converged": bool(fit["converged"]),
            "fit_iterations": int(fit["n_iter"]),
            "fit_loglik": float(fit["loglik"]),
            "n_items_fit_before_policy": int(policy_diag["n_items_before_policy"]),
            "n_items_fit_after_policy": int(policy_diag["n_items_after_policy"]),
            "n_nonpositive_items": int(policy_diag["n_nonpositive_items"]),
            "stop_reason": result["stop_reason"],
            "precision_reached": bool(result["precision_reached"]),
            "scenarios_administered": int(result["scenarios_administered"]),
            "criteria_administered": int(result["criteria_administered"]),
            "mwle_converged": bool(result["mwle_converged"]),
            "mwle_message": result["mwle_message"],
            "scenario_order": json.dumps(result["scenario_order"], ensure_ascii=False),
            "criterion_order": json.dumps(result["criterion_order"], ensure_ascii=False),
            "n_observed_fitted_items": int(result["n_observed_fitted_items"]),
            "observed_fitted_pass_rate": float(result["observed_fitted_pass_rate"]),
        }
        for estimator in PASS_RATE_ESTIMATORS:
            check = result["pass_rate_checks"][estimator]
            row[f"pirt_predicted_pass_rate_{estimator}"] = check["predicted_pass_rate"]
            row[f"pirt_pass_rate_error_{estimator}"] = check["error"]
        for dim in structure.labels:
            row[f"theta_ref_{dim}"] = result["theta_full_eap"][dim]
            row[f"se_ref_{dim}"] = result["se_full_eap"][dim]
            for estimator in ESTIMATORS:
                row[f"theta_{estimator}_{dim}"] = result[f"theta_{estimator}"][dim]
                row[f"se_{estimator}_{dim}"] = result[f"se_{estimator}"][dim]
        rows.append(row)
    fold_summary = {
        "fold": fold_index,
        "train_models": list(train_models),
        "test_models": list(test_models),
        "fit": {
            "n_items": len(fit["items"]),
            "loglik": float(fit["loglik"]),
            "n_params": int(fit["n_params"]),
            "n_iter": int(fit["n_iter"]),
            "converged": bool(fit["converged"]),
            "latent_correlation": np.asarray(fit["R"]).tolist(),
            "diagnostics": fit.get("diag") or {},
        },
        "bank_policy": policy_diag,
        "n_replay_errors": sum(row["status"] != "ok" for row in rows),
        "n_mwle_failures": sum(
            row["status"] == "ok" and not row["mwle_converged"] for row in rows
        ),
    }
    return rows, fold_summary, fold_bank


def _validate_args(args: argparse.Namespace, n_models: int, n_dims: int) -> None:
    if args.k < 2 or args.k > n_models:
        raise ValueError(f"--k must be between 2 and the {n_models} available models")
    if args.grid < 2 or (args.eap_grid is not None and args.eap_grid < 2):
        raise ValueError("fit/eap grids must have at least two nodes per dimension")
    if args.max_iter < 1 or args.tol <= 0 or args.ridge < 0 or args.mwle_ridge <= 0:
        raise ValueError("invalid fitter or MWLE optimization settings")
    if args.top_n < 1 or args.max_scenarios < 1:
        raise ValueError("top_n and max_scenarios must be positive")
    if args.min_scenarios < 0 or args.min_evals_per_skill < 0:
        raise ValueError("minimum scenario/evaluation counts cannot be negative")
    fit_nodes = args.grid**n_dims
    eap_nodes = (args.eap_grid or args.grid) ** n_dims
    if max(fit_nodes, eap_nodes) > args.max_grid_nodes and not args.allow_large_grid:
        raise ValueError(
            f"requested up to {max(fit_nodes, eap_nodes):,} quadrature nodes, above "
            f"--max-grid-nodes={args.max_grid_nodes:,}"
        )


def run(args: argparse.Namespace) -> int:
    for path in (args.matrix, args.rubrics, args.scenarios):
        if not path.is_file():
            raise FileNotFoundError(path)
    named_outputs = (
        "oos_per_model.csv",
        "recovery_per_fold.csv",
        "pass_rate_calibration.csv",
        "metrics_aggregate.json",
        "fold_assignments.json",
        "manifest.json",
        "failure_cases.csv",
    )
    if not args.overwrite and any((args.out_dir / name).exists() for name in named_outputs):
        raise scat.OfflineStudyError(
            f"outputs already exist under {args.out_dir}; pass --overwrite deliberately"
        )

    source_skills = cm.configure_skills(args.skills)
    structure = cell_cv.build_structure(
        tuple(source_skills), args.dimensions, args.structure_name
    )
    matrix = cm.load_matrix_strict(args.matrix)
    q_by = cm.load_q_matrix(args.rubrics)
    alignment = cm.validate_matrix_bank_alignment(
        matrix, q_by, args.require_complete_bank
    )
    _validate_args(args, len(matrix), structure.n_dims)
    source_records = source_records_by_id(args.rubrics)
    scenarios = scat.load_scenario_records(args.scenarios)

    folds = (
        cell_cv.make_stratified_folds(matrix, args.k, args.seed)
        if args.fold_strategy == "stratified"
        else cell_cv.make_folds(list(matrix.index), args.k, args.seed)
    )
    fold_of = {model: fold for fold, models in enumerate(folds) for model in models}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    models = list(matrix.index)
    for fold_index, test_models in enumerate(folds):
        test_set = set(test_models)
        train_models = [model for model in models if model not in test_set]
        print(
            f"\n=== fold {fold_index}: train={len(train_models)} test={len(test_models)} ===",
            flush=True,
        )
        rows, summary, fold_bank = run_fold(
            fold_index,
            train_models,
            test_models,
            matrix,
            q_by,
            structure,
            source_records,
            scenarios,
            args,
        )
        all_rows.extend(rows)
        fold_summaries.append(summary)
        fold_item_frame(fold_bank).to_csv(
            args.out_dir / f"fold_{fold_index}_item_params.csv", index=False
        )

    per_model = pd.DataFrame(all_rows).sort_values(["fold", "model"])
    per_model.to_csv(args.out_dir / "oos_per_model.csv", index=False)
    recovery = recovery_table(per_model, structure.labels)
    recovery.to_csv(args.out_dir / "recovery_per_fold.csv", index=False)
    pass_rate = pass_rate_table(per_model)
    pass_rate.to_csv(args.out_dir / "pass_rate_calibration.csv", index=False)

    assignment_rows = [
        {
            "model": model,
            "fold": fold_of[model],
            "observed_pass_rate": float(matrix.loc[model].mean(skipna=True)),
        }
        for model in models
    ]
    pd.DataFrame(assignment_rows).sort_values(["fold", "model"]).to_csv(
        args.out_dir / "fold_assignments.csv", index=False
    )
    (args.out_dir / "fold_assignments.json").write_text(
        json.dumps(
            {
                "k": args.k,
                "seed": args.seed,
                "strategy": args.fold_strategy,
                "folds": {str(i): fold for i, fold in enumerate(folds)},
                "model_to_fold": fold_of,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ok = per_model[per_model["status"] == "ok"]
    pooled = recovery[recovery["fold"] == "pooled"].drop(columns=["fold"])
    pooled_pass_rate = pass_rate[pass_rate["fold"] == "pooled"].drop(columns=["fold"])
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric_kind": "heldout-person scenario-CAT theta recovery",
        "reference": (
            "full-bank correlated-prior EAP for each held-out model under its "
            "training-fold item parameters"
        ),
        "n_models": len(models),
        "n_successful_replays": int(len(ok)),
        "n_replay_errors": int((per_model["status"] != "ok").sum()),
        "n_mwle_failures": int(
            ((per_model["status"] == "ok") & (~per_model["mwle_converged"].fillna(False))).sum()
        ),
        "precision_reached": int(ok["precision_reached"].sum()) if len(ok) else 0,
        "mean_scenarios": float(ok["scenarios_administered"].mean()) if len(ok) else None,
        "mean_criteria": float(ok["criteria_administered"].mean()) if len(ok) else None,
        "pooled_recovery": pooled.to_dict(orient="records"),
        "pooled_pass_rate_calibration": pooled_pass_rate.to_dict(orient="records"),
        "folds": fold_summaries,
        "notes": [
            "Held-out people never contribute to their fold's item calibration.",
            "Full-bank EAP and shortened CAT estimates use only that held-out person's responses.",
            "MWLE recovery excludes rows where mwle_converged is false.",
            "bias is estimate minus full-bank EAP; slope near one indicates low compression.",
            "Pass-rate bias is p-IRT predicted minus observed fitted-bank pass rate.",
            "Full-EAP pass-rate calibration is posterior reconstruction; the shortened "
            "estimators are the relevant CAT observable checks.",
        ],
    }
    (args.out_dir / "metrics_aggregate.json").write_text(
        json.dumps(_json_ready(metrics), indent=2), encoding="utf-8"
    )
    failure_mask = (per_model["status"] != "ok") | (
        (per_model["status"] == "ok") & (~per_model["mwle_converged"].fillna(False))
    )
    per_model[failure_mask].to_csv(args.out_dir / "failure_cases.csv", index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/scenario_kfold_estimator_cv.py",
        "argv": sys.argv[1:],
        "inputs": {
            "matrix": str(args.matrix),
            "matrix_sha256": _sha256(args.matrix),
            "rubrics": str(args.rubrics),
            "rubrics_sha256": _sha256(args.rubrics),
            "scenarios": str(args.scenarios),
            "scenarios_sha256": _sha256(args.scenarios),
        },
        "matrix_bank_alignment": alignment,
        "structure": structure.as_dict(),
        "config": {
            "k": args.k,
            "seed": args.seed,
            "fold_strategy": args.fold_strategy,
            "fit_grid": args.grid,
            "eap_grid": args.eap_grid or args.grid,
            "estimate_latent_corr": args.estimate_latent_corr,
            "ridge": args.ridge,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "negative_policy": args.negative_policy,
            "allow_unconverged_fit": args.allow_unconverged_fit,
            "selection": args.selection,
            "top_n": args.top_n,
            "max_se": args.max_se,
            "min_evals_per_skill": args.min_evals_per_skill,
            "min_scenarios": args.min_scenarios,
            "max_scenarios": args.max_scenarios,
            "mwle_ridge": args.mwle_ridge,
        },
        "folds": fold_summaries,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2), encoding="utf-8"
    )
    print(f"\nwrote scenario-level k-fold outputs -> {args.out_dir}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--skills",
        default=None,
        help="ordered source skills; InFoBench: content,format,number,style,linguistic",
    )
    parser.add_argument(
        "--dimensions",
        default=None,
        help="modeled partition: label=source+source,label2=source",
    )
    parser.add_argument("--structure-name", default=None)
    parser.add_argument("--require-complete-bank", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--fold-strategy", choices=("stratified", "random"), default="stratified"
    )
    parser.add_argument("--fit-grid", dest="grid", type=int, default=5)
    parser.add_argument("--eap-grid", type=int, default=None)
    parser.add_argument("--max-grid-nodes", type=int, default=DEFAULT_MAX_GRID_NODES)
    parser.add_argument("--allow-large-grid", action="store_true")
    parser.add_argument(
        "--estimate-latent-corr",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--negative-policy", choices=("error", "drop", "keep"), default="error"
    )
    parser.add_argument("--allow-unconverged-fit", action="store_true")
    parser.add_argument("--selection", choices=("trace", "dopt"), default="trace")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-se", type=float, default=0.30)
    parser.add_argument("--min-evals-per-skill", type=int, default=15)
    parser.add_argument("--min-scenarios", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int, default=50)
    parser.add_argument("--mwle-ridge", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    try:
        return run(build_argparser().parse_args())
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        scat.OfflineStudyError,
        cm.CalibrationError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
