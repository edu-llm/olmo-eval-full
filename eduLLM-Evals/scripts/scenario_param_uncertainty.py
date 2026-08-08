"""Estimate CAT ability uncertainty caused by calibration-sample variability.

This script performs an explicit nonparametric *model bootstrap*. Each replicate
resamples calibration tutor models with replacement, refits the selected M2PL item
bank, and re-estimates every tutor's ability on the scenario set selected by the
original fitted bank. Holding the administered set fixed isolates item-parameter
uncertainty from the separate scenario-path uncertainty experiment.

``--se-targets`` can replay several CAT precision targets in one study.  The
expensive bootstrap item-bank refit is performed once per bootstrap replicate and
then reused to re-estimate abilities on each target-specific administered set.
This makes an SE-target sweep cost roughly ``B`` fits, not ``B * n_targets`` fits.

For estimator ``e`` and dimension ``d`` it reports::

    SE_total[e,d] = sqrt(SE_ability[e,d]**2 + SE_param[e,d]**2)

``SE_ability`` is conditional on the original frozen item parameters;
``SE_param`` is the across-bootstrap SD of re-estimated theta. The quadrature sum
uses the conventional independence approximation and is labeled as such. Batch EAP
uses each bootstrap fit's estimated correlated latent prior; MWLE is also rerun and
its convergence failures remain visible.

The response matrix already contains judge verdicts, so this is CPU-only and makes
no tutor-model or judge-model calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


cm = cell_cv.cm
ESTIMATORS = ("eap", "mwle")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
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
    return value


def _sample_signature(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def resolve_se_targets(args: argparse.Namespace) -> tuple[float, ...]:
    """Return a validated, de-duplicated precision-target sequence."""
    raw = args.se_targets
    if raw is None:
        targets = [float(args.max_se)]
    else:
        try:
            targets = [float(part.strip()) for part in raw.split(",") if part.strip()]
        except ValueError as error:
            raise ValueError("--se-targets must be a comma-separated list of numbers") from error
    if not targets:
        raise ValueError("--se-targets must contain at least one value")
    if any(not math.isfinite(target) or target <= 0 for target in targets):
        raise ValueError("every SE target must be finite and positive")
    if len(set(targets)) != len(targets):
        raise ValueError("--se-targets contains duplicate values")
    slugs = [_target_slug(target) for target in targets]
    if len(set(slugs)) != len(slugs):
        raise ValueError("SE targets are too close to produce distinct output filenames")
    return tuple(targets)


def _target_slug(target: float) -> str:
    """Stable filename label for a target, e.g. 0.20 -> 0p20."""
    return f"{target:.2f}".replace("-", "m").replace(".", "p")


def _variant_items(y: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Columns with at least one observed pass and failure in this bootstrap."""
    successes = np.where(observed, y, 0.0).sum(axis=0)
    counts = observed.sum(axis=0)
    return (counts >= 2) & (successes > 0) & (successes < counts)


def bootstrap_refit(
    matrix_values: np.ndarray,
    bank: scat.FittedBank,
    sample_indices: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Refit one bootstrap bank and return parameters aligned to kept source items."""
    sampled = matrix_values[sample_indices]
    observed = np.isfinite(sampled)
    y = np.nan_to_num(sampled, nan=0.0)
    variant = _variant_items(y, observed)
    keep = np.where(variant)[0]
    if keep.size == 0:
        raise scat.OfflineStudyError("bootstrap has no variant fitted-bank items")
    q_fit = bank.Q[keep]
    empty_dims = np.where(q_fit.sum(axis=0) == 0)[0]
    if empty_dims.size:
        raise scat.OfflineStudyError(
            "bootstrap variant filter leaves no item for dimensions "
            f"{[bank.dims[index] for index in empty_dims]}"
        )

    fit = cm.fit_m2pl_em(
        y[:, keep],
        observed[:, keep],
        q_fit,
        args.fit_grid,
        estimate_corr=bank.n_dims > 1,
        ridge=args.ridge,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    if not fit["converged"] and not args.allow_unconverged_fit:
        raise scat.OfflineStudyError("bootstrap M2PL fit did not converge")
    A = np.asarray(fit["A"], dtype=float)
    b = np.asarray(fit["b"], dtype=float)
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise scat.OfflineStudyError("bootstrap M2PL fit returned non-finite parameters")
    nonpositive = np.asarray(
        [bool(np.any(A[row][q_fit[row] == 1] <= 0)) for row in range(len(keep))],
        dtype=bool,
    )
    affected = keep[nonpositive]
    if affected.size and args.negative_policy == "error":
        raise scat.OfflineStudyError(
            f"bootstrap fit has {affected.size} nonpositive-loading item(s)"
        )
    if args.negative_policy == "drop":
        keep_rows = ~nonpositive
        keep = keep[keep_rows]
        q_fit = q_fit[keep_rows]
        A = A[keep_rows]
        b = b[keep_rows]
    if keep.size == 0 or np.any(q_fit.sum(axis=0) == 0):
        raise scat.OfflineStudyError(
            "bootstrap loading policy leaves no usable item for at least one dimension"
        )
    return {
        "item_indices": keep,
        "A": q_fit * A,
        "b": b,
        "R": np.asarray(fit["R"], dtype=float),
        "n_iter": int(fit["n_iter"]),
        "converged": bool(fit["converged"]),
        "n_variant_before_loading_policy": int(variant.sum()),
        "n_nonpositive": int(nonpositive.sum()),
    }


def summarize_uncertainty(
    base_rows: pd.DataFrame,
    draw_rows: pd.DataFrame,
    dims: Sequence[str],
    n_boot: int,
    min_valid_fraction: float,
) -> pd.DataFrame:
    """Build long-form SE_ability/SE_param/SE_total results."""
    rows: list[dict[str, Any]] = []
    required = max(2, int(math.ceil(min_valid_fraction * n_boot)))
    for _, base in base_rows.iterrows():
        model = str(base["model"])
        model_draws = draw_rows[
            (draw_rows["model"] == model) & (draw_rows["status"] == "ok")
        ]
        for estimator in ESTIMATORS:
            base_estimator_valid = (
                estimator != "mwle" or bool(base.get("mwle_converged", False))
            )
            eligible = model_draws
            if estimator == "mwle":
                eligible = eligible[eligible["mwle_converged"] == True]  # noqa: E712
            for dim in dims:
                values = pd.to_numeric(
                    eligible.get(f"theta_{estimator}_{dim}", pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna().to_numpy(float)
                se_param = float(values.std(ddof=1)) if values.size >= 2 else None
                se_ability = float(base[f"se_{estimator}_{dim}"])
                se_total = (
                    math.sqrt(se_ability**2 + se_param**2)
                    if se_param is not None and math.isfinite(se_ability)
                    else None
                )
                rows.append(
                    {
                        "model": model,
                        "estimator": estimator,
                        "dimension": dim,
                        "theta": float(base[f"theta_{estimator}_{dim}"]),
                        "se_ability": se_ability,
                        "se_param": se_param,
                        "se_total": se_total,
                        "n_valid_bootstrap": int(values.size),
                        "n_failed_bootstrap": int(n_boot - values.size),
                        "min_valid_required": required,
                        "base_estimator_valid": base_estimator_valid,
                        "reliable": bool(
                            base_estimator_valid and values.size >= required
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _validate_args(args: argparse.Namespace, bank: scat.FittedBank) -> None:
    resolve_se_targets(args)
    if args.n_boot < 2:
        raise ValueError("--n-boot must be at least 2")
    if not 0.0 < args.min_valid_fraction <= 1.0:
        raise ValueError("--min-valid-fraction must be in (0,1]")
    if args.fit_grid < 2 or args.eap_grid < 2:
        raise ValueError("fit/eap grids must have at least two nodes per dimension")
    if max(args.fit_grid**bank.n_dims, args.eap_grid**bank.n_dims) > args.max_grid_nodes:
        raise ValueError("fit/eap quadrature exceeds --max-grid-nodes")
    if args.ridge < 0 or args.mwle_ridge <= 0 or args.max_iter < 1 or args.tol <= 0:
        raise ValueError("invalid fit or MWLE optimizer settings")
    if args.top_n < 1 or args.max_scenarios < 1:
        raise ValueError("--top-n and --max-scenarios must be positive")


def run(args: argparse.Namespace) -> int:
    for path in (args.bank, args.matrix, args.scenarios):
        if not path.is_file():
            raise FileNotFoundError(path)
    se_targets = resolve_se_targets(args)
    named_outputs = (
        "base_cat_results.csv",
        "bootstrap_replicates.csv",
        "bootstrap_theta_draws.csv",
        "ability_se_components.csv",
        "total_se_vs_target.csv",
        "metrics.json",
    ) + tuple(
        f"ability_se_components_se_{_target_slug(target)}.csv"
        for target in se_targets
    )
    if not args.overwrite and any((args.out_dir / name).exists() for name in named_outputs):
        raise scat.OfflineStudyError(
            f"outputs already exist under {args.out_dir}; pass --overwrite deliberately"
        )

    skills = tuple(part.strip() for part in args.skills.split(",") if part.strip()) \
        if args.skills else None
    bank = scat.load_fitted_bank(
        args.bank,
        skills=skills,
        negative_policy=args.bank_negative_policy,
        require_calibrated=not args.allow_unmarked_bank,
    )
    _validate_args(args, bank)
    matrix = scat.load_response_matrix(args.matrix)
    scenarios = scat.load_scenario_records(args.scenarios)
    missing = sorted(set(bank.criterion_ids) - set(matrix.columns))
    if missing:
        raise scat.OfflineStudyError(
            f"matrix is missing {len(missing)} fitted-bank criteria; first: {missing[:10]}"
        )
    aligned = matrix.reindex(columns=bank.criterion_ids).to_numpy(float)

    original_quad = scat.build_quadrature(
        bank.n_dims, args.eap_grid, bank.latent_correlation, max_nodes=args.max_grid_nodes
    )
    base_nested: dict[tuple[float, str], Mapping[str, Any]] = {}
    base_rows: list[dict[str, Any]] = []
    for target in se_targets:
        spec = scat.RunSpec(
            seed=args.cat_seed,
            top_n=args.top_n,
            max_se=target,
            min_evals_per_skill=args.min_evals_per_skill,
            min_scenarios=args.min_scenarios,
            max_scenarios=args.max_scenarios,
            selection=args.selection,
            mode="cat",
        )
        for model in matrix.index:
            try:
                result = scat.run_recorded_model(
                    str(model), matrix.loc[model], bank, scenarios, original_quad, spec,
                    mwle_ridge=args.mwle_ridge,
                )
                base_nested[(target, str(model))] = result
                base_rows.append(
                    {
                        "se_target": target,
                        "status": "ok",
                        "error": "",
                        **scat.flatten_result(result, bank.dims),
                    }
                )
            except (scat.OfflineStudyError, ValueError, np.linalg.LinAlgError) as error:
                base_rows.append(
                    {
                        "se_target": target,
                        "model": str(model),
                        "status": "replay_error",
                        "error": f"{type(error).__name__}: {error}",
                        "mwle_converged": False,
                    }
                )
    base_frame = pd.DataFrame(base_rows)
    successful_base = base_frame[base_frame["status"] == "ok"].copy()
    if successful_base.empty:
        raise scat.OfflineStudyError("original-bank CAT replay failed for every model")

    rng = np.random.default_rng(args.seed)
    replicate_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    n_models = len(matrix)
    for replicate in range(args.n_boot):
        sample_indices = rng.integers(0, n_models, size=n_models)
        counts = Counter(str(matrix.index[index]) for index in sample_indices)
        rep_base: dict[str, Any] = {
            "replicate": replicate,
            "sample_signature": _sample_signature(sample_indices),
            "sampled_model_counts": json.dumps(dict(sorted(counts.items()))),
        }
        try:
            fit = bootstrap_refit(aligned, bank, sample_indices, args)
            quad = scat.build_quadrature(
                bank.n_dims, args.eap_grid, fit["R"], max_nodes=args.max_grid_nodes
            )
        except (scat.OfflineStudyError, ValueError, np.linalg.LinAlgError) as error:
            replicate_rows.append(
                {
                    **rep_base,
                    "status": "fit_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        fit_indices = np.asarray(fit["item_indices"], dtype=int)
        local_by_source = {int(source): local for local, source in enumerate(fit_indices)}
        replicate_rows.append(
            {
                **rep_base,
                "status": "ok",
                "error": "",
                "fit_converged": fit["converged"],
                "fit_iterations": fit["n_iter"],
                "n_items_fit": int(len(fit_indices)),
                "n_variant_before_loading_policy": fit["n_variant_before_loading_policy"],
                "n_nonpositive": fit["n_nonpositive"],
                "latent_correlation": json.dumps(np.asarray(fit["R"]).tolist()),
            }
        )
        for target in se_targets:
            target_models = successful_base.loc[
                successful_base["se_target"] == target, "model"
            ]
            for model in target_models:
                original_result = base_nested[(target, str(model))]
                administered_source = [
                    bank.index[cid] for cid in original_result["criterion_order"]
                ]
                administered_local = np.asarray(
                    [
                        local_by_source[index]
                        for index in administered_source
                        if index in local_by_source
                    ],
                    dtype=int,
                )
                responses = aligned[matrix.index.get_loc(model), fit_indices]
                loading_rank = (
                    int(np.linalg.matrix_rank(fit["A"][administered_local]))
                    if administered_local.size
                    else 0
                )
                if administered_local.size == 0 or loading_rank < bank.n_dims:
                    draw_rows.append(
                        {
                            "replicate": replicate,
                            "se_target": target,
                            "model": str(model),
                            "status": "insufficient_administered_rank",
                            "error": f"rank {loading_rank}/{bank.n_dims}",
                            "n_administered_fit_items": int(administered_local.size),
                            "mwle_converged": False,
                        }
                    )
                    continue
                try:
                    eap = scat.batch_eap(
                        responses, fit["A"], fit["b"], quad, administered_local
                    )
                    weighted = scat.mwle(
                        responses,
                        fit["A"],
                        fit["b"],
                        administered_local,
                        theta0=eap.theta,
                        ridge=args.mwle_ridge,
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    draw_rows.append(
                        {
                            "replicate": replicate,
                            "se_target": target,
                            "model": str(model),
                            "status": "ability_error",
                            "error": f"{type(error).__name__}: {error}",
                            "n_administered_fit_items": int(administered_local.size),
                            "mwle_converged": False,
                        }
                    )
                    continue
                row: dict[str, Any] = {
                    "replicate": replicate,
                    "se_target": target,
                    "model": str(model),
                    "status": "ok",
                    "error": "",
                    "n_administered_fit_items": int(administered_local.size),
                    "mwle_converged": bool(weighted.converged),
                    "mwle_message": weighted.message,
                }
                for index, dim in enumerate(bank.dims):
                    row[f"theta_eap_{dim}"] = float(eap.theta[index])
                    row[f"theta_mwle_{dim}"] = float(weighted.theta[index])
                draw_rows.append(row)

    replicate_frame = pd.DataFrame(replicate_rows)
    draw_frame = pd.DataFrame(draw_rows)
    if draw_frame.empty:
        # Preserve a schema and produce explicit unreliable summaries rather than
        # crashing in the aggregation code.
        draw_frame = pd.DataFrame(
            columns=["replicate", "se_target", "model", "status", "mwle_converged"]
        )
    component_frames: list[pd.DataFrame] = []
    for target in se_targets:
        target_components = summarize_uncertainty(
            successful_base[successful_base["se_target"] == target],
            draw_frame[draw_frame["se_target"] == target],
            bank.dims,
            args.n_boot,
            args.min_valid_fraction,
        )
        target_components.insert(0, "se_target", target)
        component_frames.append(target_components)
    components = pd.concat(component_frames, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_frame.to_csv(args.out_dir / "base_cat_results.csv", index=False)
    replicate_frame.to_csv(args.out_dir / "bootstrap_replicates.csv", index=False)
    draw_frame.to_csv(args.out_dir / "bootstrap_theta_draws.csv", index=False)
    components.to_csv(args.out_dir / "ability_se_components.csv", index=False)
    for target in se_targets:
        components[components["se_target"] == target].to_csv(
            args.out_dir
            / f"ability_se_components_se_{_target_slug(target)}.csv",
            index=False,
        )

    aggregate: list[dict[str, Any]] = []
    for target in se_targets:
        target_base = successful_base[successful_base["se_target"] == target]
        for estimator in ESTIMATORS:
            for dim in bank.dims:
                subset = components[
                    (components["se_target"] == target)
                    & (components["estimator"] == estimator)
                    & (components["dimension"] == dim)
                ]
                reliable = subset[subset["reliable"]]
                aggregate.append(
                    {
                        "se_target": target,
                        "estimator": estimator,
                        "dimension": dim,
                        "n_models": int(len(subset)),
                        "n_reliable_models": int(len(reliable)),
                        "mean_scenarios": (
                            float(target_base["scenarios_administered"].mean())
                            if len(target_base)
                            else None
                        ),
                        "precision_reached_rate": (
                            float(target_base["precision_reached"].astype(bool).mean())
                            if len(target_base) and "precision_reached" in target_base
                            else None
                        ),
                        "median_se_ability": (
                            float(reliable["se_ability"].median()) if len(reliable) else None
                        ),
                        "median_se_param": (
                            float(reliable["se_param"].median()) if len(reliable) else None
                        ),
                        "median_se_total": (
                            float(reliable["se_total"].median()) if len(reliable) else None
                        ),
                        "mean_se_total": (
                            float(reliable["se_total"].mean()) if len(reliable) else None
                        ),
                    }
                )
    aggregate_frame = pd.DataFrame(aggregate)
    aggregate_frame.to_csv(args.out_dir / "total_se_vs_target.csv", index=False)
    summary = {
        "generated_at": _utcnow(),
        "purpose": "scenario-CAT item-parameter uncertainty via model bootstrap refits",
        "method": "nonparametric model-row bootstrap; administered scenario set held fixed",
        "se_total_formula": "sqrt(se_ability^2 + se_param^2)",
        "se_total_assumption": "conditional ability and item-parameter errors treated as independent",
        "bank": str(args.bank),
        "matrix": str(args.matrix),
        "scenarios": str(args.scenarios),
        "dimensions": list(bank.dims),
        "n_models": int(len(matrix)),
        "se_targets": list(se_targets),
        "n_boot_requested": args.n_boot,
        "n_boot_fit_success": int((replicate_frame["status"] == "ok").sum()),
        "n_boot_fit_failures": int((replicate_frame["status"] != "ok").sum()),
        "n_base_replay_failures": int((base_frame["status"] != "ok").sum()),
        "n_base_replay_failures_by_target": {
            str(target): int(
                (
                    base_frame.loc[base_frame["se_target"] == target, "status"]
                    != "ok"
                ).sum()
            )
            for target in se_targets
        },
        "config": {
            "seed": args.seed,
            "cat_seed": args.cat_seed,
            "fit_grid": args.fit_grid,
            "eap_grid": args.eap_grid,
            "ridge": args.ridge,
            "mwle_ridge": args.mwle_ridge,
            "negative_policy_bootstrap": args.negative_policy,
            "negative_policy_bank": args.bank_negative_policy,
            "min_valid_fraction": args.min_valid_fraction,
            "selection": args.selection,
            "se_targets": list(se_targets),
            "bootstrap_refits_reused_across_targets": True,
            "min_evals_per_skill": args.min_evals_per_skill,
            "min_scenarios": args.min_scenarios,
            "max_scenarios": args.max_scenarios,
        },
        "aggregate": aggregate,
        "failure_policy": (
            "fit and ability failures are retained in bootstrap_replicates.csv and "
            "bootstrap_theta_draws.csv; unreliable model/estimator/dimension rows are flagged"
        ),
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8"
    )
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--skills", default=None, help="optional fitted dimension order")
    parser.add_argument("--bank-negative-policy", choices=("error", "drop", "keep"), default="error")
    parser.add_argument("--negative-policy", choices=("error", "drop", "keep"), default="drop")
    parser.add_argument("--allow-unmarked-bank", action="store_true")
    parser.add_argument("--allow-unconverged-fit", action="store_true")
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--min-valid-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--cat-seed", type=int, default=42)
    parser.add_argument("--fit-grid", type=int, default=5)
    parser.add_argument("--eap-grid", type=int, default=5)
    parser.add_argument("--max-grid-nodes", type=int, default=5_000)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--mwle-ridge", type=float, default=1e-6)
    parser.add_argument("--selection", choices=("trace", "dopt"), default="trace")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--max-se",
        type=float,
        default=0.30,
        help="single precision target used when --se-targets is omitted",
    )
    parser.add_argument(
        "--se-targets",
        default=None,
        help=(
            "comma-separated precision targets, e.g. 0.20,0.25,0.30,0.35; "
            "bootstrap item-bank refits are reused across all targets"
        ),
    )
    parser.add_argument("--min-evals-per-skill", type=int, default=15)
    parser.add_argument("--min-scenarios", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    try:
        return run(build_argparser().parse_args())
    except (FileNotFoundError, ValueError, scat.OfflineStudyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
