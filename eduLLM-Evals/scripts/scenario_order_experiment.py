"""Measure scenario-CAT path dependence across deterministic run seeds.

The fitted bank and recorded binary response matrix are held fixed. For every seed,
the real scenario-level CAT engine chooses its adaptive path again, then the script
reports the resulting online-Gaussian, batch-EAP, and MWLE abilities. Across-seed SD
and range quantify how much the selected scenario set/order changes a model's score.

This is an offline replay: it makes no tutor-model or judge-model calls. Batch EAP
uses the fitted bank's correlated latent prior. The production online update retains
the engine's current Gaussian initialization, so its spread is reported separately.
"""

from __future__ import annotations

import argparse
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

from scripts import scenario_cat_lib as scat  # noqa: E402


ESTIMATORS = ("online", "eap", "mwle")


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


def seed_sequence(base_seed: int, n_seeds: int) -> list[int]:
    if n_seeds < 2:
        raise ValueError("--n-seeds must be at least 2 to estimate across-seed spread")
    return [int(base_seed + offset) for offset in range(n_seeds)]


def _pairwise_jaccard(serialized_sets: Sequence[str]) -> float | None:
    sets = [set(json.loads(value)) for value in serialized_sets if value]
    scores: list[float] = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            scores.append(len(sets[left] & sets[right]) / len(union) if union else 1.0)
    return float(np.mean(scores)) if scores else None


def per_model_spread(run_rows: pd.DataFrame, dims: Sequence[str]) -> pd.DataFrame:
    """Summarize estimator spread while exposing estimator-specific failures."""
    rows: list[dict[str, Any]] = []
    for model, model_rows in run_rows.groupby("model", sort=True):
        successful = model_rows[model_rows["status"] == "ok"]
        base: dict[str, Any] = {
            "model": str(model),
            "n_seed_runs": int(len(model_rows)),
            "n_successful_runs": int(len(successful)),
            "n_run_failures": int((model_rows["status"] != "ok").sum()),
            "mean_scenarios": (
                float(successful["scenarios_administered"].mean())
                if len(successful)
                else None
            ),
            "mean_pairwise_scenario_jaccard": _pairwise_jaccard(
                successful["scenario_order"].tolist()
            ),
        }
        for estimator in ESTIMATORS:
            eligible = successful
            if estimator == "mwle":
                eligible = successful[successful["mwle_converged"] == True]  # noqa: E712
            base[f"{estimator}_valid_runs"] = int(len(eligible))
            base[f"{estimator}_failed_runs"] = int(len(model_rows) - len(eligible))
            for dim in dims:
                values = pd.to_numeric(
                    eligible.get(f"theta_{estimator}_{dim}", pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna().to_numpy(float)
                prefix = f"{estimator}_{dim}"
                base[f"{prefix}_mean"] = float(values.mean()) if values.size else None
                base[f"{prefix}_sd"] = (
                    float(values.std(ddof=1)) if values.size >= 2 else None
                )
                base[f"{prefix}_min"] = float(values.min()) if values.size else None
                base[f"{prefix}_max"] = float(values.max()) if values.size else None
                base[f"{prefix}_range"] = (
                    float(values.max() - values.min()) if values.size else None
                )
        rows.append(base)
    return pd.DataFrame(rows)


def aggregate_spread(per_model: pd.DataFrame, dims: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for estimator in ESTIMATORS:
        for dim in dims:
            sd = pd.to_numeric(per_model[f"{estimator}_{dim}_sd"], errors="coerce").dropna()
            ranges = pd.to_numeric(
                per_model[f"{estimator}_{dim}_range"], errors="coerce"
            ).dropna()
            rows.append(
                {
                    "estimator": estimator,
                    "dimension": dim,
                    "n_models_with_spread": int(len(sd)),
                    "mean_sd": float(sd.mean()) if len(sd) else None,
                    "median_sd": float(sd.median()) if len(sd) else None,
                    "max_sd": float(sd.max()) if len(sd) else None,
                    "mean_range": float(ranges.mean()) if len(ranges) else None,
                    "median_range": float(ranges.median()) if len(ranges) else None,
                    "max_range": float(ranges.max()) if len(ranges) else None,
                }
            )
    return rows


def _validate_args(args: argparse.Namespace, bank: scat.FittedBank) -> None:
    seed_sequence(args.base_seed, args.n_seeds)
    if args.grid < 2 or args.grid ** bank.n_dims > args.max_grid_nodes:
        raise ValueError(
            f"invalid quadrature: {args.grid}^{bank.n_dims} must be <= "
            f"--max-grid-nodes={args.max_grid_nodes}"
        )
    if args.top_n < 1 or args.max_scenarios < 1:
        raise ValueError("--top-n and --max-scenarios must be positive")
    if args.min_scenarios < 0 or args.min_evals_per_skill < 0:
        raise ValueError("minimum scenario/evaluation counts cannot be negative")
    if args.mwle_ridge <= 0:
        raise ValueError("--mwle-ridge must be positive")


def run(args: argparse.Namespace) -> int:
    for path in (args.bank, args.matrix, args.scenarios):
        if not path.is_file():
            raise FileNotFoundError(path)
    named_outputs = ("seed_runs.csv", "per_model_spread.csv", "spread_summary.csv", "metrics.json")
    if not args.overwrite and any((args.out_dir / name).exists() for name in named_outputs):
        raise scat.OfflineStudyError(
            f"outputs already exist under {args.out_dir}; pass --overwrite deliberately"
        )

    skills = tuple(part.strip() for part in args.skills.split(",") if part.strip()) \
        if args.skills else None
    bank = scat.load_fitted_bank(
        args.bank,
        skills=skills,
        negative_policy=args.negative_policy,
        require_calibrated=not args.allow_unmarked_bank,
    )
    _validate_args(args, bank)
    matrix = scat.load_response_matrix(args.matrix)
    scenarios = scat.load_scenario_records(args.scenarios)
    unknown = sorted(set(bank.criterion_ids) - set(matrix.columns))
    if unknown:
        raise scat.OfflineStudyError(
            f"matrix is missing {len(unknown)} fitted-bank criteria; first: {unknown[:10]}"
        )
    quadrature = scat.build_quadrature(
        bank.n_dims,
        args.grid,
        bank.latent_correlation,
        max_nodes=args.max_grid_nodes,
    )

    run_rows: list[dict[str, Any]] = []
    seeds = seed_sequence(args.base_seed, args.n_seeds)
    for seed in seeds:
        spec = scat.RunSpec(
            seed=seed,
            top_n=args.top_n,
            max_se=args.max_se,
            min_evals_per_skill=args.min_evals_per_skill,
            min_scenarios=args.min_scenarios,
            max_scenarios=args.max_scenarios,
            selection=args.selection,
            mode="cat",
        )
        for model in matrix.index:
            try:
                result = scat.run_recorded_model(
                    str(model), matrix.loc[model], bank, scenarios, quadrature, spec,
                    mwle_ridge=args.mwle_ridge,
                )
                flat = scat.flatten_result(result, bank.dims)
                row: dict[str, Any] = {"status": "ok", "error": "", **flat}
            except (scat.OfflineStudyError, ValueError, np.linalg.LinAlgError) as error:
                row = {
                    "model": str(model),
                    "seed": seed,
                    "status": "replay_error",
                    "error": f"{type(error).__name__}: {error}",
                    "scenarios_administered": np.nan,
                    "scenario_order": "[]",
                    "mwle_converged": False,
                }
            run_rows.append(row)

    run_frame = pd.DataFrame(run_rows)
    per_model = per_model_spread(run_frame, bank.dims)
    aggregate_rows = aggregate_spread(per_model, bank.dims)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_frame.sort_values(["model", "seed"]).to_csv(args.out_dir / "seed_runs.csv", index=False)
    per_model.to_csv(args.out_dir / "per_model_spread.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(args.out_dir / "spread_summary.csv", index=False)

    summary = {
        "generated_at": _utcnow(),
        "purpose": "scenario-level CAT seed/path dependence",
        "bank": str(args.bank),
        "matrix": str(args.matrix),
        "scenarios": str(args.scenarios),
        "dimensions": list(bank.dims),
        "latent_correlation": bank.latent_correlation.tolist(),
        "n_models": int(len(matrix)),
        "seeds": seeds,
        "n_run_failures": int((run_frame["status"] != "ok").sum()),
        "n_mwle_failures": int(
            ((run_frame["status"] == "ok") & (run_frame["mwle_converged"] != True)).sum()  # noqa: E712
        ),
        "config": {
            "selection": args.selection,
            "top_n": args.top_n,
            "max_se": args.max_se,
            "min_evals_per_skill": args.min_evals_per_skill,
            "min_scenarios": args.min_scenarios,
            "max_scenarios": args.max_scenarios,
            "grid": args.grid,
            "max_grid_nodes": args.max_grid_nodes,
            "negative_policy": args.negative_policy,
            "mwle_ridge": args.mwle_ridge,
        },
        "spread": aggregate_rows,
        "notes": {
            "eap_prior": "fitted correlated latent prior",
            "online_prior": "production engine Gaussian initialization",
            "failed_runs": "retained in seed_runs.csv and excluded estimator-wise from spread",
        },
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
    parser.add_argument("--negative-policy", choices=("error", "drop", "keep"), default="error")
    parser.add_argument("--allow-unmarked-bank", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--selection", choices=("trace", "dopt"), default="trace")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-se", type=float, default=0.30)
    parser.add_argument("--min-evals-per-skill", type=int, default=15)
    parser.add_argument("--min-scenarios", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int, default=50)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--max-grid-nodes", type=int, default=5_000)
    parser.add_argument("--mwle-ridge", type=float, default=1e-6)
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
