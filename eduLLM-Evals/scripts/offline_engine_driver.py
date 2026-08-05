"""Replay calibrated scenario CAT against a recorded binary response matrix.

This command is post-calibration only.  It makes no model or judge calls.  For every
recorded tutor model it runs the real scenario-level CAT engine (and optionally a
seeded-random baseline), then reports production-online, batch-EAP, and MWLE abilities.

Example::

    python scripts/offline_engine_driver.py \
      --bank runs/calibration/.../fitted_bank.jsonl \
      --matrix runs/calibration/.../response_matrix.csv \
      --scenarios data/InFoBench/scenarios.jsonl \
      --mode both --grid 5 --out-dir runs/calibration/.../cat_replay

For ``--mode both``, the CAT arm uses ``--max-scenarios`` while the random arm defaults
to the full available scenario bank (same precision stop, no artificial 50-scenario cap).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scenario_cat_lib import (  # noqa: E402
    OfflineStudyError,
    RunSpec,
    build_quadrature,
    flatten_result,
    load_fitted_bank,
    load_response_matrix,
    load_scenario_records,
    run_recorded_model,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(v) for v in value]
    if isinstance(value, (np.floating, float)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _recovery(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or float(pair.iloc[:, 0].std()) == 0:
        return {"n": int(len(pair)), "r": None, "slope": None, "mae": None}
    xv = pair.iloc[:, 0].to_numpy(float)
    yv = pair.iloc[:, 1].to_numpy(float)
    return {
        "n": int(len(pair)),
        "r": float(np.corrcoef(xv, yv)[0, 1]),
        "slope": float(np.polyfit(xv, yv, 1)[0]),
        "mae": float(np.mean(np.abs(yv - xv))),
    }


def _pass_rate_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    pair = pd.concat([observed, predicted], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if pair.empty:
        return {"n": 0, "mae": None, "bias": None, "r": None}
    actual = pair.iloc[:, 0].to_numpy(float)
    estimate = pair.iloc[:, 1].to_numpy(float)
    correlation = None
    if len(pair) >= 3 and float(actual.std()) > 0 and float(estimate.std()) > 0:
        correlation = float(np.corrcoef(actual, estimate)[0, 1])
    return {
        "n": int(len(pair)),
        "mae": float(np.mean(np.abs(estimate - actual))),
        "bias": float(np.mean(estimate - actual)),
        "r": correlation,
    }


def summarize(rows: pd.DataFrame, dims: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "modes": {},
        "pass_rate_calibration_note": (
            "Observed rate and every p-IRT prediction use the same model-specific set "
            "of observed fitted items. Full-bank EAP is a posterior-reconstruction "
            "check, not an out-of-sample prediction."
        ),
    }
    for mode, frame in rows.groupby("mode", sort=True):
        mode_summary: dict[str, Any] = {
            "n_models": int(len(frame)),
            "precision_reached": int(frame["precision_reached"].sum()),
            "precision_rate": float(frame["precision_reached"].mean()),
            "scenarios": {
                "mean": float(frame["scenarios_administered"].mean()),
                "median": float(frame["scenarios_administered"].median()),
                "min": int(frame["scenarios_administered"].min()),
                "max": int(frame["scenarios_administered"].max()),
            },
            "criteria_mean": float(frame["criteria_administered"].mean()),
            "mwle_converged": int(frame["mwle_converged"].sum()),
            "recovery_vs_full_bank_eap": {},
            "pass_rate_calibration": {},
        }
        for estimator in ("online", "eap", "mwle"):
            eligible = frame
            if estimator == "mwle":
                eligible = frame[frame["mwle_converged"] == True]  # noqa: E712
            mode_summary["recovery_vs_full_bank_eap"][estimator] = {
                dim: _recovery(
                    eligible[f"theta_full_eap_{dim}"],
                    eligible[f"theta_{estimator}_{dim}"],
                )
                for dim in dims
            }
        for estimator in ("online", "eap", "mwle", "full_eap"):
            eligible = frame
            if estimator == "mwle":
                eligible = frame[frame["mwle_converged"] == True]  # noqa: E712
            mode_summary["pass_rate_calibration"][estimator] = _pass_rate_metrics(
                eligible["observed_fitted_pass_rate"],
                eligible[f"pirt_predicted_pass_rate_{estimator}"],
            )
        summary["modes"][str(mode)] = mode_summary
    return summary


def paired_table(rows: pd.DataFrame, dims: tuple[str, ...]) -> pd.DataFrame:
    cat = rows[rows["mode"] == "cat"].set_index("model")
    baseline = rows[rows["mode"] == "baseline"].set_index("model")
    shared = cat.index.intersection(baseline.index)
    output = pd.DataFrame(index=shared)
    output.index.name = "model"
    output["cat_scenarios"] = cat.loc[shared, "scenarios_administered"]
    output["random_scenarios"] = baseline.loc[shared, "scenarios_administered"]
    output["cat_minus_random_scenarios"] = (
        output["cat_scenarios"] - output["random_scenarios"]
    )
    output["cat_precision_reached"] = cat.loc[shared, "precision_reached"]
    output["random_precision_reached"] = baseline.loc[shared, "precision_reached"]
    for dim in dims:
        for estimator in ("online", "eap", "mwle"):
            col = f"theta_{estimator}_{dim}"
            output[f"cat_minus_random_{estimator}_{dim}"] = (
                cat.loc[shared, col] - baseline.loc[shared, col]
            )
    return output.reset_index()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bank", type=Path, required=True, help="fitted-only rubric bank")
    parser.add_argument("--matrix", type=Path, required=True, help="model x criterion CSV")
    parser.add_argument("--scenarios", type=Path, required=True, help="scenario JSONL")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--skills", default=None, help="ordered comma-separated skill names")
    parser.add_argument("--mode", choices=("cat", "baseline", "both"), default="cat")
    parser.add_argument("--selection", choices=("trace", "dopt"), default="trace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-se", type=float, default=0.30)
    parser.add_argument("--min-evals-per-skill", type=int, default=15)
    parser.add_argument("--min-scenarios", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int, default=50)
    parser.add_argument(
        "--baseline-max-scenarios",
        type=int,
        default=0,
        help="random-arm cap; 0 means all available scenarios (recommended)",
    )
    parser.add_argument("--grid", type=int, default=5, help="GH nodes per latent dimension")
    parser.add_argument("--max-grid-nodes", type=int, default=50_000)
    parser.add_argument("--mwle-ridge", type=float, default=1e-6)
    parser.add_argument(
        "--negative-policy", choices=("error", "drop", "keep"), default="error"
    )
    parser.add_argument(
        "--allow-unverified-bank",
        action="store_true",
        help="allow a bank not marked calibrated (unsafe; intended for synthetic tests only)",
    )
    parser.add_argument("--models", default=None, help="comma-separated model IDs")
    parser.add_argument("--limit-models", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing this command's named artifacts in a non-empty out-dir",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    for path in (args.bank, args.matrix, args.scenarios):
        if not path.is_file():
            raise OfflineStudyError(f"input does not exist: {path}")
    if args.top_n < 1 or args.max_scenarios < 1 or args.grid < 2:
        raise OfflineStudyError("top_n/max_scenarios/grid must be positive (grid >=2)")
    if args.min_scenarios < 0 or args.min_evals_per_skill < 0 or args.mwle_ridge <= 0:
        raise OfflineStudyError("minimums must be nonnegative and mwle_ridge must be positive")

    artifacts = [
        args.out_dir / "per_model.csv",
        args.out_dir / "metrics.json",
        args.out_dir / "manifest.json",
    ]
    if args.mode == "both":
        artifacts.append(args.out_dir / "paired_cat_vs_random.csv")
    if not args.overwrite and any(path.exists() for path in artifacts):
        raise OfflineStudyError(
            f"output artifacts already exist under {args.out_dir}; pass --overwrite deliberately"
        )

    skills = tuple(x.strip() for x in args.skills.split(",") if x.strip()) if args.skills else None
    fitted = load_fitted_bank(
        args.bank,
        skills=skills,
        negative_policy=args.negative_policy,
        require_calibrated=not args.allow_unverified_bank,
    )
    matrix = load_response_matrix(args.matrix)
    missing_columns = [cid for cid in fitted.criterion_ids if cid not in matrix.columns]
    if missing_columns:
        raise OfflineStudyError(
            f"matrix is missing {len(missing_columns)} fitted-bank columns; "
            f"first: {missing_columns[:10]}"
        )
    scenarios = load_scenario_records(args.scenarios)
    quadrature = build_quadrature(
        fitted.n_dims,
        args.grid,
        fitted.latent_correlation,
        max_nodes=args.max_grid_nodes,
    )

    models = list(matrix.index)
    if args.models:
        requested = [x.strip() for x in args.models.split(",") if x.strip()]
        unknown = [model for model in requested if model not in matrix.index]
        if unknown:
            raise OfflineStudyError(f"unknown requested models: {unknown}")
        models = requested
    if args.limit_models is not None:
        if args.limit_models < 1:
            raise OfflineStudyError("--limit-models must be positive")
        models = models[: args.limit_models]

    modes = ("cat", "baseline") if args.mode == "both" else (args.mode,)
    full_bank_scenarios = len(fitted.scenario_groups())
    results: list[dict[str, Any]] = []
    for mode in modes:
        cap = args.max_scenarios
        if mode == "baseline" and args.baseline_max_scenarios == 0:
            cap = full_bank_scenarios
        elif mode == "baseline":
            cap = args.baseline_max_scenarios
        spec = RunSpec(
            seed=args.seed,
            top_n=args.top_n,
            max_se=args.max_se,
            min_evals_per_skill=args.min_evals_per_skill,
            min_scenarios=args.min_scenarios,
            max_scenarios=cap,
            selection=args.selection,
            mode=mode,
        )
        for pos, model in enumerate(models, 1):
            print(f"[{mode} {pos}/{len(models)}] {model}", flush=True)
            result = run_recorded_model(
                model,
                matrix.loc[model],
                fitted,
                scenarios,
                quadrature,
                spec,
                mwle_ridge=args.mwle_ridge,
            )
            results.append(flatten_result(result, fitted.dims))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results).sort_values(["mode", "model"])
    frame.to_csv(args.out_dir / "per_model.csv", index=False)
    metrics = summarize(frame, fitted.dims)
    if args.mode == "both":
        paired = paired_table(frame, fitted.dims)
        paired.to_csv(args.out_dir / "paired_cat_vs_random.csv", index=False)
        metrics["paired_cat_vs_random"] = {
            "n_models": int(len(paired)),
            "mean_cat_minus_random_scenarios": float(
                paired["cat_minus_random_scenarios"].mean()
            ),
            "median_cat_minus_random_scenarios": float(
                paired["cat_minus_random_scenarios"].median()
            ),
        }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(_finite_or_none(metrics), indent=2), encoding="utf-8"
    )
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "script": "scripts/offline_engine_driver.py",
        "argv": sys.argv[1:],
        "inputs": {
            "bank": str(args.bank),
            "bank_sha256": _sha256(args.bank),
            "matrix": str(args.matrix),
            "matrix_sha256": _sha256(args.matrix),
            "scenarios": str(args.scenarios),
            "scenarios_sha256": _sha256(args.scenarios),
        },
        "bank": {
            "dims": list(fitted.dims),
            "n_items": fitted.n_items,
            "n_scenarios": len(fitted.scenario_groups()),
            "latent_correlation": fitted.latent_correlation.tolist(),
            "negative_policy": args.negative_policy,
            "dropped_negative_items": list(fitted.dropped_negative_items),
        },
        "study": {
            "n_models": len(models),
            "modes": list(modes),
            "selection": args.selection,
            "seed": args.seed,
            "top_n": args.top_n,
            "max_se": args.max_se,
            "min_evals_per_skill": args.min_evals_per_skill,
            "min_scenarios": args.min_scenarios,
            "cat_max_scenarios": args.max_scenarios,
            "baseline_max_scenarios": (
                full_bank_scenarios
                if args.baseline_max_scenarios == 0
                else args.baseline_max_scenarios
            ),
            "grid_nodes_per_dim": args.grid,
            "grid_total_nodes": int(quadrature.grid.shape[0]),
            "mwle_ridge": args.mwle_ridge,
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(_finite_or_none(manifest), indent=2), encoding="utf-8"
    )
    print(f"wrote replay artifacts -> {args.out_dir}")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (OfflineStudyError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
