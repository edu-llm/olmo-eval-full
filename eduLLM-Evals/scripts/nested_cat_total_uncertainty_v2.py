"""Phase-4 uncertainty and order validation for the frozen InFoBench v2 study.

This driver is deliberately separate from the historical Phase-4 command.  It
consumes the completed 5-repeat x 5-fold v2 Phase-3 handoff, uses each panel's
inner-selected *exact* calibration specification, and evaluates only the
prospectively frozen primary CAT policy.  Sensitivity policies are never fitted,
replayed, selected, or promoted here.

For every repeat/outer-fold panel the driver performs 100 family-clustered
bootstrap refits using outer-training families only.  The primary CAT path is
rerun under every bootstrap bank.  CAT and full-administration uncertainty are
both computed with the law of total variance.  The full-administration result is
diagnostic only.  The base outer-fit bank is also replayed under the 20 frozen
order seeds.

``--plan-only`` is strictly read-only.  ``--resume`` accepts only checkpoints
and caches bearing the exact study signature.  Completed or failed output is
never deleted in place; a new attempt requires a new versioned output leaf.
Tests use small synthetic panels by calling the public calculation helpers; the
official thresholds are never relaxed for those fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
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
from scripts import nested_scenario_cat_cv as v1_phase3  # noqa: E402
from scripts import nested_scenario_cat_cv_v2 as phase3  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402
from scripts import scenario_kfold_estimator_cv as scenario_cv  # noqa: E402

SCRIPT_SCHEMA = "infobench-nested-cat-total-uncertainty-v2-v1"
CHECKPOINT_SCHEMA = "infobench-v2-phase4-checkpoint-v1"
BOOTSTRAP_CACHE_SCHEMA = "infobench-v2-phase4-bootstrap-fit-v1"
DECISION_SCHEMA = "infobench-v2-phase4-decision-v1"

DEFAULT_CONFIG = ROOT / "configs" / "infobench_calibration_cat_v2.json"
DEFAULT_PHASE3 = ROOT / "runs" / "calibration" / "InFoBench_v2" / "phase3"
DEFAULT_OUTPUT = ROOT / "runs" / "calibration" / "InFoBench_v2" / "phase4"
DEFAULT_NUMERICAL_FOLLOWUP_CONFIG = phase3.DEFAULT_NUMERICAL_FOLLOWUP_CONFIG
DEFAULT_NUMERICAL_LOCK = phase3.DEFAULT_NUMERICAL_LOCK

EXPECTED_REPEATS = 5
EXPECTED_OUTER_FOLDS = 5
EXPECTED_MODELS = 52
EXPECTED_PANELS = EXPECTED_REPEATS * EXPECTED_OUTER_FOLDS
PRIMARY_POLICY_ID = phase3.PRIMARY_POLICY_ID

FINAL_OUTPUTS = (
    "parameter_bootstrap_draws.csv",
    "outer_total_se.csv",
    "order_seed_runs.csv",
    "outer_order_stability.csv",
    "repetition_gate_results.csv",
    "model_repeat_aggregate.csv",
    "family_bootstrap_summary.csv",
    "phase4_decision.json",
    "phase4_summary.json",
)

CODE_DEPENDENCIES = (
    ROOT / "scripts" / "nested_cat_total_uncertainty_v2.py",
    ROOT / "scripts" / "nested_scenario_cat_cv_v2.py",
    ROOT / "scripts" / "nested_scenario_cat_cv.py",
    ROOT / "scripts" / "calibrate_mirt.py",
    ROOT / "scripts" / "kfold_cv_mirt.py",
    ROOT / "scripts" / "scenario_cat_lib.py",
    ROOT / "scripts" / "scenario_kfold_estimator_cv.py",
    ROOT / "scripts" / "run_infobench_v2_numerical_followup_v3.py",
    ROOT / "scripts" / "run_infobench_v2_numerical_followup_v2.py",
    ROOT / "tutor_cat" / "mirt.py",
    ROOT / "tutor_cat" / "engine.py",
)


class V2Phase4Error(RuntimeError):
    """A frozen-design, provenance, leakage, or completeness invariant failed."""


class RecoverableNumericalFailure(RuntimeError):
    """One preregistered bootstrap fit or replay failed numerically."""


@dataclass(frozen=True)
class FamilyBootstrapSample:
    family_ids: tuple[str, ...]
    model_ids: tuple[str, ...]


@dataclass(frozen=True)
class Panel:
    panel_id: str
    repeat: int
    outer_fold: int
    training_model_ids: tuple[str, ...]
    test_model_ids: tuple[str, ...]
    training_family_ids: tuple[str, ...]
    test_family_ids: tuple[str, ...]
    selected_spec: phase3.CalibrationSpec
    selected_specification: dict[str, Any]
    outer_fit_cache_key: str
    outer_fit_cache_dir: Path
    outer_fit_manifest_path: Path
    outer_fit_manifest_sha256: str


@dataclass
class BankBundle:
    bank: scat.FittedBank
    quadrature: scat.Quadrature
    fit: dict[str, Any]
    cache_key: str


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
        raise V2Phase4Error(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise V2Phase4Error(f"expected a JSON object in {path}")
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


def _repo_path(value: str | Path) -> Path:
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


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("numpy", "pandas", "scipy", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _environment_provenance() -> dict[str, Any]:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "dependencies": _dependency_versions(),
    }
    return {**payload, "canonical_sha256": _canonical_hash(payload)}


def _code_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for path in CODE_DEPENDENCIES:
        if not path.is_file():
            raise V2Phase4Error(f"required code dependency is missing: {path}")
        output[_display_path(path)] = _sha256(path)
    return output


def _tree_hash(path: Path) -> str:
    if not path.is_dir():
        return _canonical_hash({})
    inventory = {
        str(item.relative_to(path)): _sha256(item)
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    }
    return _canonical_hash(inventory)


def _manifest_output_hash(manifest: Mapping[str, Any], name: str) -> str:
    raw = (manifest.get("outputs") or {}).get(name)
    if isinstance(raw, Mapping):
        return str(raw.get("sha256") or "")
    return str(raw or "")


def _provenance_hash(payload: Mapping[str, Any], name: str) -> str:
    """Read a required hash from either flat or named provenance fields."""

    direct = payload.get(f"{name}_sha256")
    if direct:
        return str(direct)
    raw = payload.get(name)
    if isinstance(raw, Mapping) and raw.get("sha256"):
        return str(raw["sha256"])
    provenance = payload.get("provenance") or {}
    nested = provenance.get(name) if isinstance(provenance, Mapping) else None
    if isinstance(nested, Mapping) and nested.get("sha256"):
        return str(nested["sha256"])
    return ""


def assert_fit_boundary(
    training_model_ids: Sequence[str],
    test_model_ids: Sequence[str],
    sampled_model_ids: Sequence[str] | None = None,
) -> None:
    """Fail closed if an outer-test or foreign model can enter calibration."""

    training = set(map(str, training_model_ids))
    test = set(map(str, test_model_ids))
    if not training or not test:
        raise V2Phase4Error("outer training and test sets must both be non-empty")
    overlap = sorted(training & test)
    if overlap:
        raise V2Phase4Error(f"outer train/test leakage: {overlap[:10]}")
    if sampled_model_ids is None:
        return
    sampled = list(map(str, sampled_model_ids))
    if not sampled:
        raise V2Phase4Error("family bootstrap produced an empty model sample")
    leaked = sorted(set(sampled) & test)
    foreign = sorted(set(sampled) - training)
    if leaked:
        raise V2Phase4Error(f"outer-test IDs entered bootstrap fitting: {leaked[:10]}")
    if foreign:
        raise V2Phase4Error(
            f"bootstrap contains IDs outside its outer-training set: {foreign[:10]}"
        )


def draw_outer_training_family_sample(
    training_model_ids: Sequence[str],
    test_model_ids: Sequence[str],
    model_to_family: Mapping[str, str],
    *,
    seed: int,
) -> FamilyBootstrapSample:
    """Draw families with replacement, expanding every selected family in full."""

    training = tuple(sorted(map(str, training_model_ids)))
    test = tuple(sorted(map(str, test_model_ids)))
    assert_fit_boundary(training, test)
    mapping = {str(model): str(family) for model, family in model_to_family.items()}
    missing = sorted((set(training) | set(test)) - set(mapping))
    if missing:
        raise V2Phase4Error(f"models missing tutor-family assignments: {missing[:10]}")
    train_families = tuple(sorted({mapping[model] for model in training}))
    test_families = {mapping[model] for model in test}
    crossed = sorted(set(train_families) & test_families)
    if crossed:
        raise V2Phase4Error(f"tutor families cross the outer boundary: {crossed[:10]}")
    members = {
        family: tuple(model for model in training if mapping[model] == family)
        for family in train_families
    }
    if any(not values for values in members.values()):
        raise V2Phase4Error("outer-training family has no models")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(train_families), size=len(train_families))
    sampled_families = tuple(train_families[int(index)] for index in indices)
    sampled_models = tuple(model for family in sampled_families for model in members[family])
    assert_fit_boundary(training, test, sampled_models)
    return FamilyBootstrapSample(sampled_families, sampled_models)


def _replicate_seed(master_seed: int, repeat: int, outer_fold: int, replicate: int) -> int:
    digest = hashlib.sha256(
        f"{master_seed}:repeat={repeat}:outer={outer_fold}:bootstrap={replicate}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def validate_order_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) != 20:
        raise V2Phase4Error("v2 order stability requires exactly 20 frozen seeds")
    if len(set(normalized)) != len(normalized):
        raise V2Phase4Error("order stability seeds must be unique")
    return normalized


def law_total_variance(
    draw_rows: pd.DataFrame,
    *,
    model_ids: Sequence[str],
    dimensions: Sequence[str],
    n_boot: int,
    minimum_valid_rate: float,
    estimator: str = "mwle",
) -> pd.DataFrame:
    """Compute E[Var(theta|bank)] + Var(E[theta|bank]) for one panel."""

    if estimator not in {"mwle", "full_eap"}:
        raise V2Phase4Error(f"unsupported total-variance estimator: {estimator}")
    context_columns = ("panel_id", "repeat", "outer_fold", "spec_id")
    context: dict[str, Any] = {}
    for name in context_columns:
        if name in draw_rows.columns:
            values = draw_rows[name].drop_duplicates().tolist()
            if len(values) != 1:
                raise V2Phase4Error(f"total-variance input mixes {name} values")
            context[name] = values[0]
    output: list[dict[str, Any]] = []
    for model in sorted(map(str, model_ids)):
        model_rows = draw_rows[draw_rows["model"].astype(str) == model]
        if "replicate" in model_rows.columns and model_rows["replicate"].nunique() != n_boot:
            raise V2Phase4Error(f"{model}: incomplete bootstrap replicate support")
        for dim in dimensions:
            theta = pd.to_numeric(
                model_rows.get(
                    f"theta_{estimator}_{dim}", pd.Series(np.nan, index=model_rows.index)
                ),
                errors="coerce",
            )
            se = pd.to_numeric(
                model_rows.get(f"se_{estimator}_{dim}", pd.Series(np.nan, index=model_rows.index)),
                errors="coerce",
            )
            status_ok = (
                model_rows.get("status", pd.Series("", index=model_rows.index)).astype(str).eq("ok")
            )
            valid = status_ok & theta.notna() & se.notna() & (se >= 0)
            if estimator == "mwle":
                valid &= model_rows.get(
                    "mwle_converged", pd.Series(False, index=model_rows.index)
                ).astype(bool)
            theta_values = theta[valid].to_numpy(float)
            se_values = se[valid].to_numpy(float)
            conditional_variance = float(np.mean(np.square(se_values))) if se_values.size else None
            parameter_variance = (
                float(np.var(theta_values, ddof=1)) if theta_values.size >= 2 else None
            )
            total_variance = (
                conditional_variance + parameter_variance
                if conditional_variance is not None and parameter_variance is not None
                else None
            )
            valid_rate = theta_values.size / n_boot if n_boot else 0.0
            output.append(
                {
                    **context,
                    "model": model,
                    "dimension": str(dim),
                    "estimator": estimator,
                    "n_boot_requested": int(n_boot),
                    "n_valid_draws": int(theta_values.size),
                    "n_invalid_draws": int(n_boot - theta_values.size),
                    "valid_draw_rate": float(valid_rate),
                    "mean_theta": float(theta_values.mean()) if theta_values.size else None,
                    "mean_conditional_variance": conditional_variance,
                    "parameter_variance": parameter_variance,
                    "total_variance": total_variance,
                    "se_ability": (
                        math.sqrt(conditional_variance)
                        if conditional_variance is not None
                        else None
                    ),
                    "se_parameter": (
                        math.sqrt(parameter_variance) if parameter_variance is not None else None
                    ),
                    "se_total": (math.sqrt(total_variance) if total_variance is not None else None),
                    "valid_draw_gate": bool(valid_rate >= minimum_valid_rate),
                }
            )
    return pd.DataFrame(output).sort_values(["model", "dimension"]).reset_index(drop=True)


def combine_cat_and_full_total_se(cat: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    keys = ["panel_id", "repeat", "outer_fold", "spec_id", "model", "dimension"]
    for frame, label in ((cat, "CAT"), (full, "full-administration")):
        if frame.duplicated(keys).any():
            raise V2Phase4Error(f"{label} total-SE rows are not unique")
    cat_columns = {
        column: f"cat_{column}"
        for column in cat.columns
        if column not in keys and column != "estimator"
    }
    full_columns = {
        column: f"full_{column}"
        for column in full.columns
        if column not in keys and column != "estimator"
    }
    merged = (
        cat.drop(columns=["estimator"])
        .rename(columns=cat_columns)
        .merge(
            full.drop(columns=["estimator"]).rename(columns=full_columns),
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
    )
    if not merged["_merge"].eq("both").all():
        raise V2Phase4Error("CAT and full-administration total-SE support differs")
    return merged.drop(columns=["_merge"]).sort_values(keys).reset_index(drop=True)


def order_stability_frame(
    seed_rows: pd.DataFrame,
    *,
    model_ids: Sequence[str],
    dimensions: Sequence[str],
    order_seeds: Sequence[int],
) -> pd.DataFrame:
    """Retain one explicit stability row per model/dimension, including failures."""

    seeds = validate_order_seeds(order_seeds)
    context: dict[str, Any] = {}
    for name in ("panel_id", "repeat", "outer_fold", "spec_id"):
        values = seed_rows[name].drop_duplicates().tolist()
        if len(values) != 1:
            raise V2Phase4Error(f"order input mixes {name} values")
        context[name] = values[0]
    output: list[dict[str, Any]] = []
    for model in sorted(map(str, model_ids)):
        subset = seed_rows[seed_rows["model"].astype(str) == model]
        attempts = Counter(pd.to_numeric(subset["order_seed"], errors="coerce").tolist())
        all_attempts = len(subset) == len(seeds) and attempts == Counter(seeds)
        for dim in dimensions:
            theta = pd.to_numeric(
                subset.get(f"theta_mwle_{dim}", pd.Series(np.nan, index=subset.index)),
                errors="coerce",
            )
            se = pd.to_numeric(
                subset.get(f"se_mwle_{dim}", pd.Series(np.nan, index=subset.index)),
                errors="coerce",
            )
            valid = (
                subset["status"].astype(str).eq("ok")
                & subset.get("mwle_converged", pd.Series(False, index=subset.index)).astype(bool)
                & theta.notna()
                & se.notna()
                & (se >= 0)
            )
            values = theta[valid].to_numpy(float)
            all_valid = bool(all_attempts and len(values) == len(seeds))
            output.append(
                {
                    **context,
                    "model": model,
                    "dimension": str(dim),
                    "n_seeds_expected": len(seeds),
                    "n_seed_attempts": int(len(subset)),
                    "n_valid_seed_runs": int(len(values)),
                    "all_seed_attempts_present": bool(all_attempts),
                    "all_seed_runs_valid": all_valid,
                    "order_path_sd": (float(np.std(values, ddof=1)) if len(values) >= 2 else None),
                    "order_path_range": (float(np.ptp(values)) if len(values) >= 1 else None),
                }
            )
    return pd.DataFrame(output).sort_values(["model", "dimension"]).reset_index(drop=True)


def repetition_gate_results(
    total_se: pd.DataFrame,
    order: pd.DataFrame,
    *,
    expected_model_ids: Sequence[str],
    repeats: Sequence[int],
    minimum_valid_rate: float,
    maximum_p90_total_se: float,
    maximum_median_order_path_sd: float,
) -> pd.DataFrame:
    """Apply every acceptance gate independently inside every repetition."""

    expected = set(map(str, expected_model_ids))
    rows: list[dict[str, Any]] = []
    for repeat in repeats:
        total = total_se[pd.to_numeric(total_se["repeat"], errors="coerce") == repeat]
        stable = order[pd.to_numeric(order["repeat"], errors="coerce") == repeat]
        total_models = list(total["model"].astype(str))
        order_models = list(stable["model"].astype(str))
        complete_total = len(total_models) == len(expected) and set(total_models) == expected
        complete_order = len(order_models) == len(expected) and set(order_models) == expected
        cat_se = pd.to_numeric(total.get("cat_se_total"), errors="coerce")
        order_sd = pd.to_numeric(stable.get("order_path_sd"), errors="coerce")
        finite_total = bool(complete_total and cat_se.notna().all())
        finite_order = bool(complete_order and order_sd.notna().all())
        valid_rates = pd.to_numeric(total.get("cat_valid_draw_rate"), errors="coerce")
        valid_gate = bool(
            complete_total
            and valid_rates.notna().all()
            and (valid_rates >= minimum_valid_rate).all()
            and total.get("cat_valid_draw_gate", pd.Series(False, index=total.index))
            .astype(bool)
            .all()
        )
        p90 = float(np.quantile(cat_se.to_numpy(float), 0.9)) if finite_total else None
        p90_gate = bool(p90 is not None and p90 <= maximum_p90_total_se)
        every_seed = bool(
            complete_order
            and stable["all_seed_attempts_present"].astype(bool).all()
            and stable["all_seed_runs_valid"].astype(bool).all()
        )
        median_order = float(np.median(order_sd.to_numpy(float))) if finite_order else None
        order_gate = bool(
            every_seed and median_order is not None and median_order <= maximum_median_order_path_sd
        )
        passed = bool(valid_gate and p90_gate and order_gate)
        rows.append(
            {
                "repeat": int(repeat),
                "n_models_expected": len(expected),
                "n_total_se_rows": int(len(total)),
                "n_order_rows": int(len(stable)),
                "complete_total_se_support": complete_total,
                "complete_order_support": complete_order,
                "minimum_valid_draw_rate_observed": (
                    float(valid_rates.min())
                    if complete_total and valid_rates.notna().all()
                    else None
                ),
                "minimum_valid_draw_rate_required": float(minimum_valid_rate),
                "every_model_valid_draw_gate": valid_gate,
                "p90_cat_total_se": p90,
                "maximum_p90_cat_total_se": float(maximum_p90_total_se),
                "p90_total_se_gate": p90_gate,
                "every_order_seed_present_and_valid": every_seed,
                "median_order_path_sd": median_order,
                "maximum_median_order_path_sd": float(maximum_median_order_path_sd),
                "order_stability_gate": order_gate,
                "repetition_phase4_pass": passed,
            }
        )
    return pd.DataFrame(rows).sort_values("repeat").reset_index(drop=True)


def aggregate_repeated_models(
    total_se: pd.DataFrame,
    order: pd.DataFrame,
    *,
    expected_model_ids: Sequence[str],
    expected_repeats: Sequence[int],
) -> pd.DataFrame:
    """Aggregate repeat rows per model before any population-level summary."""

    expected_models = set(map(str, expected_model_ids))
    repeats = set(map(int, expected_repeats))
    keys = ["model", "repeat"]
    for frame, label in ((total_se, "total-SE"), (order, "order")):
        if frame.duplicated(keys).any():
            raise V2Phase4Error(f"{label} has duplicate model-repeat rows")
        if set(frame["model"].astype(str)) != expected_models:
            raise V2Phase4Error(f"{label} lacks complete model support")
        for model, subset in frame.groupby(frame["model"].astype(str), sort=True):
            if set(pd.to_numeric(subset["repeat"], errors="coerce").astype(int)) != repeats:
                raise V2Phase4Error(f"{label}: {model} lacks complete repeat support")
    merged = total_se.merge(
        order[["model", "repeat", "order_path_sd", "all_seed_runs_valid"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    output: list[dict[str, Any]] = []
    for model, subset in merged.groupby("model", sort=True):
        output.append(
            {
                "model": str(model),
                "n_repeats": int(len(subset)),
                "all_repeat_rows_present": len(subset) == len(repeats),
                "mean_cat_valid_draw_rate": float(subset["cat_valid_draw_rate"].mean()),
                "minimum_cat_valid_draw_rate": float(subset["cat_valid_draw_rate"].min()),
                "mean_cat_total_se": float(subset["cat_se_total"].mean()),
                "p90_across_repeats_cat_total_se": float(
                    np.quantile(subset["cat_se_total"].to_numpy(float), 0.9)
                ),
                "mean_full_total_se": float(subset["full_se_total"].mean()),
                "mean_order_path_sd": float(subset["order_path_sd"].mean()),
                "median_order_path_sd": float(subset["order_path_sd"].median()),
                "all_order_seed_runs_valid": bool(subset["all_seed_runs_valid"].all()),
            }
        )
    return pd.DataFrame(output).sort_values("model").reset_index(drop=True)


def family_cluster_bootstrap_summary(
    model_frame: pd.DataFrame,
    model_to_family: Mapping[str, str],
    *,
    seed: int,
    replicates: int,
) -> pd.DataFrame:
    """Bootstrap population diagnostics by family after per-model aggregation."""

    if replicates < 2:
        raise V2Phase4Error("metric family bootstrap requires at least two draws")
    mapping = {str(model): str(family) for model, family in model_to_family.items()}
    models = set(model_frame["model"].astype(str))
    if models != set(mapping):
        raise V2Phase4Error("family bootstrap mapping does not match model aggregate")
    families = tuple(sorted(set(mapping.values())))
    members = {
        family: tuple(sorted(model for model, value in mapping.items() if value == family))
        for family in families
    }
    indexed = model_frame.set_index(model_frame["model"].astype(str), drop=False)
    metrics: dict[str, Callable[[pd.DataFrame], float]] = {
        "mean_cat_valid_draw_rate": lambda frame: float(frame["mean_cat_valid_draw_rate"].mean()),
        "mean_cat_total_se": lambda frame: float(frame["mean_cat_total_se"].mean()),
        "p90_model_cat_total_se": lambda frame: float(
            np.quantile(frame["mean_cat_total_se"].to_numpy(float), 0.9)
        ),
        "mean_full_total_se": lambda frame: float(frame["mean_full_total_se"].mean()),
        "median_model_order_path_sd": lambda frame: float(frame["median_order_path_sd"].median()),
    }
    rng = np.random.default_rng(int(seed))
    draws = {metric: [] for metric in metrics}
    for _ in range(int(replicates)):
        selected = rng.integers(0, len(families), size=len(families))
        sampled_models = [model for index in selected for model in members[families[int(index)]]]
        sample = indexed.loc[sampled_models]
        for metric, statistic in metrics.items():
            draws[metric].append(statistic(sample))
    output = []
    for metric, statistic in metrics.items():
        values = np.asarray(draws[metric], dtype=float)
        output.append(
            {
                "metric": metric,
                "estimate": statistic(model_frame),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_replicates": int(replicates),
                "bootstrap_unit": "tutor_family",
                "repeated_rows_aggregated_per_model_first": True,
                "n_models": int(len(models)),
                "n_families": int(len(families)),
            }
        )
    return pd.DataFrame(output).sort_values("metric").reset_index(drop=True)


def validate_phase3_authorization(
    decision: Mapping[str, Any],
    selected: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Fail closed unless Phase 3 explicitly and completely authorizes Phase 4."""

    if manifest.get("schema_version") != phase3.SCRIPT_SCHEMA:
        raise V2Phase4Error("Phase-3 manifest schema is not the v2 driver")
    if manifest.get("status") != "phase3_complete":
        raise V2Phase4Error("Phase 3 is not complete")
    if selected.get("schema_version") != phase3.SELECTED_SPECS_SCHEMA:
        raise V2Phase4Error("selected calibration-spec handoff schema mismatch")
    if decision.get("schema_version") != phase3.DECISION_SCHEMA:
        raise V2Phase4Error("Phase-3 decision schema mismatch")
    if (
        decision.get("status") != "pass"
        or decision.get("phase3_pass") is not True
        or decision.get("phase4_authorized") is not True
    ):
        raise V2Phase4Error("Phase 3 did not pass and authorize Phase 4")
    failed = decision.get("failed_conditions") or []
    if failed:
        raise V2Phase4Error("Phase-3 decision retains failed conditions")
    coverage = decision.get("coverage") or {}
    required_coverage = (
        "all_25_panels_evaluated",
        "all_52_models_each_repetition",
    )
    if any(coverage.get(field) is not True for field in required_coverage):
        raise V2Phase4Error("Phase-3 coverage is incomplete")
    stability = decision.get("calibration_stability") or {}
    observed_fraction = stability.get("modal_fraction_all_25_panels", stability.get("fraction", 0))
    required_fraction = stability.get("required_fraction", stability.get("threshold", 0.8))
    if stability.get("passed") is not True or float(observed_fraction) < float(required_fraction):
        raise V2Phase4Error("Phase-3 exact calibration specification is unstable")
    primary = decision.get("primary_policy") or {}
    if (
        primary.get("all_outer_panels_pass") is not True
        or primary.get("all_repetitions_pass") is not True
        or (primary.get("failed_panel_ids") or primary.get("failed_panels") or [])
        or (primary.get("failed_repetitions") or primary.get("failed_repeats") or [])
    ):
        raise V2Phase4Error("frozen primary CAT policy did not pass complete Phase 3")
    sensitivities = decision.get("sensitivities") or {}
    if (
        sensitivities.get("diagnostic_only") is not True
        or sensitivities.get("can_promote") is not False
    ):
        raise V2Phase4Error("Phase-3 sensitivity policies are not non-promotable")
    signature = str(manifest.get("study_signature") or "")
    if not signature or selected.get("study_signature") != signature:
        raise V2Phase4Error("selected-spec handoff does not match Phase-3 study")
    if decision.get("study_signature") != signature:
        raise V2Phase4Error("Phase-3 decision does not match its manifest")


def _validate_fit_arrays(fit: Mapping[str, Any], *, context: str) -> None:
    items = list(map(str, fit.get("items") or []))
    dims = list(map(str, fit.get("dim_labels") or []))
    A = np.asarray(fit.get("A"), dtype=float)
    b = np.asarray(fit.get("b"), dtype=float)
    R = np.asarray(fit.get("R"), dtype=float)
    if not items or len(items) != len(set(items)) or not dims:
        raise V2Phase4Error(f"invalid item/dimension IDs in fit: {context}")
    if A.shape != (len(items), len(dims)) or b.shape != (len(items),):
        raise V2Phase4Error(f"item parameter shapes do not match fit provenance: {context}")
    if R.shape != (len(dims), len(dims)):
        raise V2Phase4Error(f"latent-correlation shape is invalid: {context}")
    if not all(np.all(np.isfinite(array)) for array in (A, b, R)):
        raise V2Phase4Error(f"fit arrays contain non-finite values: {context}")


def _validate_exact_spec_fit(
    fit: Mapping[str, Any], spec: phase3.CalibrationSpec, *, context: str
) -> None:
    if fit.get("calibration_specification") != spec.canonical:
        raise V2Phase4Error(f"fitter changed the exact calibration spec: {context}")
    active = np.asarray(fit["A"], dtype=float)
    active = active[np.abs(active) > 0]
    if spec.family == cm.ONE_PL and (
        active.size == 0 or not np.allclose(active, 1.0, rtol=0, atol=1e-12)
    ):
        raise V2Phase4Error(f"1PL discrimination is not fixed at one: {context}")
    if spec.family == cm.LOG_SHRINKAGE_2PL and (active.size == 0 or np.any(active <= 0)):
        raise V2Phase4Error(f"shrinkage-2PL discrimination is invalid: {context}")


def require_phase3_numerical_profile(
    phase3_manifest: Mapping[str, Any], current_profile: Mapping[str, Any]
) -> None:
    """Require byte-semantic equality with Phase 3's consumed numerical lock."""

    observed = phase3_manifest.get("numerical_lock")
    if not isinstance(observed, Mapping) or dict(observed) != dict(current_profile):
        raise V2Phase4Error(
            "Phase 4 numerical lock/profile differs from the completed Phase-3 study"
        )


def replay_primary_model(
    *,
    model: str,
    row: pd.Series,
    bank: scat.FittedBank,
    scenario_records: Mapping[str, dict[str, Any]],
    quadrature: scat.Quadrature,
    policy: phase3.Policy,
    seed: int,
    top_n: int,
    max_scenarios: int,
    minimum_scored_criteria: int,
    mwle_ridge: float,
) -> dict[str, Any]:
    if policy.policy_id != PRIMARY_POLICY_ID or policy.role != "primary":
        raise V2Phase4Error("Phase 4 may replay only the frozen primary policy")
    result = scat.run_recorded_model(
        str(model),
        row,
        bank,
        scenario_records,
        quadrature,
        scat.RunSpec(
            seed=int(seed),
            top_n=int(top_n),
            max_se=float(policy.conditional_se_target),
            min_evals_per_skill=int(minimum_scored_criteria),
            min_scenarios=int(policy.minimum_scenarios),
            max_scenarios=int(max_scenarios),
            selection=str(policy.selector),
            mode="cat",
        ),
        mwle_ridge=float(mwle_ridge),
    )
    if not set(result["scenario_order"]) <= set(bank.scenario_ids):
        raise V2Phase4Error(f"{model}: CAT replay left the administration bank")
    return result


class V2Phase4Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if args.plan_only and args.resume:
            raise V2Phase4Error("--plan-only cannot combine with --resume")
        self.config_path = args.config.resolve()
        self.phase3_dir = args.phase3_dir.resolve()
        self.output_dir = args.out_dir.resolve()
        self.config = _read_json(self.config_path)
        phase3._check_frozen_config(self.config)
        configured_outputs = self.config.get("outputs") or {}
        if configured_outputs.get("never_overwrite_v1") is not True:
            raise V2Phase4Error("v2 config does not protect historical v1 outputs")
        if self.phase3_dir != _repo_path(str(configured_outputs.get("phase3") or "")):
            raise V2Phase4Error("Phase-3 directory differs from the frozen v2 output")
        if self.output_dir != _repo_path(str(configured_outputs.get("phase4") or "")):
            raise V2Phase4Error("Phase-4 directory differs from the frozen v2 output")
        self.specs = {spec.spec_id: spec for spec in phase3.load_calibration_specs(self.config)}
        policies = phase3.load_policies(self.config)
        self.primary_policy = policies[0]
        if self.primary_policy.policy_id != PRIMARY_POLICY_ID or any(
            policy.role == "primary" for policy in policies[1:]
        ):
            raise V2Phase4Error("v2 config does not contain one frozen primary policy")
        self.numerical = phase3._validate_numerical_lock(
            self.config,
            list(self.specs.values()),
            base_config_path=self.config_path,
            lock_path=getattr(args, "numerical_lock", DEFAULT_NUMERICAL_LOCK),
            followup_config_path=getattr(
                args,
                "numerical_followup_config",
                DEFAULT_NUMERICAL_FOLLOWUP_CONFIG,
            ),
            allow_pending=False,
        )

        self.phase3_manifest_path = self.phase3_dir / "manifest.json"
        self.selection_lock_path = self.phase3_dir / "pre_outer_selection_lock.json"
        self.selected_path = self.phase3_dir / "selected_calibration_specs.json"
        self.phase3_decision_path = self.phase3_dir / "phase3_decision.json"
        for path in (
            self.phase3_manifest_path,
            self.selection_lock_path,
            self.selected_path,
            self.phase3_decision_path,
        ):
            if not path.is_file():
                raise V2Phase4Error(f"required completed Phase-3 artifact is missing: {path}")
        self.phase3_manifest = _read_json(self.phase3_manifest_path)
        self.selection_lock = _read_json(self.selection_lock_path)
        self.selected = _read_json(self.selected_path)
        self.phase3_decision = _read_json(self.phase3_decision_path)
        validate_phase3_authorization(self.phase3_decision, self.selected, self.phase3_manifest)
        require_phase3_numerical_profile(self.phase3_manifest, self.numerical)
        for name, path in (
            ("pre_outer_selection_lock.json", self.selection_lock_path),
            ("selected_calibration_specs.json", self.selected_path),
            ("phase3_decision.json", self.phase3_decision_path),
        ):
            expected = _manifest_output_hash(self.phase3_manifest, name)
            if not expected or expected != _sha256(path):
                raise V2Phase4Error(f"Phase-3 manifest output hash mismatch: {name}")
        if (
            self.selection_lock.get("schema_version") != phase3.SCRIPT_SCHEMA
            or self.selection_lock.get("study_signature")
            != self.phase3_manifest.get("study_signature")
            or self.selection_lock.get("status") != "locked_before_any_outer_cat_evaluation"
            or self.selection_lock.get("outer_outcomes_used") is not False
        ):
            raise V2Phase4Error("pre-outer calibration selection lock is invalid")
        selected_ref = self.phase3_decision.get("selected_specs") or self.phase3_decision.get(
            "selected_calibration_specs"
        )
        if not isinstance(selected_ref, Mapping) or selected_ref.get("sha256") != _sha256(
            self.selected_path
        ):
            raise V2Phase4Error("Phase-3 decision does not hash its selected-spec handoff")

        self.split_path = _repo_path(
            str((self.config.get("cross_validation") or {}).get("split_manifest"))
        )
        split_hash = str(
            (self.config.get("cross_validation") or {}).get("split_manifest_sha256") or ""
        )
        if not self.split_path.is_file() or _sha256(self.split_path) != split_hash:
            raise V2Phase4Error("frozen repeated split manifest is missing or hash-mismatched")
        self.split = _read_json(self.split_path)

        self.input_paths: dict[str, Path] = {}
        self.input_hashes: dict[str, str] = {}
        for name in ("response_matrix", "rubrics", "scenarios", "judge_manifest"):
            path, expected_hash = phase3._input_entry(self.config, name)
            if not path.is_file() or _sha256(path) != expected_hash:
                raise V2Phase4Error(f"frozen {name} is missing or hash-mismatched")
            self.input_paths[name] = path
            self.input_hashes[name] = expected_hash
        self.matrix = cm.load_matrix_strict(self.input_paths["response_matrix"])
        self.scenario_records = scat.load_scenario_records(self.input_paths["scenarios"])
        self.split_audit = phase3.audit_repeated_split_manifest(
            self.split,
            expected_models=self.matrix.index,
            expected_scenarios=self.scenario_records,
        )
        self.model_to_family = {
            str(model): str(family) for model, family in self.split_audit["model_to_family"].items()
        }
        self.administration_scenario_ids = tuple(self.split_audit["administration_scenario_ids"])

        source_skills = tuple(
            (self.config.get("latent_structure") or {}).get("source_skills") or []
        )
        dimensions = (self.config.get("latent_structure") or {}).get("dimensions") or []
        if len(dimensions) != 1:
            raise V2Phase4Error("v2 Phase 4 is frozen to one ability dimension")
        dimension = dimensions[0]
        dimension_spec = f"{dimension['label']}=" + "+".join(dimension["members"])
        self.structure = cell_cv.build_structure(
            source_skills,
            dimension_spec,
            str((self.config.get("latent_structure") or {}).get("name")),
        )
        if self.structure.n_dims != 1 or tuple(self.structure.labels) != ("instruction_following",):
            raise V2Phase4Error("v2 Phase 4 latent structure changed")
        self.q_by = cm.load_q_matrix(self.input_paths["rubrics"])
        cm.validate_matrix_bank_alignment(self.matrix, self.q_by, True)
        self.source_records = scenario_cv.source_records_by_id(self.input_paths["rubrics"])

        uncertainty = self.config.get("uncertainty") or {}
        if uncertainty.get("phase4_runs_only_after_complete_phase3_pass") is not True:
            raise V2Phase4Error("config does not fail closed before Phase 4")
        if uncertainty.get("parameter_bootstrap_unit") != "outer_training_model_family":
            raise V2Phase4Error("parameter bootstrap unit is not outer-training family")
        if uncertainty.get("total_variance_method") != "law_of_total_variance":
            raise V2Phase4Error("total-variance method changed")
        if uncertainty.get("full_administration_total_se_ratio_is_diagnostic_only") is not True:
            raise V2Phase4Error("full-administration uncertainty is not diagnostic-only")
        self.n_boot = int(uncertainty.get("parameter_bootstrap_replicates", 0))
        self.minimum_valid_rate = float(
            uncertainty.get("minimum_valid_parameter_bootstrap_rate", 0)
        )
        self.maximum_p90_total_se = float(uncertainty.get("absolute_p90_total_se_maximum"))
        self.order_seeds = validate_order_seeds(uncertainty.get("order_seeds") or [])
        self.maximum_median_order_path_sd = float(uncertainty.get("maximum_median_order_path_sd"))
        if self.n_boot != 100 or not math.isclose(
            self.minimum_valid_rate, 0.90, rel_tol=0, abs_tol=0
        ):
            raise V2Phase4Error("v2 requires exactly 100 bootstraps and a 0.90 valid-draw gate")
        if not math.isclose(self.maximum_p90_total_se, 0.50, rel_tol=0, abs_tol=0):
            raise V2Phase4Error("v2 absolute p90 total-SE threshold changed")
        if not math.isclose(self.maximum_median_order_path_sd, 0.20, rel_tol=0, abs_tol=0):
            raise V2Phase4Error("v2 median order-path SD threshold changed")

        runtime = self.config.get("runtime") or {}
        cat = self.config.get("cat_policies") or {}
        self.master_seed = int(runtime.get("master_seed"))
        self.bootstrap_seed = self.master_seed
        self.cat_seed = self.master_seed
        self.top_n = int(cat.get("top_n"))
        self.max_scenarios = int(cat.get("maximum_adaptive_scenarios"))
        self.minimum_scored_criteria = int(cat.get("minimum_scored_criteria"))
        self.max_iter = int(runtime.get("fit_max_iter"))
        self.tol = float(runtime.get("fit_tolerance"))
        self.mwle_ridge = float(runtime.get("mwle_ridge"))
        self.negative_policy = str(runtime.get("negative_loading_policy"))
        self.metric_bootstrap_replicates = int(runtime.get("metric_family_bootstrap_replicates"))
        phase3_runtime = self.phase3_manifest.get("runtime") or {}
        self.max_grid_nodes = int(phase3_runtime.get("max_grid_nodes", 0))
        if self.max_grid_nodes < int(self.numerical["eap_grid"]):
            raise V2Phase4Error("Phase-3 max-grid-nodes cannot hold the frozen EAP grid")
        self._validate_phase3_runtime()

        self.panels = self._load_panels()
        self.code_hashes = _code_hashes()
        self.environment = _environment_provenance()
        self.study_signature = self._signature()
        self._base_banks: dict[str, BankBundle] = {}
        self._already_complete = False
        self._preflight_base_fits()

    def _validate_phase3_runtime(self) -> None:
        runtime = self.phase3_manifest.get("runtime") or {}
        expected = {
            "seed": self.cat_seed,
            "top_n": self.top_n,
            "max_scenarios": self.max_scenarios,
            "minimum_scored_criteria": self.minimum_scored_criteria,
            "max_iter": self.max_iter,
            "negative_policy": self.negative_policy,
        }
        for name, value in expected.items():
            if runtime.get(name) != value:
                raise V2Phase4Error(
                    f"Phase-4 {name}={value!r} differs from Phase-3 runtime {runtime.get(name)!r}"
                )
        for name, value in (("tol", self.tol), ("mwle_ridge", self.mwle_ridge)):
            try:
                observed = float(runtime.get(name))
            except (TypeError, ValueError) as error:
                raise V2Phase4Error(f"Phase-3 manifest lacks numeric runtime {name}") from error
            if not math.isclose(observed, value, rel_tol=0, abs_tol=0):
                raise V2Phase4Error(f"Phase-4 {name} differs from Phase 3")

    def _load_panels(self) -> tuple[Panel, ...]:
        config_hash = _sha256(self.config_path)
        split_hash = _sha256(self.split_path)
        if _provenance_hash(self.selected, "config") != config_hash:
            raise V2Phase4Error("selected-spec handoff config hash mismatch")
        if _provenance_hash(self.selected, "split_manifest") != split_hash:
            raise V2Phase4Error("selected-spec handoff split hash mismatch")
        selected_inputs = self.selected.get("input_hashes") or {}
        if selected_inputs != self.input_hashes:
            raise V2Phase4Error("selected-spec handoff input hashes changed")
        phase3_code_hash = str(
            (self.phase3_manifest.get("code_provenance") or {}).get("canonical_sha256") or ""
        )
        phase3_environment_hash = str(
            (self.phase3_manifest.get("environment") or {}).get("canonical_sha256") or ""
        )
        if not phase3_code_hash or self.selected.get("code_sha256") != phase3_code_hash:
            raise V2Phase4Error("selected-spec handoff code hash mismatch")
        if (
            not phase3_environment_hash
            or self.selected.get("environment_sha256") != phase3_environment_hash
        ):
            raise V2Phase4Error("selected-spec handoff environment hash mismatch")
        decision_provenance = {
            "config_sha256": config_hash,
            "split_sha256": split_hash,
            "input_hashes": self.input_hashes,
            "code_sha256": phase3_code_hash,
            "environment_sha256": phase3_environment_hash,
        }
        for name, value in decision_provenance.items():
            if self.phase3_decision.get(name) != value:
                raise V2Phase4Error(f"Phase-3 decision {name} mismatch")
        if self.selected.get("primary_policy") != asdict(self.primary_policy):
            raise V2Phase4Error("selected-spec handoff primary policy changed")
        if self.selected.get("sensitivity_promotion_allowed") is not False:
            raise V2Phase4Error("selected-spec handoff permits sensitivity promotion")
        raw_panels = self.selected.get("panels")
        if not isinstance(raw_panels, list) or len(raw_panels) != EXPECTED_PANELS:
            raise V2Phase4Error("selected-spec handoff must contain exactly 25 panels")
        locked_panels = {
            str(panel.get("panel_id")): panel for panel in (self.selection_lock.get("panels") or [])
        }
        if len(locked_panels) != EXPECTED_PANELS:
            raise V2Phase4Error("pre-outer lock must contain exactly 25 panels")
        split_panels = {
            (int(repetition["repeat"]), int(outer["outer_fold"])): outer
            for repetition in self.split_audit["repetitions"]
            for outer in repetition["outer_folds"]
        }
        output: list[Panel] = []
        seen: set[tuple[int, int]] = set()
        for raw in raw_panels:
            repeat = int(raw.get("repeat", -1))
            outer_fold = int(raw.get("outer_fold", -1))
            key = (repeat, outer_fold)
            if key in seen or key not in split_panels:
                raise V2Phase4Error(f"foreign or duplicate Phase-3 panel: {key}")
            seen.add(key)
            expected_id = f"repeat_{repeat:02d}_outer_{outer_fold:02d}"
            if raw.get("panel_id") != expected_id or raw.get("status") != ("evaluation_complete"):
                raise V2Phase4Error(f"Phase-3 panel is not complete: {expected_id}")
            frozen = split_panels[key]
            train = tuple(sorted(map(str, raw.get("train_model_ids") or [])))
            test = tuple(sorted(map(str, raw.get("test_model_ids") or [])))
            if train != tuple(sorted(frozen["train_model_ids"])) or test != tuple(
                sorted(frozen["test_model_ids"])
            ):
                raise V2Phase4Error(f"Phase-3 panel changed frozen split: {expected_id}")
            assert_fit_boundary(train, test)
            train_families = tuple(sorted({self.model_to_family[model] for model in train}))
            test_families = tuple(sorted({self.model_to_family[model] for model in test}))
            if train_families != tuple(sorted(map(str, raw.get("train_family_ids") or []))):
                raise V2Phase4Error(f"training-family handoff mismatch: {expected_id}")
            if test_families != tuple(sorted(map(str, raw.get("test_family_ids") or []))):
                raise V2Phase4Error(f"test-family handoff mismatch: {expected_id}")
            if set(train_families) & set(test_families):
                raise V2Phase4Error(f"family leakage in Phase-3 handoff: {expected_id}")
            spec_id = str(raw.get("selected_spec_id") or "")
            spec = self.specs.get(spec_id)
            selected_specification = raw.get("selected_calibration_specification")
            if spec is None or selected_specification != spec.canonical:
                raise V2Phase4Error(f"exact calibration specification changed: {expected_id}")
            locked = locked_panels.get(expected_id) or {}
            for name in (
                "repeat",
                "outer_fold",
                "selected_spec_id",
                "selected_calibration_specification",
                "inner_selection",
            ):
                if locked.get(name) != raw.get(name):
                    raise V2Phase4Error(
                        f"post-lock calibration selection changed {name}: {expected_id}"
                    )
            inner = raw.get("inner_selection") or {}
            if inner.get("used_outer_outcomes") is not False or not inner.get("evidence_sha256"):
                raise V2Phase4Error(f"inner-only selection provenance missing: {expected_id}")
            outer_evaluation = raw.get("outer_evaluation") or {}
            if outer_evaluation.get("status") not in {"complete", "pass", "complete_pass"}:
                raise V2Phase4Error(f"outer evaluation incomplete: {expected_id}")
            if outer_evaluation.get("primary_rows_complete") is not True:
                raise V2Phase4Error(f"primary rows incomplete: {expected_id}")
            evaluation_checkpoint = _repo_path(str(outer_evaluation.get("checkpoint_path") or ""))
            evaluation_checkpoint_hash = str(outer_evaluation.get("checkpoint_sha256") or "")
            if (
                not evaluation_checkpoint.is_file()
                or not evaluation_checkpoint_hash
                or _sha256(evaluation_checkpoint) != evaluation_checkpoint_hash
            ):
                raise V2Phase4Error(f"outer-evaluation checkpoint failed provenance: {expected_id}")
            expected_evaluation_rows = len(test) * (1 + len(phase3.SENSITIVITY_POLICY_IDS))
            if int(outer_evaluation.get("n_rows", -1)) != expected_evaluation_rows:
                raise V2Phase4Error(f"outer-evaluation row count changed: {expected_id}")
            outer_fit = raw.get("outer_fit") or {}
            cache_key = str(outer_fit.get("cache_key") or "")
            cache_dir = _repo_path(str(outer_fit.get("cache_dir") or ""))
            manifest_path = _repo_path(str(outer_fit.get("manifest_path") or ""))
            manifest_hash = str(outer_fit.get("manifest_sha256") or "")
            if (
                not cache_key
                or not manifest_hash
                or manifest_path != cache_dir / "fit_manifest.json"
            ):
                raise V2Phase4Error(f"outer-fit provenance incomplete: {expected_id}")
            if tuple(sorted(map(str, outer_fit.get("training_model_ids") or []))) != train:
                raise V2Phase4Error(f"outer-fit training IDs changed: {expected_id}")
            if (
                outer_fit.get("spec_id") != spec.spec_id
                or outer_fit.get("calibration_specification") != spec.canonical
            ):
                raise V2Phase4Error(f"outer-fit exact specification changed: {expected_id}")
            output.append(
                Panel(
                    panel_id=expected_id,
                    repeat=repeat,
                    outer_fold=outer_fold,
                    training_model_ids=train,
                    test_model_ids=test,
                    training_family_ids=train_families,
                    test_family_ids=test_families,
                    selected_spec=spec,
                    selected_specification=dict(selected_specification),
                    outer_fit_cache_key=cache_key,
                    outer_fit_cache_dir=cache_dir,
                    outer_fit_manifest_path=manifest_path,
                    outer_fit_manifest_sha256=manifest_hash,
                )
            )
        if seen != set(split_panels):
            raise V2Phase4Error("selected-spec handoff lacks complete panel support")
        for repeat in range(EXPECTED_REPEATS):
            test_models = [
                model
                for panel in output
                if panel.repeat == repeat
                for model in panel.test_model_ids
            ]
            if len(test_models) != EXPECTED_MODELS or set(test_models) != set(self.matrix.index):
                raise V2Phase4Error(f"repeat {repeat} does not outer-test all 52 models once")
        return tuple(sorted(output, key=lambda panel: (panel.repeat, panel.outer_fold)))

    def _signature(self) -> str:
        return _canonical_hash(
            {
                "schema_version": SCRIPT_SCHEMA,
                "config_sha256": _sha256(self.config_path),
                "phase3_manifest_sha256": _sha256(self.phase3_manifest_path),
                "pre_outer_selection_lock_sha256": _sha256(self.selection_lock_path),
                "selected_specs_sha256": _sha256(self.selected_path),
                "phase3_decision_sha256": _sha256(self.phase3_decision_path),
                "split_sha256": _sha256(self.split_path),
                "input_hashes": self.input_hashes,
                "code_hashes": self.code_hashes,
                "environment": self.environment,
                "primary_policy": asdict(self.primary_policy),
                "panels": [
                    {
                        "panel_id": panel.panel_id,
                        "train": panel.training_model_ids,
                        "test": panel.test_model_ids,
                        "spec": panel.selected_specification,
                        "outer_fit_cache_key": panel.outer_fit_cache_key,
                    }
                    for panel in self.panels
                ],
                "gates": {
                    "n_boot": self.n_boot,
                    "minimum_valid_rate": self.minimum_valid_rate,
                    "maximum_p90_total_se": self.maximum_p90_total_se,
                    "order_seeds": self.order_seeds,
                    "maximum_median_order_path_sd": self.maximum_median_order_path_sd,
                },
                "runtime": {
                    "master_seed": self.master_seed,
                    "top_n": self.top_n,
                    "max_scenarios": self.max_scenarios,
                    "minimum_scored_criteria": self.minimum_scored_criteria,
                    "max_iter": self.max_iter,
                    "tol": self.tol,
                    "mwle_ridge": self.mwle_ridge,
                    "negative_policy": self.negative_policy,
                },
            }
        )

    def _load_phase3_fit(self, panel: Panel) -> dict[str, Any]:
        path = panel.outer_fit_manifest_path
        arrays_path = panel.outer_fit_cache_dir / "fit_arrays.npz"
        if not path.is_file() or not arrays_path.is_file():
            raise V2Phase4Error(f"outer-fit cache is missing: {panel.panel_id}")
        if _sha256(path) != panel.outer_fit_manifest_sha256:
            raise V2Phase4Error(f"outer-fit manifest hash mismatch: {panel.panel_id}")
        manifest = _read_json(path)
        content_hash = str(manifest.pop("manifest_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(manifest):
            raise V2Phase4Error(f"outer-fit manifest content hash mismatch: {panel.panel_id}")
        expected = {
            "schema_version": phase3.CACHE_SCHEMA,
            "study_signature": self.phase3_manifest["study_signature"],
            "cache_key": panel.outer_fit_cache_key,
            "repeat": panel.repeat,
            "outer_fold": panel.outer_fold,
            "training_model_ids": list(panel.training_model_ids),
            "spec_id": panel.selected_spec.spec_id,
            "calibration_specification": panel.selected_specification,
            "config_sha256": _sha256(self.config_path),
            "split_sha256": _sha256(self.split_path),
            "input_hashes": self.input_hashes,
            "code_sha256": str(
                (self.phase3_manifest.get("code_provenance") or {}).get("canonical_sha256")
            ),
            "environment_sha256": str(
                (self.phase3_manifest.get("environment") or {}).get("canonical_sha256")
            ),
            "numerical_lock": self.numerical,
        }
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise V2Phase4Error(f"outer-fit manifest changed {name}: {panel.panel_id}")
        if manifest.get("arrays_sha256") != _sha256(arrays_path):
            raise V2Phase4Error(f"outer-fit arrays hash mismatch: {panel.panel_id}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            fit = {
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
                "calibration_specification": dict(manifest["calibration_specification"]),
                "diag": manifest.get("diagnostics") or {},
            }
        _validate_fit_arrays(fit, context=panel.panel_id)
        _validate_exact_spec_fit(fit, panel.selected_spec, context=panel.panel_id)
        if not fit["converged"]:
            raise V2Phase4Error(f"outer-fit calibration did not converge: {panel.panel_id}")
        return fit

    def _bundle_from_fit(self, fit: Mapping[str, Any], cache_key: str) -> BankBundle:
        bank, _policy = scenario_cv.build_fold_bank(
            fit,
            self.structure,
            self.source_records,
            negative_policy=self.negative_policy,
        )
        bank = v1_phase3.subset_fitted_bank(bank, self.administration_scenario_ids)
        quadrature = scat.build_quadrature(
            bank.n_dims,
            int(self.numerical["eap_grid"]),
            bank.latent_correlation,
            max_nodes=self.max_grid_nodes,
            method=str(self.numerical["quadrature_method"]),
            linear_bound=float(self.numerical["linear_bound"]),
        )
        return BankBundle(bank=bank, quadrature=quadrature, fit=dict(fit), cache_key=cache_key)

    def _preflight_base_fits(self) -> None:
        for panel in self.panels:
            bundle = self._bundle_from_fit(self._load_phase3_fit(panel), panel.outer_fit_cache_key)
            if set(bundle.bank.scenario_ids) - set(self.administration_scenario_ids):
                raise V2Phase4Error(f"base bank escaped administration split: {panel.panel_id}")
            self._base_banks[panel.panel_id] = bundle

    def _bootstrap_cache_key(
        self, panel: Panel, replicate: int, sample: FamilyBootstrapSample
    ) -> str:
        return _canonical_hash(
            {
                "schema_version": BOOTSTRAP_CACHE_SCHEMA,
                "study_signature": self.study_signature,
                "panel_id": panel.panel_id,
                "repeat": panel.repeat,
                "outer_fold": panel.outer_fold,
                "replicate": replicate,
                "sampled_family_ids": sample.family_ids,
                "sampled_model_ids": sample.model_ids,
                "outer_training_model_ids": panel.training_model_ids,
                "outer_test_model_ids": panel.test_model_ids,
                "exact_spec_id": panel.selected_spec.spec_id,
                "exact_specification": panel.selected_specification,
                "config_sha256": _sha256(self.config_path),
                "input_hashes": self.input_hashes,
                "code_sha256": _canonical_hash(self.code_hashes),
                "environment_sha256": self.environment["canonical_sha256"],
                "numerical": self.numerical,
                "optimizer": {"max_iter": self.max_iter, "tol": self.tol},
            }
        )

    def _bootstrap_cache_dir(self, panel: Panel, replicate: int) -> Path:
        return (
            self.output_dir
            / "phase4_cache"
            / "bootstrap_fits"
            / panel.panel_id
            / f"replicate_{replicate:04d}"
        )

    def _load_bootstrap_fit(
        self,
        panel: Panel,
        replicate: int,
        sample: FamilyBootstrapSample,
        cache_key: str,
    ) -> dict[str, Any] | None:
        cache_dir = self._bootstrap_cache_dir(panel, replicate)
        manifest_path = cache_dir / "fit_manifest.json"
        arrays_path = cache_dir / "fit_arrays.npz"
        if not manifest_path.exists() and not arrays_path.exists():
            return None
        if not manifest_path.is_file() or not arrays_path.is_file():
            raise V2Phase4Error(f"partial bootstrap cache: {cache_dir}")
        manifest = _read_json(manifest_path)
        content_hash = str(manifest.pop("manifest_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(manifest):
            raise V2Phase4Error(f"bootstrap manifest content hash mismatch: {cache_dir}")
        expected = {
            "schema_version": BOOTSTRAP_CACHE_SCHEMA,
            "study_signature": self.study_signature,
            "cache_key": cache_key,
            "panel_id": panel.panel_id,
            "repeat": panel.repeat,
            "outer_fold": panel.outer_fold,
            "replicate": replicate,
            "outer_training_model_ids": list(panel.training_model_ids),
            "outer_test_model_ids_forbidden": list(panel.test_model_ids),
            "sampled_family_ids": list(sample.family_ids),
            "sampled_family_counts": dict(sorted(Counter(sample.family_ids).items())),
            "sampled_model_ids": list(sample.model_ids),
            "sampled_model_counts": dict(sorted(Counter(sample.model_ids).items())),
            "spec_id": panel.selected_spec.spec_id,
            "calibration_specification": panel.selected_specification,
            "config_sha256": _sha256(self.config_path),
            "phase3_selected_specs_sha256": _sha256(self.selected_path),
            "input_hashes": self.input_hashes,
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
            "numerical_lock": self.numerical,
            "optimizer": {"max_iter": self.max_iter, "tol": self.tol},
        }
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise V2Phase4Error(f"bootstrap cache mismatch for {name}: {cache_dir}")
        assert_fit_boundary(panel.training_model_ids, panel.test_model_ids, sample.model_ids)
        if manifest.get("arrays_sha256") != _sha256(arrays_path):
            raise V2Phase4Error(f"bootstrap arrays hash mismatch: {cache_dir}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            fit = {
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
                "calibration_specification": dict(manifest["calibration_specification"]),
                "diag": manifest.get("diagnostics") or {},
            }
        _validate_fit_arrays(fit, context=str(cache_dir))
        _validate_exact_spec_fit(fit, panel.selected_spec, context=str(cache_dir))
        return fit

    def _save_bootstrap_fit(
        self,
        panel: Panel,
        replicate: int,
        sample: FamilyBootstrapSample,
        cache_key: str,
        fit: Mapping[str, Any],
        bank_policy: Mapping[str, Any],
    ) -> None:
        cache_dir = self._bootstrap_cache_dir(panel, replicate)
        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays_path = cache_dir / "fit_arrays.npz"
        temporary = arrays_path.with_name(arrays_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                A=np.asarray(fit["A"], dtype=float),
                b=np.asarray(fit["b"], dtype=float),
                R=np.asarray(fit["R"], dtype=float),
            )
        temporary.replace(arrays_path)
        payload = {
            "schema_version": BOOTSTRAP_CACHE_SCHEMA,
            "study_signature": self.study_signature,
            "cache_key": cache_key,
            "panel_id": panel.panel_id,
            "repeat": panel.repeat,
            "outer_fold": panel.outer_fold,
            "replicate": replicate,
            "outer_training_model_ids": list(panel.training_model_ids),
            "outer_test_model_ids_forbidden": list(panel.test_model_ids),
            "sampled_family_ids": list(sample.family_ids),
            "sampled_family_counts": dict(sorted(Counter(sample.family_ids).items())),
            "sampled_model_ids": list(sample.model_ids),
            "sampled_model_counts": dict(sorted(Counter(sample.model_ids).items())),
            "spec_id": panel.selected_spec.spec_id,
            "calibration_specification": panel.selected_specification,
            "config_sha256": _sha256(self.config_path),
            "phase3_selected_specs_sha256": _sha256(self.selected_path),
            "input_hashes": self.input_hashes,
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
            "numerical_lock": self.numerical,
            "optimizer": {"max_iter": self.max_iter, "tol": self.tol},
            "items": list(map(str, fit["items"])),
            "dim_labels": list(map(str, fit["dim_labels"])),
            "loglik": fit["loglik"],
            "n_params": fit["n_params"],
            "n_iter": fit["n_iter"],
            "converged": fit["converged"],
            "diagnostics": fit.get("diag") or {},
            "bank_policy": dict(bank_policy),
            "arrays_sha256": _sha256(arrays_path),
        }
        payload["manifest_content_sha256"] = _canonical_hash(payload)
        _atomic_json(cache_dir / "fit_manifest.json", payload)

    def _fit_bootstrap(
        self, panel: Panel, replicate: int, sample: FamilyBootstrapSample
    ) -> BankBundle:
        assert_fit_boundary(panel.training_model_ids, panel.test_model_ids, sample.model_ids)
        cache_key = self._bootstrap_cache_key(panel, replicate, sample)
        fit = self._load_bootstrap_fit(panel, replicate, sample, cache_key)
        if fit is None:
            fit_args = argparse.Namespace(
                grid=int(self.numerical["fit_grid"]),
                estimate_latent_corr=False,
                ridge=(0.0 if panel.selected_spec.ridge is None else panel.selected_spec.ridge),
                log_a_shrinkage=(
                    cm.DEFAULT_LOG_A_SHRINKAGE
                    if panel.selected_spec.log_a_shrinkage is None
                    else panel.selected_spec.log_a_shrinkage
                ),
                calibration_model=panel.selected_spec.family,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            sampled_matrix = self.matrix.loc[list(sample.model_ids)]
            if set(map(str, sampled_matrix.index)) & set(panel.test_model_ids):
                raise V2Phase4Error("outer-test model reached bootstrap calibration matrix")
            fit = cell_cv.fit_structure(sampled_matrix, self.q_by, fit_args, self.structure)
            if fit.get("calibration_specification") != panel.selected_specification:
                raise V2Phase4Error("bootstrap fitter changed the selected exact specification")
            if not fit.get("converged"):
                raise RecoverableNumericalFailure("bootstrap calibration did not converge")
            _validate_fit_arrays(fit, context=f"{panel.panel_id}/{replicate}")
            _validate_exact_spec_fit(
                fit,
                panel.selected_spec,
                context=f"{panel.panel_id}/{replicate}",
            )
            _bank, policy = scenario_cv.build_fold_bank(
                fit,
                self.structure,
                self.source_records,
                negative_policy=self.negative_policy,
            )
            self._save_bootstrap_fit(panel, replicate, sample, cache_key, fit, policy)
        elif not fit.get("converged"):
            raise RecoverableNumericalFailure("cached bootstrap calibration did not converge")
        return self._bundle_from_fit(fit, cache_key)

    def _checkpoint_path(self, kind: str, panel: Panel, index: int) -> Path:
        name = f"replicate_{index:04d}.json" if kind == "bootstrap" else f"seed_{index}.json"
        return self.output_dir / "phase4_checkpoints" / kind / panel.panel_id / name

    def _checkpoint_context(
        self,
        kind: str,
        panel: Panel,
        index: int,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "panel_id": panel.panel_id,
            "repeat": panel.repeat,
            "outer_fold": panel.outer_fold,
            "index": int(index),
            "test_model_ids": list(panel.test_model_ids),
            "primary_policy": asdict(self.primary_policy),
            "spec_id": panel.selected_spec.spec_id,
            "calibration_specification": panel.selected_specification,
            "config_sha256": _sha256(self.config_path),
            "phase3_selected_specs_sha256": _sha256(self.selected_path),
            "input_bundle_sha256": _canonical_hash(self.input_hashes),
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
            **dict(extra),
        }

    def _load_checkpoint(
        self,
        path: Path,
        *,
        context: Mapping[str, Any],
        panel: Panel,
        allowed_statuses: set[str],
    ) -> list[dict[str, Any]] | None:
        if not (self.args.resume and path.is_file()):
            return None
        payload = _read_json(path)
        content_hash = str(payload.pop("checkpoint_content_sha256", ""))
        if not content_hash or content_hash != _canonical_hash(payload):
            raise V2Phase4Error(f"checkpoint content hash mismatch: {path}")
        expected = {
            "schema_version": CHECKPOINT_SCHEMA,
            "study_signature": self.study_signature,
            "context": dict(context),
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise V2Phase4Error(f"checkpoint mismatch for {name}: {path}")
        rows = payload.get("rows")
        if not isinstance(rows, list) or payload.get("rows_sha256") != _canonical_hash(rows):
            raise V2Phase4Error(f"checkpoint rows failed integrity: {path}")
        if len(rows) != len(panel.test_model_ids):
            raise V2Phase4Error(f"checkpoint lacks complete outer-test support: {path}")
        if set(str(row.get("model")) for row in rows) != set(panel.test_model_ids):
            raise V2Phase4Error(f"checkpoint model support changed: {path}")
        if any(str(row.get("status")) not in allowed_statuses for row in rows):
            raise V2Phase4Error(f"checkpoint contains unsupported status: {path}")
        row_common = {
            "panel_id": panel.panel_id,
            "repeat": panel.repeat,
            "outer_fold": panel.outer_fold,
            "policy_id": PRIMARY_POLICY_ID,
            "spec_id": panel.selected_spec.spec_id,
            "exact_specification_sha256": _canonical_hash(panel.selected_specification),
            "input_bundle_sha256": _canonical_hash(self.input_hashes),
            "code_sha256": _canonical_hash(self.code_hashes),
            "environment_sha256": self.environment["canonical_sha256"],
        }
        kind = str(context["kind"])
        index = int(context["index"])
        for row in rows:
            expected_row = {
                **row_common,
                ("replicate" if kind == "bootstrap" else "order_seed"): index,
                (
                    "bootstrap_fit_cache_key" if kind == "bootstrap" else "base_fit_cache_key"
                ): context[
                    "bootstrap_fit_cache_key" if kind == "bootstrap" else "base_fit_cache_key"
                ],
            }
            for name, value in expected_row.items():
                if row.get(name) != value:
                    raise V2Phase4Error(f"checkpoint row mismatch for {name}: {path}")
            if row.get("status") == "ok" and "mwle_converged" not in row:
                raise V2Phase4Error(f"successful checkpoint lacks MWLE provenance: {path}")
        fit_failures = [row.get("status") == "fit_numerical_error" for row in rows]
        if any(fit_failures) and not all(fit_failures):
            raise V2Phase4Error(f"bootstrap fit failure must cover the complete panel: {path}")
        return [dict(row) for row in rows]

    def _write_checkpoint(
        self, path: Path, *, context: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
    ) -> None:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "study_signature": self.study_signature,
            "context": dict(context),
            "rows_sha256": _canonical_hash(rows),
            "rows": list(rows),
        }
        payload["checkpoint_content_sha256"] = _canonical_hash(payload)
        _atomic_json(path, payload)

    def _run_bootstraps(self) -> pd.DataFrame:
        all_rows: list[dict[str, Any]] = []
        for panel in self.panels:
            for replicate in range(self.n_boot):
                seed = _replicate_seed(
                    self.bootstrap_seed, panel.repeat, panel.outer_fold, replicate
                )
                sample = draw_outer_training_family_sample(
                    panel.training_model_ids,
                    panel.test_model_ids,
                    self.model_to_family,
                    seed=seed,
                )
                cache_key = self._bootstrap_cache_key(panel, replicate, sample)
                extra = {
                    "bootstrap_seed": seed,
                    "sampled_family_ids": list(sample.family_ids),
                    "sampled_model_ids": list(sample.model_ids),
                    "bootstrap_fit_cache_key": cache_key,
                }
                context = self._checkpoint_context("bootstrap", panel, replicate, extra)
                checkpoint = self._checkpoint_path("bootstrap", panel, replicate)
                rows = self._load_checkpoint(
                    checkpoint,
                    context=context,
                    panel=panel,
                    allowed_statuses={"ok", "fit_numerical_error", "replay_numerical_error"},
                )
                if rows is not None:
                    if any(row.get("status") != "fit_numerical_error" for row in rows):
                        cached_fit = self._load_bootstrap_fit(panel, replicate, sample, cache_key)
                        if cached_fit is None:
                            raise V2Phase4Error(
                                "successful bootstrap checkpoint lacks its exact fit "
                                f"cache: {checkpoint}"
                            )
                    all_rows.extend(rows)
                    continue
                try:
                    bundle = self._fit_bootstrap(panel, replicate, sample)
                except (
                    RecoverableNumericalFailure,
                    cm.CalibrationError,
                    np.linalg.LinAlgError,
                    FloatingPointError,
                ) as error:
                    rows = [
                        {
                            "panel_id": panel.panel_id,
                            "repeat": panel.repeat,
                            "outer_fold": panel.outer_fold,
                            "replicate": replicate,
                            "bootstrap_seed": seed,
                            "model": model,
                            "model_family": self.model_to_family[model],
                            "policy_id": PRIMARY_POLICY_ID,
                            "spec_id": panel.selected_spec.spec_id,
                            "exact_specification_sha256": _canonical_hash(
                                panel.selected_specification
                            ),
                            "input_bundle_sha256": _canonical_hash(self.input_hashes),
                            "code_sha256": _canonical_hash(self.code_hashes),
                            "environment_sha256": self.environment["canonical_sha256"],
                            "bootstrap_fit_cache_key": cache_key,
                            "status": "fit_numerical_error",
                            "error": f"{type(error).__name__}: {error}",
                            "mwle_converged": False,
                        }
                        for model in panel.test_model_ids
                    ]
                else:
                    if bundle.cache_key != cache_key:
                        raise V2Phase4Error("bootstrap fit returned a foreign cache key")
                    rows = []
                    for model in panel.test_model_ids:
                        try:
                            result = replay_primary_model(
                                model=model,
                                row=self.matrix.loc[model],
                                bank=bundle.bank,
                                scenario_records=self.scenario_records,
                                quadrature=bundle.quadrature,
                                policy=self.primary_policy,
                                seed=self.cat_seed,
                                top_n=self.top_n,
                                max_scenarios=self.max_scenarios,
                                minimum_scored_criteria=self.minimum_scored_criteria,
                                mwle_ridge=self.mwle_ridge,
                            )
                            flattened = scat.flatten_result(result, bundle.bank.dims)
                            row = {"status": "ok", "error": "", **flattened}
                        except (
                            scat.OfflineStudyError,
                            np.linalg.LinAlgError,
                            FloatingPointError,
                        ) as error:
                            row = {
                                "model": model,
                                "status": "replay_numerical_error",
                                "error": f"{type(error).__name__}: {error}",
                                "mwle_converged": False,
                                "scenario_order": "[]",
                                "criterion_order": "[]",
                            }
                        row.update(
                            {
                                "panel_id": panel.panel_id,
                                "repeat": panel.repeat,
                                "outer_fold": panel.outer_fold,
                                "replicate": replicate,
                                "bootstrap_seed": seed,
                                "model_family": self.model_to_family[model],
                                "policy_id": PRIMARY_POLICY_ID,
                                "spec_id": panel.selected_spec.spec_id,
                                "exact_specification_sha256": _canonical_hash(
                                    panel.selected_specification
                                ),
                                "input_bundle_sha256": _canonical_hash(self.input_hashes),
                                "code_sha256": _canonical_hash(self.code_hashes),
                                "environment_sha256": self.environment["canonical_sha256"],
                                "bootstrap_fit_cache_key": cache_key,
                            }
                        )
                        rows.append(row)
                self._write_checkpoint(checkpoint, context=context, rows=rows)
                all_rows.extend(rows)
        return (
            pd.DataFrame(all_rows)
            .sort_values(["repeat", "outer_fold", "replicate", "model"])
            .reset_index(drop=True)
        )

    def _run_order_seeds(self) -> pd.DataFrame:
        all_rows: list[dict[str, Any]] = []
        for panel in self.panels:
            bundle = self._base_banks[panel.panel_id]
            for seed in self.order_seeds:
                extra = {"order_seed": seed, "base_fit_cache_key": bundle.cache_key}
                context = self._checkpoint_context("order", panel, seed, extra)
                checkpoint = self._checkpoint_path("order", panel, seed)
                rows = self._load_checkpoint(
                    checkpoint,
                    context=context,
                    panel=panel,
                    allowed_statuses={"ok", "replay_numerical_error"},
                )
                if rows is not None:
                    all_rows.extend(rows)
                    continue
                rows = []
                for model in panel.test_model_ids:
                    try:
                        result = replay_primary_model(
                            model=model,
                            row=self.matrix.loc[model],
                            bank=bundle.bank,
                            scenario_records=self.scenario_records,
                            quadrature=bundle.quadrature,
                            policy=self.primary_policy,
                            seed=seed,
                            top_n=self.top_n,
                            max_scenarios=self.max_scenarios,
                            minimum_scored_criteria=self.minimum_scored_criteria,
                            mwle_ridge=self.mwle_ridge,
                        )
                        flattened = scat.flatten_result(result, bundle.bank.dims)
                        row = {"status": "ok", "error": "", **flattened}
                    except (
                        scat.OfflineStudyError,
                        np.linalg.LinAlgError,
                        FloatingPointError,
                    ) as error:
                        row = {
                            "model": model,
                            "status": "replay_numerical_error",
                            "error": f"{type(error).__name__}: {error}",
                            "mwle_converged": False,
                            "scenario_order": "[]",
                            "criterion_order": "[]",
                        }
                    row.update(
                        {
                            "panel_id": panel.panel_id,
                            "repeat": panel.repeat,
                            "outer_fold": panel.outer_fold,
                            "order_seed": seed,
                            "model_family": self.model_to_family[model],
                            "policy_id": PRIMARY_POLICY_ID,
                            "spec_id": panel.selected_spec.spec_id,
                            "exact_specification_sha256": _canonical_hash(
                                panel.selected_specification
                            ),
                            "input_bundle_sha256": _canonical_hash(self.input_hashes),
                            "code_sha256": _canonical_hash(self.code_hashes),
                            "environment_sha256": self.environment["canonical_sha256"],
                            "base_fit_cache_key": bundle.cache_key,
                        }
                    )
                    rows.append(row)
                self._write_checkpoint(checkpoint, context=context, rows=rows)
                all_rows.extend(rows)
        return (
            pd.DataFrame(all_rows)
            .sort_values(["repeat", "outer_fold", "order_seed", "model"])
            .reset_index(drop=True)
        )

    def _total_se(self, draws: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for panel in self.panels:
            subset = draws[
                (draws["repeat"] == panel.repeat) & (draws["outer_fold"] == panel.outer_fold)
            ]
            cat = law_total_variance(
                subset,
                model_ids=panel.test_model_ids,
                dimensions=self.structure.labels,
                n_boot=self.n_boot,
                minimum_valid_rate=self.minimum_valid_rate,
                estimator="mwle",
            )
            full = law_total_variance(
                subset,
                model_ids=panel.test_model_ids,
                dimensions=self.structure.labels,
                n_boot=self.n_boot,
                minimum_valid_rate=self.minimum_valid_rate,
                estimator="full_eap",
            )
            frames.append(combine_cat_and_full_total_se(cat, full))
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["repeat", "outer_fold", "model", "dimension"])
            .reset_index(drop=True)
        )

    def _order_stability(self, rows: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for panel in self.panels:
            subset = rows[
                (rows["repeat"] == panel.repeat) & (rows["outer_fold"] == panel.outer_fold)
            ]
            frames.append(
                order_stability_frame(
                    subset,
                    model_ids=panel.test_model_ids,
                    dimensions=self.structure.labels,
                    order_seeds=self.order_seeds,
                )
            )
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["repeat", "outer_fold", "model", "dimension"])
            .reset_index(drop=True)
        )

    def _prepare_output(self) -> None:
        manifest_path = self.output_dir / "manifest.json"
        if self.args.resume:
            if not manifest_path.is_file():
                raise V2Phase4Error("--resume requires an existing v2 Phase-4 manifest")
            manifest = _read_json(manifest_path)
            if (
                manifest.get("schema_version") != SCRIPT_SCHEMA
                or manifest.get("study_signature") != self.study_signature
            ):
                raise V2Phase4Error("resume target does not match this exact Phase-4 study")
            if str(manifest.get("status", "")).startswith("phase4_complete"):
                for name in FINAL_OUTPUTS:
                    path = self.output_dir / name
                    if not path.is_file() or _manifest_output_hash(manifest, name) != _sha256(path):
                        raise V2Phase4Error(f"completed output failed hash validation: {name}")
                if manifest.get("checkpoint_tree_sha256") != _tree_hash(
                    self.output_dir / "phase4_checkpoints"
                ):
                    raise V2Phase4Error("completed checkpoint inventory failed hash validation")
                if manifest.get("bootstrap_cache_tree_sha256") != _tree_hash(
                    self.output_dir / "phase4_cache"
                ):
                    raise V2Phase4Error(
                        "completed bootstrap-cache inventory failed hash validation"
                    )
                self._already_complete = True
                return
        elif self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise V2Phase4Error(
                f"output directory is not empty: {self.output_dir}; use --resume for "
                "an interrupted identical study or choose a new versioned output leaf"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _base_manifest(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": SCRIPT_SCHEMA,
            "status": status,
            "created_at": _utcnow(),
            "study_signature": self.study_signature,
            "script": "scripts/nested_cat_total_uncertainty_v2.py",
            "command": sys.argv,
            "git_commit": _git_commit(),
            "inputs": {
                "config": {
                    "path": _display_path(self.config_path),
                    "sha256": _sha256(self.config_path),
                },
                "phase3_manifest": {
                    "path": _display_path(self.phase3_manifest_path),
                    "sha256": _sha256(self.phase3_manifest_path),
                },
                "pre_outer_selection_lock": {
                    "path": _display_path(self.selection_lock_path),
                    "sha256": _sha256(self.selection_lock_path),
                },
                "selected_calibration_specs": {
                    "path": _display_path(self.selected_path),
                    "sha256": _sha256(self.selected_path),
                },
                "phase3_decision": {
                    "path": _display_path(self.phase3_decision_path),
                    "sha256": _sha256(self.phase3_decision_path),
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
            "phase3_study_signature": self.phase3_manifest["study_signature"],
            "code_provenance": {
                "files": self.code_hashes,
                "canonical_sha256": _canonical_hash(self.code_hashes),
            },
            "environment": self.environment,
            "primary_policy": asdict(self.primary_policy),
            "sensitivity_policies_entered_phase4": False,
            "panels": [
                {
                    "panel_id": panel.panel_id,
                    "repeat": panel.repeat,
                    "outer_fold": panel.outer_fold,
                    "training_model_ids": list(panel.training_model_ids),
                    "test_model_ids": list(panel.test_model_ids),
                    "training_family_ids": list(panel.training_family_ids),
                    "test_family_ids": list(panel.test_family_ids),
                    "selected_spec_id": panel.selected_spec.spec_id,
                    "selected_calibration_specification": panel.selected_specification,
                    "outer_fit_cache_key": panel.outer_fit_cache_key,
                    "outer_fit_manifest_sha256": panel.outer_fit_manifest_sha256,
                }
                for panel in self.panels
            ],
            "uncertainty_design": {
                "parameter_bootstrap_replicates_per_panel": self.n_boot,
                "parameter_bootstrap_unit": "outer_training_model_family",
                "outer_test_models_forbidden_from_refits": True,
                "rerun_primary_cat_path_under_every_bootstrap_bank": True,
                "cat_total_variance": (
                    "mean conditional variance + between-bootstrap theta variance"
                ),
                "full_administration_total_variance": "same formula; diagnostic only",
                "order_seeds": list(self.order_seeds),
                "model_repeat_rows_treated_as_independent": False,
                "aggregate_per_model_before_family_bootstrap": True,
            },
            "gates": {
                "minimum_per_model_valid_draw_rate": self.minimum_valid_rate,
                "maximum_p90_total_se_within_each_repetition": self.maximum_p90_total_se,
                "require_every_order_seed_valid": True,
                "maximum_median_order_path_sd_within_each_repetition": (
                    self.maximum_median_order_path_sd
                ),
                "require_every_repetition_pass": True,
            },
            "limitations": [
                "same 52-model cohort is internal development, not independent confirmation",
                "results are conditional on frozen Qwen labels not human-validated on InFoBench",
            ],
        }

    def plan_payload(self) -> dict[str, Any]:
        return {
            **self._base_manifest("plan_only"),
            "read_only": True,
            "phase3_authorized": True,
            "panel_count": len(self.panels),
            "estimated_bootstrap_refits": len(self.panels) * self.n_boot,
            "estimated_bootstrap_cat_replays": EXPECTED_REPEATS * EXPECTED_MODELS * self.n_boot,
            "estimated_order_cat_replays": EXPECTED_REPEATS
            * EXPECTED_MODELS
            * len(self.order_seeds),
            "output_directory": _display_path(self.output_dir),
        }

    def run(self) -> int:
        if self.args.plan_only:
            print(json.dumps(_json_ready(self.plan_payload()), indent=2, sort_keys=True))
            return 0
        self._prepare_output()
        if self._already_complete:
            return 0
        running = self._base_manifest("phase4_running")
        _atomic_json(self.output_dir / "manifest.json", running)
        try:
            draws = self._run_bootstraps()
            _atomic_csv(self.output_dir / "parameter_bootstrap_draws.csv", draws)
            total = self._total_se(draws)
            total["policy_id"] = PRIMARY_POLICY_ID
            total["input_bundle_sha256"] = _canonical_hash(self.input_hashes)
            total["code_sha256"] = _canonical_hash(self.code_hashes)
            total["environment_sha256"] = self.environment["canonical_sha256"]
            _atomic_csv(self.output_dir / "outer_total_se.csv", total)

            seed_rows = self._run_order_seeds()
            _atomic_csv(self.output_dir / "order_seed_runs.csv", seed_rows)
            order = self._order_stability(seed_rows)
            order["policy_id"] = PRIMARY_POLICY_ID
            order["input_bundle_sha256"] = _canonical_hash(self.input_hashes)
            order["code_sha256"] = _canonical_hash(self.code_hashes)
            order["environment_sha256"] = self.environment["canonical_sha256"]
            _atomic_csv(self.output_dir / "outer_order_stability.csv", order)

            gates = repetition_gate_results(
                total,
                order,
                expected_model_ids=list(map(str, self.matrix.index)),
                repeats=range(EXPECTED_REPEATS),
                minimum_valid_rate=self.minimum_valid_rate,
                maximum_p90_total_se=self.maximum_p90_total_se,
                maximum_median_order_path_sd=self.maximum_median_order_path_sd,
            )
            panel_specs = {
                repeat: {
                    str(panel.outer_fold): panel.selected_spec.spec_id
                    for panel in self.panels
                    if panel.repeat == repeat
                }
                for repeat in range(EXPECTED_REPEATS)
            }
            gates["outer_fold_selected_specs"] = gates["repeat"].map(
                lambda repeat: json.dumps(
                    panel_specs[int(repeat)], sort_keys=True, separators=(",", ":")
                )
            )
            gates["policy_id"] = PRIMARY_POLICY_ID
            gates["input_bundle_sha256"] = _canonical_hash(self.input_hashes)
            gates["code_sha256"] = _canonical_hash(self.code_hashes)
            gates["environment_sha256"] = self.environment["canonical_sha256"]
            _atomic_csv(self.output_dir / "repetition_gate_results.csv", gates)
            aggregate = aggregate_repeated_models(
                total,
                order,
                expected_model_ids=list(map(str, self.matrix.index)),
                expected_repeats=range(EXPECTED_REPEATS),
            )
            _atomic_csv(self.output_dir / "model_repeat_aggregate.csv", aggregate)
            family_summary = family_cluster_bootstrap_summary(
                aggregate,
                self.model_to_family,
                seed=self.master_seed,
                replicates=self.metric_bootstrap_replicates,
            )
            _atomic_csv(self.output_dir / "family_bootstrap_summary.csv", family_summary)

            passed = bool(len(gates) == EXPECTED_REPEATS and gates["repetition_phase4_pass"].all())
            failed_repeats = (
                gates.loc[~gates["repetition_phase4_pass"].astype(bool), "repeat"]
                .astype(int)
                .tolist()
            )
            decision = {
                "schema_version": DECISION_SCHEMA,
                "status": "pass" if passed else "fail",
                "phase4_pass": passed,
                "phase3_study_signature": self.phase3_manifest["study_signature"],
                "study_signature": self.study_signature,
                "config_sha256": _sha256(self.config_path),
                "phase3_manifest_sha256": _sha256(self.phase3_manifest_path),
                "pre_outer_selection_lock_sha256": _sha256(self.selection_lock_path),
                "selected_calibration_specs_sha256": _sha256(self.selected_path),
                "phase3_decision_sha256": _sha256(self.phase3_decision_path),
                "input_hashes": self.input_hashes,
                "code_sha256": _canonical_hash(self.code_hashes),
                "environment_sha256": self.environment["canonical_sha256"],
                "primary_policy": asdict(self.primary_policy),
                "panel_selected_specifications": [
                    {
                        "panel_id": panel.panel_id,
                        "repeat": panel.repeat,
                        "outer_fold": panel.outer_fold,
                        "spec_id": panel.selected_spec.spec_id,
                        "calibration_specification": panel.selected_specification,
                    }
                    for panel in self.panels
                ],
                "sensitivity_policies_considered": False,
                "all_five_repetitions_pass": passed,
                "failed_repeats": failed_repeats,
                "gate_table": {
                    "path": "repetition_gate_results.csv",
                    "sha256": _sha256(self.output_dir / "repetition_gate_results.csv"),
                },
                "full_administration_total_se_is_diagnostic_only": True,
                "model_repeat_rows_treated_as_independent": False,
                "family_bootstrap_unit": "tutor_family",
                "final_fit_authorized": passed,
            }
            _atomic_json(self.output_dir / "phase4_decision.json", decision)
            summary = {
                "schema_version": SCRIPT_SCHEMA,
                "status": decision["status"],
                "phase4_pass": passed,
                "n_panels": len(self.panels),
                "n_bootstrap_refits_requested": len(self.panels) * self.n_boot,
                "n_bootstrap_draw_rows": len(draws),
                "n_total_se_model_repeat_rows": len(total),
                "n_order_seed_rows": len(seed_rows),
                "n_order_model_repeat_rows": len(order),
                "n_model_aggregates": len(aggregate),
                "failed_repeats": failed_repeats,
                "repetition_gates": gates.to_dict(orient="records"),
            }
            _atomic_json(self.output_dir / "phase4_summary.json", summary)
        except Exception as error:
            failed = {
                **running,
                "status": "phase4_aborted",
                "aborted_at": _utcnow(),
                "error": f"{type(error).__name__}: {error}",
            }
            _atomic_json(self.output_dir / "manifest.json", failed)
            raise

        manifest = {
            **running,
            "status": "phase4_complete_pass" if passed else "phase4_complete_fail",
            "completed_at": _utcnow(),
            "decision": decision,
            "outputs": {
                name: {"path": name, "sha256": _sha256(self.output_dir / name)}
                for name in FINAL_OUTPUTS
            },
            "checkpoint_tree_sha256": _tree_hash(self.output_dir / "phase4_checkpoints"),
            "bootstrap_cache_tree_sha256": _tree_hash(self.output_dir / "phase4_cache"),
        }
        _atomic_json(self.output_dir / "manifest.json", manifest)
        return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--numerical-followup-config",
        type=Path,
        default=DEFAULT_NUMERICAL_FOLLOWUP_CONFIG,
    )
    parser.add_argument("--numerical-lock", type=Path, default=DEFAULT_NUMERICAL_LOCK)
    parser.add_argument("--plan-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        return V2Phase4Runner(args).run()
    except (
        V2Phase4Error,
        phase3.V2Phase3Error,
        v1_phase3.NestedCVError,
        scat.OfflineStudyError,
        cm.CalibrationError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
