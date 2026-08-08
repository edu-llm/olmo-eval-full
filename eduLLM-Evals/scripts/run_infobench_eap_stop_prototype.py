"""Run the frozen InFoBench online-SE versus EAP-posterior stop prototype.

This is an offline, post-calibration experiment. It reuses the 25 selected
family-held-out 2PL banks from the completed 2PL-only Phase 3 study, makes no
tutor/judge calls, and never refits item parameters. The online theta/covariance
state continues to drive item selection in every arm; only the stopping
statistic changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import nested_scenario_cat_cv as nested  # noqa: E402
from scripts import scenario_cat_lib as scat  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "infobench_eap_stop_prototype_v1.json"
SCHEMA = "infobench-eap-stop-prototype-v1"


class PrototypeError(RuntimeError):
    """Raised when a frozen input, replay, or acceptance contract is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrototypeError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PrototypeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PrototypeError(f"{path}:{line_number}: row is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise PrototypeError(f"cannot read JSONL {path}: {error}") from error
    return rows


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _verify_inputs(config: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, record in (config.get("inputs") or {}).items():
        if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
            raise PrototypeError(f"invalid frozen input record: {name}")
        path = _resolve(str(record["path"]))
        if not path.is_file():
            raise PrototypeError(f"missing frozen input {name}: {path}")
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise PrototypeError(
                f"frozen input hash mismatch for {name}: {observed} != {record['sha256']}"
            )
        paths[str(name)] = path
    return paths


def _validate_contract(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if config.get("schema_version") != SCHEMA or config.get("status") != "frozen_pre_run":
        raise PrototypeError("prototype config is not the frozen v1 pre-run contract")
    parent = _read_json(paths["parent_config"])
    policy = config["policy"]
    if parent.get("cat_policies", {}).get("primary") != {
        "policy_id": policy["policy_id"],
        "role": "primary",
        "minimum_scenarios": policy["minimum_scenarios"],
        "conditional_se_target": policy["conditional_se_target"],
        "selector": policy["selector"],
    }:
        raise PrototypeError("frozen prototype policy does not reproduce the parent primary policy")
    parent_policy = parent["cat_policies"]
    if (
        parent_policy.get("top_n") != policy["top_n"]
        or parent_policy.get("maximum_adaptive_scenarios") != policy["maximum_scenarios"]
        or parent_policy.get("minimum_scored_criteria") != policy["minimum_scored_criteria"]
        or parent.get("runtime", {}).get("master_seed") != policy["master_seed"]
    ):
        raise PrototypeError("prototype runtime settings differ from the frozen parent study")

    numerical = _read_json(paths["numerical_lock"])
    quadrature = config["quadrature"]
    if not (
        numerical.get("passed") is True
        and numerical.get("status") == "complete_pass"
        and numerical.get("reference_eap_scoring")
        == [quadrature["method"], quadrature["primary_nodes"], quadrature["linear_bound"]]
    ):
        raise PrototypeError("V4 dense-grid prerequisite is not the expected complete pass")

    selected = _read_json(paths["selected_calibration_specs"])
    folds = _read_json(paths["fold_assignments"])
    panels = selected.get("panels") or []
    expected = config["expected_scope"]
    if len(panels) != expected["panels"]:
        raise PrototypeError("selected panel count differs from the frozen scope")
    if any(panel.get("selected_spec_id") != expected["selected_spec_id"] for panel in panels):
        raise PrototypeError("one or more panels selected an unexpected calibration specification")
    if any(panel.get("status") != "evaluation_complete" for panel in panels):
        raise PrototypeError("one or more source panels are incomplete")
    return selected, folds, [dict(panel) for panel in panels]


def _source_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        criterion_id = str(row.get("criterion_id") or "")
        if not criterion_id or criterion_id in records:
            raise PrototypeError(f"blank or duplicate criterion_id in {path}: {criterion_id!r}")
        records[criterion_id] = row
    return records


def _load_panel_bank(
    panel: Mapping[str, Any],
    source: Mapping[str, dict[str, Any]],
) -> tuple[scat.FittedBank, dict[str, Any]]:
    outer_fit = panel.get("outer_fit") or {}
    manifest_path = _resolve(str(outer_fit.get("manifest_path") or ""))
    cache_dir = _resolve(str(outer_fit.get("cache_dir") or ""))
    array_path = cache_dir / "fit_arrays.npz"
    if not manifest_path.is_file() or not array_path.is_file():
        raise PrototypeError(f"missing selected outer fit for panel {panel.get('panel_id')}")
    if _sha256(manifest_path) != outer_fit.get("manifest_sha256"):
        raise PrototypeError(f"outer-fit manifest hash mismatch: {panel.get('panel_id')}")
    manifest = _read_json(manifest_path)
    content = dict(manifest)
    claimed_content_hash = str(content.pop("manifest_content_sha256", ""))
    if not claimed_content_hash or _canonical_hash(content) != claimed_content_hash:
        raise PrototypeError(f"fit-manifest content hash mismatch: {panel.get('panel_id')}")
    if _sha256(array_path) != manifest.get("arrays_sha256"):
        raise PrototypeError(f"fit-array hash mismatch: {panel.get('panel_id')}")
    expected_role = (
        f"repeat_{int(panel['repeat']):02d}/"
        f"outer_{int(panel['outer_fold']):02d}/selected_outer_fit"
    )
    if (
        manifest.get("spec_id") != panel.get("selected_spec_id")
        or manifest.get("role") != expected_role
        or sorted(manifest.get("training_model_ids") or [])
        != sorted(panel.get("train_model_ids") or [])
    ):
        raise PrototypeError(f"outer-fit provenance mismatch: {panel.get('panel_id')}")

    with np.load(array_path, allow_pickle=False) as arrays:
        A = np.asarray(arrays["A"], dtype=float)
        b = np.asarray(arrays["b"], dtype=float)
        correlation = np.asarray(arrays["R"], dtype=float)
    items = tuple(map(str, manifest.get("items") or []))
    dims = tuple(map(str, manifest.get("dim_labels") or []))
    if dims != ("instruction_following",) or A.shape != (len(items), 1) or b.shape != (len(items),):
        raise PrototypeError(f"unexpected fitted-bank shape: {panel.get('panel_id')}")
    if correlation.shape != (1, 1) or not np.allclose(correlation, np.eye(1)):
        raise PrototypeError(f"unexpected 1-D prior correlation: {panel.get('panel_id')}")
    if not (np.all(np.isfinite(A)) and np.all(A > 0) and np.all(np.isfinite(b))):
        raise PrototypeError(f"invalid fitted 2PL parameters: {panel.get('panel_id')}")
    missing = [criterion_id for criterion_id in items if criterion_id not in source]
    if missing:
        raise PrototypeError(f"fit references missing rubric records: {missing[:10]}")
    records = [source[criterion_id] for criterion_id in items]
    scenarios = tuple(str(record.get("scenario_id") or "") for record in records)
    if any(not scenario for scenario in scenarios):
        raise PrototypeError(f"fit contains a blank scenario ID: {panel.get('panel_id')}")
    bank = scat.FittedBank(
        records=records,
        dims=dims,
        criterion_ids=items,
        scenario_ids=scenarios,
        Q=(np.abs(A) > 0).astype(int),
        A=A,
        b=b,
        latent_correlation=correlation,
        source_path=str(manifest_path),
    )
    provenance = {
        "panel_id": panel["panel_id"],
        "repeat": int(panel["repeat"]),
        "outer_fold": int(panel["outer_fold"]),
        "selected_spec_id": panel["selected_spec_id"],
        "fit_cache_key": outer_fit["cache_key"],
        "fit_manifest": str(manifest_path.relative_to(ROOT)),
        "fit_manifest_sha256": _sha256(manifest_path),
        "fit_arrays": str(array_path.relative_to(ROOT)),
        "fit_arrays_sha256": _sha256(array_path),
        "n_items": len(items),
    }
    return bank, provenance


def _quadrature(nodes: int, correlation: np.ndarray, config: Mapping[str, Any]) -> scat.Quadrature:
    settings = config["quadrature"]
    return scat.build_quadrature(
        1,
        int(nodes),
        correlation,
        max_nodes=max(2000, int(nodes)),
        method=str(settings["method"]),
        linear_bound=float(settings["linear_bound"]),
    )


def _empty_eval_stats() -> dict[str, Any]:
    return {
        "n_cells": 0,
        "log_loss_sum": 0.0,
        "brier_sum": 0.0,
        "correct_count": 0,
        "observed_pass_rate": None,
        "predicted_pass_rate": None,
    }


def _historical_index(path: Path, policy_id: str) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[rows["policy_id"] == policy_id].copy()
    keys = ["repeat", "outer_fold", "model"]
    if rows.duplicated(keys).any():
        raise PrototypeError("historical primary rows contain duplicate panel/model keys")
    return rows.set_index(keys, drop=False)


def _check_online_reproduction(
    result: Mapping[str, Any],
    reference_theta: float,
    historical: pd.Series,
) -> dict[str, Any]:
    observed_path = list(map(str, json.loads(str(result["scenario_order"]))))
    expected_path = list(map(str, json.loads(str(historical["cat_scenario_order"]))))
    checks = {
        "scenario_order_exact": observed_path == expected_path,
        "scenario_count_exact": int(result["scenarios_administered"])
        == int(historical["cat_scenarios_administered"]),
        "criterion_count_exact": int(result["criteria_administered"])
        == int(historical["cat_criteria_administered"]),
        "stop_reason_exact": str(result["stop_reason"]) == str(historical["cat_stop_reason"]),
        "precision_flag_exact": bool(result["precision_reached"])
        == bool(historical["cat_precision_reached"]),
        "mwle_theta_close": math.isclose(
            float(result["theta_mwle"]),
            float(historical["theta_cat_mwle"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "reference_theta_close": math.isclose(
            float(reference_theta),
            float(historical["theta_reference"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
    }
    return {**checks, "all_exact": all(checks.values())}


def _run_one(
    *,
    model: str,
    row: pd.Series,
    bank: scat.FittedBank,
    administration_bank: scat.FittedBank,
    evaluation_indices: np.ndarray,
    scenario_records: Mapping[str, dict[str, Any]],
    quadrature: scat.Quadrature,
    arm: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    policy = config["policy"]
    spec = scat.RunSpec(
        seed=int(policy["master_seed"]),
        top_n=int(policy["top_n"]),
        max_se=float(policy["conditional_se_target"]),
        min_evals_per_skill=int(policy["minimum_scored_criteria"]),
        min_scenarios=int(policy["minimum_scenarios"]),
        max_scenarios=int(policy["maximum_scenarios"]),
        selection=str(policy["selector"]),
        mode="cat",
        stop_se_method=str(arm["stop_se_method"]),
    )
    result = scat.run_recorded_model(
        model,
        row,
        administration_bank,
        scenario_records,
        quadrature,
        spec,
        mwle_ridge=float(policy["mwle_ridge"]),
    )
    responses = scat.responses_for_bank(row, bank)
    administration_scenarios = set(administration_bank.scenario_ids)
    administration_indices = np.asarray(
        [
            index
            for index, scenario_id in enumerate(bank.scenario_ids)
            if scenario_id in administration_scenarios
        ],
        dtype=int,
    )
    reference = scat.batch_eap(
        responses,
        bank.A,
        bank.b,
        quadrature,
        item_indices=administration_indices,
    )
    if result["mwle_converged"]:
        theta = np.asarray([result["theta_mwle"]["instruction_following"]], dtype=float)
        eval_stats = nested.prediction_sufficient_statistics(
            row, bank, evaluation_indices, theta
        )
    else:
        eval_stats = _empty_eval_stats()
    final_eap_se = float(result["se_eap"]["instruction_following"])
    return {
        "arm_id": arm["arm_id"],
        "stop_se_method": arm["stop_se_method"],
        "quadrature_nodes": int(quadrature.nodes_per_dim),
        "status": "ok",
        "stop_reason": result["stop_reason"],
        "precision_reached": bool(result["precision_reached"]),
        "honest_eap_target_reached": final_eap_se
        < float(policy["conditional_se_target"]),
        "scenarios_administered": int(result["scenarios_administered"]),
        "criteria_administered": int(result["criteria_administered"]),
        "scenario_order": json.dumps(result["scenario_order"], ensure_ascii=False),
        "criterion_order": json.dumps(result["criterion_order"], ensure_ascii=False),
        "final_online_se": float(result["se_online"]["instruction_following"]),
        "final_eap_se": final_eap_se,
        "final_stop_se": float(result["se_stop"]["instruction_following"]),
        "theta_online": float(result["theta_online"]["instruction_following"]),
        "theta_eap": float(result["theta_eap"]["instruction_following"]),
        "theta_mwle": float(result["theta_mwle"]["instruction_following"]),
        "mwle_converged": bool(result["mwle_converged"]),
        "mwle_message": result["mwle_message"],
        "theta_reference": float(reference.theta[0]),
        "reference_eap_se": float(reference.se[0]),
        **{f"eval_{key}": value for key, value in eval_stats.items()},
    }


def _recovery(reference: Sequence[float], estimate: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(reference, dtype=float)
    y = np.asarray(estimate, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or float(x.std()) == 0 or float(y.std()) == 0:
        return {"n": int(len(x)), "r": None, "slope": None, "mae": None, "bias": None}
    return {
        "n": int(len(x)),
        "r": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(np.polyfit(x, y, 1)[0]),
        "mae": float(np.mean(np.abs(y - x))),
        "bias": float(np.mean(y - x)),
    }


def _aggregate_per_model(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    frame["eval_abs_pass_rate_error"] = (
        frame["eval_predicted_pass_rate"] - frame["eval_observed_pass_rate"]
    ).abs()
    grouped = (
        frame.groupby(["model", "model_family", "arm_id"], as_index=False)
        .agg(
            repeats=("repeat", "nunique"),
            theta_reference=("theta_reference", "mean"),
            theta_mwle=("theta_mwle", "mean"),
            mwle_convergence_rate=("mwle_converged", "mean"),
            honest_target_rate=("honest_eap_target_reached", "mean"),
            nominal_precision_rate=("precision_reached", "mean"),
            scenarios_mean=("scenarios_administered", "mean"),
            scenarios_median=("scenarios_administered", "median"),
            final_eap_se_mean=("final_eap_se", "mean"),
            eval_pass_rate_mae=("eval_abs_pass_rate_error", "mean"),
        )
        .sort_values(["arm_id", "model"])
    )
    return grouped


def _arm_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_models": int(len(frame)),
        "mean_honest_target_rate": float(frame["honest_target_rate"].mean()),
        "mean_nominal_precision_rate": float(frame["nominal_precision_rate"].mean()),
        "mean_mwle_convergence_rate": float(frame["mwle_convergence_rate"].mean()),
        "scenarios": {
            "mean_of_model_means": float(frame["scenarios_mean"].mean()),
            "median_of_model_means": float(frame["scenarios_mean"].median()),
            "p90_of_model_means": float(np.percentile(frame["scenarios_mean"], 90)),
        },
        "mean_final_eap_se": float(frame["final_eap_se_mean"].mean()),
        "mean_disjoint_pass_rate_mae": float(frame["eval_pass_rate_mae"].mean()),
        "recovery": _recovery(frame["theta_reference"], frame["theta_mwle"]),
    }


def _paired_models(per_model: pd.DataFrame) -> pd.DataFrame:
    online = per_model[per_model["arm_id"] == "online"].copy()
    eap = per_model[per_model["arm_id"] == "eap_801"].copy()
    keys = ["model", "model_family"]
    keep = [
        "theta_reference",
        "theta_mwle",
        "honest_target_rate",
        "scenarios_mean",
        "final_eap_se_mean",
        "eval_pass_rate_mae",
    ]
    paired = online[keys + keep].merge(
        eap[keys + keep], on=keys, suffixes=("_online", "_eap"), validate="one_to_one"
    )
    paired["target_rate_difference"] = (
        paired["honest_target_rate_eap"] - paired["honest_target_rate_online"]
    )
    paired["scenario_difference"] = (
        paired["scenarios_mean_eap"] - paired["scenarios_mean_online"]
    )
    paired["theta_absolute_error_online"] = (
        paired["theta_mwle_online"] - paired["theta_reference_online"]
    ).abs()
    paired["theta_absolute_error_eap"] = (
        paired["theta_mwle_eap"] - paired["theta_reference_eap"]
    ).abs()
    return paired


def _pair_statistics(frame: pd.DataFrame) -> dict[str, float | None]:
    online_recovery = _recovery(frame["theta_reference_online"], frame["theta_mwle_online"])
    eap_recovery = _recovery(frame["theta_reference_eap"], frame["theta_mwle_eap"])
    r_difference = (
        float(eap_recovery["r"] - online_recovery["r"])
        if eap_recovery["r"] is not None and online_recovery["r"] is not None
        else None
    )
    return {
        "target_rate_improvement": float(frame["target_rate_difference"].mean()),
        "mean_scenario_difference": float(frame["scenario_difference"].mean()),
        "median_paired_scenario_difference": float(
            frame["scenario_difference"].median()
        ),
        "difference_of_arm_median_scenarios": float(
            frame["scenarios_mean_eap"].median()
            - frame["scenarios_mean_online"].median()
        ),
        "recovery_r_difference": r_difference,
        "theta_mae_difference": float(
            frame["theta_absolute_error_eap"].mean()
            - frame["theta_absolute_error_online"].mean()
        ),
    }


def _family_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float | None]]:
    families = sorted(map(str, frame["model_family"].unique()))
    if len(families) < 2:
        raise PrototypeError("family bootstrap requires at least two model families")
    groups = {family: frame[frame["model_family"] == family] for family in families}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "target_rate_improvement": [],
        "mean_scenario_difference": [],
        "median_paired_scenario_difference": [],
        "difference_of_arm_median_scenarios": [],
        "recovery_r_difference": [],
        "theta_mae_difference": [],
    }
    for _ in range(replicates):
        sampled = rng.choice(families, size=len(families), replace=True)
        draw = pd.concat([groups[str(family)] for family in sampled], ignore_index=True)
        stats = _pair_statistics(draw)
        for key, value in stats.items():
            if value is not None and math.isfinite(float(value)):
                values[key].append(float(value))
    intervals: dict[str, list[float | None]] = {}
    for key, observed in values.items():
        intervals[key] = (
            [float(np.percentile(observed, 2.5)), float(np.percentile(observed, 97.5))]
            if observed
            else [None, None]
        )
    return intervals


def _prefix_consistent(first: str, second: str) -> bool:
    a = list(map(str, json.loads(first)))
    b = list(map(str, json.loads(second)))
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return shorter == longer[: len(shorter)]


def _numerical_sensitivity(rows: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    keys = ["repeat", "outer_fold", "model"]
    primary = rows[rows["arm_id"] == "eap_801"].copy()
    dense = rows[rows["arm_id"] == "eap_1601"].copy()
    paired = primary.merge(dense, on=keys, suffixes=("_801", "_1601"), validate="one_to_one")
    paired["stop_agreement"] = (
        (paired["scenarios_administered_801"] == paired["scenarios_administered_1601"])
        & (paired["stop_reason_801"] == paired["stop_reason_1601"])
        & (paired["precision_reached_801"] == paired["precision_reached_1601"])
        & (paired["scenario_order_801"] == paired["scenario_order_1601"])
    )
    paired["path_prefix_consistent"] = [
        _prefix_consistent(a, b)
        for a, b in zip(
            paired["scenario_order_801"], paired["scenario_order_1601"], strict=True
        )
    ]
    paired["final_eap_se_difference"] = (
        paired["final_eap_se_1601"] - paired["final_eap_se_801"]
    ).abs()
    matching = paired[paired["stop_agreement"]]
    metrics = {
        "n_cases": int(len(paired)),
        "stop_agreement_rate": float(paired["stop_agreement"].mean()),
        "path_prefix_consistency_rate": float(paired["path_prefix_consistent"].mean()),
        "maximum_final_eap_se_difference_on_matching_stops": (
            float(matching["final_eap_se_difference"].max()) if len(matching) else None
        ),
    }
    return metrics, paired


def _acceptance(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    eap = metrics["arms"]["eap_801"]
    paired = metrics["paired_online_vs_eap"]
    ci = metrics["family_bootstrap_95ci"]
    numerical = metrics["numerical_801_vs_1601"]
    gates = {
        "online_reproduction_exact": metrics["online_reproduction"]["all_exact"],
        "honest_target_rate_strictly_higher": paired["target_rate_improvement"] > 0,
        "honest_target_improvement_ci_nonnegative": (
            ci["target_rate_improvement"][0]
            is not None
            and ci["target_rate_improvement"][0]
            >= thresholds["minimum_target_rate_improvement_ci_lower"]
        ),
        "recovery_r_noninferior": paired["recovery_r_difference"]
        >= -float(thresholds["maximum_recovery_r_drop"]),
        "theta_mae_noninferior": paired["theta_mae_difference"]
        <= float(thresholds["maximum_theta_mae_increase"]),
        "eap_recovery_slope_in_range": float(thresholds["minimum_eap_recovery_slope"])
        <= eap["recovery"]["slope"]
        <= float(thresholds["maximum_eap_recovery_slope"]),
        "median_length_not_increased": paired["difference_of_arm_median_scenarios"]
        <= float(thresholds["maximum_median_scenario_increase"]),
        "dense_grid_stop_agreement": numerical["stop_agreement_rate"]
        >= float(thresholds["minimum_801_vs_1601_stop_agreement"]),
        "dense_grid_final_se_stable": numerical[
            "maximum_final_eap_se_difference_on_matching_stops"
        ]
        is not None
        and numerical["maximum_final_eap_se_difference_on_matching_stops"]
        <= float(thresholds["maximum_801_vs_1601_final_se_difference_on_matching_stops"]),
    }
    return {
        "gates": gates,
        "all_passed": all(gates.values()),
        "decision": (
            "adopt_stop_rule_for_followup_validation"
            if all(gates.values())
            else "do_not_adopt"
        ),
        "note": (
            "Even a passing prototype would not authorize deployment because the underlying "
            "CAT policy previously failed Phase-3 validation."
        ),
    }


def _render_figures(rows: pd.DataFrame, per_model: pd.DataFrame, output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    paired = _paired_models(per_model)
    fig, axis = plt.subplots(figsize=(6.2, 5.4))
    axis.scatter(paired["scenarios_mean_online"], paired["scenarios_mean_eap"], alpha=0.75)
    limit = float(
        max(paired["scenarios_mean_online"].max(), paired["scenarios_mean_eap"].max()) + 1
    )
    axis.plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1)
    axis.set(xlabel="Online-stop mean scenarios", ylabel="EAP-stop mean scenarios")
    axis.set_title("Paired InFoBench CAT length (52 models)")
    fig.tight_layout()
    path = output / "01_paired_test_length.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(6.6, 4.8))
    for arm, color in (("online", "#4C78A8"), ("eap_801", "#F58518")):
        values = rows.loc[rows["arm_id"] == arm, "final_eap_se"].to_numpy(float)
        axis.hist(values, bins=24, alpha=0.55, label=arm, color=color)
    axis.axvline(0.20, linestyle="--", color="black", linewidth=1, label="target 0.20")
    axis.set(xlabel="Final EAP posterior SD", ylabel="Model-repeat cases")
    axis.set_title("Honest posterior uncertainty at each arm's stop")
    axis.legend()
    fig.tight_layout()
    path = output / "02_final_eap_se_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=True)
    for axis, arm in zip(axes, ("online", "eap_801"), strict=True):
        frame = per_model[per_model["arm_id"] == arm]
        axis.scatter(frame["theta_reference"], frame["theta_mwle"], alpha=0.75)
        low = float(min(frame["theta_reference"].min(), frame["theta_mwle"].min()))
        high = float(max(frame["theta_reference"].max(), frame["theta_mwle"].max()))
        axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        recovery = _recovery(frame["theta_reference"], frame["theta_mwle"])
        axis.set_title(f"{arm}: r={recovery['r']:.3f}, slope={recovery['slope']:.3f}")
        axis.set_xlabel("Full administration-pool EAP theta")
    axes[0].set_ylabel("Short-test MWLE theta")
    fig.suptitle("Out-of-fold ability recovery")
    fig.tight_layout()
    path = output / "03_oos_recovery.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _format_number(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _write_report(path: Path, metrics: Mapping[str, Any]) -> None:
    online = metrics["arms"]["online"]
    eap = metrics["arms"]["eap_801"]
    paired = metrics["paired_online_vs_eap"]
    decision = metrics["acceptance"]
    gates = decision["gates"]
    ci = metrics["family_bootstrap_95ci"]
    lines = [
        "# InFoBench EAP-posterior stopping prototype",
        "",
        f"**Decision:** `{decision['decision']}`",
        "",
        "This offline experiment changed only the CAT stopping statistic. It reused the "
        "same 25 family-held-out shrinkage-2PL banks, frozen response matrix, scenario split, "
        "online selection state, policy, and seeds. No calibration, tutor calls, or judge "
        "calls ran.",
        "",
        "## Primary results",
        "",
        "| Metric | Online SE stop | EAP posterior-SD stop | EAP - online |",
        "|---|---:|---:|---:|",
        f"| Honest EAP target rate | {_format_number(online['mean_honest_target_rate'])} | "
        f"{_format_number(eap['mean_honest_target_rate'])} | "
        f"{_format_number(paired['target_rate_improvement'])} |",
        f"| Arm median of model-average scenarios | "
        f"{_format_number(online['scenarios']['median_of_model_means'], 1)} | "
        f"{_format_number(eap['scenarios']['median_of_model_means'], 1)} | "
        f"{_format_number(paired['difference_of_arm_median_scenarios'], 1)} |",
        "| Median paired per-model increase | — | — | "
        f"{_format_number(paired['median_paired_scenario_difference'], 1)} |",
        f"| Mean scenarios | {_format_number(online['scenarios']['mean_of_model_means'], 2)} | "
        f"{_format_number(eap['scenarios']['mean_of_model_means'], 2)} | "
        f"{_format_number(paired['mean_scenario_difference'], 2)} |",
        f"| Recovery r | {_format_number(online['recovery']['r'])} | "
        f"{_format_number(eap['recovery']['r'])} | "
        f"{_format_number(paired['recovery_r_difference'])} |",
        f"| Recovery slope | {_format_number(online['recovery']['slope'])} | "
        f"{_format_number(eap['recovery']['slope'])} | — |",
        f"| Theta MAE | {_format_number(online['recovery']['mae'])} | "
        f"{_format_number(eap['recovery']['mae'])} | "
        f"{_format_number(paired['theta_mae_difference'])} |",
        "",
        "The family-bootstrap 95% CI for the median paired per-model increase was "
        f"[{_format_number(ci['median_paired_scenario_difference'][0], 1)}, "
        f"{_format_number(ci['median_paired_scenario_difference'][1], 1)}] scenarios.",
        "",
        "## Preregistered acceptance gates",
        "",
    ]
    for name, passed in gates.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            decision["note"],
            "The handoff's historical floor-0/SE-0.25 setting was not used. This run uses the "
            "current floor-15/SE-0.20/trace 2PL candidate and the V4-validated 801-node "
            "normal-trapezoid grid; 1601 nodes are included as a numerical sensitivity.",
            "",
            "## Artifacts",
            "",
            "Raw rows and machine-readable metrics are in "
            "`runs/calibration/InFoBench_eap_stop_prototype_v1/`; figures and this summary "
            "are in `reports/infobench_eap_stop_prototype_v1/`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--limit-panels",
        type=int,
        default=None,
        help="deterministic engineering check only; full results require all 25 panels",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    started = datetime.now(UTC)
    config_path = _resolve(args.config)
    config = _read_json(config_path)
    paths = _verify_inputs(config)
    _, folds, panels = _validate_contract(config, paths)
    source = _source_records(paths["rubrics"])
    scenario_records = scat.load_scenario_records(paths["scenarios"])
    matrix = scat.load_response_matrix(paths["response_matrix"])
    model_to_family = {str(key): str(value) for key, value in folds["model_to_family"].items()}
    scenario_split = folds["scenario_split"]
    administration_ids = set(map(str, scenario_split["administration_scenario_ids"]))
    evaluation_ids = set(map(str, scenario_split["evaluation_scenario_ids"]))
    if (
        len(administration_ids) != 400
        or len(evaluation_ids) != 100
        or administration_ids & evaluation_ids
    ):
        raise PrototypeError("frozen scenario partition is not an exact disjoint 400/100 split")
    if set(matrix.index) != set(model_to_family):
        raise PrototypeError("response-matrix models differ from the frozen family map")

    panels.sort(key=lambda panel: (int(panel["repeat"]), int(panel["outer_fold"])))
    if args.limit_panels is not None:
        if args.limit_panels < 1:
            raise PrototypeError("--limit-panels must be positive")
        panels = panels[: args.limit_panels]

    panel_banks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for panel in panels:
        bank, provenance = _load_panel_bank(panel, source)
        if not set(panel["test_model_ids"]) <= set(matrix.index):
            raise PrototypeError(f"panel has unknown test models: {panel['panel_id']}")
        if set(panel["test_model_ids"]) & set(panel["train_model_ids"]):
            raise PrototypeError(f"train/test leakage in panel {panel['panel_id']}")
        panel_banks.append((panel, provenance))
        del bank
    if args.plan_only:
        print(
            f"validated {len(panels)} panels, {len(matrix)} models, frozen inputs, "
            "and all fit hashes"
        )
        return 0

    output_settings = config["outputs"]
    out_dir = _resolve(args.out_dir or output_settings["run_dir"])
    report_dir = _resolve(args.report_dir or output_settings["report_dir"])
    for directory in (out_dir, report_dir):
        if directory.exists() and any(directory.iterdir()):
            raise PrototypeError(f"output directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)

    historical = _historical_index(paths["historical_outer_rows"], config["policy"]["policy_id"])
    arms = {str(arm["arm_id"]): dict(arm) for arm in config["arms"]}
    rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    panel_provenance: list[dict[str, Any]] = []
    total_cases = sum(len(panel["test_model_ids"]) for panel in panels)
    completed_cases = 0

    for panel, _ in panel_banks:
        bank, provenance = _load_panel_bank(panel, source)
        panel_provenance.append(provenance)
        administration_bank = nested.subset_fitted_bank(bank, administration_ids)
        evaluation_indices = nested.evaluation_item_indices(bank, evaluation_ids)
        quadratures = {
            801: _quadrature(801, bank.latent_correlation, config),
            1601: _quadrature(1601, bank.latent_correlation, config),
        }
        for model in sorted(map(str, panel["test_model_ids"])):
            completed_cases += 1
            print(
                f"[{completed_cases}/{total_cases}] {panel['panel_id']} :: {model}",
                flush=True,
            )
            model_row = matrix.loc[model]
            model_results: dict[str, dict[str, Any]] = {}
            for arm_id in ("online", "eap_801", "eap_1601"):
                arm = arms[arm_id]
                nodes = int(
                    arm.get("eap_grid") or arm.get("eap_grid_for_posthoc_scoring") or 801
                )
                result = _run_one(
                    model=model,
                    row=model_row,
                    bank=bank,
                    administration_bank=administration_bank,
                    evaluation_indices=evaluation_indices,
                    scenario_records=scenario_records,
                    quadrature=quadratures[nodes],
                    arm=arm,
                    config=config,
                )
                result.update(
                    {
                        "repeat": int(panel["repeat"]),
                        "outer_fold": int(panel["outer_fold"]),
                        "panel_id": panel["panel_id"],
                        "model": model,
                        "model_family": model_to_family[model],
                        "selected_spec_id": panel["selected_spec_id"],
                        "fit_cache_key": provenance["fit_cache_key"],
                    }
                )
                rows.append(result)
                model_results[arm_id] = result

            key = (int(panel["repeat"]), int(panel["outer_fold"]), model)
            if key not in historical.index:
                raise PrototypeError(f"historical primary row is missing: {key}")
            check = _check_online_reproduction(
                model_results["online"],
                model_results["online"]["theta_reference"],
                historical.loc[key],
            )
            reproduction_rows.append(
                {
                    "repeat": key[0],
                    "outer_fold": key[1],
                    "model": key[2],
                    **check,
                }
            )
            if not check["all_exact"]:
                failed = [name for name, passed in check.items() if passed is False]
                raise PrototypeError(f"online reproduction failed for {key}: {failed}")

    frame = pd.DataFrame(rows).sort_values(["arm_id", "repeat", "outer_fold", "model"])
    reproduction = pd.DataFrame(reproduction_rows).sort_values(["repeat", "outer_fold", "model"])
    per_model = _aggregate_per_model(frame)
    paired = _paired_models(per_model)
    arm_metrics = {
        arm_id: _arm_metrics(per_model[per_model["arm_id"] == arm_id])
        for arm_id in ("online", "eap_801", "eap_1601")
    }
    pair_stats = _pair_statistics(paired)
    bootstrap = _family_bootstrap(
        paired,
        replicates=int(config["aggregation"]["bootstrap_replicates"]),
        seed=int(config["aggregation"]["bootstrap_seed"]),
    )
    numerical_metrics, numerical_rows = _numerical_sensitivity(frame)
    metrics: dict[str, Any] = {
        "scope": {
            "panels": len(panels),
            "model_repeat_cases": int(len(frame) // 3),
            "unique_models": int(frame["model"].nunique()),
            "unique_model_families": int(frame["model_family"].nunique()),
            "full_frozen_scope": args.limit_panels is None,
        },
        "online_reproduction": {
            "n_cases": int(len(reproduction)),
            "all_exact": bool(reproduction["all_exact"].all()),
        },
        "arms": arm_metrics,
        "paired_online_vs_eap": pair_stats,
        "family_bootstrap_95ci": bootstrap,
        "numerical_801_vs_1601": numerical_metrics,
    }
    metrics["acceptance"] = _acceptance(metrics, config["acceptance"])

    raw_paths = {
        "per_model_repeat.csv": out_dir / "per_model_repeat.csv",
        "per_model_aggregated.csv": out_dir / "per_model_aggregated.csv",
        "paired_online_vs_eap.csv": out_dir / "paired_online_vs_eap.csv",
        "online_reproduction.csv": out_dir / "online_reproduction.csv",
        "eap_801_vs_1601.csv": out_dir / "eap_801_vs_1601.csv",
        "panel_provenance.json": out_dir / "panel_provenance.json",
        "metrics.json": out_dir / "metrics.json",
    }
    frame.to_csv(raw_paths["per_model_repeat.csv"], index=False)
    per_model.to_csv(raw_paths["per_model_aggregated.csv"], index=False)
    paired.to_csv(raw_paths["paired_online_vs_eap.csv"], index=False)
    reproduction.to_csv(raw_paths["online_reproduction.csv"], index=False)
    numerical_rows.to_csv(raw_paths["eap_801_vs_1601.csv"], index=False)
    raw_paths["panel_provenance.json"].write_text(
        json.dumps(_json_ready(panel_provenance), indent=2), encoding="utf-8"
    )
    raw_paths["metrics.json"].write_text(
        json.dumps(_json_ready(metrics), indent=2), encoding="utf-8"
    )

    summary_path = report_dir / "EAP_STOP_PROTOTYPE_SUMMARY.md"
    _write_report(summary_path, metrics)
    figure_paths = _render_figures(frame, per_model, report_dir / "figures")

    code_paths = [
        ROOT / "tutor_cat" / "mirt.py",
        ROOT / "tutor_cat" / "engine.py",
        ROOT / "scripts" / "scenario_cat_lib.py",
        Path(__file__).resolve(),
    ]
    finished = datetime.now(UTC)
    manifest = {
        "schema_version": SCHEMA,
        "status": "complete",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "git_head": _git_head(),
        "config": {
            "path": _manifest_path(config_path),
            "sha256": _sha256(config_path),
        },
        "inputs": config["inputs"],
        "code_sha256": {_manifest_path(path): _sha256(path) for path in code_paths},
        "panel_provenance": panel_provenance,
        "outputs": {
            _manifest_path(path): _sha256(path)
            for path in [*raw_paths.values(), summary_path, *figure_paths]
        },
        "acceptance": metrics["acceptance"],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")
    print(f"wrote raw prototype artifacts -> {out_dir}")
    print(f"wrote shareable report -> {report_dir}")
    print(f"decision: {metrics['acceptance']['decision']}")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (
        PrototypeError,
        scat.OfflineStudyError,
        nested.NestedCVError,
        OSError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
