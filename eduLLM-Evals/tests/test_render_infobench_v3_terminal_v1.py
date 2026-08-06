# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import render_infobench_v3_terminal_v1 as report  # noqa: E402

PHASE4_COMMAND = "\n".join(
    [
        "uv run --frozen --extra irt python scripts/nested_cat_total_uncertainty_v3.py \\",
        "  --config configs/infobench_calibration_cat_v3.json \\",
        "  --phase3-dir runs/calibration/InFoBench_v3/phase3 \\",
        "  --out-dir runs/calibration/InFoBench_v3/phase4 \\",
        "  --numerical-followup-config configs/infobench_v2_numerical_followup_v4.json \\",
        "  --numerical-lock runs/calibration/InFoBench_v2_numerical_followup_v4/numerical_followup_lock.json",
    ]
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _config(path: Path) -> None:
    sources = path.parent / "phase3_sources"
    response_matrix = sources / "response_matrix.csv"
    rubrics = sources / "rubrics.jsonl"
    scenarios = sources / "scenarios.jsonl"
    judge_manifest = sources / "judge_manifest.json"
    split_manifest = sources / "split_manifest.json"
    response_matrix.parent.mkdir(parents=True, exist_ok=True)
    response_matrix.write_text("model,c001\nm00,1\n", encoding="utf-8")
    rubrics.write_text('{"criterion_id":"c001"}\n', encoding="utf-8")
    scenarios.write_text('{"scenario_id":"s001"}\n', encoding="utf-8")
    _json(judge_manifest, {"judge": "fixture"})
    _json(split_manifest, {"fixture": "split-source"})
    _json(
        path,
        {
            "schema_version": report.V3_CONFIG_SCHEMA,
            "code_dependencies": list(report.V3_CODE_DEPENDENCIES),
            "benchmark": "InFoBench",
            "baseline": {
                "response_matrix": str(response_matrix.resolve()),
                "response_matrix_sha256": _sha(response_matrix),
                "rubrics": str(rubrics.resolve()),
                "rubrics_sha256": _sha(rubrics),
                "scenarios": str(scenarios.resolve()),
                "scenarios_sha256": _sha(scenarios),
                "judge_manifest": str(judge_manifest.resolve()),
                "judge_manifest_sha256": _sha(judge_manifest),
            },
            "cross_validation": {
                "split_manifest": str(split_manifest.resolve()),
                "split_manifest_sha256": _sha(split_manifest),
                "repetitions": 5,
                "outer_folds_per_repetition": 5,
                "inner_folds_per_outer_panel": 4,
                "group_related_model_families": True,
                "aggregate_repeated_predictions_per_model": True,
                "family_cluster_bootstrap": True,
                "treat_model_repeat_rows_as_independent": False,
            },
            "outputs": {
                "numerical_checks": "runs/calibration/InFoBench_v2_numerical_followup_v4",
                "phase3": "runs/calibration/InFoBench_v3/phase3",
                "phase4": "runs/calibration/InFoBench_v3/phase4",
                "final_fit": "runs/calibration/InFoBench_v3/final_fit",
                "report": "reports/infobench_calibration_cat_v3",
                "never_overwrite_v1_or_v2": True,
            },
            "latent_structure": {
                "name": "overall_1d",
                "source_skills": ["content", "format", "number", "style", "linguistic"],
                "dimensions": [
                    {
                        "label": "instruction_following",
                        "members": ["content", "format", "number", "style", "linguistic"],
                    }
                ],
            },
            "calibration_specifications": [
                {
                    "spec_id": spec_id,
                    "family": "1pl" if index == 0 else "log-shrinkage-2pl",
                    "ridge": None,
                    "log_a_shrinkage": None if index == 0 else (16 if index == 1 else 4),
                    "simplicity_rank": index,
                    "canonical_cache_key": cm.calibration_specification(
                        "1pl" if index == 0 else "log-shrinkage-2pl",
                        log_a_shrinkage=(
                            cm.DEFAULT_LOG_A_SHRINKAGE
                            if index == 0
                            else (16.0 if index == 1 else 4.0)
                        ),
                    )["cache_key"],
                }
                for index, spec_id in enumerate(report.SPEC_IDS)
            ],
            "numerical_lock": {
                "fit_quadrature_method": "normal_trapezoid",
                "fit_linear_bound": 8,
                "fit_grid": 401,
                "convergence_mode": "returned_iterate",
                "objective_tolerance": 1e-4,
                "parameter_tolerance": 5e-5,
                "consecutive_convergence_passes": 2,
                "eap_quadrature_method": "normal_trapezoid",
                "eap_linear_bound": 8,
                "eap_grid": 801,
                "tail_region": 7.5,
                "eligible_spec_ids": list(report.SPEC_IDS),
            },
            "cat_policies": {
                "primary": {
                    "policy_id": report.PRIMARY_POLICY,
                    "role": "primary",
                    "minimum_scenarios": 15,
                    "conditional_se_target": 0.2,
                    "selector": "trace",
                },
                "sensitivities": [
                    {
                        "policy_id": "sensitivity_floor12_se0p20_trace",
                        "role": "sensitivity_floor",
                        "minimum_scenarios": 12,
                        "conditional_se_target": 0.2,
                        "selector": "trace",
                    },
                    {
                        "policy_id": "sensitivity_floor15_se0p20_dopt",
                        "role": "sensitivity_selector",
                        "minimum_scenarios": 15,
                        "conditional_se_target": 0.2,
                        "selector": "dopt",
                    },
                ],
                "sensitivities_can_replace_primary": False,
                "top_n": 5,
                "maximum_adaptive_scenarios": 50,
                "minimum_scored_criteria": 15,
            },
            "selection_gates": {
                "apply_to_primary_only": True,
                "allow_fallback_if_none_pass": False,
                "allow_fallback_if_primary_fails": False,
                "require_every_outer_panel": True,
                "require_every_repetition_pooled_gate": True,
                "minimum_nominal_precision_lower_95_ci": 0.90,
                "minimum_scenario_reduction_vs_random": 0.50,
                "outer_panel_applied_gate_names": list(report.PANEL_APPLIED_GATE_NAMES),
                "outer_panel_diagnostic_gate_names": list(report.PANEL_DIAGNOSTIC_GATE_NAMES),
                "repetition_applied_gate_names": list(report.REPETITION_APPLIED_GATE_NAMES),
                "outer_panel_inference_unit": (
                    "unique_outer_test_tutor_within_panel; confidence intervals diagnostic only"
                ),
                "repetition_inference_unit": "52_unique_out_of_fold_tutors_once_per_repetition",
                "repetition_expected_unique_oof_tutors": 52,
                "cross_repeat_gate_role": "diagnostic_only_after_per_model_aggregation",
                "treat_260_model_repeat_rows_as_independent": False,
            },
            "runtime": {
                "master_seed": 20260805,
                "fit_max_iter": 1500,
                "fit_tolerance": 1e-4,
                "fit_parameter_tolerance": 5e-5,
                "fit_consecutive_convergence_passes": 2,
                "fit_convergence_mode": "returned_iterate",
                "fit_quadrature_method": "normal_trapezoid",
                "fit_linear_bound": 8,
                "mwle_ridge": 1e-6,
                "negative_loading_policy": "drop",
                "metric_family_bootstrap_replicates": 2000,
            },
        },
    )


def _inner_evidence_audit(*, passed: bool) -> dict[str, object]:
    validation_by_fold = {str(inner_fold): [f"m{inner_fold:02d}"] for inner_fold in range(4)}
    combinations: list[dict[str, object]] = []
    failures: list[str] = []
    for inner_fold in range(4):
        for spec_id in report.SPEC_IDS:
            checks = {name: True for name in report.INNER_COMBINATION_CHECK_NAMES}
            if not passed and inner_fold == 0 and spec_id == report.SPEC_IDS[0]:
                checks["status_ok"] = False
                failures.append(f"inner_{inner_fold}_{spec_id}_status_ok")
            combinations.append(
                {
                    "inner_fold": inner_fold,
                    "spec_id": spec_id,
                    "expected_validation_models": validation_by_fold[str(inner_fold)],
                    "observed_validation_models": validation_by_fold[str(inner_fold)],
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    valid = sum(bool(combination["passed"]) for combination in combinations)
    return {
        "passed": passed,
        "require_all_specs_every_inner_fold": True,
        "survivor_selection_allowed": False,
        "expected_inner_folds": list(range(4)),
        "expected_spec_ids": list(report.SPEC_IDS),
        "expected_spec_fold_combinations": 12,
        "valid_spec_fold_combinations": valid,
        "validation_models_by_inner_fold": validation_by_fold,
        "failed_checks": sorted(failures),
        "combination_audits": combinations,
    }


def _gate_scope_contract() -> dict[str, object]:
    return {
        "outer_panel": {
            "decision_role": "applied_point_estimate_gates",
            "inference_unit": (
                "unique_outer_test_tutor_within_panel; confidence intervals diagnostic only"
            ),
            "applied_gate_names": list(report.PANEL_APPLIED_GATE_NAMES),
            "diagnostic_only_gate_names": list(report.PANEL_DIAGNOSTIC_GATE_NAMES),
            "confidence_intervals_affect_decision": False,
        },
        "repetition": {
            "decision_role": "applied_all_absolute_gates",
            "inference_unit": "52_unique_out_of_fold_tutors_once_per_repetition",
            "expected_unique_oof_tutors": 52,
            "applied_gate_names": list(report.REPETITION_APPLIED_GATE_NAMES),
            "diagnostic_only_gate_names": [],
            "nominal_precision_lower_95_ci_threshold": 0.90,
        },
        "cross_repeat": {
            "decision_role": "diagnostic_only_after_per_model_aggregation",
            "inference_unit": "52_unique_tutors_after_within_tutor_mean_across_5_repetitions",
            "raw_model_repeat_rows": 260,
            "raw_model_repeat_rows_treated_as_independent": False,
        },
    }


def _v4_fixture(root: Path, *, passed: bool) -> Path:
    v4 = root / "v4"
    v4.mkdir(parents=True, exist_ok=True)
    v4_config = root / "v4_config.json"
    frozen_contract = {"fixture": "dense-numerical-contract"}
    _json(
        v4_config,
        {
            "schema_version": report.V4_CONFIG_SCHEMA,
            "output_dir": str(v4.resolve()),
            "frozen_contract": frozen_contract,
        },
    )
    frozen_input = root / "v4_frozen_input.txt"
    frozen_code = root / "v4_frozen_code.py"
    frozen_input.write_text("fixture input\n", encoding="utf-8")
    frozen_code.write_text("VALUE = 1\n", encoding="utf-8")
    environment_payload = {"python": "fixture"}
    environment = {
        **environment_payload,
        "canonical_sha256": report._canonical_hash(environment_payload),
    }
    signature_payload = {
        "schema_version": report.V4_RUN_SCHEMA,
        "config_sha256": _sha(v4_config),
        "frozen_contract_sha256": report._canonical_hash(frozen_contract),
        "input_sha256": {str(frozen_input.resolve()): _sha(frozen_input)},
        "code_sha256": {str(frozen_code.resolve()): _sha(frozen_code)},
        "historical_evidence_sha256": {"fixture": True},
        "environment_sha256": environment["canonical_sha256"],
        "spec_ids": list(report.SPEC_IDS),
        "scope_ids": list(report.SCOPE_IDS),
        "fit_nodes": [401, 801],
        "required_total_new_fits": 54,
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "adaptive_runs": 0,
    }
    signature = {
        **signature_payload,
        "environment": environment,
        "canonical_sha256": report._canonical_hash(signature_payload),
    }
    evidence_paths: dict[str, str] = {}
    evidence_hashes: dict[str, str] = {}
    fit_validity: dict[str, bool] = {}
    start_validity: dict[str, bool] = {}

    def evidence(key: str, path: Path) -> None:
        evidence_paths[key] = str(path.resolve())
        evidence_hashes[key] = _sha(path)

    def checkpoint(stage_id: str, inputs: dict[str, object], outputs: list[Path]) -> None:
        _json(
            v4 / "checkpoints" / f"{stage_id.replace('/', '__')}.json",
            {
                "schema_version": report.V4_CHECKPOINT_SCHEMA,
                "status": "completed",
                "stage_id": stage_id,
                "study_signature_sha256": signature["canonical_sha256"],
                "stage_inputs": inputs,
                "stage_inputs_sha256": report._canonical_hash(inputs),
                "output_sha256": {str(output.resolve()): _sha(output) for output in outputs},
            },
        )

    for spec_index, spec_id in enumerate(report.SPEC_IDS):
        item_rows = [
            {
                "criterion_id": f"c{index:03d}",
                "a_instruction_following": (
                    1.0 if spec_index == 0 else 0.75 + 0.02 * index + 0.1 * spec_index
                ),
                "b": -2.0 + 0.1 * index + 0.05 * spec_index,
                "exportable": True,
            }
            for index in range(40)
        ]
        for nodes, start in ((401, "cold"), (801, "cold"), (801, "continuation")):
            for scope_index, scope in enumerate(
                ("full", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4")
            ):
                fit_dir = v4 / "fits" / spec_id / f"nodes_{nodes:04d}" / scope / start
                fit_path = fit_dir / "fit.npz"
                parameter_path = fit_dir / "item_params.csv"
                trace_path = fit_dir / "convergence_trace.json"
                manifest_path = fit_dir / "fit_manifest.json"
                fit_path.parent.mkdir(parents=True, exist_ok=True)
                fit_path.write_bytes(f"fit:{spec_id}:{nodes}:{scope}:{start}".encode())
                _csv(parameter_path, item_rows)
                _json(trace_path, {"trace": [], "spec_id": spec_id})
                stage_id = f"fit/{spec_id}/nodes_{nodes:04d}/{scope}/{start}"
                stage_inputs = {
                    "spec_id": spec_id,
                    "nodes": nodes,
                    "scope": scope,
                    "start": start,
                }
                _json(
                    manifest_path,
                    {
                        "schema_version": report.V4_RUN_SCHEMA,
                        "stage_inputs": stage_inputs,
                        "stage_inputs_sha256": report._canonical_hash(stage_inputs),
                        "spec_id": spec_id,
                        "scope": scope,
                        "nodes": nodes,
                        "start": start,
                        "n_iter": 20 + spec_index * 3 + scope_index,
                        "fit_validity": {"fit_valid": True},
                        "fit_sha256": _sha(fit_path),
                        "parameter_csv_sha256": _sha(parameter_path),
                        "trace_sha256": _sha(trace_path),
                    },
                )
                checkpoint(
                    stage_id,
                    stage_inputs,
                    [fit_path, parameter_path, trace_path, manifest_path],
                )
                key = f"fit/{spec_id}/{scope}/{nodes}/{start}"
                fit_validity[key] = True
                evidence(key, manifest_path)

        for scope in ("full", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4"):
            start_path = v4 / "start_selection" / spec_id / f"{scope}.json"
            _json(
                start_path,
                {
                    "schema_version": report.V4_RUN_SCHEMA,
                    "spec_id": spec_id,
                    "scope": scope,
                    "selection_valid": True,
                },
            )
            checkpoint(
                f"start_selection/{spec_id}/{scope}",
                {"spec_id": spec_id, "scope": scope},
                [start_path],
            )
            key = f"start/{spec_id}/{scope}"
            start_validity[key] = True
            evidence(key, start_path)

        comparison = v4 / "fit_comparisons" / "401_vs_801" / spec_id
        parameter_rows = []
        theta_rows = []
        for scope_index, scope in enumerate(
            ("full", "outer_0", "outer_1", "outer_2", "outer_3", "outer_4")
        ):
            parameter_rows.append(
                {
                    "scope": scope,
                    "a_spearman": np.nan if spec_index == 0 else 0.997 - 0.0001 * scope_index,
                    "b_spearman": 0.998 - 0.0001 * scope_index,
                    "exportability_agreement": 0.999,
                    "minimum_parameter_spearman": 0.99,
                    "minimum_exportability_agreement": 0.995,
                    "passed": passed,
                }
            )
            for model_index in range(4):
                theta_rows.append(
                    {
                        "scope": scope,
                        "model": f"m{model_index:02d}",
                        "support": "full_bank",
                        "theta_401": -1.0 + 0.5 * model_index,
                        "theta_801": -1.0 + 0.5 * model_index + 0.003 + 0.001 * spec_index,
                    }
                )
        _csv(comparison / "parameter_scope_gates.csv", parameter_rows)
        _csv(comparison / "common_support.csv", [{"scope": "full", "n_items": 40}])
        _csv(comparison / "theta_common_support.csv", theta_rows)
        _csv(
            comparison / "heldout_cells.csv",
            [
                {
                    "outer_fold": index % 5,
                    "model": f"m{index % 4:02d}",
                    "criterion_id": f"c{index:03d}",
                    "log_loss_delta_801_minus_401": 0.0002,
                    "brier_delta_801_minus_401": 0.0001,
                }
                for index in range(20)
            ],
        )
        fit_pair = {
            "schema_version": report.V4_RUN_SCHEMA,
            "spec_id": spec_id,
            "comparison": "normal_trapezoid_bound8_401_vs_801",
            "all_scope_start_agreement_gates_passed": True,
            "all_scope_parameter_gates_passed": passed,
            "heldout_common_cell_gate": {
                "n_common_cells": 20,
                "passed": passed,
                "metrics": {
                    "log_loss": {
                        "pooled_shift_801_minus_401": 0.0002,
                        "family_cluster_bootstrap_ci_95": [-0.0002, 0.0006],
                        "equivalence_margin": 0.005,
                    },
                    "brier": {
                        "pooled_shift_801_minus_401": 0.0001,
                        "family_cluster_bootstrap_ci_95": [-0.0001, 0.0003],
                        "equivalence_margin": 0.005,
                    },
                },
            },
            "common_support_refit_theta_gate": {
                "median_absolute_theta_shift": 0.004,
                "p95_absolute_theta_shift": 0.008,
                "maximum_absolute_theta_shift_diagnostic_only": 0.012,
                "thresholds": {
                    "maximum_median_absolute_theta_shift": 0.02,
                    "maximum_p95_absolute_theta_shift": 0.05,
                },
                "passed": passed,
            },
            "passed": passed,
        }
        fit_pair_path = comparison / "fit_pair_gate.json"
        _json(fit_pair_path, fit_pair)
        checkpoint(
            f"fit_comparison/{spec_id}/401_vs_801",
            {"spec_id": spec_id, "comparison": "401_vs_801"},
            [
                comparison / "parameter_scope_gates.csv",
                comparison / "common_support.csv",
                comparison / "theta_common_support.csv",
                comparison / "heldout_cells.csv",
                fit_pair_path,
            ],
        )
        evidence(f"refit/{spec_id}", fit_pair_path)

        if passed:
            fixed = v4 / "fixed_bank" / spec_id
            _csv(
                fixed / "theta_profiles.csv",
                [
                    {
                        "scope": "full",
                        "model": f"m{index:02d}",
                        "support": "full_bank",
                        "theta_bound8_801": -1 + index * 0.2,
                        "theta_bound8_1601": -1 + index * 0.2 + 1e-4,
                        "theta_bound10_1001": -1 + index * 0.2 + 2e-4,
                        "tail_bound8_801": 1e-7,
                        "tail_bound8_1601": 2e-7,
                        "tail_bound10_1001": 1e-8,
                    }
                    for index in range(10)
                ],
            )
            _csv(fixed / "support.csv", [{"scope": "full", "n_score_models": 10}])
            fixed_gate = {
                "schema_version": report.V4_RUN_SCHEMA,
                "spec_id": spec_id,
                "all_six_fixed_banks_valid": True,
                "numerical_scoring_gate": {
                    "comparisons": {
                        "bound8_801_vs_bound8_1601": {
                            "maximum_absolute_theta_shift": 1e-4,
                            "threshold": 0.005,
                            "passed": True,
                        },
                        "bound8_801_vs_bound10_1001": {
                            "maximum_absolute_theta_shift": 2e-4,
                            "threshold": 0.005,
                            "passed": True,
                        },
                    },
                    "tail_mass": {
                        "maximum_posterior_tail_mass": 2e-7,
                        "threshold": 1e-5,
                        "passed": True,
                    },
                    "all_values_finite": True,
                    "passed": True,
                },
                "passed": True,
            }
            fixed_path = fixed / "fixed_bank_gate.json"
            _json(fixed_path, fixed_gate)
            checkpoint(
                f"fixed_bank/{spec_id}",
                {"spec_id": spec_id, "fixed_bank_nodes": 401},
                [fixed / "theta_profiles.csv", fixed / "support.csv", fixed_path],
            )
            evidence(f"fixed_bank/{spec_id}", fixed_path)

    decision = {
        "schema_version": report.V4_RUN_SCHEMA,
        "status": "complete_pass" if passed else "blocked_numerical_followup",
        "passed": passed,
        "eligible_spec_ids": list(report.SPEC_IDS),
        "scope_ids": list(report.SCOPE_IDS),
        "fit_validity": fit_validity,
        "start_validity": start_validity,
        "all_54_fits_valid": True,
        "all_18_grid801_start_gates_passed": True,
        "all_three_refit_bank_gates_passed": passed,
        "all_three_fixed_bank_scoring_and_tail_gates_passed": passed,
        "refit_gate_passed": {spec_id: passed for spec_id in report.SPEC_IDS},
        "fixed_bank_gate_passed": {spec_id: passed for spec_id in report.SPEC_IDS},
        "config_sha256": _sha(v4_config),
        "frozen_contract_sha256": report._canonical_hash(frozen_contract),
        "study_signature_sha256": signature["canonical_sha256"],
        "code_sha256_reverified_before_finalization": signature["code_sha256"],
        "evidence_paths": evidence_paths,
        "evidence_sha256": evidence_hashes,
    }
    decision_path = v4 / "numerical_followup_decision.json"
    _json(decision_path, decision)
    manifest = {
        "schema_version": report.V4_RUN_SCHEMA,
        "status": decision["status"],
        "config": str(v4_config.resolve()),
        "config_sha256": _sha(v4_config),
        "study_signature": signature,
        "study_signature_sha256": signature["canonical_sha256"],
        "decision_sha256": _sha(decision_path),
        "lock_sha256": None,
    }
    if passed:
        lock = {
            "schema_version": report.V4_LOCK_SCHEMA,
            "status": "complete_pass",
            "passed": True,
            "eligible_spec_ids": list(report.SPEC_IDS),
            "scope_ids": list(report.SCOPE_IDS),
            "config_sha256": _sha(v4_config),
            "frozen_contract_sha256": report._canonical_hash(frozen_contract),
            "study_signature_sha256": signature["canonical_sha256"],
            "followup_decision_sha256": _sha(decision_path),
            "evidence_paths": evidence_paths,
            "evidence_sha256": evidence_hashes,
        }
        lock_path = v4 / "numerical_followup_lock.json"
        _json(lock_path, lock)
        (v4 / "numerical_followup_lock.sha256").write_text(
            f"{_sha(lock_path)}  numerical_followup_lock.json\n", encoding="utf-8"
        )
        manifest["lock_sha256"] = _sha(lock_path)
    _json(v4 / "study_manifest.json", manifest)
    return v4


def _phase3_fixture(root: Path, *, passed: bool) -> Path:
    phase3 = root / "phase3"
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    baseline = config["baseline"]
    split_path = Path(config["cross_validation"]["split_manifest"])
    input_paths = {
        "response_matrix": Path(baseline["response_matrix"]),
        "rubrics": Path(baseline["rubrics"]),
        "scenarios": Path(baseline["scenarios"]),
        "judge_manifest": Path(baseline["judge_manifest"]),
    }
    input_hashes = {name: _sha(path) for name, path in input_paths.items()}
    code_hashes = {path: _sha(ROOT / path) for path in report.V3_CODE_DEPENDENCIES}
    code_hash = report._canonical_hash(code_hashes)
    environment_payload = {"python": "fixture-phase3"}
    environment = {
        **environment_payload,
        "canonical_sha256": report._canonical_hash(environment_payload),
    }

    v4_dir = root / "v4"
    v4_lock = json.loads((v4_dir / "numerical_followup_lock.json").read_text())
    v4_manifest = json.loads((v4_dir / "study_manifest.json").read_text())
    numerical = {
        "fit_grid": 401,
        "fit_quadrature_method": "normal_trapezoid",
        "fit_linear_bound": 8.0,
        "fit_convergence_mode": "returned_iterate",
        "fit_max_iter": 1500,
        "fit_objective_tolerance": 1e-4,
        "fit_parameter_tolerance": 5e-5,
        "fit_consecutive_convergence_passes": 2,
        "eap_grid": 801,
        "quadrature_method": "normal_trapezoid",
        "linear_bound": 8.0,
        "tail_region": 7.5,
        "status": "passed",
        "verification_path": str((v4_dir / "numerical_followup_lock.json").resolve()),
        "verification_sha256": _sha(v4_dir / "numerical_followup_lock.json"),
        "followup_config_path": v4_manifest["config"],
        "followup_config_sha256": _sha(Path(v4_manifest["config"])),
        "followup_decision_sha256": _sha(v4_dir / "numerical_followup_decision.json"),
        "followup_manifest_sha256": _sha(v4_dir / "study_manifest.json"),
        "study_signature_sha256": v4_lock["study_signature_sha256"],
        "frozen_contract_sha256": v4_lock["frozen_contract_sha256"],
        "evidence_sha256": v4_lock["evidence_sha256"],
        "passed_spec_ids": list(report.SPEC_IDS),
        "eligible_spec_ids": list(report.SPEC_IDS),
        "historical_fit_or_checkpoint_artifacts_reused": 0,
    }
    calibration_specifications = []
    signature_specs = []
    for index, configured in enumerate(config["calibration_specifications"]):
        spec_id = configured["spec_id"]
        complete_specification = cm.calibration_specification(
            configured["family"],
            ridge=0.0 if configured.get("ridge") is None else float(configured["ridge"]),
            log_a_shrinkage=(
                cm.DEFAULT_LOG_A_SHRINKAGE
                if configured.get("log_a_shrinkage") is None
                else float(configured["log_a_shrinkage"])
            ),
        )
        calibration_specifications.append(
            {
                "spec_id": spec_id,
                "simplicity_rank": index,
                "complete_specification": complete_specification,
            }
        )
        signature_specs.append(
            {
                "spec_id": spec_id,
                "family": configured.get("family"),
                "ridge": (None if configured.get("ridge") is None else float(configured["ridge"])),
                "log_a_shrinkage": (
                    None
                    if configured.get("log_a_shrinkage") is None
                    else float(configured["log_a_shrinkage"])
                ),
                "simplicity_rank": index,
                "canonical": complete_specification,
            }
        )
    cat_policies = [
        {
            "policy_id": report.PRIMARY_POLICY,
            "role": "primary",
            "minimum_scenarios": 15,
            "conditional_se_target": 0.2,
            "selector": "trace",
        },
        {
            "policy_id": "sensitivity_floor12_se0p20_trace",
            "role": "sensitivity",
            "minimum_scenarios": 12,
            "conditional_se_target": 0.2,
            "selector": "trace",
        },
        {
            "policy_id": "sensitivity_floor15_se0p20_dopt",
            "role": "sensitivity",
            "minimum_scenarios": 15,
            "conditional_se_target": 0.2,
            "selector": "dopt",
        },
    ]
    runtime = {
        "seed": 20260805,
        "top_n": 5,
        "max_scenarios": 50,
        "minimum_scored_criteria": 15,
        "max_iter": 1500,
        "tol": 1e-4,
        "fit_parameter_tolerance": 5e-5,
        "fit_consecutive_convergence_passes": 2,
        "fit_convergence_mode": "returned_iterate",
        "fit_quadrature_method": "normal_trapezoid",
        "fit_linear_bound": 8.0,
        "mwle_ridge": 1e-6,
        "max_grid_nodes": 100_000,
        "metric_bootstrap_replicates": 2_000,
        "negative_policy": "drop",
    }
    signature_runtime = {key: value for key, value in runtime.items() if key != "max_grid_nodes"}
    latent = config["latent_structure"]
    study_signature = report._canonical_hash(
        {
            "schema": report.V3_RUN_SCHEMA,
            "config_sha256": _sha(config_path),
            "split_sha256": _sha(split_path),
            "inputs": input_hashes,
            "code": code_hashes,
            "environment": environment,
            "structure": {
                "name": report.V3_STRUCTURE_NAME,
                "source_skills": latent["source_skills"],
                "dimensions": latent["dimensions"],
            },
            "numerical": numerical,
            "specifications": signature_specs,
            "policies": cat_policies,
            "runtime": signature_runtime,
        }
    )
    policies = list(report.POLICY_LABELS)
    models = [f"m{index:02d}" for index in range(52)]

    inner = []
    panels = []
    for repeat in range(5):
        for outer_fold in range(5):
            panels.append(
                {
                    "panel_id": f"repeat_{repeat:02d}_outer_{outer_fold:02d}",
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "status": "evaluation_complete",
                    "selected_spec_id": report.SPEC_IDS[0],
                    "inner_selection": {
                        "complete_inner_evidence_audit": _inner_evidence_audit(passed=True)
                    },
                    "outer_evaluation": {
                        "status": "complete",
                        "primary_rows_complete": True,
                        "n_rows": 3 * sum(index % 5 == outer_fold for index in range(52)),
                    },
                }
            )
            for spec_index, spec_id in enumerate(report.SPEC_IDS):
                inner.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "spec_id": spec_id,
                        "mean_log_loss": 0.35 + 0.01 * spec_index,
                        "family_cluster_se": 0.015,
                        "selected": spec_index == 0,
                    }
                )

    outer_rows = []
    for repeat in range(5):
        for model_index, model in enumerate(models):
            outer_fold = model_index % 5
            theta = -2 + 4 * model_index / 51
            for policy_index, policy_id in enumerate(policies):
                outer_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "policy_id": policy_id,
                        "model": model,
                        "theta_reference": theta,
                        "theta_cat_mwle": theta * (0.98 if passed else 0.80) + 0.01,
                        "cat_scenarios_administered": 15 - min(policy_index, 1),
                        "baseline_scenarios_administered": 34,
                        "cat_eval_observed_pass_rate": 0.2 + 0.6 * model_index / 51,
                        "cat_eval_predicted_pass_rate": 0.21 + 0.59 * model_index / 51,
                        "baseline_eval_observed_pass_rate": 0.2 + 0.6 * model_index / 51,
                        "baseline_eval_predicted_pass_rate": 0.22 + 0.57 * model_index / 51,
                    }
                )

    gate_rows = []
    outer_gate_checks = {
        name: passed if name == "recovery_slope" else True
        for name in report.PANEL_APPLIED_GATE_NAMES
    }
    outer_gate_checks["inference_scope_complete"] = True
    repeat_gate_checks = {
        name: passed if name == "recovery_slope" else True
        for name in report.REPETITION_APPLIED_GATE_NAMES
    }
    repeat_gate_checks["inference_scope_complete"] = True
    for repeat in range(5):
        for outer_fold in range(5):
            panel_models = sum(index % 5 == outer_fold for index in range(52))
            for policy_index, policy_id in enumerate(policies):
                primary = policy_id == report.PRIMARY_POLICY
                gate_rows.append(
                    {
                        "scope": "outer_panel",
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "policy_id": policy_id,
                        "policy_role": "primary" if primary else "sensitivity",
                        "recovery_slope": 0.98 if passed else 0.80,
                        "disjoint_pass_rate_mae": 0.04 + 0.005 * policy_index,
                        "mean_scenario_count": 15 - min(policy_index, 1),
                        "mean_random_scenario_count": 34,
                        "scenario_reduction_vs_random": 0.56,
                        "gate_status": "pass"
                        if (primary and passed)
                        else ("fail" if primary else "diagnostic_only_non_promotable"),
                        "all_gates_pass": passed if primary else None,
                        "gate_checks": json.dumps(outer_gate_checks if primary else {}),
                        "n_inference_rows": panel_models,
                        "n_unique_tutor_models": panel_models,
                        "expected_unique_tutor_models": panel_models,
                        "inference_scope_complete": True,
                        "inference_unit": _gate_scope_contract()["outer_panel"]["inference_unit"],
                        "model_repeat_rows_treated_as_independent": False,
                        "gate_scope_contract_sha256": report._canonical_hash(
                            _gate_scope_contract()
                        ),
                    }
                )
        for policy_index, policy_id in enumerate(policies):
            primary = policy_id == report.PRIMARY_POLICY
            gate_rows.append(
                {
                    "scope": "repetition",
                    "repeat": repeat,
                    "outer_fold": np.nan,
                    "policy_id": policy_id,
                    "policy_role": "primary" if primary else "sensitivity",
                    "recovery_slope": 0.98 if passed else 0.80,
                    "disjoint_pass_rate_mae": 0.04 + 0.005 * policy_index,
                    "mean_scenario_count": 15 - min(policy_index, 1),
                    "mean_random_scenario_count": 34,
                    "scenario_reduction_vs_random": 0.56,
                    "gate_status": "pass"
                    if (primary and passed)
                    else ("fail" if primary else "diagnostic_only_non_promotable"),
                    "all_gates_pass": passed if primary else None,
                    "gate_checks": json.dumps(repeat_gate_checks if primary else {}),
                    "n_inference_rows": 52,
                    "n_unique_tutor_models": 52,
                    "expected_unique_tutor_models": 52,
                    "inference_scope_complete": True,
                    "inference_unit": _gate_scope_contract()["repetition"]["inference_unit"],
                    "model_repeat_rows_treated_as_independent": False,
                    "gate_scope_contract_sha256": report._canonical_hash(_gate_scope_contract()),
                }
            )

    predictions = []
    for repeat in range(5):
        for outer_fold in range(5):
            panel_models = sum(index % 5 == outer_fold for index in range(52))
            for policy_id in policies:
                for mode in ("cat", "baseline"):
                    predictions.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "candidate_id": policy_id,
                            "mode": mode,
                            "n_models": panel_models,
                            "n_cells": 100,
                            "log_loss": 0.40 if mode == "cat" else 0.44,
                        }
                    )

    cross_model = []
    for policy_id in policies:
        for model_index, model in enumerate(models):
            theta = -2 + 4 * model_index / 51
            cross_model.append(
                {
                    "policy_id": policy_id,
                    "model": model,
                    "model_family": f"family_{model_index % 22:02d}",
                    "n_repetitions": 5,
                    "expected_repetitions": 5,
                    "repeat_rows_treated_as_independent": False,
                    "mean_theta_reference": theta,
                    "mean_theta_cat_mwle": theta * (0.98 if passed else 0.80) + 0.01,
                }
            )
    cross_metrics = [
        {
            "policy_id": policy_id,
            "policy_role": "primary" if policy_id == report.PRIMARY_POLICY else "sensitivity",
            "n_models": 52,
            "n_models_with_all_repetitions": 52,
            "n_families": 22,
            "aggregation_unit": "model_after_mean_across_repetitions",
            "bootstrap_unit": "tutor_family",
            "model_repeat_rows_treated_as_independent": False,
            "recovery_n": 52,
            "recovery_correlation": 0.98,
            "recovery_correlation_lower_95_ci": 0.90,
            "recovery_slope": 0.98 if passed else 0.80,
            "scenario_reduction_vs_random": 0.56,
            "scenario_reduction_lower_95_ci": 0.50,
            "scenario_reduction_upper_95_ci": 0.62,
            "gate_role": "diagnostic_only_cross_repeat_summary",
        }
        for policy_id in policies
    ]

    files: dict[str, Path] = {}
    files["inner_calibration_model_results.csv"] = phase3 / "inner_calibration_model_results.csv"
    _csv(files["inner_calibration_model_results.csv"], inner)
    files["outer_oof_per_model.csv"] = phase3 / "outer_oof_per_model.csv"
    _csv(files["outer_oof_per_model.csv"], outer_rows)
    files["disjoint_prediction_metrics.csv"] = phase3 / "disjoint_prediction_metrics.csv"
    _csv(files["disjoint_prediction_metrics.csv"], predictions)
    files["repeat_fold_gate_results.csv"] = phase3 / "repeat_fold_gate_results.csv"
    _csv(files["repeat_fold_gate_results.csv"], gate_rows)
    files["repeat_metrics.csv"] = phase3 / "repeat_metrics.csv"
    _csv(files["repeat_metrics.csv"], [row for row in gate_rows if row["scope"] == "repetition"])
    files["calibration_spec_selection_frequency.csv"] = (
        phase3 / "calibration_spec_selection_frequency.csv"
    )
    _csv(
        files["calibration_spec_selection_frequency.csv"],
        [
            {
                "spec_id": spec_id,
                "selected_outer_panels": 25 if index == 0 else 0,
                "selection_fraction_all_25_panels": 1.0 if index == 0 else 0.0,
            }
            for index, spec_id in enumerate(report.SPEC_IDS)
        ],
    )
    files["duplicate_cat_paths.csv"] = phase3 / "duplicate_cat_paths.csv"
    _csv(files["duplicate_cat_paths.csv"], [], columns=["panel_id", "left_policy", "right_policy"])
    files["cross_repeat_per_model.csv"] = phase3 / "cross_repeat_per_model.csv"
    _csv(files["cross_repeat_per_model.csv"], cross_model)
    files["cross_repeat_metrics.csv"] = phase3 / "cross_repeat_metrics.csv"
    _csv(files["cross_repeat_metrics.csv"], cross_metrics)

    common_provenance = {
        "study_signature": study_signature,
        "config_sha256": _sha(config_path),
        "split_sha256": _sha(split_path),
        "input_hashes": input_hashes,
        "code_sha256": code_hash,
        "environment_sha256": environment["canonical_sha256"],
    }
    selected = {
        "schema_version": report.V3_SELECTED_SPECS_SCHEMA,
        "study_signature": study_signature,
        "config": {"path": str(config_path.resolve()), "sha256": _sha(config_path)},
        "split_manifest": {"path": str(split_path.resolve()), "sha256": _sha(split_path)},
        "input_hashes": input_hashes,
        "code_sha256": code_hash,
        "environment_sha256": environment["canonical_sha256"],
        "primary_policy": cat_policies[0],
        "sensitivities": cat_policies[1:],
        "sensitivity_promotion_allowed": False,
        "require_all_specs_every_inner_fold": True,
        "survivor_selection_allowed": False,
        "panels": panels,
    }
    choices = {"schema_version": report.V3_RUN_SCHEMA, "panels": panels}
    lock_panels = json.loads(json.dumps(panels))
    for panel in lock_panels:
        panel["status"] = "selection_complete"
        panel.pop("outer_evaluation", None)
    files["selected_calibration_specs.json"] = phase3 / "selected_calibration_specs.json"
    _json(files["selected_calibration_specs.json"], selected)
    files["calibration_model_choices.json"] = phase3 / "calibration_model_choices.json"
    _json(files["calibration_model_choices.json"], choices)

    decision = {
        "schema_version": report.V3_DECISION_SCHEMA,
        **common_provenance,
        "gate_scope_contract": _gate_scope_contract(),
        "status": "pass" if passed else "fail",
        "phase3_pass": passed,
        "phase4_authorized": passed,
        "failed_conditions": (
            []
            if passed
            else [
                "primary_policy_failed_one_or_more_outer_panels",
                "primary_policy_failed_one_or_more_pooled_repetitions",
            ]
        ),
        "inner_selection_fail_closed": {
            "require_all_specs_every_inner_fold": True,
            "expected_spec_fold_combinations_per_panel": 12,
            "survivor_selection_allowed": False,
            "all_panel_selections_locked_before_outer_outcomes": True,
            "all_25_panels_complete": True,
        },
        "coverage": {
            "all_25_panels_evaluated": True,
            "all_52_models_each_repetition": True,
            "by_repetition": {
                str(repeat): {
                    "n_primary_rows": 52,
                    "n_unique_models": 52,
                    "all_52_models_exactly_once": True,
                }
                for repeat in range(5)
            },
            "model_repeat_rows_treated_as_independent": False,
        },
        "calibration_stability": {
            "unique_modal_spec_id": report.SPEC_IDS[0],
            "tied_modal_spec_ids": [],
            "modal_count": 25,
            "modal_fraction_all_25_panels": 1.0,
            "required_fraction": 0.8,
            "passed": True,
        },
        "primary_policy": {
            "policy": cat_policies[0],
            "all_outer_panels_pass": passed,
            "all_repetitions_pass": passed,
            "failed_panel_ids": (
                []
                if passed
                else [
                    f"repeat_{repeat:02d}_outer_{outer_fold:02d}"
                    for repeat in range(5)
                    for outer_fold in range(5)
                ]
            ),
            "failed_repetitions": [] if passed else list(range(5)),
        },
    }
    files["phase3_decision.json"] = phase3 / "phase3_decision.json"
    _json(files["phase3_decision.json"], decision)
    files["v3_decision.json"] = phase3 / "v3_decision.json"
    _json(files["v3_decision.json"], decision)
    files["fold_assignments.json"] = phase3 / "fold_assignments.json"
    _json(
        files["fold_assignments.json"],
        {
            "schema_version": report.V3_RUN_SCHEMA,
            "source_split_manifest": str(split_path.resolve()),
            "source_split_sha256": _sha(split_path),
            "model_to_family": {model: f"family_{int(model[1:]) % 22:02d}" for model in models},
            "repetitions": [
                {
                    "repeat": repeat,
                    "outer_folds": [
                        {
                            "outer_fold": outer_fold,
                            "test_model_ids": [
                                model
                                for model_index, model in enumerate(models)
                                if model_index % 5 == outer_fold
                            ],
                        }
                        for outer_fold in range(5)
                    ],
                }
                for repeat in range(5)
            ],
        },
    )
    files["pre_outer_selection_lock.json"] = phase3 / "pre_outer_selection_lock.json"
    _json(
        files["pre_outer_selection_lock.json"],
        {
            "schema_version": report.V3_RUN_SCHEMA,
            "study_signature": study_signature,
            "panels": lock_panels,
        },
    )

    manifest = {
        "schema_version": report.V3_RUN_SCHEMA,
        "status": "phase3_complete",
        "study_signature": study_signature,
        "script": "scripts/nested_scenario_cat_cv_v3.py",
        "inputs": {
            "config": {"path": str(config_path.resolve()), "sha256": _sha(config_path)},
            "split_manifest": {"path": str(split_path.resolve()), "sha256": _sha(split_path)},
            **{
                name: {"path": str(path.resolve()), "sha256": input_hashes[name]}
                for name, path in input_paths.items()
            },
        },
        "code_provenance": {
            "dependency_inventory": list(code_hashes),
            "files": code_hashes,
            "canonical_sha256": code_hash,
        },
        "environment": environment,
        "numerical_lock": numerical,
        "cross_validation": {
            "repetitions": 5,
            "outer_folds_per_repeat": 5,
            "inner_folds_per_outer": 4,
            "bootstrap_unit": "tutor_family",
            "repeated_rows_treated_as_independent": False,
        },
        "inner_selection_fail_closed": {
            "require_all_specs_every_inner_fold": True,
            "expected_spec_fold_combinations_per_panel": 12,
            "survivor_selection_allowed": False,
            "outer_outcomes_opened_only_after_all_panel_selections_locked": True,
        },
        "calibration_specifications": calibration_specifications,
        "cat_policies": cat_policies,
        "primary_policy_only_can_pass_phase3": True,
        "sensitivity_promotion_allowed": False,
        "runtime": runtime,
        "gate_scope_contract": _gate_scope_contract(),
        "phase3_decision": {
            "status": decision["status"],
            "phase3_pass": decision["phase3_pass"],
            "phase4_authorized": decision["phase4_authorized"],
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": _sha(path)}
            for name, path in files.items()
        },
    }
    _json(phase3 / "manifest.json", manifest)
    return phase3


def _selection_failed_phase3_fixture(root: Path) -> Path:
    phase3 = _phase3_fixture(root, passed=False)
    base_manifest = json.loads((phase3 / "manifest.json").read_text())
    selection_files = set(report.PHASE3_SELECTION_OUTPUTS)
    for name in report.PHASE3_OUTCOME_OUTPUTS:
        (phase3 / name).unlink()

    selected_path = phase3 / "selected_calibration_specs.json"
    selected = json.loads(selected_path.read_text())
    panels = selected["panels"]
    for index, panel in enumerate(panels):
        panel["status"] = (
            "selection_blocked_incomplete_inner_evidence" if index == 0 else "selection_complete"
        )
        panel["selected_spec_id"] = None if index == 0 else report.SPEC_IDS[0]
        panel.pop("outer_evaluation", None)
        panel["inner_selection"] = {
            "complete_inner_evidence_audit": _inner_evidence_audit(passed=index != 0)
        }
    _json(selected_path, selected)
    choices_path = phase3 / "calibration_model_choices.json"
    _json(choices_path, {"schema_version": report.V3_RUN_SCHEMA, "panels": panels})
    lock_path = phase3 / "pre_outer_selection_lock.json"
    _json(
        lock_path,
        {
            "schema_version": report.V3_RUN_SCHEMA,
            "study_signature": selected["study_signature"],
            "panels": panels,
        },
    )

    decision = {
        "schema_version": report.V3_DECISION_SCHEMA,
        "study_signature": selected["study_signature"],
        "config_sha256": selected["config"]["sha256"],
        "split_sha256": selected["split_manifest"]["sha256"],
        "input_hashes": selected["input_hashes"],
        "code_sha256": selected["code_sha256"],
        "environment_sha256": selected["environment_sha256"],
        "gate_scope_contract": _gate_scope_contract(),
        "status": "fail_selection_incomplete",
        "decision_scope": "terminal_selection_only",
        "phase3_pass": False,
        "phase4_authorized": False,
        "terminal_stop": {
            "stage": "pre_outer_selection_gate",
            "reason": "one_or_more_panel_selections_incomplete",
            "preregistered_fail_closed_stop": True,
            "outer_outcomes_opened": False,
            "outer_policy_evaluation_called": False,
        },
        "calibration_selection": {
            "status": "incomplete_fail_closed",
            "expected_panels_denominator": 25,
            "attempted_panels_numerator": 25,
            "complete_panels_numerator": 24,
            "failed_panels_numerator": 1,
            "complete_panel_fraction": 24 / 25,
            "status_counts": {
                "selection_blocked_incomplete_inner_evidence": 1,
                "selection_complete": 24,
            },
            "failed_panel_ids": ["repeat_00_outer_00"],
            "failed_panel_statuses": {
                "repeat_00_outer_00": "selection_blocked_incomplete_inner_evidence"
            },
            "require_all_specs_every_inner_fold": True,
            "survivor_selection_allowed": False,
            "expected_spec_fold_combinations_denominator": 300,
            "valid_spec_fold_combinations_numerator": 299,
            "valid_spec_fold_fraction": 299 / 300,
        },
        "outer_evaluation": {
            "status": "not_run_preregistered_selection_gate",
            "expected_outer_panels_denominator": 25,
            "evaluated_outer_panels_numerator": 0,
            "expected_unique_oof_tutors_per_repetition_denominator": 52,
            "evaluated_unique_oof_tutors_numerator": 0,
            "expected_model_repeat_rows_per_policy_denominator": 260,
            "evaluated_model_repeat_rows_per_policy_numerator": 0,
            "cat_or_baseline_metrics_estimable": False,
        },
        "primary_policy": {
            "policy": selected["primary_policy"],
            "status": "not_evaluated_selection_gate_failed",
            "absolute_gates_applied": False,
        },
        "sensitivities": {
            "policies": selected["sensitivities"],
            "status": "not_evaluated_selection_gate_failed",
            "diagnostic_only": True,
            "can_promote": False,
        },
        "failed_conditions": ["incomplete_calibration_spec_selection"],
    }
    _json(phase3 / "phase3_decision.json", decision)
    _json(phase3 / "v3_decision.json", decision)
    outputs = {
        name: {"path": str((phase3 / name).resolve()), "sha256": _sha(phase3 / name)}
        for name in selection_files
    }
    _json(
        phase3 / "manifest.json",
        {
            "schema_version": report.V3_RUN_SCHEMA,
            "status": "phase3_terminal_selection_failed",
            **{
                key: base_manifest[key]
                for key in (
                    "study_signature",
                    "script",
                    "inputs",
                    "code_provenance",
                    "environment",
                    "numerical_lock",
                    "cross_validation",
                    "inner_selection_fail_closed",
                    "calibration_specifications",
                    "cat_policies",
                    "primary_policy_only_can_pass_phase3",
                    "sensitivity_promotion_allowed",
                    "runtime",
                )
            },
            "gate_scope_contract": _gate_scope_contract(),
            "terminal_stop": decision["terminal_stop"],
            "phase3_decision": {
                "status": decision["status"],
                "phase3_pass": False,
                "phase4_authorized": False,
            },
            "outputs": outputs,
        },
    )
    return phase3


def _refresh_phase3_hashes(phase3: Path, *names: str) -> None:
    manifest_path = phase3 / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in names:
        manifest["outputs"][name]["sha256"] = _sha(phase3 / name)
    _json(manifest_path, manifest)


def _refresh_v4_decision_hash(v4: Path) -> None:
    manifest_path = v4 / "study_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["decision_sha256"] = _sha(v4 / "numerical_followup_decision.json")
    _json(manifest_path, manifest)


def _rewrite_phase3_decision(phase3: Path, decision: dict[str, object]) -> None:
    _json(phase3 / "phase3_decision.json", decision)
    _json(phase3 / "v3_decision.json", decision)
    _refresh_phase3_hashes(phase3, "phase3_decision.json", "v3_decision.json")


def test_blocked_v4_never_reads_phase3_and_emits_numerical_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=False)
    phase3 = tmp_path / "phase3"
    phase3.mkdir()
    (phase3 / "phase3_decision.json").write_text("{not json", encoding="utf-8")
    output = tmp_path / "report"

    original_resolve = Path.resolve

    def reject_phase3_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        if self == phase3:
            raise AssertionError("blocked V4 must not even resolve the Phase-3 path")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", reject_phase3_resolve)

    paths = report.render_terminal_report(
        config_path=config,
        v4_dir=v4,
        phase3_dir=phase3,
        output_dir=output,
    )

    assert len(paths) == 10  # 3 report files, guide, manifest, and 5 figures
    status = json.loads((output / "gate_status.json").read_text())
    assert status["overall_status"] == "blocked_after_v4_numerical_followup"
    assert status["gates"]["v4_numerical_verification"]["decision"] == str(
        (v4 / "numerical_followup_decision.json").resolve()
    )
    assert (
        status["gates"]["v4_numerical_verification"]["fixed_bank_scoring_status"]
        == "not_applicable_refit_gate_failed"
    )
    manifest = json.loads((output / "figures" / "figure_source_manifest.json").read_text())
    assert manifest["phase3_artifacts_opened"] is False
    assert not any("phase3" in name.lower() for name in manifest["source_files"])
    assert len(manifest["figure_files"]) == 5
    assert not any(path.name.startswith("07_") for path in (output / "figures").iterdir())
    assert "CAT NOT RUN" in (output / "figures" / "FIGURE_GUIDE.md").read_text()
    assert (
        "N/A (not run because a refitted-bank gate failed)"
        in (output / "EXECUTION_SUMMARY.md").read_text()
    )


@pytest.mark.parametrize(
    ("passed", "state", "suffix"),
    [
        (False, "blocked_after_phase3_validation", "_INCOMPLETE"),
        (True, "phase3_pass_phase4_pending", ""),
    ],
)
def test_passing_v4_renders_complete_phase3_terminal_state(
    tmp_path: Path, passed: bool, state: str, suffix: str
) -> None:
    case = tmp_path / ("pass" if passed else "fail")
    case.mkdir()
    config = case / "config.json"
    _config(config)
    v4 = _v4_fixture(case, passed=True)
    phase3 = _phase3_fixture(case, passed=passed)
    output = case / "report"

    report.render_terminal_report(
        config_path=config,
        v4_dir=v4,
        phase3_dir=phase3,
        output_dir=output,
    )

    status = json.loads((output / "gate_status.json").read_text())
    assert status["overall_status"] == state
    assert status["release_claims_authorized"] is False
    figures = json.loads((output / "figures" / "figure_source_manifest.json").read_text())
    assert figures["phase3_artifacts_opened"] is True
    assert len(figures["figure_files"]) == 13
    expected = output / "figures" / f"10_oos_ability_recovery{suffix}.png"
    assert expected.is_file()
    if passed:
        assert status["gates"]["phase4_total_uncertainty_and_order"]["status"] == "pending_not_run"
        assert not any("INCOMPLETE" in path.name for path in (output / "figures").glob("*.png"))
    else:
        assert all(
            (output / "figures" / f"{prefix}{suffix}.png").is_file()
            for prefix in (
                "10_oos_ability_recovery",
                "11_oos_pass_rate_calibration",
                "12_cat_vs_random",
                "13_outer_coverage",
            )
        )
    commands = (output / "reproduction_commands.txt").read_text()
    if passed:
        assert PHASE4_COMMAND in commands
        assert "Phase 4 is authorized by the complete passing Phase-3 decision" in commands
        assert "Final fit/export/replay remains unavailable until Phase 4 completes and passes" in commands
    else:
        assert PHASE4_COMMAND not in commands
        assert "Phase 4 is not authorized by this terminal state" in commands
    assert "scripts/finalize_infobench_calibration_cat_v3.py" not in commands


def test_passing_v4_requires_terminal_phase3(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    with pytest.raises(report.TerminalReportError, match="required source artifact is missing"):
        report.render_terminal_report(
            config_path=config,
            v4_dir=v4,
            phase3_dir=tmp_path / "missing_phase3",
            output_dir=tmp_path / "report",
        )


def test_selection_failure_renders_selection_only_and_never_opens_outcomes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _selection_failed_phase3_fixture(tmp_path)
    output = tmp_path / "report"

    paths = report.render_terminal_report(
        config_path=config,
        v4_dir=v4,
        phase3_dir=phase3,
        output_dir=output,
    )

    assert len(paths) == 12  # 5 report/guide files plus 7 numerical/selection figures
    status = json.loads((output / "gate_status.json").read_text())
    assert status["overall_status"] == "blocked_after_phase3_selection"
    phase3_gate = status["gates"]["phase3_repeated_nested_cat"]
    assert phase3_gate["status"] == "failed_selection_incomplete_cat_not_run"
    assert phase3_gate["outcome_artifacts_available"] is False
    figures = json.loads((output / "figures" / "figure_source_manifest.json").read_text())
    assert figures["phase3_artifacts_opened"] is True
    assert figures["phase3_outcome_artifacts_opened"] is False
    assert len(figures["figure_files"]) == 7
    assert figures["denominators"]["phase3_selection_panels_complete"] == 24
    assert figures["denominators"]["phase3_spec_fold_blocks_valid"] == 299
    assert figures["denominators"]["phase3_outer_panels_evaluated"] == 0
    assert not any(
        any(name in source for name in report.PHASE3_OUTCOME_OUTPUTS)
        for source in figures["source_files"]
    )
    assert not (output / "figures" / "08_cat_gate_outcomes.png").exists()
    assert "CAT NOT RUN" in (output / "figures" / "FIGURE_GUIDE.md").read_text()


def test_failed_phase3_accepts_partial_outcomes_and_plots_actual_denominators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)

    outer_path = phase3 / "outer_oof_per_model.csv"
    outer = pd.read_csv(outer_path)
    outer = outer[outer["outer_fold"].isin([0, 1])].copy()
    outer.to_csv(outer_path, index=False)

    evaluated_folds = {0, 1}
    for name in ("selected_calibration_specs.json", "calibration_model_choices.json"):
        path = phase3 / name
        payload = json.loads(path.read_text())
        for panel in payload["panels"]:
            if int(panel["outer_fold"]) not in evaluated_folds:
                panel["status"] = "outer_fit_failed"
                panel["outer_evaluation"] = {
                    "status": "not_run_outer_fit_failed",
                    "primary_rows_complete": False,
                    "n_rows": 0,
                }
        _json(path, payload)

    cross_path = phase3 / "cross_repeat_per_model.csv"
    cross = pd.read_csv(cross_path)
    allowed_models = set(outer["model"].astype(str))
    cross = cross[cross["model"].astype(str).isin(allowed_models)].copy()
    cross.to_csv(cross_path, index=False)
    cross_metrics_path = phase3 / "cross_repeat_metrics.csv"
    cross_metrics = pd.read_csv(cross_metrics_path)
    cross_metrics["n_models"] = len(allowed_models)
    cross_metrics["n_models_with_all_repetitions"] = len(allowed_models)
    cross_metrics["n_families"] = cross["model_family"].nunique()
    cross_metrics["recovery_n"] = len(allowed_models)
    cross_metrics.to_csv(cross_metrics_path, index=False)
    prediction_path = phase3 / "disjoint_prediction_metrics.csv"
    prediction = pd.read_csv(prediction_path)
    unavailable = ~prediction["outer_fold"].isin([0, 1])
    prediction.loc[unavailable, "n_models"] = 0
    prediction.loc[unavailable, "n_cells"] = 0
    prediction.loc[unavailable, "log_loss"] = np.nan
    prediction.to_csv(prediction_path, index=False)

    gates_path = phase3 / "repeat_fold_gate_results.csv"
    gates = pd.read_csv(gates_path)
    for index, row in gates.iterrows():
        scope = str(row["scope"])
        repeat = int(row["repeat"])
        policy = str(row["policy_id"])
        if scope == "outer_panel":
            outer_fold = int(row["outer_fold"])
            observed_models = outer[
                (outer["repeat"] == repeat)
                & (outer["outer_fold"] == outer_fold)
                & (outer["policy_id"] == policy)
            ]["model"].nunique()
            inference_complete = outer_fold in evaluated_folds
        else:
            observed_models = outer[(outer["repeat"] == repeat) & (outer["policy_id"] == policy)][
                "model"
            ].nunique()
            inference_complete = observed_models == 52
        gates.loc[index, "n_inference_rows"] = observed_models
        gates.loc[index, "n_unique_tutor_models"] = observed_models
        gates.loc[index, "inference_scope_complete"] = inference_complete
        if policy == report.PRIMARY_POLICY:
            checks = json.loads(str(row["gate_checks"]))
            checks["inference_scope_complete"] = inference_complete
            all_pass = all(checks.values())
            gates.loc[index, "gate_checks"] = json.dumps(checks)
            gates.loc[index, "all_gates_pass"] = all_pass
            gates.loc[index, "gate_status"] = "pass" if all_pass else "fail"
    gates.to_csv(gates_path, index=False)
    repeat_metrics_path = phase3 / "repeat_metrics.csv"
    gates[gates["scope"] == "repetition"].to_csv(repeat_metrics_path, index=False)

    decision_path = phase3 / "phase3_decision.json"
    decision = json.loads(decision_path.read_text())
    decision["coverage"] = {
        "all_25_panels_evaluated": False,
        "all_52_models_each_repetition": False,
        "by_repetition": {
            str(repeat): {
                "n_primary_rows": 22,
                "n_unique_models": 22,
                "all_52_models_exactly_once": False,
            }
            for repeat in range(5)
        },
        "model_repeat_rows_treated_as_independent": False,
    }
    decision["failed_conditions"] = [
        "incomplete_outer_coverage",
        "primary_policy_failed_one_or_more_outer_panels",
        "primary_policy_failed_one_or_more_pooled_repetitions",
    ]
    decision["primary_policy"] = {
        "policy": decision["primary_policy"]["policy"],
        "all_outer_panels_pass": False,
        "all_repetitions_pass": False,
        "failed_panel_ids": [
            f"repeat_{repeat:02d}_outer_{outer_fold:02d}"
            for repeat in range(5)
            for outer_fold in range(5)
        ],
        "failed_repetitions": list(range(5)),
    }
    _json(decision_path, decision)
    _json(phase3 / "v3_decision.json", decision)

    _refresh_phase3_hashes(
        phase3,
        "outer_oof_per_model.csv",
        "selected_calibration_specs.json",
        "calibration_model_choices.json",
        "cross_repeat_per_model.csv",
        "cross_repeat_metrics.csv",
        "disjoint_prediction_metrics.csv",
        "repeat_fold_gate_results.csv",
        "repeat_metrics.csv",
        "phase3_decision.json",
        "v3_decision.json",
    )

    inputs = report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)
    assert len(inputs.phase3["outer"]) == 330
    assert len(inputs.phase3["cross_model"]) == 66

    titles: dict[str, list[str]] = {}
    annotations: dict[str, list[str]] = {}

    def capture(fig, path: Path) -> None:
        titles[path.name] = [axis.get_title() for axis in fig.axes]
        annotations[path.name] = [text.get_text() for axis in fig.axes for text in axis.texts]
        report.plt.close(fig)

    monkeypatch.setattr(report, "_save", capture)
    report._plot_recovery(inputs, tmp_path / "recovery.png")
    report._plot_pass_rate(inputs, tmp_path / "pass_rate.png")
    report._plot_cat_vs_random(inputs, tmp_path / "cat_random.png")

    assert any("22/52 models with all 5 repetitions" in title for title in titles["recovery.png"])
    assert all("valid rows=110/260" in title for title in titles["pass_rate.png"])
    assert "110/260 paired model-repeats" in titles["cat_random.png"][0]
    assert "CAT 10/25; random 10/25 panels" in titles["cat_random.png"][1]
    assert any("family-bootstrap lower 95%" in value for value in annotations["recovery.png"])
    assert any("family-bootstrap 95% CI" in value for value in annotations["cat_random.png"])


@pytest.mark.parametrize("mutation", ["duplicate", "above_maximum"])
def test_phase3_outer_rows_reject_duplicates_and_counts_above_maximum(
    tmp_path: Path, mutation: str
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    outer_path = phase3 / "outer_oof_per_model.csv"
    outer = pd.read_csv(outer_path)
    if mutation == "duplicate":
        outer = pd.concat([outer.iloc[:-1], outer.iloc[[0]]], ignore_index=True)
        match = "duplicate repeat/policy/model"
    else:
        extra = outer.iloc[[0]].copy()
        extra["model"] = "m_extra"
        outer = pd.concat([outer, extra], ignore_index=True)
        match = "maximum is 780"
    outer.to_csv(outer_path, index=False)
    _refresh_phase3_hashes(phase3, "outer_oof_per_model.csv")

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_phase3_rejects_imbalanced_cat_random_prediction_keys(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    prediction_path = phase3 / "disjoint_prediction_metrics.csv"
    prediction = pd.read_csv(prediction_path)
    target = prediction.index[
        (prediction["candidate_id"] == report.PRIMARY_POLICY) & (prediction["mode"] == "cat")
    ][0]
    prediction.loc[target, "mode"] = "baseline"
    prediction.to_csv(prediction_path, index=False)
    _refresh_phase3_hashes(phase3, "disjoint_prediction_metrics.csv")

    with pytest.raises(report.TerminalReportError, match="prediction composite-key roster"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_parameter_equivalence_axis_exposes_low_failed_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=False)
    inputs = report.load_terminal_inputs(
        config_path=config, v4_dir=v4, phase3_dir=tmp_path / "must_not_resolve"
    )
    inputs.numerical["refit_parameters"][report.SPEC_IDS[0]].loc[0, "b_spearman"] = 0.40
    observed_ylim: tuple[float, float] | None = None

    def capture(fig, _path: Path) -> None:
        nonlocal observed_ylim
        observed_ylim = fig.axes[0].get_ylim()
        report.plt.close(fig)

    monkeypatch.setattr(report, "_save", capture)
    report._plot_parameter_equivalence(inputs, tmp_path / "parameter.png")
    assert observed_ylim is not None
    assert observed_ylim[0] < 0.40


@pytest.mark.parametrize("mutation", ["item_parameters", "missing_checkpoint"])
def test_v4_strict_hash_and_checkpoint_chain_rejects_mutation(
    tmp_path: Path, mutation: str
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=False)
    if mutation == "item_parameters":
        item_path = (
            v4 / "fits" / report.SPEC_IDS[0] / "nodes_0401" / "full" / "cold" / "item_params.csv"
        )
        item_path.write_text(item_path.read_text() + "tampered,1,0,true\n", encoding="utf-8")
        match = "checkpoint output hash mismatch"
    else:
        checkpoint = next((v4 / "checkpoints").glob("fit__*.json"))
        checkpoint.unlink()
        match = "checkpoint inventory differs"

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(
            config_path=config,
            v4_dir=v4,
            phase3_dir=tmp_path / "must_not_resolve",
        )


def test_phase3_manifest_rejects_noncanonical_output_path(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    manifest_path = phase3 / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["outer_oof_per_model.csv"]["path"] = str((tmp_path / "wrong.csv").resolve())
    _json(manifest_path, manifest)

    with pytest.raises(report.TerminalReportError, match="path is not canonical"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_phase3_accepts_driver_selected_spec_nested_config_split_shape(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    selected = json.loads((phase3 / "selected_calibration_specs.json").read_text())

    assert "config_sha256" not in selected
    assert "split_sha256" not in selected
    assert selected["config"] == {
        "path": str(config.resolve()),
        "sha256": _sha(config),
    }
    split = Path(json.loads(config.read_text())["cross_validation"]["split_manifest"])
    assert selected["split_manifest"] == {
        "path": str(split.resolve()),
        "sha256": _sha(split),
    }

    inputs = report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)
    assert inputs.terminal_state == "blocked_after_phase3_validation"


@pytest.mark.parametrize(
    ("artifact", "field"),
    [
        ("decision", "config_sha256"),
        ("decision", "split_sha256"),
        ("decision", "input_hashes"),
        ("decision", "code_sha256"),
        ("decision", "environment_sha256"),
        ("selected", "input_hashes"),
        ("selected", "code_sha256"),
        ("selected", "environment_sha256"),
        ("selected_nested", "config"),
        ("selected_nested", "split_manifest"),
    ],
)
def test_phase3_live_provenance_shape_remains_fail_closed(
    tmp_path: Path, artifact: str, field: str
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)

    if artifact == "decision":
        decision = json.loads((phase3 / "phase3_decision.json").read_text())
        decision[field] = {} if field == "input_hashes" else "0" * 64
        _rewrite_phase3_decision(phase3, decision)
        match = rf"Phase-3 decision {field} differs"
    else:
        selected_path = phase3 / "selected_calibration_specs.json"
        selected = json.loads(selected_path.read_text())
        if artifact == "selected_nested":
            selected[field]["sha256"] = "0" * 64
            label = "config" if field == "config" else "split"
            match = rf"selected-spec {label} hash mismatch"
        else:
            selected[field] = {} if field == "input_hashes" else "0" * 64
            match = rf"selected-spec handoff {field} differs"
        _json(selected_path, selected)
        _refresh_phase3_hashes(phase3, "selected_calibration_specs.json")

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


@pytest.mark.parametrize(
    "component",
    [
        "specification",
        "policy",
        "runtime",
        "numerical",
        "cross_validation",
        "gate_scope",
        "selection_policy",
    ],
)
def test_phase3_rejects_mutated_study_contract(tmp_path: Path, component: str) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    manifest_path = phase3 / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if component == "specification":
        manifest["calibration_specifications"][0]["simplicity_rank"] = 9
    elif component == "policy":
        manifest["cat_policies"][0]["minimum_scenarios"] = 14
    elif component == "runtime":
        manifest["runtime"]["seed"] += 1
    elif component == "numerical":
        manifest["numerical_lock"]["fit_grid"] = 400
    elif component == "cross_validation":
        manifest["cross_validation"]["bootstrap_unit"] = "tutor_model"
    elif component == "gate_scope":
        manifest["gate_scope_contract"]["cross_repeat"]["raw_model_repeat_rows"] = 259
    else:
        manifest["primary_policy_only_can_pass_phase3"] = False
    _json(manifest_path, manifest)

    with pytest.raises(report.TerminalReportError, match="Phase-3 .* differs"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_phase3_rejects_auxiliary_schema_mutation(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    selected_path = phase3 / "selected_calibration_specs.json"
    selected = json.loads(selected_path.read_text())
    selected["schema_version"] = "wrong-schema"
    _json(selected_path, selected)
    _refresh_phase3_hashes(phase3, "selected_calibration_specs.json")

    with pytest.raises(report.TerminalReportError, match="selected-spec handoff schema"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_phase3_rejects_code_inventory_mutation(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    manifest_path = phase3 / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["code_provenance"]["dependency_inventory"] = manifest["code_provenance"][
        "dependency_inventory"
    ][:-1]
    _json(manifest_path, manifest)

    with pytest.raises(report.TerminalReportError, match="code inventory differs"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_passing_phase3_rejects_missing_outer_row_even_with_rehashed_manifest(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=True)
    outer_path = phase3 / "outer_oof_per_model.csv"
    pd.read_csv(outer_path).iloc[:-1].to_csv(outer_path, index=False)
    _refresh_phase3_hashes(phase3, "outer_oof_per_model.csv")

    with pytest.raises(report.TerminalReportError, match="partially represented|lacks 780"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_passing_phase3_rejects_nonpositive_prediction_support(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=True)
    prediction_path = phase3 / "disjoint_prediction_metrics.csv"
    prediction = pd.read_csv(prediction_path)
    prediction.loc[0, "n_cells"] = 0
    prediction.loc[0, "log_loss"] = 0
    prediction.to_csv(prediction_path, index=False)
    _refresh_phase3_hashes(phase3, "disjoint_prediction_metrics.csv")

    with pytest.raises(report.TerminalReportError, match="invalid prediction panels"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("refit_aggregate", "aggregate refit-bank gate"),
        ("fixed_aggregate", "fixed-bank flags"),
        ("overall_pass", "aggregate pass flag"),
        ("blocked_lock", "blocked V4 manifest claims a lock hash"),
    ],
)
def test_v4_rejects_inconsistent_aggregate_flags_and_blocked_lock(
    tmp_path: Path, mutation: str, match: str
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    passed = mutation == "overall_pass"
    v4 = _v4_fixture(tmp_path, passed=passed)
    if mutation == "blocked_lock":
        manifest_path = v4 / "study_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["lock_sha256"] = "0" * 64
        _json(manifest_path, manifest)
    else:
        decision_path = v4 / "numerical_followup_decision.json"
        decision = json.loads(decision_path.read_text())
        field = {
            "refit_aggregate": "all_three_refit_bank_gates_passed",
            "fixed_aggregate": "all_three_fixed_bank_scoring_and_tail_gates_passed",
            "overall_pass": "all_three_fixed_bank_scoring_and_tail_gates_passed",
        }[mutation]
        decision[field] = not bool(decision[field])
        _json(decision_path, decision)
        _refresh_v4_decision_hash(v4)

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(
            config_path=config,
            v4_dir=v4,
            phase3_dir=tmp_path / "must_not_resolve",
        )


@pytest.mark.parametrize("stage", ["refit", "fixed_bank"])
def test_v4_rejects_component_gate_tamper_with_rehashed_provenance(
    tmp_path: Path, stage: str
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    passed = stage == "fixed_bank"
    v4 = _v4_fixture(tmp_path, passed=passed)
    spec_id = report.SPEC_IDS[0]
    if stage == "refit":
        record_path = v4 / "fit_comparisons" / "401_vs_801" / spec_id / "fit_pair_gate.json"
        checkpoint_path = v4 / "checkpoints" / f"fit_comparison__{spec_id}__401_vs_801.json"
        evidence_key = f"refit/{spec_id}"
        record = json.loads(record_path.read_text())
        record["all_scope_parameter_gates_passed"] = True
        match = "refit component gates disagree"
    else:
        record_path = v4 / "fixed_bank" / spec_id / "fixed_bank_gate.json"
        checkpoint_path = v4 / "checkpoints" / f"fixed_bank__{spec_id}.json"
        evidence_key = f"fixed_bank/{spec_id}"
        record = json.loads(record_path.read_text())
        record["numerical_scoring_gate"]["comparisons"]["bound8_801_vs_bound8_1601"]["passed"] = (
            False
        )
        match = "fixed-bank component gates disagree"
    _json(record_path, record)
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["output_sha256"][str(record_path.resolve())] = _sha(record_path)
    _json(checkpoint_path, checkpoint)
    decision_path = v4 / "numerical_followup_decision.json"
    decision = json.loads(decision_path.read_text())
    decision["evidence_sha256"][evidence_key] = _sha(record_path)
    _json(decision_path, decision)
    _refresh_v4_decision_hash(v4)

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(
            config_path=config,
            v4_dir=v4,
            phase3_dir=tmp_path / "must_not_resolve",
        )


@pytest.mark.parametrize("mutation", ["coverage", "panel_status"])
def test_phase3_rejects_derived_coverage_or_panel_status_tamper(
    tmp_path: Path, mutation: str
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    if mutation == "coverage":
        decision = json.loads((phase3 / "phase3_decision.json").read_text())
        decision["coverage"]["by_repetition"]["0"]["n_primary_rows"] = 51
        _rewrite_phase3_decision(phase3, decision)
        match = "decision coverage differs"
    else:
        for name in ("selected_calibration_specs.json", "calibration_model_choices.json"):
            path = phase3 / name
            payload = json.loads(path.read_text())
            payload["panels"][0]["status"] = "outer_fit_failed"
            payload["panels"][0]["outer_evaluation"] = {
                "status": "not_run_outer_fit_failed",
                "primary_rows_complete": False,
            }
            _json(path, payload)
        _refresh_phase3_hashes(
            phase3, "selected_calibration_specs.json", "calibration_model_choices.json"
        )
        match = "evaluated panel status differs"

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_phase3_rejects_choice_outer_evaluation_tamper(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    choices_path = phase3 / "calibration_model_choices.json"
    choices = json.loads(choices_path.read_text())
    choices["panels"][0]["outer_evaluation"]["n_rows"] = -1
    _json(choices_path, choices)
    _refresh_phase3_hashes(phase3, "calibration_model_choices.json")

    with pytest.raises(report.TerminalReportError, match="choice outer-evaluation record differs"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


@pytest.mark.parametrize(
    "mutation", ["valid_total", "fraction", "status", "terminal_stop", "policy_disposition"]
)
def test_selection_only_reconciles_panel_audits_and_decision_totals(
    tmp_path: Path, mutation: str
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _selection_failed_phase3_fixture(tmp_path)
    if mutation in {"valid_total", "fraction", "terminal_stop", "policy_disposition"}:
        decision = json.loads((phase3 / "phase3_decision.json").read_text())
        if mutation == "valid_total":
            decision["calibration_selection"]["valid_spec_fold_combinations_numerator"] = 300
            match = "panel denominators differ"
        elif mutation == "fraction":
            decision["calibration_selection"]["complete_panel_fraction"] = 1.0
            match = "panel denominators differ"
        elif mutation == "terminal_stop":
            decision["terminal_stop"]["outer_outcomes_opened"] = True
            match = "terminal-stop contract differs"
        else:
            decision["primary_policy"]["status"] = "evaluated"
            match = "policy disposition differs"
        _rewrite_phase3_decision(phase3, decision)
        if mutation == "terminal_stop":
            manifest_path = phase3 / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["terminal_stop"] = decision["terminal_stop"]
            _json(manifest_path, manifest)
    else:
        selected_path = phase3 / "selected_calibration_specs.json"
        selected = json.loads(selected_path.read_text())
        selected["panels"][0]["status"] = "selection_failed"
        _json(selected_path, selected)
        _refresh_phase3_hashes(phase3, "selected_calibration_specs.json")
        match = "panel status differs across artifacts"

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_phase3_rejects_manifest_decision_summary_tamper(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    manifest_path = phase3 / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["phase3_decision"]["phase4_authorized"] = True
    _json(manifest_path, manifest)

    with pytest.raises(report.TerminalReportError, match="manifest decision summary differs"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


@pytest.mark.parametrize("mutation", ["frequency", "stability"])
def test_phase3_rejects_selection_frequency_or_stability_tamper(
    tmp_path: Path, mutation: str
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    if mutation == "frequency":
        frequency_path = phase3 / "calibration_spec_selection_frequency.csv"
        frequency = pd.read_csv(frequency_path)
        frequency.loc[0, "selection_fraction_all_25_panels"] = 0.96
        frequency.to_csv(frequency_path, index=False)
        _refresh_phase3_hashes(phase3, "calibration_spec_selection_frequency.csv")
        match = "selection frequency differs"
    else:
        decision = json.loads((phase3 / "phase3_decision.json").read_text())
        decision["calibration_stability"]["modal_fraction_all_25_panels"] = 0.96
        _rewrite_phase3_decision(phase3, decision)
        match = "calibration stability differs"

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_phase3_rejects_failed_condition_reordering(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    decision = json.loads((phase3 / "phase3_decision.json").read_text())
    decision["failed_conditions"] = list(reversed(decision["failed_conditions"]))
    _rewrite_phase3_decision(phase3, decision)

    with pytest.raises(report.TerminalReportError, match="failed conditions differ"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


@pytest.mark.parametrize(
    "mutation",
    [
        "malformed",
        "duplicate_key",
        "wrong_key",
        "nonboolean",
        "nonboolean_inference_flag",
    ],
)
def test_phase3_strictly_parses_primary_gate_checks(tmp_path: Path, mutation: str) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    gates_path = phase3 / "repeat_fold_gate_results.csv"
    gates = pd.read_csv(gates_path)
    target = gates.index[
        (gates["scope"] == "outer_panel") & (gates["policy_id"] == report.PRIMARY_POLICY)
    ][0]
    if mutation == "nonboolean_inference_flag":
        gates["inference_scope_complete"] = gates["inference_scope_complete"].astype(object)
        gates.loc[target, "inference_scope_complete"] = "yes"
        match = "gate inference flag .* is not boolean"
    elif mutation == "malformed":
        gates.loc[target, "gate_checks"] = "{not-json"
        match = "gate_checks is malformed JSON"
    elif mutation == "duplicate_key":
        raw_checks = str(gates.loc[target, "gate_checks"])
        gates.loc[target, "gate_checks"] = raw_checks[:-1] + ', "recovery_slope": true}'
        match = "gate_checks contains duplicate JSON keys"
    else:
        checks = json.loads(str(gates.loc[target, "gate_checks"]))
        if mutation == "wrong_key":
            checks["unexpected_gate"] = checks.pop("recovery_slope")
            match = "gate_checks key roster differs"
        else:
            checks["recovery_slope"] = 1
            match = "gate_checks values are not boolean"
        gates.loc[target, "gate_checks"] = json.dumps(checks)
    gates.to_csv(gates_path, index=False)
    _refresh_phase3_hashes(phase3, "repeat_fold_gate_results.csv")

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


@pytest.mark.parametrize("mutation", ["interval", "bootstrap_metadata"])
def test_phase3_rejects_cross_repeat_interval_or_bootstrap_metadata_tamper(
    tmp_path: Path, mutation: str
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    metrics_path = phase3 / "cross_repeat_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    if mutation == "interval":
        metrics.loc[0, "scenario_reduction_lower_95_ci"] = 0.70
        match = "scenario-reduction family-bootstrap interval differs"
    else:
        metrics.loc[0, "bootstrap_unit"] = "model_repeat_row"
        match = "cross-repeat denominators differ"
    metrics.to_csv(metrics_path, index=False)
    _refresh_phase3_hashes(phase3, "cross_repeat_metrics.csv")

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


def test_policy_sensitivity_plot_uses_family_bootstrap_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    inputs = report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)
    cross = inputs.phase3["cross_metrics"]
    expected_ranges: dict[int, tuple[float, float]] = {}
    expected_counts = [52, 45, 38]
    for index, policy_id in enumerate(report.POLICY_LABELS):
        mask = cross["policy_id"].astype(str).eq(policy_id)
        mean = 0.51 + 0.02 * index
        lower = 0.40 + 0.02 * index
        upper = 0.63 + 0.02 * index
        cross.loc[mask, "scenario_reduction_vs_random"] = mean
        cross.loc[mask, "scenario_reduction_lower_95_ci"] = lower
        cross.loc[mask, "scenario_reduction_upper_95_ci"] = upper
        cross.loc[mask, "n_models_with_all_repetitions"] = expected_counts[index]
        expected_ranges[index] = (lower, upper)

    captured: dict[str, object] = {}

    def capture(fig, _path: Path) -> None:
        captured["figure"] = fig

    monkeypatch.setattr(report, "_save", capture)
    report._plot_policy_sensitivities(inputs, tmp_path / "sensitivity.png")
    figure = captured["figure"]
    axis = figure.axes[0]
    assert "family-bootstrap 95% CI" in axis.get_title()
    assert f"complete models {expected_counts}/52" in axis.get_title()
    observed_ranges: dict[int, tuple[float, float]] = {}
    for collection in axis.collections:
        for segment in getattr(collection, "get_segments", lambda: [])():
            if len(segment) == 2 and np.isclose(segment[0][0], segment[1][0]):
                x_value = int(round(float(segment[0][0])))
                if x_value in expected_ranges:
                    observed_ranges[x_value] = tuple(sorted(map(float, segment[:, 1])))
    assert set(observed_ranges) == set(expected_ranges)
    for index in expected_ranges:
        np.testing.assert_allclose(observed_ranges[index], expected_ranges[index])
    report.plt.close(figure)


def test_passing_phase3_requires_finite_cross_repeat_bootstrap_intervals(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=True)
    metrics_path = phase3 / "cross_repeat_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics["recovery_correlation"] = np.nan
    metrics["recovery_correlation_lower_95_ci"] = np.nan
    metrics.to_csv(metrics_path, index=False)
    _refresh_phase3_hashes(phase3, "cross_repeat_metrics.csv")

    with pytest.raises(report.TerminalReportError, match="lacks family-bootstrap"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


@pytest.mark.parametrize("mutation", ["combination", "failure_name"])
def test_phase3_rejects_inner_combination_audit_tamper(tmp_path: Path, mutation: str) -> None:
    config = tmp_path / "config.json"
    _config(config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)
    selected_path = phase3 / "selected_calibration_specs.json"
    selected = json.loads(selected_path.read_text())
    audit = selected["panels"][0]["inner_selection"]["complete_inner_evidence_audit"]
    if mutation == "combination":
        combination = audit["combination_audits"][0]
        combination["checks"]["status_ok"] = False
        combination["passed"] = False
    else:
        audit["failed_checks"] = ["made_up_failure"]
        audit["passed"] = False
    _json(selected_path, selected)
    _refresh_phase3_hashes(phase3, "selected_calibration_specs.json")

    with pytest.raises(report.TerminalReportError, match="combination-audit totals differ"):
        report.load_terminal_inputs(config_path=config, v4_dir=v4, phase3_dir=phase3)


@pytest.mark.parametrize("component", ["cross_validation", "outputs", "selection_gates"])
def test_phase3_rejects_mutated_frozen_config_contract(tmp_path: Path, component: str) -> None:
    config_path = tmp_path / "config.json"
    _config(config_path)
    config = json.loads(config_path.read_text())
    if component == "cross_validation":
        config[component]["repetitions"] = 4
        match = "cross-validation contract differs"
    elif component == "outputs":
        config[component]["phase4"] = "runs/calibration/wrong_phase4"
        match = "output directories differ"
    else:
        config[component]["minimum_scenario_reduction_vs_random"] = 0.49
        match = "gate-scope contract differs"
    _json(config_path, config)
    v4 = _v4_fixture(tmp_path, passed=True)
    phase3 = _phase3_fixture(tmp_path, passed=False)

    with pytest.raises(report.TerminalReportError, match=match):
        report.load_terminal_inputs(config_path=config_path, v4_dir=v4, phase3_dir=phase3)


def test_renderer_rejects_stale_output_inventory_before_reading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "report"
    output.mkdir()
    (output / "stale.txt").write_text("old report\n", encoding="utf-8")
    with pytest.raises(report.TerminalReportError, match="absent or empty"):
        report.render_terminal_report(
            config_path=tmp_path / "missing-config.json",
            v4_dir=tmp_path / "missing-v4",
            phase3_dir=tmp_path / "missing-phase3",
            output_dir=output,
        )
