from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import calibrate_mirt as cm  # noqa: E402
from scripts import nested_scenario_cat_cv_v2 as v2  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_numerical_lock(
    tmp_path: Path, *, first_pass: bool, second_pass: bool
) -> tuple[Path, Path, dict, list[v2.CalibrationSpec]]:
    """Build a complete hash-linked v3 lock without fitting or CAT."""

    base_path = ROOT / "configs" / "infobench_calibration_cat_v2.json"
    base = _load(base_path)
    specs = v2.load_calibration_specs(base)
    overlay_path = tmp_path / "followup_config.json"
    output = tmp_path / "followup"
    lock_path = output / "numerical_followup_lock.json"
    overlay = _load(ROOT / "configs" / "infobench_v2_numerical_followup_v3.json")
    overlay["output_dir"] = str(output)
    overlay["lock_path"] = str(lock_path)
    _write_json(overlay_path, overlay)
    effective = v2._materialize_followup_config(
        overlay, followup_path=overlay_path, lock_path=lock_path
    )

    evidence: dict[str, str] = {}
    for comparison, passed in (
        ("81_vs_101", first_pass),
        ("101_vs_121", second_pass),
    ):
        for spec in specs:
            path = (
                output
                / "fit_comparisons"
                / comparison
                / spec.spec_id
                / "fit_pair_gate.json"
            )
            _write_json(
                path,
                {
                    "schema_version": v2.FOLLOWUP_RUN_SCHEMA,
                    "spec_id": spec.spec_id,
                    "canonical_specification": spec.canonical,
                    "comparison": comparison,
                    "selection_performed": False,
                    "cat_results_inspected": False,
                    "passed": passed,
                },
            )
            evidence[f"fit/{comparison}/{spec.spec_id}"] = v2._sha256(path)

    fit_grid = 81 if first_pass and second_pass else 101
    for spec in specs:
        path = output / "eap_bound" / spec.spec_id / "eap_bound_gate.json"
        _write_json(
            path,
            {
                "schema_version": v2.FOLLOWUP_RUN_SCHEMA,
                "spec_id": spec.spec_id,
                "fit_grid": fit_grid,
                "all_theta_comparisons_passed": True,
                "tail_gates": {
                    "bound8": {"passed": True},
                    "bound10": {"passed": True},
                },
                "selection_performed": False,
                "cat_results_inspected": False,
            },
        )
        evidence[f"eap/{spec.spec_id}"] = v2._sha256(path)

    profile = {
        "fit_grid": fit_grid,
        "quadrature_method": "normal_trapezoid",
        "linear_bound": 8.0,
        "eap_grid": 401,
        "tail_region": 7.5,
    }
    parent = effective["parent_study"]
    history = overlay["historical_evidence"]
    v1_history = history["v1_aborted"]
    v2_history = history["v2_terminal"]
    decision_path = output / "numerical_followup_decision.json"
    decision = {
        "schema_version": v2.FOLLOWUP_RUN_SCHEMA,
        "status": "complete_pass",
        "passed": True,
        "parent_failure_preserved": True,
        "v1_abort_preserved": True,
        "v2_terminal_preserved": True,
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "historical_v1_sha256": {
            "aborted_marker_sha256": v1_history["aborted_marker_sha256"],
            "output_tree_sha256": v1_history["output_tree_sha256"],
        },
        "historical_v2_sha256": {
            "manifest_sha256": v2_history["manifest_sha256"],
            "decision_sha256": v2_history["decision_sha256"],
            "launched_runner_sha256": v2_history["launched_runner_sha256"],
            "output_tree_sha256": v2_history["output_tree_sha256"],
        },
        "parent_manifest_sha256": parent["study_manifest_sha256"],
        "parent_decision_sha256": parent["decision_sha256"],
        "effective_global_profile": profile,
        "evidence_sha256": evidence,
        "selection_performed": False,
        "cat_results_inspected": False,
        "phase3_ready": False,
        "passed_spec_ids": [spec.spec_id for spec in specs],
    }
    _write_json(decision_path, decision)

    environment = v2.numerical_followup.v2.v1.parent_runner._environment()
    signature_payload = {
        "schema_version": v2.FOLLOWUP_RUN_SCHEMA,
        "config_sha256": v2._sha256(overlay_path),
        "effective_config_sha256": v2._canonical_hash(effective),
        "frozen_contract_sha256": v2.FOLLOWUP_FROZEN_CONTRACT_SHA256,
        "historical_v1_sha256": {
            "aborted_marker_sha256": v1_history["aborted_marker_sha256"],
            "output_tree_sha256": v1_history["output_tree_sha256"],
        },
        "historical_v2_sha256": v2.numerical_followup._verify_v2_history(overlay),
        "parent_manifest_sha256": parent["study_manifest_sha256"],
        "parent_decision_sha256": parent["decision_sha256"],
        "parent_checkpoint_sha256": parent["grid81_checkpoint_sha256"],
        "input_sha256": {
            v2.numerical_followup._display(v2._resolve_repo_path(effective["frozen_inputs"][name])):
            effective["frozen_inputs"][f"{name}_sha256"]
            for name in ("response_matrix", "rubrics", "scenarios", "split_manifest")
        },
        "code_sha256": {
            v2.numerical_followup._display(path): v2._sha256(path)
            for path in v2.numerical_followup.CODE_DEPENDENCIES
        },
        "environment_sha256": environment["canonical_sha256"],
        "exact_specifications": [
            {
                "spec_id": spec.spec_id,
                "family": spec.family,
                "ridge": spec.ridge,
                "log_a_shrinkage": spec.log_a_shrinkage,
                "canonical_specification": spec.canonical,
            }
            for spec in specs
        ],
        "fit_grids": [81, 101, 121],
        "comparison_schedule": [[81, 101], [101, 121]],
        "eap_profiles": v2._json_ready(
            v2.numerical_followup.v2.v1.EAP_PROFILES
        ),
        "v1_fit_or_checkpoint_artifacts_reused": 0,
        "v2_fit_or_checkpoint_artifacts_reused": 0,
        "required_total_new_fits": 72,
    }
    signature = {
        **signature_payload,
        "environment": environment,
        "canonical_sha256": v2._canonical_hash(signature_payload),
    }
    lock = {
        "schema_version": v2.FOLLOWUP_LOCK_SCHEMA,
        "status": "complete_pass",
        "passed_spec_ids": [spec.spec_id for spec in specs],
        "common_fit_grid": fit_grid,
        "quadrature_method": "normal_trapezoid",
        "linear_bound": 8.0,
        "eap_grid": 401,
        "tail_region": 7.5,
        "fit_comparison_all_six_passed": {
            "81_vs_101": first_pass,
            "101_vs_121": second_pass,
        },
        "effective_global_profile": profile,
        "all_six_exact_specifications_share_profile": True,
        "config_sha256": v2._sha256(overlay_path),
        "study_signature_sha256": signature["canonical_sha256"],
        "parent_manifest_sha256": parent["study_manifest_sha256"],
        "parent_decision_sha256": parent["decision_sha256"],
        "historical_v1_aborted_marker_sha256": v1_history[
            "aborted_marker_sha256"
        ],
        "historical_v1_output_tree_sha256": v1_history["output_tree_sha256"],
        "historical_v2_manifest_sha256": v2_history["manifest_sha256"],
        "historical_v2_decision_sha256": v2_history["decision_sha256"],
        "historical_v2_output_tree_sha256": v2_history["output_tree_sha256"],
        "historical_fit_or_checkpoint_artifacts_reused": 0,
        "followup_decision_sha256": v2._sha256(decision_path),
        "evidence_sha256": evidence,
        "selection_performed": False,
        "cat_results_inspected": False,
    }
    _write_json(lock_path, lock)
    lock_path.with_suffix(".sha256").write_text(
        f"{v2._sha256(lock_path)}  {lock_path.name}\n", encoding="utf-8"
    )
    _write_json(
        output / "study_manifest.json",
        {
            "schema_version": v2.FOLLOWUP_RUN_SCHEMA,
            "status": "complete_pass",
            "study_signature_sha256": signature["canonical_sha256"],
            "study_signature": signature,
            "decision_sha256": v2._sha256(decision_path),
            "lock_sha256": v2._sha256(lock_path),
            "v1_abort_preserved": True,
            "v2_terminal_preserved": True,
            "historical_fit_or_checkpoint_artifacts_reused": 0,
        },
    )
    return overlay_path, lock_path, base, specs


def test_frozen_exact_specs_and_policies_match_decision_record() -> None:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v2.json")
    specs = v2.load_calibration_specs(config)
    assert [spec.family for spec in specs] == [
        cm.ONE_PL,
        cm.LOG_SHRINKAGE_2PL,
        cm.LOG_SHRINKAGE_2PL,
        cm.FREE_2PL,
        cm.FREE_2PL,
        cm.FREE_2PL,
    ]
    assert [spec.log_a_shrinkage for spec in specs] == [None, 16.0, 4.0, None, None, None]
    assert [spec.ridge for spec in specs] == [None, None, None, 0.1, 0.01, 0.001]
    assert [spec.simplicity_rank for spec in specs] == list(range(6))
    assert all(
        spec.canonical["cache_key"]
        == config["calibration_specifications"][index]["canonical_cache_key"]
        for index, spec in enumerate(specs)
    )

    policies = v2.load_policies(config)
    observed_policies = [
        (item.minimum_scenarios, item.conditional_se_target, item.selector)
        for item in policies
    ]
    assert observed_policies == [
        (15, 0.2, "trace"),
        (12, 0.2, "trace"),
        (15, 0.2, "dopt"),
    ]
    assert policies[0].role == "primary"
    assert all(item.role == "sensitivity" for item in policies[1:])


@pytest.mark.parametrize(
    "first_pass,second_pass,expected_grid",
    [(True, True, 81), (False, True, 101)],
)
def test_complete_followup_lock_selects_only_the_frozen_common_grid_cases(
    tmp_path: Path,
    first_pass: bool,
    second_pass: bool,
    expected_grid: int,
) -> None:
    followup, lock, config, specs = _synthetic_numerical_lock(
        tmp_path, first_pass=first_pass, second_pass=second_pass
    )
    profile = v2._validate_numerical_lock(
        config,
        specs,
        base_config_path=ROOT / "configs" / "infobench_calibration_cat_v2.json",
        lock_path=lock,
        followup_config_path=followup,
    )
    assert profile["status"] == "passed"
    assert profile["fit_grid"] == expected_grid
    assert profile["effective_global_profile"]["fit_grid"] == expected_grid
    assert profile["verification_sha256"] == v2._sha256(lock)


def test_followup_lock_rejects_tampered_evidence_and_overlay(
    tmp_path: Path,
) -> None:
    followup, lock, config, specs = _synthetic_numerical_lock(
        tmp_path, first_pass=True, second_pass=True
    )
    evidence = (
        lock.parent
        / "fit_comparisons"
        / "81_vs_101"
        / specs[0].spec_id
        / "fit_pair_gate.json"
    )
    payload = _load(evidence)
    payload["comparison"] = "mislabeled"
    _write_json(evidence, payload)
    with pytest.raises(v2.V2Phase3Error, match="evidence hash mismatch"):
        v2._validate_numerical_lock(
            config,
            specs,
            base_config_path=ROOT / "configs" / "infobench_calibration_cat_v2.json",
            lock_path=lock,
            followup_config_path=followup,
        )

    followup2, lock2, config2, specs2 = _synthetic_numerical_lock(
        tmp_path / "second", first_pass=True, second_pass=True
    )
    overlay = _load(followup2)
    overlay["orchestration_source"]["required_corrections"][
        "dynamic_common_cell_labels"
    ] = False
    _write_json(followup2, overlay)
    with pytest.raises(v2.V2Phase3Error, match="config contract"):
        v2._validate_numerical_lock(
            config2,
            specs2,
            base_config_path=ROOT / "configs" / "infobench_calibration_cat_v2.json",
            lock_path=lock2,
            followup_config_path=followup2,
        )


def test_v2_phase_parsers_have_no_destructive_fresh_option() -> None:
    from scripts import nested_cat_total_uncertainty_v2 as phase4

    for parser in (v2.build_argparser(), phase4.build_argparser()):
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        assert "--resume" in options
        assert "--fresh" not in options


def test_repeated_split_audit_covers_every_model_and_rejects_family_leakage() -> None:
    split = _load(ROOT / "configs" / "infobench_v2_splits.manifest.json")
    audit = v2.audit_repeated_split_manifest(split)
    assert len(audit["repetitions"]) == 5
    assert len({row["partition_sha256"] for row in audit["repetitions"]}) == 5
    assert all(len(row["outer_folds"]) == 5 for row in audit["repetitions"])
    assert all(
        len({model for fold in row["outer_folds"] for model in fold["test_model_ids"]})
        == 52
        for row in audit["repetitions"]
    )

    contaminated = copy.deepcopy(split)
    outer = contaminated["repetitions"][0]["outer_folds"][0]
    family = outer["test_family_ids"][0]
    member = contaminated["model_families"][family][0]
    replacement = outer["train_model_ids"][0]
    outer["test_model_ids"].remove(member)
    outer["test_model_ids"].append(replacement)
    outer["train_model_ids"].remove(replacement)
    outer["train_model_ids"].append(member)
    with pytest.raises(v2.V2Phase3Error, match="family|partition"):
        v2.audit_repeated_split_manifest(contaminated)


def _selection_rows(
    specs: list[v2.CalibrationSpec], *, losses: list[float]
) -> list[dict]:
    rows = []
    for spec, loss in zip(specs, losses, strict=True):
        rows.append(
            {
                "spec_id": spec.spec_id,
                "calibration_specification": spec.canonical,
                "mean_log_loss": loss,
                "family_cluster_se": 0.02,
                "eligible": True,
            }
        )
    return rows


def test_inner_one_se_selection_prefers_frozen_simplicity_and_has_no_outer_input() -> None:
    specs = v2.expected_calibration_specs()
    # Free 2PL is the raw minimum, but all candidates are inside its one-SE band.
    evidence = _selection_rows(specs, losses=[0.410, 0.405, 0.404, 0.403, 0.402, 0.400])
    selected, annotated, detail = v2.select_calibration_spec_one_se(evidence, specs)
    assert selected.family == cm.ONE_PL
    assert detail["empirical_best_cache_key"] == specs[-1].canonical["cache_key"]
    assert sum(bool(row["selected"]) for row in annotated) == 1
    assert all(row["selection_used_outer_outcomes"] is False for row in annotated)


def test_calibration_evidence_uses_family_jackknife_and_fails_incomplete_support() -> None:
    specs = v2.expected_calibration_specs()
    families = {"m0": "f0", "m1": "f0", "m2": "f1", "m3": "f2"}
    rows: list[dict] = []
    for spec in specs:
        for index, model in enumerate(families):
            rows.append(
                {
                    "spec_id": spec.spec_id,
                    "model": model,
                    "status": "ok",
                    "model_log_loss": 0.2 + 0.01 * index,
                    "log_loss_sum": 2.0 + index,
                    "n_cells": 10,
                    "fit_eligible": True,
                    "common_support_verified": True,
                    "fit_cache_key": f"fit-{spec.spec_id}",
                }
            )
    aggregate = v2.aggregate_calibration_evidence(rows, specs, families)
    assert all(row["eligible"] for row in aggregate)
    assert all(row["n_families_scored"] == 3 for row in aggregate)
    assert all(float(row["family_cluster_se"]) > 0 for row in aggregate)

    rows[-1]["common_support_verified"] = False
    aggregate = v2.aggregate_calibration_evidence(rows, specs, families)
    assert aggregate[-1]["eligible"] is False


def test_duplicate_nominal_policies_are_flagged_not_independent() -> None:
    policies = [
        v2.Policy(v2.PRIMARY_POLICY_ID, "primary", 15, 0.2, "trace"),
        v2.Policy(v2.SENSITIVITY_POLICY_IDS[0], "sensitivity", 12, 0.2, "trace"),
        v2.Policy(v2.SENSITIVITY_POLICY_IDS[1], "sensitivity", 15, 0.2, "dopt"),
    ]
    rows = []
    for policy in policies:
        for model in ("a", "b"):
            order = '["s1", "s2"]'
            if policy.policy_id == v2.SENSITIVITY_POLICY_IDS[0] and model == "b":
                order = '["s1"]'
            rows.append(
                {"policy_id": policy.policy_id, "model": model, "cat_scenario_order": order}
            )
    duplicates = v2.detect_duplicate_cat_paths(rows, policies)
    primary_dopt = next(
        row
        for row in duplicates
        if row["left_policy_id"] == v2.PRIMARY_POLICY_ID
        and row["right_policy_id"] == v2.SENSITIVITY_POLICY_IDS[1]
    )
    assert primary_dopt["all_paths_identical"] is True
    assert primary_dopt["counted_as_independent_support"] is False


def test_family_cluster_bootstrap_is_deterministic_and_reports_family_unit() -> None:
    rows = []
    model_to_family = {}
    for index in range(8):
        model = f"m{index}"
        model_to_family[model] = f"f{index // 2}"
        rows.append(
            {
                "model": model,
                "cat_replay_success": True,
                "baseline_replay_success": True,
                "cat_mwle_converged": True,
                "cat_precision_reached": True,
                "theta_reference": float(index),
                "theta_cat_mwle": float(index) * 0.98,
                "cat_scenarios_administered": 15 + index % 2,
                "baseline_scenarios_administered": 40,
                "cat_eval_n_cells": 10,
                "cat_eval_log_loss_sum": 3.0,
                "cat_eval_brier_sum": 1.0,
                "cat_eval_correct_count": 8,
                "cat_eval_predicted_pass_rate": 0.51,
                "cat_eval_observed_pass_rate": 0.50,
                "fit_cache_key": "fit",
            }
        )
    left = v2.aggregate_policy_rows_clustered(
        rows, model_to_family, seed=12, replicates=100, label="test"
    )
    right = v2.aggregate_policy_rows_clustered(
        rows, model_to_family, seed=12, replicates=100, label="test"
    )
    assert left == right
    assert left["bootstrap_unit"] == "tutor_family"
    assert left["n_families"] == 4


def test_cache_key_is_v2_only_and_changes_by_spec_repeat_fold_and_environment() -> None:
    runner = v2.V2Phase3Runner.__new__(v2.V2Phase3Runner)
    runner.study_signature = "study"
    runner.input_hashes = {"response_matrix": "matrix", "rubrics": "rubrics"}
    runner.config_path = ROOT / "configs" / "infobench_calibration_cat_v2.json"
    runner.split_path = ROOT / "configs" / "infobench_v2_splits.manifest.json"
    runner.code_hashes = {"v2": "code"}
    runner.environment = {"canonical_sha256": "env-a"}
    runner.numerical = {
        "fit_grid": 81,
        "eap_grid": 401,
        "quadrature_method": "normal_trapezoid",
        "linear_bound": 8.0,
        "verification_sha256": "lock-a",
    }
    runner.args = argparse.Namespace(
        max_iter=200, tol=1e-4, negative_policy="drop"
    )
    specs = v2.expected_calibration_specs()
    common = dict(
        outer_fold=0,
        inner_fold=0,
        role="inner",
        training_model_ids=["a", "b"],
    )
    key = runner._fit_cache_key(repeat=0, spec=specs[0], **common)
    assert key != runner._fit_cache_key(repeat=1, spec=specs[0], **common)
    assert key != runner._fit_cache_key(repeat=0, spec=specs[1], **common)
    runner.numerical = {**runner.numerical, "verification_sha256": "lock-b"}
    assert key != runner._fit_cache_key(repeat=0, spec=specs[0], **common)
    runner.numerical = {**runner.numerical, "verification_sha256": "lock-a"}
    runner.environment = {"canonical_sha256": "env-b"}
    assert key != runner._fit_cache_key(repeat=0, spec=specs[0], **common)
    assert v2.CACHE_SCHEMA != v2.v1.SCRIPT_SCHEMA


def test_real_plan_only_reports_pending_numerical_lock_without_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan-only attempted calibration")

    monkeypatch.setattr(v2.cell_cv, "fit_structure", forbidden)
    output = tmp_path / "plan"
    status = v2.main(["--plan-only", "--out-dir", str(output)])
    assert status == 0
    manifest = _load(output / "manifest.json")
    assert manifest["status"] in {"plan_only_ready", "plan_only_blocked"}
    assert manifest["not_run"]["item_fits"] is True


def test_plan_only_rejects_posthoc_config(tmp_path: Path) -> None:
    config = _load(ROOT / "configs" / "infobench_calibration_cat_v2.json")
    config["status"] = "edited_after_results"
    path = tmp_path / "posthoc.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    status = v2.main(
        ["--plan-only", "--config", str(path), "--out-dir", str(tmp_path / "out")]
    )
    assert status == 2


def test_synthetic_cli_preflight_writes_nonofficial_artifact(tmp_path: Path) -> None:
    assert v2.main(["--synthetic-preflight", "--out-dir", str(tmp_path)]) == 0
    artifact = _load(tmp_path / "synthetic_phase3_preflight.json")
    assert artifact["status"] == "passed"
    assert artifact["official_result"] is False
    assert artifact["selected_spec_id"] == v2.expected_calibration_specs()[0].spec_id


def test_failed_all_panels_still_writes_complete_decision_and_never_promotes_sensitivity(
    tmp_path: Path,
) -> None:
    runner = v2.V2Phase3Runner.__new__(v2.V2Phase3Runner)
    runner.output_dir = tmp_path / "phase3"
    runner.output_dir.mkdir()
    runner.config_path = tmp_path / "config.json"
    runner.split_path = tmp_path / "splits.json"
    runner.config_path.write_text("{}", encoding="utf-8")
    runner.split_path.write_text("{}", encoding="utf-8")
    input_path = tmp_path / "input"
    input_path.write_text("fixture", encoding="utf-8")
    runner.input_paths = {
        name: input_path
        for name in ("response_matrix", "rubrics", "scenarios", "judge_manifest")
    }
    runner.input_hashes = {name: v2._sha256(input_path) for name in runner.input_paths}
    runner.specs = v2.expected_calibration_specs()
    runner.policies = [
        v2.Policy(v2.PRIMARY_POLICY_ID, "primary", 15, 0.2, "trace"),
        v2.Policy(v2.SENSITIVITY_POLICY_IDS[0], "sensitivity", 12, 0.2, "trace"),
        v2.Policy(v2.SENSITIVITY_POLICY_IDS[1], "sensitivity", 15, 0.2, "dopt"),
    ]
    runner.config = {
        "selection_gates": {
            "minimum_replay_success_rate": 0.99,
            "minimum_mwle_convergence_rate": 0.95,
            "minimum_nominal_precision_rate": 0.95,
            "minimum_nominal_precision_lower_95_ci": 0.90,
            "minimum_recovery_correlation_lower_95_ci": 0.85,
            "minimum_recovery_slope": 0.90,
            "maximum_recovery_slope": 1.10,
            "maximum_disjoint_pass_rate_mae": 0.07,
            "maximum_absolute_disjoint_pass_rate_bias": 0.03,
            "minimum_scenario_reduction_vs_random": 0.50,
            "require_paired_family_bootstrap_ci_favors_cat": True,
        }
    }
    models = [f"m{index:02d}" for index in range(52)]
    model_to_family = {
        model: f"f{min(index // 2, 21):02d}" for index, model in enumerate(models)
    }
    runner.matrix = pd.DataFrame(index=models)
    runner.split_audit = {"model_to_family": model_to_family}
    runner.study_signature = "synthetic-study"
    runner.code_hashes = {"v2": "hash"}
    runner.environment = {"canonical_sha256": "environment"}
    runner.numerical = {
        "fit_grid": 81,
        "eap_grid": 401,
        "quadrature_method": "normal_trapezoid",
        "linear_bound": 8.0,
        "verification_sha256": "synthetic-lock",
        "status": "passed",
    }
    runner.args = argparse.Namespace(
        seed=1,
        top_n=5,
        max_scenarios=50,
        minimum_scored_criteria=15,
        max_iter=200,
        tol=1e-4,
        mwle_ridge=1e-6,
        max_grid_nodes=50_000,
        negative_policy="drop",
        metric_bootstrap_replicates=100,
    )
    (runner.output_dir / "fold_assignments.json").write_text("{}\n", encoding="utf-8")
    (runner.output_dir / "pre_outer_selection_lock.json").write_text(
        "{}\n", encoding="utf-8"
    )
    panels = []
    inner = []
    for repeat in range(5):
        for outer_fold in range(5):
            panel_id = f"repeat_{repeat:02d}_outer_{outer_fold:02d}"
            panels.append(
                {
                    "panel_id": panel_id,
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "status": "selection_failed",
                    "selected_spec_id": None,
                    "selected_calibration_specification": None,
                    "inner_selection": {},
                    "outer_fit": None,
                    "outer_evaluation": {"status": "not_run_selection_failed"},
                }
            )
            for spec in runner.specs:
                inner.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "simplicity_rank": spec.simplicity_rank,
                        "spec_id": spec.spec_id,
                        "eligible": False,
                    }
                )
    runner._write_phase3_outputs(
        panels=panels, inner_results=inner, outer_rows=[], started_at="start"
    )
    decision = _load(runner.output_dir / "phase3_decision.json")
    assert decision["phase3_pass"] is False
    assert decision["phase4_authorized"] is False
    assert decision["sensitivities"]["can_promote"] is False
    assert "incomplete_outer_coverage" in decision["failed_conditions"]
    manifest = _load(runner.output_dir / "manifest.json")
    assert manifest["status"] == "phase3_complete"
    assert set(v2.REQUIRED_OUTPUTS) - {"manifest.json"} <= set(manifest["outputs"])
