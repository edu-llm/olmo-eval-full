from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import finalize_infobench_calibration_cat_v3 as final


def _frozen_config() -> dict:
    return json.loads(final.DEFAULT_CONFIG.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("section", "field", "mutant"),
    [
        ("selection_gates", "minimum_replay_success_rate", 0.98),
        ("selection_gates", "minimum_mwle_convergence_rate", 0.94),
        ("selection_gates", "minimum_nominal_precision_rate", 0.94),
        ("selection_gates", "minimum_nominal_precision_lower_95_ci", 0.89),
        ("selection_gates", "minimum_recovery_correlation_lower_95_ci", 0.84),
        ("selection_gates", "minimum_recovery_slope", 0.89),
        ("selection_gates", "maximum_recovery_slope", 1.11),
        ("selection_gates", "maximum_disjoint_pass_rate_mae", 0.08),
        (
            "selection_gates",
            "maximum_absolute_disjoint_pass_rate_bias",
            0.04,
        ),
        ("selection_gates", "minimum_scenario_reduction_vs_random", 0.49),
        ("selection_gates", "minimum_valid_parameter_bootstrap_rate", 0.89),
        ("selection_gates", "require_paired_family_bootstrap_ci_favors_cat", False),
        ("selection_gates", "require_every_outer_panel", False),
        ("selection_gates", "require_every_repetition_pooled_gate", False),
        ("selection_gates", "allow_fallback_if_primary_fails", True),
        ("calibration_selection", "allow_fallback", True),
        ("calibration_selection", "require_all_specs_every_inner_fold", False),
        ("calibration_selection", "survivor_selection_allowed", True),
        ("uncertainty", "minimum_valid_parameter_bootstrap_rate", 0.89),
    ],
)
def test_every_frozen_scientific_gate_is_mutation_resistant(
    section: str, field: str, mutant: object
) -> None:
    config = _frozen_config()
    final.validate_frozen_scientific_contract(config)

    changed = copy.deepcopy(config)
    changed[section][field] = mutant
    with pytest.raises(final.FinalFitError, match="frozen|gate|fallback|bootstrap"):
        final.validate_frozen_scientific_contract(changed)


def test_scientific_thresholds_reject_text_that_only_coerces_to_the_number() -> None:
    config = _frozen_config()
    config["selection_gates"]["minimum_replay_success_rate"] = "0.99"
    with pytest.raises(final.FinalFitError, match="minimum_replay_success_rate"):
        final.validate_frozen_scientific_contract(config)


def test_scientific_gate_name_order_and_denominator_are_exact() -> None:
    config = _frozen_config()
    config["selection_gates"]["outer_panel_applied_gate_names"] = list(
        reversed(config["selection_gates"]["outer_panel_applied_gate_names"])
    )
    with pytest.raises(final.FinalFitError, match="inventory"):
        final.validate_frozen_scientific_contract(config)

    config = _frozen_config()
    config["selection_gates"]["repetition_expected_unique_oof_tutors"] = 51
    with pytest.raises(final.FinalFitError, match="denominator"):
        final.validate_frozen_scientific_contract(config)


def test_final_output_is_pinned_to_one_non_symlink_v3_leaf(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    canonical = root / "runs" / "calibration" / "InFoBench_v3" / "final_fit"
    canonical.parent.mkdir(parents=True)
    assert final.validate_canonical_output_leaf(
        "runs/calibration/InFoBench_v3/final_fit",
        canonical,
        root=root,
        canonical=canonical,
    ) == canonical

    with pytest.raises(final.FinalFitError, match="canonical"):
        final.validate_canonical_output_leaf(
            "runs/calibration/InFoBench_v2/final_fit",
            root / "runs" / "calibration" / "InFoBench_v2" / "final_fit",
            root=root,
            canonical=canonical,
        )
    with pytest.raises(final.FinalFitError, match="canonical|outside"):
        final.validate_canonical_output_leaf(
            canonical,
            tmp_path / "external-final-fit",
            root=root,
            canonical=canonical,
        )


def test_final_output_rejects_a_symlinked_parent_even_if_it_resolves_to_leaf(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    real_parent = root / "real" / "InFoBench_v3"
    real_parent.mkdir(parents=True)
    calibration = root / "runs" / "calibration"
    calibration.mkdir(parents=True)
    (calibration / "InFoBench_v3").symlink_to(real_parent, target_is_directory=True)
    lexical = calibration / "InFoBench_v3" / "final_fit"
    with pytest.raises(final.FinalFitError, match="symlink|canonical"):
        final.validate_canonical_output_leaf(
            "runs/calibration/InFoBench_v3/final_fit",
            lexical,
            root=root,
            canonical=lexical,
        )


def test_output_preparation_rechecks_that_leaf_was_not_swapped_to_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "final_fit"
    linked_output.symlink_to(real_output, target_is_directory=True)
    runner = object.__new__(final.FinalFitRunner)
    runner.output_dir = linked_output
    runner.config = {"outputs": {"final_fit": "canonical-final-fit"}}
    runner.args = SimpleNamespace(resume=False)
    runner._reverify_sources = lambda _stage: None

    def reject_swapped_leaf(configured: str, requested: Path) -> Path:
        assert configured == "canonical-final-fit"
        assert requested == linked_output
        assert requested.is_symlink()
        raise final.FinalFitError("final output traverses a symlink")

    monkeypatch.setattr(final, "validate_canonical_output_leaf", reject_swapped_leaf)
    with pytest.raises(final.FinalFitError, match="symlink|output"):
        runner._prepare_output()


def test_runtime_axis_defaults_are_exact_and_each_axis_mutation_fails() -> None:
    args = final.build_argparser().parse_args([])
    final.validate_exact_runtime_axis(args)
    mutations = {
        "skills": "format,content,number,style,linguistic",
        "dimensions": "instruction_following=format+content+number+style+linguistic",
        "structure_name": "another_1d_structure",
        "require_complete_bank": not args.require_complete_bank,
        "max_grid_nodes": args.max_grid_nodes + 1,
    }
    for field, mutant in mutations.items():
        changed = argparse.Namespace(**vars(args))
        setattr(changed, field, mutant)
        with pytest.raises(final.FinalFitError, match="runtime axis"):
            final.validate_exact_runtime_axis(changed)


def test_captured_provenance_rejects_aliases_symlinks_and_later_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"version": 1}\n', encoding="utf-8")
    captured = final.CapturedProvenance.capture({"source": source})
    assert captured.sha256("source") == final._sha256(source)
    captured.reverify("before synthetic use")

    with pytest.raises(final.FinalFitError, match="aliases"):
        final.CapturedProvenance.capture({"a": source, "b": source})

    alias = tmp_path / "alias.json"
    alias.symlink_to(source)
    with pytest.raises(final.FinalFitError, match="symlink"):
        final.CapturedProvenance.capture({"source": alias})

    source.write_text('{"version": 2}\n', encoding="utf-8")
    with pytest.raises(final.FinalFitError, match="changed"):
        captured.reverify("before synthetic finalization")


def test_captured_provenance_inventory_itself_is_tamper_evident(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    captured = final.CapturedProvenance.capture({"source": source})
    captured.entries["source"]["sha256"] = "0" * 64
    with pytest.raises(final.FinalFitError, match="mutated in memory"):
        captured.reverify("synthetic boundary")


def _valid_final_fit() -> tuple[dict, final.phase3.CalibrationSpec, tuple[str, ...]]:
    config = _frozen_config()
    spec = final.phase3.load_calibration_specs(config)[0]
    items = ("criterion-1", "criterion-2")
    trace = [
        {
            "iteration": iteration,
            "marginal_loglik": -10.2 + 0.1 * iteration,
            "penalty": 0.0,
            "penalized_objective": -10.2 + 0.1 * iteration,
            "objective_change": 0.1 / iteration,
            "objective_decreased_beyond_tolerance": False,
            "max_abs_parameter_change": {
                "A": 1e-6,
                "b": 1e-6,
                "R": 0.0,
                "overall": 1e-6,
            },
            "objective_converged": iteration == 2,
            "parameter_converged": iteration == 2,
            "consecutive_passes": iteration,
            "required_consecutive_passes": 2,
            "mstep_optimizer": {
                "method": "vectorized_1d_1pl_newton",
                "n_items": len(items),
                "n_converged": len(items),
                "n_failed": 0,
                "max_abs_gradient": 1e-8,
            },
        }
        for iteration in (1, 2)
    ]
    fit = {
        "items": list(items),
        "A": np.ones((len(items), 1)),
        "b": np.array([-0.1, 0.2]),
        "R": np.eye(1),
        "dim_labels": ["instruction_following"],
        "loglik": -10.0,
        "n_params": len(items),
        "n_iter": len(trace),
        "converged": True,
        "grid_nodes": 401,
        "quadrature_method": final.cm.NORMAL_TRAPEZOID_QUADRATURE,
        "quadrature_linear_bound": 8.0,
        "convergence_mode": final.cm.RETURNED_ITERATE_CONVERGENCE,
        "penalized_objective": -10.0,
        "calibration_specification": spec.canonical,
        "convergence_diagnostics": {
            "objective_tolerance": 1e-4,
            "parameter_tolerance": 5e-5,
            "required_consecutive_passes": 2,
            "final_consecutive_passes": 2,
            "returned_iterate_matches_last_trace": True,
            "final_exact_recomputation": True,
            "stopped_before_extra_mstep": True,
            "all_objective_changes_monotone_within_tolerance": True,
            "trace": trace,
        },
    }
    return fit, spec, items


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda fit: fit.update(items=list(reversed(fit["items"]))), "roster"),
        (lambda fit: fit.update(loglik=float("nan")), "log likelihood"),
        (lambda fit: fit.update(n_params=3), "parameter count"),
        (lambda fit: fit.update(n_params=2.0), "parameter count"),
        (lambda fit: fit.update(n_iter=2.0), "iteration count"),
        (lambda fit: fit.update(R=np.array([[0.99]])), "correlation"),
        (lambda fit: fit.update(A=np.ones((2, 2))), "array shapes"),
        (
            lambda fit: fit["convergence_diagnostics"]["trace"][0][
                "mstep_optimizer"
            ].update(n_items=1),
            "optimizer n_items",
        ),
        (
            lambda fit: fit["convergence_diagnostics"]["trace"][0].update(
                iteration=2
            ),
            "iteration sequence",
        ),
        (
            lambda fit: fit["convergence_diagnostics"].update(
                trace=fit["convergence_diagnostics"]["trace"][:1]
            ),
            "trace length",
        ),
    ],
)
def test_final_fit_contract_rejects_roster_numeric_and_optimizer_mutations(
    mutation, match: str
) -> None:
    fit, spec, items = _valid_final_fit()
    assert final.validate_final_fit_contract(fit, spec, items)["fit_valid"] is True
    changed = copy.deepcopy(fit)
    mutation(changed)
    with pytest.raises(final.FinalFitError, match=match):
        final.validate_final_fit_contract(changed, spec, items)


def test_final_fit_contract_requires_family_specific_optimizer_method() -> None:
    fit, spec, items = _valid_final_fit()
    fit["convergence_diagnostics"]["trace"][0]["mstep_optimizer"][
        "method"
    ] = "unrelated-optimizer"
    with pytest.raises(final.FinalFitError, match="optimizer.*method"):
        final.validate_final_fit_contract(fit, spec, items)


def test_final_fit_loglik_must_match_final_trace_even_when_flag_claims_it_does() -> None:
    fit, spec, items = _valid_final_fit()
    assert fit["convergence_diagnostics"]["returned_iterate_matches_last_trace"] is True
    fit["loglik"] = -9.75
    with pytest.raises(final.FinalFitError, match="loglik.*final optimizer trace"):
        final.validate_final_fit_contract(fit, spec, items)


def test_final_fit_penalized_objective_must_match_final_trace_row() -> None:
    fit, spec, items = _valid_final_fit()
    fit["penalized_objective"] = -9.5
    with pytest.raises(
        final.FinalFitError, match="penalized_objective.*final optimizer trace"
    ):
        final.validate_final_fit_contract(fit, spec, items)


def _coverage_row() -> dict:
    return {
        "model": "model-a",
        "replay_seed": 1000,
        "status": "ok",
        "cat_replay_success": True,
        "baseline_replay_success": True,
        "cat_mwle_converged": True,
        "baseline_mwle_converged": True,
        "theta_cat_mwle": 0.25,
        "theta_baseline_mwle": 0.20,
    }


@pytest.mark.parametrize(
    ("field", "mutant", "paired_replay_succeeded"),
    [
        ("status", "replay_error", False),
        ("cat_replay_success", False, False),
        ("cat_mwle_converged", False, True),
        ("theta_cat_mwle", None, True),
        ("theta_cat_mwle", float("nan"), True),
        ("theta_cat_mwle", float("inf"), True),
        ("theta_cat_mwle", -float("inf"), True),
    ],
)
def test_valid_cat_score_requires_success_convergence_and_finite_theta(
    field: str, mutant: object, paired_replay_succeeded: bool
) -> None:
    row = _coverage_row()
    row[field] = mutant
    coverage = final.replay_coverage(
        [row], expected_models=["model-a"], expected_seeds=[1000]
    )
    assert coverage["attempt_grid_complete"] is True
    assert coverage["n_valid_cat_scores"] == 0
    assert coverage["all_cat_scores_valid"] is False
    assert coverage["all_paired_replays_successful"] is paired_replay_succeeded
    assert coverage["paired_evidence_available"] is False


def test_paired_evidence_is_available_only_for_a_complete_all_success_grid() -> None:
    rows = []
    for model in ("model-a", "model-b"):
        for seed in (1000, 1001):
            row = _coverage_row()
            row.update(model=model, replay_seed=seed)
            rows.append(row)
    coverage = final.replay_coverage(
        rows,
        expected_models=["model-a", "model-b"],
        expected_seeds=[1000, 1001],
    )
    assert coverage["attempt_grid_complete"] is True
    assert coverage["all_paired_replays_successful"] is True
    assert coverage["paired_evidence_available"] is True

    rows[0]["baseline_replay_success"] = False
    failed = final.replay_coverage(
        rows,
        expected_models=["model-a", "model-b"],
        expected_seeds=[1000, 1001],
    )
    assert failed["attempt_grid_complete"] is True
    assert failed["all_paired_replays_successful"] is False
    assert failed["paired_evidence_available"] is False


def _phase4_validation_fixture(tmp_path: Path) -> dict:
    paths = {
        "config_path": tmp_path / "config.json",
        "phase3_manifest_path": tmp_path / "p3_manifest.json",
        "phase3_decision_path": tmp_path / "p3_decision.json",
        "selected_path": tmp_path / "selected.json",
    }
    phase3_signature = "3" * 64
    phase4_signature = "4" * 64
    payloads = {
        "config_path": {"config": True},
        "phase3_manifest_path": {"study_signature": phase3_signature},
        "phase3_decision_path": {"phase3_pass": True},
        "selected_path": {"panels": []},
    }
    for name, path in paths.items():
        path.write_text(json.dumps(payloads[name]), encoding="utf-8")
    phase3_manifest = {
        "study_signature": phase3_signature,
        "inputs": {
            "response_matrix": {"sha256": "matrix-hash"},
            "rubrics": {"sha256": "rubrics-hash"},
            "scenarios": {"sha256": "scenarios-hash"},
            "judge_manifest": {"sha256": "judge-hash"},
        },
    }
    gates = pd.DataFrame(
        [
            {
                "repeat": repeat,
                "complete_total_se_support": True,
                "complete_order_support": True,
                "every_model_valid_draw_gate": True,
                "p90_total_se_gate": True,
                "every_order_seed_present_and_valid": True,
                "order_stability_gate": True,
                "repetition_phase4_pass": True,
                "n_models_expected": 52,
                "n_total_se_rows": 52,
                "n_order_rows": 52,
                "minimum_valid_draw_rate_required": 0.9,
                "minimum_valid_draw_rate_observed": 0.95,
                "maximum_p90_cat_total_se": 0.5,
                "p90_cat_total_se": 0.4,
                "maximum_median_order_path_sd": 0.2,
                "median_order_path_sd": 0.1,
            }
            for repeat in range(5)
        ]
    )
    gate_path = tmp_path / "repetition_gate_results.csv"
    gates.to_csv(gate_path, index=False)
    decision = {
        "schema_version": final.phase4.DECISION_SCHEMA,
        "status": "pass",
        "phase4_pass": True,
        "all_five_repetitions_pass": True,
        "final_fit_authorized": True,
        "sensitivity_policies_considered": False,
        "model_repeat_rows_treated_as_independent": False,
        "failed_repeats": [],
        "phase3_study_signature": phase3_signature,
        "study_signature": phase4_signature,
        "primary_policy": {
            "policy_id": final.PRIMARY_POLICY_ID,
            "role": "primary",
            "minimum_scenarios": 15,
            "conditional_se_target": 0.2,
            "selector": "trace",
        },
        "input_hashes": {
            "response_matrix": "matrix-hash",
            "rubrics": "rubrics-hash",
            "scenarios": "scenarios-hash",
            "judge_manifest": "judge-hash",
        },
        "config_sha256": final._sha256(paths["config_path"]),
        "phase3_manifest_sha256": final._sha256(paths["phase3_manifest_path"]),
        "selected_calibration_specs_sha256": final._sha256(paths["selected_path"]),
        "phase3_decision_sha256": final._sha256(paths["phase3_decision_path"]),
        "gate_table": {
            "path": gate_path.name,
            "sha256": final._sha256(gate_path),
        },
    }
    return {
        **paths,
        "phase4_dir": tmp_path,
        "phase3_manifest": phase3_manifest,
        "gates": gates,
        "gate_path": gate_path,
        "decision": decision,
        "manifest": {"study_signature": phase4_signature, "decision": copy.deepcopy(decision)},
    }


def _validate_phase4_fixture(fixture: dict) -> None:
    final.validate_phase4_decision(
        fixture["decision"],
        fixture["manifest"],
        phase4_dir=fixture["phase4_dir"],
        phase3_manifest=fixture["phase3_manifest"],
        config_path=fixture["config_path"],
        phase3_manifest_path=fixture["phase3_manifest_path"],
        phase3_decision_path=fixture["phase3_decision_path"],
        selected_path=fixture["selected_path"],
    )


@pytest.mark.parametrize(
    ("field", "mutant"),
    [
        ("repeat", 0.5),
        ("n_models_expected", 52.5),
        ("n_total_se_rows", 51),
        ("n_order_rows", -1),
        ("minimum_valid_draw_rate_observed", -0.1),
        ("minimum_valid_draw_rate_observed", 1.1),
        ("minimum_valid_draw_rate_observed", float("nan")),
        ("minimum_valid_draw_rate_observed", float("inf")),
        ("p90_cat_total_se", -0.1),
        ("p90_cat_total_se", float("inf")),
        ("median_order_path_sd", -0.1),
        ("median_order_path_sd", float("inf")),
    ],
)
def test_phase4_gate_table_rejects_noninteger_and_nonfinite_domain_mutations(
    tmp_path: Path, field: str, mutant: object
) -> None:
    fixture = _phase4_validation_fixture(tmp_path)
    _validate_phase4_fixture(fixture)
    fixture["gates"][field] = fixture["gates"][field].astype(object)
    fixture["gates"].loc[0, field] = mutant
    fixture["gates"].to_csv(fixture["gate_path"], index=False)
    fixture["decision"]["gate_table"]["sha256"] = final._sha256(
        fixture["gate_path"]
    )
    fixture["manifest"]["decision"] = copy.deepcopy(fixture["decision"])
    with pytest.raises(final.FinalFitError, match="Phase-4"):
        _validate_phase4_fixture(fixture)


@pytest.mark.parametrize(
    ("field", "mutant"),
    [
        ("phase4_pass", 1),
        ("all_five_repetitions_pass", "true"),
        ("failed_repeats", ()),
    ],
)
def test_phase4_decision_requires_exact_json_types(
    tmp_path: Path, field: str, mutant: object
) -> None:
    fixture = _phase4_validation_fixture(tmp_path)
    fixture["decision"][field] = mutant
    fixture["manifest"]["decision"] = copy.deepcopy(fixture["decision"])
    with pytest.raises(final.FinalFitError, match="Phase-4"):
        _validate_phase4_fixture(fixture)


def test_phase4_study_signature_must_be_64_lowercase_hex_characters(
    tmp_path: Path,
) -> None:
    fixture = _phase4_validation_fixture(tmp_path)
    fixture["decision"]["study_signature"] = "z" * 64
    fixture["manifest"]["study_signature"] = "z" * 64
    fixture["manifest"]["decision"] = copy.deepcopy(fixture["decision"])
    with pytest.raises(final.FinalFitError, match="signature|SHA-256"):
        _validate_phase4_fixture(fixture)


def _authorization_payloads_hardening() -> tuple[dict, dict, dict, dict, dict]:
    config = _frozen_config()
    specs = final.phase3.load_calibration_specs(config)
    primary = final.phase3.load_policies(config)[0]
    selected_ids = [specs[0].spec_id] * 20 + [specs[1].spec_id] * 3 + [specs[2].spec_id] * 2
    panels: list[dict] = []
    phase4_panels: list[dict] = []
    for position, spec_id in enumerate(selected_ids):
        repeat, outer = divmod(position, 5)
        spec = next(candidate for candidate in specs if candidate.spec_id == spec_id)
        panels.append(
            {
                "panel_id": f"repeat_{repeat:02d}_outer_{outer:02d}",
                "repeat": repeat,
                "outer_fold": outer,
                "status": "evaluation_complete",
                "selected_spec_id": spec_id,
                "selected_calibration_specification": spec.canonical,
                "inner_selection": {"complete_inner_evidence_audit": {"passed": True}},
            }
        )
        phase4_panels.append(
            {
                "panel_id": f"repeat_{repeat:02d}_outer_{outer:02d}",
                "repeat": repeat,
                "outer_fold": outer,
                "spec_id": spec_id,
                "calibration_specification": spec.canonical,
            }
        )
    signature = "3" * 64
    manifest = {
        "schema_version": final.phase3.SCRIPT_SCHEMA,
        "status": "phase3_complete",
        "study_signature": signature,
    }
    selected = {
        "schema_version": final.phase3.SELECTED_SPECS_SCHEMA,
        "study_signature": signature,
        "primary_policy": final.asdict(primary),
        "sensitivity_promotion_allowed": False,
        "panels": panels,
    }
    coverage_by_repetition = {
        str(repeat): {
            "n_primary_rows": 52,
            "n_unique_models": 52,
            "all_52_models_exactly_once": True,
        }
        for repeat in range(5)
    }
    phase3_decision = {
        "schema_version": final.phase3.DECISION_SCHEMA,
        "status": "pass",
        "study_signature": signature,
        "phase3_pass": True,
        "phase4_authorized": True,
        "failed_conditions": [],
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
            "by_repetition": coverage_by_repetition,
            "model_repeat_rows_treated_as_independent": False,
        },
        "calibration_stability": {
            "passed": True,
            "unique_modal_spec_id": specs[0].spec_id,
            "tied_modal_spec_ids": [],
            "modal_count": 20,
            "modal_fraction_all_25_panels": 0.8,
            "required_fraction": 0.8,
        },
        "primary_policy": {
            "policy": final.asdict(primary),
            "all_outer_panels_pass": True,
            "all_repetitions_pass": True,
            "failed_panel_ids": [],
            "failed_repetitions": [],
        },
        "sensitivities": {
            "diagnostic_only": True,
            "can_promote": False,
            "affected_phase3_decision": False,
        },
    }
    phase4_decision = {"panel_selected_specifications": phase4_panels}
    return config, selected, phase3_decision, phase4_decision, manifest


@pytest.mark.parametrize(
    ("path", "mutant"),
    [
        (("primary_policy", "all_outer_panels_pass"), False),
        (("primary_policy", "all_repetitions_pass"), False),
        (("coverage", "all_25_panels_evaluated"), False),
        (("coverage", "all_52_models_each_repetition"), False),
        (("inner_selection_fail_closed", "all_25_panels_complete"), False),
        (("inner_selection_fail_closed", "survivor_selection_allowed"), True),
        (("sensitivities", "can_promote"), True),
        (("sensitivities", "affected_phase3_decision"), True),
    ],
)
def test_phase3_authorization_requires_every_panel_and_repetition_flag(
    path: tuple[str, str], mutant: object
) -> None:
    config, selected, p3, p4, manifest = _authorization_payloads_hardening()
    final.select_authorized_final_spec(config, selected, p3, p4, manifest)
    p3[path[0]][path[1]] = mutant
    with pytest.raises(final.FinalFitError, match="Phase-3|authorized|complete|policy"):
        final.select_authorized_final_spec(config, selected, p3, p4, manifest)


def test_phase3_and_phase4_panel_coordinates_must_be_exact_integers() -> None:
    config, selected, p3, p4, manifest = _authorization_payloads_hardening()
    selected["panels"][0]["repeat"] = 0.5
    with pytest.raises(final.FinalFitError, match="panel|integer"):
        final.select_authorized_final_spec(config, selected, p3, p4, manifest)

    config, selected, p3, p4, manifest = _authorization_payloads_hardening()
    p4["panel_selected_specifications"][0]["outer_fold"] = 0.5
    with pytest.raises(final.FinalFitError, match="panel|integer"):
        final.select_authorized_final_spec(config, selected, p3, p4, manifest)


def _valid_semantic_replay_row() -> tuple[dict, final.phase3.Policy]:
    policy = final.phase3.load_policies(_frozen_config())[0]
    row = {
        "model": "model-a",
        "model_family": "family-a",
        "replay_seed": 1000,
        "candidate_id": policy.as_candidate().candidate_id,
        "policy_id": policy.policy_id,
        "policy_role": policy.role,
        "replay_scope": "paired_cat_random",
        "spec_id": "1pl_fixed_a1",
        "fit_cache_key": "cache-key",
        "final_fit_cache_key": "cache-key",
        "selector": policy.selector,
        "minimum_scenarios": policy.minimum_scenarios,
        "conditional_se_target": policy.conditional_se_target,
        "same_cohort_final_bank": True,
        "out_of_sample": False,
        "status": "ok",
        "error": "",
        "theta_reference": 0.3,
        "theta_reference_scope": "all_observed_administration_pool_items",
    }
    for arm, theta in (("cat", 0.25), ("baseline", 0.20)):
        row.update(
            {
                f"{arm}_replay_success": True,
                f"{arm}_precision_reached": True,
                f"{arm}_mwle_converged": True,
                f"{arm}_scenario_order": json.dumps(["scenario-a"]),
                f"{arm}_scenarios_administered": 1,
                f"{arm}_criteria_administered": 2,
                f"{arm}_stop_reason": "precision_reached",
                f"theta_{arm}_mwle": theta,
                f"{arm}_eval_n_cells": 2,
                f"{arm}_eval_correct_count": 1,
                f"{arm}_eval_log_loss_sum": 0.5,
                f"{arm}_eval_brier_sum": 0.25,
                f"{arm}_eval_observed_pass_rate": 0.5,
                f"{arm}_eval_predicted_pass_rate": 0.6,
            }
        )
    return row, policy


def _validate_semantic_replay(row: dict, policy: final.phase3.Policy) -> None:
    final.validate_replay_rows(
        [row],
        expected_model_to_family={"model-a": "family-a"},
        expected_seeds=[1000],
        expected_scope="paired_cat_random",
        expected_policy=policy,
        expected_spec_id="1pl_fixed_a1",
        expected_cache_key="cache-key",
        administration_scenario_ids=["scenario-a", "scenario-b"],
        maximum_scenarios=50,
    )


@pytest.mark.parametrize(
    ("field", "mutant"),
    [
        ("model_family", "wrong-family"),
        ("replay_seed", 1000.0),
        ("candidate_id", "wrong-candidate"),
        ("policy_id", "wrong-policy"),
        ("policy_role", "sensitivity"),
        ("replay_scope", "deployment"),
        ("spec_id", "wrong-spec"),
        ("fit_cache_key", "wrong-cache"),
        ("final_fit_cache_key", "wrong-cache"),
        ("same_cohort_final_bank", False),
        ("out_of_sample", True),
        ("status", "complete"),
        ("error", "contradictory error"),
        ("theta_reference", float("nan")),
        ("theta_reference_scope", "evaluation_pool"),
        ("cat_replay_success", 1),
        ("cat_scenario_order", json.dumps(["scenario-a", "scenario-a"])),
        ("cat_scenario_order", json.dumps(["outside-administration-pool"])),
        ("cat_scenarios_administered", 2),
        ("cat_criteria_administered", 0),
        ("theta_cat_mwle", float("inf")),
        ("cat_eval_correct_count", 3),
        ("cat_eval_log_loss_sum", -0.1),
        ("cat_eval_brier_sum", 3.0),
        ("cat_eval_observed_pass_rate", 1.1),
    ],
)
def test_replay_row_semantics_reject_identity_scope_and_domain_mutations(
    field: str, mutant: object
) -> None:
    row, policy = _valid_semantic_replay_row()
    _validate_semantic_replay(row, policy)
    row[field] = mutant
    with pytest.raises(final.FinalFitError, match="replay row"):
        _validate_semantic_replay(row, policy)


@pytest.mark.parametrize("arm", ["cat", "baseline"])
@pytest.mark.parametrize("mutation", ["loss", "brier", "complete_nonempty"])
def test_replay_error_rows_cannot_contain_any_evaluation_evidence(
    arm: str, mutation: str
) -> None:
    row, policy = _valid_semantic_replay_row()
    row.update(status="replay_error", error="OfflineStudyError: synthetic failure")
    row["theta_reference"] = None
    for name in ("cat", "baseline"):
        row.update(
            {
                f"{name}_replay_success": False,
                f"{name}_precision_reached": False,
                f"{name}_mwle_converged": False,
                f"{name}_scenario_order": "[]",
                f"{name}_scenarios_administered": None,
                f"{name}_criteria_administered": None,
                f"theta_{name}_mwle": None,
                f"{name}_eval_n_cells": 0,
                f"{name}_eval_correct_count": 0,
                f"{name}_eval_log_loss_sum": 0.0,
                f"{name}_eval_brier_sum": 0.0,
                f"{name}_eval_observed_pass_rate": None,
                f"{name}_eval_predicted_pass_rate": None,
            }
        )
    _validate_semantic_replay(row, policy)

    if mutation == "loss":
        row[f"{arm}_eval_log_loss_sum"] = 0.1
    elif mutation == "brier":
        row[f"{arm}_eval_brier_sum"] = 0.1
    else:
        row.update(
            {
                f"{arm}_eval_n_cells": 1,
                f"{arm}_eval_correct_count": 1,
                f"{arm}_eval_log_loss_sum": 0.1,
                f"{arm}_eval_brier_sum": 0.1,
                f"{arm}_eval_observed_pass_rate": 1.0,
                f"{arm}_eval_predicted_pass_rate": 0.9,
            }
        )
    with pytest.raises(final.FinalFitError, match="nonempty evaluation statistics"):
        _validate_semantic_replay(row, policy)


def test_replay_error_attempt_cannot_contaminate_scientific_metrics() -> None:
    rows: list[dict] = []
    families: dict[str, str] = {}
    for index, model in enumerate(("model-a", "model-b", "model-c")):
        row, _policy = _valid_semantic_replay_row()
        row.update(
            model=model,
            model_family=f"family-{index}",
            theta_reference=float(index - 1),
            theta_cat_mwle=float(index - 1) + 0.1,
            cat_scenarios_administered=10 + index,
            baseline_scenarios_administered=20 + index,
        )
        rows.append(row)
        families[model] = f"family-{index}"

    replay_error, policy = _valid_semantic_replay_row()
    replay_error.update(
        model="model-error",
        model_family="family-error",
        status="replay_error",
        error="OfflineStudyError: baseline failed after CAT",
        theta_reference=None,
        theta_cat_mwle=None,
        baseline_replay_success=False,
        baseline_precision_reached=False,
        baseline_mwle_converged=False,
        baseline_scenario_order="[]",
        baseline_scenarios_administered=None,
        baseline_criteria_administered=None,
        theta_baseline_mwle=None,
    )
    for arm in ("cat", "baseline"):
        replay_error.update(
            {
                f"{arm}_eval_n_cells": 0,
                f"{arm}_eval_correct_count": 0,
                f"{arm}_eval_log_loss_sum": 0.0,
                f"{arm}_eval_brier_sum": 0.0,
                f"{arm}_eval_observed_pass_rate": None,
                f"{arm}_eval_predicted_pass_rate": None,
            }
        )
    final.validate_replay_rows(
        [replay_error],
        expected_model_to_family={"model-error": "family-error"},
        expected_seeds=[1000],
        expected_scope="paired_cat_random",
        expected_policy=policy,
        expected_spec_id="1pl_fixed_a1",
        expected_cache_key="cache-key",
        administration_scenario_ids=["scenario-a", "scenario-b"],
        maximum_scenarios=50,
    )
    families["model-error"] = "family-error"

    base_eligible = final.scientific_replay_rows(rows)
    mutated_eligible = final.scientific_replay_rows([*rows, replay_error])
    assert mutated_eligible == base_eligible
    base_metrics = final.phase3.aggregate_policy_rows_clustered(
        base_eligible,
        families,
        seed=123,
        replicates=100,
        label="synthetic-filter",
    )
    mutated_metrics = final.phase3.aggregate_policy_rows_clustered(
        mutated_eligible,
        families,
        seed=123,
        replicates=100,
        label="synthetic-filter",
    )
    assert mutated_metrics == base_metrics

    _base_per_model, base_lengths = final.aggregate_paired_lengths(
        rows, families, seed=123, bootstrap_replicates=100
    )
    _mutated_per_model, mutated_lengths = final.aggregate_paired_lengths(
        [*rows, replay_error], families, seed=123, bootstrap_replicates=100
    )
    for field in (
        "mean_cat_scenarios",
        "mean_random_scenarios",
        "scenario_reduction_vs_random",
        "scenario_reduction_lower_95_ci",
        "scenario_reduction_upper_95_ci",
        "mean_paired_scenarios_saved",
        "mean_paired_scenarios_saved_lower_95_ci",
        "mean_paired_scenarios_saved_upper_95_ci",
    ):
        assert mutated_lengths[field] == base_lengths[field]
    assert mutated_lengths["n_replay_error_rows"] == 1
    assert mutated_lengths["n_scientific_length_rows"] == len(rows)


def test_code_provenance_inventory_cannot_silently_drop_a_direct_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashes = final._code_hashes()
    assert "scripts/scenario_cat_lib.py" in hashes
    incomplete = tuple(
        path for path in final.CODE_DEPENDENCIES if path.name != "scenario_cat_lib.py"
    )
    monkeypatch.setattr(final, "CODE_DEPENDENCIES", incomplete)
    with pytest.raises(final.FinalFitError, match="code.*inventory|dependency"):
        final._code_hashes()


def _synthetic_transaction_runner(tmp_path: Path) -> tuple[final.FinalFitRunner, dict]:
    fit, spec, items = _valid_final_fit()
    runner = object.__new__(final.FinalFitRunner)
    runner.output_dir = tmp_path
    runner.output_dir.mkdir(exist_ok=True)
    runner.args = SimpleNamespace(resume=False)
    runner.study_signature = "f" * 64
    runner.matrix = pd.DataFrame(index=["model-a", "model-b"])
    runner.final_spec = spec
    runner.numerical = {"profile": "synthetic-fit-401"}
    runner.input_hashes = {
        "response_matrix": "matrix-hash",
        "rubrics": "rubrics-hash",
        "scenarios": "scenarios-hash",
        "judge_manifest": "judge-hash",
    }
    runner.code_hashes = {"synthetic.py": "code-hash"}
    runner.environment = {"canonical_sha256": "environment-hash"}
    runner.runtime_axis = {
        "source_skills_order": list(final.phase3.DEFAULT_SKILLS),
        "source_skills_csv": final.EXACT_SKILLS,
        "dimensions": final.EXACT_DIMENSIONS,
        "structure_name": final.EXACT_STRUCTURE_NAME,
        "require_complete_bank": True,
        "max_grid_nodes": final.EXACT_MAX_GRID_NODES,
    }
    runner.structure_contract = {
        "name": final.EXACT_STRUCTURE_NAME,
        "labels": ["instruction_following"],
    }
    runner.structure_contract_sha256 = final._canonical_hash(
        runner.structure_contract
    )
    runner.expected_fit_items = items
    runner.expected_fit_items_sha256 = final._canonical_hash(items)
    runner.expected_crosswalk = {
        "criterion-1": "scenario-a",
        "criterion-2": "scenario-b",
    }
    runner.expected_crosswalk_sha256 = final._canonical_hash(
        runner.expected_crosswalk
    )
    for name in ("config_path", "phase3_decision_path", "phase4_decision_path"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        setattr(runner, name, path)
    provenance_paths = {
        "config": runner.config_path,
        "phase3_decision": runner.phase3_decision_path,
        "phase4_decision": runner.phase4_decision_path,
    }
    runner.provenance = final.CapturedProvenance.capture(provenance_paths)
    runner.provenance_keys = {name: name for name in provenance_paths}
    runner._revalidate_output_leaf = lambda: None
    return runner, fit


def test_final_fit_bundle_publishes_atomically_and_resume_hash_checks_every_file(
    tmp_path: Path,
) -> None:
    runner, fit = _synthetic_transaction_runner(tmp_path)
    validity = final.validate_final_fit_contract(
        fit, runner.final_spec, runner.expected_fit_items
    )
    runner._save_final_fit(fit, validity, {"n_nonpositive_items": 0})
    bundle = tmp_path / final.FINAL_FIT_BUNDLE
    assert {path.name for path in bundle.iterdir()} == {
        "final_fit_arrays.npz",
        "final_fit_manifest.json",
        "TRANSACTION.json",
    }
    assert not runner._orphan_fit_staging_dirs()
    runner._validate_fit_transaction()

    runner.args.resume = True
    loaded = runner._load_final_fit()
    assert loaded is not None
    assert loaded["items"] == list(runner.expected_fit_items)

    arrays = bundle / "final_fit_arrays.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tamper")
    with pytest.raises(final.FinalFitError, match="transaction.*hash|arrays.*hash"):
        runner._load_final_fit()


def test_interrupted_fit_publication_never_exposes_a_partial_committed_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, fit = _synthetic_transaction_runner(tmp_path)
    validity = final.validate_final_fit_contract(
        fit, runner.final_spec, runner.expected_fit_items
    )
    real_write = final._write_fsynced_json

    def interrupt_before_commit(path: Path, value: object) -> None:
        if path.name == "TRANSACTION.json":
            raise OSError("synthetic interruption")
        real_write(path, value)

    monkeypatch.setattr(final, "_write_fsynced_json", interrupt_before_commit)
    with pytest.raises(OSError, match="synthetic interruption"):
        runner._save_final_fit(fit, validity, {"n_nonpositive_items": 0})
    assert not (tmp_path / final.FINAL_FIT_BUNDLE).exists()
    orphaned = runner._orphan_fit_staging_dirs()
    assert len(orphaned) == 1
    assert (orphaned[0] / "final_fit_arrays.npz").is_file()
    assert (orphaned[0] / "final_fit_manifest.json").is_file()
    assert not (orphaned[0] / "TRANSACTION.json").exists()

    runner.args.resume = True
    with pytest.raises(final.FinalFitError, match="orphan.*preserved"):
        runner._load_final_fit()


def test_fit_publication_rehashes_captured_sources_before_writing(
    tmp_path: Path,
) -> None:
    runner, fit = _synthetic_transaction_runner(tmp_path)
    validity = final.validate_final_fit_contract(
        fit, runner.final_spec, runner.expected_fit_items
    )
    runner.config_path.write_text('{"mutated": true}\n', encoding="utf-8")
    with pytest.raises(final.FinalFitError, match="captured provenance changed"):
        runner._save_final_fit(fit, validity, {"n_nonpositive_items": 0})
    assert not (tmp_path / final.FINAL_FIT_BUNDLE).exists()
    assert not runner._orphan_fit_staging_dirs()


def _write_jsonl_fixture(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _export_roundtrip_fixture(tmp_path: Path) -> dict:
    records = [
        {
            "criterion_id": "criterion-1",
            "scenario_id": "scenario-a",
            "criterion": "First requirement",
            "q_modeled": {"instruction_following": 1},
            "discrimination": {"instruction_following": 1.0},
            "difficulty": -0.25,
            "irt_params": {
                "source": final.FINAL_BANK_SOURCE,
                "calibrated": True,
                "skills_order": ["instruction_following"],
                "latent_correlation": [[1.0]],
            },
        },
        {
            "criterion_id": "criterion-2",
            "scenario_id": "scenario-b",
            "criterion": "Second requirement",
            "q_modeled": {"instruction_following": 1},
            "discrimination": {"instruction_following": 1.0},
            "difficulty": 0.5,
            "irt_params": {
                "source": final.FINAL_BANK_SOURCE,
                "calibrated": True,
                "skills_order": ["instruction_following"],
                "latent_correlation": [[1.0]],
            },
        },
    ]
    scenarios = [
        {"scenario_id": "scenario-a", "criterion_ids": ["criterion-1"]},
        {"scenario_id": "scenario-b", "criterion_ids": ["criterion-2"]},
    ]
    rubric_path = tmp_path / "rubrics.jsonl"
    scenario_path = tmp_path / "scenarios.jsonl"
    _write_jsonl_fixture(rubric_path, records)
    _write_jsonl_fixture(scenario_path, scenarios)
    bank = final.scat.FittedBank(
        records=copy.deepcopy(records),
        dims=("instruction_following",),
        criterion_ids=("criterion-1", "criterion-2"),
        scenario_ids=("scenario-a", "scenario-b"),
        Q=np.ones((2, 1), dtype=int),
        A=np.ones((2, 1), dtype=float),
        b=np.array([-0.25, 0.5]),
        latent_correlation=np.eye(1),
    )
    return {
        "rubric_path": rubric_path,
        "scenario_path": scenario_path,
        "records": records,
        "scenarios": scenarios,
        "bank": bank,
        "crosswalk": {
            "criterion-1": "scenario-a",
            "criterion-2": "scenario-b",
        },
        "scenario_ids": ("scenario-a", "scenario-b"),
    }


def _validate_export_fixture(fixture: dict) -> dict:
    return final.validate_export_roundtrip(
        fixture["rubric_path"],
        fixture["scenario_path"],
        expected_bank=fixture["bank"],
        expected_crosswalk=fixture["crosswalk"],
        expected_scenario_ids=fixture["scenario_ids"],
    )


def test_export_roundtrip_accepts_only_exact_order_crosswalk_and_arrays(
    tmp_path: Path,
) -> None:
    fixture = _export_roundtrip_fixture(tmp_path)
    audit = _validate_export_fixture(fixture)
    assert audit["exact_roundtrip"] is True
    assert audit["n_items"] == 2
    assert audit["n_scenarios"] == 2

    _write_jsonl_fixture(fixture["rubric_path"], list(reversed(fixture["records"])))
    with pytest.raises(final.FinalFitError, match="criterion order"):
        _validate_export_fixture(fixture)


def test_export_roundtrip_rejects_scenario_order_and_crosswalk_mutations(
    tmp_path: Path,
) -> None:
    fixture = _export_roundtrip_fixture(tmp_path)
    _write_jsonl_fixture(
        fixture["scenario_path"], list(reversed(fixture["scenarios"]))
    )
    with pytest.raises(final.FinalFitError, match="scenario order"):
        _validate_export_fixture(fixture)

    fixture = _export_roundtrip_fixture(tmp_path)
    fixture["records"][0]["scenario_id"] = "scenario-b"
    _write_jsonl_fixture(fixture["rubric_path"], fixture["records"])
    with pytest.raises(final.FinalFitError, match="scenario IDs|crosswalk"):
        _validate_export_fixture(fixture)


@pytest.mark.parametrize(
    ("scenario_index", "criterion_ids"),
    [
        (0, ["criterion-1", "criterion-1"]),
        (1, ["criterion-1"]),
        (0, ["criterion-1", "unknown-criterion"]),
    ],
)
def test_export_roundtrip_rejects_duplicate_missing_or_extra_criterion_support(
    tmp_path: Path, scenario_index: int, criterion_ids: list[str]
) -> None:
    fixture = _export_roundtrip_fixture(tmp_path)
    fixture["scenarios"][scenario_index]["criterion_ids"] = criterion_ids
    _write_jsonl_fixture(fixture["scenario_path"], fixture["scenarios"])
    with pytest.raises(final.FinalFitError, match="criterion|crosswalk|support"):
        _validate_export_fixture(fixture)


@pytest.mark.parametrize("array_name", ["Q", "A", "b"])
def test_export_roundtrip_rejects_each_item_parameter_array_mutation(
    tmp_path: Path, array_name: str
) -> None:
    fixture = _export_roundtrip_fixture(tmp_path)
    mutated = copy.deepcopy(fixture["bank"])
    value = np.asarray(getattr(mutated, array_name)).copy()
    value.flat[0] = 0 if array_name == "Q" else value.flat[0] + 0.25
    setattr(mutated, array_name, value)
    fixture["bank"] = mutated
    with pytest.raises(final.FinalFitError, match=f"{array_name} parameters"):
        _validate_export_fixture(fixture)


def test_export_roundtrip_reads_and_validates_recorded_latent_correlation(
    tmp_path: Path,
) -> None:
    fixture = _export_roundtrip_fixture(tmp_path)
    fixture["records"][0]["irt_params"]["latent_correlation"] = [[0.9]]
    _write_jsonl_fixture(fixture["rubric_path"], fixture["records"])
    with pytest.raises(final.FinalFitError, match="R parameters|correlation"):
        _validate_export_fixture(fixture)
