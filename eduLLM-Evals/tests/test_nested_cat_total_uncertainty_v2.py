from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import nested_cat_total_uncertainty_v2 as phase4


def test_family_bootstrap_is_deterministic_clustered_and_training_only() -> None:
    mapping = {
        "a1": "family-a",
        "a2": "family-a",
        "b1": "family-b",
        "c1": "family-c",
        "c2": "family-c",
        "held": "family-held",
    }
    first = phase4.draw_outer_training_family_sample(
        ["c2", "a1", "b1", "a2", "c1"], ["held"], mapping, seed=123
    )
    second = phase4.draw_outer_training_family_sample(
        ["a1", "a2", "b1", "c1", "c2"], ["held"], mapping, seed=123
    )

    assert first == second
    assert len(first.family_ids) == 3
    assert set(first.model_ids) <= {"a1", "a2", "b1", "c1", "c2"}
    assert "held" not in first.model_ids
    for family in set(first.family_ids):
        expected_members = sorted(
            model for model in {"a1", "a2", "b1", "c1", "c2"} if mapping[model] == family
        )
        assert first.model_ids.count(expected_members[0]) == first.family_ids.count(family)
        for model in expected_members:
            assert first.model_ids.count(model) == first.family_ids.count(family)


def test_family_bootstrap_fails_on_family_or_model_leakage() -> None:
    with pytest.raises(phase4.V2Phase4Error, match="train/test leakage"):
        phase4.assert_fit_boundary(["a", "held"], ["held"])
    with pytest.raises(phase4.V2Phase4Error, match="outer-test IDs"):
        phase4.assert_fit_boundary(["a"], ["held"], ["held"])
    with pytest.raises(phase4.V2Phase4Error, match="outside"):
        phase4.assert_fit_boundary(["a"], ["held"], ["foreign"])
    with pytest.raises(phase4.V2Phase4Error, match="families cross"):
        phase4.draw_outer_training_family_sample(
            ["a"], ["held"], {"a": "same", "held": "same"}, seed=1
        )


def test_law_total_variance_computes_cat_and_full_diagnostic() -> None:
    rows = pd.DataFrame(
        {
            "panel_id": ["repeat_00_outer_00"] * 4,
            "repeat": [0] * 4,
            "outer_fold": [0] * 4,
            "spec_id": ["1pl_fixed_a1"] * 4,
            "replicate": [0, 1, 2, 3],
            "model": ["m"] * 4,
            "status": ["ok"] * 4,
            "mwle_converged": [True] * 4,
            "theta_mwle_instruction_following": [0.0, 2.0, 4.0, 6.0],
            "se_mwle_instruction_following": [1.0] * 4,
            "theta_full_eap_instruction_following": [0.0, 1.0, 2.0, 3.0],
            "se_full_eap_instruction_following": [0.5] * 4,
        }
    )
    cat = phase4.law_total_variance(
        rows,
        model_ids=["m"],
        dimensions=["instruction_following"],
        n_boot=4,
        minimum_valid_rate=0.9,
        estimator="mwle",
    )
    full = phase4.law_total_variance(
        rows,
        model_ids=["m"],
        dimensions=["instruction_following"],
        n_boot=4,
        minimum_valid_rate=0.9,
        estimator="full_eap",
    )
    combined = phase4.combine_cat_and_full_total_se(cat, full).iloc[0]

    assert combined["cat_mean_conditional_variance"] == pytest.approx(1.0)
    assert combined["cat_parameter_variance"] == pytest.approx(20.0 / 3.0)
    assert combined["cat_se_total"] == pytest.approx(math.sqrt(23.0 / 3.0))
    assert combined["full_mean_conditional_variance"] == pytest.approx(0.25)
    assert combined["full_parameter_variance"] == pytest.approx(5.0 / 3.0)
    assert combined["full_se_total"] == pytest.approx(math.sqrt(23.0 / 12.0))


def _gate_frames(cat_se: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = pd.DataFrame(
        {
            "repeat": [0, 0],
            "model": ["m1", "m2"],
            "cat_valid_draw_rate": [0.9, 1.0],
            "cat_valid_draw_gate": [True, True],
            "cat_se_total": [cat_se, cat_se],
            "full_se_total": [0.3, 0.3],
        }
    )
    order = pd.DataFrame(
        {
            "repeat": [0, 0],
            "model": ["m1", "m2"],
            "all_seed_attempts_present": [True, True],
            "all_seed_runs_valid": [True, True],
            "order_path_sd": [0.2, 0.2],
        }
    )
    return total, order


def test_frozen_gate_thresholds_are_inclusive_but_not_loosened() -> None:
    total, order = _gate_frames(0.5)
    exact = phase4.repetition_gate_results(
        total,
        order,
        expected_model_ids=["m1", "m2"],
        repeats=[0],
        minimum_valid_rate=0.9,
        maximum_p90_total_se=0.5,
        maximum_median_order_path_sd=0.2,
    ).iloc[0]
    assert bool(exact["repetition_phase4_pass"])

    too_large, order = _gate_frames(0.5000001)
    failed = phase4.repetition_gate_results(
        too_large,
        order,
        expected_model_ids=["m1", "m2"],
        repeats=[0],
        minimum_valid_rate=0.9,
        maximum_p90_total_se=0.5,
        maximum_median_order_path_sd=0.2,
    ).iloc[0]
    assert not bool(failed["p90_total_se_gate"])
    assert not bool(failed["repetition_phase4_pass"])


def test_gate_retains_complete_support_failure() -> None:
    total, order = _gate_frames(0.4)
    result = phase4.repetition_gate_results(
        total.iloc[:1],
        order,
        expected_model_ids=["m1", "m2"],
        repeats=[0],
        minimum_valid_rate=0.9,
        maximum_p90_total_se=0.5,
        maximum_median_order_path_sd=0.2,
    ).iloc[0]
    assert not bool(result["complete_total_se_support"])
    assert not bool(result["every_model_valid_draw_gate"])
    assert not bool(result["repetition_phase4_pass"])


def test_failed_order_seed_is_not_silently_dropped() -> None:
    seeds = list(range(1000, 1020))
    rows = []
    for seed in seeds:
        rows.append(
            {
                "panel_id": "repeat_00_outer_00",
                "repeat": 0,
                "outer_fold": 0,
                "spec_id": "1pl_fixed_a1",
                "model": "m",
                "order_seed": seed,
                "status": "replay_numerical_error" if seed == seeds[-1] else "ok",
                "mwle_converged": seed != seeds[-1],
                "theta_mwle_instruction_following": (
                    np.nan if seed == seeds[-1] else float(seed) / 1000
                ),
                "se_mwle_instruction_following": (np.nan if seed == seeds[-1] else 0.1),
            }
        )
    result = phase4.order_stability_frame(
        pd.DataFrame(rows),
        model_ids=["m"],
        dimensions=["instruction_following"],
        order_seeds=seeds,
    ).iloc[0]
    assert bool(result["all_seed_attempts_present"])
    assert result["n_valid_seed_runs"] == 19
    assert not bool(result["all_seed_runs_valid"])


def test_model_repeat_rows_are_aggregated_before_population_summary() -> None:
    total_rows = []
    order_rows = []
    for repeat in range(5):
        for model, value in (("m1", 0.3), ("m2", 0.4)):
            total_rows.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "cat_valid_draw_rate": 1.0,
                    "cat_se_total": value + repeat * 0.01,
                    "full_se_total": value / 2,
                }
            )
            order_rows.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "order_path_sd": 0.1,
                    "all_seed_runs_valid": True,
                }
            )
    aggregate = phase4.aggregate_repeated_models(
        pd.DataFrame(total_rows),
        pd.DataFrame(order_rows),
        expected_model_ids=["m1", "m2"],
        expected_repeats=range(5),
    )
    assert len(aggregate) == 2
    assert aggregate["n_repeats"].tolist() == [5, 5]
    assert aggregate.loc[aggregate["model"] == "m1", "mean_cat_total_se"].iloc[0] == pytest.approx(
        0.32
    )

    duplicated = pd.concat([pd.DataFrame(total_rows), pd.DataFrame(total_rows).iloc[:1]])
    with pytest.raises(phase4.V2Phase4Error, match="duplicate model-repeat"):
        phase4.aggregate_repeated_models(
            duplicated,
            pd.DataFrame(order_rows),
            expected_model_ids=["m1", "m2"],
            expected_repeats=range(5),
        )


def test_family_metric_bootstrap_is_deterministic_and_uses_family_unit() -> None:
    frame = pd.DataFrame(
        {
            "model": ["a1", "a2", "b1", "c1"],
            "mean_cat_valid_draw_rate": [1.0, 0.9, 1.0, 0.95],
            "mean_cat_total_se": [0.2, 0.3, 0.4, 0.5],
            "mean_full_total_se": [0.1, 0.2, 0.2, 0.3],
            "median_order_path_sd": [0.05, 0.1, 0.15, 0.2],
        }
    )
    mapping = {"a1": "a", "a2": "a", "b1": "b", "c1": "c"}
    first = phase4.family_cluster_bootstrap_summary(frame, mapping, seed=7, replicates=100)
    second = phase4.family_cluster_bootstrap_summary(
        frame.sample(frac=1, random_state=3), mapping, seed=7, replicates=100
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(first["bootstrap_unit"]) == {"tutor_family"}
    assert first["repeated_rows_aggregated_per_model_first"].all()


def _authorized_phase3_payloads() -> tuple[dict, dict, dict]:
    signature = "phase3-signature"
    manifest = {
        "schema_version": phase4.phase3.SCRIPT_SCHEMA,
        "status": "phase3_complete",
        "study_signature": signature,
    }
    selected = {
        "schema_version": phase4.phase3.SELECTED_SPECS_SCHEMA,
        "study_signature": signature,
    }
    decision = {
        "schema_version": phase4.phase3.DECISION_SCHEMA,
        "status": "pass",
        "phase3_pass": True,
        "phase4_authorized": True,
        "study_signature": signature,
        "failed_conditions": [],
        "coverage": {
            "all_25_panels_evaluated": True,
            "all_52_models_each_repetition": True,
        },
        "calibration_stability": {
            "passed": True,
            "modal_fraction_all_25_panels": 0.8,
            "required_fraction": 0.8,
        },
        "primary_policy": {
            "all_outer_panels_pass": True,
            "all_repetitions_pass": True,
            "failed_panels": [],
            "failed_repeats": [],
        },
        "sensitivities": {"diagnostic_only": True, "can_promote": False},
    }
    return decision, selected, manifest


def test_phase3_authorization_fails_closed() -> None:
    decision, selected, manifest = _authorized_phase3_payloads()
    phase4.validate_phase3_authorization(decision, selected, manifest)

    decision = json.loads(json.dumps(decision))
    decision["phase4_authorized"] = False
    with pytest.raises(phase4.V2Phase4Error, match="did not pass"):
        phase4.validate_phase3_authorization(decision, selected, manifest)


def test_phase4_requires_exact_phase3_numerical_profile() -> None:
    profile = {
        "fit_grid": 101,
        "eap_grid": 401,
        "verification_sha256": "lock-a",
        "followup_config_sha256": "config-a",
    }
    phase4.require_phase3_numerical_profile({"numerical_lock": profile}, profile)
    with pytest.raises(phase4.V2Phase4Error, match="numerical lock/profile differs"):
        phase4.require_phase3_numerical_profile(
            {"numerical_lock": profile},
            {**profile, "verification_sha256": "lock-b"},
        )


def test_sensitivity_policy_cannot_enter_replay() -> None:
    sensitivity = phase4.phase3.Policy(
        policy_id="sensitivity_floor12_se0p20_trace",
        role="sensitivity",
        minimum_scenarios=12,
        conditional_se_target=0.2,
        selector="trace",
    )
    with pytest.raises(phase4.V2Phase4Error, match="only the frozen primary"):
        phase4.replay_primary_model(
            model="m",
            row=pd.Series(dtype=float),
            bank=SimpleNamespace(),
            scenario_records={},
            quadrature=SimpleNamespace(),
            policy=sensitivity,
            seed=1,
            top_n=5,
            max_scenarios=50,
            minimum_scored_criteria=15,
            mwle_ridge=1e-6,
        )


def _tiny_panel() -> phase4.Panel:
    spec = phase4.phase3.expected_calibration_specs()[0]
    return phase4.Panel(
        panel_id="repeat_00_outer_00",
        repeat=0,
        outer_fold=0,
        training_model_ids=("train-a", "train-b"),
        test_model_ids=("test-a", "test-b"),
        training_family_ids=("family-a", "family-b"),
        test_family_ids=("family-test-a", "family-test-b"),
        selected_spec=spec,
        selected_specification=spec.canonical,
        outer_fit_cache_key="outer-fit",
        outer_fit_cache_dir=Path("unused"),
        outer_fit_manifest_path=Path("unused/fit_manifest.json"),
        outer_fit_manifest_sha256="hash",
    )


def test_tiny_synthetic_panel_runs_primary_bootstrap_without_relaxing_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = _tiny_panel()
    runner = object.__new__(phase4.V2Phase4Runner)
    runner.panels = (panel,)
    runner.n_boot = 2
    runner.bootstrap_seed = 9
    runner.model_to_family = {
        "train-a": "family-a",
        "train-b": "family-b",
        "test-a": "family-test-a",
        "test-b": "family-test-b",
    }
    runner.primary_policy = phase4.phase3.Policy(
        policy_id=phase4.PRIMARY_POLICY_ID,
        role="primary",
        minimum_scenarios=15,
        conditional_se_target=0.2,
        selector="trace",
    )
    runner.matrix = pd.DataFrame(index=["test-a", "test-b"])
    runner.scenario_records = {}
    runner.cat_seed = 11
    runner.top_n = 5
    runner.max_scenarios = 50
    runner.minimum_scored_criteria = 15
    runner.mwle_ridge = 1e-6
    runner.args = SimpleNamespace(resume=False)
    runner.output_dir = tmp_path
    runner.study_signature = "synthetic"
    runner.input_hashes = {"synthetic": "input-hash"}
    runner.code_hashes = {"synthetic.py": "code-hash"}
    runner.environment = {"canonical_sha256": "environment-hash"}
    runner.config_path = tmp_path / "synthetic-config.json"
    runner.selected_path = tmp_path / "synthetic-selected.json"
    runner.config_path.write_text("{}\n", encoding="utf-8")
    runner.selected_path.write_text("{}\n", encoding="utf-8")
    fit_calls: list[tuple[int, tuple[str, ...]]] = []

    def fake_cache_key(self, panel_arg, replicate, sample):
        return f"cache-{replicate}"

    def fake_fit(self, panel_arg, replicate, sample):
        fit_calls.append((replicate, sample.family_ids))
        return phase4.BankBundle(
            bank=SimpleNamespace(dims=("instruction_following",), scenario_ids=()),
            quadrature=object(),
            fit={},
            cache_key=f"cache-{replicate}",
        )

    monkeypatch.setattr(runner, "_bootstrap_cache_key", MethodType(fake_cache_key, runner))
    monkeypatch.setattr(runner, "_fit_bootstrap", MethodType(fake_fit, runner))
    monkeypatch.setattr(runner, "_load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_write_checkpoint", lambda *args, **kwargs: None)

    def fake_replay(**kwargs):
        assert kwargs["policy"].policy_id == phase4.PRIMARY_POLICY_ID
        return {
            "model": kwargs["model"],
            "scenario_order": [],
            "criterion_order": [],
        }

    monkeypatch.setattr(phase4, "replay_primary_model", fake_replay)
    monkeypatch.setattr(
        phase4.scat,
        "flatten_result",
        lambda result, dims: {
            "model": result["model"],
            "mwle_converged": True,
            "theta_mwle_instruction_following": 0.0,
            "se_mwle_instruction_following": 0.1,
            "theta_full_eap_instruction_following": 0.0,
            "se_full_eap_instruction_following": 0.1,
        },
    )

    rows = phase4.V2Phase4Runner._run_bootstraps(runner)
    assert len(rows) == 2 * 2
    assert len(fit_calls) == 2
    assert set(rows["policy_id"]) == {phase4.PRIMARY_POLICY_ID}
    assert {"test-a", "test-b"} == set(rows["model"])


def test_plan_only_run_is_read_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runner = object.__new__(phase4.V2Phase4Runner)
    runner.args = SimpleNamespace(plan_only=True, resume=False)
    runner.plan_payload = MethodType(
        lambda self: {"read_only": True, "output": str(tmp_path / "never-created")}, runner
    )
    assert phase4.V2Phase4Runner.run(runner) == 0
    assert not (tmp_path / "never-created").exists()
    assert json.loads(capsys.readouterr().out)["read_only"] is True


def test_synthetic_phase3_handoff_through_phase4_decision(
    tmp_path: Path,
) -> None:
    """Exercise the contiguous handoff without fitting or changing frozen gates."""

    decision, selected, manifest = _authorized_phase3_payloads()
    selected["panels"] = [
        {
            "panel_id": f"repeat_{repeat:02d}_outer_{fold:02d}",
            "repeat": repeat,
            "outer_fold": fold,
            "status": "evaluation_complete",
            "selected_spec_id": "1pl_fixed_a1",
        }
        for repeat in range(5)
        for fold in range(5)
    ]
    selected_path = tmp_path / "selected_calibration_specs.json"
    selected_path.write_text(json.dumps(selected, sort_keys=True), encoding="utf-8")
    decision["selected_calibration_specs"] = {
        "path": str(selected_path),
        "sha256": phase4._sha256(selected_path),
    }
    manifest_path = tmp_path / "manifest.json"
    decision_path = tmp_path / "phase3_decision.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    decision_path.write_text(json.dumps(decision, sort_keys=True), encoding="utf-8")

    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded_selected = json.loads(selected_path.read_text(encoding="utf-8"))
    loaded_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    phase4.validate_phase3_authorization(loaded_decision, loaded_selected, loaded_manifest)
    assert len(loaded_selected["panels"]) == 25
    assert loaded_decision["selected_calibration_specs"]["sha256"] == phase4._sha256(selected_path)

    models = ["model-a", "model-b"]
    all_total: list[pd.DataFrame] = []
    all_order: list[pd.DataFrame] = []
    for repeat in range(5):
        draw_rows = []
        for replicate in range(100):
            for model_index, model in enumerate(models):
                draw_rows.append(
                    {
                        "panel_id": f"repeat_{repeat:02d}_outer_00",
                        "repeat": repeat,
                        "outer_fold": 0,
                        "spec_id": "1pl_fixed_a1",
                        "replicate": replicate,
                        "model": model,
                        "status": "ok",
                        "mwle_converged": True,
                        "theta_mwle_instruction_following": (
                            model_index * 0.2 + replicate * 0.0005
                        ),
                        "se_mwle_instruction_following": 0.1,
                        "theta_full_eap_instruction_following": (
                            model_index * 0.2 + replicate * 0.00025
                        ),
                        "se_full_eap_instruction_following": 0.08,
                    }
                )
        draws = pd.DataFrame(draw_rows)
        cat = phase4.law_total_variance(
            draws,
            model_ids=models,
            dimensions=["instruction_following"],
            n_boot=100,
            minimum_valid_rate=0.9,
            estimator="mwle",
        )
        full = phase4.law_total_variance(
            draws,
            model_ids=models,
            dimensions=["instruction_following"],
            n_boot=100,
            minimum_valid_rate=0.9,
            estimator="full_eap",
        )
        all_total.append(phase4.combine_cat_and_full_total_se(cat, full))

        order_rows = []
        for seed_offset, seed in enumerate(range(1000, 1020)):
            for model_index, model in enumerate(models):
                order_rows.append(
                    {
                        "panel_id": f"repeat_{repeat:02d}_outer_00",
                        "repeat": repeat,
                        "outer_fold": 0,
                        "spec_id": "1pl_fixed_a1",
                        "model": model,
                        "order_seed": seed,
                        "status": "ok",
                        "mwle_converged": True,
                        "theta_mwle_instruction_following": (
                            model_index * 0.2 + seed_offset * 0.001
                        ),
                        "se_mwle_instruction_following": 0.1,
                    }
                )
        all_order.append(
            phase4.order_stability_frame(
                pd.DataFrame(order_rows),
                model_ids=models,
                dimensions=["instruction_following"],
                order_seeds=range(1000, 1020),
            )
        )

    total = pd.concat(all_total, ignore_index=True)
    order = pd.concat(all_order, ignore_index=True)
    gates = phase4.repetition_gate_results(
        total,
        order,
        expected_model_ids=models,
        repeats=range(5),
        minimum_valid_rate=0.9,
        maximum_p90_total_se=0.5,
        maximum_median_order_path_sd=0.2,
    )
    aggregate = phase4.aggregate_repeated_models(
        total,
        order,
        expected_model_ids=models,
        expected_repeats=range(5),
    )
    family_summary = phase4.family_cluster_bootstrap_summary(
        aggregate,
        {"model-a": "family-a", "model-b": "family-b"},
        seed=20260805,
        replicates=100,
    )
    decision_like = {
        "schema_version": phase4.DECISION_SCHEMA,
        "status": "pass" if gates["repetition_phase4_pass"].all() else "fail",
        "phase4_pass": bool(gates["repetition_phase4_pass"].all()),
        "final_fit_authorized": bool(gates["repetition_phase4_pass"].all()),
        "n_model_aggregates": len(aggregate),
        "family_bootstrap_unit": family_summary["bootstrap_unit"].unique().tolist(),
    }
    phase4_decision_path = tmp_path / "phase4_decision.json"
    phase4_decision_path.write_text(json.dumps(decision_like, sort_keys=True), encoding="utf-8")

    assert gates["repetition_phase4_pass"].all()
    assert len(aggregate) == 2
    assert set(family_summary["bootstrap_unit"]) == {"tutor_family"}
    assert json.loads(phase4_decision_path.read_text(encoding="utf-8"))["final_fit_authorized"]
