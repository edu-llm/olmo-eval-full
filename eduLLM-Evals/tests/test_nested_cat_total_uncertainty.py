from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import nested_cat_total_uncertainty as phase4


def test_training_bootstrap_is_deterministic_and_never_draws_outer_test() -> None:
    training = ["train-c", "train-a", "train-b", "train-d"]
    test = ["held-out-a", "held-out-b"]
    first = phase4.draw_outer_training_sample(training, test, seed=31415)
    second = phase4.draw_outer_training_sample(reversed(training), test, seed=31415)

    assert first == second
    assert len(first) == len(training)
    assert set(first) <= set(training)
    assert set(first).isdisjoint(test)


def test_fit_boundary_fails_closed_on_outer_test_or_foreign_ids() -> None:
    with pytest.raises(phase4.Phase4Error, match="train/test leakage"):
        phase4.assert_fit_boundary(["a", "held"], ["held"])
    with pytest.raises(phase4.Phase4Error, match="outer-test IDs"):
        phase4.assert_fit_boundary(["a", "b"], ["held"], ["a", "held"])
    with pytest.raises(phase4.Phase4Error, match="outside"):
        phase4.assert_fit_boundary(["a", "b"], ["held"], ["a", "foreign"])


def test_replay_model_reruns_path_using_each_supplied_bootstrap_bank(monkeypatch) -> None:
    calls: list[object] = []

    def fake_run(model, row, bank, scenarios, quadrature, spec, *, mwle_ridge):
        calls.append(bank)
        return {
            "scenario_order": list(bank.scenario_ids),
            "criterion_order": [],
            "mwle_converged": True,
        }

    monkeypatch.setattr(phase4.scat, "run_recorded_model", fake_run)
    candidate = phase4.nested.Candidate(5, 0.25, "trace")
    bank_a = SimpleNamespace(scenario_ids=("scenario-a",), dims=("ability",))
    bank_b = SimpleNamespace(scenario_ids=("scenario-b",), dims=("ability",))
    common = dict(
        model="held-out",
        row=pd.Series(dtype=float),
        scenario_records={},
        quadrature=object(),
        candidate=candidate,
        seed=9,
        top_n=5,
        max_scenarios=50,
        minimum_scored_criteria=15,
        mwle_ridge=1e-6,
    )

    first = phase4.replay_model(bank=bank_a, **common)
    second = phase4.replay_model(bank=bank_b, **common)

    assert calls == [bank_a, bank_b]
    assert first["scenario_order"] == ["scenario-a"]
    assert second["scenario_order"] == ["scenario-b"]


def test_law_total_variance_uses_mean_conditional_variance_plus_theta_variance() -> None:
    draws = pd.DataFrame(
        {
            "model": ["m"] * 5,
            "status": ["ok", "ok", "ok", "ok", "replay_error"],
            "mwle_converged": [True, True, True, True, False],
            "theta_mwle_ability": [0.0, 2.0, 4.0, 6.0, np.nan],
            "se_mwle_ability": [1.0, 1.0, 1.0, 1.0, np.nan],
        }
    )
    result = phase4.law_total_variance(
        draws,
        model_ids=["m"],
        dimensions=["ability"],
        n_boot=5,
        minimum_valid_rate=0.9,
    ).iloc[0]

    assert result["n_valid_draws"] == 4
    assert result["valid_draw_rate"] == pytest.approx(0.8)
    assert result["mean_conditional_variance"] == pytest.approx(1.0)
    assert result["parameter_variance"] == pytest.approx(20.0 / 3.0)
    assert result["se_ability"] == pytest.approx(1.0)
    assert result["se_param"] == pytest.approx(math.sqrt(20.0 / 3.0))
    assert result["se_total"] == pytest.approx(math.sqrt(23.0 / 3.0))
    assert not bool(result["valid_draw_gate"])


def test_total_variance_retains_explicit_unreliable_row_when_every_draw_fails() -> None:
    draws = pd.DataFrame(
        {
            "model": ["m", "m"],
            "status": ["fit_error", "fit_error"],
            "mwle_converged": [False, False],
        }
    )
    result = phase4.law_total_variance(
        draws,
        model_ids=["m"],
        dimensions=["ability"],
        n_boot=2,
        minimum_valid_rate=0.9,
    ).iloc[0]

    assert result["n_valid_draws"] == 0
    assert result["valid_draw_rate"] == 0.0
    assert pd.isna(result["se_total"])
    assert not bool(result["valid_draw_gate"])


def test_order_stability_requires_at_least_twenty_unique_seeds() -> None:
    with pytest.raises(phase4.Phase4Error, match="at least 20"):
        phase4.validate_order_seeds(range(19))
    with pytest.raises(phase4.Phase4Error, match="unique"):
        phase4.validate_order_seeds([*range(19), 18])
    assert phase4.validate_order_seeds(range(1000, 1020)) == tuple(range(1000, 1020))


def test_total_variance_output_is_sorted_and_byte_deterministic() -> None:
    draws = pd.DataFrame(
        {
            "model": ["z", "a", "z", "a"],
            "status": ["ok"] * 4,
            "mwle_converged": [True] * 4,
            "theta_mwle_ability": [1.0, -1.0, 2.0, 0.0],
            "se_mwle_ability": [0.2] * 4,
        }
    )
    first = phase4.law_total_variance(
        draws,
        model_ids=["z", "a"],
        dimensions=["ability"],
        n_boot=2,
        minimum_valid_rate=0.9,
    )
    second = phase4.law_total_variance(
        draws.sample(frac=1, random_state=7),
        model_ids=["a", "z"],
        dimensions=["ability"],
        n_boot=2,
        minimum_valid_rate=0.9,
    )
    assert first["model"].tolist() == ["a", "z"]
    assert first.to_csv(index=False) == second.to_csv(index=False)


def _panel_with_two_candidates() -> phase4.FoldPanel:
    return phase4.FoldPanel(
        outer_fold=0,
        training_model_ids=("train-a", "train-b"),
        test_model_ids=("test-a", "test-b"),
        selected_ridge=0.01,
        candidates=(
            phase4.nested.Candidate(5, 0.25, "trace"),
            phase4.nested.Candidate(5, 0.25, "dopt"),
        ),
        fit_cache_key="base-fit",
    )


def test_one_bootstrap_fit_is_reused_across_every_candidate_and_model(
    tmp_path: Path, monkeypatch
) -> None:
    panel = _panel_with_two_candidates()
    runner = object.__new__(phase4.Phase4Runner)
    runner.prereq = SimpleNamespace(
        panels=(panel,),
        matrix=pd.DataFrame(index=["test-a", "test-b"]),
        scenario_records={},
        nested_dir=tmp_path,
    )
    runner.n_boot = 1
    runner.phase4_signature = "signature"
    runner.args = SimpleNamespace(
        bootstrap_seed=11,
        cat_seed=12,
        top_n=5,
        max_scenarios=50,
        minimum_scored_criteria=15,
        mwle_ridge=1e-6,
        resume=False,
    )
    calls = {"fit": 0, "replay": 0}

    def fake_fit(panel_arg, replicate, sample):
        calls["fit"] += 1
        assert panel_arg is panel and replicate == 0 and len(sample) == 2
        return SimpleNamespace(dims=("ability",)), object(), "bootstrap-key"

    def fake_replay(**kwargs):
        calls["replay"] += 1
        return {
            "model": kwargs["model"],
            "mwle_converged": True,
            "scenario_order": [],
        }

    monkeypatch.setattr(phase4, "_bootstrap_cache_key", lambda *args: "bootstrap-key")
    monkeypatch.setattr(runner, "_fit_bootstrap", fake_fit)
    monkeypatch.setattr(phase4, "replay_model", fake_replay)
    monkeypatch.setattr(phase4.scat, "flatten_result", lambda result, dims: dict(result))
    monkeypatch.setattr(runner, "_write_checkpoint", lambda *args, **kwargs: None)

    rows = phase4.Phase4Runner._run_bootstraps(runner)

    assert calls == {"fit": 1, "replay": 4}
    assert len(rows) == 4
    assert set(zip(rows["candidate_id"], rows["model"], strict=True)) == {
        (candidate.candidate_id, model)
        for candidate in panel.candidates
        for model in panel.test_model_ids
    }


def test_phase4_invariant_failure_aborts_instead_of_becoming_failed_draw(
    tmp_path: Path, monkeypatch
) -> None:
    panel = _panel_with_two_candidates()
    runner = object.__new__(phase4.Phase4Runner)
    runner.prereq = SimpleNamespace(
        panels=(panel,),
        matrix=pd.DataFrame(index=["test-a", "test-b"]),
        scenario_records={},
        nested_dir=tmp_path,
    )
    runner.n_boot = 1
    runner.phase4_signature = "signature"
    runner.args = SimpleNamespace(
        bootstrap_seed=11,
        cat_seed=12,
        top_n=5,
        max_scenarios=50,
        minimum_scored_criteria=15,
        mwle_ridge=1e-6,
        resume=False,
    )
    monkeypatch.setattr(phase4, "_bootstrap_cache_key", lambda *args: "bootstrap-key")
    monkeypatch.setattr(
        runner,
        "_fit_bootstrap",
        lambda *args: (_ for _ in ()).throw(phase4.Phase4Error("leakage")),
    )

    with pytest.raises(phase4.Phase4Error, match="leakage"):
        phase4.Phase4Runner._run_bootstraps(runner)


def test_balanced_support_rejects_missing_or_duplicate_candidate_cells() -> None:
    panel = _panel_with_two_candidates()
    rows = pd.DataFrame(
        [
            {
                "outer_fold": panel.outer_fold,
                "candidate_id": candidate.candidate_id,
                "model": model,
                "replicate": replicate,
            }
            for candidate in panel.candidates
            for model in panel.test_model_ids
            for replicate in (0, 1)
        ]
    )
    phase4.assert_balanced_panel_support(
        rows, panels=[panel], index_column="replicate", index_values=[0, 1]
    )
    with pytest.raises(phase4.Phase4Error, match="balanced"):
        phase4.assert_balanced_panel_support(
            pd.concat([rows.iloc[:-1], rows.iloc[[0]]], ignore_index=True),
            panels=[panel],
            index_column="replicate",
            index_values=[0, 1],
        )


def test_policy_uses_exact_length_then_p90_total_se_then_trace_and_stays_blocked() -> None:
    runner = object.__new__(phase4.Phase4Runner)
    panel = _panel_with_two_candidates()
    outer_rows = pd.DataFrame(
        [
            {
                "outer_fold": 0,
                "candidate_id": candidate.candidate_id,
                "model": model,
                "theta_reference": 0.0 if model == "test-a" else 1.0,
                "theta_cat_mwle": 0.0 if model == "test-a" else 1.0,
                "cat_scenarios_administered": 8,
            }
            for candidate in panel.candidates
            for model in panel.test_model_ids
        ]
    )
    runner.prereq = SimpleNamespace(
        panels=(panel,),
        outer_rows=outer_rows,
        matrix=pd.DataFrame(index=list(panel.test_model_ids)),
        structure=SimpleNamespace(n_dims=1),
    )
    dopt_id = panel.candidates[1].candidate_id
    trace_id = panel.candidates[0].candidate_id
    pareto = pd.DataFrame(
        [
            {
                "candidate_id": dopt_id,
                "outer_fold": 0,
                "phase3_inner_p90_scenario_count": 10.0,
                "phase3_inner_mean_scenario_count": 8.0,
                "p90_se_total": 0.4,
                "selector": "dopt",
                "valid_bootstrap_gate_all_models": True,
                "order_path_gate": True,
            },
            {
                "candidate_id": trace_id,
                "outer_fold": 0,
                "phase3_inner_p90_scenario_count": 10.0,
                "phase3_inner_mean_scenario_count": 8.0,
                "p90_se_total": 0.4,
                "selector": "trace",
                "valid_bootstrap_gate_all_models": True,
                "order_path_gate": True,
            },
        ]
    )
    total_se = pd.DataFrame(
        [
            {
                "outer_fold": 0,
                "candidate_id": candidate.candidate_id,
                "model": model,
                "se_total": 0.4,
                "valid_draw_gate": True,
            }
            for candidate in panel.candidates
            for model in panel.test_model_ids
        ]
    )
    order = pd.DataFrame(
        [
            {
                "outer_fold": 0,
                "candidate_id": candidate.candidate_id,
                "model": model,
                "order_path_sd": 0.1,
                "order_path_gate": True,
            }
            for candidate in panel.candidates
            for model in panel.test_model_ids
        ]
    )

    policy = phase4.Phase4Runner._policy_payload(runner, pareto, total_se, order)

    assert policy["provisional_fold_choices_not_frozen"][0][
        "provisional_candidate_id"
    ] == trace_id
    assert policy["status"] == "blocked_missing_preregistered_total_se_tolerance"
    assert policy["total_uncertainty_tiebreak_metric"] == "p90_se_total"
    assert policy["distinct_total_se_tolerance"] is None
    assert policy["final_policy_freeze_allowed"] is False
    assert policy["final_policy"] is None
    assert policy["provisional_choice_uses_outer_outcomes"] is True
    assert policy["not_an_unbiased_nested_performance_estimate"] is True


def test_failed_order_seed_is_retained_in_order_gate() -> None:
    candidate = phase4.nested.Candidate(5, 0.25, "trace")
    panel = phase4.FoldPanel(
        outer_fold=0,
        training_model_ids=("train",),
        test_model_ids=("test",),
        selected_ridge=0.01,
        candidates=(candidate,),
        fit_cache_key="fit",
    )
    runner = object.__new__(phase4.Phase4Runner)
    runner.prereq = SimpleNamespace(
        panels=(panel,), structure=SimpleNamespace(labels=("ability",))
    )
    runner.order_seeds = tuple(range(20))
    rows = []
    for seed in runner.order_seeds:
        ok = seed != 7
        rows.append(
            {
                "outer_fold": 0,
                "candidate_id": candidate.candidate_id,
                "model": "test",
                "seed": seed,
                "status": "ok" if ok else "replay_numerical_error",
                "scenarios_administered": 10 if ok else np.nan,
                "scenario_order": "[]",
                "mwle_converged": ok,
                "theta_mwle_ability": float(seed) / 100 if ok else np.nan,
            }
        )

    result = phase4.Phase4Runner._order_frame(runner, pd.DataFrame(rows)).iloc[0]

    assert result["n_seed_runs"] == 20
    assert result["mwle_valid_runs"] == 19
    assert bool(result["all_seed_attempts_present"])
    assert not bool(result["all_seed_runs_valid"])
    assert not bool(result["order_path_gate"])


def test_checkpoint_rows_require_exact_unique_panel_and_provenance() -> None:
    panel = _panel_with_two_candidates()
    rows = [
        {
            "outer_fold": 0,
            "replicate": 0,
            "candidate_id": candidate.candidate_id,
            "model": model,
            "selected_ridge": panel.selected_ridge,
            "minimum_scenarios": candidate.minimum_scenarios,
            "conditional_se_target": candidate.conditional_se_target,
            "selector": candidate.selector,
            "status": "ok",
            "mwle_converged": True,
        }
        for candidate in panel.candidates
        for model in panel.test_model_ids
    ]
    phase4.Phase4Runner._validate_checkpoint_rows(
        rows,
        path=Path("checkpoint.json"),
        kind="parameter_bootstrap_cat_path",
        panel=panel,
        index_name="replicate",
        index_value=0,
    )
    with pytest.raises(phase4.Phase4Error, match="exactly one row"):
        phase4.Phase4Runner._validate_checkpoint_rows(
            [*rows[:-1], rows[0]],
            path=Path("checkpoint.json"),
            kind="parameter_bootstrap_cat_path",
            panel=panel,
            index_name="replicate",
            index_value=0,
        )


def test_preflight_rejects_max_grid_nodes_below_locked_grid() -> None:
    runner = object.__new__(phase4.Phase4Runner)
    runner.args = SimpleNamespace(max_grid_nodes=400)
    runner.prereq = SimpleNamespace(
        effective_node_count=401,
        eap_grid=401,
        panels=(),
    )
    runner._base_banks = {}

    with pytest.raises(phase4.Phase4Error, match="below the locked"):
        phase4.Phase4Runner._preflight_outer_caches_and_grid(runner)


def test_completed_output_and_intermediate_hashes_are_reverified(tmp_path: Path) -> None:
    runner = object.__new__(phase4.Phase4Runner)
    runner.prereq = SimpleNamespace(nested_dir=tmp_path)
    outputs = {}
    for name in phase4.FINAL_OUTPUTS:
        path = tmp_path / name
        path.write_text(f"{name}\n", encoding="utf-8")
        if name != "phase4_manifest.json":
            outputs[name] = {"sha256": phase4._sha256(path)}
    for directory in ("phase4_checkpoints", "phase4_cache"):
        path = tmp_path / directory
        path.mkdir()
        (path / "artifact.json").write_text("{}\n", encoding="utf-8")
    inventories = {}
    for directory in ("phase4_checkpoints", "phase4_cache"):
        inventory = phase4._tree_inventory(tmp_path / directory)
        inventories[directory] = {
            "file_count": len(inventory),
            "inventory_sha256": phase4._canonical_hash(inventory),
        }
    manifest = {
        "outputs": outputs,
        "completed_artifact_inventories": inventories,
    }

    phase4.Phase4Runner._verify_completed_outputs(runner, manifest)
    (tmp_path / "phase4_summary.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(phase4.Phase4Error, match="hash mismatch"):
        phase4.Phase4Runner._verify_completed_outputs(runner, manifest)


def test_plan_only_reports_missing_phase3_as_blocker_without_writing(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    code = phase4.main(["--nested-dir", str(tmp_path), "--plan-only"])
    assert code == 2
    assert list(tmp_path.iterdir()) == before


def test_plan_only_validates_explicit_completed_nested_fixture_without_fitting(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "infobench_calibration_remediation_v1.json"
    split_path = root / "configs" / "infobench_remediation_splits_v1.manifest.json"
    matrix_path = (
        root / "runs" / "calibration" / "InFoBench_full_20260804" / "inputs" / "response_matrix.csv"
    )
    rubrics_path = root / "data" / "InFoBench" / "rubrics.jsonl"
    scenarios_path = root / "data" / "InFoBench" / "scenarios.jsonl"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    candidate = phase4.nested.Candidate(5, 0.25, "trace")

    folds_payload = {
        "schema_version": phase4.NESTED_SCHEMA,
        "leakage_audit": "passed",
        "outer_folds": split["outer_folds"],
        "scenario_split": split["scenario_split"],
    }
    finalist_folds = []
    inner_rows = []
    outer_rows = []
    for outer in split["outer_folds"]:
        fold = int(outer["outer_fold"])
        finalist_folds.append(
            {
                "outer_fold": fold,
                "selected_ridge": 0.01,
                "ridge_selected_before_outer_scoring": True,
                "shortlist_status": "finalists_ready",
                "outer_outcomes_available_at_shortlisting": False,
                "configured_selection_order": [
                    "lowest_p90_scenario_count",
                    "lowest_mean_scenario_count",
                    "lowest_total_uncertainty",
                    "prefer_trace",
                ],
                "applied_in_phase3": [
                    "absolute_gates",
                    "lowest_p90_scenario_count",
                    "lowest_mean_scenario_count",
                ],
                "deferred_to_phase4": ["lowest_total_uncertainty", "prefer_trace"],
                "phase3_declares_winner": False,
                "final_policy_freeze_authorized": False,
                "ridge_evidence_sha256": "b" * 64,
                "inner_metrics_sha256": "c" * 64,
                "n_finalists": 1,
                "finalist_candidate_ids": [candidate.candidate_id],
                "outer_evaluation_candidate_ids": [candidate.candidate_id],
                "outer_evaluation_support": (
                    "fold_local_finalists_on_corresponding_outer_test_models_only"
                ),
                "outer_training_fit_cache_key": f"fit-{fold}",
                "finalists": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "minimum_scenarios": candidate.minimum_scenarios,
                        "conditional_se_target": candidate.conditional_se_target,
                        "selector": candidate.selector,
                        "p90_scenario_count": 10.0,
                        "mean_scenario_count": 10.0,
                        "all_gates_pass": True,
                        "phase3_length_filter_status": "retained_through_p90_and_mean",
                        "deferred_selection_fields": [
                            "lowest_total_uncertainty",
                            "prefer_trace",
                        ],
                        "outer_training_fit_cache_key": f"fit-{fold}",
                    }
                ],
            }
        )
        inner_rows.append(
            {
                "outer_fold": fold,
                "candidate_id": candidate.candidate_id,
                "minimum_scenarios": candidate.minimum_scenarios,
                "conditional_se_target": candidate.conditional_se_target,
                    "selector": candidate.selector,
                    "all_gates_pass": True,
                    "p90_scenario_count": 10.0,
                    "mean_scenario_count": 10.0,
            }
        )
        outer_rows.extend(
            {
                "outer_fold": fold,
                "model": model,
                "candidate_id": candidate.candidate_id,
                "selected_ridge": 0.01,
                "fit_cache_key": f"fit-{fold}",
                "cat_scenarios_administered": 10,
                "theta_reference": 0.0,
                "theta_cat_mwle": 0.0,
            }
            for model in outer["test_model_ids"]
        )

    fold_path = tmp_path / "fold_assignments.json"
    finalists_path = tmp_path / "outer_fold_finalists.json"
    inner_path = tmp_path / "inner_candidate_results.csv"
    outer_path = tmp_path / "outer_oof_per_model.csv"
    fold_path.write_text(json.dumps(folds_payload), encoding="utf-8")
    finalists_path.write_text(
        json.dumps(
            {
                "schema_version": phase4.NESTED_SCHEMA,
                "artifact_schema_version": phase4.FINALIST_SCHEMA,
                "selection_source": "inner-fold metrics only",
                "outer_outcomes_used_for_shortlisting": False,
                "allow_fallback_if_none_pass": False,
                "configured_selection_order": [
                    "lowest_p90_scenario_count",
                    "lowest_mean_scenario_count",
                    "lowest_total_uncertainty",
                    "prefer_trace",
                ],
                "applied_in_phase3": [
                    "absolute_gates",
                    "lowest_p90_scenario_count",
                    "lowest_mean_scenario_count",
                ],
                "deferred_to_phase4": ["lowest_total_uncertainty", "prefer_trace"],
                "length_tie_policy": "exact_numeric_equality",
                "phase3_declares_winner": False,
                "final_policy_freeze_authorized": False,
                "outer_evaluation_support": (
                    "each fold-local finalist set on only that fold's outer-test models"
                ),
                "cross_fold_candidate_id_comparison_authorized": False,
                "folds": finalist_folds,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(inner_rows).to_csv(inner_path, index=False)
    pd.DataFrame(outer_rows).to_csv(outer_path, index=False)
    manifest = {
        "schema_version": phase4.NESTED_SCHEMA,
        "status": "phase3_finalists_complete",
        "study_signature": "a" * 64,
        "cross_validation": {"outer_test_used_for_selection": False},
        "phase3_finalist_handoff": {
            "artifact": "outer_fold_finalists.json",
            "artifact_schema_version": phase4.FINALIST_SCHEMA,
            "configured_selection_order": [
                "lowest_p90_scenario_count",
                "lowest_mean_scenario_count",
                "lowest_total_uncertainty",
                "prefer_trace",
            ],
            "applied_in_phase3": [
                "absolute_gates",
                "lowest_p90_scenario_count",
                "lowest_mean_scenario_count",
            ],
            "deferred_to_phase4": ["lowest_total_uncertainty", "prefer_trace"],
            "length_tie_policy": "exact_numeric_equality",
            "outer_evaluation_panel": (
                "each fold-local exact-tie finalist set, evaluated only on that "
                "fold's outer-test models"
            ),
            "cross_fold_candidate_id_comparison_authorized": False,
            "phase3_declares_winner": False,
            "final_policy_freeze_authorized": False,
        },
        "ridge_selection": {"headline_eligible": True},
        "dense_grid_lock": {"fit_grid": 25, "eap_grid": 41},
        "runtime": {
            "seed": 20260729,
            "top_n": 5,
            "maximum_adaptive_scenarios": 50,
            "minimum_scored_criteria": 15,
            "max_iter": 200,
            "tol": 1e-4,
            "mwle_ridge": 1e-6,
            "negative_policy": "drop",
            "estimate_latent_corr": True,
            "allow_unconverged_fit": False,
        },
        "structure": {
            "name": "infobench_overall_1d",
            "source_skills": ["content", "format", "number", "style", "linguistic"],
            "dimensions": [
                {
                    "label": "instruction_following",
                    "members": ["content", "format", "number", "style", "linguistic"],
                }
            ],
        },
        "inputs": {
            name: {"path": str(path), "sha256": phase4._sha256(path)}
            for name, path in (
                ("config", config_path),
                ("response_matrix", matrix_path),
                ("rubrics", rubrics_path),
                ("scenarios", scenarios_path),
            )
        },
        "outputs": {
            path.name: {"sha256": phase4._sha256(path)}
            for path in (fold_path, finalists_path, inner_path, outer_path)
        },
    }
    code_hashes = phase4.nested._code_dependency_hashes()
    dependency_versions = phase4.nested._dependency_versions()
    manifest["code_provenance"] = {
        "files": code_hashes,
        "canonical_sha256": phase4._canonical_hash(code_hashes),
    }
    manifest["environment"] = {
        **dependency_versions,
        "canonical_dependency_versions_sha256": phase4._canonical_hash(
            dependency_versions
        ),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        phase4.cell_cv,
        "fit_structure",
        lambda *args, **kwargs: pytest.fail("plan-only attempted an item fit"),
    )
    preflighted: list[int] = []

    def fake_outer_cache(self, panel):
        preflighted.append(panel.outer_fold)
        return SimpleNamespace(), SimpleNamespace(grid=np.zeros((41, 1)))

    monkeypatch.setattr(phase4.Phase4Runner, "_load_outer_base_bank", fake_outer_cache)
    before = {path.name: phase4._sha256(path) for path in tmp_path.iterdir()}
    code = phase4.main(["--nested-dir", str(tmp_path), "--plan-only", "--n-boot", "2"])
    after = {path.name: phase4._sha256(path) for path in tmp_path.iterdir()}

    assert code == 0
    assert before == after
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "plan_validated"
    assert payload["outer_test_in_fit"] is False
    assert payload["estimated_item_fits"] == 10
    assert len(payload["order_seeds"]) == 20
    assert preflighted == [0, 1, 2, 3, 4]
    assert payload["outer_fit_cache_preflight"]["status"] == "passed"


def test_fresh_removes_only_phase4_owned_paths(tmp_path: Path) -> None:
    historical = tmp_path / "outer_oof_per_model.csv"
    historical.write_text("historical\n", encoding="utf-8")
    for name in phase4.FINAL_OUTPUTS:
        (tmp_path / name).write_text("phase4\n", encoding="utf-8")
    cache = tmp_path / "phase4_cache"
    cache.mkdir()
    (cache / "cached").write_text("x", encoding="utf-8")

    runner = object.__new__(phase4.Phase4Runner)
    runner.args = SimpleNamespace(fresh=True, resume=False)
    runner.prereq = SimpleNamespace(nested_dir=tmp_path)
    runner.phase4_signature = "signature"

    # _prepare_output removes owned files first, then sees a clean Phase-4 scope.
    phase4.Phase4Runner._prepare_output(runner)
    assert historical.read_text(encoding="utf-8") == "historical\n"
    assert not cache.exists()
    assert all(not (tmp_path / name).exists() for name in phase4.FINAL_OUTPUTS)
