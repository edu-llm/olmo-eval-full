from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import nested_scenario_cat_cv as nested

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "infobench_calibration_remediation_v1.json"
SPLITS = ROOT / "configs" / "infobench_remediation_splits_v1.manifest.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _split() -> dict:
    return json.loads(SPLITS.read_text(encoding="utf-8"))


def _tiny_bank() -> tuple[nested.scat.FittedBank, dict[str, dict], pd.Series, set[str], set[str]]:
    records = []
    scenarios: dict[str, dict] = {}
    criterion_ids = []
    scenario_ids = []
    for index in range(8):
        scenario_id = f"s{index}"
        criterion_id = f"{scenario_id}_c01"
        records.append(
            {
                "criterion_id": criterion_id,
                "scenario_id": scenario_id,
                "criterion": f"criterion {index}",
                "q_mapping": {"instruction_following": 1},
                "q_modeled": {"instruction_following": 1},
                "difficulty": -0.7 + 0.2 * index,
                "discrimination": {"instruction_following": 1.0 + 0.05 * index},
                "irt_params": {
                    "source": "calibrated-test",
                    "calibrated": True,
                    "skills_order": ["instruction_following"],
                },
            }
        )
        scenarios[scenario_id] = {
            "scenario_id": scenario_id,
            "prompt": scenario_id,
            "criterion_ids": [criterion_id],
            "modality": "text",
        }
        criterion_ids.append(criterion_id)
        scenario_ids.append(scenario_id)
    q = np.ones((8, 1), dtype=int)
    a = np.asarray([[record["discrimination"]["instruction_following"]] for record in records])
    b = np.asarray([record["difficulty"] for record in records])
    bank = nested.scat.FittedBank(
        records=records,
        dims=("instruction_following",),
        criterion_ids=tuple(criterion_ids),
        scenario_ids=tuple(scenario_ids),
        Q=q,
        A=a,
        b=b,
        latent_correlation=np.eye(1),
        source_path="synthetic-test",
    )
    row = pd.Series(
        [1, 0, 1, 0, 0, 0, 0, 0], index=criterion_ids, name="model-a", dtype=float
    )
    return bank, scenarios, row, set(scenario_ids[:4]), set(scenario_ids[4:])


def test_full_factorial_contains_exactly_48_deterministic_candidates() -> None:
    first = nested.load_candidates(_config())
    second = nested.load_candidates(_config())
    assert first == second
    assert len(first) == 48
    assert len({candidate.candidate_id for candidate in first}) == 48
    assert {
        candidate.minimum_scenarios for candidate in first
    } == {0, 5, 8, 12, 15, 20}
    assert {candidate.selector for candidate in first} == {"trace", "dopt"}


def test_frozen_manifest_passes_and_leakage_contamination_fails_closed() -> None:
    split = _split()
    audit = nested.audit_split_manifest(split)
    assert len(audit["outer_folds"]) == 5
    assert all(len(outer["inner_folds"]) == 4 for outer in audit["outer_folds"])
    assert len(audit["administration_scenario_ids"]) == 400
    assert len(audit["evaluation_scenario_ids"]) == 100

    contaminated = copy.deepcopy(split)
    leaked = contaminated["outer_folds"][0]["test_model_ids"][0]
    contaminated["outer_folds"][0]["inner_folds"][0]["fit_model_ids"].append(leaked)
    with pytest.raises(nested.NestedCVError, match="leaked"):
        nested.audit_split_manifest(contaminated)

    broken_family = copy.deepcopy(split)
    family, members = next(
        (family, models)
        for family, models in broken_family["model_families"].items()
        if len(models) > 1
    )
    moved = members[0]
    original_fold = broken_family["model_to_outer_fold"][moved]
    target_fold = (original_fold + 1) % 5
    broken_family["outer_folds"][original_fold]["test_model_ids"].remove(moved)
    broken_family["outer_folds"][original_fold]["train_model_ids"].append(moved)
    broken_family["outer_folds"][target_fold]["train_model_ids"].remove(moved)
    broken_family["outer_folds"][target_fold]["test_model_ids"].append(moved)
    with pytest.raises(nested.NestedCVError):
        nested.audit_split_manifest(broken_family)


def test_evaluation_outcomes_cannot_change_cat_or_random_path() -> None:
    bank, scenarios, row, administration, evaluation = _tiny_bank()
    administration_bank = nested.subset_fitted_bank(bank, administration)
    evaluation_indices = nested.evaluation_item_indices(bank, evaluation)
    quadrature = nested.scat.build_quadrature(1, 9, np.eye(1))
    candidate = nested.Candidate(0, 0.01, "trace")

    common = dict(
        model="model-a",
        full_bank=bank,
        administration_bank=administration_bank,
        evaluation_indices=evaluation_indices,
        scenario_records=scenarios,
        quadrature=quadrature,
        candidate=candidate,
        seed=17,
        top_n=2,
        maximum_scenarios=4,
        minimum_scored_criteria=0,
        mwle_ridge=1e-6,
    )
    first = nested.evaluate_model_pair(row=row, **common)
    changed = row.copy()
    changed.loc[[bank.criterion_ids[index] for index in evaluation_indices]] = 1.0
    second = nested.evaluate_model_pair(row=changed, **common)

    assert first["cat_scenario_order"] == second["cat_scenario_order"]
    assert first["baseline_scenario_order"] == second["baseline_scenario_order"]
    assert first["cat_scenarios_administered"] == second["cat_scenarios_administered"]
    assert first["theta_reference"] == pytest.approx(second["theta_reference"])
    assert set(json.loads(first["cat_scenario_order"])) <= administration
    assert set(json.loads(first["baseline_scenario_order"])) <= administration
    # The held-out outcomes are opened only afterward, so their scoring statistics
    # are allowed (and expected) to change.
    assert first["cat_eval_observed_pass_rate"] != second["cat_eval_observed_pass_rate"]


def test_prediction_scoring_preserves_missing_cells() -> None:
    bank, _scenarios, row, _administration, evaluation = _tiny_bank()
    indices = nested.evaluation_item_indices(bank, evaluation)
    row.loc[bank.criterion_ids[indices[0]]] = np.nan
    stats = nested.prediction_sufficient_statistics(row, bank, indices, np.array([0.0]))
    assert stats["n_cells"] == len(indices) - 1


def _passing_metrics() -> dict:
    return {
        "replay_success_rate": 1.0,
        "mwle_convergence_rate": 1.0,
        "nominal_precision_rate": 1.0,
        "nominal_precision_lower_95_ci": 0.95,
        "recovery_correlation_lower_95_ci": 0.90,
        "recovery_slope": 1.0,
        "disjoint_pass_rate_mae": 0.03,
        "disjoint_pass_rate_bias": 0.01,
        "scenario_reduction_vs_random": 0.60,
        "paired_ci_favors_cat": True,
    }


def test_absolute_gates_have_no_fallback_and_phase3_retains_exact_length_ties() -> None:
    gates = _config()["selection_gates"]
    passed, failures, outcomes = nested.apply_absolute_gates(_passing_metrics(), gates)
    assert passed and not failures and all(outcomes.values())

    failed = _passing_metrics()
    failed["disjoint_pass_rate_mae"] = 0.5
    passed, failures, _ = nested.apply_absolute_gates(failed, gates)
    assert not passed
    assert failures == ["disjoint_pass_rate_mae"]

    shortlist, stage = nested.shortlist_candidates(
        [{"candidate_id": "x", "all_gates_pass": False}],
        selection_order=nested.EXPECTED_SELECTION_ORDER,
    )
    assert shortlist == []
    assert stage["n_candidates_passing_all_absolute_gates"] == 0

    tied = [
        {
            "candidate_id": "dopt",
            "selector": "dopt",
            "p90_scenario_count": 20.0,
            "mean_scenario_count": 15.0,
            "all_gates_pass": True,
        },
        {
            "candidate_id": "trace",
            "selector": "trace",
            "p90_scenario_count": 20.0,
            "mean_scenario_count": 15.0,
            "all_gates_pass": True,
        },
    ]
    shortlist, stage = nested.shortlist_candidates(
        list(reversed(tied)), selection_order=nested.EXPECTED_SELECTION_ORDER
    )
    assert [row["candidate_id"] for row in shortlist] == ["dopt", "trace"]
    assert stage["n_after_p90_length_filter"] == 2
    assert stage["n_after_mean_length_filter"] == 2
    assert all(
        row["deferred_selection_fields"]
        == ["lowest_total_uncertainty", "prefer_trace"]
        for row in shortlist
    )


def test_phase3_shortlist_applies_p90_then_mean_and_requires_exact_ties() -> None:
    rows = [
        {
            "candidate_id": "best-mean",
            "selector": "dopt",
            "p90_scenario_count": 10.0,
            "mean_scenario_count": 7.0,
            "all_gates_pass": True,
        },
        {
            "candidate_id": "same-exact",
            "selector": "trace",
            "p90_scenario_count": 10.0,
            "mean_scenario_count": 7.0,
            "all_gates_pass": True,
        },
        {
            "candidate_id": "near-not-exact",
            "selector": "trace",
            "p90_scenario_count": 10.0,
            "mean_scenario_count": 7.0 + 1e-13,
            "all_gates_pass": True,
        },
        {
            "candidate_id": "lower-mean-but-worse-p90",
            "selector": "trace",
            "p90_scenario_count": 11.0,
            "mean_scenario_count": 1.0,
            "all_gates_pass": True,
        },
    ]
    shortlist, stage = nested.shortlist_candidates(
        rows, selection_order=nested.EXPECTED_SELECTION_ORDER
    )
    assert [row["candidate_id"] for row in shortlist] == [
        "best-mean",
        "same-exact",
    ]
    assert stage == {
        "n_candidates_passing_all_absolute_gates": 4,
        "minimum_eligible_p90_scenario_count": 10.0,
        "n_after_p90_length_filter": 3,
        "minimum_p90_tied_mean_scenario_count": 7.0,
        "n_after_mean_length_filter": 2,
        "length_tie_policy": "exact_numeric_equality",
    }


def test_phase3_shortlist_fails_closed_if_selection_order_changes() -> None:
    with pytest.raises(nested.NestedCVError, match="selection_order"):
        nested.shortlist_candidates(
            [], selection_order=["lowest_p90_scenario_count", "prefer_trace"]
        )


def test_each_selected_inner_fit_is_reused_for_all_48_candidates(tmp_path: Path) -> None:
    candidates = nested.load_candidates(_config())
    runner = object.__new__(nested.NestedCVRunner)
    runner.args = SimpleNamespace(resume=False)
    runner.output_dir = tmp_path
    runner.study_signature = "study"
    runner.code_hashes = {"runner": "hash"}
    runner.dependency_versions = {"python": "test"}
    runner.candidates = candidates

    panel_candidate_counts: list[int] = []

    def fake_panel(*, bundle, model_ids, candidates, outer_fold, inner_fold):
        panel_candidate_counts.append(len(candidates))
        return [
            {
                "candidate_id": candidate.candidate_id,
                "model": model,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "fit_cache_key": bundle.cache_key,
            }
            for candidate in candidates
            for model in model_ids
        ]

    runner._evaluate_panel = fake_panel
    outer = {
        "outer_fold": 0,
        "test_model_ids": ["held-out"],
        "inner_folds": [
            {
                "inner_fold": index,
                "fit_model_ids": [f"fit-{index}"],
                "validation_model_ids": [f"val-{index}"],
            }
            for index in range(4)
        ],
    }
    bundles = {
        (inner_fold, 0.01): SimpleNamespace(cache_key=f"fit-{inner_fold + 1}")
        for inner_fold in range(4)
    }
    rows = nested.NestedCVRunner._inner_rows_for_outer(
        runner, outer, selected_ridge=0.01, bundles=bundles
    )
    assert panel_candidate_counts == [48, 48, 48, 48]
    assert len(rows) == 4 * 48
    for inner_fold in range(4):
        keys = {
            row["fit_cache_key"] for row in rows if row["inner_fold"] == inner_fold
        }
        assert keys == {f"fit-{inner_fold + 1}"}


def test_plan_only_validates_real_inputs_without_fitting(tmp_path: Path, monkeypatch) -> None:
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("plan-only attempted an item fit")

    monkeypatch.setattr(nested.cell_cv, "fit_structure", forbidden_fit)
    code = nested.main(
        [
            "--fit-grid",
            "25",
            "--eap-grid",
            "41",
            "--plan-only",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert {path.name for path in tmp_path.iterdir()} == {
        "fold_assignments.json",
        "manifest.json",
    }
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "plan_only"
    assert manifest["schema_version"] == "infobench-nested-scenario-cat-cv-v3"
    assert manifest["candidate_count"] == 48
    assert manifest["cross_validation"]["outer_test_used_for_selection"] is False
    assert manifest["runtime"]["metric_bootstrap_replicates"] == 2000
    assert manifest["runtime"]["estimate_latent_corr"] is True
    assert manifest["runtime"]["max_grid_nodes"] == 50_000
    assert manifest["runtime"]["allow_unconverged_fit"] is False
    assert manifest["runtime"]["require_complete_bank"] is False
    assert "scripts/nested_scenario_cat_cv.py" in manifest["code_provenance"]["files"]
    assert "tutor_cat/mcq_irt/calibrate.py" in manifest["code_provenance"]["files"]
    assert set(manifest["environment"]) >= {
        "python",
        "numpy",
        "pandas",
        "scipy",
        "canonical_dependency_versions_sha256",
    }
    assert manifest["phase3_finalist_handoff"] == {
        "artifact": "outer_fold_finalists.json",
        "artifact_schema_version": nested.FINALIST_SCHEMA,
        "configured_selection_order": list(nested.EXPECTED_SELECTION_ORDER),
        "applied_in_phase3": [
            "absolute_gates",
            "lowest_p90_scenario_count",
            "lowest_mean_scenario_count",
        ],
        "deferred_to_phase4": ["lowest_total_uncertainty", "prefer_trace"],
        "length_tie_policy": "exact_numeric_equality",
        "outer_evaluation_panel": (
            "each fold-local exact-tie finalist set, evaluated only on that fold's "
            "outer-test models"
        ),
        "cross_fold_candidate_id_comparison_authorized": False,
        "phase3_declares_winner": False,
        "final_policy_freeze_authorized": False,
    }
    assert set(manifest["deferred_outputs"]) == {
        "outer_total_se.csv",
        "outer_order_stability.csv",
    }


def test_dense_settings_must_be_explicit_or_pre_nested_locked(tmp_path: Path) -> None:
    parser = nested.build_argparser()
    with pytest.raises(nested.NestedCVError, match="pass --fit-grid"):
        nested.resolve_dense_settings(parser.parse_args([]))

    bad = tmp_path / "dense.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": "infobench-dense-grid-lock-v1",
                "status": "locked",
                "selection_scope": "phase2_dense_grid",
                "uses_outer_results": True,
                "selected": {"fit_grid": 25, "eap_grid": 41, "ridge": 0.01},
            }
        ),
        encoding="utf-8",
    )
    args = parser.parse_args(["--dense-grid-manifest", str(bad)])
    with pytest.raises(nested.NestedCVError, match="legacy dense-grid locks"):
        nested.resolve_dense_settings(args)

    dense_dir = tmp_path / "native" / "dense_grid"
    dense_dir.mkdir(parents=True)
    native = dense_dir / "grid_lock.json"
    locked_quadrature = nested.scat.build_quadrature(
        1,
        81,
        np.eye(1),
        method="normal_trapezoid",
        linear_bound=8.0,
    )
    lock = {
        "fit_grid": 41,
        "eap_grid": 81,
        "quadrature_method": "normal_trapezoid",
        "linear_bound": 8.0,
        "quadrature_axis_sha256": nested._array_sha256(
            locked_quadrature.grid[:, 0]
        ),
        "quadrature_log_prior_sha256": nested._array_sha256(
            locked_quadrature.log_prior
        ),
        "effective_node_count": len(locked_quadrature.grid),
        "initial_ridge": 0.1,
        "fit_gate": {
            "passed": True,
            "require_every_comparison": True,
            "locked_fit_grid": 41,
            "comparisons": {
                "41_vs_61": {
                    "passed": True,
                    "common_cell_keys_identical_after_explicit_intersection": True,
                    "common_cell_labels_identical": True,
                }
            },
        },
        "eap_gate": {
            "passed": True,
            "locked_eap_grid": 81,
            "comparisons": {"81_vs_161": {"passed": True}},
        },
        "bound_gate": {
            "passed": True,
            "model_keys_identical": True,
            "posterior_tail_mass_passed": True,
        },
        "cross_family_gate": {"passed": True, "model_keys_identical": True},
        "ridge_scheduled_only_after_this_lock": True,
        "ridge_scheduled_only_after_every_numerical_gate": True,
        "production_ridge_selection": "deferred_to_nested_cv",
    }
    native.write_text(json.dumps(lock), encoding="utf-8")
    (tmp_path / "native" / "study_manifest.json").write_text(
        json.dumps(
            {
                "status": "phase2_complete",
                "result": {"fit_grid": 41, "eap_grid": 81, "grid_lock": lock},
            }
        ),
        encoding="utf-8",
    )
    locked = nested.resolve_dense_settings(
        parser.parse_args(["--dense-grid-manifest", str(native)])
    )
    assert locked.fit_grid == 41
    assert locked.eap_grid == 81
    assert locked.quadrature_method == "normal_trapezoid"
    assert locked.linear_bound == 8.0
    assert locked.effective_node_count == 81
    assert locked.manifest_ridge_field_ignored is None
    assert locked.native_lock_verified is True

    broken = dict(lock)
    broken["ridge_scheduled_only_after_every_numerical_gate"] = False
    native.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(nested.NestedCVError, match="every common/EAP/bound"):
        nested.resolve_dense_settings(
            parser.parse_args(["--dense-grid-manifest", str(native)])
        )


def test_study_signature_covers_code_dependencies_and_behavior_flags(
    tmp_path: Path, monkeypatch
) -> None:
    parser = nested.build_argparser()

    def signature(extra: list[str]) -> str:
        args = parser.parse_args(
            [
                "--fit-grid",
                "25",
                "--eap-grid",
                "41",
                "--plan-only",
                "--out-dir",
                str(tmp_path / "unused"),
                *extra,
            ]
        )
        runner = nested.NestedCVRunner(args)
        return runner.study_signature

    try:
        baseline = signature([])
        assert signature(["--metric-bootstrap-replicates", "1000"]) != baseline
        assert signature(["--no-estimate-latent-corr"]) != baseline
        assert signature(["--max-grid-nodes", "60000"]) != baseline
        assert signature(["--allow-unconverged-fit"]) != baseline
        assert signature(["--require-complete-bank"]) != baseline
        code_hashes = nested._code_dependency_hashes()
        first_path = next(iter(code_hashes))
        with monkeypatch.context() as patcher:
            patcher.setattr(
                nested,
                "_code_dependency_hashes",
                lambda: {**code_hashes, first_path: "changed"},
            )
            assert signature([]) != baseline
        versions = nested._dependency_versions()
        with monkeypatch.context() as patcher:
            patcher.setattr(
                nested,
                "_dependency_versions",
                lambda: {**versions, "numpy": "changed"},
            )
            assert signature([]) != baseline
    finally:
        nested.cm.configure_skills(None)


def test_fresh_refuses_to_delete_outside_runs_calibration(tmp_path: Path) -> None:
    target = tmp_path / "unsafe-output"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    runner = object.__new__(nested.NestedCVRunner)
    runner.args = SimpleNamespace(fresh=True, resume=False)
    runner.output_dir_requested = target
    runner.output_dir = target

    with pytest.raises(nested.NestedCVError, match="unsafe --fresh target"):
        nested.NestedCVRunner._prepare_output(runner)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_fresh_allows_only_exact_nonsymlink_leaf(
    tmp_path: Path, monkeypatch
) -> None:
    exact = tmp_path / "study" / "nested_cat_cv"
    exact.mkdir(parents=True)
    (exact / "replace.txt").write_text("replace\n", encoding="utf-8")
    monkeypatch.setattr(nested, "DEFAULT_OUTPUT", exact)
    runner = object.__new__(nested.NestedCVRunner)
    runner.args = SimpleNamespace(fresh=True, resume=False)
    runner.output_dir_requested = exact.absolute()
    runner.output_dir = exact.resolve()

    nested.NestedCVRunner._prepare_output(runner)
    assert exact.is_dir()
    assert list(exact.iterdir()) == []


def test_fresh_rejects_exact_leaf_symlink(tmp_path: Path, monkeypatch) -> None:
    real = tmp_path / "real-output"
    real.mkdir()
    sentinel = real / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    link = tmp_path / "study" / "nested_cat_cv"
    link.parent.mkdir()
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(nested, "DEFAULT_OUTPUT", link)
    runner = object.__new__(nested.NestedCVRunner)
    runner.args = SimpleNamespace(fresh=True, resume=False)
    runner.output_dir_requested = link.absolute()
    runner.output_dir = link.resolve()

    with pytest.raises(nested.NestedCVError, match="unsafe --fresh target"):
        nested.NestedCVRunner._prepare_output(runner)
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_fresh_allows_exact_verified_lock_study_leaf(tmp_path: Path) -> None:
    lock = tmp_path / "native-study" / "dense_grid" / "grid_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text("{}\n", encoding="utf-8")
    target = tmp_path / "native-study" / "nested_cat_cv"
    target.mkdir()
    (target / "replace.txt").write_text("replace\n", encoding="utf-8")
    runner = object.__new__(nested.NestedCVRunner)
    runner.args = SimpleNamespace(
        fresh=True,
        resume=False,
        dense_grid_manifest=lock,
    )
    runner.output_dir_requested = target.absolute()
    runner.output_dir = target.resolve()

    nested.NestedCVRunner._prepare_output(runner)
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_checkpoint_hash_and_exact_support_for_all_stages(tmp_path: Path) -> None:
    runner = object.__new__(nested.NestedCVRunner)
    runner.args = SimpleNamespace(resume=False)
    runner.study_signature = "study-signature"
    runner.code_hashes = {"runner": "abc"}
    runner.dependency_versions = {"python": "test"}
    models = ["model-a", "model-b"]
    fit_key = "fit-key"
    cases = [
        (
            "ridge_validation",
            0,
            1,
            0.01,
            ["ridge_0p01_eap_disjoint"],
            [
                {
                    "outer_fold": 0,
                    "inner_fold": 1,
                    "model": model,
                    "ridge": 0.01,
                    "fit_cache_key": fit_key,
                }
                for model in models
            ],
        ),
        (
            "inner_cat",
            2,
            3,
            None,
            ["candidate-a", "candidate-b"],
            [
                {
                    "outer_fold": 2,
                    "inner_fold": 3,
                    "candidate_id": candidate,
                    "model": model,
                    "fit_cache_key": fit_key,
                }
                for candidate in ("candidate-a", "candidate-b")
                for model in models
            ],
        ),
        (
            "outer_cat",
            4,
            None,
            None,
            ["candidate-a", "candidate-b"],
            [
                {
                    "outer_fold": 4,
                    "inner_fold": None,
                    "candidate_id": candidate,
                    "model": model,
                    "fit_cache_key": fit_key,
                }
                for candidate in ("candidate-a", "candidate-b")
                for model in models
            ],
        ),
    ]
    for stage, outer_fold, inner_fold, ridge, candidate_ids, rows in cases:
        path = tmp_path / f"{stage}.json"
        runner._write_checkpoint(
            path,
            stage=stage,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            ridge=ridge,
            fit_cache_key=fit_key,
            candidate_ids=candidate_ids,
            model_ids=models,
            rows=rows,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == nested.CHECKPOINT_SCHEMA
        assert payload["checkpoint_stage"] == stage
        assert payload["rows_sha256"] == nested._canonical_hash(rows)
        runner.args.resume = True
        assert runner._load_checkpoint(
            path,
            stage=stage,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            ridge=ridge,
            fit_cache_key=fit_key,
            candidate_ids=candidate_ids,
            model_ids=models,
        ) == rows
        runner.args.resume = False

    broken = cases[1][-1][:-1]
    with pytest.raises(nested.NestedCVError, match="candidate×model support"):
        runner._write_checkpoint(
            tmp_path / "broken.json",
            stage="inner_cat",
            outer_fold=2,
            inner_fold=3,
            ridge=None,
            fit_cache_key=fit_key,
            candidate_ids=["candidate-a", "candidate-b"],
            model_ids=models,
            rows=broken,
        )

    tampered_path = tmp_path / "outer_cat.json"
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["rows"][0]["model"] = "tampered"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    runner.args.resume = True
    with pytest.raises(nested.NestedCVError, match="content hash mismatch"):
        runner._load_checkpoint(
            tampered_path,
            stage="outer_cat",
            outer_fold=4,
            inner_fold=None,
            ridge=None,
            fit_cache_key=fit_key,
            candidate_ids=["candidate-a", "candidate-b"],
            model_ids=models,
        )


def test_cached_and_fresh_unconverged_fits_fail_identically(
    tmp_path: Path, monkeypatch
) -> None:
    runner = object.__new__(nested.NestedCVRunner)
    runner.matrix = pd.DataFrame(index=["model-a"])
    runner.output_dir = tmp_path
    runner.dense = SimpleNamespace(fit_grid=3)
    runner.args = SimpleNamespace(
        estimate_latent_corr=False,
        max_iter=10,
        tol=1e-4,
        allow_unconverged_fit=False,
    )
    runner.q_by = {}
    runner.structure = object()
    runner._fit_cache_key = lambda training_model_ids, ridge: "fit-key"
    unconverged = {"converged": False}
    runner._load_fit_cache = lambda *args, **kwargs: unconverged

    with pytest.raises(nested.NestedCVError, match="cached item fit did not converge"):
        runner.fit_or_load(
            role="cached-role",
            training_model_ids=["model-a"],
            forbidden_model_ids=[],
            ridge=0.01,
        )

    runner._load_fit_cache = lambda *args, **kwargs: None
    monkeypatch.setattr(
        nested.cell_cv, "fit_structure", lambda *args, **kwargs: unconverged
    )
    with pytest.raises(nested.NestedCVError, match="fresh item fit did not converge"):
        runner.fit_or_load(
            role="fresh-role",
            training_model_ids=["model-a"],
            forbidden_model_ids=[],
            ridge=0.01,
        )


def test_ridge_one_se_rule_uses_only_evidence_and_prefers_stronger_regularization() -> None:
    evidence = [
        {
            "ridge": 0.001,
            "scoring_coverage": 1.0,
            "mean_model_disjoint_log_loss": 0.40,
            "se_model_disjoint_log_loss": 0.05,
        },
        {
            "ridge": 0.01,
            "scoring_coverage": 1.0,
            "mean_model_disjoint_log_loss": 0.42,
            "se_model_disjoint_log_loss": 0.02,
        },
        {
            "ridge": 0.1,
            "scoring_coverage": 1.0,
            "mean_model_disjoint_log_loss": 0.44,
            "se_model_disjoint_log_loss": 0.02,
        },
    ]
    selected, annotated = nested.select_ridge_one_se(evidence)
    assert selected == pytest.approx(0.1)
    assert sum(bool(row["selected"]) for row in annotated) == 1
    assert all(row["within_one_se"] for row in annotated)


def test_mocked_complete_run_writes_every_phase3_output(
    tmp_path: Path, monkeypatch
) -> None:
    args = nested.build_argparser().parse_args(
        [
            "--fit-grid",
            "25",
            "--eap-grid",
            "41",
            "--debug-allow-explicit-numerical-settings",
            "--metric-bootstrap-replicates",
            "100",
            "--out-dir",
            str(tmp_path),
        ]
    )
    runner = nested.NestedCVRunner(args)
    active_runner = runner
    outer_variant = False
    fit_calls: list[tuple[str, tuple[str, ...], float]] = []
    counters = {"inner_panels": 0, "shortlists": 0, "outer_panels": 0}
    real_shortlist = nested.shortlist_candidates

    def tracked_shortlist(*args, **kwargs):
        counters["shortlists"] += 1
        return real_shortlist(*args, **kwargs)

    monkeypatch.setattr(nested, "shortlist_candidates", tracked_shortlist)

    def fake_fit_or_load(*, role, training_model_ids, forbidden_model_ids, ridge):
        assert set(training_model_ids).isdisjoint(forbidden_model_ids)
        fit_calls.append((role, tuple(sorted(training_model_ids)), float(ridge)))
        return SimpleNamespace(cache_key=nested._canonical_hash(fit_calls[-1]))

    model_rank = {model: index for index, model in enumerate(sorted(runner.matrix.index))}

    def fake_ridge_panel(*, bundle, model_ids, ridge, outer_fold, inner_fold):
        return [
            {
                "model": model,
                "ridge": ridge,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "fit_cache_key": bundle.cache_key,
                "status": "ok",
                "error": "",
                "n_cells": 10,
                "log_loss_sum": 1.0,
                "brier_sum": 0.5,
                "model_log_loss": 0.1,
            }
            for model in model_ids
        ]

    def fake_panel(*, bundle, model_ids, candidates, outer_fold, inner_fold):
        outer = active_runner.split_audit["outer_folds"][outer_fold]
        test_ids = set(map(str, outer["test_model_ids"]))
        if inner_fold is not None:
            assert set(map(str, model_ids)).isdisjoint(test_ids)
            counters["inner_panels"] += 1
        else:
            assert set(map(str, model_ids)) == test_ids
            if counters["outer_panels"] == 0:
                assert counters["inner_panels"] == 20
                assert counters["shortlists"] == 5
            counters["outer_panels"] += 1
        rows = []
        for candidate in candidates:
            candidate_rank = active_runner.candidates.index(candidate)
            cat_length = (
                10
                if inner_fold is None
                or candidate_rank in {2 * outer_fold, 2 * outer_fold + 1}
                else 11
            )
            for model in sorted(model_ids):
                theta = (model_rank[model] - 25.5) / 10.0
                outer_shift = 50.0 if outer_variant and inner_fold is None else 0.0
                rows.append(
                    {
                        "model": model,
                        "candidate_id": candidate.candidate_id,
                        "minimum_scenarios": candidate.minimum_scenarios,
                        "conditional_se_target": candidate.conditional_se_target,
                        "selector": candidate.selector,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "fit_cache_key": bundle.cache_key,
                        "status": "ok",
                        "cat_replay_success": True,
                        "baseline_replay_success": True,
                        "cat_precision_reached": True,
                        "baseline_precision_reached": True,
                        "cat_mwle_converged": True,
                        "baseline_mwle_converged": True,
                        "cat_scenarios_administered": cat_length,
                        "baseline_scenarios_administered": 30,
                        "theta_reference": theta,
                        "theta_cat_mwle": theta + outer_shift,
                        "theta_baseline_mwle": theta,
                        "cat_eval_n_cells": 10,
                        "cat_eval_log_loss_sum": 1.0 + outer_shift,
                        "cat_eval_brier_sum": 0.5,
                        "cat_eval_correct_count": 9,
                        "cat_eval_observed_pass_rate": 0.5,
                        "cat_eval_predicted_pass_rate": 0.5,
                        "baseline_eval_n_cells": 10,
                        "baseline_eval_log_loss_sum": 1.0,
                        "baseline_eval_brier_sum": 0.5,
                        "baseline_eval_correct_count": 9,
                        "baseline_eval_observed_pass_rate": 0.5,
                        "baseline_eval_predicted_pass_rate": 0.5,
                    }
                )
        return rows

    runner.fit_or_load = fake_fit_or_load
    runner._evaluate_ridge_panel = fake_ridge_panel
    runner._evaluate_panel = fake_panel
    try:
        assert runner.run() == 0
    finally:
        nested.cm.configure_skills(None)

    assert set(nested.REQUIRED_OUTPUTS) <= {path.name for path in tmp_path.iterdir()}
    # 20 inner splits x 3 ridges plus five selected-ridge outer-training fits;
    # never one fit per CAT candidate.
    assert len(fit_calls) == 65
    assert len({role for role, _ids, _ridge in fit_calls}) == 65
    assert counters == {"inner_panels": 20, "shortlists": 5, "outer_panels": 5}
    ridge = pd.read_csv(tmp_path / "inner_ridge_results.csv")
    assert len(ridge) == 5 * 3
    assert ridge.groupby("outer_fold")["selected"].sum().eq(1).all()
    inner = pd.read_csv(tmp_path / "inner_candidate_results.csv")
    assert len(inner) == 5 * 48
    assert inner["all_gates_pass"].all()
    finalists = json.loads((tmp_path / "outer_fold_finalists.json").read_text())
    assert finalists["artifact_schema_version"] == nested.FINALIST_SCHEMA
    assert finalists["phase3_declares_winner"] is False
    assert finalists["deferred_to_phase4"] == [
        "lowest_total_uncertainty",
        "prefer_trace",
    ]
    assert len(finalists["folds"]) == 5
    assert all(len(fold["finalist_candidate_ids"]) == 2 for fold in finalists["folds"])
    assert all("selected_candidate_id" not in fold for fold in finalists["folds"])
    assert all(fold["outer_training_fit_cache_key"] for fold in finalists["folds"])
    assert all(
        fold["outer_evaluation_candidate_ids"] == fold["finalist_candidate_ids"]
        for fold in finalists["folds"]
    )
    assert "global_evaluation_candidate_ids" not in finalists
    outer = pd.read_csv(tmp_path / "outer_oof_per_model.csv")
    assert len(outer) == 52 * 2
    for fold in finalists["folds"]:
        fold_rows = outer[outer["outer_fold"] == fold["outer_fold"]]
        assert set(fold_rows["candidate_id"]) == set(fold["finalist_candidate_ids"])
        assert fold_rows.groupby("candidate_id")["model"].nunique().nunique() == 1
    assert not outer.duplicated(["outer_fold", "candidate_id", "model"]).any()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "phase3_finalists_complete"
    assert manifest["shortlist_summary"]["no_fallback_used"] is True
    assert manifest["shortlist_summary"]["phase3_declared_winner"] is False
    assert manifest["shortlist_summary"][
        "cross_fold_candidate_id_comparison_authorized"
    ] is False
    assert set(manifest["deferred_outputs"]) == set(nested.DEFERRED_PHASE4_OUTPUTS)

    first_finalists = (tmp_path / "outer_fold_finalists.json").read_bytes()
    changed_out = tmp_path / "changed_outer_outcomes"
    changed_args = nested.build_argparser().parse_args(
        [
            "--fit-grid",
            "25",
            "--eap-grid",
            "41",
            "--debug-allow-explicit-numerical-settings",
            "--metric-bootstrap-replicates",
            "100",
            "--out-dir",
            str(changed_out),
        ]
    )
    active_runner = nested.NestedCVRunner(changed_args)
    active_runner.fit_or_load = fake_fit_or_load
    active_runner._evaluate_ridge_panel = fake_ridge_panel
    active_runner._evaluate_panel = fake_panel
    outer_variant = True
    counters.update({"inner_panels": 0, "shortlists": 0, "outer_panels": 0})
    try:
        assert active_runner.run() == 0
    finally:
        nested.cm.configure_skills(None)
    assert counters == {"inner_panels": 20, "shortlists": 5, "outer_panels": 5}
    assert (changed_out / "outer_fold_finalists.json").read_bytes() == first_finalists
