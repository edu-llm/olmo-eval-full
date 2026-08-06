from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import nested_scenario_cat_cv_2pl_only_v1 as two_pl  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "infobench_calibration_cat_2pl_only_v1.json"
PARENT_PATH = ROOT / "configs" / "infobench_calibration_cat_v3.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_inner_evidence() -> tuple[
    list[dict], list[two_pl.CalibrationSpec], dict[int, list[str]]
]:
    specs = two_pl.load_calibration_specs(_load(CONFIG_PATH))
    expected = {
        inner_fold: [f"model_{inner_fold}_a", f"model_{inner_fold}_b"]
        for inner_fold in range(two_pl.EXPECTED_INNER_FOLDS)
    }
    rows: list[dict] = []
    for inner_fold, models in expected.items():
        common_hash = two_pl._canonical_hash({"inner_fold": inner_fold})
        for spec in specs:
            for model in models:
                rows.append(
                    {
                        "inner_fold": inner_fold,
                        "spec_id": spec.spec_id,
                        "model": model,
                        "status": "ok",
                        "fit_eligible": True,
                        "all_specs_fitted": True,
                        "common_support_verified": True,
                        "fit_cache_key": f"fit-{inner_fold}-{spec.spec_id}",
                        "common_support_sha256": common_hash,
                        "n_common_admin_items": 10,
                        "n_common_evaluation_items": 5,
                        "n_cells": 5,
                        "log_loss_sum": 1.0,
                        "brier_sum": 0.5,
                        "model_log_loss": 0.2,
                        "observed_common_cells_sha256": two_pl._canonical_hash(
                            {"inner_fold": inner_fold, "model": model}
                        ),
                    }
                )
    return rows, specs, expected


def _complete_panel(repeat: int, outer_fold: int) -> dict:
    return {
        "panel_id": two_pl._panel_id(repeat, outer_fold),
        "repeat": repeat,
        "outer_fold": outer_fold,
        "status": "selection_complete",
        "error": "",
        "selected_spec_id": two_pl.EXPECTED_PHASE3_SPEC_IDS[0],
        "selected_calibration_specification": {},
        "inner_selection": {
            "complete_inner_evidence_audit": {
                "passed": True,
                "expected_spec_fold_combinations": 8,
                "valid_spec_fold_combinations": 8,
            }
        },
    }


def test_config_is_exact_allowlisted_child_of_hash_frozen_parent() -> None:
    config = _load(CONFIG_PATH)
    parent = _load(PARENT_PATH)
    provenance = two_pl._check_frozen_config(config)
    assert two_pl._sha256(PARENT_PATH) == two_pl.PARENT_CONFIG_SHA256
    assert provenance["all_non_allowlisted_fields_equal"] is True
    assert {
        path.split(".", 1)[0].split("[", 1)[0] for path in provenance["observed_diff_paths"]
    } <= set(two_pl.PARENT_ALLOWED_DIFF_ROOTS)
    for section in (
        "baseline",
        "limitations",
        "latent_structure",
        "numerical_lock",
        "cross_validation",
        "cat_policies",
        "selection_gates",
        "uncertainty",
        "runtime",
    ):
        assert config[section] == parent[section]


def test_parent_allowlist_rejects_gate_or_runtime_changes() -> None:
    for section, field in (
        ("selection_gates", "minimum_replay_success_rate"),
        ("runtime", "fit_max_iter"),
        ("cat_policies", "top_n"),
    ):
        config = copy.deepcopy(_load(CONFIG_PATH))
        config[section][field] = 999
        with pytest.raises(two_pl.V3Phase3Error, match="non-allowlisted"):
            two_pl._validate_parent_config_provenance(config)


def test_exact_two_spec_order_ranks_and_canonical_keys() -> None:
    config = _load(CONFIG_PATH)
    specs = two_pl.load_calibration_specs(config)
    assert [spec.spec_id for spec in specs] == list(two_pl.EXPECTED_PHASE3_SPEC_IDS)
    assert [spec.simplicity_rank for spec in specs] == [0, 1]
    assert [spec.canonical["cache_key"] for spec in specs] == [
        config["calibration_specifications"][0]["canonical_cache_key"],
        config["calibration_specifications"][1]["canonical_cache_key"],
    ]
    assert [spec.spec_id for spec in two_pl.expected_v4_calibration_specs()] == list(
        two_pl.EXPECTED_V4_SPEC_IDS
    )


@pytest.mark.parametrize("mutation", ["reverse", "missing", "add_1pl"])
def test_candidate_subset_fails_closed_when_changed(mutation: str) -> None:
    config = copy.deepcopy(_load(CONFIG_PATH))
    if mutation == "reverse":
        config["calibration_specifications"].reverse()
    elif mutation == "missing":
        config["calibration_specifications"].pop()
    else:
        config["calibration_specifications"].insert(
            0, copy.deepcopy(_load(PARENT_PATH)["calibration_specifications"][0])
        )
    with pytest.raises(two_pl.V3Phase3Error):
        two_pl.load_calibration_specs(config)


def test_real_v4_chain_is_full_three_spec_and_authorizes_exact_subset() -> None:
    config = _load(CONFIG_PATH)
    profile = two_pl._validate_numerical_lock(
        config,
        two_pl.load_calibration_specs(config),
    )
    assert profile["status"] == "passed"
    assert profile["passed_spec_ids"] == list(two_pl.EXPECTED_V4_SPEC_IDS)
    assert profile["phase3_candidate_spec_ids"] == list(two_pl.EXPECTED_PHASE3_SPEC_IDS)
    assert profile["phase3_subset_of_v4_exact"] is True
    assert profile["verification_sha256"] == (
        "2bdc6abcc01b0ef97542673ffdd09374577e1eaaf1dd5fea5c05126f6f723994"
    )


@pytest.mark.parametrize("mutation", ["reverse", "missing"])
def test_v4_consumer_rejects_reordered_or_missing_phase3_subset(mutation: str) -> None:
    config = _load(CONFIG_PATH)
    specs = two_pl.load_calibration_specs(config)
    specs = list(reversed(specs)) if mutation == "reverse" else specs[:-1]
    with pytest.raises(two_pl.V3Phase3Error, match="subset or order"):
        two_pl._validate_numerical_lock(config, specs)


def test_v4_consumer_rejects_missing_evidence_key(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _load(CONFIG_PATH)
    original = two_pl._read_json
    lock_path = two_pl.DEFAULT_NUMERICAL_LOCK.resolve()

    def altered(path: Path) -> dict:
        value = original(path)
        if path.resolve() == lock_path:
            value = copy.deepcopy(value)
            key = next(iter(value["evidence_sha256"]))
            value["evidence_sha256"].pop(key)
            value["evidence_paths"].pop(key)
        return value

    monkeypatch.setattr(two_pl, "_read_json", altered)
    with pytest.raises(two_pl.V3Phase3Error, match="evidence panel is incomplete"):
        two_pl._validate_numerical_lock(config, two_pl.load_calibration_specs(config))


def test_v4_consumer_rejects_tampered_evidence_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _load(CONFIG_PATH)
    lock = _load(two_pl.DEFAULT_NUMERICAL_LOCK)
    first_relative = next(iter(lock["evidence_paths"].values()))
    target = two_pl._resolve_repo_path(first_relative).resolve()
    original = two_pl._sha256

    def altered(path: Path) -> str:
        if path.resolve() == target:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(two_pl, "_sha256", altered)
    with pytest.raises(two_pl.V3Phase3Error, match="evidence hash mismatch"):
        two_pl._validate_numerical_lock(config, two_pl.load_calibration_specs(config))


def test_inner_evidence_requires_all_eight_blocks_and_total_is_200() -> None:
    rows, specs, expected = _valid_inner_evidence()
    audit = two_pl.audit_complete_inner_evidence(rows, specs, expected)
    assert audit["passed"] is True
    assert audit["expected_spec_fold_combinations"] == 8
    assert audit["valid_spec_fold_combinations"] == 8
    assert two_pl.EXPECTED_SPEC_FOLD_BLOCKS_PER_PANEL == 8
    assert two_pl.EXPECTED_TOTAL_SPEC_FOLD_BLOCKS == 200


@pytest.mark.parametrize("missing_spec", list(two_pl.EXPECTED_PHASE3_SPEC_IDS))
def test_missing_either_candidate_in_any_inner_fold_blocks_selection(missing_spec: str) -> None:
    rows, specs, expected = _valid_inner_evidence()
    rows = [row for row in rows if not (row["inner_fold"] == 2 and row["spec_id"] == missing_spec)]
    audit = two_pl.audit_complete_inner_evidence(rows, specs, expected)
    assert audit["passed"] is False
    assert audit["valid_spec_fold_combinations"] == 7


def _selection_rows(lambda16_loss: float, lambda4_loss: float, best_se: float) -> list[dict]:
    specs = two_pl.expected_calibration_specs()
    return [
        {
            "spec_id": spec.spec_id,
            "calibration_specification": spec.canonical,
            "mean_log_loss": loss,
            "family_cluster_se": best_se,
            "eligible": True,
        }
        for spec, loss in zip(specs, (lambda16_loss, lambda4_loss), strict=True)
    ]


def test_one_se_prefers_lambda16_when_both_are_within_cutoff() -> None:
    specs = two_pl.expected_calibration_specs()
    selected, _rows, detail = two_pl.select_calibration_spec_one_se(
        _selection_rows(0.405, 0.400, 0.010), specs
    )
    assert selected.spec_id == "log_shrinkage_2pl_lambda16"
    assert len(detail["within_one_se_cache_keys"]) == 2


def test_one_se_selects_lambda4_only_when_lambda16_is_outside_cutoff() -> None:
    specs = two_pl.expected_calibration_specs()
    selected, _rows, detail = two_pl.select_calibration_spec_one_se(
        _selection_rows(0.420, 0.400, 0.010), specs
    )
    assert selected.spec_id == "log_shrinkage_2pl_lambda4"
    assert len(detail["within_one_se_cache_keys"]) == 1


def test_output_isolation_rejects_parent_v3_leaf() -> None:
    config = _load(CONFIG_PATH)
    parent_leaf = ROOT / "runs" / "calibration" / "InFoBench_v3" / "phase3"
    with pytest.raises(two_pl.V3Phase3Error, match="must equal"):
        two_pl._resolve_phase3_output_dir(
            config,
            parent_leaf,
            plan_only=False,
        )
    assert (
        two_pl._resolve_phase3_output_dir(config, None, plan_only=False)
        == (ROOT / "runs" / "calibration" / "InFoBench_2pl_only_v1" / "phase3").resolve()
    )


def test_cache_and_checkpoint_namespaces_are_new() -> None:
    runner = two_pl.V3Phase3Runner.__new__(two_pl.V3Phase3Runner)
    runner.output_dir = ROOT / "runs" / "calibration" / "InFoBench_2pl_only_v1" / "phase3"
    spec = two_pl.expected_calibration_specs()[0]
    assert "two_pl_only_v1_fit_cache" in str(
        runner._cache_dir(repeat=0, outer_fold=0, inner_fold=0, spec=spec)
    )
    assert "two_pl_only_v1_checkpoints" in str(runner._checkpoint_path("inner", 0, 0, "fixture"))
    assert "InFoBench_v3/phase3" not in str(
        runner._cache_dir(repeat=0, outer_fold=0, inner_fold=0, spec=spec)
    )


def test_panel_lock_is_exclusive_and_releases_only_owner(tmp_path: Path) -> None:
    with (
        two_pl._exclusive_panel_lock(
            tmp_path,
            study_signature="study",
            repeat=0,
            outer_fold=0,
        ),
        pytest.raises(two_pl.V3Phase3Error, match="already owned"),
        two_pl._exclusive_panel_lock(
            tmp_path,
            study_signature="study",
            repeat=0,
            outer_fold=0,
        ),
    ):
        pass
    with two_pl._exclusive_panel_lock(
        tmp_path,
        study_signature="study",
        repeat=0,
        outer_fold=0,
    ):
        assert two_pl._panel_lock_path(tmp_path, 0, 0).is_file()
    assert not two_pl._panel_lock_path(tmp_path, 0, 0).exists()


def test_unique_atomic_json_temporaries_do_not_collide(tmp_path: Path) -> None:
    target = tmp_path / "shared.json"

    def write(index: int) -> None:
        two_pl._atomic_json(target, {"index": index, "payload": "x" * 1000})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(40)))
    observed = _load(target)
    assert observed["index"] in range(40)
    assert observed["payload"] == "x" * 1000
    assert list(tmp_path.glob(".shared.json.*.tmp")) == []


def _checkpoint_runner(tmp_path: Path, signature: str = "study-a") -> two_pl.V3Phase3Runner:
    runner = two_pl.V3Phase3Runner.__new__(two_pl.V3Phase3Runner)
    runner.output_dir = tmp_path
    runner.study_signature = signature
    runner.specs = two_pl.expected_calibration_specs()
    runner.config_path = CONFIG_PATH
    runner.split_path = two_pl.DEFAULT_SPLITS
    return runner


def test_official_reuse_rejects_mismatched_panel_study_signature(tmp_path: Path) -> None:
    outer = {"outer_fold": 0, "train_model_ids": ["a"], "test_model_ids": ["b"]}
    writer = _checkpoint_runner(tmp_path, "study-a")
    panel = _complete_panel(0, 0)
    writer._write_panel_selection_checkpoint(
        repeat=0,
        outer=outer,
        panel=panel,
        evidence=[{"spec_id": two_pl.EXPECTED_PHASE3_SPEC_IDS[0]}],
    )
    reader = _checkpoint_runner(tmp_path, "study-b")
    with pytest.raises(two_pl.V3Phase3Error, match="provenance mismatch"):
        reader._load_panel_selection_checkpoint(repeat=0, outer=outer)


def test_precompute_panel_never_calls_outer_evaluation(tmp_path: Path) -> None:
    runner = two_pl.V3Phase3Runner.__new__(two_pl.V3Phase3Runner)
    runner.numerical = {"status": "passed"}
    runner.output_dir = tmp_path / "phase3"
    runner.split_audit = {
        "repetitions": [
            {"repeat": 0, "outer_folds": [{"outer_fold": 0}]},
        ]
    }
    runner._select_or_load_panel = lambda *_args, **_kwargs: (_complete_panel(0, 0), [])
    runner._evaluate_outer_panel = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("precompute must not open outer outcomes")
    )
    assert runner.precompute_panel(0, 0) == 0
    assert not (runner.output_dir / "manifest.json").exists()
    assert not (runner.output_dir / "pre_outer_selection_lock.json").exists()


def test_precompute_mode_is_mutually_exclusive_with_plan_and_resume() -> None:
    parser = two_pl.build_argparser()
    args = parser.parse_args(["--precompute-panel", "0", "0", "--plan-only"])
    with pytest.raises(two_pl.V3Phase3Error, match="cannot combine"):
        two_pl._validate_runtime_args(args)
    with pytest.raises(SystemExit):
        parser.parse_args(["--precompute-panel", "0", "0", "--resume"])


@pytest.mark.parametrize("failed_panel", [None, (0, 0)])
def test_all_25_selections_precede_every_outer_call_and_failure_blocks_outer(
    tmp_path: Path, failed_panel: tuple[int, int] | None
) -> None:
    runner = two_pl.V3Phase3Runner.__new__(two_pl.V3Phase3Runner)
    runner.args = argparse.Namespace(plan_only=False, resume=False)
    runner.numerical = {"status": "passed"}
    runner.output_dir = tmp_path / "phase3"
    runner.study_signature = "barrier-study"
    runner._already_complete = False
    runner.split_audit = {
        "repetitions": [
            {
                "repeat": repeat,
                "outer_folds": [{"outer_fold": fold} for fold in range(5)],
            }
            for repeat in range(5)
        ]
    }
    runner._prepare_output = lambda: runner.output_dir.mkdir(parents=True)
    runner._fold_assignment_payload = lambda: {"schema_version": two_pl.SCRIPT_SCHEMA}
    runner._base_manifest = lambda status, **_kwargs: {
        "schema_version": two_pl.SCRIPT_SCHEMA,
        "status": status,
        "study_signature": runner.study_signature,
    }
    events: list[str] = []

    def select(repeat: int, outer: dict, **_kwargs):
        fold = int(outer["outer_fold"])
        events.append(f"select-{repeat}-{fold}")
        panel = _complete_panel(repeat, fold)
        if failed_panel == (repeat, fold):
            panel["status"] = "selection_blocked_incomplete_inner_evidence"
            panel["selected_spec_id"] = None
        return panel, []

    runner._select_or_load_panel = select

    def evaluate(panel: dict, _outer: dict) -> list[dict]:
        events.append(f"outer-{panel['repeat']}-{panel['outer_fold']}")
        return []

    runner._evaluate_outer_panel = evaluate
    terminal_calls: list[bool] = []
    runner._write_terminal_selection_failure_outputs = lambda **_kwargs: terminal_calls.append(True)
    runner._write_phase3_outputs = lambda **_kwargs: None
    assert runner.run() == 0
    assert events[:25] == [f"select-{repeat}-{fold}" for repeat in range(5) for fold in range(5)]
    if failed_panel is None:
        assert len([event for event in events if event.startswith("outer-")]) == 25
        assert terminal_calls == []
    else:
        assert not any(event.startswith("outer-") for event in events)
        assert terminal_calls == [True]


def test_manifest_free_resume_accepts_only_isolated_precompute_tree(tmp_path: Path) -> None:
    runner = _checkpoint_runner(tmp_path)
    runner.args = SimpleNamespace(resume=True)
    runner._already_complete = False
    outer = {"outer_fold": 0, "train_model_ids": ["a"], "test_model_ids": ["b"]}
    runner.split_audit = {"repetitions": [{"repeat": 0, "outer_folds": [outer]}]}
    runner._write_panel_selection_checkpoint(
        repeat=0,
        outer=outer,
        panel=_complete_panel(0, 0),
        evidence=[{"spec_id": two_pl.EXPECTED_PHASE3_SPEC_IDS[0]}],
    )
    runner._prepare_output()
    (tmp_path / "unexpected_parent_v3_cache").mkdir()
    with pytest.raises(two_pl.V3Phase3Error, match="non-precompute"):
        runner._prepare_output()


def test_all_25_hash_identical_precomputed_panels_are_resumable(tmp_path: Path) -> None:
    runner = _checkpoint_runner(tmp_path)
    runner.args = SimpleNamespace(resume=True)
    repetitions = []
    for repeat in range(5):
        outer_folds = []
        for outer_fold in range(5):
            outer = {
                "outer_fold": outer_fold,
                "train_model_ids": [f"train-{repeat}-{outer_fold}"],
                "test_model_ids": [f"test-{repeat}-{outer_fold}"],
            }
            outer_folds.append(outer)
            runner._write_panel_selection_checkpoint(
                repeat=repeat,
                outer=outer,
                panel=_complete_panel(repeat, outer_fold),
                evidence=[{"panel": two_pl._panel_id(repeat, outer_fold)}],
            )
        repetitions.append({"repeat": repeat, "outer_folds": outer_folds})
    runner.split_audit = {"repetitions": repetitions}
    runner._validate_precompute_only_tree()
    runner._select_panel = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("hash-identical completed panel must be reused")
    )
    loaded = [
        runner._select_or_load_panel(repeat, outer, permit_precomputed=True)[0]["panel_id"]
        for repeat, repetition in enumerate(repetitions)
        for outer in repetition["outer_folds"]
    ]
    assert loaded == [
        two_pl._panel_id(repeat, outer_fold) for repeat in range(5) for outer_fold in range(5)
    ]


def test_plan_only_is_read_only_and_official_leaf_was_not_started(tmp_path: Path) -> None:
    requested = tmp_path / "plan"
    canonical_existed_before = two_pl.DEFAULT_OUTPUT.exists()
    canonical_children_before = (
        sorted(
            str(path.relative_to(two_pl.DEFAULT_OUTPUT))
            for path in two_pl.DEFAULT_OUTPUT.rglob("*")
        )
        if canonical_existed_before
        else []
    )
    assert two_pl.main(["--plan-only", "--out-dir", str(requested)]) == 0
    assert not requested.exists()
    assert two_pl.DEFAULT_OUTPUT.exists() is canonical_existed_before
    canonical_children_after = (
        sorted(
            str(path.relative_to(two_pl.DEFAULT_OUTPUT))
            for path in two_pl.DEFAULT_OUTPUT.rglob("*")
        )
        if canonical_existed_before
        else []
    )
    assert canonical_children_after == canonical_children_before
