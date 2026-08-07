from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_infobench_remediation_splits.py"
CONFIG = ROOT / "configs" / "infobench_remediation_splits_v1.json"
MANIFEST = ROOT / "configs" / "infobench_remediation_splits_v1.manifest.json"
MODEL_CSV = ROOT / "configs" / "infobench_remediation_splits_v1.models.csv"
SCENARIO_CSV = ROOT / "configs" / "infobench_remediation_splits_v1.scenarios.csv"

SPEC = importlib.util.spec_from_file_location("freeze_infobench_remediation_splits", SCRIPT)
assert SPEC and SPEC.loader
splits = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = splits
SPEC.loader.exec_module(splits)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_split_artifacts_reproduce_exactly_from_pinned_inputs() -> None:
    first, first_models, first_scenarios = splits.build_manifest(CONFIG)
    second, second_models, second_scenarios = splits.build_manifest(CONFIG)

    assert first == second == _manifest()
    assert first_models == second_models == MODEL_CSV.read_bytes()
    assert first_scenarios == second_scenarios == SCENARIO_CSV.read_bytes()
    assert first["inputs"]["response_matrix"]["sha256"] == (
        "087948fcaa884cde6660df1fb4964072ec60fe8aee398e293ed5f74db8f3f27c"
    )


def test_outer_test_ids_never_enter_fit_or_inner_sets() -> None:
    manifest = _manifest()
    all_models = set(manifest["model_to_family"])
    outer_test_union: set[str] = set()

    for outer in manifest["outer_folds"]:
        test = set(outer["test_model_ids"])
        train = set(outer["train_model_ids"])
        assert test.isdisjoint(train)
        assert test | train == all_models
        assert outer_test_union.isdisjoint(test)
        outer_test_union.update(test)

        validated: list[str] = []
        for inner in outer["inner_folds"]:
            validation = set(inner["validation_model_ids"])
            fit = set(inner["fit_model_ids"])
            assert test.isdisjoint(validation)
            assert test.isdisjoint(fit)
            assert validation.isdisjoint(fit)
            assert validation | fit == train
            validated.extend(validation)
        assert sorted(validated) == sorted(train)

    assert outer_test_union == all_models


def test_model_families_never_cross_outer_or_inner_fold_boundaries() -> None:
    manifest = _manifest()
    family_groups = {
        family: set(models) for family, models in manifest["model_families"].items()
    }
    model_to_outer = manifest["model_to_outer_fold"]

    for family, models in family_groups.items():
        assert {model_to_outer[model] for model in models} == {
            model_to_outer[next(iter(models))]
        }, family

    for outer in manifest["outer_folds"]:
        test = set(outer["test_model_ids"])
        for family in outer["test_family_ids"]:
            assert family_groups[family] <= test
        for inner in outer["inner_folds"]:
            validation = set(inner["validation_model_ids"])
            for family in inner["validation_family_ids"]:
                assert family_groups[family] <= validation


def test_four_inner_folds_are_the_four_remaining_outer_folds() -> None:
    manifest = _manifest()
    outer_test = {
        int(outer["outer_fold"]): set(outer["test_model_ids"])
        for outer in manifest["outer_folds"]
    }
    assert sorted(len(models) for models in outer_test.values()) == [10, 10, 10, 11, 11]

    for outer in manifest["outer_folds"]:
        assert len(outer["inner_folds"]) == 4
        sources = set()
        for inner in outer["inner_folds"]:
            source = int(inner["source_outer_fold"])
            sources.add(source)
            assert set(inner["validation_model_ids"]) == outer_test[source]
        assert sources == set(outer_test) - {int(outer["outer_fold"])}


def test_scenario_split_is_disjoint_scenario_level_80_20() -> None:
    frame = pd.read_csv(SCENARIO_CSV)
    assert len(frame) == 500
    assert frame["scenario_id"].is_unique
    assert frame["criterion_count"].sum() == 2250
    assert frame["role"].value_counts().to_dict() == {
        "administration": 400,
        "evaluation": 100,
    }
    by_subset = frame.groupby(["subset", "role"]).size().to_dict()
    assert by_subset == {
        ("Easy_set", "administration"): 202,
        ("Easy_set", "evaluation"): 50,
        ("Hard_set", "administration"): 198,
        ("Hard_set", "evaluation"): 50,
    }


def test_leakage_audit_fails_closed_on_outer_model_contamination() -> None:
    manifest = _manifest()
    contaminated = copy.deepcopy(manifest)
    leaked = contaminated["outer_folds"][0]["test_model_ids"][0]
    contaminated["outer_folds"][0]["inner_folds"][0]["fit_model_ids"].append(leaked)

    with pytest.raises(splits.SplitError, match="leaked"):
        splits.audit_manifest(contaminated)


def test_leakage_audit_fails_closed_on_family_or_scenario_contamination() -> None:
    manifest = _manifest()
    broken_family = copy.deepcopy(manifest)
    family = broken_family["outer_folds"][0]["test_family_ids"][0]
    member = broken_family["model_families"][family][0]
    broken_family["outer_folds"][0]["test_model_ids"].remove(member)
    broken_family["outer_folds"][0]["train_model_ids"].append(member)
    with pytest.raises(splits.SplitError, match="split across outer folds"):
        splits.audit_manifest(broken_family)

    broken_scenario = copy.deepcopy(manifest)
    leaked_scenario = broken_scenario["scenario_split"]["evaluation_scenario_ids"][0]
    broken_scenario["scenario_split"]["administration_scenario_ids"].append(
        leaked_scenario
    )
    with pytest.raises(splits.SplitError, match="scenario"):
        splits.audit_manifest(broken_scenario)
