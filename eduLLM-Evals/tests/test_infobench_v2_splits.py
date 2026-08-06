from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_infobench_v2_splits.py"
CONFIG = ROOT / "configs" / "infobench_v2_splits.json"
MANIFEST = ROOT / "configs" / "infobench_v2_splits.manifest.json"
MODEL_CSV = ROOT / "configs" / "infobench_v2_splits.models.csv"
SCENARIO_CSV = ROOT / "configs" / "infobench_v2_splits.scenarios.csv"
V1_MANIFEST = ROOT / "configs" / "infobench_remediation_splits_v1.manifest.json"
V1_SCENARIO_CSV = ROOT / "configs" / "infobench_remediation_splits_v1.scenarios.csv"

SPEC = importlib.util.spec_from_file_location("freeze_infobench_v2_splits", SCRIPT)
assert SPEC and SPEC.loader
splits = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = splits
SPEC.loader.exec_module(splits)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_v2_artifacts_reproduce_exactly() -> None:
    first, first_models, first_scenarios = splits.build_artifacts(CONFIG)
    second, second_models, second_scenarios = splits.build_artifacts(CONFIG)

    assert first == second == _manifest()
    assert first_models == second_models == MODEL_CSV.read_bytes()
    assert first_scenarios == second_scenarios == SCENARIO_CSV.read_bytes()
    assert first["configuration_sha256"] == splits.sha256_file(CONFIG)
    assert first["generator_sha256"] == splits.sha256_file(SCRIPT)


def test_repeat_zero_exactly_reuses_v1_outer_partition() -> None:
    v2 = _manifest()
    v1 = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    repeat_zero = v2["repetitions"][0]

    assert repeat_zero["source"] == "exact_pinned_v1_partition"
    assert repeat_zero["derived_repeat_seed"] is None
    assert [outer["test_family_ids"] for outer in repeat_zero["outer_folds"]] == [
        outer["test_family_ids"] for outer in v1["outer_folds"]
    ]
    assert [outer["test_model_ids"] for outer in repeat_zero["outer_folds"]] == [
        outer["test_model_ids"] for outer in v1["outer_folds"]
    ]


def test_each_repeat_groups_families_and_tests_all_52_models_once() -> None:
    manifest = _manifest()
    all_models = set(manifest["model_to_family"])
    family_groups = {
        family: set(models) for family, models in manifest["model_families"].items()
    }

    assert len(all_models) == 52
    assert len(manifest["repetitions"]) == 5
    for repetition in manifest["repetitions"]:
        test_counts = {model: 0 for model in all_models}
        family_counts = {family: 0 for family in family_groups}
        for outer in repetition["outer_folds"]:
            test = set(outer["test_model_ids"])
            train = set(outer["train_model_ids"])
            assert test.isdisjoint(train)
            assert test | train == all_models
            for model in test:
                test_counts[model] += 1
            for family in outer["test_family_ids"]:
                family_counts[family] += 1
                assert family_groups[family] <= test

            validated: list[str] = []
            for inner in outer["inner_folds"]:
                validation = set(inner["validation_model_ids"])
                fit = set(inner["fit_model_ids"])
                assert test.isdisjoint(validation | fit)
                assert validation.isdisjoint(fit)
                assert validation | fit == train
                validated.extend(validation)
            assert sorted(validated) == sorted(train)

        assert set(test_counts.values()) == {1}
        assert set(family_counts.values()) == {1}


def test_generated_repeats_use_one_master_seed_and_are_distinct() -> None:
    config = _config()
    manifest = _manifest()
    master_seed = int(config["cross_validation"]["master_seed"])
    canonical = []

    for repetition in manifest["repetitions"]:
        repeat = int(repetition["repeat"])
        family_folds = [outer["test_family_ids"] for outer in repetition["outer_folds"]]
        canonical.append(splits._canonical_partition(family_folds))
        if repeat > 0:
            assert repetition["derived_repeat_seed"] == splits.derive_repeat_seed(
                master_seed, repeat
            )

    assert len(set(canonical)) == 5
    assert manifest["construction"]["seed_search"] is False
    assert manifest["construction"]["candidate_partition_scoring"] is False


def test_generated_partition_uses_only_frozen_group_and_stratum_fields() -> None:
    config = _config()
    manifest = _manifest()
    cv = config["cross_validation"]
    family_to_stratum = {
        family: stratum
        for stratum, families in config["balance_strata"].items()
        for family in families
    }

    for repeat in range(1, 5):
        regenerated = splits.generate_repeat_family_folds(
            family_groups=config["model_families"],
            family_to_stratum=family_to_stratum,
            target_model_counts=cv["outer_fold_model_counts"],
            minimum_families=cv["minimum_families_per_generated_fold"],
            maximum_families=cv["maximum_families_per_generated_fold"],
            maximum_per_stratum=(
                cv["maximum_families_from_each_balance_stratum_per_generated_fold"]
            ),
            maximum_stratum_rank_deviation=(
                cv["maximum_absolute_model_weighted_stratum_rank_deviation"]
            ),
            repeat_seed=splits.derive_repeat_seed(cv["master_seed"], repeat),
        )
        frozen = [
            outer["test_family_ids"]
            for outer in manifest["repetitions"][repeat]["outer_folds"]
        ]
        assert regenerated == frozen

    # Exact pass-rate values are diagnostic after the categorical strata freeze;
    # changing those diagnostics cannot make an otherwise valid split pass/fail.
    diagnostic_change = copy.deepcopy(manifest)
    diagnostic_change["family_balance"][
        "family_mean_observed_pass_rate_diagnostic_only"
    ] = {family: -1.0 for family in config["model_families"]}
    for repetition in diagnostic_change["repetitions"]:
        for outer in repetition["outer_folds"]:
            outer["observed_test_pass_rate_mean_diagnostic_only"] = -1.0
    splits.audit_manifest(diagnostic_change)


def test_generated_fold_balance_contract_is_satisfied() -> None:
    config = _config()
    manifest = _manifest()
    cv = config["cross_validation"]
    targets = cv["outer_fold_model_counts"]
    family_sizes = {
        family: len(models) for family, models in config["model_families"].items()
    }
    ordered_strata = list(config["balance_strata"])
    family_to_stratum = {
        family: stratum
        for stratum, families in config["balance_strata"].items()
        for family in families
    }
    stratum_rank = {
        stratum: rank for rank, stratum in enumerate(ordered_strata, start=1)
    }
    cohort_mean = sum(
        stratum_rank[family_to_stratum[family]] * family_sizes[family]
        for family in family_sizes
    ) / 52

    for repetition in manifest["repetitions"][1:]:
        for fold, outer in enumerate(repetition["outer_folds"]):
            families = outer["test_family_ids"]
            assert len(outer["test_model_ids"]) == targets[fold]
            assert 4 <= len(families) <= 5
            assert max(outer["balance_stratum_family_counts"].values()) <= 1
            fold_mean = sum(
                stratum_rank[family_to_stratum[family]] * family_sizes[family]
                for family in families
            ) / targets[fold]
            assert abs(fold_mean - cohort_mean) <= 0.4 + 1e-12


def test_scenario_split_is_an_exact_copy_of_v1() -> None:
    v2 = _manifest()
    v1 = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))

    assert v2["scenario_split"] == v1["scenario_split"]
    assert SCENARIO_CSV.read_bytes() == V1_SCENARIO_CSV.read_bytes()
    assert splits.sha256_file(SCENARIO_CSV) == (
        "10075f8de40cd6677ae233c27e52de803fd9288773fe605a3d52a2159d2d2e4b"
    )


def test_model_assignment_csv_has_52_unique_rows_per_repeat() -> None:
    frame = pd.read_csv(MODEL_CSV)
    assert len(frame) == 260
    assert frame.groupby("repeat").size().to_dict() == {repeat: 52 for repeat in range(5)}
    assert frame.groupby("repeat")["model_id"].nunique().to_dict() == {
        repeat: 52 for repeat in range(5)
    }


def test_audit_fails_closed_on_cross_family_or_outer_leakage() -> None:
    manifest = _manifest()
    contaminated = copy.deepcopy(manifest)
    leaked = contaminated["repetitions"][1]["outer_folds"][0]["test_model_ids"][0]
    contaminated["repetitions"][1]["outer_folds"][0]["inner_folds"][0][
        "fit_model_ids"
    ].append(leaked)
    with pytest.raises(splits.SplitError, match="leaked"):
        splits.audit_manifest(contaminated)

    split_family = copy.deepcopy(manifest)
    outer = split_family["repetitions"][2]["outer_folds"][0]
    family = outer["test_family_ids"][0]
    member = split_family["model_families"][family][0]
    outer["test_model_ids"].remove(member)
    outer["train_model_ids"].append(member)
    with pytest.raises(splits.SplitError, match="splits a related model family"):
        splits.audit_manifest(split_family)
