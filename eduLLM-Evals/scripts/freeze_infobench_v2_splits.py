#!/usr/bin/env python3
"""Freeze repeated family-grouped nested-CV splits for InFoBench v2.

Repeat 0 reuses the exact v1 outer family partition.  Repeats 1-4 are
generated once from a single master seed.  The generator balances only model
counts, family counts, and the categorical family pass-rate quintiles frozen in
the config; it never searches seeds or scores candidate partitions against
response outcomes.

The scenario administration/evaluation split is copied exactly from the pinned
v1 artifacts.  This script fits no IRT model and runs no CAT evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "infobench_v2_splits.json"


class SplitError(RuntimeError):
    """Raised when a pinned input or split invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitError(f"could not read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SplitError(f"expected a JSON object in {path}")
    return value


def _resolve(configured: str) -> Path:
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _verify_pinned(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise SplitError(f"pinned {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise SplitError(
            f"{label} hash mismatch: expected {expected_sha256}, got {actual} ({path})"
        )


def _validate_matrix(
    matrix: pd.DataFrame,
    *,
    expected_models: int,
    expected_criteria: int,
) -> pd.DataFrame:
    matrix.index = matrix.index.map(str)
    matrix.columns = matrix.columns.map(str)
    if matrix.index.has_duplicates:
        raise SplitError("response matrix contains duplicate model IDs")
    if matrix.columns.has_duplicates:
        raise SplitError("response matrix contains duplicate criterion IDs")
    if matrix.shape != (expected_models, expected_criteria):
        raise SplitError(
            "response-matrix shape mismatch: "
            f"expected {(expected_models, expected_criteria)}, got {matrix.shape}"
        )
    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    invalid = numeric.notna() & ~numeric.isin([0.0, 1.0])
    if bool(invalid.any().any()):
        location = invalid.stack()[lambda values: values].index[0]
        raise SplitError(f"response matrix contains a non-binary value at {location}")
    empty_models = numeric.index[numeric.notna().sum(axis=1).eq(0)].tolist()
    if empty_models:
        raise SplitError(f"models with no observed outcomes: {empty_models}")
    return numeric


def _model_to_family(
    models: Iterable[str], family_groups: dict[str, list[str]]
) -> dict[str, str]:
    expected = {str(model) for model in models}
    seen: dict[str, str] = {}
    for family_id, members in family_groups.items():
        if not family_id or not members:
            raise SplitError(f"family {family_id!r} must contain at least one model")
        for model in members:
            if model in seen:
                raise SplitError(
                    f"model {model!r} appears in both {seen[model]!r} and {family_id!r}"
                )
            seen[model] = family_id
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    if missing or extra:
        raise SplitError(
            f"family roster differs from matrix (missing={missing}, extra={extra})"
        )
    return seen


def _family_to_stratum(
    family_groups: dict[str, list[str]], balance_strata: dict[str, list[str]]
) -> dict[str, str]:
    seen: dict[str, str] = {}
    for stratum, families in balance_strata.items():
        if not families:
            raise SplitError(f"balance stratum {stratum!r} is empty")
        for family in families:
            if family in seen:
                raise SplitError(
                    f"family {family!r} appears in balance strata "
                    f"{seen[family]!r} and {stratum!r}"
                )
            seen[family] = stratum
    missing = sorted(set(family_groups) - set(seen))
    extra = sorted(set(seen) - set(family_groups))
    if missing or extra:
        raise SplitError(
            "balance strata do not partition families exactly "
            f"(missing={missing}, extra={extra})"
        )
    return seen


def _validate_frozen_rank_strata(
    family_pass_rates: dict[str, float],
    balance_strata: dict[str, list[str]],
    stratum_sizes: list[int],
) -> None:
    """Verify that config labels are the declared rank quintiles.

    Exact rates are used only to verify the frozen categorical field and for
    diagnostics.  The split generator receives only the resulting labels.
    """

    if len(stratum_sizes) != len(balance_strata):
        raise SplitError("balance-stratum sizes and labels differ in length")
    if sum(stratum_sizes) != len(family_pass_rates):
        raise SplitError("balance-stratum sizes do not cover all model families")
    ranked = sorted(family_pass_rates, key=lambda family: (family_pass_rates[family], family))
    offset = 0
    for (stratum, frozen_families), size in zip(
        balance_strata.items(), stratum_sizes, strict=True
    ):
        expected = ranked[offset : offset + size]
        if frozen_families != expected:
            raise SplitError(
                f"frozen balance stratum {stratum!r} is not the declared rank slice: "
                f"expected {expected}, found {frozen_families}"
            )
        offset += size


def derive_repeat_seed(master_seed: int, repeat: int) -> int:
    if repeat <= 0:
        raise SplitError("derived seeds are defined only for generated repeats")
    digest = hashlib.sha256(f"{master_seed}\0repeat\0{repeat}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _stable_rank(seed: int, *parts: object) -> str:
    material = "\0".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(material.encode()).hexdigest()


def generate_repeat_family_folds(
    *,
    family_groups: dict[str, list[str]],
    family_to_stratum: dict[str, str],
    target_model_counts: list[int],
    minimum_families: int,
    maximum_families: int,
    maximum_per_stratum: int,
    maximum_stratum_rank_deviation: float,
    repeat_seed: int,
) -> list[list[str]]:
    """Generate one exact-capacity grouped partition without outcome scoring.

    There is no seed search and no candidate objective.  A single seeded family
    order and seeded fold tie-break are used in a deterministic depth-first
    feasibility search.  Only family sizes and frozen categorical strata enter.
    """

    if sum(target_model_counts) != sum(len(models) for models in family_groups.values()):
        raise SplitError("target outer-fold sizes do not cover the model roster")
    if minimum_families <= 0 or maximum_families < minimum_families:
        raise SplitError("invalid generated-fold family-count bounds")
    if maximum_per_stratum <= 0:
        raise SplitError("maximum families per stratum must be positive")
    if maximum_stratum_rank_deviation <= 0.0:
        raise SplitError("maximum stratum-rank deviation must be positive")

    family_sizes = {family: len(models) for family, models in family_groups.items()}
    ordered_families = sorted(
        family_groups,
        key=lambda family: (
            -family_sizes[family],
            _stable_rank(repeat_seed, "family-order", family),
            family,
        ),
    )
    folds: list[list[str]] = [[] for _ in target_model_counts]
    remaining_capacity = list(target_model_counts)
    strata = sorted(set(family_to_stratum.values()))
    configured_strata = list(dict.fromkeys(family_to_stratum.values()))
    stratum_rank = {
        stratum: rank for rank, stratum in enumerate(configured_strata, start=1)
    }
    cohort_stratum_rank_mean = sum(
        stratum_rank[family_to_stratum[family]] * family_sizes[family]
        for family in family_groups
    ) / sum(family_sizes.values())
    stratum_counts = [{stratum: 0 for stratum in strata} for _ in target_model_counts]
    fold_stratum_rank_totals = [0 for _ in target_model_counts]

    def search(index: int) -> bool:
        if index == len(ordered_families):
            valid_counts = all(capacity == 0 for capacity in remaining_capacity) and all(
                minimum_families <= len(fold) <= maximum_families for fold in folds
            )
            valid_stratum_balance = all(
                abs(
                    fold_stratum_rank_totals[fold] / target_model_counts[fold]
                    - cohort_stratum_rank_mean
                )
                <= maximum_stratum_rank_deviation
                for fold in range(len(folds))
            )
            return valid_counts and valid_stratum_balance

        family = ordered_families[index]
        size = family_sizes[family]
        stratum = family_to_stratum[family]
        candidates = [
            fold
            for fold, capacity in enumerate(remaining_capacity)
            if capacity >= size
            and len(folds[fold]) < maximum_families
            and stratum_counts[fold][stratum] < maximum_per_stratum
        ]
        candidates.sort(
            key=lambda fold: (
                len(folds[fold]),
                -remaining_capacity[fold],
                _stable_rank(repeat_seed, "fold-tie", index, family, fold),
                fold,
            )
        )

        for fold in candidates:
            folds[fold].append(family)
            remaining_capacity[fold] -= size
            stratum_counts[fold][stratum] += 1
            fold_stratum_rank_totals[fold] += stratum_rank[stratum] * size

            remaining = ordered_families[index + 1 :]
            remaining_slots = len(remaining)
            minimum_slots_needed = sum(
                max(0, minimum_families - len(assigned)) for assigned in folds
            )
            maximum_slots_available = sum(
                maximum_families - len(assigned) for assigned in folds
            )
            exact_capacity = sum(remaining_capacity) == sum(
                family_sizes[value] for value in remaining
            )
            feasible_family_counts = (
                minimum_slots_needed <= remaining_slots <= maximum_slots_available
            )
            if exact_capacity and feasible_family_counts and search(index + 1):
                return True

            folds[fold].pop()
            remaining_capacity[fold] += size
            stratum_counts[fold][stratum] -= 1
            fold_stratum_rank_totals[fold] -= stratum_rank[stratum] * size
        return False

    if not search(0):
        raise SplitError(
            "the single seeded assignment has no feasible exact-capacity partition; "
            "change the preregistered design rather than searching alternate seeds"
        )
    return [sorted(fold) for fold in folds]


def _family_models(
    families: Iterable[str], family_groups: dict[str, list[str]]
) -> list[str]:
    return sorted(model for family in families for model in family_groups[family])


def _canonical_partition(folds: list[list[str]]) -> tuple[tuple[str, ...], ...]:
    """Ignore arbitrary fold labels when checking repeated partitions."""

    return tuple(sorted(tuple(sorted(fold)) for fold in folds))


def _outer_records(
    *,
    family_folds: list[list[str]],
    family_groups: dict[str, list[str]],
    family_to_stratum: dict[str, str],
    model_pass_rates: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    all_models = set(model_pass_rates.index)
    model_folds = [_family_models(families, family_groups) for families in family_folds]
    model_to_outer = {
        model: fold for fold, models in enumerate(model_folds) for model in models
    }
    records: list[dict[str, Any]] = []
    for outer_fold, test_models_list in enumerate(model_folds):
        test = set(test_models_list)
        train = all_models - test
        inner_records: list[dict[str, Any]] = []
        for inner_fold, source_outer_fold in enumerate(
            fold for fold in range(len(model_folds)) if fold != outer_fold
        ):
            validation = set(model_folds[source_outer_fold])
            fit = train - validation
            fit_families = sorted(
                family
                for fold, families in enumerate(family_folds)
                if fold not in {outer_fold, source_outer_fold}
                for family in families
            )
            inner_records.append(
                {
                    "inner_fold": inner_fold,
                    "source_outer_fold": source_outer_fold,
                    "validation_family_ids": family_folds[source_outer_fold],
                    "validation_model_ids": sorted(validation),
                    "fit_family_ids": fit_families,
                    "fit_model_ids": sorted(fit),
                }
            )
        test_families = family_folds[outer_fold]
        records.append(
            {
                "outer_fold": outer_fold,
                "test_family_ids": test_families,
                "test_model_ids": sorted(test),
                "train_family_ids": sorted(
                    family
                    for fold, families in enumerate(family_folds)
                    if fold != outer_fold
                    for family in families
                ),
                "train_model_ids": sorted(train),
                "balance_stratum_family_counts": {
                    stratum: sum(
                        family_to_stratum[family] == stratum for family in test_families
                    )
                    for stratum in sorted(set(family_to_stratum.values()))
                },
                "observed_test_pass_rate_mean_diagnostic_only": float(
                    model_pass_rates.loc[sorted(test)].mean()
                ),
                "inner_folds": inner_records,
            }
        )
    return records, model_to_outer


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _validate_scenario_artifact(
    scenario_bytes: bytes, scenario_split: dict[str, Any], expected_scenarios: int
) -> None:
    rows = list(csv.DictReader(io.StringIO(scenario_bytes.decode("utf-8"))))
    if len(rows) != expected_scenarios:
        raise SplitError(
            f"v1 scenario assignment count is {len(rows)}, expected {expected_scenarios}"
        )
    if len({row["scenario_id"] for row in rows}) != len(rows):
        raise SplitError("v1 scenario assignments contain duplicate scenario IDs")
    administration = {
        row["scenario_id"] for row in rows if row["role"] == "administration"
    }
    evaluation = {row["scenario_id"] for row in rows if row["role"] == "evaluation"}
    if administration != set(scenario_split["administration_scenario_ids"]):
        raise SplitError("v1 scenario CSV and manifest administration IDs disagree")
    if evaluation != set(scenario_split["evaluation_scenario_ids"]):
        raise SplitError("v1 scenario CSV and manifest evaluation IDs disagree")
    if administration & evaluation or len(administration | evaluation) != expected_scenarios:
        raise SplitError("v1 scenario administration/evaluation split is invalid")


def build_artifacts(config_path: Path) -> tuple[dict[str, Any], bytes, bytes]:
    config_path = config_path.resolve()
    config = _load_json(config_path)
    inputs = config["inputs"]
    input_paths: dict[str, Path] = {}
    for label, specification in inputs.items():
        path = _resolve(str(specification["path"]))
        _verify_pinned(path, str(specification["sha256"]), label)
        input_paths[label] = path

    expected = config["expectations"]
    matrix = _validate_matrix(
        pd.read_csv(input_paths["response_matrix"], index_col=0),
        expected_models=int(expected["models"]),
        expected_criteria=int(expected["criteria"]),
    )
    family_groups = {
        str(family): [str(model) for model in models]
        for family, models in config["model_families"].items()
    }
    if len(family_groups) != int(expected["model_families"]):
        raise SplitError("unexpected number of model families")
    model_to_family = _model_to_family(matrix.index, family_groups)

    v1_config = _load_json(input_paths["v1_split_config"])
    v1_manifest = _load_json(input_paths["v1_split_manifest"])
    if v1_config.get("model_families") != config.get("model_families"):
        raise SplitError("v2 family roster is not identical to the pinned v1 roster")
    if v1_manifest.get("model_to_family") != {
        model: model_to_family[model] for model in sorted(model_to_family)
    }:
        raise SplitError("v1 manifest family mapping differs from the v2 roster")

    model_pass_rates = matrix.mean(axis=1, skipna=True)
    family_pass_rates = {
        family: float(model_pass_rates.loc[models].mean())
        for family, models in family_groups.items()
    }
    balance_strata = {
        str(stratum): [str(family) for family in families]
        for stratum, families in config["balance_strata"].items()
    }
    family_to_stratum = _family_to_stratum(family_groups, balance_strata)
    _validate_frozen_rank_strata(
        family_pass_rates,
        balance_strata,
        [int(value) for value in config["balance_stratum_sizes"]],
    )

    cv = config["cross_validation"]
    repetitions = int(cv["repetitions"])
    outer_folds = int(cv["outer_folds"])
    inner_folds = int(cv["inner_folds"])
    generated_indices = [int(value) for value in cv["generated_repeat_indices"]]
    if repetitions != 5 or outer_folds != 5 or inner_folds != outer_folds - 1:
        raise SplitError("v2 requires exactly five repeats of five-by-four nested CV")
    if generated_indices != list(range(1, repetitions)):
        raise SplitError("generated repeat indices must be exactly 1 through 4")

    repeat_zero_folds = [
        [str(family) for family in outer["test_family_ids"]]
        for outer in v1_manifest["outer_folds"]
    ]
    if len(repeat_zero_folds) != outer_folds:
        raise SplitError("pinned v1 manifest does not contain five outer folds")

    master_seed = int(cv["master_seed"])
    target_model_counts = [int(value) for value in cv["outer_fold_model_counts"]]
    if len(target_model_counts) != outer_folds:
        raise SplitError("outer-fold count and target model counts disagree")

    repeated_family_folds: list[list[list[str]]] = [repeat_zero_folds]
    repeat_seeds: list[int | None] = [None]
    for repeat in generated_indices:
        repeat_seed = derive_repeat_seed(master_seed, repeat)
        generated = generate_repeat_family_folds(
            family_groups=family_groups,
            family_to_stratum=family_to_stratum,
            target_model_counts=target_model_counts,
            minimum_families=int(cv["minimum_families_per_generated_fold"]),
            maximum_families=int(cv["maximum_families_per_generated_fold"]),
            maximum_per_stratum=int(
                cv["maximum_families_from_each_balance_stratum_per_generated_fold"]
            ),
            maximum_stratum_rank_deviation=float(
                cv["maximum_absolute_model_weighted_stratum_rank_deviation"]
            ),
            repeat_seed=repeat_seed,
        )
        repeated_family_folds.append(generated)
        repeat_seeds.append(repeat_seed)

    canonical_partitions = [_canonical_partition(folds) for folds in repeated_family_folds]
    if len(set(canonical_partitions)) != repetitions:
        raise SplitError("repeated outer partitions are not all distinct")

    repeat_records: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for repeat, (family_folds, repeat_seed) in enumerate(
        zip(repeated_family_folds, repeat_seeds, strict=True)
    ):
        outer_records, model_to_outer = _outer_records(
            family_folds=family_folds,
            family_groups=family_groups,
            family_to_stratum=family_to_stratum,
            model_pass_rates=model_pass_rates,
        )
        source = "exact_pinned_v1_partition" if repeat == 0 else "single_seed_generation"
        repeat_records.append(
            {
                "repeat": repeat,
                "source": source,
                "derived_repeat_seed": repeat_seed,
                "model_to_outer_fold": {
                    model: model_to_outer[model] for model in sorted(model_to_outer)
                },
                "outer_folds": outer_records,
            }
        )
        for model in sorted(model_to_family):
            model_rows.append(
                {
                    "repeat": repeat,
                    "source": source,
                    "derived_repeat_seed": "" if repeat_seed is None else repeat_seed,
                    "model_id": model,
                    "family_id": model_to_family[model],
                    "balance_stratum": family_to_stratum[model_to_family[model]],
                    "outer_fold": model_to_outer[model],
                    "observed_pass_rate_diagnostic_only": (
                        f"{float(model_pass_rates.loc[model]):.12f}"
                    ),
                }
            )

    model_csv = _csv_bytes(
        model_rows,
        [
            "repeat",
            "source",
            "derived_repeat_seed",
            "model_id",
            "family_id",
            "balance_stratum",
            "outer_fold",
            "observed_pass_rate_diagnostic_only",
        ],
    )

    scenario_bytes = input_paths["v1_scenario_assignments"].read_bytes()
    scenario_split = deepcopy(v1_manifest["scenario_split"])
    _validate_scenario_artifact(
        scenario_bytes, scenario_split, int(expected["scenarios"])
    )

    manifest = {
        "schema_version": "infobench-v2-repeated-splits-v1",
        "benchmark": "InFoBench",
        "generated_by": "scripts/freeze_infobench_v2_splits.py",
        "generator_sha256": sha256_file(Path(__file__)),
        "configuration_sha256": sha256_file(config_path),
        "inputs": {
            label: {
                "configured_path": str(specification["path"]),
                "sha256": str(specification["sha256"]),
            }
            for label, specification in inputs.items()
        },
        "construction": {
            "repetitions": repetitions,
            "outer_folds_per_repeat": outer_folds,
            "inner_folds_per_outer": inner_folds,
            "repeat_zero": "exact_pinned_v1_outer_family_partition",
            "generated_repeats": generated_indices,
            "master_seed": master_seed,
            "derived_repeat_seed_method": (
                "uint64_from_first_8_bytes_sha256(master_seed\\0repeat\\0index)"
            ),
            "generated_partition_method": str(cv["generation_algorithm"]),
            "assignment_fields": [
                "model_count_per_family",
                "family_count_per_fold",
                "frozen_family_pass_rate_quintile",
                "sha256_seeded_tie_break",
            ],
            "fields_excluded_from_assignment": [
                "per_criterion_response_outcomes",
                "exact_family_or_model_pass_rate_within_frozen_quintile",
                "IRT_fit_metrics",
                "CAT_metrics",
                "outer_fold_results",
            ],
            "seed_search": False,
            "candidate_partition_scoring": False,
            "scenario_split": "exact_copy_of_pinned_v1_administration_evaluation_split",
        },
        "model_families": {
            family: family_groups[family] for family in sorted(family_groups)
        },
        "model_to_family": {
            model: model_to_family[model] for model in sorted(model_to_family)
        },
        "family_balance": {
            "field": str(cv["balance_field"]),
            "role": str(cv["balance_field_role"]),
            "strata": balance_strata,
            "family_mean_observed_pass_rate_diagnostic_only": {
                family: family_pass_rates[family] for family in sorted(family_pass_rates)
            },
            "generated_fold_constraints": {
                "target_model_counts": target_model_counts,
                "minimum_families": int(cv["minimum_families_per_generated_fold"]),
                "maximum_families": int(cv["maximum_families_per_generated_fold"]),
                "maximum_families_per_stratum": int(
                    cv["maximum_families_from_each_balance_stratum_per_generated_fold"]
                ),
                "maximum_absolute_model_weighted_stratum_rank_deviation": float(
                    cv["maximum_absolute_model_weighted_stratum_rank_deviation"]
                ),
            },
        },
        "repetitions": repeat_records,
        "scenario_split": scenario_split,
        "artifact_hashes": {
            "model_assignments_csv_sha256": sha256_bytes(model_csv),
            "scenario_assignments_csv_sha256": sha256_bytes(scenario_bytes),
        },
        "leakage_contract": [
            "Each model is outer-test exactly once per repetition.",
            "Every related model family remains intact in every outer and inner partition.",
            "An outer-test model never enters that panel's fit or inner-validation sets.",
            "Each outer-training model is inner-validation exactly once.",
            "The v1 scenario administration/evaluation boundary is unchanged.",
            (
                "No response outcomes beyond the frozen categorical balance stratum "
                "influence generated assignments."
            ),
            (
                "All five repetitions are retained; no split may be removed after v2 "
                "outcomes are observed."
            ),
        ],
    }
    audit_manifest(manifest)
    return manifest, model_csv, scenario_bytes


def audit_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed on coverage, grouped-family, inner, or scenario leakage."""

    family_groups = {
        family: set(models) for family, models in manifest["model_families"].items()
    }
    all_models = set(manifest["model_to_family"])
    if set().union(*family_groups.values()) != all_models:
        raise SplitError("family groups do not cover the model roster exactly")

    constraints = manifest["family_balance"]["generated_fold_constraints"]
    target_sizes = [int(value) for value in constraints["target_model_counts"]]
    min_families = int(constraints["minimum_families"])
    max_families = int(constraints["maximum_families"])
    max_per_stratum = int(constraints["maximum_families_per_stratum"])
    maximum_stratum_rank_deviation = float(
        constraints["maximum_absolute_model_weighted_stratum_rank_deviation"]
    )
    ordered_strata = list(manifest["family_balance"]["strata"])
    stratum_rank = {
        stratum: rank for rank, stratum in enumerate(ordered_strata, start=1)
    }
    family_to_stratum = {
        family: stratum
        for stratum, families in manifest["family_balance"]["strata"].items()
        for family in families
    }
    family_sizes = {family: len(models) for family, models in family_groups.items()}
    cohort_stratum_rank_mean = sum(
        stratum_rank[family_to_stratum[family]] * family_sizes[family]
        for family in family_groups
    ) / sum(family_sizes.values())

    repeats = manifest["repetitions"]
    if [int(record["repeat"]) for record in repeats] != list(range(5)):
        raise SplitError("manifest does not contain repeats 0 through 4 exactly once")
    canonical_partitions: list[tuple[tuple[str, ...], ...]] = []
    for repeat_record in repeats:
        repeat = int(repeat_record["repeat"])
        outer_records = repeat_record["outer_folds"]
        if len(outer_records) != 5:
            raise SplitError(f"repeat {repeat} does not contain five outer folds")
        test_seen: set[str] = set()
        family_seen: set[str] = set()
        family_folds: list[list[str]] = []
        for expected_fold, outer in enumerate(outer_records):
            if int(outer["outer_fold"]) != expected_fold:
                raise SplitError(f"repeat {repeat} outer-fold numbering is not canonical")
            test = set(outer["test_model_ids"])
            train = set(outer["train_model_ids"])
            families = list(outer["test_family_ids"])
            family_folds.append(families)
            if test & train or test | train != all_models:
                raise SplitError(f"repeat {repeat} fold {expected_fold} partition is invalid")
            if test_seen & test:
                raise SplitError(f"repeat {repeat} tests a model more than once")
            if family_seen & set(families):
                raise SplitError(f"repeat {repeat} tests a family more than once")
            test_seen.update(test)
            family_seen.update(families)
            sorted_family_groups = {
                family: sorted(models) for family, models in family_groups.items()
            }
            expected_test = set(_family_models(families, sorted_family_groups))
            if expected_test != test:
                raise SplitError(
                    f"repeat {repeat} fold {expected_fold} splits a related model family"
                )

            if repeat > 0:
                if len(test) != target_sizes[expected_fold]:
                    raise SplitError(
                        f"repeat {repeat} fold {expected_fold} has the wrong model count"
                    )
                if not min_families <= len(families) <= max_families:
                    raise SplitError(
                        f"repeat {repeat} fold {expected_fold} has the wrong family count"
                    )
                stratum_counts: dict[str, int] = {}
                for family in families:
                    stratum = family_to_stratum[family]
                    stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
                if any(count > max_per_stratum for count in stratum_counts.values()):
                    raise SplitError(
                        f"repeat {repeat} fold {expected_fold} violates frozen stratum balance"
                    )
                fold_stratum_rank_mean = sum(
                    stratum_rank[family_to_stratum[family]] * family_sizes[family]
                    for family in families
                ) / len(test)
                if (
                    abs(fold_stratum_rank_mean - cohort_stratum_rank_mean)
                    > maximum_stratum_rank_deviation
                ):
                    raise SplitError(
                        f"repeat {repeat} fold {expected_fold} exceeds the frozen "
                        "model-weighted stratum-rank balance limit"
                    )

            validations: list[set[str]] = []
            for inner in outer["inner_folds"]:
                validation = set(inner["validation_model_ids"])
                fit = set(inner["fit_model_ids"])
                if test & (validation | fit):
                    raise SplitError(
                        f"outer-test model leaked in repeat {repeat}, fold {expected_fold}"
                    )
                if validation & fit or validation | fit != train:
                    raise SplitError(
                        f"invalid inner partition in repeat {repeat}, fold {expected_fold}"
                    )
                for family in inner["validation_family_ids"]:
                    if not family_groups[family] <= validation:
                        raise SplitError(
                            f"family {family} is split in repeat {repeat} inner validation"
                        )
                validations.append(validation)
            if len(validations) != 4:
                raise SplitError(f"repeat {repeat} fold {expected_fold} lacks four inner folds")
            if set().union(*validations) != train or sum(map(len, validations)) != len(train):
                raise SplitError(
                    f"repeat {repeat} fold {expected_fold} does not validate train models once"
                )

        if test_seen != all_models or family_seen != set(family_groups):
            raise SplitError(f"repeat {repeat} does not cover every model/family exactly once")
        canonical_partitions.append(_canonical_partition(family_folds))
    if len(set(canonical_partitions)) != len(canonical_partitions):
        raise SplitError("repeated partitions are not distinct")

    split = manifest["scenario_split"]
    administration = set(split["administration_scenario_ids"])
    evaluation = set(split["evaluation_scenario_ids"])
    if not administration or not evaluation or administration & evaluation:
        raise SplitError("scenario administration/evaluation split is invalid")
    expected_total = sum(
        int(counts["total"]) for counts in split["stratum_counts"].values()
    )
    if len(administration | evaluation) != expected_total:
        raise SplitError("scenario split does not cover its declared scenario count")


def _artifact_bytes(
    config_path: Path, manifest: dict[str, Any], model_csv: bytes, scenario_csv: bytes
) -> dict[Path, bytes]:
    outputs = _load_json(config_path)["outputs"]
    return {
        _resolve(str(outputs["manifest"])): canonical_json_bytes(manifest),
        _resolve(str(outputs["model_assignments_csv"])): model_csv,
        _resolve(str(outputs["scenario_assignments_csv"])): scenario_csv,
    }


def write_or_check(artifacts: dict[Path, bytes], *, write: bool) -> None:
    for path, content in artifacts.items():
        if write:
            if path.exists() and path.read_bytes() != content:
                raise SplitError(f"refusing to replace different frozen artifact: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        else:
            if not path.is_file():
                raise SplitError(f"frozen artifact is missing: {path}")
            if path.read_bytes() != content:
                raise SplitError(f"frozen artifact does not reproduce exactly: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--write",
        action="store_true",
        help="create missing artifacts; never overwrite different frozen content",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    manifest, model_csv, scenario_csv = build_artifacts(config_path)
    artifacts = _artifact_bytes(config_path, manifest, model_csv, scenario_csv)
    write_or_check(artifacts, write=args.write)
    action = "wrote" if args.write else "verified"
    for path in artifacts:
        print(f"{action}: {path.relative_to(ROOT)}")
    print("repetitions: 5; outer folds per repetition: 5; models per repetition: 52")
    print("scenario split: exact pinned v1 copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
