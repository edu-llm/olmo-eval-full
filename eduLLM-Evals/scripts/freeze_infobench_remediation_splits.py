#!/usr/bin/env python3
"""Freeze the leakage-safe nested-CV splits for InFoBench remediation v1.

This script does not fit IRT parameters or run CAT.  It only:

* verifies the pinned matrix/rubric/scenario inputs;
* assigns explicit, human-reviewed model-family groups to five balanced outer
  folds;
* uses the four remaining outer folds as the four inner validation folds;
* freezes a scenario-level 80/20 administration/evaluation split; and
* writes deterministic JSON/CSV artifacts after a structural leakage audit.

The response matrix is used solely to balance observed pass rates across folds.
Once written, the manifest is the immutable split boundary for the study.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "infobench_remediation_splits_v1.json"


class SplitError(RuntimeError):
    """Raised when inputs or frozen assignments violate the split contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SplitError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SplitError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def resolve_input_path(config_path: Path, configured: str, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    path = Path(configured)
    if path.is_absolute():
        return path
    # Repository-relative paths are intentional even when the config lives below
    # configs/.  This keeps the manifest portable to any checkout.
    return (ROOT / path).resolve()


def _verify_input(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise SplitError(f"{label} input does not exist: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise SplitError(
            f"{label} hash mismatch: expected {expected_sha256}, got {actual} ({path})"
        )


def _validate_matrix(matrix: pd.DataFrame, expected: dict[str, Any]) -> None:
    if matrix.index.has_duplicates:
        duplicates = matrix.index[matrix.index.duplicated()].tolist()
        raise SplitError(f"duplicate model IDs in matrix: {duplicates[:10]}")
    if matrix.columns.has_duplicates:
        duplicates = matrix.columns[matrix.columns.duplicated()].tolist()
        raise SplitError(f"duplicate criterion IDs in matrix: {duplicates[:10]}")
    if matrix.shape != (int(expected["models"]), int(expected["criteria"])):
        raise SplitError(
            "matrix shape mismatch: "
            f"expected {(expected['models'], expected['criteria'])}, got {matrix.shape}"
        )
    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    invalid = numeric.notna() & ~numeric.isin([0.0, 1.0])
    if bool(invalid.any().any()):
        row, column = np.argwhere(invalid.to_numpy())[0]
        raise SplitError(
            f"matrix contains non-binary value at {numeric.index[row]}/{numeric.columns[column]}"
        )
    if bool(numeric.notna().sum(axis=1).eq(0).any()):
        bad = numeric.index[numeric.notna().sum(axis=1).eq(0)].tolist()
        raise SplitError(f"cannot stratify model rows with no observed responses: {bad}")


def validate_family_groups(
    models: Iterable[str], family_groups: dict[str, list[str]]
) -> dict[str, str]:
    expected = {str(model) for model in models}
    seen: dict[str, str] = {}
    for family_id, members in family_groups.items():
        if not family_id or not isinstance(members, list) or not members:
            raise SplitError(f"family {family_id!r} must contain at least one model")
        for model in members:
            model = str(model)
            if model in seen:
                raise SplitError(
                    f"model {model!r} appears in both {seen[model]!r} and {family_id!r}"
                )
            seen[model] = str(family_id)
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    if missing or extra:
        raise SplitError(
            f"family roster differs from matrix (missing={missing}, extra={extra})"
        )
    return seen


def validate_frozen_grouped_folds(
    family_groups: dict[str, list[str]],
    pass_rates: pd.Series,
    frozen_family_folds: list[list[str]],
    target_fold_sizes: list[int],
    maximum_pass_rate_range: float,
) -> list[list[str]]:
    """Validate the preregistered, explicit family folds and their stratification."""

    if len(frozen_family_folds) < 2:
        raise SplitError("at least two outer folds are required")
    if len(frozen_family_folds) != len(target_fold_sizes):
        raise SplitError("frozen family-fold and model-count lists differ in length")
    if sum(target_fold_sizes) != len(pass_rates):
        raise SplitError(
            f"target fold sizes sum to {sum(target_fold_sizes)}, not {len(pass_rates)}"
        )
    assigned = [family for fold in frozen_family_folds for family in fold]
    duplicates = sorted({family for family in assigned if assigned.count(family) > 1})
    missing = sorted(set(family_groups) - set(assigned))
    extra = sorted(set(assigned) - set(family_groups))
    if duplicates or missing or extra:
        raise SplitError(
            "frozen family folds are not an exact partition "
            f"(duplicates={duplicates}, missing={missing}, extra={extra})"
        )

    folds = [sorted(fold) for fold in frozen_family_folds]
    observed_sizes: list[int] = []
    observed_means: list[float] = []
    for fold in folds:
        models = _family_models(fold, family_groups)
        observed_sizes.append(len(models))
        observed_means.append(float(pass_rates.loc[models].mean()))
    if observed_sizes != target_fold_sizes:
        raise SplitError(
            f"frozen outer sizes {observed_sizes} differ from targets {target_fold_sizes}"
        )
    pass_rate_range = max(observed_means) - min(observed_means)
    if pass_rate_range > maximum_pass_rate_range:
        raise SplitError(
            f"outer-fold pass-rate range {pass_rate_range:.6f} exceeds "
            f"the frozen tolerance {maximum_pass_rate_range:.6f}"
        )
    return folds


def _allocate_stratified_counts(
    strata: dict[str, list[str]], fraction: float
) -> dict[str, int]:
    target_total = int(round(sum(len(ids) for ids in strata.values()) * fraction))
    exact = {name: len(ids) * fraction for name, ids in strata.items()}
    counts = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = target_total - sum(counts.values())
    order = sorted(strata, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def split_scenarios(
    scenario_rows: list[dict[str, Any]],
    evaluation_fraction: float,
    seed: int,
    stratify_by: str,
) -> tuple[list[str], list[str], dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    if not 0.0 < evaluation_fraction < 1.0:
        raise SplitError("evaluation fraction must lie strictly between zero and one")
    by_id: dict[str, dict[str, Any]] = {}
    strata: dict[str, list[str]] = {}
    for row in scenario_rows:
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            raise SplitError("scenario record is missing scenario_id")
        if scenario_id in by_id:
            raise SplitError(f"duplicate scenario_id: {scenario_id}")
        by_id[scenario_id] = row
        stratum = str(row.get(stratify_by) or "__missing__")
        strata.setdefault(stratum, []).append(scenario_id)

    evaluation_counts = _allocate_stratified_counts(strata, evaluation_fraction)
    evaluation: set[str] = set()
    counts: dict[str, dict[str, int]] = {}
    for stratum, scenario_ids in sorted(strata.items()):
        ranked = sorted(
            scenario_ids,
            key=lambda scenario_id: hashlib.sha256(
                f"{seed}\0{stratum}\0{scenario_id}".encode()
            ).hexdigest(),
        )
        n_evaluation = evaluation_counts[stratum]
        evaluation.update(ranked[:n_evaluation])
        counts[stratum] = {
            "total": len(ranked),
            "administration": len(ranked) - n_evaluation,
            "evaluation": n_evaluation,
        }

    all_ids = set(by_id)
    administration = all_ids - evaluation
    return sorted(administration), sorted(evaluation), counts, by_id


def _family_models(
    family_ids: Iterable[str], family_groups: dict[str, list[str]]
) -> list[str]:
    return sorted(
        model for family_id in family_ids for model in family_groups[family_id]
    )


def build_manifest(
    config_path: Path,
    *,
    matrix_override: Path | None = None,
    rubrics_override: Path | None = None,
    scenarios_override: Path | None = None,
) -> tuple[dict[str, Any], bytes, bytes]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    expected = config["expectations"]

    matrix_path = resolve_input_path(
        config_path, inputs["response_matrix"]["path"], matrix_override
    )
    rubrics_path = resolve_input_path(
        config_path, inputs["rubrics"]["path"], rubrics_override
    )
    scenarios_path = resolve_input_path(
        config_path, inputs["scenarios"]["path"], scenarios_override
    )
    resolved = {
        "response_matrix": matrix_path,
        "rubrics": rubrics_path,
        "scenarios": scenarios_path,
    }
    for label, path in resolved.items():
        _verify_input(path, str(inputs[label]["sha256"]), label)

    matrix = pd.read_csv(matrix_path, index_col=0)
    matrix.index = matrix.index.map(str)
    matrix.columns = matrix.columns.map(str)
    _validate_matrix(matrix, expected)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    rubric_rows = read_jsonl(rubrics_path)
    scenario_rows = read_jsonl(scenarios_path)
    if len(rubric_rows) != int(expected["criteria"]):
        raise SplitError(
            f"expected {expected['criteria']} rubric records, found {len(rubric_rows)}"
        )
    if len(scenario_rows) != int(expected["scenarios"]):
        raise SplitError(
            f"expected {expected['scenarios']} scenarios, found {len(scenario_rows)}"
        )

    rubric_ids = [str(row.get("criterion_id") or "") for row in rubric_rows]
    if len(set(rubric_ids)) != len(rubric_ids) or "" in rubric_ids:
        raise SplitError("rubric criterion IDs are missing or duplicated")
    if set(rubric_ids) != set(matrix.columns):
        raise SplitError("matrix and rubric criterion IDs do not match exactly")

    family_groups = {
        str(family): [str(model) for model in members]
        for family, members in config["model_families"].items()
    }
    model_to_family = validate_family_groups(matrix.index, family_groups)
    pass_rates = matrix.mean(axis=1, skipna=True)

    cv = config["cross_validation"]
    target_sizes = [int(value) for value in cv["outer_fold_model_counts"]]
    if len(target_sizes) != int(cv["outer_folds"]):
        raise SplitError("outer fold count and target-size list disagree")
    outer_family_folds = validate_frozen_grouped_folds(
        family_groups,
        pass_rates,
        [[str(value) for value in fold] for fold in cv["outer_fold_family_ids"]],
        target_sizes,
        float(cv["maximum_outer_fold_mean_pass_rate_range"]),
    )
    outer_model_folds = [
        _family_models(families, family_groups) for families in outer_family_folds
    ]
    model_to_outer = {
        model: fold
        for fold, models in enumerate(outer_model_folds)
        for model in models
    }

    all_models = set(matrix.index)
    outer_records: list[dict[str, Any]] = []
    for outer_fold in range(len(outer_model_folds)):
        outer_test = set(outer_model_folds[outer_fold])
        outer_train = all_models - outer_test
        inner_records: list[dict[str, Any]] = []
        remaining_outer_folds = [
            source for source in range(len(outer_model_folds)) if source != outer_fold
        ]
        if len(remaining_outer_folds) != int(cv["inner_folds"]):
            raise SplitError(
                "inner-fold design requires inner_folds == outer_folds - 1"
            )
        for inner_fold, source_outer_fold in enumerate(remaining_outer_folds):
            validation = set(outer_model_folds[source_outer_fold])
            fit = outer_train - validation
            inner_records.append(
                {
                    "inner_fold": inner_fold,
                    "source_outer_fold": source_outer_fold,
                    "validation_family_ids": outer_family_folds[source_outer_fold],
                    "validation_model_ids": sorted(validation),
                    "fit_family_ids": sorted(
                        family_id
                        for family_id, fold in {
                            family: f
                            for f, families in enumerate(outer_family_folds)
                            for family in families
                        }.items()
                        if fold not in {outer_fold, source_outer_fold}
                    ),
                    "fit_model_ids": sorted(fit),
                }
            )
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "test_family_ids": outer_family_folds[outer_fold],
                "test_model_ids": sorted(outer_test),
                "train_family_ids": sorted(
                    family
                    for fold, families in enumerate(outer_family_folds)
                    if fold != outer_fold
                    for family in families
                ),
                "train_model_ids": sorted(outer_train),
                "observed_test_pass_rate_mean": float(pass_rates.loc[sorted(outer_test)].mean()),
                "inner_folds": inner_records,
            }
        )

    administration, evaluation, stratum_counts, scenario_by_id = split_scenarios(
        scenario_rows,
        float(cv["evaluation_scenario_fraction"]),
        int(cv["scenario_split_seed"]),
        str(cv["scenario_stratify_by"]),
    )
    scenario_ids = set(scenario_by_id)
    rubric_scenarios = {str(row.get("scenario_id") or "") for row in rubric_rows}
    if rubric_scenarios != scenario_ids:
        raise SplitError(
            "rubric and scenario banks disagree on scenario IDs: "
            f"rubric_only={sorted(rubric_scenarios - scenario_ids)[:10]}, "
            f"scenario_only={sorted(scenario_ids - rubric_scenarios)[:10]}"
        )

    manifest = {
        "schema_version": "infobench-remediation-nested-splits-v1",
        "benchmark": "InFoBench",
        "generated_by": "scripts/freeze_infobench_remediation_splits.py",
        "configuration_sha256": sha256_file(config_path),
        "inputs": {
            label: {
                "configured_path": str(inputs[label]["path"]),
                "sha256": str(inputs[label]["sha256"]),
            }
            for label in ("response_matrix", "rubrics", "scenarios")
        },
        "construction": {
            "outer": {
                "folds": int(cv["outer_folds"]),
                "seed": int(cv["seed"]),
                "strategy": "explicit_frozen_family_groups_balanced_on_observed_pass_rate_v1",
                "target_model_counts": target_sizes,
                "stratification_value": "observed_model_pass_rate",
                "maximum_mean_pass_rate_range": float(
                    cv["maximum_outer_fold_mean_pass_rate_range"]
                ),
            },
            "inner": {
                "folds_per_outer": int(cv["inner_folds"]),
                "strategy": "each_remaining_outer_family_fold_once",
            },
            "scenarios": {
                "seed": int(cv["scenario_split_seed"]),
                "strategy": "sha256_rank_within_metadata_stratum_v1",
                "stratify_by": str(cv["scenario_stratify_by"]),
                "administration_fraction": 1.0
                - float(cv["evaluation_scenario_fraction"]),
                "evaluation_fraction": float(cv["evaluation_scenario_fraction"]),
            },
        },
        "model_families": {key: family_groups[key] for key in sorted(family_groups)},
        "model_to_family": {key: model_to_family[key] for key in sorted(model_to_family)},
        "model_to_outer_fold": {key: model_to_outer[key] for key in sorted(model_to_outer)},
        "outer_folds": outer_records,
        "scenario_split": {
            "administration_scenario_ids": administration,
            "evaluation_scenario_ids": evaluation,
            "stratum_counts": stratum_counts,
        },
        "leakage_contract": [
            (
                "An outer-test model may not appear in that fold's item fit, inner fit, "
                "or inner validation sets."
            ),
            (
                "Each outer-training model is inner-validation data exactly once and "
                "inner-fit data in every other inner fold."
            ),
            (
                "Every explicit model family belongs to exactly one outer fold and "
                "therefore remains intact in all inner folds."
            ),
            (
                "Evaluation-scenario responses may be scored only after CAT "
                "administration and may not alter the CAT path."
            ),
        ],
    }
    audit_manifest(manifest)

    model_rows: list[dict[str, Any]] = []
    for outer in outer_records:
        validation_fold = {
            model: inner["inner_fold"]
            for inner in outer["inner_folds"]
            for model in inner["validation_model_ids"]
        }
        for model in sorted(all_models):
            is_test = model in set(outer["test_model_ids"])
            model_rows.append(
                {
                    "outer_fold": outer["outer_fold"],
                    "model_id": model,
                    "family_id": model_to_family[model],
                    "outer_role": "test" if is_test else "train",
                    "inner_validation_fold": "" if is_test else validation_fold[model],
                    "observed_pass_rate": f"{float(pass_rates.loc[model]):.12f}",
                }
            )
    model_csv = _csv_bytes(
        model_rows,
        [
            "outer_fold",
            "model_id",
            "family_id",
            "outer_role",
            "inner_validation_fold",
            "observed_pass_rate",
        ],
    )

    criterion_counts: dict[str, int] = {}
    for row in rubric_rows:
        scenario_id = str(row["scenario_id"])
        criterion_counts[scenario_id] = criterion_counts.get(scenario_id, 0) + 1
    evaluation_set = set(evaluation)
    scenario_rows_csv = [
        {
            "scenario_id": scenario_id,
            "subset": str(scenario_by_id[scenario_id].get(cv["scenario_stratify_by"]) or ""),
            "role": "evaluation" if scenario_id in evaluation_set else "administration",
            "criterion_count": criterion_counts[scenario_id],
        }
        for scenario_id in sorted(scenario_by_id)
    ]
    scenario_csv = _csv_bytes(
        scenario_rows_csv, ["scenario_id", "subset", "role", "criterion_count"]
    )
    return manifest, model_csv, scenario_csv


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def audit_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed if any model/family/scenario leakage is present."""

    family_groups = {
        family: set(models) for family, models in manifest["model_families"].items()
    }
    all_models = set(manifest["model_to_family"])
    if set().union(*family_groups.values()) != all_models:
        raise SplitError("family groups do not cover the model roster exactly")

    outer_folds = manifest["outer_folds"]
    test_seen: set[str] = set()
    family_fold: dict[str, int] = {}
    for outer in outer_folds:
        outer_index = int(outer["outer_fold"])
        test = set(outer["test_model_ids"])
        train = set(outer["train_model_ids"])
        if test & train or test | train != all_models:
            raise SplitError(f"outer fold {outer_index} train/test partition is invalid")
        if test_seen & test:
            raise SplitError("a model appears in more than one outer-test fold")
        test_seen.update(test)

        for family_id in outer["test_family_ids"]:
            if family_id in family_fold:
                raise SplitError(f"family {family_id} appears in multiple outer folds")
            family_fold[family_id] = outer_index
            if not family_groups[family_id] <= test:
                raise SplitError(f"family {family_id} is split across outer folds")

        validations: list[set[str]] = []
        for inner in outer["inner_folds"]:
            validation = set(inner["validation_model_ids"])
            fit = set(inner["fit_model_ids"])
            if test & (validation | fit):
                raise SplitError(
                    f"outer-test model leaked into outer {outer_index}, inner {inner['inner_fold']}"
                )
            if validation & fit or validation | fit != train:
                raise SplitError(
                    f"outer {outer_index}, inner {inner['inner_fold']} is not a train partition"
                )
            for family_id in inner["validation_family_ids"]:
                members = family_groups[family_id]
                if not members <= validation:
                    raise SplitError(
                        f"family {family_id} is split in outer {outer_index} inner validation"
                    )
            validations.append(validation)
        validation_union = set().union(*validations)
        validation_total = sum(len(values) for values in validations)
        if validation_union != train or validation_total != len(train):
            raise SplitError(
                f"outer fold {outer_index} does not validate each training model exactly once"
            )

    if test_seen != all_models:
        raise SplitError("outer-test folds do not cover every model exactly once")
    if set(family_fold) != set(family_groups):
        raise SplitError("outer folds do not cover every family exactly once")

    split = manifest["scenario_split"]
    administration = set(split["administration_scenario_ids"])
    evaluation = set(split["evaluation_scenario_ids"])
    if not administration or not evaluation or administration & evaluation:
        raise SplitError("scenario administration/evaluation split is invalid")
    expected_total = sum(
        int(counts["total"]) for counts in split["stratum_counts"].values()
    )
    if len(administration | evaluation) != expected_total:
        raise SplitError("scenario split does not cover the declared scenario count")


def _artifact_bytes(
    config_path: Path,
    manifest: dict[str, Any],
    model_csv: bytes,
    scenario_csv: bytes,
) -> dict[Path, bytes]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    return {
        (ROOT / outputs["manifest"]).resolve(): canonical_json_bytes(manifest),
        (ROOT / outputs["model_assignments_csv"]).resolve(): model_csv,
        (ROOT / outputs["scenario_assignments_csv"]).resolve(): scenario_csv,
    }


def write_or_check(artifacts: dict[Path, bytes], *, write: bool) -> None:
    for path, content in artifacts.items():
        if write:
            if path.exists() and path.read_bytes() != content:
                raise SplitError(
                    f"refusing to replace frozen artifact with different content: {path}"
                )
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
    parser.add_argument("--matrix", type=Path, default=None)
    parser.add_argument("--rubrics", type=Path, default=None)
    parser.add_argument("--scenarios", type=Path, default=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help="create missing frozen artifacts; existing different files are never overwritten",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    manifest, model_csv, scenario_csv = build_manifest(
        config_path,
        matrix_override=args.matrix,
        rubrics_override=args.rubrics,
        scenarios_override=args.scenarios,
    )
    artifacts = _artifact_bytes(config_path, manifest, model_csv, scenario_csv)
    write_or_check(artifacts, write=args.write)
    action = "wrote" if args.write else "verified"
    for path in artifacts:
        print(f"{action}: {path.relative_to(ROOT)}")
    print(
        f"outer fold sizes: {[len(fold['test_model_ids']) for fold in manifest['outer_folds']]}"
    )
    print(
        "scenario split: "
        f"{len(manifest['scenario_split']['administration_scenario_ids'])} administration / "
        f"{len(manifest['scenario_split']['evaluation_scenario_ids'])} evaluation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
