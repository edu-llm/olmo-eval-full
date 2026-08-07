"""Tests for the read-only InFoBench baseline reproducer."""

from __future__ import annotations

import csv
import io
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "verify_infobench_baseline.py"
SPEC = importlib.util.spec_from_file_location("verify_infobench_baseline", SCRIPT)
assert SPEC and SPEC.loader
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture(root: Path) -> Path:
    rubrics = [
        {"criterion_id": "c1", "scenario_id": "s1"},
        {"criterion_id": "c2", "scenario_id": "s1"},
        {"criterion_id": "c3", "scenario_id": "s2"},
    ]
    scenarios = [
        {"scenario_id": "s1", "criterion_ids": ["c1", "c2"]},
        {"scenario_id": "s2", "criterion_ids": ["c3"]},
    ]
    _write_jsonl(root / "rubrics.jsonl", rubrics)
    _write_jsonl(root / "scenarios.jsonl", scenarios)
    _write_csv(
        root / "matrix.csv",
        ["model", "c1", "c2", "c3"],
        [
            {"model": "m1", "c1": 1, "c2": 0, "c3": 0},
            {"model": "m2", "c1": 1, "c2": 1, "c3": 0},
            {"model": "m3", "c1": 0, "c2": "", "c3": 0},
        ],
    )

    judge_expected = {
        "judge_name": "qwen",
        "judge_model": "qwen-test",
        "judge_revision": "rev",
        "adapter": "generic-binary",
        "prompt_version": "prompt",
        "normalization_version": "normalization",
        "evidence_policy_version": "evidence",
        "prompt_variant": "canonical",
        "replicate_id": "r1",
    }
    judge_observed = {key: [value] for key, value in judge_expected.items()}
    judge_observed.update(
        {"configuration_hash": ["configuration"], "frozen_configuration_hash": ["frozen"]}
    )
    _write_json(
        root / "judge_manifest.json",
        {
            "judge_expected": judge_expected,
            "judge_observed": judge_observed,
            "counts": {
                "total_cells": 9,
                "this_run": {"no_decision_policy": "missing"},
            },
            "coverage": {"n_holes": 1, "n_filled": 8},
            "models": ["m1", "m2", "m3"],
            "n_criteria": 3,
        },
    )

    _write_csv(
        root / "fit.csv",
        ["criterion_id", "a_instruction_following", "b"],
        [
            {"criterion_id": "c1", "a_instruction_following": 1.2, "b": 0.1},
            {"criterion_id": "c2", "a_instruction_following": -0.2, "b": -0.1},
        ],
    )
    _write_jsonl(root / "exported.jsonl", [rubrics[0]])
    _write_json(
        root / "export_manifest.json",
        {
            "counts": {
                "source_criteria": 3,
                "calibration_rows": 2,
                "exported_criteria": 1,
                "excluded_criteria": 2,
            },
            "exclusion_reason_counts": {"nonpositive_a": 1, "unfitted": 1},
            "excluded_criteria": [
                {"criterion_id": "c2", "reasons": ["nonpositive_a"]},
                {"criterion_id": "c3", "reasons": ["unfitted"]},
            ],
            "inputs": {
                "sha256": {
                    "calibration_csv": baseline.sha256_file(root / "fit.csv"),
                    "rubrics": baseline.sha256_file(root / "rubrics.jsonl"),
                }
            },
            "outputs": {
                "sha256": {"rubrics": baseline.sha256_file(root / "exported.jsonl")}
            },
        },
    )

    structure_fields = [
        "structure",
        "n_dims",
        "primary_metric",
        "cv_mean",
        "cv_se",
        "core_converged",
        "all_fold_fits_converged",
        "selected",
    ]
    _write_csv(
        root / "structure.csv",
        structure_fields,
        [
            {
                "structure": "overall_1d",
                "n_dims": 1,
                "primary_metric": "log_loss",
                "cv_mean": 0.4,
                "cv_se": 0.02,
                "core_converged": True,
                "all_fold_fits_converged": True,
                "selected": True,
            },
            {
                "structure": "full_2d",
                "n_dims": 2,
                "primary_metric": "log_loss",
                "cv_mean": 0.39,
                "cv_se": 0.03,
                "core_converged": True,
                "all_fold_fits_converged": True,
                "selected": False,
            },
        ],
    )
    _write_json(
        root / "structure_selection.json",
        {"selected_structure": "overall_1d", "one_standard_error_threshold": 0.42},
    )

    cat_fields = [
        "model",
        "scenarios_administered",
        "criteria_administered",
        "precision_reached",
        "mwle_converged",
        "theta_full_eap_instruction_following",
        "theta_mwle_instruction_following",
    ]
    selected_cat = [
        {
            "model": model,
            "scenarios_administered": scenarios_administered,
            "criteria_administered": scenarios_administered * 2,
            "precision_reached": True,
            "mwle_converged": True,
            "theta_full_eap_instruction_following": reference,
            "theta_mwle_instruction_following": estimate,
        }
        for model, scenarios_administered, reference, estimate in zip(
            ("m1", "m2", "m3"), (2, 3, 4), (-1.0, 0.0, 1.0), (-0.8, 0.1, 1.2)
        )
    ]
    random_cat = [
        {
            "model": model,
            "scenarios_administered": scenarios_administered,
            "criteria_administered": scenarios_administered * 2,
            "precision_reached": True,
            "mwle_converged": True,
            "theta_full_eap_instruction_following": reference,
            "theta_mwle_instruction_following": estimate,
        }
        for model, scenarios_administered, reference, estimate in zip(
            ("m1", "m2", "m3"), (4, 5, 6), (-1.0, 0.0, 1.0), (-0.5, 0.0, 0.5)
        )
    ]
    _write_csv(root / "selected_cat.csv", cat_fields, selected_cat)
    _write_csv(root / "random_cat.csv", cat_fields, random_cat)
    selected_recovery = baseline._recovery(
        [-1.0, 0.0, 1.0], [-0.8, 0.1, 1.2]
    )
    _write_json(
        root / "cat_configuration.json",
        {
            "selected_min_scenarios": 0,
            "selected_se_target": 0.25,
            "selected_selection_rule": "trace",
        },
    )
    _write_json(
        root / "cat_validation.json",
        {
            "metrics": {
                "mean_scenarios": 3.0,
                "mwle_recovery_r": selected_recovery["r"],
                "mwle_recovery_slope": selected_recovery["slope"],
                "precision_rate": 1.0,
            }
        },
    )

    oos_fields = [
        "model",
        "status",
        "mwle_converged",
        "theta_ref_instruction_following",
        "theta_eap_instruction_following",
        "theta_mwle_instruction_following",
        "theta_online_instruction_following",
    ]
    references = [-1.0, 0.0, 1.0]
    estimates = {
        "eap": [-0.9, 0.1, 1.0],
        "mwle": [-0.8, 0.0, 1.1],
        "online": [-0.7, 0.2, 0.8],
    }
    _write_csv(
        root / "oos.csv",
        oos_fields,
        [
            {
                "model": model,
                "status": "ok",
                "mwle_converged": True,
                "theta_ref_instruction_following": references[index],
                "theta_eap_instruction_following": estimates["eap"][index],
                "theta_mwle_instruction_following": estimates["mwle"][index],
                "theta_online_instruction_following": estimates["online"][index],
            }
            for index, model in enumerate(("m1", "m2", "m3"))
        ],
    )
    metric_fields = ["estimator", "skill", "metric", "estimate", "n_models"]
    metric_rows: list[dict[str, object]] = []
    for estimator, estimator_values in estimates.items():
        for metric, value in baseline._recovery(references, estimator_values).items():
            metric_rows.append(
                {
                    "estimator": estimator,
                    "skill": "instruction_following",
                    "metric": metric,
                    "estimate": value,
                    "n_models": 3,
                }
            )
    _write_csv(root / "oos_metrics.csv", metric_fields, metric_rows)

    uncertainty_fields = [
        "se_target",
        "model",
        "estimator",
        "dimension",
        "se_total",
        "reliable",
    ]
    _write_csv(
        root / "uncertainty.csv",
        uncertainty_fields,
        [
            {
                "se_target": 0.25,
                "model": model,
                "estimator": "mwle",
                "dimension": "instruction_following",
                "se_total": value,
                "reliable": True,
            }
            for model, value in zip(("m1", "m2", "m3"), (0.4, 0.5, 0.6))
        ],
    )
    _write_csv(
        root / "uncertainty_summary.csv",
        ["se_target", "estimator", "dimension", "n_reliable_models", "median_se_total"],
        [
            {
                "se_target": 0.25,
                "estimator": "mwle",
                "dimension": "instruction_following",
                "n_reliable_models": 3,
                "median_se_total": 0.5,
            }
        ],
    )

    config = {
        "schema_version": baseline.SCHEMA_VERSION,
        "frozen_inputs": {
            name: {"path": path, "sha256": baseline.sha256_file(root / path)}
            for name, path in {
                "response_matrix": "matrix.csv",
                "judge_manifest": "judge_manifest.json",
                "rubrics": "rubrics.jsonl",
                "scenarios": "scenarios.jsonl",
            }.items()
        },
        "stored_artifacts": {
            "export_manifest": "export_manifest.json",
            "selected_calibration_csv": "fit.csv",
            "exported_rubrics": "exported.jsonl",
            "structure_comparison": "structure.csv",
            "structure_selection": "structure_selection.json",
            "cat_configuration": "cat_configuration.json",
            "cat_validation": "cat_validation.json",
            "selected_cat_per_model": "selected_cat.csv",
            "random_per_model": "random_cat.csv",
            "oos_per_model": "oos.csv",
            "oos_metrics": "oos_metrics.csv",
            "uncertainty_components": "uncertainty.csv",
            "uncertainty_summary": "uncertainty_summary.csv",
        },
        "outputs": {
            "report": "report.json",
            "criterion_flow_csv": "criterion_flow.csv",
        },
        "expected": {
            "matrix": {"models": 3, "criteria": 3, "scenarios": 2, "missing_cells": 1},
            "criterion_flow": {
                "source_criteria": 3,
                "fitted_criteria": 2,
                "exported_criteria": 1,
                "exclusions": {"unfitted": 1, "nonpositive_discrimination": 1},
            },
            "headline_metrics": {
                "structure_selection": {
                    "selected_structure": "overall_1d",
                    "best_mean_structure": "full_2d",
                },
                "mwle_uncertainty": {"median_total_se": 0.5},
            },
        },
    }
    config_path = root / "config.json"
    _write_json(config_path, config)
    return config_path


def _add_phase0_bundle(root: Path, config_path: Path) -> None:
    payload_bytes = {
        "inputs/response_matrix.csv": (root / "matrix.csv").read_bytes(),
        "inputs/judge_manifest.json": (root / "judge_manifest.json").read_bytes(),
        "inputs/normalized_verdicts.jsonl": b"{}\n",
        **{
            f"inputs/responses/{model}.responses.jsonl": b"{}\n"
            for model in ("m1", "m2", "m3")
        },
    }
    archive_path = root / "bundle.tar.gz"
    with tarfile.open(archive_path, "w:gz") as handle:
        for logical_name, data in sorted(payload_bytes.items()):
            member = tarfile.TarInfo(f"fixture/{logical_name}")
            member.size = len(data)
            member.mode = 0o444
            handle.addfile(member, io.BytesIO(data))
    payload = {
        name: {"sha256": baseline.hashlib.sha256(data).hexdigest()}
        for name, data in payload_bytes.items()
    }
    judge = json.loads((root / "judge_manifest.json").read_text(encoding="utf-8"))
    _write_json(
        root / "phase0_artifact.json",
        {
            "schema_version": "infobench-phase0-input-bundle-v1",
            "archive": {
                "sha256": baseline.sha256_file(archive_path),
                "bytes": archive_path.stat().st_size,
                "members": len(payload),
            },
            "payload": payload,
            "cohort": {
                "models": 3,
                "criteria": 3,
                "matrix_cells": 9,
                "matrix_value_counts": {"0": 5, "1": 3, "missing": 1},
                "response_shards": [
                    {"model": model, "rows": 500} for model in ("m1", "m2", "m3")
                ],
            },
            "judge_provenance": judge["judge_expected"],
            "criterion_count_flow": {
                "source_criteria": 3,
                "fitted_criteria": 2,
                "exported_criteria": 1,
                "source_to_fit": {"excluded_all_fail": 1},
                "fit_to_export": {"excluded_nonpositive_discrimination": 1},
            },
            "storage": {"status": "test", "uri": "test://bundle"},
        },
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["stored_artifacts"].update(
        {
            "phase0_input_artifact": "phase0_artifact.json",
            "phase0_input_archive": "bundle.tar.gz",
        }
    )
    config["expected"]["phase0_input_bundle"] = {
        "archive_members": len(payload),
        "response_shards": 3,
        "normalized_verdicts_present": True,
    }
    _write_json(config_path, config)


def test_end_to_end_reproduction_is_deterministic_and_checks_outputs(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)

    first_report, first_flow = baseline.reproduce_baseline(tmp_path, config_path)
    report_path, flow_path = baseline.output_paths(tmp_path, config_path)
    baseline.write_outputs(report_path, flow_path, first_report, first_flow)
    first_bytes = (report_path.read_bytes(), flow_path.read_bytes())

    second_report, second_flow = baseline.reproduce_baseline(tmp_path, config_path)
    baseline.check_outputs(report_path, flow_path, second_report, second_flow)

    assert first_report == second_report
    assert first_bytes == (report_path.read_bytes(), flow_path.read_bytes())
    assert first_report["criterion_flow"]["arithmetic"] == {
        "source_minus_unfitted_equals_fitted": 2,
        "fitted_minus_nonpositive_equals_exported": 1,
    }
    assert [row["criteria"] for row in first_flow] == [3, 2, 1]


def test_frozen_input_hash_tampering_fails_before_reproduction(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    with (tmp_path / "matrix.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(baseline.BaselineVerificationError, match="response_matrix SHA-256"):
        baseline.reproduce_baseline(tmp_path, config_path)


def test_stale_stored_report_is_rejected(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    report, flow = baseline.reproduce_baseline(tmp_path, config_path)
    report_path, flow_path = baseline.output_paths(tmp_path, config_path)
    baseline.write_outputs(report_path, flow_path, report, flow)
    flow_path.write_text("stale\n", encoding="utf-8")

    with pytest.raises(baseline.BaselineVerificationError, match="stale or modified"):
        baseline.check_outputs(report_path, flow_path, report, flow)


def test_phase0_archive_payload_is_cross_checked(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    _add_phase0_bundle(tmp_path, config_path)

    report, _ = baseline.reproduce_baseline(tmp_path, config_path)

    assert report["phase0_input_bundle"]["archive_root"] == "fixture"
    assert report["phase0_input_bundle"]["payload_members_verified"] == 6
    assert report["phase0_input_bundle"]["response_shards"] == 3


def test_checked_in_reproduction_report_locks_count_flow_and_headlines() -> None:
    report = json.loads(
        (ROOT / "reports/infobench_calibration_20260804/baseline_reproduction.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "verified"
    assert report["criterion_flow"] == {
        "arithmetic": {
            "fitted_minus_nonpositive_equals_exported": 2096,
            "source_minus_unfitted_equals_fitted": 2105,
        },
        "exclusions": {"nonpositive_discrimination": 9, "unfitted": 145},
        "exported_criteria": 2096,
        "fitted_criteria": 2105,
        "identities_reconciled": True,
        "source_criteria": 2250,
        "total_excluded": 154,
    }
    headlines = report["headline_metrics"]
    assert headlines["structure_selection"]["selected_structure"] == "overall_1d"
    assert headlines["cat_vs_random"]["selected_cat"]["mean_scenarios"] == pytest.approx(
        5.288461538461538
    )
    assert headlines["cat_vs_random"]["random_baseline"]["mean_scenarios"] == pytest.approx(
        23.692307692307693
    )
    assert headlines["mwle_uncertainty"]["median_total_se"] == pytest.approx(
        0.536703549746096
    )
