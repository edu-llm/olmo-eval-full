"""Synthetic tests for scenario-level person k-fold estimator recovery."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import scenario_kfold_estimator_cv as scenario_cv
from tutor_cat.skill_structure import SkillStructure


def _study_files(tmp_path):
    source_skills = ("content", "format", "style")
    records = []
    scenarios = []
    criterion_ids = []
    for j in range(12):
        sid = f"s{j:02d}"
        cid = f"{sid}_c01"
        source = source_skills[j % len(source_skills)]
        q = {skill: int(skill == source) for skill in source_skills}
        records.append(
            {
                "criterion_id": cid,
                "scenario_id": sid,
                "criterion": f"criterion {j}",
                "q_mapping": q,
                "difficulty": 0.0,
                "discrimination": {skill: float(q[skill]) for skill in source_skills},
                "status": "approved",
            }
        )
        scenarios.append(
            {"scenario_id": sid, "prompt": sid, "criterion_ids": [cid], "modality": "text"}
        )
        criterion_ids.append(cid)

    # Deterministic but heterogeneous response profiles; each fold contains a broad
    # pass-rate range under stratification.
    models = [f"model-{i:02d}" for i in range(10)]
    values = np.empty((len(models), len(criterion_ids)), dtype=int)
    for i in range(len(models)):
        for j in range(len(criterion_ids)):
            values[i, j] = int(((3 * i + 2 * j + (j % 3) * i) % 11) < (2 + i))
    matrix = pd.DataFrame(values, index=models, columns=criterion_ids)
    matrix.index.name = "model"

    rubric_path = tmp_path / "rubrics.jsonl"
    scenario_path = tmp_path / "scenarios.jsonl"
    matrix_path = tmp_path / "matrix.csv"
    rubric_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    scenario_path.write_text(
        "".join(json.dumps(r) + "\n" for r in scenarios), encoding="utf-8"
    )
    matrix.to_csv(matrix_path)
    return source_skills, records, matrix, matrix_path, rubric_path, scenario_path


def test_build_fold_bank_applies_arbitrary_structure_and_explicit_negative_policy():
    source = ("content", "format", "number", "style", "linguistic")
    structure = SkillStructure.from_groups(
        "two_axis",
        source,
        [
            ("content_style", ("content", "style")),
            ("instruction", ("format", "number", "linguistic")),
        ],
    )
    records = {}
    for j, skill in enumerate(source):
        cid = f"c{j}"
        records[cid] = {
            "criterion_id": cid,
            "scenario_id": f"s{j}",
            "criterion": cid,
            "q_mapping": {s: int(s == skill) for s in source},
        }
    fit = {
        "items": list(records),
        "A": np.array([[1.0, 0.0], [0.0, 1.1], [0.0, 1.2], [-0.4, 0.0], [0.0, 1.3]]),
        "b": np.linspace(-0.5, 0.5, 5),
        "R": np.array([[1.0, 0.3], [0.3, 1.0]]),
        "dim_labels": list(structure.labels),
    }
    with pytest.raises(scenario_cv.scat.OfflineStudyError, match="nonpositive"):
        scenario_cv.build_fold_bank(fit, structure, records, negative_policy="error")
    bank, diag = scenario_cv.build_fold_bank(fit, structure, records, negative_policy="drop")
    assert bank.dims == ("content_style", "instruction")
    assert bank.n_items == 4
    assert "c3" not in bank.criterion_ids
    assert diag["n_nonpositive_items"] == 1
    assert np.allclose(bank.latent_correlation, fit["R"])


def test_recovery_stats_reports_compression_error_and_signed_bias():
    reference = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    estimate = 0.6 * reference + 0.2
    stats = scenario_cv.recovery_stats(reference, estimate)
    assert stats["r"] == pytest.approx(1.0)
    assert stats["slope"] == pytest.approx(0.6)
    assert stats["bias"] == pytest.approx(0.2)
    assert stats["mae"] > 0.0
    pass_rate = scenario_cv.pass_rate_stats(
        [0.2, 0.5, 0.8], [0.3, 0.5, 0.7]
    )
    assert pass_rate["mae"] == pytest.approx(0.2 / 3.0)
    assert pass_rate["bias"] == pytest.approx(0.0)
    assert pass_rate["r"] == pytest.approx(1.0)


def test_recovery_table_excludes_failed_mwle_rows():
    frame = pd.DataFrame(
        {
            "fold": [0, 0, 1, 1],
            "status": ["ok"] * 4,
            "mwle_converged": [True, False, True, False],
            "theta_ref_axis": [-1.0, -0.2, 0.4, 1.2],
            "theta_online_axis": [-0.8, -0.1, 0.3, 0.9],
            "theta_eap_axis": [-0.9, -0.1, 0.35, 1.0],
            "theta_mwle_axis": [-1.0, 99.0, 0.4, 99.0],
        }
    )
    table = scenario_cv.recovery_table(frame, ("axis",))
    pooled_mwle = table[(table["fold"] == "pooled") & (table.estimator == "mwle")].iloc[0]
    assert pooled_mwle["n"] == 2
    assert pooled_mwle["mae"] == pytest.approx(0.0)


def test_full_study_refits_only_training_models_and_writes_recovery_outputs(
    tmp_path, monkeypatch
):
    source_skills, records, matrix, matrix_path, rubric_path, scenario_path = _study_files(
        tmp_path
    )
    fit_calls: list[set[str]] = []

    def fake_fit_structure(train_df, q_by, args, structure):
        fit_calls.append(set(train_df.index))
        items = list(train_df.columns)
        q_source = np.asarray([q_by[cid] for cid in items], dtype=int)
        q_modeled = structure.transform_q(q_source)
        A = q_modeled.astype(float) * 1.25
        b = np.linspace(-0.8, 0.8, len(items))
        return {
            "items": items,
            "A": A,
            "b": b,
            "R": np.array([[1.0, 0.2], [0.2, 1.0]]),
            "dim_labels": list(structure.labels),
            "loglik": -123.0,
            "n_params": int(np.count_nonzero(q_modeled) + len(items)),
            "n_iter": 4,
            "converged": True,
            "diag": {},
        }

    monkeypatch.setattr(scenario_cv.cell_cv, "fit_structure", fake_fit_structure)
    out_dir = tmp_path / "out"
    args = scenario_cv.build_argparser().parse_args(
        [
            "--matrix",
            str(matrix_path),
            "--rubrics",
            str(rubric_path),
            "--scenarios",
            str(scenario_path),
            "--out-dir",
            str(out_dir),
            "--skills",
            ",".join(source_skills),
            "--dimensions",
            "content_style=content+style,format=format",
            "--structure-name",
            "test_two_axis",
            "--k",
            "2",
            "--fit-grid",
            "3",
            "--eap-grid",
            "3",
            "--top-n",
            "1",
            "--max-se",
            "0.01",
            "--min-evals-per-skill",
            "0",
            "--max-scenarios",
            "12",
        ]
    )
    try:
        assert scenario_cv.run(args) == 0
    finally:
        # ``calibrate_mirt`` keeps the configured source axis process-locally.
        scenario_cv.cm.configure_skills(None)

    assignments = json.loads((out_dir / "fold_assignments.json").read_text())
    assert len(fit_calls) == 2
    for fold_index, train_models in enumerate(fit_calls):
        held_out = set(assignments["folds"][str(fold_index)])
        assert train_models.isdisjoint(held_out)
        assert train_models | held_out == set(matrix.index)

    expected = {
        "oos_per_model.csv",
        "recovery_per_fold.csv",
        "pass_rate_calibration.csv",
        "metrics_aggregate.json",
        "fold_assignments.csv",
        "fold_assignments.json",
        "manifest.json",
        "fold_0_item_params.csv",
        "fold_1_item_params.csv",
    }
    assert expected <= {path.name for path in out_dir.iterdir()}
    per_model = pd.read_csv(out_dir / "oos_per_model.csv")
    assert len(per_model) == len(matrix)
    assert set(per_model["status"]) == {"ok"}
    assert set(per_model["fold"]) == {0, 1}
    recovery = pd.read_csv(out_dir / "recovery_per_fold.csv")
    assert set(recovery["estimator"]) == {"online", "eap", "mwle"}
    assert set(recovery["dimension"]) == {"content_style", "format"}
    assert {"r", "slope", "mae", "bias"} <= set(recovery.columns)
    assert {
        "observed_fitted_pass_rate",
        "pirt_predicted_pass_rate_online",
        "pirt_predicted_pass_rate_eap",
        "pirt_predicted_pass_rate_mwle",
        "pirt_predicted_pass_rate_full_eap",
    } <= set(per_model.columns)
    pass_rate = pd.read_csv(out_dir / "pass_rate_calibration.csv")
    assert set(pass_rate["estimator"]) == {"online", "eap", "mwle", "full_eap"}
    assert {"mae", "bias", "r"} <= set(pass_rate.columns)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["structure"]["name"] == "test_two_axis"
    assert manifest["config"]["estimate_latent_corr"] is True
