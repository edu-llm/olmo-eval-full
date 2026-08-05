"""Focused tests for generic skill structures and MIRT person-fold validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tutor_cat.skill_structure import SkillStructure, parse_dimension_spec


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "kfold_cv_mirt.py"
SPEC = importlib.util.spec_from_file_location("kfold_cv_mirt_test_module", SCRIPT)
assert SPEC and SPEC.loader
kf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kf
SPEC.loader.exec_module(kf)


INFO_SKILLS = ("content", "format", "number", "style", "linguistic")


def test_skill_structure_or_merges_multiple_groups() -> None:
    structure = parse_dimension_spec(
        "content_style=content+style,format=format,number_linguistic=number+linguistic",
        INFO_SKILLS,
        name="correlated_3d",
    )
    q_source = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ],
        dtype=int,
    )
    assert structure.labels == ("content_style", "format", "number_linguistic")
    assert structure.groups == {
        "content_style": ["content", "style"],
        "format": ["format"],
        "number_linguistic": ["number", "linguistic"],
    }
    assert structure.transform_q(q_source).tolist() == [
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 0],
    ]


@pytest.mark.parametrize(
    "spec,match",
    [
        ("first=content+format,second=number+style", "unassigned"),
        ("first=content+format,second=format+number+style+linguistic", "more than once"),
        ("first=content+unknown,second=format+number+style+linguistic", "unknown"),
        ("content+format", "label=skill"),
    ],
)
def test_skill_structure_rejects_nonpartitions(spec: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_dimension_spec(spec, INFO_SKILLS)


def test_structure_defaults_preserve_legacy_and_make_explicit_axis_identity() -> None:
    legacy = kf.build_structure(("content", "diagnosis", "scaffolding"), None, None)
    assert legacy.labels == ("correctness", "scaffolding")
    assert legacy.groups["correctness"] == ["content", "diagnosis"]

    explicit_historical_axis = kf.build_structure(
        ("content", "diagnosis", "scaffolding"),
        None,
        None,
        historical_default=False,
    )
    assert explicit_historical_axis.labels == ("content", "diagnosis", "scaffolding")

    identity = kf.build_structure(INFO_SKILLS, None, None)
    assert identity == SkillStructure.identity(INFO_SKILLS)
    assert identity.labels == INFO_SKILLS


def test_stratified_folds_are_deterministic_balanced_and_complete() -> None:
    models = [f"m{i:02d}" for i in range(17)]
    matrix = pd.DataFrame(
        {"c1": np.linspace(0.0, 1.0, len(models)), "c2": np.linspace(0.0, 1.0, len(models))},
        index=models,
    )
    first = kf.make_stratified_folds(matrix, k=5, seed=19)
    second = kf.make_stratified_folds(matrix, k=5, seed=19)
    assert first == second
    assert sorted(model for fold in first for model in fold) == models
    assert max(map(len, first)) - min(map(len, first)) <= 1


def test_disjoint_eap_does_not_use_evaluation_responses_for_theta() -> None:
    items = ["score", "evaluate"]
    item_to_scenario = {"score": "s_score", "evaluate": "s_evaluate"}
    A = np.array([[1.2], [0.8]])
    b = np.array([0.0, 0.1])
    grid = kf.cm.build_grid(1, 5)
    log_prior = kf.cm.base_log_weights(1, 5)

    fail_eval = pd.DataFrame([[1.0, 0.0]], index=["m"], columns=items)
    pass_eval = pd.DataFrame([[1.0, 1.0]], index=["m"], columns=items)
    y0, _, theta0, _ = kf.eap_predict_disjoint(
        fail_eval, items, A, b, grid, log_prior, item_to_scenario, {"s_evaluate"}
    )
    y1, _, theta1, _ = kf.eap_predict_disjoint(
        pass_eval, items, A, b, grid, log_prior, item_to_scenario, {"s_evaluate"}
    )
    assert y0.tolist() == [0.0]
    assert y1.tolist() == [1.0]
    assert np.allclose(theta0, theta1)


def _write_generic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    n_models = 15
    n_items = 10
    models = [f"model_{index:02d}" for index in range(n_models)]
    items = [f"item_{index:02d}" for index in range(n_items)]
    values = np.empty((n_models, n_items), dtype=int)
    for person in range(n_models):
        for item in range(n_items):
            values[person, item] = (person + item + person // 3) % 2
    matrix = pd.DataFrame(values, index=models, columns=items)
    matrix.index.name = "model"
    matrix_path = tmp_path / "matrix.csv"
    matrix.to_csv(matrix_path)

    rubrics_path = tmp_path / "rubrics.jsonl"
    with rubrics_path.open("w", encoding="utf-8") as handle:
        for item_index, criterion_id in enumerate(items):
            skill = INFO_SKILLS[item_index % len(INFO_SKILLS)]
            record = {
                "criterion_id": criterion_id,
                "scenario_id": f"scenario_{item_index // 2:02d}",
                "criterion": criterion_id,
                "q_mapping": {name: int(name == skill) for name in INFO_SKILLS},
            }
            handle.write(json.dumps(record) + "\n")
    return matrix_path, rubrics_path


def test_generic_three_dimensional_kfold_run_writes_dynamic_outputs(tmp_path: Path) -> None:
    matrix_path, rubrics_path = _write_generic_fixture(tmp_path)
    out_dir = tmp_path / "out"
    args = kf.build_argparser().parse_args(
        [
            "--matrix", str(matrix_path),
            "--rubrics", str(rubrics_path),
            "--skills", ",".join(INFO_SKILLS),
            "--dimensions",
            "content_style=content+style,format=format,number_linguistic=number+linguistic",
            "--structure-name", "correlated_3d",
            "--k", "3",
            "--grid", "2",
            "--max-iter", "2",
            "--tol", "0.01",
            "--ridge", "0.01",
            "--evaluation-scenario-fraction", "0.4",
            "--require-complete-bank",
            "--out-dir", str(out_dir),
        ]
    )
    try:
        assert kf.run(args) == 0
    finally:
        kf.cm.configure_skills(None)

    fold_params = pd.read_csv(out_dir / "fold_0_item_params.csv")
    assert list(fold_params.columns) == [
        "criterion_id", "a_content_style", "a_format", "a_number_linguistic", "b"
    ]
    summary = json.loads((out_dir / "kfold_summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["structure"]["name"] == "correlated_3d"
    assert [dim["label"] for dim in summary["config"]["structure"]["dimensions"]] == [
        "content_style", "format", "number_linguistic"
    ]
    assert summary["metric_kind"] == "heldout_person_disjoint_scenario_prediction"
    assert summary["pooled_oos"]["n_cells"] > 0
    assert summary["pooled_oos_reconstruction"]["n_cells"] > summary["pooled_oos"]["n_cells"]
    stability = pd.read_csv(out_dir / "item_param_stability.csv")
    assert set(stability["parameter"]) == {
        "a_content_style", "a_format", "a_number_linguistic", "b"
    }
