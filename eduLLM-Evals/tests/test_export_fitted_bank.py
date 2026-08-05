"""Tests for the generic fitted-only calibrated-bank exporter."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "export_fitted_bank.py"
SPEC = importlib.util.spec_from_file_location("export_fitted_bank", SCRIPT)
assert SPEC and SPEC.loader
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "csv": tmp_path / "calibration.csv",
        "rubrics": tmp_path / "rubrics.jsonl",
        "scenarios": tmp_path / "scenarios.jsonl",
        "calibration_manifest": tmp_path / "calibration_manifest.json",
        "out_rubrics": tmp_path / "out" / "rubrics_fitted.jsonl",
        "out_scenarios": tmp_path / "out" / "scenarios_fitted.jsonl",
        "manifest": tmp_path / "out" / "manifest.json",
    }


def _config(paths: dict[str, Path], **overrides):
    values = {
        "calibration_csv": paths["csv"],
        "rubrics": paths["rubrics"],
        "scenarios": paths["scenarios"],
        "out_rubrics": paths["out_rubrics"],
        "out_scenarios": paths["out_scenarios"],
        "out_manifest": paths["manifest"],
        "calibration_manifest": paths["calibration_manifest"],
    }
    values.update(overrides)
    return exporter.ExportConfig(**values)


def _write_calibration_manifest(
    paths: dict[str, Path], skills: list[str], n_items: int, corr: list[list[float]] | None = None
) -> None:
    paths["calibration_manifest"].write_text(
        json.dumps(
            {
                "method": "confirmatory-m2pl-mml-em",
                "skills_order": skills,
                "n_items_fit": n_items,
                "n_persons_fit": 52,
                "estimate_latent_corr": corr is not None,
                "latent_correlation": corr,
            }
        ),
        encoding="utf-8",
    )


def _source_record(cid: str, sid: str, q: dict[str, int]) -> dict:
    return {
        "criterion_id": cid,
        "scenario_id": sid,
        "criterion": f"criterion {cid}",
        "q_mapping": q,
        "difficulty": -999.0,
        "discrimination": {skill: 999.0 for skill in q},
        "irt_params": {"source": "synthetic", "method": "placeholder"},
    }


def test_export_is_fitted_only_filters_scenarios_and_preserves_sources(tmp_path: Path):
    paths = _paths(tmp_path)
    rubrics = [
        _source_record("c_valid", "s1", {"alpha": 1, "beta": 0}),
        _source_record("c_unfitted", "s1", {"alpha": 0, "beta": 1}),
        _source_record("c_nonpositive", "s2", {"alpha": 0, "beta": 1}),
        _source_record("c_extreme", "s3", {"alpha": 1, "beta": 1}),
    ]
    scenarios = [
        {"scenario_id": "s1", "prompt": "one", "criterion_ids": ["c_valid", "c_unfitted"]},
        {"scenario_id": "s2", "prompt": "two", "criterion_ids": ["c_nonpositive"]},
        {"scenario_id": "s3", "prompt": "three", "criterion_ids": ["c_extreme"]},
    ]
    _write_jsonl(paths["rubrics"], rubrics)
    _write_jsonl(paths["scenarios"], scenarios)
    _write_csv(
        paths["csv"],
        ["criterion_id", "a_alpha", "a_beta", "b", "n_persons", "flags"],
        [
            {"criterion_id": "c_valid", "a_alpha": 1.25, "a_beta": 0, "b": 0.4,
             "n_persons": 52, "flags": "low_n"},
            {"criterion_id": "c_nonpositive", "a_alpha": 0, "a_beta": -0.2, "b": 0.1,
             "n_persons": 52, "flags": "nonpositive_a|low_n"},
            {"criterion_id": "c_extreme", "a_alpha": 7.0, "a_beta": 1.1, "b": -0.2,
             "n_persons": 52, "flags": "extreme_a|low_n"},
        ],
    )
    _write_calibration_manifest(
        paths, ["alpha", "beta"], 3, [[1.0, 0.4], [0.4, 1.0]]
    )
    before = {"rubrics": _sha(paths["rubrics"]), "scenarios": _sha(paths["scenarios"])}

    manifest = exporter.export_fitted_bank(
        _config(paths)
    )

    assert {"rubrics": _sha(paths["rubrics"]), "scenarios": _sha(paths["scenarios"])} == before
    output = _read_jsonl(paths["out_rubrics"])
    assert [row["criterion_id"] for row in output] == ["c_valid"]
    row = output[0]
    assert row["difficulty"] == 0.4
    assert row["discrimination"] == {"alpha": 1.25, "beta": 0.0}
    assert row["q_modeled"] == {"alpha": 1, "beta": 0}
    assert row["irt_params"]["calibrated"] is True
    assert row["irt_params"]["fitted"] is True
    assert row["irt_params"]["synthetic"] is False
    assert row["irt_params"]["source"] == exporter.CALIBRATION_SOURCE
    assert row["irt_params"]["skills_order"] == ["alpha", "beta"]
    assert row["irt_params"]["latent_correlation"] == [[1.0, 0.4], [0.4, 1.0]]
    assert row["irt_params"]["provenance"]["latent_correlation"] == [
        [1.0, 0.4], [0.4, 1.0]
    ]
    assert -999.0 not in (row["difficulty"], *row["discrimination"].values())
    assert _read_jsonl(paths["out_scenarios"]) == [
        {"scenario_id": "s1", "prompt": "one", "criterion_ids": ["c_valid"]}
    ]
    assert manifest["counts"] == {
        "source_criteria": 4,
        "calibration_rows": 3,
        "exported_criteria": 1,
        "excluded_criteria": 3,
        "source_scenarios": 3,
        "exported_scenarios": 1,
        "dropped_empty_scenarios": 2,
    }
    assert manifest["exclusion_reason_counts"] == {
        "extreme_a": 1,
        "nonpositive_a": 1,
        "unfitted": 1,
    }
    assert manifest["invariants"]["contains_synthetic_parameters"] is False
    assert json.loads(paths["manifest"].read_text()) == manifest


def test_collapsed_skill_map_is_arbitrary_and_ordered(tmp_path: Path):
    paths = _paths(tmp_path)
    rubrics = [
        _source_record("c_content", "s1", {"content": 1, "diagnosis": 0, "scaffolding": 0}),
        _source_record("c_diagnosis", "s1", {"content": 0, "diagnosis": 1, "scaffolding": 0}),
        _source_record("c_scaffolding", "s2", {"content": 0, "diagnosis": 0, "scaffolding": 1}),
    ]
    scenarios = [
        {"scenario_id": "s1", "criterion_ids": ["c_content", "c_diagnosis"]},
        {"scenario_id": "s2", "criterion_ids": ["c_scaffolding"]},
    ]
    _write_jsonl(paths["rubrics"], rubrics)
    _write_jsonl(paths["scenarios"], scenarios)
    _write_csv(
        paths["csv"],
        ["criterion_id", "a_correctness", "a_support", "b", "flags"],
        [
            {
                "criterion_id": "c_content", "a_correctness": 1.1,
                "a_support": 0, "b": 0, "flags": "",
            },
            {
                "criterion_id": "c_diagnosis", "a_correctness": 1.2,
                "a_support": 0, "b": 0.1, "flags": "",
            },
            {
                "criterion_id": "c_scaffolding", "a_correctness": 0,
                "a_support": 1.3, "b": 0.2, "flags": "",
            },
        ],
    )
    _write_calibration_manifest(
        paths, ["correctness", "support"], 3, [[1.0, 0.2], [0.2, 1.0]]
    )

    manifest = exporter.export_fitted_bank(
        _config(
            paths,
            dimensions="correctness=content+diagnosis,support=scaffolding",
        )
    )

    output = {row["criterion_id"]: row for row in _read_jsonl(paths["out_rubrics"])}
    assert list(output["c_content"]["discrimination"]) == ["correctness", "support"]
    assert output["c_content"]["q_modeled"] == {"correctness": 1, "support": 0}
    assert output["c_diagnosis"]["q_modeled"] == {"correctness": 1, "support": 0}
    assert output["c_scaffolding"]["q_modeled"] == {"correctness": 0, "support": 1}
    assert manifest["skill_structure"] == {
        "name": "export",
        "source_skills": ["content", "diagnosis", "scaffolding"],
        "dimensions": [
            {"label": "correctness", "members": ["content", "diagnosis"]},
            {"label": "support", "members": ["scaffolding"]},
        ],
    }
    assert manifest["per_skill_items"] == {"correctness": 2, "support": 1}


def test_uncovered_source_skill_and_off_q_loading_block_export(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_jsonl(
        paths["rubrics"],
        [
            _source_record("c1", "s1", {"alpha": 1, "beta": 0}),
            _source_record("c2", "s1", {"alpha": 0, "beta": 1}),
        ],
    )
    _write_jsonl(
        paths["scenarios"],
        [{"scenario_id": "s1", "criterion_ids": ["c1", "c2"]}],
    )
    _write_csv(
        paths["csv"],
        ["criterion_id", "a_alpha", "a_beta", "b"],
        [
            {"criterion_id": "c1", "a_alpha": 1.0, "a_beta": 0.5, "b": 0.0},
            {"criterion_id": "c2", "a_alpha": 0.0, "a_beta": 1.0, "b": 0.0},
        ],
    )
    _write_calibration_manifest(paths, ["alpha", "beta"], 2)

    with pytest.raises(exporter.ExportError, match="off_q_nonzero"):
        exporter.export_fitted_bank(_config(paths))
    assert not paths["out_rubrics"].exists()


def test_skill_structure_cannot_silently_drop_positive_source_dimension(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_jsonl(
        paths["rubrics"],
        [
            _source_record("c1", "s1", {"alpha": 1, "beta": 0}),
            _source_record("c2", "s1", {"alpha": 0, "beta": 1}),
        ],
    )
    _write_jsonl(
        paths["scenarios"],
        [{"scenario_id": "s1", "criterion_ids": ["c1", "c2"]}],
    )
    _write_csv(
        paths["csv"],
        ["criterion_id", "a_overall", "b"],
        [{"criterion_id": "c1", "a_overall": 1.0, "b": 0.0}],
    )
    _write_calibration_manifest(paths, ["overall"], 1)

    with pytest.raises(exporter.ExportError, match="unassigned source skills"):
        exporter.export_fitted_bank(
            _config(paths, dimensions="overall=alpha")
        )


def test_refuses_input_collision_and_existing_outputs(tmp_path: Path):
    paths = _paths(tmp_path)
    _write_jsonl(paths["rubrics"], [_source_record("c1", "s1", {"alpha": 1})])
    _write_jsonl(paths["scenarios"], [{"scenario_id": "s1", "criterion_ids": ["c1"]}])
    _write_csv(
        paths["csv"], ["criterion_id", "a_alpha", "b"],
        [{"criterion_id": "c1", "a_alpha": 1.0, "b": 0.0}],
    )
    _write_calibration_manifest(paths, ["alpha"], 1)

    with pytest.raises(exporter.ExportError, match="overwrite an input"):
        exporter.export_fitted_bank(_config(paths, out_rubrics=paths["rubrics"]))

    exporter.export_fitted_bank(_config(paths))
    with pytest.raises(exporter.ExportError, match="output already exists"):
        exporter.export_fitted_bank(_config(paths))
