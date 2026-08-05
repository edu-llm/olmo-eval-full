from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_infobench_calibration_study.py"
SPEC = importlib.util.spec_from_file_location("run_infobench_calibration_study", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def _config(tmp_path: Path) -> Path:
    inputs = {}
    for name in ("matrix", "rubrics", "scenarios", "judge_manifest"):
        path = tmp_path / name
        path.write_text("placeholder\n", encoding="utf-8")
        inputs[name] = str(path)
    value = {
        "schema_version": "calibration-study-v1",
        "benchmark": "InFoBench",
        "inputs": inputs,
        "source_skills": ["a", "b", "c"],
        "structures": [
            {"name": "full", "dimensions": "a=a,b=b,c=c", "status": "confirmatory"},
            {"name": "reduced", "dimensions": "ab=a+b,c=c", "status": "exploratory"},
        ],
        "fit": {"grid": 3, "ridge": 0.01},
        "kfold": {"folds": 2, "evaluation_scenario_fraction": 0.2},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_config_requires_partitioned_structures_and_preserves_native_axis(tmp_path: Path) -> None:
    path = _config(tmp_path)
    config, candidates = study.load_and_validate_config(path)
    assert config["source_skills"] == ["a", "b", "c"]
    assert candidates[0].structure.labels == ("a", "b", "c")
    assert candidates[1].structure.labels == ("ab", "c")
    assert candidates[1].structure.groups == {"ab": ["a", "b"], "c": ["c"]}


def test_config_rejects_structure_that_drops_a_source_skill(tmp_path: Path) -> None:
    path = _config(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["structures"][1]["dimensions"] = "ab=a+b"
    path.write_text(json.dumps(value), encoding="utf-8")
    try:
        study.load_and_validate_config(path)
    except study.StudyError as exc:
        assert "unassigned source skills" in str(exc)
    else:
        raise AssertionError("invalid skill partition was accepted")


def test_commands_use_full_matrix_and_explicit_structure(tmp_path: Path) -> None:
    path = _config(tmp_path)
    config, candidates = study.load_and_validate_config(path)
    runner = study.Orchestrator(
        path, config, candidates, tmp_path / "out", resume=False, plan_only=True
    )
    command = runner.fit_command(candidates[1], tmp_path / "fit", grid=3, ridge=0.01)
    assert "--matrix" in command
    assert str(tmp_path / "matrix") in command
    assert command[command.index("--dimensions") + 1] == "ab=a+b,c=c"
    assert "--require-complete-bank" in command
    assert "--write-params" not in command


def test_cat_and_uncertainty_stage_commands_are_well_formed(tmp_path: Path) -> None:
    path = _config(tmp_path)
    config, candidates = study.load_and_validate_config(path)
    config["cat"] = {
        "min_scenarios_values": [0, 12],
        "se_target_values": [0.2, 0.3],
        "selection_rules": ["trace", "dopt"],
        "order_seeds": [1000, 1001],
    }
    config["estimator"] = {"quadrature_grid": 3, "max_grid_nodes": 5000}
    config["uncertainty"] = {"parameter_bootstrap_replicates": 4, "seed": 7}
    runner = study.Orchestrator(
        path, config, candidates, tmp_path / "out", resume=False, plan_only=False
    )
    captured = []
    runner.run_command = lambda stage, command, outputs: captured.append(  # type: ignore[method-assign]
        (stage, command, outputs)
    )
    runner._preselect_cat_floor = lambda: 12  # type: ignore[method-assign]
    runner._preselect_cat_se_target = lambda: 0.3  # type: ignore[method-assign]
    runner.run_cat_studies(tmp_path / "bank.jsonl", tmp_path / "scenarios.jsonl")
    assert len(captured) == 6  # 2 floors + 2 SE targets + 2 selectors

    runner.run_final_cat_comparison(
        tmp_path / "bank.jsonl",
        tmp_path / "scenarios.jsonl",
        min_scenarios=15,
        max_se=0.25,
        selection="dopt",
    )
    assert len(captured) == 9  # paired run + final CAT-only + random-only

    runner.run_order_and_uncertainty(
        tmp_path / "bank.jsonl",
        tmp_path / "scenarios.jsonl",
        fit_grid=3,
        ridge=0.01,
        min_scenarios=15,
        max_se=0.25,
        selection="dopt",
    )
    uncertainty = captured[-1]
    assert uncertainty[0] == "parameter_uncertainty"
    command = uncertainty[1]
    assert command[command.index("--se-targets") + 1] == "0.2,0.3"
    assert command[command.index("--min-scenarios") + 1] == "15"
    assert command[command.index("--max-se") + 1] == "0.25"
    assert command[command.index("--selection") + 1] == "dopt"
    assert tmp_path / "out" / "parameter_uncertainty" / "total_se_vs_target.csv" in uncertainty[2]


def test_cat_configuration_selection_uses_coverage_recovery_and_length(
    tmp_path: Path,
) -> None:
    path = _config(tmp_path)
    config, candidates = study.load_and_validate_config(path)
    config["cat"] = {
        "min_scenarios_values": [0, 12],
        "se_target_values": [0.2, 0.3],
        "selection_rules": ["trace", "dopt"],
        "minimum_precision_rate": 0.95,
        "minimum_mwle_convergence_rate": 0.95,
        "maximum_recovery_drop": 0.01,
        "maximum_slope_error_increase": 0.05,
        "selector_minimum_length_reduction": 0.05,
        "selector_maximum_recovery_drop": 0.01,
    }
    runner = study.Orchestrator(
        path, config, candidates, tmp_path / "out", resume=False, plan_only=False
    )

    def write_metrics(path: Path, *, precision: float, mean: float, r: float, slope: float) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "modes": {
                "cat": {
                    "n_models": 20,
                    "precision_rate": precision,
                    "scenarios": {"mean": mean},
                    "mwle_converged": 20,
                    "recovery_vs_full_bank_eap": {
                        "mwle": {
                            "a": {"r": r, "slope": slope},
                            "b": {"r": r, "slope": slope},
                            "c": {"r": r, "slope": slope},
                        }
                    },
                }
            }
        }
        (path / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    root = tmp_path / "out" / "cat_sweeps"
    write_metrics(
        root / "min_scenarios" / "floor_0",
        precision=0.96,
        mean=10,
        r=0.90,
        slope=0.70,
    )
    write_metrics(
        root / "min_scenarios" / "floor_12",
        precision=0.96,
        mean=15,
        r=0.96,
        slope=0.98,
    )
    write_metrics(
        root / "se_target" / "se_0p2",
        precision=0.90,
        mean=30,
        r=0.97,
        slope=0.99,
    )
    write_metrics(
        root / "se_target" / "se_0p3",
        precision=0.96,
        mean=20,
        r=0.96,
        slope=0.98,
    )
    write_metrics(
        root / "selection" / "trace",
        precision=0.96,
        mean=20,
        r=0.96,
        slope=0.98,
    )
    write_metrics(
        root / "selection" / "dopt",
        precision=0.96,
        mean=18,
        r=0.955,
        slope=0.97,
    )

    assert runner.select_cat_configuration() == (12, 0.3, "dopt")
