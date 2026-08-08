from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_infobench_dense_grid_study.py"
SPEC = importlib.util.spec_from_file_location("run_infobench_dense_grid_study", SCRIPT)
assert SPEC and SPEC.loader
dense = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dense
SPEC.loader.exec_module(dense)


def _context():
    matrix = (
        ROOT
        / "runs"
        / "calibration"
        / "InFoBench_full_20260804"
        / "inputs"
        / "response_matrix.csv"
    )
    if not matrix.is_file():
        pytest.skip("integration check requires the separately frozen response matrix")
    return dense.load_context(dense.DEFAULT_CONFIG, dense.DEFAULT_SPLITS)


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_fit_and_eap_grids_propagate_as_separate_worker_arguments(tmp_path: Path) -> None:
    context = _context()
    fit = dense.build_fit_command(context, tmp_path / "new", 25, 0.1)
    score = dense.build_score_command(context, tmp_path / "new", 25, 81, 0.1)

    assert _argument(fit, "--fit-grid") == "25"
    assert "--eap-grid" not in fit
    assert "--fit-cache-dir" not in fit

    assert _argument(score, "--fit-grid") == "25"
    assert _argument(score, "--eap-grid") == "81"
    assert _argument(score, "--fit-cache-dir").endswith("fit_cache/grid_025/ridge_0p1")


def test_plan_only_is_read_only_and_defers_ridge_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context = _context()
    out_dir = tmp_path / "never-created"
    historical = ROOT / "reports" / "infobench_calibration_20260804" / "baseline_reproduction.json"
    before_hash = dense._sha256(historical)
    before_mtime = historical.stat().st_mtime_ns

    runner = dense.Orchestrator(context, out_dir, resume=False, plan_only=True)
    assert runner.run() == 0

    assert not out_dir.exists()
    assert dense._sha256(historical) == before_hash
    assert historical.stat().st_mtime_ns == before_mtime
    assert len(runner.commands) == 10  # five fits + five common-grid held-out scores
    assert all("--ridge" in entry["command"] for entry in runner.commands)
    assert not any("ridge_0p001" in entry["stage"] for entry in runner.commands)

    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    fit_gate = next(row for row in payloads if row["stage"] == "fit_grid_gate")
    conditional = fit_gate["conditional_eap_commands"]
    assert [row["eap_grid"] for row in conditional] == [21, 41, 81]
    assert all(_argument(row["command"], "--fit-grid") == "25" for row in conditional)
    ridge = next(row for row in payloads if row["stage"] == "ridge_candidates")
    assert ridge["status"] == "deferred_until_both_grid_locks_exist"
    assert ridge["commands"] is None


def test_cached_fit_round_trip_preserves_one_dimensional_parameters(tmp_path: Path) -> None:
    path = tmp_path / "fit.npz"
    original = {
        "A": np.asarray([[1.2], [0.8]]),
        "b": np.asarray([-0.5, 0.3]),
        "R": np.eye(1),
        "items": ["c1", "c2"],
        "dim_labels": [dense.DIMENSION],
        "loglik": -12.5,
        "n_params": 4,
        "n_iter": 7,
        "converged": True,
    }
    dense._save_fit(path, original)
    restored = dense.load_cached_fit(path)

    np.testing.assert_allclose(restored["A"], original["A"])
    np.testing.assert_allclose(restored["b"], original["b"])
    assert restored["items"] == original["items"]
    assert restored["dim_labels"] == [dense.DIMENSION]
    assert restored["converged"] is True


def test_resume_reuses_only_a_matching_completed_stage(tmp_path: Path) -> None:
    context = _context()
    out_dir = tmp_path / "study"
    first = dense.Orchestrator(context, out_dir, resume=False, plan_only=False)
    first.prepare()
    output = out_dir / "tiny.txt"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
        str(output),
    ]
    first.run_stage("tiny", command, [output])
    before_mtime = output.stat().st_mtime_ns

    resumed = dense.Orchestrator(context, out_dir, resume=True, plan_only=False)
    resumed.prepare()
    resumed.run_stage("tiny", command, [output])
    assert output.read_text(encoding="utf-8") == "ok"
    assert output.stat().st_mtime_ns == before_mtime


def test_fresh_output_and_historical_output_guards_fail_closed(tmp_path: Path) -> None:
    context = _context()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(dense.DenseGridError, match="already exists"):
        dense.Orchestrator(context, existing, resume=False, plan_only=False).prepare()

    protected = ROOT / context.config["baseline"]["run_dir"]
    with pytest.raises(dense.DenseGridError, match="protected historical"):
        dense.Orchestrator(context, protected, resume=False, plan_only=True).prepare()


def _write_synthetic_score(
    context,
    out_dir: Path,
    eap_grid: int,
    *,
    theta: list[float],
    orders: list[str],
) -> None:
    score_dir = dense._score_dir(out_dir, 25, eap_grid, context.initial_ridge)
    dense._write_json(
        score_dir / "score_manifest.json",
        {
            "pooled_oos": {"log_loss": 0.4, "brier": 0.12},
            "quantization": {
                "n_models": len(theta),
                "unique_theta_rounded_6": 10,
                "largest_rounded_value_count": 1,
                "largest_rounded_value_fraction": 0.1,
                "fraction_within_1e_6_of_node": 0.0,
                "largest_nearest_node_fraction": 0.2,
            },
        },
    )
    dense._write_csv(
        score_dir / "recovery.csv",
        pd.DataFrame(
            [
                {
                    "fold": "pooled",
                    "estimator": "mwle",
                    "n": len(theta),
                    "r": 0.95,
                    "slope": 1.0,
                    "mae": 0.1,
                    "bias": 0.0,
                }
            ]
        ),
    )
    dense._write_csv(
        score_dir / "pass_rate.csv",
        pd.DataFrame(
            [
                {
                    "fold": "pooled",
                    "estimator": "mwle",
                    "n": len(theta),
                    "mae": 0.04,
                    "bias": 0.0,
                    "r": 0.9,
                }
            ]
        ),
    )
    dense._write_csv(
        score_dir / "per_model.csv",
        pd.DataFrame(
            {
                "model": [f"m{index}" for index in range(len(theta))],
                "fold": [index % 5 for index in range(len(theta))],
                "theta_reference": theta,
                "scenario_order": orders,
                "criterion_order": orders,
            }
        ),
    )


def test_eap_stability_gate_passes_small_shifts_and_fails_order_flip(tmp_path: Path) -> None:
    context = _context()
    base = [index / 10 for index in range(10)]
    same_orders = ['["s1", "s2"]'] * 10
    _write_synthetic_score(context, tmp_path, 21, theta=base, orders=same_orders)
    _write_synthetic_score(context, tmp_path, 41, theta=base, orders=same_orders)
    _write_synthetic_score(
        context,
        tmp_path,
        81,
        theta=[value + 0.001 for value in base],
        orders=same_orders,
    )
    passed = dense.summarize_eap_grids(context, tmp_path, 25)
    assert passed["passed"] is True
    assert passed["locked_eap_grid"] == 41

    flipped = same_orders.copy()
    flipped[0] = '["s2", "s1"]'
    _write_synthetic_score(
        context,
        tmp_path,
        81,
        theta=[value + 0.001 for value in base],
        orders=flipped,
    )
    failed = dense.summarize_eap_grids(context, tmp_path, 25)
    assert failed["passed"] is False
    assert failed["fixed_bank_orders_identical"] is False
    assert failed["locked_eap_grid"] is None


def test_paired_fold_metric_gate_enforces_logloss_and_brier_stability() -> None:
    context = _context()
    policy = context.config["dense_grid"]["fit_grid_stability"]["heldout_metric_stability"]
    lower = pd.DataFrame(
        {
            "fold": ["pooled", 0, 1, 2, 3, 4],
            "log_loss": [0.4, 0.39, 0.4, 0.41, 0.42, 0.38],
            "brier": [0.12, 0.11, 0.12, 0.13, 0.14, 0.1],
        }
    )
    stable = lower.copy()
    stable["log_loss"] += 0.003
    stable["brier"] += 0.003
    result = dense.paired_fold_metric_stability(lower, stable, policy)
    assert result["passed"] is True
    assert result["metrics"]["log_loss"]["passed_absolute_limit"] is True

    uncertain = lower.copy()
    differences = np.asarray([-0.01, 0.03, -0.01, 0.03, 0.01])
    uncertain.loc[0, ["log_loss", "brier"]] += 0.01
    uncertain.loc[1:, "log_loss"] += differences
    uncertain.loc[1:, "brier"] += differences
    result = dense.paired_fold_metric_stability(lower, uncertain, policy)
    assert result["passed"] is True
    assert result["metrics"]["log_loss"]["passed_absolute_limit"] is False
    assert result["metrics"]["log_loss"]["passed_uncertainty_limit"] is True

    unstable = lower.copy()
    unstable[["log_loss", "brier"]] += 0.02
    result = dense.paired_fold_metric_stability(lower, unstable, policy)
    assert result["passed"] is False
    assert result["metrics"]["log_loss"]["passed"] is False
    assert result["metrics"]["brier"]["passed"] is False
