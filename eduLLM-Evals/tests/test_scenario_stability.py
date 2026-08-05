"""Focused tests for scenario path and calibration-parameter uncertainty studies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import scenario_order_experiment as order
from scripts import scenario_param_uncertainty as param


DIMS = ("content", "format")


def test_order_spread_is_dynamic_and_excludes_failed_mwle() -> None:
    rows = []
    for seed, value, mwle_ok in [(10, 0.0, True), (11, 1.0, True), (12, 2.0, False)]:
        rows.append(
            {
                "model": "m",
                "seed": seed,
                "status": "ok",
                "scenarios_administered": 2,
                "scenario_order": json.dumps([f"s{seed % 2}", "s2"]),
                "mwle_converged": mwle_ok,
                "theta_online_content": value,
                "theta_online_format": value + 0.1,
                "theta_eap_content": value / 2,
                "theta_eap_format": value / 2 + 0.1,
                "theta_mwle_content": value + 1,
                "theta_mwle_format": value + 1.1,
            }
        )
    spread = order.per_model_spread(pd.DataFrame(rows), DIMS).iloc[0]
    assert spread["online_content_sd"] == pytest.approx(1.0)
    assert spread["online_content_range"] == pytest.approx(2.0)
    assert spread["mwle_valid_runs"] == 2
    assert spread["mwle_failed_runs"] == 1
    assert 0.0 <= spread["mean_pairwise_scenario_jaccard"] <= 1.0


def test_uncertainty_summary_uses_quadrature_formula_and_valid_counts() -> None:
    base = pd.DataFrame(
        [
            {
                "model": "m",
                "theta_eap_content": 0.5,
                "se_eap_content": 0.3,
                "theta_mwle_content": 0.6,
                "se_mwle_content": 0.4,
                "mwle_converged": True,
            }
        ]
    )
    draws = pd.DataFrame(
        [
            {"model": "m", "status": "ok", "mwle_converged": True,
             "theta_eap_content": 0.0, "theta_mwle_content": 1.0},
            {"model": "m", "status": "ok", "mwle_converged": True,
             "theta_eap_content": 2.0, "theta_mwle_content": 2.0},
            {"model": "m", "status": "ok", "mwle_converged": False,
             "theta_eap_content": 4.0, "theta_mwle_content": 9.0},
        ]
    )
    result = param.summarize_uncertainty(base, draws, ("content",), 3, 0.66)
    eap = result[result["estimator"] == "eap"].iloc[0]
    mwle = result[result["estimator"] == "mwle"].iloc[0]
    assert eap["se_param"] == pytest.approx(2.0)
    assert eap["se_total"] == pytest.approx(math.sqrt(0.3**2 + 2.0**2))
    assert eap["n_valid_bootstrap"] == 3
    assert bool(eap["reliable"])
    assert mwle["n_valid_bootstrap"] == 2
    assert mwle["se_param"] == pytest.approx(math.sqrt(0.5))


def test_uncertainty_summary_never_marks_failed_base_mwle_reliable() -> None:
    base = pd.DataFrame(
        [
            {
                "model": "m",
                "theta_eap_content": 0.0,
                "se_eap_content": 0.2,
                "theta_mwle_content": 0.0,
                "se_mwle_content": 0.2,
                "mwle_converged": False,
            }
        ]
    )
    draws = pd.DataFrame(
        [
            {
                "model": "m",
                "status": "ok",
                "mwle_converged": True,
                "theta_eap_content": value,
                "theta_mwle_content": value,
            }
            for value in (-0.2, 0.0, 0.2)
        ]
    )

    result = param.summarize_uncertainty(base, draws, ("content",), 3, 2 / 3)
    mwle = result[result["estimator"] == "mwle"].iloc[0]

    assert mwle["n_valid_bootstrap"] == 3
    assert not bool(mwle["base_estimator_valid"])
    assert not bool(mwle["reliable"])


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    bank_path = tmp_path / "bank.jsonl"
    scenarios_path = tmp_path / "scenarios.jsonl"
    matrix_path = tmp_path / "matrix.csv"
    rng = np.random.default_rng(44)
    n_models = 12
    n_scenarios = 4
    criterion_ids: list[str] = []
    records: list[dict] = []
    scenarios: list[dict] = []
    correlation = [[1.0, 0.2], [0.2, 1.0]]
    for scenario_index in range(n_scenarios):
        cids: list[str] = []
        for dim_index, dim in enumerate(DIMS):
            cid = f"c{scenario_index}_{dim}"
            cids.append(cid)
            criterion_ids.append(cid)
            q = {name: int(name == dim) for name in DIMS}
            records.append(
                {
                    "criterion_id": cid,
                    "scenario_id": f"s{scenario_index}",
                    "criterion": cid,
                    "q_mapping": q,
                    "q_modeled": q,
                    "discrimination": {
                        name: (1.0 + 0.1 * scenario_index if name == dim else 0.0)
                        for name in DIMS
                    },
                    "difficulty": -0.4 + 0.25 * scenario_index,
                    "criticality": "standard",
                    "status": "approved",
                    "irt_params": {
                        "source": "calibrated-m2pl-test",
                        "calibrated": True,
                        "skills_order": list(DIMS),
                        "latent_correlation": correlation,
                    },
                }
            )
        scenarios.append(
            {
                "scenario_id": f"s{scenario_index}",
                "prompt": f"scenario {scenario_index}",
                "criterion_ids": cids,
                "modality": "text",
            }
        )
    bank_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    scenarios_path.write_text(
        "".join(json.dumps(row) + "\n" for row in scenarios), encoding="utf-8"
    )

    probabilities = np.linspace(0.25, 0.75, len(criterion_ids))
    values = (rng.random((n_models, len(criterion_ids))) < probabilities).astype(int)
    # Guarantee full-sample variation for every item.
    values[0, :] = 0
    values[-1, :] = 1
    matrix = pd.DataFrame(
        values,
        index=[f"model_{index:02d}" for index in range(n_models)],
        columns=criterion_ids,
    )
    matrix.index.name = "model"
    matrix.to_csv(matrix_path)
    return {"bank": bank_path, "scenarios": scenarios_path, "matrix": matrix_path}


def test_order_experiment_tiny_dynamic_run(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    out_dir = tmp_path / "order"
    args = order.build_argparser().parse_args(
        [
            "--bank", str(paths["bank"]),
            "--matrix", str(paths["matrix"]),
            "--scenarios", str(paths["scenarios"]),
            "--out-dir", str(out_dir),
            "--n-seeds", "3",
            "--base-seed", "90",
            "--grid", "3",
            "--top-n", "2",
            "--min-evals-per-skill", "0",
            "--max-se", "0.01",
            "--max-scenarios", "3",
        ]
    )
    assert order.run(args) == 0
    seed_runs = pd.read_csv(out_dir / "seed_runs.csv")
    assert len(seed_runs) == 12 * 3
    assert set(seed_runs["seed"]) == {90, 91, 92}
    summary = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert summary["dimensions"] == list(DIMS)
    assert len(summary["spread"]) == 3 * len(DIMS)


def test_parameter_uncertainty_tiny_model_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_fixture(tmp_path)
    out_dir = tmp_path / "uncertainty"
    original_refit = param.bootstrap_refit
    refit_calls = 0

    def counted_refit(*args, **kwargs):
        nonlocal refit_calls
        refit_calls += 1
        return original_refit(*args, **kwargs)

    monkeypatch.setattr(param, "bootstrap_refit", counted_refit)
    args = param.build_argparser().parse_args(
        [
            "--bank", str(paths["bank"]),
            "--matrix", str(paths["matrix"]),
            "--scenarios", str(paths["scenarios"]),
            "--out-dir", str(out_dir),
            "--n-boot", "4",
            "--min-valid-fraction", "0.25",
            "--fit-grid", "2",
            "--eap-grid", "3",
            "--max-iter", "3",
            "--tol", "0.01",
            "--negative-policy", "keep",
            "--allow-unconverged-fit",
            "--top-n", "2",
            "--min-evals-per-skill", "0",
            "--max-se", "0.01",
            "--se-targets", "0.20,0.35",
            "--max-scenarios", "4",
            "--seed", "1234",
        ]
    )
    assert param.run(args) == 0
    replicates = pd.read_csv(out_dir / "bootstrap_replicates.csv")
    assert len(replicates) == 4
    assert refit_calls == 4
    assert "sample_signature" in replicates
    components = pd.read_csv(out_dir / "ability_se_components.csv")
    assert set(components["estimator"]) == {"eap", "mwle"}
    assert set(components["dimension"]) == set(DIMS)
    assert set(components["se_target"]) == {0.20, 0.35}
    assert (out_dir / "ability_se_components_se_0p20.csv").is_file()
    assert (out_dir / "ability_se_components_se_0p35.csv").is_file()
    total_by_target = pd.read_csv(out_dir / "total_se_vs_target.csv")
    assert set(total_by_target["se_target"]) == {0.20, 0.35}
    finite = components.dropna(subset=["se_param", "se_total"])
    assert len(finite) > 0
    assert np.allclose(
        finite["se_total"] ** 2,
        finite["se_ability"] ** 2 + finite["se_param"] ** 2,
    )
    summary = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert summary["method"].startswith("nonparametric model-row bootstrap")
    assert summary["n_boot_requested"] == 4
    assert summary["se_targets"] == [0.20, 0.35]
    assert summary["config"]["bootstrap_refits_reused_across_targets"] is True
