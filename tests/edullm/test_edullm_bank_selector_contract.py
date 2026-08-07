"""Focused parity and fail-closed tests for the EduLLM bank and selector port."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from olmo_eval.edullm.bank import (
    FITTED_BANK_SCHEMA_VERSION,
    BankValidationError,
    FittedBank,
    Rubric,
    Scenario,
    load_fitted_bank,
)
from olmo_eval.edullm.selector import (
    criterion_information,
    scenario_dopt_value,
    scenario_value,
    select_next,
    total_information_value,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rows)


def _criterion(
    criterion_id: str,
    scenario_id: str,
    q: tuple[int, int],
    a: tuple[float, float],
    b: float = 0.0,
    *,
    correlation: list[list[float]] | None = None,
) -> dict:
    skills = ["reasoning", "scaffolding"]
    corr = correlation or [[1.0, 0.25], [0.25, 1.0]]
    return {
        "criterion_id": criterion_id,
        "scenario_id": scenario_id,
        "criterion": f"criterion {criterion_id}",
        "q_modeled": dict(zip(skills, q, strict=True)),
        "discrimination": dict(zip(skills, a, strict=True)),
        "difficulty": b,
        "calibration_version": "calibration-2026-08-07",
        "scoring_type": "binary",
        "irt_params": {
            "source": "calibrated-m2pl-fitted-only",
            "calibrated": True,
            "fitted": True,
            "synthetic": False,
            "skills_order": skills,
            "latent_correlation": corr,
        },
    }


def _write_bundle(
    tmp_path: Path,
    *,
    rubrics: list[dict] | None = None,
    scenarios: list[dict] | None = None,
    correlation: list[list[float]] | None = None,
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    corr = correlation or [[1.0, 0.25], [0.25, 1.0]]
    rubric_rows = rubrics or [
        _criterion("c1", "s1", (1, 0), (1.2, 0.0), -0.2, correlation=corr),
        _criterion("c2", "s1", (0, 1), (0.0, 1.1), 0.1, correlation=corr),
        _criterion("c3", "s2", (1, 1), (0.8, 0.9), 0.4, correlation=corr),
    ]
    scenario_rows = scenarios or [
        {"scenario_id": "s1", "prompt": "prompt 1", "criterion_ids": ["c1", "c2"]},
        {"scenario_id": "s2", "prompt": "prompt 2", "criterion_ids": ["c3"]},
    ]
    rubric_bytes = _jsonl(rubric_rows)
    scenario_bytes = _jsonl(scenario_rows)
    rubric_path = tmp_path / "rubrics.jsonl"
    scenario_path = tmp_path / "scenarios.jsonl"
    manifest_path = tmp_path / "manifest.json"
    rubric_path.write_bytes(rubric_bytes)
    scenario_path.write_bytes(scenario_bytes)
    manifest = {
        "schema_version": FITTED_BANK_SCHEMA_VERSION,
        "skills_order": ["reasoning", "scaffolding"],
        "latent_correlation": corr,
        "counts": {
            "exported_criteria": len(rubric_rows),
            "exported_scenarios": len(scenario_rows),
        },
        "outputs": {
            "sha256": {
                "rubrics": _sha(rubric_bytes),
                "scenarios": _sha(scenario_bytes),
            }
        },
        "policies": {
            "extreme_a_threshold": 6.0,
            "off_q_tolerance": 1e-10,
        },
        "invariants": {
            "fitted_only": True,
            "contains_synthetic_parameters": False,
            "scenario_criterion_ids_exactly_match_exported_rubrics": True,
            "all_active_loadings_positive_and_within_threshold": True,
            "all_inactive_loadings_zero": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return rubric_path, scenario_path, manifest_path


def test_edullm_bank_loads_dynamic_versioned_manifest_bundle(tmp_path: Path) -> None:
    rubrics, scenarios, manifest = _write_bundle(tmp_path)

    bank = load_fitted_bank(rubrics, scenarios, manifest_path=manifest)

    assert bank.skills == ("reasoning", "scaffolding")
    assert bank.n_dims == 2
    assert bank.n_items == 3
    assert bank.criterion_ids == ("c1", "c2", "c3")
    assert np.array_equal(bank.Q, [[1, 0], [0, 1], [1, 1]])
    assert np.allclose(bank.A, [[1.2, 0.0], [0.0, 1.1], [0.8, 0.9]])
    assert np.allclose(bank.b, [-0.2, 0.1, 0.4])
    assert np.allclose(bank.latent_correlation, [[1.0, 0.25], [0.25, 1.0]])
    assert [rubric.criterion_id for rubric in bank.rubrics_for("s1")] == ["c1", "c2"]
    assert bank.manifest_sha256 == _sha(manifest.read_bytes())
    assert bank.Q.flags.writeable is False


def test_edullm_bank_manifest_hash_detects_relocated_file_tampering(tmp_path: Path) -> None:
    rubrics, scenarios, manifest = _write_bundle(tmp_path)
    rubrics.write_text(rubrics.read_text() + "\n", encoding="utf-8")

    with pytest.raises(BankValidationError, match="SHA-256 mismatch for rubrics"):
        load_fitted_bank(rubrics, scenarios, manifest_path=manifest)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(q_modeled={"reasoning": 0, "scaffolding": 0}), "at least one"),
        (lambda row: row["irt_params"].update(synthetic=True), "synthetic=false"),
        (lambda row: row.update(calibration_version=""), "calibration_version"),
        (
            lambda row: row.update(calibration_version="different-fit-v2"),
            "mix calibration_version",
        ),
        (
            lambda row: row.update(discrimination={"reasoning": 1.2, "scaffolding": 0.2}),
            "inactive discrimination",
        ),
        (lambda row: row.update(difficulty=float("nan")), "finite"),
    ],
)
def test_edullm_bank_rejects_entire_bundle_for_one_invalid_item(
    tmp_path: Path, mutation, message: str
) -> None:
    rows = [
        _criterion("c1", "s1", (1, 0), (1.2, 0.0)),
        _criterion("c2", "s2", (1, 0), (1.1, 0.0)),
    ]
    mutation(rows[1])
    rubric_path, scenario_path, _ = _write_bundle(
        tmp_path,
        rubrics=rows,
        scenarios=[
            {"scenario_id": "s1", "criterion_ids": ["c1"]},
            {"scenario_id": "s2", "criterion_ids": ["c2"]},
        ],
    )

    with pytest.raises(BankValidationError, match=message):
        load_fitted_bank(rubric_path, scenario_path)


def test_edullm_bank_rejects_duplicate_ids_and_inconsistent_correlation(tmp_path: Path) -> None:
    duplicate = [
        _criterion("c1", "s1", (1, 0), (1.0, 0.0)),
        _criterion("c1", "s1", (0, 1), (0.0, 1.0)),
    ]
    rubric_path, scenario_path, _ = _write_bundle(
        tmp_path,
        rubrics=duplicate,
        scenarios=[{"scenario_id": "s1", "criterion_ids": ["c1"]}],
    )
    with pytest.raises(BankValidationError, match="duplicate criterion_id"):
        load_fitted_bank(rubric_path, scenario_path)

    disagreeing = [
        _criterion("c1", "s1", (1, 0), (1.0, 0.0)),
        _criterion(
            "c2",
            "s2",
            (0, 1),
            (0.0, 1.0),
            correlation=[[1.0, 0.6], [0.6, 1.0]],
        ),
    ]
    rubric_path, scenario_path, _ = _write_bundle(
        tmp_path / "corr",
        rubrics=disagreeing,
        scenarios=[
            {"scenario_id": "s1", "criterion_ids": ["c1"]},
            {"scenario_id": "s2", "criterion_ids": ["c2"]},
        ],
    )
    with pytest.raises(BankValidationError, match="declarations disagree"):
        load_fitted_bank(rubric_path, scenario_path)


def test_edullm_bank_requires_explicit_positive_definite_correlation(tmp_path: Path) -> None:
    singular = [[1.0, 1.0], [1.0, 1.0]]
    rows = [
        _criterion("c1", "s1", (1, 0), (1.0, 0.0), correlation=singular),
    ]
    rubric_path, scenario_path, _ = _write_bundle(
        tmp_path,
        rubrics=rows,
        scenarios=[{"scenario_id": "s1", "criterion_ids": ["c1"]}],
        correlation=singular,
    )
    with pytest.raises(BankValidationError, match="positive definite"):
        load_fitted_bank(rubric_path, scenario_path)


def _runtime_rubric(
    cid: str, sid: str, q: tuple[int, int], a: tuple[float, float], b: float
) -> Rubric:
    return Rubric(
        criterion_id=cid,
        scenario_id=sid,
        criterion=cid,
        q=np.asarray(q),
        a=np.asarray(a),
        b=b,
        calibration_version="test-v1",
    )


def _runtime_bank(rubrics: list[Rubric]) -> FittedBank:
    by_scenario: dict[str, list[str]] = {}
    for rubric in rubrics:
        by_scenario.setdefault(rubric.scenario_id, []).append(rubric.criterion_id)
    scenarios = {
        sid: Scenario(sid, sid, tuple(criterion_ids)) for sid, criterion_ids in by_scenario.items()
    }
    return FittedBank(
        scenarios=scenarios,
        rubrics={rubric.criterion_id: rubric for rubric in rubrics},
        skills=("reasoning", "scaffolding"),
        latent_correlation=np.eye(2),
        records=tuple(),
        source_path="test",
        scenarios_path="test",
    )


def test_edullm_selector_preserves_fisher_cost_normalization_and_seeded_top_n() -> None:
    theta = np.zeros(2)
    bank = _runtime_bank(
        [
            _runtime_rubric("s0_c", "s0", (1, 0), (1.5, 0.0), 0.0),
            _runtime_rubric("s1_c", "s1", (1, 0), (1.5, 0.0), 0.8),
            _runtime_rubric("s2_c", "s2", (1, 0), (1.5, 0.0), 1.6),
            _runtime_rubric("s3_c", "s3", (1, 0), (1.5, 0.0), 2.4),
        ]
    )
    value = scenario_value(theta, bank.rubrics_for("s0"), 0)
    assert value == pytest.approx(0.25 * 1.5**2)
    assert criterion_information(theta, bank.rubrics["s0_c"], 1) == 0.0

    first = select_next(
        theta,
        bank,
        list(bank.scenarios),
        0,
        np.random.default_rng(42),
        top_n=2,
    )
    second = select_next(
        theta,
        bank,
        list(bank.scenarios),
        0,
        np.random.default_rng(42),
        top_n=2,
    )
    assert first == second
    assert {sid for sid, _ in first.top_candidates} == {"s0", "s1"}
    assert first.mode == "target_skill"


def test_edullm_selector_preserves_deterministic_fallback_and_optional_dopt() -> None:
    theta = np.zeros(2)
    bank = _runtime_bank(
        [
            _runtime_rubric("s1_c", "s1", (1, 0), (0.8, 0.0), 0.0),
            _runtime_rubric("s2_c", "s2", (1, 0), (1.9, 0.0), 0.0),
        ]
    )
    fallback = select_next(
        theta,
        bank,
        ["s1", "s2"],
        1,
        np.random.default_rng(0),
    )
    assert fallback.mode == "fallback_total_info"
    assert fallback.scenario_id == "s2"
    assert total_information_value(theta, bank.rubrics_for("s2")) > (
        total_information_value(theta, bank.rubrics_for("s1"))
    )

    covariance = np.diag([4.0, 1.0])
    assert scenario_dopt_value(theta, covariance, bank.rubrics_for("s2")) > 0
    dopt = select_next(
        theta,
        bank,
        ["s1", "s2"],
        1,
        np.random.default_rng(7),
        top_n=1,
        selection="dopt",
        U=covariance,
    )
    assert dopt.mode == "dopt"
    assert dopt.scenario_id == "s2"
    assert dopt.target_skill_index is None


def test_edullm_selector_rejects_unknown_modes_and_invalid_covariance() -> None:
    bank = _runtime_bank([_runtime_rubric("c1", "s1", (1, 0), (1.0, 0.0), 0.0)])
    with pytest.raises(ValueError, match="selection must"):
        select_next(
            np.zeros(2),
            bank,
            ["s1"],
            0,
            np.random.default_rng(0),
            selection="mystery",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="positive definite"):
        scenario_dopt_value(np.zeros(2), np.array([[1.0, 1.0], [1.0, 1.0]]), bank.rubrics_for("s1"))
