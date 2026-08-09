"""Behavioral tests for the provider-independent EduLLM CAT state machine."""

from __future__ import annotations

import asyncio
from types import MappingProxyType

import numpy as np
import pytest

from olmo_eval.edullm.ability import build_quadrature
from olmo_eval.edullm.bank import FittedBank, Rubric, Scenario
from olmo_eval.edullm.cat import CatConfig, CatSession, run_cat


def _bank(
    scenario_difficulties: tuple[tuple[str, tuple[float, ...]], ...],
    *,
    critical: str | None = None,
) -> FittedBank:
    scenarios: dict[str, Scenario] = {}
    rubrics: dict[str, Rubric] = {}
    for scenario_id, difficulties in scenario_difficulties:
        criterion_ids = tuple(
            f"{scenario_id}_c{index:02d}" for index in range(1, len(difficulties) + 1)
        )
        scenarios[scenario_id] = Scenario(
            scenario_id=scenario_id,
            prompt=f"Prompt for {scenario_id}",
            criterion_ids=criterion_ids,
        )
        for criterion_id, difficulty in zip(criterion_ids, difficulties, strict=True):
            rubrics[criterion_id] = Rubric(
                criterion_id=criterion_id,
                scenario_id=scenario_id,
                criterion=f"Requirement for {criterion_id}",
                q=np.array([1], dtype=np.int8),
                a=np.array([2.0]),
                b=difficulty,
                calibration_version="fixture-v1",
                criticality=("critical_safety" if criterion_id == critical else "standard"),
            )
    return FittedBank(
        scenarios=MappingProxyType(scenarios),
        rubrics=MappingProxyType(rubrics),
        skills=("instruction_following",),
        latent_correlation=np.eye(1),
        records=(),
        source_path="fixture-rubrics.jsonl",
        scenarios_path="fixture-scenarios.jsonl",
    )


def _config(
    *,
    max_se: float = 0.01,
    min_scenarios: int = 0,
    max_scenarios: int = 3,
) -> CatConfig:
    return CatConfig(
        max_se=max_se,
        min_evals_per_skill=0,
        min_scenarios=min_scenarios,
        max_scenarios=max_scenarios,
        top_n=1,
        seed=7,
        selection="trace",
        stop_se_method="eap",
    )


def _quadrature():  # type annotation would obscure this compact fixture
    return build_quadrature(1, 21, np.eye(1))


def test_cat_branches_adaptively_after_first_binary_observation() -> None:
    bank = _bank((("mid", (0.0,)), ("easy", (-2.0,)), ("hard", (2.0,))))

    failed = CatSession(bank, _quadrature(), _config(max_scenarios=2))
    assert failed.next_scenario().scenario_id == "mid"  # type: ignore[union-attr]
    failed.record_scenario({"mid_c01": 0})
    assert failed.next_scenario().scenario_id == "easy"  # type: ignore[union-attr]

    passed = CatSession(bank, _quadrature(), _config(max_scenarios=2))
    assert passed.next_scenario().scenario_id == "mid"  # type: ignore[union-attr]
    passed.record_scenario({"mid_c01": 1})
    assert passed.next_scenario().scenario_id == "hard"  # type: ignore[union-attr]


def test_eap_stop_is_checked_only_after_complete_scenario_boundary() -> None:
    bank = _bank((("multi", (0.0, 0.0)), ("unused", (0.0,))))
    calls: list[str] = []

    async def responder(scenario, rubrics):  # type: ignore[no-untyped-def]
        calls.append(scenario.scenario_id)
        return {rubric.criterion_id: 1 for rubric in rubrics}

    result = asyncio.run(
        run_cat(
            bank,
            _quadrature(),
            _config(max_se=2.0, min_scenarios=1),
            responder,
        )
    )

    assert calls == ["multi"]
    assert result.stop_reason == "precision_reached"
    assert result.criteria_observed == 2
    assert len(result.steps[0].observations) == 2


def test_minimum_scenario_floor_is_part_of_final_precision() -> None:
    bank = _bank((("first", (0.0,)), ("second", (-1.0,)), ("third", (1.0,))))

    async def responder(_scenario, rubrics):  # type: ignore[no-untyped-def]
        return {rubric.criterion_id: 1 for rubric in rubrics}

    result = asyncio.run(
        run_cat(
            bank,
            _quadrature(),
            _config(max_se=2.0, min_scenarios=2),
            responder,
        )
    )

    assert result.precision_reached is True
    assert result.stop_reason == "precision_reached"
    assert len(result.scenarios_administered) == 2


def test_no_decision_is_excluded_not_converted_to_failure() -> None:
    bank = _bank((("only", (0.0,)),))
    session = CatSession(
        bank,
        _quadrature(),
        _config(max_se=0.01, max_scenarios=1),
    )
    assert session.next_scenario() is not None
    session.record_scenario({"only_c01": None})
    result = session.result()

    assert np.isnan(session.responses[0])
    assert result.criteria_observed == 0
    assert result.criteria_no_decision == 1
    assert result.counts == {"instruction_following": 0}
    assert result.theta_online == {"instruction_following": 0.0}
    assert result.mwle_converged is False
    assert result.theta_mwle is None


def test_final_result_reports_mwle_and_critical_failures() -> None:
    bank = _bank(
        (("low", (-1.0,)), ("mid", (0.0,)), ("high", (1.0,))),
        critical="mid_c01",
    )
    session = CatSession(
        bank,
        _quadrature(),
        _config(max_se=0.01, max_scenarios=3),
    )
    while (selection := session.next_scenario()) is not None:
        value = 0 if selection.scenario_id == "mid" else 1
        criterion_id = bank.scenarios[selection.scenario_id].criterion_ids[0]
        session.record_scenario({criterion_id: value})
    result = session.result()

    assert result.mwle_converged is True
    assert result.theta_mwle is not None
    assert result.se_mwle is not None
    assert result.critical_failures == ("mid_c01",)


def test_seeded_selection_is_reproducible() -> None:
    bank = _bank(tuple((f"s{index}", (0.0,)) for index in range(6)))
    config = CatConfig(
        max_se=0.01,
        min_evals_per_skill=0,
        min_scenarios=0,
        max_scenarios=4,
        top_n=5,
        seed=1234,
    )

    def path() -> list[str]:
        session = CatSession(bank, _quadrature(), config)
        selected: list[str] = []
        while (choice := session.next_scenario()) is not None:
            selected.append(choice.scenario_id)
            criterion_id = bank.scenarios[choice.scenario_id].criterion_ids[0]
            session.record_scenario({criterion_id: len(selected) % 2})
        return selected

    assert path() == path()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("min_scenarios", 1.5),
        ("max_scenarios", True),
        ("top_n", 2.5),
        ("seed", False),
        ("min_evals_per_skill", True),
    ),
)
def test_cat_rejects_noninteger_policy_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "max_se": 1.0,
        "min_evals_per_skill": 0,
        "min_scenarios": 0,
        "max_scenarios": 3,
        "top_n": 1,
        "seed": 7,
    }
    values[field] = value
    if field == "min_evals_per_skill":
        config = CatConfig(**values)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-negative integers"):
            CatSession(_bank((("only", (0.0,)),)), _quadrature(), config)
    else:
        with pytest.raises(ValueError):
            CatConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "diag",
    ((0.0,), (-1.0,), (float("nan"),)),
)
def test_cat_rejects_invalid_initial_covariance(diag: tuple[float, ...]) -> None:
    config = CatConfig(
        max_se=1.0,
        min_evals_per_skill=0,
        min_scenarios=0,
        max_scenarios=1,
        covariance_init_diag=diag,
    )
    with pytest.raises(ValueError, match="finite positive"):
        CatSession(_bank((("only", (0.0,)),)), _quadrature(), config)
