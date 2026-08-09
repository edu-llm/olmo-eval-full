"""Stateful scenario-level CAT for EduLLM's confirmatory MIRT banks.

This is a runtime extraction of the state machine in
``origin/frq/infobench:eduLLM-Evals/tutor_cat/engine.py`` (blob
``2bc74f68``), with its numerical work delegated to the extracted MIRT,
joint-EAP, MWLE, bank, and selector modules.

The session is provider-independent.  It selects a scenario, accepts one
explicit Pass/Fail/no-decision observation per calibrated criterion, and only
then advances.  OLMo adapters are responsible for obtaining the tutor response
and judge decisions.  Missing judge decisions never become failures.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import numpy as np

from olmo_eval.edullm.ability import AbilityEstimate, Quadrature, batch_eap, mwle
from olmo_eval.edullm.bank import FittedBank, Rubric, Scenario
from olmo_eval.edullm.mirt import initial_state, standard_errors, update
from olmo_eval.edullm.selector import SelectionResult, select_next

StopSEMethod = Literal["eap", "online"]
Observation = int | None
ScenarioResponder = Callable[
    [Scenario, tuple[Rubric, ...]],
    Awaitable[Mapping[str, Observation]],
]


def _threshold_map(
    value: float | Mapping[str, float] | int,
    skills: Sequence[str],
    *,
    label: str,
    integer: bool,
) -> np.ndarray:
    if isinstance(value, Mapping):
        thresholds = cast(Mapping[str, float], value)
        missing = [skill for skill in skills if skill not in thresholds]
        extra = sorted(set(thresholds) - set(skills))
        if missing or extra:
            raise ValueError(
                f"{label} keys differ from bank skills; missing={missing}, extra={extra}"
            )
        values = [thresholds[skill] for skill in skills]
    else:
        values = [value] * len(skills)

    if integer and any(isinstance(item, (bool, np.bool_)) for item in values):
        raise ValueError(f"{label} values must be non-negative integers")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} values must be numeric") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} values must be finite")
    if integer:
        if np.any(result < 0) or np.any(result != np.floor(result)):
            raise ValueError(f"{label} values must be non-negative integers")
        return result.astype(int)
    if np.any(result <= 0):
        raise ValueError(f"{label} values must be strictly positive")
    return result


@dataclass(frozen=True, slots=True)
class CatConfig:
    """Explicit CAT policy; benchmark-specific operating points have no defaults."""

    max_se: float | Mapping[str, float]
    min_evals_per_skill: int | Mapping[str, int]
    min_scenarios: int
    max_scenarios: int
    seed: int = 42
    top_n: int = 5
    selection: Literal["trace", "dopt"] = "trace"
    stop_se_method: StopSEMethod = "eap"
    theta_init: tuple[float, ...] | None = None
    covariance_init_diag: tuple[float, ...] | None = None
    mwle_ridge: float = 1e-6

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_scenarios, bool)
            or not isinstance(self.min_scenarios, (int, np.integer))
            or self.min_scenarios < 0
        ):
            raise ValueError("min_scenarios must be a non-negative integer")
        if (
            isinstance(self.max_scenarios, bool)
            or not isinstance(self.max_scenarios, (int, np.integer))
            or self.max_scenarios < 1
        ):
            raise ValueError("max_scenarios must be a positive integer")
        if self.min_scenarios > self.max_scenarios:
            raise ValueError("min_scenarios cannot exceed max_scenarios")
        if (
            isinstance(self.top_n, bool)
            or not isinstance(self.top_n, (int, np.integer))
            or self.top_n < 1
        ):
            raise ValueError("top_n must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if self.selection not in {"trace", "dopt"}:
            raise ValueError("selection must be 'trace' or 'dopt'")
        if self.stop_se_method not in {"eap", "online"}:
            raise ValueError("stop_se_method must be 'eap' or 'online'")
        if not math.isfinite(self.mwle_ridge) or self.mwle_ridge <= 0:
            raise ValueError("mwle_ridge must be finite and strictly positive")


@dataclass(frozen=True, slots=True)
class CriterionObservation:
    criterion_id: str
    scenario_id: str
    value: Observation
    probability_before: float | None
    theta_after: tuple[float, ...]
    online_se_after: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ScenarioStep:
    step: int
    scenario_id: str
    selection_mode: str
    target_skill: str | None
    selection_value: float
    observations: tuple[CriterionObservation, ...]
    theta_after: tuple[float, ...]
    online_se_after: tuple[float, ...]
    stop_se_after: tuple[float, ...]
    counts_after: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts_after", MappingProxyType(dict(self.counts_after)))


@dataclass(frozen=True, slots=True)
class CatResult:
    """Complete adaptive result without replacing unavailable MWLE values."""

    stop_reason: str
    precision_reached: bool
    stop_se_method: StopSEMethod
    scenarios_administered: tuple[str, ...]
    criteria_observed: int
    criteria_no_decision: int
    counts: Mapping[str, int]
    theta_online: Mapping[str, float]
    se_online: Mapping[str, float]
    theta_eap: Mapping[str, float]
    se_eap: Mapping[str, float]
    theta_mwle: Mapping[str, float] | None
    se_mwle: Mapping[str, float] | None
    mwle_converged: bool
    mwle_message: str
    critical_failures: tuple[str, ...]
    steps: tuple[ScenarioStep, ...]

    def __post_init__(self) -> None:
        for name in (
            "counts",
            "theta_online",
            "se_online",
            "theta_eap",
            "se_eap",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if self.theta_mwle is not None:
            object.__setattr__(self, "theta_mwle", MappingProxyType(dict(self.theta_mwle)))
        if self.se_mwle is not None:
            object.__setattr__(self, "se_mwle", MappingProxyType(dict(self.se_mwle)))


class CatSession:
    """One deterministic adaptive session over an immutable fitted bank."""

    def __init__(
        self,
        bank: FittedBank,
        quadrature: Quadrature,
        config: CatConfig,
    ) -> None:
        if quadrature.grid.ndim != 2 or quadrature.grid.shape[1] != bank.n_dims:
            raise ValueError("quadrature dimensions do not match fitted bank")
        if not np.allclose(
            np.asarray(quadrature.correlation, dtype=float),
            bank.latent_correlation,
            atol=1e-8,
        ):
            raise ValueError("quadrature latent correlation does not match fitted bank")

        self.bank = bank
        self.quadrature = quadrature
        self.config = config
        self._max_se = _threshold_map(config.max_se, bank.skills, label="max_se", integer=False)
        self._minimum_counts = _threshold_map(
            config.min_evals_per_skill,
            bank.skills,
            label="min_evals_per_skill",
            integer=True,
        )
        self.theta, self.covariance = initial_state(
            list(config.theta_init) if config.theta_init is not None else None,
            (
                list(config.covariance_init_diag)
                if config.covariance_init_diag is not None
                else None
            ),
            bank.n_dims,
        )
        if not np.all(np.isfinite(self.theta)):
            raise ValueError("theta_init must contain only finite values")
        covariance_diagonal = np.diag(self.covariance)
        if not np.all(np.isfinite(covariance_diagonal)) or np.any(covariance_diagonal <= 0):
            raise ValueError("covariance_init_diag must contain only finite positive values")
        self.counts = np.zeros(bank.n_dims, dtype=int)
        self.responses = np.full(bank.n_items, np.nan, dtype=float)
        self.rng = np.random.default_rng(config.seed)
        self.administered: list[str] = []
        self.steps: list[ScenarioStep] = []
        self.critical_failures: list[str] = []
        self.criteria_no_decision = 0
        self._pending: SelectionResult | None = None
        self.stop_reason: str | None = None

    @property
    def pending_scenario_id(self) -> str | None:
        return self._pending.scenario_id if self._pending is not None else None

    @property
    def observed_indices(self) -> np.ndarray:
        return np.flatnonzero(np.isfinite(self.responses))

    def _eap(self) -> AbilityEstimate:
        return batch_eap(
            self.responses,
            self.bank.A,
            self.bank.b,
            self.quadrature,
            self.observed_indices,
        )

    def stop_standard_errors(self) -> np.ndarray:
        if self.config.stop_se_method == "online":
            return standard_errors(self.covariance)
        return self._eap().se

    def precision_reached(self) -> bool:
        return bool(
            len(self.administered) >= self.config.min_scenarios
            and np.all(self.counts >= self._minimum_counts)
            and np.all(self.stop_standard_errors() <= self._max_se)
        )

    def _terminal_reason(self) -> str | None:
        if self.precision_reached():
            return "precision_reached"
        if len(self.administered) >= self.config.max_scenarios:
            return "max_scenarios_reached"
        if len(self.administered) >= len(self.bank.scenarios):
            return "bank_exhausted"
        return None

    def next_scenario(self) -> SelectionResult | None:
        """Select one scenario; the caller must record it before selecting again."""

        if self._pending is not None:
            raise RuntimeError(f"scenario {self._pending.scenario_id!r} is awaiting observations")
        if self.stop_reason is not None:
            return None
        terminal = self._terminal_reason()
        if terminal is not None:
            self.stop_reason = terminal
            return None

        unused = [sid for sid in self.bank.scenarios if sid not in self.administered]
        if not unused:
            self.stop_reason = "bank_exhausted"
            return None
        target = int(np.argmax(standard_errors(self.covariance)))
        selection = select_next(
            self.theta,
            self.bank,
            unused,
            target,
            self.rng,
            self.config.top_n,
            selection=self.config.selection,
            U=self.covariance,
        )
        if selection is None:
            self.stop_reason = "bank_exhausted"
            return None
        self._pending = selection
        return selection

    def record_scenario(self, observations: Mapping[str, Observation]) -> ScenarioStep:
        """Record every criterion for the selected scenario at one scenario boundary."""

        if self._pending is None:
            raise RuntimeError("record_scenario requires a pending selected scenario")
        selection = self._pending
        rubrics = tuple(self.bank.rubrics_for(selection.scenario_id))
        required_ids = [rubric.criterion_id for rubric in rubrics]
        missing = [
            criterion_id for criterion_id in required_ids if criterion_id not in observations
        ]
        extra = sorted(set(observations) - set(required_ids))
        if missing or extra:
            raise ValueError(
                "scenario observations must name every calibrated criterion exactly once; "
                f"missing={missing}, extra={extra}"
            )

        recorded: list[CriterionObservation] = []
        for rubric in rubrics:
            value = observations[rubric.criterion_id]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value not in {0, 1}
            ):
                raise ValueError(f"{rubric.criterion_id}: observation must be 0, 1, or None")
            probability_before: float | None = None
            if value is None:
                self.criteria_no_decision += 1
            else:
                index = self.bank.index[rubric.criterion_id]
                if np.isfinite(self.responses[index]):
                    raise RuntimeError(f"criterion {rubric.criterion_id} was already observed")
                self.theta, self.covariance, probability_before = update(
                    self.theta,
                    self.covariance,
                    rubric.a,
                    rubric.q,
                    rubric.b,
                    int(value),
                )
                self.responses[index] = int(value)
                self.counts += rubric.q
                if rubric.criticality.startswith("critical") and int(value) == 0:
                    self.critical_failures.append(rubric.criterion_id)
            recorded.append(
                CriterionObservation(
                    criterion_id=rubric.criterion_id,
                    scenario_id=rubric.scenario_id,
                    value=None if value is None else int(value),
                    probability_before=probability_before,
                    theta_after=tuple(float(item) for item in self.theta),
                    online_se_after=tuple(float(item) for item in standard_errors(self.covariance)),
                )
            )

        self.administered.append(selection.scenario_id)
        self._pending = None
        target_skill = (
            self.bank.skills[selection.target_skill_index]
            if selection.target_skill_index is not None
            else None
        )
        step = ScenarioStep(
            step=len(self.administered),
            scenario_id=selection.scenario_id,
            selection_mode=selection.mode,
            target_skill=target_skill,
            selection_value=float(selection.value),
            observations=tuple(recorded),
            theta_after=tuple(float(item) for item in self.theta),
            online_se_after=tuple(float(item) for item in standard_errors(self.covariance)),
            stop_se_after=tuple(float(item) for item in self.stop_standard_errors()),
            counts_after={
                skill: int(self.counts[index]) for index, skill in enumerate(self.bank.skills)
            },
        )
        self.steps.append(step)
        return step

    def result(self) -> CatResult:
        if self._pending is not None:
            raise RuntimeError("cannot finalize while a selected scenario is pending")
        if self.stop_reason is None:
            terminal = self._terminal_reason()
            self.stop_reason = terminal or "stopped_by_caller"

        eap = self._eap()
        weighted = mwle(
            self.responses,
            self.bank.A,
            self.bank.b,
            self.observed_indices,
            theta0=eap.theta,
            ridge=self.config.mwle_ridge,
        )
        skills = self.bank.skills
        theta_mwle = (
            {skill: float(weighted.theta[index]) for index, skill in enumerate(skills)}
            if weighted.converged
            else None
        )
        se_mwle = (
            {skill: float(weighted.se[index]) for index, skill in enumerate(skills)}
            if weighted.converged and np.all(np.isfinite(weighted.se))
            else None
        )
        return CatResult(
            stop_reason=self.stop_reason,
            precision_reached=self.precision_reached(),
            stop_se_method=self.config.stop_se_method,
            scenarios_administered=tuple(self.administered),
            criteria_observed=int(self.observed_indices.size),
            criteria_no_decision=self.criteria_no_decision,
            counts={skill: int(self.counts[index]) for index, skill in enumerate(skills)},
            theta_online={skill: float(self.theta[index]) for index, skill in enumerate(skills)},
            se_online={
                skill: float(standard_errors(self.covariance)[index])
                for index, skill in enumerate(skills)
            },
            theta_eap={skill: float(eap.theta[index]) for index, skill in enumerate(skills)},
            se_eap={skill: float(eap.se[index]) for index, skill in enumerate(skills)},
            theta_mwle=theta_mwle,
            se_mwle=se_mwle,
            mwle_converged=weighted.converged,
            mwle_message=weighted.message,
            critical_failures=tuple(self.critical_failures),
            steps=tuple(self.steps),
        )


async def run_cat(
    bank: FittedBank,
    quadrature: Quadrature,
    config: CatConfig,
    responder: ScenarioResponder,
) -> CatResult:
    """Drive a session; only adaptively selected scenarios reach ``responder``."""

    session = CatSession(bank, quadrature, config)
    while (selection := session.next_scenario()) is not None:
        scenario = bank.scenarios[selection.scenario_id]
        rubrics = tuple(bank.rubrics_for(selection.scenario_id))
        observations = await responder(scenario, rubrics)
        session.record_scenario(observations)
    return session.result()


__all__ = [
    "CatConfig",
    "CatResult",
    "CatSession",
    "CriterionObservation",
    "Observation",
    "ScenarioResponder",
    "ScenarioStep",
    "StopSEMethod",
    "run_cat",
]
