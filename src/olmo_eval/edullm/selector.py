"""Fisher-information scenario selection for calibrated EduLLM CAT.

Ported from ``origin/frq/infobench:eduLLM-Evals/tutor_cat/selector.py`` (blob
``19e634ad``).  This preserves the production trace/Fisher rule, seeded uniform
sampling within the ranked top-N pool, deterministic total-information fallback, and
the optional covariance-aware D-optimal rule.  The implementation is latent-dimension
generic and uses the strict runtime schemas from :mod:`olmo_eval.edullm.bank`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from olmo_eval.edullm.bank import FittedBank, Rubric
from olmo_eval.edullm.mirt import pass_probability as mirt_pass_probability

SelectionMode = Literal["trace", "dopt"]


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Recorded result of one scenario-selection decision."""

    scenario_id: str
    mode: Literal["target_skill", "fallback_total_info", "dopt"]
    target_skill_index: int | None
    value: float
    top_candidates: tuple[tuple[str, float], ...]


def pass_probability(theta: np.ndarray, rubric: Rubric) -> float:
    """Return ``sigmoid((q * a) @ theta - b)`` with strict shape checks."""

    ability = _theta(theta, rubric.q.shape[0])
    return mirt_pass_probability(ability, rubric.a, rubric.q, rubric.b)


def criterion_information(theta: np.ndarray, rubric: Rubric, k: int) -> float:
    """Return ``q[k] * P(1-P) * a[k]^2`` for one criterion and skill."""

    _skill_index(k, rubric.q.shape[0])
    if rubric.q[k] == 0:
        return 0.0
    probability = pass_probability(theta, rubric)
    return float(probability * (1.0 - probability) * rubric.a[k] ** 2)


def scenario_value(theta: np.ndarray, rubrics: list[Rubric], k: int) -> float | None:
    """Average target-skill information per applicable criterion.

    ``None`` identifies a scenario that does not measure the target skill.  Criteria
    loading only on other skills do not enter the cost denominator, matching the
    authoritative selector.
    """

    _nonempty_rubrics(rubrics)
    n_dims = rubrics[0].q.shape[0]
    _aligned_rubrics(rubrics, n_dims)
    _skill_index(k, n_dims)
    applicable = sum(int(rubric.q[k]) for rubric in rubrics)
    if applicable == 0:
        return None
    return float(sum(criterion_information(theta, rubric, k) for rubric in rubrics) / applicable)


def total_information_value(theta: np.ndarray, rubrics: list[Rubric]) -> float:
    """Return total Fisher trace per administered/scorable criterion."""

    _nonempty_rubrics(rubrics)
    n_dims = rubrics[0].q.shape[0]
    _aligned_rubrics(rubrics, n_dims)
    ability = _theta(theta, n_dims)
    total = 0.0
    for rubric in rubrics:
        probability = pass_probability(ability, rubric)
        loading = rubric.q * rubric.a
        total += probability * (1.0 - probability) * float(loading @ loading)
    return float(total / len(rubrics))


def scenario_dopt_value(
    theta: np.ndarray,
    U: np.ndarray,
    rubrics: list[Rubric],
) -> float:
    """Return per-criterion D-optimal log-determinant information gain.

    ``U`` is the current positive-definite posterior covariance.  Administering the
    complete scenario adds each criterion's rank-one Fisher information to ``U^-1``.
    """

    _nonempty_rubrics(rubrics)
    n_dims = rubrics[0].q.shape[0]
    _aligned_rubrics(rubrics, n_dims)
    ability = _theta(theta, n_dims)
    covariance = _covariance(U, n_dims)
    information = np.linalg.inv(covariance)
    base_sign, base_logdet = np.linalg.slogdet(information)
    if base_sign <= 0 or not math.isfinite(float(base_logdet)):
        raise ValueError("U inverse must have a finite positive determinant")

    updated = information.copy()
    for rubric in rubrics:
        loading = rubric.q * rubric.a
        probability = pass_probability(ability, rubric)
        updated += probability * (1.0 - probability) * np.outer(loading, loading)
    new_sign, new_logdet = np.linalg.slogdet(updated)
    if new_sign <= 0 or not math.isfinite(float(new_logdet)):
        raise ValueError("updated information matrix has no finite positive determinant")
    return float(new_logdet - base_logdet) / len(rubrics)


def select_next(
    theta: np.ndarray,
    bank: FittedBank,
    unused_scenario_ids: list[str],
    target_skill_index: int,
    rng: np.random.Generator,
    top_n: int = 5,
    selection: SelectionMode = "trace",
    U: np.ndarray | None = None,
) -> SelectionResult | None:
    """Choose the next scenario using the explicitly selected policy.

    Trace/Fisher ranking samples uniformly from the top ``top_n`` applicable
    scenarios using ``rng``.  If no unused scenario measures the target skill, the
    highest total-information scenario is chosen deterministically.  D-optimal ranking
    also samples from its seeded top-N pool and requires covariance ``U``.
    """

    if selection not in ("trace", "dopt"):
        raise ValueError("selection must be 'trace' or 'dopt'")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    _theta(theta, bank.n_dims)
    if len(unused_scenario_ids) != len(set(unused_scenario_ids)):
        raise ValueError("unused_scenario_ids contains duplicates")
    unknown = [sid for sid in unused_scenario_ids if sid not in bank.scenarios]
    if unknown:
        raise ValueError(f"unknown unused scenario_id(s): {unknown[:10]}")
    if not unused_scenario_ids:
        return None

    if selection == "dopt":
        if U is None:
            raise ValueError("selection='dopt' requires posterior covariance U")
        scored = [
            (sid, scenario_dopt_value(theta, U, bank.rubrics_for(sid)))
            for sid in unused_scenario_ids
        ]
        top = _ranked_top(scored, top_n)
        scenario_id, value = top[int(rng.integers(len(top)))]
        return SelectionResult(
            scenario_id=scenario_id,
            mode="dopt",
            target_skill_index=None,
            value=value,
            top_candidates=tuple(top),
        )

    _skill_index(target_skill_index, bank.n_dims)
    scored_applicable: list[tuple[str, float]] = []
    for scenario_id in unused_scenario_ids:
        value = scenario_value(theta, bank.rubrics_for(scenario_id), target_skill_index)
        if value is not None:
            scored_applicable.append((scenario_id, value))

    if scored_applicable:
        top = _ranked_top(scored_applicable, top_n)
        scenario_id, value = top[int(rng.integers(len(top)))]
        return SelectionResult(
            scenario_id=scenario_id,
            mode="target_skill",
            target_skill_index=target_skill_index,
            value=value,
            top_candidates=tuple(top),
        )

    fallback = _ranked_top(
        [
            (scenario_id, total_information_value(theta, bank.rubrics_for(scenario_id)))
            for scenario_id in unused_scenario_ids
        ],
        top_n,
    )
    scenario_id, value = fallback[0]
    return SelectionResult(
        scenario_id=scenario_id,
        mode="fallback_total_info",
        target_skill_index=None,
        value=value,
        top_candidates=tuple(fallback),
    )


def _ranked_top(values: list[tuple[str, float]], top_n: int) -> list[tuple[str, float]]:
    if not values:
        raise ValueError("cannot rank an empty candidate set")
    if any(not math.isfinite(value) for _, value in values):
        raise ValueError("scenario information values must be finite")
    return sorted(values, key=lambda item: (-item[1], item[0]))[:top_n]


def _theta(theta: np.ndarray, n_dims: int) -> np.ndarray:
    ability = np.asarray(theta, dtype=float)
    if ability.shape != (n_dims,):
        raise ValueError(f"theta shape {ability.shape} != ({n_dims},)")
    if not np.isfinite(ability).all():
        raise ValueError("theta must be finite")
    return ability


def _covariance(U: np.ndarray, n_dims: int) -> np.ndarray:
    covariance = np.asarray(U, dtype=float)
    if covariance.shape != (n_dims, n_dims):
        raise ValueError(f"U shape {covariance.shape} != ({n_dims}, {n_dims})")
    if not np.isfinite(covariance).all():
        raise ValueError("U must be finite")
    if not np.allclose(covariance, covariance.T, atol=1e-9, rtol=0.0):
        raise ValueError("U must be symmetric")
    if float(np.linalg.eigvalsh(covariance).min()) <= 0:
        raise ValueError("U must be positive definite")
    return covariance


def _skill_index(k: int, n_dims: int) -> None:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("skill index must be an integer")
    if not 0 <= int(k) < n_dims:
        raise IndexError(f"skill index {k} outside [0, {n_dims})")


def _nonempty_rubrics(rubrics: list[Rubric]) -> None:
    if not rubrics:
        raise ValueError("scenario must contain at least one rubric")


def _aligned_rubrics(rubrics: list[Rubric], n_dims: int) -> None:
    if any(rubric.q.shape != (n_dims,) or rubric.a.shape != (n_dims,) for rubric in rubrics):
        raise ValueError("scenario rubrics do not share one latent-dimension axis")
