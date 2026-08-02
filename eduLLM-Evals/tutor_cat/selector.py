"""Choosing Next Scenario (PRD): Fisher-information scenario selection.

    V_kc(theta) = q_kc × P_c(theta) × (1 − P_c(theta)) × a_kc²
    ScenarioValue(S, k) = sum_{c in S} V_kc(theta) / #{c in S : q_kc = 1}

Selection: rank eligible scenarios descending, take top n (default 5), pick one
uniformly at random with the run's seeded RNG.

Fallback (no eligible scenario for the target skill): the unused scenario with
the highest total information summed across all three skills, per criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataio import ItemBank
from .mirt import pass_probability
from .schemas import Rubric


@dataclass
class SelectionResult:
    scenario_id: str
    mode: str                              # "target_skill" | "fallback_total_info"
    target_skill_index: int | None
    value: float
    top_candidates: list[tuple[str, float]]  # (scenario_id, value) of the ranked pool


def criterion_information(theta: np.ndarray, rubric: Rubric, k: int) -> float:
    """V_kc = q_kc * P(1-P) * a_kc^2 for target skill index k."""
    if rubric.q[k] == 0:
        return 0.0
    p = pass_probability(theta, rubric.a, rubric.q, rubric.b)
    return float(p * (1.0 - p) * rubric.a[k] ** 2)


def scenario_value(theta: np.ndarray, rubrics: list[Rubric], k: int) -> float | None:
    """Average information about skill k per applicable criterion; None if inapplicable."""
    applicable = sum(int(r.q[k]) for r in rubrics)
    if applicable == 0:
        return None
    total = sum(criterion_information(theta, r, k) for r in rubrics)
    return total / applicable


def total_information_value(theta: np.ndarray, rubrics: list[Rubric]) -> float:
    """Fallback score: information summed across all skills, per criterion."""
    total = 0.0
    for r in rubrics:
        p = pass_probability(theta, r.a, r.q, r.b)
        total += p * (1.0 - p) * float(((r.q * r.a) ** 2).sum())
    return total / len(rubrics)


def scenario_dopt_value(theta: np.ndarray, U: np.ndarray, rubrics: list[Rubric]) -> float:
    """D-optimality score for administering an ENTIRE scenario (all its criteria).

    Administering the scenario updates the information matrix by a sum of rank-1 terms:
        I_new = U^-1 + sum_c p_c(1-p_c) (q_c ⊙ a_c)(q_c ⊙ a_c)^T
    D-optimality maximises det(I_new); we score the log-det gain over U^-1 so the value
    is comparable across steps. Unlike the trace/`scenario_value` rule, this uses the
    current posterior covariance U, so it prefers scenarios that shrink the widest
    remaining uncertainty direction. Normalised per scorable criterion so it measures
    information per unit judging cost, matching `scenario_value`'s cost normalisation.
    """
    info = np.linalg.inv(U)
    _, base_logdet = np.linalg.slogdet(info)
    m_new = info.copy()
    n_scorable = 0
    for r in rubrics:
        if int(r.q.sum()) == 0:
            continue
        n_scorable += 1
        m = r.q * r.a
        p = pass_probability(theta, r.a, r.q, r.b)
        m_new = m_new + (p * (1.0 - p)) * np.outer(m, m)
    if n_scorable == 0:
        return 0.0
    _, new_logdet = np.linalg.slogdet(m_new)
    return float(new_logdet - base_logdet) / n_scorable


def select_next(
    theta: np.ndarray,
    bank: ItemBank,
    unused_scenario_ids: list[str],
    target_skill_index: int,
    rng: np.random.Generator,
    top_n: int = 5,
    selection: str = "trace",
    U: np.ndarray | None = None,
) -> SelectionResult | None:
    """Pick the next scenario. Returns None if the bank is exhausted.

    ``selection``:
      * ``"trace"`` (default, PRD production): rank scenarios by per-criterion Fisher
        information for the ``target_skill_index`` at the plug-in ``theta``.
      * ``"dopt"``: rank scenarios by the D-optimality log-det gain of administering the
        whole scenario, using the posterior covariance ``U`` (uncertainty-aware). The
        target skill is not used; ``U`` is required.
    """
    if not unused_scenario_ids:
        return None

    if selection == "dopt":
        if U is None:
            raise ValueError("selection='dopt' requires the posterior covariance U")
        scored = [
            (sid, scenario_dopt_value(theta, U, bank.rubrics_for(sid)))
            for sid in unused_scenario_ids
        ]
        scored.sort(key=lambda t: (-t[1], t[0]))
        top = scored[:top_n]
        sid, value = top[int(rng.integers(len(top)))]
        return SelectionResult(sid, "dopt", None, value, top)

    scored_opt: list[tuple[str, float]] = []
    for sid in unused_scenario_ids:
        value = scenario_value(theta, bank.rubrics_for(sid), target_skill_index)
        if value is not None:
            scored_opt.append((sid, value))

    if scored_opt:
        # Deterministic order: value descending, scenario_id as tie-break.
        scored_opt.sort(key=lambda t: (-t[1], t[0]))
        top = scored_opt[:top_n]
        sid, value = top[int(rng.integers(len(top)))]
        return SelectionResult(sid, "target_skill", target_skill_index, value, top)

    # Fallback: no unused scenario touches the target skill.
    fallback = [
        (sid, total_information_value(theta, bank.rubrics_for(sid)))
        for sid in unused_scenario_ids
    ]
    fallback.sort(key=lambda t: (-t[1], t[0]))
    sid, value = fallback[0]
    return SelectionResult(sid, "fallback_total_info", None, value, fallback[:top_n])
