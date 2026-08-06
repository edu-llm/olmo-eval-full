"""The generic free-response Computerized Adaptive Testing engine.

This engine is style-agnostic: it drives any
:class:`~diagnostics.frq_cat.base.FrqCatStyle` through the standard CAT loop and
returns the style's report. FRQ grading is two-stage and shared: the engine asks
the style to select the next criterion, ensures the criterion's scenario has a
generated response (cached per scenario, since one response covers all of a
scenario's criteria), grades that criterion with the frozen judge, updates the
ability estimate, then tests the stopping rule. All model- and selection-specific
behavior lives behind the ``base.py`` interface, so this module never changes when
a style is added.
"""

from __future__ import annotations

import logging

from ..base import (
    CATReport,
    CATState,
    CriterionResponse,
    FrqBank,
    FrqCatStyle,
    IRTBank,
    Judge,
    RespGenModel,
)

log = logging.getLogger("frq_cat.cat_loop")


def run_cat(
    style: FrqCatStyle,
    *,
    bank: FrqBank,
    irt_bank: IRTBank,
    respgen: RespGenModel,
    judge: Judge,
    se_threshold: float | None = None,
    max_items: int | None = None,
) -> CATReport:
    """Run one adaptive FRQ test session and return the style's report.

    Args:
        style: The resolved CAT style supplying the IRT model and selector.
        bank: The prepared scenarios + criteria.
        irt_bank: IRT parameters for the criteria.
        respgen: The checkpoint under test (generates responses per scenario).
        judge: The frozen LLM-as-a-judge (grades a response per criterion).
        se_threshold: Standard-error stop threshold, passed to the style's rule.
        max_items: Hard cap on administered criteria, enforced by the engine.

    Returns:
        The completed :class:`CATReport` produced by ``style.report``.
    """
    state = CATState(benchmark=bank.name, se_threshold=se_threshold, max_items=max_items)
    response_cache: dict[str, str] = {}

    while True:
        if max_items is not None and state.step >= max_items:
            log.info("Stopping: reached max_items=%d", max_items)
            break

        next_id = style.select_next_item(irt_bank, state)
        if next_id is None:
            log.info("Stopping: no further criteria to administer")
            break

        criterion = bank.get(next_id)
        scenario = bank.get_scenario(criterion.scenario_id)

        # One response covers all of a scenario's criteria: generate once, then cache.
        if scenario.scenario_id not in response_cache:
            response_cache[scenario.scenario_id] = respgen.generate([scenario])[
                scenario.scenario_id
            ]
        response_text = response_cache[scenario.scenario_id]

        verdict = judge.evaluate(scenario, criterion, response_text)
        state.administered.append(
            CriterionResponse(
                criterion_id=next_id,
                correct=verdict.passed,
                verdict=verdict,
            )
        )

        state.ability = style.estimate_ability(irt_bank, state.administered, previous=state.ability)
        log.info(
            "Administered %s (step %d): theta=%s se=%s",
            next_id,
            state.step,
            state.ability.theta,
            state.ability.standard_error,
        )

        if style.stopping_rule(state):
            log.info("Stopping: style stopping rule satisfied at step %d", state.step)
            break

    return style.report(state)
