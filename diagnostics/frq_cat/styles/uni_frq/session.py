"""A CAT session that survives bad scenarios instead of dying on the first one.

The shared engine calls respgen and the judge with no error handling, so one HTTP 400 (for
example a TutorEval prompt that overflows the tutor's context window) raises out of the
whole run and the runner writes nothing. Measured on the shipped bank, 16.3% of scenarios
exceed a 4096-token window, so that is the expected case rather than an edge case.

This session keeps the same adaptive logic and the same style interface, and adds:

- a scenario that cannot be served (or answers with nothing) is dropped along with all of
  its criteria, and the run continues with the next most informative item;
- a judge that cannot produce a scorable verdict abstains; the criterion is excluded and
  is *not* fed to the ability estimator as a failure. A judge that *raises* is treated the
  same way, because no single bad reply should be able to end a paid run;
- every branch is flushed through the sink as it happens, so a crash keeps partial results
  no matter which path the run was on;
- per-call timings are recorded so wall-clock can be projected before a large run.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Protocol

from ...base import CATState, CriterionResponse, FrqBank, IRTBank, JudgeVerdict
from .respgen_client import ScenarioUnservable, TutorUnavailable
from .result_sink import ResultSink

log = logging.getLogger("uni_frq.session")


class _RespGen(Protocol):
    """The tutor seam: anything that can answer one scenario (real client or a stub)."""

    def generate_one(self, scenario: Any) -> str: ...


class _Judge(Protocol):
    """The judge seam: anything that can grade one criterion."""

    def evaluate(self, scenario: Any, criterion: Any, response_text: str) -> Any: ...


def _filtered(bank: IRTBank, excluded: set[str]) -> IRTBank:
    """Return ``bank`` without the excluded criteria, so they cannot be re-selected."""
    if not excluded:
        return bank
    kept = {k: v for k, v in bank.params.items() if k not in excluded}
    return IRTBank(params=kept, dimensions=bank.dimensions)


#: Attempts allowed per intended scored criterion. A 4096-window tutor cannot serve about
#: 16% of TutorEval scenarios, and a judge abstains occasionally, so a run needs headroom
#: above `max_items`; it does not need the whole bank.
_ATTEMPTS_PER_ITEM = 4

#: Consecutive unscorable verdicts after which the judge is presumed broken. Well above
#: any plausible run of genuinely ambiguous replies, far below the cost of finding out the
#: slow way.
_ABSTAIN_STREAK_LIMIT = 12


def run_session(
    style: Any,
    *,
    bank: FrqBank,
    irt_bank: IRTBank,
    respgen: _RespGen,
    judge: _Judge,
    sink: ResultSink,
    se_threshold: float | None,
    max_items: int | None,
    attempts_per_item: int = _ATTEMPTS_PER_ITEM,
    abstain_streak_limit: int = _ABSTAIN_STREAK_LIMIT,
) -> dict[str, Any]:
    """Run one adaptive session, persisting as it goes. Returns a summary dict."""
    criteria_by_scenario: dict[str, list[str]] = defaultdict(list)
    for criterion in bank.criteria:
        criteria_by_scenario[criterion.scenario_id].append(criterion.criterion_id)

    state = CATState(benchmark=bank.name, se_threshold=se_threshold, max_items=max_items)
    responses: dict[str, str] = {}
    excluded: set[str] = set()
    selectable = irt_bank
    resp_seconds: list[float] = []
    judge_seconds: list[float] = []
    skipped_scenarios: list[dict[str, str]] = []
    abstentions: list[dict[str, str]] = []
    stopped_because = "exhausted-bank"
    attempts = 0
    abstain_streak = 0
    last_abstain_reason = ""
    # Every iteration costs a judge call, and most cost a tutor generation too, so the
    # budget is on attempts rather than on results. The multiplier leaves room for the
    # ~16% of scenarios a 4096-window tutor cannot serve and for occasional abstentions,
    # while still bounding a pathological run to a small multiple of the intended test.
    attempt_budget = (
        max_items * attempts_per_item if max_items is not None else len(irt_bank.params)
    )

    def drop_scenario(scenario_id: str, reason: str) -> None:
        """Exclude a scenario and everything that depends on its response."""
        nonlocal selectable
        excluded.update(criteria_by_scenario[scenario_id])
        selectable = _filtered(irt_bank, excluded)
        record = {"scenario_id": scenario_id, "reason": reason}
        skipped_scenarios.append(record)
        sink.append("skipped.jsonl", record)
        sink.sync()
        log.warning("skipping scenario %s: %s", scenario_id, reason)

    while True:
        if max_items is not None and state.step >= max_items:
            stopped_because = "max-items"
            break
        if attempts >= attempt_budget:
            # `max_items` counts criteria that were SCORED, so on its own it bounds nothing:
            # an abstention or a dropped scenario never increments it. A judge that cannot
            # be reached therefore walks the entire bank, paying for a tutor generation and
            # a judge call per criterion, and still reports the prior. Measured on the
            # shipped bank: 1,186 judge calls and 702 generations under `--max-items 5`.
            stopped_because = f"attempt-budget-exhausted: {attempts} attempts, {state.step} scored"
            log.error(
                "stopping: %d attempts made for %d scored criteria (budget %d). "
                "Nothing is being scored; check the judge.",
                attempts,
                state.step,
                attempt_budget,
            )
            break
        if abstain_streak >= abstain_streak_limit:
            # A run that cannot score anything is a broken judge, not a bad tutor, and the
            # sooner it says so the less of the window it spends proving it.
            stopped_because = f"judge-unusable: {abstain_streak} consecutive unscorable verdicts"
            log.error(
                "stopping: %d consecutive criteria came back unscorable; the judge is not "
                "grading. Last reason: %s",
                abstain_streak,
                last_abstain_reason,
            )
            break

        attempts += 1
        next_id = style.select_next_item(selectable, state)
        if next_id is None:
            if excluded:
                stopped_because = "no-scorable-items-left"
            break
        criterion = bank.get(next_id)
        scenario = bank.get_scenario(criterion.scenario_id)

        # --- step 1: the tutor answers the scenario (once per scenario) --------------
        if scenario.scenario_id not in responses:
            started = time.perf_counter()
            try:
                text = respgen.generate_one(scenario)
            except ScenarioUnservable as exc:
                drop_scenario(scenario.scenario_id, str(exc))
                continue
            except TutorUnavailable as exc:
                # The endpoint itself looks unhealthy: stop, but keep what we have.
                stopped_because = f"tutor-unavailable: {exc}"
                log.error("stopping early, tutor unavailable: %s", exc)
                break
            if not text.strip():
                # Caching this would grade every criterion of the scenario against nothing.
                drop_scenario(scenario.scenario_id, "tutor returned an empty response")
                continue
            elapsed = time.perf_counter() - started
            resp_seconds.append(elapsed)
            responses[scenario.scenario_id] = text
            sink.append(
                "responses.jsonl",
                {
                    "scenario_id": scenario.scenario_id,
                    "response": text,
                    "response_chars": len(text),
                    "seconds": round(elapsed, 3),
                },
            )
            sink.sync()

        # --- step 2: the judge grades this criterion --------------------------------
        started = time.perf_counter()
        try:
            verdict = judge.evaluate(scenario, criterion, responses[scenario.scenario_id])
        except Exception as exc:  # noqa: BLE001 - one bad reply must not end a paid run
            verdict = JudgeVerdict(
                criterion_id=criterion.criterion_id,
                passed=False,
                unscorable_reason=f"judge raised {type(exc).__name__}: {exc}",
            )
            log.warning("judge raised on %s: %s", criterion.criterion_id, exc)
        judge_elapsed = time.perf_counter() - started
        judge_seconds.append(judge_elapsed)
        sink.append(
            "judgments.jsonl",
            {
                "criterion_id": criterion.criterion_id,
                "scenario_id": scenario.scenario_id,
                # null, not false, when there is no verdict: an abstention is missing
                # data and must not read as a failed criterion downstream.
                "passed": None if verdict.unscorable_reason else bool(verdict.passed),
                "unscorable_reason": verdict.unscorable_reason,
                "rationale": verdict.rationale,
                # The evidence-gated contracts return the quote the verdict rests on;
                # without it a disputed judgment cannot be checked without the raw body.
                "evidence": verdict.evidence,
                "raw_output": verdict.raw_output,
                "seconds": round(judge_elapsed, 3),
            },
        )

        if verdict.unscorable_reason:
            abstain_streak += 1
            last_abstain_reason = verdict.unscorable_reason
            excluded.add(criterion.criterion_id)
            selectable = _filtered(irt_bank, excluded)
            record = {
                "criterion_id": criterion.criterion_id,
                "scenario_id": scenario.scenario_id,
                "reason": verdict.unscorable_reason,
            }
            abstentions.append(record)
            sink.append("abstentions.jsonl", record)
            sink.sync()
            continue

        # --- ability update + stopping ----------------------------------------------
        abstain_streak = 0  # only a scored criterion clears it
        state.administered.append(
            CriterionResponse(
                criterion_id=criterion.criterion_id, correct=verdict.passed, verdict=verdict
            )
        )
        state.ability = style.estimate_ability(irt_bank, state.administered, previous=state.ability)
        sink.append(
            "trajectory.jsonl",
            {
                "step": state.step,
                "criterion_id": criterion.criterion_id,
                "scenario_id": scenario.scenario_id,
                "theta": state.ability.theta,
                "standard_error": state.ability.standard_error,
                "boundary_limited": bool(state.ability.metadata.get("boundary_limited", False)),
            },
        )
        sink.sync()

        if style.stopping_rule(state):
            stopped_because = "se-threshold"
            break

    # Hand the style what only the session knows, so the report is self-contained under
    # either engine. `ungradable` is the honesty signal: a theta from 8 scored criteria
    # reads very differently once you know 30 more could not be scored at all.
    dropped_with_scenarios = sum(
        len(criteria_by_scenario[entry["scenario_id"]]) for entry in skipped_scenarios
    )
    unscorable = len(abstentions) + dropped_with_scenarios
    attempted = state.step + unscorable
    rate = (unscorable / attempted) if attempted else 0.0
    state.metadata["stop_reason"] = stopped_because
    state.metadata["bank_size"] = len(irt_bank.params)
    # The report says whether theta is on the calibrated scale, which is a fact about the
    # judge that actually graded rather than about this style. Read off the judge so a
    # config change cannot leave the report describing a contract that did not run.
    provenance = getattr(judge, "provenance", None)
    if isinstance(provenance, dict):
        state.metadata["judge_contract"] = {
            "adapter": provenance.get("adapter"),
            "prompt_version": provenance.get("prompt_version"),
            "evidence_gated": provenance.get("evidence_gated"),
            "unparseable_policy": provenance.get("unparseable_policy"),
        }
    state.metadata["attempts"] = {
        "made": attempts,
        "budget": attempt_budget,
        "scored": state.step,
        "budget_exhausted": attempts >= attempt_budget,
    }
    state.metadata["ungradable"] = {
        "count": unscorable,
        "rate": round(rate, 4),
        "criteria_abstained": len(abstentions),
        "criteria_dropped_with_scenario": dropped_with_scenarios,
        "scenarios_skipped": len(skipped_scenarios),
        "item_ids": [entry["criterion_id"] for entry in abstentions][:50],
        "reasons": sorted({entry["reason"][:80] for entry in abstentions + skipped_scenarios}),
        "alert": rate > 0.20,
    }

    report = style.report(state)
    return {
        "report": report,
        "stopped_because": stopped_because,
        "scenarios_used": len(responses),
        "scenarios_skipped": len(skipped_scenarios),
        "criteria_administered": state.step,
        "criteria_abstained": len(abstentions),
        "attempts_made": attempts,
        "attempt_budget": attempt_budget,
        # Full lists live in skipped.jsonl / abstentions.jsonl; these are a preview.
        "skipped_sample": skipped_scenarios[:20],
        "abstention_sample": abstentions[:20],
        "timing": {
            "respgen_calls": len(resp_seconds),
            "respgen_total_s": round(sum(resp_seconds), 2),
            "judge_calls": len(judge_seconds),
            "judge_total_s": round(sum(judge_seconds), 2),
        },
    }
