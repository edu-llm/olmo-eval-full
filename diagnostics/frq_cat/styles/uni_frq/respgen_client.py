"""Tutor client that separates "this scenario will never fit" from "try again".

The shared ``ServedRespGen`` retries a 400 by shrinking ``max_tokens``. That helps only
when the *completion budget* overflows the window; it does nothing when the prompt alone
is too long, and it applies the same shrink to 429s and timeouts, which need a wait
instead. It also discards the server's response body, so the operator sees a bare
``Client error '400 Bad Request'`` with no reason.

TutorEval prompts embed book passages (p90 ~4.5k tokens, max ~9.7k), so over-length
prompts are expected on a small context window and must be skippable, not fatal. Getting
the classification backwards is equally bad: a sick endpoint misreported as a bad
scenario makes the session skip its way through the whole bank instead of stopping.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

from ...base import Scenario

log = logging.getLogger("uni_frq.respgen")

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Smallest completion budget worth trying. A 400 that survives this proves the prompt
#: itself does not fit, which is what makes a scenario permanently unservable.
_BUDGET_FLOOR = 32


class TutorUnavailable(RuntimeError):
    """Transient failure: the endpoint may succeed later (rate limit, 5xx, timeout)."""


class ScenarioUnservable(RuntimeError):
    """Permanent failure for this scenario (e.g. prompt exceeds the context window)."""


@dataclass
class RespGenConfig:
    """Generation settings; deterministic to match the calibration convention."""

    endpoint: str
    served_model: str
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    timeout: float = 600.0
    max_attempts: int = 4


def _chat_url(endpoint: str) -> str:
    """Return the chat-completions URL for a served ``/v1`` base endpoint."""
    root = endpoint.rstrip("/")
    return f"{root}/chat/completions" if root.endswith("/v1") else f"{root}/v1/chat/completions"


def _messages_for(scenario: Scenario) -> list[dict[str, str]]:
    """Return the chat messages for a scenario (fall back to a bare user prompt)."""
    if scenario.messages:
        return [dict(message) for message in scenario.messages]
    return [{"role": "user", "content": scenario.prompt}]


class ResilientRespGen:
    """Generate one deterministic tutor response per scenario, with typed failures."""

    def __init__(self, config: RespGenConfig) -> None:
        self.config = config
        self._url = _chat_url(config.endpoint)

    def budget_ladder(self) -> list[int]:
        """Completion budgets to try, strictly decreasing, smallest last.

        Deduplicated, and never above the configured ``max_tokens`` (asking for a larger
        completion than requested would be a surprising side effect of a small setting).
        With a single attempt we try the configured budget, because succeeding matters
        more than being able to prove permanence.
        """
        floor = min(_BUDGET_FLOOR, self.config.max_tokens)
        rungs = sorted(
            {
                b
                for b in (self.config.max_tokens, 512, 128, floor)
                if floor <= b <= self.config.max_tokens
            },
            reverse=True,
        )
        if not rungs:
            rungs = [floor]
        if len(rungs) > self.config.max_attempts:
            rungs = (
                [*rungs[: self.config.max_attempts - 1], floor]
                if self.config.max_attempts > 1
                else [rungs[0]]
            )
        return rungs

    def _post(self, client: object, messages: list[dict[str, str]], budget: int) -> tuple[int, str]:
        """POST once. Returns ``(status, content_or_error_body)``; status 0 = try again."""
        import httpx

        assert isinstance(client, httpx.Client)
        payload = {
            "model": self.config.served_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": budget,
            "seed": self.config.seed,
        }
        try:
            resp = client.post(self._url, json=payload)
        except httpx.TransportError as exc:
            return 0, f"{type(exc).__name__}: {exc}"
        if resp.status_code != 200:
            return resp.status_code, resp.text[:400]
        try:
            body = resp.json()
        except ValueError:
            return 0, "non-JSON 200 body from the tutor"
        choices = body.get("choices") if isinstance(body, dict) else None
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            # A 200 we cannot read is a server problem, not a property of the scenario.
            return 0, "200 response with no usable message content"
        return 200, content

    def _post_with_retries(
        self, client: object, messages: list[dict[str, str]], budget: int
    ) -> tuple[int, str]:
        """Retry only genuinely transient faults; never shrink the budget for a 429."""
        reason = "no attempt made"
        for attempt in range(self.config.max_attempts):
            status, payload = self._post(client, messages, budget)
            if status != 0 and status not in _RETRYABLE_STATUS:
                return status, payload
            reason = payload if status == 0 else f"HTTP {status}: {payload}"
            if attempt < self.config.max_attempts - 1:
                time.sleep(min(2.0**attempt, 8.0) + random.uniform(0.0, 0.5))
        return 0, reason

    def generate_one(self, scenario: Scenario) -> str:
        """Return the tutor's response text for ``scenario``.

        Raises :class:`ScenarioUnservable` when the scenario cannot be served at all (the
        caller should drop it and continue) and :class:`TutorUnavailable` when the endpoint
        looks temporarily unhealthy (the caller should stop). A transient fault always
        wins over an earlier 400: if the endpoint went sick partway down the ladder we
        have not proven anything about the prompt.
        """
        import httpx

        messages = _messages_for(scenario)
        overflow_body: str | None = None
        overflow_budget: int | None = None
        transient_reason: str | None = None
        with httpx.Client(
            headers={"Content-Type": "application/json"}, timeout=self.config.timeout
        ) as client:
            for budget in self.budget_ladder():
                status, payload = self._post_with_retries(client, messages, budget)
                if status == 200:
                    return payload
                if status == 400:
                    overflow_body, overflow_budget = payload, budget
                    continue  # maybe only the completion budget; try a smaller one
                if status == 0:
                    transient_reason = payload
                    break  # endpoint unhealthy; do not burn the rest of the ladder
                raise TutorUnavailable(f"{scenario.scenario_id}: HTTP {status}: {payload}")

        if transient_reason is not None:
            raise TutorUnavailable(f"{scenario.scenario_id}: {transient_reason}")
        if overflow_body is not None:
            raise ScenarioUnservable(
                f"{scenario.scenario_id}: HTTP 400 down to max_tokens={overflow_budget} "
                f"(the prompt itself does not fit): {overflow_body}"
            )
        raise TutorUnavailable(f"{scenario.scenario_id}: no attempt made")
