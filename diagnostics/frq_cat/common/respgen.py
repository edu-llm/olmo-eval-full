"""Response generation from the checkpoint under test (the tutor endpoint).

FRQ evaluation serves the checkpoint under test as a vLLM OpenAI-compatible
endpoint and generates one greedy/deterministic response per scenario, matching
the calibration convention (``temperature=0``, fixed seed). ``httpx`` is imported
lazily so the package imports without it installed; ``--dry-run`` never calls it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..base import Scenario

log = logging.getLogger("frq_cat.respgen")


@dataclass
class RespGenConfig:
    """Configuration for served response generation."""

    endpoint: str
    served_model: str
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    timeout: float = 600.0


def _chat_url(endpoint: str) -> str:
    """Return the chat-completions URL for a served ``/v1`` base endpoint."""
    root = endpoint.rstrip("/")
    return f"{root}/chat/completions" if root.endswith("/v1") else f"{root}/v1/chat/completions"


def _messages_for(scenario: Scenario) -> list[dict[str, str]]:
    """Return the chat messages for a scenario (fall back to a user prompt)."""
    if scenario.messages:
        return [dict(message) for message in scenario.messages]
    return [{"role": "user", "content": scenario.prompt}]


class ServedRespGen:
    """A :class:`~diagnostics.frq_cat.base.RespGenModel` backed by a served endpoint."""

    def __init__(self, config: RespGenConfig) -> None:
        self.config = config
        self._url = _chat_url(config.endpoint)

    def generate(self, scenarios: Sequence[Scenario]) -> dict[str, str]:
        """Generate one deterministic response per scenario via the tutor endpoint."""
        import httpx

        responses: dict[str, str] = {}
        with httpx.Client(
            headers={"Content-Type": "application/json"}, timeout=self.config.timeout
        ) as client:
            for scenario in scenarios:
                responses[scenario.scenario_id] = self._generate_one(client, scenario)
        return responses

    def _generate_one(self, client: object, scenario: Scenario) -> str:
        """Generate a single response, backing off max_tokens on context overflow."""
        import httpx

        assert isinstance(client, httpx.Client)
        messages = _messages_for(scenario)
        last_exc: Exception | None = None
        started = time.perf_counter()
        for attempt, max_tokens in enumerate((self.config.max_tokens, 512, 128, 32)):
            payload = {
                "model": self.config.served_model,
                "messages": messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": max_tokens,
                "seed": self.config.seed,
            }
            try:
                resp = client.post(self._url, json=payload)
                if resp.status_code == 400 and max_tokens != 32:
                    last_exc = httpx.HTTPStatusError(resp.text, request=resp.request, response=resp)
                    continue
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"] or ""
                log.debug(
                    "respgen %s ok in %.2fs (max_tokens=%d)",
                    scenario.scenario_id,
                    time.perf_counter() - started,
                    max_tokens,
                )
                return text
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < 3:
                    time.sleep(1.0 + attempt)
                    continue
        raise RuntimeError(f"generation failed for {scenario.scenario_id}: {last_exc}")
