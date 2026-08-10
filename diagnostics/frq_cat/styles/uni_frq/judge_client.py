"""Judge transport for the uni_frq pipeline (frozen self-hosted or hosted frontier).

This module is transport only. Which prompt is sent and which parser reads the reply is
the ``adapter`` named in the judge YAML, resolved through :mod:`.adapters`, so a contract
can never be half-applied: sending the JSON prompt and reading it with the text parser
abstains on every criterion. Only the pieces the shared ``ServedJudge`` gets wrong for a
hosted model are replaced:

- it sends no ``Authorization`` header, so any frontier API answers 401;
- its parser searches for "pass" anywhere, so "does not pass" scores as a PASS;
- it has no retry/backoff, so a single 429 or 502 kills a whole CAT session;
- it has no abstain path, so an empty or unparseable reply silently becomes a failed
  criterion and is fed to the ability estimator as real evidence.

Nothing here may raise on a bad reply: an unusable response must become an abstention,
because the caller treats an exception as fatal to the run.

``httpx`` is imported lazily so this module can be imported without it installed.
"""

from __future__ import annotations

import copy
import logging
import os
import random
import re
import time
from typing import Any, cast

from ...base import Criterion, JudgeVerdict, Scenario
from ...common.judge import JudgeSpec
from .adapters import get_adapter, parse_pass_fail_text

log = logging.getLogger("uni_frq.judge")

#: What to do with a reply no rung of the parser could read.
#:
#: ``abstain`` drops the criterion: inside an adaptive session a parse failure is a fact
#: about the judge, not evidence about the tutor, and scoring it as a fail would feed the
#: estimator an answer nobody gave. ``fail_closed`` scores it as a fail, which is what the
#: team's grader does and what a rectangular calibration matrix needs. Selectable so a run
#: can reproduce either, and recorded in the manifest so the reader knows which happened.
_ABSTAIN = "abstain"
_FAIL_CLOSED = "fail_closed"
_UNPARSEABLE_POLICIES = (_ABSTAIN, _FAIL_CLOSED)

#: Redacted before any body text reaches a log line or a persisted artifact.
_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|Bearer\s+\S+)", re.IGNORECASE)

#: Statuses worth retrying: rate limits and transient server/proxy faults.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_LOCAL_HOSTS = ("http://127.0.0.1", "http://localhost", "http://[::1]", "http://0.0.0.0")


def redact(text: str) -> str:
    """Strip anything that looks like an API key out of text we log or persist."""
    return _SECRET_RE.sub("[redacted]", text or "")


def parse_verdict(raw: object) -> tuple[bool | None, str]:
    """Parse a ``served-pass-fail`` completion into ``(passed, rationale_or_reason)``.

    Returns ``(None, reason)`` when the reply cannot be scored so the caller can record an
    abstention. Kept as the text contract's entry point; the JSON contracts are read by
    :func:`.adapters.parse_json_verdict`, which ``ResilientJudge`` selects by adapter.
    """
    if not isinstance(raw, str):
        return None, f"non-text judge output: {type(raw).__name__}"
    parsed = parse_pass_fail_text(raw)
    if parsed.passed is None:
        return None, parsed.reason
    return parsed.passed, parsed.rationale


def extract_content(payload: object) -> str | None:
    """Pull the assistant text out of an OpenAI-shaped body, or ``None`` if unusable.

    Providers differ: ``message`` may be absent or null, and ``content`` may be a list of
    parts. Every one of those shapes used to raise ``AttributeError`` and abort the run.
    """
    if not isinstance(payload, dict):
        return None
    choices = cast("dict[str, Any]", payload).get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content-parts style
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(parts) if parts else None
    return None


def _chat_url(endpoint: str) -> str:
    """Return the chat-completions URL for a served ``/v1`` base endpoint."""
    root = endpoint.rstrip("/")
    return f"{root}/chat/completions" if root.endswith("/v1") else f"{root}/v1/chat/completions"


class ResilientJudge:
    """A :class:`~diagnostics.frq_cat.base.Judge` usable against a frontier API.

    ``api_key_env`` names an environment variable (never a literal secret). When set, its
    value is sent as a bearer token; when the variable is missing the constructor fails
    immediately rather than after a GPU session has already been spent.
    """

    def __init__(
        self,
        spec: JudgeSpec,
        *,
        endpoint: str,
        api_key_env: str = "",
        timeout: float = 120.0,
        max_attempts: int = 4,
    ) -> None:
        self.spec = spec
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.adapter = get_adapter(spec.adapter)
        if spec.max_tokens < self.adapter.min_max_tokens:
            # Refused here rather than absorbed: under the floor the reply is cut before
            # the verdict field, every criterion comes back unscorable, and the run burns
            # its GPU window to report an untouched prior.
            raise RuntimeError(
                f"judge adapter {self.adapter.name!r} needs max_tokens >= "
                f"{self.adapter.min_max_tokens} (its verdict is the last field of a JSON "
                f"object, so a shorter budget truncates it away); the judge YAML asks for "
                f"{spec.max_tokens}"
            )
        policy = str(spec.metadata.get("unparseable_policy", _ABSTAIN)).strip() or _ABSTAIN
        if policy not in _UNPARSEABLE_POLICIES:
            raise RuntimeError(
                f"judge metadata.unparseable_policy={policy!r} is not one of "
                f"{', '.join(_UNPARSEABLE_POLICIES)}"
            )
        self.unparseable_policy = policy
        self._url = _chat_url(endpoint)
        self._headers = {"Content-Type": "application/json"}
        if api_key_env:
            key = os.environ.get(api_key_env, "").strip()
            if not key:
                raise RuntimeError(
                    f"judge api_key_env={api_key_env!r} is set but that environment "
                    f"variable is empty; export it before starting the run"
                )
            self._headers["Authorization"] = f"Bearer {key}"
            if endpoint.startswith("http://") and not endpoint.startswith(_LOCAL_HOSTS):
                log.warning(
                    "sending the judge API key over cleartext http to %s; use https", endpoint
                )

    @property
    def provenance(self) -> dict[str, Any]:
        """Full judge identity for the run manifest (no secrets, only the env var name)."""
        return {
            "name": self.spec.name,
            "model_id": self.spec.model_id,
            "revision": self.spec.revision,
            "adapter": self.adapter.name,
            "prompt_version": self.adapter.prompt_version,
            "evidence_gated": self.adapter.evidence_gated,
            "unparseable_policy": self.unparseable_policy,
            "temperature": self.spec.temperature,
            "top_p": self.spec.top_p,
            "max_tokens": self.spec.max_tokens,
            "endpoint": self.endpoint,
            "authenticated": "Authorization" in self._headers,
            "api_key_env": self.api_key_env or None,
            "metadata": copy.deepcopy(dict(self.spec.metadata)),
        }

    def _abstain(self, criterion: Criterion, reason: str, raw: str = "") -> JudgeVerdict:
        """Record missing data. Never raises, so one bad reply cannot end the session."""
        log.warning("judge abstained on %s: %s", criterion.criterion_id, reason)
        return JudgeVerdict(
            criterion_id=criterion.criterion_id,
            passed=False,
            unscorable_reason=reason,
            raw_output=raw[:2000],
        )

    def evaluate(
        self, scenario: Scenario, criterion: Criterion, response_text: str
    ) -> JudgeVerdict:
        """Grade one criterion, retrying transient faults and abstaining when unscorable."""
        import httpx

        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": self.adapter.build_messages(scenario, criterion, response_text),
            "temperature": self.spec.temperature,
            "top_p": self.spec.top_p,
            "max_tokens": self.spec.max_tokens,
            "seed": 42,
        }
        if self.adapter.response_format is not None:
            payload["response_format"] = dict(self.adapter.response_format)
        last_reason = "no attempt made"
        with httpx.Client(headers=self._headers, timeout=self.timeout) as client:
            for attempt in range(self.max_attempts):
                try:
                    resp = client.post(self._url, json=payload)
                except httpx.TransportError as exc:
                    last_reason = f"{type(exc).__name__}: {exc}"
                else:
                    if resp.status_code == 200:
                        try:
                            body = resp.json()
                        except ValueError:
                            return self._abstain(
                                criterion, "judge returned a non-JSON 200 body", redact(resp.text)
                            )
                        content = extract_content(body)
                        if content is None:
                            return self._abstain(
                                criterion,
                                "judge returned a 200 with no usable message content",
                                redact(resp.text),
                            )
                        parsed = self.adapter.parse(content)
                        if parsed.passed is None:
                            if self.unparseable_policy == _FAIL_CLOSED:
                                # Scored, not dropped: the reason still travels, but in
                                # metadata, because `unscorable_reason` is what marks a
                                # verdict as missing data downstream.
                                log.warning(
                                    "judge output on %s was unparseable and fail-closed: %s",
                                    criterion.criterion_id,
                                    parsed.reason,
                                )
                                return JudgeVerdict(
                                    criterion_id=criterion.criterion_id,
                                    passed=False,
                                    rationale=parsed.reason,
                                    raw_output=content[:2000],
                                    metadata={"fail_closed_reason": parsed.reason},
                                )
                            return self._abstain(criterion, parsed.reason, content)
                        return JudgeVerdict(
                            criterion_id=criterion.criterion_id,
                            passed=parsed.passed,
                            rationale=parsed.rationale,
                            evidence=parsed.evidence,
                            raw_output=content[:2000],
                        )
                    last_reason = f"HTTP {resp.status_code}: {redact(resp.text)[:300]}"
                    if resp.status_code not in _RETRYABLE_STATUS:
                        break  # 400/401/403/404 will not fix themselves
                if attempt < self.max_attempts - 1:
                    time.sleep(min(2.0**attempt, 8.0) + random.uniform(0.0, 0.5))

        return self._abstain(criterion, f"judge call failed: {last_reason}")
