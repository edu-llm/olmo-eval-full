"""Judge transport for the uni_frq pipeline (frozen self-hosted or hosted frontier).

The grading *contract* stays single-sourced: this module imports the shared
``build_messages`` prompt and ``JudgeSpec`` from ``common/judge.py`` rather than forking
them, so the criterion prompt cannot drift from the calibrated one. Only the pieces the
shared ``ServedJudge`` gets wrong for a hosted model are replaced:

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
from typing import Any

from ...base import Criterion, JudgeVerdict, Scenario
from ...common.judge import JudgeSpec, build_messages

log = logging.getLogger("uni_frq.judge")

#: A scorable reply must *begin* with the verdict, per the shared prompt's instruction
#: ("Answer with PASS or FAIL on the first line"). Anything else is an abstention.
_VERDICT_RE = re.compile(r"^\W*(PASS|FAIL)\b", re.IGNORECASE)

#: The model restating its options ("pass/fail unclear", "PASS or FAIL") has not decided.
#: Anchored at the start so a decided verdict that merely goes on to use the other word
#: ("PASS - it does pass and fail to cite") is still scored.
_ENUMERATION_RE = re.compile(
    r"^\W*(?:pass|fail)\b\s*(?:/|\||,|\bor\b|\band\b)\s*\b(?:pass|fail)\b", re.IGNORECASE
)

#: Redacted before any body text reaches a log line or a persisted artifact.
_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{6,}|Bearer\s+\S+)", re.IGNORECASE)

#: Statuses worth retrying: rate limits and transient server/proxy faults.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_LOCAL_HOSTS = ("http://127.0.0.1", "http://localhost", "http://[::1]", "http://0.0.0.0")


def redact(text: str) -> str:
    """Strip anything that looks like an API key out of text we log or persist."""
    return _SECRET_RE.sub("[redacted]", text or "")


def parse_verdict(raw: object) -> tuple[bool | None, str]:
    """Parse a judge completion into ``(passed, rationale_or_reason)``.

    Returns ``(None, reason)`` when the reply cannot be scored so the caller can record an
    abstention. Scoring an unparseable reply as FAIL would feed the ability estimator
    evidence the judge never actually gave.
    """
    if not isinstance(raw, str):
        return None, f"non-text judge output: {type(raw).__name__}"
    text = raw.strip()
    if not text:
        return None, "empty judge output"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_line = lines[0]
    if _ENUMERATION_RE.match(first_line):
        return None, f"undecided verdict line: {first_line[:120]!r}"
    match = _VERDICT_RE.match(first_line)
    if match is None:
        return None, f"unparseable verdict: {first_line[:120]!r}"
    passed = match.group(1).upper() == "PASS"
    return passed, lines[1] if len(lines) > 1 else ""


def extract_content(payload: object) -> str | None:
    """Pull the assistant text out of an OpenAI-shaped body, or ``None`` if unusable.

    Providers differ: ``message`` may be absent or null, and ``content`` may be a list of
    parts. Every one of those shapes used to raise ``AttributeError`` and abort the run.
    """
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
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
            "adapter": self.spec.adapter,
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

        payload = {
            "model": self.spec.model_id,
            "messages": build_messages(scenario, criterion, response_text),
            "temperature": self.spec.temperature,
            "top_p": self.spec.top_p,
            "max_tokens": self.spec.max_tokens,
            "seed": 42,
        }
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
                        passed, note = parse_verdict(content)
                        if passed is None:
                            return self._abstain(criterion, note, content)
                        return JudgeVerdict(
                            criterion_id=criterion.criterion_id,
                            passed=passed,
                            rationale=note,
                            raw_output=content[:2000],
                        )
                    last_reason = f"HTTP {resp.status_code}: {redact(resp.text)[:300]}"
                    if resp.status_code not in _RETRYABLE_STATUS:
                        break  # 400/401/403/404 will not fix themselves
                if attempt < self.max_attempts - 1:
                    time.sleep(min(2.0**attempt, 8.0) + random.uniform(0.0, 0.5))

        return self._abstain(criterion, f"judge call failed: {last_reason}")
