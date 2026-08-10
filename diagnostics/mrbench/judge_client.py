"""TrueFoundry AI Gateway judge client (OpenAI-compatible).

All inference is routed through the TrueFoundry AI Gateway, which exposes an
OpenAI-compatible Chat Completions API, so we use the ``openai`` SDK pointed at
the gateway base URL rather than any provider-native SDK.

Three knobs, all read from the environment (never hardcoded):
  * base URL  : ``OPENAI_BASE_URL`` | ``TFY_GATEWAY_BASE_URL`` | ``TRUEFOUNDRY_BASE_URL``
                (default ``https://gateway.truefoundry.ai``)
  * API key   : ``OPENAI_API_KEY`` | ``TFY_API_KEY`` | ``TRUEFOUNDRY_API_KEY``
  * model id  : ``MRBENCH_JUDGE_MODEL`` (TrueFoundry ``provider_account/model_name``;
                the user copies the exact Sonnet 4.6 id from their Playground
                "Code Snippet" — there is no usable hardcoded default).

Nothing here performs a network call at import time. ``openai`` is imported
lazily inside :func:`make_client` so dry-run / metrics / tests need no live SDK.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

# Placeholder model id. It is intentionally NOT a working model string: the live
# path refuses to run until the user supplies their real TrueFoundry model id.
PLACEHOLDER_MODEL = "REPLACE_WITH_TFY_PROVIDER_ACCOUNT/claude-sonnet-4-6"

DEFAULT_BASE_URL = "https://gateway.truefoundry.ai"

_BASE_URL_ENVS = ("OPENAI_BASE_URL", "TFY_GATEWAY_BASE_URL", "TRUEFOUNDRY_BASE_URL")
_API_KEY_ENVS = ("OPENAI_API_KEY", "TFY_API_KEY", "TRUEFOUNDRY_API_KEY")
_MODEL_ENV = "MRBENCH_JUDGE_MODEL"


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class JudgeConfig:
    """Resolved judge configuration. Faithful to the paper: temperature 0."""

    base_url: str
    api_key: str | None
    model: str
    temperature: float = 0.0
    # Bounds output cost; a one-sentence feedback + "[RESULT] N" fits easily.
    # Operational cap (not specified by the paper); adjust via from_env override.
    max_tokens: int = 256
    timeout: float = 60.0
    max_retries: int = 6
    backoff_base: float = 2.0

    @classmethod
    def from_env(cls) -> JudgeConfig:
        return cls(
            base_url=_first_env(_BASE_URL_ENVS) or DEFAULT_BASE_URL,
            api_key=_first_env(_API_KEY_ENVS),
            model=os.environ.get(_MODEL_ENV, PLACEHOLDER_MODEL),
        )

    @property
    def model_is_placeholder(self) -> bool:
        return self.model == PLACEHOLDER_MODEL

    def missing_for_live(self) -> list[str]:
        """Return human-readable descriptions of what is missing to run live."""
        missing: list[str] = []
        if not self.api_key:
            missing.append(f"a TrueFoundry API key in one of {list(_API_KEY_ENVS)}")
        if self.model_is_placeholder:
            missing.append(
                f"the Sonnet 4.6 TrueFoundry model id in {_MODEL_ENV} "
                "(copy from your Playground 'Code Snippet')"
            )
        return missing


def make_client(config: JudgeConfig):
    """Construct an OpenAI client pointed at the TrueFoundry gateway.

    Imported lazily so importing this module never requires the SDK to be present.
    """
    from openai import OpenAI

    return OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout)


# Exception types worth retrying (resolved lazily to avoid a hard SDK dependency).
def _retryable_exceptions() -> tuple[type[BaseException], ...]:
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

        return (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    except Exception:  # pragma: no cover - SDK not installed
        return ()


def score_messages(client, config: JudgeConfig, messages: list[dict[str, str]]) -> str:
    """Call the gateway once for one (response, dimension) and return raw text.

    Retries transient/rate-limit errors with exponential backoff + jitter.
    """
    retryable = _retryable_exceptions()
    last_exc: BaseException | None = None
    for attempt in range(config.max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            return completion.choices[0].message.content or ""
        except retryable as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == config.max_retries:
                break
            sleep_s = config.backoff_base**attempt + random.uniform(0, 1)
            time.sleep(sleep_s)
    raise RuntimeError(f"Judge call failed after {config.max_retries + 1} attempts") from last_exc
