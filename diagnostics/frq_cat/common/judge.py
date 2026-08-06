"""The shared, frozen LLM-as-a-judge for FRQ grading.

One calibrated judge grades every FRQ style: a served ``generic-binary`` judge is
sent the scenario + the criterion + the checkpoint's response and returns a
pass/fail verdict per criterion (one judge call == one criterion). The judge
identity is defined ONCE here; styles select it via their ``config.yaml`` and never
fork it, so calibration provenance (model id, revision, prompt/normalization
versions) is stable across styles.

``httpx`` and ``yaml`` are imported lazily so the package imports without them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..base import Criterion, JudgeVerdict, Scenario

log = logging.getLogger("frq_cat.judge")

#: Versioned knobs stamped into provenance so a judge change is always traceable.
PROMPT_VERSION = "generic-binary/v1"
NORMALIZATION_VERSION = "v1"
EVIDENCE_POLICY_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    """The frozen identity of the calibrated judge."""

    name: str
    model_id: str
    revision: str = ""
    adapter: str = "generic-binary"
    enable_thinking: bool = False
    language_model_only: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    metadata: dict[str, Any] = field(default_factory=dict)


def spec_from_config(config: dict[str, Any]) -> JudgeSpec:
    """Build a :class:`JudgeSpec` from a parsed ``judge_frozen.yaml`` mapping."""
    return JudgeSpec(
        name=config.get("name", "frozen-judge"),
        model_id=config["model_id"],
        revision=config.get("revision", ""),
        adapter=config.get("adapter", "generic-binary"),
        enable_thinking=bool(config.get("enable_thinking", False)),
        language_model_only=bool(config.get("language_model_only", True)),
        temperature=float(config.get("temperature", 0.0)),
        top_p=float(config.get("top_p", 1.0)),
        max_tokens=int(config.get("max_tokens", 1024)),
        metadata=dict(config.get("metadata", {})),
    )


def load_spec(config_path: str | Path) -> JudgeSpec:
    """Load and parse a ``judge_frozen.yaml`` config into a :class:`JudgeSpec`."""
    import yaml

    text = Path(config_path).read_text(encoding="utf-8")
    return spec_from_config(yaml.safe_load(text))


def build_messages(
    scenario: Scenario, criterion: Criterion, response_text: str
) -> list[dict[str, str]]:
    """Build the ``generic-binary`` judge prompt for one criterion."""
    system = (
        "You are a strict grader. Decide whether the tutor RESPONSE satisfies the "
        "single CRITERION. Answer with PASS or FAIL on the first line, then a one-line "
        "rationale."
    )
    user = (
        f"QUESTION:\n{scenario.prompt}\n\n"
        f"CRITERION:\n{criterion.text}\n\n"
        f"RESPONSE:\n{response_text}\n\n"
        "Verdict (PASS/FAIL):"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_judgment(raw_output: str) -> tuple[bool, str]:
    """Parse a raw judge completion into ``(passed, rationale)``."""
    text = (raw_output or "").strip()
    passed = bool(re.search(r"\bpass\b", text, flags=re.IGNORECASE)) and not re.match(
        r"\s*fail\b", text, flags=re.IGNORECASE
    )
    rationale = ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        rationale = lines[1]
    return passed, rationale


class ServedJudge:
    """A :class:`~diagnostics.frq_cat.base.Judge` backed by a served endpoint."""

    def __init__(self, spec: JudgeSpec, *, endpoint: str, timeout: float = 600.0) -> None:
        self.spec = spec
        self.timeout = timeout
        root = endpoint.rstrip("/")
        self._url = (
            f"{root}/chat/completions" if root.endswith("/v1") else f"{root}/v1/chat/completions"
        )

    def evaluate(
        self, scenario: Scenario, criterion: Criterion, response_text: str
    ) -> JudgeVerdict:
        """Grade ``response_text`` for ``criterion`` and return a verdict."""
        import httpx

        messages = build_messages(scenario, criterion, response_text)
        payload = {
            "model": self.spec.model_id,
            "messages": messages,
            "temperature": self.spec.temperature,
            "top_p": self.spec.top_p,
            "max_tokens": self.spec.max_tokens,
            "seed": 42,
        }
        with httpx.Client(
            headers={"Content-Type": "application/json"}, timeout=self.timeout
        ) as client:
            resp = client.post(self._url, json=payload)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"] or ""
        passed, rationale = parse_judgment(raw)
        return JudgeVerdict(
            criterion_id=criterion.criterion_id,
            passed=passed,
            rationale=rationale,
            raw_output=raw,
        )
