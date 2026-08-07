"""No-network tests for the frozen-Qwen runtime preflight."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners import mode_orchestrator
from olmo_eval.runners.mode_orchestrator import (
    FROZEN_QWEN_VLLM_VERSION,
    validate_qwen_vllm_runtime,
)


def _local_judge() -> ProviderConfig:
    return ProviderConfig(kind="vllm_server", model="Qwen/Qwen3.5-9B")


def _external_judge() -> ProviderConfig:
    return ProviderConfig(
        kind="vllm_server",
        model="Qwen/Qwen3.5-9B",
        base_url="http://judge.invalid:8000/v1",
    )


def _mock_local_probe(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setenv("VLLM_PYTHON", sys.executable)

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": 30,
        }
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=f"startup noise\n{json.dumps(payload)}\n",
            stderr="",
        )

    monkeypatch.setattr(mode_orchestrator.subprocess, "run", fake_run)
    return calls


def test_local_qwen_runtime_accepts_exact_version_and_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_local_probe(
        monkeypatch,
        {
            "version": FROZEN_QWEN_VLLM_VERSION,
            "chat_logprob_token_ids": True,
            "completion_logprob_token_ids": True,
        },
    )

    result = validate_qwen_vllm_runtime(_local_judge())

    assert result == {
        "deployment": "managed_subprocess",
        "version": FROZEN_QWEN_VLLM_VERSION,
        "python": str(Path(sys.executable).resolve()),
        "chat_logprob_token_ids": True,
        "completion_logprob_token_ids": True,
    }
    assert len(calls) == 1
    assert calls[0][0:2] == [sys.executable, "-c"]
    assert "ChatCompletionRequest.model_fields" in calls[0][2]
    assert "CompletionRequest.model_fields" in calls[0][2]


def test_local_qwen_runtime_rejects_wrong_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_local_probe(
        monkeypatch,
        {
            "version": "0.25.0",
            "chat_logprob_token_ids": True,
            "completion_logprob_token_ids": True,
        },
    )

    with pytest.raises(RuntimeError, match=r"requires vLLM 0\.26\.0.*0\.25\.0"):
        validate_qwen_vllm_runtime(_local_judge())


@pytest.mark.parametrize(
    "missing_capability",
    ["chat_logprob_token_ids", "completion_logprob_token_ids"],
)
def test_local_qwen_runtime_rejects_missing_explicit_token_logprob_capability(
    monkeypatch: pytest.MonkeyPatch,
    missing_capability: str,
) -> None:
    payload = {
        "version": FROZEN_QWEN_VLLM_VERSION,
        "chat_logprob_token_ids": True,
        "completion_logprob_token_ids": True,
    }
    payload[missing_capability] = False
    _mock_local_probe(monkeypatch, payload)

    with pytest.raises(RuntimeError, match="lacks explicit logprob_token_ids support"):
        validate_qwen_vllm_runtime(_local_judge())


class _VersionResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


def test_external_qwen_runtime_accepts_exact_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float]] = []

    def fake_get(url: str, *, timeout: float) -> _VersionResponse:
        calls.append((url, timeout))
        return _VersionResponse({"version": FROZEN_QWEN_VLLM_VERSION})

    monkeypatch.setattr(mode_orchestrator.httpx, "get", fake_get)

    result = validate_qwen_vllm_runtime(_external_judge())

    assert calls == [("http://judge.invalid:8000/version", 10.0)]
    assert result == {
        "deployment": "external_endpoint",
        "version": FROZEN_QWEN_VLLM_VERSION,
        "version_url": "http://judge.invalid:8000/version",
        "explicit_token_logprobs": "required_by_frozen_request_contract",
    }


@pytest.mark.parametrize(
    ("get_result", "message"),
    [
        (
            lambda: _VersionResponse({"version": "0.25.0"}),
            r"requires vLLM 0\.26\.0.*0\.25\.0",
        ),
        (
            lambda: httpx.ConnectError(
                "offline", request=httpx.Request("GET", "http://judge.invalid/version")
            ),
            "could not verify the external frozen-Qwen vLLM runtime",
        ),
    ],
)
def test_external_qwen_runtime_rejects_wrong_or_unreachable_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    get_result: Callable[[], _VersionResponse | httpx.ConnectError],
    message: str,
) -> None:
    def fake_get(url: str, *, timeout: float) -> _VersionResponse:
        del url, timeout
        result = get_result()
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(mode_orchestrator.httpx, "get", fake_get)

    with pytest.raises(RuntimeError, match=message):
        validate_qwen_vllm_runtime(_external_judge())


def test_gpu_detection_uses_process_visible_device_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7, 3,5")

    assert mode_orchestrator._detect_gpu_ids([_local_judge()]) == (0, 1, 2)


def test_gpu_detection_skips_cuda_probe_when_no_local_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,3")

    assert mode_orchestrator._detect_gpu_ids([_external_judge()]) == ()
