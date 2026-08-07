"""Focused regression tests for inference-provider lifecycle safeguards."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmo_eval.inference.manager import InferenceManager, ServerInfo
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.providers.litellm import LiteLLMProvider
from olmo_eval.inference.providers.vllm_server import RemoteTokenizer, VLLMServerProvider
from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess


@pytest.mark.parametrize("tensor_parallel_size", [0, -1, True, 1.5, "2"])
def test_inference_manager_rejects_invalid_tensor_parallel_size_before_creation(
    tensor_parallel_size: object,
) -> None:
    manager = InferenceManager(
        configs={
            "judge": ProviderConfig(
                kind="vllm_server",
                model="org/model",
                kwargs={"tensor_parallel_size": tensor_parallel_size},
            )
        },
        available_gpu_ids=[0, 1],
    )

    with (
        patch("olmo_eval.inference.manager._create_server") as create_server,
        pytest.raises(ValueError, match="tensor_parallel_size must be a positive integer"),
    ):
        manager.start()

    create_server.assert_not_called()


def test_inference_manager_rejects_insufficient_gpus_before_creation() -> None:
    manager = InferenceManager(
        configs={
            "judge": ProviderConfig(
                kind="vllm_server",
                model="org/model",
                num_instances=2,
                kwargs={"tensor_parallel_size": 2},
            )
        },
        available_gpu_ids=[0, 1, 2],
    )

    with (
        patch("olmo_eval.inference.manager._create_server") as create_server,
        pytest.raises(RuntimeError, match=r"Need 4, available: 3"),
    ):
        manager.start()

    create_server.assert_not_called()


def test_inference_manager_stops_every_created_server_after_parallel_start_failure() -> None:
    successful_server = MagicMock()
    successful_server.start.return_value = "http://127.0.0.1:8000/v1"
    failing_server = MagicMock()
    failing_server.start.side_effect = RuntimeError("startup failed")
    manager = InferenceManager(
        configs={
            "judge": ProviderConfig(
                kind="vllm_server",
                model="org/model",
                num_instances=2,
            )
        },
        available_gpu_ids=[0, 1],
    )

    with (
        patch(
            "olmo_eval.inference.manager._create_server",
            side_effect=[successful_server, failing_server],
        ) as create_server,
        pytest.raises(RuntimeError, match="startup failed"),
    ):
        manager.start()

    assert create_server.call_count == 2
    successful_server.stop.assert_called_once_with()
    failing_server.stop.assert_called_once_with()
    assert manager.get_resolved_configs() == {}
    assert manager._started is False


def test_inference_manager_shutdown_reports_failures_and_retains_retry_ownership() -> None:
    successful_server = MagicMock()
    failing_server = MagicMock()
    failing_server.stop.side_effect = RuntimeError("stop failed")
    config = ProviderConfig(kind="vllm_server", model="org/model")
    manager = InferenceManager()
    manager._started = True
    manager._servers = {
        "judge": ServerInfo(
            name="judge",
            resolved_configs=[config, config],
            servers=[successful_server, failing_server],
        )
    }

    with pytest.raises(ExceptionGroup, match="failed to stop one or more"):
        manager.shutdown()

    successful_server.stop.assert_called_once_with()
    failing_server.stop.assert_called_once_with()
    assert manager._started is True
    assert "judge" in manager._servers

    failing_server.stop.side_effect = None
    manager.shutdown()

    assert successful_server.stop.call_count == 2
    assert failing_server.stop.call_count == 2
    assert manager._started is False
    assert manager._servers == {}


def test_vllm_environment_unset_only_changes_child_environment() -> None:
    process = MagicMock()
    process.pid = 12345
    process.poll.return_value = None

    with (
        patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "1"}),
        patch(
            "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
            return_value=23456,
        ),
        patch("subprocess.Popen", return_value=process) as popen,
        patch(
            "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
            return_value=(True, None, None),
        ),
        patch("atexit.register"),
    ):
        server = VLLMServerProcess(
            model_name="org/model",
            port=8000,
            environment_unset=["VLLM_BATCH_INVARIANT"],
        )

        server.start()

        child_environment = popen.call_args.kwargs["env"]
        assert "VLLM_BATCH_INVARIANT" not in child_environment
        assert os.environ["VLLM_BATCH_INVARIANT"] == "1"


def test_vllm_server_can_select_a_per_server_interpreter() -> None:
    process = MagicMock()
    process.pid = 12345
    process.poll.return_value = None

    with (
        patch.dict(os.environ, {"VLLM_PYTHON": "/global/python"}),
        patch(
            "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
            return_value=23456,
        ),
        patch("subprocess.Popen", return_value=process) as popen,
        patch(
            "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
            return_value=(True, None, None),
        ),
        patch("atexit.register"),
    ):
        VLLMServerProcess(
            model_name="org/model",
            port=8000,
            python_executable="/isolated/qwen-python",
        ).start()

    assert popen.call_args.args[0][0] == "/isolated/qwen-python"


def test_vllm_server_retains_resolved_global_interpreter_for_capability_checks() -> None:
    with (
        patch.dict(os.environ, {"VLLM_PYTHON": "/global/vllm/bin/python"}),
        patch(
            "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
            return_value=23456,
        ),
    ):
        server = VLLMServerProcess(model_name="org/model", port=8000)

    assert server.python_executable == "/global/vllm/bin/python"


@pytest.mark.anyio
async def test_litellm_aclose_closes_and_discards_cached_client() -> None:
    client = MagicMock()
    client.close = AsyncMock()
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    provider._client = client

    await provider.aclose()
    await provider.aclose()

    client.close.assert_awaited_once_with()
    assert provider._client is None


@pytest.mark.anyio
async def test_vllm_server_aclose_releases_remote_tokenizer_client() -> None:
    tokenizer_client = MagicMock()
    tokenizer = RemoteTokenizer("http://127.0.0.1:8000/v1", "org/model")
    tokenizer._client = tokenizer_client
    provider = VLLMServerProvider.__new__(VLLMServerProvider)
    provider._tokenizer = tokenizer
    provider._raw_http_client = None
    provider._http_client = None
    provider._client = None

    await provider.aclose()

    tokenizer_client.close.assert_called_once_with()
    assert tokenizer._client is None
    assert provider._tokenizer is None
