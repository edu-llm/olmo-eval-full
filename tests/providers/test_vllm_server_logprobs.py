"""Unit tests for VLLMServerProvider completion and logprobs behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams


class TestVLLMServerProviderLogprobs:
    """Tests for VLLMServerProvider._logprobs_single_impl."""

    def _make_provider(self, **kwargs):
        """Create a real provider instance configured for an existing server."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        with patch("olmo_eval.inference.providers.vllm_server.BeakerStatusReporter"):
            provider = VLLMServerProvider(
                "test-model", base_url="http://localhost:8000/v1", **kwargs
            )
        provider._max_length = 4096
        return provider

    def _make_fake_transformers(self, tokenizer):
        """Build a stub transformers module for local tokenizer loads."""
        from_pretrained = MagicMock(return_value=tokenizer)
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)
        )
        return from_pretrained, fake_transformers

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, add_special_tokens=False: list(
            range(len(text.split()))
        )
        tokenizer.decode.side_effect = lambda ids: " ".join(f"tok{i}" for i in ids)
        tokenizer.bos_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer

    @pytest.fixture
    def provider(self, mock_tokenizer):
        """Create a VLLMServerProvider with __init__ bypassed."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        with patch.object(VLLMServerProvider, "__init__", lambda self, *a, **kw: None):
            p = VLLMServerProvider.__new__(VLLMServerProvider)
            p.model_name = "test-model"
            p.base_url = "http://localhost:8000/v1"
            p._tokenizer = mock_tokenizer
            p._add_bos_token = None
            p._client = None
            p._http_client = None
            p._raw_http_client = None
            p._server = None
            p._max_length = 4096
            p._prompt_logprobs = 5
            p.chat_template_kwargs = None
            p._completion_use_prompt_token_ids = False
            p._completion_client_side_stop_trim = False
            p._completion_sentencepiece_cleanup = False
            p._get_tokenizer = MagicMock(return_value=mock_tokenizer)
            return p

    def _make_vllm_response(self, prompt_logprobs):
        """Build a JSON response matching vLLM's prompt_logprobs format."""
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"prompt_logprobs": prompt_logprobs}],
        }
        return resp

    def _make_completion_response(self, text="test completion"):
        """Build a completion response matching the OpenAI client shape."""
        choice = MagicMock()
        choice.text = text
        choice.logprobs = None
        choice.finish_reason = None
        choice.token_ids = None

        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None
        return resp

    def _make_chat_response(self, text="test reply"):
        """Build a chat completion response matching the OpenAI client shape."""
        message = MagicMock()
        message.content = text
        message.tool_calls = None

        choice = MagicMock()
        choice.message = message
        choice.logprobs = None
        choice.finish_reason = None
        choice.token_ids = None

        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None
        return resp

    @pytest.mark.anyio
    async def test_generate_completion_sets_add_special_tokens_false(self, provider):
        """Completion payloads should disable server-side special token insertion."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        params = SamplingParams(max_tokens=32, temperature=0.6, top_p=0.6)

        await provider._generate_completion(client, request, params)

        call_kwargs = client.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"]["add_special_tokens"] is False

    @pytest.mark.anyio
    async def test_generation_forwards_seed_and_preserves_finish_reason(self, provider):
        completion_response = self._make_completion_response("done")
        completion_response.choices[0].finish_reason = "length"
        completion_response.choices[0].token_ids = [47]
        completion_client = MagicMock()
        completion_client.completions.create = AsyncMock(return_value=completion_response)

        completion_outputs = await provider._generate_completion(
            completion_client,
            LMRequest(request_type=RequestType.COMPLETION, prompt="Judge"),
            SamplingParams(max_tokens=1, seed=42),
        )

        assert completion_client.completions.create.call_args.kwargs["extra_body"]["seed"] == 42
        assert completion_outputs[0].finish_reason == "length"
        assert completion_outputs[0].token_ids == (47,)

        chat_response = self._make_chat_response("P")
        chat_response.choices[0].finish_reason = "stop"
        chat_response.choices[0].token_ids = [47]
        chat_client = MagicMock()
        chat_client.chat.completions.create = AsyncMock(return_value=chat_response)

        chat_outputs = await provider._generate_chat(
            chat_client,
            LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": "Judge"},),
            ),
            SamplingParams(max_tokens=1, seed=42),
        )

        assert chat_client.chat.completions.create.call_args.kwargs["extra_body"]["seed"] == 42
        assert chat_outputs[0].finish_reason == "stop"
        assert chat_outputs[0].token_ids == (47,)

    @pytest.mark.anyio
    async def test_generate_completion_omits_max_tokens_when_none(self, provider):
        """max_tokens=None means uncapped: omit the field rather than sending null."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        await provider._generate_completion(client, request, SamplingParams(max_tokens=None))

        assert "max_tokens" not in client.completions.create.call_args.kwargs

    @pytest.mark.anyio
    async def test_generate_completion_includes_max_tokens_when_set(self, provider):
        """An explicit max_tokens is still forwarded to the completion payload."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        await provider._generate_completion(client, request, SamplingParams(max_tokens=64))

        assert client.completions.create.call_args.kwargs["max_tokens"] == 64

    @pytest.mark.anyio
    async def test_generate_completion_requests_configured_top_logprobs(self, provider):
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        await provider._generate_completion(
            client,
            request,
            SamplingParams(max_tokens=1, logprobs=20),
        )

        assert client.completions.create.call_args.kwargs["logprobs"] == 20

    @pytest.mark.anyio
    async def test_generate_completion_passes_json_schema(self, provider):
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())
        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")

        await provider._generate_completion(
            client,
            request,
            SamplingParams(
                max_tokens=1,
                structured_output_json_schema={"type": "object"},
            ),
        )

        assert client.completions.create.call_args.kwargs["extra_body"]["structured_outputs"] == {
            "json": {"type": "object"}
        }

    @pytest.mark.anyio
    async def test_generate_chat_omits_max_tokens_when_none(self, provider):
        """max_tokens=None means uncapped: omit the field rather than sending null."""
        provider.chat_template_kwargs = None
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._make_chat_response())

        request = LMRequest(
            request_type=RequestType.CHAT, messages=[{"role": "user", "content": "Hi"}]
        )
        await provider._generate_chat(client, request, SamplingParams(max_tokens=None))

        assert "max_tokens" not in client.chat.completions.create.call_args.kwargs

    @pytest.mark.anyio
    async def test_generate_chat_includes_max_tokens_when_set(self, provider):
        """An explicit max_tokens is still forwarded to the chat payload."""
        provider.chat_template_kwargs = None
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._make_chat_response())

        request = LMRequest(
            request_type=RequestType.CHAT, messages=[{"role": "user", "content": "Hi"}]
        )
        await provider._generate_chat(client, request, SamplingParams(max_tokens=128))

        assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 128

    @pytest.mark.anyio
    async def test_generate_chat_preserves_configured_top_logprobs(self, provider):
        provider.chat_template_kwargs = None
        message = SimpleNamespace(content="P", tool_calls=None)
        token_logprob = SimpleNamespace(
            token="P",
            logprob=-0.2,
            bytes=[80],
            top_logprobs=[
                SimpleNamespace(token="P", logprob=-0.2, bytes=[80]),
                SimpleNamespace(token="F", logprob=-0.7, bytes=[70]),
            ],
        )
        choice = SimpleNamespace(
            message=message,
            logprobs=SimpleNamespace(content=[token_logprob]),
        )
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(choices=[choice], usage=None)
        )

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Judge"},),
        )
        outputs = await provider._generate_chat(
            client,
            request,
            SamplingParams(max_tokens=1, logprobs=20),
        )

        assert client.chat.completions.create.call_args.kwargs["top_logprobs"] == 20
        assert outputs[0].logprobs == [
            {
                "token": "P",
                "logprob": -0.2,
                "bytes": [80],
                "top_logprobs": [
                    {"token": "P", "logprob": -0.2, "bytes": [80]},
                    {"token": "F", "logprob": -0.7, "bytes": [70]},
                ],
            }
        ]

    @pytest.mark.anyio
    async def test_generate_chat_passes_explicit_ids_regex_and_template_kwargs(self, provider):
        provider.chat_template_kwargs = {"enable_thinking": True, "provider_only": "kept"}
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._make_chat_response("P"))
        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Judge"},),
            chat_template_kwargs={"enable_thinking": False},
        )

        await provider._generate_chat(
            client,
            request,
            SamplingParams(
                max_tokens=1,
                logprobs=2,
                logprob_token_ids=(10, 11),
                structured_output_regex="[PF]",
            ),
        )

        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["logprobs"] is True
        assert "top_logprobs" not in call_kwargs
        assert call_kwargs["extra_body"] == {
            "logprob_token_ids": [10, 11],
            "structured_outputs": {"regex": "[PF]"},
            "chat_template_kwargs": {
                "enable_thinking": False,
                "provider_only": "kept",
            },
        }

    def test_managed_server_rejects_explicit_ids_when_protocol_is_too_old(self, provider):
        provider._server = SimpleNamespace(
            stop=lambda: None,
            python_executable="/isolated/vllm026/bin/python",
        )
        params = SamplingParams(max_tokens=1, logprobs=2, logprob_token_ids=(10, 11))

        with (
            patch(
                "olmo_eval.inference.providers.vllm_server.subprocess.run",
                return_value=SimpleNamespace(returncode=3, stderr="", stdout=""),
            ) as run,
            pytest.raises(RuntimeError, match="requires vLLM >= 0.26.0"),
        ):
            provider._require_managed_explicit_logprobs_support(
                params,
                request_type=RequestType.CHAT,
            )
        assert run.call_args.args[0][0] == "/isolated/vllm026/bin/python"

    def test_managed_server_probes_isolated_interpreter_once(self, provider):
        provider._server = SimpleNamespace(
            stop=lambda: None,
            python_executable="/isolated/vllm026/bin/python",
        )
        params = SamplingParams(max_tokens=1, logprobs=2, logprob_token_ids=(10, 11))

        with patch(
            "olmo_eval.inference.providers.vllm_server.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stderr="", stdout=""),
        ) as run:
            provider._require_managed_explicit_logprobs_support(
                params,
                request_type=RequestType.CHAT,
            )
            provider._require_managed_explicit_logprobs_support(
                params,
                request_type=RequestType.CHAT,
            )

        run.assert_called_once()
        command = run.call_args.args[0]
        assert command[0] == "/isolated/vllm026/bin/python"
        assert "ChatCompletionRequest" in command[2]

    def test_describe_request_includes_chat_template_kwargs(self):
        """Chat traces should preserve template kwargs in generation metadata."""
        provider = self._make_provider(chat_template_kwargs={"enable_thinking": False})
        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=[{"role": "user", "content": "Hello"}],
        )
        params = SamplingParams(max_tokens=32, temperature=0.6, top_p=0.6)

        trace = provider.describe_request(request, params)

        assert trace is not None
        assert trace["generation_kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}

    @pytest.mark.anyio
    async def test_generate_completion_appends_eos_to_stop_sequences(self, provider):
        """Completion payloads should include EOS in the stop list when available."""
        client = MagicMock()
        client.completions.create = AsyncMock(return_value=self._make_completion_response())

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        params = SamplingParams(max_tokens=32, stop_sequences=("Question:", "</s>"))

        await provider._generate_completion(client, request, params)

        call_kwargs = client.completions.create.call_args.kwargs
        assert call_kwargs["stop"] == ["Question:", "</s>", "tok1"]

    @pytest.mark.anyio
    async def test_generate_completion_with_explicit_runtime_flags_uses_token_ids(self):
        """Explicit completion runtime flags should tokenize locally and post-process output."""
        provider = self._make_provider(
            add_bos_token=False,
            completion_use_prompt_token_ids=True,
            completion_client_side_stop_trim=True,
            completion_sentencepiece_cleanup=True,
        )
        local_tokenizer = MagicMock()
        local_tokenizer.encode.return_value = [11, 12, 13]
        local_tokenizer.eos_token_id = None
        local_tokenizer.eos_token = None
        provider._get_tokenizer = MagicMock(return_value=local_tokenizer)

        raw_response = MagicMock()
        raw_response.raise_for_status = MagicMock()
        raw_response.json.return_value = {
            "choices": [{"text": "\u2581helloSTOP trailing", "logprobs": None}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        mock_http = AsyncMock()
        mock_http.post.return_value = raw_response
        provider._get_raw_http_client = MagicMock(return_value=mock_http)

        client = MagicMock()
        request = LMRequest(request_type=RequestType.COMPLETION, prompt="Test prompt")
        params = SamplingParams(max_tokens=8, stop_sequences=("STOP",))

        outputs = await provider._generate_completion(client, request, params)

        assert len(outputs) == 1
        assert outputs[0].text == " hello"
        payload = mock_http.post.call_args.kwargs["json"]
        assert payload["prompt"] == [11, 12, 13]
        assert payload["add_special_tokens"] is False
        client.completions.create.assert_not_called()

    def test_build_completion_output_sets_is_greedy_from_top_logprobs(self, provider):
        """Completion metadata should expose greedy status when top logprobs are available."""
        output = provider._build_completion_output(
            text=" yes",
            logprobs_payload={
                "tokens": [" yes"],
                "token_logprobs": [-0.1],
                "top_logprobs": [{" yes": -0.1}],
            },
            usage=None,
            stop_sequences=None,
        )

        assert output.metadata["is_greedy"] is True
        assert output.logprobs == [
            {
                "token": " yes",
                "logprob": -0.1,
                "top_logprobs": [{"token": " yes", "logprob": -0.1}],
            }
        ]

    def test_build_completion_output_detects_non_greedy_from_top_logprobs(self, provider):
        """Completion metadata should mark sampled non-argmax tokens as non-greedy."""
        output = provider._build_completion_output(
            text=" no",
            logprobs_payload={
                "tokens": [" no"],
                "token_logprobs": [-0.7],
                "top_logprobs": [{" yes": -0.1}],
            },
            usage=None,
            stop_sequences=None,
        )

        assert output.metadata["is_greedy"] is False

    def test_build_completion_output_omits_unknown_is_greedy(self, provider):
        """Completion metadata should not invent greedy status without top logprobs."""
        output = provider._build_completion_output(
            text=" yes",
            logprobs_payload={
                "tokens": [" yes"],
                "token_logprobs": [-0.1],
            },
            usage=None,
            stop_sequences=None,
        )

        assert "is_greedy" not in output.metadata

    def test_completion_eos_uses_revision_for_local_tokenizer(self):
        """EOS stop detection should respect the configured tokenizer revision."""
        provider = self._make_provider(
            tokenizer="custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
        )
        local_tokenizer = MagicMock()
        local_tokenizer.eos_token_id = 1
        local_tokenizer.decode.return_value = "</s>"
        from_pretrained, fake_transformers = self._make_fake_transformers(local_tokenizer)

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            assert provider._get_completion_eos_stop() == "</s>"

        from_pretrained.assert_called_once_with(
            "custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
        )

    def test_local_tokenizer_load_uses_force_download(self):
        """Local tokenizer loads should refresh cache when force_download is enabled."""
        provider = self._make_provider(
            tokenizer="custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
            force_download=True,
        )
        local_tokenizer = MagicMock()
        local_tokenizer.eos_token_id = 1
        local_tokenizer.decode.return_value = "</s>"
        from_pretrained, fake_transformers = self._make_fake_transformers(local_tokenizer)

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            assert provider._get_completion_eos_stop() == "</s>"

        from_pretrained.assert_called_once_with(
            "custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
            force_download=True,
        )

    @pytest.mark.anyio
    async def test_logprobs_extracts_continuation_tokens(self, provider, mock_tokenizer):
        """Test that logprobs are correctly extracted for continuation tokens only."""
        # Context: 3 tokens [0, 1, 2], Continuation: 1 token [3]
        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0, 1, 2], [3])

            # prompt_logprobs has one entry per token position.
            # First 3 are context (skipped), 4th is the continuation token.
            prompt_logprobs = [
                None,
                {"0": {"logprob": -0.5, "decoded_token": "answer"}},
                {"1": {"logprob": -0.3, "decoded_token": "is"}},
                {"3": {"logprob": -0.1, "decoded_token": "Paris"}},
            ]

            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="The answer is",
                continuations=[" Paris"],
            )

            outputs = await provider._logprobs_single_impl(request)

            assert len(outputs) == 1
            output = outputs[0]
            assert output.text == " Paris"
            assert output.logprobs is not None
            assert len(output.logprobs) == 1
            assert output.logprobs[0]["token"] == "Paris"
            assert output.logprobs[0]["logprob"] == -0.1
            assert output.metadata["sum_logits"] == pytest.approx(-0.1)
            assert output.metadata["num_tokens"] == 1

    @pytest.mark.anyio
    async def test_logprobs_threads_sampling_temperature(self, provider):
        """Logprob requests should honor the threaded sampling temperature."""
        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0, 1], [2])

            prompt_logprobs = [
                None,
                {"1": {"logprob": -0.2, "decoded_token": "ctx"}},
                {"2": {"logprob": -0.1, "decoded_token": "choice"}},
            ]

            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="Prompt",
                continuations=(" choice",),
            )
            params = SamplingParams(temperature=1.0)

            await provider._logprobs_single_impl(request, params)

            payload = mock_http.post.call_args.kwargs["json"]
            assert payload["temperature"] == 1.0

            trace = provider.describe_request(request, params)
            assert trace is not None
            assert trace["generation_kwargs"]["temperature"] == 1.0
            assert trace["generation_kwargs"]["max_gen_toks"] == 1

    @pytest.mark.anyio
    async def test_logprobs_multiple_continuations(self, provider, mock_tokenizer):
        """Test logprobs computation for multiple continuations."""
        call_count = [0]

        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            # Context is always 2 tokens, continuation is 1 token
            mock_encode.side_effect = lambda tok, ctx, cont: ([0, 1], [2])

            async def mock_post(url, json=None):
                call_count[0] += 1
                # Different logprobs per continuation
                lp_val = {1: -0.1, 2: -0.5, 3: -0.8}[call_count[0]]
                prompt_logprobs = [
                    None,
                    {"1": {"logprob": -0.2, "decoded_token": "is"}},
                    {"2": {"logprob": lp_val, "decoded_token": "cont"}},
                ]
                return self._make_vllm_response(prompt_logprobs)

            mock_http = AsyncMock()
            mock_http.post.side_effect = mock_post
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Capital is",
                continuations=[" Paris", " London", " Berlin"],
            )

            outputs = await provider._logprobs_single_impl(request)

            assert len(outputs) == 3
            logprobs = [o.metadata["sum_logits"] for o in outputs]
            assert logprobs[0] > logprobs[1] > logprobs[2]

    @pytest.mark.anyio
    async def test_logprobs_empty_continuations(self, provider):
        """Test handling of empty continuations list."""
        mock_http = AsyncMock()
        provider._get_raw_http_client = MagicMock(return_value=mock_http)

        request = LMRequest(
            request_type=RequestType.COMPLETION,
            prompt="Test prompt",
            continuations=[],
        )

        outputs = await provider._logprobs_single_impl(request)

        assert len(outputs) == 0
        mock_http.post.assert_not_called()

    @pytest.mark.anyio
    async def test_logprobs_uses_completions_endpoint(self, provider, mock_tokenizer):
        """Test that logprobs uses the raw completions endpoint."""
        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0], [1])

            prompt_logprobs = [
                None,
                {"1": {"logprob": -0.1, "decoded_token": "yes"}},
            ]
            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            await provider._logprobs_single_impl(request)

            mock_http.post.assert_called_once()
            call_args = mock_http.post.call_args
            assert "/completions" in call_args[0][0]

    @pytest.mark.anyio
    async def test_logprobs_passes_prompt_logprobs_param(self, provider, mock_tokenizer):
        """Test that prompt_logprobs parameter is passed correctly."""
        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0], [1])

            prompt_logprobs = [
                None,
                {"1": {"logprob": -0.1, "decoded_token": "yes"}},
            ]
            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            await provider._logprobs_single_impl(request)

            call_kwargs = mock_http.post.call_args[1]
            json_body = call_kwargs["json"]
            assert json_body["prompt_logprobs"] == 5
            assert json_body["max_tokens"] == 1
            assert json_body["add_special_tokens"] is False

    @pytest.mark.anyio
    async def test_logprobs_respects_explicit_prompt_logprobs_override(self, mock_tokenizer):
        """Explicit prompt_logprobs override should be passed through to vLLM."""
        provider = self._make_provider(prompt_logprobs=1)
        provider._tokenizer = mock_tokenizer
        provider._get_tokenizer = MagicMock(return_value=mock_tokenizer)

        with patch(
            "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
        ) as mock_encode:
            mock_encode.return_value = ([0], [1])

            prompt_logprobs = [
                None,
                {"1": {"logprob": -0.1, "decoded_token": "yes"}},
            ]
            mock_http = AsyncMock()
            mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
            provider._get_raw_http_client = MagicMock(return_value=mock_http)

            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            await provider._logprobs_single_impl(request)

            json_body = mock_http.post.call_args.kwargs["json"]
            assert json_body["prompt_logprobs"] == 1

    @pytest.mark.anyio
    async def test_logprobs_loads_local_tokenizer_with_revision(self):
        """Prompt logprob tokenization should use the configured tokenizer revision."""
        provider = self._make_provider(
            tokenizer="custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
        )
        local_tokenizer = MagicMock()
        from_pretrained, fake_transformers = self._make_fake_transformers(local_tokenizer)

        prompt_logprobs = [
            None,
            {"1": {"logprob": -0.1, "decoded_token": "yes"}},
        ]
        mock_http = AsyncMock()
        mock_http.post.return_value = self._make_vllm_response(prompt_logprobs)
        provider._get_raw_http_client = MagicMock(return_value=mock_http)

        with (
            patch.dict("sys.modules", {"transformers": fake_transformers}),
            patch(
                "olmo_eval.inference.providers.vllm_server.encode_context_and_continuation"
            ) as mock_encode,
        ):
            mock_encode.return_value = ([0], [1])
            request = LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=[" yes"],
            )

            outputs = await provider._logprobs_single_impl(request)

        assert len(outputs) == 1
        from_pretrained.assert_called_once_with(
            "custom-tokenizer",
            revision="stage2-step47684",
            trust_remote_code=True,
        )
