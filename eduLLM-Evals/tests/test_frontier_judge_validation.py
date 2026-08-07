"""Offline tests for the cross-family frontier-judge runner."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types as module_types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_frontier_judge_validation.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_frontier_judge_validation_for_tests", RUNNER_PATH
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
frontier = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = frontier
RUNNER_SPEC.loader.exec_module(frontier)


def _case(case_id: str, response_id: str) -> dict:
    return {
        "case_id": case_id,
        "response_id": response_id,
        "scenario_id": "scenario-1",
        "criterion_id": f"criterion-{case_id}",
        "scenario_prompt": "Explain the idea.",
        "conversation_context": [],
        "reference_solution": "A concise reference.",
        "candidate_response": f"Candidate response for {case_id}.",
        "criterion": "The response must explain the idea.",
        "expected_evidence": [],
        "primary_skill": "content",
        "criticality": "not_critical",
    }


def _input_files(tmp_path: Path, *, per_family: int = 1) -> tuple[Path, Path]:
    identities = (
        (
            "openai",
            "gpt-5.5",
            "openai-group/gpt-5.5",
            "Tutor A",
        ),
        (
            "anthropic",
            "opus-4.8",
            "claude-group/claude-opus-4-8",
            "Tutor B",
        ),
        (
            "google",
            "gemini-3.5-flash",
            "gemini-group/gemini-3.5-flash",
            "Tutor C",
        ),
    )
    cases = []
    human_rows = []
    for family, model, slug, tutor in identities:
        for index in range(per_family):
            case = _case(f"{family}-{index}", f"response-{family}-{index}")
            cases.append(case)
            human_rows.append(
                {
                    "case_id": case["case_id"],
                    "case_input_hash": frontier.base.stable_hash(case),
                    "response_id": case["response_id"],
                    "scenario_id": case["scenario_id"],
                    "criterion_id": case["criterion_id"],
                    "candidate_model": model,
                    "candidate_model_slug": slug,
                    "anonymous_tutor": tutor,
                    "human_label": "pass",
                    "human_notes": "must not leak",
                }
            )

    cases_path = tmp_path / "source_cases.jsonl"
    frontier.base.write_jsonl(cases_path, cases)
    labels_path = tmp_path / "human_labels.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(human_rows[0]))
        writer.writeheader()
        writer.writerows(human_rows)
    return cases_path, labels_path


def _prepare(tmp_path: Path, *, per_family: int = 1) -> tuple[Path, Path]:
    cases_path, labels_path = _input_files(tmp_path, per_family=per_family)
    out_dir = tmp_path / "prepared"
    frontier.prepare_artifacts(cases_path, labels_path, out_dir)
    return out_dir / "judge_cases.blinded.jsonl", out_dir / "case_routes.blinded.jsonl"


class FakeGenerator:
    def __init__(self, spec, calls: list, **_: object) -> None:
        self.spec = spec
        self.calls = calls
        self.closed = False

    def generate(self, message_batches):
        self.calls.extend((self.spec.name, messages) for messages in message_batches)
        return [
            frontier.FrontierGenerationResult(
                text=json.dumps(
                    {
                        "verdict": "pass",
                        "rationale": "The criterion is directly satisfied.",
                        "evidence": "Candidate response",
                    }
                ),
                raw_response={"id": f"fake-{index}", "model": self.spec.name},
                resolved_model=f"resolved-{self.spec.name}",
                usage={"input_tokens": 10, "output_tokens": 5},
                latency_ms=1.25,
                provider_attempts=1,
            )
            for index, _ in enumerate(message_batches)
        ]

    def close(self):
        self.closed = True


def _fake_factory(calls: list):
    def factory(spec, **kwargs):
        return FakeGenerator(spec, calls, **kwargs)

    return factory


def test_prepare_emits_family_only_routes_and_plan_is_cross_family(
    tmp_path: Path, monkeypatch
):
    for spec in frontier.FRONTIER_JUDGES.values():
        monkeypatch.delenv(spec.model_env, raising=False)
    for env_name in frontier.DIRECT_MODEL_ENVS.values():
        monkeypatch.delenv(env_name, raising=False)
    for env_name in (
        frontier.TRUEFOUNDRY_BASE_URL_ENV,
        frontier.TRUEFOUNDRY_TOKEN_PARAM_ENV,
        frontier.TRUEFOUNDRY_TEMPERATURE_MODE_ENV,
    ):
        monkeypatch.delenv(env_name, raising=False)
    cases_path, labels_path = _input_files(tmp_path, per_family=2)
    out_dir = tmp_path / "prepared"

    report = frontier.prepare_artifacts(cases_path, labels_path, out_dir)

    assert report["case_count"] == 6
    assert report["candidate_family_counts"] == {
        "openai": 2,
        "anthropic": 2,
        "google": 2,
    }
    route_rows = frontier.base.load_jsonl(out_dir / "case_routes.blinded.jsonl")
    assert all(
        not (frontier.FORBIDDEN_ROUTE_FIELDS & set(row)) for row in route_rows
    )
    assert {tuple(sorted(row)) for row in route_rows} == {
        ("candidate_family", "case_id", "input_hash", "routing_version")
    }

    plan = frontier.build_plan(
        out_dir / "judge_cases.blinded.jsonl",
        out_dir / "case_routes.blinded.jsonl",
    )
    assert plan["backend"] == "truefoundry"
    assert plan["base_url"] == frontier.TRUEFOUNDRY_BASE_URL
    assert plan["request_profile"] == {
        "temperature": 0.0,
        "token_param": "max_completion_tokens",
    }
    assert plan["total_planned_api_calls"] == 72
    for name, judge in plan["judges"].items():
        assert judge["model_id"] == frontier.TRUEFOUNDRY_DEFAULT_MODELS[name]
        assert judge["api_surface"] == "chat.completions"
        assert judge["eligible_cases_per_wave"] == 4, name
        assert judge["own_family_excluded_per_wave"] == 2, name
        assert judge["graded_candidate_family_counts"][judge["judge_family"]] == 0

    direct_plan = frontier.build_plan(
        out_dir / "judge_cases.blinded.jsonl",
        out_dir / "case_routes.blinded.jsonl",
        backend="direct",
    )
    assert direct_plan["backend"] == "direct"
    assert direct_plan["base_url"] is None
    assert direct_plan["judges"]["gpt-5.5"]["model_id"] == "gpt-5.5-2026-04-23"
    assert direct_plan["judges"]["gpt-5.5"]["api_surface"] == "responses"


def test_prepare_rejects_unknown_or_conflicting_tutor_identity(tmp_path: Path):
    cases_path, labels_path = _input_files(tmp_path)
    rows = list(csv.DictReader(labels_path.open(encoding="utf-8")))
    rows[0]["candidate_model_slug"] = "unknown/model"
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="unsupported candidate_model_slug"):
        frontier.prepare_artifacts(cases_path, labels_path, tmp_path / "prepared")


@pytest.mark.parametrize(
    ("text", "verdict", "status"),
    [
        (
            '{"verdict":"pass","rationale":"met","evidence":"quoted text"}',
            "pass",
            "ok",
        ),
        (
            '{"verdict":"fail","rationale":"missing","evidence":"NONE"}',
            "fail",
            "ok",
        ),
        (
            '```json\n{"verdict":"pass","rationale":"met","evidence":"x"}\n```',
            "pass",
            "ok",
        ),
        (
            '{"verdict":"pass","rationale":"met","evidence":"NONE"}',
            "no_decision",
            "parse_error",
        ),
        (
            '{"verdict":"pass","rationale":"met","evidence":"x","extra":1}',
            "no_decision",
            "parse_error",
        ),
    ],
)
def test_strict_frontier_parser(text: str, verdict: str, status: str):
    parsed = frontier.parse_frontier_judgment(text)
    assert parsed.verdict == verdict
    assert parsed.status == status


def test_one_wave_excludes_own_family_writes_raw_response_and_resumes(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv(frontier.TRUEFOUNDRY_API_KEY_ENV, "artifact-secret-key")
    cases_path, routing_path = _prepare(tmp_path)
    output = tmp_path / "results" / "gpt.jsonl"
    calls: list = []

    report = frontier.run_wave(
        cases_path=cases_path,
        routing_path=routing_path,
        judge_name="gpt-5.5",
        wave="canonical_r1",
        output_path=output,
        batch_size=1,
        generator_factory=_fake_factory(calls),
    )

    rows = frontier.base.load_jsonl(output)
    assert report["eligible_case_count"] == 2
    assert report["own_family_excluded_count"] == 1
    assert len(calls) == len(rows) == 2
    assert {row["candidate_family"] for row in rows} == {"anthropic", "google"}
    assert all(row["judge_family"] == "openai" for row in rows)
    assert all(row["backend"] == "truefoundry" for row in rows)
    assert all(row["api_surface"] == "chat.completions" for row in rows)
    assert all(row["base_url"] == frontier.TRUEFOUNDRY_BASE_URL for row in rows)
    assert all(row["request_token_param"] == "max_completion_tokens" for row in rows)
    assert all(row["request_temperature"] == 0.0 for row in rows)
    assert all(row["raw_response"]["id"].startswith("fake-") for row in rows)
    assert all(row["status"] == "ok" and row["verdict"] == "pass" for row in rows)
    rendered_prompts = "\n".join(
        message["content"]
        for _judge, messages in calls
        for message in messages
    )
    assert "must not leak" not in rendered_prompts
    assert "candidate_family" not in rendered_prompts
    assert "Tutor A" not in rendered_prompts
    assert "Tutor B" not in rendered_prompts
    assert "Tutor C" not in rendered_prompts
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["resolved_provider_models"] == ["resolved-gpt-5.5"]
    assert manifest["model_provenance_consistent"] is True
    assert manifest["configuration"]["backend"] == "truefoundry"
    assert manifest["configuration"]["api_key_envs"] == ["TFY_API_KEY"]
    assert "artifact-secret-key" not in output.read_text()
    assert "artifact-secret-key" not in json.dumps(manifest)

    def must_not_build(*_args, **_kwargs):
        raise AssertionError("a complete resume must not instantiate an API client")

    resumed = frontier.run_wave(
        cases_path=cases_path,
        routing_path=routing_path,
        judge_name="gpt-5.5",
        wave="canonical_r1",
        output_path=output,
        batch_size=1,
        resume=True,
        generator_factory=must_not_build,
    )
    assert resumed["new_rows"] == 0
    assert resumed["resumed_rows"] == 2


def test_truefoundry_key_is_redacted_at_artifact_boundary(tmp_path: Path, monkeypatch):
    secret = "tfy-key-that-must-never-be-written"
    monkeypatch.setenv(frontier.TRUEFOUNDRY_API_KEY_ENV, secret)
    cases_path, routing_path = _prepare(tmp_path)
    output = tmp_path / "redacted.jsonl"

    class LeakyFakeGenerator:
        def generate(self, message_batches):
            return [
                frontier.FrontierGenerationResult(
                    text=json.dumps(
                        {
                            "verdict": "pass",
                            "rationale": f"accidentally included {secret}",
                            "evidence": "Candidate response",
                        }
                    ),
                    raw_response={"authorization": f"Bearer {secret}"},
                    resolved_model="resolved-gpt-5.5",
                    usage={"debug": secret},
                    provider_attempts=1,
                )
                for _ in message_batches
            ]

        def close(self):
            pass

    frontier.run_wave(
        cases_path=cases_path,
        routing_path=routing_path,
        judge_name="gpt-5.5",
        wave="canonical_r1",
        output_path=output,
        limit=1,
        generator_factory=lambda *_args, **_kwargs: LeakyFakeGenerator(),
    )

    artifact_text = output.read_text() + output.with_suffix(
        ".manifest.json"
    ).read_text()
    assert secret not in artifact_text
    assert "[REDACTED]" in output.read_text()


def test_wave_manifest_fails_when_gateway_model_drifts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(frontier.TRUEFOUNDRY_API_KEY_ENV, "fake-key")
    cases_path, routing_path = _prepare(tmp_path)
    output = tmp_path / "drift.jsonl"

    class DriftingGenerator:
        def generate(self, message_batches):
            return [
                frontier.FrontierGenerationResult(
                    text=json.dumps(
                        {
                            "verdict": "pass",
                            "rationale": "The criterion is satisfied.",
                            "evidence": "Candidate response",
                        }
                    ),
                    resolved_model=f"resolved-model-{index}",
                    provider_attempts=1,
                )
                for index, _ in enumerate(message_batches)
            ]

        def close(self):
            pass

    report = frontier.run_wave(
        cases_path=cases_path,
        routing_path=routing_path,
        judge_name="gpt-5.5",
        wave="canonical_r1",
        output_path=output,
        generator_factory=lambda *_args, **_kwargs: DriftingGenerator(),
    )

    assert report["status"] == "failed_model_drift"
    assert report["resolved_provider_models"] == [
        "resolved-model-0",
        "resolved-model-1",
    ]
    assert report["model_provenance_consistent"] is False


def test_truefoundry_key_is_redacted_from_setup_errors(tmp_path: Path, monkeypatch):
    secret = "tfy-key-in-constructor-error"
    monkeypatch.setenv(frontier.TRUEFOUNDRY_API_KEY_ENV, secret)
    cases_path, routing_path = _prepare(tmp_path)
    output = tmp_path / "failed.jsonl"

    def fail_to_build(*_args, **_kwargs):
        raise ValueError(f"gateway rejected credential {secret}")

    with pytest.raises(RuntimeError, match=r"\[REDACTED\]") as caught:
        frontier.run_wave(
            cases_path=cases_path,
            routing_path=routing_path,
            judge_name="gpt-5.5",
            wave="canonical_r1",
            output_path=output,
            limit=1,
            generator_factory=fail_to_build,
        )

    assert secret not in str(caught.value)
    manifest_text = output.with_suffix(".manifest.json").read_text()
    assert secret not in manifest_text
    assert "[REDACTED]" in manifest_text


def test_suite_runs_all_three_judges_and_six_waves_offline(tmp_path: Path):
    cases_path, routing_path = _prepare(tmp_path)
    output_dir = tmp_path / "suite"
    calls: list = []

    reports = frontier.run_suite(
        cases_path=cases_path,
        routing_path=routing_path,
        output_dir=output_dir,
        judges=list(frontier.FRONTIER_JUDGES),
        generator_factory=_fake_factory(calls),
        limit=1,
        batch_size=1,
    )

    assert len(reports) == 18
    assert len(calls) == 18
    for judge_name, spec in frontier.FRONTIER_JUDGES.items():
        for wave, (variant, replicate) in frontier.WAVES.items():
            path = output_dir / judge_name / wave / f"{wave}.jsonl"
            rows = frontier.base.load_jsonl(path)
            assert len(rows) == 1
            assert rows[0]["candidate_family"] != spec.family
            assert rows[0]["prompt_variant"] == variant
            assert rows[0]["replicate_id"] == replicate


def test_transient_provider_error_retries_without_network(monkeypatch):
    class RetryingGenerator(frontier.BaseFrontierGenerator):
        def __init__(self):
            self.calls = 0
            super().__init__(
                spec=frontier.FRONTIER_JUDGES["gpt-5.5"],
                model_id="fake-model",
                api_key="secret-value",
                concurrency=1,
                timeout=1,
                max_output_tokens=10,
                reasoning_level="medium",
                max_retries=2,
                retry_base_seconds=0,
                sleep=lambda _seconds: None,
            )

        def _call_once(self, _messages):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout containing secret-value")
            return "{}", {"ok": True}, "fake-model", None

    generator = RetryingGenerator()
    result = generator.generate([[{"role": "user", "content": "test"}]])[0]
    assert result.error is None
    assert result.provider_attempts == 2
    assert generator.calls == 2

    class NonRetryingGenerator(RetryingGenerator):
        def _call_once(self, _messages):
            raise ValueError("bad request using secret-value")

    failed = NonRetryingGenerator().generate(
        [[{"role": "user", "content": "test"}]]
    )[0]
    assert "secret-value" not in failed.error
    assert "[REDACTED]" in failed.error


def test_model_id_precedence_is_cli_then_environment_then_default(monkeypatch):
    spec = frontier.FRONTIER_JUDGES["gpt-5.5"]
    monkeypatch.delenv(spec.model_env, raising=False)
    direct_model_env = frontier.DIRECT_MODEL_ENVS[spec.name]
    monkeypatch.delenv(direct_model_env, raising=False)
    assert frontier.effective_model_id(spec) == (
        "openai-group/gpt-5.5",
        "built_in_default",
    )
    assert frontier.effective_model_id(spec, backend="direct") == (
        "gpt-5.5-2026-04-23",
        "built_in_default",
    )

    monkeypatch.setenv(spec.model_env, "openai-group/account-specific-gpt")
    assert frontier.effective_model_id(spec) == (
        "openai-group/account-specific-gpt",
        spec.model_env,
    )
    assert frontier.effective_model_id(spec, backend="direct") == (
        "gpt-5.5-2026-04-23",
        "built_in_default",
    )
    monkeypatch.setenv(direct_model_env, "direct-account-specific-gpt")
    assert frontier.effective_model_id(spec, backend="direct") == (
        "direct-account-specific-gpt",
        direct_model_env,
    )
    assert frontier.effective_model_id(spec, "openai-group/cli-gpt") == (
        "openai-group/cli-gpt",
        "cli",
    )
    with pytest.raises(ValueError, match="must start with"):
        frontier.effective_model_id(spec, "claude-group/wrong-family")


def _provider_kwargs(spec, **overrides):
    values = {
        "spec": spec,
        "model_id": spec.default_model_id,
        "api_key": "fake-key",
        "concurrency": 1,
        "timeout": 10,
        "max_output_tokens": 4096,
        "reasoning_level": "medium",
        "max_retries": 0,
        "retry_base_seconds": 0,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("judge_name", "expected_model"),
    list(frontier.TRUEFOUNDRY_DEFAULT_MODELS.items()),
)
def test_truefoundry_adapter_uses_chat_completions_for_every_judge(
    monkeypatch, judge_name: str, expected_model: str
):
    captured = {}

    class FakeMessage:
        content = '{"verdict":"pass","rationale":"met","evidence":"quote"}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        model = f"resolved/{expected_model}"
        usage = {"completion_tokens": 7}

        def model_dump(self, **_kwargs):
            return {"model": self.model}

    class Completions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return FakeResponse()

    class Chat:
        completions = Completions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = Chat()

        def close(self):
            pass

    module = module_types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    spec = frontier.FRONTIER_JUDGES[judge_name]
    settings = frontier.TrueFoundrySettings(
        base_url=frontier.TRUEFOUNDRY_BASE_URL,
        token_param="max_completion_tokens",
        temperature_mode="zero",
    )
    generator = frontier.TrueFoundryGenerator(
        settings=settings,
        **_provider_kwargs(spec, model_id=expected_model),
    )
    messages = [{"role": "user", "content": "judge this"}]
    text, _raw, model, usage = generator._call_once(messages)

    assert captured["client"] == {
        "base_url": frontier.TRUEFOUNDRY_BASE_URL,
        "api_key": "fake-key",
        "max_retries": 0,
        "timeout": 10,
    }
    request = captured["request"]
    assert request["model"] == expected_model
    assert request["messages"] == messages
    assert request["temperature"] == 0.0
    assert request["max_completion_tokens"] == 4096
    forbidden = {
        "response_format",
        "reasoning",
        "reasoning_effort",
        "thinking",
        "max_tokens",
    }
    assert forbidden.isdisjoint(request)
    assert text.startswith("{") and model == f"resolved/{expected_model}"
    assert usage == {"completion_tokens": 7}


def test_truefoundry_frozen_compatibility_profile_can_be_explicitly_overridden(
    monkeypatch,
):
    captured = {}

    class FakeResponse:
        choices = [
            module_types.SimpleNamespace(
                message=module_types.SimpleNamespace(content="{}")
            )
        ]
        model = "resolved"
        usage = None

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = module_types.SimpleNamespace(completions=Completions())

        def close(self):
            pass

    module = module_types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    generator = frontier.TrueFoundryGenerator(
        settings=frontier.TrueFoundrySettings(
            base_url=frontier.TRUEFOUNDRY_BASE_URL,
            token_param="max_tokens",
            temperature_mode="omit",
        ),
        **_provider_kwargs(
            frontier.FRONTIER_JUDGES["gpt-5.5"],
            model_id="openai-group/gpt-5.5",
        ),
    )
    generator._call_once([{"role": "user", "content": "judge"}])

    assert captured["max_tokens"] == 4096
    assert "max_completion_tokens" not in captured
    assert "temperature" not in captured
    assert "response_format" not in captured


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (frontier.TRUEFOUNDRY_BASE_URL, frontier.TRUEFOUNDRY_BASE_URL),
        (
            frontier.TRUEFOUNDRY_BASE_URL + "/",
            frontier.TRUEFOUNDRY_BASE_URL,
        ),
        ("https://gateway.example", "https://gateway.example/v1"),
        ("https://gateway.example/v1/", "https://gateway.example/v1"),
        (
            "https://gateway.example/v1/chat/completions",
            "https://gateway.example/v1",
        ),
        (
            frontier.TRUEFOUNDRY_BASE_URL + "/chat/completions",
            frontier.TRUEFOUNDRY_BASE_URL,
        ),
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
        ("http://localhost:8000", "http://localhost:8000/v1"),
    ],
)
def test_truefoundry_base_url_normalization(raw: str, expected: str):
    assert frontier.normalize_truefoundry_base_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "https://user:password@gateway.example/v1",
        "https://gateway.example/unknown",
        "https://gateway.example/v1?key=value",
        "https://gateway.example/v1#fragment",
        "http://gateway.example/v1",
        "not-a-url",
    ],
)
def test_truefoundry_base_url_rejects_unsafe_or_unknown_shapes(raw: str):
    with pytest.raises(ValueError, match="TFY_BASE_URL"):
        frontier.normalize_truefoundry_base_url(raw)


def test_truefoundry_settings_and_credentials_are_backend_specific(monkeypatch):
    spec = frontier.FRONTIER_JUDGES["gpt-5.5"]
    monkeypatch.setenv(frontier.TRUEFOUNDRY_BASE_URL_ENV, "https://gateway.example")
    monkeypatch.setenv(frontier.TRUEFOUNDRY_TOKEN_PARAM_ENV, "max_tokens")
    monkeypatch.setenv(frontier.TRUEFOUNDRY_TEMPERATURE_MODE_ENV, "omit")
    monkeypatch.setenv(frontier.TRUEFOUNDRY_API_KEY_ENV, "tfy-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    settings = frontier.resolve_truefoundry_settings()
    assert settings == frontier.TrueFoundrySettings(
        base_url="https://gateway.example/v1",
        token_param="max_tokens",
        temperature_mode="omit",
    )
    overridden = frontier.resolve_truefoundry_settings(
        token_param="max_completion_tokens", temperature_mode="zero"
    )
    assert overridden.token_param == "max_completion_tokens"
    assert overridden.temperature == 0.0
    assert frontier._credential(spec, "truefoundry") == (
        "tfy-secret",
        "TFY_API_KEY",
    )
    assert frontier._credential(spec, "direct") == (
        "openai-secret",
        "OPENAI_API_KEY",
    )


def test_openai_adapter_uses_responses_api_without_network(monkeypatch):
    captured = {}

    class FakeResponse:
        output_text = '{"verdict":"fail","rationale":"missing","evidence":"NONE"}'
        model = "resolved-gpt"
        usage = {"output_tokens": 9}

        def model_dump(self, **_kwargs):
            return {"id": "response-1", "model": self.model}

    class Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = Responses()

        def close(self):
            pass

    module = module_types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    generator = frontier.OpenAIFrontierGenerator(
        **_provider_kwargs(frontier.FRONTIER_JUDGES["gpt-5.5"])
    )
    messages = [{"role": "user", "content": "judge this"}]
    text, _raw, model, usage = generator._call_once(messages)

    assert captured["input"] == messages
    assert captured["reasoning"] == {"effort": "medium"}
    assert captured["text"]["format"]["schema"] == frontier.JUDGMENT_SCHEMA
    assert captured["store"] is False
    assert "temperature" not in captured
    assert text.startswith("{") and model == "resolved-gpt"
    assert usage == {"output_tokens": 9}


def test_anthropic_adapter_uses_messages_api_without_network(monkeypatch):
    captured = {}

    class TextBlock:
        type = "text"
        text = '{"verdict":"pass","rationale":"met","evidence":"quote"}'

    class FakeResponse:
        content = [TextBlock()]
        model = "resolved-opus"
        usage = {"output_tokens": 8}

        def model_dump(self, **_kwargs):
            return {"id": "message-1", "model": self.model}

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.messages = Messages()

        def close(self):
            pass

    module = module_types.ModuleType("anthropic")
    module.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    generator = frontier.AnthropicFrontierGenerator(
        **_provider_kwargs(frontier.FRONTIER_JUDGES["opus-4.8"])
    )
    text, _raw, model, usage = generator._call_once(
        [{"role": "user", "content": "judge this"}]
    )

    assert captured["messages"] == [{"role": "user", "content": "judge this"}]
    assert captured["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert captured["output_config"]["effort"] == "medium"
    assert captured["output_config"]["format"]["schema"] == frontier.JUDGMENT_SCHEMA
    assert "temperature" not in captured
    assert text.startswith("{") and model == "resolved-opus"
    assert usage == {"output_tokens": 8}


def test_google_adapter_uses_generate_content_without_network(monkeypatch):
    captured = {}

    class FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeResponse:
        text = '{"verdict":"pass","rationale":"met","evidence":"quote"}'
        model_version = "resolved-gemini"
        usage_metadata = {"candidates_token_count": 7}

        def model_dump(self, **_kwargs):
            return {"model_version": self.model_version}

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.models = Models()

        def close(self):
            pass

    google_module = module_types.ModuleType("google")
    genai_module = module_types.ModuleType("google.genai")
    types_module = module_types.ModuleType("google.genai.types")
    genai_module.Client = FakeClient
    types_module.ThinkingConfig = FakeThinkingConfig
    types_module.GenerateContentConfig = FakeGenerateContentConfig
    genai_module.types = types_module
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
    generator = frontier.GoogleFrontierGenerator(
        **_provider_kwargs(frontier.FRONTIER_JUDGES["gemini-3.6-flash"])
    )
    text, _raw, model, usage = generator._call_once(
        [{"role": "user", "content": "judge this"}]
    )

    assert captured["contents"] == "judge this"
    config = captured["config"].kwargs
    assert config["max_output_tokens"] == 4096
    assert config["thinking_config"].kwargs == {"thinking_budget": -1}
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == frontier.JUDGMENT_SCHEMA
    assert text.startswith("{") and model == "resolved-gemini"
    assert usage == {"candidates_token_count": 7}
