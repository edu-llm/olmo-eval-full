"""No-GPU end-to-end coverage for the production multi-mode wrappers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

import olmo_eval.runners.mode_orchestrator as orchestrator_module
from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.edullm.bank import FITTED_BANK_SCHEMA_VERSION
from olmo_eval.edullm.judge import (
    ADAPTER_VERSION,
    FAILURE_PROBABILITY_THRESHOLD,
    PROMPT_VERSION,
    QWEN_FAIL_TOKEN_IDS,
    QWEN_JUDGE_MODEL,
    QWEN_JUDGE_REVISION,
    QWEN_PASS_TOKEN_IDS,
)
from olmo_eval.edullm.mode import (
    ATOMIC_REQUIREMENT_POLICY,
)
from olmo_eval.edullm.mode import (
    MODE_CONFIG_SCHEMA_VERSION as ADAPTIVE_CONFIG_SCHEMA_VERSION,
)
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.manager import InferenceManager
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.mode_config import MODE_CONFIG_SCHEMA_VERSION, parse_mapping
from olmo_eval.runners.mode_orchestrator import ModeRunOrchestrator
from olmo_eval.runners.standard_mode import StandardOlmoRunRequest

CANDIDATE_MODEL = "fixture/candidate"
CANDIDATE_REVISION = "fixture-candidate-revision"
CANDIDATE_ENDPOINT = "http://candidate.fixture:8000/v1"
JUDGE_ENDPOINT = "http://judge.fixture:8001/v1"
RUN_ID = "combined-wrapper-e2e"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)


def _write_fitted_bank(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True)
    rubric_rows = [
        {
            "criterion_id": "scenario-1_c01",
            "scenario_id": "scenario-1",
            "criterion": "The tutor clearly states that two plus two equals four.",
            "expected_evidence": ["Explicitly state 2 + 2 = 4."],
            "q_modeled": {"instruction_following": 1},
            "discrimination": {"instruction_following": 1.5},
            "difficulty": 0.0,
            "calibration_version": "fixture-calibration-v1",
            "scoring_type": "binary",
            "irt_params": {
                "source": "fixture-fitted-m2pl",
                "calibrated": True,
                "fitted": True,
                "synthetic": False,
                "skills_order": ["instruction_following"],
                "latent_correlation": [[1.0]],
            },
        }
    ]
    scenario_rows = [
        {
            "scenario_id": "scenario-1",
            "prompt": "Help the student understand what 2 + 2 equals.",
            "criterion_ids": ["scenario-1_c01"],
            "benchmark": "FixtureBench",
        }
    ]
    rubric_bytes = _jsonl_bytes(rubric_rows)
    scenario_bytes = _jsonl_bytes(scenario_rows)
    rubrics_path = root / "rubrics.jsonl"
    scenarios_path = root / "scenarios.jsonl"
    manifest_path = root / "bank_manifest.json"
    rubrics_path.write_bytes(rubric_bytes)
    scenarios_path.write_bytes(scenario_bytes)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": FITTED_BANK_SCHEMA_VERSION,
                "skills_order": ["instruction_following"],
                "latent_correlation": [[1.0]],
                "counts": {"exported_criteria": 1, "exported_scenarios": 1},
                "outputs": {
                    "sha256": {
                        "rubrics": _sha256(rubric_bytes),
                        "scenarios": _sha256(scenario_bytes),
                    }
                },
                "policies": {"extreme_a_threshold": 6.0, "off_q_tolerance": 1e-10},
                "invariants": {
                    "fitted_only": True,
                    "contains_synthetic_parameters": False,
                    "scenario_criterion_ids_exactly_match_exported_rubrics": True,
                    "all_active_loadings_positive_and_within_threshold": True,
                    "all_inactive_loadings_zero": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return rubrics_path, scenarios_path, manifest_path


def _adaptive_config(bank_paths: tuple[Path, Path, Path]) -> dict[str, Any]:
    rubrics, scenarios, manifest = bank_paths
    return {
        "schema_version": ADAPTIVE_CONFIG_SCHEMA_VERSION,
        "bank": {
            "rubrics_path": str(rubrics),
            "scenarios_path": str(scenarios),
            "manifest_path": str(manifest),
            "skills_order": ["instruction_following"],
            "benchmark_id": "FixtureBench",
            "calibration_version": "fixture-calibration-v1",
            "policy_id": "fixture-cat-policy-v1",
            "scientific_status": "experimental",
        },
        "tutor": {
            "expected_model": CANDIDATE_MODEL,
            "model_family": "fixture-family",
            "model_provenance": {
                "source": "fixture-checkpoint",
                "revision": CANDIDATE_REVISION,
            },
            "generation": {
                "max_tokens": 64,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": None,
                "stop_sequences": None,
                "num_samples": 1,
                "do_sample": False,
            },
        },
        "judge": {
            "provider": "judge",
            "model": QWEN_JUDGE_MODEL,
            "model_family": "qwen",
            "revision": QWEN_JUDGE_REVISION,
            "prompt_version": PROMPT_VERSION,
            "adapter_version": ADAPTER_VERSION,
            "enable_thinking": False,
            "language_model_only": True,
            "pass_token_ids": list(QWEN_PASS_TOKEN_IDS),
            "fail_token_ids": list(QWEN_FAIL_TOKEN_IDS),
            "failure_probability_threshold": FAILURE_PROBABILITY_THRESHOLD,
            "atomic_requirement_policy": ATOMIC_REQUIREMENT_POLICY,
        },
        "quadrature": {
            "nodes_per_dim": 9,
            "max_nodes": 20,
            "method": "gauss_hermite",
            "linear_bound": 8.0,
        },
        "cat": {
            "max_se": 0.01,
            "min_evals_per_skill": 0,
            "min_scenarios": 0,
            "max_scenarios": 1,
            "seed": 7,
            "top_n": 1,
            "selection": "trace",
            "stop_se_method": "eap",
            "theta_init": None,
            "covariance_init_diag": None,
            "mwle_ridge": 1e-6,
        },
    }


def _public_config(
    output_dir: Path,
    bank_paths: tuple[Path, Path, Path],
) -> dict[str, Any]:
    return {
        "schema_version": MODE_CONFIG_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "output_dir": str(output_dir),
        "harness": {
            "name": "combined-wrapper-e2e",
            "provider": {
                "kind": "vllm_server",
                "model": CANDIDATE_MODEL,
                "base_url": CANDIDATE_ENDPOINT,
                "revision": CANDIDATE_REVISION,
            },
            "auxiliary_providers": {
                "judge": {
                    "kind": "vllm_server",
                    "model": QWEN_JUDGE_MODEL,
                    "base_url": JUDGE_ENDPOINT,
                    "revision": QWEN_JUDGE_REVISION,
                    "tokenizer": QWEN_JUDGE_MODEL,
                    "dtype": "bfloat16",
                    "max_model_len": 32768,
                    "trust_remote_code": False,
                    "num_instances": 1,
                    "kwargs": {
                        "language_model_only": True,
                        "tensor_parallel_size": 1,
                        "gpu_memory_utilization": 0.9,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                }
            },
        },
        "modes": [
            {
                "name": "standard_olmo",
                "config": {
                    "task_specs": ["arc:mc:olmo3base"],
                    "save_predictions": False,
                },
            },
            {
                "name": "edullm_adaptive",
                "config": _adaptive_config(bank_paths),
            },
        ],
        "metadata": {"test_kind": "no-gpu-wrapper-e2e"},
    }


class _ScriptedProvider(InferenceProvider):
    def __init__(
        self,
        config: ProviderConfig,
        handler: Callable[[LMRequest, SamplingParams | None], LMOutput],
    ) -> None:
        super().__init__(config.model)
        self.config = config
        self.handler = handler
        self.requests: list[tuple[LMRequest, SamplingParams | None]] = []
        self.close_calls = 0
        self.endpoint = config.base_url
        if config.model == CANDIDATE_MODEL:
            self._tokenizer_revision = config.revision
        else:
            self.revision = config.revision
            self.chat_template_kwargs = dict(config.kwargs.get("chat_template_kwargs", {}))
            self.language_model_only = config.kwargs.get("language_model_only")

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        outputs: list[list[LMOutput]] = []
        for request in requests:
            self.requests.append((request, sampling_params))
            outputs.append([self.handler(request, sampling_params)])
        return outputs

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        return self.generate(requests, sampling_params)

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        del requests, sampling_params
        raise AssertionError("the integration fixture does not use logprobs()")

    async def aclose(self) -> None:
        self.close_calls += 1


def _candidate_output(
    request: LMRequest,
    sampling_params: SamplingParams | None,
) -> LMOutput:
    assert request.messages[-1]["content"] == "Help the student understand what 2 + 2 equals."
    assert sampling_params is not None and sampling_params.max_tokens == 64
    return LMOutput(text="Two plus two equals four, so 2 + 2 = 4.", finish_reason="stop")


def _judge_output(
    request: LMRequest,
    sampling_params: SamplingParams | None,
) -> LMOutput:
    assert request.chat_template_kwargs == {"enable_thinking": False}
    assert sampling_params is not None and sampling_params.seed == 42
    if sampling_params.structured_output_json_schema is not None:
        return LMOutput(
            text=json.dumps(
                {
                    "verdict": "pass",
                    "rationale": "The tutor states the required arithmetic fact.",
                    "evidence": "2 + 2 = 4",
                }
            ),
            finish_reason="stop",
        )

    assert sampling_params.structured_output_regex == "[PF]"
    assert sampling_params.logprob_token_ids == (*QWEN_PASS_TOKEN_IDS, *QWEN_FAIL_TOKEN_IDS)
    p_token_probability = 0.9 / len(QWEN_PASS_TOKEN_IDS)
    f_token_probability = 0.1 / len(QWEN_FAIL_TOKEN_IDS)
    alternatives = [
        {
            "token": " P",
            "token_id": token_id,
            "logprob": math.log(p_token_probability),
        }
        for token_id in QWEN_PASS_TOKEN_IDS
    ] + [
        {
            "token": " F",
            "token_id": token_id,
            "logprob": math.log(f_token_probability),
        }
        for token_id in QWEN_FAIL_TOKEN_IDS
    ]
    return LMOutput(
        text="P",
        token_ids=(QWEN_PASS_TOKEN_IDS[0],),
        logprobs=[
            {
                "token": " P",
                "token_id": QWEN_PASS_TOKEN_IDS[0],
                "logprob": math.log(p_token_probability),
                "top_logprobs": alternatives,
            }
        ],
        finish_reason="stop",
    )


class _FakeNativeRunner:
    def __init__(
        self,
        request: StandardOlmoRunRequest,
        provider_config: ProviderConfig,
    ) -> None:
        self.request = request
        self.provider_config = provider_config
        self.output_dir = str(request.output_dir)
        self.task_specs = list(request.task_specs)
        self.model_name = provider_config.model
        self.validate_calls = 0

    def validate(self) -> None:
        self.validate_calls += 1

    async def run_async(self) -> Mapping[str, Any]:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "summary": {
                "arc:mc:olmo3base": {
                    "metric": "primary_score:average",
                    "score": 1.0,
                }
            },
            "tasks": [{"task": "fixture-arc", "num_instances": 1}],
            "errors": [],
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics) + "\n",
            encoding="utf-8",
        )
        (output_dir / "requests.jsonl").write_text(
            json.dumps({"run_id": self.request.experiment_id, "fixture": True}) + "\n",
            encoding="utf-8",
        )
        return {"tasks": {"fixture-arc": {"score": 1.0}}}


class _FakeNativeRunnerFactory:
    instances: list[_FakeNativeRunnerFactory] = []

    def __init__(
        self,
        resolved_provider_config: ProviderConfig | None = None,
        resolved_auxiliary_configs: Mapping[str, ProviderConfig] | None = None,
    ) -> None:
        self.resolved_provider_config = resolved_provider_config
        self.resolved_auxiliary_configs = dict(resolved_auxiliary_configs or {})
        self.requests: list[StandardOlmoRunRequest] = []
        self.runners: list[_FakeNativeRunner] = []
        self.instances.append(self)

    def __call__(self, request: StandardOlmoRunRequest) -> _FakeNativeRunner:
        assert self.resolved_provider_config is not None
        self.requests.append(request)
        runner = _FakeNativeRunner(request, self.resolved_provider_config)
        self.runners.append(runner)
        return runner


class _TrackingManager:
    def __init__(self, **kwargs: Any) -> None:
        self.inner = InferenceManager(**kwargs)
        self.start_calls = 0
        self.shutdown_calls = 0

    def start(self) -> dict[str, list[dict[str, Any]]]:
        self.start_calls += 1
        return self.inner.start()

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.inner.shutdown()


def test_real_standard_and_adaptive_wrappers_share_providers_artifacts_and_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_paths = _write_fitted_bank(tmp_path / "bank")
    config = parse_mapping(_public_config(tmp_path / "run", bank_paths))
    constructed_configs: dict[str, list[ProviderConfig]] = {}
    scripted_providers: dict[str, _ScriptedProvider] = {}

    def scripted_provider(self: ProviderConfig) -> _ScriptedProvider:
        handler = _candidate_output if self.model == CANDIDATE_MODEL else _judge_output
        provider = _ScriptedProvider(self, handler)
        constructed_configs.setdefault(self.model, []).append(self)
        scripted_providers[self.model] = provider
        return provider

    monkeypatch.setattr(ProviderConfig, "create_provider", scripted_provider)
    _FakeNativeRunnerFactory.instances = []
    monkeypatch.setattr(
        orchestrator_module,
        "AsyncEvalRunnerFactory",
        _FakeNativeRunnerFactory,
    )
    managers: list[_TrackingManager] = []

    def manager_factory(**kwargs: Any) -> _TrackingManager:
        manager = _TrackingManager(**kwargs)
        managers.append(manager)
        return manager

    results = ModeRunOrchestrator(
        config,
        manager_factory=manager_factory,
        runtime_checker=lambda provider: {
            "deployment": "scripted-external-endpoint",
            "version": "0.26.0",
            "endpoint": provider.base_url,
        },
        available_gpu_ids=(),
    ).run()

    assert list(results) == ["standard_olmo", "edullm_adaptive"]
    assert all(result.succeeded for result in results.values())
    assert len(managers) == 1
    assert managers[0].start_calls == managers[0].shutdown_calls == 1

    runtime_factory = next(
        factory
        for factory in _FakeNativeRunnerFactory.instances
        if factory.resolved_provider_config is not None
    )
    assert runtime_factory.resolved_provider_config.base_url == CANDIDATE_ENDPOINT
    assert runtime_factory.resolved_auxiliary_configs["judge"].base_url == JUDGE_ENDPOINT
    assert len(constructed_configs[CANDIDATE_MODEL]) == 1
    assert len(constructed_configs[QWEN_JUDGE_MODEL]) == 1
    assert runtime_factory.resolved_provider_config is constructed_configs[CANDIDATE_MODEL][0]
    assert (
        runtime_factory.resolved_auxiliary_configs["judge"]
        is constructed_configs[QWEN_JUDGE_MODEL][0]
    )
    assert {request.experiment_id for request in runtime_factory.requests} == {RUN_ID}
    assert {request.experiment_name for request in runtime_factory.requests} == {RUN_ID}

    candidate = scripted_providers[CANDIDATE_MODEL]
    judge = scripted_providers[QWEN_JUDGE_MODEL]
    assert candidate.endpoint == CANDIDATE_ENDPOINT
    assert judge.endpoint == JUDGE_ENDPOINT
    assert len(candidate.requests) == 1
    assert len(judge.requests) == 2
    assert candidate.close_calls == judge.close_calls == 1

    standard_dir = config.output_dir / "modes" / "standard_olmo"
    adaptive_dir = config.output_dir / "modes" / "edullm_adaptive"
    assert set(results["standard_olmo"].artifacts) == {"metrics.json", "requests.jsonl"}
    assert set(results["edullm_adaptive"].artifacts) == {
        "manifest.json",
        "tutor_responses.jsonl",
        "judge_rows.jsonl",
        "cat_result.json",
        "cat_trace.jsonl",
    }
    assert (standard_dir / "mode_result.json").is_file()
    assert (adaptive_dir / "mode_result.json").is_file()
    assert json.loads((adaptive_dir / "manifest.json").read_text())["run_id"] == RUN_ID
    assert json.loads((adaptive_dir / "cat_result.json").read_text())["criteria_observed"] == 1
    judge_row = json.loads((adaptive_dir / "judge_rows.jsonl").read_text())
    assert judge_row["verdict"] == "pass"
    assert judge_row["atomic_results"][0]["classification_token_id"] == 47

    shared_manifest = json.loads((config.output_dir / "manifest.json").read_text())
    assert shared_manifest["run_id"] == RUN_ID
    assert shared_manifest["status"] == "succeeded"
    assert shared_manifest["selected_modes"] == ["standard_olmo", "edullm_adaptive"]
    assert shared_manifest["completed_modes"] == ["standard_olmo", "edullm_adaptive"]
    report_index = json.loads((config.output_dir / "report_index.json").read_text())
    assert set(report_index["modes"]) == {"standard_olmo", "edullm_adaptive"}
