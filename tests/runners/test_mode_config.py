"""Strict, no-provider tests for the multi-mode configuration file."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from olmo_eval.edullm.judge import (
    ADAPTER_VERSION,
    FAILURE_PROBABILITY_THRESHOLD,
    PROMPT_VERSION,
    QWEN_JUDGE_MODEL,
    QWEN_JUDGE_REVISION,
)
from olmo_eval.edullm.mode import (
    ATOMIC_REQUIREMENT_POLICY,
)
from olmo_eval.edullm.mode import (
    MODE_CONFIG_SCHEMA_VERSION as ADAPTIVE_CONFIG_SCHEMA_VERSION,
)
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.mode_config import (
    MODE_CONFIG_SCHEMA_VERSION,
    load_mode_config,
    parse_mapping,
)


def _candidate() -> dict[str, Any]:
    return {
        "kind": "mock",
        "model": "fixture-tutor",
        "revision": "fixture-revision",
        "num_instances": 1,
    }


def _judge() -> dict[str, Any]:
    return {
        "kind": "vllm_server",
        "model": QWEN_JUDGE_MODEL,
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


def _config(tmp_path: Path, *, adaptive: bool = False) -> dict[str, Any]:
    harness: dict[str, Any] = {
        "name": "fixture",
        "provider": _candidate(),
    }
    modes: list[dict[str, Any]] = [
        {
            "name": "standard_olmo",
            "config": {
                "task_specs": ["arc:mc:olmo3base"],
                "save_predictions": False,
            },
        }
    ]
    if adaptive:
        harness["auxiliary_providers"] = {"judge": _judge()}
        modes.append(
            {
                "name": "edullm_adaptive",
                "config": {
                    "schema_version": ADAPTIVE_CONFIG_SCHEMA_VERSION,
                    "bank": {
                        "rubrics_path": str(tmp_path / "bank/rubrics.json"),
                        "scenarios_path": str(tmp_path / "bank/scenarios.json"),
                        "manifest_path": str(tmp_path / "bank/manifest.json"),
                        "skills_order": ["instruction_following"],
                        "benchmark_id": "FixtureBench",
                        "calibration_version": "fixture-calibration-v1",
                        "policy_id": "fixture-policy-v1",
                        "scientific_status": "experimental",
                    },
                    "tutor": {
                        "expected_model": "fixture-tutor",
                        "model_family": "fixture-family",
                        "model_provenance": {
                            "source": "fixture-provider",
                            "revision": "fixture-revision",
                        },
                        "generation": {
                            "max_tokens": 128,
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
                        "pass_token_ids": [47, 387, 9729],
                        "fail_token_ids": [37, 426, 12362],
                        "failure_probability_threshold": FAILURE_PROBABILITY_THRESHOLD,
                        "atomic_requirement_policy": ATOMIC_REQUIREMENT_POLICY,
                    },
                    "quadrature": {
                        "nodes_per_dim": 7,
                        "max_nodes": 1000,
                        "method": "gauss_hermite",
                        "linear_bound": 8.0,
                    },
                    "cat": {
                        "max_se": 0.5,
                        "min_evals_per_skill": 0,
                        "min_scenarios": 1,
                        "max_scenarios": 2,
                        "seed": 42,
                        "top_n": 1,
                        "selection": "trace",
                        "stop_se_method": "eap",
                        "theta_init": None,
                        "covariance_init_diag": None,
                        "mwle_ridge": 0.000001,
                    },
                },
            }
        )
    return {
        "schema_version": MODE_CONFIG_SCHEMA_VERSION,
        "run_id": "fixture-run",
        "output_dir": str(tmp_path / "results"),
        "harness": harness,
        "modes": modes,
    }


def _use_precomputed_responses(raw: dict[str, Any], tmp_path: Path) -> None:
    """Convert an adaptive fixture to the provider-free tutor replay mode."""

    adaptive_mode = next(mode for mode in raw["modes"] if mode["name"] == "edullm_adaptive")
    tutor = adaptive_mode["config"]["tutor"]
    tutor["expected_model"] = "archived-tutor-model"
    tutor["model_provenance"] = {
        "source": "fixture-response-archive",
        "revision": "archive-revision",
    }
    tutor["response_source"] = {
        "kind": "precomputed_jsonl",
        "schema_version": "edullm-precomputed-tutor-responses-v1",
        "path": str(tmp_path / "responses.jsonl"),
        "sha256": "a" * 64,
    }
    tutor["generation"] = None


def test_parse_injects_shared_harness_into_standard_mode(tmp_path: Path) -> None:
    raw = _config(tmp_path)
    parsed = parse_mapping(raw)

    assert parsed.run_id == "fixture-run"
    assert parsed.output_dir == (tmp_path / "results").resolve()
    assert parsed.continue_on_mode_failure is True
    assert parsed.mode_names == ("standard_olmo",)
    standard = parsed.modes[0]
    assert standard.config["task_specs"] == ["arc:mc:olmo3base"]
    assert standard.config["harness_config"] == parsed.harness.to_dict()
    assert "harness_config" not in raw["modes"][0]["config"]


def test_parse_never_creates_a_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_: ProviderConfig) -> None:
        raise AssertionError("configuration parsing must not create providers")

    monkeypatch.setattr(ProviderConfig, "create_provider", forbidden)

    assert parse_mapping(_config(tmp_path)).harness.provider.model == "fixture-tutor"


def test_standard_mode_rejects_user_supplied_duplicate_harness(tmp_path: Path) -> None:
    raw = _config(tmp_path)
    raw["modes"][0]["config"]["harness_config"] = {}

    with pytest.raises(ValueError, match="must not contain harness_config"):
        parse_mapping(raw)


def test_future_mode_name_and_config_pass_through_unchanged(tmp_path: Path) -> None:
    raw = _config(tmp_path)
    future_config = {"new_policy": {"alpha": 0.2}, "enabled": True}
    raw["modes"] = [{"name": "future_eval", "config": future_config}]

    parsed = parse_mapping(raw)

    assert parsed.mode_names == ("future_eval",)
    assert dict(parsed.modes[0].config) == future_config


def test_adaptive_mode_accepts_exact_frozen_qwen_declaration(tmp_path: Path) -> None:
    raw = _config(tmp_path, adaptive=True)
    raw["harness"]["auxiliary_providers"]["judge"]["tokenizer"] = None
    parsed = parse_mapping(raw)

    assert parsed.mode_names == ("standard_olmo", "edullm_adaptive")
    assert parsed.harness.auxiliary_providers["judge"].model == QWEN_JUDGE_MODEL
    assert parsed.modes[1].config["schema_version"] == ADAPTIVE_CONFIG_SCHEMA_VERSION


def test_adaptive_precomputed_responses_accept_mock_without_provider_identity_match(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path, adaptive=True)
    _use_precomputed_responses(raw, tmp_path)
    raw["modes"] = [mode for mode in raw["modes"] if mode["name"] == "edullm_adaptive"]
    raw["harness"]["provider"] = {
        "kind": "mock",
        "model": "precomputed-response-sentinel",
        "num_instances": 1,
    }

    parsed = parse_mapping(raw)

    assert parsed.mode_names == ("edullm_adaptive",)
    assert parsed.harness.provider.kind == "mock"
    assert parsed.harness.provider.revision is None
    assert parsed.modes[0].config["tutor"]["expected_model"] == "archived-tutor-model"


def test_adaptive_precomputed_responses_require_mock_provider(tmp_path: Path) -> None:
    raw = _config(tmp_path, adaptive=True)
    _use_precomputed_responses(raw, tmp_path)
    raw["modes"] = [mode for mode in raw["modes"] if mode["name"] == "edullm_adaptive"]
    raw["harness"]["provider"] = {
        "kind": "vllm_server",
        "model": "live-tutor-model",
        "revision": "live-revision",
        "num_instances": 1,
    }

    with pytest.raises(ValueError, match="precomputed_jsonl requires.*kind='mock'"):
        parse_mapping(raw)


def test_adaptive_precomputed_responses_reject_standard_olmo(tmp_path: Path) -> None:
    raw = _config(tmp_path, adaptive=True)
    _use_precomputed_responses(raw, tmp_path)

    with pytest.raises(ValueError, match="standard_olmo cannot run.*precomputed_jsonl"):
        parse_mapping(raw)


def test_adaptive_precomputed_responses_still_require_declared_provenance(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path, adaptive=True)
    _use_precomputed_responses(raw, tmp_path)
    raw["modes"] = [mode for mode in raw["modes"] if mode["name"] == "edullm_adaptive"]
    raw["modes"][0]["config"]["tutor"]["model_provenance"]["revision"] = ""

    with pytest.raises(ValueError, match="model_provenance.revision"):
        parse_mapping(raw)


def test_adaptive_precomputed_responses_still_require_frozen_qwen(tmp_path: Path) -> None:
    raw = _config(tmp_path, adaptive=True)
    _use_precomputed_responses(raw, tmp_path)
    raw["modes"] = [mode for mode in raw["modes"] if mode["name"] == "edullm_adaptive"]
    raw["harness"]["auxiliary_providers"]["judge"]["model"] = "other-judge"

    with pytest.raises(ValueError, match="frozen Qwen judge model"):
        parse_mapping(raw)


def test_explicit_provider_response_source_keeps_candidate_identity_checks(
    tmp_path: Path,
) -> None:
    raw = _config(tmp_path, adaptive=True)
    raw["modes"][1]["config"]["tutor"]["response_source"] = {"kind": "provider"}
    raw["modes"][1]["config"]["tutor"]["expected_model"] = "other-model"

    with pytest.raises(ValueError, match="expected_model must equal"):
        parse_mapping(raw)


def test_adaptive_mode_requires_explicit_candidate_revision(tmp_path: Path) -> None:
    raw = _config(tmp_path, adaptive=True)
    raw["harness"]["provider"].pop("revision")

    with pytest.raises(ValueError, match="immutable harness.provider.revision"):
        parse_mapping(raw)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("expected_model", "other-model", "expected_model must equal"),
        (
            "model_provenance.revision",
            "other-revision",
            "model_provenance.revision must equal",
        ),
    ],
)
def test_adaptive_mode_cross_checks_candidate_identity(
    tmp_path: Path,
    field: str,
    replacement: str,
    message: str,
) -> None:
    raw = _config(tmp_path, adaptive=True)
    tutor = raw["modes"][1]["config"]["tutor"]
    _set_path(tutor, field, replacement)

    with pytest.raises(ValueError, match=message):
        parse_mapping(raw)


def _set_path(value: dict[str, Any], path: str, replacement: Any) -> None:
    parts = path.split(".")
    target: dict[str, Any] = value
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        ("kind", "litellm", "kind must be 'vllm_server'"),
        ("model", "other", "judge model"),
        ("revision", "other", "judge revision"),
        ("tokenizer", "other", "tokenizer must be null or equal"),
        ("dtype", "float16", "dtype must be 'bfloat16'"),
        ("max_model_len", 8192, "max_model_len must equal 32768"),
        ("trust_remote_code", True, "trust_remote_code must be false"),
        ("num_instances", 2, "num_instances must equal 1"),
        ("kwargs.language_model_only", False, "language_model_only must be true"),
        ("kwargs.tensor_parallel_size", 2, "tensor_parallel_size must equal 1"),
        ("kwargs.gpu_memory_utilization", 0.8, "gpu_memory_utilization must equal 0.9"),
        (
            "kwargs.chat_template_kwargs.enable_thinking",
            True,
            "enable_thinking must be false",
        ),
    ],
)
def test_adaptive_mode_rejects_nonfrozen_qwen_engine_setting(
    tmp_path: Path,
    path: str,
    replacement: Any,
    message: str,
) -> None:
    raw = _config(tmp_path, adaptive=True)
    judge = raw["harness"]["auxiliary_providers"]["judge"]
    _set_path(judge, path, replacement)

    with pytest.raises(ValueError, match=message):
        parse_mapping(raw)


def test_adaptive_mode_requires_named_judge(tmp_path: Path) -> None:
    raw = _config(tmp_path)
    raw["modes"] = [{"name": "edullm_adaptive", "config": {}}]

    with pytest.raises(ValueError, match="requires harness.auxiliary_providers.judge"):
        parse_mapping(raw)


@pytest.mark.parametrize(
    "kwargs_mutation",
    [
        lambda kwargs: kwargs.pop("language_model_only"),
        lambda kwargs: kwargs.update({"unfrozen_engine_option": True}),
    ],
)
def test_adaptive_judge_kwargs_have_exact_keys(
    tmp_path: Path,
    kwargs_mutation: Any,
) -> None:
    raw = _config(tmp_path, adaptive=True)
    kwargs = raw["harness"]["auxiliary_providers"]["judge"]["kwargs"]
    kwargs_mutation(kwargs)

    with pytest.raises(ValueError, match="unknown field|missing required field"):
        parse_mapping(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c.update({"extra": True}), "unknown field"),
        (lambda c: c.pop("run_id"), "missing required field"),
        (
            lambda c: c.update({"schema_version": "wrong"}),
            "unsupported schema_version",
        ),
        (lambda c: c.update({"continue_on_mode_failure": 1}), "must be a boolean"),
        (lambda c: c.update({"modes": []}), "non-empty list"),
        (lambda c: c["harness"].update({"unknown": True}), "harness has unknown field"),
        (lambda c: c["modes"][0].update({"unknown": True}), r"modes\[0\] has unknown"),
        (
            lambda c: c["modes"].append(copy.deepcopy(c["modes"][0])),
            "selected more than once",
        ),
    ],
)
def test_top_level_schema_is_exact(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    raw = _config(tmp_path)
    mutation(raw)

    with pytest.raises(ValueError, match=message):
        parse_mapping(raw)


@pytest.mark.parametrize(
    ("provider_change", "message"),
    [
        ({"model": ""}, "model must be non-blank"),
        ({"num_instances": 2}, "num_instances must equal 1"),
        ({"num_instances": 1.0}, "num_instances must equal 1"),
        ({"kind": "vllm"}, "unresolved local provider"),
        ({"kind": "hf"}, "unresolved local provider"),
        ({"kind": "olmo_core"}, "unresolved local provider"),
        ({"unexpected": True}, "unknown field"),
    ],
)
def test_candidate_provider_contract(
    tmp_path: Path,
    provider_change: dict[str, Any],
    message: str,
) -> None:
    raw = _config(tmp_path)
    raw["harness"]["provider"].update(provider_change)

    with pytest.raises(ValueError, match=message):
        parse_mapping(raw)


def test_unresolved_vllm_server_is_allowed_for_orchestrator_resolution(tmp_path: Path) -> None:
    raw = _config(tmp_path)
    raw["harness"]["provider"] = {
        "kind": "vllm_server",
        "model": "fixture-local-model",
        "num_instances": 1,
    }

    parsed = parse_mapping(raw)

    assert parsed.harness.provider.kind == "vllm_server"
    assert parsed.harness.provider.base_url is None


@pytest.mark.parametrize("bad_value", [float("nan"), ("not", "json")])
def test_config_requires_strict_json_values(tmp_path: Path, bad_value: Any) -> None:
    raw = _config(tmp_path)
    raw["metadata"] = {"bad": bad_value}

    with pytest.raises(ValueError, match="non-finite|non-JSON"):
        parse_mapping(raw)


def test_output_directory_cannot_be_root_dot_or_existing_file(tmp_path: Path) -> None:
    existing_file = tmp_path / "file.txt"
    existing_file.write_text("fixture", encoding="utf-8")

    for value in ("/", ".", str(existing_file)):
        raw = _config(tmp_path)
        raw["output_dir"] = value
        with pytest.raises(ValueError, match="root|directory path"):
            parse_mapping(raw)


def test_loads_yaml_and_json_without_creating_providers(tmp_path: Path) -> None:
    raw = _config(tmp_path)
    json_path = tmp_path / "mode.json"
    yaml_path = tmp_path / "mode.yaml"
    json_path.write_text(json.dumps(raw), encoding="utf-8")
    yaml_path.write_text(
        "\n".join(
            [
                f"schema_version: {MODE_CONFIG_SCHEMA_VERSION}",
                "run_id: yaml-run",
                f"output_dir: {tmp_path / 'yaml-results'}",
                "harness:",
                "  name: fixture",
                "  provider:",
                "    kind: mock",
                "    model: fixture-tutor",
                "    num_instances: 1",
                "modes:",
                "  - name: future_eval",
                "    config:",
                "      enabled: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_mode_config(json_path).run_id == "fixture-run"
    assert load_mode_config(yaml_path).mode_names == ("future_eval",)


def test_loader_rejects_interpolation_and_non_config_extension(tmp_path: Path) -> None:
    interpolation = tmp_path / "interpolation.yaml"
    interpolation.write_text(
        "\n".join(
            [
                f"schema_version: {MODE_CONFIG_SCHEMA_VERSION}",
                "run_id: interpolation",
                "output_dir: ${oc.env:HOME}",
                "harness:",
                "  provider: {kind: mock, model: fixture}",
                "modes: [{name: future, config: {}}]",
            ]
        ),
        encoding="utf-8",
    )
    unsupported = tmp_path / "config.txt"
    unsupported.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot use OmegaConf interpolation"):
        load_mode_config(interpolation)
    with pytest.raises(ValueError, match="must use .yaml"):
        load_mode_config(unsupported)
