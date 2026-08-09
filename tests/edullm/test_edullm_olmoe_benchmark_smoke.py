"""GPU-free contracts for the native OLMoE GSM8K benchmark smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from olmo_eval.edullm.olmoe_benchmark_smoke import (
    BENCHMARK_DATASET,
    BENCHMARK_DATASET_REVISION,
    BENCHMARK_NAME,
    BENCHMARK_TASK,
    DTYPE,
    EXPECTED_INSTANCES,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    MODEL_ARCHITECTURE,
    MODEL_ID,
    MODEL_REVISION,
    SCHEMA_VERSION,
    TASK_OVERRIDES,
    TASK_SEED,
    TENSOR_PARALLEL_SIZE,
    BenchmarkValidationError,
    build_argument_parser,
    build_mode_run_config,
    run_olmoe_benchmark_smoke,
    validate_benchmark_artifacts,
)
from olmo_eval.runners.modes import ModeResult, ModeStatus


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_native_fixture(
    root: Path,
    *,
    run_id: str = "olmoe_gsm8k_smoke",
    count: int = EXPECTED_INSTANCES,
    blank_index: int | None = None,
    write_log: bool = True,
) -> None:
    config = build_mode_run_config(root, run_id=run_id)
    root.mkdir(parents=True, exist_ok=True)
    mode_root = root / "modes" / "standard_olmo"
    mode_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "olmo-eval-mode-run-v1",
        "run_id": run_id,
        "status": "succeeded",
        "continue_on_mode_failure": False,
        "selected_modes": ["standard_olmo"],
        "completed_modes": ["standard_olmo"],
        "context_metadata": {
            "mode_runner": {
                "candidate_model": MODEL_ID,
                "candidate_revision": MODEL_REVISION,
                "candidate_provider": "vllm_server",
            }
        },
        "results": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    preflight = {
        "schema_version": "edullm-olmoe-benchmark-preflight-v1",
        "gpu": {
            "selected_visible_gpu_id": 0,
            "visible_gpu_count": 1,
            "name": "fixture BF16 GPU",
            "compute_capability": [8, 0],
            "bfloat16_supported": True,
            "scheduler_mask_present": True,
        },
        "model": {
            "source": "huggingface_config_json",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "architectures": [MODEL_ARCHITECTURE],
            "model_type": "olmoe",
            "torch_dtype": DTYPE,
        },
    }
    (root / "olmoe_preflight.json").write_text(json.dumps(preflight) + "\n", encoding="utf-8")
    mode_result = {
        "mode": "standard_olmo",
        "implementation_version": "olmo-async-runner-v1",
        "status": "succeeded",
        "metrics": {},
        "artifacts": [],
        "warnings": [],
        "completed_units": count,
        "error": None,
    }
    (mode_root / "mode_result.json").write_text(json.dumps(mode_result) + "\n", encoding="utf-8")
    metrics = {
        "config": config.harness.to_dict(),
        "tasks": [
            {
                "task": BENCHMARK_TASK,
                "metrics": {"accuracy": {"exact_match": 0.5}},
                "num_instances": count,
                "primary_metric": "accuracy:exact_match",
            }
        ],
        "summary": {BENCHMARK_TASK: {"metric": "accuracy:exact_match", "score": 0.5}},
        "errors": [],
    }
    (mode_root / "metrics.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")

    request_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for index in range(count):
        native_id = 100 + index
        request_rows.append(
            {
                "request_type": "generate_until",
                "task_name": BENCHMARK_TASK,
                "native_id": native_id,
                "request": {
                    "context": f"eight-shot native GSM8K prompt {index}",
                    "generation_kwargs": {
                        "max_gen_toks": MAX_TOKENS,
                        "do_sample": False,
                        "temperature": 0.0,
                        "top_p": 1.0,
                    },
                },
            }
        )
        output = "   " if blank_index == index else f"Reasoning for case {index}. Answer: 4"
        prediction_rows.append(
            {
                "native_id": native_id,
                "model_output": [{"text": output, "extracted_answer": "4"}],
                "final_output": output,
                "instance_metrics": {},
                "label": "4",
            }
        )

    _jsonl(
        mode_root / "requests" / "allenai_OLMoE" / "gsm8k_fixture-requests.jsonl",
        request_rows,
    )
    _jsonl(
        mode_root / "predictions" / "allenai_OLMoE" / "gsm8k_fixture-predictions.jsonl",
        prediction_rows,
    )
    if write_log:
        log = root / "logs" / "vllm_server_olmoe" / "server.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("vLLM server ready and stopped cleanly\n", encoding="utf-8")


def test_config_uses_native_gsm8k_and_frozen_olmoe_provider(tmp_path: Path) -> None:
    config = build_mode_run_config(tmp_path / "run")

    provider = config.harness.provider
    assert provider.get_provider_name() == "vllm_server"
    assert provider.model == provider.tokenizer == MODEL_ID
    assert provider.revision == MODEL_REVISION
    assert provider.dtype == DTYPE == "bfloat16"
    assert provider.max_model_len == MAX_MODEL_LEN == 4096
    assert provider.num_instances == 1
    assert provider.kwargs == {
        "tokenizer_revision": MODEL_REVISION,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "enable_prefix_caching": True,
    }

    assert config.mode_names == ("standard_olmo",)
    mode = config.modes[0]
    assert mode.config["task_specs"] == [BENCHMARK_TASK]
    assert dict(mode.config["task_overrides"][BENCHMARK_TASK]) == dict(TASK_OVERRIDES)
    assert mode.config["save_predictions"] is True
    assert mode.config["save_requests"] is True
    assert config.metadata["benchmark"] == BENCHMARK_NAME
    assert config.metadata["benchmark_dataset"] == BENCHMARK_DATASET
    assert config.metadata["benchmark_dataset_revision_observed_at_handoff"] == (
        BENCHMARK_DATASET_REVISION
    )
    assert config.metadata["expected_model_architecture"] == MODEL_ARCHITECTURE
    assert config.metadata["score_is_acceptance_gate"] is False
    assert TASK_SEED == 42

    with pytest.raises(ValueError, match="frozen to 'bfloat16'"):
        build_mode_run_config(tmp_path / "wrong-dtype", dtype="float16")


def test_valid_native_artifacts_pass_all_generation_gates(tmp_path: Path) -> None:
    root = tmp_path / "passed"
    _write_native_fixture(root)

    result = validate_benchmark_artifacts(root)

    assert result.validation["schema_version"] == SCHEMA_VERSION
    assert result.validation["status"] == "passed"
    assert all(result.validation["gates"].values())
    assert result.validation["scope"] == {
        "native_olmo_benchmark": True,
        "benchmark_score_recorded": True,
        "benchmark_score_is_acceptance_gate": False,
        "judge": False,
        "calibration": False,
        "cat": False,
    }
    assert result.predictions_path.name.endswith("-predictions.jsonl")
    assert result.requests_path.name.endswith("-requests.jsonl")
    reloaded = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert reloaded == result.validation


def test_blank_native_output_fails_and_persists_failed_validation(tmp_path: Path) -> None:
    root = tmp_path / "blank"
    _write_native_fixture(root, blank_index=1)

    with pytest.raises(BenchmarkValidationError, match="blank final_output"):
        validate_benchmark_artifacts(root)

    validation = json.loads((root / "olmoe_benchmark_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "failed"
    assert validation["gates"]["all_outputs_nonblank"] is False
    assert "blank final_output" in validation["error"]["message"]


def test_missing_diagnostic_score_cannot_claim_score_was_recorded(tmp_path: Path) -> None:
    root = tmp_path / "missing-score"
    _write_native_fixture(root)
    metrics_path = root / "modes" / "standard_olmo" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["tasks"][0]["metrics"] = {}
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")

    with pytest.raises(BenchmarkValidationError, match="exact-match accuracy"):
        validate_benchmark_artifacts(root)


@pytest.mark.parametrize(
    ("count", "write_log", "message"),
    [
        (2, True, "exactly 3 instances"),
        (EXPECTED_INSTANCES, False, "server log"),
    ],
)
def test_incomplete_native_run_cannot_pass(
    tmp_path: Path,
    count: int,
    write_log: bool,
    message: str,
) -> None:
    root = tmp_path / f"incomplete-{count}-{write_log}"
    _write_native_fixture(root, count=count, write_log=write_log)

    with pytest.raises(BenchmarkValidationError, match=message):
        validate_benchmark_artifacts(root)


def test_wrapper_dispatches_native_mode_on_selected_visible_gpu(tmp_path: Path) -> None:
    created: list[tuple[Any, tuple[int, ...]]] = []

    class _FixtureOrchestrator:
        def __init__(self, config: Any, *, available_gpu_ids: tuple[int, ...]) -> None:
            self.config = config
            created.append((config, available_gpu_ids))

        def run(self) -> dict[str, ModeResult]:
            _write_native_fixture(self.config.output_dir, run_id=self.config.run_id)
            return {
                "standard_olmo": ModeResult(
                    mode="standard_olmo",
                    implementation_version="olmo-async-runner-v1",
                    status=ModeStatus.SUCCEEDED,
                    completed_units=EXPECTED_INSTANCES,
                )
            }

    result = run_olmoe_benchmark_smoke(
        tmp_path / "wrapper",
        gpu_id=2,
        run_id="fixture_run",
        reproduction_command="fixture benchmark command",
        orchestrator_factory=_FixtureOrchestrator,
        gpu_checker=lambda gpu_id: {
            "selected_visible_gpu_id": gpu_id,
            "visible_gpu_count": 3,
            "name": "fixture BF16 GPU",
            "compute_capability": [8, 0],
            "bfloat16_supported": True,
            "scheduler_mask_present": True,
        },
        model_checker=lambda: {
            "source": "huggingface_config_json",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "architectures": [MODEL_ARCHITECTURE],
            "model_type": "olmoe",
            "torch_dtype": DTYPE,
        },
    )

    assert len(created) == 1
    config, gpu_ids = created[0]
    assert config.mode_names == ("standard_olmo",)
    assert gpu_ids == (2,)
    assert result.validation["status"] == "passed"
    assert (result.output_dir / "reproduction_command.txt").read_text(encoding="utf-8") == (
        "fixture benchmark command\n"
    )


def test_preflight_failure_leaves_command_and_structured_forensics(tmp_path: Path) -> None:
    root = tmp_path / "preflight-failure"

    def forbidden_orchestrator(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("orchestrator must not start after failed model preflight")

    with pytest.raises(RuntimeError, match="bad pinned config"):
        run_olmoe_benchmark_smoke(
            root,
            reproduction_command="fixture failed command",
            orchestrator_factory=forbidden_orchestrator,
            gpu_checker=lambda gpu_id: {
                "selected_visible_gpu_id": gpu_id,
                "visible_gpu_count": 1,
                "name": "fixture BF16 GPU",
                "compute_capability": [8, 0],
                "bfloat16_supported": True,
                "scheduler_mask_present": True,
            },
            model_checker=lambda: (_ for _ in ()).throw(RuntimeError("bad pinned config")),
        )

    assert (root / "reproduction_command.txt").read_text(encoding="utf-8") == (
        "fixture failed command\n"
    )
    validation = json.loads((root / "olmoe_benchmark_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "failed"
    assert validation["error"] == {
        "stage": "model_config_preflight",
        "type": "RuntimeError",
        "message": "bad pinned config",
    }
    assert validation["gates"]["bf16_gpu_verified"] is True
    assert validation["gates"]["model_architecture_verified"] is False


@pytest.mark.parametrize("gpu_id", [-1, True, 1.5, "0"])
def test_wrapper_rejects_invalid_gpu_id_before_creating_output(
    tmp_path: Path,
    gpu_id: object,
) -> None:
    output_dir = tmp_path / "invalid-gpu"

    with pytest.raises(ValueError, match="gpu_id"):
        run_olmoe_benchmark_smoke(
            output_dir,
            gpu_id=gpu_id,  # ty: ignore[invalid-argument-type]
        )

    assert not output_dir.exists()


def test_cli_rejects_negative_gpu_and_nonfrozen_dtype(tmp_path: Path) -> None:
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", str(tmp_path / "a"), "--gpu-id", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", str(tmp_path / "b"), "--dtype", "float16"])
