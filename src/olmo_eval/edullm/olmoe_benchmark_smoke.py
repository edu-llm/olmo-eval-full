"""Native OLMo Eval benchmark smoke for the pinned OLMoE checkpoint.

This module deliberately delegates benchmark preparation, request formatting,
inference, scoring, and artifact writing to OLMo Eval's ``standard_olmo`` mode.
Its only custom behavior is freezing a tiny run contract and validating the
native artifacts after the managed provider has shut down.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shlex
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from olmo_eval.runners.mode_config import (
    MODE_CONFIG_SCHEMA_VERSION,
    ModeRunConfig,
    parse_mapping,
)
from olmo_eval.runners.mode_orchestrator import ModeRunOrchestrator
from olmo_eval.runners.modes import ModeResult

SCHEMA_VERSION = "edullm-olmoe-benchmark-smoke-v1"
MODE_NAME = "standard_olmo"
MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
MODEL_REVISION = "7f1c97f440f06ce36705e4f2b843edb5925f4498"
MODEL_ARCHITECTURE = "OlmoeForCausalLM"
DTYPE = "bfloat16"
MAX_MODEL_LEN = 4096
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.8

BENCHMARK_NAME = "GSM8K"
BENCHMARK_TASK = "gsm8k"
BENCHMARK_DATASET = "openai/gsm8k"
BENCHMARK_DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
EXPECTED_INSTANCES = 3
TASK_SEED = 42
SHUFFLE_SEED = 42
MAX_TOKENS = 128

TASK_OVERRIDES: Mapping[str, Any] = {
    "limit": EXPECTED_INSTANCES,
    "seed": TASK_SEED,
    "max_tokens": MAX_TOKENS,
    "temperature": 0.0,
    "top_p": 1.0,
    "num_samples": 1,
    "do_sample": False,
}


class Orchestrator(Protocol):
    """Subset of the native mode orchestrator used by this handoff."""

    def run(self) -> Mapping[str, ModeResult]: ...


OrchestratorFactory = Callable[..., Orchestrator]
GPUChecker = Callable[[int], Mapping[str, Any]]
ModelChecker = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class BenchmarkSmokeResult:
    """Paths and validation record for a successful benchmark smoke."""

    output_dir: Path
    manifest_path: Path
    metrics_path: Path
    predictions_path: Path
    requests_path: Path
    validation_path: Path
    validation: Mapping[str, Any]


class BenchmarkValidationError(RuntimeError):
    """Raised after a failed validation document has been persisted."""


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise BenchmarkValidationError(f"required artifact is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkValidationError(f"artifact is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"JSON artifact must contain an object: {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise BenchmarkValidationError(f"required artifact is missing or unsafe: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkValidationError(f"could not read artifact: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise BenchmarkValidationError(f"{path} has a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkValidationError(f"{path} line {line_number} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise BenchmarkValidationError(f"{path} line {line_number} must contain an object")
        rows.append(value)
    return rows


def _one_artifact(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise BenchmarkValidationError(
            f"expected exactly one {pattern!r} artifact under {root}, found {len(matches)}"
        )
    if matches[0].is_symlink():
        raise BenchmarkValidationError(f"artifact cannot be a symlink: {matches[0]}")
    return matches[0]


def _provider_mapping() -> dict[str, Any]:
    return {
        "kind": "vllm_server",
        "model": MODEL_ID,
        "tokenizer": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": DTYPE,
        "max_model_len": MAX_MODEL_LEN,
        "num_instances": 1,
        "kwargs": {
            "tokenizer_revision": MODEL_REVISION,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "enable_prefix_caching": True,
        },
    }


def _validate_output_root(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError("benchmark smoke output path exists and is not a directory")
    if path.is_dir() and any(path.iterdir()):
        raise ValueError("benchmark smoke output directory must be new or empty")


def inspect_bf16_gpu(gpu_id: int) -> Mapping[str, Any]:
    """Resolve one scheduler-visible GPU and reject hardware without BF16 support."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the GPU smoke requires an environment with torch installed") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("the GPU smoke requires at least one CUDA-visible GPU")
    visible_count = torch.cuda.device_count()
    if gpu_id >= visible_count:
        raise ValueError(
            f"gpu_id {gpu_id} is outside the scheduler-visible GPU set (count={visible_count})"
        )
    with torch.cuda.device(gpu_id):
        name = torch.cuda.get_device_name(gpu_id)
        capability = torch.cuda.get_device_capability(gpu_id)
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    if not bf16_supported:
        raise RuntimeError(
            f"GPU {gpu_id} ({name}) does not support the frozen bfloat16 model contract"
        )
    return {
        "selected_visible_gpu_id": gpu_id,
        "visible_gpu_count": visible_count,
        "name": name,
        "compute_capability": [int(capability[0]), int(capability[1])],
        "bfloat16_supported": True,
        "scheduler_mask_present": bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()),
    }


def inspect_pinned_model_config() -> Mapping[str, Any]:
    """Fetch the pinned HF config and prove that it is the intended OLMoE architecture."""

    from huggingface_hub import hf_hub_download

    try:
        config_path = Path(
            hf_hub_download(
                MODEL_ID,
                "config.json",
                revision=MODEL_REVISION,
            )
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("could not load the pinned OLMoE config.json") from exc
    if not isinstance(config, Mapping):
        raise RuntimeError("the pinned OLMoE config.json is not a JSON object")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or MODEL_ARCHITECTURE not in architectures:
        raise RuntimeError(
            f"pinned model config does not declare architecture {MODEL_ARCHITECTURE!r}"
        )
    if config.get("model_type") != "olmoe":
        raise RuntimeError("pinned model config does not declare model_type='olmoe'")
    if config.get("torch_dtype") != DTYPE:
        raise RuntimeError(f"pinned model config does not declare torch_dtype={DTYPE!r}")
    return {
        "source": "huggingface_config_json",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "architectures": list(architectures),
        "model_type": config["model_type"],
        "torch_dtype": config["torch_dtype"],
    }


def _write_preflight_artifact(
    root: Path,
    *,
    gpu: Mapping[str, Any] | None,
    model: Mapping[str, Any] | None,
) -> None:
    _atomic_write_json(
        root / "olmoe_preflight.json",
        {
            "schema_version": "edullm-olmoe-benchmark-preflight-v1",
            "gpu": dict(gpu) if gpu is not None else None,
            "model": dict(model) if model is not None else None,
        },
    )


def _write_runtime_failure(
    root: Path,
    *,
    run_id: str,
    command: str,
    stage: str,
    exc: BaseException,
    gpu: Mapping[str, Any] | None,
    model: Mapping[str, Any] | None,
) -> None:
    """Best-effort failure forensics after ownership of an empty output root."""

    _atomic_write_text(root / "reproduction_command.txt", command + "\n")
    _write_preflight_artifact(root, gpu=gpu, model=model)
    gates = {
        "native_run_succeeded": False,
        "model_architecture_verified": model is not None,
        "bf16_gpu_verified": gpu is not None,
        "frozen_provider_recorded": False,
        "benchmark_metrics_complete": False,
        "exact_native_request_count": False,
        "deterministic_request_settings": False,
        "exact_native_prediction_count": False,
        "all_outputs_nonblank": False,
        "request_prediction_ids_match": False,
        "managed_server_logs_captured": False,
    }
    _atomic_write_json(
        root / "olmoe_benchmark_validation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "run_id": run_id,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "expected_architecture": MODEL_ARCHITECTURE,
            },
            "benchmark": {
                "name": BENCHMARK_NAME,
                "task_spec": BENCHMARK_TASK,
                "instances": EXPECTED_INSTANCES,
            },
            "gates": gates,
            "artifacts": {
                "preflight": "olmoe_preflight.json",
                "reproduction_command": "reproduction_command.txt",
            },
            "error": {
                "stage": stage,
                "type": type(exc).__name__,
                "message": str(exc),
            },
        },
    )


def build_mode_run_config(
    output_dir: str | Path,
    *,
    run_id: str = "olmoe_gsm8k_smoke",
    dtype: str = DTYPE,
) -> ModeRunConfig:
    """Build and strictly validate the frozen native benchmark configuration."""

    if dtype != DTYPE:
        raise ValueError(f"OLMoE benchmark smoke dtype is frozen to {DTYPE!r}")
    root = Path(output_dir).expanduser().resolve()
    provider = _provider_mapping()
    provider["dtype"] = dtype
    return parse_mapping(
        {
            "schema_version": MODE_CONFIG_SCHEMA_VERSION,
            "run_id": run_id,
            "output_dir": str(root),
            "continue_on_mode_failure": False,
            "metadata": {
                "purpose": "olmoe_native_benchmark_generation_smoke",
                "benchmark": BENCHMARK_NAME,
                "benchmark_task": BENCHMARK_TASK,
                "benchmark_dataset": BENCHMARK_DATASET,
                "benchmark_dataset_revision_observed_at_handoff": (BENCHMARK_DATASET_REVISION),
                "expected_instances": EXPECTED_INSTANCES,
                "score_is_acceptance_gate": False,
                "expected_model_architecture": MODEL_ARCHITECTURE,
            },
            "harness": {
                "name": "olmoe-benchmark-smoke",
                "provider": provider,
            },
            "modes": [
                {
                    "name": MODE_NAME,
                    "config": {
                        "task_specs": [BENCHMARK_TASK],
                        "task_overrides": {
                            BENCHMARK_TASK: dict(TASK_OVERRIDES),
                        },
                        "save_predictions": True,
                        "save_requests": True,
                        "shuffle_seed": SHUFFLE_SEED,
                    },
                }
            ],
        }
    )


def _validate_manifest(manifest: Mapping[str, Any], run_id: str) -> None:
    if manifest.get("run_id") != run_id:
        raise BenchmarkValidationError("shared manifest run_id does not match the smoke config")
    if manifest.get("status") != "succeeded":
        raise BenchmarkValidationError("native OLMo run did not finish with status=succeeded")
    if manifest.get("selected_modes") != [MODE_NAME]:
        raise BenchmarkValidationError("shared manifest selected an unexpected mode set")
    if manifest.get("completed_modes") != [MODE_NAME]:
        raise BenchmarkValidationError("standard_olmo did not complete")
    metadata = manifest.get("context_metadata")
    if not isinstance(metadata, Mapping):
        raise BenchmarkValidationError("shared manifest context_metadata is missing")
    mode_runner = metadata.get("mode_runner")
    if not isinstance(mode_runner, Mapping):
        raise BenchmarkValidationError("shared manifest mode_runner provenance is missing")
    expected = {
        "candidate_model": MODEL_ID,
        "candidate_revision": MODEL_REVISION,
        "candidate_provider": "vllm_server",
    }
    for field, value in expected.items():
        if mode_runner.get(field) != value:
            raise BenchmarkValidationError(
                f"shared manifest {field} does not match the frozen contract"
            )


def _validate_metrics(metrics: Mapping[str, Any]) -> None:
    errors = metrics.get("errors")
    if errors not in (None, []):
        raise BenchmarkValidationError("native metrics contain task errors")
    tasks = metrics.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise BenchmarkValidationError("native metrics must contain exactly one task")
    task = tasks[0]
    if not isinstance(task, Mapping):
        raise BenchmarkValidationError("native metrics task entry is malformed")
    if task.get("task") != BENCHMARK_TASK:
        raise BenchmarkValidationError("native metrics recorded an unexpected benchmark task")
    if task.get("num_instances") != EXPECTED_INSTANCES:
        raise BenchmarkValidationError(
            f"native metrics did not record exactly {EXPECTED_INSTANCES} instances"
        )
    task_metrics = task.get("metrics")
    accuracy = task_metrics.get("accuracy") if isinstance(task_metrics, Mapping) else None
    exact_match = accuracy.get("exact_match") if isinstance(accuracy, Mapping) else None
    if (
        isinstance(exact_match, bool)
        or not isinstance(exact_match, (int, float))
        or not math.isfinite(float(exact_match))
        or not 0.0 <= float(exact_match) <= 1.0
    ):
        raise BenchmarkValidationError("native metrics did not record GSM8K exact-match accuracy")
    if task.get("primary_metric") != "accuracy:exact_match":
        raise BenchmarkValidationError("native metrics recorded the wrong GSM8K primary metric")
    summary = metrics.get("summary")
    task_summary = summary.get(BENCHMARK_TASK) if isinstance(summary, Mapping) else None
    if not isinstance(task_summary, Mapping):
        raise BenchmarkValidationError("native GSM8K score summary is missing")
    if (
        task_summary.get("metric") != "accuracy:exact_match"
        or task_summary.get("score") != exact_match
    ):
        raise BenchmarkValidationError("native GSM8K score summary contradicts task metrics")
    config = metrics.get("config")
    if not isinstance(config, Mapping):
        raise BenchmarkValidationError("native metrics provider config is missing")
    provider = config.get("provider")
    if not isinstance(provider, Mapping):
        raise BenchmarkValidationError("native metrics provider provenance is missing")
    frozen_provider = _provider_mapping()
    for field in ("kind", "model", "tokenizer", "revision", "dtype", "max_model_len"):
        if provider.get(field) != frozen_provider[field]:
            raise BenchmarkValidationError(
                f"native metrics provider.{field} does not match the frozen contract"
            )
    kwargs = provider.get("kwargs")
    if not isinstance(kwargs, Mapping):
        raise BenchmarkValidationError("native metrics provider kwargs are missing")
    for field, value in frozen_provider["kwargs"].items():
        if kwargs.get(field) != value:
            raise BenchmarkValidationError(
                f"native metrics provider kwargs.{field} does not match the frozen contract"
            )


def _validate_preflight(preflight: Mapping[str, Any]) -> None:
    if preflight.get("schema_version") != "edullm-olmoe-benchmark-preflight-v1":
        raise BenchmarkValidationError("OLMoE preflight schema is unsupported")
    gpu = preflight.get("gpu")
    if not isinstance(gpu, Mapping) or gpu.get("bfloat16_supported") is not True:
        raise BenchmarkValidationError("preflight did not verify a BF16-capable GPU")
    selected_gpu = gpu.get("selected_visible_gpu_id")
    visible_count = gpu.get("visible_gpu_count")
    if (
        isinstance(selected_gpu, bool)
        or not isinstance(selected_gpu, int)
        or isinstance(visible_count, bool)
        or not isinstance(visible_count, int)
        or selected_gpu < 0
        or selected_gpu >= visible_count
    ):
        raise BenchmarkValidationError("preflight recorded an invalid visible GPU selection")

    model = preflight.get("model")
    if not isinstance(model, Mapping):
        raise BenchmarkValidationError("preflight model-config evidence is missing")
    if model.get("model") != MODEL_ID or model.get("revision") != MODEL_REVISION:
        raise BenchmarkValidationError("preflight inspected the wrong model revision")
    architectures = model.get("architectures")
    if not isinstance(architectures, list) or MODEL_ARCHITECTURE not in architectures:
        raise BenchmarkValidationError("preflight did not verify the OLMoE architecture")
    if model.get("model_type") != "olmoe" or model.get("torch_dtype") != DTYPE:
        raise BenchmarkValidationError("preflight model type or dtype is not the frozen contract")


def _validate_requests(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    if len(rows) != EXPECTED_INSTANCES:
        raise BenchmarkValidationError(
            f"expected {EXPECTED_INSTANCES} native request rows, found {len(rows)}"
        )
    native_ids: list[Any] = []
    for index, row in enumerate(rows):
        if row.get("task_name") != BENCHMARK_TASK:
            raise BenchmarkValidationError(f"request row {index} has the wrong task_name")
        if row.get("request_type") != "generate_until":
            raise BenchmarkValidationError(f"request row {index} is not native generation")
        request = row.get("request")
        if not isinstance(request, Mapping):
            raise BenchmarkValidationError(f"request row {index} has no request object")
        context = request.get("context")
        if not isinstance(context, str) or not context.strip():
            raise BenchmarkValidationError(f"request row {index} has a blank native prompt")
        generation = request.get("generation_kwargs")
        expected_generation = {
            "max_gen_toks": MAX_TOKENS,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        if not isinstance(generation, Mapping) or any(
            generation.get(field) != value for field, value in expected_generation.items()
        ):
            raise BenchmarkValidationError(
                f"request row {index} generation settings do not match the frozen contract"
            )
        native_ids.append(row.get("native_id"))
    if len(set(native_ids)) != EXPECTED_INSTANCES:
        raise BenchmarkValidationError("native request IDs are missing or duplicated")
    return native_ids


def _validate_predictions(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    if len(rows) != EXPECTED_INSTANCES:
        raise BenchmarkValidationError(
            f"expected {EXPECTED_INSTANCES} native prediction rows, found {len(rows)}"
        )
    native_ids: list[Any] = []
    for index, row in enumerate(rows):
        final_output = row.get("final_output")
        if not isinstance(final_output, str) or not final_output.strip():
            raise BenchmarkValidationError(f"prediction row {index} has a blank final_output")
        model_output = row.get("model_output")
        if not isinstance(model_output, list) or len(model_output) != 1:
            raise BenchmarkValidationError(
                f"prediction row {index} must contain exactly one model output"
            )
        raw = model_output[0]
        if not isinstance(raw, Mapping):
            raise BenchmarkValidationError(f"prediction row {index} model output is malformed")
        raw_text = raw.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise BenchmarkValidationError(f"prediction row {index} raw output is blank")
        if raw_text != final_output:
            raise BenchmarkValidationError(
                f"prediction row {index} raw output and final_output disagree"
            )
        native_ids.append(row.get("native_id"))
    if len(set(native_ids)) != EXPECTED_INSTANCES:
        raise BenchmarkValidationError("native prediction IDs are missing or duplicated")
    return native_ids


def validate_benchmark_artifacts(
    output_dir: str | Path,
    *,
    run_id: str = "olmoe_gsm8k_smoke",
) -> BenchmarkSmokeResult:
    """Validate native OLMo artifacts and always write a validation document."""

    root = Path(output_dir).expanduser().resolve()
    validation_path = root / "olmoe_benchmark_validation.json"
    manifest_path = root / "manifest.json"
    mode_root = root / "modes" / MODE_NAME
    metrics_path = mode_root / "metrics.json"

    gates = {
        "native_run_succeeded": False,
        "model_architecture_verified": False,
        "bf16_gpu_verified": False,
        "frozen_provider_recorded": False,
        "benchmark_metrics_complete": False,
        "exact_native_request_count": False,
        "deterministic_request_settings": False,
        "exact_native_prediction_count": False,
        "all_outputs_nonblank": False,
        "request_prediction_ids_match": False,
        "managed_server_logs_captured": False,
    }
    paths: dict[str, str | None] = {
        "manifest": "manifest.json",
        "preflight": "olmoe_preflight.json",
        "mode_result": f"modes/{MODE_NAME}/mode_result.json",
        "metrics": f"modes/{MODE_NAME}/metrics.json",
        "predictions": None,
        "requests": None,
        "logs": "logs/",
        "reproduction_command": "reproduction_command.txt",
    }
    error: dict[str, str] | None = None
    predictions_path = mode_root / "predictions"
    requests_path = mode_root / "requests"
    try:
        manifest = _read_json_object(manifest_path)
        _validate_manifest(manifest, run_id)
        mode_result = _read_json_object(mode_root / "mode_result.json")
        if mode_result.get("mode") != MODE_NAME or mode_result.get("status") != "succeeded":
            raise BenchmarkValidationError("standard_olmo mode_result is not successful")
        gates["native_run_succeeded"] = True

        preflight = _read_json_object(root / "olmoe_preflight.json")
        _validate_preflight(preflight)
        gates["model_architecture_verified"] = True
        gates["bf16_gpu_verified"] = True

        metrics = _read_json_object(metrics_path)
        _validate_metrics(metrics)
        gates["frozen_provider_recorded"] = True
        gates["benchmark_metrics_complete"] = True

        predictions_path = _one_artifact(mode_root / "predictions", "*-predictions.jsonl")
        requests_path = _one_artifact(mode_root / "requests", "*-requests.jsonl")
        paths["predictions"] = predictions_path.relative_to(root).as_posix()
        paths["requests"] = requests_path.relative_to(root).as_posix()

        request_rows = _read_jsonl_objects(requests_path)
        request_ids = _validate_requests(request_rows)
        gates["exact_native_request_count"] = True
        gates["deterministic_request_settings"] = True

        prediction_rows = _read_jsonl_objects(predictions_path)
        prediction_ids = _validate_predictions(prediction_rows)
        gates["exact_native_prediction_count"] = True
        gates["all_outputs_nonblank"] = True
        if request_ids != prediction_ids:
            raise BenchmarkValidationError("native request and prediction IDs do not match")
        gates["request_prediction_ids_match"] = True

        log_root = root / "logs"
        log_files = (
            [path for path in log_root.rglob("*") if path.is_file() and path.stat().st_size > 0]
            if log_root.is_dir()
            else []
        )
        if not log_files:
            raise BenchmarkValidationError("no nonempty managed vLLM server log was captured")
        gates["managed_server_logs_captured"] = True
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}

    validation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "run_id": run_id,
        "scope": {
            "native_olmo_benchmark": True,
            "benchmark_score_recorded": True,
            "benchmark_score_is_acceptance_gate": False,
            "judge": False,
            "calibration": False,
            "cat": False,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "expected_architecture": MODEL_ARCHITECTURE,
        },
        "benchmark": {
            "name": BENCHMARK_NAME,
            "task_spec": BENCHMARK_TASK,
            "dataset": BENCHMARK_DATASET,
            "dataset_revision_observed_at_handoff": BENCHMARK_DATASET_REVISION,
            "instances": EXPECTED_INSTANCES,
            "task_seed": TASK_SEED,
        },
        "provider": _provider_mapping(),
        "task_overrides": dict(TASK_OVERRIDES),
        "gates": gates,
        "artifacts": paths,
        "error": error,
    }
    _atomic_write_json(validation_path, validation)
    if validation["status"] != "passed":
        message = error["message"] if error is not None else "one or more gates failed"
        raise BenchmarkValidationError(f"OLMoE benchmark smoke validation failed: {message}")

    return BenchmarkSmokeResult(
        output_dir=root,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        requests_path=requests_path,
        validation_path=validation_path,
        validation=validation,
    )


def run_olmoe_benchmark_smoke(
    output_dir: str | Path,
    *,
    gpu_id: int = 0,
    dtype: str = DTYPE,
    run_id: str = "olmoe_gsm8k_smoke",
    reproduction_command: str | None = None,
    orchestrator_factory: OrchestratorFactory = ModeRunOrchestrator,
    gpu_checker: GPUChecker = inspect_bf16_gpu,
    model_checker: ModelChecker = inspect_pinned_model_config,
) -> BenchmarkSmokeResult:
    """Run the native benchmark and gate its stored outputs after cleanup."""

    if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
        raise ValueError("gpu_id must be a non-negative integer")
    config = build_mode_run_config(output_dir, run_id=run_id, dtype=dtype)
    _validate_output_root(config.output_dir)
    command = reproduction_command or shlex.join(
        [
            "uv",
            "run",
            "python",
            "scripts/edullm/run_olmoe_benchmark_smoke.py",
            "--output-dir",
            str(config.output_dir),
            "--gpu-id",
            str(gpu_id),
            "--dtype",
            dtype,
            "--run-id",
            run_id,
        ]
    )
    gpu_info: Mapping[str, Any] | None = None
    model_info: Mapping[str, Any] | None = None
    stage = "gpu_preflight"
    try:
        gpu_info = dict(gpu_checker(gpu_id))
        stage = "model_config_preflight"
        model_info = dict(model_checker())
        stage = "native_olmo_run"
        orchestrator = orchestrator_factory(config, available_gpu_ids=(gpu_id,))
        results = orchestrator.run()
        result = results.get(MODE_NAME)
        if result is None or not result.succeeded:
            detail = result.error if result is not None else "standard_olmo returned no result"
            raise RuntimeError(f"native OLMo benchmark run failed: {detail}")
    except BaseException as exc:
        with contextlib.suppress(Exception):
            _write_runtime_failure(
                config.output_dir,
                run_id=run_id,
                command=command,
                stage=stage,
                exc=exc,
                gpu=gpu_info,
                model=model_info,
            )
        raise

    _atomic_write_text(config.output_dir / "reproduction_command.txt", command + "\n")
    _write_preflight_artifact(config.output_dir, gpu=gpu_info, model=model_info)
    return validate_benchmark_artifacts(config.output_dir, run_id=run_id)


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser separately for GPU-free contract tests."""

    parser = argparse.ArgumentParser(
        description=(
            "Load the pinned OLMoE checkpoint and run three GSM8K cases through "
            "OLMo Eval's native standard benchmark runner."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New or empty directory for native OLMo artifacts and vLLM logs.",
    )
    parser.add_argument(
        "--gpu-id",
        type=_nonnegative_int,
        default=0,
        help="GPU index inside the scheduler-visible CUDA device set (default: 0).",
    )
    parser.add_argument(
        "--dtype",
        choices=(DTYPE,),
        default=DTYPE,
        help="Frozen model dtype.",
    )
    parser.add_argument(
        "--run-id",
        default="olmoe_gsm8k_smoke",
        help="Run identifier stored in native OLMo manifests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the GPU handoff."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    command = shlex.join(
        [
            "uv",
            "run",
            "python",
            "scripts/edullm/run_olmoe_benchmark_smoke.py",
            *effective_argv,
        ]
    )
    try:
        result = run_olmoe_benchmark_smoke(
            args.output_dir,
            gpu_id=args.gpu_id,
            dtype=args.dtype,
            run_id=args.run_id,
            reproduction_command=command,
        )
    except Exception as exc:
        print(f"OLMoE benchmark smoke FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"OLMoE benchmark smoke PASSED: {result.validation_path}")
    return 0


__all__ = [
    "BENCHMARK_DATASET",
    "BENCHMARK_DATASET_REVISION",
    "BENCHMARK_NAME",
    "BENCHMARK_TASK",
    "BenchmarkSmokeResult",
    "BenchmarkValidationError",
    "DTYPE",
    "EXPECTED_INSTANCES",
    "GPU_MEMORY_UTILIZATION",
    "MAX_MODEL_LEN",
    "MAX_TOKENS",
    "MODEL_ARCHITECTURE",
    "MODEL_ID",
    "MODEL_REVISION",
    "SCHEMA_VERSION",
    "TASK_OVERRIDES",
    "TASK_SEED",
    "TENSOR_PARALLEL_SIZE",
    "build_argument_parser",
    "build_mode_run_config",
    "inspect_bf16_gpu",
    "inspect_pinned_model_config",
    "main",
    "run_olmoe_benchmark_smoke",
    "validate_benchmark_artifacts",
]
