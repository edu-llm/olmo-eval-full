"""Production lifecycle for one OLMo-owned multi-mode evaluation run."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from olmo_eval.edullm.batch_resume import (
    build_resume_contract,
    exclusive_run_lock,
    validate_resume_contract,
    write_or_validate_resume_contract,
)
from olmo_eval.edullm.mode import (
    EduLLMAdaptiveMode,
    build_batch_resume_contract_payload,
    validate_batch_resume_artifacts,
)
from olmo_eval.inference.manager import InferenceManager
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.inference.registry import ProviderRegistry
from olmo_eval.runners.mode_config import ModeRunConfig
from olmo_eval.runners.modes import (
    MODE_RUN_SCHEMA_VERSION,
    EvaluationModeRegistry,
    EvaluationModeRunner,
    ModeResult,
    ModeRunContext,
    ModeSpec,
)
from olmo_eval.runners.standard_mode import AsyncEvalRunnerFactory, StandardOlmoMode

logger = logging.getLogger(__name__)

PRIMARY_PROVIDER_NAME = "candidate"
FROZEN_QWEN_VLLM_VERSION = "0.26.0"
QWEN_UNSET_ENVIRONMENT = ("VLLM_BATCH_INVARIANT",)
QWEN_VLLM_PYTHON_ENV = "EDULLM_QWEN_VLLM_PYTHON"


class InferenceManagerLike(Protocol):
    """Lifecycle subset used by the orchestrator and no-GPU tests."""

    def start(self) -> dict[str, list[dict[str, Any]]]: ...

    def shutdown(self) -> None: ...


InferenceManagerFactory = Callable[..., InferenceManagerLike]
RuntimeChecker = Callable[[ProviderConfig], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ModeRunPreflight:
    """Read-only preflight result used for display and run provenance."""

    selected_modes: tuple[str, ...]
    required_auxiliary_providers: tuple[str, ...]
    provider_names: tuple[str, ...]
    available_gpu_ids: tuple[int, ...]
    judge_runtime: Mapping[str, Any] | None = None
    resume_contract_payload: Mapping[str, Any] | None = None
    resume_contract_fingerprint: str | None = None


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _external_version_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/version", "", ""))


def _validate_external_qwen_runtime(provider: ProviderConfig) -> Mapping[str, Any]:
    assert provider.base_url is not None
    version_url = _external_version_url(provider.base_url)
    try:
        response = httpx.get(version_url, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(
            f"could not verify the external frozen-Qwen vLLM runtime at {version_url}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("external vLLM /version response must be a JSON object")
    version = payload.get("version")
    if version != FROZEN_QWEN_VLLM_VERSION:
        raise RuntimeError(
            "frozen Qwen requires vLLM "
            f"{FROZEN_QWEN_VLLM_VERSION}, external endpoint reported {version!r}"
        )
    return {
        "deployment": "external_endpoint",
        "version": version,
        "version_url": version_url,
        "explicit_token_logprobs": "required_by_frozen_request_contract",
    }


def _validate_local_qwen_runtime(provider: ProviderConfig) -> Mapping[str, Any]:
    del provider
    python_executable = (
        os.environ.get(QWEN_VLLM_PYTHON_ENV) or os.environ.get("VLLM_PYTHON") or sys.executable
    )
    executable = Path(python_executable).expanduser()
    if not executable.is_file():
        raise RuntimeError(
            "the frozen Qwen vLLM interpreter does not exist: "
            f"{python_executable!r}; set {QWEN_VLLM_PYTHON_ENV} to an isolated "
            "vLLM 0.26.0 environment"
        )
    probe = """
import json
from importlib.metadata import version
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
print(json.dumps({
    "version": version("vllm"),
    "chat_logprob_token_ids": "logprob_token_ids" in ChatCompletionRequest.model_fields,
    "completion_logprob_token_ids": "logprob_token_ids" in CompletionRequest.model_fields,
}))
""".strip()
    try:
        completed = subprocess.run(
            [str(executable), "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not inspect frozen-Qwen vLLM runtime: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "frozen Qwen requires an importable isolated vLLM 0.26.0 runtime; "
            f"probe failed: {detail[-2000:]}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("frozen-Qwen vLLM runtime probe returned invalid JSON") from exc
    version = payload.get("version")
    if version != FROZEN_QWEN_VLLM_VERSION:
        raise RuntimeError(
            "frozen Qwen requires vLLM "
            f"{FROZEN_QWEN_VLLM_VERSION}, interpreter reported {version!r}; "
            f"use {QWEN_VLLM_PYTHON_ENV} instead of upgrading OLMo's default vLLM"
        )
    if not payload.get("chat_logprob_token_ids") or not payload.get("completion_logprob_token_ids"):
        raise RuntimeError("frozen-Qwen vLLM lacks explicit logprob_token_ids support")
    return {
        "deployment": "managed_subprocess",
        "version": version,
        "python": str(executable.resolve()),
        "chat_logprob_token_ids": True,
        "completion_logprob_token_ids": True,
    }


def validate_qwen_vllm_runtime(provider: ProviderConfig) -> Mapping[str, Any]:
    """Verify the exact vLLM runtime required by the frozen judge contract."""

    if provider.base_url:
        return _validate_external_qwen_runtime(provider)
    return _validate_local_qwen_runtime(provider)


def _detect_gpu_ids(configs: Sequence[ProviderConfig]) -> tuple[int, ...]:
    if not any(config.requires_local_gpu for config in configs):
        return ()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        devices = tuple(value.strip() for value in visible.split(",") if value.strip())
        if devices:
            return tuple(range(len(devices)))

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "local model providers require visible GPUs, but torch is unavailable and "
            "CUDA_VISIBLE_DEVICES is not set"
        ) from exc
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("local model providers require at least one visible GPU")
    return tuple(range(count))


def _registry(
    resolved_primary: ProviderConfig | None = None,
    resolved_auxiliaries: Mapping[str, ProviderConfig] | None = None,
) -> EvaluationModeRegistry:
    registry = EvaluationModeRegistry()
    registry.register(
        StandardOlmoMode(
            AsyncEvalRunnerFactory(
                resolved_provider_config=resolved_primary,
                resolved_auxiliary_configs=dict(resolved_auxiliaries or {}),
            )
        )
    )
    registry.register(EduLLMAdaptiveMode())
    return registry


class ModeRunOrchestrator:
    """Start providers once, dispatch selected modes, and always clean up."""

    def __init__(
        self,
        config: ModeRunConfig,
        *,
        manager_factory: InferenceManagerFactory = InferenceManager,
        runtime_checker: RuntimeChecker = validate_qwen_vllm_runtime,
        available_gpu_ids: Sequence[int] | None = None,
        resume: bool = False,
    ) -> None:
        self.config = config
        self._manager_factory = manager_factory
        self._runtime_checker = runtime_checker
        self._available_gpu_ids = (
            tuple(available_gpu_ids) if available_gpu_ids is not None else None
        )
        if not isinstance(resume, bool):
            raise ValueError("resume must be boolean")
        self.resume = resume

    def _normalized_modes(self) -> tuple[ModeSpec, ...]:
        normalized: list[ModeSpec] = []
        for spec in self.config.modes:
            mode_config = dict(spec.config)
            if spec.name == "standard_olmo":
                experiment_name = mode_config.get("experiment_name")
                if experiment_name is not None and experiment_name != self.config.run_id:
                    raise ValueError("standard_olmo.experiment_name must equal the shared run_id")
                mode_config["experiment_name"] = self.config.run_id
            normalized.append(ModeSpec(spec.name, mode_config))
        return tuple(normalized)

    def _required_auxiliaries(self, modes: Sequence[ModeSpec]) -> tuple[str, ...]:
        registry = _registry()
        required: set[str] = set()
        for spec in modes:
            required.update(registry.get(spec.name).required_auxiliary_providers)
            if spec.name == "standard_olmo":
                # Native OLMo scorers may reference any auxiliary declared by
                # the shared harness.  Resolve those once at the parent level.
                required.update(self.config.harness.auxiliary_providers)
        return tuple(sorted(required))

    def _provider_configs(
        self,
        required_auxiliaries: Sequence[str],
    ) -> dict[str, ProviderConfig]:
        auxiliaries = self.config.harness.auxiliary_providers
        if PRIMARY_PROVIDER_NAME in auxiliaries:
            raise ValueError(
                f"{PRIMARY_PROVIDER_NAME!r} is reserved for the shared candidate provider"
            )
        missing = sorted(set(required_auxiliaries) - set(auxiliaries))
        if missing:
            raise ValueError(f"selected modes require undeclared provider(s): {missing}")

        configs = {PRIMARY_PROVIDER_NAME: self.config.harness.provider}
        configs.update({name: auxiliaries[name] for name in required_auxiliaries})
        for name, provider in configs.items():
            missing_secrets = provider.validate_secrets()
            if missing_secrets:
                raise ValueError(
                    f"provider {name!r} is missing required secret(s): {missing_secrets}"
                )
        return configs

    @staticmethod
    def _validate_output_root(output_root: Path, *, resume: bool) -> None:
        if resume:
            if not output_root.is_dir() or not any(output_root.iterdir()):
                raise ValueError("resume requires an existing non-empty mode output root")
            return
        if output_root.exists():
            if not output_root.is_dir():
                raise ValueError("mode output root must be a directory")
            if any(output_root.iterdir()):
                raise ValueError(
                    "mode output root must be new or empty; choose a new run directory"
                )

    def _batch_resume_payload(self, modes: Sequence[ModeSpec]) -> Mapping[str, Any] | None:
        adaptive = [mode for mode in modes if mode.name == "edullm_adaptive"]
        if len(adaptive) != 1:
            return None
        tutor = adaptive[0].config.get("tutor")
        if not isinstance(tutor, Mapping):
            return None
        source = tutor.get("response_source")
        if not isinstance(source, Mapping) or source.get("kind") != "precomputed_batch_jsonl":
            return None
        return build_batch_resume_contract_payload(
            adaptive[0].config,
            run_id=self.config.run_id,
            metadata=self.config.metadata,
            harness_config=self.config.harness.to_dict(),
            runtime_contract={
                "vllm_version": FROZEN_QWEN_VLLM_VERSION,
                "explicit_token_logprobs": True,
                "environment_unset": list(QWEN_UNSET_ENVIRONMENT),
            },
        )

    def preflight(self) -> ModeRunPreflight:
        """Perform read-only shared validation without starting a provider."""

        self._validate_output_root(self.config.output_dir, resume=self.resume)
        modes = self._normalized_modes()
        required = self._required_auxiliaries(modes)
        configs = self._provider_configs(required)
        mode_registry = _registry()
        for spec in modes:
            mode_registry.get(spec.name).preflight_config(spec.config)
        resume_payload = self._batch_resume_payload(modes)
        if self.resume and (
            resume_payload is None or len(modes) != 1 or modes[0].name != "edullm_adaptive"
        ):
            raise ValueError(
                "resume is supported only for a single edullm_adaptive precomputed_batch_jsonl mode"
            )
        resume_fingerprint: str | None = None
        if resume_payload is not None:
            resume_document = build_resume_contract(resume_payload)
            resume_fingerprint = str(resume_document["fingerprint_sha256"])
            if self.resume:
                validate_resume_contract(
                    self.config.output_dir / "resume_contract.json",
                    resume_payload,
                )
                validate_batch_resume_artifacts(
                    modes[0].config,
                    run_id=self.config.run_id,
                    mode_root=self.config.output_dir / "modes" / "edullm_adaptive",
                    fingerprint_sha256=resume_fingerprint,
                )
        gpu_ids = (
            self._available_gpu_ids
            if self._available_gpu_ids is not None
            else _detect_gpu_ids(tuple(configs.values()))
        )

        judge_runtime: Mapping[str, Any] | None = None
        if "edullm_adaptive" in {mode.name for mode in modes}:
            judge_runtime = dict(self._runtime_checker(configs["judge"]))

        return ModeRunPreflight(
            selected_modes=tuple(mode.name for mode in modes),
            required_auxiliary_providers=required,
            provider_names=tuple(configs),
            available_gpu_ids=tuple(gpu_ids),
            judge_runtime=judge_runtime,
            resume_contract_payload=resume_payload,
            resume_contract_fingerprint=resume_fingerprint,
        )

    def _runtime_provider_configs(
        self,
        preflight: ModeRunPreflight,
    ) -> dict[str, ProviderConfig]:
        configs = self._provider_configs(preflight.required_auxiliary_providers)
        judge = configs.get("judge")
        if judge is not None and judge.requires_local_gpu:
            kwargs = dict(judge.kwargs)
            kwargs["environment_unset"] = list(QWEN_UNSET_ENVIRONMENT)
            if preflight.judge_runtime is None:
                raise RuntimeError("managed Qwen runtime provenance is missing")
            python_executable = preflight.judge_runtime.get("python")
            if not isinstance(python_executable, str) or not python_executable:
                raise RuntimeError("managed Qwen runtime did not resolve a Python interpreter")
            kwargs["python_executable"] = python_executable
            configs["judge"] = replace(judge, kwargs=kwargs)
        return configs

    def _assert_batch_inputs_unchanged(
        self,
        preflight: ModeRunPreflight,
        modes: Sequence[ModeSpec],
    ) -> Mapping[str, Any] | None:
        """Re-hash mutable scientific inputs at the locked execution boundary."""

        observed = self._batch_resume_payload(modes)
        expected = preflight.resume_contract_payload
        if observed != expected:
            raise RuntimeError(
                "batch resume-contract inputs changed between preflight and execution; "
                "restart from a stable fitted bank and response batch"
            )
        if observed is not None:
            fingerprint = str(build_resume_contract(observed)["fingerprint_sha256"])
            if fingerprint != preflight.resume_contract_fingerprint:
                raise RuntimeError("batch resume-contract fingerprint changed during execution")
        return observed

    def _context_metadata(self, preflight: ModeRunPreflight) -> dict[str, Any]:
        metadata = dict(self.config.metadata)
        metadata["mode_runner"] = {
            "schema_version": self.config.schema_version,
            "candidate_model": self.config.harness.provider.model,
            "candidate_revision": self.config.harness.provider.revision,
            "candidate_provider": self.config.harness.provider.get_provider_name(),
            "provider_names": list(preflight.provider_names),
            "judge_runtime": dict(preflight.judge_runtime or {}),
            "resume_requested": self.resume,
            "resume_contract_fingerprint": preflight.resume_contract_fingerprint,
        }
        return metadata

    def _write_shared_failure(
        self,
        exc: BaseException,
        *,
        stage: str,
        preflight: ModeRunPreflight,
    ) -> None:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        manifest_path = self.config.output_dir / "manifest.json"
        report_path = self.config.output_dir / "report_index.json"

        manifest: dict[str, Any]
        if manifest_path.is_file():
            try:
                loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded_manifest = None
            manifest = dict(loaded_manifest) if isinstance(loaded_manifest, Mapping) else {}
        else:
            manifest = {}
        manifest.setdefault("schema_version", MODE_RUN_SCHEMA_VERSION)
        manifest.setdefault("run_id", self.config.run_id)
        manifest.setdefault("continue_on_mode_failure", self.config.continue_on_mode_failure)
        manifest.setdefault("selected_modes", list(self.config.mode_names))
        manifest.setdefault("completed_modes", [])
        manifest.setdefault("context_metadata", self._context_metadata(preflight))
        manifest.setdefault("results", [])
        manifest["status"] = status
        manifest["shared_error"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
        }

        report_index: dict[str, Any]
        if report_path.is_file():
            try:
                loaded_report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded_report = None
            report_index = dict(loaded_report) if isinstance(loaded_report, Mapping) else {}
        else:
            report_index = {}
        report_index.setdefault("schema_version", MODE_RUN_SCHEMA_VERSION)
        report_index.setdefault("run_id", self.config.run_id)
        report_index.setdefault("modes", {})
        report_index["status"] = status

        _atomic_write_json(manifest_path, manifest)
        _atomic_write_json(report_path, report_index)

    def _record_cleanup_errors(
        self,
        cleanup_errors: Sequence[tuple[str, Exception]],
    ) -> None:
        """Append structured cleanup provenance without replacing the primary failure."""

        manifest_path = self.config.output_dir / "manifest.json"
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("cannot record cleanup errors without a valid manifest") from exc
        if not isinstance(loaded, Mapping):
            raise RuntimeError("cannot record cleanup errors in a non-object manifest")
        manifest = dict(loaded)
        manifest["cleanup_errors"] = [
            {
                "stage": cleanup_stage,
                "type": type(cleanup_error).__name__,
                "message": str(cleanup_error),
            }
            for cleanup_stage, cleanup_error in cleanup_errors
        ]
        _atomic_write_json(manifest_path, manifest)

    async def run_async(self) -> dict[str, ModeResult]:
        """Execute every selected mode under one provider and artifact lifecycle."""

        preflight = self.preflight()
        modes = self._normalized_modes()
        provider_configs = self._runtime_provider_configs(preflight)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_run_lock(self.config.output_dir, self.config.run_id):
            locked_resume_payload = self._assert_batch_inputs_unchanged(preflight, modes)
            if locked_resume_payload is not None:
                contract = write_or_validate_resume_contract(
                    self.config.output_dir,
                    locked_resume_payload,
                    resume=self.resume,
                )
                fingerprint = str(contract["fingerprint_sha256"])
                if fingerprint != preflight.resume_contract_fingerprint:
                    raise RuntimeError("resume contract changed between preflight and execution")
                if self.resume:
                    validate_batch_resume_artifacts(
                        modes[0].config,
                        run_id=self.config.run_id,
                        mode_root=self.config.output_dir / "modes" / "edullm_adaptive",
                        fingerprint_sha256=fingerprint,
                    )
            elif self.resume:
                raise RuntimeError("resume contract is unavailable")
            return await self._run_with_providers(
                preflight,
                modes,
                provider_configs,
            )

    async def _run_with_providers(
        self,
        preflight: ModeRunPreflight,
        modes: Sequence[ModeSpec],
        provider_configs: Mapping[str, ProviderConfig],
    ) -> dict[str, ModeResult]:
        """Run providers and modes while the caller holds the exclusive run lock."""

        manager: InferenceManagerLike | None = None
        providers: ProviderRegistry | None = None
        results: dict[str, ModeResult] | None = None
        active_error: BaseException | None = None
        stage = "provider_manager_creation"
        try:
            manager = self._manager_factory(
                configs=provider_configs,
                available_gpu_ids=list(preflight.available_gpu_ids),
                log_dir=str(self.config.output_dir / "logs"),
            )
            stage = "provider_startup"
            serialized = manager.start()
            providers = ProviderRegistry.from_serialized(serialized)
            if providers is None:
                raise RuntimeError("inference manager returned no resolved providers")
            self._assert_batch_inputs_unchanged(preflight, modes)
            resolved_primary = providers.get_replica_set(PRIMARY_PROVIDER_NAME).get_config(0)
            resolved_auxiliaries = {
                name: providers.get_replica_set(name).get_config(0)
                for name in preflight.required_auxiliary_providers
            }
            context = ModeRunContext(
                run_id=self.config.run_id,
                output_root=self.config.output_dir,
                provider=providers.get(PRIMARY_PROVIDER_NAME),
                inference_pool=providers,
                metadata=self._context_metadata(preflight),
                resume=self.resume,
                resume_contract_fingerprint=preflight.resume_contract_fingerprint,
            )

            def validate_inputs_after_mode_preflight() -> None:
                self._assert_batch_inputs_unchanged(preflight, modes)

            stage = "mode_dispatch"
            results = await EvaluationModeRunner(
                registry=_registry(resolved_primary, resolved_auxiliaries),
                context=context,
                modes=modes,
                continue_on_mode_failure=self.config.continue_on_mode_failure,
                pre_dispatch_validation=validate_inputs_after_mode_preflight,
            ).run_async()
        except BaseException as exc:
            active_error = exc
            self._write_shared_failure(exc, stage=stage, preflight=preflight)
            raise
        finally:
            cleanup_errors: list[tuple[str, Exception]] = []
            if providers is not None:
                try:
                    await providers.aclose()
                except Exception as exc:
                    cleanup_errors.append(("provider_registry_close", exc))
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception as exc:
                    cleanup_errors.append(("inference_manager_shutdown", exc))
            if cleanup_errors:
                message = "; ".join(
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                    for _, cleanup_error in cleanup_errors
                )
                cleanup_failure = RuntimeError(f"provider cleanup failed: {message}")
                if active_error is None:
                    self._write_shared_failure(
                        cleanup_failure,
                        stage="provider_cleanup",
                        preflight=preflight,
                    )
                    self._record_cleanup_errors(cleanup_errors)
                    raise cleanup_failure
                self._record_cleanup_errors(cleanup_errors)
                for _, cleanup_error in cleanup_errors:
                    logger.error("mode-run cleanup failed after run failure: %s", cleanup_error)

        if results is None:
            raise RuntimeError("mode dispatcher completed without returning results")
        return results

    def run(self) -> dict[str, ModeResult]:
        """Synchronous entry point used by the CLI."""

        return asyncio.run(self.run_async())


__all__ = [
    "FROZEN_QWEN_VLLM_VERSION",
    "ModeRunOrchestrator",
    "ModeRunPreflight",
    "PRIMARY_PROVIDER_NAME",
    "QWEN_UNSET_ENVIRONMENT",
    "QWEN_VLLM_PYTHON_ENV",
    "validate_qwen_vllm_runtime",
]
