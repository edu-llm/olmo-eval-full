"""Mock-provider integration tests for the OLMo-owned EduLLM mode."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

import olmo_eval.edullm.judge as judge_module
import olmo_eval.edullm.mode as mode_module
from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.edullm.bank import FITTED_BANK_SCHEMA_VERSION
from olmo_eval.edullm.judge import (
    ADAPTER_VERSION,
    FAILURE_PROBABILITY_THRESHOLD,
    PROMPT_VERSION,
    QWEN_JUDGE_MODEL,
    QWEN_JUDGE_REVISION,
)
from olmo_eval.edullm.mode import (
    ATOMIC_REQUIREMENT_POLICY,
    MODE_CONFIG_SCHEMA_VERSION,
    EduLLMAdaptiveMode,
)
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.runners.modes import ModeRunContext, ModeStatus


class _Provider(InferenceProvider):
    def __init__(self, model_name: str, handler: Callable[[LMRequest], LMOutput]) -> None:
        super().__init__(model_name)
        self.handler = handler
        self.requests: list[LMRequest] = []

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        self.requests.extend(requests)
        return [[self.handler(request)] for request in requests]

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
        raise AssertionError("the frozen judge uses generated-token alternatives")


class _Lookup:
    def __init__(self, judge: InferenceProvider) -> None:
        self.judge = judge

    @property
    def names(self) -> list[str]:
        return ["judge"]

    def get(self, name: str) -> InferenceProvider:
        assert name == "judge"
        return self.judge


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha(payload.encode())


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rows)


def _write_bank(
    root: Path,
    *,
    scenario_ids: tuple[str, ...] = ("mid", "easy", "hard"),
    reviewed_checklist: bool = False,
    checklist_status: str = "curation_v1_finalized",
    criteria_per_scenario: int = 1,
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    difficulties = {"mid": 0.0, "easy": -2.0, "hard": 2.0}
    rubrics: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        criterion_ids: list[str] = []
        for criterion_index in range(1, criteria_per_scenario + 1):
            criterion_id = f"{scenario_id}_c{criterion_index:02d}"
            criterion_ids.append(criterion_id)
            record: dict[str, Any] = {
                "criterion_id": criterion_id,
                "scenario_id": scenario_id,
                "criterion": f"Requirement for {criterion_id}",
                "expected_evidence": (
                    [f"Expected evidence for {criterion_id}"] if criteria_per_scenario > 1 else []
                ),
                "q_modeled": {"instruction_following": 1},
                "discrimination": {"instruction_following": 2.0},
                "difficulty": difficulties.get(scenario_id, 0.0),
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
            if reviewed_checklist:
                requirements = [
                    f"First reviewed requirement for {criterion_id}",
                    "Second reviewed requirement",
                ]
                record["atomic_checklist"] = {
                    "requirements": requirements,
                    "review_status": checklist_status,
                    "source": "fixture-review-sheet",
                    "sha256": _canonical_hash(requirements),
                }
            rubrics.append(record)
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "prompt": f"Prompt for {scenario_id}",
                "criterion_ids": criterion_ids,
                "benchmark": "FixtureBench",
            }
        )
    rubric_bytes = _jsonl(rubrics)
    scenario_bytes = _jsonl(scenarios)
    rubric_path = root / "rubrics.jsonl"
    scenario_path = root / "scenarios.jsonl"
    manifest_path = root / "bank_manifest.json"
    rubric_path.write_bytes(rubric_bytes)
    scenario_path.write_bytes(scenario_bytes)
    manifest = {
        "schema_version": FITTED_BANK_SCHEMA_VERSION,
        "skills_order": ["instruction_following"],
        "latent_correlation": [[1.0]],
        "counts": {
            "exported_criteria": len(rubrics),
            "exported_scenarios": len(scenarios),
        },
        "outputs": {
            "sha256": {
                "rubrics": _sha(rubric_bytes),
                "scenarios": _sha(scenario_bytes),
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
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return rubric_path, scenario_path, manifest_path


def _config(
    bank_paths: tuple[Path, Path, Path],
    *,
    tutor_model: str = "fixture-tutor",
    max_scenarios: int = 2,
) -> dict[str, Any]:
    rubrics, scenarios, manifest = bank_paths
    return {
        "schema_version": MODE_CONFIG_SCHEMA_VERSION,
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
            "expected_model": tutor_model,
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
            "nodes_per_dim": 21,
            "max_nodes": 100,
            "method": "gauss_hermite",
            "linear_bound": 8.0,
        },
        "cat": {
            "max_se": 0.01,
            "min_evals_per_skill": 0,
            "min_scenarios": 0,
            "max_scenarios": max_scenarios,
            "seed": 7,
            "top_n": 1,
            "selection": "trace",
            "stop_se_method": "eap",
            "theta_init": None,
            "covariance_init_diag": None,
            "mwle_ridge": 1e-6,
        },
    }


def _use_precomputed_responses(
    config: dict[str, Any],
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    payload = _jsonl(rows)
    path.write_bytes(payload)
    config["tutor"]["generation"] = None
    config["tutor"]["model_provenance"] = {
        "source": "uploaded_response_archive",
        "revision": "uploaded-revision",
    }
    config["tutor"]["response_source"] = {
        "kind": "precomputed_jsonl",
        "schema_version": "edullm-precomputed-tutor-responses-v1",
        "path": str(path),
        "sha256": _sha(payload),
    }


def _use_precomputed_response_batch(
    config: dict[str, Any],
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    payload = _jsonl(rows)
    path.write_bytes(payload)
    config["tutor"] = {
        "generation": None,
        "response_source": {
            "kind": "precomputed_batch_jsonl",
            "schema_version": "edullm-precomputed-tutor-response-batch-v1",
            "path": str(path),
            "sha256": _sha(payload),
            "provenance": {
                "source": "uploaded_batch_archive",
                "revision": "uploaded-batch-revision",
            },
        },
    }


def _classification_output(label: str) -> LMOutput:
    p_fail = 0.1 if label == "P" else 0.9
    chosen_probability = 1.0 - p_fail if label == "P" else p_fail
    other = "F" if label == "P" else "P"
    other_probability = p_fail if label == "P" else 1.0 - p_fail
    return LMOutput(
        text=label,
        logprobs=[
            {
                "token": f" {label}",
                "logprob": math.log(chosen_probability),
                "top_logprobs": [{"token": f" {other}", "logprob": math.log(other_probability)}],
            }
        ],
    )


def _judge_provider(first_verdict: str = "pass", *, malformed: bool = False) -> _Provider:
    def handler(request: LMRequest) -> LMOutput:
        is_classification = len(request.messages) >= 3
        if is_classification:
            native = str(request.messages[-2]["content"])
            if malformed:
                return LMOutput(text="maybe")
            label = "F" if '"verdict": "fail"' in native else "P"
            return _classification_output(label)
        prompt = str(request.messages[-1]["content"])
        verdict = first_verdict if "Prompt for mid" in prompt else "pass"
        return LMOutput(
            text=json.dumps(
                {
                    "verdict": verdict,
                    "rationale": "criterion-specific fixture rationale",
                    "evidence": "direct fixture evidence",
                }
            )
        )

    return _Provider(QWEN_JUDGE_MODEL, handler)


def _batch_judge_provider(*, raise_on: str | None = None) -> _Provider:
    def handler(request: LMRequest) -> LMOutput:
        is_classification = len(request.messages) >= 3
        if is_classification:
            native = str(request.messages[-2]["content"])
            label = "F" if '"verdict": "fail"' in native else "P"
            return _classification_output(label)
        prompt = str(request.messages[-1]["content"])
        if raise_on is not None and raise_on in prompt:
            raise RuntimeError("fixture model-specific judge failure")
        verdict = "fail" if "A mid response" in prompt else "pass"
        return LMOutput(
            text=json.dumps(
                {
                    "verdict": verdict,
                    "rationale": "batch fixture rationale",
                    "evidence": "batch fixture evidence",
                }
            )
        )

    return _Provider(QWEN_JUDGE_MODEL, handler)


def _context(
    root: Path,
    judge: InferenceProvider,
    *,
    tutor_handler: Callable[[LMRequest], LMOutput] | None = None,
) -> tuple[ModeRunContext, _Provider]:
    tutor = _Provider(
        "fixture-tutor",
        tutor_handler
        or (lambda request: LMOutput(text=f"Response to {request.messages[-1]['content']}")),
    )
    context = ModeRunContext(
        run_id="fixture-run",
        output_root=root,
        provider=tutor,
        inference_pool=_Lookup(judge),
        metadata={
            "code_revision": "fixture-code",
            "mode_runner": {"candidate_revision": "fixture-revision"},
        },
    )
    return context, tutor


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize(
    ("first_verdict", "expected_path"),
    [("fail", ["mid", "easy"]), ("pass", ["mid", "hard"])],
)
def test_mode_branches_and_uses_primary_tutor_plus_named_judge(
    tmp_path: Path,
    first_verdict: str,
    expected_path: list[str],
) -> None:
    bank_paths = _write_bank(tmp_path / "bank")
    judge = _judge_provider(first_verdict)
    context, tutor = _context(tmp_path / "run", judge)
    output = tmp_path / "run/modes/edullm_adaptive"

    result = asyncio.run(EduLLMAdaptiveMode().run(context, _config(bank_paths), output))

    assert result.status == ModeStatus.SUCCEEDED
    cat_result = json.loads((output / "cat_result.json").read_text())
    assert cat_result["scenarios_administered"] == expected_path
    assert len(tutor.requests) == 2
    assert len(judge.requests) == 4
    assert all(
        "criterion" not in json.dumps(request.messages).lower() for request in tutor.requests
    )
    assert all(request.messages for request in judge.requests)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["tutor"]["model"] == "fixture-tutor"
    assert manifest["tutor"]["provenance"]["revision"] == "fixture-revision"
    assert manifest["judge"]["model"] == QWEN_JUDGE_MODEL
    assert manifest["judge"]["failure_probability_threshold"] == 0.33
    assert manifest["bank"]["bundle_sha256"]
    assert manifest["prompts"]["judge_contract_sha256"]
    provider_tutor_row = _read_jsonl(output / "tutor_responses.jsonl")[0]
    assert provider_tutor_row["request_provenance"] == "provider_generation_request"
    assert provider_tutor_row["messages_provenance"] == "provider_generation_request"
    assert provider_tutor_row["request_sha256"] == provider_tutor_row["evaluation_context_sha256"]
    first_judge_row = _read_jsonl(output / "judge_rows.jsonl")[0]
    assert first_judge_row["requirement_source"] == "criterion_text"
    assert first_judge_row["requirements"] == [
        {"requirement_id": "R1", "text": "Requirement for mid_c01"}
    ]
    assert set(result.artifacts) == {
        "manifest.json",
        "tutor_responses.jsonl",
        "judge_rows.jsonl",
        "cat_result.json",
        "cat_trace.jsonl",
    }


def test_prompt_contract_digest_changes_with_material_prompt_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = mode_module._prompt_provenance()["judge_contract_sha256"]
    monkeypatch.setattr(
        judge_module,
        "ATOMIC_INSTRUCTION",
        judge_module.ATOMIC_INSTRUCTION + " CONTRACT_CHANGE",
    )

    after = mode_module._prompt_provenance()["judge_contract_sha256"]

    assert after != before


def test_missing_judgment_remains_none_and_is_excluded(tmp_path: Path) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid",))
    judge = _judge_provider(malformed=True)
    context, _ = _context(tmp_path / "run", judge)
    output = tmp_path / "run/modes/edullm_adaptive"

    result = asyncio.run(
        EduLLMAdaptiveMode().run(
            context,
            _config(bank_paths, max_scenarios=1),
            output,
        )
    )

    assert result.status == ModeStatus.SUCCEEDED
    rows = _read_jsonl(output / "judge_rows.jsonl")
    assert rows[0]["verdict"] == "no_decision"
    assert rows[0]["observation"] is None
    cat_result = json.loads((output / "cat_result.json").read_text())
    assert cat_result["criteria_observed"] == 0
    assert cat_result["criteria_no_decision"] == 1


def test_blank_tutor_response_is_missing_without_calling_judge(tmp_path: Path) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid",))
    judge = _judge_provider()
    context, _ = _context(
        tmp_path / "run",
        judge,
        tutor_handler=lambda _request: LMOutput(text="   "),
    )
    output = tmp_path / "run/modes/edullm_adaptive"

    result = asyncio.run(
        EduLLMAdaptiveMode().run(
            context,
            _config(bank_paths, max_scenarios=1),
            output,
        )
    )

    assert result.status == ModeStatus.SUCCEEDED
    assert not judge.requests
    row = _read_jsonl(output / "judge_rows.jsonl")[0]
    assert row["status"] == "blank_tutor_response"
    assert row["observation"] is None
    assert len(row["atomic_prompt_sha256"]) == 1
    assert row["classification_prompt_sha256"] == []


def test_precomputed_responses_skip_tutor_generation_and_reach_qwen_cat(
    tmp_path: Path,
) -> None:
    bank_paths = _write_bank(tmp_path / "bank")
    config = _config(bank_paths)
    _use_precomputed_responses(
        config,
        tmp_path / "uploaded.jsonl",
        [
            {
                "scenario_id": "mid",
                "response": "Uploaded mid response",
                "metadata": {
                    "n": 1,
                    "rendered_prompt": "original generation prompt",
                    "rendered_prompt_sha256": hashlib.sha256(
                        b"original generation prompt"
                    ).hexdigest(),
                },
            },
            {"scenario_id": "easy", "response": "Uploaded easy response"},
            {"scenario_id": "hard", "response": "Uploaded hard response"},
        ],
    )
    judge = _judge_provider("fail")
    context, tutor = _context(
        tmp_path / "run",
        judge,
        tutor_handler=lambda _request: (_ for _ in ()).throw(
            AssertionError("precomputed mode must not call the tutor provider")
        ),
    )
    output = tmp_path / "run/modes/edullm_adaptive"

    result = asyncio.run(EduLLMAdaptiveMode().run(context, config, output))

    assert result.status == ModeStatus.SUCCEEDED
    assert tutor.requests == []
    assert len(judge.requests) == 4
    cat_result = json.loads((output / "cat_result.json").read_text())
    assert cat_result["scenarios_administered"] == ["mid", "easy"]
    tutor_rows = _read_jsonl(output / "tutor_responses.jsonl")
    assert [row["raw_output"] for row in tutor_rows] == [
        "Uploaded mid response",
        "Uploaded easy response",
    ]
    assert tutor_rows[0]["response_source"] == "precomputed_jsonl"
    assert tutor_rows[0]["source_metadata"]["n"] == 1
    assert (
        tutor_rows[0]["request_sha256"] == hashlib.sha256(b"original generation prompt").hexdigest()
    )
    assert tutor_rows[0]["request_provenance"] == "uploaded_rendered_prompt"
    assert tutor_rows[0]["messages_provenance"] == (
        "reconstructed_evaluation_context_not_generation_request"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    source = manifest["tutor"]["response_source"]
    assert manifest["tutor"]["model"] == "fixture-tutor"
    assert manifest["tutor"]["provenance"]["revision"] == "uploaded-revision"
    assert source["kind"] == "precomputed_jsonl"
    assert source["declared_sha256"] == source["observed_sha256"]
    assert source["row_count"] == 3
    assert source["blank_count"] == 0
    assert source["exact_bank_coverage"] is True


def test_blank_precomputed_response_remains_no_decision_without_qwen(
    tmp_path: Path,
) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid",))
    config = _config(bank_paths, max_scenarios=1)
    _use_precomputed_responses(
        config,
        tmp_path / "uploaded.jsonl",
        [{"scenario_id": "mid", "response": "   "}],
    )
    judge = _judge_provider()
    context, tutor = _context(tmp_path / "run", judge)
    output = tmp_path / "run/modes/edullm_adaptive"

    result = asyncio.run(EduLLMAdaptiveMode().run(context, config, output))

    assert result.status == ModeStatus.SUCCEEDED
    assert tutor.requests == []
    assert judge.requests == []
    row = _read_jsonl(output / "judge_rows.jsonl")[0]
    assert row["status"] == "blank_tutor_response"
    assert row["observation"] is None
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["tutor"]["response_source"]["blank_count"] == 1


def test_precomputed_batch_reuses_qwen_and_runs_independent_cat_per_model(
    tmp_path: Path,
) -> None:
    bank_paths = _write_bank(tmp_path / "bank")
    config = _config(bank_paths)
    rows: list[dict[str, Any]] = []
    for model_id, family, revision, prefix in (
        ("vendor/z-model", "family-z", "revision-z", "Z"),
        ("vendor/a-model", "family-a", "revision-a", "A"),
    ):
        for scenario_id in ("mid", "easy", "hard"):
            rows.append(
                {
                    "model_id": model_id,
                    "model_family": family,
                    "model_revision": revision,
                    "scenario_id": scenario_id,
                    "response": f"{prefix} {scenario_id} response",
                    "metadata": {"source_index": len(rows)},
                }
            )
    _use_precomputed_response_batch(config, tmp_path / "batch.jsonl", rows)
    judge = _batch_judge_provider()
    context, tutor = _context(
        tmp_path / "run",
        judge,
        tutor_handler=lambda _request: (_ for _ in ()).throw(
            AssertionError("batch mode must not call the tutor provider")
        ),
    )
    output = tmp_path / "run/modes/edullm_adaptive"

    result = asyncio.run(EduLLMAdaptiveMode().run(context, config, output))

    assert result.status == ModeStatus.SUCCEEDED
    assert result.completed_units == 2
    assert result.metrics["models_total"] == 2
    assert result.metrics["models_succeeded"] == 2
    assert tutor.requests == []
    assert len(judge.requests) == 8
    model_rows = _read_jsonl(output / "model_results.jsonl")
    assert [row["model_id"] for row in model_rows] == [
        "vendor/a-model",
        "vendor/z-model",
    ]
    assert all("/" not in Path(row["output_dir"]).name for row in model_rows)
    assert len({row["output_dir"] for row in model_rows}) == 2
    cat_paths: dict[str, list[str]] = {}
    for row in model_rows:
        candidate_dir = output / row["output_dir"]
        candidate_manifest = json.loads((candidate_dir / "manifest.json").read_text())
        candidate_cat = json.loads((candidate_dir / "cat_result.json").read_text())
        assert candidate_manifest["tutor"]["model"] == row["model_id"]
        assert candidate_manifest["tutor"]["provenance"]["revision"] == row["model_revision"]
        assert candidate_manifest["tutor"]["response_source"]["kind"] == "precomputed_batch_jsonl"
        cat_paths[row["model_id"]] = candidate_cat["scenarios_administered"]
    assert cat_paths["vendor/a-model"] == ["mid", "easy"]
    assert cat_paths["vendor/z-model"] == ["mid", "hard"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["response_batch"]["model_count"] == 2
    assert manifest["response_batch"]["row_count"] == 6
    assert manifest["progress"]["models_completed"] == 2
    assert set(result.artifacts) == {
        "manifest.json",
        "model_results.jsonl",
        "attempt_history.jsonl",
        "batch_summary.json",
        "progress.json",
        "batch_results.json",
        "batch_results.csv",
        "BATCH_REPORT.md",
        "checkpoints/latest.json",
    }


def test_precomputed_batch_continues_after_one_model_runtime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid",))
    config = _config(bank_paths, max_scenarios=1)
    _use_precomputed_response_batch(
        config,
        tmp_path / "batch.jsonl",
        [
            {
                "model_id": "a-broken",
                "model_family": "family-a",
                "model_revision": "revision-a",
                "scenario_id": "mid",
                "response": "BROKEN response",
            },
            {
                "model_id": "z-good",
                "model_family": "family-z",
                "model_revision": "revision-z",
                "scenario_id": "mid",
                "response": "Good response",
            },
        ],
    )
    judge = _batch_judge_provider()
    context, tutor = _context(tmp_path / "run", judge)
    output = tmp_path / "run/modes/edullm_adaptive"
    original_run_candidate = EduLLMAdaptiveMode._run_candidate

    async def controlled_run_candidate(
        self: EduLLMAdaptiveMode,
        candidate_context: ModeRunContext,
        prepared: Any,
        candidate_output: Path,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Any:
        if prepared.tutor_identity.model == "a-broken":
            candidate_output.mkdir(parents=True)
            (candidate_output / "orphaned_partial.txt").write_text("preserved\n", encoding="utf-8")
            raise RuntimeError("fixture model-specific failure")
        return await original_run_candidate(
            self,
            candidate_context,
            prepared,
            candidate_output,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(EduLLMAdaptiveMode, "_run_candidate", controlled_run_candidate)

    result = asyncio.run(EduLLMAdaptiveMode().run(context, config, output))

    assert result.status == ModeStatus.FAILED
    assert result.completed_units == 2
    assert result.metrics["models_failed"] == 1
    assert result.metrics["models_succeeded"] == 1
    assert tutor.requests == []
    model_rows = _read_jsonl(output / "model_results.jsonl")
    assert [(row["model_id"], row["status"]) for row in model_rows] == [
        ("a-broken", "failed"),
        ("z-good", "succeeded"),
    ]
    broken_dir = output / model_rows[0]["output_dir"]
    good_dir = output / model_rows[1]["output_dir"]
    assert (broken_dir / "failure.json").is_file()
    assert (good_dir / "cat_result.json").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["progress"]["models_pending"] == 0


def test_reviewed_atomic_checklist_is_used_without_synthetic_splitting(
    tmp_path: Path,
) -> None:
    bank_paths = _write_bank(
        tmp_path / "bank",
        scenario_ids=("mid",),
        reviewed_checklist=True,
    )
    judge = _judge_provider()
    context, _ = _context(tmp_path / "run", judge)
    output = tmp_path / "run/modes/edullm_adaptive"

    result = asyncio.run(
        EduLLMAdaptiveMode().run(
            context,
            _config(bank_paths, max_scenarios=1),
            output,
        )
    )

    assert result.status == ModeStatus.SUCCEEDED
    row = _read_jsonl(output / "judge_rows.jsonl")[0]
    assert row["requirement_source"] == "reviewed_atomic_checklist"
    assert len(row["requirements"]) == 2
    assert row["requirement_provenance"]["review_status"] == "curation_v1_finalized"
    assert len(judge.requests) == 4


def test_unfinalized_checklist_is_ignored_in_favor_of_original_criterion(
    tmp_path: Path,
) -> None:
    bank_paths = _write_bank(
        tmp_path / "bank",
        scenario_ids=("mid",),
        reviewed_checklist=True,
        checklist_status="needs_independent_human_review",
    )
    judge = _judge_provider()
    context, _ = _context(tmp_path / "run", judge)
    output = tmp_path / "run/modes/edullm_adaptive"

    result = asyncio.run(
        EduLLMAdaptiveMode().run(
            context,
            _config(bank_paths, max_scenarios=1),
            output,
        )
    )

    assert result.status == ModeStatus.SUCCEEDED
    row = _read_jsonl(output / "judge_rows.jsonl")[0]
    assert row["requirement_source"] == "criterion_text"
    assert row["requirements"] == [{"requirement_id": "R1", "text": "Requirement for mid_c01"}]
    assert row["requirement_provenance"]["ignored_checklist_review_status"] == (
        "needs_independent_human_review"
    )
    assert len(judge.requests) == 2


def test_preflight_rejects_provider_identity_and_incomplete_config(tmp_path: Path) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid",))
    wrong_judge = _Provider("not-qwen", lambda _request: LMOutput(text="unused"))
    context, _ = _context(tmp_path / "run", wrong_judge)
    config = _config(bank_paths, max_scenarios=1)

    with pytest.raises(ValueError, match="does not expose the frozen Qwen"):
        EduLLMAdaptiveMode().preflight(context, config)

    del config["quadrature"]["max_nodes"]
    with pytest.raises(ValueError, match="quadrature fields differ"):
        EduLLMAdaptiveMode().preflight(context, config)


def test_preflight_checks_declared_thinking_mode_and_exposed_revision(tmp_path: Path) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid",))
    judge = _judge_provider()
    context, _ = _context(tmp_path / "run", judge)
    config = _config(bank_paths, max_scenarios=1)
    config["judge"]["enable_thinking"] = True
    with pytest.raises(ValueError, match="enable_thinking=false"):
        EduLLMAdaptiveMode().preflight(context, config)

    config["judge"]["enable_thinking"] = False
    judge.chat_template_kwargs = {"enable_thinking": True}  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        EduLLMAdaptiveMode().preflight(context, config)

    judge.chat_template_kwargs = {"enable_thinking": False}  # type: ignore[attr-defined]
    judge.revision = "wrong-revision"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="frozen revision"):
        EduLLMAdaptiveMode().preflight(context, config)

    del judge.revision  # type: ignore[attr-defined]
    judge.language_model_only = False  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="language_model_only=false"):
        EduLLMAdaptiveMode().preflight(context, config)


def test_preflight_rejects_candidate_revision_mismatch(tmp_path: Path) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid",))
    context, _ = _context(tmp_path / "run", _judge_provider())
    config = _config(bank_paths, max_scenarios=1)
    config["tutor"]["model_provenance"]["revision"] = "wrong-revision"

    with pytest.raises(ValueError, match="shared runner revision does not match"):
        EduLLMAdaptiveMode().preflight(context, config)


def test_run_consumes_the_exact_snapshot_built_during_preflight(tmp_path: Path) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid",))
    rubric_path = bank_paths[0]
    prepared_rubric_sha256 = _sha(rubric_path.read_bytes())
    context, _ = _context(tmp_path / "run", _judge_provider())
    config = _config(bank_paths, max_scenarios=1)
    mode = EduLLMAdaptiveMode()

    mode.preflight(context, config)
    # A mutation after the orchestrator's final validation must not make the
    # execution reload a different scientific bank. The real orchestrator also
    # detects mutations that happen before this boundary.
    rubric_path.write_text("{}\n", encoding="utf-8")

    result = asyncio.run(mode.run(context, config, tmp_path / "run/modes/edullm_adaptive"))

    assert result.status == ModeStatus.SUCCEEDED
    manifest = json.loads(
        (tmp_path / "run/modes/edullm_adaptive/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bank"]["files"]["rubrics"]["sha256"] == prepared_rubric_sha256


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [{"scenario_id": "mid", "response": "one"}],
            "must exactly cover the fitted bank",
        ),
        (
            [
                {"scenario_id": "mid", "response": "one"},
                {"scenario_id": "easy", "response": "two"},
                {"scenario_id": "extra", "response": "three"},
            ],
            "must exactly cover the fitted bank",
        ),
        (
            [
                {"scenario_id": "mid", "response": "one"},
                {"scenario_id": "mid", "response": "two"},
                {"scenario_id": "easy", "response": "three"},
            ],
            "duplicate precomputed tutor response",
        ),
        (
            [
                {"scenario_id": "mid", "response": 1},
                {"scenario_id": "easy", "response": "two"},
            ],
            "response must be a string",
        ),
        (
            [
                {"scenario_id": "mid", "response": "one", "unknown": True},
                {"scenario_id": "easy", "response": "two"},
            ],
            "fields differ",
        ),
    ],
)
def test_precomputed_response_preflight_rejects_invalid_rows_or_coverage(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    message: str,
) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid", "easy"))
    config = _config(bank_paths, max_scenarios=1)
    _use_precomputed_responses(config, tmp_path / "uploaded.jsonl", rows)

    with pytest.raises(ValueError, match=message):
        EduLLMAdaptiveMode().preflight_config(config)


def test_precomputed_response_preflight_rejects_hash_and_malformed_jsonl(
    tmp_path: Path,
) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid",))
    config = _config(bank_paths, max_scenarios=1)
    path = tmp_path / "uploaded.jsonl"
    _use_precomputed_responses(
        config,
        path,
        [{"scenario_id": "mid", "response": "one"}],
    )
    config["tutor"]["response_source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        EduLLMAdaptiveMode().preflight_config(config)

    malformed = b'{"scenario_id":"mid","response":}\n'
    path.write_bytes(malformed)
    config["tutor"]["response_source"]["sha256"] = _sha(malformed)
    with pytest.raises(ValueError, match="invalid JSON"):
        EduLLMAdaptiveMode().preflight_config(config)


def test_precomputed_response_source_requires_null_generation_and_valid_schema(
    tmp_path: Path,
) -> None:
    bank_paths = _write_bank(tmp_path / "bank", scenario_ids=("mid",))
    config = _config(bank_paths, max_scenarios=1)
    generation = config["tutor"]["generation"]
    _use_precomputed_responses(
        config,
        tmp_path / "uploaded.jsonl",
        [{"scenario_id": "mid", "response": "one"}],
    )
    config["tutor"]["generation"] = generation
    with pytest.raises(ValueError, match="generation must be null"):
        EduLLMAdaptiveMode().preflight_config(config)

    config["tutor"]["generation"] = None
    config["tutor"]["response_source"]["schema_version"] = "future-version"
    with pytest.raises(ValueError, match="unsupported tutor.response_source.schema_version"):
        EduLLMAdaptiveMode().preflight_config(config)


def test_each_criterion_gets_only_its_own_expected_evidence(tmp_path: Path) -> None:
    bank_paths = _write_bank(
        tmp_path / "bank",
        scenario_ids=("mid",),
        criteria_per_scenario=2,
    )
    judge = _judge_provider()
    context, _ = _context(tmp_path / "run", judge)
    output = tmp_path / "run/modes/edullm_adaptive"

    result = asyncio.run(
        EduLLMAdaptiveMode().run(
            context,
            _config(bank_paths, max_scenarios=1),
            output,
        )
    )

    assert result.status == ModeStatus.SUCCEEDED
    main_prompts = [
        str(request.messages[0]["content"])
        for request in judge.requests
        if len(request.messages) == 1
    ]
    assert "Expected evidence for mid_c01" in main_prompts[0]
    assert "Expected evidence for mid_c02" not in main_prompts[0]
    assert "Expected evidence for mid_c02" in main_prompts[1]
    assert "Expected evidence for mid_c01" not in main_prompts[1]


def test_runtime_exception_writes_structured_partial_failure(tmp_path: Path) -> None:
    bank_paths = _write_bank(tmp_path / "bank")
    judge = _judge_provider("pass")
    calls = 0

    def tutor_handler(request: LMRequest) -> LMOutput:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture tutor outage")
        return LMOutput(text=f"Response to {request.messages[-1]['content']}")

    context, _ = _context(tmp_path / "run", judge, tutor_handler=tutor_handler)
    output = tmp_path / "run/modes/edullm_adaptive"

    result = asyncio.run(EduLLMAdaptiveMode().run(context, _config(bank_paths), output))

    assert result.status == ModeStatus.FAILED
    assert "fixture tutor outage" in str(result.error)
    failure = json.loads((output / "failure.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    partial = json.loads((output / "cat_result.json").read_text())
    assert failure["status"] == "failed"
    assert failure["completed_tutor_responses"] == 1
    assert failure["completed_scenarios"] == 1
    assert result.completed_units == 1
    assert manifest["status"] == "failed"
    assert partial["partial"] is True
    assert "failure.json" in result.artifacts
