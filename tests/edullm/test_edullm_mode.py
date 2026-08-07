"""Mock-provider integration tests for the OLMo-owned EduLLM mode."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

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
