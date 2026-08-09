"""No-GPU feasibility tests for an EduLLM adaptive mode on OLMo contracts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from olmo_eval.common.types import LMRequest, ProviderKind
from olmo_eval.evals.external import ExternalEval, ExternalEvalContext, ExternalEvalResult
from olmo_eval.experimental.edullm_adaptive import (
    AdaptiveRunConfig,
    CallableEvaluationMode,
    EvaluationModeRegistry,
    ScriptedProvider,
    StaticProviderLookup,
    adaptive_artifact_directory,
    fixture_items,
    run_adaptive_spike,
)
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.external import ExternalEvalRunner
from olmo_eval.runners.external import runner as external_runner_module


def _scenario_id(request: LMRequest) -> str:
    for line in request.prompt.splitlines():
        if line.startswith("SCENARIO_ID: "):
            return line.removeprefix("SCENARIO_ID: ").strip()
    raise AssertionError("scripted request omitted SCENARIO_ID")


def _json_judgment(verdict: str, scenario_id: str) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "evidence": f"Evidence for {scenario_id}",
            "rationale": f"Fixture rationale for {scenario_id}",
        }
    )


def _providers(
    decisions: dict[str, str],
) -> tuple[ScriptedProvider, ScriptedProvider, ExternalEvalContext]:
    tutor = ScriptedProvider(
        model_name="scripted-tutor",
        handler=lambda request: f"Tutor response for {_scenario_id(request)}",
    )

    def judge_handler(request: LMRequest) -> str:
        scenario_id = _scenario_id(request)
        verdict = decisions[scenario_id]
        if verdict == "no_decision":
            return "not valid judge JSON"
        return _json_judgment(verdict=verdict, scenario_id=scenario_id)

    judge = ScriptedProvider(model_name="scripted-judge", handler=judge_handler)
    context = ExternalEvalContext(
        provider=tutor,
        inference_pool=StaticProviderLookup({"judge": judge}),
    )
    return tutor, judge, context


def _run_fixture(
    tmp_path: Path,
    run_id: str,
    decisions: dict[str, str],
) -> tuple[ScriptedProvider, ScriptedProvider, ExternalEvalResult]:
    tutor, judge, context = _providers(decisions=decisions)
    result = asyncio.run(
        run_adaptive_spike(
            context=context,
            items=fixture_items(),
            config=AdaptiveRunConfig(run_id=run_id),
            output_dir=tmp_path,
        )
    )
    return tutor, judge, result


def _artifact_dir(output_dir: Path, result: ExternalEvalResult) -> Path:
    return output_dir / str(result.metadata["artifact_namespace"])


def test_adaptive_selection_branches_on_prior_judgment_and_estimates_ability(
    tmp_path: Path,
) -> None:
    _, _, strong = _run_fixture(
        tmp_path=tmp_path / "strong",
        run_id="strong",
        decisions={"mid": "pass", "hard": "fail", "easy": "no_decision"},
    )
    _, _, weak = _run_fixture(
        tmp_path=tmp_path / "weak",
        run_id="weak",
        decisions={"mid": "fail", "easy": "pass", "hard": "no_decision"},
    )

    assert strong.metadata["scenarios_administered"] == ["mid", "hard", "easy"]
    assert weak.metadata["scenarios_administered"] == ["mid", "easy", "hard"]
    assert strong.metadata["final_eap"]["theta"] == pytest.approx(0.283622568, abs=1e-6)
    assert weak.metadata["final_eap"]["theta"] == pytest.approx(-0.283622568, abs=1e-6)
    assert strong.metadata["final_eap"]["se"] == pytest.approx(0.660388755, abs=1e-6)
    assert weak.metadata["final_eap"]["se"] == pytest.approx(0.660388755, abs=1e-6)
    assert strong.metadata["final_mwle"]["theta"] == pytest.approx(0.499999796, abs=1e-5)
    assert weak.metadata["final_mwle"]["theta"] == pytest.approx(-0.499999796, abs=1e-5)
    assert strong.metadata["final_mwle"]["se"] == pytest.approx(0.797351713, abs=1e-5)
    assert weak.metadata["final_mwle"]["se"] == pytest.approx(0.797351713, abs=1e-5)
    assert strong.metadata["final_mwle"]["converged"] is True
    assert weak.metadata["final_mwle"]["converged"] is True


def test_no_decision_is_preserved_skipped_by_estimators_and_written(
    tmp_path: Path,
) -> None:
    tutor, judge, result = _run_fixture(
        tmp_path=tmp_path,
        run_id="tri-state",
        decisions={"mid": "pass", "hard": "fail", "easy": "no_decision"},
    )

    assert result.metrics["coverage"] == pytest.approx(2 / 3)
    assert result.metadata["criteria_observed"] == 2
    assert result.metadata["criteria_no_decision"] == 1
    assert result.predictions is not None
    undecided = result.predictions[-1]
    assert undecided["judgment"]["verdict"] == "no_decision"
    assert undecided["judgment"]["y"] is None
    assert undecided["judgment"]["error_code"] == "invalid_json"
    assert undecided["observed_before"] == undecided["observed_after"] == 2
    assert undecided["selection_theta"] == undecided["online_theta_after"]
    assert undecided["selection_variance"] == undecided["online_variance_after"]
    assert result.metadata["final_eap"]["n_items"] == 2
    assert result.metadata["final_mwle"]["n_items"] == 2
    assert len(tutor.requests) == len(judge.requests) == 3

    artifact_dir = _artifact_dir(tmp_path, result)
    assert artifact_dir == adaptive_artifact_directory(
        tmp_path,
        AdaptiveRunConfig(run_id="tri-state"),
    )
    assert {path.name for path in artifact_dir.iterdir()} == {
        "calibrated_bank.json",
        "final_result.json",
        "judgments.jsonl",
        "manifest.json",
        "steps.jsonl",
    }
    judgment_lines = (artifact_dir / "judgments.jsonl").read_text().splitlines()
    judgments = [json.loads(line) for line in judgment_lines]
    assert len(judgments) == 3
    assert judgments[-1]["verdict"] == "no_decision"
    assert ExternalEvalResult.from_dict(result.to_dict()).to_dict() == result.to_dict()

    calibrated_bank = json.loads((artifact_dir / "calibrated_bank.json").read_text())
    assert calibrated_bank == [item.to_dict() for item in fixture_items()]


def test_all_no_decision_run_writes_strict_json_with_null_unavailable_mwle(
    tmp_path: Path,
) -> None:
    _, _, result = _run_fixture(
        tmp_path=tmp_path,
        run_id="all-invalid",
        decisions={"mid": "no_decision", "easy": "no_decision", "hard": "no_decision"},
    )

    assert result.metadata["criteria_observed"] == 0
    assert result.metadata["criteria_no_decision"] == 3
    assert result.metadata["final_mwle"]["theta"] is None
    assert result.metadata["final_mwle"]["se"] is None
    assert "mwle_theta" not in result.metrics
    assert "mwle_se" not in result.metrics

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant {value}")

    artifact_dir = _artifact_dir(tmp_path, result)
    for path in artifact_dir.iterdir():
        if path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                json.loads(line, parse_constant=reject_constant)
        else:
            json.loads(path.read_text(), parse_constant=reject_constant)


def test_adaptive_sidecars_are_namespaced_by_evaluation_and_run(tmp_path: Path) -> None:
    _, _, first = _run_fixture(
        tmp_path=tmp_path,
        run_id="first/run",
        decisions={"mid": "pass", "hard": "fail", "easy": "no_decision"},
    )
    _, _, second = _run_fixture(
        tmp_path=tmp_path,
        run_id="second/run",
        decisions={"mid": "fail", "easy": "pass", "hard": "no_decision"},
    )

    first_dir = _artifact_dir(tmp_path, first)
    second_dir = _artifact_dir(tmp_path, second)
    assert first_dir != second_dir
    assert first_dir.is_dir()
    assert second_dir.is_dir()
    assert "%2F" in str(first.metadata["artifact_namespace"])
    assert "%2F" in str(second.metadata["artifact_namespace"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "."),
        ("run_id", ".."),
        ("eval_name", "."),
        ("eval_name", ".."),
    ],
)
def test_adaptive_artifact_namespace_rejects_dot_segments(field: str, value: str) -> None:
    kwargs = {"run_id": "safe-run", "eval_name": "safe-eval", field: value}
    with pytest.raises(ValueError, match="dot segment"):
        AdaptiveRunConfig(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stop_eap_se": float("nan")}, "stop_eap_se"),
        ({"quadrature_bound": float("inf")}, "quadrature_bound"),
        ({"mwle_ridge": 0.0}, "mwle_ridge"),
    ],
)
def test_adaptive_config_rejects_unsafe_numeric_values(
    overrides: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveRunConfig(run_id="invalid-config", **overrides)


def test_mode_registry_exposes_two_modes_and_accepts_future_registration() -> None:
    async def standard_handler(**kwargs: Any) -> dict[str, Any]:
        return {"delegated_to": "olmo", **kwargs}

    async def adaptive_handler(**kwargs: Any) -> dict[str, Any]:
        return {"delegated_to": "edullm", **kwargs}

    registry = EvaluationModeRegistry()
    registry.register(CallableEvaluationMode("standard_olmo", standard_handler))
    registry.register(CallableEvaluationMode("edullm_adaptive", adaptive_handler))

    assert registry.names == ("edullm_adaptive", "standard_olmo")
    assert asyncio.run(registry.run("standard_olmo", suite="mmlu")) == {
        "delegated_to": "olmo",
        "suite": "mmlu",
    }
    assert asyncio.run(registry.run("edullm_adaptive", bank="frozen")) == {
        "delegated_to": "edullm",
        "bank": "frozen",
    }

    registry.register(CallableEvaluationMode("future_mode", standard_handler))
    assert "future_mode" in registry.names


class AdaptiveSpikeExternalEval(ExternalEval):
    @property
    def name(self) -> str:
        return "edullm_adaptive_spike"

    @property
    def description(self) -> str:
        return "Three-scenario no-GPU adaptive integration spike."

    @property
    def timeout_seconds(self) -> float:
        return 30.0

    async def execute(
        self,
        provider: InferenceProvider,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        raise RuntimeError("adaptive spike requires a named judge provider")

    async def execute_with_context(
        self,
        context: ExternalEvalContext,
        args: dict[str, Any],
        output_dir: str | None = None,
        container_runtime: str = "podman",
    ) -> ExternalEvalResult:
        return await run_adaptive_spike(
            context=context,
            items=fixture_items(),
            config=AdaptiveRunConfig(run_id=str(args["run_id"])),
            output_dir=output_dir,
        )


def test_actual_external_runner_executes_adaptive_loop_but_drops_rich_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = AdaptiveSpikeExternalEval()

    def judge_handler(request: LMRequest) -> str:
        scenario_id = _scenario_id(request)
        verdicts = {"mid": "pass", "hard": "fail", "easy": "no_decision"}
        verdict = verdicts[scenario_id]
        return (
            "malformed"
            if verdict == "no_decision"
            else _json_judgment(verdict=verdict, scenario_id=scenario_id)
        )

    judge = ScriptedProvider(model_name="scripted-judge", handler=judge_handler)
    runner = ExternalEvalRunner(
        provider_config=ProviderConfig(kind=ProviderKind.MOCK, model="olmo-mock-tutor"),
        external_eval_names=[evaluator.name],
        output_dir=str(tmp_path),
        eval_args={"run_id": "runner-path"},
        metrics=None,
        inference_pool=StaticProviderLookup({"judge": judge}),
    )
    monkeypatch.setattr(external_runner_module, "get_external_eval", lambda _name: evaluator)

    results = asyncio.run(runner.run_async())

    result = results[evaluator.name]
    assert result.metadata["candidate_model"] == "olmo-mock-tutor"
    assert result.metadata["judge_model"] == "scripted-judge"
    assert result.metadata["scenarios_administered"] == ["mid", "hard", "easy"]
    assert len(judge.requests) == 3

    sidecar = json.loads((_artifact_dir(tmp_path, result) / "final_result.json").read_text())
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert result.name == evaluator.name
    assert metrics["tasks"][0]["task"] == result.name
    assert metrics["tasks"][0]["config"]["eval_name"] == result.name
    assert sidecar["metadata"]["stop_reason"] == "bank_exhausted"
    assert sidecar["metadata"]["final_mwle"]["n_items"] == 2
    assert "stop_reason" not in json.dumps(metrics)
    assert "final_mwle" not in json.dumps(metrics)
