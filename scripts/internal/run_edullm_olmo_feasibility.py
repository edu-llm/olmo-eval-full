#!/usr/bin/env python3
"""Run the deterministic no-GPU EduLLM-on-OLMo feasibility experiment.

Run from the repository root with the optional analysis dependencies available::

    uv run --extra analysis python scripts/internal/run_edullm_olmo_feasibility.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from olmo_eval.common.types import LMRequest, ProviderKind
from olmo_eval.evals.external import (
    ExternalEval,
    ExternalEvalContext,
    ExternalEvalResult,
    register_external_eval,
)
from olmo_eval.experimental.edullm_adaptive import (
    AdaptiveRunConfig,
    CallableEvaluationMode,
    EvaluationModeRegistry,
    ScriptedProvider,
    StaticProviderLookup,
    fixture_items,
    run_adaptive_spike,
)
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.external import ExternalEvalRunner

EVAL_NAME = "edullm_adaptive_feasibility"
DEFAULT_LEGACY_REFERENCE = Path(__file__).with_name("edullm_olmo_legacy_regression.json")


def _scenario_id(request: LMRequest) -> str:
    for line in request.prompt.splitlines():
        if line.startswith("SCENARIO_ID: "):
            return line.removeprefix("SCENARIO_ID: ").strip()
    raise RuntimeError("request omitted SCENARIO_ID")


def _judge_output(verdict: str, scenario_id: str) -> str:
    if verdict == "no_decision":
        return "deliberately malformed judge output"
    return json.dumps(
        {
            "verdict": verdict,
            "evidence": f"Evidence for {scenario_id}",
            "rationale": f"Deterministic fixture rationale for {scenario_id}",
        }
    )


class AdaptiveFeasibilityEval(ExternalEval):
    @property
    def name(self) -> str:
        return EVAL_NAME

    @property
    def description(self) -> str:
        return "Three-scenario no-GPU EduLLM adaptive feasibility experiment."

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
        raise RuntimeError("EduLLM adaptive evaluation requires a named judge provider")

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
            config=AdaptiveRunConfig(
                run_id=str(args["run_id"]),
                eval_name=self.name,
            ),
            output_dir=output_dir,
        )


def _run_arm(root: Path, arm: str, decisions: dict[str, str]) -> ExternalEvalResult:
    arm_dir = root / arm

    def handler(request: LMRequest) -> str:
        scenario_id = _scenario_id(request)
        return _judge_output(verdict=decisions[scenario_id], scenario_id=scenario_id)

    judge = ScriptedProvider(model_name=f"scripted-qwen-{arm}", handler=handler)
    runner = ExternalEvalRunner(
        provider_config=ProviderConfig(kind=ProviderKind.MOCK, model=f"mock-tutor-{arm}"),
        external_eval_names=[EVAL_NAME],
        output_dir=str(arm_dir),
        eval_args={"run_id": arm},
        metrics=None,
        inference_pool=StaticProviderLookup({"judge": judge}),
    )
    runner.validate()
    return runner.run()[EVAL_NAME]


async def _exercise_mode_registry() -> tuple[str, ...]:
    async def delegate(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    registry = EvaluationModeRegistry()
    registry.register(CallableEvaluationMode("standard_olmo", delegate))
    registry.register(CallableEvaluationMode("edullm_adaptive", delegate))
    await registry.run("standard_olmo", suite="fixture")
    await registry.run("edullm_adaptive", bank="fixture")
    return registry.names


def _metric_metadata_survives(root: Path, arm: str) -> bool:
    metrics = json.loads((root / arm / "metrics.json").read_text())
    encoded = json.dumps(metrics)
    return "stop_reason" in encoded and "final_mwle" in encoded


def _strict_json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    return json.loads(text, parse_constant=reject_constant)


def _read_jsonl(path: Path) -> list[Any]:
    return [_strict_json(line) for line in path.read_text().splitlines()]


def _sidecars_round_trip(
    root: Path,
    arm: str,
    result: ExternalEvalResult,
    expected_bank: list[dict[str, Any]],
) -> bool:
    """Reload and compare every adaptive sidecar written for one run."""

    destination = root / arm / str(result.metadata["artifact_namespace"])
    expected_names = {
        "manifest.json",
        "calibrated_bank.json",
        "steps.jsonl",
        "judgments.jsonl",
        "final_result.json",
    }
    if not destination.is_dir() or {path.name for path in destination.iterdir()} != expected_names:
        return False

    manifest = _strict_json((destination / "manifest.json").read_text())
    calibrated_bank = _strict_json((destination / "calibrated_bank.json").read_text())
    steps = _read_jsonl(destination / "steps.jsonl")
    judgments = _read_jsonl(destination / "judgments.jsonl")
    final_result = _strict_json((destination / "final_result.json").read_text())
    expected_manifest = {key: result.metadata[key] for key in manifest}
    expected_judgments = [
        {
            "step": step["step"],
            "scenario_id": step["scenario_id"],
            "criterion_id": step["criterion_id"],
            **step["judgment"],
        }
        for step in (result.predictions or [])
    ]
    return (
        manifest == expected_manifest
        and calibrated_bank == expected_bank
        and steps == result.predictions
        and judgments == expected_judgments
        and final_result == result.to_dict()
    )


def _evaluation_names_align(root: Path, arm: str, result: ExternalEvalResult) -> bool:
    metrics = _strict_json((root / arm / "metrics.json").read_text())
    tasks = metrics.get("tasks", [])
    return (
        result.name == EVAL_NAME
        and len(tasks) == 1
        and tasks[0].get("task") == result.name
        and tasks[0].get("config", {}).get("eval_name") == result.name
    )


def _load_legacy_reference(path: Path) -> dict[str, Any]:
    reference = _strict_json(path.read_text())
    source = reference.get("source", {})
    if not source.get("commit") or not source.get("files"):
        raise ValueError(f"legacy reference lacks source provenance: {path}")
    return reference


def _legacy_provenance_resolves(reference: dict[str, Any]) -> bool:
    """Verify each recorded path resolves to its recorded blob at the source commit."""

    source = reference["source"]
    commit = str(source["commit"])
    repository_root = Path(__file__).resolve().parents[2]
    try:
        for file_record in source["files"]:
            resolved = subprocess.run(
                ["git", "rev-parse", f"{commit}:{file_record['path']}"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if resolved != file_record["blob"]:
                return False
    except (KeyError, OSError, subprocess.CalledProcessError):
        return False
    return True


def _fixture_definition_matches_reference(reference: dict[str, Any]) -> bool:
    recorded = reference["fixture"]
    actual_items = [
        {
            "scenario_id": item.scenario_id,
            "discrimination": item.discrimination,
            "difficulty": item.difficulty,
        }
        for item in fixture_items()
    ]
    default_config = AdaptiveRunConfig(run_id="reference-check")
    return (
        recorded["dimensions"] == 1
        and recorded["items"] == actual_items
        and recorded["quadrature"]
        == {
            "method": "normal_trapezoid",
            "nodes": default_config.quadrature_nodes,
            "bound": default_config.quadrature_bound,
        }
        and recorded["mwle_ridge"] == default_config.mwle_ridge
        and recorded["initial_online_theta"] == 0.0
        and recorded["initial_online_variance"] == 1.0
        and recorded["selection_policy"]["top_n"] == 1
    )


def _arm_matches_reference(result: ExternalEvalResult, expected: dict[str, Any]) -> bool:
    predictions = result.predictions or []
    decisions = {step["scenario_id"]: step["judgment"]["verdict"] for step in predictions}
    observations = [
        [step["scenario_id"], step["judgment"]["y"]]
        for step in predictions
        if step["judgment"]["y"] is not None
    ]
    return (
        result.metadata["scenarios_administered"] == expected["scenario_order"]
        and decisions == expected["decisions"]
        and observations == expected["observations"]
    )


def _render_summary(payload: dict[str, Any]) -> str:
    gates = payload["gates"]
    core_status = "feasible" if payload["adaptive_core_feasible"] else "not feasible"
    native_status = "yes" if payload["native_integration_ready"] else "no"
    rows = "\n".join(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in gates.items()
    )
    return f"""# EduLLM–OLMo adaptive feasibility spike

## Verdict

- Adaptive CAT core on OLMo provider contracts: **{core_status}**
- Production-ready native OLMo integration today: **{native_status}**
- Recommended architecture now: **{payload["recommended_architecture"]}**

The three-scenario CPU-only experiment made tutor and judge calls through OLMo's
inference-provider interface. The first verdict changed the next selected scenario,
EAP and MWLE matched recorded, provenance-bearing legacy-derived regression values,
and malformed judge output stayed `no_decision` without updating ability.

## Gate results

| Gate | Result |
| --- | --- |
{rows}

## Observed paths

- Strong trajectory: `{" -> ".join(payload["strong"]["scenario_order"])}`
- Weak trajectory: `{" -> ".join(payload["weak"]["scenario_order"])}`
- Strong EAP theta: `{payload["strong"]["eap_theta"]:.9f}`
- Weak EAP theta: `{payload["weak"]["eap_theta"]:.9f}`
- Strong MWLE theta: `{payload["strong"]["mwle_theta"]:.9f}`
- Weak MWLE theta: `{payload["weak"]["mwle_theta"]:.9f}`

## Scope and limitations

This was an interface-feasibility experiment, not a validation of the full scientific
study. It used one latent dimension, three already-calibrated scenarios, one criterion
per scenario, and mock providers. It did not refit item parameters, run a real Qwen or
tutor model, allocate GPUs, or test multidimensional recovery. Those behaviors remain
covered by the legacy EduLLM studies and must be migrated with their tests.

The spike used deterministic top-1 information selection to make branching easy to
characterize. It did not reproduce or validate the legacy default seeded-random choice
among the top five scenarios.

The regression oracle records its source commit, file paths, and Git blob hashes in
`scripts/internal/edullm_olmo_legacy_regression.json`. Matching it is compatibility
evidence, not independent validation of the legacy method.

## Remaining integration work

The prototype injects the judge lookup manually. OLMo's external runner still needs to
own the auxiliary provider lifecycle/GPU planning, preserve rich adaptive metadata in
its standard result storage, and support discoverable external mode plugins. Until
those gates pass, the safe product design is one two-mode dispatcher with standard
OLMo delegated natively and EduLLM adaptive execution kept behind its own adapter.

The branch audit also found that the legacy live verdict property maps every non-pass
label to zero. The production migration therefore needs the explicit tri-state boundary
demonstrated here so `no_decision` is skipped rather than treated as a failed criterion.
"""


def run(output_dir: Path, legacy_reference_path: Path = DEFAULT_LEGACY_REFERENCE) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    register_external_eval(AdaptiveFeasibilityEval())
    legacy_reference = _load_legacy_reference(legacy_reference_path)
    expected = legacy_reference["expected"]
    tolerances = legacy_reference["tolerances"]

    strong = _run_arm(
        root=output_dir,
        arm="strong",
        decisions={"mid": "pass", "hard": "fail", "easy": "no_decision"},
    )
    weak = _run_arm(
        root=output_dir,
        arm="weak",
        decisions={"mid": "fail", "easy": "pass", "hard": "no_decision"},
    )
    assert strong.predictions is not None and weak.predictions is not None

    strong_order = list(strong.metadata["scenarios_administered"])
    weak_order = list(weak.metadata["scenarios_administered"])
    direct_round_trip = (
        ExternalEvalResult.from_dict(strong.to_dict()).to_dict() == strong.to_dict()
        and ExternalEvalResult.from_dict(weak.to_dict()).to_dict() == weak.to_dict()
    )
    mode_names = asyncio.run(_exercise_mode_registry())
    expected_bank = [item.to_dict() for item in fixture_items()]

    gates = {
        "legacy_provenance_resolves": _legacy_provenance_resolves(legacy_reference),
        "fixture_definition_matches_recorded_oracle": (
            _fixture_definition_matches_reference(legacy_reference)
        ),
        "sequential_selection_uses_prior_judgments": (
            strong_order == expected["strong"]["scenario_order"]
            and weak_order == expected["weak"]["scenario_order"]
            and _arm_matches_reference(strong, expected["strong"])
            and _arm_matches_reference(weak, expected["weak"])
        ),
        "distinct_tutor_and_judge_providers": (
            strong.metadata["candidate_model"] != strong.metadata["judge_model"]
            and weak.metadata["candidate_model"] != weak.metadata["judge_model"]
        ),
        "eap_matches_recorded_legacy_regression": (
            math.isclose(
                strong.metadata["final_eap"]["theta"],
                expected["strong"]["eap_theta"],
                abs_tol=tolerances["eap_absolute"],
            )
            and math.isclose(
                weak.metadata["final_eap"]["theta"],
                expected["weak"]["eap_theta"],
                abs_tol=tolerances["eap_absolute"],
            )
            and math.isclose(
                strong.metadata["final_eap"]["se"],
                expected["strong"]["eap_se"],
                abs_tol=tolerances["eap_absolute"],
            )
            and math.isclose(
                weak.metadata["final_eap"]["se"],
                expected["weak"]["eap_se"],
                abs_tol=tolerances["eap_absolute"],
            )
        ),
        "mwle_matches_recorded_legacy_regression": (
            math.isclose(
                strong.metadata["final_mwle"]["theta"],
                expected["strong"]["mwle_theta"],
                abs_tol=tolerances["mwle_absolute"],
            )
            and math.isclose(
                weak.metadata["final_mwle"]["theta"],
                expected["weak"]["mwle_theta"],
                abs_tol=tolerances["mwle_absolute"],
            )
            and math.isclose(
                strong.metadata["final_mwle"]["se"],
                expected["strong"]["mwle_se"],
                abs_tol=tolerances["mwle_absolute"],
            )
            and math.isclose(
                weak.metadata["final_mwle"]["se"],
                expected["weak"]["mwle_se"],
                abs_tol=tolerances["mwle_absolute"],
            )
            and strong.metadata["final_mwle"]["converged"]
            and weak.metadata["final_mwle"]["converged"]
        ),
        "no_decision_is_not_a_failure": (
            strong.metadata["criteria_no_decision"] == 1
            and weak.metadata["criteria_no_decision"] == 1
            and strong.metadata["criteria_observed"] == 2
            and weak.metadata["criteria_observed"] == 2
        ),
        "adaptive_sidecars_round_trip_losslessly": (
            _sidecars_round_trip(output_dir, "strong", strong, expected_bank)
            and _sidecars_round_trip(output_dir, "weak", weak, expected_bank)
            and direct_round_trip
        ),
        "evaluation_names_align": (
            _evaluation_names_align(output_dir, "strong", strong)
            and _evaluation_names_align(output_dir, "weak", weak)
        ),
        "two_mode_dispatcher_is_extensible": mode_names == ("edullm_adaptive", "standard_olmo"),
        "standard_external_metrics_preserve_adaptive_metadata": (
            _metric_metadata_survives(output_dir, "strong")
            and _metric_metadata_survives(output_dir, "weak")
        ),
        # These are deliberately false in the bounded spike: the named judge was
        # injected, not started/planned/stopped by ExternalEvalRunner, and this
        # external eval was registered explicitly rather than discovered as a plugin.
        "external_runner_owns_judge_lifecycle": False,
        "third_party_mode_plugin_is_auto_discovered": False,
    }
    core_gate_names = (
        "legacy_provenance_resolves",
        "fixture_definition_matches_recorded_oracle",
        "sequential_selection_uses_prior_judgments",
        "distinct_tutor_and_judge_providers",
        "eap_matches_recorded_legacy_regression",
        "mwle_matches_recorded_legacy_regression",
        "no_decision_is_not_a_failure",
        "adaptive_sidecars_round_trip_losslessly",
        "evaluation_names_align",
        "two_mode_dispatcher_is_extensible",
    )
    adaptive_core_feasible = all(gates[name] for name in core_gate_names)
    native_integration_ready = all(gates.values())
    payload = {
        "schema_version": "edullm-olmo-feasibility-result-v1",
        "adaptive_core_feasible": adaptive_core_feasible,
        "native_integration_ready": native_integration_ready,
        "recommended_architecture": (
            "two-mode dispatcher with standard OLMo native execution and an "
            "EduLLM adaptive adapter; continue the integrated plugin only after "
            "provider-lifecycle, artifact, and discovery gates pass"
        ),
        "legacy_reference": {
            "schema_version": legacy_reference["schema_version"],
            "source": legacy_reference["source"],
        },
        "gates": gates,
        "strong": {
            "scenario_order": strong_order,
            "eap_theta": strong.metadata["final_eap"]["theta"],
            "eap_se": strong.metadata["final_eap"]["se"],
            "mwle_theta": strong.metadata["final_mwle"]["theta"],
            "mwle_se": strong.metadata["final_mwle"]["se"],
            "coverage": strong.metrics["coverage"],
        },
        "weak": {
            "scenario_order": weak_order,
            "eap_theta": weak.metadata["final_eap"]["theta"],
            "eap_se": weak.metadata["final_eap"]["se"],
            "mwle_theta": weak.metadata["final_mwle"]["theta"],
            "mwle_se": weak.metadata["final_mwle"]["se"],
            "coverage": weak.metrics["coverage"],
        },
    }
    (output_dir / "feasibility_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output_dir / "FEASIBILITY_SUMMARY.md").write_text(_render_summary(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/edullm_olmo_feasibility_spike"),
    )
    parser.add_argument(
        "--legacy-reference",
        type=Path,
        default=DEFAULT_LEGACY_REFERENCE,
    )
    args = parser.parse_args()
    payload = run(
        output_dir=args.output_dir,
        legacy_reference_path=args.legacy_reference,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["adaptive_core_feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
