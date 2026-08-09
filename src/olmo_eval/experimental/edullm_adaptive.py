"""No-GPU feasibility spike for an EduLLM adaptive evaluation in OLMo Eval.

This is deliberately a small characterization prototype, not production CAT code. It
uses OLMo's real inference-provider and external-result contracts while transcribing
the one-dimensional IRT/EAP/MWLE behavior needed to test the integration boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.evals.external import ExternalEvalContext, ExternalEvalResult
from olmo_eval.inference.base import InferenceProvider

SPIKE_SCHEMA_VERSION = "edullm-olmo-adaptive-spike-v1"
Verdict = Literal["pass", "fail", "no_decision"]


@dataclass(frozen=True, slots=True)
class CalibratedItem:
    """Minimal calibrated one-dimensional criterion used by the spike."""

    scenario_id: str
    criterion_id: str
    prompt: str
    criterion: str
    discrimination: float
    difficulty: float

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.criterion_id:
            raise ValueError("scenario_id and criterion_id must be non-empty")
        if not math.isfinite(self.discrimination) or self.discrimination <= 0:
            raise ValueError("discrimination must be finite and positive")
        if not math.isfinite(self.difficulty):
            raise ValueError("difficulty must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "criterion_id": self.criterion_id,
            "prompt": self.prompt,
            "criterion": self.criterion,
            "discrimination": self.discrimination,
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True, slots=True)
class JudgeDecision:
    """Tri-state normalization boundary; no-decision never aliases to fail."""

    verdict: Verdict
    evidence: str | None
    rationale: str | None
    raw_output: str
    parser_status: Literal["parsed", "parse_error"]
    error_code: str | None = None

    @property
    def y(self) -> int | None:
        if self.verdict == "pass":
            return 1
        if self.verdict == "fail":
            return 0
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "y": self.y,
            "evidence": self.evidence,
            "rationale": self.rationale,
            "raw_output": self.raw_output,
            "parser_status": self.parser_status,
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class AbilityEstimate:
    theta: float | None
    se: float | None
    n_items: int
    converged: bool
    method: str
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta": self.theta,
            "se": self.se,
            "n_items": self.n_items,
            "converged": self.converged,
            "method": self.method,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveRunConfig:
    run_id: str
    eval_name: str = "edullm_adaptive_spike"
    judge_provider_name: str = "judge"
    max_scenarios: int = 3
    min_scenarios: int = 1
    stop_eap_se: float | None = None
    quadrature_nodes: int = 801
    quadrature_bound: float = 8.0
    mwle_ridge: float = 1e-6

    def __post_init__(self) -> None:
        if not self.run_id or self.run_id in {".", ".."}:
            raise ValueError("run_id must be non-empty and cannot be a dot segment")
        if not self.eval_name or self.eval_name in {".", ".."}:
            raise ValueError("eval_name must be non-empty and cannot be a dot segment")
        if self.max_scenarios < 1:
            raise ValueError("max_scenarios must be positive")
        if not 1 <= self.min_scenarios <= self.max_scenarios:
            raise ValueError("min_scenarios must be in [1, max_scenarios]")
        if self.stop_eap_se is not None and (
            not math.isfinite(self.stop_eap_se) or self.stop_eap_se <= 0
        ):
            raise ValueError("stop_eap_se must be finite and positive")
        if self.quadrature_nodes < 3 or self.quadrature_nodes % 2 == 0:
            raise ValueError("quadrature_nodes must be an odd integer >= 3")
        if not math.isfinite(self.quadrature_bound) or self.quadrature_bound <= 0:
            raise ValueError("quadrature_bound must be finite and positive")
        if not math.isfinite(self.mwle_ridge) or self.mwle_ridge <= 0:
            raise ValueError("mwle_ridge must be finite and positive")


class EvaluationMode(Protocol):
    """Small mode contract used to prove future modes need no dispatcher branch."""

    @property
    def name(self) -> str: ...

    async def run(self, **kwargs: Any) -> Any: ...


class EvaluationModeRegistry:
    """Prototype registry for standard OLMo and EduLLM adaptive modes."""

    def __init__(self) -> None:
        self._modes: dict[str, EvaluationMode] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._modes))

    def register(self, mode: EvaluationMode) -> None:
        if not mode.name:
            raise ValueError("evaluation mode name must be non-empty")
        if mode.name in self._modes:
            raise ValueError(f"evaluation mode {mode.name!r} is already registered")
        self._modes[mode.name] = mode

    async def run(self, name: str, **kwargs: Any) -> Any:
        try:
            mode = self._modes[name]
        except KeyError:
            raise KeyError(f"unknown evaluation mode {name!r}; available: {self.names}") from None
        return await mode.run(**kwargs)


@dataclass(frozen=True)
class CallableEvaluationMode:
    name: str
    handler: Callable[..., Awaitable[Any]]

    async def run(self, **kwargs: Any) -> Any:
        return await self.handler(**kwargs)


class StaticProviderLookup:
    """In-memory ProviderLookup used to inject a distinct judge in no-GPU tests."""

    def __init__(self, providers: Mapping[str, InferenceProvider]) -> None:
        self._providers = dict(providers)

    @property
    def names(self) -> list[str]:
        return sorted(self._providers)

    def get(self, name: str) -> InferenceProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(f"unknown provider {name!r}; available: {self.names}") from None


class ScriptedProvider(InferenceProvider):
    """Request-aware mock provider with a complete OLMo provider interface."""

    def __init__(self, model_name: str, handler: Callable[[LMRequest], str]) -> None:
        super().__init__(model_name=model_name)
        self._handler = handler
        self.requests: list[LMRequest] = []

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        params = self._default_sampling_params(sampling_params)
        outputs: list[list[LMOutput]] = []
        for request in requests:
            self.requests.append(request)
            text = self._handler(request)
            outputs.append([LMOutput(text=text) for _ in range(params.num_samples)])
        return outputs

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        return self.generate(requests=requests, sampling_params=sampling_params)

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        raise NotImplementedError("the adaptive spike uses generation only")


def fixture_items() -> tuple[CalibratedItem, ...]:
    """Return the symmetric three-scenario fixture from the legacy CAT math."""

    return (
        CalibratedItem(
            scenario_id="mid",
            criterion_id="mid_c01",
            prompt="Give a medium-difficulty tutoring response.",
            criterion="The response satisfies the medium requirement.",
            discrimination=2.0,
            difficulty=0.0,
        ),
        CalibratedItem(
            scenario_id="easy",
            criterion_id="easy_c01",
            prompt="Give an easy tutoring response.",
            criterion="The response satisfies the easy requirement.",
            discrimination=2.0,
            difficulty=-2.0,
        ),
        CalibratedItem(
            scenario_id="hard",
            criterion_id="hard_c01",
            prompt="Give a hard tutoring response.",
            criterion="The response satisfies the hard requirement.",
            discrimination=2.0,
            difficulty=2.0,
        ),
    )


def parse_binary_judgment(raw_output: str) -> JudgeDecision:
    """Parse strict binary judge JSON and return no-decision on any ambiguity."""

    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return JudgeDecision(
            verdict="no_decision",
            evidence=None,
            rationale=None,
            raw_output=raw_output,
            parser_status="parse_error",
            error_code="invalid_json",
        )
    if not isinstance(payload, dict):
        return JudgeDecision(
            verdict="no_decision",
            evidence=None,
            rationale=None,
            raw_output=raw_output,
            parser_status="parse_error",
            error_code="not_an_object",
        )
    verdict = payload.get("verdict")
    evidence = payload.get("evidence")
    rationale = payload.get("rationale")
    if verdict not in {"pass", "fail"}:
        return JudgeDecision(
            verdict="no_decision",
            evidence=evidence if isinstance(evidence, str) else None,
            rationale=rationale if isinstance(rationale, str) else None,
            raw_output=raw_output,
            parser_status="parse_error",
            error_code="invalid_verdict",
        )
    if not isinstance(evidence, str) or not evidence.strip():
        return JudgeDecision(
            verdict="no_decision",
            evidence=None,
            rationale=rationale if isinstance(rationale, str) else None,
            raw_output=raw_output,
            parser_status="parse_error",
            error_code="missing_evidence",
        )
    if not isinstance(rationale, str) or not rationale.strip():
        return JudgeDecision(
            verdict="no_decision",
            evidence=evidence,
            rationale=None,
            raw_output=raw_output,
            parser_status="parse_error",
            error_code="missing_rationale",
        )
    return JudgeDecision(
        verdict=verdict,
        evidence=evidence,
        rationale=rationale,
        raw_output=raw_output,
        parser_status="parsed",
    )


def _pass_probability(theta: float, item: CalibratedItem) -> float:
    return float(expit(item.discrimination * theta - item.difficulty))


def _information(theta: float, item: CalibratedItem) -> float:
    probability = _pass_probability(theta=theta, item=item)
    return item.discrimination**2 * probability * (1.0 - probability)


def _select_next(
    theta: float,
    items: tuple[CalibratedItem, ...],
    used_scenarios: set[str],
) -> tuple[CalibratedItem, list[dict[str, float | str]]]:
    candidates = [item for item in items if item.scenario_id not in used_scenarios]
    if not candidates:
        raise ValueError("cannot select from an exhausted bank")
    ranked = sorted(
        ((item, _information(theta=theta, item=item)) for item in candidates),
        key=lambda pair: (-pair[1], pair[0].scenario_id),
    )
    trace = [
        {"scenario_id": item.scenario_id, "information": float(value)} for item, value in ranked
    ]
    return ranked[0][0], trace


def _online_update(
    theta: float,
    variance: float,
    item: CalibratedItem,
    y: int,
) -> tuple[float, float, float]:
    probability = _pass_probability(theta=theta, item=item)
    information = _information(theta=theta, item=item)
    new_variance = 1.0 / (1.0 / variance + information)
    new_theta = theta + new_variance * item.discrimination * (float(y) - probability)
    return float(new_theta), float(new_variance), probability


def _normal_quadrature(nodes: int, bound: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-bound, bound, nodes)
    spacing = float(grid[1] - grid[0])
    integration = np.full(nodes, spacing)
    integration[[0, -1]] *= 0.5
    log_prior = -0.5 * grid**2 - 0.5 * math.log(2.0 * math.pi) + np.log(integration)
    log_prior -= logsumexp(log_prior)
    return grid, log_prior


def _batch_eap(
    observations: list[tuple[CalibratedItem, int]],
    nodes: int,
    bound: float,
) -> AbilityEstimate:
    grid, log_prior = _normal_quadrature(nodes=nodes, bound=bound)
    joint = log_prior.copy()
    for item, y in observations:
        eta = item.discrimination * grid - item.difficulty
        joint += log_expit(eta) if y == 1 else log_expit(-eta)
    posterior = np.exp(joint - logsumexp(joint))
    theta = float(posterior @ grid)
    variance = float(posterior @ ((grid - theta) ** 2))
    return AbilityEstimate(
        theta=theta,
        se=math.sqrt(max(variance, 0.0)),
        n_items=len(observations),
        converged=True,
        method="batch_eap",
    )


def _mwle_objective(
    theta_array: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    response: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray]:
    theta = float(theta_array[0])
    eta = discrimination * theta - difficulty
    probability = expit(eta)
    weight = probability * (1.0 - probability)
    information = float(np.sum(discrimination**2 * weight) + ridge)
    if not math.isfinite(information) or information <= 0:
        return 1e100, np.zeros(1)
    log_likelihood = float(response @ log_expit(eta) + (1.0 - response) @ log_expit(-eta))
    objective = log_likelihood + 0.5 * math.log(information)
    adjustment = float(np.sum(discrimination**3 * weight * (1.0 - 2.0 * probability)) / information)
    gradient = float(discrimination @ (response - probability)) + 0.5 * adjustment
    return -objective, np.asarray([-gradient])


def _mwle(
    observations: list[tuple[CalibratedItem, int]],
    theta0: float,
    ridge: float,
) -> AbilityEstimate:
    if not observations:
        return AbilityEstimate(
            theta=None,
            se=None,
            n_items=0,
            converged=False,
            method="mwle",
            message="no observed judgments",
        )
    discrimination = np.asarray([item.discrimination for item, _ in observations])
    difficulty = np.asarray([item.difficulty for item, _ in observations])
    response = np.asarray([float(y) for _, y in observations])
    result = minimize(
        _mwle_objective,
        np.asarray([theta0]),
        args=(discrimination, difficulty, response, ridge),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-12.0, 12.0)],
    )
    theta = float(result.x[0])
    probability = expit(discrimination * theta - difficulty)
    information = float(np.sum(discrimination**2 * probability * (1.0 - probability)) + ridge)
    return AbilityEstimate(
        theta=theta,
        se=math.sqrt(1.0 / information),
        n_items=len(observations),
        converged=bool(result.success and math.isfinite(theta)),
        method="mwle",
        message="" if result.success else str(result.message),
    )


async def _generate_one(
    provider: InferenceProvider,
    request: LMRequest,
    sampling_params: SamplingParams,
) -> LMOutput:
    outputs = await provider.agenerate(requests=[request], sampling_params=sampling_params)
    if len(outputs) != 1 or len(outputs[0]) != 1:
        raise RuntimeError("adaptive spike requires exactly one output per request")
    return outputs[0][0]


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def adaptive_artifact_directory(
    output_dir: str | Path,
    config: AdaptiveRunConfig,
) -> Path:
    """Return the collision-safe sidecar directory for one evaluation run."""

    eval_component = quote(config.eval_name, safe="._-")
    run_component = quote(config.run_id, safe="._-")
    root = Path(output_dir)
    destination = root / eval_component / run_component
    if not destination.resolve().is_relative_to(root.resolve()):  # pragma: no cover
        raise ValueError("adaptive artifact namespace escaped its output directory")
    return destination


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    )


async def run_adaptive_spike(
    context: ExternalEvalContext,
    items: tuple[CalibratedItem, ...],
    config: AdaptiveRunConfig,
    output_dir: str | Path | None = None,
) -> ExternalEvalResult:
    """Run a sequential tutor -> judge -> estimate -> select loop."""

    if len({item.scenario_id for item in items}) != len(items):
        raise ValueError("fixture requires one uniquely identified criterion per scenario")
    if config.max_scenarios > len(items):
        raise ValueError("max_scenarios exceeds the calibrated bank")

    candidate = context.provider
    judge = context.get_provider(config.judge_provider_name)
    candidate_sampling = SamplingParams(max_tokens=128, temperature=0.0, num_samples=1)
    judge_sampling = SamplingParams(max_tokens=128, temperature=0.0, num_samples=1)

    online_theta = 0.0
    online_variance = 1.0
    used_scenarios: set[str] = set()
    observations: list[tuple[CalibratedItem, int]] = []
    steps: list[dict[str, Any]] = []
    stop_reason = "max_scenarios"

    eap = _batch_eap(
        observations=observations,
        nodes=config.quadrature_nodes,
        bound=config.quadrature_bound,
    )
    if eap.theta is None or eap.se is None:  # pragma: no cover - EAP is always defined
        raise RuntimeError("batch EAP unexpectedly returned no estimate")
    mwle = _mwle(observations=observations, theta0=eap.theta, ridge=config.mwle_ridge)

    for step_index in range(config.max_scenarios):
        item, candidates = _select_next(
            theta=online_theta,
            items=items,
            used_scenarios=used_scenarios,
        )
        used_scenarios.add(item.scenario_id)
        theta_before = online_theta
        variance_before = online_variance
        observed_before = len(observations)

        tutor_request = LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=(
                f"SCENARIO_ID: {item.scenario_id}\n"
                f"CRITERION_ID: {item.criterion_id}\n\n{item.prompt}"
            ),
        )
        tutor_output = await _generate_one(
            provider=candidate,
            request=tutor_request,
            sampling_params=candidate_sampling,
        )
        judge_request = LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=(
                "Decide whether the tutor response satisfies exactly one criterion. "
                "Return strict JSON with verdict, evidence, and rationale.\n\n"
                f"SCENARIO_ID: {item.scenario_id}\n"
                f"CRITERION_ID: {item.criterion_id}\n"
                f"CRITERION: {item.criterion}\n"
                f"TUTOR_RESPONSE: {tutor_output.text}"
            ),
        )
        judge_output = await _generate_one(
            provider=judge,
            request=judge_request,
            sampling_params=judge_sampling,
        )
        decision = parse_binary_judgment(judge_output.text)

        pre_update_probability = _pass_probability(theta=online_theta, item=item)
        if decision.y is not None:
            online_theta, online_variance, pre_update_probability = _online_update(
                theta=online_theta,
                variance=online_variance,
                item=item,
                y=decision.y,
            )
            observations.append((item, decision.y))

        eap = _batch_eap(
            observations=observations,
            nodes=config.quadrature_nodes,
            bound=config.quadrature_bound,
        )
        if eap.theta is None or eap.se is None:  # pragma: no cover - EAP is always defined
            raise RuntimeError("batch EAP unexpectedly returned no estimate")
        mwle = _mwle(
            observations=observations,
            theta0=eap.theta,
            ridge=config.mwle_ridge,
        )
        steps.append(
            {
                "step": step_index + 1,
                "scenario_id": item.scenario_id,
                "criterion_id": item.criterion_id,
                "selection_candidates": candidates,
                "selection_theta": theta_before,
                "selection_variance": variance_before,
                "pre_update_pass_probability": pre_update_probability,
                "tutor_request": tutor_request.prompt,
                "raw_tutor_output": tutor_output.text,
                "judge_request": judge_request.prompt,
                "judgment": decision.to_dict(),
                "online_theta_after": online_theta,
                "online_variance_after": online_variance,
                "eap_after": eap.to_dict(),
                "mwle_after": mwle.to_dict(),
                "observed_before": observed_before,
                "observed_after": len(observations),
            }
        )

        if (
            config.stop_eap_se is not None
            and len(steps) >= config.min_scenarios
            and eap.se <= config.stop_eap_se
        ):
            stop_reason = "eap_posterior_se"
            break
        if len(used_scenarios) == len(items):
            stop_reason = "bank_exhausted"
            break

    bank_payload = [item.to_dict() for item in items]
    bank_hash = _canonical_hash(bank_payload)
    coverage = len(observations) / len(steps) if steps else 0.0
    artifact_namespace = str(adaptive_artifact_directory(Path(), config=config))
    artifact_names = (
        "manifest.json",
        "calibrated_bank.json",
        "steps.jsonl",
        "judgments.jsonl",
        "final_result.json",
    )
    manifest = {
        "schema_version": SPIKE_SCHEMA_VERSION,
        "eval_name": config.eval_name,
        "run_id": config.run_id,
        "candidate_model": candidate.model_name,
        "judge_model": judge.model_name,
        "judge_provider_name": config.judge_provider_name,
        "calibrated_bank_hash": bank_hash,
        "artifact_namespace": artifact_namespace,
        "config": {
            "max_scenarios": config.max_scenarios,
            "min_scenarios": config.min_scenarios,
            "stop_eap_se": config.stop_eap_se,
            "quadrature_nodes": config.quadrature_nodes,
            "quadrature_bound": config.quadrature_bound,
            "mwle_ridge": config.mwle_ridge,
        },
    }
    metadata = {
        **manifest,
        "num_tasks": len(steps),
        "stop_reason": stop_reason,
        "scenarios_administered": [step["scenario_id"] for step in steps],
        "criteria_observed": len(observations),
        "criteria_no_decision": len(steps) - len(observations),
        "final_online_theta": online_theta,
        "final_online_se": math.sqrt(online_variance),
        "final_eap": eap.to_dict(),
        "final_mwle": mwle.to_dict(),
        "artifacts": [f"{artifact_namespace}/{name}" for name in artifact_names],
    }
    metrics: dict[str, float] = {
        "coverage": float(coverage),
        "scenarios_administered": float(len(steps)),
        "criteria_observed": float(len(observations)),
        "eap_theta": eap.theta,
        "eap_se": eap.se,
        "mwle_converged": float(mwle.converged),
    }
    if mwle.theta is not None:
        metrics["mwle_theta"] = mwle.theta
    if mwle.se is not None:
        metrics["mwle_se"] = mwle.se
    result = ExternalEvalResult(
        name=config.eval_name,
        metrics=metrics,
        metadata=metadata,
        predictions=steps,
    )

    if output_dir is not None:
        destination = adaptive_artifact_directory(output_dir, config=config)
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "manifest.json", manifest)
        _write_json(destination / "calibrated_bank.json", bank_payload)
        _write_jsonl(destination / "steps.jsonl", steps)
        _write_jsonl(
            destination / "judgments.jsonl",
            [
                {
                    "step": step["step"],
                    "scenario_id": step["scenario_id"],
                    "criterion_id": step["criterion_id"],
                    **step["judgment"],
                }
                for step in steps
            ],
        )
        _write_json(destination / "final_result.json", result.to_dict())
    return result
