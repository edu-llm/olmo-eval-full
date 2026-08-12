"""MRBench Phase-B: score a model-under-test as a tutor, then judge on 8 dims.

This is the *generation* half of MRBench (see ``Plan/mrbench/README.md``): for
each MRBench conversation the model under test (``olmo-eval run -m <model>``)
generates a one-sentence tutor remediation using the paper's Figure-2 template;
each generation is then judged on the eight pedagogical dimensions with the
paper's Figure-6 protocol, and per-dimension + aggregate **DAMR** (Desired
Annotation Match Rate) are reported.

Phase A (validating the judge against human gold) stays standalone in
``diagnostics/mrbench``; this task deliberately reuses that package as the single
source of truth for the judge and DAMR arithmetic — the Figure-6 prompt
(``diagnostics.mrbench.judge_prompt``), the ``[RESULT] N`` parser
(``diagnostics.mrbench.parse``), the TrueFoundry-compatible OpenAI client
(``diagnostics.mrbench.judge_client``), and the desired-label rule
(``diagnostics.mrbench.reference.DESIRED_LABELS``).

Judge-agnostic: the judge model, gateway base URL, and API key are all read from
the environment by ``judge_client`` (never hardcoded); there is no usable default
judge model, so a live run requires the user to supply ``MRBENCH_JUDGE_MODEL``.

Usage:
    olmo-eval run -m <model-under-test> -t mrbench            # + --store/--s3-*

Cross-package import note: ``diagnostics`` lives at the repo root, not under
``src/``. When ``olmo-eval`` runs as a console script the repo root is not on
``sys.path``, so this module puts it there (computed from ``__file__``). This
keeps a single source of truth for the judge/DAMR logic instead of duplicating
it. If ``olmo_eval`` were ever installed as a standalone wheel without the
``diagnostics`` tree beside it, the task registers but refuses to score with a
clear error (see ``score_responses``); it never breaks task discovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
)
from olmo_eval.evals.tasks.common import Task, register

logger = logging.getLogger(__name__)

# --- Bridge to the repo-root ``diagnostics`` package ----------------------- #
# <root>/src/olmo_eval/evals/tasks/mrbench.py -> parents[4] == <root>.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MRBENCH_DATA_PATH = _REPO_ROOT / "diagnostics" / "mrbench" / "data" / "MRBench_V1.json"

# Env var that, when truthy, makes ``score_responses`` skip the judge entirely
# (no diagnostics import, no API, no network) so a run only emits its generations
# for off-cluster judging. Default OFF => byte-identical to the inline-judge path.
_GENERATE_ONLY_ENV = "MRBENCH_GENERATE_ONLY"
# Env var override for the data file, complementing ``-o data_source=``.
_DATA_SOURCE_ENV = "MRBENCH_DATA_SOURCE"


def _env_truthy(value: str | None) -> bool:
    """Return True for the usual affirmative env-var spellings, False otherwise."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


# Authoritative dimension order is ``diagnostics.mrbench.reference.DIMENSIONS``.
# Imported here so the 8 per-dimension metrics can be declared at registration
# time. A small hard-coded mirror keeps task *discovery* resilient if the
# diagnostics tree is absent; ``score_responses`` re-imports and would raise
# before doing anything, and a consistency check guards against drift.
_FALLBACK_DIMENSIONS: tuple[str, ...] = (
    "Mistake_Identification",
    "Mistake_Location",
    "Revealing_of_the_Answer",
    "Providing_Guidance",
    "Actionability",
    "Coherence",
    "Tutor_Tone",
    "Humanlikeness",
)
try:
    from diagnostics.mrbench import reference as _reference

    DIMENSIONS: tuple[str, ...] = tuple(_reference.DIMENSIONS)
    _DIAGNOSTICS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # keep task discovery bulletproof
    DIMENSIONS = _FALLBACK_DIMENSIONS
    _DIAGNOSTICS_IMPORT_ERROR = exc

# --- Figure-2 tutor-generation prompt (paper Appendix B, byte-faithful) ---- #
# This is the *tutor-under-test* prompt, independent of whatever model later acts
# as judge. The assistant response cue is folded onto the user turn (same
# convention as the Figure-6 judge prompt). NOTE: the MathDial cue's "hisotry" is
# a verbatim paper typo, preserved intentionally.
_GEN_SYSTEM_BRIDGE = (
    "You are an experienced elementary school math teacher, and you are going to "
    "respond to a student's mistake in a useful and caring way"
)
_GEN_SYSTEM_MATHDIAL = (
    "You are an experienced middle school math teacher, and you are going to "
    "respond to a student's mistake in a useful and caring way"
)
_GEN_USER_BRIDGE = (
    "The problem your student is solving is on the topic: {topic}\n"
    "Conversation History: {history}\n"
    "Tutor response (maximum one sentence that is most appropriate given topic "
    "and conversation history):"
)
_GEN_USER_MATHDIAL = (
    "The conversation history of the problem your student is solving is: {history}\n"
    "Tutor response (maximum one sentence that aligns most appropriately with "
    "conversation hisotry):"
)

# Metric names.
AGGREGATE_METRIC_NAME = "damr"


def _dim_metric_name(dimension: str) -> str:
    return f"damr_{dimension}"


def build_generation_messages(
    data_source: str, history: str, topic: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the (system, user) Figure-2 messages for one conversation.

    ``Bridge`` (elementary) conditions on ``{topic}``; ``MathDial`` (middle
    school) has no topic. Anything that is not ``Bridge`` uses the MathDial form.
    """
    if data_source == "Bridge":
        system = _GEN_SYSTEM_BRIDGE
        user = _GEN_USER_BRIDGE.format(topic=topic, history=history)
    else:
        system = _GEN_SYSTEM_MATHDIAL
        user = _GEN_USER_MATHDIAL.format(history=history)
    return ({"role": "system", "content": system}, {"role": "user", "content": user})


# --- Metrics --------------------------------------------------------------- #
@dataclass(frozen=True)
class MRBenchScorer(Scorer):
    """Placeholder scorer; DAMR is precomputed by the task in ``score_responses``."""

    name: str = "mrbench"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return (output.metadata or {}).get(f"score:{AGGREGATE_METRIC_NAME}", 0.0)


@dataclass(frozen=True)
class MRBenchDamrMetric(Metric):
    """Mean desired-annotation match rate over responses for one score key.

    ``name`` selects which precomputed value in ``response.scores`` to average:
    ``"damr"`` for the aggregate, ``"damr_<Dimension>"`` for a single dimension.
    Values are 0/1 indicators per response, so the mean is a proportion in [0, 1].
    """

    name: str = AGGREGATE_METRIC_NAME
    scorer: type[Scorer] | Scorer = MRBenchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        return sum(r.scores.get(self.name, 0.0) for r in responses) / len(responses)

    def pairwise_display_format(self) -> str:
        return "percentage"

    def pairwise_unit(self) -> str:
        return "proportion"


_AGGREGATE_METRIC = MRBenchDamrMetric(name=AGGREGATE_METRIC_NAME)
_MRBENCH_METRICS: tuple[Metric, ...] = (
    _AGGREGATE_METRIC,
    *(MRBenchDamrMetric(name=_dim_metric_name(dim)) for dim in DIMENSIONS),
)


# --- Task ------------------------------------------------------------------ #
@register("mrbench")
class MRBench(Task):
    """MRBench Phase-B tutor-generation task, judged on 8 dimensions -> DAMR."""

    # A filesystem path to the vendored MRBench V1 data (not re-downloaded).
    # Overridable with ``-o data_source=/path/to/MRBench_V1.json`` or the
    # ``MRBENCH_DATA_SOURCE`` env var; falls back to this checkout path otherwise.
    data_source = str(MRBENCH_DATA_PATH)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    metrics = _MRBENCH_METRICS
    primary_metric = _AGGREGATE_METRIC
    # The gateway key the judge needs; beaker mounts it as a user-scoped secret.
    # Aliases TFY_API_KEY / TRUEFOUNDRY_API_KEY are also accepted by judge_client,
    # which additionally needs MRBENCH_JUDGE_MODEL (and optionally a base-URL env).
    required_secrets = ("OPENAI_API_KEY",)

    def _resolve_data_path(self) -> Path:
        """Resolve MRBench_V1.json with explicit-override-then-checkout precedence.

        Works both for a source checkout (the file ships beside the ``diagnostics``
        tree) and an installed wheel (where the ``__file__``-relative path points
        into site-packages and does not exist): supply an explicit path via
        ``-o data_source=/abs/MRBench_V1.json`` or the ``MRBENCH_DATA_SOURCE`` env
        var. The checkout path stays the default fallback and is never hardcoded to
        a single deployment location.
        """
        source = self.config.data_source
        if isinstance(source, str) and source and Path(source) != MRBENCH_DATA_PATH:
            return Path(source)
        env_override = os.getenv(_DATA_SOURCE_ENV)
        if env_override:
            return Path(env_override)
        return MRBENCH_DATA_PATH

    def _load_conversations(self) -> list[dict[str, Any]]:
        """Load the MRBench V1 conversation list from the resolved path."""
        path = self._resolve_data_path()
        if not path.exists():
            raise FileNotFoundError(
                f"MRBench data not found at {path}. Pass an explicit path with "
                "-o data_source=/abs/path/MRBench_V1.json or set "
                f"{_DATA_SOURCE_ENV} (required when running from an installed wheel "
                "where the checkout-relative path is unavailable)."
            )
        with path.open(encoding="utf-8") as fh:
            conversations = json.load(fh)
        if not isinstance(conversations, list):
            raise ValueError(f"Expected a JSON list of conversations, got {type(conversations)}.")
        return conversations

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            cache: list[Instance] = []
            for index, doc in enumerate(self._load_conversations()):
                instance = self.process_doc(doc, index)
                if instance is not None:
                    cache.append(instance)
            self._instances_cache = cache
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        history = doc.get("conversation_history", "")
        if not history:
            return None
        data_source = doc.get("Data", "MathDial")
        topic = doc.get("Topic", "")
        conversation_id = str(doc.get("conversation_id", f"mrbench_{index}"))
        _system, user = build_generation_messages(data_source, history, topic)
        return Instance(
            question=user["content"],
            metadata={
                "id": conversation_id,
                "conversation_id": conversation_id,
                "data_source": data_source,
                "topic": topic,
                "conversation_history": history,
                "ground_truth_solution": doc.get("Ground_Truth_Solution", ""),
                "index": index,
            },
        )

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT

    def format_request(self, instance: Instance) -> LMRequest:
        system, user = build_generation_messages(
            instance.metadata.get("data_source", "MathDial"),
            instance.metadata.get("conversation_history", ""),
            instance.metadata.get("topic", ""),
        )
        return LMRequest(request_type=RequestType.CHAT, messages=(system, user))

    async def score_responses(
        self,
        responses: Sequence[Response],
        context: Any = None,
    ) -> Sequence[Response]:
        """Judge every generation on all 8 dimensions and store per-dim + aggregate DAMR.

        Reuses ``diagnostics.mrbench`` for the judge call and desired-label rule.
        A parse failure counts as "desired not met" (indicator 0) and is recorded,
        mirroring the conservative handling in the standalone pipeline.
        """
        self._extract_answers(responses)
        if not responses:
            return responses

        # OPT-IN, default OFF: generate-only. When set, skip the judge entirely
        # (no diagnostics import, no API, no network). The runner still writes the
        # predictions (native_id = conversation_id, final_output = tutor text), so
        # the generations can be judged off-cluster later (see
        # diagnostics/mrbench/phase_b_judge.py). Unset => behaviour is identical to
        # the inline-judge path below.
        if _env_truthy(os.getenv(_GENERATE_ONLY_ENV)):
            logger.info(
                "%s is set: skipping the MRBench judge; %d generations emitted for "
                "off-cluster judging.",
                _GENERATE_ONLY_ENV,
                len(responses),
            )
            return responses

        if _DIAGNOSTICS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "MRBench scoring requires the 'diagnostics.mrbench' package at the repo "
                f"root ({_REPO_ROOT}); it could not be imported."
            ) from _DIAGNOSTICS_IMPORT_ERROR

        from diagnostics.mrbench import judge_client, judge_prompt
        from diagnostics.mrbench.parse import aggregate_samples, parse_result
        from diagnostics.mrbench.reference import DESIRED_LABELS
        from diagnostics.mrbench.reference import DIMENSIONS as REF_DIMENSIONS

        if tuple(REF_DIMENSIONS) != DIMENSIONS:
            raise RuntimeError(
                "MRBench dimension list drifted from diagnostics.mrbench.reference; "
                "regenerate the metric set."
            )

        # OPT-IN, default OFF: reference-guided judging + self-consistency (k).
        # With both env vars unset this path is byte-identical to the paper default
        # (single Figure-6 call parsed by parse_result, no reference injection).
        reference_guided = judge_prompt.reference_guided_default()
        k = max(1, int(os.getenv("MRBENCH_JUDGE_SAMPLES", "1")))

        config = judge_client.JudgeConfig.from_env()
        client = judge_client.make_client(config)
        concurrency = max(1, int(os.getenv("MRBENCH_JUDGE_CONCURRENCY", "8")))
        semaphore = asyncio.Semaphore(concurrency)

        async def judge_one(history: str, answer: str, solution: str, dimension: str) -> Any:
            messages = judge_prompt.build_messages(
                history,
                answer,
                dimension,
                reference_solution=solution,
                reference_guided=reference_guided,
            )
            async with semaphore:
                if k == 1:
                    raw = await asyncio.to_thread(
                        judge_client.score_messages, client, config, messages
                    )
                    return parse_result(raw, dimension), None
                raws = [
                    await asyncio.to_thread(judge_client.score_messages, client, config, messages)
                    for _ in range(k)
                ]
            parsed, per_sample = aggregate_samples(raws, dimension)
            return parsed, [s.raw for s in per_sample]

        jobs: list[Any] = []
        job_index: list[tuple[int, str]] = []
        for resp_idx, response in enumerate(responses):
            history = response.instance.metadata.get("conversation_history", "")
            solution = response.instance.metadata.get("ground_truth_solution", "")
            answer = self._response_text(response)
            for dimension in DIMENSIONS:
                jobs.append(judge_one(history, answer, solution, dimension))
                job_index.append((resp_idx, dimension))

        results = await asyncio.gather(*jobs)

        parsed_by_response: dict[int, dict[str, Any]] = {}
        for (resp_idx, dimension), result in zip(job_index, results, strict=True):
            parsed_by_response.setdefault(resp_idx, {})[dimension] = result

        for resp_idx, response in enumerate(responses):
            parsed_by_dim = parsed_by_response.get(resp_idx, {})
            indicators: list[float] = []
            judgements: dict[str, dict[str, Any]] = {}
            for dimension in DIMENSIONS:
                result = parsed_by_dim.get(dimension)
                parsed = None if result is None else result[0]
                samples = None if result is None else result[1]
                desired = bool(
                    parsed is not None and parsed.ok and parsed.label == DESIRED_LABELS[dimension]
                )
                indicator = 1.0 if desired else 0.0
                indicators.append(indicator)
                response.scores[_dim_metric_name(dimension)] = indicator
                judgements[dimension] = {
                    "score": None if parsed is None else parsed.score,
                    "label": None if parsed is None else parsed.label,
                    "ok": bool(parsed is not None and parsed.ok),
                    "desired": desired,
                    "samples": samples,
                }
            aggregate = sum(indicators) / len(indicators) if indicators else 0.0
            response.scores[AGGREGATE_METRIC_NAME] = aggregate

            output = response.outputs[0] if response.outputs else None
            if output is not None:
                if output.metadata is None:
                    output.metadata = {}
                output.metadata["mrbench_judgements"] = judgements
                output.metadata["mrbench_judge_model"] = config.model
                output.metadata[f"score:{AGGREGATE_METRIC_NAME}"] = aggregate
                for dimension in DIMENSIONS:
                    key = _dim_metric_name(dimension)
                    output.metadata[f"score:{key}"] = response.scores[key]
        return responses

    @staticmethod
    def _response_text(response: Response) -> str:
        output = response.outputs[0] if response.outputs else None
        if output is None:
            return ""
        if isinstance(output.extracted_answer, str):
            return output.extracted_answer
        return output.text or ""
