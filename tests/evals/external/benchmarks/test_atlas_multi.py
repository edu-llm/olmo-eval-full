"""Tests for the config-driven online ATLAS external evals (multi-benchmark)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from olmo_eval.adaptive.benchmarks import get_benchmark
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.evals.external.benchmarks.atlas.eval import AtlasExternalEval
from olmo_eval.evals.external.registry import get_external_eval, list_external_evals
from olmo_eval.evals.tasks.common import get_task

_MCQ_EVALS = ["atlas_hellaswag", "atlas_winogrande", "atlas_csqa", "atlas_piqa"]


def _write_bank(tmp_path: Path, n: int) -> Path:
    with open(tmp_path / "irt_item_parameters_combined.csv", "w", newline="") as fh:
        fh.write("X,a1,d,g,u\n")
        for i in range(1, n + 1):
            fh.write(f"X{i},1.3,{0.6 - i * 0.05:.3f},0.2,1\n")
    with open(tmp_path / "atlas_idx_to_question_id.csv", "w", newline="") as fh:
        fh.write("atlas_idx,question_id\n")
        for i in range(1, n + 1):
            fh.write(f"{i},q{i - 1}\n")
    return tmp_path


class _FakeTask:
    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="",
            continuations=(" a", " b"),
        )


class _FakeProvider:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def alogprobs(self, requests: list[LMRequest], sampling_params: Any = None):
        self.calls.append(requests)
        return [
            [
                LMOutput(text="a", logprobs=[{"token": "a", "logprob": -1.0}]),
                LMOutput(text="b", logprobs=[{"token": "b", "logprob": -2.0}]),
            ]
        ]


def _items(n: int) -> dict[str, Instance]:
    return {
        f"q{i}": Instance(
            question="q",
            choices=("a", "b"),
            gold_answer="A",
            metadata={"id": f"q{i}", "gold_idx": i % 2},
        )
        for i in range(n)
    }


@pytest.mark.parametrize("eval_name", [*_MCQ_EVALS, "atlas_gsm8k"])
def test_registered(eval_name: str) -> None:
    ev = get_external_eval(eval_name)
    assert ev.name == eval_name


def test_full_atlas_eval_set() -> None:
    atlas = {e for e in list_external_evals() if e.startswith("atlas")}
    assert atlas == {"atlas_arc", "atlas_gsm8k", *_MCQ_EVALS}


@pytest.mark.parametrize("bench", ["hellaswag", "winogrande", "csqa", "piqa"])
def test_online_cat_only_queries_selected_items(bench: str, tmp_path: Path) -> None:
    bank_dir = _write_bank(tmp_path, n=30)
    ev = AtlasExternalEval(get_benchmark(bench))
    provider = _FakeProvider()
    ev._load_items = lambda args: (_FakeTask(), _items(30))  # type: ignore[method-assign]

    result = asyncio.run(
        ev.execute(
            provider,
            {"bank_path": str(bank_dir), "se_stop": 0.0, "min_items": 8, "max_items": 15},
        )
    )

    assert result.success
    assert result.metrics["n_items"] == 15.0
    assert result.metrics["bank_size"] == 30.0
    assert len(provider.calls) == 15  # one provider call per administered item
    assert result.metadata["benchmark"] == bench
    selected = result.metadata["selected_question_ids"]
    assert len(selected) == len(set(selected))  # never re-administered


def test_missing_bank_returns_error_result(tmp_path: Path) -> None:
    """Banks arrive via S3 at run time; an absent bank yields a clean error."""
    ev = AtlasExternalEval(get_benchmark("hellaswag"))
    result = asyncio.run(
        ev.execute(_FakeProvider(), {"bank_path": str(tmp_path / "not_synced_yet")})
    )
    assert not result.success
    assert result.error is not None
    assert "bank unavailable" in result.error


class _GenProvider:
    """Fake provider whose generation always returns the same completion text."""

    def __init__(self, text: str = " so the answer is 42") -> None:
        self.text = text
        self.gen_calls: list[Any] = []

    async def agenerate(self, requests: list[LMRequest], sampling_params: Any = None):
        self.gen_calls.append(requests)
        return [[LMOutput(text=self.text)] for _ in requests]


def _gsm8k_items(n: int) -> dict[str, Instance]:
    # Even ids have gold 42 (match the provider text), odd ids have gold 7.
    return {
        f"q{i}": Instance(
            question="what is 40 + 2?",
            gold_answer="42" if i % 2 == 0 else "7",
            metadata={"id": f"q{i}"},
        )
        for i in range(n)
    }


def test_gsm8k_generative_scores_per_item() -> None:
    """The generative scorer reuses gsm8k's own extraction + exact match: 1 vs 0."""
    ev = AtlasExternalEval(get_benchmark("gsm8k"))
    task = get_task("gsm8k")  # real base task: format/extract/score, no network
    provider = _GenProvider(" ...so the final answer is 42")

    correct = Instance(question="q", gold_answer="42", metadata={"id": "q0"})
    wrong = Instance(question="q", gold_answer="7", metadata={"id": "q1"})
    assert asyncio.run(ev._score_item(task, provider, correct)) == 1
    assert asyncio.run(ev._score_item(task, provider, wrong)) == 0
    assert provider.gen_calls  # used the generation API, not alogprobs


def test_gsm8k_online_cat_runs_and_stops(tmp_path: Path) -> None:
    """The generative CAT loop administers items one by one and stops at max_items."""
    bank_dir = _write_bank(tmp_path, n=30)
    ev = AtlasExternalEval(get_benchmark("gsm8k"))
    task = get_task("gsm8k")
    provider = _GenProvider(" ...so the final answer is 42")
    ev._load_items = lambda args: (task, _gsm8k_items(30))  # type: ignore[method-assign]

    result = asyncio.run(
        ev.execute(
            provider,
            {"bank_path": str(bank_dir), "se_stop": 0.0, "min_items": 8, "max_items": 15},
        )
    )

    assert result.success
    assert result.metrics["n_items"] == 15.0
    assert result.metrics["bank_size"] == 30.0
    assert len(provider.gen_calls) == 15  # one generation per administered item
    assert result.metadata["benchmark"] == "gsm8k"
    scores = [p["score"] for p in result.predictions]
    assert set(scores) <= {0, 1}  # binary per-item correctness
    assert any(s == 1 for s in scores) and any(s == 0 for s in scores)
