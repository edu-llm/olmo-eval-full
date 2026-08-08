"""Tests for the online ATLAS adaptive-testing external eval (Phase 2)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.evals.external.benchmarks.atlas_arc.eval import AtlasArcExternalEval
from olmo_eval.evals.external.registry import get_external_eval


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
    """Minimal stand-in that formats a two-choice loglik request."""

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="",
            continuations=(" a", " b"),
        )


class _FakeProvider:
    """Always prefers continuation 0; correctness is decided by gold_idx."""

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


def test_registered() -> None:
    assert get_external_eval("atlas_arc").name == "atlas_arc"


def test_online_cat_only_queries_selected_items(tmp_path: Path) -> None:
    bank_dir = _write_bank(tmp_path, n=30)
    ev = AtlasArcExternalEval()
    provider = _FakeProvider()
    ev._load_items = lambda args: (_FakeTask(), _items(30))  # type: ignore[method-assign]

    result = asyncio.run(
        ev.execute(
            provider,
            {"bank_path": str(bank_dir), "se_stop": 0.0, "min_items": 8, "max_items": 15},
        )
    )

    assert result.success
    # se_stop=0 forces the loop to exhaust max_items, which is < bank size (30):
    # that gap is the inference saving online CAT provides.
    assert result.metrics["n_items"] == 15.0
    assert result.metrics["bank_size"] == 30.0
    # The provider was called exactly once per administered item, never for the rest.
    assert len(provider.calls) == 15
    selected = result.metadata["selected_question_ids"]
    assert len(selected) == len(set(selected))  # no item administered twice
    assert result.predictions is not None
    assert len(result.predictions) == 15


def test_online_cat_stops_on_se_threshold(tmp_path: Path) -> None:
    bank_dir = _write_bank(tmp_path, n=50)
    ev = AtlasArcExternalEval()
    provider = _FakeProvider()
    ev._load_items = lambda args: (_FakeTask(), _items(50))  # type: ignore[method-assign]

    result = asyncio.run(
        ev.execute(
            provider,
            {"bank_path": str(bank_dir), "se_stop": 0.3, "min_items": 8, "max_items": 200},
        )
    )
    assert result.success
    assert result.metrics["n_items"] >= 8
    assert result.metrics["se"] <= 0.3 or result.metrics["n_items"] == 50.0


def test_no_bank_overlap_returns_error(tmp_path: Path) -> None:
    bank_dir = _write_bank(tmp_path, n=5)
    ev = AtlasArcExternalEval()
    ev._load_items = lambda args: (_FakeTask(), _items(0))  # type: ignore[method-assign]
    result = asyncio.run(ev.execute(_FakeProvider(), {"bank_path": str(bank_dir)}))
    assert not result.success
    assert result.error is not None
