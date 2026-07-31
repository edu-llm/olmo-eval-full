"""Tests for the config-driven offline ATLAS tasks (multi-benchmark, Phase 3)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from olmo_eval.common.metrics import LogprobMCAccuracyMetric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.atlas import AtlasAdaptiveMixin
from olmo_eval.evals.tasks.common import get_task, list_tasks

# The MCQ benchmarks that get an offline ``atlas_<name>`` task (arc excluded here
# because it is exercised by test_atlas_arc.py).
_MCQ_TASKS = ["atlas_hellaswag", "atlas_winogrande", "atlas_csqa", "atlas_piqa"]


def _write_bank(tmp_path: Path, n: int = 12) -> Path:
    params = tmp_path / "irt_item_parameters_combined.csv"
    idx_map = tmp_path / "atlas_idx_to_question_id.csv"
    with open(params, "w", newline="") as fh:
        fh.write("X,a1,d,g,u\n")
        for i in range(1, n + 1):
            fh.write(f"X{i},1.2,{0.5 - i * 0.1:.3f},0.2,1\n")
    with open(idx_map, "w", newline="") as fh:
        fh.write("atlas_idx,question_id\n")
        for i in range(1, n + 1):
            fh.write(f"{i},q{i - 1}\n")
    return tmp_path


def _mc_response(qid: str, correct: bool) -> Response:
    lp0 = -1.0 if correct else -5.0
    lp1 = -5.0 if correct else -1.0
    return Response(
        instance=Instance(
            question="q?",
            choices=("a", "b"),
            gold_answer="A",
            metadata={"id": qid, "gold_idx": 0},
        ),
        request=LMRequest(request_type=RequestType.LOGLIKELIHOOD),
        outputs=[
            LMOutput(text="a", logprobs=[{"token": "a", "logprob": lp0}]),
            LMOutput(text="b", logprobs=[{"token": "b", "logprob": lp1}]),
        ],
    )


@pytest.mark.parametrize("task_name", _MCQ_TASKS)
def test_task_registers_with_mixin(task_name: str) -> None:
    task = get_task(task_name)
    assert isinstance(task, AtlasAdaptiveMixin)
    assert task.atlas_benchmark.offline_task_name == task_name


def test_gsm8k_registered_truthfulqa_not() -> None:
    """gsm8k (generative) is wired; truthfulqa (varies / no base task) is skipped."""
    tasks = set(list_tasks())
    assert "atlas_gsm8k" in tasks
    assert "atlas_truthfulqa" not in tasks


def _gsm8k_response(qid: str, correct: bool) -> Response:
    """A gsm8k-style generation response: text ends in the (in)correct number."""
    text = " ...so the answer is 42" if correct else " ...so the answer is 7"
    return Response(
        instance=Instance(question="q?", gold_answer="42", metadata={"id": qid}),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt=""),
        outputs=[LMOutput(text=text)],
    )


def test_gsm8k_generative_offline_appends_atlas_report(tmp_path: Path) -> None:
    """Offline atlas_gsm8k needs no MCQ scoring: the base task's exact-match 0/1
    (produced by score_responses, as the runner does) flows straight into the CAT."""
    bank_dir = _write_bank(tmp_path, n=12)
    task = get_task("atlas_gsm8k")
    assert isinstance(task, AtlasAdaptiveMixin)
    task.atlas_bank_dir = str(bank_dir)
    task.atlas_se_stop = 0.0
    task.atlas_min_items = 8
    task.atlas_max_items = 12

    responses = [_gsm8k_response(f"q{i}", correct=(i % 2 == 0)) for i in range(12)]
    # The runner scores responses before compute_metrics; do the same so the
    # generative primary metric (exact-match) has per-item 0/1 to read.
    asyncio.run(task.score_responses(responses))
    # Sanity: the base task's own extraction + scorer produced binary correctness.
    assert responses[0].scores["exact_match"] == 1.0
    assert responses[1].scores["exact_match"] == 0.0

    result = task.compute_metrics(responses)
    for key in ("atlas_theta", "atlas_se", "atlas_n_items", "atlas_pirt_accuracy"):
        assert key in result, key
    assert result["atlas_n_items"]["atlas"] == 12.0
    assert 0.0 <= result["atlas_pirt_accuracy"]["atlas"] <= 1.0


def test_gsm8k_missing_bank_degrades_gracefully(tmp_path: Path) -> None:
    task = get_task("atlas_gsm8k")
    assert isinstance(task, AtlasAdaptiveMixin)
    task.atlas_bank_dir = str(tmp_path / "not_synced_yet")

    responses = [_gsm8k_response(f"q{i}", correct=True) for i in range(6)]
    asyncio.run(task.score_responses(responses))
    result = task.compute_metrics(responses)
    assert "accuracy" in result  # base metric preserved
    assert "atlas_theta" not in result  # adaptive report skipped, no exception


@pytest.mark.parametrize("task_name", _MCQ_TASKS)
def test_compute_metrics_appends_atlas_report(task_name: str, tmp_path: Path) -> None:
    bank_dir = _write_bank(tmp_path, n=12)
    task = get_task(task_name)
    assert isinstance(task, AtlasAdaptiveMixin)
    task.atlas_bank_dir = str(bank_dir)
    task.atlas_se_stop = 0.0  # exhaust max_items deterministically
    task.atlas_min_items = 8
    task.atlas_max_items = 12
    # Normalize scoring so the synthetic responses score uniformly across
    # benchmarks; each base task's own metric fidelity is covered elsewhere.
    task.config.metrics = (LogprobMCAccuracyMetric(),)

    responses = [_mc_response(f"q{i}", correct=(i % 2 == 0)) for i in range(12)]
    result = task.compute_metrics(responses)

    for key in ("atlas_theta", "atlas_se", "atlas_n_items", "atlas_pirt_accuracy"):
        assert key in result, key
    assert result["atlas_n_items"]["atlas"] == 12.0
    assert 0.0 <= result["atlas_pirt_accuracy"]["atlas"] <= 1.0
    assert any("restrict12" in k for k in result["atlas_bank_version"])


@pytest.mark.parametrize("task_name", _MCQ_TASKS)
def test_missing_bank_degrades_gracefully(task_name: str, tmp_path: Path) -> None:
    """Banks arrive via S3 at run time; an absent bank must not crash the task."""
    task = get_task(task_name)
    assert isinstance(task, AtlasAdaptiveMixin)
    task.atlas_bank_dir = str(tmp_path / "not_synced_yet")  # no CSVs on disk
    task.config.metrics = (LogprobMCAccuracyMetric(),)

    responses = [_mc_response(f"q{i}", correct=True) for i in range(10)]
    result = task.compute_metrics(responses)

    assert "accuracy" in result  # base metric preserved
    assert "atlas_theta" not in result  # adaptive report skipped, no exception


def test_no_bank_overlap_is_graceful(tmp_path: Path) -> None:
    bank_dir = _write_bank(tmp_path, n=4)
    task = get_task("atlas_hellaswag")
    assert isinstance(task, AtlasAdaptiveMixin)
    task.atlas_bank_dir = str(bank_dir)
    task.config.metrics = (LogprobMCAccuracyMetric(),)

    responses = [_mc_response(f"missing{i}", correct=True) for i in range(3)]
    result = task.compute_metrics(responses)
    assert "accuracy" in result
    assert "atlas_theta" not in result
