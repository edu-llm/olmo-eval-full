"""Tests for the offline ATLAS adaptive-testing task (Phase 1)."""

from __future__ import annotations

from pathlib import Path

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.atlas_arc import AtlasARCChallenge
from olmo_eval.evals.tasks.common import get_task


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
    # Two choices; gold is index 0. Make choice 0 win when `correct`.
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


def test_task_registers() -> None:
    task = get_task("atlas_arc_challenge")
    assert isinstance(task, AtlasARCChallenge)


def test_compute_metrics_appends_atlas_report(tmp_path: Path) -> None:
    bank_dir = _write_bank(tmp_path, n=12)
    task = get_task("atlas_arc_challenge")
    assert isinstance(task, AtlasARCChallenge)
    task.atlas_bank_dir = str(bank_dir)
    task.atlas_se_stop = 0.0  # force the run to exhaust max_items
    task.atlas_min_items = 8
    task.atlas_max_items = 12

    responses = [_mc_response(f"q{i}", correct=(i % 2 == 0)) for i in range(12)]
    result = task.compute_metrics(responses)

    assert "accuracy" in result  # inherited base metric preserved
    for key in ("atlas_theta", "atlas_se", "atlas_n_items", "atlas_pirt_accuracy"):
        assert key in result, key
    assert result["atlas_n_items"]["atlas"] == 12.0
    pirt = result["atlas_pirt_accuracy"]["atlas"]
    assert 0.0 <= pirt <= 1.0
    # bank provenance recorded as a version-keyed map
    assert any("restrict12" in k for k in result["atlas_bank_version"])


def test_compute_metrics_no_bank_overlap_is_graceful(tmp_path: Path) -> None:
    bank_dir = _write_bank(tmp_path, n=4)
    task = get_task("atlas_arc_challenge")
    assert isinstance(task, AtlasARCChallenge)
    task.atlas_bank_dir = str(bank_dir)

    # ids that do not exist in the bank -> no overlap -> base metrics only
    responses = [_mc_response(f"missing{i}", correct=True) for i in range(3)]
    result = task.compute_metrics(responses)
    assert "accuracy" in result
    assert "atlas_theta" not in result
