"""The MuSR bank: its prompt, its ``acc_norm`` scoring, and the content key behind it.

Three things about MuSR are unlike every other bank here and each of them is a way to
get a well-formed run out of the wrong measurement.

Its prompt puts the choice list *inside* the text and then scores the choice text as a
continuation of it, which no other MCQ style does. Its metric is ``acc_norm`` rather than
the unnormalized sum, so the same log-probabilities rank differently -- and both the
prompt and the metric were what its difficulties were estimated behind, which EAP treats
as fixed. And it is the first Route B bank re-keyed by content hash, so nothing about an
item's id says which subtask it came from any more: the bridge's own ``subtask`` column
does, which is why the tests below read it rather than splitting an id.

That re-key is also why ``object_placements`` no longer needs special handling here. Its
254 calibrated rows sat over a 256-long enumeration with positions 136 and 140 dropped in
the fit, which every positional guard had to be taught to tolerate; under a content key
the two dropped positions are simply rows the bank does not have.

The parity tests build the task's own ``LMRequest`` from a committed item rather than
comparing against a string typed out here, for the reason
:mod:`.test_mcq_prompts` does: a hardcoded expectation keeps passing after the task
changes, which is the failure being guarded against. They skip where ``olmo_eval`` is not
importable.
"""

from __future__ import annotations

import csv
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ....base import BenchmarkItem
from ....common import cat_loop, grading, inference
from ..datasets import SUPPORTED
from ..style import UniMcqStyle
from .conftest import CALIBRATED_DATASETS, SimMcqTaker, vendored_params

DATASET = "musr"

BRIDGE = Path(__file__).resolve().parents[1] / "bridges" / f"{DATASET}.csv"

#: What each subtask contributes, as the bridge counts it and as the task enumerates it.
#: ``object_placements`` is the pair that differs, because two of its items were dropped
#: mid-run during calibration.
SUBTASK_ROWS = {"murder_mysteries": 250, "object_placements": 254, "team_allocation": 250}
SUBTASK_SPANS = {"murder_mysteries": 250, "object_placements": 256, "team_allocation": 250}

#: Bank rows surviving ``a > 0`` per subtask, and the count of each after the task join.
SUBTASK_ITEMS = {"murder_mysteries": 132, "object_placements": 158, "team_allocation": 142}


def bridge_rows() -> list[dict[str, str]]:
    """The committed bridge, which is what says where an item came from."""
    with BRIDGE.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def subtask_of(item_id: str) -> str:
    """Return the subtask the bridge attributes an item to.

    A content hash names the item and nothing else, so this is a lookup rather than the
    string split the composite key allowed. That is the intended trade: an id that no
    longer encodes a position also no longer encodes a subtask, and the bridge is the
    right place to ask, being the artifact that recorded both.
    """
    return {row["item_id"]: row["subtask"] for row in bridge_rows()}[item_id]


def vendored_items() -> list[BenchmarkItem]:
    """The committed MuSR items, loaded through the real loader."""
    from ....common.benchmark_download import load_items_from_jsonl

    path = CALIBRATED_DATASETS / DATASET / "items.jsonl"
    if not path.is_file():
        pytest.skip("musr has not been vendored")
    return list(load_items_from_jsonl(path, name=DATASET).items)


def vendored_bank_params() -> list[dict[str, Any]]:
    """The committed MuSR parameter records, whose metadata names each item's subtask."""
    path = CALIBRATED_DATASETS / DATASET / "params.json"
    if not path.is_file():
        pytest.skip("musr has not been vendored")
    return json.loads(path.read_text(encoding="utf-8"))


def manifest() -> dict[str, Any]:
    """The committed MuSR manifest, or skip."""
    path = CALIBRATED_DATASETS / DATASET / "manifest.json"
    if not path.is_file():
        pytest.skip("musr has not been vendored")
    return json.loads(path.read_text(encoding="utf-8"))


def resolved_mcq_config() -> inference.InferenceConfig:
    """The MCQ settings the committed ``config.yaml`` resolves for MuSR."""
    settings = grading._apply_mcq_overrides(
        grading.GradingSettings(), UniMcqStyle().mcq_settings, dataset=DATASET
    )
    return settings.mcq


def task_pairs(item: BenchmarkItem) -> tuple[tuple[str, str], ...]:
    """Return the subtask's own ``(prompt, continuation)`` pairs for ``item``.

    The task is chosen by what the bridge says the item is, so a bank re-vendored against
    a different set of subtasks compares against the tasks it actually names.
    """
    registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
    types = pytest.importorskip("olmo_eval.common.types")
    subtask = subtask_of(item.item_id)
    instance = types.Instance(
        question=item.question,
        choices=item.choices,
        gold_answer=item.choices[item.gold_index],
        metadata={"gold_idx": item.gold_index},
    )
    request = registry.get_task(f"musr_{subtask}").format_request(instance)
    prompts = request.continuation_prompts or ((request.prompt,) * len(request.continuations))
    return tuple(zip(prompts, request.continuations, strict=True))


def our_pairs(item: BenchmarkItem) -> tuple[tuple[str, str], ...]:
    """Return this harness's pairs for ``item`` under the resolved MuSR settings."""
    choices = inference.scored_choices(item, resolved_mcq_config())
    return tuple((c.prompt, c.continuation) for c in choices)


class TestParityWithTheTask:
    """Our pairs must be the task's pairs, down to the blank line before the cue."""

    def test_the_first_committed_item_of_each_subtask_matches(self) -> None:
        seen: set[str] = set()
        for item in vendored_items():
            subtask = subtask_of(item.item_id)
            if subtask in seen:
                continue
            seen.add(subtask)
            assert our_pairs(item) == task_pairs(item), item.item_id
        assert seen == set(SUBTASK_ITEMS)

    def test_a_sample_of_the_bank_matches(self) -> None:
        """Every twentieth item, so a layout that only breaks on some choice counts shows."""
        for item in vendored_items()[::20]:
            assert our_pairs(item) == task_pairs(item), item.item_id

    def test_the_choices_are_listed_in_the_prompt_and_scored_after_it(self) -> None:
        """Both halves of what makes this style its own: the block and the cue."""
        item = vendored_items()[0]
        (prompt, _), *_ = our_pairs(item)

        assert prompt.startswith(item.question)
        assert prompt.endswith("\nAnswer:")
        for number, choice in enumerate(item.choices, start=1):
            assert f"{number} - {choice}\n" in prompt

    def test_the_blank_line_before_the_cue_survives(self) -> None:
        """Upstream joins a newline-terminated block to the cue with another newline.

        It is inside the string every calibrated difficulty was estimated against, so a
        tidier layout would be a different prompt.
        """
        prompt, _ = our_pairs(vendored_items()[0])[0]
        assert prompt.endswith("\n\nAnswer:")

    def test_the_scored_span_is_the_choice_text_and_not_its_number(self) -> None:
        item = vendored_items()[0]
        continuations = [continuation for _, continuation in our_pairs(item)]
        assert continuations == [f" {choice}" for choice in item.choices]

    def test_one_prompt_serves_every_choice(self) -> None:
        """Unlike WinoGrande's substitution, the prompt does not vary per candidate."""
        prompts = {prompt for prompt, _ in our_pairs(vendored_items()[0])}
        assert len(prompts) == 1


class TestTheConfiguredConvention:
    def test_it_is_scored_by_acc_norm(self) -> None:
        config = resolved_mcq_config()
        assert config.prompt_style == "musr"
        assert config.score_normalization == "continuation_logprob_per_character"

    def test_the_task_declares_the_metric_the_config_reproduces(self) -> None:
        """The config entry is only right if the task it mirrors still says so."""
        metrics = pytest.importorskip("olmo_eval.common.metrics")
        registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
        for subtask in SUBTASK_ITEMS:
            (metric,) = registry.get_task(f"musr_{subtask}").config.metrics
            assert isinstance(metric, metrics.LogprobPerCharMCAccuracyMetric), subtask

    def test_the_task_is_zero_shot(self) -> None:
        registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
        assert registry.get_task("musr_murder_mysteries").config.num_fewshot == 0
        assert inference.NUM_FEWSHOT == 0

    def test_the_manifest_records_both(self) -> None:
        runtime = manifest()["scoring_convention"]["runtime"]
        assert runtime["prompt_style"] == "musr"
        assert runtime["score_normalization"] == "continuation_logprob_per_character"

    def test_the_calibration_metric_is_the_one_it_is_graded_under(self) -> None:
        """The only MCQ bank here whose recorded calibration metric this harness matches."""
        assert SUPPORTED[DATASET].calibration.metric == "acc_norm"


class TestTheVendoredBank:
    def test_the_counts_are_the_ones_upstream_reports(self) -> None:
        recorded = manifest()
        assert recorded["upstream_bank_rows"] == 754
        assert recorded["bridge_rows"] == 754
        assert recorded["dropped"]["non_positive_discrimination"] == 322
        assert recorded["dropped"]["not_in_task"] == 0
        assert recorded["dropped"]["not_in_bridge"] == 0
        assert recorded["dropped"]["ambiguous_item_id"] == 0
        assert recorded["items"] == 432

    def test_it_spans_the_three_subtasks(self) -> None:
        recorded = manifest()
        assert recorded["task"] == (
            "musr_murder_mysteries, musr_object_placements, musr_team_allocation"
        )
        counts: dict[str, int] = {}
        for record in vendored_bank_params():
            subtask = record["metadata"]["subtask"]
            counts[subtask] = counts.get(subtask, 0) + 1
        assert counts == SUBTASK_ITEMS

    def test_every_item_records_the_subtask_its_id_no_longer_carries(self) -> None:
        """The one thing a content hash costs, paid back where a reader can reach it.

        A composite id said which task rendered an item; a hash says only which item it
        is. Vendoring therefore copies the bridge's own ``subtask`` onto each parameter
        record, and without that the per-subtask overlap floor has nothing to group by
        and a session's items cannot be attributed to a skill at all.
        """
        bridge = {row["item_id"]: row["subtask"] for row in bridge_rows()}
        for record in vendored_bank_params():
            assert record["metadata"]["subtask"] == bridge[record["item_id"]]

    def test_the_sparse_subtask_kept_the_items_past_its_gap(self) -> None:
        """Calibration dropped two ``object_placements`` items mid-run, not at the tail.

        Positions 136 and 140 are absent from a split of 256, which is upstream's right
        and was the awkward case for every positional guard: a gapless run was demanded
        before enumeration started, and the enumerated count was later compared against
        254 rows rather than the 256 instances the task correctly yields. A content key
        removes the question rather than answering it, and what is checked here is that
        the gap is still where it was and that the items above it are still in the bank.
        """
        by_id = {row["item_id"]: row for row in bridge_rows()}
        positions = sorted(
            int(by_id[record["item_id"]]["split_index"])
            for record in vendored_bank_params()
            if record["metadata"]["subtask"] == "object_placements"
        )
        assert len(positions) == SUBTASK_ITEMS["object_placements"]
        assert max(positions) == SUBTASK_SPANS["object_placements"] - 1
        assert sum(1 for position in positions if position > 140) > 0

    def test_every_item_carries_a_choice_set_and_a_gold_index(self) -> None:
        """MCQ modality: a choiceless item here would be graded by the wrong scheme."""
        for item in vendored_items():
            assert len(item.choices) >= 2
            assert 0 <= item.gold_index < len(item.choices)

    def test_the_modality_guard_admits_it(self) -> None:
        request = grading.GradingRequest(dataset=DATASET, modality=grading.MCQ)
        grading.check_bank_modality(request, vendored_items())

    def test_the_narrative_is_in_the_stem(self) -> None:
        """A MuSR question without its story is unanswerable, not merely harder.

        Graded that way every model looks uniformly weak, which reads as a depressed
        ability estimate rather than as an error, so the stems are checked for length
        rather than only for presence.
        """
        assert min(len(item.question) for item in vendored_items()) > 500


def run_real_bank(true_theta: float, *, max_items: int = 40) -> dict:
    """A full CAT over the committed 432-item bank, with only the forward pass simulated."""
    style = UniMcqStyle()
    bank = style.download_benchmark(DATASET)
    irt = style.load_irt_params(DATASET)
    config = resolved_mcq_config()

    model = inference._HFScoringModel.__new__(inference._HFScoringModel)
    model.config = config
    model._continuation_logprob = SimMcqTaker(
        true_theta, vendored_params(DATASET), list(bank.items), config
    )

    report = cat_loop.run_cat(
        style, bank=bank, irt_bank=irt, model=model, se_threshold=0.3, max_items=max_items
    )
    return report.to_dict()


class TestAbilityRecoveryOnTheRealBank:
    """Theta recovery over the committed bank, through the real engine and scorer."""

    run = staticmethod(run_real_bank)

    def test_a_session_converges_and_stops_on_precision(self) -> None:
        report = self.run(0.5)

        assert report["metadata"]["stop_reason"] == "precision_reached"
        assert report["metadata"]["standard_error"] <= 0.3
        assert report["metadata"]["bank_size"] == 432
        assert np.isfinite(report["ability"]["theta"])

    @pytest.mark.parametrize("true_theta", [-1.0, 0.0, 1.0])
    def test_the_estimate_lands_near_the_truth(self, true_theta: float) -> None:
        assert self.run(true_theta)["metadata"]["theta"] == pytest.approx(true_theta, abs=0.6)

    def test_theta_recovers_monotonically(self) -> None:
        """The property a checkpoint-to-checkpoint comparison actually rests on.

        The absolute scale is the bank's and moves with the grading convention; the
        ordering is what says one checkpoint is stronger than another, and it has to
        hold over the parameters actually shipped rather than over a synthetic ladder.
        """
        estimates = [self.run(theta)["metadata"]["theta"] for theta in (-1.5, -0.5, 0.5, 1.5)]

        assert all(b > a for a, b in pairwise(estimates)), estimates

    def test_the_report_names_the_convention_that_produced_it(self) -> None:
        """Both halves, because a theta is a statement about a prompt and a ranking rule.

        The note is the part a reader sees first, and it used to assert that MCQ items
        are "not length-normalized" -- true of the three ATLAS banks and false here.
        """
        metadata = self.run(0.5)["metadata"]

        assert metadata["modality"] == "mcq"
        assert metadata["prompt_style"] == "musr"
        assert metadata["score_normalization"] == "continuation_logprob_per_character"
        assert "acc_norm" in metadata["scoring_note"]
        assert "not length-normalized" not in metadata["scoring_note"]

    def test_every_response_records_one_score_per_choice(self) -> None:
        report = self.run(0.5)
        by_id = {item["id"]: item for item in _items_by_id()}
        for response in report["responses"]:
            assert len(response["choice_logprobs"]) == len(by_id[response["item_id"]]["choices"])


def _items_by_id() -> list[dict[str, Any]]:
    """The committed records as written, for a length check the loader would hide."""
    path = CALIBRATED_DATASETS / DATASET / "items.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
