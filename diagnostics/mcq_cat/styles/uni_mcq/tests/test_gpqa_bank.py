"""The GPQA bank as multiple choice, and the option ordering the whole thing rests on.

GPQA was vendored here as a generative bank until 2026-08-08, on the registered
olmo-eval task's chat chain-of-thought default. It should never have been: the
difficulties came from Open LLM Leaderboard v2, whose ``leaderboard_gpqa`` is
``output_type: multiple_choice`` and ranks "(A)".."(D)" by log-likelihood under
``acc_norm``. Scoring a chain of thought against those parameters was a convention
mismatch EAP absorbs entirely into theta, and it also put the bank out of reach of a base
checkpoint, since a chain of thought needs a chat template and a base checkpoint has none.

**The one thing that can go wrong here quietly is the option order**, and most of this
module is about it. ``GPQATask.process_doc`` shuffles the choices per question, so
``gold_index`` indexes one permutation and means nothing against any other. A rotated
choice list is not a broken bank: every count matches, the join is total, Fisher selection
runs on the real difficulties, the CAT converges, the standard error collapses on
schedule, and the theta is noise with a healthy interval printed beside it. Nothing
downstream computes a quantity that moves. ``Plan/flows/uni_mcq/README.md`` records the
same failure hiding in the ATLAS bridges until someone correlated p-values against
difficulty.

So the ordering is not re-derived and sanity-checked, it is transferred and then held to,
and three independent things say it is right. Each has a class below.

:class:`TestTheOrderSurvivedTheModalityChange` is the **transfer**. The generative stems
carried the four options as a lettered block in exactly the order the gold letter indexed;
``scripts/freeze_choice_order.py`` parsed all 395 back out -- demanding a unique
decomposition and a byte-identical round trip for each -- into
``bridges/gpqa.choice_order.json``, and ``vendor_bank.check_choice_order`` holds every
re-vendored item to it. Parsing is a transformation with no ordering assumption in it,
which re-enumerating the source dataset and re-applying a shuffle is not.

:class:`TestTheOutsideRecordNamesTheSameAnswer` is the part the transfer cannot do. Both
halves of a transfer are this repo's, so a mapping shifted before the stems were written
would be reproduced faithfully and every byte-equality test would pass. The control
therefore also carries GPQA's own ``Correct Answer`` and ``Incorrect Answer`` columns as
Open LLM Leaderboard v2's per-example records deliver them, and those are compared against
our option set and our gold on every item.

:class:`TestMutationsAreCaught` is what makes either of those a test rather than a
formality. Rotating the choices, swapping two, shifting the gold index and rewriting an
option text are each applied to the whole committed bank, and each must be refused by
vendoring *and* contradicted by the outside record.

The remaining classes are the ordinary bank checks -- counts, the prompt, the metric, the
manifest -- and :class:`TestItIsReachableWithoutAChatTemplate`, which is the reason the
modality changed at all.

Nothing here touches the network. ``Idavidrein/gpqa`` is gated and the leaderboard's
records are large, so what the outside checks read is the committed control, which froze
them at a pinned revision.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ....base import BenchmarkItem
from ....common import cat_loop, grading, inference
from ..datasets import SUPPORTED
from ..scripts import freeze_choice_order, vendor_bank
from ..style import UniMcqStyle
from .conftest import CALIBRATED_DATASETS, SimMcqTaker, vendored_params

DATASET = "gpqa"

#: The three registered tasks the bank spans, in the order its bridge names them.
SUBTASKS = ("gpqa_diamond", "gpqa_extended", "gpqa_main")

#: What each subtask contributes to the bridge and to the deduplicated bank.
SUBTASK_BRIDGE_ROWS = {"diamond": 198, "extended": 546, "main": 448}
SUBTASK_ITEMS = {"diamond": 24, "extended": 252, "main": 119}

#: The order a repeated question's surviving calibration is chosen in.
PRECEDENCE = ["extended", "main", "diamond"]

ITEMS = 395

#: How often the gold sits at each position, which is what makes an order check able to
#: fail. Under a shuffle that put the answer first every time, a rotated choice list would
#: agree with a correct one and every assertion below would pass on nothing.
GOLD_POSITIONS = {0: 103, 1: 87, 2: 110, 3: 95}

#: Items whose four options are not four distinguishable strings once both harnesses'
#: preprocessing is reconciled, and therefore the exact items on which no comparison of
#: answer *text* can tell a correct gold from a rotated one.
#:
#: Three of them are GPQA's own doing -- extended|105 offers "200 mL" twice, extended|219
#: repeats a whole spectroscopy answer, extended|310 repeats a IUPAC name -- and on those
#: the ambiguity is harmless, because the two candidate golds are the same string and a
#: model cannot tell them apart either. The rest collapse only under
#: :func:`~..scripts.freeze_choice_order.comparable`, which deletes bracketed spans
#: because lm-evaluation-harness's own preprocessing does.
INDISTINGUISHABLE = {
    "extended|105",
    "extended|152",
    "extended|219",
    "extended|29",
    "extended|310",
    "extended|404",
    "extended|476",
}

#: Items whose gold survives a one-position rotation of the choice list, which is
#: :data:`INDISTINGUISHABLE` minus the two whose duplicate options are not adjacent in
#: the frozen order. Pinned as a number rather than named, because what it measures is
#: the sharpness of the check and not a property of any one item.
ROTATION_SURVIVORS = 5


def control() -> dict[str, freeze_choice_order.FrozenOrder]:
    """The committed option-ordering control."""
    return freeze_choice_order.load_control(DATASET)


def records() -> list[dict[str, Any]]:
    """The committed item records as written, before the loader normalizes them."""
    path = CALIBRATED_DATASETS / DATASET / "items.jsonl"
    if not path.is_file():
        pytest.skip("gpqa has not been vendored")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def vendored_items() -> list[BenchmarkItem]:
    """The committed GPQA items, loaded through the real loader."""
    from ....common.benchmark_download import load_items_from_jsonl

    path = CALIBRATED_DATASETS / DATASET / "items.jsonl"
    if not path.is_file():
        pytest.skip("gpqa has not been vendored")
    return list(load_items_from_jsonl(path, name=DATASET).items)


def manifest() -> dict[str, Any]:
    """The committed GPQA manifest, or skip."""
    path = CALIBRATED_DATASETS / DATASET / "manifest.json"
    if not path.is_file():
        pytest.skip("gpqa has not been vendored")
    return json.loads(path.read_text(encoding="utf-8"))


def resolved_mcq_config() -> inference.InferenceConfig:
    """The MCQ settings the committed ``config.yaml`` resolves for GPQA."""
    settings = grading._apply_mcq_overrides(
        grading.GradingSettings(), UniMcqStyle().mcq_settings, dataset=DATASET
    )
    return settings.mcq


def normalized(text: str) -> str:
    """Shorthand for the comparison the two harnesses can both be read in."""
    return freeze_choice_order.comparable(text)


def gold_disagreements(items: list[dict[str, Any]]) -> list[str]:
    """Items whose gold does not name the answer the benchmark's own column gives.

    The check that does not go through the control's ``choices``, and therefore the one
    a hand-edited control cannot talk its way past.
    """
    frozen = control()
    return [
        str(item["id"])
        for item in items
        if normalized(list(item["choices"])[int(item["gold_index"])])
        != normalized(frozen[str(item["id"])].source_gold)
    ]


def option_set_disagreements(items: list[dict[str, Any]]) -> list[str]:
    """Items whose four options are not the benchmark's own four, as a set."""
    frozen = control()
    return [
        str(item["id"])
        for item in items
        if sorted(normalized(choice) for choice in item["choices"])
        != sorted(normalized(choice) for choice in frozen[str(item["id"])].source_options)
    ]


def mutated(mutate: Callable[[list[str], int], tuple[list[str], int]]) -> list[dict[str, Any]]:
    """The committed records with every item's choices or gold moved by ``mutate``."""
    out: list[dict[str, Any]] = []
    for record in records():
        choices, gold = mutate(list(record["choices"]), int(record["gold_index"]))
        out.append({**record, "choices": choices, "gold_index": gold})
    return out


def rotate_choices(choices: list[str], gold: int) -> tuple[list[str], int]:
    """Turn the options by one and leave the gold index where it was."""
    return choices[1:] + choices[:1], gold


def swap_first_two(choices: list[str], gold: int) -> tuple[list[str], int]:
    """Exchange the first two options."""
    choices[0], choices[1] = choices[1], choices[0]
    return choices, gold


def shift_gold(choices: list[str], gold: int) -> tuple[list[str], int]:
    """Leave a correct choice list and move the answer off it."""
    return choices, (gold + 1) % len(choices)


def retext_first(choices: list[str], gold: int) -> tuple[list[str], int]:
    """Edit an option's text without moving anything."""
    choices[0] = f"{choices[0]} (approximately)"
    return choices, gold


class TestTheControlCoversTheBank:
    """The frozen ordering, before anything is compared against it.

    A control that covered 394 items would let one through unchecked, and an entry
    naming an item the bank no longer has would mean the two artifacts had drifted --
    which is the moment a control stops being evidence and starts being decoration.
    """

    def test_it_holds_one_ordering_per_committed_item(self) -> None:
        frozen = control()
        assert len(frozen) == ITEMS
        assert set(frozen) == {str(record["id"]) for record in records()}

    def test_every_entry_names_four_options_and_one_of_them(self) -> None:
        for item_id, entry in control().items():
            assert len(entry.choices) == 4, item_id
            assert entry.gold_letter in "ABCD", item_id
            assert 0 <= entry.gold_index < 4, item_id

    def test_it_records_where_it_was_read_from(self) -> None:
        """A control whose source nobody can name cannot be rebuilt or disputed."""
        payload = json.loads(freeze_choice_order.control_path(DATASET).read_text(encoding="utf-8"))
        assert payload["source"]["path"] == "calibrated_datasets/gpqa/items.jsonl"
        assert len(payload["source"]["commit"]) == 40
        assert payload["cross_check"]["revision"]


class TestTheOrderSurvivedTheModalityChange:
    """The acceptance criterion. Every item, every option, every position.

    Not a sample. A permutation that reached one item in twenty would still be a theta
    built partly on distractors, and sampling is exactly how it would be missed.
    """

    def test_every_item_carries_the_frozen_options_in_the_frozen_order(self) -> None:
        frozen = control()
        checked = 0
        for record in records():
            entry = frozen[str(record["id"])]
            assert list(record["choices"]) == entry.choices, record["id"]
            checked += 1
        assert checked == ITEMS

    def test_every_gold_index_names_what_the_frozen_letter_named(self) -> None:
        frozen = control()
        for record in records():
            entry = frozen[str(record["id"])]
            assert int(record["gold_index"]) == entry.gold_index, record["id"]
            assert record["choices"][record["gold_index"]] == entry.choices[entry.gold_index]

    def test_every_question_and_its_block_reassemble_the_old_stem(self) -> None:
        """Ties each ordering to the whole stem it was read out of, not just to a list.

        The generative bank's stem was the question, a blank line and the lettered
        block. Rebuilding that from today's question and today's choices and hashing it
        is what would catch a question that has drifted, a block parsed at the wrong
        boundary, or a control regenerated against some other bank -- none of which the
        comparisons above can see, because each of those would agree with itself.
        """
        frozen = control()
        for record in records():
            entry = frozen[str(record["id"])]
            stem = str(record["question"]) + freeze_choice_order.BLOCK_SEPARATOR
            stem += freeze_choice_order.render_block(list(record["choices"]))
            assert freeze_choice_order.sha256(stem) == entry.stem_sha256, record["id"]

    def test_the_gold_lands_on_all_four_positions(self) -> None:
        """What makes every assertion above capable of failing.

        If the shuffle had put the answer first every time, a rotated choice list would
        select the same option as a correct one on every item and none of this would
        measure anything.
        """
        positions = dict.fromkeys(range(4), 0)
        for record in records():
            positions[int(record["gold_index"])] += 1
        assert positions == GOLD_POSITIONS

    def test_vendoring_would_hold_todays_bank_to_it(self) -> None:
        """The guard itself, run over the committed bank rather than over a fixture."""
        assert vendor_bank.check_choice_order(SUPPORTED[DATASET], records()) == ITEMS


class TestTheOutsideRecordNamesTheSameAnswer:
    """The half a transfer cannot supply: whether the ordering transferred was right.

    Read off the control's frozen copy of GPQA's own ``Correct Answer`` and
    ``Incorrect Answer`` columns, as Open LLM Leaderboard v2's per-example records
    deliver them. Those columns share no code with this pipeline and no ordering with
    it either, which is what makes them able to contradict it.
    """

    def test_our_four_options_are_the_benchmarks_four(self) -> None:
        assert option_set_disagreements(records()) == []

    def test_our_gold_names_the_benchmarks_correct_answer(self) -> None:
        assert gold_disagreements(records()) == []

    def test_it_reaches_every_item(self) -> None:
        """An outside check covering most of the bank would leave the rest asserted only
        against this repo's own copy of itself."""
        frozen = control()
        assert sum(1 for entry in frozen.values() if entry.source_options) == ITEMS
        assert all(len(entry.source_options) == 4 for entry in frozen.values())

    def test_the_evaluation_that_was_scored_agrees_where_it_can_be_read(self) -> None:
        """The stronger claim, and the incomplete one.

        ``leaderboard_gold`` is the option the harness actually scored as correct rather
        than the column the benchmark ships, so agreeing with it means our gold names an
        option a real evaluation counted. It is empty on one item because
        lm-evaluation-harness's ``preprocess`` deletes bracketed spans and extended|476's
        four options are written entirely inside brackets, so the leaderboard showed the
        model four blank options and there is nothing left to compare.
        """
        frozen = control()
        with_scored = {item_id for item_id, entry in frozen.items() if entry.leaderboard_gold}
        assert set(frozen) - with_scored == {"extended|476"}
        for record in records():
            entry = frozen[str(record["id"])]
            if not entry.leaderboard_gold:
                continue
            chosen = record["choices"][record["gold_index"]]
            assert normalized(chosen) == normalized(entry.leaderboard_gold), record["id"]

    def test_a_rotation_would_contradict_it_almost_everywhere(self) -> None:
        """The check's sharpness, which is the only thing that makes the two passes above
        worth anything.

        A comparison that agreed with a rotated bank as readily as with a correct one
        would be measuring the presence of the file rather than the contents of it.
        """
        survivors = gold_disagreements(mutated(rotate_choices))
        assert len(survivors) == ITEMS - ROTATION_SURVIVORS

    def test_what_it_cannot_speak_for_is_named(self) -> None:
        """The limit, pinned rather than left implicit.

        On these seven the four options are not four distinguishable strings, so no
        comparison of answer text can separate a correct gold from a rotated one. Three
        are GPQA repeating an option outright, where the ambiguity costs nothing because
        both candidates are the same string; the rest collapse only under the
        bracket-stripping the comparison itself has to do.
        """
        collapsed = {
            str(record["id"])
            for record in records()
            if len({normalized(choice) for choice in record["choices"]}) < 4
        }
        assert collapsed == INDISTINGUISHABLE


class TestMutationsAreCaught:
    """Move the ordering deliberately, four ways, and watch both defences fire.

    Each mutation is applied to the whole committed bank rather than to a fixture, so
    what is exercised is the data actually shipped. Both defences are asserted for each,
    because they fail independently: the guard compares against a file this repo wrote,
    and the outside record does not.
    """

    def test_rotating_every_choice_list_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="the order is not"):
            vendor_bank.check_choice_order(SUPPORTED[DATASET], mutated(rotate_choices))

    def test_rotating_every_choice_list_contradicts_the_benchmark(self) -> None:
        assert len(gold_disagreements(mutated(rotate_choices))) == ITEMS - ROTATION_SURVIVORS

    def test_swapping_two_choices_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="the order is not"):
            vendor_bank.check_choice_order(SUPPORTED[DATASET], mutated(swap_first_two))

    def test_swapping_two_choices_contradicts_the_benchmark_where_it_can(self) -> None:
        """Only where the gold is one of the two moved, which is the honest expectation.

        A swap of positions 0 and 1 leaves an item whose answer sits at 2 or 3 answering
        correctly, so this catches 188 rather than 390 -- every item whose gold is at one
        of the swapped positions, and no more. The guard above catches the rest.
        """
        caught = gold_disagreements(mutated(swap_first_two))
        assert len(caught) == GOLD_POSITIONS[0] + GOLD_POSITIONS[1] - 2

    def test_shifting_the_gold_index_is_refused(self) -> None:
        """The quietest mutation of the four: the choice list stays correct.

        Nothing downstream compares a gold index against anything, so this is the one
        that would otherwise run to completion and report a confident wrong theta.
        """
        with pytest.raises(SystemExit, match="the answer does not"):
            vendor_bank.check_choice_order(SUPPORTED[DATASET], mutated(shift_gold))

    def test_shifting_the_gold_index_contradicts_the_benchmark(self) -> None:
        assert len(gold_disagreements(mutated(shift_gold))) == ITEMS - ROTATION_SURVIVORS

    def test_rewriting_an_option_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="option texts themselves differ"):
            vendor_bank.check_choice_order(SUPPORTED[DATASET], mutated(retext_first))

    def test_rewriting_an_option_breaks_the_option_set_on_every_item(self) -> None:
        """The one mutation the set comparison catches everywhere, gold or not."""
        assert len(option_set_disagreements(mutated(retext_first))) == ITEMS

    def test_editing_a_question_is_refused(self) -> None:
        """Caught by the digest, since the choices and the gold are both untouched."""
        edited = [{**record, "question": record["question"] + " "} for record in records()]
        with pytest.raises(SystemExit, match="does not reassemble the stem"):
            vendor_bank.check_choice_order(SUPPORTED[DATASET], edited)

    def test_a_bank_the_control_does_not_cover_is_refused(self) -> None:
        """An item with no frozen ordering is one nothing above can speak for."""
        with pytest.raises(SystemExit, match="no frozen ordering"):
            vendor_bank.check_choice_order(SUPPORTED[DATASET], records()[:-1])


class TestThePromptIsTheHarvestsOwn:
    """The layout, against the arguments Open LLM Leaderboard v2 recorded for a *base*
    checkpoint.

    Frozen from ``open-llm-leaderboard/meta-llama__Meta-Llama-3-8B-details`` at revision
    7edb46e12947, ``samples_leaderboard_gpqa_diamond_2024-06-16T19-10-04.926831.json``,
    document 0. A pretrained submission rather than an instruction-tuned one on purpose:
    an instruct model's ``arg_0`` has been through its chat template, which hides exactly
    the whitespace this is checking, and a base checkpoint is the case this bank now has
    to serve.
    """

    #: The document as that record's ``doc`` gives it, options in the order it showed them.
    QUESTION = (
        "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and "
        "10^-8 sec, respectively. We want to clearly distinguish these two energy "
        "levels. Which one of the following options could be their energy difference "
        "so that they can be clearly resolved?"
    )
    CHOICES = ("10^-9 eV", "10^-4 eV", "10^-8 eV", "10^-11 eV")

    #: Its ``arguments.gen_args_0.arg_0``, byte for byte.
    ARG_0 = (
        "What is the correct answer to this question:Two quantum states with energies "
        "E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to "
        "clearly distinguish these two energy levels. Which one of the following options "
        "could be their energy difference so that they can be clearly resolved?\n\n"
        "Choices:\n(A) 10^-9 eV\n(B) 10^-4 eV\n(C) 10^-8 eV\n(D) 10^-11 eV\nAnswer: "
    )

    #: Its four ``arg_1`` values, which are what was ranked.
    ARG_1 = (" (A)", " (B)", " (C)", " (D)")

    def item(self) -> BenchmarkItem:
        return BenchmarkItem(
            item_id="diamond|0",
            question=self.QUESTION,
            choices=self.CHOICES,
            gold_index=1,
            metadata={},
        )

    def test_our_prompt_is_the_one_that_was_harvested(self) -> None:
        (first, *_) = inference.get_mcq_prompt_style("gpqa").scored_choices(self.item())
        assert first.prompt == self.ARG_0

    def test_what_is_ranked_is_the_label_and_not_the_option_text(self) -> None:
        """The reason GPQA needs a style of its own.

        Its options are bare quantities that are frequently permutations of each other's
        digits, so scoring the option text ranks numerals rather than answers.
        """
        scored = inference.get_mcq_prompt_style("gpqa").scored_choices(self.item())
        assert tuple(choice.continuation for choice in scored) == self.ARG_1

    def test_one_prompt_serves_all_four_labels(self) -> None:
        scored = inference.get_mcq_prompt_style("gpqa").scored_choices(self.item())
        assert len({choice.prompt for choice in scored}) == 1

    def test_the_double_space_at_the_join_is_upstreams_and_is_kept(self) -> None:
        """The cue ends with a space and the continuation begins with one.

        It reads like a defect and it is the string every difficulty here was estimated
        behind, so tidying it would be a different prompt. Asserted on the join rather
        than on either half, because that is where a well-meant strip would show.
        """
        (first, *_) = inference.get_mcq_prompt_style("gpqa").scored_choices(self.item())
        assert (first.prompt + first.continuation).endswith("Answer:  (A)")

    def test_the_committed_bank_renders_through_the_same_style(self) -> None:
        """The fixture is one document; this is the 395 the bank actually holds."""
        config = resolved_mcq_config()
        for item in vendored_items()[::40]:
            scored = inference.scored_choices(item, config)
            assert len(scored) == 4
            assert scored[0].prompt.startswith("What is the correct answer to this question:")
            assert scored[0].prompt.endswith("\nAnswer: ")
            for index, choice in enumerate(item.choices):
                assert f"({'ABCD'[index]}) {choice}" in scored[0].prompt


class TestTheConfiguredConvention:
    def test_it_is_scored_the_way_the_leaderboard_scored_it(self) -> None:
        config = resolved_mcq_config()
        assert config.prompt_style == "gpqa"
        assert config.score_normalization == "continuation_logprob_per_character"

    def test_it_is_zero_shot(self) -> None:
        """0-shot, as the harvest was, and not BBH's frozen 3-shot arrangement."""
        assert inference.fewshot_count("gpqa") == 0
        assert manifest()["scoring_convention"]["runtime"]["num_fewshot"] == 0

    def test_the_manifest_records_both_halves(self) -> None:
        recorded = manifest()["scoring_convention"]
        assert recorded["runtime"]["modality"] == "mcq"
        assert recorded["runtime"]["prompt_style"] == "gpqa"
        assert recorded["runtime"]["score_normalization"] == "continuation_logprob_per_character"
        assert recorded["calibration"]["metric"] == "acc_norm"

    def test_the_calibration_note_says_what_changed_and_on_what_evidence(self) -> None:
        """The old note conceded the metric was unrecorded; this one has to do better.

        What replaced it is not a stronger assertion of the same guess. The harvest was
        identified -- the fit's own response matrix is the leaderboard's acc_norm outcome
        for that document -- so the note names the evidence, names the modality it
        implies, and says plainly that the bank was scored the other way until this
        change rather than implying it was always known.
        """
        note = SUPPORTED[DATASET].calibration.note
        assert "leaderboard_gpqa" in note
        assert "multiple_choice" in note
        assert "acc_norm" in note
        assert "generative until 2026-08-08" in note
        assert "unrecorded" not in note

    def test_the_generative_entry_is_gone_rather_than_stale(self) -> None:
        """A live generative block for gpqa would still resolve and still run."""
        yaml = pytest.importorskip("yaml")
        from ..convention import CONFIG_PATH

        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        assert DATASET not in config["generative"][grading.PER_DATASET_KEY]
        assert DATASET in config["mcq"][grading.PER_DATASET_KEY]

    def test_the_committed_bank_passes_the_startup_check(self) -> None:
        style = UniMcqStyle()
        style.download_benchmark(DATASET)
        request = style.grading_request()
        style.check_scoring_convention(
            request, grading.resolve_settings(request, grading.GradingSettings())
        )


class TestTheVendoredBank:
    """Counts and keys, which the modality change was required to leave alone."""

    def test_the_item_count_did_not_move(self) -> None:
        """The join is to IRT parameters keyed by item id; a different count is a
        different bank."""
        recorded = manifest()
        assert recorded["items"] == ITEMS
        assert len(records()) == ITEMS

    def test_the_counts_are_the_ones_upstream_reports(self) -> None:
        recorded = manifest()
        assert recorded["upstream_bank_rows"] == 1192
        assert recorded["bridge_rows"] == 1192
        assert recorded["dropped"]["non_positive_discrimination"] == 613
        assert recorded["dropped"]["not_in_task"] == 0
        assert recorded["dropped"]["not_in_bridge"] == 0
        assert recorded["dropped"]["ambiguous_item_id"] == 0
        assert recorded["dropped"]["duplicate_question"] == 184

    def test_the_extended_first_deduplication_still_chose_the_survivors(self) -> None:
        recorded = manifest()
        assert recorded["duplicate_question_precedence"] == PRECEDENCE
        assert recorded["subtask_items"] == SUBTASK_ITEMS
        counts: dict[str, int] = {}
        for item in vendored_items():
            label = item.item_id.split("|", 1)[0]
            counts[label] = counts.get(label, 0) + 1
        assert counts == SUBTASK_ITEMS

    def test_the_ids_are_the_composite_the_bridge_uses(self) -> None:
        """A bare integer here would mean the single-task fallback had keyed the bank."""
        for item in vendored_items():
            label, sep, position = item.item_id.partition("|")
            assert sep == "|"
            assert label in SUBTASK_BRIDGE_ROWS
            assert 0 <= int(position) < SUBTASK_BRIDGE_ROWS[label]

    def test_no_question_is_administered_twice(self) -> None:
        stems = [item.question for item in vendored_items()]
        assert len(set(stems)) == len(stems)

    def test_every_item_carries_four_options_and_a_gold_index(self) -> None:
        """What the modality change was for: there was nothing to rank before it."""
        for item in vendored_items():
            assert len(item.choices) == 4, item.item_id
            assert 0 <= item.gold_index < 4, item.item_id

    def test_no_stem_still_carries_the_frozen_choice_block(self) -> None:
        """The block moved out of the prompt and into a choice list.

        Left behind it would be rendered twice -- once by the stem and once by the
        prompt style -- and the model would be shown each option two times.
        """
        for item in vendored_items():
            assert "\n(A) " not in item.question, item.item_id
            assert "\n(D) " not in item.question, item.item_id

    def test_the_modality_guard_admits_it(self) -> None:
        request = grading.GradingRequest(dataset=DATASET, modality=grading.MCQ)
        grading.check_bank_modality(request, vendored_items())

    def test_the_manifest_records_how_many_orderings_were_checked(self) -> None:
        """Provenance for the claim this whole module is about."""
        assert manifest()["choice_order_checked"] == ITEMS


class TestItIsReachableWithoutAChatTemplate:
    """The reason the modality changed, driven through the code that used to refuse.

    A chain of thought is carried by a system turn, a system turn needs a chat template,
    and a base checkpoint has none -- so the generative framing put this bank out of
    reach of exactly the checkpoints it is wanted for. A log-likelihood ranking needs no
    template at all.
    """

    @pytest.fixture
    def fake_stack(self, monkeypatch):
        """``torch`` and ``transformers`` replaced, with a template-less tokenizer."""
        from .test_generative_grading import FakeModel, FakeTokenizer

        loaded: list[str] = []
        model = FakeModel()

        torch = types.ModuleType("torch")
        torch.no_grad = contextlib.nullcontext
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: FakeTokenizer()
        )

        def load_model(*args: object, **kwargs: object) -> FakeModel:
            loaded.append("weights")
            return model

        transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=load_model)
        transformers.set_seed = lambda seed: None
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        return loaded

    def test_a_checkpoint_with_no_chat_template_can_now_be_scored(self, fake_stack) -> None:
        """Built from the resolved config rather than a hand-written one, so flipping the
        setting back fails here instead of passing on a copy of what the file used to
        say."""
        inference._HFScoringModel(Path("/ckpt"), resolved_mcq_config())
        assert fake_stack == ["weights"]

    def test_the_native_checkpoint_kind_is_allowed(self) -> None:
        """MCQ registers both backends, so ``--checkpoint-kind olmo_core`` reaches this
        bank; the generative path registers only ``hf`` and would have refused it."""
        request = grading.GradingRequest(dataset=DATASET, modality=grading.MCQ)
        settings = grading.GradingSettings(
            mcq=inference.InferenceConfig(checkpoint_kind="olmo_core")
        )
        grading.check_checkpoint_kind(request, settings)
        assert set(inference.MCQ_SCORING_BACKENDS) == {"hf", "olmo_core"}


class TestTheSpec:
    def test_it_declares_the_mcq_modality_and_records_no_grader(self) -> None:
        """``answer_type`` keeps its default and stops being read.

        It named ``gpqa_letter`` while the bank was generative, which selected the
        chain-of-thought grader. An MCQ bank is decided by an argmax over choices and
        never reaches a grader at all, which is why the manifest writes null rather than
        carrying the field's default forward as though it meant something.
        """
        assert SUPPORTED[DATASET].modality == "mcq"
        assert manifest()["modality"] == "mcq"
        assert manifest()["answer_type"] is None

    def test_it_is_held_to_an_ordering_control(self) -> None:
        """The only bank here that is, and the flag is what turns the guard on."""
        spec = SUPPORTED[DATASET]
        assert spec.choice_order_control
        assert freeze_choice_order.control_path(DATASET).is_file()
        assert [name for name, s in SUPPORTED.items() if s.choice_order_control] == [DATASET]

    def test_it_is_still_multi_task(self) -> None:
        spec = SUPPORTED[DATASET]
        assert spec.is_multi_task
        assert spec.task_names == SUBTASKS

    def test_it_can_run_today(self) -> None:
        from ..datasets import ready_names, supported_names

        assert SUPPORTED[DATASET].blocked is None
        assert DATASET in supported_names()
        assert DATASET in ready_names()


def run_real_bank(true_theta: float, *, max_items: int = 40) -> dict:
    """A full CAT over the committed 395-item bank, with only the forward pass simulated."""
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
        assert report["metadata"]["bank_size"] == ITEMS
        assert np.isfinite(report["ability"]["theta"])

    @pytest.mark.parametrize("true_theta", [0.0, 1.0])
    def test_the_estimate_lands_near_the_truth(self, true_theta: float) -> None:
        assert self.run(true_theta)["metadata"]["theta"] == pytest.approx(true_theta, abs=0.6)

    def test_theta_recovers_monotonically(self) -> None:
        """The ordering is what a checkpoint-to-checkpoint comparison rests on."""
        estimates = [self.run(theta)["metadata"]["theta"] for theta in (-1.0, 0.0, 1.0, 1.5)]

        assert all(b > a for a, b in pairwise(estimates)), estimates

    def test_the_report_names_the_convention_that_produced_it(self) -> None:
        metadata = self.run(0.5)["metadata"]

        assert metadata["modality"] == "mcq"
        assert metadata["prompt_style"] == "gpqa"
        assert metadata["score_normalization"] == "continuation_logprob_per_character"
        assert "acc_norm" in metadata["scoring_note"]

    def test_every_response_records_one_score_per_option(self) -> None:
        for response in self.run(0.5)["responses"]:
            assert len(response["choice_logprobs"]) == 4
