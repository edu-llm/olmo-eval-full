"""GPQA's grading convention, and the two things about it that bite.

GPQA looks like a multiple-choice benchmark and is not scored as one. Its registered
tasks default to chat plus chain of thought -- an expert-scientist system prompt asking
for step-by-step reasoning ending in ``ANSWER: X``, then letter extraction and
``MultipleChoiceScorer`` -- and the log-likelihood formulation exists only as the ``:mc``
variant. There is nothing to weigh that against: Research never wired GPQA into its CAT,
so the task's own default is the whole of the convention. :class:`TestTheConvention`
reads that off the registry rather than trusting it, because the entire modality choice
rests on it.

The two hazards are both about the choice block. It is *part of the prompt*, since a
chain of thought has to see the options it is choosing among; and it is shuffled per
question, so the letter that answers it is meaningful only against one ordering.
Vendoring freezes both together into the item and checks they agree, which is what
:class:`TestVendoringFreezesTheShuffle` pins.

A third thing is peculiar to this bank and is checked here too. The three subsets are
nested quality filters over one pool of questions rather than different content, so
upstream calibrated 184 of its 579 surviving rows twice or three times over;
:class:`TestTheVendoredBank` pins that the committed bank is one calibration per
question and that the precedence which chose them is recorded beside it.

Nothing here enumerates the dataset. ``Idavidrein/gpqa`` is gated -- the bank was blocked
until this checkout's token was granted the gated-repository scope -- so the tests work
from a constructed instance, from the task's configuration, or from the committed bank,
none of which touches the network. :class:`TestTheVendoredBank` is where the frozen
shuffle is checked against what vendoring actually wrote for all 395 items.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from ....base import BenchmarkItem
from ....common import cat_loop, generative, grading
from ..convention import CONFIG_PATH
from ..datasets import SUPPORTED
from ..scripts import vendor_bank
from ..style import UniMcqStyle
from .conftest import CALIBRATED_DATASETS, SimGenerativeTaker, vendored_params

DATASET = "gpqa"
ANSWER_TYPE = "gpqa_letter"
GRADER_NAME = "gpqa_cot_letter"

#: The three registered tasks the bank spans, in the order its bridge names them.
SUBTASKS = ("gpqa_diamond", "gpqa_extended", "gpqa_main")


def olmo_eval_types() -> Any:
    """``olmo_eval.common.types``, or skip."""
    return pytest.importorskip("olmo_eval.common.types")


def gpqa_config() -> dict[str, Any]:
    """The committed generative entry for gpqa."""
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return dict((config["generative"][grading.PER_DATASET_KEY])[DATASET])


def instance(**overrides: Any) -> Any:
    """A GPQA-shaped instance: four choices, a gold letter, a matching ``gold_idx``."""
    types = olmo_eval_types()
    fields: dict[str, Any] = {
        "question": "Which particle mediates the strong interaction?",
        "choices": ("photon", "gluon", "W boson", "graviton"),
        "gold_answer": "B",
        "metadata": {"index": 7, "gold_idx": 1, "subdomain": "Physics"},
    }
    fields.update(overrides)
    return types.Instance(**fields)


class TestTheConvention:
    """Read off the task registry, because the modality choice rests entirely on it."""

    @pytest.fixture
    def task(self) -> Any:
        registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
        return registry.get_task("gpqa_diamond")

    def test_the_default_is_chat_not_log_likelihood(self, task: Any) -> None:
        formatters = pytest.importorskip("olmo_eval.common.formatters")
        types = olmo_eval_types()
        assert isinstance(task.config.formatter, formatters.MCQAChatFormatter)
        assert task.request_type == types.RequestType.CHAT

    def test_the_system_prompt_asks_for_reasoning_and_an_answer_line(self, task: Any) -> None:
        prompt = task.config.formatter.system_prompt
        assert "step by step" in prompt
        assert "ANSWER: X" in prompt

    def test_the_default_metric_matches_an_extracted_letter(self, task: Any) -> None:
        scorers = pytest.importorskip("olmo_eval.common.scorers")
        (metric,) = task.config.metrics
        assert metric.scorer is scorers.MultipleChoiceScorer

    def test_greedy_decoding_inside_1024_tokens(self, task: Any) -> None:
        assert task.config.sampling_params.temperature == 0.0
        assert task.config.sampling_params.max_tokens == 1024
        assert not task.config.sampling_params.stop_sequences

    def test_the_log_likelihood_formulation_is_a_variant_we_do_not_select(self) -> None:
        registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
        assert "mc" in registry.list_variants("gpqa_diamond")["gpqa_diamond"]
        assert SUPPORTED[DATASET].task_names == SUBTASKS
        assert all(":" not in name for name in SUBTASKS)

    def test_every_subtask_the_spec_names_is_registered(self) -> None:
        registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
        for name in SUBTASKS:
            assert registry.task_exists(name), name


class TestTheGrader:
    """Extraction and comparison, both olmo-eval's own."""

    @pytest.fixture(autouse=True)
    def _needs_olmo_eval(self) -> None:
        pytest.importorskip("olmo_eval.evals.tasks.gpqa")

    @pytest.fixture
    def grader(self) -> generative.GPQACoTLetter:
        return generative.GPQACoTLetter()

    def test_it_reads_the_answer_line_the_system_prompt_teaches(self, grader) -> None:
        verdict = grader.grade("Gluons carry colour charge.\n\nANSWER: B", "B")
        assert verdict.correct
        assert verdict.extracted == "B"
        assert verdict.gold == "B"

    def test_a_lowercase_answer_line_still_matches(self, grader) -> None:
        assert grader.grade("...so answer: b", "B").correct

    def test_the_last_answer_line_wins(self, grader) -> None:
        """A chain of thought that corrects itself states the answer more than once."""
        assert grader.grade("ANSWER: A\nOn reflection, ANSWER: B", "B").correct

    def test_a_parenthesized_letter_is_the_second_fallback(self, grader) -> None:
        assert grader.grade("The strong force is mediated by (B).", "B").correct

    def test_the_wrong_letter_is_wrong(self, grader) -> None:
        verdict = grader.grade("ANSWER: D", "B")
        assert not verdict.correct
        assert verdict.extracted == "D"

    def test_a_completion_with_no_letter_scores_incorrect_rather_than_raising(self, grader) -> None:
        """Model behaviour, and behaviour the calibration counted; not a harness fault."""
        verdict = grader.grade("I am not sure about this one.", "B")
        assert not verdict.correct
        assert verdict.extracted is None

    def test_it_agrees_with_the_task_extractor_it_wraps(self, grader) -> None:
        base = pytest.importorskip("olmo_eval.evals.tasks.common.base")
        gpqa = pytest.importorskip("olmo_eval.evals.tasks.gpqa")
        types = olmo_eval_types()
        task = gpqa.GPQATask(base.TaskConfig(name="gpqa"))

        for text in ("ANSWER: c", "we conclude (D)", "the answer is A", "no letters"):
            expected = task.extract_answer(types.LMOutput(text=text))
            assert grader.grade(text, "A").extracted == expected, text


class TestTheGraderIsRegistered:
    def test_the_answer_type_resolves_to_it(self) -> None:
        grader = generative.get_answer_grader(ANSWER_TYPE)
        assert isinstance(grader, generative.GPQACoTLetter)
        assert grader.name == GRADER_NAME

    def test_it_is_gold_matched_rather_than_item_graded(self) -> None:
        """``MultipleChoiceScorer`` compares two letters; nothing else about the item."""
        grader = generative.get_answer_grader(ANSWER_TYPE)
        assert isinstance(grader, generative.AnswerGrader)
        assert not isinstance(grader, generative.ItemGrader)

    def test_the_report_can_say_how_an_item_was_decided(self) -> None:
        note = generative.grader_note(GRADER_NAME)
        assert "chain of thought" in note
        assert ":mc" in note

    def test_an_item_carrying_a_gold_letter_is_gradable(self) -> None:
        item = BenchmarkItem(
            item_id="diamond|0",
            question="Q\n\n(A) a\n(B) b",
            choices=(),
            gold_index=-1,
            metadata={"answer_type": ANSWER_TYPE, "gold_answer": "B"},
        )
        assert generative.is_gradable(item)


class TestTheSystemPrompt:
    """Loaded from olmo-eval, because it is what produces the format the grader reads."""

    def test_it_is_the_registered_task_s_own(self) -> None:
        registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
        expected = registry.get_task("gpqa_diamond").config.formatter.system_prompt
        assert generative.get_system_prompt("gpqa") == expected

    def test_an_unknown_source_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="Unknown system_prompt_source 'minerva'"):
            generative.get_system_prompt("minerva")

    def test_a_system_prompt_without_chat_format_is_refused(self) -> None:
        """A completion prompt has no system turn, so it would silently drop it."""
        with pytest.raises(ValueError, match="no system turn"):
            generative.GenerationConfig(system_prompt_source="gpqa", chat_format=False)

    def test_chat_format_with_no_source_is_still_fine(self) -> None:
        """IFEval's shape: chat, and deliberately no standing instruction."""
        config = generative.GenerationConfig(chat_format=True, num_fewshot=0)
        assert config.system_prompt_source is None


class TestTheConfigEntry:
    def test_it_reproduces_the_task_defaults(self) -> None:
        entry = gpqa_config()
        assert entry["num_fewshot"] == 0
        assert entry["prompt_style"] == "gpqa"
        assert entry["chat_format"] is True
        assert entry["system_prompt_source"] == "gpqa"
        assert entry["max_new_tokens"] == 1024

    def test_it_declares_no_stop_sequences(self) -> None:
        """The answer is the last thing a chain of thought says; a stop would cut it."""
        assert gpqa_config()["stop_sequences"] == []

    def test_it_reaches_the_generation_config(self, tmp_path, monkeypatch) -> None:
        from ..style import UniMcqStyle

        seen: list[generative.GenerationConfig] = []
        monkeypatch.setattr(
            generative,
            "load_generative_model",
            lambda _dir, config: (
                seen.append(config) or generative.GenerativeScorer(lambda _: "", config)
            ),
        )
        request = grading.GradingRequest(
            dataset=DATASET,
            modality=grading.GENERATIVE,
            generation=UniMcqStyle().generation_settings,
        )
        grading.load_grader(request, tmp_path, grading.GradingSettings())

        config = seen[0]
        assert config.prompt_style == "gpqa"
        assert config.chat_format is True
        assert config.system_prompt_source == "gpqa"
        assert config.max_new_tokens == 1024
        assert config.stop_sequences == ()

    def test_the_prompt_template_leaves_the_stem_alone(self) -> None:
        """The stem is already the user turn, choice block included."""
        template = generative.get_prompt_template("gpqa")
        assert template.render("Q\n\n(A) a\n(B) b", []) == "Q\n\n(A) a\n(B) b"

    def test_a_few_shot_block_under_it_raises(self) -> None:
        """0-shot only: the tasks prepend nothing and there is no worked-example form."""
        with pytest.raises(ValueError, match="0-shot only"):
            generative.get_prompt_template("gpqa").render("Q", [{"question": "q"}])


class TestVendoringFreezesTheShuffle:
    """The choice block and the letter that answers it must leave as one artifact."""

    @pytest.fixture(autouse=True)
    def _needs_olmo_eval(self) -> None:
        pytest.importorskip("olmo_eval.common.formatters")

    def record(self, **overrides: Any) -> dict[str, Any] | None:
        return vendor_bank.generative_record(
            "diamond|0", instance(**overrides), answer_type=ANSWER_TYPE
        )

    def test_the_choice_block_is_written_into_the_stem(self) -> None:
        record = self.record()
        assert record is not None
        assert record["question"] == (
            "Which particle mediates the strong interaction?\n\n"
            "(A) photon\n(B) gluon\n(C) W boson\n(D) graviton"
        )

    def test_it_is_the_layout_the_task_s_own_formatter_writes(self) -> None:
        formatters = pytest.importorskip("olmo_eval.common.formatters")
        expected = formatters.MCQAChatFormatter().format(instance()).messages[-1]["content"]
        assert self.record()["question"] == expected

    def test_the_record_stays_choiceless_so_it_reaches_the_generative_grader(self) -> None:
        record = self.record()
        assert record["choices"] == []
        assert record["gold_index"] == -1

    def test_the_post_shuffle_gold_letter_is_recorded(self) -> None:
        metadata = self.record()["metadata"]
        assert metadata["gold_answer"] == "B"
        assert metadata["answer_type"] == ANSWER_TYPE
        assert metadata["modality"] == "generative"

    def test_a_gold_letter_outside_the_choice_block_aborts(self) -> None:
        with pytest.raises(SystemExit, match="must be one of"):
            self.record(gold_answer="E")

    def test_a_gold_letter_disagreeing_with_gold_idx_aborts(self) -> None:
        """Both come from one shuffle upstream; if they part, one ordering is stale."""
        with pytest.raises(SystemExit, match="disagree about which"):
            self.record(metadata={"index": 7, "gold_idx": 2})

    def test_a_choiceless_generative_instance_is_untouched(self) -> None:
        """GSM8K and MATH must be vendored exactly as before."""
        record = vendor_bank.generative_record(
            "0", instance(choices=(), gold_answer="72", metadata={}), answer_type="numeric"
        )
        assert record["question"] == "Which particle mediates the strong interaction?"
        assert record["metadata"]["gold_answer"] == "72"

    def test_the_grader_reads_back_what_vendoring_wrote(self) -> None:
        """End to end on one item: the frozen gold and a completion that names it."""
        record = self.record()
        item = BenchmarkItem(
            item_id=record["id"],
            question=record["question"],
            choices=(),
            gold_index=-1,
            metadata=dict(record["metadata"]),
        )
        config = generative.GenerationConfig(
            num_fewshot=0,
            prompt_style="gpqa",
            chat_format=True,
            system_prompt_source="gpqa",
            max_new_tokens=1024,
            stop_sequences=(),
        )
        prompt = generative.format_generative_prompt(item, config)
        assert "(B) gluon" in prompt

        response = generative.grade_completion(item, "Colour charge.\n\nANSWER: B", config)
        assert response.correct
        assert response.metadata["grader"] == GRADER_NAME
        assert response.chosen_index == generative.NO_CHOICE_INDEX


#: What each subtask contributes to the bridge, to the bank after the ``a > 0`` filter,
#: and to the vendored bank after the nested repeats are deduplicated. The join is
#: total, so the second row is also what each subtask matched.
#:
#: Extended is unchanged across the last two rows and that is the precedence working
#: rather than a coincidence: it is the most preferred tier, so every extended row that
#: survived the filter survives deduplication too. What the other two lose is the copies
#: of questions extended already carries.
SUBTASK_BRIDGE_ROWS = {"diamond": 198, "extended": 546, "main": 448}
SUBTASK_FILTERED_ROWS = {"diamond": 99, "extended": 252, "main": 228}
SUBTASK_ITEMS = {"diamond": 24, "extended": 252, "main": 119}

#: The order a repeated question's surviving calibration is chosen in.
PRECEDENCE = ["extended", "main", "diamond"]


def manifest() -> dict[str, Any]:
    """The committed GPQA manifest, or skip."""
    path = CALIBRATED_DATASETS / DATASET / "manifest.json"
    if not path.is_file():
        pytest.skip("gpqa has not been vendored")
    return json.loads(path.read_text(encoding="utf-8"))


def vendored_items() -> list[BenchmarkItem]:
    """The committed GPQA items, loaded through the real loader."""
    from ....common.benchmark_download import load_items_from_jsonl

    path = CALIBRATED_DATASETS / DATASET / "items.jsonl"
    if not path.is_file():
        pytest.skip("gpqa has not been vendored")
    return list(load_items_from_jsonl(path, name=DATASET).items)


class TestTheVendoredBank:
    """The committed bank, and the frozen shuffle as it survived onto disk."""

    def test_the_counts_are_the_ones_upstream_reports(self) -> None:
        recorded = manifest()
        assert recorded["upstream_bank_rows"] == 1192
        assert recorded["bridge_rows"] == 1192
        assert recorded["dropped"]["non_positive_discrimination"] == 613
        assert recorded["dropped"]["not_in_task"] == 0
        assert recorded["dropped"]["not_in_bridge"] == 0
        assert recorded["dropped"]["ambiguous_item_id"] == 0
        assert recorded["dropped"]["duplicate_question"] == 184
        assert recorded["items"] == 395

    def test_the_nested_repeats_are_accounted_for_separately(self) -> None:
        """The two filters cost different things and a reader has to be able to tell.

        613 rows went because a negative discrimination scores backwards; 184 went
        because they are a second or third calibration of a question the bank already
        carries. Folding them into one number would read as a bank that discriminates
        even worse than it does.
        """
        recorded = manifest()
        assert sum(SUBTASK_FILTERED_ROWS.values()) == 579
        assert recorded["items"] + recorded["dropped"]["duplicate_question"] == 579
        assert recorded["items"] + sum(recorded["dropped"].values()) == 1192

    def test_the_precedence_that_chose_the_survivors_is_recorded(self) -> None:
        """A distribution nobody can re-derive the rule for is not provenance."""
        recorded = manifest()
        assert recorded["duplicate_question_precedence"] == PRECEDENCE
        assert recorded["subtask_items"] == SUBTASK_ITEMS

    def test_it_spans_the_three_subsets(self) -> None:
        counts: dict[str, int] = {}
        for item in vendored_items():
            label = item.item_id.split("|", 1)[0]
            counts[label] = counts.get(label, 0) + 1
        assert counts == SUBTASK_ITEMS
        assert sum(SUBTASK_BRIDGE_ROWS.values()) == manifest()["bridge_rows"]

    def test_the_diamond_only_questions_survived(self) -> None:
        """The 24 the precedence must not silently drop.

        Diamond is the least preferred tier, so a row of it reaching the bank means the
        extended and main calibrations of that question both failed the ``a > 0`` filter
        and this is the only estimate the question has. A rule that dropped every repeat
        rather than keeping one would lose all 24 questions outright.
        """
        diamond = [i for i in vendored_items() if i.item_id.startswith("diamond|")]
        assert len(diamond) == SUBTASK_ITEMS["diamond"]

    def test_no_question_is_administered_twice(self) -> None:
        """What the deduplication exists for, read off the committed stems.

        The CAT masks administered items by id, so two ids over one question let a
        session score it twice and give EAP the second scoring as fresh evidence. The
        stems carry a per-subset shuffle of the choices, so equal stems here understate
        the repeats rather than overstating them.
        """
        stems = [item.question for item in vendored_items()]
        assert len(set(stems)) == len(stems)

    def test_the_ids_are_the_composite_the_bridge_uses(self) -> None:
        """A bare integer here would mean the single-task fallback had keyed the bank."""
        for item in vendored_items():
            label, sep, position = item.item_id.partition("|")
            assert sep == "|"
            assert label in SUBTASK_BRIDGE_ROWS
            assert 0 <= int(position) < SUBTASK_BRIDGE_ROWS[label]

    def test_every_stem_carries_its_frozen_choice_block(self) -> None:
        """The letters only mean anything against the ordering the shuffle produced."""
        for item in vendored_items():
            assert "\n(A) " in item.question
            assert "\n(D) " in item.question
            assert item.choices == ()
            assert item.gold_index == -1

    def test_every_gold_letter_indexes_that_block(self) -> None:
        for item in vendored_items():
            assert item.metadata["gold_answer"] in {"A", "B", "C", "D"}
            assert item.metadata["answer_type"] == ANSWER_TYPE

    def test_the_frozen_gold_is_a_real_shuffle_rather_than_a_constant(self) -> None:
        """What makes the vendoring-time cross-check a test rather than a formality.

        ``check_choice_gold`` compares the letter written beside the block against the
        instance's own ``gold_idx``. If the shuffle put the correct answer first every
        time the comparison would pass on both a correct and a pre-shuffle letter, and
        would be evidence of nothing. It does not: the letters land on all four
        positions, so most of the bank would fail the check under the wrong ordering.
        """
        letters = [item.metadata["gold_answer"] for item in vendored_items()]
        assert set(letters) == {"A", "B", "C", "D"}
        assert max(letters.count(letter) for letter in "ABCD") < 0.5 * len(letters)

    def test_the_cross_check_fires_on_a_committed_item(self) -> None:
        """Rotate one real gold letter by one position and vendoring must refuse it.

        Driven through the committed stem rather than a constructed instance, so what is
        exercised is the data actually on disk: the block is real, the letter is the one
        the shuffle produced, and only the pairing is broken.
        """
        types = olmo_eval_types()
        item = vendored_items()[0]
        gold = item.metadata["gold_answer"]
        rotated = chr(ord("A") + (ord(gold) - ord("A") + 1) % 4)
        instance = types.Instance(
            question=item.question,
            choices=("w", "x", "y", "z"),
            gold_answer=rotated,
            metadata={"gold_idx": ord(gold) - ord("A")},
        )

        with pytest.raises(SystemExit, match="disagree about which"):
            vendor_bank.generative_record(item.item_id, instance, answer_type=ANSWER_TYPE)


def gpqa_answer(item: BenchmarkItem, correct: bool) -> str:
    """A chain of thought that ends in the taught format, naming the right letter or not."""
    gold = str(item.metadata["gold_answer"])
    letter = gold if correct else chr(ord("A") + (ord(gold) - ord("A") + 1) % 4)
    return f"Considering each option in turn, the reasoning points one way.\n\nANSWER: {letter}"


def run_real_bank(true_theta: float, *, max_items: int = 40) -> dict:
    """A full CAT over the committed 395-item bank, tokens simulated and nothing else."""
    style = UniMcqStyle()
    bank = style.download_benchmark(DATASET)
    irt = style.load_irt_params(DATASET)
    settings = grading._apply_generation_overrides(
        grading.GradingSettings(), UniMcqStyle().generation_settings, dataset=DATASET
    )
    taker = SimGenerativeTaker(
        true_theta,
        vendored_params(DATASET),
        bank.items,
        recover_question=lambda prompt: prompt,
        answer=gpqa_answer,
    )
    report = cat_loop.run_cat(
        style,
        bank=bank,
        irt_bank=irt,
        model=generative.GenerativeScorer(taker, settings.generation),
        se_threshold=0.3,
        max_items=max_items,
    )
    return report.to_dict()


class TestAbilityRecoveryOnTheRealBank:
    """Theta recovery over the committed bank, through the real engine and extractor."""

    @pytest.fixture(autouse=True)
    def _needs_olmo_eval(self) -> None:
        pytest.importorskip("olmo_eval.evals.tasks.gpqa")

    run = staticmethod(run_real_bank)

    def test_a_session_converges_and_stops_on_precision(self) -> None:
        report = self.run(0.5)

        assert report["metadata"]["stop_reason"] == "precision_reached"
        assert report["metadata"]["standard_error"] <= 0.3
        assert report["metadata"]["bank_size"] == 395
        assert np.isfinite(report["ability"]["theta"])

    @pytest.mark.parametrize("true_theta", [-1.0, 0.0, 1.0])
    def test_the_estimate_lands_near_the_truth(self, true_theta: float) -> None:
        assert self.run(true_theta)["metadata"]["theta"] == pytest.approx(true_theta, abs=0.6)

    def test_theta_recovers_monotonically(self) -> None:
        """The ordering is what a checkpoint-to-checkpoint comparison rests on."""
        estimates = [self.run(theta)["metadata"]["theta"] for theta in (-1.5, -0.5, 0.5, 1.5)]

        assert all(b > a for a, b in pairwise(estimates)), estimates

    def test_the_report_names_the_grader_that_produced_it(self) -> None:
        report = self.run(0.5)

        assert report["metadata"]["modality"] == "generative"
        assert "chain of thought" in report["metadata"]["scoring_note"]
        assert all(r["metadata"]["grader"] == GRADER_NAME for r in report["responses"])

    def test_nothing_came_back_ungradable(self) -> None:
        """Every item carries a gold letter, so a fabricated zero would be a bank fault."""
        assert self.run(0.5)["metadata"]["ungradable"]["count"] == 0


class TestTheSpec:
    def test_it_declares_the_generative_modality_and_its_grader(self) -> None:
        spec = SUPPORTED[DATASET]
        assert spec.modality == "generative"
        assert spec.answer_type == ANSWER_TYPE
        assert spec.answer_type in generative.ANSWER_GRADERS

    def test_it_is_still_multi_task(self) -> None:
        spec = SUPPORTED[DATASET]
        assert spec.is_multi_task
        assert spec.task_names == SUBTASKS

    def test_it_can_run_today(self) -> None:
        from ..datasets import ready_names, supported_names

        assert SUPPORTED[DATASET].blocked is None
        assert DATASET in supported_names()
        assert DATASET in ready_names()

    def test_the_notes_still_record_what_the_blocker_was(self) -> None:
        """The scope is granted and the diagnosis is worth keeping.

        A fine-grained HuggingFace token without the gated-repository scope resolves
        ``dataset_info`` and then 403s every data file, which ``datasets`` surfaces as a
        FileNotFoundError reading like a network fault. Anyone who hits it again on
        another gated dataset should not have to re-derive that.
        """
        notes = SUPPORTED[DATASET].notes
        assert "token scope" in notes
        assert "403" in notes
