"""The chain-of-thought grading GPQA is *not* scored by, and why the machinery survives.

Until 2026-08-08 this bank was vendored as generative and graded by
:class:`~diagnostics.mcq_cat.common.generative.GPQACoTLetter`: the registered olmo-eval
tasks default to an ``MCQAChatFormatter`` carrying an expert-scientist system prompt that
asks for step-by-step reasoning ending in ``ANSWER: X``, then letter extraction and
``MultipleChoiceScorer``. That default was taken as the convention because Research never
wired GPQA into its CAT, so there was nothing to weigh it against.

There was. The harvest the difficulties came from is itself a record of the convention and
is readable, and it says multiple choice. :mod:`.test_gpqa_bank` is where the bank lives
now; what remains here is the road not taken, still tested, for two reasons.

The first is that the decision is only legible beside the thing it rejected. A reader
asking why an obviously multiple-choice benchmark was ever graded by chain of thought is
answered by :class:`TestTheTaskStillDefaultsToChainOfThought`, which reads that default off
the live registry rather than off a claim in a note -- and would fail if olmo-eval ever
changed it, at which point the note in ``datasets.py`` becomes wrong too.

The second is that the grader is registered code. ``gpqa_letter`` still resolves and
``get_system_prompt("gpqa")`` still returns the task's own prompt; nothing selects either
today, and leaving registered machinery untested is how it rots into something that
raises the first time a future bank reaches for it. GPQA is not the only benchmark whose
tasks default to this shape.

Nothing here enumerates the dataset. ``Idavidrein/gpqa`` is gated, so the tests work from
a constructed instance or from the task's configuration, neither of which touches the
network.
"""

from __future__ import annotations

from typing import Any

import pytest

from ....base import BenchmarkItem
from ....common import generative
from ..datasets import SUPPORTED

DATASET = "gpqa"
ANSWER_TYPE = "gpqa_letter"
GRADER_NAME = "gpqa_cot_letter"

#: The three registered tasks the bank spans, in the order its bridge names them.
SUBTASKS = ("gpqa_diamond", "gpqa_extended", "gpqa_main")


def olmo_eval_types() -> Any:
    """``olmo_eval.common.types``, or skip."""
    return pytest.importorskip("olmo_eval.common.types")


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


class TestTheTaskStillDefaultsToChainOfThought:
    """Read off the registry, because the rejected convention has to stay identifiable.

    If olmo-eval changes this default the reasoning recorded in ``datasets.py`` stops
    describing anything, and a reader would have no way to tell that from a note that was
    always wrong.
    """

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

    def test_the_spec_still_names_the_plain_tasks_rather_than_a_variant(self) -> None:
        """The ``:mc`` variant is not what this bank switched to, and the difference
        matters.

        olmo-eval's log-likelihood formulation exists, but it writes ``A.`` labels and
        is not the presentation Open LLM Leaderboard v2 used. What
        :mod:`.test_gpqa_bank` reproduces is lm-evaluation-harness's
        ``leaderboard_gpqa``, which is where the difficulties came from, so the spec goes
        on naming the plain tasks and the MCQ layout is this style's own.
        """
        registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
        assert "mc" in registry.list_variants("gpqa_diamond")["gpqa_diamond"]
        assert SUPPORTED[DATASET].task_names == SUBTASKS
        assert all(":" not in name for name in SUBTASKS)

    def test_every_subtask_the_spec_names_is_registered(self) -> None:
        registry = pytest.importorskip("olmo_eval.evals.tasks.common.registry")
        for name in SUBTASKS:
            assert registry.task_exists(name), name


class TestNothingSelectsItAnyMore:
    """The retirement itself, asserted rather than left to be inferred from an absence."""

    def test_the_bank_is_multiple_choice(self) -> None:
        assert SUPPORTED[DATASET].modality == "mcq"

    def test_no_dataset_asks_for_this_grader(self) -> None:
        selected = {
            spec.answer_type for spec in SUPPORTED.values() if spec.modality == "generative"
        }
        assert ANSWER_TYPE not in selected

    def test_the_grader_is_nonetheless_still_reachable(self) -> None:
        """Registered code with no caller is still code, and is tested below."""
        assert ANSWER_TYPE in generative.ANSWER_GRADERS


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

    def test_the_refusal_names_the_alternative_that_was_eventually_taken(self) -> None:
        """The message has to survive the reader who is trying to score a base model.

        Dropping ``chat_format`` and keeping the source is what someone reaches for after
        ifeval's flip made a base checkpoint runnable, and the guard's job is to be more
        than a locked door: pasting a system prompt into a completion prompt is a
        presentation nothing was calibrated behind. So the message names the alternative
        that exists -- lm-evaluation-harness's GPQA is a log-likelihood ranking with no
        system prompt at all -- and that alternative is what this bank went on to adopt.
        """
        with pytest.raises(ValueError) as exc:
            generative.GenerationConfig(system_prompt_source="gpqa", chat_format=False)
        message = str(exc.value)

        assert "instruct checkpoint" in message
        assert "log-likelihood" in message
        assert "ifeval" in message

    def test_chat_format_with_no_source_is_still_fine(self) -> None:
        """Chat with no standing instruction stays a legal shape.

        It is nothing's shape today -- ifeval had it until the bank's mixed calibration
        made the completion framing the better half to match -- and it stays permitted
        because the pairing is a per-benchmark fact rather than a rule: a chat bank whose
        instructions are all in the prompt text wants exactly this.
        """
        config = generative.GenerationConfig(chat_format=True, num_fewshot=0)
        assert config.system_prompt_source is None


class TestVendoringCanStillFreezeAShuffle:
    """``generative_record``'s choice-carrying branch, which nothing reaches today.

    A generative bank whose instances carry choices has to put the block *in the prompt*,
    because a chain of thought must see the options it is choosing among, and has to
    check the frozen letter against the shuffle that produced it. GPQA was the only bank
    that did, and the equivalent guard for the MCQ bank is now
    ``vendor_bank.check_choice_order``, which is a stronger one -- it holds the ordering
    to a committed control rather than only to the instance's own two statements about
    itself.

    Kept under test because the branch is still live code on the path every generative
    bank takes, and an unreachable branch that raises is worse than one that works.
    """

    @pytest.fixture(autouse=True)
    def _needs_olmo_eval(self) -> None:
        pytest.importorskip("olmo_eval.common.formatters")

    def record(self, **overrides: Any) -> dict[str, Any] | None:
        from ..scripts import vendor_bank

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

    def test_a_gold_letter_outside_the_choice_block_aborts(self) -> None:
        with pytest.raises(SystemExit, match="must be one of"):
            self.record(gold_answer="E")

    def test_a_gold_letter_disagreeing_with_gold_idx_aborts(self) -> None:
        """Both come from one shuffle upstream; if they part, one ordering is stale."""
        with pytest.raises(SystemExit, match="disagree about which"):
            self.record(metadata={"index": 7, "gold_idx": 2})

    def test_a_choiceless_generative_instance_is_untouched(self) -> None:
        """GSM8K, MATH and IFEval must be vendored exactly as before."""
        from ..scripts import vendor_bank

        record = vendor_bank.generative_record(
            "0", instance(choices=(), gold_answer="72", metadata={}), answer_type="numeric"
        )
        assert record["question"] == "Which particle mediates the strong interaction?"
        assert record["metadata"]["gold_answer"] == "72"


class TestThePromptTemplate:
    """The generative ``gpqa`` template, which is a different registry from the MCQ style
    of the same name and is no longer selected by anything."""

    def test_it_leaves_the_stem_alone(self) -> None:
        """It existed because the stem was already the whole user turn, block included."""
        template = generative.get_prompt_template("gpqa")
        assert template.render("Q\n\n(A) a\n(B) b", []) == "Q\n\n(A) a\n(B) b"

    def test_a_few_shot_block_under_it_raises(self) -> None:
        """0-shot only: the tasks prepend nothing and there is no worked-example form."""
        with pytest.raises(ValueError, match="0-shot only"):
            generative.get_prompt_template("gpqa").render("Q", [{"question": "q"}])
