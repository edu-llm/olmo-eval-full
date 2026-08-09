"""Verifier-scored grading for the IFEval bank, and the interface it widened.

IFEval is the first bank here with no gold answer anywhere in it. Its items carry the
instruction ids and per-instruction arguments the prompt states, and a response is
correct when every one of those verifiers accepts it -- ``prompt_level_strict_acc``,
the metric the bank was calibrated under. That is a second shape of grader, not a
second modality: one sampled completion still yields one binary, so the frozen CAT
engine, the EAP update and p-IRT are all untouched.

Three failure modes are what these guard against, in rising order of quietness. A
grader called with the wrong signature crashes, which is fine. An item admitted without
its constraints scores incorrect for every model and reads as a weak checkpoint. And
loose scoring in place of strict, or a stop sequence cutting a multi-paragraph answer,
produces a perfectly well-formed report whose theta is on a different scale from the
bank's difficulties.

The verifier registry itself lives in ``ifbench``, a declared dependency installed from
a git URL. Tests that need real verifier behaviour skip without it; the rest install a
stub registry in ``sys.modules`` so the real ``IFEvalScorer``, the real grader and the
real CAT engine all run on machines that do not have it.
"""

from __future__ import annotations

import json
import re
import sys
import types
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ....base import BenchmarkItem, CATState
from ....common import cat_loop, generative, grading, inference
from .. import style as style_mod
from ..datasets import SUPPORTED
from ..style import UniMcqStyle
from .conftest import CALIBRATED_DATASETS, SimGenerativeTaker, vendored_params

DATASET = "ifeval"

#: The stub registry accepts a response only if it contains this. It stands in for
#: "every instruction passed", so a completion carrying it is correct under strict
#: scoring and one without it is not.
FOLLOWED = "[followed]"


@pytest.fixture(autouse=True)
def _bank_reachable_despite_the_block(monkeypatch):
    """Let this module resolve the ifeval bank while the dataset is blocked.

    ``ifeval`` was blocked on 2026-08-08 because a model reciting its prompt passes 129
    of its 511 items, so ``resolve_bank`` now refuses it and every test here that reaches
    the committed bank through ``download_benchmark`` would error before grading anything.

    Skipping them instead would be the wrong trade. Nothing in this module asserts that
    the dataset is *available* -- that claim is tested once, deliberately, in
    ``test_resolve.TestAllowlist`` -- and what these tests actually exercise is the
    generative grading path against real vendored items and the real verifier registry.
    That path is still live for ``leaderboard_math`` and the block is meant to be
    temporary, lifted when the echo-baseline guard lands. Letting the coverage lapse for
    the duration is how the code would rot in the interval.

    Scoped to this module and applied to the registry entry rather than to
    ``resolve_bank``, so production refusal is untouched and no test-only argument leaks
    into the resolver. ``DatasetSpec`` is frozen, so this swaps in a replaced copy rather
    than assigning to the field.
    """
    monkeypatch.setitem(SUPPORTED, DATASET, replace(SUPPORTED[DATASET], blocked=None))


def ifeval_item(
    instruction_ids: tuple[str, ...] = ("punctuation:no_comma",),
    kwargs: tuple[dict, ...] = ({},),
    question: str = "Write a summary with no commas.",
) -> BenchmarkItem:
    """A generative item shaped as a vendored IFEval bank record."""
    return BenchmarkItem(
        item_id="0",
        question=question,
        choices=(),
        gold_index=-1,
        metadata={
            "answer_type": "ifeval_strict",
            "modality": "generative",
            "instruction_id_list": list(instruction_ids),
            "kwargs": [dict(kw) for kw in kwargs],
        },
    )


class StubInstruction:
    """The slice of an IFBench verifier that ``IFEvalScorer`` actually calls."""

    def __init__(self, instruction_id: str) -> None:
        self.instruction_id = instruction_id
        self.built: list[dict] = []

    def get_instruction_args_keys(self) -> list[str]:
        return ["prompt_to_repeat"]

    def build_description(self, **kwargs: object) -> str:
        self.built.append(dict(kwargs))
        return self.instruction_id

    def get_instruction_args(self) -> dict:
        return {}

    def check_following(self, response: str) -> bool:
        return FOLLOWED in response


@pytest.fixture
def stub_ifbench(monkeypatch):
    """Install a fake ``ifbench`` whose every verifier looks for :data:`FOLLOWED`.

    Everything above the verifier stays real: olmo-eval's ``IFEvalScorer`` resolves the
    ids, fills in ``prompt_to_repeat``, runs strict and all eight loose variants, and
    writes the pass lists that :class:`IFEvalPromptStrict` reads. Only the verifiers'
    own bodies are simulated, which is the part ``ifbench`` owns and this machine may
    not have.

    Because the scorer is real, ``olmo_eval`` has to be importable. Absent it the grader
    raises rather than returning a verdict, which pytest reports as a failure -- so a
    bare checkout would show thirteen red tests for a missing optional package instead
    of the clean skip every other file here gives. Guarding in the fixture rather than
    per test keeps the two from drifting as tests are added.
    """
    pytest.importorskip(
        "olmo_eval.common.scorers",
        reason="IFEval grading runs olmo-eval's real IFEvalScorer; run with PYTHONPATH=src",
    )
    registry = types.ModuleType("ifbench.instructions_registry")
    registry.INSTRUCTION_DICT = _AlwaysRegistered()
    package = types.ModuleType("ifbench")
    package.instructions_registry = registry
    monkeypatch.setitem(sys.modules, "ifbench", package)
    monkeypatch.setitem(sys.modules, "ifbench.instructions_registry", registry)
    return registry


class _AlwaysRegistered(dict):
    """A registry that knows every instruction id the bank happens to name."""

    def __missing__(self, key: str) -> type[StubInstruction]:
        return StubInstruction


def ifeval_answer(item: BenchmarkItem, correct: bool) -> str:
    """A response that satisfies the item's constraints, or plainly does not."""
    if correct:
        return f"Here is the response.\n\n{FOLLOWED}\n\nIt closes here."
    return "Here is a response that ignores what was asked."


class TestTheGraderShape:
    def test_ifeval_strict_resolves_to_the_verifier_grader(self) -> None:
        grader = generative.get_answer_grader("ifeval_strict")
        assert isinstance(grader, generative.IFEvalPromptStrict)
        assert grader.name == "ifeval_prompt_strict"

    def test_it_is_an_item_grader_and_not_an_answer_grader(self) -> None:
        """The dispatch in ``apply_grader`` turns on exactly this."""
        grader = generative.get_answer_grader("ifeval_strict")
        assert isinstance(grader, generative.ItemGrader)
        assert not isinstance(grader, generative.AnswerGrader)

    def test_the_gold_matched_graders_are_untouched(self) -> None:
        """Widening the interface must not have moved the other two into it."""
        for answer_type in ("numeric", "math_latex"):
            grader = generative.get_answer_grader(answer_type)
            assert isinstance(grader, generative.AnswerGrader)
            assert not isinstance(grader, generative.ItemGrader)

    def test_it_declares_the_metadata_it_decides_on(self) -> None:
        """Vendoring copies exactly these keys, so the two cannot drift apart."""
        grader = generative.get_answer_grader("ifeval_strict")
        assert grader.required_metadata == ("instruction_id_list", "kwargs")


class TestStrictVerdicts:
    def test_all_instructions_passing_is_correct(self, stub_ifbench) -> None:
        item = ifeval_item(("punctuation:no_comma", "length_constraints:number_words"), ({}, {}))
        verdict = generative.get_answer_grader("ifeval_strict").grade_item(
            f"A response. {FOLLOWED}", item
        )

        assert verdict.correct is True
        assert verdict.detail["strict"] == [True, True]
        assert verdict.gold is None
        assert verdict.extracted is None

    def test_one_failing_instruction_fails_the_item(self, stub_ifbench, monkeypatch) -> None:
        """Prompt-level strict is all-or-nothing; a partial pass is still a 0."""

        class OnlyFirstPasses(StubInstruction):
            def check_following(self, response: str) -> bool:
                return self.instruction_id.endswith("first")

        monkeypatch.setitem(stub_ifbench.INSTRUCTION_DICT, "rule:first", OnlyFirstPasses)
        monkeypatch.setitem(stub_ifbench.INSTRUCTION_DICT, "rule:second", OnlyFirstPasses)

        verdict = generative.get_answer_grader("ifeval_strict").grade_item(
            f"A response. {FOLLOWED}", ifeval_item(("rule:first", "rule:second"), ({}, {}))
        )

        assert verdict.correct is False
        assert verdict.detail["strict"] == [True, False]

    def test_no_instruction_results_is_a_bank_fault_not_a_wrong_answer(self, stub_ifbench) -> None:
        """The 0 enters the response pattern, so it has to be marked as not the model's."""
        verdict = generative.get_answer_grader("ifeval_strict").grade_item(
            "anything", ifeval_item((), ())
        )

        assert verdict.correct is False
        assert verdict.ungradable_reason is not None
        assert "instruction" in verdict.ungradable_reason

    def test_the_prompt_is_handed_back_for_the_verifiers_that_quote_it(self, stub_ifbench) -> None:
        """``prompt_to_repeat`` is filled from the item's question, not left empty.

        The verifiers that check a response repeats the instruction read it from there,
        and would reject every response if it arrived blank.
        """
        seen: list[dict] = []

        class Recording(StubInstruction):
            def build_description(self, **kwargs: object) -> str:
                seen.append(dict(kwargs))
                return self.instruction_id

        stub_ifbench.INSTRUCTION_DICT["combination:repeat_prompt"] = Recording
        item = ifeval_item(("combination:repeat_prompt",), ({},), question="Repeat this exactly.")
        generative.get_answer_grader("ifeval_strict").grade_item(f"x {FOLLOWED}", item)

        assert seen and seen[0]["prompt_to_repeat"] == "Repeat this exactly."


class TestGradeCompletionDispatch:
    def test_a_gold_less_item_reaches_the_verifier_grader(self, stub_ifbench) -> None:
        graded = generative.grade_completion(
            ifeval_item(), f"ok {FOLLOWED}", generative.GenerationConfig(stop_sequences=())
        )

        assert graded.correct is True
        assert graded.metadata["grader"] == "ifeval_prompt_strict"
        assert graded.metadata["gold_answer"] is None
        assert graded.metadata["grader_detail"]["instruction_id_list"] == ["punctuation:no_comma"]

    def test_the_gold_matched_response_shape_gains_no_grader_detail(self) -> None:
        """A gold-matched bank stays free of the verifier grader's per-item detail."""
        item = BenchmarkItem(
            item_id="g0",
            question="Q?",
            choices=(),
            gold_index=-1,
            metadata={"gold_answer": "72", "answer_type": "numeric"},
        )
        graded = generative.grade_completion(
            item, " So the answer is 72.", generative.GenerationConfig(num_fewshot=8)
        )

        assert set(graded.metadata) == {
            "modality",
            "grader",
            "completion",
            "extracted_answer",
            "gold_answer",
            "num_fewshot",
            "ungradable",
        }

    def test_a_gold_matched_item_with_no_gold_is_marked_rather_than_scored(self) -> None:
        """Widening the interface must not have made a missing gold pass for an answer."""
        bare = BenchmarkItem(item_id="g0", question="Q?", choices=(), gold_index=-1)
        graded = generative.grade_completion(bare, "72", generative.GenerationConfig())

        assert graded.correct is False
        assert graded.metadata[generative.UNGRADABLE_KEY] is True

    def test_the_report_records_which_constraints_failed(self, stub_ifbench, monkeypatch) -> None:
        """A wrong IFEval answer has no wrong answer in it, so the reason must be stored."""

        class NeverPasses(StubInstruction):
            def check_following(self, response: str) -> bool:
                return False

        monkeypatch.setitem(stub_ifbench.INSTRUCTION_DICT, "punctuation:no_comma", NeverPasses)
        graded = generative.grade_completion(
            ifeval_item(), "a response", generative.GenerationConfig(stop_sequences=())
        )

        assert graded.correct is False
        assert graded.metadata["grader_detail"] == {
            "instruction_id_list": ["punctuation:no_comma"],
            "strict": [False],
        }


class TestModalityGuard:
    def test_a_verifier_scored_bank_passes_the_generative_guard(self) -> None:
        """It has no gold, and the guard used to require one of every generative item."""
        request = grading.GradingRequest(dataset=DATASET, modality=grading.GENERATIVE)
        grading.check_bank_modality(request, [ifeval_item()])

    def test_an_item_stripped_of_its_constraints_is_still_refused(self) -> None:
        """Admitting it would score it incorrect for every model, silently."""
        stripped = BenchmarkItem(
            item_id="0",
            question="q",
            choices=(),
            gold_index=-1,
            metadata={"answer_type": "ifeval_strict", "modality": "generative"},
        )
        request = grading.GradingRequest(dataset=DATASET, modality=grading.GENERATIVE)
        with pytest.raises(grading.ModalityMismatch, match="answer_type"):
            grading.check_bank_modality(request, [stripped])

    def test_an_unknown_answer_type_is_refused_before_a_checkpoint_loads(self) -> None:
        unknown = BenchmarkItem(
            item_id="0",
            question="q",
            choices=(),
            gold_index=-1,
            metadata={"answer_type": "essay_rubric", "gold_answer": "x"},
        )
        request = grading.GradingRequest(dataset=DATASET, modality=grading.GENERATIVE)
        with pytest.raises(grading.ModalityMismatch):
            grading.check_bank_modality(request, [unknown])


class TestPromptAndSampling:
    def test_the_prompt_is_the_instruction_verbatim(self) -> None:
        """Any framing is an extra constraint the verifiers were never told about."""
        item = ifeval_item(question="Write three sections. Do not use commas.")
        config = generative.GenerationConfig(num_fewshot=0, prompt_style="ifeval")

        assert generative.format_generative_prompt(item, config) == item.question

    def test_a_fewshot_block_behind_the_ifeval_template_is_refused(self, monkeypatch) -> None:
        """Silently borrowing gsm8k's block would prompt IFEval as a Q/A benchmark."""
        monkeypatch.setitem(
            generative.FEWSHOT_SOURCES, "gsm8k", lambda: ({"question": "q", "answer": "a"},)
        )
        config = generative.GenerationConfig(num_fewshot=1, prompt_style="ifeval")

        with pytest.raises(ValueError, match="0-shot only"):
            generative.format_generative_prompt(ifeval_item(), config)

    def test_the_committed_config_matches_the_leaderboard_task(self, tmp_path: Path) -> None:
        """The settings a run would actually use, read off config.yaml.

        Every one of these that decides whether an item passes is
        lm-evaluation-harness's ``leaderboard_ifeval`` verbatim: ``doc_to_text`` is the
        bare ``prompt`` field, ``num_fewshot: 0``, and generation kwargs of
        ``until: []``, ``do_sample: false``. That includes the completion framing, which
        the harness applies outside the task through ``--apply_chat_template`` rather
        than in it.

        The budget is the exception and is pinned separately below, because lm-eval
        sends a flat 1280 and this style sends a per-item budget.
        """
        config = _resolved_generation_config(DATASET)

        assert config.num_fewshot == 0
        assert config.prompt_style == "ifeval"
        assert config.chat_format is False
        assert config.stop_sequences == ()

    def test_no_stop_sequence_survives_a_multi_paragraph_answer(self) -> None:
        """A blank-line stop would cut most responses into a length-constraint failure."""
        config = _resolved_generation_config(DATASET)
        answer = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        assert generative.truncate_at_stop(answer, config.stop_sequences) == answer


def _resolved_generation_config(dataset: str) -> generative.GenerationConfig:
    """The generation settings the committed config.yaml resolves to for ``dataset``."""
    settings = grading._apply_generation_overrides(
        grading.GradingSettings(), UniMcqStyle().generation_settings, dataset=dataset
    )
    return settings.generation


FLOOR = generative.UNCONSTRAINED_FLOOR_TOKENS

#: The ceiling ``config.yaml`` pins for this bank, matching olmo-eval and lm-eval.
CEILING = 1280

#: Tokens per word the fake tokenizer below charges. Exact, so an expected budget is
#: arithmetic rather than an approximation, and deliberately not any real tokenizer's
#: figure: there is no longer a constant to match, and a test that hard-coded one
#: model's ratio would be pinning the thing the implementation stopped doing.
RATIO = 2


def counting_tokenizer(tokens_per_word: int = RATIO):
    """A ``text -> token count`` charging ``tokens_per_word`` for every whitespace word.

    Stands in for the live tokenizer the cascade now measures with. Being exact is the
    point: every budget below is a closed-form expression a reader can check, and no test
    needs network access to a real tokenizer.
    """

    def count(text: str) -> int:
        return tokens_per_word * len(text.split())

    return count


#: Effectively no ceiling. The cascade's arithmetic and the ceiling are separate claims, so
#: the mechanics are tested with the clamp out of the way and the clamp gets its own class.
NO_CEILING = 1_000_000


def budget(
    *declared: tuple[str, dict],
    question: str = "Write something.",
    ceiling: int = NO_CEILING,
    ratio: float = RATIO,
) -> int:
    """The per-item budget for an item declaring exactly ``declared``.

    Built from pairs rather than a mapping, so a test can declare the same instruction id
    twice -- which two items in the real bank do.
    """
    item = ifeval_item(
        instruction_ids=tuple(name for name, _ in declared),
        kwargs=tuple(kw for _, kw in declared),
        question=question,
    )
    return generative.ifeval_token_budget(
        item,
        ceiling=ceiling,
        tokens_per_word=ratio,
        count_tokens=counting_tokenizer(int(ratio)),
    )


def words(count: int, relation: str = "at least") -> tuple[str, dict]:
    return ("length_constraints:number_words", {"relation": relation, "num_words": count})


class TestWordsToTokens:
    """The conversion itself, which is the whole reason this is not the declared count."""

    def test_a_word_count_is_converted_to_tokens_and_not_used_raw(self) -> None:
        """The units differ, and using the raw count is the bug this exists to avoid.

        ``num_words`` is words and ``max_new_tokens`` is tokens, and English prose runs well
        above one token per word on every tokenizer. A budget of 900 for "at least 900
        words" would cut a compliant answer short and the verifier would then grade the item
        as violating a length constraint the model had actually met.
        """
        assert budget(words(900)) > 900
        assert budget(words(900)) == 900 * RATIO + generative.DETECTION_HEADROOM_TOKENS

    def test_the_ratio_is_whatever_the_live_tokenizer_charges(self) -> None:
        """No constant: the conversion tracks the tokenizer it is given.

        This is the whole of the change from a surveyed multiplier. A coarser tokenizer must
        produce a larger budget for the same declared count, because the same 300 words cost
        it more tokens, and a budget fitted to one model would under-budget the other.
        """
        assert budget(words(300), ratio=1) == 300 + generative.DETECTION_HEADROOM_TOKENS
        assert budget(words(300), ratio=3) == 900 + generative.DETECTION_HEADROOM_TOKENS
        assert budget(words(300), ratio=3) > budget(words(300), ratio=1)

    def test_no_fallback_multiplier_is_left_behind(self) -> None:
        """The surveyed constants are gone rather than unused.

        An unused constant next to a run-time measurement is worse than no constant: the
        next reader cannot tell which one the budget is using.
        """
        for gone in ("TOKENS_PER_WORD", "CAPITAL_TOKENS_PER_WORD", "NON_ENGLISH_FLOOR_TOKENS"):
            assert not hasattr(generative, gone), f"{gone} should have been removed"

    def test_an_all_caps_item_is_measured_under_uppercase(self) -> None:
        """Uppercase is a different tokenization regime, not a rounding difference.

        ``change_case:english_capital`` obliges the whole response into capitals. The
        penalty is *measured* on the live tokenizer rather than assumed, because it does not
        transfer between them -- an uncased WordPiece tokenizer pays nothing, Llama-2 pays
        84%. Here the fake charges double for an uppercased prompt, and the budget follows.
        """

        def shouty(text: str) -> int:
            return (2 * RATIO if text.isupper() else RATIO) * len(text.split())

        item = ifeval_item(
            instruction_ids=("length_constraints:number_words", "change_case:english_capital"),
            kwargs=({"num_words": 100}, {}),
            question="write it loudly",
        )
        resolved = generative.ifeval_token_budget(
            item, ceiling=NO_CEILING, tokens_per_word=RATIO, count_tokens=shouty
        )

        assert resolved == 100 * (2 * RATIO) + generative.DETECTION_HEADROOM_TOKENS
        assert resolved > budget(words(100))

    def test_the_lowercase_and_no_comma_constraints_get_no_multiplier(self) -> None:
        """Measured, and both leave the ratio within 1% of as-written."""
        plain = budget(words(100))

        assert budget(words(100), ("change_case:english_lowercase", {})) == plain
        assert budget(words(100), ("punctuation:no_comma", {})) == plain

    def test_the_running_ratio_never_falls_below_the_corpus_aggregate(self) -> None:
        """A single prompt is a bad estimator, and the naive version truncates.

        Measured over the real bank, per-item prompt ratios run 1.08 to 2.10 against a
        corpus aggregate of 1.285, and taking each item's own ratio alone under-budgets 77
        items by up to 185 tokens. So the scorer takes the maximum of the item's own ratio
        and the running aggregate over everything it has measured. Here the first item is
        dense and the second sparse; the second must not be budgeted at its own thin rate.
        """
        scorer = generative.GenerativeScorer(
            _BudgetedCompleter(), generative.GenerationConfig(max_new_tokens=CEILING)
        )

        dense = scorer._tokens_per_word("aaaa bbbb")
        sparse = scorer._tokens_per_word("a b c d e f g h i j")

        assert dense == RATIO
        assert sparse == RATIO, "a fixed-rate tokenizer gives one ratio either way"

        uneven = generative.GenerativeScorer(
            _BudgetedCompleter(count=lambda text: 20 if "dense" in text else 1),
            generative.GenerationConfig(max_new_tokens=CEILING),
        )
        first = uneven._tokens_per_word("dense text here")
        second = uneven._tokens_per_word("a b c d e f g h i j")

        assert first == pytest.approx(20 / 3)
        assert second > 1 / 10, "not the sparse item's own thin ratio"
        assert second == pytest.approx(21 / 13), "the running aggregate"


class TestDetectionHeadroom:
    """Why the budget is the converted count *plus* something."""

    def test_an_upper_bound_gets_room_to_be_exceeded(self) -> None:
        """A budget of exactly the limit cannot tell obedience from truncation.

        "less than 20 words" graded inside a 20-word budget passes a model that would
        have run past it and was cut at the boundary by the harness. The headroom is what
        makes over-running observable, which is the measurement this bank is for.
        """
        assert budget(words(20, "less than")) > 20 * RATIO
        assert budget(words(20, "less than")) == 20 * RATIO + (
            generative.DETECTION_HEADROOM_TOKENS
        )

    def test_it_is_added_for_both_relations(self) -> None:
        """"at least N" needs it too, for the ending constraints a cut at N destroys.

        Six of the 30 "at least" items also constrain how the response ends
        (``startend:end_checker``, ``startend:quotation``,
        ``detectable_content:postscript``), and truncation at exactly N words fails that
        instruction instead of the length one.
        """
        assert budget(words(300, "at least")) == budget(words(300, "less than"))

    def test_the_headroom_is_real_rather_than_absorbed_by_rounding(self) -> None:
        assert generative.DETECTION_HEADROOM_TOKENS >= 50


class TestTheSignalCascade:
    """Every declared signal is converted; the largest demand wins."""

    def test_an_item_declaring_nothing_and_safe_to_cut_falls_to_the_floor(self) -> None:
        """84 of the 511 items: no declared size, and no constraint a cut would break.

        ``punctuation:no_comma`` is a *negative* constraint, so truncation can only help
        satisfy it. This is the only branch that economises, and the only one that could
        produce a wrong grade by being too small -- which is why the fragile ids are routed
        away from it rather than sharing it.
        """
        assert budget(("punctuation:no_comma", {})) == FLOOR
        assert budget() == FLOOR

    def test_the_floor_sits_well_above_the_measured_demand(self) -> None:
        """A prior measurement put these items near 52 tokens at the median.

        52 is a median and not a budget -- half of such answers run longer, and prose
        lengths are right-skewed -- so the floor is a multiple of it. Pinned as a ratio so
        lowering the floor toward the median fails here.
        """
        assert FLOOR >= 4 * 52
        assert FLOOR < CEILING, "the floor has to actually save something"

    def test_the_largest_of_several_signals_governs(self) -> None:
        """Not the first, not the last, and not the word count by privilege."""
        bullets = ("detectable_format:number_bullet_lists", {"num_bullets": 10})

        assert budget(words(100), bullets) == budget(bullets)
        assert budget(words(100), bullets) > budget(words(100))

    def test_a_word_count_does_not_short_circuit_the_other_signals(self) -> None:
        """The real failure this prevents, on two real items.

        ``9250210cf50a985a`` and ``0d42e8b266e3933f`` each ask for at least 300 words in
        *two* separate responses. A "has a word count, use it" rule gives them one
        300-word budget and truncates the second response.
        """
        pair = ("combination:two_responses", {})

        assert budget(words(300), pair) == 2 * budget(words(300))

    def test_sentences_paragraphs_bullets_and_sections_each_convert(self) -> None:
        """Each has its own words-per-unit figure, and each must exceed the floor."""
        assert budget(
            ("length_constraints:number_sentences", {"relation": "at least", "num_sentences": 40})
        ) == 40 * generative.WORDS_PER_SENTENCE * RATIO + (
            generative.DETECTION_HEADROOM_TOKENS
        )
        assert budget(("length_constraints:number_paragraphs", {"num_paragraphs": 8})) == (
            8 * generative.WORDS_PER_PARAGRAPH * RATIO
            + generative.DETECTION_HEADROOM_TOKENS
        )
        assert budget(("detectable_format:multiple_sections", {"num_sections": 6})) == (
            6 * generative.WORDS_PER_PARAGRAPH * RATIO
            + generative.DETECTION_HEADROOM_TOKENS
        )
        assert budget(("detectable_format:number_bullet_lists", {"num_bullets": 9})) == (
            9 * generative.WORDS_PER_BULLET * RATIO
            + generative.DETECTION_HEADROOM_TOKENS
        )

    def test_the_nth_paragraph_signal_reads_its_paragraph_count(self) -> None:
        """It carries ``num_paragraphs`` alongside the first word it pins. 12 items."""
        assert budget(
            (
                "length_constraints:nth_paragraph_first_word",
                {"num_paragraphs": 8, "nth_paragraph": 2, "first_word": "weekend"},
            )
        ) == budget(("length_constraints:number_paragraphs", {"num_paragraphs": 8}))

    def test_the_echo_is_added_on_top_rather_than_maximised(self) -> None:
        """``combination:repeat_prompt`` obliges prompt *and then* answer.

        Treating the echo as the item's whole demand is not a rounding error: it gives
        these 35 items enough for the echo and nothing else, and truncates every one.
        """
        echoed = "one two three four five six seven eight nine ten"
        repeat = ("combination:repeat_prompt", {"prompt_to_repeat": echoed})

        assert budget(repeat) == FLOOR + 10 * RATIO
        assert budget(repeat) > FLOOR
        assert budget(words(300), repeat) == budget(words(300)) + 10 * RATIO

    def test_the_echo_is_counted_exactly_rather_than_converted(self) -> None:
        """The echoed text is in hand and so is the tokenizer, so nothing is estimated.

        Distinct from every other branch, which converts a *declared count* it cannot
        tokenize. Charging the echo at the words-to-tokens ratio would be an approximation
        with a real measurement available.
        """
        echoed = "alpha beta gamma"
        seen: list[str] = []

        def count(text: str) -> int:
            seen.append(text)
            return RATIO * len(text.split())

        item = ifeval_item(
            instruction_ids=("combination:repeat_prompt",),
            kwargs=({"prompt_to_repeat": echoed},),
        )
        generative.ifeval_token_budget(
            item, ceiling=NO_CEILING, tokens_per_word=RATIO, count_tokens=count
        )

        assert echoed in seen, "the echoed text was tokenized, not converted"

    def test_the_echo_falls_back_to_the_question_when_the_kwarg_is_absent(self) -> None:
        assert budget(
            ("combination:repeat_prompt", {}), question="alpha beta gamma"
        ) == FLOOR + 3 * RATIO

    def test_two_responses_doubles_the_whole_of_the_rest(self) -> None:
        """Including the echo, since both responses have to carry it."""
        repeat = ("combination:repeat_prompt", {"prompt_to_repeat": "one two three"})
        pair = ("combination:two_responses", {})

        assert budget(pair) == 2 * FLOOR
        assert budget(repeat, pair) == 2 * budget(repeat)

    def test_a_non_english_item_bypasses_ratio_conversion_entirely(self) -> None:
        """The ratio is an English number and there is no target-language text to measure.

        A run-time tokenizer does not rescue this: the ratio would have to be measured on
        text in the item's target language, and the prompt is English asking for a Marathi
        response, so tokenizing it measures the wrong language. Measured across seven
        tokenizers on target-language samples, a non-English response costs 2.0x
        (Portuguese) to 22.3x (Tamil) per word, and Thai 158x because it is unspaced.

        So these items take the ceiling rather than a converted budget -- and take it
        whatever else they declare, which is the part that matters, since a converted
        paragraph count would land 2 to 22 times too small.
        """
        other = ("language:response_language", {"language": "ta"})

        assert budget(other, ceiling=CEILING) == CEILING
        assert budget(words(100), other, ceiling=CEILING) == CEILING
        assert budget(
            ("length_constraints:number_paragraphs", {"num_paragraphs": 3}),
            other,
            ceiling=CEILING,
        ) == CEILING

    def test_the_ratio_is_never_consulted_for_a_non_english_item(self) -> None:
        """Stronger than the budget being right: the English ratio must not be used at all.

        The Marathi item ``914b4ebdd73c5cc1`` is the concrete hazard -- 3 paragraphs, which
        converts to 450 English words. A rate this test can detect is one the implementation
        should never reach for.
        """
        marathi = ifeval_item(
            instruction_ids=("language:response_language", "length_constraints:number_paragraphs"),
            kwargs=({"language": "mr"}, {"num_paragraphs": 3}),
        )

        def forbidden(text: str) -> int:
            pytest.fail("a non-English item must not be tokenized at an English rate")

        assert generative.ifeval_token_budget(
            marathi, ceiling=CEILING, tokens_per_word=RATIO, count_tokens=forbidden
        ) == CEILING

    def test_a_malformed_count_falls_through_instead_of_budgeting_nothing(self) -> None:
        """A zero budget would generate nothing and grade the item on an empty response."""
        for bad in (0, -5, None, "300", True):
            assert budget(("length_constraints:number_words", {"num_words": bad})) == FLOOR

    def test_a_short_kwargs_list_does_not_raise(self) -> None:
        """A malformed bank is IFEvalPromptStrict's to mark, not budgeting's to crash on."""
        item = ifeval_item(
            instruction_ids=("length_constraints:number_words", "punctuation:no_comma"),
            kwargs=({"relation": "at least", "num_words": 200},),
        )

        assert generative.ifeval_token_budget(
            item,
            ceiling=NO_CEILING,
            tokens_per_word=RATIO,
            count_tokens=counting_tokenizer(),
        ) == (200 * RATIO + generative.DETECTION_HEADROOM_TOKENS)


class TestTheTruncationFragilitySplit:
    """The no-signal group is split by whether truncation breaks the grade."""

    def test_an_ending_dependent_item_takes_the_ceiling(self) -> None:
        """The grader reads the end, so a cut tail fails a constraint at any budget.

        The same failure class as the ``leaderboard_math`` blank-line stop that deleted
        40.9% of answers, which was treated as a correctness bug rather than a tuning
        question.
        """
        for instruction_id in sorted(generative.ENDING_DEPENDENT_IDS):
            assert budget((instruction_id, {}), ceiling=CEILING) == CEILING, instruction_id

    def test_a_count_dependent_item_takes_the_ceiling(self) -> None:
        """Truncation lowers a count the grader is checking."""
        for instruction_id in sorted(generative.COUNT_DEPENDENT_IDS):
            assert budget((instruction_id, {}), ceiling=CEILING) == CEILING, instruction_id

    def test_a_tolerant_item_takes_the_floor(self) -> None:
        """The six instructions the floor actually governs, asserted one by one.

        Anything absent from the fragile set is being *asserted* truncation-tolerant, so the
        assertion is made explicitly rather than left as an absence.
        """
        for instruction_id in (
            "keywords:forbidden_words",
            "punctuation:no_comma",
            "change_case:english_lowercase",
            "change_case:english_capital",
            "detectable_format:constrained_response",
            "detectable_format:title",
        ):
            assert instruction_id not in generative.TRUNCATION_FRAGILE_IDS
            assert budget((instruction_id, {}), ceiling=CEILING) == FLOOR, instruction_id

    def test_moving_an_id_out_of_the_fragile_set_is_caught(self, monkeypatch) -> None:
        """The mutation the set exists to prevent, exercised rather than trusted.

        A future reader trimming this set for cost is the failure mode; dropping
        ``startend:end_checker`` from it must change a budget, and this is what notices.
        """
        assert budget(("startend:end_checker", {}), ceiling=CEILING) == CEILING

        monkeypatch.setattr(
            generative,
            "ENDING_DEPENDENT_IDS",
            generative.ENDING_DEPENDENT_IDS - {"startend:end_checker"},
        )
        monkeypatch.setattr(
            generative,
            "TRUNCATION_FRAGILE_IDS",
            generative.TRUNCATION_FRAGILE_IDS - {"startend:end_checker"},
        )

        assert budget(("startend:end_checker", {}), ceiling=CEILING) == FLOOR, (
            "with the id removed the item drops to the floor, which is the bug the set "
            "prevents -- so the set is load-bearing rather than decorative"
        )

    def test_an_ending_constraint_overrides_a_declared_word_count(self) -> None:
        """A declared minimum is not an upper bound on a compliant answer.

        6 items state "at least N words" *and* an ending constraint -- f0da3bf76abece71 is
        "at least 400 words" wrapped in quotes. Converting the minimum gives it a budget a
        compliant 600-word answer would overrun, losing the closing quote the grader wants.
        Deriving from a lower bound would be a regression against the flat cap.
        """
        quoted = ("startend:quotation", {})

        assert budget(words(400), quoted, ceiling=CEILING) == CEILING
        assert budget(words(400), ceiling=CEILING) < CEILING

    def test_a_count_constraint_rides_inside_a_declared_length(self) -> None:
        """Count-dependent ids do *not* override a declared length, and that is checked.

        Verified against the bank: the smallest such derived budgets belong to items like
        563b267f4a5a3451 (1 highlight in "less than 30 words") and 58c326abfd36afc1 (2
        keywords in "less than 50"), where the count is comfortably satisfiable inside the
        declared length. Overriding these too would spend the ceiling for nothing.
        """
        highlights = ("detectable_format:number_highlighted_sections", {"num_highlights": 1})

        assert budget(words(30, "less than"), highlights, ceiling=CEILING) < CEILING

    def test_the_fragile_set_is_the_union_of_the_two_kinds(self) -> None:
        assert generative.TRUNCATION_FRAGILE_IDS == (
            generative.ENDING_DEPENDENT_IDS | generative.COUNT_DEPENDENT_IDS
        )
        assert not (generative.ENDING_DEPENDENT_IDS & generative.COUNT_DEPENDENT_IDS)


class TestTheCeiling:
    """1280 bounds every budget, and the cascade only ever lowers."""

    def test_the_config_pins_the_upstream_value(self) -> None:
        """1280 is what olmo-eval and lm-eval generate IFEval at.

        A harness convention rather than a paper value -- arXiv 2311.07911 specifies no cap
        at all -- but it is the scale the 1,102-model harvest was produced on, and theta is
        only interpretable on the scale the difficulties came from.
        """
        assert _resolved_generation_config(DATASET).max_new_tokens == CEILING

    def test_a_demand_above_the_ceiling_is_capped(self) -> None:
        """7 items convert to more than 1280 and are truncated there deliberately.

        ``ab5ca590f2d20f37`` wants about 3,400 tokens. It is cut at exactly the point the
        calibration population was cut; letting the cascade raise it instead would score it
        on a scale no fitted model was measured on.
        """
        assert budget(words(5000), ceiling=CEILING) == CEILING
        assert budget(words(5000)) > CEILING, "the demand really does exceed it"

    def test_nothing_the_cascade_produces_exceeds_the_ceiling(self) -> None:
        """Including the branches that multiply and add."""
        pair = ("combination:two_responses", {})
        repeat = ("combination:repeat_prompt", {"prompt_to_repeat": "one " * 400})

        assert budget(words(900), pair, repeat, ceiling=CEILING) == CEILING
        for item in _vendored_items():
            assert _real_budget(item) <= CEILING


class TestTheRealBankBudgets:
    """The cascade against the committed 511 items, not against constructed ones."""

    def test_the_two_items_that_declare_a_word_count_twice_follow_the_upper_bound(self) -> None:
        """46 ``number_words`` constraints across 44 items, so two are declared twice.

        Both are a bracketing pair. The budget follows the *upper* bound, because a model
        has to be able to reach the upper limit for the verifier's judgement of it to mean
        anything; a budget at the lower bound would cut the response there and record a
        pass on "less than N" that the harness produced rather than the model.

        A dict keyed by instruction id would keep one of the pair and drop the other
        without anything to show it, which is why the implementation iterates pairs.
        """
        found = {}
        for item in _vendored_items():
            declared = [
                kw
                for name, kw in zip(
                    item.metadata["instruction_id_list"],
                    item.metadata["kwargs"],
                    strict=False,
                )
                if name == "length_constraints:number_words"
            ]
            if len(declared) > 1:
                found[item.item_id] = (declared, _real_budget(item))

        assert set(found) == {"af6a27b556603a01", "1e657beb0dd15c56"}
        for declared, resolved in found.values():
            upper = max(kw["num_words"] for kw in declared)
            lower = min(kw["num_words"] for kw in declared)
            expected = min(upper * RATIO + generative.DETECTION_HEADROOM_TOKENS, CEILING)

            assert resolved == expected
            assert resolved > lower * RATIO, "the lower bound does not set the budget"

    def test_a_900_word_item_can_hold_a_compliant_answer(self) -> None:
        """The largest word count in the bank, and the case the units bug would break.

        ``9baf1cb36bafc2bb`` also carries ``startend:end_checker``, so it needs to reach 900
        words *and* still emit its closing phrase -- which is why it takes the ceiling rather
        than a converted budget.
        """
        resolved = _real_budget(self._item("9baf1cb36bafc2bb"))

        assert resolved == CEILING
        assert resolved > 900, "cannot be the raw word count"

    def test_a_20_word_item_gets_room_to_visibly_overshoot(self) -> None:
        """``b0d5c4ed20a557c6`` is "less than 20 words" plus a title."""
        resolved = _real_budget(self._item("b0d5c4ed20a557c6"))

        assert resolved == 20 * RATIO + generative.DETECTION_HEADROOM_TOKENS
        assert resolved < FLOOR, "a bounded item is the one case that may be shrunk"

    def test_the_largest_item_in_the_bank_is_capped_not_expanded(self) -> None:
        """``ab5ca590f2d20f37`` asks for 4 sections of at least 100 sentences.

        Its demand is far above the ceiling and it is truncated there deliberately, at the
        same point the calibration population was truncated. Asserted because an earlier
        design let the cascade raise it above the harness value, which put it on a scale no
        fitted model was measured on.
        """
        item = self._item("ab5ca590f2d20f37")

        assert _real_budget(item) == CEILING
        assert generative.ifeval_token_budget(
            item,
            ceiling=NO_CEILING,
            tokens_per_word=RATIO,
            count_tokens=counting_tokenizer(),
        ) > CEILING, "the demand really does exceed the ceiling"

    def test_every_item_gets_a_usable_budget(self) -> None:
        """No item may be budgeted at zero, and none above the ceiling."""
        for item in _vendored_items():
            resolved = _real_budget(item)
            assert isinstance(resolved, int)
            assert 0 < resolved <= CEILING, item.item_id

    def test_the_bank_is_cheaper_than_the_flat_cap_it_replaces(self) -> None:
        """The saving is the point of the floor, so it is asserted rather than assumed.

        Deliberately a modest claim. Half this bank is a correctness case that keeps the
        ceiling, so the honest headline is that the median budget is *at* the ceiling and the
        saving comes from the other half.
        """
        items = _vendored_items()
        total = sum(_real_budget(item) for item in items)

        assert total < 1536 * len(items), "cheaper than the flat 1536 it replaces"
        assert total < CEILING * len(items), "and cheaper than a flat ceiling"

    def test_the_branch_partition_is_the_one_the_cascade_was_built_for(self) -> None:
        """If the bank is re-vendored into a different shape, this is what notices.

        Measured 2026-08-08 over the committed 511. The ordering matters and mirrors the
        implementation's: the non-English exemption and the ending override are both taken
        before any count is read.
        """
        length_signals = {
            "length_constraints:number_words",
            "length_constraints:number_sentences",
            "length_constraints:number_paragraphs",
            "length_constraints:nth_paragraph_first_word",
            "detectable_format:number_bullet_lists",
            "detectable_format:multiple_sections",
        }
        counts: dict[str, int] = {}
        for item in _vendored_items():
            ids = set(item.metadata["instruction_id_list"])
            if "language:response_language" in ids:
                branch = "non_english"
            elif ids & generative.ENDING_DEPENDENT_IDS:
                branch = "ending"
            elif ids & length_signals:
                branch = "derived"
            elif ids & {"combination:repeat_prompt", "combination:two_responses"}:
                branch = "echo_only"
            elif ids & generative.COUNT_DEPENDENT_IDS:
                branch = "counting"
            else:
                branch = "tolerant"
            counts[branch] = counts.get(branch, 0) + 1

        assert counts == {
            "derived": 133,
            "ending": 106,
            "counting": 105,
            "tolerant": 84,
            "echo_only": 52,
            "non_english": 31,
        }
        assert sum(counts.values()) == 511

    def test_the_non_english_items_are_the_ones_measured(self) -> None:
        """31 items, 5 of which declare a length signal that would otherwise be converted."""
        signals = {
            "length_constraints:number_paragraphs",
            "combination:repeat_prompt",
            "combination:two_responses",
        }
        non_english = [
            item
            for item in _vendored_items()
            if "language:response_language" in item.metadata["instruction_id_list"]
        ]
        with_signal = {
            item.item_id
            for item in non_english
            if set(item.metadata["instruction_id_list"]) & signals
        }

        assert len(non_english) == 31
        assert with_signal == {
            "914b4ebdd73c5cc1",  # Marathi, 3 paragraphs -- the concrete hazard
            "79f8d92443e51c9e",  # Hindi, repeat_prompt
            "aa5496cf74dc2695",  # Vietnamese, two_responses
            "6ce2fcd3fa8e2bd0",  # Bulgarian, two_responses
            "b232f5603f3f2f50",  # Tamil, two_responses
        }
        for item in non_english:
            assert _real_budget(item) == CEILING

    @staticmethod
    def _item(item_id: str) -> BenchmarkItem:
        for item in _vendored_items():
            if item.item_id == item_id:
                return item
        pytest.fail(f"{item_id} is not in the committed bank")


def _real_budget(item: BenchmarkItem) -> int:
    """A committed item's budget at the ceiling and the fake tokenizer's fixed rate."""
    return generative.ifeval_token_budget(
        item,
        ceiling=CEILING,
        tokens_per_word=RATIO,
        count_tokens=counting_tokenizer(),
    )


class _BudgetedCompleter:
    """A completer publishing everything :class:`generative.GenerativeScorer` reads.

    The four attributes are the whole of the extended protocol -- a budget-aware call, a
    token counter, an end-token text and a context window -- so a test can drop any one of
    them to exercise the corresponding degraded path.
    """

    accepts_token_budget = True

    def __init__(
        self,
        *,
        count=None,
        context: int | None = None,
        eos: str | None = None,
        tokenizer: str | None = "fake/tokenizer",
    ) -> None:
        self.seen: list[int] = []
        self.prompts: list[str] = []
        if count is not None:
            self.count_tokens = count
        elif count is not False:
            self.count_tokens = counting_tokenizer()
        if context is not None:
            self.max_context_tokens = context
        if eos is not None:
            self.eos_text = eos
        if tokenizer is not None:
            self.tokenizer_id = tokenizer

    def __call__(self, prompt: str, max_new_tokens: int | None = None) -> str:
        self.prompts.append(prompt)
        if max_new_tokens is not None:
            self.seen.append(max_new_tokens)
        return FOLLOWED


def _ifeval_config(**overrides) -> generative.GenerationConfig:
    return generative.GenerationConfig(
        num_fewshot=0,
        prompt_style="ifeval",
        stop_sequences=(),
        max_new_tokens=overrides.pop("max_new_tokens", CEILING),
        **overrides,
    )


class TestTheBudgetReachesTheModel:
    """A budget nothing passes down is a comment, so the seam is pinned too."""

    def test_a_budget_aware_completer_is_called_with_the_per_item_budget(
        self, stub_ifbench
    ) -> None:
        completer = _BudgetedCompleter()
        items = [
            ifeval_item(("length_constraints:number_words",), ({"num_words": 300},)),
            ifeval_item(("punctuation:no_comma",), ({},)),
            ifeval_item(("startend:end_checker",), ({"end_phrase": "the end"},)),
        ]

        generative.GenerativeScorer(completer, _ifeval_config()).score_items(items)

        assert completer.seen == [
            300 * RATIO + generative.DETECTION_HEADROOM_TOKENS,
            FLOOR,
            CEILING,
        ]

    def test_without_a_tokenizer_the_cascade_does_not_run_at_all(self, stub_ifbench) -> None:
        """Every item takes the flat ceiling, and none takes a converted budget.

        This is the mode ``generative.require_live_tokenizer`` refuses in a real run. It stays
        reachable here on purpose: the refusal lives at the point a checkpoint is loaded, so a
        scorer built directly can still exercise the degraded path.
        """
        completer = _BudgetedCompleter(count=False)
        items = [
            ifeval_item(("length_constraints:number_words",), ({"num_words": 300},)),
            ifeval_item(("punctuation:no_comma",), ({},)),
        ]

        scorer = generative.GenerativeScorer(completer, _ifeval_config())
        scorer.score_items(items)

        assert scorer.cascade_active is False
        assert completer.seen == [CEILING, CEILING]

    def test_the_item_token_budget_dispatcher_needs_both_halves(self) -> None:
        """Either missing piece disables the cascade rather than half-running it."""
        item = ifeval_item(("length_constraints:number_words",), ({"num_words": 300},))
        config = _ifeval_config()

        assert generative.item_token_budget(item, config) == CEILING
        assert generative.item_token_budget(item, config, tokens_per_word=RATIO) == CEILING
        assert generative.item_token_budget(
            item, config, count_tokens=counting_tokenizer()
        ) == CEILING
        assert generative.item_token_budget(
            item, config, tokens_per_word=RATIO, count_tokens=counting_tokenizer()
        ) == 300 * RATIO + generative.DETECTION_HEADROOM_TOKENS

    def test_a_one_argument_completer_is_still_called_with_one_argument(
        self, stub_ifbench
    ) -> None:
        """The extension is opt-in because one-argument completers are the norm here.

        Every completer in this tree other than ``_HFCompleter`` is a one-argument
        callable, so always passing a second would be a ``TypeError`` raised from inside
        the scorer rather than a widened interface.
        """
        seen: list[str] = []

        def complete(prompt: str) -> str:
            seen.append(prompt)
            return FOLLOWED

        config = generative.GenerationConfig(
            num_fewshot=0, prompt_style="ifeval", stop_sequences=()
        )

        responses = generative.GenerativeScorer(complete, config).score_items([ifeval_item()])

        assert len(seen) == 1
        assert responses[0].correct is True

    def test_the_hf_completer_declares_itself_budget_aware(self) -> None:
        assert getattr(generative._HFCompleter, generative.BUDGET_AWARE_ATTR) is True

    def test_another_bank_keeps_its_flat_budget(self) -> None:
        """The cascade must not reach a bank whose items carry no constraints.

        A MATH item has no ``instruction_id_list``, and handing it the unconstrained floor
        would cut its longest derivations off at 512 against a derived 2048.
        """
        config = generative.GenerationConfig(max_new_tokens=2048)
        math_item = BenchmarkItem(
            item_id="m",
            question="Problem",
            choices=(),
            gold_index=-1,
            metadata={"answer_type": "math_latex", "modality": "generative"},
        )

        assert generative.item_token_budget(math_item, config) == 2048
        assert generative.item_token_budget(
            math_item, config, tokens_per_word=RATIO, count_tokens=counting_tokenizer()
        ) == 2048, "even with a tokenizer available, a non-ifeval bank is untouched"


class TestTheContextClamp:
    """The last stage: no budget may exceed what the context window leaves."""

    def test_it_is_inert_on_a_roomy_window(self) -> None:
        """Honest about what this is: insurance, not a fix for a live bug.

        IFEval prompts run 42/75/356 tokens median/p90/max, so against a 2048-token window
        the longest still leaves 1692 -- clear of the 1280 ceiling. Nothing in this bank
        trips the clamp on any model with a context of 1636 or more.
        """
        assert generative.fit_budget_to_context(CEILING, 51, 4096) == CEILING
        assert generative.fit_budget_to_context(CEILING, 51, 2048) == CEILING
        assert generative.fit_budget_to_context(CEILING, 356, 2048) == CEILING

    def test_a_window_smaller_than_the_budget_does_not_collapse_to_one(self) -> None:
        """The bug Research recorded, which this formula exists to avoid.

        Their earlier version computed ``max(1, max_model_len - max_new_tokens)``, which
        collapsed to 1 whenever the window was smaller than the nominal budget -- every model
        they ran at 4096 or below -- and then left-truncated the prompt to its final token. So
        the clamp is a function of the measured prompt length and never of the nominal budget.
        """
        for window in (512, 1024, 1279):
            resolved = generative.fit_budget_to_context(CEILING, 51, window)

            assert resolved is not None
            assert resolved == window - 51
            assert resolved > 1, f"collapsed at window={window}"
            assert resolved != max(1, window - CEILING), "not the old formula"

    def test_a_small_demand_is_not_refused_for_room_it_never_wanted(self) -> None:
        """``MIN_GENERATION_TOKENS`` is a reserve *capped by* the budget, not applied over it.

        A "less than 20 words" item wants 76 tokens. Reserving a flat 256 for it would refuse
        it on a window with 129 tokens free -- room for its whole compliant answer twice over
        -- turning the floor meant to prevent collapse into a source of refusals.
        """
        assert generative.MIN_GENERATION_TOKENS > 129, "the case below is the one that bites"
        assert generative.fit_budget_to_context(76, 51, 180) == 76

    def test_a_prompt_that_leaves_too_little_is_refused_rather_than_truncated(self) -> None:
        """Left-truncating would delete the instruction the verifiers grade against.

        Research keeps the tail of an over-long prompt, which is right for a chat transcript
        where the front is stale history. Here the instruction *is* the prompt, and the
        verifiers read ``metadata['kwargs']`` rather than the prompt -- so cutting its front
        does not soften what is checked, it just grades a model on a constraint it was never
        shown.
        """
        assert generative.fit_budget_to_context(CEILING, 500, 512) is None
        assert generative.fit_budget_to_context(CEILING, 600, 512) is None
        assert generative.fit_budget_to_context(76, 300, 320) is None

    def test_the_clamp_runs_after_the_ceiling(self, stub_ifbench) -> None:
        """Both only ever lower, so the smaller wins -- which is the ordering question."""
        completer = _BudgetedCompleter(context=600)
        item = ifeval_item(("startend:end_checker",), ({"end_phrase": "done"},))

        generative.GenerativeScorer(completer, _ifeval_config()).score_items([item])

        prompt_tokens = RATIO * len(completer.prompts[0].split())
        assert completer.seen == [600 - prompt_tokens]
        assert completer.seen[0] < CEILING

    def test_a_refused_item_is_recorded_ungradable_and_never_sent(
        self, stub_ifbench
    ) -> None:
        """Scored 0 and marked, so the fabricated zero is visible in the report."""
        completer = _BudgetedCompleter(context=8)
        item = ifeval_item(("punctuation:no_comma",), ({},))

        responses = generative.GenerativeScorer(
            completer, _ifeval_config()
        ).score_items([item])

        assert completer.seen == [], "the item was not sent to the model"
        assert responses[0].correct is False
        assert responses[0].metadata[generative.UNGRADABLE_KEY] is True
        assert "context window" in responses[0].metadata[generative.UNGRADABLE_REASON_KEY]

    def test_a_refused_item_is_not_graded_on_an_empty_response(self, stub_ifbench) -> None:
        """Several verifiers *pass* on empty input, which would invent a partial score.

        ``keywords:forbidden_words`` and ``punctuation:no_comma`` are both satisfied by a
        response with no text in it, so grading a non-attempt would credit the model for one.
        """
        completer = _BudgetedCompleter(context=8)
        item = ifeval_item(("keywords:forbidden_words",), ({"forbidden_words": ["x"]},))

        responses = generative.GenerativeScorer(
            completer, _ifeval_config()
        ).score_items([item])

        assert responses[0].correct is False

    def test_no_published_window_means_no_clamp(self, stub_ifbench) -> None:
        """A wrong window is worse than no window, so an unresolvable one is skipped."""
        completer = _BudgetedCompleter()
        item = ifeval_item(("punctuation:no_comma",), ({},))

        generative.GenerativeScorer(completer, _ifeval_config()).score_items([item])

        assert completer.seen == [FLOOR]


class TestTheLoadTimeTokenizerGuard:
    """A real run without a tokenizer is refused, not quietly degraded."""

    def test_a_completer_with_no_token_counter_is_refused(self) -> None:
        """Inference is impossible without a tokenizer, so a missing one is a fault.

        Continuing would hand every item the flat ceiling while the report looked entirely
        normal -- a plausible theta with a healthy standard error and nothing recording that
        the budgets were defaulted rather than computed.
        """
        scorer = generative.GenerativeScorer(
            _BudgetedCompleter(count=False), _ifeval_config()
        )

        with pytest.raises(RuntimeError, match="no tokenizer could be resolved"):
            generative.require_live_tokenizer(scorer, Path("/ckpt"))

    def test_a_token_counter_that_raises_is_the_same_fault(self) -> None:
        """Probed rather than merely present, since a broken one is the likelier defect."""

        def broken(text: str) -> int:
            raise OSError("tokenizer.json could not be fetched")

        scorer = generative.GenerativeScorer(
            _BudgetedCompleter(count=broken), _ifeval_config()
        )

        with pytest.raises(RuntimeError, match="tokenizer.json could not be fetched"):
            generative.require_live_tokenizer(scorer, Path("/ckpt"))

    def test_a_working_completer_passes(self) -> None:
        scorer = generative.GenerativeScorer(_BudgetedCompleter(), _ifeval_config())

        generative.require_live_tokenizer(scorer, Path("/ckpt"))

        assert scorer.cascade_active is True

    def test_the_guard_is_at_the_loader_and_not_in_scoring(self, stub_ifbench) -> None:
        """Which is what keeps the degraded path testable without an ``is_testing`` flag.

        A scorer built directly -- every prompt-building and grading test in this tree -- must
        still run with a one-argument completer and no tokenizer at all.
        """
        scorer = generative.GenerativeScorer(lambda _prompt: FOLLOWED, _ifeval_config())

        responses = scorer.score_items([ifeval_item()])

        assert scorer.cascade_active is False
        assert responses[0].correct is True

    def test_the_loader_applies_it(self, monkeypatch) -> None:
        """Registered through ``GENERATIVE_BACKENDS``, so the guard cannot be bypassed."""
        monkeypatch.setitem(
            generative.GENERATIVE_BACKENDS,
            "fake",
            lambda checkpoint_dir, config: generative.GenerativeScorer(
                _BudgetedCompleter(count=False), config
            ),
        )
        config = _ifeval_config(checkpoint_kind="fake")

        with pytest.raises(RuntimeError, match="no tokenizer could be resolved"):
            generative.load_generative_model(Path("/ckpt"), config)


class TestTheBudgetIsRecordedInTheReport:
    """Whether budgets were computed or defaulted has to be readable from the report."""

    def test_each_response_carries_its_own_budget_provenance(self, stub_ifbench) -> None:
        completer = _BudgetedCompleter(context=4096)
        items = [
            ifeval_item(("length_constraints:number_words",), ({"num_words": 300},)),
            ifeval_item(("punctuation:no_comma",), ({},)),
        ]

        responses = generative.GenerativeScorer(
            completer, _ifeval_config()
        ).score_items(items)

        derived = responses[0].metadata[generative.BUDGET_KEY]
        assert derived["tokens"] == 300 * RATIO + generative.DETECTION_HEADROOM_TOKENS
        assert derived["source"] == generative.BUDGET_FROM_CASCADE
        assert derived["cascade_active"] is True
        assert derived["context_window"] == 4096
        assert derived["tokenizer"] == "fake/tokenizer"

    def test_a_defaulted_budget_says_so(self, stub_ifbench) -> None:
        """The whole point: a degraded run must not look like a computed one."""
        completer = _BudgetedCompleter(count=False)

        responses = generative.GenerativeScorer(
            completer, _ifeval_config()
        ).score_items([ifeval_item(("length_constraints:number_words",), ({"num_words": 300},))])

        record = responses[0].metadata[generative.BUDGET_KEY]
        assert record["source"] == generative.BUDGET_FROM_CEILING
        assert record["cascade_active"] is False
        assert record["tokens"] == CEILING

    def test_a_clamped_budget_says_so(self, stub_ifbench) -> None:
        completer = _BudgetedCompleter(context=600)

        responses = generative.GenerativeScorer(
            completer, _ifeval_config()
        ).score_items([ifeval_item(("startend:end_checker",), ({"end_phrase": "done"},))])

        assert responses[0].metadata[generative.BUDGET_KEY]["source"] == (
            generative.BUDGET_FROM_CONTEXT
        )

    def test_an_offline_grade_carries_no_budget_at_all(self) -> None:
        """Omitted rather than null, so the key's presence means a decision was made."""
        response = generative.grade_completion(
            ifeval_item(), FOLLOWED, _ifeval_config()
        )

        assert generative.BUDGET_KEY not in response.metadata

    def test_the_report_block_summarises_the_mode(self, stub_ifbench) -> None:
        """Aggregated off the responses, the way ``_ungradable_block`` counts off them."""
        completer = _BudgetedCompleter(context=4096)
        items = [
            ifeval_item(("length_constraints:number_words",), ({"num_words": 300},)),
            ifeval_item(("punctuation:no_comma",), ({},)),
        ]
        responses = generative.GenerativeScorer(
            completer, _ifeval_config()
        ).score_items(items)

        state = CATState(benchmark=DATASET)
        state.administered.extend(responses)
        block = UniMcqStyle()._generation_budget_block(state)

        assert block["cascade_active"] is True
        assert block["tokenizer"] == "fake/tokenizer"
        assert block["context_window"] == 4096
        assert block["context_clamp_fired"] is False
        assert block["sources"] == {generative.BUDGET_FROM_CASCADE: 2}
        assert block["ceiling"] == CEILING
        assert block["tokens"]["min"] == FLOOR
        assert block["tokens"]["max"] == 300 * RATIO + generative.DETECTION_HEADROOM_TOKENS
        assert block["tokens"]["at_ceiling"] == 0

    def test_the_source_is_provenance_rather_than_value(self, stub_ifbench) -> None:
        """A fragile item left at the ceiling *on purpose* is still a computed budget.

        Labelling it flat would make it indistinguishable from the degraded mode, which is the
        one distinction the block exists to draw. How many items ended at the ceiling is
        reported separately.
        """
        completer = _BudgetedCompleter()
        responses = generative.GenerativeScorer(
            completer, _ifeval_config()
        ).score_items([ifeval_item(("startend:quotation",), ({},))])

        state = CATState(benchmark=DATASET)
        state.administered.extend(responses)
        block = UniMcqStyle()._generation_budget_block(state)

        assert block["sources"] == {generative.BUDGET_FROM_CASCADE: 1}
        assert block["tokens"]["at_ceiling"] == 1
        assert block["cascade_active"] is True

    def test_the_report_block_flags_a_clamp_that_actually_fired(self, stub_ifbench) -> None:
        """A window being *available* says nothing; one having bound is worth knowing."""
        completer = _BudgetedCompleter(context=600)
        responses = generative.GenerativeScorer(
            completer, _ifeval_config()
        ).score_items([ifeval_item(("startend:end_checker",), ({"end_phrase": "done"},))])

        state = CATState(benchmark=DATASET)
        state.administered.extend(responses)

        assert UniMcqStyle()._generation_budget_block(state)["context_clamp_fired"] is True

    def test_an_mcq_session_gets_no_block(self) -> None:
        """``None``, and then omitted, rather than an empty block on a bank with no budget."""
        state = CATState(benchmark="arc_challenge")

        assert UniMcqStyle()._generation_budget_block(state) is None


EOS = "<|endoftext|>"


class TestTheLeakedEndTokenStop:
    """The second stopping mechanism: keeping a leaked end marker out of the graded span."""

    @staticmethod
    def _stop(item: BenchmarkItem, eos: str | None = EOS) -> tuple[str, ...]:
        config = generative.GenerationConfig(
            num_fewshot=0, prompt_style="ifeval", stop_sequences=()
        )
        return generative.eos_stop_sequences(item, config, eos)

    def test_the_end_token_is_appended_at_runtime(self) -> None:
        assert self._stop(ifeval_item()) == (EOS,)

    def test_the_configured_list_stays_empty(self) -> None:
        """The committed config and the manifest must not name a checkpoint's token.

        A stop string is a benchmark property only when the benchmark names the string.
        This one is the model's, so recording it would make the bank's convention
        checkpoint-specific and refuse every model that spells its end token otherwise.
        """
        assert _resolved_generation_config(DATASET).stop_sequences == ()

        path = CALIBRATED_DATASETS / DATASET / "manifest.json"
        if path.is_file():
            recorded = json.loads(path.read_text(encoding="utf-8"))
            assert recorded["scoring_convention"]["runtime"]["stop_sequences"] == []

    def test_no_end_token_leaves_the_list_empty(self) -> None:
        """Then the per-item cap is the only bound, which is today's actual state."""
        assert self._stop(ifeval_item(), None) == ()
        assert self._stop(ifeval_item(), "") == ()

    def test_a_leaked_literal_is_cut_out_of_the_graded_span(self) -> None:
        """It survives ``skip_special_tokens`` because it is not the special token."""
        completion = f"A tidy response.{EOS}\nUser: another question entirely"

        assert generative.truncate_at_stop(
            completion, self._stop(ifeval_item())
        ) == "A tidy response."

    def test_it_rescues_an_end_anchored_constraint(self, stub_ifbench) -> None:
        """26 end_checker plus 40 quotation plus 27 parsed-span items turn on this.

        ``EndChecker`` does ``value.strip().strip('"').lower().endswith(phrase)``, so a
        trailing marker fails a response that ended exactly as asked.
        """
        item = ifeval_item(
            ("startend:end_checker",), ({"end_phrase": FOLLOWED},), question="End as told."
        )
        config = generative.GenerationConfig(
            num_fewshot=0, prompt_style="ifeval", stop_sequences=()
        )
        leaked = f"Body of the answer. {FOLLOWED}{EOS} Assistant: hello again"

        assert generative.grade_completion(item, leaked, config, EOS).metadata[
            "completion"
        ].strip().endswith(FOLLOWED)
        assert not generative.grade_completion(item, leaked, config, None).metadata[
            "completion"
        ].strip().endswith(FOLLOWED)

    def test_it_is_suppressed_for_two_responses(self) -> None:
        """Cutting at the first leak would delete the second response.

        ``TwoResponsesChecker`` splits on ``******`` and requires exactly two non-empty
        parts, so a checkpoint leaking its marker between them would fail the very
        constraint being checked -- a harness-manufactured failure.
        """
        item = ifeval_item((generative.TWO_RESPONSES_ID,), ({},))

        assert self._stop(item) == ()

    def test_the_second_response_survives_because_of_that(self, stub_ifbench) -> None:
        """The interaction end to end, on the real verifier's own splitting rule."""
        item = ifeval_item(
            (generative.TWO_RESPONSES_ID,), ({},), question="Give two responses."
        )
        config = generative.GenerationConfig(
            num_fewshot=0, prompt_style="ifeval", stop_sequences=()
        )
        leaked = f"First response {FOLLOWED}{EOS}\n******\nSecond response {FOLLOWED}"

        graded = generative.grade_completion(item, leaked, config, EOS)

        assert "Second response" in graded.metadata["completion"]
        assert graded.metadata["completion"].count("******") == 1

    def test_suppressing_it_there_gives_up_nothing(self) -> None:
        """0 of the 24 two-responses items carry an end-anchored constraint.

        Which is what makes the exception free rather than a trade. If a re-vendor ever
        produces an item with both, this fails and the rule needs rethinking.
        """
        anchored = {
            "startend:end_checker",
            "startend:quotation",
            "detectable_format:json_format",
            "detectable_format:constrained_response",
        }
        clashing = [
            item.item_id
            for item in _vendored_items()
            if generative.TWO_RESPONSES_ID in item.metadata["instruction_id_list"]
            and anchored & set(item.metadata["instruction_id_list"])
        ]

        assert clashing == []

    def test_a_bank_that_declares_its_own_stops_keeps_them(self) -> None:
        """The end token is appended to the configured list, never a replacement for it."""
        item = ifeval_item()
        config = generative.GenerationConfig(stop_sequences=("Question:", "\n\n"))

        assert generative.eos_stop_sequences(item, config, EOS) == (
            "Question:",
            "\n\n",
            EOS,
        )

    def test_an_already_configured_end_token_is_not_duplicated(self) -> None:
        config = generative.GenerationConfig(stop_sequences=(EOS,))

        assert generative.eos_stop_sequences(ifeval_item(), config, EOS) == (EOS,)

    def test_the_scorer_reads_the_end_token_off_the_completer(self, stub_ifbench) -> None:
        def complete(prompt: str) -> str:
            return f"{FOLLOWED}{EOS} trailing junk"

        complete.eos_text = EOS
        config = generative.GenerationConfig(
            num_fewshot=0, prompt_style="ifeval", stop_sequences=()
        )

        scorer = generative.GenerativeScorer(complete, config)
        graded = scorer.score_items([ifeval_item()])[0]

        assert scorer.eos_text == EOS
        assert graded.metadata["completion"] == FOLLOWED

    def test_a_completer_that_publishes_nothing_changes_nothing(self, stub_ifbench) -> None:
        config = generative.GenerationConfig(
            num_fewshot=0, prompt_style="ifeval", stop_sequences=()
        )
        scorer = generative.GenerativeScorer(lambda _: f"{FOLLOWED}{EOS}", config)

        assert scorer.eos_text is None
        assert scorer.score_items([ifeval_item()])[0].metadata["completion"] == (
            f"{FOLLOWED}{EOS}"
        )

    def test_the_hf_completer_publishes_its_tokenizers_end_token(self) -> None:
        """And reports ``None`` for a checkpoint that defines none, which is today's case.

        ``hf_config_patch._llama_config`` writes ``eos_token_id=None``, so this is the
        live state rather than a hypothetical.
        """
        completer = generative._HFCompleter.__new__(generative._HFCompleter)
        completer.tokenizer = types.SimpleNamespace(eos_token=EOS)
        assert completer.eos_text == EOS

        completer.tokenizer = types.SimpleNamespace(eos_token=None)
        assert completer.eos_text is None

    def test_another_bank_is_left_alone_by_the_suppression_rule(self) -> None:
        """A MATH item has no instruction list, so it simply gains the end token."""
        math_item = BenchmarkItem(
            item_id="m",
            question="Problem",
            choices=(),
            gold_index=-1,
            metadata={"answer_type": "math_latex", "modality": "generative"},
        )
        config = generative.GenerationConfig(stop_sequences=("Problem:",))

        assert generative.eos_stop_sequences(math_item, config, EOS) == ("Problem:", EOS)


class TestTheBudgetPolicyIsRecorded:
    """The manifest guard has to stay meaningful once the budget is not one integer."""

    def test_the_manifest_records_the_policy_constants(self) -> None:
        path = CALIBRATED_DATASETS / DATASET / "manifest.json"
        if not path.is_file():
            pytest.skip("ifeval has not been vendored")
        recorded = json.loads(path.read_text(encoding="utf-8"))["scoring_convention"]["runtime"]

        assert recorded["max_new_tokens"] == CEILING, "the ceiling, still one integer"
        policy = recorded["per_item_token_budget"]
        assert policy["policy"] == "ifeval_length_signal_cascade"
        assert policy["detection_headroom_tokens"] == generative.DETECTION_HEADROOM_TOKENS
        assert policy["words_per_sentence"] == generative.WORDS_PER_SENTENCE
        assert policy["words_per_paragraph"] == generative.WORDS_PER_PARAGRAPH
        assert policy["words_per_bullet"] == generative.WORDS_PER_BULLET
        assert policy["unconstrained_floor_tokens"] == generative.UNCONSTRAINED_FLOOR_TOKENS
        assert policy["truncation_fragile_ids"] == sorted(generative.TRUNCATION_FRAGILE_IDS)

    def test_the_runtime_ratio_is_recorded_as_unpinnable(self) -> None:
        """It is a model property, so pinning it would re-stamp the bank per checkpoint.

        Recorded as prose rather than omitted, so its absence reads as a decision instead of
        an oversight -- the same treatment the run-time EOS gets.
        """
        path = CALIBRATED_DATASETS / DATASET / "manifest.json"
        if not path.is_file():
            pytest.skip("ifeval has not been vendored")
        recorded = json.loads(path.read_text(encoding="utf-8"))["scoring_convention"]["runtime"]

        assert recorded["per_item_token_budget"]["tokens_per_word"] == (
            "resolved at runtime from the checkpoint tokenizer"
        )
        assert recorded["stop_sequences"] == [], "the run-time EOS is not recorded either"

    def test_changing_a_constant_would_trip_the_guard(self, monkeypatch) -> None:
        """Which is the whole point of recording them rather than only the ceiling."""
        from .. import convention

        before = convention.configured_convention(
            SUPPORTED[DATASET], convention.load_config()
        )
        monkeypatch.setattr(generative, "UNCONSTRAINED_FLOOR_TOKENS", 1)
        after = convention.configured_convention(
            SUPPORTED[DATASET], convention.load_config()
        )

        assert before != after
        assert convention._differences(before, after)

    def test_trimming_the_fragile_set_would_trip_the_guard(self, monkeypatch) -> None:
        """The set is a budget policy, so quietly shrinking it must not go unnoticed."""
        from .. import convention

        before = convention.configured_convention(
            SUPPORTED[DATASET], convention.load_config()
        )
        monkeypatch.setattr(
            generative,
            "TRUNCATION_FRAGILE_IDS",
            generative.TRUNCATION_FRAGILE_IDS - {"startend:end_checker"},
        )
        after = convention.configured_convention(
            SUPPORTED[DATASET], convention.load_config()
        )

        assert convention._differences(before, after)

    def test_no_other_bank_gains_the_key(self) -> None:
        """Emitting it everywhere would invalidate eight manifests for no policy."""
        from .. import convention

        config = convention.load_config()
        for name, spec in SUPPORTED.items():
            if spec.modality != grading.GENERATIVE or name == DATASET:
                continue
            recorded = convention.configured_convention(spec, config)
            assert convention.PER_ITEM_BUDGET_KEY not in recorded, name


class FakeChatTokenizer:
    """A tokenizer that renders a chat turn the way an instruction-tuned one would."""

    pad_token_id = None
    eos_token_id = 7
    chat_template = "{{ messages }}"

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = True, add_generation_prompt: bool = False
    ) -> str:
        self.last = (messages, tokenize, add_generation_prompt)
        return f"<|user|>{messages[0]['content']}<|assistant|>"

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        from .test_generative_grading import FakeIds

        return {"input_ids": FakeIds(text.split(" "))}

    def decode(self, tokens: list[str], skip_special_tokens: bool = False) -> str:
        return " ".join(tokens)


class TestChatFormatting:
    """The one part of the pipeline that is the checkpoint's rather than the bank's.

    Exercised through ``gpqa``'s settings rather than ``ifeval``'s, which is a change of
    example and not of subject. IFEval is scored in completion format now -- the
    leaderboard it was harvested from templated its chat submissions and not its
    pretrained ones, so the completion half is the one a base checkpoint can be measured
    against -- which leaves gpqa as the only chat-format bank here and therefore the
    only honest way to drive this path. The two tests at the bottom hold the other side:
    that ifeval really does bypass the template now, and that gpqa really does still
    refuse a checkpoint without one.
    """

    @pytest.fixture
    def fake_stack(self, monkeypatch):
        import contextlib

        from .test_generative_grading import FakeModel, FakeTokenizer

        loaded: list[str] = []

        def install(tokenizer: object) -> FakeModel:
            model = FakeModel()
            torch = types.ModuleType("torch")
            torch.no_grad = contextlib.nullcontext
            transformers = types.ModuleType("transformers")
            transformers.AutoTokenizer = types.SimpleNamespace(
                from_pretrained=lambda *a, **k: tokenizer
            )

            def load_model(*args: object, **kwargs: object) -> FakeModel:
                loaded.append("weights")
                return model

            transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=load_model)
            transformers.set_seed = lambda seed: None
            monkeypatch.setitem(sys.modules, "torch", torch)
            monkeypatch.setitem(sys.modules, "transformers", transformers)
            return model

        install.plain = FakeTokenizer
        install.loaded = loaded
        return install

    @staticmethod
    def chat_config() -> generative.GenerationConfig:
        """The shape gpqa was scored in until its modality changed.

        Constructed rather than resolved from ``config.yaml``, because no bank selects
        it any more and the guard still has to hold for the next one that would.
        """
        return generative.GenerationConfig(
            num_fewshot=0,
            prompt_style="gpqa",
            chat_format=True,
            system_prompt_source="gpqa",
            stop_sequences=(),
        )

    def test_the_prompt_goes_through_the_checkpoints_template(self, fake_stack) -> None:
        tokenizer = FakeChatTokenizer()
        fake_stack(tokenizer)
        config = generative.GenerationConfig(
            num_fewshot=0, prompt_style="gpqa", chat_format=True, stop_sequences=()
        )

        generative._HFCompleter(Path("/ckpt"), config)("Write a summary.")

        messages, tokenize, add_generation_prompt = tokenizer.last
        assert messages == [{"role": "user", "content": "Write a summary."}]
        assert tokenize is False
        assert add_generation_prompt is True

    def test_a_checkpoint_without_a_chat_template_is_refused(self, fake_stack) -> None:
        """Sending the prompt raw would still complete, still grade, and still report."""
        fake_stack(fake_stack.plain())

        with pytest.raises(ValueError, match="defines no chat template"):
            generative._HFCompleter(Path("/ckpt"), self.chat_config())

    def test_it_is_refused_before_the_weights_are_loaded(self, fake_stack) -> None:
        """The tokenizer is cheap and the model is not; the check goes between them."""
        fake_stack(fake_stack.plain())

        with pytest.raises(ValueError, match="defines no chat template"):
            generative._HFCompleter(Path("/ckpt"), self.chat_config())
        assert fake_stack.loaded == []

    def test_a_completion_bank_never_touches_the_template(self, fake_stack) -> None:
        """gsm8k and MATH prompts carry their own framing and must arrive unwrapped."""
        tokenizer = FakeChatTokenizer()
        fake_stack(tokenizer)
        tokenizer.last = None
        config = generative.GenerationConfig(num_fewshot=0, chat_format=False)

        generative._HFCompleter(Path("/ckpt"), config)("Question: how many?\nAnswer:")

        assert tokenizer.last is None

    def test_a_checkpoint_with_no_template_can_now_be_scored_on_ifeval(self, fake_stack) -> None:
        """The whole point of the flip, driven through the real completer.

        A tokenizer carrying no ``chat_template`` is what a base checkpoint has --
        SmolLM2-135M among them -- and while ifeval was chat-format this raised before
        the weights were reached, which put every generative bank here out of reach of a
        base model. Built from the committed ``config.yaml`` rather than a constructed
        config, so it fails if the setting is flipped back rather than passing on a
        hand-written copy of what the file used to say.
        """
        fake_stack(fake_stack.plain())
        config = _resolved_generation_config(DATASET)

        generative._HFCompleter(Path("/ckpt"), config)("Write a summary with no commas.")

        assert config.chat_format is False
        assert fake_stack.loaded == ["weights"]

    def test_the_chat_shape_is_still_refused_for_anything_that_asks_for_it(
        self, fake_stack
    ) -> None:
        """The guard outlived both banks that needed it, and has to keep working.

        GPQA was this test's subject until 2026-08-08: it kept ``chat_format`` after
        ifeval gave it up, so a base checkpoint was refused here and the bank was
        unreachable. That was resolved by changing its modality rather than its framing
        -- lm-eval scores GPQA as a log-likelihood ranking, which needs no template --
        so no bank sets ``chat_format`` today. The refusal still matters, because the
        thing it prevents is silent: sending the prompt raw would complete, grade and
        report, with the standing instruction simply missing.
        """
        fake_stack(fake_stack.plain())

        assert not any(
            grading._apply_generation_overrides(
                grading.GradingSettings(), UniMcqStyle().generation_settings, dataset=name
            ).generation.chat_format
            for name, spec in SUPPORTED.items()
            if spec.modality == grading.GENERATIVE
        )
        with pytest.raises(ValueError, match="defines no chat template"):
            generative._HFCompleter(Path("/ckpt"), self.chat_config())


class TestVendoredBank:
    """The committed bank, asserted against what the join was supposed to produce."""

    @pytest.fixture
    def manifest(self) -> dict:
        path = CALIBRATED_DATASETS / DATASET / "manifest.json"
        if not path.is_file():
            pytest.skip("ifeval has not been vendored")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_content_hashed_bridge_was_used(self, manifest) -> None:
        """Neither of the two wrong joins this bank had available is what was read.

        The bank once joined through the bare integer in
        ``ATLAS/ifeval/atlas_idx_to_question_id.csv``, and that integer is a position in
        an enumeration performed at harvest time rather than anything the prompt says.
        Its two available substitutes are worse -- the composite ``item_id_map`` joins
        nothing, and ``scenarios.jsonl`` is a dense 0..540 run over the same prompts in a
        different order, so it joins near-totally against mostly wrong questions.
        """
        assert manifest["bridge_kind"] == "content_hash"
        assert manifest["bridge_path"].endswith("bridges/ifeval.csv")
        assert manifest["bridge_in_repo"] is True

    def test_the_counts_are_the_ones_upstream_reports(self, manifest) -> None:
        assert manifest["upstream_bank_rows"] == 535
        assert manifest["bridge_rows"] == 535
        assert manifest["dropped"]["non_positive_discrimination"] == 24
        assert manifest["dropped"]["not_in_task"] == 0
        assert manifest["items"] == 511

    def test_item_ids_are_content_hashes(self, manifest) -> None:
        """A bare integer here would mean the superseded positional bridge was read.

        Checked as 16 hex digits rather than as "not a number", because roughly one
        content hash in 1,800 happens to be all decimal digits and a bank of 511 would
        fail that on about a quarter of re-vendors.
        """
        items = _vendored_items()
        assert all(re.fullmatch(r"[0-9a-f]{16}", item.item_id) for item in items)
        assert len({item.item_id for item in items}) == len(items)

    def test_every_item_carries_its_constraints_and_no_gold(self, manifest) -> None:
        for item in _vendored_items():
            assert item.metadata["answer_type"] == "ifeval_strict"
            assert item.metadata["instruction_id_list"]
            assert len(item.metadata["kwargs"]) == len(item.metadata["instruction_id_list"])
            assert "gold_answer" not in item.metadata


def _vendored_items() -> list[BenchmarkItem]:
    """The committed IFEval items, loaded through the real loader."""
    from ....common.benchmark_download import load_items_from_jsonl

    path = CALIBRATED_DATASETS / DATASET / "items.jsonl"
    if not path.is_file():
        pytest.skip("ifeval has not been vendored")
    return list(load_items_from_jsonl(path, name=DATASET).items)


def run_real_bank(true_theta: float, *, max_items: int = 40) -> dict:
    """A full CAT over the committed 511-item bank, tokens simulated and nothing else."""
    style = UniMcqStyle()
    bank = style.download_benchmark(DATASET)
    irt = style.load_irt_params(DATASET)
    config = _resolved_generation_config(DATASET)
    taker = SimGenerativeTaker(
        true_theta,
        vendored_params(DATASET),
        bank.items,
        recover_question=lambda prompt: prompt,
        answer=ifeval_answer,
    )
    report = cat_loop.run_cat(
        style,
        bank=bank,
        irt_bank=irt,
        model=generative.GenerativeScorer(taker, config),
        se_threshold=0.3,
        max_items=max_items,
    )
    return report.to_dict()


class TestAbilityRecoveryOnTheRealBank:
    """Theta recovery over the committed bank, through the real engine and grader."""

    run = staticmethod(run_real_bank)

    def test_a_session_converges_and_stops_on_precision(self, stub_ifbench) -> None:
        """Well inside the 40-item cap, though faster here than upstream's 27.

        A taker drawn from the bank's own 3PL fits it perfectly and collapses the
        posterior at the ``min_items`` floor; the cross-validated 27 is over real
        models, whose responses misfit. What this pins is that the cap is not what
        ends the session.
        """
        report = self.run(0.5)

        assert report["metadata"]["stop_reason"] == "precision_reached"
        assert report["metadata"]["standard_error"] <= 0.3
        assert report["metadata"]["bank_size"] == 511
        assert np.isfinite(report["ability"]["theta"])

    @pytest.mark.parametrize("true_theta", [-1.5, 0.0, 1.5])
    def test_the_estimate_lands_near_the_truth(self, stub_ifbench, true_theta: float) -> None:
        report = self.run(true_theta)
        assert report["metadata"]["theta"] == pytest.approx(true_theta, abs=0.6)

    def test_theta_recovers_monotonically(self, stub_ifbench) -> None:
        """A stronger simulated taker must produce a strictly higher estimate.

        The property that matters for a diagnostic. The absolute scale is the bank's
        and can shift with the grader; the ordering is what a comparison between two
        checkpoints actually rests on. The ladder stays inside the bank's range, which
        for IFEval reaches down to about -1.5 -- far lower than MATH's, which is what a
        median difficulty of 1.03 against MATH's 2.78 means in practice.
        """
        estimates = [self.run(theta)["metadata"]["theta"] for theta in (-1.5, -0.5, 0.5, 1.5)]

        assert all(b > a for a, b in pairwise(estimates)), estimates

    def test_the_report_names_the_grader_that_produced_it(self, stub_ifbench) -> None:
        report = self.run(0.5)
        note = report["metadata"]["scoring_note"]

        assert "prompt_level_strict_acc" in note
        assert "last number" not in note
        assert report["metadata"]["modality"] == "generative"

    def test_every_response_records_the_constraints_it_was_judged_on(self, stub_ifbench) -> None:
        for response in self.run(0.5)["responses"]:
            assert response["chosen_index"] == -1
            assert response["metadata"]["grader"] == "ifeval_prompt_strict"
            assert response["metadata"]["grader_detail"]["strict"]


class TestASessionSurvivesUngradableItems:
    """One malformed item must cost that item, not the whole checkpoint's diagnostic.

    Driven through :func:`run_real_bank` on purpose: the point is that the *same* run,
    over the same committed bank, still reaches a written report when grading stops
    producing outcomes. Before this the first such item raised out through the CAT
    engine into ``runner.main``, which logged and returned 1 -- after the checkpoint had
    been staged and the GPU booted, and with nothing written down about how far the
    session got.
    """

    run = staticmethod(run_real_bank)

    @pytest.fixture
    def silent_scorer(self, monkeypatch, stub_ifbench):
        """An ``IFEvalScorer`` that verifies nothing, as a missing registry would."""
        from olmo_eval.common.types import Instance, LMOutput

        class VerifiesNothing:
            def score(self, instance: object, output: object) -> float:
                output.metadata["ifeval"] = {"strict": [], "loose": []}
                return 0.0

        monkeypatch.setattr(
            generative, "_ifeval_scoring", lambda: (VerifiesNothing, Instance, LMOutput)
        )

    def test_the_report_is_still_written(self, silent_scorer) -> None:
        report = self.run(0.5)

        assert report["metadata"]["n_items_administered"] > 0
        assert np.isfinite(report["ability"]["theta"])

    def test_every_zero_is_marked_as_not_the_models(self, silent_scorer) -> None:
        """Otherwise the pattern is indistinguishable from a checkpoint answering wrong."""
        responses = self.run(0.5)["responses"]

        assert responses
        assert all(r["correct"] is False for r in responses)
        assert all(r["metadata"][generative.UNGRADABLE_KEY] is True for r in responses)
        assert all(r["metadata"][generative.UNGRADABLE_REASON_KEY] for r in responses)

    def test_the_report_says_the_theta_is_not_about_the_checkpoint(self, silent_scorer) -> None:
        """A floor theta with a healthy standard error is what this run would look like."""
        metadata = self.run(0.5)["metadata"]
        block = metadata["ungradable"]

        assert block["count"] == metadata["n_items_administered"]
        assert block["rate"] == 1.0
        assert "harness rather than the checkpoint" in block["alert"]

    def test_a_broken_dependency_is_still_an_error_rather_than_a_floor(
        self, monkeypatch, stub_ifbench
    ) -> None:
        """The other side of the line: no verifiers at all is not 511 wrong answers.

        An ungradable item is one the bank failed to describe. A grader that cannot run
        is a different claim -- every item would score 0, the report would carry a
        confident floor theta, and the ungradable count would be the only thing saying
        so. That is too quiet for a fault whose fix is installing a package, so the
        import failure keeps propagating.
        """
        monkeypatch.delitem(sys.modules, "ifbench")
        monkeypatch.setattr(
            generative,
            "_ifeval_scoring",
            lambda: (_ for _ in ()).throw(RuntimeError("ifbench is not importable")),
        )

        with pytest.raises(RuntimeError, match="ifbench"):
            self.run(0.5)


#: A response used to drive every verifier in the bank at once: no commas, several
#: sections, and long enough that a minimum-word constraint has a chance. It is not
#: expected to satisfy most items -- IFEval prompts ask for specific things -- only to be
#: something every verifier can reach a real verdict about.
NEUTRAL_RESPONSE = (
    "Section 1\n\nHere is a plain answer written without any commas.\n\n"
    "Section 2\n\nIt continues for a while so that a length constraint has a chance of "
    "passing and it keeps going with more words to be safe about minimum word counts."
)


class TestAgainstTheRealVerifiers:
    """Skipped without ``ifbench``. Nothing here can be simulated and still mean anything.

    Two dependencies, not one, and the second is easy to miss now that the first is
    installed. Every test here runs olmo-eval's real ``IFEvalScorer`` over the real
    registry, so a bare diagnostics-only checkout has to skip on ``olmo_eval`` as well --
    otherwise installing ``ifbench`` turns what used to be a clean skip into a wall of
    failures about a package these tests never claimed to need.
    """

    @pytest.fixture(autouse=True)
    def _needs_olmo_eval(self) -> None:
        pytest.importorskip(
            "olmo_eval.common.scorers",
            reason="IFEval grading runs olmo-eval's real IFEvalScorer; run with PYTHONPATH=src",
        )

    @pytest.fixture
    def real_ifbench(self):
        return pytest.importorskip(
            "ifbench",
            reason="ifbench (a git-URL dependency) is not installed in this environment",
        )

    @pytest.fixture
    def real_ifbench_over_the_bank(self, real_ifbench):
        """``ifbench`` plus the NLTK corpora its verifiers load on first use.

        A second prerequisite that the package alone does not satisfy, and one worth
        skipping on rather than failing over. Several IFEval verifiers -- the ones that
        count sentences or capitalised words -- tokenize with NLTK and fetch ``punkt_tab``
        and ``averaged_perceptron_tagger_eng`` the first time they run. The fetch is
        silent when it works and raises a ``LookupError`` mid-grade when it does not,
        which on a GPU run is an aborted session rather than a missing package. Probing
        one such item here turns that into a named skip with the fix in it.
        """
        item = ifeval_item(
            ("change_case:capital_word_frequency",),
            ({"capital_relation": "less than", "capital_frequency": 2},),
        )
        try:
            generative.get_answer_grader("ifeval_strict").grade_item("A short answer.", item)
        except LookupError as exc:
            pytest.skip(
                f"an ifbench verifier needs NLTK data this machine does not have and "
                f"could not download ({str(exc).splitlines()[1].strip()}); run "
                f"python -c \"import nltk; nltk.download('punkt_tab'); "
                f"nltk.download('averaged_perceptron_tagger_eng')\""
            )
        return real_ifbench

    def test_a_comma_free_response_passes_and_a_comma_fails(self, real_ifbench) -> None:
        grader = generative.get_answer_grader("ifeval_strict")
        item = ifeval_item(("punctuation:no_comma",), ({},))

        assert grader.grade_item("No commas anywhere in this sentence.", item).correct is True
        assert grader.grade_item("There is, unmistakably, a comma.", item).correct is False

    def test_strict_and_loose_disagree_where_the_metric_choice_matters(self, real_ifbench) -> None:
        """Markdown stars are stripped by the loose variants and not by strict scoring.

        The bank was calibrated on strict, so this grader has to be the stricter of the
        two; if it ever started reading the loose list every item would look easier
        than its difficulty.
        """
        grader = generative.get_answer_grader("ifeval_strict")
        item = ifeval_item(("punctuation:no_comma",), ({},))

        assert grader.grade_item("*A starred, comma-bearing line.*", item).correct is False

    def test_the_registry_covers_every_instruction_the_bank_names(self, real_ifbench) -> None:
        """A missing verifier would raise mid-session on whichever item selected it."""
        from ifbench import instructions_registry

        named = {
            instruction
            for item in _vendored_items()
            for instruction in item.metadata["instruction_id_list"]
        }
        assert named <= set(instructions_registry.INSTRUCTION_DICT)

    def test_every_item_in_the_bank_reaches_a_real_verdict(self, real_ifbench_over_the_bank):
        """The registry resolving is not the same claim as every verifier running.

        A verifier can be present and still fail on the arguments this bank hands it --
        a missing corpus, an argument name the vendored kwargs spell differently -- and
        the failure surfaces only on the item that selects it, mid-session. Driving all
        511 items through the real grader turns that from a run-time surprise into a
        property of the committed bank.
        """
        grader = generative.get_answer_grader("ifeval_strict")
        verdicts = [grader.grade_item(NEUTRAL_RESPONSE, item) for item in _vendored_items()]

        assert len(verdicts) == 511
        assert all(verdict.ungradable_reason is None for verdict in verdicts)
        for verdict, item in zip(verdicts, _vendored_items(), strict=True):
            assert len(verdict.detail["strict"]) == len(item.metadata["instruction_id_list"])

    def test_the_verifiers_discriminate_rather_than_refusing_everything(
        self, real_ifbench_over_the_bank
    ) -> None:
        """A registry that rejected every response would look exactly like a weak model.

        One canned response passes some items and fails most, which is what real
        verification looks like; a bank where the same response scored 0 everywhere
        would produce a floor theta with a healthy standard error and no other sign.
        """
        grader = generative.get_answer_grader("ifeval_strict")
        outcomes = [grader.grade_item(NEUTRAL_RESPONSE, item).correct for item in _vendored_items()]

        assert any(outcomes)
        assert not all(outcomes)

    def test_a_simulated_session_runs_on_the_real_verifiers(
        self, real_ifbench_over_the_bank
    ) -> None:
        """The whole harness end to end with nothing stubbed but the tokens.

        Everywhere else in this file the verifier bodies are replaced, because a stub is
        what lets the grader, the CAT and the report be exercised on a machine without
        ``ifbench``. This is the run that says the stub was standing in for something
        that works: the same committed bank, the same engine, and the real IFBench
        registry deciding every item.

        It asserts mechanics rather than recovery. The simulated taker's responses are
        not built to satisfy any particular constraint, so which items come back correct
        is not a statement about the bank -- what matters is that every administered item
        produced one real verdict per instruction it names, and that no zero in the
        response pattern was fabricated by the harness.
        """
        report = run_real_bank(0.5)

        assert report["metadata"]["n_items_administered"] >= UniMcqStyle().min_items
        assert report["metadata"]["ungradable"]["count"] == 0
        assert np.isfinite(report["ability"]["theta"])
        for response in report["responses"]:
            detail = response["metadata"]["grader_detail"]
            assert response["metadata"]["grader"] == "ifeval_prompt_strict"
            assert len(detail["strict"]) == len(detail["instruction_id_list"])

    def test_that_session_did_not_reach_the_stub_registry(self, real_ifbench_over_the_bank) -> None:
        """The claim the test above rests on: nothing replaced ``ifbench`` in sys.modules.

        The stub is installed with ``monkeypatch.setitem(sys.modules, ...)``, so a
        fixture ordering mistake would leave it in place and the session would pass on
        the marker text instead of on verified constraints -- which is precisely the
        difference this class exists to establish.
        """
        assert sys.modules["ifbench"].__spec__ is not None
        assert "site-packages" in (sys.modules["ifbench"].__file__ or "")
        assert FOLLOWED not in NEUTRAL_RESPONSE


class TestTheScoringNote:
    """A report's note has to describe the grader that produced its theta, not a modality."""

    def test_each_grader_supplies_its_own_clause(self) -> None:
        for name in ("last_number_exact_match", "math_latex_equivalence", "ifeval_prompt_strict"):
            assert generative.grader_note(name) != f"graded by {name}"

    def test_every_registered_grader_has_one(self) -> None:
        """Otherwise a new bank's reports would describe it only by its internal name."""
        for grader in generative.ANSWER_GRADERS.values():
            assert grader.name in generative.GRADER_NOTES

    def test_the_mcq_note_names_its_own_ranking_rule(self) -> None:
        """The two modalities fill their template from different places; neither leaks."""
        note = style_mod._scoring_note(
            grading.MCQ, [], score_normalization=inference.DEFAULT_SCORE_NORMALIZATION
        )
        assert "not length-normalized" in note
        assert "{scoring}" not in note
        assert "{grading}" not in note

    def test_a_session_that_administered_nothing_says_so_rather_than_naming_a_grader(
        self,
    ) -> None:
        """No response means no evidence of which grader ran, and none is invented."""
        note = style_mod._scoring_note(grading.GENERATIVE, [])
        assert style_mod.NO_GRADER_NOTE in note
        assert "{grading}" not in note


class TestWithoutOlmoEval:
    def test_the_grader_says_where_the_verifiers_live(self, monkeypatch) -> None:
        """A diagnostics-only checkout must get a message, not an ImportError."""
        monkeypatch.setitem(sys.modules, "olmo_eval.common.scorers", None)
        with pytest.raises(RuntimeError, match="IFEvalScorer"):
            generative.get_answer_grader("ifeval_strict").grade_item("x", ifeval_item())
