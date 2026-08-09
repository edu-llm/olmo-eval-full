"""Symbolic grading for the MATH bank, and the GSM8K numeric path it must not disturb.

Two graders now share one generative pipeline, and the failure this file guards against
is not a crash. ``vendor_bank.answer_type`` reads a gold answer's type off the string,
so a MATH gold of ``7`` looks numeric and a gold of ``\\frac{1}{2}`` looks like text; a
bank graded that way would run half its items through last-number exact match and raise
on the other half. Both halves have to reach :class:`MathLatexEquivalence` instead, and
GSM8K has to keep reaching :class:`LastNumberExactMatch` unchanged, because the two
scales are only comparable to their own calibrations.

Anything asserting real symbolic equivalence needs ``olmo_eval``, whose extractor and
``is_equiv`` this grader deliberately reuses rather than reimplements, and skips when it
is not importable. Everything else -- table lookups, prompt layout, the composition of
extraction and comparison, the missing-answer verdict -- runs against a stubbed
``olmo_eval`` in ``sys.modules`` and needs neither the package nor the network.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from itertools import pairwise
from pathlib import Path

import pytest

from ....base import BenchmarkItem
from ....common import cat_loop, generative, grading
from ....common.benchmark_download import load_items_from_jsonl
from ..style import UniMcqStyle
from .conftest import CALIBRATED_DATASETS, SimGenerativeTaker, vendored_params
from .test_generative_grading import STUB_FEWSHOT

#: A stand-in for the Minerva block. The real four examples live in olmo_eval; these
#: pin the prompt's shape, not its content, and carry the ``solution`` key the
#: leaderboard_math template reads its worked answers from.
MATH_STUB_FEWSHOT = (
    {
        "question": "Find the domain of $\\sqrt{x-2}$.",
        "solution": "We need $x \\ge 2$, so the domain is $\\boxed{[2,\\infty)}$.",
    },
    {
        "question": "If $\\det A = 2$ and $\\det B = 12$, find $\\det(AB)$.",
        "solution": "$\\det(AB) = (2)(12) = \\boxed{24}$.",
    },
)


def boxed(answer: str) -> str:
    """A one-line solution ending the way the Minerva few-shot block teaches."""
    return f" Working through it, the value is $\\boxed{{{answer}}}$."


def _boxed_content(text: str) -> str | None:
    """Return what the last ``\\boxed{}`` in ``text`` wraps, matching braces.

    Only the stubbed extractor needs this. Splitting on the first ``}`` instead would
    truncate every fraction, which is most of what a MATH answer is.
    """
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth = 0
    for index in range(start + len(marker) - 1, len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[start + len(marker) : index]
    return None


def math_item(gold: str = "\\frac{1}{2}") -> BenchmarkItem:
    """A generative item shaped as a vendored MATH bank record."""
    return BenchmarkItem(
        item_id="algebra_hard|0",
        question="What is one half?",
        choices=(),
        gold_index=-1,
        metadata={"gold_answer": gold, "answer_type": "math_latex"},
    )


@pytest.fixture
def real_math_extract():
    """olmo_eval's math extractor, or skip: there is no substitute for the real one."""
    return pytest.importorskip(
        "olmo_eval.evals.extract.math",
        reason="olmo_eval is not importable in this environment",
    )


@pytest.fixture
def stub_math_extract(monkeypatch):
    """Install a fake ``olmo_eval.evals.extract.math`` and return it.

    Lets the grader's composition be exercised without the package: what it extracts,
    how many candidates it tries, which one it reports, and what it does when nothing
    answer-shaped was produced.
    """
    fake = types.ModuleType("olmo_eval.evals.extract.math")
    fake.calls = []

    def extract_math_answer(text: str) -> list[str]:
        inner = _boxed_content(text)
        if inner is None:
            return [text.strip()]
        return [inner, f"{inner} (alternate)"]

    def is_equiv(gen: str, correct: str) -> bool:
        fake.calls.append((gen, correct))
        return gen == correct

    fake.extract_math_answer = extract_math_answer
    fake.is_equiv = is_equiv
    fake.last_boxed_only_string = lambda text: "\\boxed{}" if "\\boxed{" in text else None
    fake.get_unnormalized_answer = lambda text: (
        "found" if "Final Answer:" in text else "[invalidanswer]"
    )

    extract_pkg = types.ModuleType("olmo_eval.evals.extract")
    extract_pkg.math = fake
    for name, module in (
        ("olmo_eval", types.ModuleType("olmo_eval")),
        ("olmo_eval.evals", types.ModuleType("olmo_eval.evals")),
        ("olmo_eval.evals.extract", extract_pkg),
        ("olmo_eval.evals.extract.math", fake),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return fake


class TestTheGraderTable:
    def test_math_latex_resolves_to_the_symbolic_grader(self) -> None:
        grader = generative.get_answer_grader("math_latex")
        assert isinstance(grader, generative.MathLatexEquivalence)
        assert grader.name == "math_latex_equivalence"

    def test_numeric_still_resolves_to_last_number_exact_match(self) -> None:
        assert isinstance(generative.get_answer_grader("numeric"), generative.LastNumberExactMatch)

    def test_a_text_answer_type_still_raises_and_now_names_math_latex(self) -> None:
        """The one wiring gap left: a MATH bank must emit math_latex, not text."""
        with pytest.raises(ValueError, match="math_latex") as excinfo:
            generative.get_answer_grader("text")
        assert "No answer grader for answer_type 'text'" in str(excinfo.value)

    def test_every_grader_satisfies_exactly_one_of_the_two_protocols(self) -> None:
        """Nothing in the table may be neither shape, and nothing may be both.

        ``apply_grader`` dispatches on which one an entry is, so an entry answering to
        neither would be called with a signature it does not have, and one answering to
        both would take whichever branch is written first regardless of which of its
        two methods the benchmark actually needs.
        """
        for answer_type, grader in generative.ANSWER_GRADERS.items():
            shapes = [
                isinstance(grader, generative.AnswerGrader),
                isinstance(grader, generative.ItemGrader),
            ]
            assert sum(shapes) == 1, f"{answer_type} matches {sum(shapes)} grader protocols"


class TestSymbolicEquivalence:
    """Answers that are the same number written differently must score the same."""

    @pytest.mark.parametrize(
        "answer",
        [
            "\\frac{1}{2}",
            "0.5",
            "\\dfrac{1}{2}",
            "\\frac{1}{ 2}",
            "\\tfrac{1}{2}",
        ],
    )
    def test_equivalent_spellings_of_one_half_are_all_correct(
        self, real_math_extract, answer: str
    ) -> None:
        verdict = generative.MathLatexEquivalence().grade(boxed(answer), "\\frac{1}{2}")
        assert verdict.correct is True
        assert verdict.gold == "\\frac{1}{2}"

    def test_left_and_right_wrappers_do_not_change_the_answer(self, real_math_extract) -> None:
        """A gold carrying the delimiters must match a completion that omits them."""
        verdict = generative.MathLatexEquivalence().grade(
            boxed("(3,\\frac{\\pi}{2})"), "\\left(3,\\frac{\\pi}{2}\\right)"
        )
        assert verdict.correct is True

    @pytest.mark.parametrize("answer", ["\\frac{1}{3}", "2", "-\\frac{1}{2}", "x"])
    def test_different_answers_are_rejected(self, real_math_extract, answer: str) -> None:
        assert (
            generative.MathLatexEquivalence().grade(boxed(answer), "\\frac{1}{2}").correct is False
        )

    def test_the_minerva_final_answer_line_is_also_extracted(self, real_math_extract) -> None:
        """The few-shot block teaches both endings, so the grader honours both."""
        completion = " Adding them gives 24.\nFinal Answer: The final answer is $24$."
        assert generative.MathLatexEquivalence().grade(completion, "24").correct is True


class TestBoxedExtractionBeatsLastNumber:
    """The reason a MATH bank cannot borrow the GSM8K grader."""

    @pytest.mark.parametrize(
        ("answer", "trailing_digit"),
        [("x^2", "2"), ("\\frac{1}{2}", "2"), ("\\sqrt{5}", "5"), ("2\\pi", "2")],
    )
    def test_the_answer_is_read_whole_not_as_its_trailing_digit(
        self, real_math_extract, answer: str, trailing_digit: str
    ) -> None:
        completion = boxed(answer)
        assert generative.extract_last_number(completion) == trailing_digit

        verdict = generative.MathLatexEquivalence().grade(completion, answer)
        assert verdict.extracted == answer
        assert verdict.correct is True

    @pytest.mark.parametrize(
        ("gold", "wrong_answer"),
        [("x^2", "y^2"), ("\\frac{1}{2}", "\\frac{3}{2}"), ("\\sqrt{5}", "\\sqrt{7}5")],
    )
    def test_last_number_grading_would_pass_a_wrong_boxed_answer(
        self, real_math_extract, gold: str, wrong_answer: str
    ) -> None:
        """Not merely inaccurate on boxed answers: it scores them on the wrong string.

        ``LastNumberExactMatch`` reduces both sides to their last number, so a gold of
        ``x^2`` and an answer of ``y^2`` both become ``2`` and the item is marked
        correct. Every MATH item whose answer ends in a digit -- most of them -- would
        be scored on a coincidence, and the bank would look far easier than its
        calibrated difficulties say it is.
        """
        completion = boxed(wrong_answer)
        assert generative.LastNumberExactMatch().grade(completion, gold).correct is True
        assert generative.MathLatexEquivalence().grade(completion, gold).correct is False


class TestNumericGoldsStayOnTheSymbolicPath:
    """A MATH gold that parses as a float must not fall back to exact match."""

    def test_an_integer_gold_is_graded_symbolically(self, real_math_extract) -> None:
        assert generative.MathLatexEquivalence().grade(boxed("7"), "7").correct is True
        assert generative.MathLatexEquivalence().grade(boxed("8"), "7").correct is False

    def test_an_integer_gold_still_tolerates_reformatting(self, real_math_extract) -> None:
        """What exact match would lose: one bank, two grading conventions."""
        assert generative.MathLatexEquivalence().grade(boxed("1{,}000"), "1000").correct is True

    def test_the_item_declares_the_grader_not_the_answer_shape(self, stub_math_extract) -> None:
        graded = generative.grade_completion(
            math_item(gold="7"), boxed("7"), generative.GenerationConfig()
        )
        assert graded.metadata["grader"] == "math_latex_equivalence"


class TestACompletionWithNoBoxedAnswer:
    """Scored incorrect and warned about, never raised on."""

    def test_prose_with_no_answer_is_incorrect(self, stub_math_extract) -> None:
        verdict = generative.MathLatexEquivalence().grade(" I do not know how.", "24")
        assert verdict.correct is False

    def test_an_empty_completion_extracts_nothing(self, stub_math_extract) -> None:
        verdict = generative.MathLatexEquivalence().grade("", "24")
        assert verdict.correct is False
        assert verdict.extracted is None

    def test_it_warns_so_a_formatting_failure_is_not_read_as_low_ability(
        self, stub_math_extract, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="mcq_cat.generative"):
            generative.MathLatexEquivalence().grade(" I do not know how.", "24")
        assert "No \\boxed{} answer" in caplog.text

    def test_a_boxed_answer_warns_about_nothing(self, stub_math_extract, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="mcq_cat.generative"):
            generative.MathLatexEquivalence().grade(boxed("24"), "24")
        assert caplog.text == ""

    def test_the_cat_loop_sees_an_ordinary_incorrect_response(self, stub_math_extract) -> None:
        """Raising here would drop exactly the items the calibration counted wrong."""
        graded = generative.grade_completion(
            math_item(gold="24"), " I refuse to answer.", generative.GenerationConfig()
        )
        assert graded.correct is False
        assert graded.metadata["grader"] == "math_latex_equivalence"

    def test_has_final_answer_recognises_both_taught_endings(self, stub_math_extract) -> None:
        assert generative.has_final_answer(boxed("24")) is True
        assert generative.has_final_answer("Final Answer: The final answer is $24$.") is True
        assert generative.has_final_answer("It is about twenty four.") is False


class TestExtractionAndComparisonAreComposed:
    """Every candidate is tried against gold, as MinervaMathScorer does."""

    def test_a_later_candidate_can_carry_the_verdict(self, stub_math_extract) -> None:
        verdict = generative.MathLatexEquivalence().grade(boxed("24"), "24 (alternate)")
        assert verdict.correct is True
        assert stub_math_extract.calls == [
            ("24", "24 (alternate)"),
            ("24 (alternate)", "24 (alternate)"),
        ]

    def test_the_reported_answer_is_the_primary_candidate(self, stub_math_extract) -> None:
        assert generative.MathLatexEquivalence().grade(boxed("24"), "99").extracted == "24"

    def test_gold_is_compared_stripped_but_otherwise_untouched(self, stub_math_extract) -> None:
        """``clean_answer_text`` would reduce a LaTeX gold to its last digit."""
        verdict = generative.MathLatexEquivalence().grade(boxed("\\frac{1}{2}"), "  \\frac{1}{2} ")
        assert verdict.gold == "\\frac{1}{2}"
        assert verdict.correct is True

    def test_a_missing_olmo_eval_says_so_rather_than_guessing(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "olmo_eval.evals.extract", None)
        with pytest.raises(RuntimeError, match="no string-equality fallback"):
            generative.MathLatexEquivalence().grade(boxed("24"), "24")


class TestPromptTemplates:
    def test_gsm8k_is_the_default_style(self) -> None:
        assert generative.GenerationConfig().prompt_style == "gsm8k"

    def test_an_unknown_style_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="Unknown prompt_style"):
            generative.get_prompt_template("minerva")

    def test_math_asks_for_a_solution_not_an_answer(self, monkeypatch) -> None:
        monkeypatch.setitem(
            generative.FEWSHOT_SOURCES, "leaderboard_math", lambda: MATH_STUB_FEWSHOT
        )
        config = generative.GenerationConfig(
            num_fewshot=2, fewshot_source="leaderboard_math", prompt_style="leaderboard_math"
        )
        prompt = generative.format_generative_prompt(math_item(), config)

        assert prompt.startswith("Problem:\nFind the domain of $\\sqrt{x-2}$.\n\nSolution: We need")
        assert prompt.endswith("Problem:\nWhat is one half?\n\nSolution:")
        assert "Question:" not in prompt
        assert prompt.count("Problem:\n") == 3

    def test_a_latex_stem_is_substituted_not_reparsed(self, monkeypatch) -> None:
        """A braced question would be a format-spec error if the stem were a template."""
        monkeypatch.setitem(generative.FEWSHOT_SOURCES, "leaderboard_math", lambda: ())
        item = BenchmarkItem(
            item_id="algebra_hard|1",
            question="Evaluate $\\frac{a}{b}$ when $\\{a,b\\} = \\{1,2\\}$.",
            choices=(),
            gold_index=-1,
            metadata={"gold_answer": "\\frac{1}{2}", "answer_type": "math_latex"},
        )
        config = generative.GenerationConfig(num_fewshot=0, prompt_style="leaderboard_math")
        assert item.question in generative.format_generative_prompt(item, config)

    def test_mismatched_style_and_fewshot_source_are_refused(self, monkeypatch) -> None:
        """A gsm8k block behind the MATH template teaches an answer format nothing reads."""
        monkeypatch.setitem(generative.FEWSHOT_SOURCES, "gsm8k", lambda: STUB_FEWSHOT)
        config = generative.GenerationConfig(
            num_fewshot=1, fewshot_source="gsm8k", prompt_style="leaderboard_math"
        )
        with pytest.raises(ValueError, match="do not belong together"):
            generative.format_generative_prompt(math_item(), config)

    def test_the_math_fewshot_source_is_registered(self) -> None:
        assert "leaderboard_math" in generative.FEWSHOT_SOURCES

    def test_every_fewshot_source_has_a_template_of_the_same_name(self) -> None:
        """The two are selected by separate config keys, and a block needs a layout.

        Not the reverse: a template with no source of its own is a 0-shot benchmark,
        and it declares that by leaving fewshot_answer_key unset rather than by
        registering an empty block.
        """
        assert set(generative.FEWSHOT_SOURCES) <= set(generative.PROMPT_TEMPLATES)
        for name in set(generative.PROMPT_TEMPLATES) - set(generative.FEWSHOT_SOURCES):
            assert generative.PROMPT_TEMPLATES[name].fewshot_answer_key is None


class TestGsm8kIsUnchanged:
    """Byte-for-byte, against the layout the template table replaced."""

    @pytest.fixture(autouse=True)
    def _stub(self, monkeypatch):
        monkeypatch.setitem(generative.FEWSHOT_SOURCES, "gsm8k", lambda: STUB_FEWSHOT)

    def test_the_prompt_is_identical_to_the_hardcoded_layout(self) -> None:
        item = BenchmarkItem(
            item_id="g0",
            question="How many clips?",
            choices=(),
            gold_index=-1,
            metadata={"gold_answer": "72", "answer_type": "numeric"},
        )
        config = generative.GenerationConfig(num_fewshot=2)

        expected = "\n\n".join(
            [
                *(
                    f"Question: {example['question']}\nAnswer: {example['answer']}"
                    for example in STUB_FEWSHOT
                ),
                f"Question: {item.question}\nAnswer:",
            ]
        )
        assert generative.format_generative_prompt(item, config) == expected

    def test_zero_shot_is_still_the_bare_question(self) -> None:
        item = BenchmarkItem(item_id="g0", question="How many?", choices=(), gold_index=-1)
        config = generative.GenerationConfig(num_fewshot=0)
        assert generative.format_generative_prompt(item, config) == "Question: How many?\nAnswer:"

    def test_the_numeric_grader_still_takes_the_last_number(self) -> None:
        verdict = generative.LastNumberExactMatch().grade("So the answer is 1,234.", "1234")
        assert verdict.correct is True
        assert verdict.extracted == "1234"

    def test_a_numeric_item_grades_exactly_as_before(self) -> None:
        item = BenchmarkItem(
            item_id="g0",
            question="How many?",
            choices=(),
            gold_index=-1,
            metadata={"gold_answer": "72", "answer_type": "numeric"},
        )
        graded = generative.grade_completion(
            item, " ... So the answer is 73.", generative.GenerationConfig(num_fewshot=8)
        )
        assert graded.metadata == {
            "modality": "generative",
            "grader": "last_number_exact_match",
            "completion": " ... So the answer is 73.",
            "extracted_answer": "73",
            "gold_answer": "72",
            "num_fewshot": 8,
            "ungradable": False,
        }

    def test_the_sampling_defaults_are_untouched(self) -> None:
        config = generative.GenerationConfig()
        assert config.max_new_tokens == 512
        assert config.num_fewshot == 8
        assert config.fewshot_source == "gsm8k"
        assert config.stop_sequences == ("Question:", "\n\n")


DATASET = "leaderboard_math"


def _resolved_generation_config(dataset: str) -> generative.GenerationConfig:
    """The generation settings the committed config.yaml resolves to for ``dataset``."""
    settings = grading._apply_generation_overrides(
        grading.GradingSettings(), UniMcqStyle().generation_settings, dataset=dataset
    )
    return settings.generation


def _recover_math_question(prompt: str) -> str:
    """Pull the live problem back out of a rendered Minerva prompt.

    The last ``Problem:`` block is the live one; everything before it is the four-shot
    worked block, which the template joins with the same framing.
    """
    return prompt.rsplit("Problem:\n", 1)[-1].split("\n\nSolution:")[0]


def math_answer(item: BenchmarkItem, correct: bool) -> str:
    """A solution ending the two ways the Minerva block teaches, right or wrong.

    The wrong branch answers with a token that cannot be equivalent to any MATH gold
    under either half of ``is_equiv``, so a simulated failure is a genuine failure
    rather than a near miss the grader might forgive.
    """
    answer = item.metadata["gold_answer"] if correct else "\\text{nonsense}"
    return (
        f" Working through it, the value is $\\boxed{{{answer}}}$.\n"
        f"Final Answer: The final answer is ${answer}$. I hope it is correct."
    )


class TestVendoredMathBank:
    """The committed bank, asserted against what the join was supposed to produce."""

    @pytest.fixture
    def manifest(self) -> dict:
        path = CALIBRATED_DATASETS / DATASET / "manifest.json"
        if not path.is_file():
            pytest.skip("leaderboard_math has not been vendored")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_it_took_the_single_task_path_on_a_composite_key(self, manifest) -> None:
        """The task emits the composite id itself, so nothing is rebuilt by counting.

        That composite supplies the enumeration and no longer the join. The bank is
        keyed by the problem's own text, because the two orderings the composite
        conflated are unrelated: exactly 8 of 1,324 problems sit where it assumed.

        The multi-task path would abort here, and correctly: its per-subtask guard
        demands positions be a gapless 0..n-1 run, and calibration left these sparse.
        """
        assert manifest["bridge_kind"] == "content_hash"
        assert manifest["task"] == "leaderboard_math"

    def test_the_counts_are_the_ones_upstream_reports(self, manifest) -> None:
        assert manifest["upstream_bank_rows"] == 1206
        assert manifest["bridge_rows"] == 1206
        assert manifest["dropped"]["non_positive_discrimination"] == 23
        assert manifest["dropped"]["not_in_task"] == 0
        assert manifest["dropped"]["ambiguous_item_id"] == 0
        assert manifest["items"] == 1183

    def test_every_item_records_a_subject_spanning_all_seven_subtasks(self, manifest) -> None:
        """A content hash names only the item, so the subject rides on the parameters.

        Vendoring copies the bridge's subtask column onto each parameter record for
        that reason, the composite id having carried the attribution until now. Looking
        each item up by it is also what says the two artifacts are keyed alike; a subject
        short of seven is a partially vendored MATH-Hard, which nothing else would see.
        """
        subject_of = {row["item_id"]: row["metadata"]["subtask"] for row in _vendored_math_params()}
        subjects = {subject_of[item.item_id] for item in _vendored_math_items()}

        assert len(subjects) == 7
        assert "algebra_hard" in subjects

    def test_every_item_is_graded_symbolically_whatever_its_gold_looks_like(self, manifest) -> None:
        """The wiring this spec field exists for: a gold of ``7`` must not read numeric."""
        items = _vendored_math_items()
        assert all(item.metadata["answer_type"] == "math_latex" for item in items)
        assert any(_parses_as_float(item.metadata["gold_answer"]) for item in items)


def _parses_as_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _vendored_math_items() -> list[BenchmarkItem]:
    """The committed MATH items, loaded through the real loader."""
    path = CALIBRATED_DATASETS / DATASET / "items.jsonl"
    if not path.is_file():
        pytest.skip("leaderboard_math has not been vendored")
    return list(load_items_from_jsonl(path, name=DATASET).items)


def _vendored_math_params() -> list[dict]:
    """The committed MATH parameter records, which carry each item's subject."""
    path = CALIBRATED_DATASETS / DATASET / "params.json"
    if not path.is_file():
        pytest.skip("leaderboard_math has not been vendored")
    return json.loads(path.read_text(encoding="utf-8"))


class TestAbilityRecoveryOnTheRealMathBank:
    """A full CAT over the committed 1,183-item bank, tokens simulated and nothing else.

    These need ``olmo_eval``: the grader is its extractor and ``is_equiv``, and the
    four-shot block is its Minerva constant. Stubbing either would leave the test
    measuring the stub.
    """

    @pytest.fixture(autouse=True)
    def _needs_olmo_eval(self, real_math_extract) -> None:
        pytest.importorskip(
            "olmo_eval.evals.tasks.constants.minerva_math",
            reason="olmo_eval is not importable in this environment",
        )

    def run(self, true_theta: float, *, max_items: int = 40) -> dict:
        style = UniMcqStyle()
        bank = style.download_benchmark(DATASET)
        irt = style.load_irt_params(DATASET)
        taker = SimGenerativeTaker(
            true_theta,
            vendored_params(DATASET),
            bank.items,
            recover_question=_recover_math_question,
            answer=math_answer,
        )
        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt,
            model=generative.GenerativeScorer(taker, _resolved_generation_config(DATASET)),
            se_threshold=0.3,
            max_items=max_items,
        )
        return report.to_dict()

    def test_a_session_converges_inside_the_banks_range(self) -> None:
        """Fast here, and that speed is the simulation's rather than the bank's.

        A taker drawn from the bank's own 3PL fits it perfectly, so the posterior
        collapses at the ``min_items`` floor. Upstream's ten-fold cross-validation
        needs 96 items to the same SE target over real models, whose responses misfit;
        the gap between the two numbers is that misfit and nothing else.
        """
        report = self.run(1.0)

        assert report["metadata"]["bank_size"] == 1183
        assert report["metadata"]["stop_reason"] == "precision_reached"
        assert report["metadata"]["standard_error"] <= 0.3

    def test_a_taker_below_the_banks_floor_exhausts_the_cap_without_precision(self) -> None:
        """MATH-Hard measures nothing at theta -2, and the report has to say so.

        Its difficulties run from 0.85 at the 5th percentile to 2.78 at the median, so
        a weak taker answers every item wrong however many are administered and the
        posterior never tightens. The value to read off such a run is ``stop_reason``
        and the standard error, not the point estimate, which is the prior pulled
        slightly down rather than a measurement.
        """
        report = self.run(-2.0)

        assert report["metadata"]["stop_reason"] == "max_items_reached"
        assert report["num_items_administered"] == 40
        assert report["metadata"]["standard_error"] > 0.3

    @pytest.mark.parametrize("true_theta", [0.0, 1.0, 2.0])
    def test_the_estimate_lands_near_the_truth(self, true_theta: float) -> None:
        assert self.run(true_theta)["metadata"]["theta"] == pytest.approx(true_theta, abs=0.5)

    def test_theta_recovers_monotonically(self) -> None:
        """A stronger simulated taker must produce a strictly higher estimate.

        The ladder sits above zero because that is where this bank has items. Below
        roughly -1 every question is out of reach and two different weak takers return
        the same floored estimate, which is the bank's range rather than a failure of
        the CAT -- and is why the floor is pinned by its own test above.
        """
        estimates = [self.run(theta)["metadata"]["theta"] for theta in (-0.5, 0.5, 1.5, 2.5)]

        assert all(b > a for a, b in pairwise(estimates)), estimates

    def test_the_report_names_the_symbolic_grader(self) -> None:
        note = self.run(0.5)["metadata"]["scoring_note"]

        assert "symbolic equivalence" in note
        assert "last number" not in note

    def test_every_response_was_graded_symbolically(self) -> None:
        for response in self.run(0.5)["responses"]:
            assert response["metadata"]["grader"] == "math_latex_equivalence"
            assert response["chosen_index"] == -1


class TestPerDatasetGenerationSettings:
    """config.yaml holds one generative block; two benchmarks disagree about prompting."""

    @pytest.fixture
    def captured(self, monkeypatch) -> list[generative.GenerationConfig]:
        seen: list[generative.GenerationConfig] = []
        monkeypatch.setattr(
            generative,
            "load_generative_model",
            lambda _dir, config: (
                seen.append(config) or generative.GenerativeScorer(lambda _: "", config)
            ),
        )
        return seen

    def load(self, dataset: str, tmp_path: Path) -> None:
        """Build the grader for ``dataset`` from the committed config.yaml."""
        request = grading.GradingRequest(
            dataset=dataset,
            modality="generative",
            generation=UniMcqStyle().generation_settings,
        )
        grading.load_grader(request, tmp_path, grading.GradingSettings())

    def test_the_math_entry_overrides_the_shared_keys(self, captured, tmp_path: Path) -> None:
        self.load("leaderboard_math", tmp_path)
        config = captured[0]
        assert config.num_fewshot == 4
        assert config.max_new_tokens == 1024
        assert config.prompt_style == "leaderboard_math"
        assert config.fewshot_source == "leaderboard_math"
        assert config.stop_sequences == ("Problem:", "problem:")

    def test_a_dataset_with_no_entry_keeps_the_shared_keys(self, captured, tmp_path: Path) -> None:
        self.load("gsm8k", tmp_path)
        config = captured[0]
        assert config.num_fewshot == 8
        assert config.max_new_tokens == 512
        assert config.prompt_style == "gsm8k"
        assert config.stop_sequences == ("Question:", "\n\n")

    def test_the_per_dataset_map_is_not_mistaken_for_a_setting(self, tmp_path: Path) -> None:
        settings = grading._apply_generation_overrides(
            grading.GradingSettings(),
            {"num_fewshot": 2, grading.PER_DATASET_KEY: {"other": {"num_fewshot": 5}}},
            dataset="leaderboard_math",
        )
        assert settings.generation.num_fewshot == 2

    def test_a_misspelled_key_inside_a_dataset_entry_still_raises(self, tmp_path: Path) -> None:
        request = grading.GradingRequest(
            dataset="leaderboard_math",
            modality="generative",
            generation={grading.PER_DATASET_KEY: {"leaderboard_math": {"prompt_stile": "x"}}},
        )
        with pytest.raises(ValueError, match="Unknown generation settings"):
            grading.load_grader(request, tmp_path, grading.GradingSettings())

    def test_a_non_mapping_per_dataset_block_raises(self, tmp_path: Path) -> None:
        request = grading.GradingRequest(
            dataset="leaderboard_math",
            modality="generative",
            generation={grading.PER_DATASET_KEY: ["leaderboard_math"]},
        )
        with pytest.raises(ValueError, match="must be a mapping"):
            grading.load_grader(request, tmp_path, grading.GradingSettings())


#: A MATH completion whose derivation contains a blank line before the answer, which is
#: how 594 of the 1,324 Level-5 MATH-lighteval solutions are written. Everything the
#: grader reads -- the ``\boxed{}`` and the Minerva ``Final Answer:`` line -- is on the
#: far side of it.
BLANK_LINE_SOLUTION = (
    " Multiplying the first equation by $-\\frac{3}{2}$ gives\n"
    "\n"
    "$$6y-9x=-\\frac{3}{2}a.$$Since $6y-9x=b$, we have $\\boxed{-\\frac{2}{3}}$.\n"
    "Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."
)

#: The same answer, followed by the model running on into a question nobody asked. The
#: hazard the stop sequence exists for, and the reason ``Problem:`` cannot simply be
#: dropped too.
RUN_ON_SOLUTION = (
    " The value is $\\boxed{5}$.\n"
    "Final Answer: The final answer is $5$. I hope it is correct.\n"
    "\n"
    "Problem:\n"
    "What is $3+4$?\n"
    "\n"
    "Solution: The value is $\\boxed{7}$.\n"
    "Final Answer: The final answer is $7$. I hope it is correct."
)

#: The same run-on with the header miscased, which is why the lowercase stop exists. A
#: 135M base model copies the few-shot skeleton approximately, and a header it renders
#: as ``problem:`` would otherwise leave the hallucinated question inside the graded
#: completion -- where ``last_boxed_only_string``, an ``rfind``, reads its answer in
#: preference to the real one.
LOWERCASE_RUN_ON_SOLUTION = RUN_ON_SOLUTION.replace("Problem:", "problem:")

#: The reason the stop carries a colon and the bare word is not in the list. Six of the
#: 1,183 vendored questions and four of the exemplars use "problem" in exactly this way,
#: so a bare stop would cut this solution before its answer and grade it wrong.
PROSE_PROBLEM_SOLUTION = (
    " This problem requires the quadratic formula, and the problem states $a=1$.\n"
    "Rewriting the problem in standard form gives $x^2-5x+6=0$, so $x=\\boxed{3}$.\n"
    "Final Answer: The final answer is $3$. I hope it is correct."
)


class TestTheMathBankStopsAtTheHeaderNotTheBlankLine:
    """``["Problem:", "problem:"]``, not the task's ``["Problem:", "\\n\\n"]``.

    The olmo-eval task adds a blank line to lm-eval's single stop, and the addition was
    deleting answers rather than bounding run-on text. Both of the extractor's paths read
    the *end* of a solution -- ``last_boxed_only_string`` is an ``rfind`` and the Minerva
    ``Final Answer:`` line is written last -- so a cut at the first blank line leaves
    neither. Measured over the 1,324 Level-5 ``MATH-lighteval`` solutions this bank is
    drawn from, 594 contain a blank line and 541 lose their answer to the cut, a
    guaranteed wrong verdict on 40.9% of the bank.

    Dropping it is also what the calibration says: these difficulties came from Open LLM
    Leaderboard v2, whose MATH task is lm-eval's ``leaderboard_math``, whose
    ``generation_kwargs`` are ``until: ["Problem:"]`` and nothing else.

    The lowercase header is the one addition to that, for a base model that reproduces
    the few-shot skeleton imperfectly. It is a deviation from the calibration harness, so
    it is bounded by measurement rather than taste: over the 1,183 vendored questions and
    the four ``MINERVA_MATH_FIXED_FEWSHOT`` exemplars, ``problem:`` occurs zero times and
    ``Problem:`` zero times, while the bare word occurs 6 and 4 times as ordinary prose.
    The colon is therefore what separates a header from a sentence, and
    :data:`PROSE_PROBLEM_SOLUTION` pins that a bare stop is never added.
    """

    def config(self, **overrides) -> generative.GenerationConfig:
        """The committed MATH settings, optionally with one field forced."""
        settings = grading._apply_generation_overrides(
            grading.GradingSettings(),
            UniMcqStyle().generation_settings,
            dataset="leaderboard_math",
        )
        for field, value in overrides.items():
            setattr(settings.generation, field, value)
        return settings.generation

    def test_the_committed_config_stops_only_at_the_next_problem_header(self) -> None:
        assert self.config().stop_sequences == ("Problem:", "problem:")

    def test_the_bare_word_is_not_a_stop(self) -> None:
        """The colon is the whole difference between a header and a sentence."""
        assert "problem" not in self.config().stop_sequences
        assert all(stop.endswith(":") for stop in self.config().stop_sequences)

    def test_a_miscased_header_is_still_cut(self, real_math_extract) -> None:
        """Grading against the run-on's answer must fail, or the lowercase stop is dead."""
        config = self.config()

        assert not generative.grade_completion(
            math_item("7"), LOWERCASE_RUN_ON_SOLUTION, config
        ).correct
        assert generative.grade_completion(
            math_item("5"), LOWERCASE_RUN_ON_SOLUTION, config
        ).correct

    def test_a_solution_discussing_its_own_problem_survives(self, real_math_extract) -> None:
        """The regression a bare ``problem`` stop would cause, pinned rather than argued."""
        response = generative.grade_completion(
            math_item("3"), PROSE_PROBLEM_SOLUTION, self.config()
        )

        assert response.correct
        assert response.metadata["completion"] == PROSE_PROBLEM_SOLUTION

    def test_a_bare_problem_stop_would_have_graded_that_one_wrong(
        self, real_math_extract
    ) -> None:
        """The counterfactual, so the cost of adding the bare word stays visible."""
        response = generative.grade_completion(
            math_item("3"),
            PROSE_PROBLEM_SOLUTION,
            self.config(stop_sequences=("Problem:", "problem")),
        )

        assert not response.correct
        assert "\\boxed" not in response.metadata["completion"]

    def test_the_blank_line_is_gone_rather_than_reordered(self) -> None:
        """Order does not matter to ``truncate_at_stop``; presence does."""
        assert "\n\n" not in self.config().stop_sequences

    def test_a_solution_with_a_blank_line_in_it_survives_to_be_graded(
        self, real_math_extract
    ) -> None:
        response = generative.grade_completion(
            math_item("-\\frac{2}{3}"), BLANK_LINE_SOLUTION, self.config()
        )

        assert response.correct
        assert response.metadata["completion"] == BLANK_LINE_SOLUTION

    def test_the_task_stop_would_have_graded_that_same_answer_wrong(
        self, real_math_extract
    ) -> None:
        """The counterfactual, pinned so the cost of restoring the stop is visible."""
        response = generative.grade_completion(
            math_item("-\\frac{2}{3}"),
            BLANK_LINE_SOLUTION,
            self.config(stop_sequences=("Problem:", "\n\n")),
        )

        assert not response.correct
        assert "\\boxed" not in response.metadata["completion"]

    def test_the_surviving_stop_still_cuts_a_hallucinated_next_question(
        self, real_math_extract
    ) -> None:
        """Grading against the run-on's answer must fail, or nothing was cut."""
        config = self.config()

        assert not generative.grade_completion(math_item("7"), RUN_ON_SOLUTION, config).correct
        assert generative.grade_completion(math_item("5"), RUN_ON_SOLUTION, config).correct

    def test_the_run_on_text_is_not_carried_into_the_report(self) -> None:
        cut = generative.truncate_at_stop(RUN_ON_SOLUTION, self.config().stop_sequences)

        assert "\\boxed{5}" in cut
        assert "\\boxed{7}" not in cut

    def test_the_extractor_takes_the_last_boxed_expression(self, real_math_extract) -> None:
        """Why a cut before the answer is fatal rather than merely lossy.

        ``rfind`` means an intermediate ``\\boxed{}`` left standing by a truncation is
        read as the answer, so the failure is a confident wrong verdict and not a
        detectable blank.
        """
        two_boxes = "First $\\boxed{1}$, then on reflection $\\boxed{2}$."

        assert real_math_extract.last_boxed_only_string(two_boxes) == "\\boxed{2}"

    def test_a_truncated_derivation_states_no_answer_at_all(self, real_math_extract) -> None:
        """What the old stop actually handed the grader."""
        cut = generative.truncate_at_stop(BLANK_LINE_SOLUTION, ("Problem:", "\n\n"))

        assert not generative.has_final_answer(cut)

    def test_the_fewshot_block_teaches_the_header_the_stop_relies_on(self) -> None:
        """``Problem:`` only bounds a run-on if the model has been shown it as a boundary."""
        prompt = generative.format_generative_prompt(math_item(), self.config())

        assert "\n\nProblem:\n" in prompt

    def test_no_exemplar_solution_contains_the_stop_string(self) -> None:
        """A stop that can occur inside an answer is the bug being fixed, not the fix.

        ``Problem:`` appears in 0 of the 1,324 reference solutions; the four exemplars
        are the part of that distribution this test can reach without the dataset.
        """
        examples = generative.fewshot_examples(self.config())

        assert len(examples) == 4
        assert not any("Problem:" in example["solution"] for example in examples)


#: SmolLM2-135M's arrangement, which is this checkpoint family's. Written here as a fake
#: tokenizer's attributes rather than as a constant in ``generative``, which is the whole
#: design: the string is the model's and the benchmark never learns it.
SMOL_EOS = "<|endoftext|>"
SMOL_EOS_ID = 0

#: A different family's spelling, used to prove the resolution is a resolution. Nothing
#: in the source under test may prefer one of these two.
LLAMA_EOS = "<|eot_id|>"
LLAMA_EOS_ID = 128009

#: The two benchmark-level artifacts that may not name any model's end-of-text token.
CONFIG_YAML = Path(__file__).resolve().parents[1] / "config.yaml"
MATH_MANIFEST = CALIBRATED_DATASETS / "leaderboard_math" / "manifest.json"


#: What closing the exemplars with an end-of-text token risks, and the only case where the
#: run-time stop entry does any work. A model shown ``<|endoftext|>`` four times can learn
#: to spell those thirteen characters out of ordinary vocabulary instead of emitting id 0;
#: they are then not special, they survive ``skip_special_tokens=True``, and generation
#: does not halt on them.
#:
#: Deliberately without a ``Problem:`` header in the continuation. A run-on that reprints
#: the header is already cut by the two committed stops, so including one would let a test
#: pass whether or not the end-of-text entry exists.
#:
#: The harm runs in one direction, and it is not the direction the header stop's comments
#: describe. ``MathLatexEquivalence`` scores ``any(is_equiv(candidate, gold))`` over every
#: candidate ``extract_math_answer`` finds, so extra text can only add candidates:
#: an uncut leak turns wrong answers right, never right answers wrong. Here the model's
#: own answer is 5 and its hallucinated continuation says 7, so against a gold of 7 the
#: uncut completion is credited for an answer the model did not give. Across a bank that
#: is accuracy inflation, and it lands in theta as ability the checkpoint does not have.
LEAKED_EOS_RUN_ON = (
    " The value is $\\boxed{5}$.\n"
    "Final Answer: The final answer is $5$. I hope it is correct."
    f"{SMOL_EOS}Next, evaluate $2+5$. That gives $\\boxed{{7}}$.\n"
    "Final Answer: The final answer is $7$. I hope it is correct."
)


def long_math_item() -> BenchmarkItem:
    """A stem long enough to force the ladder where a short one does not.

    Real: the framed stems run to 1,527 SmolLM2 tokens over the 1,183 vendored items
    against a median of 69, and it is that spread -- not the exemplar block -- that makes
    a single context window produce different shot counts within one session.
    """
    return BenchmarkItem(
        item_id="intermediate_algebra_hard|913",
        question="Consider the following. " + "Suppose additionally that $x_i > 0$. " * 60,
        choices=(),
        gold_index=-1,
        metadata={"gold_answer": "\\frac{1}{2}", "answer_type": "math_latex"},
    )


def uncommented(text: str) -> str:
    """``text`` with ``#`` comment lines dropped, so prose about a token is not a token.

    ``config.yaml`` argues at length about why the token is absent from its settings, and
    naming it there is how that argument gets made. The settings themselves are the claim
    under test.
    """
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


class FakeTensor:
    """The slice of a torch tensor ``_HFCompleter`` touches: shape, ``.to``, row 0."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = list(ids)

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.ids))

    def to(self, device: object) -> FakeTensor:
        return self

    def __getitem__(self, index: int) -> list[int]:
        if index != 0:
            raise IndexError(index)
        return self.ids


class SpellingTokenizer:
    """A tokenizer that encodes its own end-of-text spelling to its own id.

    Ordinary words get ids from 1 upwards so that a count of ``eos_token_id`` in an
    encoded prompt is a count of real end-of-text tokens and not an artefact of the
    stub -- which matters here, because this family's id is 0 and a stub that emitted 0
    for anything else would make the round-trip assertion vacuous.
    """

    #: Matches :attr:`HaltingModel.config.vocab_size`, so the consistency check passes
    #: quietly on the ordinary path and only the subclass below trips it.
    size = 49152

    def __init__(self, eos_token: str | None, eos_token_id: int | None) -> None:
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.pad_token_id = None

    def __len__(self) -> int:
        return self.size

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for index, chunk in enumerate(text.split(self.eos_token or "\0")):
            if index:
                ids.append(self.eos_token_id)
            ids.extend(1 + len(word) % 97 for word in chunk.split())
        return ids

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ) -> dict[str, object]:
        ids = self._encode(text)
        return {"input_ids": FakeTensor(ids) if return_tensors else ids}

    def count_tokens(self, text: str) -> int:
        """``text -> token count``, the shape ``_HFCompleter.count_tokens`` publishes.

        Offered by the fake rather than reimplemented in each test so a prompt measured
        against the context window here is measured exactly as the real completer measures
        it -- content only, no special tokens.
        """
        return len(self(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Model ``skip_special_tokens`` faithfully, because the design turns on it."""
        kept = [i for i in ids if not (skip_special_tokens and i == self.eos_token_id)]
        return " ".join(str(i) for i in kept)


class MisspellingTokenizer(SpellingTokenizer):
    """Names an end-of-text token that its own encoder reads as ordinary text.

    Not a hypothetical: a tokenizer registers its special tokens as added tokens, and one
    that does not will happily report ``eos_token`` while tokenizing those characters
    piecewise. Teaching a model to type them would be worse than teaching it nothing.
    """

    def _encode(self, text: str) -> list[int]:
        return [1 + len(word) % 97 for word in text.split()]


class HugeVocabTokenizer(SpellingTokenizer):
    """A dolma2-sized vocabulary against a SmolLM2-sized model: 100,278 against 49,152."""

    size = 100278


class HaltingModel:
    """A ``generate`` that honours ``eos_token_id`` the way ``transformers`` does.

    It emits an answer, then its end-of-text id, then filler that only a caller who
    failed to pass the id will receive. The halting itself is the library's behaviour and
    is modelled here rather than tested -- ``torch`` is not installed, so there is no way
    to exercise the real one offline. What these tests can pin, and what the bug actually
    was, is whether the id is passed at all.
    """

    device = "cpu"

    def __init__(self, eos_token_id: int = SMOL_EOS_ID, context: int | None = 8192) -> None:
        self.eos_token_id = eos_token_id
        self.calls: list[dict] = []
        self.config = types.SimpleNamespace(vocab_size=49152)
        if context is not None:
            self.config.max_position_embeddings = context

    def eval(self) -> None:
        pass

    def generate(self, input_ids: FakeTensor, **kwargs: object) -> FakeTensor:
        self.calls.append(kwargs)
        budget = int(kwargs["max_new_tokens"])
        emitted = [11, 12, self.eos_token_id, 13, 14, 15]
        if kwargs.get("eos_token_id", "absent") == self.eos_token_id:
            emitted = emitted[: emitted.index(self.eos_token_id) + 1]
        return FakeTensor([*input_ids.ids, *emitted[:budget]])


class TestTheMathExemplarsCloseWithTheCheckpointsEndOfText:
    """The prompt-level half of making a base checkpoint stop, and its cost.

    This checkpoint writes ``eos_token_id == pad_token_id == bos_token_id == 0`` and was
    converted without a ``generation_config.json``, so nothing in the directory tells
    ``generate`` what to halt on and every generative item decodes its whole budget. The
    fix is two independent halves, and the tests below keep them independent:

    - ``generate(eos_token_id=...)``, which is what actually stops a completion, applies
      to every bank, and would work with the prompt untouched.
    - the end-of-text token closing each of the four exemplars, which is the only way to
      *induce* the behaviour in a base model with no instruction-following to appeal to,
      and is a deviation from the prompt this bank's difficulties were estimated behind.

    Every assertion here is about the mechanism rather than about a spelling. The
    benchmark declares ``exemplars_end_with_eos`` and the checkpoint supplies the string,
    so both :data:`SMOL_EOS` and :data:`LLAMA_EOS` are put through the same paths: a test
    that only passed for ``<|endoftext|>`` would be pinning the frozen literal this
    design exists to avoid.
    """

    def config(self, **overrides) -> generative.GenerationConfig:
        settings = grading._apply_generation_overrides(
            grading.GradingSettings(),
            UniMcqStyle().generation_settings,
            dataset="leaderboard_math",
        )
        for field, value in overrides.items():
            setattr(settings.generation, field, value)
        return settings.generation

    def prompt(self, eos_token: str | None, **overrides) -> str:
        return generative.format_generative_prompt(
            math_item(), self.config(**overrides), eos_token=eos_token
        )

    # --- the benchmark declares the convention, the model supplies the string ---

    def test_the_math_template_declares_that_its_exemplars_close_with_eos(self) -> None:
        assert generative.PROMPT_TEMPLATES["leaderboard_math"].exemplars_end_with_eos

    def test_no_other_template_declares_it(self) -> None:
        """The deviation is argued per bank; gsm8k, ifeval and gpqa keep their prompts."""
        declaring = {
            name
            for name, template in generative.PROMPT_TEMPLATES.items()
            if template.exemplars_end_with_eos
        }

        assert declaring == {"leaderboard_math"}

    @pytest.mark.parametrize("artifact", [CONFIG_YAML, MATH_MANIFEST])
    def test_no_benchmark_artifact_names_a_models_end_of_text_token(self, artifact) -> None:
        """The design rule, as an assertion rather than as a comment.

        Both files describe the benchmark and are read before any checkpoint is fetched --
        the manifest by ``convention.check_runtime_convention``, which cannot see a
        run-time value by construction. A literal in either would be one model's answer
        recorded as the benchmark's, and would be ordinary text to every other model.
        """
        settings = uncommented(artifact.read_text(encoding="utf-8"))

        for spelling in (SMOL_EOS, LLAMA_EOS, "</s>", "<|im_end|>", "<EOS>"):
            assert spelling not in settings

    # --- every exemplar ends with it ---

    @pytest.mark.parametrize("eos", [SMOL_EOS, LLAMA_EOS])
    def test_every_exemplar_ends_with_the_resolved_token(self, eos: str) -> None:
        rendered = self.prompt(eos)
        exemplars = rendered.split("\n\n" + "Problem:\n")

        assert len(exemplars) == 5  # four worked examples, then the live question
        assert rendered.count(eos) == 4
        for exemplar in exemplars[:4]:
            assert exemplar.endswith(eos)

    @pytest.mark.parametrize("eos", [SMOL_EOS, LLAMA_EOS])
    def test_the_final_exemplar_carries_it_too(self, eos: str) -> None:
        """Decided deliberately; the reasoning is beside the template, not here.

        Pinned separately from the count above because it is the one placement with an
        argument against it -- the live question follows immediately, so this token has
        prompt text after it.
        """
        rendered = self.prompt(eos)
        last_exemplar_end = rendered.rindex(eos) + len(eos)

        assert rendered[last_exemplar_end:].startswith("\n\nProblem:\n")
        assert "Solution:" in rendered[last_exemplar_end:]  # the live question, unanswered

    def test_it_lands_after_the_final_answer_line_not_before_it(self) -> None:
        """Order matters: the grader reads the ``Final Answer`` line the exemplars teach.

        Appending ahead of that line would leave the demonstration intact and quietly stop
        teaching the one thing :class:`MathLatexEquivalence` needs.
        """
        for exemplar in self.prompt(SMOL_EOS).split("\n\nProblem:\n")[:4]:
            assert exemplar.endswith(SMOL_EOS)
            assert exemplar.index("Final Answer:") < exemplar.rindex(SMOL_EOS)

    def test_the_live_question_is_not_closed_with_it(self) -> None:
        """It is the model's job to emit the token, not the prompt's to supply it."""
        assert not self.prompt(SMOL_EOS).endswith(SMOL_EOS)

    # --- the upstream constant is untouched ---

    def test_the_upstream_constant_carries_no_end_of_text_token(self) -> None:
        """Imported fresh, so an in-place edit of olmo_eval's constant is caught here.

        The block is shared with the real olmo-eval tasks and with ``leaderboard_math.py``
        and ``minerva_math.py``. Appending there would change what those score.
        """
        from olmo_eval.evals.tasks.constants.minerva_math import MINERVA_MATH_FIXED_FEWSHOT

        for example in MINERVA_MATH_FIXED_FEWSHOT:
            for spelling in (SMOL_EOS, LLAMA_EOS, "</s>", "<|im_end|>", "<EOS>"):
                assert spelling not in example["solution"]

    def test_the_transformation_boundary_stays_a_pure_data_source(self) -> None:
        """``_leaderboard_math_fixed_fewshot`` returns the same four examples every run."""
        for example in generative._leaderboard_math_fixed_fewshot():
            assert SMOL_EOS not in example["solution"]

    # --- the round trip ---

    def test_the_rendered_prompt_encodes_to_the_real_id_once_per_exemplar(self) -> None:
        """The point of resolving a string at all: it has to survive tokenization.

        Verified against a stub that models SmolLM2-135M's arrangement -- id 0 as eos,
        bos and unk alike -- rather than against the real tokenizer, which is not
        available offline. Its ``tokenizer_config.json`` was read from the Hub and
        records ``eos_token: "<|endoftext|>"`` at id 0, ``normalized: false`` and
        ``special: true``, which is the property this stub reproduces.
        """
        tokenizer = SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID)
        encoded = tokenizer(self.prompt(SMOL_EOS), add_special_tokens=False)["input_ids"]

        assert encoded.count(SMOL_EOS_ID) == 4

    def test_the_ids_fall_where_the_string_was_put_not_merely_somewhere(self) -> None:
        """A count alone would pass on four tokens encoded in the wrong places.

        Each id has to sit at the boundary the rendered string put it at, which is the
        cumulative length of everything before it plus the ids already emitted.
        """
        tokenizer = SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID)
        rendered = self.prompt(SMOL_EOS)
        encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]

        expected: list[int] = []
        consumed = 0
        for chunk in rendered.split(SMOL_EOS)[:-1]:
            consumed += len(tokenizer(chunk, add_special_tokens=False)["input_ids"])
            expected.append(consumed + len(expected))

        assert [i for i, token in enumerate(encoded) if token == SMOL_EOS_ID] == expected

    def test_resolution_reads_the_tokenizer_and_checks_the_round_trip(self) -> None:
        assert generative.resolve_eos_token(SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID)) == (
            SMOL_EOS,
            SMOL_EOS_ID,
        )
        assert generative.resolve_eos_token(SpellingTokenizer(LLAMA_EOS, LLAMA_EOS_ID)) == (
            LLAMA_EOS,
            LLAMA_EOS_ID,
        )

    def test_a_token_that_does_not_round_trip_is_refused_as_a_teaching_signal(self, caplog) -> None:
        """Stop on the id, teach nothing: typing the characters is worse than silence."""
        with caplog.at_level(logging.WARNING, logger="mcq_cat.generative"):
            text, token_id = generative.resolve_eos_token(
                MisspellingTokenizer(SMOL_EOS, SMOL_EOS_ID)
            )

        assert text is None
        assert token_id == SMOL_EOS_ID
        assert "ordinary text" in caplog.text

    # --- the id, which is the half that actually stops generation ---

    @pytest.fixture
    def fake_stack(self, monkeypatch):
        """``torch`` and ``transformers`` stubbed; ``torch`` is not installed here."""

        def install(tokenizer: object, model: object):
            import contextlib

            torch = types.ModuleType("torch")
            torch.no_grad = contextlib.nullcontext
            transformers = types.ModuleType("transformers")
            transformers.AutoTokenizer = types.SimpleNamespace(
                from_pretrained=lambda *a, **k: tokenizer
            )
            transformers.AutoModelForCausalLM = types.SimpleNamespace(
                from_pretrained=lambda *a, **k: model
            )
            transformers.set_seed = lambda seed: None
            monkeypatch.setitem(sys.modules, "torch", torch)
            monkeypatch.setitem(sys.modules, "transformers", transformers)
            return generative._HFCompleter(Path("/ckpt"), self.config())

        return install

    def test_the_resolved_id_is_passed_to_generate(self, fake_stack) -> None:
        """The whole bug: without this, ``generate`` has nothing to halt on."""
        model = HaltingModel()
        completer = fake_stack(SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID), model)
        completer("Problem:\nx?\n\nSolution:")

        assert model.calls[0]["eos_token_id"] == SMOL_EOS_ID

    def test_an_id_of_zero_is_passed_rather_than_treated_as_absent(self, fake_stack) -> None:
        """``bool(0)`` is ``False``, and 0 is exactly this checkpoint's id.

        A truth test here would discard the one value the fix exists for, and every test
        above would still pass because the string half is unaffected.
        """
        model = HaltingModel()
        completer = fake_stack(SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID), model)

        assert completer.eos_token_id == 0
        assert completer._eos_kwargs() == {"eos_token_id": 0}
        completer("Problem:\nx?\n\nSolution:")
        assert "eos_token_id" in model.calls[0]

    def test_generation_stops_at_the_token_the_model_emits(self, fake_stack) -> None:
        model = HaltingModel()
        completer = fake_stack(SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID), model)

        assert completer("Problem:\nx?\n\nSolution:") == "11 12"

    def test_without_the_id_the_same_model_runs_past_it(self, fake_stack) -> None:
        """The counterfactual, which is what this bank was doing before the fix."""
        tokenizer = SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID)
        model = HaltingModel()
        completer = fake_stack(tokenizer, model)
        completer.eos_token_id = None

        assert completer("Problem:\nx?\n\nSolution:") == "11 12 13 14 15"

    def test_no_kwarg_is_passed_when_nothing_resolved(self) -> None:
        """The degraded path itself, which the refusal above makes unreachable in a run."""
        completer = object.__new__(generative._HFCompleter)
        completer.eos_token_id = None

        assert completer._eos_kwargs() == {}

    def test_a_tokenizer_with_no_end_of_text_id_is_refused_at_load(self, fake_stack) -> None:
        """A real run always has a tokenizer, so this is a failure and not a mode.

        Without an id there is nothing for ``generate`` to halt on, so every item would
        decode its whole budget and the report would look entirely normal. The guard is at
        checkpoint load rather than at prompt build precisely so the offline tests, which
        load no checkpoint, keep exercising the degraded path deliberately.
        """
        with pytest.raises(RuntimeError, match="no eos_token_id") as excinfo:
            fake_stack(SpellingTokenizer(None, None), HaltingModel())

        assert "full 1024 new tokens" in str(excinfo.value)

    def test_an_unloadable_tokenizer_is_refused_with_the_cost_named(self, monkeypatch) -> None:
        """The Hub-unreachable case ``HF_CONVERSION.md`` flags, since no files ship."""
        broken = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: (_ for _ in ()).throw(OSError("no tokenizer files"))
        )

        with pytest.raises(RuntimeError, match="No usable tokenizer") as excinfo:
            generative._HFCompleter._load_tokenizer(broken, Path("/ckpt"))

        assert "never halt" in str(excinfo.value)
        assert "into the exemplar block" in str(excinfo.value)

    def test_the_backend_hands_the_string_from_the_completer_to_the_scorer(
        self, fake_stack
    ) -> None:
        """The seam: one place where both the tokenizer and the config exist."""
        fake_stack(SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID), HaltingModel())
        scorer = generative.GENERATIVE_BACKENDS["hf"](Path("/ckpt"), self.config())

        assert isinstance(scorer, generative.GenerativeScorer)
        assert scorer.eos_token == SMOL_EOS

    def test_a_mismatched_tokenizer_is_reported_rather_than_scored_silently(
        self, fake_stack, caplog
    ) -> None:
        """The check that would have caught reading the wrong identifier off the config.

        A dolma2 tokenizer against this checkpoint is 100,278 tokens against 49,152, which
        is a tokenizer able to emit ids the model has no embedding for.
        """
        with caplog.at_level(logging.WARNING, logger="mcq_cat.generative"):
            fake_stack(HugeVocabTokenizer(SMOL_EOS, SMOL_EOS_ID), HaltingModel())

        assert "cannot represent" in caplog.text

    def test_a_matching_pair_is_not_complained_about(self, fake_stack, caplog) -> None:
        """One direction only: a model larger than its tokenizer is ordinary padding."""
        with caplog.at_level(logging.WARNING, logger="mcq_cat.generative"):
            fake_stack(SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID), HaltingModel())

        assert caplog.text == ""

    # --- the stop list, and what it is and is not for ---

    def test_the_committed_config_names_no_end_of_text_token(self) -> None:
        """It is checkpoint-level; ``config.yaml`` and the manifest are benchmark-level."""
        assert self.config().stop_sequences == ("Problem:", "problem:")

    def test_the_runtime_list_carries_it_and_the_committed_one_does_not(self) -> None:
        config = self.config()

        assert generative.eos_stop_sequences(math_item(), config, SMOL_EOS) == (
            "Problem:",
            "problem:",
            SMOL_EOS,
        )
        assert generative.eos_stop_sequences(math_item(), config, None) == config.stop_sequences

    def test_a_genuine_special_token_never_reaches_the_stop_list(self, fake_stack) -> None:
        """Why the entry above is a leak-catcher and not the stopping mechanism.

        ``__call__`` decodes with ``skip_special_tokens=True``, so a real end-of-text
        token is deleted from the text before ``truncate_at_stop`` sees any of it. A test
        that merely asserted the string was in the list would pass while pinning nothing.
        """
        completer = fake_stack(SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID), HaltingModel())

        assert SMOL_EOS not in completer("Problem:\nx?\n\nSolution:")

    def test_a_typed_out_end_of_text_is_cut_with_what_follows_it(self, real_math_extract) -> None:
        """The failure closing the exemplars invites; see :data:`LEAKED_EOS_RUN_ON`."""
        response = generative.grade_completion(
            math_item("5"), LEAKED_EOS_RUN_ON, self.config(), eos_text=SMOL_EOS
        )

        assert response.correct
        assert SMOL_EOS not in response.metadata["completion"]
        assert "\\boxed{7}" not in response.metadata["completion"]

    def test_without_the_runtime_entry_the_hallucination_is_credited(
        self, real_math_extract
    ) -> None:
        """The counterfactual, so the entry is not dead code asserted into existence.

        The model answered 5. Uncut, it is scored correct against a gold of 7 as well,
        because the continuation it invented after its own end-of-text token states 7 and
        the grader accepts any candidate it can find.
        """
        assert generative.grade_completion(math_item("7"), LEAKED_EOS_RUN_ON, self.config()).correct
        assert not generative.grade_completion(
            math_item("7"), LEAKED_EOS_RUN_ON, self.config(), eos_text=SMOL_EOS
        ).correct

    def test_the_committed_stops_alone_do_not_cover_it(self) -> None:
        """Why the entry is added rather than argued away: no header to catch."""
        assert not any(stop in LEAKED_EOS_RUN_ON for stop in self.config().stop_sequences)

    def test_the_stop_list_is_not_what_bounds_the_cost(self, real_math_extract) -> None:
        """``truncate_at_stop`` runs on text already generated, so every token is paid for.

        What bounds a run is the token budget and the id passed to ``generate``. The cap is
        1024, which is what ``leaderboard_math.py`` itself declares and what the
        calibration population generated behind; see the derivation in ``config.yaml``.
        """
        config = self.config()
        graded = generative.grade_completion(
            math_item("5"), boxed("5") + SMOL_EOS + " and more", config, eos_text=SMOL_EOS
        )

        assert config.max_new_tokens == 1024
        assert graded.metadata["completion"] == boxed("5")

    # --- degradation, which is most of the callers ---

    def test_a_prompt_built_with_no_tokenizer_is_the_calibrated_one(self) -> None:
        """Most callers here have no tokenizer, and offline is the default path."""
        bare = self.prompt(None)

        assert SMOL_EOS not in bare
        assert bare == generative.format_generative_prompt(math_item(), self.config())

    def test_no_token_resolved_means_nothing_appended_anywhere(self) -> None:
        config = self.config()

        assert generative.exemplar_eos_token(config, None) is None
        assert generative.exemplar_eos_token(config, "") is None
        assert generative.eos_stop_sequences(math_item(), config, None) == config.stop_sequences

    def test_a_zero_shot_run_appends_nothing(self) -> None:
        """No exemplar to close, and so no spelling demonstrated for a leak to copy."""
        assert generative.exemplar_eos_token(self.config(num_fewshot=0), SMOL_EOS) is None

    def test_a_bank_that_does_not_declare_the_convention_appends_nothing(self) -> None:
        config = generative.GenerationConfig(
            num_fewshot=8, fewshot_source="gsm8k", prompt_style="gsm8k"
        )

        assert generative.exemplar_eos_token(config, SMOL_EOS) is None

    def test_the_gsm8k_prompt_is_unchanged_by_a_resolved_token(self, monkeypatch) -> None:
        monkeypatch.setitem(generative.FEWSHOT_SOURCES, "gsm8k", lambda: STUB_FEWSHOT)
        config = generative.GenerationConfig(
            num_fewshot=1, fewshot_source="gsm8k", prompt_style="gsm8k"
        )
        item = BenchmarkItem(
            item_id="g0",
            question="How many clips?",
            choices=(),
            gold_index=-1,
            metadata={"gold_answer": "72", "answer_type": "numeric"},
        )

        assert generative.format_generative_prompt(
            item, config, eos_token=SMOL_EOS
        ) == generative.format_generative_prompt(item, config)

    def test_the_scorer_degrades_without_raising(self, real_math_extract) -> None:
        """A large number of existing tests construct this with no tokenizer at all."""
        scorer = generative.GenerativeScorer(lambda _: boxed("\\frac{1}{2}"), self.config())

        assert scorer.eos_token is None
        assert scorer.score_items([math_item()])[0].metadata["completion"]


class TestTheExemplarBlockIsNeverCutIntoToFitTheContextWindow:
    """The context clamp, and the one rule that makes it different from Research's.

    Research's ``respgen/runner.py`` ``_fit_prompt_and_budget`` left-truncates the prompt
    and keeps the tail. That is right for a chat transcript and destructive here: the four
    Minerva exemplars are what teach the ``\\boxed{}`` expression and the ``Final Answer:``
    line, which are the two forms the grader reads, so eating the front of the block would
    leave a correctly-sized prompt that no longer teaches the answer format. Grading would
    then fail while every guard passed, and the run would report a theta instead of an
    error. So the unit of reduction here is a whole exemplar, and the floor is one.

    Measured with SmolLM2-135M's own tokenizer -- downloaded, 49,152 tokens, ``<|endoftext|>``
    at id 0 -- the four exemplars are 146, 118, 221 and 195 tokens closed with end-of-text
    and joined, 680 together, and the framed stems run median 69, p90 190, p99 589, max
    1,527 over the 1,183 vendored items. Against ``hf_config_patch``'s emitted 2048 that
    puts the longest-stem items over the window, which is why this exists rather than being
    insurance: the ladder fires on this checkpoint as configured.

    The fakes below are sized in *words*, not those token counts, so the tests pin the
    mechanism and stay readable. The measured figures are recorded in ``config.yaml``.
    """

    def config(self, **overrides) -> generative.GenerationConfig:
        config = generative.GenerationConfig(
            num_fewshot=4,
            fewshot_source="leaderboard_math",
            prompt_style="leaderboard_math",
            max_new_tokens=1024,
            stop_sequences=("Problem:", "problem:"),
        )
        for field, value in overrides.items():
            setattr(config, field, value)
        return config

    def fitter(self, context: int | None) -> generative.PromptFitter:
        return generative.PromptFitter(
            count_tokens=SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID).count_tokens,
            context_length=context,
            eos_token=SMOL_EOS,
        )

    def fit(self, context: int | None, **overrides) -> generative.PromptFit:
        return self.fitter(context).fit(math_item(), self.config(**overrides))

    def exemplars(self) -> tuple[dict[str, str], ...]:
        return generative._leaderboard_math_fixed_fewshot()

    # --- a window that fits everything changes nothing ---

    def test_a_roomy_window_leaves_all_four_exemplars_and_the_full_cap(self) -> None:
        fit = self.fit(8192)

        assert fit.num_fewshot == 4
        assert fit.dropped_exemplars == 0
        assert fit.clamped is False
        assert fit.gen_budget == 1024
        assert fit.prompt == generative.format_generative_prompt(
            math_item(), self.config(), eos_token=SMOL_EOS
        )

    def test_no_context_length_means_no_clamp_and_no_budget_override(self) -> None:
        """``None`` is the undeterminable case, and it must not silently reduce anything."""
        fit = self.fit(None)

        assert fit.num_fewshot == 4
        assert fit.gen_budget is None
        assert fit.context_length is None
        assert fit.clamped is False

    def test_a_missing_context_length_is_logged_rather_than_guessed(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="mcq_cat.generative"):
            assert generative.context_length_of(HaltingModel(context=None)) is None

        assert "context clamp is disabled" in caplog.text

    # --- the budget clamp, and Research's collapse-to-1 bug ---

    def test_the_bank_cap_wins_when_it_is_the_smaller_bound(self) -> None:
        """Order matters: the cap is applied first, and the clamp can only lower it.

        Reversing them would let a large-context model generate far past 1024 on a bank
        whose difficulties were all estimated at 1024.
        """
        fit = self.fit(16384)

        assert fit.gen_budget == 1024
        assert fit.clamped is False

    def test_a_window_smaller_than_the_nominal_budget_still_gives_a_usable_budget(self) -> None:
        """Research's bug, pinned: ``max(1, context - max_new_tokens)`` collapses to 1.

        Their earlier version hit this on every model at ``max_model_len <= 4096`` and then
        left-truncated the prompt to its final token. The budget must key off the ACTUAL
        prompt length with a floor.
        """
        fit = self.fit(600)

        assert fit.gen_budget is not None
        assert fit.gen_budget >= generative.MIN_GENERATION_TOKENS
        assert fit.gen_budget != 1
        assert fit.clamped is True

    def test_the_floor_and_the_window_bound_the_budget_independently(self) -> None:
        """Both guards, exercised directly on the clamp because the ladder hides them.

        The ladder only offers a prompt the clamp accepts, so by then the room already
        exceeds the reserve and neither guard can fire through :meth:`PromptFitter.fit`.
        They are on :func:`fit_budget_to_context` for its own sake, and pinning them here
        is what keeps them from being untested claims.

        The window bound is a REFUSAL, which is where this class's own earlier arithmetic
        was wrong rather than merely different. ``min(cap, max(reserve, room))`` bounded by
        ``context - 1`` handed a 4000-token prompt in a 4096-token window a 256-token
        budget the window physically could not hold, and a 10-token prompt in a 100-token
        window 99. Both states were unreachable from :meth:`fit`, which refused them first,
        so the two halves of the clamp disagreed about cases only one of them could see.
        There is one clamp now and it gives :meth:`fit`'s answer everywhere.

        The floor bound is unchanged: a prompt that fits always gets a usable budget rather
        than the 1 token ``context - nominal_budget`` collapses to, and the reserve is
        capped by the demand so a small item is not refused for room it never wanted.
        """
        cap = self.config().max_new_tokens

        assert generative.fit_budget_to_context(cap, 4000, 4096) is None
        assert generative.fit_budget_to_context(cap, 10, 100) is None
        assert generative.fit_budget_to_context(cap, 10, 300) == 290
        assert generative.fit_budget_to_context(64, 10, 100) == 64

    # --- the ladder drops whole exemplars, never part of one ---

    def test_a_tight_window_drops_exemplars_rather_than_shrinking_one(self) -> None:
        fit = self.fit(420)

        assert 1 <= fit.num_fewshot < 4
        assert fit.dropped_exemplars == 4 - fit.num_fewshot

    @pytest.mark.parametrize("context", [500, 460, 430, 400])
    def test_every_surviving_exemplar_is_whole_and_they_are_a_suffix(self, context: int) -> None:
        """Not merely shorter: the block must be whole exemplars, and the LAST ones.

        A left-truncation that happened to land near a boundary would satisfy a
        length-only assertion. This compares the rendered exemplars against the real four.
        """
        fit = self.fit(context)
        kept = fit.prompt.split("\n\nProblem:\n")[: fit.num_fewshot]
        expected = self.exemplars()[4 - fit.num_fewshot :]

        assert len(kept) == fit.num_fewshot
        for rendered, source in zip(kept, expected, strict=True):
            assert rendered.endswith(source["solution"] + SMOL_EOS)
            assert source["question"] in rendered

    def test_the_first_exemplars_are_the_ones_dropped(self) -> None:
        """Recency dominates in-context learning; the adjacent exemplar is kept.

        The reasoning is on :class:`PromptFitter`; this pins the direction so a later
        change to drop from the back has to argue with a failing test.
        """
        fit = self.fit(420)
        dropped = self.exemplars()[: 4 - fit.num_fewshot]
        kept = self.exemplars()[4 - fit.num_fewshot :]

        assert fit.num_fewshot < 4
        assert all(example["question"] not in fit.prompt for example in dropped)
        assert all(example["question"] in fit.prompt for example in kept)

    def test_the_live_question_always_survives(self) -> None:
        for context in (400, 500, 8192):
            assert "What is one half?" in self.fit(context).prompt

    # --- the floor is one exemplar, and below it the item is ungradable ---

    def test_a_window_too_small_for_one_exemplar_is_ungradable_not_zero_shot(self) -> None:
        """A 0-shot MATH prompt teaches neither answer form, so its zero is fabricated."""
        fit = self.fit(120)

        assert fit.ungradable_reason == generative.CONTEXT_OVERFLOW_REASON
        assert fit.prompt == ""

    def test_the_ladder_never_returns_a_zero_shot_prompt(self) -> None:
        """Swept rather than spot-checked, because one shot is the whole floor."""
        for context in range(80, 900, 20):
            fit = self.fit(context)
            if fit.ungradable_reason is None:
                assert fit.num_fewshot >= 1
                assert generative.fewshot_examples(self.config(), fit.num_fewshot)

    def test_an_overflowed_item_is_scored_zero_and_marked_for_the_report(self) -> None:
        """It reaches ``style._ungradable_block``, which carries count, ids and reasons."""
        scorer = generative.GenerativeScorer(
            lambda *a: boxed("\\frac{1}{2}"), self.config(), fitter=self.fitter(120)
        )
        response = scorer.score_items([math_item()])[0]

        assert response.correct is False
        assert response.metadata[generative.UNGRADABLE_KEY] is True
        assert (
            response.metadata[generative.UNGRADABLE_REASON_KEY]
            == generative.CONTEXT_OVERFLOW_REASON
        )
        assert response.metadata["completion"] == ""

    def test_an_overflowed_item_is_not_sampled_at_all(self) -> None:
        """No prompt exists that both fits and teaches the format, so none is sent."""
        sent: list[str] = []
        generative.GenerativeScorer(
            lambda prompt, *a: sent.append(prompt) or "", self.config(), fitter=self.fitter(120)
        ).score_items([math_item()])

        assert sent == []

    # --- what the report has to carry ---

    def test_the_shot_count_used_is_recorded_per_item(self) -> None:
        """Per item, not per session: a small window mixes counts across one estimate."""
        scorer = generative.GenerativeScorer(
            lambda *a: boxed("\\frac{1}{2}"), self.config(), fitter=self.fitter(420)
        )
        metadata = scorer.score_items([math_item()])[0].metadata

        assert metadata["num_fewshot"] < 4
        assert metadata["context_fit"]["num_fewshot_configured"] == 4
        assert metadata["context_fit"]["exemplars_dropped"] == 4 - metadata["num_fewshot"]
        assert metadata["context_fit"]["context_length"] == 420

    def test_an_unclamped_item_carries_no_context_block(self) -> None:
        """Nine banks' reports keep their shape when the clamp does nothing."""
        scorer = generative.GenerativeScorer(
            lambda *a: boxed("\\frac{1}{2}"), self.config(), fitter=self.fitter(8192)
        )
        metadata = scorer.score_items([math_item()])[0].metadata

        assert "context_fit" not in metadata
        assert metadata["num_fewshot"] == 4

    def test_the_session_facts_say_whether_end_of_text_worked(self) -> None:
        """The report must answer this without the log; the cost difference is every item."""
        scorer = generative.GenerativeScorer(
            lambda *a: boxed("\\frac{1}{2}"),
            self.config(),
            eos_token=SMOL_EOS,
            fitter=self.fitter(8192),
            checkpoint_facts={"eos_token_id": SMOL_EOS_ID, "eos_token_id_passed_to_generate": True},
        )
        scorer.score_items([math_item()])
        facts = scorer.runtime_facts()

        assert facts["exemplars_end_with_eos"] is True
        assert facts["eos_token_id_passed_to_generate"] is True
        assert facts["context_clamp_active"] is True
        assert facts["context_length"] == 8192
        assert facts["num_fewshot_used"] == [4]
        assert facts["context_clamp_fired"] is False
        assert "mixed_shot_alert" not in facts

    def test_mixed_shot_counts_raise_an_alert_in_the_report(self) -> None:
        """The failure a session-level summary would hide: two scales in one theta."""
        scorer = generative.GenerativeScorer(
            lambda *a: boxed("\\frac{1}{2}"), self.config(), fitter=self.fitter(700)
        )
        scorer.score_items([math_item(), long_math_item()])
        facts = scorer.runtime_facts()

        assert len(facts["num_fewshot_used"]) > 1
        assert "NOT comparable" in facts["mixed_shot_alert"]
        assert facts["context_clamp_fired"] is True

    def test_a_run_with_no_fitter_reports_the_clamp_as_inactive(self) -> None:
        scorer = generative.GenerativeScorer(lambda *a: boxed("\\frac{1}{2}"), self.config())
        scorer.score_items([math_item()])

        assert scorer.runtime_facts()["context_clamp_active"] is False
        assert scorer.runtime_facts()["context_length"] is None

    # --- the fitted budget reaches generate ---

    def test_the_fitted_budget_is_what_generate_is_given(self, monkeypatch) -> None:
        import contextlib

        model = HaltingModel(context=700)
        torch = types.ModuleType("torch")
        torch.no_grad = contextlib.nullcontext
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: SpellingTokenizer(SMOL_EOS, SMOL_EOS_ID)
        )
        transformers.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: model
        )
        transformers.set_seed = lambda seed: None
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "transformers", transformers)

        scorer = generative.GENERATIVE_BACKENDS["hf"](Path("/ckpt"), self.config())
        scorer.score_items([math_item()])

        assert scorer.fitter is not None
        assert scorer.fitter.context_length == 700
        assert model.calls[0]["max_new_tokens"] < 1024
        assert model.calls[0]["max_new_tokens"] >= generative.MIN_GENERATION_TOKENS
