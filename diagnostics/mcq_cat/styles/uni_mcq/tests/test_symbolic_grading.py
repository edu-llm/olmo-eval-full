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
        assert config.stop_sequences == ("Problem:", "\n\n")

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
