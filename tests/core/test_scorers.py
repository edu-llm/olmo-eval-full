"""Tests for olmo_eval.core.scorers module."""

from olmo_eval.common.scorers import (
    ExactMatchScorer,
    MultipleChoiceScorer,
    SQuADExactMatchScorer,
    SQuADF1Scorer,
)
from olmo_eval.common.types import Instance, LMOutput


class TestExactMatchScorer:
    """Tests for ExactMatchScorer."""

    def test_exact_match_correct(self):
        """Test exact match with correct answer."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="Paris")
        output.extracted_answer = "Paris"

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_exact_match_incorrect(self):
        """Test exact match with incorrect answer."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="London")
        output.extracted_answer = "London"

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_case_insensitive(self):
        """Test case insensitive matching (default)."""
        scorer = ExactMatchScorer(case_sensitive=False)
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="paris")
        output.extracted_answer = "paris"

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_exact_match_case_sensitive(self):
        """Test case sensitive matching."""
        scorer = ExactMatchScorer(case_sensitive=True)
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="paris")
        output.extracted_answer = "paris"

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_strips_whitespace(self):
        """Test whitespace stripping (default)."""
        scorer = ExactMatchScorer(strip_whitespace=True)
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="  Paris  ")
        output.extracted_answer = "  Paris  "

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_exact_match_no_strip_whitespace(self):
        """Test without whitespace stripping."""
        scorer = ExactMatchScorer(strip_whitespace=False)
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="  Paris  ")
        output.extracted_answer = "  Paris  "

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_none_gold_answer(self):
        """Test with None gold answer."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="answer")
        output.extracted_answer = "answer"

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_none_extracted_answer(self):
        """Test with None extracted answer."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="text")
        output.extracted_answer = None

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_both_none(self):
        """Test with both answers None."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="text")
        output.extracted_answer = None

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_exact_match_name(self):
        """Test scorer name."""
        scorer = ExactMatchScorer()
        assert scorer.name == "exact_match"

        custom = ExactMatchScorer(name="custom_exact")
        assert custom.name == "custom_exact"

    def test_exact_match_converts_to_string(self):
        """Test that extracted answer is converted to string."""
        scorer = ExactMatchScorer()
        instance = Instance(question="Q", gold_answer="42")
        output = LMOutput(text="42")
        output.extracted_answer = 42  # Integer

        score = scorer.score(instance, output)

        assert score == 1.0


class TestMultipleChoiceScorer:
    """Tests for MultipleChoiceScorer."""

    def test_mc_correct(self):
        """Test multiple choice with correct answer."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer="B")
        output = LMOutput(text="B")
        output.extracted_answer = "B"

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_mc_incorrect(self):
        """Test multiple choice with incorrect answer."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer="B")
        output = LMOutput(text="A")
        output.extracted_answer = "A"

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_mc_case_insensitive(self):
        """Test multiple choice is case insensitive."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer="B")
        output = LMOutput(text="b")
        output.extracted_answer = "b"

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_mc_strips_whitespace(self):
        """Test multiple choice strips whitespace."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer="B")
        output = LMOutput(text=" B ")
        output.extracted_answer = " B "

        score = scorer.score(instance, output)

        assert score == 1.0

    def test_mc_none_gold_answer(self):
        """Test with None gold answer."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="A")
        output.extracted_answer = "A"

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_mc_none_extracted_answer(self):
        """Test with None extracted answer."""
        scorer = MultipleChoiceScorer()
        instance = Instance(question="Q", gold_answer="A")
        output = LMOutput(text="text")
        output.extracted_answer = None

        score = scorer.score(instance, output)

        assert score == 0.0

    def test_mc_name(self):
        """Test scorer name."""
        scorer = MultipleChoiceScorer()
        assert scorer.name == "multiple_choice"

        custom = MultipleChoiceScorer(name="custom_mc")
        assert custom.name == "custom_mc"


def _output(text: str) -> LMOutput:
    output = LMOutput(text=text)
    output.extracted_answer = text
    return output


class TestSQuADExactMatchScorer:
    """Tests for SQuADExactMatchScorer."""

    def test_identical_answer_matches(self):
        scorer = SQuADExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Shakespeare")

        assert scorer.score(instance, _output("Shakespeare")) == 1.0

    def test_different_answer_does_not_match(self):
        scorer = SQuADExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Shakespeare")

        assert scorer.score(instance, _output("Dickens")) == 0.0

    def test_articles_are_ignored(self):
        """SQuAD normalization drops a/an/the, so these are the same answer."""
        scorer = SQuADExactMatchScorer()
        instance = Instance(question="Q", gold_answer="the Great Depression")

        assert scorer.score(instance, _output("Great Depression")) == 1.0

    def test_punctuation_and_case_are_ignored(self):
        scorer = SQuADExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Washington, D.C.")

        assert scorer.score(instance, _output("washington dc")) == 1.0

    def test_extra_whitespace_is_ignored(self):
        scorer = SQuADExactMatchScorer()
        instance = Instance(question="Q", gold_answer="New York")

        assert scorer.score(instance, _output("  New   York  ")) == 1.0

    def test_partial_answer_does_not_match(self):
        """Unlike F1, exact match gives no partial credit."""
        scorer = SQuADExactMatchScorer()
        instance = Instance(question="Q", gold_answer="William Shakespeare")

        assert scorer.score(instance, _output("Shakespeare")) == 0.0

    def test_matches_any_of_multiple_references(self):
        scorer = SQuADExactMatchScorer()
        instance = Instance(
            question="Q",
            gold_answer="William Shakespeare",
            metadata={"all_answers": ["William Shakespeare", "Shakespeare", "the Bard"]},
        )

        assert scorer.score(instance, _output("Shakespeare")) == 1.0
        assert scorer.score(instance, _output("Bard")) == 1.0
        assert scorer.score(instance, _output("Marlowe")) == 0.0

    def test_falls_back_to_gold_answer_without_metadata(self):
        scorer = SQuADExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Paris", metadata={})

        assert scorer.score(instance, _output("paris")) == 1.0

    def test_none_extracted_answer(self):
        scorer = SQuADExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="Paris")
        output.extracted_answer = None

        assert scorer.score(instance, output) == 0.0

    def test_none_gold_answer_without_references(self):
        scorer = SQuADExactMatchScorer()
        instance = Instance(question="Q", gold_answer=None, metadata={})

        assert scorer.score(instance, _output("Paris")) == 0.0

    def test_name(self):
        """The scorer name becomes the sub-key under `accuracy` in metrics.json."""
        assert SQuADExactMatchScorer().name == "squad_exact_match"

    def test_agrees_with_f1_when_f1_is_perfect(self):
        """A perfect F1 implies exact match, since both share the normalization."""
        em = SQuADExactMatchScorer()
        f1 = SQuADF1Scorer()
        instance = Instance(question="Q", gold_answer="the Great Depression")
        output = _output("Great Depression.")

        assert f1.score(instance, output) == 1.0
        assert em.score(instance, output) == 1.0

    def test_diverges_from_f1_on_partial_overlap(self):
        """Partial overlap earns F1 credit but not exact match."""
        em = SQuADExactMatchScorer()
        f1 = SQuADF1Scorer()
        instance = Instance(question="Q", gold_answer="William Shakespeare")
        output = _output("Shakespeare")

        assert 0.0 < f1.score(instance, output) < 1.0
        assert em.score(instance, output) == 0.0
