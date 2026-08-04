"""Tests for olmo_eval.core.scorers module."""

from olmo_eval.common.scorers import (
    ContainmentScorer,
    ExactMatchScorer,
    MultipleChoiceScorer,
    SQuADExactMatchScorer,
    SQuADF1Scorer,
    WindowedContainmentScorer,
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


class TestContainmentScorer:
    """Tests for ContainmentScorer."""

    def test_bare_answer_matches(self):
        scorer = ContainmentScorer()
        instance = Instance(question="Q", gold_answer="Paris")

        assert scorer.score(instance, _output("Paris")) == 1.0

    def test_answer_inside_a_sentence_matches(self):
        """The reason this scorer exists: exact match reads this as wrong."""
        scorer = ContainmentScorer()
        instance = Instance(question="Q", gold_answer="Paris")
        output = _output("The capital of France is Paris.")

        assert scorer.score(instance, output) == 1.0
        assert SQuADExactMatchScorer().score(instance, output) == 0.0

    def test_wrong_answer_does_not_match(self):
        scorer = ContainmentScorer()
        instance = Instance(question="Q", gold_answer="Paris")

        assert scorer.score(instance, _output("The capital of France is Lyon.")) == 0.0

    def test_short_aliases_match_inside_longer_words(self):
        """A documented false positive of the published rule, kept on purpose.

        PopQA lists 'pol' as an alias for politician, and the paper's metric is
        "any substring of the prediction is an exact match of any of the gold
        answers" -- so 'policy' contains 'pol' and scores correct. Tightening
        this to token boundaries would depart from published numbers, so the
        behavior is asserted rather than fixed.
        """
        scorer = ContainmentScorer()
        instance = Instance(
            question="Q", gold_answer="politician", metadata={"all_answers": ["politician", "pol"]}
        )

        assert scorer.score(instance, _output("He works in policy")) == 1.0
        assert scorer.score(instance, _output("He is a politician")) == 1.0

    def test_multi_token_answer_must_be_contiguous(self):
        scorer = ContainmentScorer()
        instance = Instance(question="Q", gold_answer="New York")

        assert scorer.score(instance, _output("She lives in New York today")) == 1.0
        assert scorer.score(instance, _output("New Jersey and York")) == 0.0

    def test_matches_any_of_multiple_references(self):
        scorer = ContainmentScorer()
        instance = Instance(
            question="Q",
            gold_answer="journalist",
            metadata={"all_answers": ["journalist", "journo", "journalists"]},
        )

        assert scorer.score(instance, _output("He was a journo by trade")) == 1.0
        assert scorer.score(instance, _output("He was a chemist")) == 0.0

    def test_normalization_is_shared_with_squad_scorers(self):
        scorer = ContainmentScorer()
        instance = Instance(question="Q", gold_answer="Washington, D.C.")

        assert scorer.score(instance, _output("It is washington dc, I think")) == 1.0

    def test_is_more_lenient_than_exact_match_never_stricter(self):
        """Anything exact match accepts, containment must accept too."""
        containment = ContainmentScorer()
        em = SQuADExactMatchScorer()
        instance = Instance(
            question="Q",
            gold_answer="Tokyo",
            metadata={"all_answers": ["Tokyo", "Tokio"]},
        )
        for text in ["Tokyo", "tokyo", "  Tokyo  ", "Tokio"]:
            output = _output(text)
            if em.score(instance, output) == 1.0:
                assert containment.score(instance, output) == 1.0

    def test_hedging_scores_correct_which_is_why_it_needs_a_partner(self):
        """Listing candidates beats containment; exact match is what catches it."""
        instance = Instance(question="Q", gold_answer="Paris")
        output = _output("Paris, London, or Rome")

        assert ContainmentScorer().score(instance, output) == 1.0
        assert SQuADExactMatchScorer().score(instance, output) == 0.0

    def test_falls_back_to_gold_answer_without_metadata(self):
        scorer = ContainmentScorer()
        instance = Instance(question="Q", gold_answer="Paris", metadata={})

        assert scorer.score(instance, _output("paris")) == 1.0

    def test_none_extracted_answer(self):
        scorer = ContainmentScorer()
        instance = Instance(question="Q", gold_answer="Paris")
        output = LMOutput(text="Paris")
        output.extracted_answer = None

        assert scorer.score(instance, output) == 0.0

    def test_empty_generation(self):
        scorer = ContainmentScorer()
        instance = Instance(question="Q", gold_answer="Paris")

        assert scorer.score(instance, _output("")) == 0.0

    def test_none_gold_answer_without_references(self):
        scorer = ContainmentScorer()
        instance = Instance(question="Q", gold_answer=None, metadata={})

        assert scorer.score(instance, _output("Paris")) == 0.0

    def test_name(self):
        """The scorer name becomes the sub-key under `accuracy` in metrics.json."""
        assert ContainmentScorer().name == "containment"


class TestWindowedContainmentScorer:
    """Tests for WindowedContainmentScorer (Co-LMLM's published TriviaQA rule)."""

    def test_bare_answer_matches(self):
        scorer = WindowedContainmentScorer()
        instance = Instance(question="Q", gold_answer="Sinclair Lewis")

        assert scorer.score(instance, _output("Sinclair Lewis")) == 1.0

    def test_answer_inside_a_sentence_matches(self):
        scorer = WindowedContainmentScorer()
        instance = Instance(question="Q", gold_answer="Sinclair Lewis")

        assert scorer.score(instance, _output("Sinclair Lewis, the novelist.")) == 1.0

    def test_wrong_answer_does_not_match(self):
        scorer = WindowedContainmentScorer()
        instance = Instance(question="Q", gold_answer="Sinclair Lewis")

        assert scorer.score(instance, _output("Upton Sinclair")) == 0.0

    def test_matching_is_case_insensitive(self):
        scorer = WindowedContainmentScorer()
        instance = Instance(question="Q", gold_answer="Sinclair Lewis")

        assert scorer.score(instance, _output("sinclair lewis")) == 1.0

    def test_answer_beyond_the_window_does_not_count(self):
        """The paper searches only the first 100 characters of the output."""
        scorer = WindowedContainmentScorer()
        instance = Instance(question="Q", gold_answer="Sinclair Lewis")
        preamble = "I think the answer to this trivia question is probably going to be " * 2

        assert scorer.score(instance, _output(preamble + "Sinclair Lewis")) == 0.0
        assert scorer.score(instance, _output("x" * 80 + " Sinclair Lewis")) == 1.0

    def test_window_is_configurable(self):
        instance = Instance(question="Q", gold_answer="Sinclair Lewis")
        preamble = "word " * 40

        assert WindowedContainmentScorer(window_chars=10_000).score(
            instance, _output(preamble + "Sinclair Lewis")
        ) == 1.0

    def test_only_normalization_is_lowercasing(self):
        """Unlike the SQuAD family, articles and punctuation are left alone."""
        windowed = WindowedContainmentScorer()
        squad = ContainmentScorer()
        instance = Instance(question="Q", gold_answer="the Beatles")

        assert windowed.score(instance, _output("Beatles")) == 0.0
        assert squad.score(instance, _output("Beatles")) == 1.0
        assert windowed.score(instance, _output("the Beatles")) == 1.0

    def test_matches_any_of_multiple_references(self):
        scorer = WindowedContainmentScorer()
        instance = Instance(
            question="Q",
            gold_answer="Sinclair Lewis",
            metadata={"all_answers": ["Sinclair Lewis", "Harry Sinclair Lewis"]},
        )

        assert scorer.score(instance, _output("Harry Sinclair Lewis")) == 1.0
        assert scorer.score(instance, _output("Ernest Hemingway")) == 0.0

    def test_blank_reference_does_not_match_everything(self):
        scorer = WindowedContainmentScorer()
        instance = Instance(question="Q", gold_answer="x", metadata={"all_answers": ["", "  "]})

        assert scorer.score(instance, _output("anything at all")) == 0.0

    def test_scores_at_least_as_high_as_exact_match(self):
        scorer = WindowedContainmentScorer()
        em = SQuADExactMatchScorer()
        instance = Instance(question="Q", gold_answer="Sinclair Lewis")
        output = _output("Sinclair Lewis")

        assert em.score(instance, output) == 1.0
        assert scorer.score(instance, output) == 1.0

    def test_none_extracted_answer(self):
        scorer = WindowedContainmentScorer()
        instance = Instance(question="Q", gold_answer="Sinclair Lewis")
        output = LMOutput(text="Sinclair Lewis")
        output.extracted_answer = None

        assert scorer.score(instance, output) == 0.0

    def test_empty_generation(self):
        scorer = WindowedContainmentScorer()
        instance = Instance(question="Q", gold_answer="Sinclair Lewis")

        assert scorer.score(instance, _output("")) == 0.0

    def test_name_and_default_window(self):
        scorer = WindowedContainmentScorer()

        assert scorer.name == "windowed_containment"
        assert scorer.window_chars == 100
