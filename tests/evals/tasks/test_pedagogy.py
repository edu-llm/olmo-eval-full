"""Tests for Pedagogy task logic.

Offline throughout: the dataset is gated, so every test drives ``process_doc`` with
hand-built rows rather than fetching. The rendering assertions are the load-bearing
ones — they pin the prompt, the scored spans and the gold index to the shape the
``pedagogy`` CAT item bank was calibrated behind, so a well-meaning reformat of this
task fails here instead of silently invalidating the bank's difficulties.
"""

import pytest

from olmo_eval.common.types import Instance, RequestType
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.pedagogy import PEDAGOGY_REVISION


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


def _doc(**overrides):
    """A four-option row shaped like ``cdpk_main``, with all seven option columns."""
    doc = {
        "question_id": 447,
        "question": "What prior knowledge must students necessarily have to begin the "
        "study of Harmony?",
        "answer_a": "Intervals",
        "answer_b": "Notes of the scale",
        "answer_c": "Main functions",
        "answer_d": "Major and minor scales",
        "answer_e": None,
        "answer_f": None,
        "answer_g": None,
        "correct_answer": "a",
    }
    doc.update(overrides)
    return doc


class TestPedagogyRegistration:
    """Tests for Pedagogy task registration."""

    @pytest.mark.parametrize(
        "task_name", ["pedagogy", "pedagogy:rc", "pedagogy:mc", "pedagogy:bpb", "pedagogy:full"]
    )
    def test_task_registered(self, task_name):
        base = task_name.split(":")[0]
        assert base in list_tasks()

    @pytest.mark.parametrize(
        "task_name", ["pedagogy", "pedagogy:rc", "pedagogy:mc", "pedagogy:bpb", "pedagogy:full"]
    )
    def test_get_task(self, task_name):
        assert get_task(task_name) is not None

    def test_data_source_is_pinned(self):
        """The revision pin is the whole defence for a bank whose ids are row indices."""
        source = get_task("pedagogy").config.data_source
        assert isinstance(source, DataSource)
        assert source.path == "AI-for-Education/pedagogy-benchmark"
        assert source.subset == "cdpk_main"
        assert source.split == "train"
        assert source.revision == PEDAGOGY_REVISION

    def test_zero_shot(self):
        """Difficulties were estimated 0-shot; exemplars would change what is asked."""
        assert get_task("pedagogy").config.num_fewshot == 0


class TestProcessDoc:
    """Tests for Pedagogy.process_doc."""

    @pytest.fixture
    def task(self):
        return get_task("pedagogy")

    def test_basic_conversion(self, task):
        instance = task.process_doc(_doc(), index=0)

        assert instance is not None
        assert instance.question.startswith("What prior knowledge")
        assert instance.choices == (
            "Intervals",
            "Notes of the scale",
            "Main functions",
            "Major and minor scales",
        )
        assert instance.metadata["gold_idx"] == 0
        assert instance.metadata["gold_text"] == "Intervals"

    def test_item_id_is_the_calibrated_qid(self, task):
        instance = task.process_doc(_doc(), index=13)
        assert instance.metadata["id"] == "pedagogy_447"
        assert instance.metadata["index"] == 13

    def test_item_id_falls_back_to_position(self, task):
        """A row with no ``question_id`` still gets an id, matching the Aug-1 loader."""
        instance = task.process_doc(_doc(question_id=None), index=7)
        assert instance.metadata["id"] == "pedagogy_00007"

    def test_unused_option_columns_are_dropped(self, task):
        instance = task.process_doc(_doc(answer_e="", answer_f="None", answer_g=None), index=0)
        assert len(instance.choices) == 4
        assert instance.metadata["option_letters"] == ("a", "b", "c", "d")

    def test_seven_options_are_allowed(self, task):
        instance = task.process_doc(_doc(answer_e="E", answer_f="F", answer_g="G"), index=0)
        assert len(instance.choices) == 7
        assert instance.metadata["num_choices"] == 7

    def test_gold_is_an_index_among_survivors(self, task):
        """With ``answer_b`` empty, the key ``"c"`` is choice 1 — not choice 2."""
        instance = task.process_doc(_doc(answer_b=None, correct_answer="c"), index=0)

        assert instance.choices == ("Intervals", "Main functions", "Major and minor scales")
        assert instance.metadata["gold_idx"] == 1
        assert instance.metadata["gold_text"] == "Main functions"
        assert instance.metadata["option_letters"] == ("a", "c", "d")

    def test_gold_letter_for_a_dropped_column_falls_back(self, task):
        """A key naming a null column is mis-keyed upstream; the fallback reads it
        positionally, as the calibrating loader did."""
        instance = task.process_doc(_doc(answer_b=None, correct_answer="b"), index=0)
        assert instance.metadata["gold_idx"] == 1

    def test_gold_stated_as_a_digit(self, task):
        instance = task.process_doc(_doc(correct_answer="2"), index=0)
        assert instance.metadata["gold_idx"] == 2

    def test_gold_stated_as_option_text(self, task):
        instance = task.process_doc(_doc(correct_answer="Main functions"), index=0)
        assert instance.metadata["gold_idx"] == 2

    def test_uppercase_letter_key(self, task):
        instance = task.process_doc(_doc(correct_answer="D"), index=0)
        assert instance.metadata["gold_idx"] == 3

    def test_whitespace_is_stripped(self, task):
        instance = task.process_doc(
            _doc(question="  Padded stem  ", answer_a="  Intervals\n"), index=0
        )
        assert instance.question == "Padded stem"
        assert instance.choices[0] == "Intervals"

    def test_skip_missing_stem(self, task):
        assert task.process_doc(_doc(question=None), index=0) is None

    def test_skip_missing_answer(self, task):
        assert task.process_doc(_doc(correct_answer=None), index=0) is None

    def test_skip_single_option(self, task):
        doc = _doc(answer_b=None, answer_c=None, answer_d=None)
        assert task.process_doc(doc, index=0) is None

    def test_skip_unresolvable_gold(self, task):
        assert task.process_doc(_doc(correct_answer="zz"), index=0) is None

    def test_alternate_column_names(self, task):
        """The loader accepted ``Question`` / ``answer`` spellings; so does this."""
        doc = _doc(question=None, correct_answer=None)
        doc["Question"] = "Capitalised stem?"
        doc["answer"] = "b"
        instance = task.process_doc(doc, index=0)
        assert instance.question == "Capitalised stem?"
        assert instance.metadata["gold_idx"] == 1


class TestFormatRequest:
    """Tests that the rendered request matches the calibrated snapshot exactly."""

    def test_rc_matches_the_calibrated_rendering(self):
        """Byte-for-byte against ``pedagogy_447`` in the 2026-08-01 mcq_cache snapshot."""
        task = get_task("pedagogy")
        request = task.format_request(task.process_doc(_doc(), index=0))

        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.prompt == (
            "Question: What prior knowledge must students necessarily have to begin "
            "the study of Harmony?\nAnswer:"
        )
        assert request.continuations == (
            " Intervals",
            " Notes of the scale",
            " Main functions",
            " Major and minor scales",
        )
        assert request.continuation_prompts is None

    def test_rc_scores_one_continuation_per_option(self):
        task = get_task("pedagogy")
        instance = Instance(
            question="Q?",
            choices=("one", "two", "three", "four", "five", "six", "seven"),
            gold_answer="one",
            metadata={"gold_idx": 0, "gold_text": "one"},
        )
        request = task.format_request(instance)
        assert request.continuations == (
            " one",
            " two",
            " three",
            " four",
            " five",
            " six",
            " seven",
        )

    def test_mc_lists_options_and_scores_letters(self):
        task = get_task("pedagogy:mc")
        request = task.format_request(task.process_doc(_doc(), index=0))

        assert request.prompt.endswith(" D. Major and minor scales\nAnswer:")
        assert request.continuations == (" A", " B", " C", " D")

    def test_mc_letters_run_past_d(self):
        """Seven options mean letters through G, which a four-choice assumption misses."""
        task = get_task("pedagogy:mc")
        instance = Instance(
            question="Q?",
            choices=tuple("abcdefg"),
            gold_answer="a",
            metadata={"gold_idx": 0, "gold_text": "a"},
        )
        assert task.format_request(instance).continuations == (
            " A",
            " B",
            " C",
            " D",
            " E",
            " F",
            " G",
        )
