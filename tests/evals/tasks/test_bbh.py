"""Tests for BIG-Bench Hard task logic."""

import pytest

from olmo_eval.common.types import RequestType
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.constants.bbh import (
    BBH_CHOICES,
    BBH_DESCRIPTIONS,
    BBH_FIXED_FEWSHOT,
)

_SUBTASKS = tuple(BBH_CHOICES)

#: Leaderboard v2 reformulates 24 of BIG-Bench Hard's 27 subtasks as multiple choice.
#: The three left out have no closed answer set to rank, and no bank row either.
_GENERATIVE_SUBTASKS = ("dyck_languages", "multistep_arithmetic_two", "word_sorting")


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


class TestBBHRegistration:
    def test_twenty_four_subtasks(self):
        assert len(_SUBTASKS) == 24

    @pytest.mark.parametrize("subtask", _SUBTASKS)
    def test_task_registered(self, subtask):
        assert f"bbh_{subtask}" in list_tasks()

    @pytest.mark.parametrize("subtask", _GENERATIVE_SUBTASKS)
    def test_generative_subtask_not_registered(self, subtask):
        assert f"bbh_{subtask}" not in list_tasks()

    @pytest.mark.parametrize("subtask", _SUBTASKS)
    def test_subtask_attribute_and_subset_agree(self, subtask):
        task = get_task(f"bbh_{subtask}")
        assert task.subtask == subtask
        assert task.config.data_source.subset == subtask

    @pytest.mark.parametrize("subtask", _SUBTASKS)
    def test_leaderboard_is_three_shot(self, subtask):
        assert get_task(f"bbh_{subtask}").config.num_fewshot == 3


class TestConstants:
    @pytest.mark.parametrize("subtask", _SUBTASKS)
    def test_every_subtask_has_all_three_pieces(self, subtask):
        assert BBH_DESCRIPTIONS[subtask]
        assert BBH_CHOICES[subtask]
        assert len(BBH_FIXED_FEWSHOT[subtask]) == 3

    @pytest.mark.parametrize("subtask", _SUBTASKS)
    def test_choices_are_distinct_non_empty_strings(self, subtask):
        choices = BBH_CHOICES[subtask]
        assert all(isinstance(c, str) and c for c in choices)
        assert len(set(choices)) == len(choices)

    @pytest.mark.parametrize("subtask", _SUBTASKS)
    def test_exemplar_answers_are_rankable(self, subtask):
        for example in BBH_FIXED_FEWSHOT[subtask]:
            assert example["input"]
            assert example["target"] in BBH_CHOICES[subtask]

    @pytest.mark.parametrize(
        ("subtask", "expected"),
        (
            ("boolean_expressions", ("False", "True")),
            ("web_of_lies", ("Yes", "No")),
            ("causal_judgement", ("Yes", "No")),
            ("formal_fallacies", ("valid", "invalid")),
            ("object_counting", tuple(str(n) for n in range(19))),
            ("penguins_in_a_table", ("(A)", "(B)", "(C)", "(D)", "(E)")),
            ("logical_deduction_three_objects", ("(A)", "(B)", "(C)")),
        ),
    )
    def test_choice_set_verbatim_from_lm_eval(self, subtask, expected):
        assert BBH_CHOICES[subtask] == expected


class TestProcessDoc:
    @pytest.fixture
    def task(self):
        return get_task("bbh_boolean_expressions")

    def test_basic_conversion(self, task):
        instance = task.process_doc({"input": "not ( True ) is", "target": "False"}, index=0)
        assert instance.question == "not ( True ) is"
        assert instance.choices == ("False", "True")
        assert instance.gold_answer == "False"
        assert instance.metadata["gold_idx"] == 0

    def test_metadata_records_position_and_subtask(self, task):
        instance = task.process_doc({"input": "True is", "target": "True"}, index=42)
        assert instance.metadata["index"] == 42
        assert instance.metadata["subtask"] == "boolean_expressions"

    def test_unlisted_target_is_unanswerable_not_dropped(self, task, caplog):
        with caplog.at_level("WARNING"):
            instance = task.process_doc({"input": "True is", "target": "Maybe"}, index=5)
        assert instance is not None
        assert instance.metadata["gold_idx"] is None
        assert "choice set" in caplog.text

    @pytest.mark.parametrize("subtask", _SUBTASKS)
    def test_choices_come_from_the_subtask(self, subtask):
        task = get_task(f"bbh_{subtask}")
        instance = task.process_doc({"input": "q", "target": BBH_CHOICES[subtask][0]}, index=0)
        assert instance.choices == BBH_CHOICES[subtask]


class TestFormatRequest:
    @pytest.fixture
    def task(self):
        return get_task("bbh_boolean_expressions")

    def test_prompt_matches_leaderboard_layout(self, task):
        instance = task.process_doc({"input": "not ( True ) and ( True ) is", "target": "False"})
        assert task.format_request(instance).prompt == (
            "Evaluate the result of a random Boolean expression."
            "Q: not ( ( not not True ) ) is\n"
            "A: False\n"
            "\n"
            "Q: True and False and not True and True is\n"
            "A: False\n"
            "\n"
            "Q: not not ( not ( False ) ) is\n"
            "A: True\n"
            "\n"
            "Q: not ( True ) and ( True ) is\n"
            "A:"
        )

    def test_description_runs_into_the_first_exemplar(self, task):
        # Upstream inserts no separator here. Adding one would look like a tidy-up and
        # would change every prompt the bank's difficulties were estimated against.
        instance = task.process_doc({"input": "q", "target": "False"})
        assert "expression.Q:" in task.format_request(instance).prompt

    @pytest.mark.parametrize("subtask", _SUBTASKS)
    def test_three_exemplars_in_fixed_order(self, subtask):
        task = get_task(f"bbh_{subtask}")
        fewshot = task.get_fewshot()
        assert [ex.question for ex in fewshot] == [ex["input"] for ex in BBH_FIXED_FEWSHOT[subtask]]

    @pytest.mark.parametrize("subtask", _SUBTASKS)
    def test_choice_texts_are_ranked(self, subtask):
        task = get_task(f"bbh_{subtask}")
        instance = task.process_doc({"input": "q", "target": BBH_CHOICES[subtask][0]})
        request = task.format_request(instance)
        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.continuations == tuple(f" {c}" for c in BBH_CHOICES[subtask])
