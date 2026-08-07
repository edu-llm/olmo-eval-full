"""Tests for MuSR task logic."""

import pytest

from olmo_eval.common.types import RequestType
from olmo_eval.evals.tasks.common import get_task, list_tasks


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


_TASKS = (
    "musr_murder_mysteries",
    "musr_object_placements",
    "musr_team_allocation",
)

_DOC = {
    "narrative": "Ana and Mackenzie were both at the bungee site.\nWinston took notes.",
    "question": "Who is the most likely murderer?",
    "choices": "['Mackenzie', 'Ana']",
    "answer_choice": "Ana",
    "answer_index": 1,
}


class TestMuSRRegistration:
    @pytest.mark.parametrize("task_name", _TASKS)
    def test_task_registered(self, task_name):
        assert task_name in list_tasks()

    @pytest.mark.parametrize("task_name", _TASKS)
    def test_get_task(self, task_name):
        task = get_task(task_name)
        assert task.config.name == task_name

    @pytest.mark.parametrize("task_name", _TASKS)
    def test_subtask_matches_task_name(self, task_name):
        assert get_task(task_name).subtask == task_name.removeprefix("musr_")

    @pytest.mark.parametrize("task_name", _TASKS)
    def test_leaderboard_is_zero_shot(self, task_name):
        assert get_task(task_name).config.num_fewshot == 0


class TestProcessDoc:
    @pytest.fixture
    def task(self):
        return get_task("musr_murder_mysteries")

    def test_choices_parsed_from_repr(self, task):
        instance = task.process_doc(_DOC, index=0)
        assert instance.choices == ("Mackenzie", "Ana")

    def test_question_carries_the_narrative(self, task):
        instance = task.process_doc(_DOC, index=0)
        assert instance.question.startswith(_DOC["narrative"])
        assert instance.question.endswith(_DOC["question"])

    def test_gold_resolves_to_the_answer_text(self, task):
        instance = task.process_doc(_DOC, index=0)
        gold_idx = instance.metadata["gold_idx"]
        assert instance.choices[gold_idx] == "Ana"
        assert instance.gold_answer == "Ana"
        assert instance.metadata["gold_text"] == "Ana"

    def test_metadata_records_position_and_subtask(self, task):
        instance = task.process_doc(_DOC, index=17)
        assert instance.metadata["index"] == 17
        assert instance.metadata["subtask"] == "murder_mysteries"

    def test_disagreeing_answer_index_is_unanswerable_not_dropped(self, task, caplog):
        doc = dict(_DOC, answer_index=0)
        with caplog.at_level("WARNING"):
            instance = task.process_doc(doc, index=3)
        assert instance is not None
        assert instance.metadata["gold_idx"] is None
        assert "answer_index" in caplog.text

    def test_answer_text_absent_from_choices_is_unanswerable(self, task, caplog):
        doc = dict(_DOC, answer_choice="Winston")
        with caplog.at_level("WARNING"):
            instance = task.process_doc(doc, index=4)
        assert instance is not None
        assert instance.metadata["gold_idx"] is None


class TestFormatRequest:
    @pytest.fixture
    def task(self):
        return get_task("musr_murder_mysteries")

    def test_prompt_matches_leaderboard_layout(self, task):
        instance = task.process_doc(_DOC, index=0)
        request = task.format_request(instance)
        assert request.prompt == (
            "Ana and Mackenzie were both at the bungee site.\n"
            "Winston took notes.\n"
            "\n"
            "Who is the most likely murderer?\n"
            "\n"
            "1 - Mackenzie\n"
            "2 - Ana\n"
            "\n"
            "Answer:"
        )

    def test_choice_texts_are_ranked_not_their_numbers(self, task):
        instance = task.process_doc(_DOC, index=0)
        request = task.format_request(instance)
        assert request.request_type == RequestType.LOGLIKELIHOOD
        assert request.continuations == (" Mackenzie", " Ana")
