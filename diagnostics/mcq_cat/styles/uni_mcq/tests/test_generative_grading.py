"""The generative grading scheme: extraction, prompting, and a full CAT session.

Nothing here touches a GPU or the network. The sampling backend is injected as a
``prompt -> completion`` callable, so these exercise the real extractor, the real stop
handling and the real frozen CAT engine, and only the tokens themselves are simulated.

The extraction cases are the load-bearing ones. The GSM8K bank was calibrated under
Open LLM Leaderboard strict matching, so a grader one comma or one trailing sentence
looser than that silently re-scales every theta it produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ....base import BenchmarkItem, ScoringModel
from ....common import cat_loop, generative
from .. import datasets, resolve
from ..style import UniMcqStyle
from .conftest import (
    GENERATIVE_DATASET,
    TOY_GENERATIVE_ITEMS,
    SimCompleter,
    make_spec,
    write_bank,
)

#: A stand-in few-shot block. The real one lives in olmo_eval, which a diagnostics-only
#: checkout need not have installed, so these pin the prompt's shape, not its content.
STUB_FEWSHOT = (
    {
        "question": "Two plus two?",
        "answer": "2 + 2 = 4. So the answer is 4.",
        "short_answer": "4",
    },
    {
        "question": "Three plus three?",
        "answer": "3 + 3 = 6. So the answer is 6.",
        "short_answer": "6",
    },
)


@pytest.fixture
def stub_fewshot(monkeypatch) -> tuple[dict[str, str], ...]:
    """Serve :data:`STUB_FEWSHOT` wherever the gsm8k few-shot block is asked for."""
    monkeypatch.setitem(generative.FEWSHOT_SOURCES, "gsm8k", lambda: STUB_FEWSHOT)
    return STUB_FEWSHOT


def make_item(question: str = "Q?", **metadata: object) -> BenchmarkItem:
    """A generative item: no choices, gold answer in metadata."""
    return BenchmarkItem(
        item_id="g0",
        question=question,
        choices=(),
        gold_index=-1,
        metadata={"gold_answer": "72", "answer_type": "numeric", **metadata},
    )


class TestAnswerExtraction:
    @pytest.mark.parametrize(
        ("completion", "expected"),
        [
            ("So the answer is 72.", "72"),
            ("She sold 1,234 clips. So the answer is 1,234.", "1234"),
            ("The balance fell. So the answer is -15.", "-15"),
            ("So the answer is 3.5.", "3.5"),
            ("So the answer is 72 apples in the basket.", "72"),
            ("18 + 54 = 72. So the answer is 72.", "72"),
            ("I do not know how to solve this.", None),
            ("", None),
        ],
    )
    def test_takes_the_last_number(self, completion: str, expected: str | None) -> None:
        assert generative.extract_last_number(completion) == expected

    def test_comma_separators_are_one_number_not_two(self) -> None:
        """Without the comma pass, "1,234" extracts as 234 and grades wrong."""
        assert generative.extract_last_number("Total: 1,234") == "1234"

    def test_gold_is_normalized_the_same_way(self) -> None:
        assert generative.clean_answer_text("1,234") == "1234"

    def test_gold_without_a_number_survives_unchanged(self) -> None:
        """``_clean_short_answer`` returns the text when there is nothing numeric."""
        assert generative.clean_answer_text("yes") == "yes"

    @pytest.mark.parametrize(
        ("completion", "gold", "correct"),
        [
            ("So the answer is 72.", "72", True),
            ("So the answer is 1,234.", "1234", True),
            ("So the answer is 72 apples.", "72", True),
            ("So the answer is 73.", "72", False),
            ("So the answer is 72.5.", "72", False),
            ("No idea.", "72", False),
        ],
    )
    def test_verdicts(self, completion: str, gold: str, correct: bool) -> None:
        verdict = generative.LastNumberExactMatch().grade(completion, gold)
        assert verdict.correct is correct
        assert verdict.gold == gold

    def test_a_completion_with_no_number_is_wrong_not_an_error(self) -> None:
        verdict = generative.LastNumberExactMatch().grade("I refuse.", "72")
        assert verdict.correct is False
        assert verdict.extracted is None

    def test_unknown_answer_type_names_what_is_available(self) -> None:
        with pytest.raises(ValueError, match="No answer grader for answer_type"):
            generative.get_answer_grader("free_text")

    def test_the_grader_table_is_the_extension_point(self, monkeypatch) -> None:
        """A new answer format is one entry here and no change to sampling."""

        class AlwaysRight:
            name = "always_right"

            def grade(self, completion: str, gold_answer: str) -> generative.Verdict:
                return generative.Verdict(correct=True, extracted=completion, gold=gold_answer)

        monkeypatch.setitem(generative.ANSWER_GRADERS, "free_text", AlwaysRight())
        graded = generative.grade_completion(
            make_item(answer_type="free_text"), "anything", generative.GenerationConfig()
        )
        assert graded.correct is True
        assert graded.metadata["grader"] == "always_right"


class TestParityWithOlmoEval:
    """The diagnostic must extract exactly what the olmo-eval gsm8k task extracts."""

    def test_extraction_matches_the_task_implementation(self) -> None:
        gsm8k = pytest.importorskip(
            "olmo_eval.evals.tasks.gsm8k",
            reason="olmo_eval is not importable in this environment",
        )
        cases = [
            "So the answer is 72.",
            "She sold 1,234 clips. So the answer is 1,234.",
            "The balance fell. So the answer is -15.",
            "So the answer is 3.5.",
            "18 + 54 = 72. So the answer is 72 apples.",
            "I do not know.",
        ]
        for text in cases:
            assert generative.extract_last_number(text) == gsm8k._extract_last_number(text)


class TestStopSequences:
    def test_cuts_at_the_earliest_stop(self) -> None:
        text = "So the answer is 72.\n\nQuestion: something else"
        assert generative.truncate_at_stop(text, ("Question:", "\n\n")) == "So the answer is 72."

    def test_a_run_on_answer_is_not_graded_on_the_next_question(self) -> None:
        """The extractor takes the last number, so truncation decides the grade."""
        run_on = " So the answer is 72.\n\nQuestion: how many is 999?\nAnswer: 999"
        graded = generative.grade_completion(make_item(), run_on, generative.GenerationConfig())
        assert graded.correct is True
        assert graded.metadata["extracted_answer"] == "72"

    def test_text_without_a_stop_is_left_alone(self) -> None:
        assert generative.truncate_at_stop("plain text", ("Question:",)) == "plain text"


class TestPrompt:
    def test_few_shot_block_precedes_the_question(self, stub_fewshot) -> None:
        config = generative.GenerationConfig(num_fewshot=2)
        prompt = generative.format_generative_prompt(make_item("How many?"), config)
        assert prompt.startswith("Question: Two plus two?\nAnswer: 2 + 2 = 4.")
        assert prompt.endswith("Question: How many?\nAnswer:")
        assert prompt.count("Question: ") == 3

    def test_num_fewshot_truncates_the_block(self, stub_fewshot) -> None:
        config = generative.GenerationConfig(num_fewshot=1)
        prompt = generative.format_generative_prompt(make_item(), config)
        assert prompt.count("Question: ") == 2

    def test_zero_shot_is_available_but_not_the_default(self, stub_fewshot) -> None:
        assert generative.GenerationConfig().num_fewshot == 8
        prompt = generative.format_generative_prompt(
            make_item("How many?"), generative.GenerationConfig(num_fewshot=0)
        )
        assert prompt == "Question: How many?\nAnswer:"

    def test_unknown_fewshot_source_raises(self) -> None:
        config = generative.GenerationConfig(fewshot_source="nope")
        with pytest.raises(ValueError, match="Unknown fewshot_source"):
            generative.fewshot_examples(config)


class TestGenerationConfig:
    def test_sampling_defaults_match_the_olmo_eval_task(self) -> None:
        config = generative.GenerationConfig()
        assert config.max_new_tokens == 512
        assert config.temperature == 0.0
        assert config.stop_sequences == ("Question:", "\n\n")
        assert config.num_fewshot == 8

    def test_a_nonzero_temperature_is_rejected_not_ignored(self) -> None:
        with pytest.raises(ValueError, match="temperature must be 0"):
            generative.GenerationConfig(temperature=0.7)

    def test_stop_sequences_from_yaml_lists_become_tuples(self) -> None:
        config = generative.GenerationConfig(stop_sequences=["Question:", "\n\n"])
        assert config.stop_sequences == ("Question:", "\n\n")


class TestItemResponseShape:
    def test_no_choice_index_and_no_logprobs(self) -> None:
        graded = generative.grade_completion(
            make_item(), " So the answer is 72.", generative.GenerationConfig()
        )
        assert graded.chosen_index == -1
        assert graded.choice_logprobs == ()

    def test_metadata_makes_the_run_auditable(self) -> None:
        config = generative.GenerationConfig(num_fewshot=8)
        graded = generative.grade_completion(make_item(), " ... So the answer is 73.", config)
        assert graded.correct is False
        assert graded.metadata == {
            "modality": "generative",
            "grader": "last_number_exact_match",
            "completion": " ... So the answer is 73.",
            "extracted_answer": "73",
            "gold_answer": "72",
            "num_fewshot": 8,
            "ungradable": False,
        }

    def test_an_item_without_a_gold_answer_scores_zero_and_says_so(self) -> None:
        """A fabricated zero must not be readable as the model having answered wrong."""
        bare = BenchmarkItem(item_id="g0", question="Q?", choices=(), gold_index=-1)
        graded = generative.grade_completion(bare, "72", generative.GenerationConfig())
        assert graded.correct is False
        assert graded.metadata["ungradable"] is True
        assert "gold_answer" in graded.metadata[generative.UNGRADABLE_REASON_KEY]

    def test_a_gradable_item_is_marked_gradable_rather_than_left_silent(self) -> None:
        """Absent and False have to differ, or an old report reads as a clean one."""
        graded = generative.grade_completion(
            make_item(), " So the answer is 72.", generative.GenerationConfig()
        )
        assert graded.metadata[generative.UNGRADABLE_KEY] is False
        assert generative.UNGRADABLE_REASON_KEY not in graded.metadata


class TestGenerativeScorer:
    def test_satisfies_the_frozen_scoring_model_protocol(self) -> None:
        scorer = generative.GenerativeScorer(lambda _: "", generative.GenerationConfig())
        assert isinstance(scorer, ScoringModel)

    def test_scores_every_item_in_order(self, toy_generative_items, stub_fewshot) -> None:
        config = generative.GenerationConfig(num_fewshot=1)
        scorer = generative.GenerativeScorer(lambda _: " So the answer is 42.", config)

        responses = scorer.score_items(toy_generative_items)

        assert [r.item_id for r in responses] == [i.item_id for i in toy_generative_items]
        # Gold answers run 40..44, so a constant 42 is right for toy_2 only.
        assert [r.correct for r in responses] == [False, False, True, False, False]

    def test_the_completer_sees_the_formatted_prompt(self, stub_fewshot) -> None:
        seen: list[str] = []

        def complete(prompt: str) -> str:
            seen.append(prompt)
            return " So the answer is 72."

        config = generative.GenerationConfig(num_fewshot=0)
        generative.GenerativeScorer(complete, config).score_items([make_item("How many?")])
        assert seen == ["Question: How many?\nAnswer:"]


class TestLazyHeavyImports:
    def test_importing_the_module_does_not_pull_in_torch(self) -> None:
        """A diagnostics-only checkout has no GPU stack, and must still import this."""
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parents[5]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import diagnostics.mcq_cat.common.generative; "
                "print('torch' in sys.modules, 'transformers' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False False"

    def test_an_unknown_checkpoint_kind_is_rejected_before_any_import(self) -> None:
        config = generative.GenerationConfig(checkpoint_kind="ollama")
        with pytest.raises(ValueError, match="Unknown checkpoint_kind"):
            generative.load_generative_model(Path("/nowhere"), config)

    def test_olmo_core_points_at_the_integration_point(self) -> None:
        config = generative.GenerationConfig(checkpoint_kind="olmo_core")
        with pytest.raises(NotImplementedError, match="GenerativeScorer"):
            generative.load_generative_model(Path("/nowhere"), config)


class FakeIds:
    """Just enough of a torch tensor for the completer: shape, ``.to``, row access."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = list(tokens)

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.tokens))

    def to(self, device: object) -> FakeIds:
        return self

    def __getitem__(self, index: int) -> list[str]:
        if index != 0:
            raise IndexError(index)
        return self.tokens


class FakeTokenizer:
    """Whitespace tokenizer standing in for a checkpoint's tokenizer."""

    pad_token_id = None
    eos_token_id = 7

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, FakeIds]:
        return {"input_ids": FakeIds(text.split(" "))}

    def decode(self, tokens: list[str], skip_special_tokens: bool = False) -> str:
        return " ".join(tokens)


class FakeModel:
    """Appends a canned answer to whatever it is prompted with."""

    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def eval(self) -> None:
        pass

    def generate(self, input_ids: FakeIds, **kwargs: object) -> FakeIds:
        self.calls.append(kwargs)
        return FakeIds([*input_ids.tokens, "So", "the", "answer", "is", "72."])


class TestHuggingFaceCompleter:
    """The sampling backend, with ``torch`` and ``transformers`` stubbed out."""

    @pytest.fixture
    def fake_stack(self, monkeypatch) -> FakeModel:
        import contextlib
        import sys
        import types

        model = FakeModel()

        torch = types.ModuleType("torch")
        torch.no_grad = contextlib.nullcontext
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: FakeTokenizer()
        )
        transformers.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *a, **k: model
        )
        transformers.set_seed = lambda seed: None

        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        return model

    def test_returns_the_continuation_without_the_prompt_echo(self, fake_stack) -> None:
        completer = generative._HFCompleter(
            Path("/ckpt"), generative.GenerationConfig(num_fewshot=0)
        )
        assert completer("Question: how many?") == "So the answer is 72."

    def test_decodes_greedily_within_the_configured_token_budget(self, fake_stack) -> None:
        config = generative.GenerationConfig(num_fewshot=0, max_new_tokens=256)
        generative._HFCompleter(Path("/ckpt"), config)("Question: how many?")

        kwargs = fake_stack.calls[0]
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 256
        assert kwargs["pad_token_id"] == FakeTokenizer.eos_token_id

    def test_the_loader_wraps_the_completer_in_a_scorer(self, fake_stack) -> None:
        config = generative.GenerationConfig(num_fewshot=0)
        model = generative.load_generative_model(Path("/ckpt"), config)

        assert isinstance(model, generative.GenerativeScorer)
        responses = model.score_items([make_item()])
        assert responses[0].correct is True
        assert responses[0].metadata["extracted_answer"] == "72"


@pytest.fixture
def generative_style(monkeypatch, tmp_path: Path) -> UniMcqStyle:
    """A style pointed at a synthetic five-item generative bank.

    The spec is registered directly rather than borrowing a shipped dataset entry, so
    this does not wait on gsm8k being vendored.
    """
    monkeypatch.setitem(
        datasets.SUPPORTED, GENERATIVE_DATASET, make_spec(GENERATIVE_DATASET, modality="generative")
    )
    monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
    write_bank(tmp_path, dataset=GENERATIVE_DATASET, modality="generative")
    return UniMcqStyle()


class TestGenerativeCatSession:
    """A generative bank driven end to end by the unchanged frozen CAT engine."""

    def test_runs_to_exhaustion_and_produces_a_sane_theta(
        self, generative_style, toy_params, stub_fewshot
    ) -> None:
        bank = generative_style.download_benchmark(GENERATIVE_DATASET)
        irt = generative_style.load_irt_params(GENERATIVE_DATASET)
        completer = SimCompleter(0.8, toy_params, TOY_GENERATIVE_ITEMS)

        report = cat_loop.run_cat(
            generative_style,
            bank=bank,
            irt_bank=irt,
            model=generative.GenerativeScorer(
                completer, generative.GenerationConfig(num_fewshot=2)
            ),
            se_threshold=0.3,
            max_items=40,
        )

        assert report.num_items_administered == 5
        assert report.metadata["stop_reason"] == "bank_exhausted"
        assert completer.calls == [r.item_id for r in report.responses]
        assert np.isfinite(report.metadata["theta"])
        assert -4.0 <= report.metadata["theta"] <= 4.0
        assert report.metadata["standard_error"] < 1.0
        assert 0.0 <= report.metadata["pirt_accuracy"] <= 1.0

    def test_the_report_records_the_generative_scoring_convention(
        self, generative_style, toy_params, stub_fewshot
    ) -> None:
        """A theta is not readable without the convention that produced it."""
        bank = generative_style.download_benchmark(GENERATIVE_DATASET)
        irt = generative_style.load_irt_params(GENERATIVE_DATASET)
        report = cat_loop.run_cat(
            generative_style,
            bank=bank,
            irt_bank=irt,
            model=generative.GenerativeScorer(
                SimCompleter(0.8, toy_params, TOY_GENERATIVE_ITEMS),
                generative.GenerationConfig(num_fewshot=2),
            ),
            se_threshold=0.3,
            max_items=40,
        )
        assert report.metadata["modality"] == "generative"
        assert "greedy sampled completion" in report.metadata["scoring_note"]
        assert "log-likelihood" not in report.metadata["scoring_note"]
        # The clause is the grader's, not the modality's: a note describing last-number
        # matching on a bank graded symbolically would misstate what produced the theta.
        assert "last number in the completion" in report.metadata["scoring_note"]
        assert "symbolic equivalence" not in report.metadata["scoring_note"]

    def test_responses_carry_the_generative_shape(
        self, generative_style, toy_params, stub_fewshot
    ) -> None:
        bank = generative_style.download_benchmark(GENERATIVE_DATASET)
        irt = generative_style.load_irt_params(GENERATIVE_DATASET)
        report = cat_loop.run_cat(
            generative_style,
            bank=bank,
            irt_bank=irt,
            model=generative.GenerativeScorer(
                SimCompleter(0.8, toy_params, TOY_GENERATIVE_ITEMS),
                generative.GenerationConfig(num_fewshot=2),
            ),
            se_threshold=0.3,
            max_items=40,
        )

        for response in report.responses:
            assert response.chosen_index == -1
            assert response.choice_logprobs == ()
            assert response.metadata["grader"] == "last_number_exact_match"
            assert response.metadata["completion"].endswith(".")

    def test_the_report_is_json_serializable(
        self, generative_style, toy_params, stub_fewshot
    ) -> None:
        """``chosen_index = -1`` and empty logprobs must survive the runner's writer."""
        bank = generative_style.download_benchmark(GENERATIVE_DATASET)
        irt = generative_style.load_irt_params(GENERATIVE_DATASET)
        report = cat_loop.run_cat(
            generative_style,
            bank=bank,
            irt_bank=irt,
            model=generative.GenerativeScorer(
                SimCompleter(0.8, toy_params, TOY_GENERATIVE_ITEMS),
                generative.GenerationConfig(num_fewshot=2),
            ),
            se_threshold=0.3,
            max_items=40,
        )
        payload = json.loads(json.dumps(report.to_dict()))
        assert all(r["chosen_index"] == -1 for r in payload["responses"])
        assert all(r["choice_logprobs"] == [] for r in payload["responses"])

    def test_ability_recovers_a_known_theta_on_a_larger_generative_bank(
        self, monkeypatch, tmp_path: Path, stub_fewshot
    ) -> None:
        """Five items cannot pin an ability, so recovery needs a bank worth measuring on."""
        params = []
        items = []
        for i in range(60):
            params.append(
                {
                    "item_id": f"gen_{i}",
                    "difficulty": -3.0 + 6.0 * i / 59,
                    "discrimination": 1.8,
                    "guessing": 0.0,
                }
            )
            items.append(
                {
                    "id": f"gen_{i}",
                    "question": f"Word problem {i}?",
                    "choices": [],
                    "gold_index": -1,
                    "metadata": {
                        "gold_answer": str(100 + i),
                        "answer_type": "numeric",
                        "modality": "generative",
                    },
                }
            )
        bank_dir = tmp_path / GENERATIVE_DATASET
        bank_dir.mkdir(parents=True)
        (bank_dir / "params.json").write_text(json.dumps(params), encoding="utf-8")
        (bank_dir / "items.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in items), encoding="utf-8"
        )
        (bank_dir / "manifest.json").write_text(
            json.dumps({"fit_family": "2pl", "items": 60, "modality": "generative"}),
            encoding="utf-8",
        )
        monkeypatch.setitem(
            datasets.SUPPORTED,
            GENERATIVE_DATASET,
            make_spec(GENERATIVE_DATASET, modality="generative"),
        )
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        style = UniMcqStyle()
        bank = style.download_benchmark(GENERATIVE_DATASET)
        irt = style.load_irt_params(GENERATIVE_DATASET)
        lookup = {
            p["item_id"]: (p["discrimination"], p["difficulty"], p["guessing"]) for p in params
        }

        true_theta = 0.75
        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt,
            model=generative.GenerativeScorer(
                SimCompleter(true_theta, lookup, items, seed=7),
                generative.GenerationConfig(num_fewshot=2),
            ),
            se_threshold=0.3,
            max_items=40,
        )

        assert report.metadata["theta"] == pytest.approx(true_theta, abs=0.6)
        assert report.metadata["standard_error"] <= 0.3
        assert report.metadata["stop_reason"] == "precision_reached"
