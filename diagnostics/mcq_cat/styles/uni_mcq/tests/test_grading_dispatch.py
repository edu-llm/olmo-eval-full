"""Modality dispatch, the mismatch guard, and the MCQ log-probability regression.

Two things are pinned here. First, that naming a dataset picks the right grader all
the way from ``--benchmark NAME`` down to the constructed ``ScoringModel``. Second,
and more important, that the wrong pairing cannot happen quietly: an empty choice set
in the log-likelihood scorer, or an absent gold answer in the generative one, produces
a run that finishes cleanly and reports a theta computed from nothing.

The MCQ regression tests are the "still graded by log probability" check the branch
owes: they assert that ``arc_challenge`` and ``hellaswag`` reach the continuation
scorer and that the per-choice log-probabilities are on the response.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from .... import runner
from ....base import BenchmarkItem, ItemResponse, ScoringModel
from ....common import cat_loop, generative, grading, inference, s3_io
from .. import datasets, resolve
from ..style import UniMcqStyle
from .conftest import (
    GENERATIVE_DATASET,
    SimScorer,
    make_spec,
    stage_hf_checkpoint,
    unblock,
    write_bank,
)


class FakeLogprobModel:
    """A stand-in checkpoint whose continuation log-probabilities are known.

    Scores each choice by its length so the argmax is predictable, and otherwise
    reproduces what ``_HFScoringModel.score_items`` does: sum the continuation's token
    log-probabilities, take the argmax, compare it to ``gold_index``.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    def score_items(self, items):
        responses = []
        for item in items:
            self.seen.append(item.item_id)
            logprobs = tuple(
                -1.0 * (len(choice) + index) for index, choice in enumerate(item.choices)
            )
            chosen = max(range(len(logprobs)), key=lambda i: logprobs[i])
            responses.append(
                ItemResponse(
                    item_id=item.item_id,
                    chosen_index=chosen,
                    correct=chosen == item.gold_index,
                    choice_logprobs=logprobs,
                )
            )
        return responses


def mcq_item(item_id: str = "m0") -> BenchmarkItem:
    return BenchmarkItem(item_id=item_id, question="Q?", choices=("alpha", "beta"), gold_index=1)


def generative_item(item_id: str = "g0", **metadata: object) -> BenchmarkItem:
    return BenchmarkItem(
        item_id=item_id,
        question="Q?",
        choices=(),
        gold_index=-1,
        metadata={"gold_answer": "72", "answer_type": "numeric", **metadata},
    )


class TestTheGraderTable:
    def test_holds_exactly_the_two_modalities(self) -> None:
        assert set(grading.GRADERS) == {"mcq", "generative"}

    def test_agrees_with_the_allowlist_vocabulary(self) -> None:
        """``datasets.MODALITIES`` is the shared contract; the table must cover it."""
        assert set(grading.GRADERS) == set(datasets.MODALITIES)

    def test_each_entry_describes_how_it_grades(self) -> None:
        assert "log-likelihood" in grading.GRADERS["mcq"].summary
        assert "sampled completion" in grading.GRADERS["generative"].summary

    def test_an_unknown_modality_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="Unknown modality 'audio'"):
            grading.get_grader("audio")

    def test_mcq_loads_the_log_likelihood_scorer(self, monkeypatch, tmp_path: Path) -> None:
        built: list[str] = []
        monkeypatch.setattr(
            inference,
            "load_scoring_model",
            lambda *a, **k: built.append("mcq") or FakeLogprobModel(),
        )
        request = grading.GradingRequest(dataset="arc_challenge", modality="mcq")
        grading.load_grader(request, tmp_path, grading.GradingSettings())
        assert built == ["mcq"]

    def test_generative_loads_the_sampling_scorer(self, monkeypatch, tmp_path: Path) -> None:
        built: list[str] = []
        monkeypatch.setattr(
            generative,
            "load_generative_model",
            lambda *a, **k: (
                built.append("generative")
                or generative.GenerativeScorer(lambda _: "", generative.GenerationConfig())
            ),
        )
        request = grading.GradingRequest(dataset="gsm8k", modality="generative")
        grading.load_grader(request, tmp_path, grading.GradingSettings())
        assert built == ["generative"]


class TestGenerationOverrides:
    def test_style_settings_reach_the_generation_config(self, monkeypatch, tmp_path: Path) -> None:
        seen: list[generative.GenerationConfig] = []
        monkeypatch.setattr(
            generative,
            "load_generative_model",
            lambda _dir, config: (
                seen.append(config) or generative.GenerativeScorer(lambda _: "", config)
            ),
        )
        request = grading.GradingRequest(
            dataset="gsm8k",
            modality="generative",
            generation={"num_fewshot": 5, "max_new_tokens": 256},
        )
        grading.load_grader(request, tmp_path, grading.GradingSettings())

        assert seen[0].num_fewshot == 5
        assert seen[0].max_new_tokens == 256
        assert seen[0].stop_sequences == ("Question:", "\n\n")

    def test_a_misspelled_setting_raises_rather_than_being_ignored(self, tmp_path: Path) -> None:
        request = grading.GradingRequest(
            dataset="gsm8k", modality="generative", generation={"num_few_shot": 5}
        )
        with pytest.raises(ValueError, match="Unknown generation settings"):
            grading.load_grader(request, tmp_path, grading.GradingSettings())

    def test_mcq_banks_carry_no_generation_settings(self, monkeypatch, tmp_path: Path) -> None:
        write_bank(tmp_path, dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        style = UniMcqStyle()
        style.download_benchmark("arc_challenge")
        assert style.grading_request().generation == {}

    def test_config_yaml_pins_the_calibration_convention(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setitem(
            datasets.SUPPORTED,
            GENERATIVE_DATASET,
            make_spec(GENERATIVE_DATASET, modality="generative"),
        )
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        write_bank(tmp_path, dataset=GENERATIVE_DATASET, modality="generative")

        style = UniMcqStyle()
        style.download_benchmark(GENERATIVE_DATASET)
        request = style.grading_request()

        assert request.modality == "generative"
        assert request.generation["num_fewshot"] == 8
        assert request.generation["max_new_tokens"] == 512
        assert request.generation["stop_sequences"] == ["Question:", "\n\n"]


class TestTheMismatchGuard:
    def test_generative_items_are_refused_by_the_mcq_declaration(self) -> None:
        request = grading.GradingRequest(dataset="gsm8k", modality="mcq")
        with pytest.raises(grading.ModalityMismatch) as excinfo:
            grading.check_bank_modality(request, [generative_item("g0"), generative_item("g1")])

        message = str(excinfo.value)
        assert "gsm8k" in message
        assert "'mcq'" in message
        assert "log-likelihood" in message
        assert "g0" in message

    def test_mcq_items_are_refused_by_the_generative_declaration(self) -> None:
        request = grading.GradingRequest(dataset="arc_challenge", modality="generative")
        with pytest.raises(grading.ModalityMismatch) as excinfo:
            grading.check_bank_modality(request, [mcq_item("m0")])

        message = str(excinfo.value)
        assert "arc_challenge" in message
        assert "'generative'" in message
        assert "sampled completion" in message
        assert "m0" in message

    def test_a_generative_item_missing_its_gold_answer_is_caught(self) -> None:
        bare = BenchmarkItem(item_id="g0", question="Q?", choices=(), gold_index=-1)
        request = grading.GradingRequest(dataset="gsm8k", modality="generative")
        with pytest.raises(grading.ModalityMismatch):
            grading.check_bank_modality(request, [bare])

    def test_matching_banks_pass_quietly(self) -> None:
        grading.check_bank_modality(
            grading.GradingRequest(dataset="arc_challenge", modality="mcq"), [mcq_item()]
        )
        grading.check_bank_modality(
            grading.GradingRequest(dataset="gsm8k", modality="generative"), [generative_item()]
        )

    def test_the_mcq_scorer_itself_refuses_a_choiceless_item(self) -> None:
        """The guard of last resort: an empty argmax would otherwise be the error."""
        scorer = inference._HFScoringModel.__new__(inference._HFScoringModel)
        scorer.config = inference.InferenceConfig()
        with pytest.raises(ValueError, match="no answer choices"):
            scorer.score_items([generative_item()])

    def test_the_generative_scorer_itself_refuses_an_mcq_item(self) -> None:
        scorer = generative.GenerativeScorer(lambda _: "72", generative.GenerationConfig())
        with pytest.raises(ValueError, match="answer choices"):
            scorer.score_items([mcq_item()])


class TestStyleDeclaresItsModality:
    def test_mcq_banks_declare_mcq(self, monkeypatch, tmp_path: Path) -> None:
        write_bank(tmp_path, dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
        style = UniMcqStyle()
        style.download_benchmark("arc_challenge")
        assert style.bank_modality() == "mcq"
        assert style.grading_request().dataset == "arc_challenge"

    def test_a_bank_with_no_modality_key_falls_back_to_the_allowlist(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Banks vendored before modality existed are all MCQ, and stay runnable."""
        bank_dir = write_bank(tmp_path, dataset="arc_challenge")
        manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
        del manifest["modality"]
        (bank_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        style = UniMcqStyle()
        style.download_benchmark("arc_challenge")
        assert style.bank_modality() == "mcq"

    def test_manifest_and_allowlist_disagreeing_refuses_to_guess(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """One of the two is stale, and either guess grades real items on an assumption."""
        write_bank(tmp_path, dataset="arc_challenge", modality="generative")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        style = UniMcqStyle()
        style.download_benchmark("arc_challenge")
        with pytest.raises(resolve.DatasetNotAvailable, match="re-vendor the bank"):
            style.bank_modality()

    def test_asking_before_resolving_a_bank_explains_itself(self) -> None:
        with pytest.raises(RuntimeError, match="download_benchmark"):
            UniMcqStyle().grading_request()

    def test_styles_without_the_hook_are_treated_as_mcq(self) -> None:
        """The frozen CatStyle has no modality method, so the hook has to be optional."""

        class LegacyStyle:
            pass

        request = grading.request_for(LegacyStyle(), dataset="arc_challenge")
        assert request.modality == "mcq"
        assert request.dataset == "arc_challenge"


class TestMcqIsStillGradedByLogProbability:
    """The regression the generative work must not break."""

    @pytest.mark.parametrize("dataset", ["arc_challenge", "hellaswag"])
    def test_named_mcq_datasets_route_to_the_log_likelihood_scorer(
        self, monkeypatch, tmp_path: Path, dataset: str
    ) -> None:
        unblock(monkeypatch, dataset)
        write_bank(tmp_path, dataset=dataset)
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        style = UniMcqStyle()
        style.download_benchmark(dataset)
        request = style.grading_request()

        assert request.modality == "mcq"
        assert grading.get_grader(request.modality).load is grading._load_mcq

        loaded: list[Path] = []
        monkeypatch.setattr(
            inference,
            "load_scoring_model",
            lambda checkpoint_dir, config: loaded.append(checkpoint_dir) or FakeLogprobModel(),
        )
        model = grading.load_grader(request, tmp_path / "ckpt", grading.GradingSettings())

        assert loaded == [tmp_path / "ckpt"]
        assert isinstance(model, ScoringModel)

    def test_responses_carry_one_log_probability_per_choice(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        write_bank(tmp_path, dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        style = UniMcqStyle()
        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")

        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt,
            model=FakeLogprobModel(),
            se_threshold=0.3,
            max_items=40,
        )

        assert report.responses
        for response in report.responses:
            item = bank.get(response.item_id)
            assert len(response.choice_logprobs) == len(item.choices)
            assert all(math.isfinite(value) for value in response.choice_logprobs)
            argmax = max(
                range(len(response.choice_logprobs)),
                key=lambda i: response.choice_logprobs[i],
            )
            assert response.chosen_index == argmax
            assert response.correct == (argmax == item.gold_index)

    def test_the_mcq_report_still_says_log_likelihood(self, monkeypatch, tmp_path: Path) -> None:
        write_bank(tmp_path, dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)

        style = UniMcqStyle()
        bank = style.download_benchmark("arc_challenge")
        irt = style.load_irt_params("arc_challenge")
        report = cat_loop.run_cat(
            style, bank=bank, irt_bank=irt, model=FakeLogprobModel(), se_threshold=0.3, max_items=40
        )

        assert report.metadata["modality"] == "mcq"
        assert "continuation log-likelihood" in report.metadata["scoring_note"]
        assert "not length-normalized" in report.metadata["scoring_note"]


class TestThroughTheRunner:
    """Naming a dataset must drive the whole pipeline, whichever modality it is."""

    def test_an_mcq_dataset_runs_through_the_log_likelihood_scorer(
        self, monkeypatch, tmp_path: Path, toy_params
    ) -> None:
        banks = tmp_path / "banks"
        write_bank(banks, dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", banks)
        staged = stage_hf_checkpoint(tmp_path)
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)

        chosen: list[str] = []
        monkeypatch.setattr(
            inference,
            "load_scoring_model",
            lambda *a, **k: chosen.append("mcq") or SimScorer(0.5, toy_params),
        )
        monkeypatch.setattr(
            generative,
            "load_generative_model",
            lambda *a, **k: pytest.fail("an MCQ bank must not reach the generative grader"),
        )

        out_dir = tmp_path / "out"
        exit_code = runner.main(
            [
                "--cat-style",
                "uni_mcq",
                "--checkpoint",
                "s3://bucket/run/step_1000",
                "--s3-out",
                str(out_dir),
                "--benchmark",
                "arc_challenge",
            ]
        )

        assert exit_code == 0
        assert chosen == ["mcq"]
        payload = json.loads((out_dir / "cat_report.json").read_text(encoding="utf-8"))
        assert payload["run"]["modality"] == "mcq"
        assert "log-likelihood" in payload["run"]["grader"]

    def test_a_generative_dataset_runs_through_the_sampling_scorer(
        self, monkeypatch, tmp_path: Path, toy_params
    ) -> None:
        from .conftest import TOY_GENERATIVE_ITEMS, SimCompleter
        from .test_generative_grading import STUB_FEWSHOT

        banks = tmp_path / "banks"
        monkeypatch.setitem(
            datasets.SUPPORTED,
            GENERATIVE_DATASET,
            make_spec(GENERATIVE_DATASET, modality="generative"),
        )
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", banks)
        monkeypatch.setitem(generative.FEWSHOT_SOURCES, "gsm8k", lambda: STUB_FEWSHOT)
        write_bank(banks, dataset=GENERATIVE_DATASET, modality="generative")
        staged = stage_hf_checkpoint(tmp_path)
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)

        seen_configs: list[generative.GenerationConfig] = []
        monkeypatch.setattr(
            generative,
            "load_generative_model",
            lambda _dir, config: (
                seen_configs.append(config)
                or generative.GenerativeScorer(
                    SimCompleter(0.8, toy_params, TOY_GENERATIVE_ITEMS), config
                )
            ),
        )
        monkeypatch.setattr(
            inference,
            "load_scoring_model",
            lambda *a, **k: pytest.fail("a generative bank must not reach the MCQ scorer"),
        )

        out_dir = tmp_path / "out"
        exit_code = runner.main(
            [
                "--cat-style",
                "uni_mcq",
                "--checkpoint",
                "s3://bucket/run/step_1000",
                "--s3-out",
                str(out_dir),
                "--benchmark",
                GENERATIVE_DATASET,
            ]
        )

        assert exit_code == 0
        # config.yaml's pinned few-shot count reaches the grader through the style.
        assert seen_configs[0].num_fewshot == 8
        payload = json.loads((out_dir / "cat_report.json").read_text(encoding="utf-8"))
        assert payload["run"]["modality"] == "generative"
        assert payload["metadata"]["modality"] == "generative"
        assert all(r["chosen_index"] == -1 for r in payload["responses"])

    def test_a_mismatched_bank_fails_before_the_checkpoint_is_fetched(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """A generative bank listed as MCQ must not reach a GPU, let alone a report."""
        banks = tmp_path / "banks"
        monkeypatch.setitem(
            datasets.SUPPORTED, GENERATIVE_DATASET, make_spec(GENERATIVE_DATASET, modality="mcq")
        )
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", banks)
        write_bank(banks, dataset=GENERATIVE_DATASET, modality="mcq")

        # Items are generative even though both records say MCQ: the shape is what the
        # guard checks, because the shape is what the grader actually consumes.
        items_path = banks / GENERATIVE_DATASET / "items.jsonl"
        items_path.write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"toy_{i}",
                        "question": f"Word problem {i}?",
                        "choices": [],
                        "gold_index": -1,
                        "metadata": {"gold_answer": str(i), "answer_type": "numeric"},
                    }
                )
                + "\n"
                for i in range(5)
            ),
            encoding="utf-8",
        )

        def _boom(*args, **kwargs):
            raise AssertionError("nothing expensive may happen after a modality mismatch")

        monkeypatch.setattr(s3_io, "resolve_checkpoint", _boom)
        monkeypatch.setattr(inference, "load_scoring_model", _boom)
        monkeypatch.setattr(generative, "load_generative_model", _boom)

        out_dir = tmp_path / "out"
        assert (
            runner.main(
                [
                    "--cat-style",
                    "uni_mcq",
                    "--checkpoint",
                    "s3://bucket/run/step_1000",
                    "--s3-out",
                    str(out_dir),
                    "--benchmark",
                    GENERATIVE_DATASET,
                ]
            )
            == 1
        )
        assert not out_dir.exists()
