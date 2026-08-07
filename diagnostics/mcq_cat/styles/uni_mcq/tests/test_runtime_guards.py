"""Guards on the runtime path, each pinned against the state that reached it.

Everything here is a case a run can actually be in, found by walking the runner from
argument parsing to the written report rather than by reading the tests. They share a
shape: none of them is a crash the engine would hit anyway, and every one of them used
to end in a well-formed report whose theta was computed from something other than the
checkpoint -- a fabricated zero that passed for a wrong answer, a parameter file nobody
asked for, a scored span the length cap had eaten. The last one is the reason they are
worth tests at all: an exception is self-reporting and these were not.

Nothing here needs a GPU. ``torch`` and ``transformers`` are absent by design, so the
token arithmetic is checked through the pure helper the forward pass calls and the
scorers are driven with their sampling and forward-pass halves injected.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from .... import runner
from ....base import BenchmarkItem
from ....common import (
    benchmark_download,
    cat_loop,
    generative,
    grading,
    inference,
    irt_params,
    s3_io,
)
from .. import datasets, resolve
from ..style import UniMcqStyle
from .conftest import GENERATIVE_DATASET, TOY_PARAMS, SimScorer, make_spec, write_bank

#: The three gold-matched graders. IFEval is deliberately absent: it decides from the
#: item's constraints and has no gold to be blank.
GOLD_MATCHED_TYPES = ("numeric", "math_latex", "gpqa_letter")

#: A request carrying no style overrides, for the checks that only read the settings.
_REQUEST = grading.GradingRequest(dataset="arc_challenge", modality=grading.MCQ)


def gold_item(gold: object, answer_type: str = "numeric") -> BenchmarkItem:
    """A generative item carrying ``gold`` as its expected answer."""
    return BenchmarkItem(
        item_id="g0",
        question="Q?",
        choices=(),
        gold_index=-1,
        metadata={"gold_answer": gold, "answer_type": answer_type, "modality": "generative"},
    )


class TestABlankGoldIsNotAnAnswer:
    """A gold of ``""`` matches nothing, so every model is wrong and nothing says why.

    The zero it produces is the bank's, not the checkpoint's, which is the distinction
    the ungradable marking exists to keep. Testing ``is not None`` admitted the blank
    while the verifier-scored half of the same function tested truthiness and did not,
    so a bank could be refused for an empty constraint list and accepted for an empty
    gold.
    """

    @pytest.mark.parametrize("answer_type", GOLD_MATCHED_TYPES)
    @pytest.mark.parametrize("gold", ["", "   ", "\n\t"])
    def test_the_bank_level_guard_refuses_it(self, answer_type: str, gold: str) -> None:
        assert not generative.is_gradable(gold_item(gold, answer_type))

    @pytest.mark.parametrize("answer_type", GOLD_MATCHED_TYPES)
    def test_a_real_gold_is_still_gradable(self, answer_type: str) -> None:
        assert generative.is_gradable(gold_item("B", answer_type))

    def test_a_gold_of_zero_survives_the_emptiness_test(self) -> None:
        """``0`` is falsy and is a legitimate GSM8K answer, so the test is on its text."""
        assert generative.gradable_gold(gold_item(0)) == "0"
        assert generative.is_gradable(gold_item(0))
        assert generative.is_gradable(gold_item("0"))

    def test_a_gold_keeps_its_surrounding_space_when_it_has_content(self) -> None:
        """Only the emptiness test strips; the grader is handed what the bank holds."""
        assert generative.gradable_gold(gold_item(" 72 ")) == " 72 "

    @pytest.mark.parametrize("gold", [None, "", "   "])
    def test_the_item_level_marker_catches_one_that_got_past(self, gold: object) -> None:
        response = generative.grade_completion(
            gold_item(gold), "So the answer is 72.", generative.GenerationConfig(num_fewshot=0)
        )

        assert response.correct is False
        assert response.metadata[generative.UNGRADABLE_KEY] is True
        assert "gold_answer" in response.metadata[generative.UNGRADABLE_REASON_KEY]

    def test_a_wrong_answer_is_still_only_a_wrong_answer(self) -> None:
        """The marker must separate the two, not label every incorrect response."""
        response = generative.grade_completion(
            gold_item("72"), "So the answer is 73.", generative.GenerationConfig(num_fewshot=0)
        )

        assert response.correct is False
        assert response.metadata[generative.UNGRADABLE_KEY] is False

    def test_a_bank_carrying_one_is_refused_before_a_checkpoint_is_staged(self) -> None:
        request = grading.GradingRequest(dataset="gsm8k", modality=grading.GENERATIVE)
        with pytest.raises(grading.ModalityMismatch, match="not shaped that way"):
            grading.check_bank_modality(request, [gold_item("72"), gold_item("")])


class TestAMalformedIFEvalItemDoesNotEndTheSession:
    """The verifiers refusing one item must cost that item, not the whole diagnostic.

    Both inputs here reach olmo-eval's scorer past the bank-level guard -- it pairs the
    instruction list and the kwargs list strictly and indexes the registry directly, so
    a short kwargs list raises ``ValueError`` and an unregistered id raises ``KeyError``.
    Either one leaving the grader unwinds the CAT loop and the runner writes no report,
    which is the outcome an empty strict list is already written to avoid on a bank that
    is wrong in exactly the same way.
    """

    @pytest.fixture
    def strict_registry(self, monkeypatch):
        """A registry holding one id and raising ``KeyError`` for anything else."""
        pytest.importorskip(
            "olmo_eval.common.scorers",
            reason="IFEval grading runs olmo-eval's real IFEvalScorer; run with PYTHONPATH=src",
        )

        class Passing:
            def __init__(self, instruction_id: str) -> None:
                self.instruction_id = instruction_id

            def get_instruction_args_keys(self) -> list[str]:
                return []

            def build_description(self, **kwargs: object) -> str:
                return self.instruction_id

            def get_instruction_args(self) -> dict:
                return {}

            def check_following(self, response: str) -> bool:
                return True

        registry = types.ModuleType("ifbench.instructions_registry")
        registry.INSTRUCTION_DICT = {"rule:known": Passing}
        package = types.ModuleType("ifbench")
        package.instructions_registry = registry
        monkeypatch.setitem(sys.modules, "ifbench", package)
        monkeypatch.setitem(sys.modules, "ifbench.instructions_registry", registry)
        return registry

    def item(self, instruction_ids: list[str], kwargs: list[dict]) -> BenchmarkItem:
        return BenchmarkItem(
            item_id="i0",
            question="Write something.",
            choices=(),
            gold_index=-1,
            metadata={
                "answer_type": "ifeval_strict",
                "modality": "generative",
                "instruction_id_list": instruction_ids,
                "kwargs": kwargs,
            },
        )

    def test_fewer_kwargs_than_instructions_is_ungradable_not_fatal(self, strict_registry) -> None:
        verdict = generative.get_answer_grader("ifeval_strict").grade_item(
            "a response", self.item(["rule:known", "rule:known"], [{}])
        )

        assert verdict.correct is False
        assert verdict.ungradable_reason is not None
        assert "ValueError" in verdict.ungradable_reason
        assert verdict.detail["strict"] == []

    def test_an_unregistered_instruction_id_is_ungradable_not_fatal(self, strict_registry) -> None:
        verdict = generative.get_answer_grader("ifeval_strict").grade_item(
            "a response", self.item(["rule:missing"], [{}])
        )

        assert verdict.correct is False
        assert verdict.ungradable_reason is not None
        assert "KeyError" in verdict.ungradable_reason

    def test_a_well_formed_item_is_unaffected(self, strict_registry) -> None:
        verdict = generative.get_answer_grader("ifeval_strict").grade_item(
            "a response", self.item(["rule:known"], [{}])
        )

        assert verdict.correct is True
        assert verdict.ungradable_reason is None

    def test_the_scorer_carries_it_through_to_the_response(self, strict_registry) -> None:
        response = generative.grade_completion(
            self.item(["rule:missing"], [{}]),
            "a response",
            generative.GenerationConfig(num_fewshot=0, prompt_style="ifeval", stop_sequences=()),
        )

        assert response.metadata[generative.UNGRADABLE_KEY] is True

    def test_a_missing_ifbench_still_raises(self, monkeypatch) -> None:
        """One item's payload is a bank fault; the registry's absence is every item's.

        The broad catch must not swallow it into an ungradable verdict, which would turn
        a missing dependency into a whole run of fabricated zeroes and a very low theta.
        """

        class NoRegistry:
            def score(self, instance: object, output: object) -> float:
                raise ImportError("No module named 'ifbench'")

        class Instance:
            def __init__(self, **kwargs: object) -> None:
                pass

        class Output:
            def __init__(self, text: str) -> None:
                self.text = text
                self.metadata: dict = {}

        monkeypatch.setattr(generative, "_ifeval_scoring", lambda: (NoRegistry, Instance, Output))
        with pytest.raises(RuntimeError, match="ifbench"):
            generative.get_answer_grader("ifeval_strict").grade_item(
                "a response", self.item(["rule:known"], [{}])
            )


class TestTheLengthCapCannotEatTheScoredSpan:
    """``max_length`` truncates the prompt from the left; the continuation is the answer.

    The count has to come off the untruncated pair. Taken after the cut it shrinks by
    however much was dropped, so the score covers only the tail of the answer, and once
    the cap sits at or below the prompt it comes out non-positive and the choice scores
    ``0.0``. Every choice scoring ``0.0`` hands the argmax to the first one on every
    item, and a whole run's theta then reports a length cap rather than a checkpoint.

    Checked through the helper rather than through a forward pass because the arithmetic
    is the whole of the bug and ``transformers`` is not installed here.
    """

    def test_no_cap_counts_the_whole_continuation(self) -> None:
        assert inference.continuation_token_count(40, 46, None) == 6

    def test_a_cap_above_the_pair_changes_nothing(self) -> None:
        assert inference.continuation_token_count(40, 46, 128) == 6

    def test_a_cap_that_only_reaches_the_prompt_leaves_the_count_alone(self) -> None:
        """The dropped tokens are prompt; measuring after the cut reported 4 of 6."""
        assert inference.continuation_token_count(40, 46, 44) == 6

    def test_a_cap_below_the_prompt_is_refused_rather_than_scoring_nothing(self) -> None:
        with pytest.raises(ValueError, match="cannot be set below"):
            inference.continuation_token_count(40, 46, 6)

    def test_a_cap_equal_to_the_continuation_is_refused(self) -> None:
        """There would be no prompt left in front of the span being scored."""
        with pytest.raises(ValueError, match="cannot be set below"):
            inference.continuation_token_count(40, 46, 5)

    def test_a_continuation_that_adds_no_tokens_has_no_span(self) -> None:
        assert inference.continuation_token_count(40, 40, None) == 0

    def test_the_forward_pass_sums_over_the_untruncated_count(self) -> None:
        """The scorer with only its tokenizer, model and ``torch`` replaced.

        A 40-token prompt, a 6-token continuation and a cap of 44: the pair is cut to
        44 tokens, all four dropped from the prompt, and the span still to be summed is
        the same 6. Measured after the cut it was 4.
        """
        summed: list[int] = []
        scorer = inference._HFScoringModel.__new__(inference._HFScoringModel)
        scorer.config = inference.InferenceConfig(max_length=44)
        scorer.tokenizer = _FakeTokenizer({"prompt": 40, "promptcontinuation": 46})
        scorer.model = _FakeModel()
        scorer._torch = _FakeTorch(summed)

        scorer._continuation_logprob("prompt", "continuation")

        assert summed == [6]


class _Ids:
    """The slice of a token tensor ``_continuation_logprob`` touches."""

    def __init__(self, length: int) -> None:
        self.shape = (1, length)

    def __getitem__(self, key: object) -> _Ids:
        return self

    def to(self, device: object) -> _Ids:
        return self

    def unsqueeze(self, dim: int) -> _Ids:
        return self


class _FakeTokenizer:
    """Returns a token count per exact input string."""

    def __init__(self, lengths: dict[str, int]) -> None:
        self._lengths = lengths

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        return {"input_ids": _Ids(self._lengths[text])}


class _FakeModel:
    device = "cpu"

    def __call__(self, ids: object) -> _FakeModel:
        return self

    @property
    def logits(self) -> _Logits:
        return _Logits(None)


class _FakeTorch:
    """A ``torch`` stand-in recording the width of the span the scorer summed."""

    def __init__(self, summed: list[int]) -> None:
        self._summed = summed

    def no_grad(self):
        from contextlib import nullcontext

        return nullcontext()

    def log_softmax(self, tensor: object, dim: int) -> _Logits:
        return _Logits(self._summed)


class _Logits:
    """Records the ``[:, -cont_len:]`` slice and answers everything else with itself."""

    def __init__(self, summed: list[int] | None) -> None:
        self._summed = summed

    def __getitem__(self, key: object) -> _Logits:
        if (
            self._summed is not None
            and isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[1], slice)
            and key[1].start is not None
        ):
            self._summed.append(-key[1].start)
        return self

    def gather(self, dim: int, index: object) -> _Logits:
        return self

    def squeeze(self, dim: int) -> _Logits:
        return self

    def sum(self) -> _Logits:
        return self

    def item(self) -> float:
        return -1.0


class TestAnEmptyStopSequenceIsRefused:
    """``"".find`` matches at offset 0, so one empty entry truncates every completion.

    A blank line and ``Question:`` are the two real entries and both are meaningful; a
    third that is the empty string would cut all output to nothing and grade the whole
    run on blank responses -- accepted and then silently destroying every score, which
    is what the temperature guard beside it refuses for the same reason.
    """

    def test_an_empty_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="empty string"):
            generative.GenerationConfig(stop_sequences=("Question:", ""))

    def test_no_stop_sequences_at_all_is_allowed(self) -> None:
        """ifeval and gpqa both want the whole completion."""
        assert generative.GenerationConfig(stop_sequences=()).stop_sequences == ()

    def test_the_shipped_sequences_are_unaffected(self) -> None:
        assert generative.GenerationConfig().stop_sequences == ("Question:", "\n\n")


class TestArtifactsAreReadAsUtf8:
    """Locale decoding substitutes a different stem rather than failing.

    Vendoring writes these UTF-8. ``read_text()`` without an encoding uses the platform
    locale, which is cp1252 on Windows, and most UTF-8 byte pairs decode under it to
    something else instead of raising -- so a run would score a question the bank holds
    no difficulty for, and nothing in the report would show it. The committed banks are
    ASCII-escaped JSON today, which is what makes this latent rather than live.
    """

    def test_a_non_ascii_stem_survives_loading(self, tmp_path: Path) -> None:
        source = tmp_path / "items.jsonl"
        source.write_text(
            json.dumps(
                {"id": "u0", "question": "Wie groß ist π?", "choices": ["3", "4"], "gold_index": 0},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        bank = benchmark_download.load_items_from_jsonl(source)

        assert bank.items[0].question == "Wie groß ist π?"

    def test_a_non_ascii_item_id_still_joins(self, tmp_path: Path) -> None:
        """A mis-decoded id matches nothing and the item drops out of the bank silently."""
        source = tmp_path / "params.json"
        source.write_text(
            json.dumps([{"item_id": "größe|0", "difficulty": 0.5}], ensure_ascii=False),
            encoding="utf-8",
        )

        assert "größe|0" in irt_params.load_irt_params(source).params

    def test_the_shared_reader_agrees_with_the_s3_branch(self, tmp_path: Path) -> None:
        source = tmp_path / "text"
        source.write_bytes("µ".encode())

        assert s3_io.read_text(str(source)) == "µ"


@pytest.fixture
def mcq_style(monkeypatch, tmp_path: Path) -> UniMcqStyle:
    """A style pointed at a five-item MCQ bank named ``arc_challenge``."""
    write_bank(tmp_path, dataset="arc_challenge")
    monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
    return UniMcqStyle()


class TestTheParameterSourceIsTheOneAskedFor:
    """A source that is neither a readable file nor the resolved bank must not be ignored.

    ``--irt-params`` names where the difficulties come from, and every value that was
    not a local file fell through to the bank ``download_benchmark`` had already
    resolved: a mistyped path, and an ``s3://`` URI that the loader underneath has
    always been able to read. Neither failed. The run finished and reported an ability
    estimated against parameters the caller did not choose, which no field in the report
    distinguishes from one estimated against the parameters they did.
    """

    def test_the_resolved_bank_is_reused_when_the_name_matches(self, mcq_style) -> None:
        mcq_style.download_benchmark("arc_challenge")
        assert len(mcq_style.load_irt_params("arc_challenge")) == len(TOY_PARAMS)

    def test_an_explicit_file_still_wins(self, mcq_style, tmp_path: Path) -> None:
        mcq_style.download_benchmark("arc_challenge")
        params = tmp_path / "arc_challenge" / "params.json"
        assert len(mcq_style.load_irt_params(params)) == len(TOY_PARAMS)

    def test_a_path_that_does_not_exist_is_refused(self, mcq_style, tmp_path: Path) -> None:
        mcq_style.download_benchmark("arc_challenge")
        with pytest.raises(resolve.DatasetNotAvailable):
            mcq_style.load_irt_params(tmp_path / "nowhere" / "params.json")

    def test_another_dataset_name_resolves_to_that_dataset(self, mcq_style, tmp_path: Path) -> None:
        """Not to the bank already in hand, which is what made a typo invisible."""
        mcq_style.download_benchmark("arc_challenge")
        with pytest.raises(resolve.DatasetNotAvailable, match="piqa"):
            mcq_style.load_irt_params("piqa")

    def test_an_s3_uri_reaches_the_loader(self, mcq_style, monkeypatch, tmp_path: Path) -> None:
        mcq_style.download_benchmark("arc_challenge")
        asked: list[str] = []

        def read_text(uri: str, **kwargs: object) -> str:
            asked.append(uri)
            return (tmp_path / "arc_challenge" / "params.json").read_text(encoding="utf-8")

        monkeypatch.setattr(s3_io, "read_text", read_text)
        mcq_style.load_irt_params("s3://bucket/run/params.json")

        assert asked == ["s3://bucket/run/params.json"]


class TestAnUnloadableCheckpointFormatFailsFirst:
    """``olmo_core`` is an integration point, and the step after this check downloads.

    Nothing about the failure needs the checkpoint: the format is known from a flag and
    neither grader has a loader for it. Discovering it inside the loader means
    discovering it after ``resolve_checkpoint`` has pulled every object under an
    ``s3://`` prefix, which for a checkpoint is the expensive part of the run.
    """

    def test_hf_passes(self) -> None:
        grading.check_checkpoint_kind(_REQUEST, grading.GradingSettings())

    def test_olmo_core_is_refused(self) -> None:
        settings = grading.GradingSettings(
            mcq=inference.InferenceConfig(checkpoint_kind="olmo_core"),
            generation=generative.GenerationConfig(checkpoint_kind="olmo_core"),
        )
        with pytest.raises(NotImplementedError, match="olmo_core"):
            grading.check_checkpoint_kind(_REQUEST, settings)

    def test_a_style_override_is_folded_on_before_the_check(self) -> None:
        """The format checked has to be the one a model would be built from."""
        request = grading.GradingRequest(
            dataset="arc_challenge",
            modality=grading.MCQ,
            mcq={"datasets": {"arc_challenge": {"checkpoint_kind": "olmo_core"}}},
        )
        with pytest.raises(NotImplementedError, match="olmo_core"):
            grading.check_checkpoint_kind(request, grading.GradingSettings())

    def test_the_runner_refuses_before_fetching_the_checkpoint(
        self, monkeypatch, mcq_style, tmp_path: Path
    ) -> None:
        def _boom(*args: object, **kwargs: object):
            raise AssertionError("the checkpoint must not be fetched for an unloadable format")

        monkeypatch.setattr(s3_io, "resolve_checkpoint", _boom)
        monkeypatch.setattr(inference, "load_scoring_model", _boom)

        assert (
            runner.main(
                [
                    "--cat-style",
                    "uni_mcq",
                    "--checkpoint",
                    "s3://bucket/run/step_1000",
                    "--s3-out",
                    str(tmp_path / "out"),
                    "--benchmark",
                    "arc_challenge",
                    "--checkpoint-kind",
                    "olmo_core",
                ]
            )
            == 1
        )
        assert not (tmp_path / "out").exists()


class TestASessionThatAdministersNothing:
    """``--max-items 0`` reaches ``report`` with no ability estimate at all.

    Reachable from the CLI and not covered anywhere else. The report has to be
    well formed -- the ungradable rate divides by the administered count, p-IRT
    weights by it, and observed accuracy averages over it -- and it has to be readable
    as what it is, which is the prior rather than a measurement.
    """

    @pytest.fixture
    def report(self, mcq_style, toy_params):
        bank = mcq_style.download_benchmark("arc_challenge")
        irt = mcq_style.load_irt_params("arc_challenge")
        return cat_loop.run_cat(
            mcq_style,
            bank=bank,
            irt_bank=irt,
            model=SimScorer(0.5, toy_params),
            se_threshold=0.3,
            max_items=0,
        )

    def test_nothing_is_administered_and_nothing_divides_by_zero(self, report) -> None:
        assert report.num_items_administered == 0
        assert report.responses == ()
        assert report.metadata["observed_accuracy"] == 0.0
        assert report.metadata["ungradable"]["rate"] == 0.0
        assert report.metadata["ungradable"]["count"] == 0

    def test_the_ability_is_the_untouched_prior(self, report) -> None:
        assert report.ability.theta == 0.0
        assert report.ability.standard_error == 1.0

    def test_predicted_accuracy_is_the_whole_bank_at_the_prior(self, report) -> None:
        """Every item is unobserved, so p-IRT is the bank's mean success at theta 0."""
        assert 0.0 <= report.metadata["pirt_accuracy"] <= 1.0

    def test_the_report_still_serializes(self, report) -> None:
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["num_items_administered"] == 0
        assert payload["responses"] == []

    def test_the_stop_reason_names_the_cap(self, report) -> None:
        assert report.metadata["stop_reason"] == "max_items_reached"


@pytest.fixture
def generative_style(monkeypatch, tmp_path: Path) -> UniMcqStyle:
    """A style pointed at a five-item generative bank."""
    monkeypatch.setitem(
        datasets.SUPPORTED, GENERATIVE_DATASET, make_spec(GENERATIVE_DATASET, modality="generative")
    )
    monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path)
    write_bank(tmp_path, dataset=GENERATIVE_DATASET, modality="generative")
    return UniMcqStyle()


class TestABlankGoldReachesTheReport:
    """End to end: the fabricated zeroes are counted apart from the wrong answers."""

    def test_the_ungradable_block_counts_them_and_alerts(
        self, generative_style, tmp_path: Path
    ) -> None:
        blanked = tmp_path / GENERATIVE_DATASET / "items.jsonl"
        blanked.write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"toy_{i}",
                        "question": f"Toy word problem {i}?",
                        "choices": [],
                        "gold_index": -1,
                        "metadata": {
                            "gold_answer": "" if i % 2 else str(40 + i),
                            "answer_type": "numeric",
                            "modality": "generative",
                        },
                    }
                )
                + "\n"
                for i in range(5)
            ),
            encoding="utf-8",
        )

        bank = generative_style.download_benchmark(GENERATIVE_DATASET)
        irt = generative_style.load_irt_params(GENERATIVE_DATASET)
        report = cat_loop.run_cat(
            generative_style,
            bank=bank,
            irt_bank=irt,
            model=generative.GenerativeScorer(
                lambda prompt: "So the answer is 41.",
                generative.GenerationConfig(num_fewshot=0),
            ),
            se_threshold=0.3,
            max_items=40,
        )

        block = report.metadata["ungradable"]
        assert block["count"] == 2
        assert sorted(block["item_ids"]) == ["toy_1", "toy_3"]
        assert block["rate"] == pytest.approx(0.4)
        assert "alert" in block
        json.dumps(report.to_dict())


class TestTheBlankIsTheOneTheGuardCounts:
    """The substitution style may only cut where a blank actually stands.

    :func:`~diagnostics.mcq_cat.common.inference.check_prompt_style_fits` guards only the
    appending direction, on the stated grounds that the substitution direction raises by
    itself on a stem with no blank. That was true of a stem with no underscore at all and
    false of one with an incidental underscore, which got cut at it and scored silently.
    """

    def test_an_incidental_underscore_is_not_a_blank(self) -> None:
        item = BenchmarkItem(
            item_id="m0",
            question="When frozen carbon dioxide (CO_{2}) is heated, which is true?",
            choices=("It sublimates.", "It melts."),
            gold_index=0,
        )
        with pytest.raises(ValueError, match="standalone"):
            inference.scored_choices(
                item, inference.InferenceConfig(prompt_style="blank_substitution")
            )

    def test_a_run_of_underscores_is_not_a_blank(self) -> None:
        item = BenchmarkItem(
            item_id="m1",
            question="Many animals depend on plants for ___.",
            choices=("shelter", "sunlight"),
            gold_index=0,
        )
        with pytest.raises(ValueError, match="standalone"):
            inference.scored_choices(
                item, inference.InferenceConfig(prompt_style="blank_substitution")
            )

    def test_a_real_blank_still_splits_where_it_always_did(self) -> None:
        item = BenchmarkItem(
            item_id="m2",
            question="Sarah was a better surgeon than Maria so _ always got the easy cases.",
            choices=("Sarah", "Maria"),
            gold_index=0,
        )
        pairs = inference.scored_choices(
            item, inference.InferenceConfig(prompt_style="blank_substitution")
        )
        assert [pair.prompt for pair in pairs] == [
            "Sarah was a better surgeon than Maria so Sarah",
            "Sarah was a better surgeon than Maria so Maria",
        ]
        assert {pair.continuation for pair in pairs} == {" always got the easy cases."}

    def test_the_guard_and_the_split_agree_on_every_vendored_mcq_bank(self) -> None:
        """No bank may be called non-cloze by one and still be splittable by the other.

        The two used different tests, so a bank could pass the appending guard on the
        strict count and then be cut by the loose search anyway.
        """
        for dataset in ("arc_challenge", "hellaswag", "winogrande", "bbh"):
            path = resolve.CALIBRATED_DATASETS / dataset / "items.jsonl"
            if not path.is_file():
                continue
            items = list(benchmark_download.load_items_from_jsonl(path, name=dataset).items)
            cloze = sum(1 for item in items if inference._LONE_BLANK_RE.search(item.question))
            splittable = sum(
                1
                for item in items
                if inference.lone_blank_re("_").search(item.question) is not None
            )
            assert cloze == splittable, dataset


class TestAGoldIndexPastTheChoices:
    """An MCQ gold nothing can equal is a fabricated zero, not a wrong answer."""

    def test_the_modality_check_refuses_it(self) -> None:
        item = BenchmarkItem(item_id="m0", question="Q?", choices=("a", "b", "c"), gold_index=99)
        request = grading.GradingRequest(dataset="toy", modality="mcq")
        with pytest.raises(grading.ModalityMismatch, match="gold_index into it"):
            grading.check_bank_modality(request, [item])

    def test_an_in_range_gold_still_passes(self) -> None:
        item = BenchmarkItem(item_id="m1", question="Q?", choices=("a", "b", "c"), gold_index=2)
        request = grading.GradingRequest(dataset="toy", modality="mcq")
        grading.check_bank_modality(request, [item])

    def test_every_vendored_mcq_bank_is_in_range(self) -> None:
        for dataset in ("arc_challenge", "hellaswag", "winogrande", "bbh"):
            path = resolve.CALIBRATED_DATASETS / dataset / "items.jsonl"
            if not path.is_file():
                continue
            items = list(benchmark_download.load_items_from_jsonl(path, name=dataset).items)
            request = grading.GradingRequest(dataset=dataset, modality="mcq")
            grading.check_bank_modality(request, items)
