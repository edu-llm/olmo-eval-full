"""Verifier-scored grading for the IFEval bank, and the interface it widened.

IFEval is the first bank here with no gold answer anywhere in it. Its items carry the
instruction ids and per-instruction arguments the prompt states, and a response is
correct when every one of those verifiers accepts it -- ``prompt_level_strict_acc``,
the metric the bank was calibrated under. That is a second shape of grader, not a
second modality: one sampled completion still yields one binary, so the frozen CAT
engine, the EAP update and p-IRT are all untouched.

Three failure modes are what these guard against, in rising order of quietness. A
grader called with the wrong signature crashes, which is fine. An item admitted without
its constraints scores incorrect for every model and reads as a weak checkpoint. And
loose scoring in place of strict, or a stop sequence cutting a multi-paragraph answer,
produces a perfectly well-formed report whose theta is on a different scale from the
bank's difficulties.

The verifier registry itself lives in ``ifbench``, a declared dependency installed from
a git URL. Tests that need real verifier behaviour skip without it; the rest install a
stub registry in ``sys.modules`` so the real ``IFEvalScorer``, the real grader and the
real CAT engine all run on machines that do not have it.
"""

from __future__ import annotations

import json
import re
import sys
import types
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ....base import BenchmarkItem
from ....common import cat_loop, generative, grading, inference
from .. import style as style_mod
from ..datasets import SUPPORTED
from ..style import UniMcqStyle
from .conftest import CALIBRATED_DATASETS, SimGenerativeTaker, vendored_params

DATASET = "ifeval"

#: The stub registry accepts a response only if it contains this. It stands in for
#: "every instruction passed", so a completion carrying it is correct under strict
#: scoring and one without it is not.
FOLLOWED = "[followed]"


def ifeval_item(
    instruction_ids: tuple[str, ...] = ("punctuation:no_comma",),
    kwargs: tuple[dict, ...] = ({},),
    question: str = "Write a summary with no commas.",
) -> BenchmarkItem:
    """A generative item shaped as a vendored IFEval bank record."""
    return BenchmarkItem(
        item_id="0",
        question=question,
        choices=(),
        gold_index=-1,
        metadata={
            "answer_type": "ifeval_strict",
            "modality": "generative",
            "instruction_id_list": list(instruction_ids),
            "kwargs": [dict(kw) for kw in kwargs],
        },
    )


class StubInstruction:
    """The slice of an IFBench verifier that ``IFEvalScorer`` actually calls."""

    def __init__(self, instruction_id: str) -> None:
        self.instruction_id = instruction_id
        self.built: list[dict] = []

    def get_instruction_args_keys(self) -> list[str]:
        return ["prompt_to_repeat"]

    def build_description(self, **kwargs: object) -> str:
        self.built.append(dict(kwargs))
        return self.instruction_id

    def get_instruction_args(self) -> dict:
        return {}

    def check_following(self, response: str) -> bool:
        return FOLLOWED in response


@pytest.fixture
def stub_ifbench(monkeypatch):
    """Install a fake ``ifbench`` whose every verifier looks for :data:`FOLLOWED`.

    Everything above the verifier stays real: olmo-eval's ``IFEvalScorer`` resolves the
    ids, fills in ``prompt_to_repeat``, runs strict and all eight loose variants, and
    writes the pass lists that :class:`IFEvalPromptStrict` reads. Only the verifiers'
    own bodies are simulated, which is the part ``ifbench`` owns and this machine may
    not have.

    Because the scorer is real, ``olmo_eval`` has to be importable. Absent it the grader
    raises rather than returning a verdict, which pytest reports as a failure -- so a
    bare checkout would show thirteen red tests for a missing optional package instead
    of the clean skip every other file here gives. Guarding in the fixture rather than
    per test keeps the two from drifting as tests are added.
    """
    pytest.importorskip(
        "olmo_eval.common.scorers",
        reason="IFEval grading runs olmo-eval's real IFEvalScorer; run with PYTHONPATH=src",
    )
    registry = types.ModuleType("ifbench.instructions_registry")
    registry.INSTRUCTION_DICT = _AlwaysRegistered()
    package = types.ModuleType("ifbench")
    package.instructions_registry = registry
    monkeypatch.setitem(sys.modules, "ifbench", package)
    monkeypatch.setitem(sys.modules, "ifbench.instructions_registry", registry)
    return registry


class _AlwaysRegistered(dict):
    """A registry that knows every instruction id the bank happens to name."""

    def __missing__(self, key: str) -> type[StubInstruction]:
        return StubInstruction


def ifeval_answer(item: BenchmarkItem, correct: bool) -> str:
    """A response that satisfies the item's constraints, or plainly does not."""
    if correct:
        return f"Here is the response.\n\n{FOLLOWED}\n\nIt closes here."
    return "Here is a response that ignores what was asked."


class TestTheGraderShape:
    def test_ifeval_strict_resolves_to_the_verifier_grader(self) -> None:
        grader = generative.get_answer_grader("ifeval_strict")
        assert isinstance(grader, generative.IFEvalPromptStrict)
        assert grader.name == "ifeval_prompt_strict"

    def test_it_is_an_item_grader_and_not_an_answer_grader(self) -> None:
        """The dispatch in ``apply_grader`` turns on exactly this."""
        grader = generative.get_answer_grader("ifeval_strict")
        assert isinstance(grader, generative.ItemGrader)
        assert not isinstance(grader, generative.AnswerGrader)

    def test_the_gold_matched_graders_are_untouched(self) -> None:
        """Widening the interface must not have moved the other two into it."""
        for answer_type in ("numeric", "math_latex"):
            grader = generative.get_answer_grader(answer_type)
            assert isinstance(grader, generative.AnswerGrader)
            assert not isinstance(grader, generative.ItemGrader)

    def test_it_declares_the_metadata_it_decides_on(self) -> None:
        """Vendoring copies exactly these keys, so the two cannot drift apart."""
        grader = generative.get_answer_grader("ifeval_strict")
        assert grader.required_metadata == ("instruction_id_list", "kwargs")


class TestStrictVerdicts:
    def test_all_instructions_passing_is_correct(self, stub_ifbench) -> None:
        item = ifeval_item(("punctuation:no_comma", "length_constraints:number_words"), ({}, {}))
        verdict = generative.get_answer_grader("ifeval_strict").grade_item(
            f"A response. {FOLLOWED}", item
        )

        assert verdict.correct is True
        assert verdict.detail["strict"] == [True, True]
        assert verdict.gold is None
        assert verdict.extracted is None

    def test_one_failing_instruction_fails_the_item(self, stub_ifbench, monkeypatch) -> None:
        """Prompt-level strict is all-or-nothing; a partial pass is still a 0."""

        class OnlyFirstPasses(StubInstruction):
            def check_following(self, response: str) -> bool:
                return self.instruction_id.endswith("first")

        monkeypatch.setitem(stub_ifbench.INSTRUCTION_DICT, "rule:first", OnlyFirstPasses)
        monkeypatch.setitem(stub_ifbench.INSTRUCTION_DICT, "rule:second", OnlyFirstPasses)

        verdict = generative.get_answer_grader("ifeval_strict").grade_item(
            f"A response. {FOLLOWED}", ifeval_item(("rule:first", "rule:second"), ({}, {}))
        )

        assert verdict.correct is False
        assert verdict.detail["strict"] == [True, False]

    def test_no_instruction_results_is_a_bank_fault_not_a_wrong_answer(self, stub_ifbench) -> None:
        """The 0 enters the response pattern, so it has to be marked as not the model's."""
        verdict = generative.get_answer_grader("ifeval_strict").grade_item(
            "anything", ifeval_item((), ())
        )

        assert verdict.correct is False
        assert verdict.ungradable_reason is not None
        assert "instruction" in verdict.ungradable_reason

    def test_the_prompt_is_handed_back_for_the_verifiers_that_quote_it(self, stub_ifbench) -> None:
        """``prompt_to_repeat`` is filled from the item's question, not left empty.

        The verifiers that check a response repeats the instruction read it from there,
        and would reject every response if it arrived blank.
        """
        seen: list[dict] = []

        class Recording(StubInstruction):
            def build_description(self, **kwargs: object) -> str:
                seen.append(dict(kwargs))
                return self.instruction_id

        stub_ifbench.INSTRUCTION_DICT["combination:repeat_prompt"] = Recording
        item = ifeval_item(("combination:repeat_prompt",), ({},), question="Repeat this exactly.")
        generative.get_answer_grader("ifeval_strict").grade_item(f"x {FOLLOWED}", item)

        assert seen and seen[0]["prompt_to_repeat"] == "Repeat this exactly."


class TestGradeCompletionDispatch:
    def test_a_gold_less_item_reaches_the_verifier_grader(self, stub_ifbench) -> None:
        graded = generative.grade_completion(
            ifeval_item(), f"ok {FOLLOWED}", generative.GenerationConfig(stop_sequences=())
        )

        assert graded.correct is True
        assert graded.metadata["grader"] == "ifeval_prompt_strict"
        assert graded.metadata["gold_answer"] is None
        assert graded.metadata["grader_detail"]["instruction_id_list"] == ["punctuation:no_comma"]

    def test_the_gold_matched_response_shape_gains_no_grader_detail(self) -> None:
        """A gold-matched bank stays free of the verifier grader's per-item detail."""
        item = BenchmarkItem(
            item_id="g0",
            question="Q?",
            choices=(),
            gold_index=-1,
            metadata={"gold_answer": "72", "answer_type": "numeric"},
        )
        graded = generative.grade_completion(
            item, " So the answer is 72.", generative.GenerationConfig(num_fewshot=8)
        )

        assert set(graded.metadata) == {
            "modality",
            "grader",
            "completion",
            "extracted_answer",
            "gold_answer",
            "num_fewshot",
            "ungradable",
        }

    def test_a_gold_matched_item_with_no_gold_is_marked_rather_than_scored(self) -> None:
        """Widening the interface must not have made a missing gold pass for an answer."""
        bare = BenchmarkItem(item_id="g0", question="Q?", choices=(), gold_index=-1)
        graded = generative.grade_completion(bare, "72", generative.GenerationConfig())

        assert graded.correct is False
        assert graded.metadata[generative.UNGRADABLE_KEY] is True

    def test_the_report_records_which_constraints_failed(self, stub_ifbench, monkeypatch) -> None:
        """A wrong IFEval answer has no wrong answer in it, so the reason must be stored."""

        class NeverPasses(StubInstruction):
            def check_following(self, response: str) -> bool:
                return False

        monkeypatch.setitem(stub_ifbench.INSTRUCTION_DICT, "punctuation:no_comma", NeverPasses)
        graded = generative.grade_completion(
            ifeval_item(), "a response", generative.GenerationConfig(stop_sequences=())
        )

        assert graded.correct is False
        assert graded.metadata["grader_detail"] == {
            "instruction_id_list": ["punctuation:no_comma"],
            "strict": [False],
        }


class TestModalityGuard:
    def test_a_verifier_scored_bank_passes_the_generative_guard(self) -> None:
        """It has no gold, and the guard used to require one of every generative item."""
        request = grading.GradingRequest(dataset=DATASET, modality=grading.GENERATIVE)
        grading.check_bank_modality(request, [ifeval_item()])

    def test_an_item_stripped_of_its_constraints_is_still_refused(self) -> None:
        """Admitting it would score it incorrect for every model, silently."""
        stripped = BenchmarkItem(
            item_id="0",
            question="q",
            choices=(),
            gold_index=-1,
            metadata={"answer_type": "ifeval_strict", "modality": "generative"},
        )
        request = grading.GradingRequest(dataset=DATASET, modality=grading.GENERATIVE)
        with pytest.raises(grading.ModalityMismatch, match="answer_type"):
            grading.check_bank_modality(request, [stripped])

    def test_an_unknown_answer_type_is_refused_before_a_checkpoint_loads(self) -> None:
        unknown = BenchmarkItem(
            item_id="0",
            question="q",
            choices=(),
            gold_index=-1,
            metadata={"answer_type": "essay_rubric", "gold_answer": "x"},
        )
        request = grading.GradingRequest(dataset=DATASET, modality=grading.GENERATIVE)
        with pytest.raises(grading.ModalityMismatch):
            grading.check_bank_modality(request, [unknown])


class TestPromptAndSampling:
    def test_the_prompt_is_the_instruction_verbatim(self) -> None:
        """Any framing is an extra constraint the verifiers were never told about."""
        item = ifeval_item(question="Write three sections. Do not use commas.")
        config = generative.GenerationConfig(num_fewshot=0, prompt_style="ifeval")

        assert generative.format_generative_prompt(item, config) == item.question

    def test_a_fewshot_block_behind_the_ifeval_template_is_refused(self, monkeypatch) -> None:
        """Silently borrowing gsm8k's block would prompt IFEval as a Q/A benchmark."""
        monkeypatch.setitem(
            generative.FEWSHOT_SOURCES, "gsm8k", lambda: ({"question": "q", "answer": "a"},)
        )
        config = generative.GenerationConfig(num_fewshot=1, prompt_style="ifeval")

        with pytest.raises(ValueError, match="0-shot only"):
            generative.format_generative_prompt(ifeval_item(), config)

    def test_the_committed_config_matches_the_leaderboard_task(self, tmp_path: Path) -> None:
        """The settings a run would actually use, read off config.yaml.

        Every one of these that decides whether an item passes is
        lm-evaluation-harness's ``leaderboard_ifeval`` verbatim: ``doc_to_text`` is the
        bare ``prompt`` field, ``num_fewshot: 0``, and generation kwargs of
        ``until: []``, ``do_sample: false``. That includes the completion framing, which
        the harness applies outside the task through ``--apply_chat_template`` rather
        than in it.

        The budget is the exception and is pinned separately below, because lm-eval
        sends 1280 and this style does not.
        """
        config = _resolved_generation_config(DATASET)

        assert config.num_fewshot == 0
        assert config.prompt_style == "ifeval"
        assert config.chat_format is False
        assert config.stop_sequences == ()

    def test_the_budget_covers_the_largest_length_constraint_the_bank_states(self) -> None:
        """1536, not lm-eval's 1280, and the difference is deliberate.

        IFEval has no reference answer to measure, so the budget comes from what the
        items demand: the largest explicit ``number_words`` constraint in the bank is
        900 words, about 1,161 tokens. 1280 leaves 10% of headroom over a constraint the
        grader checks directly, and a response cut short of its own word count fails
        ``length_constraints:number_words`` as though the model had ignored it.

        Unlike a prompt change this cannot reframe an item -- a longer budget only lets
        a response finish -- which is why it is the one place this bank departs from
        lm-eval.
        """
        assert _resolved_generation_config(DATASET).max_new_tokens == 1536

    def test_no_stop_sequence_survives_a_multi_paragraph_answer(self) -> None:
        """A blank-line stop would cut most responses into a length-constraint failure."""
        config = _resolved_generation_config(DATASET)
        answer = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        assert generative.truncate_at_stop(answer, config.stop_sequences) == answer


def _resolved_generation_config(dataset: str) -> generative.GenerationConfig:
    """The generation settings the committed config.yaml resolves to for ``dataset``."""
    settings = grading._apply_generation_overrides(
        grading.GradingSettings(), UniMcqStyle().generation_settings, dataset=dataset
    )
    return settings.generation


class FakeChatTokenizer:
    """A tokenizer that renders a chat turn the way an instruction-tuned one would."""

    pad_token_id = None
    eos_token_id = 7
    chat_template = "{{ messages }}"

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = True, add_generation_prompt: bool = False
    ) -> str:
        self.last = (messages, tokenize, add_generation_prompt)
        return f"<|user|>{messages[0]['content']}<|assistant|>"

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        from .test_generative_grading import FakeIds

        return {"input_ids": FakeIds(text.split(" "))}

    def decode(self, tokens: list[str], skip_special_tokens: bool = False) -> str:
        return " ".join(tokens)


class TestChatFormatting:
    """The one part of the pipeline that is the checkpoint's rather than the bank's.

    Exercised through ``gpqa``'s settings rather than ``ifeval``'s, which is a change of
    example and not of subject. IFEval is scored in completion format now -- the
    leaderboard it was harvested from templated its chat submissions and not its
    pretrained ones, so the completion half is the one a base checkpoint can be measured
    against -- which leaves gpqa as the only chat-format bank here and therefore the
    only honest way to drive this path. The two tests at the bottom hold the other side:
    that ifeval really does bypass the template now, and that gpqa really does still
    refuse a checkpoint without one.
    """

    @pytest.fixture
    def fake_stack(self, monkeypatch):
        import contextlib

        from .test_generative_grading import FakeModel, FakeTokenizer

        loaded: list[str] = []

        def install(tokenizer: object) -> FakeModel:
            model = FakeModel()
            torch = types.ModuleType("torch")
            torch.no_grad = contextlib.nullcontext
            transformers = types.ModuleType("transformers")
            transformers.AutoTokenizer = types.SimpleNamespace(
                from_pretrained=lambda *a, **k: tokenizer
            )

            def load_model(*args: object, **kwargs: object) -> FakeModel:
                loaded.append("weights")
                return model

            transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=load_model)
            transformers.set_seed = lambda seed: None
            monkeypatch.setitem(sys.modules, "torch", torch)
            monkeypatch.setitem(sys.modules, "transformers", transformers)
            return model

        install.plain = FakeTokenizer
        install.loaded = loaded
        return install

    @staticmethod
    def chat_config() -> generative.GenerationConfig:
        """The shape gpqa was scored in until its modality changed.

        Constructed rather than resolved from ``config.yaml``, because no bank selects
        it any more and the guard still has to hold for the next one that would.
        """
        return generative.GenerationConfig(
            num_fewshot=0,
            prompt_style="gpqa",
            chat_format=True,
            system_prompt_source="gpqa",
            stop_sequences=(),
        )

    def test_the_prompt_goes_through_the_checkpoints_template(self, fake_stack) -> None:
        tokenizer = FakeChatTokenizer()
        fake_stack(tokenizer)
        config = generative.GenerationConfig(
            num_fewshot=0, prompt_style="gpqa", chat_format=True, stop_sequences=()
        )

        generative._HFCompleter(Path("/ckpt"), config)("Write a summary.")

        messages, tokenize, add_generation_prompt = tokenizer.last
        assert messages == [{"role": "user", "content": "Write a summary."}]
        assert tokenize is False
        assert add_generation_prompt is True

    def test_a_checkpoint_without_a_chat_template_is_refused(self, fake_stack) -> None:
        """Sending the prompt raw would still complete, still grade, and still report."""
        fake_stack(fake_stack.plain())

        with pytest.raises(ValueError, match="defines no chat template"):
            generative._HFCompleter(Path("/ckpt"), self.chat_config())

    def test_it_is_refused_before_the_weights_are_loaded(self, fake_stack) -> None:
        """The tokenizer is cheap and the model is not; the check goes between them."""
        fake_stack(fake_stack.plain())

        with pytest.raises(ValueError, match="defines no chat template"):
            generative._HFCompleter(Path("/ckpt"), self.chat_config())
        assert fake_stack.loaded == []

    def test_a_completion_bank_never_touches_the_template(self, fake_stack) -> None:
        """gsm8k and MATH prompts carry their own framing and must arrive unwrapped."""
        tokenizer = FakeChatTokenizer()
        fake_stack(tokenizer)
        tokenizer.last = None
        config = generative.GenerationConfig(num_fewshot=0, chat_format=False)

        generative._HFCompleter(Path("/ckpt"), config)("Question: how many?\nAnswer:")

        assert tokenizer.last is None

    def test_a_checkpoint_with_no_template_can_now_be_scored_on_ifeval(self, fake_stack) -> None:
        """The whole point of the flip, driven through the real completer.

        A tokenizer carrying no ``chat_template`` is what a base checkpoint has --
        SmolLM2-135M among them -- and while ifeval was chat-format this raised before
        the weights were reached, which put every generative bank here out of reach of a
        base model. Built from the committed ``config.yaml`` rather than a constructed
        config, so it fails if the setting is flipped back rather than passing on a
        hand-written copy of what the file used to say.
        """
        fake_stack(fake_stack.plain())
        config = _resolved_generation_config(DATASET)

        generative._HFCompleter(Path("/ckpt"), config)("Write a summary with no commas.")

        assert config.chat_format is False
        assert fake_stack.loaded == ["weights"]

    def test_the_chat_shape_is_still_refused_for_anything_that_asks_for_it(
        self, fake_stack
    ) -> None:
        """The guard outlived both banks that needed it, and has to keep working.

        GPQA was this test's subject until 2026-08-08: it kept ``chat_format`` after
        ifeval gave it up, so a base checkpoint was refused here and the bank was
        unreachable. That was resolved by changing its modality rather than its framing
        -- lm-eval scores GPQA as a log-likelihood ranking, which needs no template --
        so no bank sets ``chat_format`` today. The refusal still matters, because the
        thing it prevents is silent: sending the prompt raw would complete, grade and
        report, with the standing instruction simply missing.
        """
        fake_stack(fake_stack.plain())

        assert not any(
            grading._apply_generation_overrides(
                grading.GradingSettings(), UniMcqStyle().generation_settings, dataset=name
            ).generation.chat_format
            for name, spec in SUPPORTED.items()
            if spec.modality == grading.GENERATIVE
        )
        with pytest.raises(ValueError, match="defines no chat template"):
            generative._HFCompleter(Path("/ckpt"), self.chat_config())


class TestVendoredBank:
    """The committed bank, asserted against what the join was supposed to produce."""

    @pytest.fixture
    def manifest(self) -> dict:
        path = CALIBRATED_DATASETS / DATASET / "manifest.json"
        if not path.is_file():
            pytest.skip("ifeval has not been vendored")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_content_hashed_bridge_was_used(self, manifest) -> None:
        """Neither of the two wrong joins this bank had available is what was read.

        The bank once joined through the bare integer in
        ``ATLAS/ifeval/atlas_idx_to_question_id.csv``, and that integer is a position in
        an enumeration performed at harvest time rather than anything the prompt says.
        Its two available substitutes are worse -- the composite ``item_id_map`` joins
        nothing, and ``scenarios.jsonl`` is a dense 0..540 run over the same prompts in a
        different order, so it joins near-totally against mostly wrong questions.
        """
        assert manifest["bridge_kind"] == "content_hash"
        assert manifest["bridge_path"].endswith("bridges/ifeval.csv")
        assert manifest["bridge_in_repo"] is True

    def test_the_counts_are_the_ones_upstream_reports(self, manifest) -> None:
        assert manifest["upstream_bank_rows"] == 535
        assert manifest["bridge_rows"] == 535
        assert manifest["dropped"]["non_positive_discrimination"] == 24
        assert manifest["dropped"]["not_in_task"] == 0
        assert manifest["items"] == 511

    def test_item_ids_are_content_hashes(self, manifest) -> None:
        """A bare integer here would mean the superseded positional bridge was read.

        Checked as 16 hex digits rather than as "not a number", because roughly one
        content hash in 1,800 happens to be all decimal digits and a bank of 511 would
        fail that on about a quarter of re-vendors.
        """
        items = _vendored_items()
        assert all(re.fullmatch(r"[0-9a-f]{16}", item.item_id) for item in items)
        assert len({item.item_id for item in items}) == len(items)

    def test_every_item_carries_its_constraints_and_no_gold(self, manifest) -> None:
        for item in _vendored_items():
            assert item.metadata["answer_type"] == "ifeval_strict"
            assert item.metadata["instruction_id_list"]
            assert len(item.metadata["kwargs"]) == len(item.metadata["instruction_id_list"])
            assert "gold_answer" not in item.metadata


def _vendored_items() -> list[BenchmarkItem]:
    """The committed IFEval items, loaded through the real loader."""
    from ....common.benchmark_download import load_items_from_jsonl

    path = CALIBRATED_DATASETS / DATASET / "items.jsonl"
    if not path.is_file():
        pytest.skip("ifeval has not been vendored")
    return list(load_items_from_jsonl(path, name=DATASET).items)


def run_real_bank(true_theta: float, *, max_items: int = 40) -> dict:
    """A full CAT over the committed 511-item bank, tokens simulated and nothing else."""
    style = UniMcqStyle()
    bank = style.download_benchmark(DATASET)
    irt = style.load_irt_params(DATASET)
    config = _resolved_generation_config(DATASET)
    taker = SimGenerativeTaker(
        true_theta,
        vendored_params(DATASET),
        bank.items,
        recover_question=lambda prompt: prompt,
        answer=ifeval_answer,
    )
    report = cat_loop.run_cat(
        style,
        bank=bank,
        irt_bank=irt,
        model=generative.GenerativeScorer(taker, config),
        se_threshold=0.3,
        max_items=max_items,
    )
    return report.to_dict()


class TestAbilityRecoveryOnTheRealBank:
    """Theta recovery over the committed bank, through the real engine and grader."""

    run = staticmethod(run_real_bank)

    def test_a_session_converges_and_stops_on_precision(self, stub_ifbench) -> None:
        """Well inside the 40-item cap, though faster here than upstream's 27.

        A taker drawn from the bank's own 3PL fits it perfectly and collapses the
        posterior at the ``min_items`` floor; the cross-validated 27 is over real
        models, whose responses misfit. What this pins is that the cap is not what
        ends the session.
        """
        report = self.run(0.5)

        assert report["metadata"]["stop_reason"] == "precision_reached"
        assert report["metadata"]["standard_error"] <= 0.3
        assert report["metadata"]["bank_size"] == 511
        assert np.isfinite(report["ability"]["theta"])

    @pytest.mark.parametrize("true_theta", [-1.5, 0.0, 1.5])
    def test_the_estimate_lands_near_the_truth(self, stub_ifbench, true_theta: float) -> None:
        report = self.run(true_theta)
        assert report["metadata"]["theta"] == pytest.approx(true_theta, abs=0.6)

    def test_theta_recovers_monotonically(self, stub_ifbench) -> None:
        """A stronger simulated taker must produce a strictly higher estimate.

        The property that matters for a diagnostic. The absolute scale is the bank's
        and can shift with the grader; the ordering is what a comparison between two
        checkpoints actually rests on. The ladder stays inside the bank's range, which
        for IFEval reaches down to about -1.5 -- far lower than MATH's, which is what a
        median difficulty of 1.03 against MATH's 2.78 means in practice.
        """
        estimates = [self.run(theta)["metadata"]["theta"] for theta in (-1.5, -0.5, 0.5, 1.5)]

        assert all(b > a for a, b in pairwise(estimates)), estimates

    def test_the_report_names_the_grader_that_produced_it(self, stub_ifbench) -> None:
        report = self.run(0.5)
        note = report["metadata"]["scoring_note"]

        assert "prompt_level_strict_acc" in note
        assert "last number" not in note
        assert report["metadata"]["modality"] == "generative"

    def test_every_response_records_the_constraints_it_was_judged_on(self, stub_ifbench) -> None:
        for response in self.run(0.5)["responses"]:
            assert response["chosen_index"] == -1
            assert response["metadata"]["grader"] == "ifeval_prompt_strict"
            assert response["metadata"]["grader_detail"]["strict"]


class TestASessionSurvivesUngradableItems:
    """One malformed item must cost that item, not the whole checkpoint's diagnostic.

    Driven through :func:`run_real_bank` on purpose: the point is that the *same* run,
    over the same committed bank, still reaches a written report when grading stops
    producing outcomes. Before this the first such item raised out through the CAT
    engine into ``runner.main``, which logged and returned 1 -- after the checkpoint had
    been staged and the GPU booted, and with nothing written down about how far the
    session got.
    """

    run = staticmethod(run_real_bank)

    @pytest.fixture
    def silent_scorer(self, monkeypatch, stub_ifbench):
        """An ``IFEvalScorer`` that verifies nothing, as a missing registry would."""
        from olmo_eval.common.types import Instance, LMOutput

        class VerifiesNothing:
            def score(self, instance: object, output: object) -> float:
                output.metadata["ifeval"] = {"strict": [], "loose": []}
                return 0.0

        monkeypatch.setattr(
            generative, "_ifeval_scoring", lambda: (VerifiesNothing, Instance, LMOutput)
        )

    def test_the_report_is_still_written(self, silent_scorer) -> None:
        report = self.run(0.5)

        assert report["metadata"]["n_items_administered"] > 0
        assert np.isfinite(report["ability"]["theta"])

    def test_every_zero_is_marked_as_not_the_models(self, silent_scorer) -> None:
        """Otherwise the pattern is indistinguishable from a checkpoint answering wrong."""
        responses = self.run(0.5)["responses"]

        assert responses
        assert all(r["correct"] is False for r in responses)
        assert all(r["metadata"][generative.UNGRADABLE_KEY] is True for r in responses)
        assert all(r["metadata"][generative.UNGRADABLE_REASON_KEY] for r in responses)

    def test_the_report_says_the_theta_is_not_about_the_checkpoint(self, silent_scorer) -> None:
        """A floor theta with a healthy standard error is what this run would look like."""
        metadata = self.run(0.5)["metadata"]
        block = metadata["ungradable"]

        assert block["count"] == metadata["n_items_administered"]
        assert block["rate"] == 1.0
        assert "harness rather than the checkpoint" in block["alert"]

    def test_a_broken_dependency_is_still_an_error_rather_than_a_floor(
        self, monkeypatch, stub_ifbench
    ) -> None:
        """The other side of the line: no verifiers at all is not 511 wrong answers.

        An ungradable item is one the bank failed to describe. A grader that cannot run
        is a different claim -- every item would score 0, the report would carry a
        confident floor theta, and the ungradable count would be the only thing saying
        so. That is too quiet for a fault whose fix is installing a package, so the
        import failure keeps propagating.
        """
        monkeypatch.delitem(sys.modules, "ifbench")
        monkeypatch.setattr(
            generative,
            "_ifeval_scoring",
            lambda: (_ for _ in ()).throw(RuntimeError("ifbench is not importable")),
        )

        with pytest.raises(RuntimeError, match="ifbench"):
            self.run(0.5)


#: A response used to drive every verifier in the bank at once: no commas, several
#: sections, and long enough that a minimum-word constraint has a chance. It is not
#: expected to satisfy most items -- IFEval prompts ask for specific things -- only to be
#: something every verifier can reach a real verdict about.
NEUTRAL_RESPONSE = (
    "Section 1\n\nHere is a plain answer written without any commas.\n\n"
    "Section 2\n\nIt continues for a while so that a length constraint has a chance of "
    "passing and it keeps going with more words to be safe about minimum word counts."
)


class TestAgainstTheRealVerifiers:
    """Skipped without ``ifbench``. Nothing here can be simulated and still mean anything.

    Two dependencies, not one, and the second is easy to miss now that the first is
    installed. Every test here runs olmo-eval's real ``IFEvalScorer`` over the real
    registry, so a bare diagnostics-only checkout has to skip on ``olmo_eval`` as well --
    otherwise installing ``ifbench`` turns what used to be a clean skip into a wall of
    failures about a package these tests never claimed to need.
    """

    @pytest.fixture(autouse=True)
    def _needs_olmo_eval(self) -> None:
        pytest.importorskip(
            "olmo_eval.common.scorers",
            reason="IFEval grading runs olmo-eval's real IFEvalScorer; run with PYTHONPATH=src",
        )

    @pytest.fixture
    def real_ifbench(self):
        return pytest.importorskip(
            "ifbench",
            reason="ifbench (a git-URL dependency) is not installed in this environment",
        )

    @pytest.fixture
    def real_ifbench_over_the_bank(self, real_ifbench):
        """``ifbench`` plus the NLTK corpora its verifiers load on first use.

        A second prerequisite that the package alone does not satisfy, and one worth
        skipping on rather than failing over. Several IFEval verifiers -- the ones that
        count sentences or capitalised words -- tokenize with NLTK and fetch ``punkt_tab``
        and ``averaged_perceptron_tagger_eng`` the first time they run. The fetch is
        silent when it works and raises a ``LookupError`` mid-grade when it does not,
        which on a GPU run is an aborted session rather than a missing package. Probing
        one such item here turns that into a named skip with the fix in it.
        """
        item = ifeval_item(
            ("change_case:capital_word_frequency",),
            ({"capital_relation": "less than", "capital_frequency": 2},),
        )
        try:
            generative.get_answer_grader("ifeval_strict").grade_item("A short answer.", item)
        except LookupError as exc:
            pytest.skip(
                f"an ifbench verifier needs NLTK data this machine does not have and "
                f"could not download ({str(exc).splitlines()[1].strip()}); run "
                f"python -c \"import nltk; nltk.download('punkt_tab'); "
                f"nltk.download('averaged_perceptron_tagger_eng')\""
            )
        return real_ifbench

    def test_a_comma_free_response_passes_and_a_comma_fails(self, real_ifbench) -> None:
        grader = generative.get_answer_grader("ifeval_strict")
        item = ifeval_item(("punctuation:no_comma",), ({},))

        assert grader.grade_item("No commas anywhere in this sentence.", item).correct is True
        assert grader.grade_item("There is, unmistakably, a comma.", item).correct is False

    def test_strict_and_loose_disagree_where_the_metric_choice_matters(self, real_ifbench) -> None:
        """Markdown stars are stripped by the loose variants and not by strict scoring.

        The bank was calibrated on strict, so this grader has to be the stricter of the
        two; if it ever started reading the loose list every item would look easier
        than its difficulty.
        """
        grader = generative.get_answer_grader("ifeval_strict")
        item = ifeval_item(("punctuation:no_comma",), ({},))

        assert grader.grade_item("*A starred, comma-bearing line.*", item).correct is False

    def test_the_registry_covers_every_instruction_the_bank_names(self, real_ifbench) -> None:
        """A missing verifier would raise mid-session on whichever item selected it."""
        from ifbench import instructions_registry

        named = {
            instruction
            for item in _vendored_items()
            for instruction in item.metadata["instruction_id_list"]
        }
        assert named <= set(instructions_registry.INSTRUCTION_DICT)

    def test_every_item_in_the_bank_reaches_a_real_verdict(self, real_ifbench_over_the_bank):
        """The registry resolving is not the same claim as every verifier running.

        A verifier can be present and still fail on the arguments this bank hands it --
        a missing corpus, an argument name the vendored kwargs spell differently -- and
        the failure surfaces only on the item that selects it, mid-session. Driving all
        511 items through the real grader turns that from a run-time surprise into a
        property of the committed bank.
        """
        grader = generative.get_answer_grader("ifeval_strict")
        verdicts = [grader.grade_item(NEUTRAL_RESPONSE, item) for item in _vendored_items()]

        assert len(verdicts) == 511
        assert all(verdict.ungradable_reason is None for verdict in verdicts)
        for verdict, item in zip(verdicts, _vendored_items(), strict=True):
            assert len(verdict.detail["strict"]) == len(item.metadata["instruction_id_list"])

    def test_the_verifiers_discriminate_rather_than_refusing_everything(
        self, real_ifbench_over_the_bank
    ) -> None:
        """A registry that rejected every response would look exactly like a weak model.

        One canned response passes some items and fails most, which is what real
        verification looks like; a bank where the same response scored 0 everywhere
        would produce a floor theta with a healthy standard error and no other sign.
        """
        grader = generative.get_answer_grader("ifeval_strict")
        outcomes = [grader.grade_item(NEUTRAL_RESPONSE, item).correct for item in _vendored_items()]

        assert any(outcomes)
        assert not all(outcomes)

    def test_a_simulated_session_runs_on_the_real_verifiers(
        self, real_ifbench_over_the_bank
    ) -> None:
        """The whole harness end to end with nothing stubbed but the tokens.

        Everywhere else in this file the verifier bodies are replaced, because a stub is
        what lets the grader, the CAT and the report be exercised on a machine without
        ``ifbench``. This is the run that says the stub was standing in for something
        that works: the same committed bank, the same engine, and the real IFBench
        registry deciding every item.

        It asserts mechanics rather than recovery. The simulated taker's responses are
        not built to satisfy any particular constraint, so which items come back correct
        is not a statement about the bank -- what matters is that every administered item
        produced one real verdict per instruction it names, and that no zero in the
        response pattern was fabricated by the harness.
        """
        report = run_real_bank(0.5)

        assert report["metadata"]["n_items_administered"] >= UniMcqStyle().min_items
        assert report["metadata"]["ungradable"]["count"] == 0
        assert np.isfinite(report["ability"]["theta"])
        for response in report["responses"]:
            detail = response["metadata"]["grader_detail"]
            assert response["metadata"]["grader"] == "ifeval_prompt_strict"
            assert len(detail["strict"]) == len(detail["instruction_id_list"])

    def test_that_session_did_not_reach_the_stub_registry(self, real_ifbench_over_the_bank) -> None:
        """The claim the test above rests on: nothing replaced ``ifbench`` in sys.modules.

        The stub is installed with ``monkeypatch.setitem(sys.modules, ...)``, so a
        fixture ordering mistake would leave it in place and the session would pass on
        the marker text instead of on verified constraints -- which is precisely the
        difference this class exists to establish.
        """
        assert sys.modules["ifbench"].__spec__ is not None
        assert "site-packages" in (sys.modules["ifbench"].__file__ or "")
        assert FOLLOWED not in NEUTRAL_RESPONSE


class TestTheScoringNote:
    """A report's note has to describe the grader that produced its theta, not a modality."""

    def test_each_grader_supplies_its_own_clause(self) -> None:
        for name in ("last_number_exact_match", "math_latex_equivalence", "ifeval_prompt_strict"):
            assert generative.grader_note(name) != f"graded by {name}"

    def test_every_registered_grader_has_one(self) -> None:
        """Otherwise a new bank's reports would describe it only by its internal name."""
        for grader in generative.ANSWER_GRADERS.values():
            assert grader.name in generative.GRADER_NOTES

    def test_the_mcq_note_names_its_own_ranking_rule(self) -> None:
        """The two modalities fill their template from different places; neither leaks."""
        note = style_mod._scoring_note(
            grading.MCQ, [], score_normalization=inference.DEFAULT_SCORE_NORMALIZATION
        )
        assert "not length-normalized" in note
        assert "{scoring}" not in note
        assert "{grading}" not in note

    def test_a_session_that_administered_nothing_says_so_rather_than_naming_a_grader(
        self,
    ) -> None:
        """No response means no evidence of which grader ran, and none is invented."""
        note = style_mod._scoring_note(grading.GENERATIVE, [])
        assert style_mod.NO_GRADER_NOTE in note
        assert "{grading}" not in note


class TestWithoutOlmoEval:
    def test_the_grader_says_where_the_verifiers_live(self, monkeypatch) -> None:
        """A diagnostics-only checkout must get a message, not an ImportError."""
        monkeypatch.setitem(sys.modules, "olmo_eval.common.scorers", None)
        with pytest.raises(RuntimeError, match="IFEvalScorer"):
            generative.get_answer_grader("ifeval_strict").grade_item("x", ifeval_item())
