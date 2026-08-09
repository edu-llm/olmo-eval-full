"""The native OLMo-core generative completer, driven without torch, olmo_core or a GPU.

What these can and cannot establish is the same split ``test_olmo_core_scoring`` states,
and it is worth restating because this file is about a *decode* and a decode is the thing
least like its fake. They pin the wiring and the contract: which helpers the loader calls
and with what, that the prompt echo is sliced off the return, that greedy decoding and a
disabled KV cache are asked for explicitly, that the context window is read from the
checkpoint's own config and refused rather than guessed, and -- the bulk of the file --
that every attribute the budget machinery reads off a completer is published. They cannot
establish that ``generate_batch`` behaves as the fake does; probe ``run_019fe316-0e47``
did that on a real L4, and the fake is shaped from what it reported rather than from the
type stubs.

**The fakes are the scoring file's**, imported rather than re-declared. That is deliberate
and load-bearing twice over. The two paths load the same checkpoint through the same
helpers, so a fake that drifted between them would let one path's assumptions rot
unobserved; and ``FakeGenerationConfig`` there mirrors the real
``GenerationConfig.__post_init__`` validator, which is the check that killed run_019fe265
on a card and is the whole reason the pad id is fabricated. Reusing it means the pad/eos
question is answered here by the same code that answers it there.

**Why so much of this file is about attributes that are merely absent.** Five things a
completer publishes are read by machinery it cannot see, and three of them fail silently
when missing: budgets are computed and then not passed, the context clamp switches off
entirely, the leak-catching stop is weakened. None of those produce an error, a warning
worth noticing, or a report that looks wrong -- they produce a theta. So every one of the
five is asserted present, and every one is also *removed* and shown to change an
observable outcome, because an assertion that an attribute exists is worth very little
next to a demonstration of what its absence costs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from .... import runner
from ....base import BenchmarkItem
from ....common import convert, generative, grading, inference, s3_io
from .. import resolve
from .conftest import SimScorer, write_bank
from .test_olmo_core_scoring import (
    FakeTokenizer,
    Tensor,
    install_fake_olmo_core,
    write_checkpoint,
)

#: This checkpoint family's end-of-text spelling and id, as
#: ``dataset.tokenizer.identifier`` resolves them: ``HuggingFaceTB/SmolLM2-135M``, which
#: spells the token ``<|endoftext|>`` at id **0**.
#:
#: Zero is the trap in this area rather than an incidental detail. ``bool(0)`` is False,
#: so every ``eos_token_id or None`` discards precisely the checkpoint the code exists
#: for, and a fake with a truthy id would let that bug through.
EOS_TEXT = "<|endoftext|>"
EOS_ID = 0

#: The window the checkpoints below declare unless a test says otherwise. 2048 because
#: that is this family's trained ``sequence_length`` -- the same number
#: ``hf_config_patch.DEFAULT_MAX_POSITION_EMBEDDINGS`` writes into a converted config,
#: arrived at here by reading the checkpoint instead of by importing the constant.
SEQUENCE_LENGTH = 2048


class GeneratingTokenizer(FakeTokenizer):
    """The scoring fake plus the four things only generation reads off a tokenizer.

    ``eos_token`` round-trips to ``eos_token_id`` because the real one does -- SmolLM2
    registers it as an added special token, ``normalized: false`` and ``special: true`` --
    and :func:`generative.resolve_eos_token` refuses to teach a spelling that does not.
    Every other word keeps the parent's ids, which start at 1, so an id of 0 in an encoded
    string is an end-of-text token and never an artefact of the stub.
    """

    name_or_path = "HuggingFaceTB/SmolLM2-135M"
    eos_token = EOS_TEXT
    eos_token_id = EOS_ID

    #: What ``len(tokenizer)`` reports, for the vocabulary-size fact in the report.
    #: SmolLM2-135M's, so the number in a test artifact is the number a real run records.
    vocab_size = 49152

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        ids: list[int] = []
        for index, chunk in enumerate(text.split(self.eos_token)):
            if index:
                ids.append(self.eos_token_id)
            ids.extend(self._token_id(word) for word in chunk.split())
        if add_special_tokens:
            ids = [*self.leading, *ids, *self.trailing]
        return {"input_ids": Tensor([ids]) if return_tensors == "pt" else ids}

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Model ``skip_special_tokens`` faithfully, because a leaked marker turns on it."""
        kept = [value for value in ids if not (skip_special_tokens and value == self.eos_token_id)]
        return " ".join(str(value) for value in kept)


class ChatTokenizer(GeneratingTokenizer):
    """A tokenizer that has a chat template, for the one bank that asks for one."""

    chat_template = "{{ messages }}"

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: object
    ) -> str:
        return " ".join(f"<{turn['role']}> {turn['content']}" for turn in messages)


def math_config(**overrides: Any) -> generative.GenerationConfig:
    """The shipped ``leaderboard_math`` generation block, as ``config.yaml`` sets it.

    Written out rather than read off the style so these tests state the configuration
    they depend on, and taken from the real one rather than invented so what the ladder
    is exercised against is the four Minerva exemplars a real run sends -- which is the
    whole reason MATH is the bank that overflows.
    """
    settings: dict[str, Any] = {
        "checkpoint_kind": "olmo_core",
        "num_fewshot": 4,
        "fewshot_source": "leaderboard_math",
        "prompt_style": "leaderboard_math",
        "max_new_tokens": 1024,
        "stop_sequences": ("Problem:", "problem:"),
    }
    settings.update(overrides)
    return generative.GenerationConfig(**settings)


def math_item(words: int = 40, item_id: str = "m0") -> BenchmarkItem:
    """A MATH-shaped item whose stem is ``words`` long, for driving the ladder."""
    return BenchmarkItem(
        item_id=item_id,
        question=" ".join(f"w{index}" for index in range(words)),
        choices=(),
        gold_index=-1,
        metadata={"gold_answer": "42"},
    )


def native_checkpoint(
    tmp_path: Path,
    *,
    sequence_length: int | None = SEQUENCE_LENGTH,
    model: dict[str, Any] | None = None,
    name: str = "step305176",
) -> Path:
    """A raw checkpoint declaring its trained sequence length the way this family does."""
    dataset = {} if sequence_length is None else {"sequence_length": sequence_length}
    return write_checkpoint(tmp_path / name, model=model, dataset=dataset)


class Loaded:
    """One loaded generative scorer plus everything the load touched."""

    def __init__(self, scorer: Any, module: Any, calls: dict[str, Any]) -> None:
        self.scorer = scorer
        self.module = module
        self.calls = calls

    @property
    def completer(self) -> Any:
        return self.scorer.complete


def load(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
    *,
    config: generative.GenerationConfig | None = None,
    tokenizer: FakeTokenizer | None = None,
) -> Loaded:
    """Build the scorer through the real registry with only ``_import_olmo_core`` faked."""
    module, calls = install_fake_olmo_core(
        monkeypatch, tokenizer=tokenizer if tokenizer is not None else GeneratingTokenizer()
    )
    scorer = generative.load_generative_model(checkpoint, config or math_config())
    return Loaded(scorer, module, calls)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    return native_checkpoint(tmp_path)


# ---------------------------------------------------------------------------
# The registry, and the guard in front of it
# ---------------------------------------------------------------------------


class TestTheRegistryReachesTheNativeCompleter:
    """``olmo_core`` is a row in :data:`generative.GENERATIVE_BACKENDS`, not a branch."""

    def test_the_dispatch_returns_a_scorer_over_the_native_completer(
        self, monkeypatch, checkpoint
    ) -> None:
        loaded = load(monkeypatch, checkpoint)
        assert isinstance(loaded.scorer, generative.GenerativeScorer)
        assert isinstance(loaded.completer, generative._OlmoCoreCompleter)

    def test_the_registry_is_what_names_it(self) -> None:
        assert set(generative.GENERATIVE_BACKENDS) == {"hf", "olmo_core"}

    def test_the_kind_guard_lets_a_generative_bank_past(self) -> None:
        """It runs before ``resolve_checkpoint``, so it decides this before a download."""
        request = grading.GradingRequest(dataset="leaderboard_math", modality=grading.GENERATIVE)
        settings = grading.GradingSettings(
            generation=generative.GenerationConfig(checkpoint_kind="olmo_core")
        )
        grading.check_checkpoint_kind(request, settings)

    def test_the_loader_is_reused_rather_than_reimplemented(
        self, monkeypatch, checkpoint
    ) -> None:
        """The tokenizer comes through ``dataset.tokenizer.identifier``, as on the MCQ side.

        A raw checkpoint ships no tokenizer files, so a loader that read the directory --
        the obvious thing, and what the HuggingFace completer does -- would find nothing.
        Asserting the resolved path is what says this went through olmo-eval's helpers
        rather than through a second implementation that happens to work today.
        """
        loaded = load(monkeypatch, checkpoint)
        assert loaded.calls["tokenizer_path"] == "allenai/dolma2-tokenizer"
        assert loaded.calls["from_checkpoint"]["checkpoint_dir"] == str(checkpoint)
        assert loaded.calls["from_checkpoint"]["device"] == "cpu"

    def test_the_pad_id_is_fabricated_distinct_from_the_checkpoint_s_eos(
        self, monkeypatch, checkpoint
    ) -> None:
        """The objection the stub raised, refuted by the loader it refused to reuse.

        The stub held that generation needs an end-of-text id distinct from the pad id and
        that this family has neither. It confused two ids: ``GenerationConfig.validate``
        rejects only ``pad == eos``, so fabricating the *pad* satisfies it and leaves the
        real end-of-text at 0 to stop on. The config asserted here is built by the shared
        fake that mirrors that validator, so if the rule ever widens this fails rather
        than the next GPU run.
        """
        loaded = load(monkeypatch, checkpoint)
        passed = loaded.calls["from_checkpoint"]["generation_config"]
        assert passed.eos_token_id == EOS_ID
        assert passed.pad_token_id != passed.eos_token_id


# ---------------------------------------------------------------------------
# The completer contract
# ---------------------------------------------------------------------------


class TestTheCompleterContract:
    """The five attributes the budget machinery reads, present and load-bearing.

    Each is asserted twice: that the completer publishes it, and that removing it changes
    something a report would show. The second half is the one worth having. Three of the
    five fail silently -- no exception, no missing field, just different numbers -- so a
    test that only checked presence would keep passing through the exact regression it
    was written to catch.
    """

    def test_the_token_counter_is_published_and_counts_content_only(
        self, monkeypatch, checkpoint
    ) -> None:
        """:data:`generative.TOKEN_COUNTER_ATTR`, and ``add_special_tokens=False``.

        Content only, because every caller measures what the model has to produce or
        echo. This family's tokenizer is resolved from an identifier and may prepend a
        BOS, which would inflate the words-to-tokens ratio on exactly the short strings
        the ratio is most sensitive to.
        """
        loaded = load(monkeypatch, checkpoint, tokenizer=GeneratingTokenizer(leading=(21,)))
        counter = getattr(loaded.completer, generative.TOKEN_COUNTER_ATTR)
        assert counter("alpha beta gamma") == 3

    def test_a_completer_without_a_token_counter_is_refused_at_load(
        self, monkeypatch, checkpoint
    ) -> None:
        """The one of the five that is caught, and it is caught by the loader.

        ``require_live_tokenizer`` runs on the scorer this backend just built, so the
        native path inherits the refusal rather than needing its own.
        """
        monkeypatch.delattr(generative._OlmoCoreCompleter, generative.TOKEN_COUNTER_ATTR)
        with pytest.raises(RuntimeError, match="publishes no 'count_tokens'"):
            load(monkeypatch, checkpoint)

    def test_the_budget_flag_is_published_on_the_class(self) -> None:
        """:data:`generative.BUDGET_AWARE_ATTR`, readable without constructing anything."""
        assert getattr(generative._OlmoCoreCompleter, generative.BUDGET_AWARE_ATTR) is True

    def test_the_per_item_budget_reaches_generate_batch(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """A clamped item must be decoded at its clamped budget, not at the bank's cap."""
        item = math_item(words=1400)
        window = self.window_for(monkeypatch, tmp_path, item, shots=2)
        loaded = load(monkeypatch, native_checkpoint(tmp_path, sequence_length=window, name="w"))

        loaded.scorer.score_items([item])

        assert loaded.module.generated[0]["max_new_tokens"] == generative.MIN_GENERATION_TOKENS

    def test_without_the_budget_flag_every_item_silently_takes_the_flat_cap(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """The first silent failure, made loud.

        Dropping the flag raises no error and loses no field. The scorer still derives a
        budget, still records it in the report, and then calls the one-argument form --
        so the number in ``generation_budget`` describes a decode that did not happen.
        """
        item = math_item(words=1400)
        window = self.window_for(monkeypatch, tmp_path, item, shots=2)
        monkeypatch.delattr(generative._OlmoCoreCompleter, generative.BUDGET_AWARE_ATTR)
        loaded = load(monkeypatch, native_checkpoint(tmp_path, sequence_length=window, name="w"))

        response = loaded.scorer.score_items([item])[0]

        assert loaded.module.generated[0]["max_new_tokens"] == 1024
        assert response.metadata[generative.BUDGET_KEY]["tokens"] == (
            generative.MIN_GENERATION_TOKENS
        )

    def test_the_context_window_is_published_and_is_the_declared_one(
        self, monkeypatch, checkpoint
    ) -> None:
        """:data:`generative.CONTEXT_WINDOW_ATTR`, and that the scorer builds a fitter on it."""
        loaded = load(monkeypatch, checkpoint)
        assert getattr(loaded.completer, generative.CONTEXT_WINDOW_ATTR) == SEQUENCE_LENGTH
        assert loaded.scorer.fitter is not None
        assert loaded.scorer.fitter.context_length == SEQUENCE_LENGTH
        assert loaded.scorer.runtime_facts()["context_clamp_active"] is True

    def test_without_the_context_window_the_ladder_and_the_clamp_switch_off(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """The dangerous silent failure, and the reason this whole file exists.

        Nothing raises. The prompt is rendered at the full four exemplars, the budget
        stays at the bank's 1024, and the pair is sent to a model that cannot hold it --
        so the model truncates into the exemplar block that teaches ``\\boxed{}`` and
        ``Final Answer:``, the grader finds nothing, and the item scores 0 as though the
        checkpoint could not do the mathematics.
        """
        item = math_item(words=1400)
        window = self.window_for(monkeypatch, tmp_path, item, shots=2)
        checkpoint = native_checkpoint(tmp_path, sequence_length=window, name="w")

        clamped = load(monkeypatch, checkpoint).scorer.score_items([item])[0]
        assert clamped.metadata["num_fewshot"] == 2

        monkeypatch.delattr(generative._OlmoCoreCompleter, generative.CONTEXT_WINDOW_ATTR)
        unclamped = load(monkeypatch, checkpoint)
        response = unclamped.scorer.score_items([item])[0]

        assert response.metadata["num_fewshot"] == 4
        assert "context_fit" not in response.metadata
        assert unclamped.scorer.runtime_facts()["context_clamp_active"] is False
        assert unclamped.module.generated[0]["max_new_tokens"] == 1024

    def test_the_end_of_text_spelling_is_published_and_reaches_grading(
        self, monkeypatch, checkpoint
    ) -> None:
        """:data:`generative.EOS_TEXT_ATTR`, the tokenizer's label as written."""
        loaded = load(monkeypatch, checkpoint)
        assert getattr(loaded.completer, generative.EOS_TEXT_ATTR) == EOS_TEXT
        assert loaded.scorer.eos_text == EOS_TEXT

    def test_without_the_end_of_text_spelling_a_leaked_marker_is_not_cut(
        self, monkeypatch, checkpoint
    ) -> None:
        """The third silent failure: the graded span quietly grows.

        ``eos_stop_sequences`` appends the spelling so a marker the model typed as
        ordinary characters is cut out of what the grader reads. Without it the marker and
        everything after it are graded, which on a bank with end-anchored constraints
        decides items.
        """
        item = math_item()
        config = math_config()
        assert generative.eos_stop_sequences(item, config, EOS_TEXT)[-1] == EOS_TEXT
        assert EOS_TEXT not in generative.eos_stop_sequences(item, config, None)

        monkeypatch.delattr(generative._OlmoCoreCompleter, generative.EOS_TEXT_ATTR)
        assert load(monkeypatch, checkpoint).scorer.eos_text is None

    def test_the_tokenizer_identity_is_published_and_recorded_per_item(
        self, monkeypatch, checkpoint
    ) -> None:
        """:data:`generative.TOKENIZER_ID_ATTR`. Report only, and the report needs it.

        The words-to-tokens rate is a property of the tokenizer, so two checkpoints do not
        give one item the same budget. A reader holding ``cat_report.json`` has to be able
        to say which tokenizer produced the numbers in it.
        """
        loaded = load(monkeypatch, checkpoint)
        assert getattr(loaded.completer, generative.TOKENIZER_ID_ATTR) == (
            "HuggingFaceTB/SmolLM2-135M"
        )

        response = loaded.scorer.score_items([math_item()])[0]
        assert response.metadata[generative.BUDGET_KEY]["tokenizer"] == (
            "HuggingFaceTB/SmolLM2-135M"
        )

    def test_without_the_tokenizer_identity_the_report_cannot_say_what_measured_it(
        self, monkeypatch, checkpoint
    ) -> None:
        monkeypatch.delattr(generative._OlmoCoreCompleter, generative.TOKENIZER_ID_ATTR)
        loaded = load(monkeypatch, checkpoint)
        response = loaded.scorer.score_items([math_item()])[0]
        assert response.metadata[generative.BUDGET_KEY]["tokenizer"] is None

    @staticmethod
    def window_for(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, item: BenchmarkItem, *, shots: int
    ) -> int:
        """A context window that admits exactly ``shots`` exemplars for ``item``.

        Measured rather than guessed. The four Minerva exemplars are olmo-eval's and
        their lengths are not this file's to know, so the window is derived from the
        prompt the fitter would render at ``shots`` plus the reserve
        ``fit_budget_to_context`` insists on. That makes the ladder land on ``shots``
        exactly, and it keeps landing there if upstream edits an exemplar -- whereas a
        hardcoded window would quietly start testing a different rung.
        """
        with monkeypatch.context() as scratch:
            probe = load(scratch, native_checkpoint(tmp_path, name="probe"))
            rendered = generative.format_generative_prompt(
                item,
                probe.scorer.config,
                eos_token=probe.scorer.eos_token,
                num_fewshot=shots,
            )
            return probe.completer.count_tokens(rendered) + generative.MIN_GENERATION_TOKENS


# ---------------------------------------------------------------------------
# Where the context window comes from
# ---------------------------------------------------------------------------


class TestTheContextWindowSource:
    """A raw checkpoint has no ``max_position_embeddings``, so the number has a provenance.

    On the converted path the window is written by ``hf_config_patch`` and defaults to
    2048. There is no converted config here, so that constant belongs to a different code
    path and importing it would freeze one checkpoint's training length into every future
    one. These pin what is read instead, in what order, and what happens when nothing
    declares anything.
    """

    def read(self, tmp_path: Path, **kwargs: Any) -> tuple[int, str]:
        checkpoint = native_checkpoint(tmp_path, **kwargs)
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        return generative.olmo_core_context_length(
            config,
            checkpoint_dir=checkpoint,
            model_keys=("max_sequence_length", "max_seq_len", "max_position_embeddings"),
        )

    def test_the_dataset_sequence_length_is_read_because_that_is_what_training_writes(
        self, tmp_path: Path
    ) -> None:
        """The live branch for this family, and the reason the model block is not enough.

        ``hf_config_patch`` exists because "the model object does not know the sequence
        length it was trained at", so the architecture block is silent on a real
        checkpoint here and the dataset block is not.
        """
        assert self.read(tmp_path, sequence_length=2048) == (2048, "dataset.sequence_length")

    def test_a_model_level_declaration_is_preferred(self, tmp_path: Path) -> None:
        """So this and ``OlmoCoreProvider`` cannot clamp one checkpoint differently."""
        length, source = self.read(
            tmp_path, sequence_length=2048, model={"max_sequence_length": 4096}
        )
        assert (length, source) == (4096, "model.max_sequence_length")

    def test_the_model_level_keys_are_olmo_eval_s_own_and_not_a_retyped_copy(self) -> None:
        """Anti-drift, and the only thing standing between the two readers.

        The completer passes ``olmo_core_utils._MAX_LENGTH_CONFIG_KEYS`` straight through.
        If upstream adds a spelling and this file had a copy of the tuple, the diagnostic
        would silently disagree with the provider about one checkpoint's window.
        """
        core_utils = pytest.importorskip("olmo_eval.inference.providers.olmo_core_utils")
        assert core_utils._MAX_LENGTH_CONFIG_KEYS == (
            "max_sequence_length",
            "max_seq_len",
            "max_position_embeddings",
        )

    def test_the_completer_passes_those_keys_through(self, monkeypatch, tmp_path: Path) -> None:
        core_utils = pytest.importorskip("olmo_eval.inference.providers.olmo_core_utils")
        checkpoint = native_checkpoint(
            tmp_path, sequence_length=None, model={core_utils._MAX_LENGTH_CONFIG_KEYS[-1]: 1536}
        )
        loaded = load(monkeypatch, checkpoint)
        assert loaded.completer.max_context_tokens == 1536
        assert loaded.completer.context_length_source == "model.max_position_embeddings"

    def test_the_environment_override_wins_and_is_named_in_the_report(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """The escape hatch that makes refusing safe, and it is the converted path's dial.

        One variable for one quantity. A second spelling would let a native run and a
        converted run of the same weights be clamped differently while both reports read
        as normal.
        """
        monkeypatch.setenv(generative.hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV, "1024")
        loaded = load(monkeypatch, native_checkpoint(tmp_path, sequence_length=2048))

        assert loaded.completer.max_context_tokens == 1024
        facts = loaded.scorer.runtime_facts()
        assert facts["context_length"] == 1024
        assert facts["context_length_source"] == "OLMO_CORE_HF_MAX_POSITION_EMBEDDINGS"

    def test_the_override_is_not_read_through_the_conversion_helper(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Because that helper defaults to 2048 and this must not acquire a default.

        ``hf_config_patch.max_position_embeddings()`` returns the constant when the
        variable is unset, which is exactly the silent fallback the checkpoint is supposed
        to be read for. Unset, this must fall through to the config -- so a checkpoint
        declaring 4096 gets 4096 rather than the converted path's 2048.
        """
        monkeypatch.delenv(
            generative.hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV, raising=False
        )
        assert generative.hf_config_patch.DEFAULT_MAX_POSITION_EMBEDDINGS == 2048
        assert self.read(tmp_path, sequence_length=4096) == (4096, "dataset.sequence_length")

    @pytest.mark.parametrize("value", ["", "2048.0", "two thousand"])
    def test_an_unparseable_override_is_refused(
        self, monkeypatch, tmp_path: Path, value: str
    ) -> None:
        monkeypatch.setenv(generative.hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV, value)
        with pytest.raises(RuntimeError, match="is not an integer"):
            self.read(tmp_path)

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_an_override_no_prompt_could_fit_is_refused(
        self, monkeypatch, tmp_path: Path, value: str
    ) -> None:
        """Zero and negative both validate as integers and then refuse every item."""
        monkeypatch.setenv(generative.hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV, value)
        with pytest.raises(RuntimeError, match="not a positive number of tokens"):
            self.read(tmp_path)

    def test_a_checkpoint_declaring_nothing_is_refused_rather_than_run_unclamped(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """The decision this makes, and the message that has to make it actionable.

        Unclamped is not a degraded run on this bank, it is a wrong one that looks right:
        four exemplars in front of every stem, the prompt over the window, the model
        truncated into the block that teaches the answer format, and a theta that reads as
        the checkpoint's mathematics. So the load fails, before any download or decode,
        naming every path it looked at and the variable that fixes it in one line.
        """
        with pytest.raises(RuntimeError, match="declares no context window") as caught:
            load(monkeypatch, native_checkpoint(tmp_path, sequence_length=None))

        message = str(caught.value)
        for path in (
            "model.max_sequence_length",
            "model.max_seq_len",
            "model.max_position_embeddings",
            "dataset.sequence_length",
        ):
            assert path in message
        assert generative.hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV in message

    def test_the_tokenizers_model_max_length_is_not_used_as_a_fallback(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """``_resolve_max_length`` falls back to it and this deliberately does not.

        A raw checkpoint names its tokenizer by identifier and ships no files, so that
        number belongs to whatever model the identifier points at. This family resolves
        SmolLM2, whose tokenizer declares 8192 against weights trained at 2048 -- taking it
        would leave the clamp nominally on, inert, and four times too generous, which is
        worse than no clamp because it looks like a resolved window.
        """

        class LongTokenizer(GeneratingTokenizer):
            model_max_length = 8192

        with pytest.raises(RuntimeError, match="declares no context window"):
            load(
                monkeypatch,
                native_checkpoint(tmp_path, sequence_length=None),
                tokenizer=LongTokenizer(),
            )


# ---------------------------------------------------------------------------
# The decode itself
# ---------------------------------------------------------------------------


class TestTheDecode:
    """Four lines of it, every one of them shaped by what the probe reported."""

    def test_the_prompt_echo_is_stripped(self, monkeypatch, checkpoint) -> None:
        """``generate_batch`` returns the prompt too, unless ``completions_only=True``.

        Slicing is chosen over that flag because slicing is what was measured: on all
        three banks the returned prefix compared equal to the input verbatim, while
        ``completions_only`` was read off the source and never exercised on these weights.
        """
        loaded = load(monkeypatch, checkpoint)
        loaded.module.completion_tokens = (91, 92, 93)

        assert loaded.completer("alpha beta gamma", 8) == "91 92 93"

    def test_a_tuple_return_is_unpacked(self, monkeypatch, checkpoint) -> None:
        """The real one returns three values when asked for logprobs and one when not."""
        loaded = load(monkeypatch, checkpoint)
        loaded.module.returns_tuple = True
        loaded.module.completion_tokens = (91, 92)

        assert loaded.completer("alpha beta", 8) == "91 92"

    def test_a_leaked_end_of_text_token_is_dropped_by_the_decode(
        self, monkeypatch, checkpoint
    ) -> None:
        """``skip_special_tokens=True``, which is why the *text* stop exists beside it.

        The id never reaches the graded string, so a genuine end-of-text token cannot be
        matched by a stop sequence. What :data:`generative.EOS_TEXT_ATTR` catches is the
        other case: a model typing those characters as ordinary text.
        """
        loaded = load(monkeypatch, checkpoint)
        loaded.module.completion_tokens = (91, EOS_ID, 92)

        assert loaded.completer("alpha", 8) == "91 92"

    def test_the_budget_is_passed_and_the_default_is_the_banks_cap(
        self, monkeypatch, checkpoint
    ) -> None:
        """Usable as a bare one-argument callable, which is what the default is for."""
        loaded = load(monkeypatch, checkpoint)

        loaded.completer("alpha", 12)
        loaded.completer("alpha")

        assert [call["max_new_tokens"] for call in loaded.module.generated] == [12, 1024]

    def test_the_kv_cache_is_off_on_every_call(self, monkeypatch, checkpoint) -> None:
        """Not a default to inherit: olmo_core's is ``True`` and this checkpoint refuses it.

        The default torch attention backend raises from ``assert_supports_kv_cache`` the
        moment a cache is prepared -- run_019fe2e6 died exactly there -- and the flash
        backends that implement caching need a wheel this image does not have. Passed per
        call rather than baked into the loaded module because ``generate_batch`` folds its
        kwargs onto the config, so turning it back on later needs no reload.
        """
        loaded = load(monkeypatch, checkpoint)
        loaded.completer("alpha", 4)
        assert loaded.module.generated[0]["use_cache"] is False

    def test_decoding_is_greedy_in_the_providers_own_spelling(
        self, monkeypatch, checkpoint
    ) -> None:
        """``GenerationConfig`` refuses a nonzero temperature saying this decodes greedily.

        The module's own config would otherwise decide, and a sampling default there would
        break that promise quietly -- the same run scoring differently twice, on a bank
        calibrated from single greedy completions. The four keys are lifted from
        ``OlmoCoreProvider._build_generation_kwargs`` at temperature 0 rather than
        invented, so the two callers ask the library for the same thing.
        """
        loaded = load(monkeypatch, checkpoint)
        loaded.completer("alpha", 4)

        call = loaded.module.generated[0]
        assert call["do_sample"] is False
        assert call["temperature"] == 0.0
        assert (call["top_k"], call["top_p"]) == (-1, 1.0)

    def test_a_configured_max_length_truncates_the_prompt_from_the_left(
        self, monkeypatch, checkpoint
    ) -> None:
        """The same cap the HuggingFace completer applies, so both send the same prompt.

        Asserted on what reached ``generate_batch`` rather than on what was asked for:
        the budget kwargs are identical either way, and the truncation is only visible in
        the length of the tensor.
        """
        loaded = load(monkeypatch, checkpoint, config=math_config(max_length=3))
        loaded.completer("alpha beta gamma delta epsilon", 2)
        assert loaded.module.prompted == [3]

        unlimited = load(monkeypatch, checkpoint, config=math_config(max_length=None))
        unlimited.completer("alpha beta gamma delta epsilon", 2)
        assert unlimited.module.prompted == [5]

    def test_a_chat_bank_on_a_template_less_checkpoint_is_refused(
        self, monkeypatch, checkpoint
    ) -> None:
        """Shared with the HuggingFace completer rather than copied into it.

        A base checkpoint sent a chat bank's prompt raw still completes and still grades,
        and the whole gap between a base continuation and an assistant reply lands in
        theta with a healthy standard error beside it.
        """
        chat = math_config(chat_format=True, num_fewshot=0, prompt_style="gpqa")
        with pytest.raises(ValueError, match="defines no chat template"):
            load(monkeypatch, checkpoint, config=chat)

    def test_a_chat_bank_on_a_templated_checkpoint_sends_the_template(
        self, monkeypatch, checkpoint
    ) -> None:
        loaded = load(
            monkeypatch,
            checkpoint,
            config=math_config(chat_format=True, num_fewshot=0, prompt_style="gpqa"),
            tokenizer=ChatTokenizer(),
        )
        loaded.completer("how many?", 2)

        assert loaded.module.generated, "nothing was generated"


# ---------------------------------------------------------------------------
# The clamp, against the bank it exists for
# ---------------------------------------------------------------------------


class TestTheLadderOnLeaderboardMath:
    """The real four-shot Minerva block against a small window, which is the whole case.

    MATH is the bank that overflows: 680 tokens of exemplars in front of stems reaching
    1,527, against a checkpoint trained at 2,048. What must survive the reduction is the
    part of the prompt that teaches ``\\boxed{}`` and ``Final Answer:``, because those two
    forms are exactly what the grader reads.
    """

    def test_a_stem_that_does_not_fit_drops_whole_exemplars_and_keeps_the_format(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        item = math_item(words=1400)
        window = TestTheCompleterContract.window_for(monkeypatch, tmp_path, item, shots=2)
        loaded = load(monkeypatch, native_checkpoint(tmp_path, sequence_length=window, name="w"))

        response = loaded.scorer.score_items([item])[0]

        assert response.metadata["num_fewshot"] == 2
        assert response.metadata["context_fit"]["exemplars_dropped"] == 2
        assert response.metadata[generative.BUDGET_KEY]["source"] == generative.BUDGET_FROM_CONTEXT

    def test_the_reduced_prompt_still_teaches_both_answer_forms(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Dropping whole exemplars from the front is what makes the reduction safe.

        Left-truncating to the same length would eat the head of the block and leave a
        correctly-sized prompt that teaches neither form, so the grader would find nothing
        and the item would score 0 with every guard still passing.
        """
        item = math_item(words=1400)
        window = TestTheCompleterContract.window_for(monkeypatch, tmp_path, item, shots=2)
        loaded = load(monkeypatch, native_checkpoint(tmp_path, sequence_length=window, name="w"))

        fit = loaded.scorer.fitter.fit(item, loaded.scorer.config, ceiling=1024)

        assert fit.num_fewshot == 2
        assert "\\boxed" in fit.prompt
        assert "Final Answer" in fit.prompt

    def test_a_short_stem_keeps_every_exemplar(self, monkeypatch, tmp_path: Path) -> None:
        """The clamp only binds where it has to; a stem that fits is prompted as configured."""
        loaded = load(monkeypatch, native_checkpoint(tmp_path))
        response = loaded.scorer.score_items([math_item(words=40)])[0]

        assert response.metadata["num_fewshot"] == 4
        assert "context_fit" not in response.metadata

    def test_a_session_mixing_shot_counts_says_so_in_the_report(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Two scales inside one theta is the failure the window makes possible.

        EAP treats every difficulty as fixed and estimated behind a 4-shot prompt, so an
        item answered at 2 shots moves theta by the whole of the difference with the
        standard error untouched. It has to be in the artifact, not only in a log line.
        """
        long_item = math_item(words=1400, item_id="long")
        window = TestTheCompleterContract.window_for(monkeypatch, tmp_path, long_item, shots=2)
        loaded = load(monkeypatch, native_checkpoint(tmp_path, sequence_length=window, name="w"))

        loaded.scorer.score_items([math_item(words=10, item_id="short"), long_item])
        facts = loaded.scorer.runtime_facts()

        assert facts["num_fewshot_used"] == [2, 4]
        assert "mixed_shot_alert" in facts


# ---------------------------------------------------------------------------
# What the report carries
# ---------------------------------------------------------------------------


class TestTheReportedFacts:
    """``generation_runtime``, which is what anyone re-reads once the log has scrolled."""

    def facts(self, monkeypatch, checkpoint: Path) -> dict[str, Any]:
        return load(monkeypatch, checkpoint).scorer.runtime_facts()

    def test_the_window_and_its_provenance_are_both_recorded(
        self, monkeypatch, checkpoint
    ) -> None:
        """2048 read out of a training config and 2048 supplied by an operator are the
        same clamp and different claims, so the number alone is not enough."""
        facts = self.facts(monkeypatch, checkpoint)
        assert facts["context_length_declared"] == SEQUENCE_LENGTH
        assert facts["context_length_source"] == "dataset.sequence_length"

    def test_the_disabled_cache_is_recorded(self, monkeypatch, checkpoint) -> None:
        """The largest single term in what a run costs, and invisible in the theta."""
        assert self.facts(monkeypatch, checkpoint)["kv_cache"] is False

    def test_the_end_of_text_id_is_recorded_although_it_is_zero(
        self, monkeypatch, checkpoint
    ) -> None:
        """Zero, and reported as present. ``bool(0)`` is False, so a truth test here would
        record this checkpoint family as having no end-of-text token at all."""
        facts = self.facts(monkeypatch, checkpoint)
        assert facts["eos_token_id"] == 0
        assert facts["eos_token_id_passed_to_generate"] is True
        assert facts["eos_token"] == EOS_TEXT

    def test_the_tokenization_the_native_loader_measured_travels_with_the_theta(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """The MCQ path publishes this under ``run.tokenization`` and the runner reads it
        off the scoring model; a generative scorer is not that object, so it rides in the
        checkpoint facts instead of being lost."""
        loaded = load(
            monkeypatch, native_checkpoint(tmp_path), tokenizer=GeneratingTokenizer(leading=(21,))
        )
        block = loaded.scorer.runtime_facts()["tokenization"]

        assert block["adds_special_tokens"] is True
        assert block["leading_token_ids"] == [21]

    def test_the_facts_survive_a_round_trip_through_json(self, monkeypatch, checkpoint) -> None:
        """They are written into ``cat_report.json``, so anything unserializable is a bug."""
        payload = json.loads(json.dumps(self.facts(monkeypatch, checkpoint)))
        assert payload["context_length_declared"] == SEQUENCE_LENGTH


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


class TestThePrecisionReachesTheGenerativeLoader:
    """``--dtype`` -> ``GenerationConfig.dtype`` -> ``from_checkpoint(dtype=...)``.

    The MCQ half of this flag was inert for a whole run and the report said otherwise:
    run_019fe277 asked for bfloat16 on the command line and scored in the checkpoint's
    float32, because ``--dtype`` reached ``prepare_checkpoint``, which converts nothing
    under ``--checkpoint-prep none``. The generative half had the identical gap the moment
    a native completer existed, so these assert on the kwargs handed to the loader rather
    than on the config -- the config was already right when the weights were fp32.
    """

    def loaded_kwargs(self, monkeypatch, checkpoint: Path, **overrides: Any) -> dict[str, Any]:
        return load(monkeypatch, checkpoint, config=math_config(**overrides)).calls[
            "from_checkpoint"
        ]

    def test_a_named_precision_reaches_from_checkpoint(self, monkeypatch, checkpoint) -> None:
        assert self.loaded_kwargs(monkeypatch, checkpoint, dtype="bfloat16")["dtype"] == "bfloat16"

    def test_the_default_omits_the_kwarg_rather_than_naming_a_precision(
        self, monkeypatch, checkpoint
    ) -> None:
        """ "auto" means "no opinion" and ``DType("auto")`` raises, so omission says it."""
        assert "dtype" not in self.loaded_kwargs(monkeypatch, checkpoint)

    def test_the_sentinel_is_the_one_the_mcq_config_uses(self) -> None:
        """Two configs, one word for "no opinion". A second spelling would be a second
        meaning nobody chose, and the two are set from one flag."""
        assert generative.GenerationConfig().dtype == inference.DTYPE_CHECKPOINT_DEFAULT
        assert inference.DTYPE_CHECKPOINT_DEFAULT not in convert.CONVERSION_DTYPES

    def test_the_runner_fills_the_generation_half_from_the_flag(
        self, monkeypatch, tmp_path: Path, toy_params
    ) -> None:
        """Through the CLI, because the config was never the broken link.

        Driven over an MCQ bank on purpose: the runner builds both halves from one
        ``--dtype`` unconditionally, so this is the claim at its narrowest and it needs
        no generative bank to make it. What is captured is the settings object the grader
        is built from, which is the last point before a model exists.
        """
        write_bank(tmp_path / "banks", dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")
        staged = native_checkpoint(tmp_path)
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)

        seen: list[grading.GradingSettings] = []

        def capture(request: Any, checkpoint_dir: Path, settings: Any) -> Any:
            seen.append(settings)
            return SimScorer(0.5, toy_params)

        monkeypatch.setattr(grading, "load_grader", capture)

        exit_code = runner.main(
            [
                "--cat-style",
                "uni_mcq",
                "--checkpoint",
                "s3://bucket/run/step_1000",
                "--s3-out",
                str(tmp_path / "out"),
                "--benchmark",
                "arc_challenge",
                "--checkpoint-prep",
                "none",
                "--checkpoint-kind",
                "olmo_core",
                "--dtype",
                "float16",
            ]
        )

        assert exit_code == 0
        assert seen[0].generation.dtype == "float16"
        assert seen[0].mcq.dtype == "float16"

    def test_the_generation_dtype_is_not_recorded_in_the_scoring_convention(self) -> None:
        """So no committed manifest changes shape over a field about the hardware.

        The convention block is the benchmark's description of itself and is compared
        against every bank's manifest at startup. ``dtype`` describes how the checkpoint
        was loaded, which is what ``checkpoint_kind`` and ``device_map`` already do from
        the same dataclass without appearing there.
        """
        from ..convention import _generative_convention
        from .conftest import make_spec

        spec = make_spec("leaderboard_math", modality="generative")
        recorded = _generative_convention(spec, math_config(dtype="float16"))

        assert "dtype" not in recorded
