"""The raw OLMo-core MCQ scorer, driven without torch, olmo_core or a GPU.

None of those are installed here and none of them can be, so what these tests can and
cannot establish is worth stating. They pin the *wiring* and the *arithmetic*: which
helpers the loader calls and with what, that the numbers it produces are the numbers the
HuggingFace scorer produces from the same logits, and that both permanent runtime
assertions fire on the inputs they exist for. They cannot establish anything about the
real library -- whether ``model_forward`` returns what the assertion demands, whether
``from_checkpoint`` accepts this argument set, or what the real tokenizer's defaults do.
Those are exactly the questions the assertions were made permanent to answer on the box,
rather than guessed at here.

The reused halves of ``olmo_core_utils`` are the *real* ones. That module imports only
stdlib at module level and puts every heavy import inside ``_import_olmo_core()``, so
substituting a fake for that one function leaves ``_resolve_checkpoint`` and
``_resolve_tokenizer_path`` running against a real ``config.json`` on disk -- which is
the only part of the loader's call signatures anything here can actually check.

:class:`TestTheNativePathComposes` is the other half of the file and is about the two
flags rather than the scorer. ``--checkpoint-prep none --checkpoint-kind olmo_core`` is
one cell of a two-by-two, and the design it comes from claims that reaching it costs a
flag and a registry entry. That claim is only true while preparation and the backend stay
independent, so what those tests pin is the composition: that preparation is skipped,
that the backend the registry chose is the native one, that it was handed the staged
directory rather than a converted copy, and that the report says both.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from .... import runner
from ....base import BenchmarkItem, ScoringModel
from ....common import convert, inference, s3_io
from .. import resolve
from .conftest import SimScorer, write_bank

core_utils = pytest.importorskip(
    "olmo_eval.inference.providers.olmo_core_utils",
    reason="the loader reuses olmo-eval's checkpoint helpers; run with PYTHONPATH=.;src",
)

#: Vocabulary the fake logits span. Small, and larger than any sequence these tests
#: build, so a slice that read the vocabulary axis where it meant the time axis does not
#: silently stay in range.
VOCAB = 24

#: What the checkpoint's ``dataset.tokenizer`` names. The loader must reach the
#: tokenizer through this rather than through the checkpoint directory, which is the
#: one thing about tokenizer resolution that differs from the HF path.
TOKENIZER_ID = "allenai/dolma2-tokenizer"


# ---------------------------------------------------------------------------
# A numeric stand-in for torch
# ---------------------------------------------------------------------------


def _map_last(node: Any, fn: Any) -> Any:
    """Apply ``fn`` to each innermost list of ``node``."""
    if node and isinstance(node[0], list):
        return [_map_last(child, fn) for child in node]
    return fn(node)


class Tensor:
    """A nested-list tensor supporting exactly the operations the scorers perform.

    Real arithmetic rather than a recorder of slices. Matching
    ``_HFScoringModel._continuation_logprob`` is the requirement, and a fake that only
    remembered which slices were taken could say the two backends sliced alike while
    they combined the results differently.
    """

    def __init__(self, data: Any) -> None:
        self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        dims: list[int] = []
        node = self.data
        while isinstance(node, list):
            dims.append(len(node))
            node = node[0] if node else None
        return tuple(dims)

    def __getitem__(self, key: Any) -> Tensor:
        keys = key if isinstance(key, tuple) else (key,)

        def cut(node: Any, remaining: tuple[Any, ...]) -> Any:
            if not remaining:
                return node
            head, *rest = remaining
            return [cut(child, tuple(rest)) for child in node[head]]

        return Tensor(cut(self.data, keys))

    def to(self, device: object) -> Tensor:
        return self

    def unsqueeze(self, dim: int) -> Tensor:
        return Tensor(_map_last(self.data, lambda row: [[value] for value in row]))

    def gather(self, dim: int, index: Tensor) -> Tensor:
        def pick(values: Any, indices: Any) -> Any:
            if indices and isinstance(indices[0], list):
                return [pick(v, i) for v, i in zip(values, indices, strict=True)]
            return [values[i] for i in indices]

        return Tensor(pick(self.data, index.data))

    def squeeze(self, dim: int) -> Tensor:
        return Tensor(_map_last(self.data, lambda row: row[0] if len(row) == 1 else row))

    def sum(self) -> Tensor:
        def total(node: Any) -> float:
            return sum(total(child) for child in node) if isinstance(node, list) else node

        return Tensor(total(self.data))

    def item(self) -> float:
        return float(self.data)


class FakeTorch:
    """``no_grad``, ``log_softmax`` and the two attributes the loader reads."""

    class cuda:  # noqa: N801 - mirrors torch.cuda, which is a module
        @staticmethod
        def is_available() -> bool:
            return False

    @staticmethod
    def device(name: str) -> str:
        return name

    @staticmethod
    def no_grad() -> Any:
        from contextlib import nullcontext

        return nullcontext()

    @staticmethod
    def log_softmax(tensor: Tensor, dim: int) -> Tensor:
        def row_log_softmax(row: list[float]) -> list[float]:
            peak = max(row)
            total = math.log(sum(math.exp(value - peak) for value in row))
            return [value - peak - total for value in row]

        return Tensor(_map_last(tensor.data, row_log_softmax))


# ---------------------------------------------------------------------------
# A checkpoint, a tokenizer and a generation module
# ---------------------------------------------------------------------------


def write_checkpoint(root: Path, *, pad_token_id: int = 0, eos_token_id: int = 0) -> Path:
    """A checkpoint directory in the layout ``olmo_core_utils`` parses.

    ``pad_token_id == eos_token_id == 0`` by default, because that is what this
    training setup writes and it is the exact pair ``_validate_token_ids`` refuses.

    The sharded marker is written too, so :func:`convert.is_olmo_core_checkpoint` reads
    this as native. That matters only for the composition tests, where the point is that
    ``--checkpoint-prep none`` hands over a directory ``auto`` would have converted.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model": {"d_model": 8, "vocab_size": VOCAB},
                "dataset": {
                    "tokenizer": {
                        "identifier": TOKENIZER_ID,
                        "pad_token_id": pad_token_id,
                        "eos_token_id": eos_token_id,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "model_and_optim").mkdir(exist_ok=True)
    (root / "model_and_optim" / ".metadata").write_bytes(b"\x00")
    return root


class FakeTokenizerConfig:
    """What OLMo-core's ``TokenizerConfig.from_dict`` yields, reduced to what is read."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.identifier = data.get("identifier")
        self.pad_token_id = data.get("pad_token_id")
        self.eos_token_id = data.get("eos_token_id")

    @classmethod
    def from_dict(cls, data: Any) -> FakeTokenizerConfig:
        return cls(dict(data))

    @classmethod
    def dolma2(cls) -> FakeTokenizerConfig:
        return cls({"identifier": "dolma2-fallback", "pad_token_id": 1, "eos_token_id": 2})


class FakeTokenizer:
    """Whitespace tokenizer with a configurable special-token convention.

    ``leading`` and ``trailing`` are what ``add_special_tokens=True`` adds, which is the
    thing :func:`~diagnostics.mcq_cat.common.inference.describe_tokenizer_defaults`
    exists to detect and the thing a real checkpoint's tokenizer decides for itself.
    """

    def __init__(self, *, leading: tuple[int, ...] = (), trailing: tuple[int, ...] = ()) -> None:
        self.leading = leading
        self.trailing = trailing

    @staticmethod
    def _token_id(word: str) -> int:
        return 1 + sum(ord(char) for char in word) % (VOCAB - 4)

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        ids = [self._token_id(word) for word in text.split()]
        if add_special_tokens:
            ids = [*self.leading, *ids, *self.trailing]
        return {"input_ids": Tensor([ids]) if return_tensors == "pt" else ids}


class RetokenizingTokenizer(FakeTokenizer):
    """Defaults that rewrite the string's interior rather than wrapping it.

    Neither a BOS nor an EOS, and the case the record has to be able to say is neither.
    """

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        result = super().__call__(text, return_tensors, add_special_tokens=False)
        if add_special_tokens:
            ids = [value + 1 for value in result["input_ids"]]
            return {"input_ids": Tensor([ids]) if return_tensors == "pt" else ids}
        return result


class FakeGenerationModule:
    """Deterministic logits, and one seam for reshaping them.

    The logits vary along both the time and the vocabulary axis, so a slice that reads
    one where it meant the other comes back with a different number rather than the
    same one.
    """

    device = "cpu"

    def __init__(self, reshape: Any = None) -> None:
        self.reshape = reshape
        self.forwarded: list[tuple[int, ...]] = []

    def _logits(self, input_ids: Tensor) -> Tensor:
        rows = input_ids.data[0]
        self.forwarded.append(input_ids.shape)
        logits = Tensor(
            [
                [
                    [((position * 5 + token * 3 + word * 7) % 11) / 4.0 for word in range(VOCAB)]
                    for position, token in enumerate(rows)
                ]
            ]
        )
        return self.reshape(logits) if self.reshape else logits

    def model_forward(self, *, input_ids: Tensor) -> Tensor:
        return self._logits(input_ids)

    def __call__(self, input_ids: Tensor) -> Any:
        """The HuggingFace calling convention, so one module can drive both scorers."""

        class Output:
            logits = self._logits(input_ids)

        return Output()


def install_fake_olmo_core(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tokenizer: FakeTokenizer | None = None,
    reshape: Any = None,
) -> tuple[FakeGenerationModule, dict[str, Any]]:
    """Replace only ``_import_olmo_core`` and report what the loader then did.

    Everything else in ``olmo_core_utils`` stays the shipped code, so the checkpoint
    config is parsed and the tokenizer path resolved exactly as they would be on the box.
    """
    module = FakeGenerationModule(reshape)
    tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
    calls: dict[str, Any] = {}

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer_path"] = path
            calls["tokenizer_kwargs"] = kwargs
            return tokenizer

    class ModuleFactory:
        @staticmethod
        def from_checkpoint(**kwargs: object) -> FakeGenerationModule:
            calls["from_checkpoint"] = kwargs
            return module

    class FakeGenerationConfig:
        """Mirrors ``GenerationConfig.__post_init__``, and that is the point of it.

        This used to be ``lambda **kwargs: kwargs``, which accepted anything -- so the
        suite was green while the real loader built a config olmo_core refuses, and the
        refusal was discovered on a GPU instead. The three checks below are copied from
        olmo_core/generate/generation_module/config.py:49-56 so the same argument fails
        here, in a test that needs no torch.
        """

        def __init__(self, *, pad_token_id: int, eos_token_id: int, **rest: object) -> None:
            if pad_token_id < 0:
                raise ValueError(f"pad_token_id must be non-negative, got {pad_token_id}")
            if eos_token_id < 0:
                raise ValueError(f"eos_token_id must be non-negative, got {eos_token_id}")
            if pad_token_id == eos_token_id:
                raise ValueError(
                    "pad_token_id and eos_token_id must be different, "
                    f"got {pad_token_id} and {eos_token_id}"
                )
            self.pad_token_id = pad_token_id
            self.eos_token_id = eos_token_id
            self.rest = rest

    imports = core_utils.OlmoCoreImports(
        AutoTokenizer=TokenizerFactory,
        AttentionBackendName=lambda backend: backend,
        GenerationConfig=FakeGenerationConfig,
        TokenizerConfig=FakeTokenizerConfig,
        TransformerGenerationModule=ModuleFactory,
        cached_path=lambda path: path,
        get_checkpoint_metadata=lambda path: None,
        torch=FakeTorch,
    )
    monkeypatch.setattr(core_utils, "_import_olmo_core", lambda: imports)
    return module, calls


class Loaded:
    """One loaded scorer plus everything the load touched, for the wiring assertions."""

    def __init__(self, scorer: Any, module: FakeGenerationModule, calls: dict[str, Any]) -> None:
        self.scorer = scorer
        self.module = module
        self.calls = calls


def load_scorer(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
    *,
    config: inference.InferenceConfig | None = None,
    tokenizer: FakeTokenizer | None = None,
    reshape: Any = None,
) -> Loaded:
    """Build the scorer through the real registry with only ``_import_olmo_core`` faked."""
    module, calls = install_fake_olmo_core(monkeypatch, tokenizer=tokenizer, reshape=reshape)
    scorer = inference.load_scoring_model(
        checkpoint,
        config or inference.InferenceConfig(checkpoint_kind="olmo_core"),
    )
    return Loaded(scorer, module, calls)


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    return write_checkpoint(tmp_path / "step305176")


def mcq_item(choices: tuple[str, ...] = ("alpha", "beta gamma")) -> BenchmarkItem:
    return BenchmarkItem(item_id="i0", question="Which one?", choices=choices, gold_index=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTheRegistryReachesIt:
    """``olmo_core`` is a row in :data:`inference.MCQ_SCORING_BACKENDS`, not a branch."""

    def test_the_dispatch_returns_a_scoring_model(self, monkeypatch, checkpoint) -> None:
        loaded = load_scorer(monkeypatch, checkpoint)
        assert isinstance(loaded.scorer, inference._OlmoCoreScoringModel)
        assert isinstance(loaded.scorer, ScoringModel)

    def test_the_registry_is_what_names_it(self) -> None:
        """The entry, not the class, is what makes the kind reachable."""
        assert set(inference.MCQ_SCORING_BACKENDS) == {"hf", "olmo_core"}

    def test_an_unknown_kind_is_still_rejected_and_names_both(self, checkpoint) -> None:
        config = inference.InferenceConfig(checkpoint_kind="ollama")
        with pytest.raises(ValueError, match="Unknown checkpoint_kind") as excinfo:
            inference.load_scoring_model(checkpoint, config)
        assert "hf, olmo_core" in str(excinfo.value)

    def test_the_generative_side_still_refuses_it(self) -> None:
        """The asymmetry the per-modality split exists to express, pinned from both ends.

        A backend registered for one modality and not the other is a real state, and it
        is this one. Collapsing the two registries would make an MCQ-only reader look
        like a generative one.
        """
        from ....common import generative

        assert "olmo_core" not in generative.GENERATIVE_BACKENDS


class TestTheLoaderWiring:
    """What the loader asks olmo-eval's helpers for, and with what."""

    def test_the_tokenizer_comes_from_the_checkpoint_config_not_the_directory(
        self, monkeypatch, checkpoint
    ) -> None:
        """``dataset.tokenizer.identifier`` names it; a raw checkpoint holds no tokenizer."""
        loaded = load_scorer(monkeypatch, checkpoint)
        assert loaded.calls["tokenizer_path"] == TOKENIZER_ID

    def test_from_checkpoint_is_given_the_checkpoint_and_a_device(
        self, monkeypatch, checkpoint
    ) -> None:
        loaded = load_scorer(monkeypatch, checkpoint)
        assert loaded.calls["from_checkpoint"]["checkpoint_dir"] == str(checkpoint)
        assert loaded.calls["from_checkpoint"]["device"] == "cpu"

    def test_a_generation_config_is_passed_with_a_pad_distinct_from_eos(
        self, monkeypatch, checkpoint
    ) -> None:
        """Supplying one is the fix; omitting it was the bug, and the card proved it.

        The earlier reading was that leaving ``generation_config`` out sidesteps the
        pad/eos validator, because that validator lives on the config. It does not:
        ``from_checkpoint`` builds its own from the checkpoint's token ids when the caller
        passes none, and these checkpoints write pad == eos == 0, so the config it builds
        refuses itself. Run run_019fe265 died exactly there, after resolving the bank,
        pulling all 140 objects and loading the tokenizer.

        So the config is supplied, with a pad id that only has to differ from eos. Nothing
        pads on this path, so the value never reaches a tensor.
        """
        loaded = load_scorer(monkeypatch, checkpoint)
        passed = loaded.calls["from_checkpoint"]["generation_config"]
        assert passed.eos_token_id == 0, "eos must stay the checkpoint's own"
        assert passed.pad_token_id != passed.eos_token_id, (
            "a pad equal to eos is what olmo_core refuses at load"
        )

    def test_the_fabricated_pad_id_is_inside_the_vocabulary(self) -> None:
        """Not a sentinel like -1 or vocab_size, either of which validate and then bite.

        ``GenerationConfig.validate`` only rejects a negative pad or one equal to eos, so
        an out-of-range id would pass here and surface later as an index error in whoever
        wires generation up. Both branches return a real token id.
        """
        assert inference._OlmoCoreScoringModel._unused_pad_token_id(0) == 1
        assert inference._OlmoCoreScoringModel._unused_pad_token_id(1) == 0
        for eos in range(0, 8):
            assert inference._OlmoCoreScoringModel._unused_pad_token_id(eos) != eos

    def test_a_checkpoint_whose_pad_equals_its_eos_still_loads(
        self, monkeypatch, checkpoint
    ) -> None:
        """``validate_checkpoint=False`` is the decision this pins.

        Preston's checkpoint has ``pad_token_id == eos_token_id == bos_token_id == 0``,
        and every checkpoint this training setup writes does. Scoring never pads and
        never generates, so the id is unread; validating would refuse the real
        checkpoints rather than the malformed ones.
        """
        loaded = load_scorer(monkeypatch, checkpoint)
        assert loaded.scorer.tokenizer is not None

    def test_the_validating_branch_would_have_refused_it(self, checkpoint) -> None:
        """The other half of the test above, so it cannot rot into a tautology.

        If ``_validate_token_ids`` ever stops rejecting this pair, the exemption above
        stops being load-bearing and should be reconsidered rather than kept out of
        habit.
        """
        with pytest.raises(ValueError, match="pad_token_id and eos_token_id must be different"):
            core_utils._validate_token_ids(
                checkpoint_dir=str(checkpoint), pad_token_id=0, eos_token_id=0
            )

    def test_scoring_a_pad_equals_eos_checkpoint_reaches_a_number(
        self, monkeypatch, checkpoint
    ) -> None:
        """The finding stated as a test rather than as a paragraph.

        The pad/eos collision was flagged as a blocker for the *generative* path, which
        has to stop on a distinct EOS. It is not one here: the scorer takes one forward
        pass over a single sequence and reads log-probabilities off it, so nothing pads
        and nothing stops.
        """
        loaded = load_scorer(monkeypatch, checkpoint)
        response = loaded.scorer.score_items([mcq_item()])[0]
        assert all(math.isfinite(value) for value in response.choice_logprobs)

    def test_a_checkpoint_with_no_config_is_refused_rather_than_guessed_at(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """``allow_tokenizer_fallback=False``: a silent dolma2 default would score a
        bank against a tokenizer nobody chose."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="missing config.json"):
            load_scorer(monkeypatch, empty)

    def test_a_diagnostics_only_checkout_says_what_is_missing(self, monkeypatch) -> None:
        """A checkout with no ``olmo_eval`` must say so plainly rather than ImportError."""
        monkeypatch.setitem(sys.modules, "olmo_eval.inference.providers", None)
        with pytest.raises(RuntimeError, match="olmo_eval is not importable"):
            inference._olmo_core_utils()


class TestTheForwardShapeAssertion:
    """``model_forward`` on a batch of one, which is untested upstream too.

    The provider only ever calls it batched, so nothing above this has exercised a
    batch of one. What the assertion buys, measured rather than assumed: against *this*
    arithmetic a squeezed ``(seq, vocab)`` does not read the wrong axis silently, it
    raises, because ``logits[:, :-1, :]`` is three indices into a two-dimensional
    tensor. The assertion's value is therefore that it raises *first* and says what
    happened, instead of an ``IndexError`` from the middle of a ``log_softmax``
    expression -- and that it holds the invariant in place for the refactor that would
    share this arithmetic with the provider. The provider's half of that works on a
    per-row ``(seq, vocab)`` tensor, where a squeezed batch axis *would* be readable and
    wrong.
    """

    def score(self, loaded: Loaded) -> float:
        return loaded.scorer._continuation_logprob("a prompt here", " beta")

    def test_the_expected_shape_passes(self, monkeypatch, checkpoint) -> None:
        loaded = load_scorer(monkeypatch, checkpoint)
        assert isinstance(self.score(loaded), float)

    def test_a_squeezed_batch_axis_raises(self, monkeypatch, checkpoint) -> None:
        loaded = load_scorer(monkeypatch, checkpoint, reshape=lambda t: Tensor(t.data[0]))
        with pytest.raises(RuntimeError, match=r"shape \(4, 24\)"):
            self.score(loaded)

    def test_a_batch_axis_wider_than_one_raises(self, monkeypatch, checkpoint) -> None:
        loaded = load_scorer(
            monkeypatch, checkpoint, reshape=lambda t: Tensor([t.data[0], t.data[0]])
        )
        with pytest.raises(RuntimeError, match=r"shape \(2, 4, 24\)"):
            self.score(loaded)

    def test_only_the_final_position_raises(self, monkeypatch, checkpoint) -> None:
        """A module returning just the last step's logits is the generation convention."""
        loaded = load_scorer(monkeypatch, checkpoint, reshape=lambda t: Tensor([t.data[0][-1:]]))
        with pytest.raises(RuntimeError, match=r"shape \(1, 1, 24\)"):
            self.score(loaded)

    def test_a_rank_two_output_whose_first_axes_look_right_still_raises(
        self, monkeypatch, checkpoint
    ) -> None:
        """One value per position rather than a distribution: ids, or an argmax.

        Both the batch axis and the sequence axis read correctly here, so the rank is
        the only thing separating it from a valid output -- and the vocabulary it is
        missing is the axis the gather reads.
        """
        loaded = load_scorer(
            monkeypatch, checkpoint, reshape=lambda t: Tensor([[row[0] for row in t.data[0]]])
        )
        with pytest.raises(RuntimeError, match=r"shape \(1, 4\)"):
            self.score(loaded)

    def test_the_message_names_what_was_expected_and_what_arrived(
        self, monkeypatch, checkpoint
    ) -> None:
        loaded = load_scorer(monkeypatch, checkpoint, reshape=lambda t: Tensor(t.data[0]))
        with pytest.raises(RuntimeError) as caught:
            self.score(loaded)

        message = str(caught.value)
        assert "returned logits of shape (4, 24)" in message
        assert "needs (1, 4, vocab)" in message
        assert "batch axis at 0 and the time axis at 1" in message

    def test_a_later_choice_of_a_different_length_is_not_refused(
        self, monkeypatch, checkpoint
    ) -> None:
        """Checked once per model, so what is remembered must be the module's property.

        Only the rank and the batch axis belong to the module; the sequence length is
        the input's and changes with every choice. Remembering the whole first shape
        would refuse the second choice of every item -- and every item has at least two.
        """
        loaded = load_scorer(monkeypatch, checkpoint)

        assert isinstance(loaded.scorer._continuation_logprob("a b c", " d"), float)
        assert isinstance(loaded.scorer._continuation_logprob("a b c d e f", " g h"), float)
        assert loaded.module.forwarded == [(1, 4), (1, 8)]
        assert loaded.scorer._forward_shape_checked is True


class TestTheTokenizerDefaultsRecord:
    """What ``tokenizer(text)`` adds that ``add_special_tokens=False`` does not.

    Recorded rather than corrected. This scorer keeps the defaults, because that is
    what the HF path does and what every bank's difficulties were calibrated behind;
    ``OlmoCoreProvider`` passes ``add_special_tokens=False``. Comparing the two cannot
    see the difference on its own, which is why it is measured at load.
    """

    def defaults(self, monkeypatch, checkpoint, tokenizer) -> inference.TokenizerDefaults:
        return load_scorer(monkeypatch, checkpoint, tokenizer=tokenizer).scorer.tokenizer_defaults

    def test_a_tokenizer_that_adds_nothing_is_recorded_as_agreeing(
        self, monkeypatch, checkpoint
    ) -> None:
        record = self.defaults(monkeypatch, checkpoint, FakeTokenizer())
        assert record.adds_special_tokens is False
        assert record.leading == ()
        assert record.trailing == ()
        assert "add no special tokens" in record.summary()

    def test_a_prepended_bos_is_caught_and_named(self, monkeypatch, checkpoint) -> None:
        record = self.defaults(monkeypatch, checkpoint, FakeTokenizer(leading=(21,)))
        assert record.adds_special_tokens is True
        assert record.leading == (21,)
        assert record.trailing == ()
        assert "prepend [21]" in record.summary()

    def test_an_appended_eos_is_caught_apart_from_a_bos(self, monkeypatch, checkpoint) -> None:
        """The more damaging half: it moves the scored span rather than shifting it."""
        record = self.defaults(monkeypatch, checkpoint, FakeTokenizer(trailing=(22,)))
        assert record.leading == ()
        assert record.trailing == (22,)
        assert "append [22]" in record.summary()

    def test_both_ends_are_reported_together(self, monkeypatch, checkpoint) -> None:
        record = self.defaults(
            monkeypatch, checkpoint, FakeTokenizer(leading=(21,), trailing=(22,))
        )
        assert (record.leading, record.trailing) == ((21,), (22,))
        assert "prepend [21], append [22]" in record.summary()

    def test_a_tokenizer_that_re_encodes_the_string_is_not_called_a_bos(
        self, monkeypatch, checkpoint
    ) -> None:
        record = self.defaults(monkeypatch, checkpoint, RetokenizingTokenizer())
        assert record.adds_special_tokens is True
        assert (record.leading, record.trailing) == ((), ())
        assert "re-encode the probe string entirely" in record.summary()

    def test_a_divergence_is_logged_loudly_and_agreement_quietly(
        self, monkeypatch, checkpoint, caplog
    ) -> None:
        with caplog.at_level("INFO", logger="mcq_cat.inference"):
            load_scorer(monkeypatch, checkpoint, tokenizer=FakeTokenizer(leading=(21,)))
        assert [record.levelname for record in caplog.records] == ["WARNING"]
        assert "prepend [21]" in caplog.text

        caplog.clear()
        with caplog.at_level("INFO", logger="mcq_cat.inference"):
            load_scorer(monkeypatch, checkpoint, tokenizer=FakeTokenizer())
        assert [record.levelname for record in caplog.records] == ["INFO"]

    def test_the_record_serializes_for_a_report(self, monkeypatch, checkpoint) -> None:
        record = self.defaults(monkeypatch, checkpoint, FakeTokenizer(leading=(21,)))
        payload = json.loads(json.dumps(record.as_dict()))
        assert payload["adds_special_tokens"] is True
        assert payload["leading_token_ids"] == [21]

    def test_the_probe_reads_the_tokenizer_and_nothing_else(self) -> None:
        """Checkable without a checkpoint, which is why it is a free function."""
        record = inference.describe_tokenizer_defaults(FakeTokenizer(leading=(21,), trailing=(22,)))
        assert (record.leading, record.trailing) == ((21,), (22,))


class TestTheNativePathComposes:
    """``--checkpoint-prep none --checkpoint-kind olmo_core``, end to end.

    Two seams, and the design's claim is that they vary independently. These run the
    real runner over a real native directory with only ``_import_olmo_core`` faked, so
    every step between the flags and the report is the shipped code: the kind guard, the
    preparation policy, the registry lookup, the CAT loop and the report writer.
    """

    def native_run(
        self,
        monkeypatch,
        tmp_path: Path,
        *extra: str,
        tokenizer: FakeTokenizer | None = None,
    ) -> tuple[int, Path, dict[str, Any]]:
        """Run the CLI against a native checkpoint and hand back the report location."""
        write_bank(tmp_path / "banks", dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")

        staged = write_checkpoint(tmp_path / "step305176")
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)
        _, calls = install_fake_olmo_core(monkeypatch, tokenizer=tokenizer)

        out_dir = tmp_path / "out"
        exit_code = runner.main(
            [
                "--cat-style",
                "uni_mcq",
                "--checkpoint",
                "s3://bucket/run/checkpoints/step305176",
                "--s3-out",
                str(out_dir),
                "--benchmark",
                "arc_challenge",
                "--checkpoint-prep",
                "none",
                "--checkpoint-kind",
                "olmo_core",
                *extra,
            ]
        )
        calls["staged"] = staged
        return exit_code, out_dir, calls

    def report(self, monkeypatch, tmp_path: Path, *extra: str) -> dict[str, Any]:
        exit_code, out_dir, _ = self.native_run(monkeypatch, tmp_path, *extra)
        assert exit_code == 0
        return json.loads((out_dir / "cat_report.json").read_text(encoding="utf-8"))

    def test_the_pair_of_flags_runs_to_a_report(self, monkeypatch, tmp_path: Path) -> None:
        payload = self.report(monkeypatch, tmp_path)
        assert isinstance(payload["metadata"]["theta"], float)

    def test_the_kind_guard_lets_it_past(self) -> None:
        """``check_checkpoint_kind`` runs before the fetch, so it decides this first."""
        from ....common import grading

        request = grading.GradingRequest(dataset="arc_challenge", modality=grading.MCQ)
        settings = grading.GradingSettings(
            mcq=inference.InferenceConfig(checkpoint_kind="olmo_core")
        )
        grading.check_checkpoint_kind(request, settings)

    def test_preparation_is_skipped_and_the_backend_gets_the_staged_directory(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """The composition itself, and the trap it avoids.

        This directory sniffs as native, so under the default ``auto`` policy
        ``prepare_checkpoint`` would convert it and hand the backend a *different* path
        holding HuggingFace weights. Asserting that the native loader was handed the
        staged path is what separates "the two flags compose" from "the second flag was
        read and the first was ignored".
        """
        _, _, calls = self.native_run(monkeypatch, tmp_path)
        assert convert.is_olmo_core_checkpoint(calls["staged"])
        assert calls["from_checkpoint"]["checkpoint_dir"] == str(calls["staged"])

    def test_the_conversion_body_is_never_called(self, monkeypatch, tmp_path: Path) -> None:
        """Belt and braces on the same claim, from the converter's side.

        The assertion above would still hold if conversion ran and happened to return
        its input; this one would not.
        """
        converted: list[Path] = []
        monkeypatch.setattr(
            convert,
            "convert_olmo_core_to_hf",
            lambda local_dir, out_dir, **kwargs: converted.append(local_dir) or out_dir,
        )
        exit_code, _, _ = self.native_run(monkeypatch, tmp_path)
        assert exit_code == 0
        assert converted == []

    def test_the_report_records_both_halves_of_the_composition(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Either field alone is ambiguous. ``checkpoint_kind: olmo_core`` under ``auto``
        would be a converted directory read by the native loader, and
        ``checkpoint_prep: none`` under ``hf`` is an already-HF checkpoint. Only the pair
        says which of the four cells produced this theta."""
        run = self.report(monkeypatch, tmp_path)["run"]
        assert run["checkpoint_prep"] == "none"
        assert run["checkpoint_kind"] == "olmo_core"

    def test_the_report_carries_the_tokenization_the_native_scorer_measured(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """A log line scrolls past; the theta artifact is what anyone re-reads.

        Which tokens were scored decides whether the bank's calibrated difficulties
        describe the questions this run actually asked, so it belongs beside the theta.
        """
        exit_code, out_dir, _ = self.native_run(
            monkeypatch, tmp_path, tokenizer=FakeTokenizer(leading=(21,))
        )
        assert exit_code == 0
        payload = json.loads((out_dir / "cat_report.json").read_text(encoding="utf-8"))
        block = payload["run"]["tokenization"]
        assert block["adds_special_tokens"] is True
        assert block["leading_token_ids"] == [21]
        assert "prepend [21]" in block["summary"]

    def test_a_backend_with_nothing_to_say_adds_no_key(
        self, monkeypatch, tmp_path: Path, toy_params
    ) -> None:
        """The HF scorer records none, and its reports must not grow an empty field."""
        from .conftest import stage_hf_checkpoint

        write_bank(tmp_path / "banks", dataset="arc_challenge")
        monkeypatch.setattr(resolve, "CALIBRATED_DATASETS", tmp_path / "banks")
        staged = stage_hf_checkpoint(tmp_path)
        monkeypatch.setattr(s3_io, "resolve_checkpoint", lambda *a, **k: staged)
        monkeypatch.setattr(
            inference, "load_scoring_model", lambda *a, **k: SimScorer(0.5, toy_params)
        )

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
                    "arc_challenge",
                ]
            )
            == 0
        )
        payload = json.loads((out_dir / "cat_report.json").read_text(encoding="utf-8"))
        assert "tokenization" not in payload["run"]

    def test_the_default_precision_is_recorded_and_says_nothing_about_the_weights(
        self, monkeypatch, tmp_path: Path, caplog
    ) -> None:
        """Nothing was converted, so ``dtype`` describes no weights on this path.

        It is still recorded, and it is readable only because ``checkpoint_prep`` is
        recorded beside it -- ``dtype: bfloat16, checkpoint_prep: none`` is a request
        that did not apply, where ``dtype: bfloat16`` alone would read as a measurement.
        The default draws no warning, because warning on what a caller gets for not
        asking would fire on every native run and tell nobody anything.
        """
        with caplog.at_level("WARNING", logger="mcq_cat.convert"):
            run = self.report(monkeypatch, tmp_path)["run"]
        assert (run["dtype"], run["checkpoint_prep"]) == ("bfloat16", "none")
        assert "no effect under" not in caplog.text

    def test_a_precision_that_was_actually_asked_for_warns(
        self, monkeypatch, tmp_path: Path, caplog
    ) -> None:
        """The case the warning exists for, reached through the CLI rather than the seam.

        Somebody sets ``--dtype float16`` because the card they were given has no
        bfloat16, gets ``none``, and the weights load at whatever training wrote. Without
        this line they find out when a kernel refuses the format, which reads like the
        flag is broken rather than inapplicable.
        """
        with caplog.at_level("WARNING", logger="mcq_cat.convert"):
            run = self.report(monkeypatch, tmp_path, "--dtype", "float16")["run"]
        assert run["dtype"] == "float16"
        assert "no effect under --checkpoint-prep none" in caplog.text


class TestTheArithmeticMatchesTheHuggingFacePath:
    """Same logits in, same number out, choice by choice.

    The two backends are deliberately separate implementations -- that is what makes
    comparing them against a real checkpoint informative rather than circular -- so the
    thing worth pinning is that they agree, not that they share code.
    """

    def hf_scorer(self, config: inference.InferenceConfig, module: FakeGenerationModule) -> Any:
        """The shipped HF scorer with its three heavy attributes supplied directly."""
        scorer = inference._HFScoringModel.__new__(inference._HFScoringModel)
        scorer.config = config
        scorer.tokenizer = FakeTokenizer()
        scorer.model = module
        scorer._torch = FakeTorch
        return scorer

    @pytest.mark.parametrize(
        ("prompt", "continuation"),
        [
            ("Question: which one?", " alpha"),
            ("Question: which one?", " beta gamma delta"),
            ("a", " b"),
            ("a much longer prompt with several words in it", " and a long continuation too"),
        ],
    )
    def test_the_two_backends_agree(
        self, monkeypatch, checkpoint, prompt: str, continuation: str
    ) -> None:
        config = inference.InferenceConfig(checkpoint_kind="olmo_core")
        loaded = load_scorer(monkeypatch, checkpoint, config=config)

        assert loaded.scorer._continuation_logprob(prompt, continuation) == pytest.approx(
            self.hf_scorer(config, loaded.module)._continuation_logprob(prompt, continuation)
        )

    def test_a_continuation_that_adds_no_tokens_scores_zero_without_a_forward_pass(
        self, monkeypatch, checkpoint
    ) -> None:
        loaded = load_scorer(monkeypatch, checkpoint)
        assert loaded.scorer._continuation_logprob("a prompt", "") == 0.0
        assert loaded.module.forwarded == []

    def test_the_length_cap_truncates_the_pair_but_not_the_scored_span(
        self, monkeypatch, checkpoint
    ) -> None:
        """The same span the untruncated count names, matching the HF path exactly."""
        config = inference.InferenceConfig(checkpoint_kind="olmo_core", max_length=6)
        loaded = load_scorer(monkeypatch, checkpoint, config=config)
        prompt, continuation = "one two three four five six seven", " eight nine"

        assert loaded.scorer._continuation_logprob(prompt, continuation) == pytest.approx(
            self.hf_scorer(config, loaded.module)._continuation_logprob(prompt, continuation)
        )
        assert loaded.module.forwarded[0] == (1, 6)

    def test_a_cap_that_would_eat_the_answer_is_refused(self, monkeypatch, checkpoint) -> None:
        config = inference.InferenceConfig(checkpoint_kind="olmo_core", max_length=2)
        loaded = load_scorer(monkeypatch, checkpoint, config=config)
        with pytest.raises(ValueError, match="cannot be set below"):
            loaded.scorer._continuation_logprob("one two three", " four five")

    def test_score_items_produces_the_same_responses_as_the_hf_scorer(
        self, monkeypatch, checkpoint
    ) -> None:
        config = inference.InferenceConfig(checkpoint_kind="olmo_core")
        loaded = load_scorer(monkeypatch, checkpoint, config=config)
        items = [mcq_item(), mcq_item(("delta epsilon", "zeta"))]

        native = loaded.scorer.score_items(items)
        hf = self.hf_scorer(config, loaded.module).score_items(items)

        assert [r.chosen_index for r in native] == [r.chosen_index for r in hf]
        assert [r.correct for r in native] == [r.correct for r in hf]
        for mine, theirs in zip(native, hf, strict=True):
            assert mine.choice_logprobs == pytest.approx(theirs.choice_logprobs)

    def test_choice_logprobs_are_populated_and_distinct(self, monkeypatch, checkpoint) -> None:
        """The report is only re-checkable if it carries the numbers the argmax compared."""
        loaded = load_scorer(monkeypatch, checkpoint)
        response = loaded.scorer.score_items([mcq_item()])[0]

        assert len(response.choice_logprobs) == 2
        assert response.choice_logprobs[0] != response.choice_logprobs[1]

    def test_the_normalization_is_applied_by_this_backend_too(
        self, monkeypatch, checkpoint
    ) -> None:
        """MuSR and BBH are scored under ``acc_norm``; a backend ignoring it is silent."""
        plain = inference.InferenceConfig(checkpoint_kind="olmo_core")
        per_char = inference.InferenceConfig(
            checkpoint_kind="olmo_core",
            score_normalization="continuation_logprob_per_character",
        )
        item = mcq_item()
        unnormalized = load_scorer(monkeypatch, checkpoint, config=plain).scorer.score_items([item])
        normalized = load_scorer(monkeypatch, checkpoint, config=per_char).scorer.score_items(
            [item]
        )

        for raw, scaled, choice in zip(
            unnormalized[0].choice_logprobs,
            normalized[0].choice_logprobs,
            inference.scored_choices(item, plain),
            strict=True,
        ):
            assert scaled == pytest.approx(raw / len(choice.continuation))

    def test_the_prompt_style_reaches_this_backend_too(self, monkeypatch, checkpoint) -> None:
        """WinoGrande substitutes into the stem, so a backend appending instead would
        score a different question with a healthy standard error beside it."""
        config = inference.InferenceConfig(
            checkpoint_kind="olmo_core", prompt_style="blank_substitution"
        )
        loaded = load_scorer(monkeypatch, checkpoint, config=config)
        item = BenchmarkItem(
            item_id="w0",
            question="Sarah beat Maria because _ trained harder.",
            choices=("Sarah", "Maria"),
            gold_index=0,
        )

        loaded.scorer.score_items([item])

        assert loaded.module.forwarded, "no forward pass was taken"
        assert len(loaded.module.forwarded) == 2

    def test_a_generative_item_reaching_this_scorer_raises(self, monkeypatch, checkpoint) -> None:
        loaded = load_scorer(monkeypatch, checkpoint)
        item = BenchmarkItem(item_id="g0", question="How many?", choices=(), gold_index=-1)
        with pytest.raises(ValueError, match="no answer choices"):
            loaded.scorer.score_items([item])


class TestTheHeavyImportsStayLazy:
    """The suite passes in a checkout with no GPU stack, and that is load-bearing."""

    def test_importing_the_module_pulls_in_neither_torch_nor_olmo_core(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import diagnostics.mcq_cat.common.inference; "
                "print(', '.join(name for name in ('torch', 'transformers', 'olmo_core') "
                "if name in sys.modules) or 'none')",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[5],
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "none"

    def test_registering_the_backend_did_not_import_anything(self) -> None:
        """The registry holds a lambda, so naming ``olmo_core`` costs nothing at import.

        A module-level ``from olmo_eval... import`` beside the entry would have been the
        obvious way to write this and would have made the whole diagnostic depend on
        ``olmo_eval`` being installed.
        """
        assert callable(inference.MCQ_SCORING_BACKENDS["olmo_core"])
