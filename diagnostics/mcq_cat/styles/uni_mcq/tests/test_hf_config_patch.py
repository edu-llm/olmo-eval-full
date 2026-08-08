"""The `get_hf_config` patch: that it is installed everywhere, and what it emits.

Without this patch the conversion in :mod:`~diagnostics.mcq_cat.common.convert` cannot run
against Preston's checkpoints at all -- upstream's exporter builds a config only for
``ReorderedNormTransformerBlock`` and these are plain pre-norm blocks -- so its absence is
a hard failure rather than a degradation. What is *not* a hard failure, and is the reason
half of this file exists, is the patch being installed in the wrong place: rebinding
``get_hf_config`` in the module that defines it does nothing for
``olmo_core.nn.hf.checkpoint``, which imported the name rather than the module and is the
only caller that matters. That mistake reports success and then raises the exact error the
patch exists to prevent.

``ai2-olmo-core`` needs torch and is not installed here, so the package is stood in for.
That is not a compromise for these tests: every property under test is a property of the
module graph and of the config that comes out, and a stand-in reproduces the graph exactly
while making the by-value import -- the thing most likely to be got wrong -- something a
test can construct on purpose.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from ....common import convert, hf_config_patch

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
needs_no_transformers = pytest.mark.skipif(
    HAS_TRANSFORMERS,
    reason="transformers is installed, so the missing-LlamaConfig branch is unreachable",
)

#: Distinguishes "the test did not say" from "the test said None", which for ``rope`` and
#: ``backend`` are different models rather than the same one written two ways.
_DEFAULT = object()


class Rope:
    def __init__(self, theta: float = 10000.0, scaling: object | None = None) -> None:
        self.theta = theta
        self.scaling = scaling


class Backend:
    def __init__(self, window_size: tuple[int, int] = (-1, -1)) -> None:
        self.window_size = window_size


class Linear:
    def __init__(self, bias: object | None = None) -> None:
        self.bias = bias


class Norm:
    def __init__(self, eps: float = 1e-05) -> None:
        self.eps = eps


class FeedForward:
    def __init__(self, hidden_size: int = 1536, w1_bias: object | None = None) -> None:
        self.hidden_size = hidden_size
        self.w1 = Linear(w1_bias)


class Attention:
    """Stands in for ``olmo_core.nn.attention.Attention``.

    Defaults are Preston's checkpoint: 9 heads over 3 KV heads at head_dim 64, rope with
    no scaling, no QK-norm, no sliding window, no biases.
    """

    def __init__(
        self,
        *,
        n_heads: int = 9,
        n_kv_heads: int = 3,
        head_dim: int | None = 64,
        rope: object = _DEFAULT,
        q_norm: object | None = None,
        k_norm: object | None = None,
        backend: object = _DEFAULT,
        out_bias: object | None = None,
    ) -> None:
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rope = Rope() if rope is _DEFAULT else rope
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.backend = Backend() if backend is _DEFAULT else backend
        self.w_out = Linear(out_bias)


class GatedDeltaNet:
    """A sequence mixer that is present and is not ``Attention``."""

    rope = None


class TransformerBlock:
    def __init__(
        self,
        attention: object | None = None,
        feed_forward: FeedForward | None = None,
        eps: float = 1e-05,
    ) -> None:
        self.attention = Attention() if attention is None else attention
        self.feed_forward = feed_forward or FeedForward()
        self.feed_forward_norm = Norm(eps)


class ReorderedNormTransformerBlock(TransformerBlock):
    """Subclasses the plain block upstream, which is why order of the checks matters."""


class Transformer:
    def __init__(
        self,
        blocks: list[TransformerBlock] | None = None,
        *,
        d_model: int = 576,
        n_layers: int = 30,
        vocab_size: int = 49152,
        tie_word_embeddings: bool = True,
    ) -> None:
        blocks = blocks if blocks is not None else [TransformerBlock()]
        self.blocks = {str(i): block for i, block in enumerate(blocks)}
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings


class MoETransformer(Transformer):
    pass


class NormalizedTransformer(Transformer):
    pass


def upstream_get_hf_config(model: Transformer) -> str:
    """What ``olmo_core`` ships, reduced to its dispatch and its refusals.

    Mirrors ``OLMo-core/src/olmo_core/nn/hf/config.py:87-101``: normalized transformers are
    refused, MoE goes to the FlexOlmo builder, and anything whose first block is not
    reordered-norm is refused. The last of those is the branch this whole module exists to
    get past.
    """
    if isinstance(model, NormalizedTransformer):
        raise NotImplementedError(
            f"Building HF config not implemented for {model.__class__.__name__}"
        )
    if isinstance(model, MoETransformer):
        return "flex-olmo-config"
    first_block = list(model.blocks.values())[0]
    if not isinstance(first_block, ReorderedNormTransformerBlock):
        raise NotImplementedError(
            f"Block is not a {ReorderedNormTransformerBlock.__name__}, unable to build HF "
            f"config for {model.__class__.__name__}"
        )
    return "olmo2-config"


class FakeLlamaConfig:
    """Records what the patch emitted, since the assertion is about the kwargs."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def olmo_core(monkeypatch) -> SimpleNamespace:
    """Install a stand-in ``olmo_core`` package tree and a recording ``LlamaConfig``.

    Three modules hold ``get_hf_config`` and they hold it differently: ``config`` defines
    it, and ``checkpoint`` and the package ``__init__`` bind the object by value the way
    the real ones do. A patch that only rebinds the first passes a naive test and fails on
    a node.
    """
    created: dict[str, types.ModuleType] = {}

    def module(name: str, **attrs: object) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        created[name] = mod
        return mod

    module("olmo_core")
    module("olmo_core.nn")
    module("olmo_core.nn.attention", Attention=Attention)
    module("olmo_core.nn.transformer")
    module(
        "olmo_core.nn.transformer.block",
        TransformerBlock=TransformerBlock,
        ReorderedNormTransformerBlock=ReorderedNormTransformerBlock,
    )
    module(
        "olmo_core.nn.transformer.model",
        Transformer=Transformer,
        MoETransformer=MoETransformer,
        NormalizedTransformer=NormalizedTransformer,
    )
    package = module("olmo_core.nn.hf", get_hf_config=upstream_get_hf_config)
    config = module("olmo_core.nn.hf.config", get_hf_config=upstream_get_hf_config)
    checkpoint = module("olmo_core.nn.hf.checkpoint", get_hf_config=upstream_get_hf_config)
    package.config = config
    package.checkpoint = checkpoint

    monkeypatch.setattr(hf_config_patch, "_llama_config_cls", lambda: FakeLlamaConfig)
    monkeypatch.delenv(hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV, raising=False)

    return SimpleNamespace(
        modules=created,
        package=package,
        config=config,
        checkpoint=checkpoint,
        upstream=upstream_get_hf_config,
    )


def emit(model: Transformer) -> FakeLlamaConfig:
    """Apply the patch and run ``model`` through it, returning the emitted config."""
    hf_config_patch.apply()
    return sys.modules["olmo_core.nn.hf.config"].get_hf_config(model)


class TestInstallingThePatch:
    """Where the replacement lands, and whether its absence can be seen."""

    def test_it_is_absent_until_it_is_applied(self, olmo_core) -> None:
        """The detectable-absence half. A run that skipped the patch must not look
        patched, because the only other symptom is a NotImplementedError raised after the
        shards have been read, which reads like a checkpoint problem."""
        assert not hf_config_patch.is_applied()
        assert hf_config_patch.apply() is True
        assert hf_config_patch.is_applied()

    def test_is_applied_is_false_when_olmo_core_was_never_imported(self, monkeypatch) -> None:
        """Not an exception. "Not patched" and "not installed" are both answerable
        without importing a package this machine does not have."""
        for name in list(sys.modules):
            if name == "olmo_core" or name.startswith("olmo_core."):
                monkeypatch.delitem(sys.modules, name)
        assert not hf_config_patch.is_applied()

    def test_it_replaces_the_function_in_the_module_that_defines_it(self, olmo_core) -> None:
        hf_config_patch.apply()
        assert olmo_core.config.get_hf_config is not olmo_core.upstream
        assert olmo_core.config.get_hf_config.__wrapped__ is olmo_core.upstream

    def test_it_rebinds_every_module_that_imported_the_name_by_value(self, olmo_core) -> None:
        """The failure this file was written around.

        ``olmo_core.nn.hf.checkpoint`` does ``from olmo_core.nn.hf.config import
        get_hf_config`` at module scope and calls it from ``save_hf_model``. Rebinding only
        the defining module leaves that call site pointing at the original, so conversion
        raises ``NotImplementedError`` while :func:`is_applied` reports success.
        """
        hf_config_patch.apply()
        patched = olmo_core.config.get_hf_config
        assert olmo_core.checkpoint.get_hf_config is patched
        assert olmo_core.package.get_hf_config is patched

    def test_it_leaves_modules_outside_olmo_core_alone(self, olmo_core, monkeypatch) -> None:
        """The scan is by module name, so an unrelated ``get_hf_config`` is not collateral."""
        outsider = types.ModuleType("not_olmo_core_at_all")
        outsider.get_hf_config = olmo_core.upstream
        monkeypatch.setitem(sys.modules, "not_olmo_core_at_all", outsider)

        hf_config_patch.apply()

        assert outsider.get_hf_config is olmo_core.upstream

    def test_applying_it_again_is_a_no_op(self, olmo_core) -> None:
        hf_config_patch.apply()
        first = olmo_core.config.get_hf_config
        assert hf_config_patch.apply() is False
        assert olmo_core.config.get_hf_config is first

    def test_it_never_wraps_itself(self, olmo_core) -> None:
        """Idempotence by marker, not by luck. Nesting would still work and would make
        every refusal message arrive through a stack of identical frames."""
        hf_config_patch.apply()
        hf_config_patch.apply()
        assert olmo_core.config.get_hf_config.__wrapped__ is olmo_core.upstream

    def test_an_olmo_core_without_get_hf_config_is_refused_by_name(
        self, olmo_core, monkeypatch
    ) -> None:
        """Upstream drift fails loudly here rather than quietly at conversion time."""
        monkeypatch.delattr(olmo_core.config, "get_hf_config")
        with pytest.raises(RuntimeError, match="defines no get_hf_config"):
            hf_config_patch.apply()

    def test_an_olmo_core_missing_the_block_classes_is_refused_by_name(
        self, olmo_core, monkeypatch
    ) -> None:
        monkeypatch.delattr(sys.modules["olmo_core.nn.transformer.block"], "TransformerBlock")
        with pytest.raises(RuntimeError, match="not the shape it was derived from"):
            hf_config_patch.apply()


class TestEverythingUpstreamHandledStillTakesTheUpstreamPath:
    """A wrapper, not a rewritten guard: no model that converted before may change."""

    def test_a_reordered_norm_model_is_unaffected(self, olmo_core) -> None:
        model = Transformer([ReorderedNormTransformerBlock()])
        assert emit(model) == "olmo2-config"

    def test_an_moe_model_is_unaffected(self, olmo_core) -> None:
        assert emit(MoETransformer([ReorderedNormTransformerBlock()])) == "flex-olmo-config"

    def test_a_normalized_transformer_still_raises_upstreams_refusal(self, olmo_core) -> None:
        """Checked before the blocks are touched, as upstream checks it, because a
        normalized transformer's blocks are plain and would otherwise be converted."""
        hf_config_patch.apply()
        with pytest.raises(NotImplementedError, match="not implemented for"):
            sys.modules["olmo_core.nn.hf.config"].get_hf_config(NormalizedTransformer())


class TestPlainBlocksConvertInsteadOfRaising:
    def test_upstream_refuses_a_plain_block(self, olmo_core) -> None:
        """The premise. If this ever stops holding, the patch is dead weight and the
        tests below would pass whether or not it were installed."""
        with pytest.raises(NotImplementedError, match="not a ReorderedNormTransformerBlock"):
            olmo_core.upstream(Transformer())

    def test_the_patch_converts_the_same_model(self, olmo_core) -> None:
        emitted = emit(Transformer())
        assert isinstance(emitted, FakeLlamaConfig)


class TestTheEmittedConfig:
    def test_it_describes_the_architecture(self, olmo_core) -> None:
        emitted = emit(Transformer())
        assert emitted.vocab_size == 49152
        assert emitted.hidden_size == 576
        assert emitted.intermediate_size == 1536
        assert emitted.num_hidden_layers == 30
        assert emitted.num_attention_heads == 9
        assert emitted.num_key_value_heads == 3
        assert emitted.head_dim == 64
        assert emitted.hidden_act == "silu"
        assert emitted.rms_norm_eps == 1e-05
        assert emitted.rope_theta == 10000.0
        assert emitted.rope_scaling is None

    def test_tied_embeddings_are_carried_through(self, olmo_core) -> None:
        """True for this family, and it has to be described rather than defaulted.

        ``tie_word_embeddings`` reaches HF as written: with it set, ``save_pretrained``
        drops the head from the safetensors and ``from_pretrained`` re-ties it. With it
        wrongly false the directory still loads and still scores -- on an untied head that
        happens to hold the same 28M-parameter matrix, written to disk twice.
        """
        assert emit(Transformer(tie_word_embeddings=True)).tie_word_embeddings is True

    def test_untied_embeddings_are_carried_through(self, olmo_core) -> None:
        assert emit(Transformer(tie_word_embeddings=False)).tie_word_embeddings is False

    def test_head_dim_falls_back_to_d_model_over_heads(self, olmo_core) -> None:
        """Older configs do not carry it, and a null head_dim reaches HF as a crash."""
        model = Transformer([TransformerBlock(Attention(head_dim=None, n_heads=9))], d_model=576)
        assert emit(model).head_dim == 64

    def test_max_position_embeddings_defaults_to_the_validated_2048(self, olmo_core) -> None:
        """Upstream writes -1 here, which is fine for a config nothing reads and not fine
        for one a server does: vLLM sizes its KV cache from this field."""
        assert emit(Transformer()).max_position_embeddings == 2048

    def test_max_position_embeddings_reads_the_environment(self, olmo_core, monkeypatch) -> None:
        monkeypatch.setenv(hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV, "4096")
        assert emit(Transformer()).max_position_embeddings == 4096

    def test_an_unparseable_max_position_embeddings_is_refused(
        self, olmo_core, monkeypatch
    ) -> None:
        """Refused at apply time, before anything is mutated, rather than at build time
        after the shards have been read."""
        monkeypatch.setenv(hf_config_patch.MAX_POSITION_EMBEDDINGS_ENV, "2048 tokens")
        with pytest.raises(RuntimeError, match="is not an integer"):
            hf_config_patch.apply()
        assert not hf_config_patch.is_applied()

    def test_biases_are_reported_rather_than_assumed(self, olmo_core) -> None:
        model = Transformer(
            [TransformerBlock(Attention(out_bias=object()), FeedForward(w1_bias=object()))]
        )
        emitted = emit(model)
        assert emitted.attention_bias is True
        assert emitted.mlp_bias is True

    def test_a_model_without_biases_says_so(self, olmo_core) -> None:
        emitted = emit(Transformer())
        assert emitted.attention_bias is False
        assert emitted.mlp_bias is False

    def test_the_token_ids_are_left_for_the_tokenizer(self, olmo_core) -> None:
        """Upstream leaves them unset and so does this. Conversion saves the tokenizer
        into the same directory, which is where the ids come from."""
        emitted = emit(Transformer())
        assert emitted.pad_token_id is None
        assert emitted.bos_token_id is None
        assert emitted.eos_token_id is None


class TestFeaturesLlamaCannotRepresentAreRefused:
    """Each of these would otherwise be dropped, and a dropped feature is the expensive
    failure: the directory loads, scores, and reports a theta computed by a model that is
    not the one that was trained."""

    def test_qk_norm(self, olmo_core) -> None:
        model = Transformer([TransformerBlock(Attention(q_norm=Norm()))])
        with pytest.raises(NotImplementedError, match="QK-norm"):
            emit(model)

    def test_rope_scaling(self, olmo_core) -> None:
        model = Transformer([TransformerBlock(Attention(rope=Rope(scaling={"factor": 8.0})))])
        with pytest.raises(NotImplementedError, match="rope scaling"):
            emit(model)

    def test_no_rope_at_all(self, olmo_core) -> None:
        model = Transformer([TransformerBlock(Attention(rope=None))])
        with pytest.raises(NotImplementedError, match="does not use rope"):
            emit(model)

    def test_sliding_window_attention(self, olmo_core) -> None:
        model = Transformer([TransformerBlock(Attention(backend=Backend((4096, 0))))])
        with pytest.raises(NotImplementedError, match="sliding-window"):
            emit(model)

    def test_a_sequence_mixer_that_is_not_attention(self, olmo_core) -> None:
        model = Transformer([TransformerBlock(GatedDeltaNet())])
        with pytest.raises(NotImplementedError, match="sequence mixer is not Attention"):
            emit(model)

    def test_mixed_block_types_name_the_offending_layer(self, olmo_core) -> None:
        """Every block is checked, not just the first.

        One LlamaConfig describes one architecture, so a model whose second layer is
        reordered-norm cannot be represented at all -- and reading the description off
        block 0 would produce a config that is right about 1 layer of 30.
        """
        model = Transformer([TransformerBlock(), ReorderedNormTransformerBlock()])
        with pytest.raises(NotImplementedError, match="Block 1 is not a plain"):
            emit(model)

    def test_a_late_sliding_window_is_caught_too(self, olmo_core) -> None:
        model = Transformer(
            [TransformerBlock(), TransformerBlock(Attention(backend=Backend((4096, 0))))]
        )
        with pytest.raises(NotImplementedError, match="Block 1 uses sliding-window"):
            emit(model)


class TestTheConverterInstallsThePatchItself:
    """The ordering the node drivers had to hold by hand, held by the call site instead."""

    @pytest.fixture
    def staged(self, tmp_path: Path) -> Path:
        d = tmp_path / "native"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model": {"d_model": 576, "n_layers": 30},
                    "dataset": {"tokenizer": {"identifier": "HuggingFaceTB/SmolLM2-135M"}},
                }
            ),
            encoding="utf-8",
        )
        (d / "model_and_optim").mkdir()
        (d / "model_and_optim" / ".metadata").write_bytes(b"\x00")
        return d

    @pytest.fixture
    def events(self, monkeypatch, tmp_path: Path) -> list[str]:
        """Drive the conversion body with the heavy dependencies injected.

        Everything the body touches arrives through the dict :func:`_olmo_core_imports`
        returns, which is what makes the order of its steps checkable where torch is not
        installed.
        """
        log: list[str] = []

        class Dist:
            def __init__(self) -> None:
                self.up = False

            def is_initialized(self) -> bool:
                return self.up

            def init_process_group(self, backend: str) -> None:
                log.append(f"init_process_group:{backend}")
                self.up = True

            def destroy_process_group(self) -> None:
                log.append("destroy_process_group")
                self.up = False

        def save_hf_model(out, state, model, *, dtype, save_overwrite):
            log.append("save_hf_model")
            (Path(out) / "model.safetensors").write_bytes(b"\x00")

        class Tokenizer:
            def save_pretrained(self, out: str) -> None:
                log.append("save_tokenizer")
                (Path(out) / "tokenizer.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            convert,
            "_olmo_core_imports",
            lambda: {
                "dist": Dist(),
                "DType": lambda name: name,
                "TokenizerConfig": SimpleNamespace(
                    from_dict=lambda d: SimpleNamespace(identifier=d["identifier"])
                ),
                "load_model_and_optim_state": lambda path, model: log.append("load_shards"),
                "save_hf_model": save_hf_model,
                "TransformerConfig": SimpleNamespace(
                    from_dict=lambda d: SimpleNamespace(
                        build=lambda init_device: SimpleNamespace(
                            eval=lambda: None, state_dict=lambda: {}
                        )
                    )
                ),
                "AutoTokenizer": SimpleNamespace(from_pretrained=lambda _id: Tokenizer()),
            },
        )
        monkeypatch.setattr(hf_config_patch, "apply", lambda: log.append("apply_patch") or True)
        return log

    def test_the_patch_is_applied_before_the_shards_are_read(
        self, staged: Path, tmp_path: Path, events: list[str]
    ) -> None:
        """Not merely before ``save_hf_model``, which is the only consumer.

        Patching costs nothing and reading 1.7 GB of shards costs minutes, so the failure
        that ``get_hf_config`` would raise belongs before the load rather than after it.
        """
        convert.convert_olmo_core_to_hf(staged, tmp_path / "out")
        assert events.index("apply_patch") < events.index("load_shards")
        assert events.index("apply_patch") < events.index("save_hf_model")

    def test_a_patch_that_cannot_be_applied_stops_before_any_work(
        self, staged: Path, tmp_path: Path, events: list[str], monkeypatch
    ) -> None:
        """No output directory, no process group, nothing to clean up after."""

        def refuse() -> bool:
            raise RuntimeError("the installed transformers provides no LlamaConfig")

        monkeypatch.setattr(hf_config_patch, "apply", refuse)
        out = tmp_path / "out"
        with pytest.raises(RuntimeError, match="LlamaConfig"):
            convert.convert_olmo_core_to_hf(staged, out)
        assert not out.exists()
        assert events == []


@needs_no_transformers
class TestWithoutTransformers:
    def test_the_missing_llama_config_says_which_class_is_missing(self) -> None:
        with pytest.raises(RuntimeError, match="no LlamaConfig"):
            hf_config_patch._llama_config_cls()
