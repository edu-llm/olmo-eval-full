"""Teach OLMo-core's HF exporter to emit a ``LlamaConfig`` for plain pre-norm blocks.

``olmo_core.nn.hf.get_hf_config`` builds a config only for
``ReorderedNormTransformerBlock`` -- the OLMo-2/OLMo-3 post-norm family -- and raises
``NotImplementedError`` for everything else. Preston's checkpoints are plain pre-norm
``TransformerBlock`` models, which is the SmolLM2 shape, so without this the conversion in
:mod:`~diagnostics.mcq_cat.common.convert` fails every time, and it fails late: after the
model has been rebuilt and 1.7 GB of sharded weights have been read into it.

Only the config is missing. ``convert_state_to_hf`` dispatches on ``config.model_type`` and
already carries ``llama`` weight mappings, so returning a ``LlamaConfig`` engages them with
no other change.

**Why a runtime monkeypatch rather than a rewrite of the installed package.** The version
this is derived from -- ``hf-converter-patch/apply_llama_config_patch.py``, which the
memory-split node drivers ran -- edits ``config.py`` in site-packages and is invoked as a
separate script. That works, and it costs two ordering hazards that have to be honoured by
whatever drives the node: the rewrite must land after ``uv sync`` and before the first
``uv run``, and ``UV_NO_SYNC=1`` must hold afterwards or a re-sync silently reinstates the
unpatched wheel and the failure comes back looking like a different bug. Both hazards are
properties of a shell script's ordering, so neither is visible to anything that could
check it.

Installing the patch from inside the converter removes them. There is no persistent
artifact for a re-sync to revert, nothing to sequence against the install, and no
``config.py.orig`` to leave behind. It is also the only version of this that can be tested
where torch is not installed and site-packages is not writable, which is where this
repository's tests run.

The trade is that a monkeypatch has to deal with something the textual rewrite got for
free. ``olmo_core.nn.hf.checkpoint`` does ``from olmo_core.nn.hf.config import
get_hf_config`` at module scope, so rebinding the name in its defining module does nothing
for the module that actually calls it. :func:`apply` therefore imports the package first,
so every by-value holder is loaded, and then rebinds all of them. That is one scan and it
has a test; the hazard it replaces had neither.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any

log = logging.getLogger("mcq_cat.convert")

#: Set on the replacement so :func:`is_applied` can recognize it and :func:`apply` can
#: decline to wrap itself. The name is checked, not the identity, because a re-import of
#: this module would produce a second marker object for the same patch.
PATCH_MARKER = "_mcq_cat_emits_llama_config_for_plain_blocks"

#: Read for ``max_position_embeddings``, which the model object does not carry. The
#: validated node configuration exported it as 2048, matching this checkpoint family's
#: ``sequence_length``, so that is also the default and the variable only has to be set to
#: override it.
MAX_POSITION_EMBEDDINGS_ENV = "OLMO_CORE_HF_MAX_POSITION_EMBEDDINGS"
DEFAULT_MAX_POSITION_EMBEDDINGS = 2048

#: The module that defines ``get_hf_config``, and the package whose import drags in every
#: module that binds it by value.
_CONFIG_MODULE = "olmo_core.nn.hf.config"
_PACKAGE = "olmo_core.nn.hf"


def max_position_embeddings() -> int:
    """Return the context length to write into the emitted config.

    Upstream's OLMo-2 path hardcodes ``-1``, which is harmless for a config that is only
    round-tripped and not harmless for one a server reads: vLLM sizes its KV cache from
    this field and rejects a negative value. The model object does not know the sequence
    length it was trained at, so it is supplied out of band.
    """
    raw = os.environ.get(MAX_POSITION_EMBEDDINGS_ENV)
    if raw is None:
        return DEFAULT_MAX_POSITION_EMBEDDINGS
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{MAX_POSITION_EMBEDDINGS_ENV}={raw!r} is not an integer. It becomes "
            f"max_position_embeddings in the converted config, so a bad value produces a "
            f"directory that loads and then truncates or over-allocates silently."
        ) from exc


def is_applied() -> bool:
    """Return whether this process's ``get_hf_config`` is the patched one.

    Answers about the process as it stands and imports nothing, so a caller can distinguish
    "not patched" from "olmo_core was never loaded" by checking the module itself.
    """
    module = sys.modules.get(_CONFIG_MODULE)
    if module is None:
        return False
    return getattr(getattr(module, "get_hf_config", None), PATCH_MARKER, False) is True


def apply() -> bool:
    """Install the patch in this process and return whether it was newly installed.

    Idempotent. Safe to call before every conversion, which is what
    :func:`~diagnostics.mcq_cat.common.convert.convert_olmo_core_to_hf` does rather than
    relying on a module-level side effect that would fire on import in processes that never
    convert anything.

    The package is imported before the defining module on purpose. Importing
    ``olmo_core.nn.hf`` executes its ``__init__``, which pulls in ``checkpoint.py`` and
    every other module that binds ``get_hf_config`` by value; the rebinding scan below then
    catches all of them at once. Anything imported *after* this point picks up the patched
    function from the defining module anyway, so the result does not depend on the order
    the caller happened to import things in.
    """
    if is_applied():
        return False

    # Both resolved before anything is mutated, so an unparseable environment or a
    # transformers without LlamaConfig refuses rather than leaving half a patch installed.
    # The value itself is read again at build time, so it can still be changed afterwards.
    max_positions = max_position_embeddings()
    llama_config_cls = _llama_config_cls()
    try:
        importlib.import_module(_PACKAGE)
        config_module = importlib.import_module(_CONFIG_MODULE)
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot patch {_CONFIG_MODULE}: ai2-olmo-core is not importable ({exc}). "
            f"Conversion needs the edu-llm/OLMo-core fork; see the sync line in "
            f".edullm/run.yaml."
        ) from exc

    original = getattr(config_module, "get_hf_config", None)
    if original is None:
        raise RuntimeError(
            f"{_CONFIG_MODULE} defines no get_hf_config. The installed olmo_core is not "
            f"the shape this patch was derived from; re-derive it rather than forcing it."
        )
    if getattr(original, PATCH_MARKER, False):
        return False

    symbols = _olmo_core_symbols()
    patched = _make_patched_get_hf_config(original, symbols, llama_config_cls)
    rebound = _rebind(original, patched)

    log.info(
        "Patched get_hf_config to emit a LlamaConfig for plain TransformerBlock models "
        "(max_position_embeddings=%d, rebound in: %s)",
        max_positions,
        ", ".join(rebound) or "<nothing>",
    )
    return True


def _llama_config_cls() -> Any:
    """Return ``transformers.LlamaConfig``, resolved before anything else happens.

    Eagerly rather than at first use. The alternative fails partway through a conversion
    that has already read the shards, and the only models this converter is pointed at are
    the ones that need this class.
    """
    try:
        from transformers import LlamaConfig
    except ImportError as exc:
        raise RuntimeError(
            "The installed transformers provides no LlamaConfig, so a plain "
            f"TransformerBlock model cannot be converted ({exc})."
        ) from exc
    return LlamaConfig


def _olmo_core_symbols() -> dict[str, Any]:
    """Collect the classes the patch dispatches on.

    Imported here rather than at module scope so this module stays importable where
    ``ai2-olmo-core`` is not installed, which is every machine the tests run on.
    """
    try:
        from olmo_core.nn.attention import Attention
        from olmo_core.nn.transformer.block import (
            ReorderedNormTransformerBlock,
            TransformerBlock,
        )
        from olmo_core.nn.transformer.model import MoETransformer, NormalizedTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot patch get_hf_config: olmo_core does not expose the block and "
            f"attention classes this patch dispatches on ({exc}). The installed version is "
            f"not the shape it was derived from."
        ) from exc
    return {
        "Attention": Attention,
        "TransformerBlock": TransformerBlock,
        "ReorderedNormTransformerBlock": ReorderedNormTransformerBlock,
        "delegating_models": (MoETransformer, NormalizedTransformer),
    }


def _rebind(original: Any, patched: Any) -> list[str]:
    """Point every loaded ``olmo_core`` module holding ``original`` at ``patched``.

    ``checkpoint.py`` imports the name rather than the module, so rebinding only the
    defining module would leave the one caller that matters untouched -- the conversion
    would run unpatched and raise exactly the ``NotImplementedError`` this exists to
    prevent, while :func:`is_applied` reported success.

    Scoped to ``olmo_core`` by module name. A wider scan would rebind any unrelated
    ``get_hf_config`` that happened to be the same object, which cannot occur but is not
    worth relying on.
    """
    rebound: list[str] = []
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "olmo_core" or name.startswith("olmo_core.")):
            continue
        if getattr(module, "get_hf_config", None) is original:
            module.get_hf_config = patched
            rebound.append(name)
    return rebound


def _make_patched_get_hf_config(original: Any, symbols: dict[str, Any], llama_config_cls: Any):
    """Wrap ``original`` so plain pre-norm blocks get a ``LlamaConfig`` and nothing else
    changes.

    A wrapper rather than a rewritten guard. Everything upstream already handled -- MoE,
    normalized transformers, reordered-norm blocks and each of their refusals -- reaches
    ``original`` unmodified, so this cannot alter the output for any model that converted
    before. Only the case that used to raise is intercepted.
    """
    attention_cls = symbols["Attention"]
    block_cls = symbols["TransformerBlock"]
    reordered_cls = symbols["ReorderedNormTransformerBlock"]
    delegating_models = symbols["delegating_models"]

    def get_hf_config(model: Any) -> Any:
        """Return an HF config for ``model``, extending upstream to plain blocks."""
        if isinstance(model, delegating_models):
            return original(model)
        blocks = list(model.blocks.values())
        if not blocks:
            return original(model)
        first_block = blocks[0]
        # ReorderedNormTransformerBlock subclasses TransformerBlock, so the reordered
        # check has to come first or every OLMo-2 model looks like a plain block and
        # silently converts through the wrong path.
        if isinstance(first_block, reordered_cls) or not isinstance(first_block, block_cls):
            return original(model)
        return _build_llama_config(
            model,
            blocks,
            first_block,
            attention_cls=attention_cls,
            block_cls=block_cls,
            reordered_cls=reordered_cls,
            llama_config_cls=llama_config_cls,
        )

    get_hf_config.__module__ = getattr(original, "__module__", _CONFIG_MODULE)
    get_hf_config.__qualname__ = "get_hf_config"
    get_hf_config.__wrapped__ = original
    setattr(get_hf_config, PATCH_MARKER, True)
    return get_hf_config


def _build_llama_config(
    model: Any,
    blocks: list[Any],
    first_block: Any,
    *,
    attention_cls: Any,
    block_cls: Any,
    reordered_cls: Any,
    llama_config_cls: Any,
) -> Any:
    """Build a ``LlamaConfig`` for a model of standard pre-norm blocks.

    Every feature Llama cannot express raises rather than being dropped. A config that
    quietly omits QK-norm, rope scaling or a sliding window would still load and still
    produce numbers, and the numbers would be wrong in a way nothing downstream can see --
    which for a pipeline whose output is an ability estimate is the worst available
    failure.

    The description is read off the first block after every block has been checked against
    it, so a model whose layers disagree is refused rather than represented by its first
    layer.
    """
    for idx, block in enumerate(blocks):
        if isinstance(block, reordered_cls) or not isinstance(block, block_cls):
            raise NotImplementedError(
                f"Block {idx} is not a plain {block_cls.__name__}; mixed block types "
                f"cannot be represented by a single LlamaConfig"
            )
        attn = block.attention
        if not isinstance(attn, attention_cls):
            raise NotImplementedError(
                f"Block {idx} sequence mixer is not {attention_cls.__name__}, unable to "
                f"build LlamaConfig"
            )
        if attn.rope is None:
            raise NotImplementedError(f"Block {idx} does not use rope, unable to build LlamaConfig")
        if getattr(attn.rope, "scaling", None) is not None:
            raise NotImplementedError(
                f"Block {idx} uses rope scaling, which this patch does not translate"
            )
        if getattr(attn, "q_norm", None) is not None or getattr(attn, "k_norm", None) is not None:
            raise NotImplementedError(
                f"Block {idx} uses QK-norm, which LlamaConfig cannot represent"
            )
        backend = getattr(attn, "backend", None)
        if backend is not None and getattr(backend, "window_size", (-1, -1)) != (-1, -1):
            raise NotImplementedError(
                f"Block {idx} uses sliding-window attention, which LlamaConfig cannot represent"
            )

    attn = first_block.attention
    feed_forward = first_block.feed_forward

    head_dim = getattr(attn, "head_dim", None) or model.d_model // attn.n_heads
    w1 = getattr(feed_forward, "w1", None)

    return llama_config_cls(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        intermediate_size=feed_forward.hidden_size,
        num_hidden_layers=model.n_layers,
        num_attention_heads=attn.n_heads,
        num_key_value_heads=attn.n_kv_heads,
        head_dim=head_dim,
        # Hardcoded as upstream's OLMo-2 path hardcodes it, and as the version of this
        # patch that converted 82 checkpoints hardcoded it. This checkpoint family's
        # config says silu; a model that used anything else would be mis-described here,
        # and detecting that reliably needs a feed-forward attribute olmo_core does not
        # expose consistently across the versions this has to run against.
        hidden_act="silu",
        max_position_embeddings=max_position_embeddings(),
        attention_bias=attn.w_out.bias is not None,
        mlp_bias=getattr(w1, "bias", None) is not None,
        rope_theta=float(attn.rope.theta),
        rope_scaling=None,
        rms_norm_eps=first_block.feed_forward_norm.eps,
        # Carried through rather than defaulted. This family ties them, and a config that
        # said otherwise would make `save_pretrained` write a second copy of a 28M-parameter
        # embedding matrix and `from_pretrained` load an untied head that happens to hold
        # the same values -- correct output, silently wrong description.
        tie_word_embeddings=model.tie_word_embeddings,
        # Left unset here, as upstream leaves them. They arrive in the output directory
        # through the tokenizer that conversion saves beside the weights.
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )
