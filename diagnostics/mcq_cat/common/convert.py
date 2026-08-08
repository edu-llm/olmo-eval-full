"""Checkpoint preparation: what happens to a staged directory before a backend loads it.

A checkpoint arrives from :func:`~diagnostics.mcq_cat.common.s3_io.resolve_checkpoint` in
whatever format training wrote, and the backends do not all read the same one. The MCQ and
generative scorers both load through ``transformers``, which needs a HuggingFace directory
-- ``config.json`` naming an architecture, weights, and a tokenizer. A checkpoint written
by OLMo-core is none of those: it is an olmo-core experiment ``config.json`` beside a
sharded distributed checkpoint under ``model_and_optim/``, and it carries no architecture
at all, which is why nothing can read it without ``ai2-olmo-core`` to rebuild the model
from the config first.

This module is the seam between those two facts, and it is a *policy* rather than a fixed
step. ``auto`` detects the layout and converts when it has to; ``none`` hands the directory
to the backend untouched. The second exists because converting is not always the right
answer -- a backend that reads OLMo-core natively wants the original, and on a
disk-constrained shape one copy of a checkpoint is meaningfully different from two.

Detection and dispatch import nothing heavier than :mod:`json` and :mod:`pathlib`, so they
are exercised wherever the tests run. Only :func:`convert_olmo_core_to_hf` needs torch and
``ai2-olmo-core``, and it imports them inside the call.

Ported from ``SteeringVectors/run_steering_eval.py`` on ``origin/SteeringVectors``, which
is the only conversion in this organization that has run against a real checkpoint on the
platform.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import hf_config_patch

log = logging.getLogger("mcq_cat.convert")

#: The preparation policies :func:`prepare_checkpoint` accepts.
PREP_AUTO = "auto"
PREP_NONE = "none"
PREP_POLICIES = (PREP_AUTO, PREP_NONE)

#: The precisions conversion will write, and the default.
#:
#: ``olmo_core.config.DType`` also names ``float8_e4m3fn`` and ``float8_e5m2``, and they
#: are deliberately not here. ``save_hf_model`` would cast to them and
#: ``save_pretrained`` would write them, but the ``LlamaConfig`` this pipeline emits
#: carries no quantization block, so ``AutoModelForCausalLM.from_pretrained`` cannot read
#: the result back -- the conversion would succeed and the load two lines later would not.
#: Offering a value whose output nothing downstream can open is worse than not offering
#: it, because the refusal arrives after the shards have been read.
DTYPE_DEFAULT = "bfloat16"
CONVERSION_DTYPES = (DTYPE_DEFAULT, "float16", "float32")

_HF_WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
_HF_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")


def _read_config(local_dir: Path) -> dict[str, Any] | None:
    """Return the parsed ``config.json``, or ``None`` if absent or unreadable.

    Unreadable is deliberately not an error here. Both detectors ask this question while
    deciding what a directory *is*, and a truncated ``config.json`` is a partially-synced
    checkpoint rather than a third format -- the ladder below has a branch that says so.
    """
    config_path = local_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_hf_checkpoint(local_dir: Path) -> bool:
    """Return ``True`` for a directory ``transformers`` can load as it stands.

    Weights *and* a tokenizer, because either alone is a checkpoint mid-upload rather than
    a loadable one, and ``from_pretrained`` fails differently for each.
    """
    has_weights = any((local_dir / name).exists() for name in _HF_WEIGHT_FILES)
    has_tokenizer = any((local_dir / name).exists() for name in _HF_TOKENIZER_FILES)
    return has_weights and has_tokenizer


def is_olmo_core_checkpoint(local_dir: Path) -> bool:
    """Return ``True`` for a raw OLMo-core checkpoint.

    Two signatures, and the sharded one is checked first because it is positive evidence:
    ``model_and_optim/.metadata`` is written by ``torch.distributed.checkpoint`` and
    nothing else here produces it.

    The config signature is the fallback, and ``architectures`` is what makes it safe. An
    olmo-core experiment config nests ``model`` and ``dataset`` blocks and names no
    architecture; an HF config names one. Without that third clause a converted directory
    would answer ``True`` to both detectors and the ladder's order would silently become
    load-bearing.
    """
    if (local_dir / "model_and_optim" / ".metadata").exists():
        return True
    config = _read_config(local_dir)
    return bool(
        config
        and isinstance(config.get("model"), dict)
        and isinstance(config.get("dataset"), dict)
        and "architectures" not in config
    )


def describe_layout(local_dir: Path) -> str:
    """Name what is on disk, for a log line or an error a person has to act on.

    Errors here are read by someone deciding whether they typed the wrong prefix or their
    upload died, and those need different responses, so this reports what was found rather
    than only that nothing matched.
    """
    if not local_dir.exists():
        return "path does not exist"
    if not local_dir.is_dir():
        return "not a directory"
    present = sorted(p.name for p in local_dir.iterdir())
    if not present:
        return "empty directory"
    weights = [name for name in _HF_WEIGHT_FILES if (local_dir / name).exists()]
    tokenizer = [name for name in _HF_TOKENIZER_FILES if (local_dir / name).exists()]
    return (
        f"config.json={'yes' if (local_dir / 'config.json').exists() else 'no'}, "
        f"weights={weights or 'none'}, tokenizer={tokenizer or 'none'}, "
        f"entries={present[:8]}{'...' if len(present) > 8 else ''}"
    )


def _check_dtype(dtype: str) -> None:
    """Refuse a precision this pipeline cannot write, before anything expensive happens.

    The runner's ``--dtype`` has ``choices``, so a bad value from the command line never
    reaches here. A bad value from a caller would otherwise reach ``DType(dtype)``, which
    is evaluated as an argument to ``save_hf_model`` -- after the model has been rebuilt
    and the shards read -- and that is the same late-refusal shape
    :mod:`~diagnostics.mcq_cat.common.hf_config_patch` exists to remove.
    """
    if dtype not in CONVERSION_DTYPES:
        raise ValueError(
            f"Unknown conversion dtype {dtype!r} (expected one of: {', '.join(CONVERSION_DTYPES)})"
        )


def ensure_hf_checkpoint(local_dir: Path, out_dir: Path, dtype: str = DTYPE_DEFAULT) -> Path:
    """Return a directory ``transformers`` can load, converting only when necessary.

    Four branches, in this order:

    1. Already HF. Returned untouched, and nothing is copied -- the common case once a
       training run adopts the OLMo-to-HF converter, and it must not cost a rewrite.
    2. Raw OLMo-core. Converted into ``out_dir``.
    3. A ``config.json`` matching neither. Passed through with a warning naming what was
       found. Refusing here would be guessing: ``transformers`` knows more about what it
       can open than this function does, and its error names the actual problem. This is
       also the branch a partially-synced checkpoint lands in, which is why the warning
       reports the contents rather than only the verdict.
    4. Anything else. Raises, naming the path and the layout.
    """
    if is_hf_checkpoint(local_dir):
        log.info("Checkpoint at %s is already HF format; using it as staged", local_dir)
        return local_dir
    if is_olmo_core_checkpoint(local_dir):
        log.info("Detected OLMo-core checkpoint at %s; converting to HF", local_dir)
        return convert_olmo_core_to_hf(local_dir, out_dir, dtype=dtype)
    if (local_dir / "config.json").exists():
        log.warning(
            "Checkpoint at %s is neither clearly HF nor OLMo-core (%s); handing it to "
            "the loader as-is, which will report what it cannot read. A checkpoint that "
            "was interrupted mid-upload looks exactly like this.",
            local_dir,
            describe_layout(local_dir),
        )
        return local_dir
    raise ValueError(
        f"Unrecognized checkpoint layout at {local_dir} ({describe_layout(local_dir)})"
    )


def prepare_checkpoint(
    local_dir: Path,
    work_dir: Path,
    *,
    policy: str = PREP_AUTO,
    dtype: str = DTYPE_DEFAULT,
) -> Path:
    """Apply the preparation ``policy`` to a staged checkpoint and return what to load.

    The seam the runner calls, and the reason the backend does not have to care what
    training wrote.

    ``auto`` runs :func:`ensure_hf_checkpoint`. ``none`` returns ``local_dir`` unchanged,
    for a backend that reads the original format or a checkpoint prepared out of band.

    ``none`` logs rather than passing silently. A run that skipped preparation and then
    failed inside a loader should say so in its own output, because the two failures --
    "this backend cannot read that format" and "preparation was turned off" -- have the
    same shape at the point they surface and different fixes.

    ``dtype`` is the precision conversion writes, so under ``none`` it describes nothing
    and is warned about rather than ignored quietly -- but only when it was moved off the
    default, because the default is what a caller gets for not asking. Someone who set
    ``--dtype float16`` because they were pointed at a card with no bfloat16 believes they
    have chosen a precision; under ``none`` they have not, the weights load at whatever
    training wrote, and the failure arrives as a kernel refusing a format on the card,
    which reads like the flag was broken rather than inapplicable.
    """
    if policy not in PREP_POLICIES:
        raise ValueError(
            f"Unknown checkpoint preparation policy {policy!r} "
            f"(expected one of: {', '.join(PREP_POLICIES)})"
        )
    _check_dtype(dtype)
    if policy == PREP_NONE:
        log.info(
            "Checkpoint preparation is off (policy=none); loading %s exactly as staged, "
            "whatever format it is in",
            local_dir,
        )
        if dtype != DTYPE_DEFAULT:
            log.warning(
                "dtype=%s has no effect under --checkpoint-prep none: nothing is "
                "converted, so the weights load at whatever precision they were written "
                "at. If this was set to avoid a precision the hardware lacks, that has "
                "not happened.",
                dtype,
            )
        return local_dir
    return ensure_hf_checkpoint(local_dir, work_dir, dtype=dtype)


def _olmo_core_imports() -> dict[str, Any]:
    """Import the conversion dependencies, or say which extra is missing.

    Lazy so everything above this line runs where torch does not, which is most places
    this package is imported: bank vendoring, resolver checks and the whole test suite.
    """
    try:
        import torch.distributed as dist
        from olmo_core.config import DType
        from olmo_core.data import TokenizerConfig
        from olmo_core.distributed.checkpoint import load_model_and_optim_state
        from olmo_core.nn.hf import save_hf_model
        from olmo_core.nn.transformer import TransformerConfig
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Converting a raw OLMo-core checkpoint needs torch, transformers and "
            "ai2-olmo-core, and one of them is not importable here "
            f"({exc}). A sharded OLMo-core checkpoint carries no architecture, so it "
            "cannot be rebuilt without the library that wrote it -- there is no "
            "pure-Python fallback. Install the olmo_core extra "
            "(`uv sync --extra olmo_core`), pre-convert the checkpoint elsewhere and "
            "pass the HF directory, or run with --checkpoint-prep none against a "
            "backend that reads the native format."
        ) from exc
    return {
        "dist": dist,
        "DType": DType,
        "TokenizerConfig": TokenizerConfig,
        "load_model_and_optim_state": load_model_and_optim_state,
        "save_hf_model": save_hf_model,
        "TransformerConfig": TransformerConfig,
        "AutoTokenizer": AutoTokenizer,
    }


def convert_olmo_core_to_hf(local_dir: Path, out_dir: Path, *, dtype: str = DTYPE_DEFAULT) -> Path:
    """Rebuild a sharded OLMo-core checkpoint as an HF directory and return its path.

    The model is reconstructed from the experiment ``config.json``, the distributed
    checkpoint is loaded into it in place, and the weights are written back out in HF
    layout with the tokenizer the run was trained on, so the result opens with
    ``AutoModelForCausalLM.from_pretrained`` and ``AutoTokenizer.from_pretrained``.

    Three details are not obvious and each one is load-bearing:

    ``load_model_and_optim_state`` reads a ``torch.distributed`` checkpoint and expects a
    process group even for a single rank. One gloo rank is enough and costs nothing, and
    it is torn down in a ``finally`` so a converted checkpoint does not leave a live group
    behind for whatever runs next in the process.

    ``save_hf_model`` calls ``get_hf_config``, which upstream refuses to build for the
    plain pre-norm blocks this checkpoint family uses, so
    :func:`~diagnostics.mcq_cat.common.hf_config_patch.apply` runs first. It is applied
    here, before any filesystem or network work, rather than by a separate step: the
    refusal would otherwise arrive after the shards have been read, and a patch that
    installs itself where it is needed cannot be sequenced wrongly.

    ``save_hf_model`` refuses a directory that already exists, and the directory has to
    exist first because the tokenizer is saved into it. ``save_overwrite`` resolves that,
    and is safe only because the caller owns ``out_dir`` -- the runner passes a path under
    its own tempdir.

    The tokenizer is an identifier in the config, not files on disk, so it is fetched and
    written out beside the weights. Without that step the directory has weights and no
    tokenizer, which is exactly the shape :func:`is_hf_checkpoint` rejects.

    Args:
        local_dir: A checkpoint directory holding ``config.json`` and ``model_and_optim/``.
        out_dir: Where to write the HF directory. Created if absent, overwritten if not.
        dtype: Saved weight precision, one of :data:`CONVERSION_DTYPES`. Defaults to
            bfloat16, which every card this runs on has -- except Turing, where torch has
            no bfloat16 at all and ``float16`` is the only option. Deliberately a
            parameter, and reachable from the runner's ``--dtype``: the reference this is
            ported from hardcodes it, and the fleet that hit a T4 had to patch the file on
            every box.
    """
    _check_dtype(dtype)
    imports = _olmo_core_imports()
    # After the imports, because the rebinding scan can only reach modules that are
    # loaded, and before everything else, because this is the cheapest way this function
    # can fail.
    hf_config_patch.apply()
    dist = imports["dist"]

    config = _read_config(local_dir)
    if not config or not isinstance(config.get("model"), dict):
        raise ValueError(
            f"No OLMo-core config.json with a 'model' block at {local_dir} "
            f"({describe_layout(local_dir)}). Conversion rebuilds the model from that "
            f"block, so there is nothing to rebuild without it."
        )

    started_group = False
    if not dist.is_initialized():
        import os

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")
        started_group = True

    try:
        model = imports["TransformerConfig"].from_dict(config["model"]).build(init_device="cpu")
        model.eval()
        imports["load_model_and_optim_state"](str(local_dir / "model_and_optim"), model)

        out_dir.mkdir(parents=True, exist_ok=True)
        imports["save_hf_model"](
            str(out_dir),
            model.state_dict(),
            model,
            dtype=imports["DType"](dtype),
            save_overwrite=True,
        )

        tokenizer_id = _resolve_tokenizer_id(config, imports["TokenizerConfig"])
        imports["AutoTokenizer"].from_pretrained(tokenizer_id).save_pretrained(str(out_dir))
        log.info(
            "Converted OLMo-core checkpoint %s -> %s (dtype=%s, tokenizer=%s)",
            local_dir,
            out_dir,
            dtype,
            tokenizer_id,
        )
    finally:
        if started_group and dist.is_initialized():
            dist.destroy_process_group()

    if not is_hf_checkpoint(out_dir):
        raise RuntimeError(
            f"Conversion wrote {out_dir} but it is not loadable as HF "
            f"({describe_layout(out_dir)}). Refusing here rather than letting the loader "
            f"fail after the model has been rebuilt once."
        )
    return out_dir


def _resolve_tokenizer_id(config: dict[str, Any], tokenizer_config_cls: Any) -> str:
    """Name the HF tokenizer this checkpoint was trained on, falling back to dolma2.

    The fallback is the OLMo default and is what the reference conversion uses, but it is
    a guess about the vocabulary a model was trained against, so it is logged rather than
    taken quietly -- a mismatched tokenizer scores every continuation against the wrong
    token ids and produces a theta rather than an error.
    """
    dataset = config.get("dataset")
    if isinstance(dataset, dict) and isinstance(dataset.get("tokenizer"), dict):
        resolved = getattr(tokenizer_config_cls.from_dict(dataset["tokenizer"]), "identifier", None)
        if resolved:
            return str(resolved)
    fallback = tokenizer_config_cls.dolma2()
    log.warning(
        "This checkpoint's config names no tokenizer identifier; falling back to %s. "
        "If the run was trained on a different vocabulary, scoring is against the wrong "
        "token ids and the ability estimate is meaningless rather than wrong-looking.",
        fallback.identifier,
    )
    return str(fallback.identifier)
