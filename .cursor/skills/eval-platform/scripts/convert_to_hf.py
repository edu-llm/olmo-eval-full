#!/usr/bin/env python3
"""Convert a native OLMo-core checkpoint to HuggingFace format, library-only.

vLLM loads HF-format weights only, so a native OLMo-core checkpoint
(``config.json`` + ``model_and_optim/``) has to be converted before it can be
evaluated. OLMo-core ships a converter at
``src/examples/huggingface/convert_checkpoint_to_hf.py``, but ``src/examples`` is
not packaged into the wheel, so reaching it means cloning a private repository
onto the eval box. That clone is the most fragile step in the pipeline: it needs
GitHub credentials inside the container and it fails late.

This script does the same job using only the installed ``olmo_core`` library, so
``uv sync --extra olmo_core`` is sufficient and nothing needs cloning.

The CLI deliberately mirrors OLMo-core's own converter, so ``$OLMO_CORE_CONVERT``
can point at either one and ``run_eval_sweep.sh`` does not care which it gets.

Two implementations, tried in order, because the library API has moved:

1. ``olmo_core.nn.hf.convert_checkpoint_to_hf``, a single call that also writes
   the tokenizer. This is what OLMo-core's own example script uses.
2. A manual path built from ``save_hf_model`` plus an explicit tokenizer save,
   matching what ``add-cat-evals/CONVERSION.md`` documents. The pinned
   ``ai2-olmo-core==2.4.0`` may not export the function in (1), and there is no
   way to tell without the package installed, so the fallback is not optional.

Scope: standard dense OLMo-2/OLMo-3 architectures, CPU by default. Conversion
needs roughly the model size in RAM at bf16. MoE and fused-attention checkpoints
push conversion onto a GPU and are not covered.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("convert_to_hf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Long names match OLMo-core's converter so the two are interchangeable.
    p.add_argument(
        "-i",
        "--checkpoint-input-path",
        required=True,
        help="Directory holding the OLMo-core checkpoint (config.json + model_and_optim/).",
    )
    p.add_argument(
        "-o",
        "--huggingface-output-dir",
        required=True,
        help="Directory to write the HF-format checkpoint into.",
    )
    p.add_argument(
        "-t",
        "--tokenizer",
        default=None,
        help="HF tokenizer id. Defaults to the one named in the checkpoint config.",
    )
    p.add_argument(
        "-s",
        "--max-sequence-length",
        type=int,
        default=None,
        help="Max sequence length to record. Defaults to the tokenizer's.",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        help="Torch dtype for the saved weights (default bfloat16).",
    )
    p.add_argument(
        "--skip-validation",
        dest="validate",
        action="store_false",
        help="Skip verifying the converted model matches the original. Roughly halves"
        " peak memory and is the default in the sweep's hot path.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device to convert on. Defaults to CPU, which is correct for dense archs.",
    )
    return p.parse_args()


def load_experiment_config(ckpt_dir: str) -> dict:
    """Read the checkpoint's experiment config.

    Prefers ``olmo_core.nn.hf.load_config``, which knows about remote paths, and
    falls back to reading ``config.json`` directly so a missing export does not
    stop us before the real work starts.
    """
    try:
        from olmo_core.nn.hf import load_config

        cfg = load_config(ckpt_dir)
        if cfg is not None:
            return cfg
        raise RuntimeError("load_config returned None")
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.info("falling back to reading config.json directly (%s)", exc)
        path = Path(ckpt_dir) / "config.json"
        if not path.is_file():
            raise SystemExit(
                f"no experiment config at {path}; is {ckpt_dir} a checkpoint directory?"
            ) from exc
        return json.loads(path.read_text(encoding="utf-8"))


def convert_via_library(args: argparse.Namespace, cfg: dict) -> bool:
    """Single-call conversion. Returns False if this API is unavailable."""
    try:
        from olmo_core.config import DType
        from olmo_core.nn.hf import convert_checkpoint_to_hf
    except ImportError as exc:
        logger.info("high-level converter unavailable (%s); trying manual path", exc)
        return False

    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise SystemExit("checkpoint config has no 'model' section; cannot convert")
    tokenizer_cfg = (cfg.get("dataset") or {}).get("tokenizer")
    if tokenizer_cfg is None and args.tokenizer is None:
        raise SystemExit(
            "checkpoint config names no dataset.tokenizer and no -t/--tokenizer was"
            " given; cannot determine which tokenizer to save"
        )

    import torch

    logger.info("converting via olmo_core.nn.hf.convert_checkpoint_to_hf")
    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=model_cfg,
        tokenizer_config_dict=tokenizer_cfg or {},
        dtype=DType(args.dtype),
        tokenizer_id=args.tokenizer,
        max_sequence_length=args.max_sequence_length,
        validate=args.validate,
        device=torch.device(args.device) if args.device else None,
    )
    return True


def convert_manually(args: argparse.Namespace, cfg: dict) -> None:
    """Build the model on meta, load the sharded state, save HF, save tokenizer.

    The path documented in add-cat-evals/CONVERSION.md. Used when the one-call
    API above is not present in the installed olmo_core.
    """
    import torch
    from olmo_core.config import DType
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.hf.checkpoint import save_hf_model
    from olmo_core.nn.transformer.config import TransformerConfig
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise SystemExit("checkpoint config has no 'model' section; cannot convert")

    ckpt = Path(args.checkpoint_input_path)
    out = Path(args.huggingface_output_dir)

    logger.info("building model on meta device")
    model = TransformerConfig.from_dict(model_cfg).build(init_device="meta")
    # to_empty allocates real storage without initializing it; the checkpoint
    # load below fills every parameter, so initialization would be wasted work.
    model.to_empty(device=torch.device(args.device or "cpu"))

    logger.info("loading sharded state from %s", ckpt / "model_and_optim")
    load_model_and_optim_state(str(ckpt / "model_and_optim"), model)

    logger.info("gathering state dict")
    state = get_model_state_dict(model, options=StateDictOptions(cpu_offload=True))

    logger.info("saving HF model to %s", out)
    save_hf_model(
        str(out),
        state,
        model,
        dtype=DType(args.dtype),
        save_overwrite=True,
    )

    # save_hf_model writes weights and config but not the tokenizer, and vLLM
    # needs one alongside the weights.
    tokenizer_id = args.tokenizer or (cfg.get("dataset") or {}).get("tokenizer", {}).get(
        "identifier"
    )
    if not tokenizer_id:
        logger.warning(
            "no tokenizer id resolved; the output has no tokenizer and vLLM will"
            " likely refuse it. Pass -t <hf-tokenizer-id>."
        )
        return
    from transformers import AutoTokenizer

    logger.info("saving tokenizer %s", tokenizer_id)
    AutoTokenizer.from_pretrained(tokenizer_id).save_pretrained(str(out))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[convert_to_hf] %(message)s")
    args = parse_args()

    ckpt = Path(args.checkpoint_input_path)
    if not ckpt.is_dir():
        raise SystemExit(f"not a directory: {ckpt}")

    cfg = load_experiment_config(args.checkpoint_input_path)

    if not convert_via_library(args, cfg):
        convert_manually(args, cfg)

    out = Path(args.huggingface_output_dir)
    weights = list(out.glob("*.safetensors")) + list(out.glob("*.bin"))
    if not weights:
        raise SystemExit(f"conversion produced no weight files in {out}")
    logger.info("done: %d weight file(s) in %s", len(weights), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
