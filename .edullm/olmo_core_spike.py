"""Phase 1 spike: ask OLMo-core what it does, rather than assume it.

Throwaway. This is not a test and nothing imports it. Its only product is the text it
prints, which becomes the written record phase 2's loader is designed against. Delete it
once that record exists.

The reason it exists at all: a previous attempt wrote both CAT loaders in an environment
with no torch and no ai2-olmo-core, inferring every OLMo-core API contract from how
``OlmoCoreProvider`` happened to use it. Roughly 430 lines were built on six assumptions
and none of them ever executed. Each probe below turns one of those assumptions into an
observation.

Every probe is independent and swallows its own failure. A run that cannot load the model
must still report the tokenizer's BOS behaviour, because a second GPU hour to learn the
second fact is the thing this is meant to avoid.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from typing import Any

FINDINGS: list[tuple[str, str]] = []


def record(question: str, answer: str) -> None:
    FINDINGS.append((question, answer))
    print(f"  -> {answer}", flush=True)


def probe(number: str, question: str):
    """Run one probe, report it, and never let it end the run."""

    def decorator(fn):
        print(f"\n{'=' * 78}\n[{number}] {question}\n{'=' * 78}", flush=True)
        started = time.monotonic()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - reporting is the point
            elapsed = time.monotonic() - started
            print(f"  FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            FINDINGS.append((question, f"FAILED - {type(exc).__name__}: {exc}"))
        else:
            print(f"  ({time.monotonic() - started:.1f}s)", flush=True)
        return fn

    return decorator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR"),
        help="OLMo-core step directory, local or s3://. Defaults to $EDULLM_CHECKPOINT_DIR.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help=(
            "Compute dtype. Named on the command line rather than set in code so the "
            "platform's bfloat16_not_in_the_hardware guard can read it out of argv; a "
            "Turing card would otherwise refuse the first kernel after being allocated."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    print("=" * 78)
    print("OLMo-core API spike")
    print("=" * 78)
    print(f"checkpoint: {args.checkpoint}")
    print(f"dtype:      {args.dtype}")

    # The platform hands the container several variables and the docs do not say how
    # EDULLM_CHECKPOINT_DIR gets populated, so print what actually arrived.
    print("\nEDULLM_* environment:")
    edullm_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("EDULLM_")}
    for key, value in edullm_env.items():
        print(f"  {key} = {value}")
    if not edullm_env:
        print("  (none)")

    state: dict[str, Any] = {}

    @probe("0", "What is in the environment?")
    def _env() -> None:
        import torch

        record("torch", f"torch {torch.__version__}, cuda build {torch.version.cuda}")
        record(
            "device",
            f"cuda available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
            + (f" name={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""),
        )
        if torch.cuda.is_available():
            record("bf16", f"bfloat16 supported in hardware: {torch.cuda.is_bf16_supported()}")

        import olmo_core

        record("ai2-olmo-core", str(getattr(olmo_core, "__version__", "unknown")))
        state["torch"] = torch

    @probe("1", "Does the checkpoint resolve, and what does its config say?")
    def _resolve() -> None:
        from olmo_eval.inference.providers import olmo_core_utils as u

        imports = u._import_olmo_core()
        state["imports"] = imports
        state["u"] = u

        # Run 1 found that full validation refuses this checkpoint at
        # olmo_core_utils.py:244, because its tokenizer has pad_token_id == eos_token_id.
        # Record that deliberately rather than route around it silently: it is a real
        # constraint on every checkpoint from this training setup, and phase 4's
        # generative path has to answer for it even though scoring does not.
        try:
            u._resolve_checkpoint(
                args.checkpoint,
                imports=imports,
                validate_checkpoint=True,
                allow_tokenizer_fallback=False,
            )
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            record("validate_checkpoint=True", f"REFUSED - {type(exc).__name__}: {exc}")
        else:
            record("validate_checkpoint=True", "accepted, so run 1's pad/eos gate is gone")

        # Scoring never pads and never generates, so pad_token_id is unused on this path.
        # The False branch still loads config.json and the tokenizer config; it only skips
        # the layout and token-id assertions.
        config, tokenizer_config = u._resolve_checkpoint(
            args.checkpoint,
            imports=imports,
            validate_checkpoint=False,
            allow_tokenizer_fallback=False,
        )
        state["config"] = config
        state["tokenizer_config"] = tokenizer_config

        record("validate_checkpoint=False", "config.json parsed, tokenizer config resolved")
        record(
            "pad / eos from config",
            f"pad={getattr(tokenizer_config, 'pad_token_id', None)} "
            f"eos={getattr(tokenizer_config, 'eos_token_id', None)}",
        )
        record("tokenizer identifier", str(getattr(tokenizer_config, "identifier", None)))
        if isinstance(config, dict):
            model_cfg = config.get("model") or {}
            interesting = {
                k: model_cfg.get(k)
                for k in ("d_model", "n_layers", "vocab_size", "dtype", "name")
                if isinstance(model_cfg, dict) and k in model_cfg
            }
            record("model config", json.dumps(interesting, default=str))
            record("config top-level keys", ", ".join(sorted(config.keys())))

    @probe("2", "Does the tokenizer prepend a BOS under the defaults our scorer uses?")
    def _bos() -> None:
        u, imports = state["u"], state["imports"]
        path, tokenizer_config = u._resolve_tokenizer_path(
            args.checkpoint,
            explicit_tokenizer=None,
            tokenizer_config=state["tokenizer_config"],
            TokenizerConfig=imports.TokenizerConfig,
            allow_tokenizer_fallback=False,
        )
        record("tokenizer path", path)
        tok = imports.AutoTokenizer.from_pretrained(path)
        state["tokenizer"] = tok

        text = "The capital of France is"
        default_ids = tok(text)["input_ids"]
        explicit_off = tok(text, add_special_tokens=False)["input_ids"]

        head = "..." if len(default_ids) > 6 else ""
        off_head = "..." if len(explicit_off) > 6 else ""
        record("tokenizer defaults", f"{text!r} -> {default_ids[:6]}{head}")
        record("add_special_tokens=False", f"-> {explicit_off[:6]}{off_head}")
        record("bos_token_id", str(getattr(tok, "bos_token_id", None)))

        differs = default_ids != explicit_off
        record(
            "VERDICT",
            "DEFAULTS ADD SPECIAL TOKENS - our scorer and OlmoCoreProvider would disagree, "
            f"{len(default_ids) - len(explicit_off)} extra token(s). This must be a decision."
            if differs
            else "defaults add nothing, so HF parity and provider parity are the same answer here",
        )

    @probe("3", "What does from_checkpoint require, and can one process read 8-rank shards?")
    def _load() -> None:
        import inspect

        torch = state["torch"]
        imports = state["imports"]

        sig = inspect.signature(imports.TransformerGenerationModule.from_checkpoint)
        required = [
            name
            for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            and name != "cls"
        ]
        record("signature", str(sig))
        record("required parameters", ", ".join(required) or "(none beyond cls)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # No generation_config, which run 1's signature dump showed is Optional. The MCQ
        # scorer only ever runs a forward pass, so it needs no sampling parameters -- and
        # skipping it sidesteps pad/eos entirely, since that is what a GenerationConfig
        # validates. Run 1 could not establish this because it built one unconditionally
        # from a tokenizer that probe 2 had failed to load, manufacturing pad=0 eos=0 and
        # hitting the same error from a second source.
        #
        # The checkpoint was written by 8 ranks as 128 .distcp shards. Whether DCP
        # reshards those into one process, with process_group left at its default None,
        # is the question that decides whether phase 2 is a loader or a distributed setup.
        started = time.monotonic()
        model = imports.TransformerGenerationModule.from_checkpoint(
            args.checkpoint,
            device=device,
        )
        record(
            "8-rank -> 1-process reshard",
            f"LOADED in {time.monotonic() - started:.1f}s from checkpoint_dir alone, "
            "no generation_config and no distributed initialization",
        )
        state["model"] = model
        state["device"] = device

        params = getattr(model, "model", None)
        if params is not None and hasattr(params, "parameters"):
            first = next(iter(params.parameters()), None)
            if first is not None:
                record("weight dtype as loaded", str(first.dtype))

    @probe("4", "What shape does model_forward return for a batch of one?")
    def _forward() -> None:
        torch, model = state["torch"], state["model"]
        tok = state["tokenizer"]
        ids = torch.tensor([tok("The capital of France is")["input_ids"]], device=state["device"])
        record("input_ids shape", str(tuple(ids.shape)))

        with torch.no_grad():
            logits = model.model_forward(input_ids=ids)

        shape = tuple(logits.shape)
        record("logits shape", f"{shape}, dtype {logits.dtype}")
        record(
            "VERDICT",
            f"3-D {shape} as the scorer assumes, batch axis present"
            if len(shape) == 3 and shape[0] == 1
            else "NOT the assumed (1, seq, vocab) - the arithmetic would index the wrong axis",
        )
        state["logits_ok"] = len(shape) == 3

    @probe("5", "Is free_inference_cache cheap enough to call once per item?")
    def _cache() -> None:
        torch, model = state["torch"], state["model"]
        tok = state["tokenizer"]
        ids = torch.tensor([tok("The capital of France is")["input_ids"]], device=state["device"])

        if not hasattr(model, "free_inference_cache"):
            record("free_inference_cache", "not present on this module")
            return

        timings = []
        for _ in range(3):
            with torch.no_grad():
                model.model_forward(input_ids=ids)
            started = time.monotonic()
            model.free_inference_cache()
            timings.append(time.monotonic() - started)
        mean_ms = 1000 * sum(timings) / len(timings)
        record("free_inference_cache", f"{mean_ms:.1f} ms mean over 3 calls")
        record(
            "VERDICT",
            f"a 40-item session would spend about {40 * mean_ms / 1000:.1f}s here",
        )

    @probe("6", "Can a GenerationConfig exist at all when pad == eos?")
    def _gen_config() -> None:
        imports, tok = state["imports"], state["tokenizer"]
        pad = getattr(tok, "pad_token_id", None)
        eos = getattr(tok, "eos_token_id", None)
        record("tokenizer pad / eos", f"pad={pad} eos={eos}")

        # This is phase 4's blocker rather than phase 2's. Scoring skips GenerationConfig
        # entirely; generation cannot. If the constructor refuses these two values, the
        # generative completer needs a distinct pad token from somewhere before it can
        # run at all, and it is better to learn that here than in phase 4.
        try:
            cfg = imports.GenerationConfig(
                pad_token_id=pad,
                eos_token_id=eos,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - the refusal is the finding
            record(
                "VERDICT",
                f"REFUSED by {type(exc).__module__}.{type(exc).__name__}: {exc} "
                "-- phase 4 needs a distinct pad token, phase 2 is unaffected",
            )
            return
        state["generation_config"] = cfg
        record("VERDICT", "constructed, so generation is not blocked by pad == eos")

    @probe("7", "What does generate_batch return, and how does it pad?")
    def _generate() -> None:
        model, tok = state["model"], state["tokenizer"]
        if not hasattr(model, "generate_batch"):
            record("generate_batch", "not present on this module")
            return
        if "generation_config" not in state:
            record("generate_batch", "skipped: no GenerationConfig could be built (probe 6)")
            return

        prompts = ["The capital of France is", "Two plus two equals"]
        batch = [tok(p)["input_ids"] for p in prompts]
        out = model.generate_batch(batch)

        record("return type", type(out).__name__)
        if isinstance(out, tuple):
            record("tuple arity", str(len(out)))
            for i, part in enumerate(out):
                record(f"  element {i}", f"{type(part).__name__} {getattr(part, 'shape', '')}")
        ids = out[0] if isinstance(out, tuple) else out
        try:
            first = ids[0]
            record("row 0 length", str(len(first)))
            record("prompt length", str(len(batch[0])))
            record(
                "prompt present in row?",
                "yes - the row contains the prompt, so completion needs slicing"
                if len(first) > args.max_new_tokens
                else "no - the row looks like completion only",
            )
            record("row 0 decoded", repr(tok.decode(list(first))[:200]))
        except Exception as exc:  # noqa: BLE001
            record("row inspection", f"could not index: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 78}\nFINDINGS\n{'=' * 78}")
    for question, answer in FINDINGS:
        print(f"{question:.<34} {answer}")

    failures = [q for q, a in FINDINGS if a.startswith("FAILED")]
    print(f"\n{len(FINDINGS)} findings, {len(failures)} probe(s) failed")
    # Exit 0 even on probe failure: a failed probe is a finding, and the run's job is to
    # report all of them rather than to pass.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
