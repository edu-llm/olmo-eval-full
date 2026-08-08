"""Phase 4 spike: ask generate_batch what it does on this checkpoint, rather than assume it.

Throwaway, on the same terms as ``.edullm/olmo_core_spike.py``. This is not a test and
nothing imports it. Its only product is the text it prints, which becomes the written
record the generative completer is designed against. Delete it once that record exists.

Every probe is independent and swallows its own failure, and inside the generation probe
every *prompt* is guarded separately as well. A checkpoint that generates for ifeval and
dies on gpqa must still report the ifeval finding, because a second GPU hour to learn the
second fact is the thing this is meant to avoid.

WHAT IS ALREADY KNOWN FROM SOURCE, AND WHY THIS RUNS ANYWAY. Reading OLMo-core 2.5.0 at
08df5aa0 answers the shape question on paper: ``generate_batch`` seeds ``generated`` with
``input_ids`` and concatenates onto it, so the prompt is in the return unless
``completions_only=True``; and the decode loop's stop test is ``next_tokens.eq(eos)``,
which never reads ``pad_token_id``. Neither claim has been observed on a real checkpoint,
and the one that matters most cannot be read off the source at all: whether THIS model,
trained with pad == eos == 0, ever emits token 0 within a short budget. A completer that
strips on eos when eos is never produced returns 1024 tokens of run-on text per item; one
that assumes eos arrives promptly truncates nothing and pays for every token. So the
question this run exists to answer is the empirical half.

THE LOADER IS NOT REINVENTED. ``_OlmoCoreScoringModel`` already resolves this checkpoint
with ``validate_checkpoint=False``, resolves the tokenizer, and hands ``from_checkpoint`` a
``GenerationConfig`` with a pad id fabricated distinct from eos. That combination is proven
-- run_019fe277 scored 8 arc_challenge items with it -- so it is imported rather than
copied, and the staging step in front of it is the runner's own ``s3_io.resolve_checkpoint``
for the same reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# This file lives in .edullm/, which is what sys.path[0] becomes when it is run as a
# script. The repository root is what holds `diagnostics` and `calibrated_datasets`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = (
    "s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/"
    "run_019fce1a-f393-70e3-ba0e-e2771c70f9c0/checkpoints/step305176/"
)

#: The three generative banks, with the prompt settings their style's config.yaml
#: records. Named here rather than read out of the yaml so the probe stays standalone,
#: and cross-checked against diagnostics/mcq_cat/styles/uni_mcq/config.yaml at the
#: commit this ran from. `configured_max_new_tokens` is what a real session would ask
#: for; this probe deliberately asks for far less, and printing both is the point.
BANKS: tuple[dict[str, Any], ...] = (
    {
        "name": "ifeval",
        "prompt_style": "ifeval",
        "num_fewshot": 0,
        "fewshot_source": None,
        "configured_max_new_tokens": 1280,
        "note": "completion format, chat_format: false",
    },
    {
        "name": "leaderboard_math",
        "prompt_style": "leaderboard_math",
        "num_fewshot": 4,
        "fewshot_source": "leaderboard_math",
        "configured_max_new_tokens": 1024,
        "note": "4-shot Minerva block, so the real prompt is long",
    },
    {
        "name": "gpqa",
        "prompt_style": "gpqa",
        "num_fewshot": 0,
        "fewshot_source": None,
        "configured_max_new_tokens": 1024,
        "note": "configured chat_format: true; this probe sends the completion form",
    },
)

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


def first_item(bank: str) -> dict[str, Any]:
    """The first item of a vendored bank, read straight off disk.

    A real item rather than an invented prompt, so what this reports is the workload the
    completer will actually see: an IFEval instruction runs to a paragraph, a MATH
    problem carries LaTeX, and a GPQA stem already has its lettered choice block frozen
    into it.
    """
    path = REPO_ROOT / "calibrated_datasets" / bank / "items.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"{path} holds no items")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR") or DEFAULT_CHECKPOINT,
        help="OLMo-core step directory, local or s3://.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help=(
            "Named on the command line rather than set in code so the platform's "
            "bfloat16_not_in_the_hardware guard can read it out of argv."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help=(
            "Small on purpose. 64 is enough to see whether eos is ever emitted, and the "
            "banks' real budgets of 1024-1280 would make this a paid eval rather than a "
            "probe."
        ),
    )
    args = parser.parse_args()

    print("=" * 78)
    print("OLMo-core generation probe")
    print("=" * 78)
    print(f"checkpoint:     {args.checkpoint}")
    print(f"dtype:          {args.dtype}")
    print(f"max_new_tokens: {args.max_new_tokens}")

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

    @probe("1", "Do the three generative banks yield real prompts?")
    def _prompts() -> None:
        from diagnostics.mcq_cat.common import generative

        prompts: list[dict[str, Any]] = []
        for bank in BANKS:
            name = bank["name"]
            try:
                item = first_item(name)
                template = generative.get_prompt_template(bank["prompt_style"])
                examples: tuple[Any, ...] = ()
                if bank["num_fewshot"]:
                    loader = generative.FEWSHOT_SOURCES[bank["fewshot_source"]]
                    examples = loader()[: bank["num_fewshot"]]
                text = template.render(item["question"], examples)
            except Exception as exc:  # noqa: BLE001 - one bank must not cost the others
                record(name, f"FAILED to build prompt - {type(exc).__name__}: {exc}")
                continue
            prompts.append({"bank": name, "item_id": item.get("id"), "prompt": text})
            record(
                name,
                f"item {item.get('id')!r}, {len(examples)}-shot, {len(text)} chars, "
                f"bank asks for max_new_tokens={bank['configured_max_new_tokens']} "
                f"({bank['note']})",
            )
        state["prompts"] = prompts
        record("prompts built", f"{len(prompts)} of {len(BANKS)}")

    @probe("2", "Does the proven loader read this checkpoint, and what pad/eos does it use?")
    def _load() -> None:
        import tempfile

        from diagnostics.mcq_cat.common import s3_io
        from diagnostics.mcq_cat.common.inference import (
            InferenceConfig,
            _OlmoCoreScoringModel,
        )

        # The runner stages to a TemporaryDirectory and loads inside it. Held open for
        # the life of the process here for the same reason: the module reads the
        # directory lazily and deleting it under a loaded model is not a thing to test
        # on a paid card.
        staging = tempfile.TemporaryDirectory(prefix="generation-probe-")
        state["staging"] = staging

        started = time.monotonic()
        checkpoint_dir = s3_io.resolve_checkpoint(
            args.checkpoint,
            Path(staging.name) / "checkpoint",
        )
        record(
            "staged",
            f"{checkpoint_dir} in {time.monotonic() - started:.1f}s "
            f"(--checkpoint-prep none has no step after this; nothing is converted)",
        )

        started = time.monotonic()
        scorer = _OlmoCoreScoringModel(
            checkpoint_dir,
            InferenceConfig(checkpoint_kind="olmo_core"),
        )
        record("load", f"LOADED in {time.monotonic() - started:.1f}s")

        state["scorer"] = scorer
        state["model"] = scorer.model
        state["tokenizer"] = scorer.tokenizer
        state["device"] = scorer.device

        record("tokenizer defaults", scorer.tokenizer_defaults.summary())

        generation_config = getattr(scorer.model, "_generation_config", None)
        if generation_config is None:
            record("generation config", "not reachable on the module")
            return
        eos = getattr(generation_config, "eos_token_id", None)
        pad = getattr(generation_config, "pad_token_id", None)
        state["eos"] = eos
        state["pad"] = pad
        record(
            "generation config",
            f"eos_token_id={eos} pad_token_id={pad} (pad is fabricated distinct from "
            f"eos; the checkpoint writes both as 0), use_cache="
            f"{getattr(generation_config, 'use_cache', None)}, "
            f"max_new_tokens={getattr(generation_config, 'max_new_tokens', None)} "
            f"which this probe overrides per call",
        )

        first = next(iter(scorer.model.model.parameters()), None)
        if first is not None:
            record("weight dtype as loaded", str(first.dtype))

    @probe("3", "What does generate_batch return, and is eos ever emitted?")
    def _generate() -> None:
        torch = state["torch"]
        model = state["model"]
        tok = state["tokenizer"]
        device = state["device"]
        eos = state.get("eos")

        prompts = state.get("prompts") or []
        if not prompts:
            record("generate_batch", "skipped: probe 1 built no prompts")
            return

        for entry in prompts:
            bank = entry["bank"]
            print(f"\n  --- {bank} ---", flush=True)
            try:
                prompt_ids = tok(entry["prompt"])["input_ids"]
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                prompt_len = input_ids.shape[1]

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.monotonic()
                out = model.generate_batch(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.monotonic() - started

                record(f"{bank}: return", f"{type(out).__name__} of {len(out)}")
                for index, part in enumerate(out):
                    record(
                        f"{bank}:   element {index}",
                        f"{type(part).__name__} shape={getattr(part, 'shape', None)} "
                        f"dtype={getattr(part, 'dtype', None)}",
                    )

                generated = out[0] if isinstance(out, tuple) else out
                row = generated[0]
                total = int(generated.shape[1])
                new_tokens = total - prompt_len

                prefix_matches = bool(
                    total >= prompt_len and torch.equal(row[:prompt_len], input_ids[0])
                )
                record(
                    f"{bank}: prompt in return?",
                    f"prompt_len={prompt_len} returned_len={total} "
                    + (
                        "YES - the first prompt_len tokens are the prompt verbatim, so a "
                        "completer must slice [prompt_len:] or pass completions_only=True"
                        if prefix_matches
                        else "NO - the return does not start with the prompt"
                    ),
                )
                record(
                    f"{bank}: new tokens",
                    f"{new_tokens} of max_new_tokens={args.max_new_tokens}"
                    + (
                        " - stopped early"
                        if new_tokens < args.max_new_tokens
                        else " - ran to the budget without stopping"
                    ),
                )

                completion = row[prompt_len:]
                if eos is None:
                    record(f"{bank}: eos", "eos id unknown (probe 2 did not report one)")
                else:
                    hits = (completion == eos).nonzero().flatten().tolist()
                    in_prompt = int((row[:prompt_len] == eos).sum().item())
                    record(
                        f"{bank}: eos ({eos}) emitted?",
                        (
                            f"YES at completion position(s) {hits} of {new_tokens} "
                            f"(absolute {[prompt_len + h for h in hits]})"
                            if hits
                            else f"NO - token {eos} never appeared in {new_tokens} new tokens"
                        )
                        + (f"; {in_prompt} occurrence(s) inside the prompt" if in_prompt else ""),
                    )

                completion_ids = completion.tolist()
                record(f"{bank}: seconds", f"{elapsed:.2f}s wall clock for {new_tokens} tokens")
                record(
                    f"{bank}: decoded",
                    repr(tok.decode(completion_ids, skip_special_tokens=True)[:200]),
                )
                record(
                    f"{bank}: decoded (specials kept)",
                    repr(tok.decode(completion_ids, skip_special_tokens=False)[:200]),
                )
            except Exception as exc:  # noqa: BLE001 - one prompt must not cost the others
                record(f"{bank}: generate", f"FAILED - {type(exc).__name__}: {exc}")
                traceback.print_exc()

    @probe("4", "Does free_inference_cache exist, and what does calling it cost?")
    def _cache() -> None:
        torch = state["torch"]
        model = state.get("model")
        if model is None:
            record("free_inference_cache", "skipped: no model loaded")
            return
        if not hasattr(model, "free_inference_cache"):
            record("free_inference_cache", "NOT PRESENT on this module")
            return
        record("free_inference_cache", "present")

        cuda = torch.cuda.is_available()
        before = torch.cuda.memory_allocated() if cuda else 0
        started = time.monotonic()
        model.free_inference_cache()
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        after = torch.cuda.memory_allocated() if cuda else 0

        record("cost", f"{1000 * elapsed:.1f} ms")
        if cuda:
            record(
                "freed",
                f"{(before - after) / 1e6:.1f} MB "
                f"({before / 1e6:.1f} MB -> {after / 1e6:.1f} MB allocated)",
            )
        record(
            "VERDICT",
            f"a 40-item sequential session would spend about {40 * elapsed:.2f}s here",
        )

    print(f"\n{'=' * 78}\nFINDINGS\n{'=' * 78}")
    for question, answer in FINDINGS:
        print(f"{question:.<44} {answer}")

    failures = [q for q, a in FINDINGS if a.startswith("FAILED")]
    print(f"\n{len(FINDINGS)} findings, {len(failures)} failed")
    # Exit 0 even on probe failure: a failed probe is a finding, and the run's job is to
    # report all of them rather than to pass.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
