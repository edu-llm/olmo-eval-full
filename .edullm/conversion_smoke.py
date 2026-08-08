"""Phase 5: prove the conversion path against real weights, and score nothing.

The pipeline's plumbing is covered by 1078 tests and its conversion body by none of
them, because ``convert_olmo_core_to_hf`` needs torch, ``ai2-olmo-core`` and a real
sharded checkpoint, and the development machine has none of the three. This is the job
that closes that gap, and it is deliberately the smallest job that can: fetch one
checkpoint, convert it, load the result, push one batch through it, and stop.

WHY IT IS SEPARATE FROM THE CAT RUN. The two failures look identical from outside -- a
job that exits non-zero and writes nothing -- but they are diagnosed differently. A CAT
run that fails could have failed in conversion, in the bank join, in item selection or
in scoring, and narrowing that down costs another submission whichever it was. This
narrows it in advance: everything downstream of the loaded model is absent, so a failure
here is the converter and a success here removes the converter from the next failure's
suspect list. The cost of buying that is one submission and a few cents.

WHAT IT DELIBERATELY DOES NOT DO. No bank, no items, no CAT loop, no theta. The forward
pass exists to prove the weights compute rather than merely deserialize, and its output
is checked for shape and finiteness and then discarded. A number that looked like a
score would invite reading it as one, and a single ungraded prompt is not a measurement
of anything.

IT CALLS THE REAL SEAM. ``resolve_checkpoint`` and ``prepare_checkpoint`` are the same
functions ``runner.py`` calls, in the same order, and the model is loaded exactly the way
``_HFScoringModel`` loads it -- ``torch_dtype="auto"``, ``device_map="auto"``, seeded.
A smoke that reimplemented any of that would prove that the reimplementation works.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# This runs as `python .edullm/conversion_smoke.py`, so sys.path[0] is .edullm/ and the
# repository root -- where `diagnostics` lives -- is not on the path at all. The runner
# never needs this because `python -m` from the root puts the root on sys.path itself.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diagnostics.mcq_cat.common import convert, s3_io  # noqa: E402

log = logging.getLogger("conversion_smoke")

#: Derived from the checkpoint's own config.json, read during Phase 1: 49152 x 576
#: embeddings tied to the head, 30 layers of 9-head GQA over 3 KV heads at head_dim 64,
#: and a 1536-wide SwiGLU feed-forward. Asserted as a band rather than an equality
#: because the count depends on how the HF mapping lays out the tied head, and the
#: failure worth catching is "loaded a different model", which is orders away.
EXPECTED_PARAMETERS = 134_515_008
PARAMETER_TOLERANCE = 0.05

#: What the checkpoint's config.json names, and therefore what the converted directory
#: must carry through if the tokenizer resolution step did its job.
EXPECTED_VOCAB_SIZE = 49152

#: The patch in diagnostics/mcq_cat/common/hf_config_patch.py exists to emit this for a
#: plain pre-norm TransformerBlock, which upstream get_hf_config refuses outright. If the
#: converted config says anything else, the patch did not fire and the run that follows
#: would be scoring an architecture nobody chose.
EXPECTED_MODEL_TYPE = "llama"

#: Stage -> exit code, so a failure is legible from the platform's exit status alone
#: without reading a log. Distinct codes because these are the distinctions the whole
#: job exists to draw.
EXIT_RESOLVE = 11
EXIT_PREPARE = 12
EXIT_LOAD = 13
EXIT_FORWARD = 14
EXIT_ASSERT = 15


def peak_rss_bytes() -> int | None:
    """Peak resident set size, or None where the platform cannot report it.

    ``resource`` is Unix-only, so this returns None on the Windows checkout where the
    script is syntax-checked and a real number on the Linux node where it runs. Linux
    reports ``ru_maxrss`` in kibibytes; macOS reports bytes, and this normalizes on the
    Linux reading because that is the only platform whose number will ever be recorded.
    """
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def directory_bytes(path: Path) -> int:
    """Total size of every file under ``path``."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@contextmanager
def stage(name: str, timings: dict[str, float], exit_code: int):
    """Time one stage, and turn its failure into a named exit rather than a traceback.

    The traceback still prints -- it is the only thing that says *what* broke -- but the
    exit code says *where*, which is what a person reading the platform's run list sees
    before they open anything.
    """
    log.info("--- %s: starting", name)
    started = time.monotonic()
    try:
        yield
    except Exception:
        elapsed = time.monotonic() - started
        timings[name] = elapsed
        log.exception("--- %s: FAILED after %.1fs", name, elapsed)
        _emit_and_exit(timings, failed_stage=name, exit_code=exit_code)
    elapsed = time.monotonic() - started
    timings[name] = elapsed
    log.info("--- %s: ok in %.1fs", name, elapsed)


_RESULT: dict[str, Any] = {}
_RESULTS_URI: str | None = None


def _emit_and_exit(timings: dict[str, float], *, failed_stage: str | None, exit_code: int) -> None:
    """Write the result artifact and leave, on the success and failure paths alike.

    A failed smoke is worth more than a silent one: the timings up to the break and the
    environment that produced it are what a second attempt is planned against.
    """
    _RESULT["timings_seconds"] = {k: round(v, 3) for k, v in timings.items()}
    _RESULT["peak_rss_bytes"] = peak_rss_bytes()
    _RESULT["failed_stage"] = failed_stage
    _RESULT["ok"] = failed_stage is None
    payload = json.dumps(_RESULT, indent=2, default=str)
    print(payload)
    if _RESULTS_URI:
        try:
            s3_io.upload_json(_RESULTS_URI, "conversion_smoke.json", _RESULT)
            log.info("Wrote conversion_smoke.json to %s", _RESULTS_URI)
        except Exception:
            # The artifact is the nice-to-have; the exit code and the printed payload are
            # the contract. An upload failure must not turn a passing smoke into a
            # failing one, or the next person debugs S3 instead of the converter.
            log.exception("Could not upload the result artifact; the payload above stands")
    sys.exit(exit_code)


def check(condition: bool, message: str) -> None:
    """Record a failed expectation and stop, naming what was expected."""
    if not condition:
        log.error("EXPECTATION FAILED: %s", message)
        _RESULT.setdefault("failed_expectations", []).append(message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="s3:// URI or local path")
    parser.add_argument(
        "--dtype",
        default=convert.DTYPE_DEFAULT,
        choices=list(convert.CONVERSION_DTYPES),
        # Named on the command line for the same reason runner.py names it: the
        # platform's bfloat16_not_in_the_hardware guard reads the text of the command and
        # cannot see a precision chosen in code.
        help="conversion precision, written into the HF weights",
    )
    parser.add_argument("--results-s3", default=os.environ.get("EDULLM_OUTPUT_PREFIX"))
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="where to stage; defaults to a temp dir under the current directory",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="load the model but do not run a batch through it",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    global _RESULTS_URI
    _RESULTS_URI = args.results_s3

    timings: dict[str, float] = {}
    _RESULT["checkpoint"] = args.checkpoint
    _RESULT["requested_dtype"] = args.dtype
    _RESULT["python"] = platform.python_version()
    _RESULT["platform"] = platform.platform()

    # --- environment ---------------------------------------------------------------
    # Recorded before anything can fail, because "which torch, which olmo_core" is the
    # first question asked of any failure below and the answer is gone once the process
    # dies. This is not in a stage(): a missing torch should fail here, plainly.
    import torch
    import transformers

    _RESULT["versions"] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None),
    }
    try:
        import importlib.metadata as md

        _RESULT["versions"]["ai2-olmo-core"] = md.version("ai2-olmo-core")
    except Exception:
        _RESULT["versions"]["ai2-olmo-core"] = "unknown"
    log.info("environment: %s", json.dumps(_RESULT["versions"], default=str))

    work_root = Path(args.work_dir) if args.work_dir else Path("smoke_work")
    staged = work_root / "checkpoint"
    converted = work_root / "checkpoint-hf"
    work_root.mkdir(parents=True, exist_ok=True)

    # --- resolve -------------------------------------------------------------------
    with stage("resolve", timings, EXIT_RESOLVE):
        local = s3_io.resolve_checkpoint(args.checkpoint, staged, region=args.aws_region)
        _RESULT["staged_bytes"] = directory_bytes(local)
        _RESULT["staged_layout"] = convert.describe_layout(local)
        log.info("staged %.2f GiB at %s", _RESULT["staged_bytes"] / 2**30, local)

    # --- prepare -------------------------------------------------------------------
    # The seam runner.py calls, with the same policy a real run uses. Detection deciding
    # "already HF" on a native checkpoint would return the input untouched and the load
    # below would fail with a transformers error about a missing config, which reads as a
    # bad checkpoint rather than a bad detector -- so the identity is checked, not assumed.
    with stage("prepare", timings, EXIT_PREPARE):
        hf_dir = convert.prepare_checkpoint(
            local, converted, policy=convert.PREP_AUTO, dtype=args.dtype
        )
        _RESULT["converted_dir"] = str(hf_dir)
        _RESULT["converted_bytes"] = directory_bytes(hf_dir)
        _RESULT["converted_entries"] = sorted(p.name for p in hf_dir.iterdir())
        log.info("converted %.2f GiB at %s", _RESULT["converted_bytes"] / 2**30, hf_dir)

    check(
        hf_dir.resolve() != local.resolve(),
        f"conversion returned the staged directory unchanged ({hf_dir}), so the layout "
        f"detector read a native OLMo-core checkpoint as already-HF",
    )

    hf_config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    _RESULT["hf_config"] = {
        k: hf_config.get(k)
        for k in (
            "model_type",
            "architectures",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "tie_word_embeddings",
            "max_position_embeddings",
            "torch_dtype",
        )
    }
    check(
        hf_config.get("model_type") == EXPECTED_MODEL_TYPE,
        f"converted config.json says model_type={hf_config.get('model_type')!r}, expected "
        f"{EXPECTED_MODEL_TYPE!r} -- the get_hf_config patch did not fire",
    )
    check(
        hf_config.get("tie_word_embeddings") is True,
        "converted config.json lost tie_word_embeddings=true, so the head is now an "
        "independent copy of the embedding matrix rather than the same tensor",
    )
    check(
        hf_config.get("vocab_size") == EXPECTED_VOCAB_SIZE,
        f"converted vocab_size is {hf_config.get('vocab_size')}, expected {EXPECTED_VOCAB_SIZE}",
    )

    # --- load ----------------------------------------------------------------------
    # Exactly how _HFScoringModel loads, down to torch_dtype and device_map, so a load
    # that works here works there. device_map="auto" needs accelerate; if that is missing
    # from the image this is where it says so, which is worth knowing before a CAT run.
    with stage("load", timings, EXIT_LOAD):
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(1234)
        tokenizer = AutoTokenizer.from_pretrained(str(hf_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(hf_dir), torch_dtype="auto", device_map="auto"
        )
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
        _RESULT["loaded"] = {
            "parameters": n_params,
            "parameter_dtypes": param_dtypes,
            "device": str(next(model.parameters()).device),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_size": tokenizer.vocab_size,
        }
        log.info("loaded %s parameters as %s", f"{n_params:,}", param_dtypes)

    low = EXPECTED_PARAMETERS * (1 - PARAMETER_TOLERANCE)
    high = EXPECTED_PARAMETERS * (1 + PARAMETER_TOLERANCE)
    check(
        low <= _RESULT["loaded"]["parameters"] <= high,
        f"loaded {_RESULT['loaded']['parameters']:,} parameters, expected about "
        f"{EXPECTED_PARAMETERS:,} for this 135M configuration",
    )
    check(
        any(args.dtype in d for d in _RESULT["loaded"]["parameter_dtypes"]),
        f"asked for {args.dtype} and loaded {_RESULT['loaded']['parameter_dtypes']}",
    )

    # --- forward -------------------------------------------------------------------
    # One batch, to separate "the weights deserialize" from "the weights compute". On a
    # card this is also the only part of the smoke that exercises the precision on the
    # hardware the CAT will use, which is the one thing a CPU-only venue could not have
    # told us.
    if not args.skip_forward:
        with stage("forward", timings, EXIT_FORWARD):
            inputs = tokenizer("The capital of France is", return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            _RESULT["forward"] = {
                "input_tokens": int(inputs["input_ids"].shape[-1]),
                "logits_shape": list(logits.shape),
                "logits_dtype": str(logits.dtype),
                "all_finite": bool(torch.isfinite(logits).all().item()),
                "logit_absmax": float(logits.abs().max().item()),
            }
            log.info("forward: %s", json.dumps(_RESULT["forward"]))

        forward = _RESULT["forward"]
        check(forward["all_finite"], "forward pass produced non-finite logits")
        check(
            list(forward["logits_shape"])
            == [1, forward["input_tokens"], _RESULT["hf_config"]["vocab_size"]],
            f"logits shape {forward['logits_shape']} is not "
            f"(1, {forward['input_tokens']}, {_RESULT['hf_config']['vocab_size']})",
        )

    failures = _RESULT.get("failed_expectations", [])
    if failures:
        log.error("%d expectation(s) failed; the conversion is not trustworthy", len(failures))
        _emit_and_exit(timings, failed_stage="expectations", exit_code=EXIT_ASSERT)

    _emit_and_exit(timings, failed_stage=None, exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
