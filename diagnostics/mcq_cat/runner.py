"""CLI entry point that drives an MCQ CAT session for one checkpoint.

Usage::

    python -m diagnostics.mcq_cat.runner --cat-style NAME \\
        --checkpoint <path|s3://...> --s3-out s3://bucket/prefix

The runner resolves the requested style through the auto-discovering registry and
runs it through the generic CAT engine. It resolves any registered style without
being modified when new styles are added.

It is also the only place that decides *how* items are graded. ``CatStyle.score``
receives an already-constructed model, so the grading scheme is fixed when the model
is built, which happens here. The runner stays style-agnostic about it: it asks the
style for a :class:`~diagnostics.mcq_cat.common.grading.GradingRequest` and hands that
to :mod:`~diagnostics.mcq_cat.common.grading`, which owns the modality table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from . import registry
from .common import cat_loop, convert, generative, grading, inference, s3_io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [mcq_cat.runner] %(levelname)s: %(message)s",
)
log = logging.getLogger("mcq_cat.runner")

#: What ``--ability-estimator`` accepts, and which value is the default.
#:
#: Spelled here rather than imported because this module resolves *any* registered style
#: and must not depend on one. ``uni_mcq`` owns the implementations and its own copy of
#: these names in ``styles/uni_mcq/irt.py``; ``tests/test_ability_estimator.py`` pins the
#: two lists equal, so the vocabulary cannot drift without a failure.
#:
#: Only the reported number moves. Item selection conditions on the sequential estimate
#: in every mode, so two runs of one checkpoint under the two values administer the same
#: items in the same order and differ only in which refit of that response set is
#: published.
ABILITY_ESTIMATORS = ("batch_eap", "batch_eap+mwle")
DEFAULT_ABILITY_ESTIMATOR = ABILITY_ESTIMATORS[0]


def build_parser() -> argparse.ArgumentParser:
    """Build the runner's argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.mcq_cat.runner",
        description="Run an MCQ CAT diagnostic on a checkpoint.",
    )
    parser.add_argument("--cat-style", help="Registered CAT style name (see --list-styles).")
    parser.add_argument("--checkpoint", help="Checkpoint local path or s3:// URI.")
    parser.add_argument("--s3-out", help="Destination s3:// URI (or local dir) for results.")
    parser.add_argument("--benchmark", help="Benchmark JSONL path/URI or registered task name.")
    parser.add_argument("--irt-params", help="IRT parameter JSON path/URI for the benchmark items.")
    parser.add_argument(
        "--se-threshold",
        type=float,
        default=0.3,
        help="Standard-error stop threshold (default: 0.3).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=40,
        help="Maximum number of items to administer (default: 40).",
    )
    parser.add_argument(
        "--ability-estimator",
        default=DEFAULT_ABILITY_ESTIMATOR,
        choices=list(ABILITY_ESTIMATORS),
        help=(
            "Which re-fit over the administered responses is reported as ability.theta. "
            f"'{ABILITY_ESTIMATORS[0]}' (default) publishes the EAP posterior mean, which "
            f"is what every run so far reported. '{ABILITY_ESTIMATORS[1]}' publishes "
            "Warm's weighted likelihood estimate seeded at that EAP value, which drops "
            "the standard-normal prior and so removes its inward pull at the tails. "
            "Selection is identical either way, and both thetas are recorded in the "
            "report whichever is asked for."
        ),
    )
    parser.add_argument(
        "--checkpoint-kind",
        default="hf",
        choices=["hf", "olmo_core"],
        help="Backend that loads the prepared checkpoint (default: hf).",
    )
    parser.add_argument(
        "--checkpoint-prep",
        default=convert.PREP_AUTO,
        choices=list(convert.PREP_POLICIES),
        help=(
            "What to do with the staged checkpoint before loading it. "
            "'auto' converts a raw OLMo-core directory to HF; 'none' hands it over "
            "untouched, for a backend that reads the native format (default: auto)."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=convert.DTYPE_DEFAULT,
        choices=list(convert.CONVERSION_DTYPES),
        help=(
            "Precision the model is scored at (default: bfloat16). Under "
            "--checkpoint-prep auto it is what converted weights are written at; under "
            "--checkpoint-prep none nothing is converted and it is instead what the "
            "native olmo_core backend builds the model at. Either way it is the "
            "precision that produced the run's numbers, and the report records it. "
            "Name it on the command line even at the default: the platform's "
            "bfloat16_not_in_the_hardware guard reads the text of the command and "
            "cannot see a precision this program picks in code, so a card without the "
            "format is refused for free here instead of dying on the first kernel."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Scoring batch size (default: 16)."
    )
    parser.add_argument(
        "--aws-region", default="us-east-1", help="AWS region for S3 (default: us-east-1)."
    )
    parser.add_argument("--s3-endpoint-url", default=None, help="Optional S3 endpoint URL.")
    parser.add_argument(
        "--list-styles", action="store_true", help="List discovered CAT styles and exit."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the style and print the plan without loading the model.",
    )
    return parser


def _apply_ability_estimator(style: object, estimator: str) -> bool:
    """Tell ``style`` which estimator to report, and say so when it cannot be told.

    Duck-typed rather than added to :class:`~diagnostics.mcq_cat.base.CatStyle`, which is
    the frozen contract every style implements: an estimator choice is not something a
    style with one estimator should have to answer, and the runner already reads
    ``tokenizer_defaults`` off a model the same way, for the same reason.

    A style that does not implement the seam is fine at the default and refused
    otherwise. Accepting the flag and ignoring it would publish a shrunk EAP theta out of
    a run whose command line asked for an unshrunk MWLE one, which is exactly the class
    of silent mismatch this file's other pre-flight checks exist to prevent.

    Returns:
        ``True`` when the run may continue.
    """
    setter = getattr(style, "set_ability_estimator", None)
    if callable(setter):
        setter(estimator)
        return True
    if estimator == DEFAULT_ABILITY_ESTIMATOR:
        return True
    log.error(
        "--ability-estimator %s was requested but style %r does not implement one. "
        "Only %s is available for it.",
        estimator,
        getattr(style, "name", type(style).__name__),
        DEFAULT_ABILITY_ESTIMATOR,
    )
    return False


def _write_report(report_dict: dict, args: argparse.Namespace) -> str:
    """Write the report to the S3 or local destination and return its location."""
    payload = json.dumps(report_dict, indent=2)
    name = "cat_report.json"
    if s3_io.is_s3_uri(args.s3_out):
        s3_io.upload_files(
            args.s3_out,
            {name: payload},
            region=args.aws_region,
            endpoint_url=args.s3_endpoint_url,
        )
        return f"{args.s3_out.rstrip('/')}/{name}"
    dest_dir = Path(args.s3_out)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / name
    out_path.write_text(payload)
    return str(out_path)


def run(args: argparse.Namespace) -> int:
    """Execute the runner for parsed ``args``.

    The bank is resolved, its modality checked, its recorded scoring convention compared
    against the resolved settings and the requested checkpoint format tested for a
    loader before the checkpoint is fetched, so an unsupported dataset, a bank that would
    be graded by the wrong scheme, one that would be graded under the wrong convention
    and a format nothing can load all fail without staging gigabytes or booting a GPU.
    """
    if args.list_styles:
        styles = registry.available_styles()
        print("Available CAT styles:", ", ".join(styles) if styles else "<none>")
        return 0

    missing = [
        flag
        for flag, value in (
            ("--cat-style", args.cat_style),
            ("--checkpoint", args.checkpoint),
            ("--s3-out", args.s3_out),
        )
        if not value
    ]
    if missing:
        log.error("Missing required arguments: %s", ", ".join(missing))
        return 2

    style = registry.get_cat(args.cat_style)
    benchmark = args.benchmark or ""

    if not _apply_ability_estimator(style, args.ability_estimator):
        return 2

    if args.dry_run:
        log.info(
            "[dry-run] style=%s benchmark=%s checkpoint=%s kind=%s prep=%s dtype=%s "
            "se_threshold=%.3f max_items=%d estimator=%s -> %s",
            args.cat_style,
            benchmark or "<unset>",
            args.checkpoint,
            args.checkpoint_kind,
            args.checkpoint_prep,
            args.dtype,
            args.se_threshold,
            args.max_items,
            args.ability_estimator,
            args.s3_out,
        )
        return 0

    # `--dtype` reaches three places from here, because there are two ways a precision
    # can be chosen and two modalities that can choose it, and the flag has to mean the
    # same thing whichever is live: below it goes to `prepare_checkpoint`, which writes
    # converted weights at it, and here it goes to both configs a native backend builds
    # its model from. Exactly one of the three applies per run -- conversion happens
    # under `--checkpoint-prep auto`, a native loader runs under `none`, and a run has
    # one modality -- so setting all three is not a conflict, and setting only the first
    # is what left the flag inert on the native path. The generative half was added with
    # the native completer rather than after it, because the MCQ half spent a whole run
    # reporting bfloat16 while scoring in float32 and nothing in the artifact said so.
    settings = grading.GradingSettings(
        mcq=inference.InferenceConfig(
            checkpoint_kind=args.checkpoint_kind,
            batch_size=args.batch_size,
            dtype=args.dtype,
        ),
        generation=generative.GenerationConfig(
            checkpoint_kind=args.checkpoint_kind,
            dtype=args.dtype,
        ),
    )

    with tempfile.TemporaryDirectory(prefix="mcq-cat-") as tmp:
        bank = style.download_benchmark(benchmark, dest=Path(tmp) / "benchmark")
        irt_bank = style.load_irt_params(args.irt_params or benchmark)

        request = grading.request_for(style, dataset=benchmark or bank.name)
        grading.check_bank_modality(request, bank.items)
        grading.check_scoring_convention(style, request, settings)
        grading.check_checkpoint_kind(request, settings)

        checkpoint_dir = s3_io.resolve_checkpoint(
            args.checkpoint,
            Path(tmp) / "checkpoint",
            region=args.aws_region,
            endpoint_url=args.s3_endpoint_url,
        )
        # Between the fetch and the load, because preparation reads what training wrote
        # and the backend reads what preparation produced. Under `auto` both directories
        # exist at once inside `tmp`, so a shape with room for the checkpoint but not for
        # its converted copy needs `--checkpoint-prep none` and a native backend.
        checkpoint_dir = convert.prepare_checkpoint(
            checkpoint_dir,
            Path(tmp) / "checkpoint-hf",
            policy=args.checkpoint_prep,
            dtype=args.dtype,
        )
        model = grading.load_grader(request, checkpoint_dir, settings)

        try:
            report = cat_loop.run_cat(
                style,
                bank=bank,
                irt_bank=irt_bank,
                model=model,
                se_threshold=args.se_threshold,
                max_items=args.max_items,
            )
        finally:
            # Neither in-process backend defines this, so it is inert today. A served one
            # would own a subprocess, and discovering that after the fact means editing
            # the runner rather than registering a backend.
            closer = getattr(model, "close", None)
            if callable(closer):
                closer()

    report_dict = report.to_dict()
    report_dict["run"] = {
        "cat_style": args.cat_style,
        "checkpoint": args.checkpoint,
        "checkpoint_kind": args.checkpoint_kind,
        "checkpoint_prep": args.checkpoint_prep,
        # True of both paths as of 2026-08-08, and true of only one before that. While
        # the native backend passed no dtype to `from_checkpoint`, a `none`-prep run
        # recorded whatever was asked for and scored in the checkpoint's own float32,
        # so this field named an intention rather than a measurement. Reports from
        # before that date under `--checkpoint-prep none` -- run_019fe277's
        # arc_challenge theta among them -- say bfloat16 and are fp32.
        "dtype": args.dtype,
        "ability_estimator": args.ability_estimator,
        "modality": request.modality,
        "grader": grading.get_grader(request.modality).summary,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Optional, and read off the model rather than asked of it, because only a backend
    # that has something to say sets it. The olmo_core scorer records what its
    # tokenizer's defaults add, which decides whether the tokens scored here are the
    # ones this bank's difficulties were calibrated behind -- a fact about the theta
    # below it, and one that is otherwise a log line nobody reads twice.
    tokenization = getattr(model, "tokenizer_defaults", None)
    if tokenization is not None:
        report_dict["run"]["tokenization"] = tokenization.as_dict()
    location = _write_report(report_dict, args)
    log.info("Done. Report written to %s", location)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the CAT diagnostic."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - surface a clear failure to the caller
        log.error("mcq_cat runner failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
