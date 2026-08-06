"""CLI entry point that drives an FRQ CAT session for one checkpoint.

Usage::

    python -m diagnostics.frq_cat.runner --cat-style NAME \\
        --checkpoint <path|s3://...> \\
        --tutor-endpoint http://localhost:8000/v1 \\
        --judge-endpoint http://localhost:8001/v1 \\
        --s3-out s3://bucket/prefix

The runner resolves the requested style through the auto-discovering registry and
runs it through the generic FRQ CAT engine. It resolves any registered style
without being modified when new styles are added. The checkpoint under test is
served at ``--tutor-endpoint`` (respgen); the frozen judge is served at
``--judge-endpoint``.
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
from .common import cat_loop, s3_io
from .common import judge as judge_mod
from .common import respgen as respgen_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [frq_cat.runner] %(levelname)s: %(message)s",
)
log = logging.getLogger("frq_cat.runner")


def build_parser() -> argparse.ArgumentParser:
    """Build the runner's argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m diagnostics.frq_cat.runner",
        description="Run an FRQ CAT diagnostic on a checkpoint.",
    )
    parser.add_argument("--cat-style", help="Registered CAT style name (see --list-styles).")
    parser.add_argument("--checkpoint", help="Checkpoint local path or s3:// URI (provenance).")
    parser.add_argument("--s3-out", help="Destination s3:// URI (or local dir) for results.")
    parser.add_argument(
        "--benchmark", help="Benchmark name/URI (else the style's bundled default)."
    )
    parser.add_argument(
        "--irt-params", help="IRT parameter source (else the style's bundled bank)."
    )
    parser.add_argument(
        "--tutor-endpoint", help="Served vLLM /v1 base URL of the checkpoint under test."
    )
    parser.add_argument(
        "--served-model", default="tutor", help="served-model-name of the tutor (default: tutor)."
    )
    parser.add_argument("--judge-endpoint", help="Served vLLM /v1 base URL of the frozen judge.")
    parser.add_argument(
        "--judge-config", help="Frozen judge config (judge_frozen.yaml); else the style's."
    )
    parser.add_argument(
        "--se-threshold",
        type=float,
        default=0.3,
        help="Standard-error stop threshold (default: 0.3).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=50,
        help="Maximum number of criteria to administer (default: 50).",
    )
    parser.add_argument(
        "--checkpoint-kind",
        default="hf",
        choices=["hf", "olmo_core"],
        help="Checkpoint format (provenance).",
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
        help="Resolve the style and print the plan without serving/generating.",
    )
    return parser


def _write_report(report_dict: dict, args: argparse.Namespace) -> str:
    """Write the report to the S3 or local destination and return its location."""
    payload = json.dumps(report_dict, indent=2)
    name = "cat_report.json"
    if s3_io.is_s3_uri(args.s3_out):
        s3_io.upload_files(
            args.s3_out, {name: payload}, region=args.aws_region, endpoint_url=args.s3_endpoint_url
        )
        return f"{args.s3_out.rstrip('/')}/{name}"
    dest_dir = Path(args.s3_out)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / name
    out_path.write_text(payload)
    return str(out_path)


def run(args: argparse.Namespace) -> int:
    """Execute the runner for parsed ``args``."""
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

    if args.dry_run:
        log.info(
            "[dry-run] style=%s benchmark=%s checkpoint=%s kind=%s tutor=%s judge=%s "
            "se_threshold=%.3f max_items=%d -> %s",
            args.cat_style,
            benchmark or "<style default>",
            args.checkpoint,
            args.checkpoint_kind,
            args.tutor_endpoint or "<unset>",
            args.judge_endpoint or "<unset>",
            args.se_threshold,
            args.max_items,
            args.s3_out,
        )
        return 0

    if not args.tutor_endpoint or not args.judge_endpoint:
        log.error("FRQ needs both --tutor-endpoint and --judge-endpoint for a live run.")
        return 2

    respgen = respgen_mod.ServedRespGen(
        respgen_mod.RespGenConfig(endpoint=args.tutor_endpoint, served_model=args.served_model)
    )
    if not args.judge_config:
        log.error("Missing --judge-config (or the style must bundle judge_frozen.yaml).")
        return 2
    judge = judge_mod.ServedJudge(
        judge_mod.load_spec(args.judge_config), endpoint=args.judge_endpoint
    )

    with tempfile.TemporaryDirectory(prefix="frq-cat-") as tmp:
        bank = style.download_bank(benchmark, dest=Path(tmp) / "bank")
        irt_bank = style.load_irt_params(args.irt_params or benchmark)

        report = cat_loop.run_cat(
            style,
            bank=bank,
            irt_bank=irt_bank,
            respgen=respgen,
            judge=judge,
            se_threshold=args.se_threshold,
            max_items=args.max_items,
        )

    report_dict = report.to_dict()
    report_dict["run"] = {
        "cat_style": args.cat_style,
        "checkpoint": args.checkpoint,
        "checkpoint_kind": args.checkpoint_kind,
        "judge": judge.spec.name,
        "timestamp": datetime.now(UTC).isoformat(),
    }
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
        log.error("frq_cat runner failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
