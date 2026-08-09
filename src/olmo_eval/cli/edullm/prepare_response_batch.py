"""CLI boundary for deterministic EduLLM response-batch preparation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import click

from olmo_eval.edullm.precomputed import PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION
from olmo_eval.edullm.response_prepare import (
    RESPONSE_BATCH_SOURCE_FORMATS,
    MissingResponsePolicy,
    ResponsePreparationError,
    build_precomputed_tutor_response_batch,
    check_precomputed_tutor_response_batch_write,
    load_fitted_scenario_roster,
    load_response_batch_source_manifest,
    write_precomputed_tutor_response_batch,
)


def _default_report_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.report.json")


def _summary(
    *,
    status: str,
    report: Mapping[str, Any],
    report_path: Path,
    report_sha256: str | None,
) -> dict[str, Any]:
    output = report["output"]
    return {
        "status": status,
        "response_source": {
            "kind": "precomputed_batch_jsonl",
            "schema_version": PRECOMPUTED_RESPONSE_BATCH_SCHEMA_VERSION,
            "path": output["path"],
            "sha256": output["sha256"],
        },
        "report_path": str(report_path.expanduser().resolve()),
        "report_sha256": report_sha256,
        "counts": {
            "models": output["model_count"],
            "rows": output["row_count"],
            "scenarios_per_model": output["scenario_count_per_model"],
            "blank_responses": output["blank_count"],
            "synthesized_blank_responses": output["synthesized_blank_count"],
        },
        "validation": dict(report["validation"]),
        "warnings": list(report["warnings"]),
    }


@click.command(name="prepare-response-batch")
@click.option(
    "--source-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help=(
        "Strict JSONL source manifest. Supported formats: "
        + ", ".join(RESPONSE_BATCH_SOURCE_FORMATS)
        + "."
    ),
)
@click.option(
    "--fitted-scenarios",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Fitted-bank scenario JSONL whose ordered IDs define required coverage.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Destination for strict edullm-precomputed-tutor-response-batch-v1 JSONL.",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help=("Validation report path; must share OUTPUT's directory (default: OUTPUT.report.json)."),
)
@click.option(
    "--missing",
    type=click.Choice(["error", "blank"]),
    default="error",
    show_default=True,
    help="How to handle an absent model-scenario pair.",
)
@click.option(
    "--check",
    is_flag=True,
    help="Validate, normalize, and hash without writing output artifacts.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace existing output/report files; source inputs remain protected.",
)
def prepare_response_batch(
    source_manifest: Path,
    fitted_scenarios: Path,
    output_path: Path,
    report_path: Path | None,
    missing: str,
    check: bool,
    overwrite: bool,
) -> None:
    """Prepare uploaded tutor responses for strict multi-model EduLLM replay."""

    report_output = report_path or _default_report_path(output_path)
    try:
        manifest = load_response_batch_source_manifest(source_manifest)
        roster = load_fitted_scenario_roster(fitted_scenarios)
        prepared = build_precomputed_tutor_response_batch(
            manifest,
            roster,
            missing=cast(MissingResponsePolicy, missing),
        )
        if check:
            report = check_precomputed_tutor_response_batch_write(
                prepared,
                output_path,
                report_output,
                overwrite=overwrite,
            )
            summary = _summary(
                status="checked",
                report=report,
                report_path=report_output,
                report_sha256=None,
            )
        else:
            written = write_precomputed_tutor_response_batch(
                prepared,
                output_path,
                report_output,
                overwrite=overwrite,
            )
            summary = _summary(
                status="written",
                report=written.report,
                report_path=written.report_path,
                report_sha256=written.report_sha256,
            )
    except ResponsePreparationError as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


__all__ = ["prepare_response_batch"]
