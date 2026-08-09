"""CLI for OLMo-owned standard and EduLLM evaluation modes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import click

from olmo_eval.cli.utils import console


@click.command(name="run-modes")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
    help="Strict YAML or JSON multi-mode run configuration.",
)
@click.option(
    "--check",
    is_flag=True,
    help="Run read-only preflight checks without starting providers or evaluation.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume a fingerprint-matched precomputed batch run at candidate boundaries.",
)
@click.option(
    "--status",
    is_flag=True,
    help="Print persisted batch progress without checking GPUs or starting providers.",
)
def run_modes(config_path: str, check: bool, resume: bool, status: bool) -> None:
    """Run standard OLMo evaluation, EduLLM adaptive evaluation, or both."""

    from olmo_eval.common.logging import configure_logging
    from olmo_eval.edullm.batch_resume import (
        load_checkpoint_chain,
        read_strict_json_object,
        read_strict_jsonl_objects,
        sha256_file,
    )
    from olmo_eval.edullm.mode import (
        BATCH_ARTIFACT_SCHEMA_VERSION,
        BATCH_PROGRESS_SCHEMA_VERSION,
        MODE_NAME,
    )
    from olmo_eval.runners.mode_config import load_mode_config
    from olmo_eval.runners.mode_orchestrator import ModeRunOrchestrator

    configure_logging(level="INFO")
    try:
        config = load_mode_config(config_path)
        if status:
            if check or resume:
                raise ValueError("--status cannot be combined with --check or --resume")
            mode_root = config.output_dir / "modes" / MODE_NAME

            def read_optional_object(path: Path) -> dict[str, object] | None:
                if not path.exists() and not path.is_symlink():
                    return None
                return read_strict_json_object(path)

            progress = read_optional_object(mode_root / "progress.json")
            batch_summary = read_optional_object(mode_root / "batch_summary.json")
            batch_manifest = read_optional_object(mode_root / "manifest.json")
            artifacts = {
                "progress": progress,
                "batch_summary": batch_summary,
                "batch_manifest": batch_manifest,
            }
            if all(artifact is None for artifact in artifacts.values()):
                raise ValueError(f"no run status artifacts exist under {config.output_dir}")
            expected_schemas = {
                "progress": BATCH_PROGRESS_SCHEMA_VERSION,
                "batch_summary": BATCH_ARTIFACT_SCHEMA_VERSION,
                "batch_manifest": BATCH_ARTIFACT_SCHEMA_VERSION,
            }
            fingerprints: dict[str, str] = {}
            generations: dict[str, int] = {}
            statuses: dict[str, str] = {}
            for name, artifact in artifacts.items():
                if artifact is None:
                    continue
                if artifact.get("schema_version") != expected_schemas[name]:
                    raise ValueError(f"{name} has an unsupported schema_version")
                artifact_run_id = artifact.get("run_id")
                if artifact_run_id != config.run_id:
                    raise ValueError(
                        f"{name}.run_id {artifact_run_id!r} does not match configured "
                        f"run_id {config.run_id!r}"
                    )
                artifact_status = artifact.get("status")
                if not isinstance(artifact_status, str) or artifact_status not in {
                    "running",
                    "succeeded",
                    "failed",
                    "cancelled",
                }:
                    raise ValueError(f"{name}.status is not a supported batch status")
                fingerprint = artifact.get("resume_contract_fingerprint")
                if (
                    not isinstance(fingerprint, str)
                    or len(fingerprint) != 64
                    or any(character not in "0123456789abcdef" for character in fingerprint)
                ):
                    raise ValueError(
                        f"{name}.resume_contract_fingerprint must be a lowercase SHA-256 digest"
                    )
                generation = artifact.get("checkpoint_generation")
                if (
                    isinstance(generation, bool)
                    or not isinstance(generation, int)
                    or generation < 0
                ):
                    raise ValueError(f"{name}.checkpoint_generation must be a non-negative integer")
                fingerprints[name] = fingerprint
                generations[name] = generation
                statuses[name] = artifact_status

            if batch_manifest is not None and (
                batch_manifest.get("mode") != MODE_NAME
                or batch_manifest.get("execution") != "precomputed_batch"
            ):
                raise ValueError("batch_manifest is not an EduLLM precomputed batch manifest")
            if len(set(fingerprints.values())) > 1:
                raise ValueError(
                    "status artifacts disagree on resume_contract_fingerprint: "
                    + ", ".join(f"{name}={value}" for name, value in fingerprints.items())
                )
            if len(set(generations.values())) > 1:
                raise ValueError(
                    "status artifacts disagree on checkpoint_generation: "
                    + ", ".join(f"{name}={value}" for name, value in generations.items())
                )
            if len(set(statuses.values())) > 1:
                raise ValueError(
                    "status artifacts disagree on status: "
                    + ", ".join(f"{name}={value}" for name, value in statuses.items())
                )
            fingerprint = next(iter(fingerprints.values()))
            checkpoint = load_checkpoint_chain(
                mode_root,
                fingerprint_sha256=fingerprint,
            )
            view_generation = next(iter(generations.values()))
            if checkpoint.generation != view_generation:
                raise ValueError(
                    "status views are stale relative to the authoritative checkpoint chain: "
                    f"views={view_generation}, checkpoint={checkpoint.generation}; "
                    "resume the run to regenerate status views"
                )
            if checkpoint.pointer_stale:
                raise ValueError(
                    "checkpoint pointer is stale; resume the run to repair it before "
                    "reporting status"
                )

            model_results_path = mode_root / "model_results.jsonl"
            attempt_history_path = mode_root / "attempt_history.jsonl"
            model_results_view = read_strict_jsonl_objects(model_results_path)
            attempt_history_view = read_strict_jsonl_objects(attempt_history_path)
            if tuple(dict(row) for row in model_results_view) != tuple(
                dict(row) for row in checkpoint.model_results
            ):
                raise ValueError(
                    "model_results.jsonl contradicts the authoritative checkpoint chain"
                )
            if tuple(dict(row) for row in attempt_history_view) != tuple(
                dict(row) for row in checkpoint.attempts
            ):
                raise ValueError(
                    "attempt_history.jsonl contradicts the authoritative checkpoint chain"
                )

            terminal_statuses = {"succeeded", "failed", "cancelled"}
            checkpoint_statuses: list[str] = []
            checkpoint_candidate_ids: set[str] = set()
            checkpoint_output_paths: set[str] = set()
            for position, row in enumerate(checkpoint.model_results):
                row_status = row.get("status")
                if not isinstance(row_status, str) or row_status not in terminal_statuses:
                    raise ValueError(f"checkpoint model_results[{position}].status is not terminal")
                candidate_id = row.get("candidate_id")
                if not isinstance(candidate_id, str) or not candidate_id:
                    raise ValueError(
                        f"checkpoint model_results[{position}].candidate_id is invalid"
                    )
                if candidate_id in checkpoint_candidate_ids:
                    raise ValueError("checkpoint model_results contains a duplicate candidate_id")
                checkpoint_candidate_ids.add(candidate_id)
                output_path = row.get("output_dir")
                if not isinstance(output_path, str) or not output_path:
                    raise ValueError(f"checkpoint model_results[{position}].output_dir is invalid")
                if output_path in checkpoint_output_paths:
                    raise ValueError("checkpoint model_results contains a duplicate output_dir")
                checkpoint_output_paths.add(output_path)
                checkpoint_statuses.append(row_status)
            authoritative_counts = {
                "models_completed": len(checkpoint.model_results),
                "models_succeeded": checkpoint_statuses.count("succeeded"),
                "models_failed": checkpoint_statuses.count("failed"),
                "models_cancelled": checkpoint_statuses.count("cancelled"),
            }

            count_views: dict[str, object] = {}
            if progress is not None:
                count_views["progress"] = progress
            if batch_summary is not None:
                count_views["batch_summary"] = batch_summary
            if batch_manifest is not None:
                count_views["batch_manifest.progress"] = batch_manifest.get("progress")
            model_totals: dict[str, int] = {}
            for name, raw_counts in count_views.items():
                if not isinstance(raw_counts, Mapping):
                    raise ValueError(f"{name} must contain an object of batch counts")
                counts = cast(Mapping[str, object], raw_counts)
                for field, expected in authoritative_counts.items():
                    observed = counts.get(field)
                    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
                        raise ValueError(f"{name}.{field} must be a non-negative integer")
                    if observed != expected:
                        raise ValueError(
                            f"{name}.{field} contradicts the authoritative checkpoint: "
                            f"view={observed}, checkpoint={expected}"
                        )
                total = counts.get("models_total")
                pending = counts.get("models_pending")
                for field, value in (("models_total", total), ("models_pending", pending)):
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(f"{name}.{field} must be a non-negative integer")
                assert isinstance(total, int) and isinstance(pending, int)
                if total < authoritative_counts["models_completed"]:
                    raise ValueError(f"{name}.models_total is smaller than the checkpoint roster")
                if pending != total - authoritative_counts["models_completed"]:
                    raise ValueError(
                        f"{name}.models_pending does not reconcile with the checkpoint"
                    )
                model_totals[name] = total
            if not model_totals:
                raise ValueError("no persisted status view contains authoritative batch counts")
            if len(set(model_totals.values())) > 1:
                raise ValueError(
                    "status artifacts disagree on models_total: "
                    + ", ".join(f"{name}={value}" for name, value in model_totals.items())
                )

            if batch_summary is not None:
                if batch_summary.get("model_results_path") != "model_results.jsonl":
                    raise ValueError("batch_summary.model_results_path is invalid")
                if batch_summary.get("attempt_history_path") != "attempt_history.jsonl":
                    raise ValueError("batch_summary.attempt_history_path is invalid")
                attempts_committed = batch_summary.get("attempts_committed")
                if (
                    isinstance(attempts_committed, bool)
                    or not isinstance(attempts_committed, int)
                    or attempts_committed < 0
                ):
                    raise ValueError(
                        "batch_summary.attempts_committed must be a non-negative integer"
                    )
                if attempts_committed != len(checkpoint.attempts):
                    raise ValueError(
                        "batch_summary.attempts_committed contradicts the authoritative "
                        f"checkpoint: view={attempts_committed!r}, "
                        f"checkpoint={len(checkpoint.attempts)}"
                    )
            if batch_manifest is not None:
                paths = batch_manifest.get("model_result_paths")
                expected_paths = [row.get("output_dir") for row in checkpoint.model_results]
                if paths != expected_paths:
                    raise ValueError(
                        "batch_manifest.model_result_paths contradict the authoritative checkpoint"
                    )
                artifact_hashes = batch_manifest.get("artifact_sha256")
                if not isinstance(artifact_hashes, Mapping):
                    raise ValueError("batch_manifest.artifact_sha256 must be an object")
                typed_artifact_hashes = cast(Mapping[str, object], artifact_hashes)
                for name, path in (
                    ("model_results.jsonl", model_results_path),
                    ("attempt_history.jsonl", attempt_history_path),
                    ("batch_summary.json", mode_root / "batch_summary.json"),
                ):
                    if typed_artifact_hashes.get(name) != sha256_file(path):
                        raise ValueError(
                            f"batch_manifest artifact hash is stale or invalid for {name}"
                        )

            view_status = next(iter(statuses.values()))
            model_total = next(iter(model_totals.values()))
            if view_status == "succeeded" and (
                authoritative_counts["models_completed"] != model_total
                or authoritative_counts["models_failed"]
                or authoritative_counts["models_cancelled"]
            ):
                raise ValueError("succeeded status contradicts the authoritative checkpoint")
            if view_status == "failed" and (
                authoritative_counts["models_completed"] != model_total
                or authoritative_counts["models_failed"] == 0
            ):
                raise ValueError("failed status contradicts the authoritative checkpoint")

            payload = {
                "status": "status_available",
                "run_id": config.run_id,
                "output_dir": str(config.output_dir),
                **artifacts,
                "checkpoint": {
                    "generation": checkpoint.generation,
                    "checkpoint_sha256": checkpoint.checkpoint_sha256,
                    "models_committed": len(checkpoint.model_results),
                    "attempts_committed": len(checkpoint.attempts),
                },
            }
            click.echo(json.dumps(payload, indent=2))
            return

        orchestrator = ModeRunOrchestrator(config, resume=resume)
        if check:
            preflight = orchestrator.preflight()
            click.echo(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "run_id": config.run_id,
                        "selected_modes": list(preflight.selected_modes),
                        "provider_names": list(preflight.provider_names),
                        "available_gpu_ids": list(preflight.available_gpu_ids),
                        "judge_runtime": dict(preflight.judge_runtime or {}),
                    },
                    indent=2,
                )
            )
            return

        results = orchestrator.run()
    except Exception as exc:
        console.print(f"[bold red]Mode evaluation failed:[/bold red] {exc}")
        raise click.ClickException(str(exc)) from None

    failed = [result for result in results.values() if not result.succeeded]
    console.print(
        f"Run {config.run_id!r} finished with {len(results) - len(failed)}/"
        f"{len(results)} modes successful. Results: {config.output_dir}"
    )
    if failed:
        raise click.ClickException(
            "one or more modes did not succeed: "
            + ", ".join(f"{result.mode}={result.status.value}" for result in failed)
        )


__all__ = ["run_modes"]
