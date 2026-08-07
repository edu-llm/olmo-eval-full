"""CLI for OLMo-owned standard and EduLLM evaluation modes."""

from __future__ import annotations

import json

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
def run_modes(config_path: str, check: bool) -> None:
    """Run standard OLMo evaluation, EduLLM adaptive evaluation, or both."""

    from olmo_eval.common.logging import configure_logging
    from olmo_eval.runners.mode_config import load_mode_config
    from olmo_eval.runners.mode_orchestrator import ModeRunOrchestrator

    configure_logging(level="INFO")
    try:
        config = load_mode_config(config_path)
        orchestrator = ModeRunOrchestrator(config)
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
