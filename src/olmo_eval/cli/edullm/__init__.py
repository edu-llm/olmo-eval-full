"""EduLLM input preparation commands."""

import click

from olmo_eval.cli.edullm.prepare_response_batch import prepare_response_batch


@click.group()
def edullm() -> None:
    """Prepare and inspect EduLLM evaluation inputs."""


edullm.add_command(prepare_response_batch)

__all__ = ["edullm", "prepare_response_batch"]
