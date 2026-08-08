"""Evaluate one checkpoint inside an eduLLM platform Batch job.

    python -m olmo_eval.platform.run_eval \
        --checkpoint s3://sbsandbox-intern-edullm-outputs/teams/T/runs/R/checkpoints/step2000/ \
        --benchmarks "csqa hellaswag piqa socialiqa arc_easy" --limit 2

One job is one checkpoint. The platform submits a job per run, so the
multi-checkpoint sweep that makes sense on a machine you own is the caller's loop
here, not this program's.

WHERE OUTPUT GOES IS NOT OURS TO CHOOSE. ``$EDULLM_OUTPUT_PREFIX`` is set by the
platform to ``s3://<outputs-bucket>/teams/<team>/runs/<run-id>/`` and the
workload role permits writes nowhere else, so a prefix computed here rather than
read would be denied at the end of a run rather than the start. The platform's
own client library states the rule: "A container should prefer
EDULLM_OUTPUT_PREFIX to calling this."

WHY THE CHECKPOINT IS NOT DOWNLOADED FIRST. The ``olmo_core`` provider resolves
remote checkpoints itself -- ``olmo_core_utils._read_checkpoint_config`` branches
on ``"://" in path`` and reads through ``cached_path``, and
``TransformerGenerationModule.from_checkpoint`` takes ``pre_download=True``. So an
``s3://`` directory goes straight to ``-m``. That is specific to this provider:
the vLLM path has no S3 loader and fails inside the model loader instead.

WHY BENCHMARK NAMES ARRIVE ALREADY RESOLVED. The registry that turns a group name
into a benchmark list, and the cost estimate that makes a wrong request cheap to
notice, live in the skill that submits the job. That validation has to happen
before the submission is dispatched to be worth anything, so by the time this
runs the argument is a plain list and there is nothing left to look up.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Platform variables recorded in provenance. Only OUTPUT_PREFIX is load-bearing;
#: the rest exist so a result can be traced back without the submitting shell.
PLATFORM_VARIABLES = (
    "EDULLM_RUN_ID",
    "EDULLM_TEAM",
    "EDULLM_COMMIT_SHA",
    "EDULLM_DATASET_RELEASE",
    "EDULLM_OUTPUT_BUCKET",
    "EDULLM_OUTPUT_PREFIX",
    "EDULLM_CHECKPOINT_DIR",
    "EDULLM_WANDB_PROJECT",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
)

PROVENANCE_OBJECT = "eval_provenance.json"
READY_MARKER = "_READY"
FAILED_MARKER = "_FAILED"

# Written before the eval starts and refreshed on every interim upload, then replaced by
# a terminal marker. _READY and _FAILED are both written by this process, so a hard kill
# leaves neither -- and a run with no marker at all is indistinguishable from one that
# never started. A stale timestamp here says "died at about this time, and what is beside
# it is everything that survived".
IN_PROGRESS_MARKER = "_IN_PROGRESS"

DEFAULT_UPLOAD_INTERVAL_SECONDS = 180.0
UPLOADER_JOIN_SECONDS = 60.0


class RunEvalError(RuntimeError):
    """A failure worth reporting as itself rather than as a traceback."""


@dataclass
class Plan:
    """Everything decided before anything is spent or written."""

    checkpoint: str
    benchmarks: tuple[str, ...]
    output_prefix: str
    provider: str
    limit: int | None
    tokenizer: str | None
    local_out: Path
    extra_overrides: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)

    @property
    def command(self) -> list[str]:
        """The olmo-eval invocation, in the one argument order that works.

        ``run`` has no top-level provider option; it goes through ``--harness``.
        Each ``-o`` binds to the *preceding* ``--harness`` or ``-t`` and the two
        accept disjoint keys, so a provider override after a task is a usage
        error and a ``limit=`` before the first task is one too.
        """
        argv = [
            "olmo-eval",
            "run",
            "-m",
            self.checkpoint,
            "--harness",
            "default",
            "-o",
            f"provider.kind={self.provider}",
        ]
        if self.tokenizer:
            argv += ["-o", f"provider.tokenizer={self.tokenizer}"]
        # One -o per override. They all bind to --harness above, which is why they
        # are emitted before the first -t rather than collected into one flag.
        for override in self.extra_overrides:
            argv += ["-o", override]
        for benchmark in self.benchmarks:
            argv += ["-t", benchmark]
            if self.limit is not None:
                argv += ["-o", f"limit={self.limit}"]
        argv += ["-O", str(self.local_out)]
        return argv


def _split_names(raw: str) -> tuple[str, ...]:
    """Benchmark names from a form field, which may be space or comma separated."""
    return tuple(name for name in raw.replace(",", " ").split() if name)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise RunEvalError(f"not an s3:// uri: {uri}")
    _, _, remainder = uri.partition("s3://")
    bucket, _, key = remainder.partition("/")
    if not bucket:
        raise RunEvalError(f"s3 uri names no bucket: {uri}")
    return bucket, key


def _client() -> Any:
    try:
        import boto3
    except ImportError as error:  # pragma: no cover - present in every image we target
        raise RunEvalError(
            "boto3 is required to write results to S3 and is not installed"
        ) from error
    return boto3.client("s3")


def build_plan(args: argparse.Namespace, environ: dict[str, str]) -> Plan:
    """Resolve arguments and environment into a plan, refusing what cannot work.

    Refuses here rather than after the model is loaded: every check below is
    free, and the cheapest failure is the one that happens before a GPU has been
    asked to do anything.
    """
    output_prefix = args.output_prefix or environ.get("EDULLM_OUTPUT_PREFIX", "")
    if not output_prefix:
        raise RunEvalError(
            "no output prefix: $EDULLM_OUTPUT_PREFIX is unset and --output-prefix "
            "was not given. Inside a platform job the variable is always set, so "
            "this usually means the program is being run outside one."
        )
    if not output_prefix.startswith("s3://"):
        raise RunEvalError(f"output prefix must be an s3:// uri, got {output_prefix!r}")
    if not output_prefix.endswith("/"):
        output_prefix += "/"

    benchmarks = _split_names(args.benchmarks)
    if not benchmarks:
        raise RunEvalError("--benchmarks resolved to an empty list")

    if args.limit is not None and args.limit < 1:
        raise RunEvalError(f"--limit must be positive, got {args.limit}")

    # Trailing slash removed either way: the provider joins "config.json" and
    # "model_and_optim" onto whatever it is given, and a doubled separator in an
    # s3 key is a different key rather than the same one.
    return Plan(
        checkpoint=args.checkpoint.rstrip("/"),
        benchmarks=benchmarks,
        output_prefix=output_prefix,
        provider=args.provider,
        limit=args.limit,
        tokenizer=args.tokenizer,
        local_out=Path(args.local_out)
        if args.local_out
        else Path(tempfile.mkdtemp(prefix="olmo-eval-")),
        extra_overrides=tuple(args.override or ()),
        environment={key: environ[key] for key in PLATFORM_VARIABLES if environ.get(key)},
    )


def provenance(plan: Plan, status: str, **extra: object) -> dict[str, object]:
    """What ran, recorded so a result in S3 can be explained without this shell.

    Carries the eval's own commit where it can be found. The platform's manifest
    names the repository whose *image* ran, which on the OLMo-core route is not
    the repository this code came from, so this is the only place the two are
    written down together.
    """
    document: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "recorded_at": datetime.now(UTC).isoformat(),
        "checkpoint": plan.checkpoint,
        "benchmarks": list(plan.benchmarks),
        "provider": plan.provider,
        "limit": plan.limit,
        "olmo_eval_command": plan.command,
        "olmo_eval_version": _installed_version(),
        "platform_environment": plan.environment,
    }
    document.update(extra)
    return document


def _installed_version() -> str:
    try:
        from importlib.metadata import version

        return version("olmo-eval")
    except Exception:  # pragma: no cover - a missing dist is itself worth recording
        return "unknown"


def put_json(prefix: str, name: str, document: dict[str, object]) -> None:
    bucket, key_prefix = _parse_s3_uri(prefix)
    _client().put_object(
        Bucket=bucket,
        Key=f"{key_prefix}{name}",
        Body=(json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def put_marker(prefix: str, name: str, body: str) -> None:
    bucket, key_prefix = _parse_s3_uri(prefix)
    _client().put_object(Bucket=bucket, Key=f"{key_prefix}{name}", Body=body.encode("utf-8"))


def remove_marker(prefix: str, name: str) -> None:
    """Drop a marker that a later one supersedes."""
    bucket, key_prefix = _parse_s3_uri(prefix)
    try:
        _client().delete_object(Bucket=bucket, Key=f"{key_prefix}{name}")
    except Exception as error:  # noqa: BLE001 - a stale marker misleads, it does not break
        print(f"[run_eval] could not remove {name}: {error}", file=sys.stderr)


def upload_tree(local: Path, prefix: str) -> int:
    """Copy everything olmo-eval wrote to the platform's prefix.

    Deliberately not olmo-eval's own ``--s3-bucket``/``--s3-prefix`` flags. Those
    work, but they build a path of their own under whatever prefix they are given
    -- group, model name, model hash, experiment id -- and the result is a layout
    that is permitted but that nobody reading the run can predict. Uploading here
    keeps the tree the shape it had on disk.
    """
    bucket, key_prefix = _parse_s3_uri(prefix)
    client = _client()
    uploaded = 0
    for path in sorted(local.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local).as_posix()
        client.upload_file(str(path), bucket, f"{key_prefix}{relative}")
        uploaded += 1
    return uploaded


def heartbeat_body() -> str:
    """Body of the in-progress marker: when this job was last known to be alive."""
    return f"last_upload={datetime.now(UTC).isoformat()}\n"


def _periodic_upload(plan: Plan, stop: threading.Event, interval: float) -> None:
    """Copy the output tree up every ``interval`` seconds until told to stop.

    Errors are printed and swallowed. A transient S3 failure partway through a run is
    not a reason to lose the run: the next tick tries again, and the final upload after
    the subprocess exits is the one whose failure is reported.
    """
    while not stop.wait(interval):
        try:
            uploaded = upload_tree(plan.local_out, plan.output_prefix)
            put_marker(plan.output_prefix, IN_PROGRESS_MARKER, heartbeat_body())
            print(f"[run_eval] interim upload: {uploaded} object(s)", flush=True)
        except Exception as error:  # noqa: BLE001 - interim uploads are best effort
            print(f"[run_eval] interim upload failed: {error}", file=sys.stderr, flush=True)


def run(plan: Plan, upload_interval: float = DEFAULT_UPLOAD_INTERVAL_SECONDS) -> int:
    """Invoke olmo-eval, streaming its output so a stuck run is visible in logs.

    The output tree is copied up while the subprocess runs, not only after it, because
    the failures worth protecting against -- out of memory, spot reclamation, hitting
    the runtime ceiling -- kill this process too and would otherwise take every scored
    instance with them.
    """
    plan.local_out.mkdir(parents=True, exist_ok=True)
    print(f"[run_eval] {shlex.join(plan.command)}", flush=True)

    stop = threading.Event()
    uploader: threading.Thread | None = None
    if upload_interval > 0:
        put_marker(plan.output_prefix, IN_PROGRESS_MARKER, heartbeat_body())
        uploader = threading.Thread(
            target=_periodic_upload,
            args=(plan, stop, upload_interval),
            name="run_eval-uploader",
            daemon=True,
        )
        uploader.start()

    try:
        completed = subprocess.run(plan.command, check=False)
        return completed.returncode
    finally:
        stop.set()
        if uploader is not None:
            # Bounded so a wedged upload cannot hold the job past its own result; the
            # thread is a daemon, so anything still running dies with the process.
            uploader.join(timeout=UPLOADER_JOIN_SECONDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m olmo_eval.platform.run_eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="OLMo-core checkpoint directory. An s3:// uri is read in place; the "
        "olmo_core provider resolves remote paths itself.",
    )
    parser.add_argument(
        "--benchmarks",
        required=True,
        help="Space or comma separated olmo-eval task names, already validated by "
        "whatever submitted this job.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Instances per task.")
    parser.add_argument(
        "--upload-interval",
        type=float,
        default=DEFAULT_UPLOAD_INTERVAL_SECONDS,
        help="Seconds between interim uploads of whatever has been scored so far. "
        "0 disables them, restoring upload-once-at-the-end behaviour.",
    )
    parser.add_argument(
        "--provider",
        default="olmo_core",
        help="Inference backend. Defaults to olmo_core, which is the one the "
        "OLMo-core image can load: it carries torch and ai2-olmo-core and not vLLM.",
    )
    parser.add_argument(
        "--tokenizer", default=None, help="HF tokenizer id, if the checkpoint names none."
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Overrides $EDULLM_OUTPUT_PREFIX. For running outside a platform job; "
        "inside one the platform's value is the only writable location.",
    )
    parser.add_argument("--local-out", default=None, help="Where olmo-eval writes before upload.")
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        help="Extra harness override, passed through before the first task.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and the olmo-eval command, write nothing, exit 0.",
    )
    args = parser.parse_args(argv)

    try:
        plan = build_plan(args, dict(os.environ))
    except RunEvalError as error:
        print(f"[run_eval] {error}", file=sys.stderr)
        return 2

    print(f"[run_eval] checkpoint     {plan.checkpoint}", flush=True)
    print(f"[run_eval] benchmarks     {' '.join(plan.benchmarks)}", flush=True)
    print(f"[run_eval] provider       {plan.provider}", flush=True)
    print(f"[run_eval] limit          {plan.limit}", flush=True)
    print(f"[run_eval] output prefix  {plan.output_prefix}", flush=True)

    if args.dry_run:
        print(f"[run_eval] would run: {shlex.join(plan.command)}", flush=True)
        return 0

    # Written before the eval, so a job killed for running out of time or memory
    # still leaves something behind saying what it was doing. The platform audits
    # for runs that saved nothing, and a run that saved nothing is
    # indistinguishable from one whose prefix was never reachable.
    try:
        put_json(plan.output_prefix, PROVENANCE_OBJECT, provenance(plan, "started"))
    except Exception as error:  # noqa: BLE001 - the reason matters more than the type
        print(
            f"[run_eval] could not write to {plan.output_prefix}: {error}\n"
            "[run_eval] refusing to run: results would have nowhere to go.",
            file=sys.stderr,
        )
        return 3

    exit_code = run(plan, upload_interval=args.upload_interval)

    uploaded = 0
    upload_error: str | None = None
    try:
        uploaded = upload_tree(plan.local_out, plan.output_prefix)
    except Exception as error:  # noqa: BLE001 - reported, never raised past here
        upload_error = f"{type(error).__name__}: {error}"

    status = "ok" if exit_code == 0 and upload_error is None else "failed"
    try:
        put_json(
            plan.output_prefix,
            PROVENANCE_OBJECT,
            provenance(
                plan,
                status,
                exit_code=exit_code,
                objects_uploaded=uploaded,
                upload_error=upload_error,
            ),
        )
        put_marker(
            plan.output_prefix,
            READY_MARKER if status == "ok" else FAILED_MARKER,
            f"exit_code={exit_code} objects_uploaded={uploaded}\n",
        )
        # Only once a terminal marker exists, so a reader never sees a prefix with no
        # marker at all and has to guess whether the job is running or gone.
        remove_marker(plan.output_prefix, IN_PROGRESS_MARKER)
    except Exception as error:  # noqa: BLE001 - never mask the eval's own result
        print(f"[run_eval] terminal record could not be written: {error}", file=sys.stderr)

    print(
        f"[run_eval] {status}: exit_code={exit_code} uploaded={uploaded} -> {plan.output_prefix}",
        flush=True,
    )
    if upload_error:
        print(f"[run_eval] upload error: {upload_error}", file=sys.stderr)
    return exit_code if exit_code != 0 else (0 if upload_error is None else 4)


if __name__ == "__main__":
    raise SystemExit(main())
