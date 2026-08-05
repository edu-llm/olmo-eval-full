"""Do --batch-size and --override survive the two rounds of shell quoting?

The submitter's `command:` line is quoted twice before anything runs it: the
platform does `shlex.split(FORM_COMMAND)` to get `bash -lc <payload>`, and then
bash splits the payload again. A value that loses or gains a quote at either
boundary produces a runner invocation nobody wrote.

test_command_string.py covers the benchmark list across that boundary. This
covers the two repeatable knobs, which is where an unbalanced quote would first
show: --override appears N times and its values are user-supplied.

This began life as a probe that printed what it saw and always exited 0. The
expectations below are the ones its output was being read for.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO, SUBMIT, find_bash

bash = find_bash()
if bash is None:
    print("SKIP: no bash found; this test drives submit_eval_run.sh")
    raise SystemExit(77)

BASE = [
    "--checkpoint",
    "s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/r-1/checkpoints/step1000",
    "--team", "eval-inference", "--experiment", "probe",
    "--wandb-project", "p", "--dry-run",
]

failures = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def command_line(extra):
    """The submitter's `command:` line for a dry run, as the platform would read it."""
    env = dict(os.environ, OLMO_EVAL_ROOT=str(REPO))
    proc = subprocess.run([bash, str(SUBMIT), *BASE, *extra],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"dry run failed:\n{proc.stdout}\n{proc.stderr}")
    lines = proc.stdout.splitlines()
    return lines[lines.index("command:") + 1].strip()


for label, extra, want_overrides in [
    ("--batch-size 64", ["--group", "smoke", "--batch-size", "64"],
     ["provider.kwargs.batch_size=64"]),
    ("--batch-size 64 plus two --override",
     ["--group", "smoke", "--batch-size", "64",
      "--override", "foo.bar=1", "--override", "baz.qux=2"],
     ["provider.kwargs.batch_size=64", "foo.bar=1", "baz.qux=2"]),
    ("neither flag", ["--group", "smoke"], []),
]:
    print(f"### {label}")
    cmd = command_line(extra)

    check("single quotes balanced", cmd.count("'") % 2 == 0, f"{cmd.count(chr(39))} quotes")
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        check("outer shlex.split succeeds", False, str(exc))
        continue
    check("outer shlex.split succeeds", True)
    check("outer split is bash -lc <payload>",
          argv[:2] == ["bash", "-lc"] and len(argv) == 3, str(argv[:2]) + f" len={len(argv)}")

    try:
        inner = shlex.split(argv[2])
    except ValueError as exc:
        check("payload splits without a quoting error", False, str(exc))
        continue
    check("payload splits without a quoting error", True)

    got = [inner[i + 1] for i, tok in enumerate(inner) if tok == "--override"]
    check(f"exactly {len(want_overrides)} --override argument(s) reach the runner",
          len(got) == len(want_overrides), str(got))
    for want in want_overrides:
        check(f"--override {want} survives as one token", want in got, str(got))
    # A lost quote would split "a=1" into "a=1" plus stray words, or swallow the
    # next flag; either way the count above moves, so also pin that each value
    # is still a single k=v with nothing appended.
    check("every override value is a bare key=value",
          all(v.count("=") == 1 and " " not in v for v in got), str(got))
    print()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("BATCH SIZE AND OVERRIDES SURVIVE BOTH QUOTING ROUNDS")
