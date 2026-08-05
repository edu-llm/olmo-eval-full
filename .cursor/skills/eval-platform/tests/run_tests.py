#!/usr/bin/env python3
"""Run the whole eval-platform verification suite. Exits non-zero on any failure.

    python .cursor/skills/eval-platform/tests/run_tests.py

Python rather than shell because the suite is half python and half bash, and the
bash half needs a real bash -- which on Windows is Git Bash or MSYS2, not the
`bash.exe` in System32 that launches WSL. Finding that once here is better than
each driver guessing, and if there is no bash at all the shell half is reported
as skipped rather than failed: an unrunnable test and a failing one mean
different things and should not look the same.

Nothing here touches AWS, the network, or a GPU. Every case runs the scripts
under --dry-run or against shims.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Set before _paths is imported, not just in the child environment below: this
# process does the import too, and would otherwise leave the one __pycache__ the
# children are told not to write.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import REPO, TESTS, find_bash

SKIP = 77  # autotools' convention, and what the bash-dependent tests exit with

# Explicit rather than globbed: the description is the only place a reader
# learns what a file called t_gt.sh is for, and a manifest cannot silently
# forget a test the way a glob can silently stop matching one.
PYTHON_TESTS = [
    ("python/test_registry_contract.py", "every registry name is a real olmo-eval task"),
    ("python/test_run_eval.py", "the argv and refusals of the in-Batch runner"),
    ("python/test_argv.py", "the sweep's argv against olmo-eval's real parser"),
    ("python/test_command_string.py", "the submitted command survives the platform's shlex.split"),
    ("python/test_passthrough.py", "--batch-size and --override survive both quoting rounds"),
    ("python/test_summarizer.py", "summarize_accuracy against a 7-benchmark run"),
    ("python/test_summarizer_contract.py", "the CSV/JSON shapes SKILL.md promises"),
    ("python/test_multiscorer.py", "wide-CSV naming when one metric has several scorers"),
    ("python/test_mc_variants.py", "the :mc variants resolve, travel and report intact"),
    ("python/test_partial_output.py", "what a crashed run leaves: partial files, heartbeat, salvage"),
    ("python/test_direct_gpu_groups.py", "eval-direct-gpu's smoke group, and the exclude key it needs"),
    ("python/check_scorer.py", "the real SQuAD normalizer and F1"),
    ("python/test_popqa_impl.py", "the real ContainmentScorer and PopQA.process_doc"),
    ("python/test_triviaqa_impl.py", "the real WindowedContainmentScorer and TriviaQA"),
    ("python/audit_io.py", "documented flags and outputs match the scripts"),
]

# A third field names the skill whose scripts a driver should be pointed at -- it
# becomes $EVALCK_SKILL, which lib.sh reads -- and is how e2e.sh comes to appear
# twice rather than once. Two skills ship run_eval_sweep.sh and the
# copies have diverged -- eval-direct-gpu's carries the interim S3 sync loop and
# the registry `exclude` its smoke group needs, neither of which exists here --
# so a suite that drives only the copy it happens to sit beside reports green for
# changes it never executed. That is not hypothetical: it is how the `exclude`
# feature shipped with every test passing. Everything without a third field runs
# against this skill, which is the only one that ships a submitter or a resolver.
SHELL_TESTS = [
    ("shell/t_syntax.sh", "every shipped script parses and self-documents"),
    ("shell/t_array.sh", "the empty-array idiom the submitter relies on"),
    ("shell/t_gt.sh", "the arithmetic guard behind the batch-size warning"),
    ("shell/t_valid.sh", "--batch-size and --override validation, and the other path's flags"),
    ("shell/t_warn.sh", "the batch-size and forgotten-selection warnings fire exactly when they should"),
    ("shell/localroot_probe.sh", "--checkpoint-root accepts a local directory"),
    ("shell/e2e.sh", "the whole sweep, end to end, with aws/uv shimmed", "eval-platform"),
    ("shell/e2e.sh", "the same sweep as eval-direct-gpu ships it", "eval-direct-gpu"),
]


def run(path: str, skill: str | None, bash: str | None, verbose: bool) -> tuple[str, float, str]:
    """Run one test. Returns (PASS|FAIL|SKIP, seconds, captured output)."""
    target = TESTS / path
    if path.endswith(".sh"):
        if bash is None:
            return "SKIP", 0.0, "no bash available"
        argv = [bash, str(target)]
    else:
        argv = [sys.executable, str(target)]

    env = dict(
        os.environ,
        OLMO_EVAL_ROOT=str(REPO),
        # Otherwise importing _paths leaves a __pycache__ in the suite directory
        # on every run, which is exactly the kind of stray output this suite was
        # rescued from being buried in.
        PYTHONDONTWRITEBYTECODE="1",
    )
    if skill:
        env["EVALCK_SKILL"] = skill
    started = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=str(TESTS))
    elapsed = time.monotonic() - started
    output = proc.stdout + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
    if verbose:
        print(output)
    if proc.returncode == SKIP:
        return "SKIP", elapsed, output.strip().splitlines()[-1] if output.strip() else "skipped"
    return ("PASS" if proc.returncode == 0 else "FAIL"), elapsed, output


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", metavar="SUBSTRING", help="only tests whose path contains this")
    ap.add_argument("-v", "--verbose", action="store_true", help="stream each test's output")
    ap.add_argument("--python-only", action="store_true", help="skip the bash half")
    args = ap.parse_args()

    bash = None if args.python_only else find_bash()
    selected = PYTHON_TESTS + ([] if args.python_only else SHELL_TESTS)
    if args.k:
        selected = [t for t in selected if args.k in t[0]]
    if not selected:
        print(f"no tests match {args.k!r}")
        return 2

    print(f"repo:   {REPO}")
    print(f"python: {sys.executable}")
    print(f"bash:   {bash or 'NOT FOUND -- the shell half will be skipped'}")
    print(f"running {len(selected)} test(s)\n")

    results, failed = [], []
    for path, description, *rest in selected:
        skill = rest[0] if rest else None
        status, elapsed, output = run(path, skill, bash, args.verbose)
        # The skill is part of the name, not just the description: two rows reading
        # "e2e.sh" would be indistinguishable in the failure report below.
        name = path.split("/", 1)[1] + (f" ({skill})" if skill else "")
        print(f"  {status}  {name:<32} {elapsed:5.1f}s  {description}")
        results.append((status, name))
        if status == "FAIL":
            failed.append((name, output))

    print()
    for name, output in failed:
        print("=" * 72)
        print(f"FAILED: {name}")
        print("=" * 72)
        print(output.rstrip())
        print()

    counts = {s: sum(1 for st, _ in results if st == s) for s in ("PASS", "FAIL", "SKIP")}
    print(f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
    if counts["SKIP"] and bash is None:
        print("\nThe skipped tests drive bash scripts and need a real bash.")
        print("Install Git for Windows, or run them on any Linux/macOS box.")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
