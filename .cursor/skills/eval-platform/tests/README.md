# Verifying the eval-platform skill

```bash
python .cursor/skills/eval-platform/tests/run_tests.py
```

About two minutes, no arguments, non-zero exit if anything fails. `-v` streams
each test's output, `-k SUBSTRING` selects a few, `--python-only` skips the half
that needs bash.

**None of this may be run against live AWS.** Every case drives the scripts with
`--dry-run` or against fake `aws`/`uv`/`python3` executables on `PATH`. Nothing
reaches the network, an AWS account, or a GPU, and `e2e.sh` asserts on an empty
`aws` call log in the cases where a call would mean real spending. If you add a
test, keep that property: the whole point of a suite you can run on a laptop is
that it costs nothing to run it often.

## What it covers

The skill owns no evaluation logic — it resolves benchmark names, stages a
checkpoint, shells out to olmo-eval once, and reports where the JSON went. So
these tests are mostly about the seams, which is where its bugs have actually
been:

| Area | Tests |
|---|---|
| The registry is honest | `test_registry_contract.py` — every name in `benchmarks.json` is a real `@register`ed olmo-eval task, every group member is defined, no alias shadows a runnable task, and BENCHMARKS.md documents exactly what the registry offers |
| The argv olmo-eval receives | `test_argv.py`, `test_run_eval.py` — built against olmo-eval's *real* parser. `-o` binds to the preceding `--harness` or `-t`, and the two accept disjoint keys, so a misplaced override is a usage error rather than a silent no-op |
| The command survives quoting | `test_command_string.py`, `test_passthrough.py` — the submitted command is split by `shlex` on the platform and again by bash in the container. A lost quote produces an invocation nobody wrote, after a reviewer has already approved it |
| The accuracy tables | `test_summarizer.py`, `test_summarizer_contract.py`, `test_multiscorer.py` — column naming, numeric step ordering, and that a failed checkpoint keeps a row with a blank score instead of vanishing |
| Scorers and tasks | `check_scorer.py`, `test_popqa_impl.py`, `test_triviaqa_impl.py` — the real source, lifted out with `ast` and run against stand-ins, since olmo-eval is not importable here |
| Docs match behaviour | `audit_io.py` — every flag each script parses is documented and every documented flag is parsed |
| The scripts themselves | `shell/` — see below |

## The bash half

`shell/` drives the two entry-point scripts as scripts. It needs a real bash;
`run_tests.py` looks for Git Bash and MSYS2 by their usual locations before
falling back to `PATH`, and deliberately ignores the `bash.exe` in
`System32` — that one launches WSL, which is a different filesystem where the
repo path does not exist. With no bash found, these are reported as **SKIP**
rather than FAIL, and the suite still exits 0.

`e2e.sh` is the substantial one: it builds fake checkpoints and shims, then runs
the whole sweep — discovery, conversion, evaluation, markers, aggregation —
including the failure paths, and checks the `_READY`/`_FAILED` markers and the
resulting accuracy table. It runs **twice**, once per skill that ships a
`run_eval_sweep.sh`, because the two copies have diverged and a suite that drives
only the one it sits beside reports green for changes it never executed — which is
how eval-direct-gpu's `exclude` resolution shipped with the suite passing.
`$EVALCK_SKILL` is what selects a skill; `lib.sh` reads it, `run_tests.py` sets it
from the third field of a manifest entry, and the driver derives every count from
the registry it finds there rather than stating one. Anything only one copy has —
eval-direct-gpu's interim S3 sync, for instance — is probed for and reported as
`n/a` where it is absent, not quietly assumed.

The rest are narrow: `t_syntax.sh` parses every shipped
script, `t_valid.sh` and `t_warn.sh` cover `--batch-size`/`--override` validation
and the out-of-memory warning, `localroot_probe.sh` covers local
`--checkpoint-root`, and `t_array.sh` and `t_gt.sh` pin two bash behaviours the
submitter quietly depends on (an empty array under `set -u`, and `[[ "" -gt N ]]`
being false rather than fatal).

## Where things are

```
tests/
  run_tests.py     the entry point; the manifest at the top says what each test is for
  _paths.py        repo-root resolution for the python half, plus the bash search
  lib.sh           the same for the shell half, plus which skill is under test
                   ($EVALCK_SKILL) and the check/has/finish helpers
  python/          needs only python
  shell/           needs a real bash
```

Both halves resolve the repo as `$OLMO_EVAL_ROOT` with a relative fallback,
matching what `submit_eval_run.sh` itself does. `run_tests.py` exports it so a
test and the script it drives can never disagree about which checkout is under
test. Nothing has an absolute path in it; the suite runs from wherever the repo
is checked out.

## Writing a new one

A test is any executable that exits 0 for pass, non-zero for fail, and 77 for
"cannot run here". Add it to the manifest in `run_tests.py` with a one-line
description — that description is the only place a reader learns what a file
called `t_gt.sh` is for. A third field names the skill to point it at, and is
worth adding whenever the thing under test exists in more than one skill.

Write outputs to a temp directory, never next to the test. The suite this grew
out of scattered `runs/`, `accuracy.csv` and a `sandbox/` tree beside its own
source, and by the time anyone looked it was no longer obvious which of those
were fixtures and which were leftovers. `lib.sh`'s `sandbox_dir` and
`tempfile.mkdtemp` handle this; `$EVALCK_SANDBOX` overrides the shell one when
you want to inspect what a run left behind.
