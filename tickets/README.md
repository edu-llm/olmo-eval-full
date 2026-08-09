# `tickets/` — one branch, one submission, one record

A **ticket** is one checkpoint's evaluation: the branch it ran from, the harness changes
that checkpoint needed, the spec that was submitted, and what came back. Branch and
directory share a name, `<submitter>-<model>-ticket`, and the directory is the record the
branch leaves behind.

```
tickets/<submitter>-<model>-ticket/
  RUNS.md                       one appended line per run, newest last
  <run_id>/spec.yaml            frozen copy of exactly what was submitted
  <run_id>/cat_report.json      what came back
```

## Why a branch per submission

The harness does not travel with a submission. The container clones this repository and
checks out one sha, so the code that runs is whatever was pushed — which means a
submission is only reproducible if the commit it ran is still identifiable afterwards. A
shared branch cannot supply that: by the time anyone reads the report, the branch has
moved.

It also gives a checkpoint somewhere to be accommodated. A tokenizer the config names
wrongly, a dtype that never reached the loader, a context clamp, a pip-installable
dependency — these are the checkpoint's business, and a ticket is where they live without
reaching anybody else's run. The container installs from the ticket's own
`pyproject.toml`, so a dependency is as much a ticket's business as a code change.

**What a ticket may not change is the calibration.** A bank's `items.jsonl` or its IRT
parameters are what a theta means: an ability estimate is a position on a scale that a
calibration population fixed. Edit those and the run returns a well-formed report with a
plausible theta and a healthy standard error that is comparable to nothing — not to
another checkpoint, and not to the same bank yesterday. If a bank looks wrong, that is a
finding for the harness branch and a refit, never an edit on a ticket.

## The branch is deliberately not under `edullm/`

`edullm/**` is the only prefix that fires
[`edullm-platform-build.yml`](../.github/workflows/edullm-platform-build.yml). A CAT run
takes its image from OLMo-core, so a ticket named that way would spend several gigabytes
publishing an immutable ECR tag nothing ever pulls. Named as above it fires nothing:
[`ci.yml`](../.github/workflows/ci.yml) is `main`-only.

## The order matters, and it is not the obvious one

A spec cannot pin the commit that contains it — committing the spec moves `HEAD` past the
sha it names. The way out is that **the spec is a record, not an input**: `--spec` is read
off the laptop and compiled into the submission, so it need not be on the branch when you
submit, only afterwards.

```
1. commit the harness change
2. push
3. repin the spec to HEAD          # uncommitted; check_submission_pin --repin writes it
4. submit
5. commit the spec and the run id  # now the record names the commit that actually ran
```

Step 3 is why [`check_submission_pin.py`](../diagnostics/mcq_cat/styles/uni_mcq/scripts/check_submission_pin.py)
exempts `.edullm/` from its clean-tree check: a spec is never cloned, so its edits cannot
reach the container, and leaving it dirty at submit time is what makes step 5 possible.

## Iterating, and why nothing is ever amended

A probe fails, the harness is amended, the run is resubmitted. Each attempt appends; none
rewrites. **Never amend or force-push a ticket branch** — a force-push orphans the shas
earlier specs pinned, and re-running one of those specs then dies at checkout with
`fatal: git upload-pack: not our ref`.

Freeze the spec per run rather than relying on `.edullm/<name>.yaml`, which is edited
between attempts and would otherwise retain only the last. Probe failures get a `RUNS.md`
line and no directory; there is no report to keep.

## `RUNS.md`, and what the report does not tell you

One line per run:

| run id | harness sha | OLMo-core sha | benchmark | outcome |
| --- | --- | --- | --- | --- |

The first three are there because **`cat_report.json` carries none of them**. Its `run`
block holds `cat_style`, `checkpoint`, `checkpoint_kind`, `checkpoint_prep`, `dtype`,
`ability_estimator`, `modality`, `grader`, `timestamp` and `tokenization` — nothing that
says which submission produced it or which code ran. Committing the report is therefore
not enough to make the branch a record; the line is what makes it one.
