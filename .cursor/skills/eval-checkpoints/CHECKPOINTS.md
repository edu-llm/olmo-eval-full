# The checkpoint path

**The checkpoint path is an input the user supplies. Ask for it. Do not go
looking for it.**

This file explains why it cannot be worked out from anything in the repository,
and what to check about a path once you have been given one.

## Why it has to come from the user

A training run's output path contains a randomly generated run id, and nothing
outside the training platform records which run produced which model. Two runs of
the same experiment that differ only in configuration are indistinguishable from
their paths: same bucket, same team, same shape, different id. The mapping from
run id back to *what was trained* lives only with whoever launched the run.

A plausible-looking guess is worse than no path at all. It either fails partway
through a sweep that has already started spending, or it succeeds against the
wrong model and produces a clean table of numbers that describe something nobody
intended to measure. Nothing downstream can detect that second case.

So if you do not have the exact prefix, **stop and ask.** Do not assemble one
from a template, do not substitute a team or bucket name that seems likely, and
do not go hunting through experiment trackers, config files or bucket listings
for something that looks close enough. Hand the question back to the user.

If the user does not have it to hand, the thing to ask them for is the value the
training run exposed as `EDULLM_CHECKPOINT_DIR`, copied verbatim. Asking them to
copy that one string is reliable; anything reconstructed from parts is not.

## What to ask for

A single S3 prefix whose **immediate children are the individual checkpoint
directories**:

```
s3://<bucket>/.../<run>/checkpoints/
                        |- step500/
                        |- step1000/
                        |- step2000/
```

`--checkpoint-root` wants that `checkpoints/` level. `--checkpoint` wants one
step directory from inside it.

The two neighbouring levels are the common mix-ups, and neither fails cleanly:

- **The run root**, whose children are `checkpoints/`, `logs/` and similar. The
  sweep finds prefixes that are not checkpoints and tries to evaluate them.
- **A single step directory**, whose children are the weight files. The sweep
  finds nothing to enumerate and reports no checkpoints.

`AWS_REGION` defaults to `us-east-1`; override it if the bucket is elsewhere.

## Checking a path you were given

`--dry-run` confirms the path for free. It needs only S3 **read**, exits before
any download, conversion, GPU work or upload, and validates four things at once:
the path exists, your credentials can read it, the `aws` CLI is present, and the
discovered checkpoints are the ones the user expected.

```bash
bash .cursor/skills/eval-checkpoints/scripts/run_eval_sweep.sh \
  --checkpoint-root s3://.../<run>/checkpoints \
  --s3-out s3://.../evals/scratch \
  --dry-run
```

**Show the discovered checkpoint list to the user and let them confirm it is the
right run** before dropping `--dry-run`. That confirmation is the only real check
that exists: two paths differing by one character in a run id look identical at a
glance, and `accuracy_wide.csv` would report two columns of near-identical
numbers without anything appearing wrong. When evaluating more than one run, give
each sweep a `--run-id-prefix` the user recognizes, so the results stay
attributable afterwards.

### What a checkpoint directory should contain

Each child prefix should hold `config.json` plus **either**:

- `model_and_optim/` (or a `.metadata` file) — native OLMo-core format, which the
  sweep converts before evaluating, or
- `*.safetensors` — already HF format, used directly.

A listing with neither means the path is at the wrong level. Ask the user for the
corrected one rather than trying adjacent prefixes.

## Ordering, and picking a subset

Checkpoints are ordered by the trailing integer in the directory name, so
`step9` sorts before `step10` rather than after it, and names with no trailing
number sort last. That ordering drives both `--latest` and the row order in
`accuracy_wide.csv`, so the output plots against training progress directly.

- `--latest 1` evaluates only the final checkpoint. Use it to pilot: it exercises
  the entire pipeline at the cost of one checkpoint.
- `--latest N` keeps the N highest-step checkpoints.
- `--pattern` filters on the directory name with a regex, for example
  `--pattern 'step[0-9]*000$'` to take only thousands.

Prefer `--latest` over `--limit` for shrinking a trial run. `--limit` caps
instances per benchmark and, on some tasks, changes which split is loaded — see
[BENCHMARKS.md](BENCHMARKS.md).

## Permissions

The box running the sweep needs:

- **read** on the checkpoint prefix, to list and download weights;
- **`s3:PutObject`** on the `--s3-out` prefix, to upload results and markers.

Those are often granted by different roles. On a managed platform, a job's role
is usually scoped to write only inside that job's own output directory, so
`--s3-out` may not be freely chosen — ask the user what their platform exposes as
the run's output root rather than picking a bucket. Running the eval under the
same team that owns the checkpoints keeps both permissions inside one role and
avoids cross-team access entirely.
