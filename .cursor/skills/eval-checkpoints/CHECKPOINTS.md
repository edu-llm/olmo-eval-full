# Finding the checkpoints to evaluate

`--checkpoint-root` needs an S3 prefix, and that prefix is not guessable. This
file explains where training writes checkpoints, how to recover the exact path,
and how to confirm you have the right one before spending GPU time.

## Why this file exists

A training run's output path contains a randomly generated run id, and nothing
outside the training platform records which run produced which model. Two runs of
the same experiment that differ only in configuration are indistinguishable from
their paths: same bucket, same team, same shape, different UUID. The mapping from
run id back to *what was trained* lives only in the experiment tracker.

So the path cannot be derived. It has to be read off the run that produced it.

## Path shape

```
s3://<bucket>/teams/<team>/runs/<run_id>/checkpoints/
```

For the eduLLM platform that is typically:

```
s3://sbsandbox-intern-edullm-outputs/teams/<team>/runs/run_<uuid>/checkpoints/
```

Region is `us-east-1`. The `run_id` looks like `run_019fca96-e03c-70c5-8a97-996b618329e3`.

`--checkpoint-root` wants the `checkpoints/` level -- the directory whose children
are the individual steps. The sweep lists immediate child prefixes and treats each
as one checkpoint.

## Recovering the exact path

Training exposes its save directory as the environment variable
**`EDULLM_CHECKPOINT_DIR`**. That string, copied verbatim, is the correct
`--checkpoint-root` for that run.

Copy it rather than reassembling it. Reconstruction means guessing both the team
and the UUID, and the team is the part people get wrong -- the path embeds
whichever team the run actually used, which is not always the one a runbook names.

### Where to find it in Weights & Biases

In rough order of reliability:

1. **Logs tab, search for `s3://`.** The training script almost certainly printed
   its save path. This works regardless of how the run was configured, which is
   why it is first.
2. **Overview then Config**, searched for `CHECKPOINT` or for the bucket name.
   Present when the training script logged its environment into `wandb.config`.
3. **Files then `wandb-metadata.json`**, which records the exact command line the
   run was launched with, including arguments.

If a run has several checkpoint-shaped paths, prefer the one ending in
`checkpoints/`; some scripts also log a separate directory for final artifacts.

### Matching runs to models

Note which W&B run you took each path from, and label the sweep accordingly with
`--run-id-prefix`. It is the only defense against silently comparing a model to
itself: two paths that differ by one UUID character look identical at a glance,
and `accuracy_wide.csv` would happily report two columns of near-identical numbers
without anything looking wrong.

## Confirming a path before spending

Listing the prefix should show step directories:

```bash
aws s3 ls s3://.../runs/run_<uuid>/checkpoints/
#                           PRE step1000/
#                           PRE step2000/
#                           PRE step500/
```

`--dry-run` does exactly this and needs only S3 **read** permission, since it
exits before any download, conversion, upload, or GPU work. It costs nothing and
confirms four things at once: the path exists, your credentials can read it, the
`aws` CLI is present, and the discovered checkpoints are the ones you expected.

```bash
bash .cursor/skills/eval-checkpoints/scripts/run_eval_sweep.sh \
  --checkpoint-root s3://.../runs/run_<uuid>/checkpoints \
  --s3-out s3://.../evals/scratch \
  --dry-run
```

### What a checkpoint directory should contain

Each child prefix should hold `config.json` plus **either**:

- `model_and_optim/` (or a `.metadata` file) -- native OLMo-core format, which the
  sweep converts before evaluating, or
- `*.safetensors` -- already HF format, used directly.

If a listing shows neither, the path is at the wrong level. Common mistakes are
pointing at the run root (whose children are `checkpoints/`, `logs/` and similar
rather than steps) or at a single step directory (whose children are the weight
files themselves). Both produce confusing results rather than clean errors: the
first finds prefixes that are not checkpoints, the second finds nothing to
enumerate.

## Ordering, and picking a subset

Checkpoints are ordered by the trailing integer in the directory name, so
`step9` sorts before `step10` rather than after it, and names with no trailing
number sort last. That ordering drives both `--latest` and the row order in
`accuracy_wide.csv`, so the output plots against training progress directly.

- `--latest 1` evaluates only the final checkpoint. Use this to pilot -- it
  exercises the entire pipeline at the cost of one checkpoint.
- `--latest N` keeps the N highest-step checkpoints.
- `--pattern` filters on the directory name with a regex, for example
  `--pattern 'step[0-9]*000$'` to take only thousands.

Prefer `--latest` over `--limit` for shrinking a trial run. `--limit` caps
instances per benchmark and, on some tasks, changes which split is loaded --
see [BENCHMARKS.md](BENCHMARKS.md).

## Permissions

The box running the sweep needs:

- **read** on the checkpoint prefix, to list and download weights;
- **`s3:PutObject`** on the `--s3-out` prefix, to upload results and markers.

Those are often granted by different roles. On a managed platform, a job's role
is usually scoped to write only inside that job's own output directory, so
`--s3-out` may not be freely chosen -- check what your platform exposes as the
run's output root. Running the eval under the same team that owns the checkpoints
keeps both permissions inside one role and avoids cross-team access entirely.

`AWS_REGION` defaults to `us-east-1` in the sweep; override it if the bucket lives
elsewhere.
