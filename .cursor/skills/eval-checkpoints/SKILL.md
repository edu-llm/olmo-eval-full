---
name: eval-checkpoints
description: >-
  Evaluate a sweep of training checkpoints from S3 on full benchmarks and report
  accuracy per checkpoint per benchmark. Use when a training team wants to measure
  model ability across checkpoints of a run, plot accuracy against training step,
  or score checkpoints stored on S3. The set of available benchmarks is data-driven
  and documented in BENCHMARKS.md.
---

# Evaluate a sweep of training checkpoints

Point this at a prefix of checkpoints from a training run and a list of
benchmarks. It discovers the checkpoints, evaluates every benchmark on each one,
and writes an accuracy table you can plot as a training curve.

Every benchmark runs its **complete evaluation split**. There is no sampling and
no adaptive shortcut, so the numbers are comparable to published figures.

**Which benchmarks exist, what they score, what they cost, and what they report
are all in [BENCHMARKS.md](BENCHMARKS.md).** This file covers only the mechanics.

## Quick start

```bash
bash .cursor/skills/eval-checkpoints/scripts/run_eval_sweep.sh \
  --checkpoint-root s3://YOUR_BUCKET/checkpoints/EXP \
  --s3-out s3://YOUR_BUCKET/evals/EXP \
  --dry-run
```

**Always `--dry-run` first.** It prints the discovered checkpoints, the resolved
benchmark list, and a cost estimate without spending anything. Drop the flag to
execute.

Omitting `--benchmarks` runs everything in the registry. To narrow it, pass a
space-separated subset. To pilot on the most recent checkpoint before committing
to a sweep, use `--latest 1`.

## Inputs

Checkpoints, one of (if both are given, `--checkpoints` wins):

| Flag | Meaning |
|---|---|
| `--checkpoint-root s3://.../EXP` | discover every immediate child prefix as a checkpoint |
| `--checkpoints "s3://a s3://b"` | explicit space-separated list; local paths also work |

Each checkpoint may be a native OLMo-core directory (`config.json` +
`model_and_optim/`) or already in HF format. Checkpoints are ordered by the
trailing integer in their name, so `step9` sorts before `step10`, and names with
no trailing number sort last.

| Flag | Default | Meaning |
|---|---|---|
| `--s3-out` | required | `s3://bucket/prefix` root for results |
| `--benchmarks` | every registry entry | space-separated olmo-eval task names |
| `--pattern` | (none) | regex filter on the checkpoint directory name |
| `--latest N` | (all) | keep only the N highest-step checkpoints |
| `--limit N` | (none) | cap instances per task; smoke tests only, see BENCHMARKS.md |
| `--tp` | `1` | vLLM tensor-parallel size |
| `--tokenizer` | (from config) | HF tokenizer id, if the checkpoint config does not resolve one |
| `--run-id-prefix` | (none) | prefix for result subdirectories |
| `--allow-any-task` | off | permit task names outside the registry, without a cost estimate |
| `--keep-local` | off | keep the local work tree for debugging |
| `--dry-run` | off | print the plan and cost estimate, then exit |

Benchmark names are validated against `scripts/benchmarks.json` before anything
is downloaded. An unrecognized name is rejected with the list of valid ones, and
common dataset-name confusions are called out specifically. olmo-eval would also
reject an unknown task, but only after the checkpoint had been fetched and
possibly converted, so catching it here saves real time.

### Environment

- Run from a checkout of this repo. The script shells out to `uv run olmo-eval`,
  which resolves the project from the working directory. Override the checkout
  with `OLMO_EVAL_ROOT`.
- `uv sync --extra vllm --extra hf`, and one GPU. About 24 GB fits models up to
  roughly 8B at bf16; use `--tp N` for larger.
- **`OLMO_CORE_CONVERT` is required if any checkpoint is native OLMo-core
  format.** It must point at OLMo-core's `convert_checkpoint_to_hf.py`. vLLM only
  loads HF-format weights, so native checkpoints have to be converted first.
  Without this variable set, those checkpoints are skipped and marked `_FAILED`.
  Already-HF checkpoints do not need it.
- AWS credentials with read on the checkpoint prefix and `s3:PutObject` on
  `--s3-out`. `AWS_REGION` defaults to `us-east-1`.
- Benchmark datasets download from HuggingFace on first use; the box needs Hub
  access and an `HF_HOME` with room.

Nothing is uploaded to the HuggingFace Hub. "HF" refers to the on-disk format.

## Outputs

```
<s3-out>/
  <run-id>/                          # one per checkpoint
    metrics.json                     # olmo-eval's standard output, one entry per task
    predictions/..., requests/...    # per-instance predictions and requests
    logs/                            # vLLM server log
    run_provenance.json              # checkpoint, git sha, benchmarks, status
    _READY | _FAILED                 # terminal marker, always written
  accuracy.csv                       # long: one row per checkpoint/benchmark/metric
  accuracy_wide.csv                  # wide: one row per checkpoint  <- plot this
  accuracy.json
  sweep.log
```

`accuracy_wide.csv` is the deliverable: one row per checkpoint, sorted by step,
one column per score. A benchmark reporting a single metric gets a bare column
name; one reporting several gets `<benchmark>.<metric>` columns so the units stay
distinguishable. See BENCHMARKS.md for which are which and how to read them.

`accuracy.csv` is the same data in long form, with the scorer behind each number,
an `is_primary` flag, and `num_instances`.

Every checkpoint gets a terminal marker. One checkpoint failing does not abort
the sweep: it is marked `_FAILED`, keeps a placeholder row in both CSVs so it
cannot silently vanish from the curve, and the run moves on. Pre-eval failures
record a reason (`fetch_failed`, `no_converter`, `conversion_failed`) so a missing
checkpoint is always explainable. The sweep exits non-zero if any checkpoint
failed. Poll for either marker — `_READY` alone is ambiguous, since its absence
cannot distinguish a failure from a run still in progress.

## Verify locally (no GPU, no AWS)

```bash
# Preview a sweep, including the cost estimate
bash .cursor/skills/eval-checkpoints/scripts/run_eval_sweep.sh \
  --checkpoints "s3://b/ck/step500 s3://b/ck/step1000" \
  --s3-out s3://b/evals --dry-run

# Confirm an invalid benchmark name is caught before anything spends
bash .cursor/skills/eval-checkpoints/scripts/run_eval_sweep.sh \
  --checkpoints s3://b/ck/step1 --s3-out s3://b/evals \
  --benchmarks "not_a_real_task" --dry-run
```

## How it maps to olmo-eval

This skill owns no evaluation logic. It discovers checkpoints, stages weights,
shells out to olmo-eval once per checkpoint, and parses the JSON that comes back.
Everything between the invocation and `metrics.json` is upstream code.

Per checkpoint the whole eval is a single `olmo-eval run` with one `-t` per
benchmark, so they share a single vLLM boot. `-t` accepts several values while
`-m` is single-model, which is why checkpoints are the loop.

Argument order is not cosmetic. `run` has no top-level `--provider`; it goes
through `--harness`. Each `-o` binds to the *preceding* `--harness` or `-t`, and
the two accept disjoint key sets, so a `provider.*` key after `-t` is a usage
error and a `limit=` before the first `-t` is too. Tensor parallelism is
`-o provider.kwargs.tensor_parallel_size=N` in the harness group: `ProviderConfig`
has no such field and `ProviderConfig.from_dict` drops keys it does not know, so
the shorter `provider.tensor_parallel_size=N` would be accepted and then silently
ignored.

Checkpoints are synced from S3 to local disk first. olmo-eval has no S3 download
on the vLLM path, so passing an `s3://` path straight to `-m` fails inside the
model loader rather than at argument parsing.

Scores are read from `tasks[].metrics` in `metrics.json`, not from the top-level
`summary` block — `summary` carries only each task's primary metric, which would
drop the second metric on any task that reports more than one.

## Files

| File | Role |
|---|---|
| `scripts/run_eval_sweep.sh` | the sweep: discover, stage, convert, eval, upload, mark |
| `scripts/summarize_accuracy.py` | reads each `metrics.json`, writes the two CSVs |
| `scripts/benchmarks.json` | the benchmark registry; the only place names live |
| `BENCHMARKS.md` | what each benchmark scores, costs and reports |

## Additional resources

- Adaptive testing (far fewer items per checkpoint) is on hold. The rationale,
  the measured limits, and what it would take to revisit:
  `AdaptiveTesting/docs/05_cat_deferred.md`