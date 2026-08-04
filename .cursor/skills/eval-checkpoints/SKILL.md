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

## If you are an agent asked to run this

Do not guess the inputs. Two of them cannot be inferred from the repository and
one of them costs money to get wrong, so **ask the user before running anything.**

Ask for these two, which have no defaults:

1. **Where the checkpoints are.** An S3 prefix whose immediate children are the
   checkpoint directories, for example
   `s3://bucket/teams/<team>/runs/run_<uuid>/checkpoints`. If the user does not
   know, point them at [CHECKPOINTS.md](CHECKPOINTS.md) — it explains how to
   recover the path from their experiment tracker. Do not attempt to construct it
   from a template; the run id is random and the team is easy to get wrong.
2. **Where results should go.** An `s3://` prefix they can write to. On a managed
   platform this is often constrained to the job's own output directory, so ask
   rather than assuming any bucket will accept writes.

Then confirm these, stating the default so the user can accept it:

3. **Which benchmarks.** Default is the registry's `default` group. Offer
   [BENCHMARKS.md](BENCHMARKS.md) if they want to choose.
4. **How many checkpoints.** Default is all of them. Recommend `--latest 1` for a
   first run, since a full sweep multiplies cost by the number of checkpoints.
5. **Whether the box is already set up.** If this is a fresh GPU box, add
   `--bootstrap`. If they have run the skill here before, skip it.

Then follow this order, which exists so a mistake is cheap:

- Run with `--dry-run` first, **always**. It needs only S3 read, spends nothing,
  and validates the checkpoint path, the credentials, the `aws` CLI, the resolved
  benchmark list and the cost estimate in one shot.
- Show the user the dry-run output — particularly the discovered checkpoints and
  the prompt-count estimate — and get confirmation before running for real.
- Only then drop `--dry-run`.

If the dry-run finds no checkpoints, the path is almost certainly at the wrong
level. See the checkpoint-layout section of [CHECKPOINTS.md](CHECKPOINTS.md)
before retrying.

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

With neither `--group` nor `--benchmarks`, the registry's `default` group runs.
Pass `--group NAME` for a named set, or `--benchmarks "a b c"` for an explicit
list. To pilot on the most recent checkpoint before committing to a sweep, use
`--latest 1`.

## Inputs

Checkpoints, one of (if both are given, `--checkpoints` wins):

| Flag | Meaning |
|---|---|
| `--checkpoint-root s3://.../EXP` | discover every immediate child prefix as a checkpoint |
| `--checkpoints "s3://a s3://b"` | explicit space-separated list; local paths also work |

Checkpoints are ordered by the trailing integer in their name, so `step9` sorts
before `step10`, and names with no trailing number sort last.

**If you do not know the S3 path**, see [CHECKPOINTS.md](CHECKPOINTS.md). Training
run output paths contain a random run id and cannot be derived; that file covers
how to recover the exact prefix, how to tell two similar runs apart, and how to
confirm a path for free before spending GPU time.

| Flag | Default | Meaning |
|---|---|---|
| `--s3-out` | required | `s3://bucket/prefix` root for results |
| `--group NAME` | the registry's `default` group | run a named set of benchmarks |
| `--benchmarks` | (see `--group`) | space-separated olmo-eval task names; mutually exclusive with `--group` |
| `--pattern` | (none) | regex filter on the checkpoint directory name |
| `--latest N` | (all) | keep only the N highest-step checkpoints |
| `--limit N` | (none) | cap instances per task; smoke tests only, see BENCHMARKS.md |
| `--tp` | `1` | vLLM tensor-parallel size |
| `--gpu-memory-utilization` | vLLM default (~0.9) | fraction of VRAM vLLM may claim; lower it to share a GPU |
| `--tokenizer` | (from config) | HF tokenizer id, if the checkpoint config does not resolve one |
| `--run-id-prefix` | (none) | prefix for result subdirectories |
| `--allow-any-task` | off | permit task names outside the registry, without a cost estimate |
| `--bootstrap` | off | install the environment before sweeping, for a bare box |
| `--keep-local` | off | keep the local work tree for debugging |
| `--dry-run` | off | print the plan and cost estimate, then exit |

Benchmark names and group names are both validated against
`scripts/benchmarks.json` before anything is downloaded. An unrecognized name is
rejected with the list of valid ones, and common dataset-name confusions are
called out specifically. olmo-eval would also reject an unknown task, but only
after the checkpoint had been fetched and possibly converted, so catching it here
saves real time. Which groups exist and what each covers is in
[BENCHMARKS.md](BENCHMARKS.md); the resolved list and its source are logged and
recorded in each checkpoint's `run_provenance.json` as `benchmark_selection`, so a
result set carries a record of how its benchmark list was chosen.

### Checkpoint format, and why pre-converting is easier

A checkpoint may be either **HF format** (`config.json` + `*.safetensors`) or a
**native OLMo-core directory** (`config.json` + `model_and_optim/`, or a
`.metadata` file). The script detects which and only converts when it has to.

vLLM can only load HF-format weights, so native checkpoints must be converted
first. **The skill converts them itself, with no external setup**, using the
bundled `scripts/convert_to_hf.py`. That script calls the installed `olmo_core`
library rather than OLMo-core's example script, which matters because
`src/examples/` is not packaged into the wheel — reaching it would mean cloning a
private repository onto the eval box.

Set `OLMO_CORE_CONVERT` only to override the bundled converter, for instance to
point at OLMo-core's own `src/examples/huggingface/convert_checkpoint_to_hf.py`.
Both take the same flags, so the sweep does not care which it gets. The
`no_converter` failure only occurs if the bundled script is missing and the
override points nowhere.

Converting is CPU-only for standard dense architectures and needs roughly the
model size in RAM at bf16. MoE and fused-attention checkpoints force conversion
onto a GPU and are not covered.

**Pre-converting outside the skill is still worth it if the same checkpoints will
be evaluated repeatedly**, since conversion then happens once instead of per
sweep, and the `olmo_core` extra stops being needed at all:

```bash
python .cursor/skills/eval-checkpoints/scripts/convert_to_hf.py \
  -i "${NATIVE_CKPT}" -o "${HF_CKPT}" --skip-validation
```

Add `-t <hf-tokenizer-id>` if the checkpoint's config does not name a tokenizer.
Point `--checkpoint-root` at the HF copies and the sweep skips conversion
entirely.

### Environment: no container image required

This skill runs on any Linux box with a GPU. It does **not** need a pre-built
platform image, and it does not need the environment provisioned in advance. On a
bare box, bootstrap once:

```bash
bash .cursor/skills/eval-checkpoints/scripts/bootstrap.sh
export PATH="${HOME}/.local/bin:${PATH}"   # only if uv was just installed
```

That installs `uv` if absent, syncs the `vllm`, `hf`, `s3` and `olmo_core` extras,
and picks an `HF_HOME` on the largest writable mount it finds. It is safe to
re-run. `--bootstrap` on the sweep does the same thing inline, which is convenient
for a one-shot run but wasteful on repeats.

The sweep then preflights `uv`, `aws`, `python3` and an importable `olmo_eval`
before touching S3, so a missing dependency fails in seconds rather than after a
multi-gigabyte download.

Still required, and not installable by the bootstrap:

- **The olmo-eval fork this skill ships inside** — not a fresh clone of AI2's
  upstream. The script locates itself, walks up out of `.cursor/skills/`, and runs
  `uv run olmo-eval` from that repo root, so by default it uses the very checkout
  the skill file is in and you do not have to point it anywhere.

  This matters: the fork carries task and scorer code that upstream does not, and
  some registry entries depend on it. Running against pristine upstream would fail
  at import. Set `OLMO_EVAL_ROOT` only if the skill directory has been copied
  somewhere the relative walk no longer lands on the repo root.
- **The `aws` CLI**, which is not a Python dependency. The sweep shells out to it
  for every S3 operation.
- **AWS credentials** with read on the checkpoint prefix and `s3:PutObject` on
  `--s3-out`. `AWS_REGION` defaults to `us-east-1`.
- **One GPU.** About 24 GB fits models up to roughly 8B at bf16; use `--tp N` for
  larger, or `--gpu-memory-utilization` to leave room for another process.
- **Hub access** and an `HF_HOME` with space, since benchmark datasets download on
  first use.

One caveat on the bootstrap: the `olmo_core` extra pins
`ai2-olmo-core==2.4.0`, which conflicts with the `openhands` dependency, which is
why it is not in the default sync. If resolution fails, re-run
`bootstrap.sh --no-olmo-core` and either pre-convert your checkpoints or point
`$OLMO_CORE_CONVERT` at a converter in its own environment. `olmo_core` is only
needed to convert native checkpoints; already-HF checkpoints do not need it.

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
| `scripts/bootstrap.sh` | installs the environment on a bare GPU box |
| `scripts/convert_to_hf.py` | native OLMo-core to HF, library-only, no repo clone |
| `scripts/summarize_accuracy.py` | reads each `metrics.json`, writes the two CSVs |
| `scripts/benchmarks.json` | the benchmark registry; the only place names live |
| `BENCHMARKS.md` | what each benchmark scores, costs and reports |
| `CHECKPOINTS.md` | how to find the S3 prefix to evaluate |

## Additional resources

- Adaptive testing (far fewer items per checkpoint) is on hold. The rationale,
  the measured limits, and what it would take to revisit:
  `AdaptiveTesting/docs/05_cat_deferred.md`