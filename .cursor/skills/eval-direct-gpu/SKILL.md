---
name: eval-direct-gpu
description: >-
  Evaluate a sweep of training checkpoints from S3 on full benchmarks and report
  accuracy per checkpoint per benchmark, running on a GPU machine you control. Use
  when you have a Linux CUDA box available to you -- your own EC2 instance or
  similar -- and want to sweep many checkpoints under a prefix, writing results to
  an S3 path you choose. Reads native OLMo-core checkpoints directly by default, so
  no conversion step is needed; vLLM remains available for HF-format checkpoints. If
  instead the work has to be submitted as an AWS Batch job through the eduLLM
  platform, because you have no GPU of your own, use the eval-platform skill. The set
  of available benchmarks is data-driven and documented in BENCHMARKS.md.
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

**This skill is not standalone.** The script shells out to `uv run olmo-eval`
from the repository it ships inside, so a copy of this folder on its own cannot
work. If you were pointed at a link to the skill rather than the repo, clone the
whole olmo-eval fork and run the script from that checkout — it finds the repo
root by walking up out of `.cursor/skills/`. Set `OLMO_EVAL_ROOT` only if the
folder was moved somewhere that walk no longer lands correctly.

Do not guess the inputs. Two of them cannot be inferred from the repository and
one of them costs money to get wrong, so **ask the user before running anything.**

Ask for these two, which have no defaults:

1. **Where the checkpoints are.** An S3 prefix whose immediate children are the
   checkpoint directories, or one such directory for `--checkpoint`.

   **This is the user's to provide, and yours only to check.** If they do not
   give it, ask; if what they give does not work, ask again with what you saw. Do
   not construct a path from a template, substitute a likely bucket or team, or
   search an experiment tracker, config file or bucket listing for something
   plausible. The run id is random, two runs differ by a few characters, and a
   wrong-but-valid path produces confident numbers for the wrong model that
   nothing downstream will catch. [CHECKPOINTS.md](CHECKPOINTS.md) covers how to
   verify the path they hand you.
2. **Where results should go.** An `s3://` prefix they can write to. On a managed
   platform this is often constrained to the job's own output directory, so ask
   rather than assuming any bucket will accept writes.

Then confirm these, stating the default so the user can accept it:

3. **Which benchmarks.** Default is the registry's `default` group. Offer
   [BENCHMARKS.md](BENCHMARKS.md) if they want to choose.

   If the user names benchmarks explicitly, **check them against
   [BENCHMARKS.md](BENCHMARKS.md) before running anything and tell the user
   about any that cannot run.** Three commonly requested ones — SimpleQA, T-REx
   and FactScore — produce no score today. The script refuses them outright, but
   raising it in your first reply is much better than surfacing it after the user
   has waited. TriviaQA and PopQA are *not* among them: both are registered and
   run, and sit in the `factual` group. **Do not reach for
   `--allow-any-task`** to force them through: it bypasses this skill's registry,
   not olmo-eval's, so it cannot run a task that does not exist.

   For a smoke test over the user's own list rather than the whole registry, use
   `--benchmarks "a b c" --limit 20`. `--group smoke` always covers every
   benchmark, and combining it with `--benchmarks` is rejected. It is also not
   quite the same thing: smoke splits a prompt budget rather than applying one
   instance cap, so its benchmarks get different instance counts.
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
- If the skill has not been pointed at this training run before, do a real run of
  `--group smoke --latest 1` next. It evaluates seven benchmarks at 506 instances
  and 1,005 prompts in total, so it exercises the entire path — fetch, model load,
  both scoring paths, both batch sizes, upload, summary — in minutes, and surfaces a
  bad checkpoint or a wrong tokenizer before a full sweep spends hours discovering
  the same thing. **Its scores are meaningless. Never report them.** A thousand
  prompts is a better plumbing check than a hundred and is no more a measurement.
- Only then drop `--dry-run` and `--group smoke`.

If the dry-run finds no checkpoints, the path is almost certainly at the wrong
level. Show the user what you ran and what came back, and ask them for the
corrected prefix — see the checkpoint-layout section of
[CHECKPOINTS.md](CHECKPOINTS.md) for the two levels people usually land on. Do
not probe neighbouring prefixes looking for one that lists.

## Quick start

```bash
bash .cursor/skills/eval-direct-gpu/scripts/run_eval_sweep.sh \
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

`--group smoke` runs seven benchmarks sharing a budget of 1,000 prompts — 506
instances and 1,005 prompts in total — covering everything except the
`fact_proxy` pair, `naturalqs` and `jeopardy`, which add download time without
exercising a path `popqa` and `triviaqa` do not already cover. A budget rather
than one instance cap because prompts are instances times answer choices, so a
flat cap would give `csqa` five times `popqa`'s coverage; an even share is about
143 prompts each, which is 29 `csqa` instances and 143 `triviaqa` ones. It is the
cheapest way to prove the whole path works end to end, and a plumbing check
rather than a measurement: 29 instances is nowhere near enough to score anything,
and neither is 143. See the `--limit` trap in [BENCHMARKS.md](BENCHMARKS.md) for
why its numbers cannot be reported.

## Inputs

Checkpoints, one of (if both are given, `--checkpoint` wins):

| Flag | Meaning |
|---|---|
| `--checkpoint-root s3://.../EXP` | discover every immediate child prefix as a checkpoint; a local directory works too |
| `--checkpoint s3://.../step1000` | exactly one checkpoint; a local path works too |

`--checkpoint` is singular by design: a space-separated list is rejected with a
pointer to `--checkpoint-root`, rather than being silently treated as one very
strange path. Sweeping several checkpoints is what `--checkpoint-root` is for.

Checkpoints are ordered by the trailing integer in their name, so `step9` sorts
before `step10`, and names with no trailing number sort last.

**If you do not have the S3 path, ask for it.** Training run output paths contain
a random run id and cannot be derived, so there is nothing to work it out from.
[CHECKPOINTS.md](CHECKPOINTS.md) covers why, what to ask for, and how to confirm
a supplied path for free before spending GPU time.

| Flag | Default | Meaning |
|---|---|---|
| `--s3-out` | required | `s3://bucket/prefix` root for results |
| `--group NAME` | the registry's `default` group | run a named set of benchmarks; `smoke` is the 1,000-prompt plumbing check over seven of them |
| `--benchmarks` | (see `--group`) | space-separated olmo-eval task names; mutually exclusive with `--group` |
| `--pattern` | (none) | regex filter on the checkpoint directory name |
| `--latest N` | (all) | keep only the N highest-step checkpoints |
| `--limit N` | (none, or the group's own) | cap instances per task; overrides a group's own limit or prompt budget, uniformly. Smoke tests only, see BENCHMARKS.md |
| `--provider` | `olmo_core` | inference backend: `olmo_core` reads native checkpoints directly, `vllm_server` needs HF format and converts first |
| `--batch-size-mcq` | `512` | requests per forward pass for multiple-choice benchmarks (`olmo_core` only) |
| `--batch-size-gen` | `192` | requests per forward pass for generative benchmarks (`olmo_core` only) |
| `--tp` | `1` | vLLM tensor-parallel size (`vllm_server` only; refused on the native path) |
| `--gpu-memory-utilization` | vLLM default (~0.9) | fraction of VRAM vLLM may claim; lower it to share a GPU (`vllm_server` only) |
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

### Choosing a provider

Two inference backends, and the choice decides whether a conversion step exists
at all.

**`olmo_core` (default) reads a native OLMo-core checkpoint as it stands.** Nothing
is converted, which is the reason it is the default: conversion is
architecture-gated, and `get_hf_config` refuses anything that is not standard dense
OLMo-2/OLMo-3 — a memory-split model, for instance — no matter which OLMo-core
version is installed. A checkpoint the vLLM path cannot accept at all is still
scored here.

**`vllm_server` is faster per prompt but only loads HF-format weights.** Prefer it
when your checkpoints are already HF format, or when a conversion is known to
succeed and the same checkpoints will be swept repeatedly. It is also the path with
tensor parallelism, via `--tp`.

Pick with `--provider`. The vLLM-only flags (`--tp`,
`--gpu-memory-utilization`) are **refused** rather than ignored on the native path,
because `ProviderConfig.from_dict` drops keys it does not recognise, so a silently
accepted one would look honoured and do nothing. `OlmoCoreProvider` requires
`tensor_parallel_size` of 1; run several single-GPU sweeps rather than one
tensor-parallel sweep.

### Batching on the native path

vLLM schedules its own batches continuously, so it takes no batch size. The native
provider does, and leaving it unset is a trap worth knowing about:
`_iter_chunks` treats `batch_size=None` as **one chunk holding every request**,
which for a full sweep means an immediate out-of-memory. The sweep therefore always
sets it explicitly.

It sets two different values, because one number cannot serve both kinds of work:

- **Multiple choice** is a single forward pass per prompt with no KV-cache growth,
  so a large batch is straightforwardly better. Default `--batch-size-mcq 512`.
- **Generation** pads every sequence in a batch out to the longest one, so past a
  point the padding costs more than the parallelism buys. Default
  `--batch-size-gen 192`.

`batch_size` is a provider setting bound to the harness, so one invocation carries
one value. The sweep consequently splits each checkpoint into **two invocations**,
one per kind, and pays a second model load to do it. Where a run is entirely one
kind, only one invocation is issued and there is no second load.

Both defaults suit a ~1B model on a 24GB card. Scale them down for a larger model
or a smaller card: roughly halve both each time parameter count doubles, and drop
them if you see an out-of-memory in the logs. They are a throughput knob only —
no metric depends on them, so a conservative value costs time and nothing else.

### Checkpoint format, and pre-converting for the vLLM path

A checkpoint may be either **HF format** (`config.json` + `*.safetensors`) or a
**native OLMo-core directory** (`config.json` + `model_and_optim/`, or a
`.metadata` file). The script detects which, and converts only on the vLLM path
and only when it has to. **On the default native path nothing below applies.**

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
python .cursor/skills/eval-direct-gpu/scripts/convert_to_hf.py \
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
bash .cursor/skills/eval-direct-gpu/scripts/bootstrap.sh
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
    metrics.json                     # every task, both halves merged  <- read this
    mcq/                             # native path only: the multiple-choice half
      metrics.json                   #   that half alone
      predictions/..., requests/...  #   per-instance predictions and requests
    generative/                      # native path only: the generative half
      metrics.json, predictions/, requests/
    logs/                            # provider log
    run_provenance.json              # checkpoint, git sha, benchmarks, status
    _READY | _FAILED                 # terminal marker, always written
    _IN_PROGRESS                     # heartbeat while evaluating; gone once terminal
  accuracy.csv                       # long: one row per checkpoint/benchmark/metric
  accuracy_wide.csv                  # wide: one row per checkpoint  <- plot this
  accuracy.json
  sweep.log
```

On the native path the two invocations each write their own `metrics.json`, and
`write_metrics_json` opens with `"w"`, so the sweep merges them into the run-root
`metrics.json` rather than letting the second overwrite the first. **Read the
run-root file**; the per-half copies are kept only so a crash between the halves
leaves the finished one legible. The merge runs after *each* half, so a run that
scores the multiple-choice benchmarks and then dies still leaves a usable
`metrics.json` covering them.

On the `vllm_server` path there is one invocation, so there are no subdirectories
and `predictions/` and `requests/` sit directly under the run directory.

`accuracy_wide.csv` is the deliverable: one row per checkpoint, sorted by step,
one column per score. Column names are qualified only as far as they need to be:
a benchmark reporting a single score gets a bare name, one reporting several
metrics gets `<benchmark>.<metric>`, and a metric carrying more than one scorer
gets `<benchmark>.<metric>.<scorer>`. See BENCHMARKS.md for which are which and
how to read them.

`accuracy.csv` is the same data in long form, with the scorer behind each number,
an `is_primary` flag, and `num_instances`.

Every checkpoint gets a terminal marker. One checkpoint failing does not abort
the sweep: it is marked `_FAILED`, keeps a placeholder row in both CSVs so it
cannot silently vanish from the curve, and the run moves on. Pre-eval failures
record a reason (`fetch_failed`, `no_converter`, `conversion_failed`) so a missing
checkpoint is always explainable. The sweep exits non-zero if any checkpoint
failed. Poll for either marker — `_READY` alone is ambiguous, since its absence
cannot distinguish a failure from a run still in progress.

### What a dying box leaves

A checkpoint's directory is synced up every `--sync-interval` seconds (default 180)
while it is still evaluating, not only when it finishes, so a box lost partway keeps the
instances it had already scored. `--sync-interval 0` restores sync-only-at-the-end.

Both terminal markers are written by this script, so a box that dies — out of memory, a
spot reclaim, a dropped session — writes neither. `_IN_PROGRESS` carries the time of the
last sync, so a stale one dates the loss and marks that checkpoint's directory as holding
everything that survived rather than everything there was.

A benchmark still running when the box died leaves
`<name>-predictions.partial.jsonl` instead of its ordinary predictions file, and no
`metrics.json` entry: a mean over whichever instances the queue reached is not that
benchmark's accuracy. **Its final line is usually cut off**, because the sync copies the
file while it is still being appended to. That is expected, not corruption. The
`eval-platform` skill ships `scripts/salvage_partial.py`, which reads these files, drops
a truncated tail with a note, and reports what each one scored.

## Verify locally (no GPU, no AWS)

```bash
# Preview a sweep, including the cost estimate
bash .cursor/skills/eval-direct-gpu/scripts/run_eval_sweep.sh \
  --checkpoint-root s3://b/ck \
  --s3-out s3://b/evals --dry-run

# Price the smoke group against one checkpoint
bash .cursor/skills/eval-direct-gpu/scripts/run_eval_sweep.sh \
  --checkpoint s3://b/ck/step1000 --s3-out s3://b/evals \
  --group smoke --dry-run

# Confirm an invalid benchmark name is caught before anything spends
bash .cursor/skills/eval-direct-gpu/scripts/run_eval_sweep.sh \
  --checkpoint s3://b/ck/step1 --s3-out s3://b/evals \
  --benchmarks "not_a_real_task" --dry-run
```

## How it maps to olmo-eval

This skill owns no evaluation logic. It discovers checkpoints, stages weights,
shells out to olmo-eval once per checkpoint, and parses the JSON that comes back.
Everything between the invocation and `metrics.json` is upstream code.

`-t` accepts several benchmarks while `-m` is single-model, which is why
checkpoints are the loop. On `vllm_server` a checkpoint is one `olmo-eval run` with
one `-t` per benchmark, sharing a single boot. On `olmo_core` it is two, split by
kind, because `batch_size` is a harness-scoped provider setting and one invocation
can carry only one value.

Argument order is not cosmetic. `run` has no top-level `--provider`; it goes
through `--harness`. Each `-o` binds to the *preceding* `--harness` or `-t`, and
the two accept disjoint key sets, so a `provider.*` key after `-t` is a usage
error and a `limit=` before the first `-t` is too. Both the batch size and tensor
parallelism go through `provider.kwargs.*` in the harness group:
`ProviderConfig` has no such fields and `ProviderConfig.from_dict` drops keys it
does not know, so the shorter `provider.batch_size=N` would be accepted and then
silently ignored — leaving `batch_size` at `None`, which `_iter_chunks` reads as one
chunk holding everything.

Checkpoints are synced from S3 to local disk first. olmo-eval has no S3 download on
either provider, so passing an `s3://` path straight to `-m` fails inside the model
loader rather than at argument parsing.

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
| `CHECKPOINTS.md` | what to ask the user for, and how to verify the prefix they give |

## Additional resources

- Adaptive testing (far fewer items per checkpoint) is on hold. The rationale,
  the measured limits, and what it would take to revisit:
  `AdaptiveTesting/docs/05_cat_deferred.md`