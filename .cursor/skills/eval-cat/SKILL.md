---
name: eval-cat
description: >-
  Submit an adaptive-testing (CAT) evaluation of one OLMo-core checkpoint to the eduLLM
  platform, and report the ability estimate it lands in S3. A CAT administers a few dozen
  items chosen to be maximally informative at the model's current ability instead of the
  benchmark's full split, so one bank costs minutes rather than hours. Use when the user
  asks to evaluate, score or diagnose a checkpoint on ARC, HellaSwag, MuSR, BBH, GPQA,
  IFEval or MATH, mentions CAT or adaptive testing, or asks how good a checkpoint is. This
  skill submits a platform job; it does not run anything on a machine you hold. The
  runnable benchmark set is data-driven and you query it, never recall it.
---

# Submit a CAT evaluation through the platform

The job runs on AWS Batch, allocated for you, so you need no AWS credential and no GPU.
In exchange the platform picks where results land (`$EDULLM_OUTPUT_PREFIX`, injected into
the container) and one submission scores **one benchmark per cell**.

Read [AGENTS.md](../../../AGENTS.md) first. It is the platform contract, it is distributed
from `edu-llm/platform` and reverts local edits, and this skill deliberately does not
restate its refusal codes, its exit codes or its prices.

---

## The mistake that produces a confident wrong answer

**The container clones this repository at a pinned sha. Your working tree is not in the
run.** The spec's `command:` does `git clone … && git checkout <sha>`, so an edit you made
and did not push is simply absent: the run executes the *previous* code, succeeds, and
writes a well-formed `cat_report.json` with a plausible theta and a healthy standard
error. Nothing downstream catches it, because nothing about the artifact is wrong — it is
an accurate measurement of code you no longer have.

So, in this order, every time: **commit → push → repin the sha in the spec → submit.**
An unpushed sha is the better of the two failures; it dies at `git checkout` a few minutes
into a paid run. An unpushed *edit* on top of a pushed sha dies silently and never.

Two asymmetries that make this easy to get wrong:

- `--spec` is read off your laptop and compiled into the submission, so the spec file
  itself need not be committed and cannot pin the commit that contains it.
- The code the run executes is a clone at a sha, **not** a wheel. `diagnostics/` sits at
  the repository root and `calibrated_datasets/` is 18 MB of item banks that `resolve.py`
  finds via `parents[4]`; neither ships in the packaged `src/`.

---

## What to get from the user

Five things, none of which you may invent. Ask for anything missing before doing anything
else, and **never construct a checkpoint path** — it cannot be derived.

| # | Input | Where it lands |
|---|---|---|
| 1 | **Checkpoint** — one or more `s3://` prefixes | `--checkpoint` inside the spec's `command:` |
| 2 | **A basic inference example** — the runner file plus ~200 tokens of its real output, verbatim, and the library and commit it was loaded with | `--dtype`, and whether generative banks are reachable at all |
| 3 | **Benchmarks** | `--benchmark`, or the `BENCHMARKS` array of a fan-out |
| 4 | **CAT settings** | `--se-threshold`, `--max-items`, `--ability-estimator`, `--batch-size` |
| 5 | **Hardware** — recommend `gpu-1xl4` | `--compute` on the submit line |

The eval job's role reads only under `s3://sbsandbox-intern-edullm-outputs/teams/<team>/runs/…`.
A checkpoint anywhere else is admitted, placed, and then fails its first read.

---

## Step 1 — read the inference example

Input 2 is not paperwork. Ask for the output more insistently than for the script: the
output is what shows where the model stops and what the call returns. Three facts come out
of it and each one changes the command.

**Does `pad_token_id == eos_token_id`?** On this checkpoint family it does — SmolLM2-135M
writes `pad == eos == bos == 0`. MCQ scoring survives that only by exemption: the native
scorer loads with `validate_checkpoint=False` and hands `from_checkpoint` a
`GenerationConfig` carrying a *fabricated* distinct pad id, because a forward-only scorer
encodes one `(prompt, continuation)` pair at a time and never pads. Omitting the config is
not the fix and was tried: `from_checkpoint` builds its own from the checkpoint's token ids
and the validator refuses it, which is how `run_019fe265` died at load having resolved the
bank and pulled all 140 objects. If the user's example shows distinct ids, none of this
applies and you should say so rather than carrying the workaround forward.

**Does the model emit EOS at all?** Read it out of the pasted output, not out of the
script. This decides whether a generative item terminates or runs to `max_new_tokens`
(1,024 for `leaderboard_math`, 1,280 for `ifeval`) — the difference between a generative
CAT that is cheap and one that is not worth submitting.

**What dtype do the weights load at?** Take it from the output, not from `config.json`.
The measured case here was 546.2 MB of weights for a 135M model — four bytes a parameter,
i.e. fp32 — from a command line that said `--dtype bfloat16`, because the flag did not
reach the loader at the time. It does now. Any theta recorded before 2026-08-08 says
bfloat16 and was produced in fp32, so it is not a baseline for a rerun.

Also record the library and commit the example was loaded with. A release wheel and `main`
disagree: PyPI `ai2-olmo-core` 2.4.0 has no schema for `sequence_mixer` or
`partial_rotary_factor` and cannot parse this config at all, while 2.5.0 — what the image
carries — can.

---

## Step 2 — preflight, free

`edullm check --json` costs a fraction of a second and reaches no network. Run the bank
resolution first, though: it is the only thing that will tell you a benchmark is not
runnable, and it is cheaper still.

**Query the ready set. Never recall it.** It changes as banks are vendored, and a
benchmark's *modality* moves too — `gpqa` is mid-reclassification from generative to MCQ
right now. Write this to a temp file and run it from the repo root; the quoting does not
survive most shells.

```python
from diagnostics.mcq_cat.styles.uni_mcq import resolve
from diagnostics.mcq_cat.styles.uni_mcq.datasets import ready_names

print("ready:", ready_names())
for name in ready_names():          # or just the names the user asked for
    try:
        r = resolve.resolve(name)
    except resolve.DatasetNotAvailable as exc:
        print("UNAVAILABLE", name, "->", exc)
        continue
    print(f"OK {name} items={r.manifest.get('items')} fit={r.fit_family} "
          f"modality={r.spec.modality}")
```

`PYTHONPATH` must contain **both** the repo root and `src` — `diagnostics` lives at the
root and `olmo_eval` under `src`, so `PYTHONPATH=src` alone fails with
`ModuleNotFoundError: No module named 'diagnostics'`.

```bash
cd <repo-root> && PYTHONPATH=.:src .venv/bin/python /tmp/preflight.py        # Unix
```
```powershell
cd <repo-root>; $env:PYTHONPATH=".;src"; & .venv\Scripts\python.exe "$env:TEMP\preflight.py"
```

Two of the nine supported banks are blocked today. When one is, the exception says
specifically why — no bank, a blocked bank, or an unknown name — and you relay that reason
verbatim rather than paraphrasing it as "not supported".

**Only `modality == "mcq"` banks can run on the native path.**
`GENERATIVE_BACKENDS` registers `hf` alone, because decoding needs a distinct EOS to stop
on and this checkpoint family has none, so the exemption that lets the MCQ scorer read the
raw format cannot be extended to a completer. `runner.run` calls `check_checkpoint_kind`
before it fetches anything, so a generative cell fails in seconds having spent nothing —
which makes it noise in a sweep, not insurance. Converting first
(`--checkpoint-prep auto --checkpoint-kind hf`) is the route to those banks and is a
separate, unverified experiment; do not fold it into an MCQ submission.

---

## Step 3 — write the spec

Specs live in `.edullm/`. Start from the closest sibling rather than from nothing:
[`run-native-cat.yaml`](../../../.edullm/run-native-cat.yaml) is the single-benchmark
control and [`run-native-cat-sweep.yaml`](../../../.edullm/run-native-cat-sweep.yaml) is
the MCQ fan-out. Their headers carry the arguments for every flag below and are worth
reading once.

The on-node command, with the parts that are yours in caps:

```yaml
schema_version: 1
workload_profile: olmo-core-check
suggested_compute: gpu-1xl4
command: >-
  bash -lc 'if ! command -v git >/dev/null; then apt-get update -qq && apt-get install -y -qq --no-install-recommends git; fi
  && git clone --filter=blob:none https://github.com/edu-llm/olmo-eval-full.git /opt/olmo-eval-full
  && cd /opt/olmo-eval-full
  && git checkout PUSHED_SHA_OF_THIS_REPO
  && python -m pip install --no-cache-dir ".[hf,s3]"
  && python -m diagnostics.mcq_cat.runner
  --cat-style uni_mcq
  --checkpoint S3_URI_FROM_THE_USER
  --benchmark ONE_NAME
  --checkpoint-prep none
  --checkpoint-kind olmo_core
  --dtype bfloat16
  --s3-out "$EDULLM_OUTPUT_PREFIX"'
```

`--checkpoint-prep none --checkpoint-kind olmo_core` is what makes this the native path:
nothing is converted, the sharded DCP is read as-is, and no HF directory is written. They
are independent seams and this is the pairing that converts nothing.

**Not `--extra olmo_core` on the install.** That extra pins `ai2-olmo-core==2.4.0`, so it
would install over the 2.5.0 the image already carries — the one release that cannot read
this config — and the run would die at config parse having looked correct all the way down.

### More than one benchmark means a fan-out

`--benchmark` is singular. Passing it four times is not an error: argparse keeps the last
value and the run silently scores that one benchmark.

```
$ python -m diagnostics.mcq_cat.runner --benchmark arc_challenge --benchmark hellaswag \
    --benchmark musr --benchmark bbh --dry-run …
[dry-run] style=uni_mcq benchmark=bbh …
```

Declare the fan-out in the spec instead, so the shape lives beside the loop it drives and
a submission cannot ask for a cell count the command has no benchmark for:

```yaml
fanout:
  size: 4
  index_parameter: benchmark
command: >-
  bash -lc 'BENCHMARKS=(arc_challenge hellaswag musr bbh)
  ; CELL="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"
  ; BENCH="${BENCHMARKS[$CELL]:-}"
  ; if [ -z "$BENCH" ]; then echo "cell $CELL has no benchmark; fanout.size must equal ${#BENCHMARKS[@]}" >&2; exit 2; fi
  …  --benchmark "$BENCH"  …'
```

`fanout.size` must equal the array length. The runtime bound applies **per cell**, which is
the reason to fan out rather than loop inside one container: four cells get the bound each,
where four benchmarks in one container would share it.

The array must be the MCQ set **at the sha you pinned**, not the one your working tree
reports. Those differ whenever a modality is mid-move, which it is today.

Several checkpoints are the same mechanism with the array holding S3 prefixes and
`index_parameter: checkpoint`. Several checkpoints *and* several benchmarks is a product,
so it is one submission per checkpoint each fanning out over benchmarks — say the total
cell count to the user before submitting any of them, because `cost` is per submission.

---

## Step 4 — commit, push, repin

Commit the paths the run actually clones — `diagnostics/`, `calibrated_datasets/`, and
anything they import — rather than `-A`; other people work in this tree and scratch files
live in it.

```bash
git log --oneline -5        # somebody else may have just pushed
git commit … && git push
git rev-parse HEAD          # → the sha that goes in the spec's git checkout
```

Put that sha in the spec and read it back before submitting. **Do not push twice inside a
minute**: the image build re-verifies source identity, and a second push kills the first.

---

## Step 5 — check, then submit

**Submit from the OLMo-core clone.** The job runs on OLMo-core's image, not this
repository's: the published `olmo-eval-full` image leaves torch out and can only run
`provider.kind=mock`, while OLMo-core's carries CUDA torch and a 2.5.0 `ai2-olmo-core`,
which is exactly what reading a sharded DCP checkpoint needs. The CLI reads `--commit`
against the clone it is run in, and this repository holds no OLMo-core objects.

```bash
cd ../OLMo-core
edullm check --experiment <slug> --dataset none \
  --workload olmo-core-check --compute gpu-1xl4 \
  --commit <OLMo-core sha> \
  --spec ../olmo-eval-full/.edullm/<your-spec>.yaml --json
```

Swap `check` for `submit` once it comes back clean. Every completed native run so far used
OLMo-core `08df5aa0142465c80b4ea48e84faa46117275d61`; confirm it is still what you want
rather than assuming it.

- **`--dataset none` is a statement**, not an omission. This run reads no corpus.
- **Do not pass `--hours` above the workload's bound.** `--hours 2` against
  `olmo-core-check` is refused outright with `runtime_above_the_workload_bound`. Read the
  bound out of `cost.maximum_runtime_hours`. (The header of `run-native-cat.yaml` still
  shows `--hours 2`; that line no longer validates.)
- **Read `cost` and `approval_class` out of the JSON and quote nothing from memory** —
  AGENTS.md forbids it, and the numbers live in reviewed configuration that moves. Also
  read `approving_environment`, and note that cell count can change the class: the same
  spec at one cell and at four came back in different classes on 2026-08-08.
- **`approval_class: automatic` does not mean it will start.** Runs classified automatic
  have repeatedly still parked at `PENDING_APPROVAL` behind the `run-approval-lead` GitHub
  environment, which the CLI cannot see. Tell the user a run may wait; do not promise it
  will not.
- Three `deferred` entries are normal. They are image-registry questions only a pushed
  commit and the submission workflow's credential can answer.

### Hardware

Recommend `gpu-1xl4`. It is what every validated run has used, and the platform's
`bfloat16_not_in_the_hardware` guard refuses the Turing shapes for this command — verified
on `gpu-1xt4`, whose refusal names the provisioned shapes whose cards do have the format.
That guard **reads the text of the command and nothing else**, which is why `--dtype` is
named explicitly even at its default: a precision the program picks in code is invisible to
it, and the run would instead die on its first kernel after being priced, released,
admitted and given a machine. Do not call any shape the cheapest without reading it out of
`check`; the ordering is not what it looks like.

---

## Step 6 — watch it

`edullm status --json` answers from GitHub, dispatches nothing, is free and **may be
polled**. Plain `edullm status`, `edullm status --ask-aws` and `edullm logs` all start a
workflow, so none of them belongs in a loop.

Real progress lives in CloudWatch, log group `/aws/batch/sbsandbox-intern-edullm-gpu`.
Reach it through the sb-aws broker, read-only — AGENTS.md rules out `boto3`, the `aws` CLI
and `curl` at an AWS endpoint, and a broker call is attributable where a laptop shell is
not. **Read the events, not `lastEventTimestamp`**: that field lags badly enough on a live
stream to make a working job look dead.

---

## Step 7 — read the report

`cat_report.json` lands under the platform's output prefix. The fields that matter:

- `ability.theta` — the ability estimate, on a logit scale centred near 0. **Comparable
  across runs of the same benchmark only**, never across benchmarks: each bank's scale is
  anchored to its own calibration population.
- `ability.standard_error` — at the default threshold an EAP-reported run lands at or below
  0.3. Under `batch_eap+mwle` this is `se_mwle`, `1/sqrt(I(theta))`, an asymptotic
  likelihood SE and not a posterior SD; the stopping rule was still applied to the
  posterior one, so do not read it against 0.3.
- `metadata.theta_batch` / `metadata.theta_mwle` — both estimators' answers are recorded
  whichever was asked for, so a run reported under one stays comparable with a run reported
  under the other. `ability_estimator_reported` says which is published.
- `metadata.pirt_accuracy` — predicted accuracy, and the number to quote to a user who
  wants "how good is it". Report it with `metadata.pirt_accuracy_denominator`, because it
  is over the **calibrated subset**, not the full benchmark split.
- `metadata.observed_accuracy` — raw fraction correct on the items actually administered.
  It differs from predicted accuracy by design: adaptive testing concentrates on items near
  the model's ability, where it is near 50/50.
- `metadata.n_items_administered`, `metadata.bank_size`, `metadata.stop_reason`.
- `metadata.cat_settings` — including `max_items_is_pinned_value`. False means this run is
  not comparable with the others and says so where a reader will not notice.
- `metadata.ungradable` — always present. A nonzero `count` means those items were scored 0
  and theta is that much lower than the response pattern warrants. Above a 20% rate the
  block gains an `alert`, and **then theta is a floor produced by grading failing, not a
  measurement of the model** — a missing dependency, a bank vendored against another
  convention and a genuinely weak checkpoint all look identical, and all of them shrink the
  standard error on schedule.
- `metadata.bank_provenance` and `metadata.scoring_note`, plus `bank_caveat` when present.
  Always surface these. Several banks were calibrated under a different prompting
  convention than we administer, which shifts theta's absolute value while leaving
  checkpoint-to-checkpoint comparison intact; a user told only the number will over-read it.

---

## Worked example

**The five inputs, as given.** (1) `s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/run_019fce1a-f393-70e3-ba0e-e2771c70f9c0/checkpoints/step305176/`.
(2) An inference example loaded with `ai2-olmo-core` 2.5.0, whose tokenizer
`HuggingFaceTB/SmolLM2-135M` reports `pad == eos == bos == 0`, and whose weights measured
546.2 MB for 135M parameters — fp32, against a command line that said bfloat16. Whether
this checkpoint ever emits EOS is the one thing the example did not settle, which is why a
separate 64-token probe exists.
(3) All MCQ banks. (4) Defaults, with Warm's estimate reported instead of EAP.
(5) `gpu-1xl4`.

Input 2 settles two things before anything is written: the pad/eos collision is why
`--checkpoint-kind olmo_core` needs the fabricated pad id and why the same exemption cannot
be extended to a completer, and the fp32 measurement is why `--dtype bfloat16` is a real
change here rather than a restatement — a rerun is not expected to reproduce the fp32
theta. The unanswered EOS question costs nothing on this submission, because MCQ scoring is
forward-only and never stops on a token; it is what would have to be answered first to cost
a generative bank. Input 3 plus Step 2's preflight gives four MCQ banks at the pinned sha,
so it becomes a four-cell fan-out rather than four submissions.

The spec is [`.edullm/run-native-cat-sweep.yaml`](../../../.edullm/run-native-cat-sweep.yaml),
already committed, pinning `6c1d8415…`:

```yaml
schema_version: 1
workload_profile: olmo-core-check
suggested_compute: gpu-1xl4
fanout:
  size: 4
  index_parameter: benchmark
command: >-
  bash -lc 'BENCHMARKS=(arc_challenge hellaswag musr bbh)
  ; CELL="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"
  ; BENCH="${BENCHMARKS[$CELL]:-}"
  ; if [ -z "$BENCH" ]; then echo "cell $CELL has no benchmark; fanout.size must equal ${#BENCHMARKS[@]}" >&2; exit 2; fi
  ; echo "cell $CELL -> $BENCH"
  ; if ! command -v git >/dev/null; then apt-get update -qq && apt-get install -y -qq --no-install-recommends git; fi
  && git clone --filter=blob:none https://github.com/edu-llm/olmo-eval-full.git /opt/olmo-eval-full
  && cd /opt/olmo-eval-full
  && git checkout 6c1d8415c306e6756de50dc6682ec1369c990b23
  && python -m pip install --no-cache-dir ".[hf,s3]"
  && python -m diagnostics.mcq_cat.runner
  --cat-style uni_mcq
  --checkpoint s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/run_019fce1a-f393-70e3-ba0e-e2771c70f9c0/checkpoints/step305176/
  --benchmark "$BENCH"
  --ability-estimator batch_eap+mwle
  --checkpoint-prep none
  --checkpoint-kind olmo_core
  --dtype bfloat16
  --s3-out "$EDULLM_OUTPUT_PREFIX"'
```

```bash
cd ../OLMo-core
edullm check --experiment native-olmo-core-cat-sweep --dataset none \
  --workload olmo-core-check --compute gpu-1xl4 \
  --commit 08df5aa0142465c80b4ea48e84faa46117275d61 \
  --spec ../olmo-eval-full/.edullm/run-native-cat-sweep.yaml --json
```

Exit 0, `"refused": false`, `"refusals": []`. **Re-run it rather than reading these numbers
off this page** — they were true on 2026-08-08 and live in configuration that moves:

```json
"approval_class": "routine",
"approving_environment": "run-approval-lead",
"cost": { "cells": 4, "nodes": 1, "hourly_rate_usd": "0.8048",
          "maximum_runtime_hours": "1", "maximum_attempts": 1,
          "maximum_compute_cost_usd": "3.22" },
"history": { "said": "3 succeeded runs of this workload, on this machine, on this
              dataset took a median of 8m, between 2m and 21m." }
```

The same spec at `--hours 2` is refused with `runtime_above_the_workload_bound`, and at
`--compute gpu-1xt4` with `bfloat16_not_in_the_hardware`. The single-benchmark sibling —
one cell, same everything else — came back `automatic` / `run-approval-automatic` instead,
which is the whole reason you read the class rather than predicting it.

Swapping `check` for `submit` is the only remaining step, and it is the one that spends
money. Show the user the resolved plan, the cost block and the class first.

---

## What not to do

- **Do not submit with uncommitted or unpushed changes to `diagnostics/` or
  `calibrated_datasets/`.** See the top of this file. This is the expensive one.
- Do not pass `--benchmark` more than once. Fan out.
- Do not guess a checkpoint path, and do not promise an output location — the platform
  mints it.
- Do not pass `--force`, and do not edit a spec to silence a refusal without reading it.
- Do not lower `--se-threshold` or raise `--max-items` to "get a better number". A wider
  cap buys precision, not accuracy, and breaks comparability with every existing run. The
  session floor is `min_items` (8, and 24 for `bbh`), which always binds before precision
  does, so an ordinary run administers 8–40 items.
- Do not run a benchmark absent from `ready_names()` by pointing at its bank directly. The
  exclusions are recorded, and several of them are that the resulting number would be
  meaningless.
- Do not compare theta across benchmarks. Predicted accuracy is the cross-benchmark
  quantity.
- Do not present a theta from a report carrying an `ungradable` alert as a measurement of
  the model.
- Do not quote a price, a runtime bound, a cost ceiling or an approver from memory or from
  a document — including this one.
