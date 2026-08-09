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
| 2 | **A basic inference example** — the runner file plus ~200 tokens of its real output, verbatim, and the library and commit it was loaded with | Step 1, which produces the same thing yourself when it does not arrive |
| 3 | **Benchmarks** | `--benchmark`, or the `BENCHMARKS` array of a fan-out |
| 4 | **CAT settings** | `--se-threshold`, `--max-items`, `--ability-estimator`, `--batch-size` |
| 5 | **Hardware** — recommend `gpu-1xl4` | `--compute` on the submit line |

The eval job's role reads only under `s3://sbsandbox-intern-edullm-outputs/teams/<team>/runs/…`.
A checkpoint anywhere else is admitted, placed, and then fails its first read.

Inputs 1 and 2 are what [`RUNNER_REQUEST.md`](../../../RUNNER_REQUEST.md) asks a submitter
for, in terms that mean something to them. Hand them that file rather than paraphrasing it.

**Input 2 is the only one you can produce yourself, and Step 1 is how.** Ask for it, because
a submitter's own decode costs nobody a card; but a missing or unreadable answer is not a
reason to stop, and an answer that arrives is not a reason to skip Step 1 on a checkpoint
whose architecture you have not read before.

---

## Step 1 — make the checkpoint generate, natively, before you evaluate it

**Get a runner and real output for the submitter's own checkpoint, through the raw native
OLMo-core path, before anything is spent on a bank.** This is an active first task, not a
form to collect: prove the weights load and decode, then use what that proves to decide
whether the pipeline needs changing for this particular model. It is a submission like
every other on this page — the weights are in S3 and the loader wants a GPU — but it is one
cell, priced by the 1.74 GB checkpoint pull rather than by generation, and every question
below is one that otherwise gets answered by a paid eval returning a number nobody can read.

[`.edullm/generation_probe.py`](../../../.edullm/generation_probe.py) is the worked example.
It is a throwaway that nothing imports, written for exactly this, and its header is the
model to copy: it separates what was already known from reading OLMo-core 2.5.0 at
`08df5aa0` from what could only be observed on a card, and it guards each prompt separately
so a checkpoint that decodes for one bank and dies on another still reports the first
finding. Its spec is
[`run-generation-probe.yaml`](../../../.edullm/run-generation-probe.yaml), which passes no
`--s3-out` on purpose — the product is the CloudWatch log, read the way Step 7 describes,
and nothing downstream consumes a file. It asks for `--max-new-tokens 64` against banks
configured for 1,024 and 1,280, which is the cost control and is also sufficient: 64 tokens
cannot show what a finished answer looks like and can show whether one ever ends. **The sha
the spec pins is the one that has to carry the probe**, not the spec, which is read off your
laptop; so Step 5 applies here first, before it applies to any eval.

**The hard constraint: nothing is converted, anywhere in this step.** Generation goes
through `TransformerGenerationModule.from_checkpoint` against the raw sharded DCP directory
— the shape `inference._OlmoCoreScoringModel` builds and the probe drives end to end — and
never through `save_hf_model` or `convert.ensure_hf_checkpoint`, and never under
`--checkpoint-prep auto`. Conversion is the fallback, documented in Step 3, and part of what
this step exists to establish is that it is not needed. The reason is generality rather than
price: `save_hf_model` calls `get_hf_config`, which upstream builds a config only for
`ReorderedNormTransformerBlock` — the OLMo-2/OLMo-3 post-norm family — and raises
`NotImplementedError` for everything else, which is why this project carries
`hf_config_patch` at all; these checkpoints are plain pre-norm `TransformerBlock` models and
the conversion fails on them, *after* the model has been rebuilt and 1.7 GB of shards read
into it. That patch is one submitter's architecture taught to one exporter. The native
reader takes whatever `TransformerConfig.from_dict` parses.

### What the probe has to answer, and what each answer costs

1. **Are `pad_token_id` and `eos_token_id` the same value?** This family writes both as 0.
   `GenerationConfig.validate` refuses `pad == eos`, and the fix is to fabricate the *pad* —
   `_OlmoCoreScoringModel._unused_pad_token_id` returns 1 when eos is 0 — leaving the real
   end-of-text where the checkpoint put it. **The older belief that this collision blocked
   generation outright was measured and withdrawn**: probe `run_019fe316-0e47` loaded with
   `pad_token_id=1, eos_token_id=0` and decoded 64 tokens on each of three prompts. Say that
   plainly, because the withdrawn claim was written down widely enough that its correction
   now sits in five places — twice in `inference.py`, in `_load_olmo_core`'s docstring, in
   `test_olmo_core_generation.py` and in `Plan/flows/uni_mcq/GENERATIVE_RUN.md`. If the
   checkpoint's ids are distinct, none of this applies and you should say so rather than
   carrying the workaround forward.
2. **Does the model ever emit its end-of-text token?** Ask it of the returned *ids* — does
   the eos id appear in them — and not of whether generation stopped. Those are different
   questions, and the second has a cause that is not the model: on the converted path
   `generate` was being handed `eos_token_id=None`, because `hf_config_patch._llama_config`
   writes no id and `save_hf_model` fills them only when handed a tokenizer object, so a
   decode could run to the cap for a reason that had nothing to do with the weights. On this
   checkpoint the answer is no: `run_019fe316-0e47` found token 0 in none of the 192 tokens
   it decoded, and all three prompts ran their full 64. The consequence is the whole price
   of a generative CAT — every item pays `max_new_tokens` instead of stopping, which is the
   difference between a cheap generative bank and one not worth submitting.
3. **How fast does it decode without a KV cache, which is the only way it decodes today?**
   OLMo-core's default torch attention backend refuses KV caching outright —
   `assert_supports_kv_cache` raises at `olmo_core/nn/attention/backend.py:288-289` the
   moment `prepare_inference_cache` runs, which is how probe `run_019fe2e6` died before
   emitting a token — and only the flash backends implement it, from a binary wheel this
   image does not carry. So `use_cache=False` is the working configuration and every step
   re-reads the whole prefix. Measured on an L4: `run_019fe78d` decoded at **39.6 tokens/sec,
   about 25.8s for a 1,024-token item**, so roughly 17 minutes for a 40-item bank against a
   1h per-cell bound with no second attempt.
   `TORCH_TODOS.md` extrapolates ~32s an item from the probe's 64-token samples instead; both
   are floors, since both extrapolate linearly from a path both call quadratic. Scale the
   *rate* when a submitter's checkpoint is larger. That file also holds the two-part fix —
   the wheel in the image, and `attention_backend` threaded through the `from_checkpoint`
   the completer reaches — and why neither is on the critical path.
4. **What context window does the checkpoint declare, and what declared it?** Read it from
   the checkpoint's own config, never from a constant. `olmo_core_context_length` tries
   `OLMO_CORE_HF_MAX_POSITION_EMBEDDINGS` first, then olmo-eval's own
   `_MAX_LENGTH_CONFIG_KEYS` under `model.*` — `max_sequence_length`, `max_seq_len`,
   `max_position_embeddings` — and only then `dataset.sequence_length`. On this family the
   last is the only one present: 2048, no model-level key, which is what `run_019fe78d`
   recorded as its `context_length_source`. A checkpoint declaring none of them is **refused
   at load** rather than run unclamped, and the tokenizer's `model_max_length` is
   deliberately not a fallback — SmolLM2's says 8192 against weights trained at 2048, and a
   window four times too large is worse than none.
5. **What precision did the weights actually load at?** Take it from the output, not from
   `config.json`: measure the bytes that land on the device. The failure here was measured —
   546.2 MB for a 135M model in `run_019fe2e6`, four bytes a parameter, from a command line
   that said `--dtype bfloat16` — because `InferenceConfig` carried no dtype and
   `from_checkpoint` was given none. The flag reaches the loader now, and the runner sends it
   to all three places a precision can be chosen: `prepare_checkpoint` on the converted path,
   `InferenceConfig` for scoring, `GenerationConfig` for decoding.
6. **Which tokenizer resolved, and does it fit the model?** A raw checkpoint ships no
   tokenizer files and names one by identifier at `dataset.tokenizer.identifier`, so it is
   fetched from the Hub — which means this step also proves the venue can reach the Hub.
   Cross-check the model's `vocab_size` against `len(tokenizer)`: 49,152 against
   `HuggingFaceTB/SmolLM2-135M` here. That check is worth the two lines because a wrong
   identifier produces a plausible tokenizer rather than an error, and an agent on this
   project did exactly that — reading `_resolve_tokenizer_id`'s fallback branch and a test
   fixture, it concluded the tokenizer was `allenai/dolma2-tokenizer`, whose ~100k against
   49,152 would have contradicted it in seconds. **Nothing on the native path makes the
   comparison for you**: `_warn_if_tokenizer_outgrows_model` belongs to `_HFCompleter`, and
   `_OlmoCoreCompleter.checkpoint_facts` records `tokenizer_vocab_size` without comparing it.
   Print both numbers.
7. **What does `generate_batch` return?** It seeds its output with `input_ids` and
   concatenates onto it, so the prompt is in the return unless `completions_only=True` — and
   the completer slices `[prompt_tokens:]` rather than passing that flag, because slicing is
   what the probe measured and the flag was only ever read off the source. Print the raw
   type and shape so the stripping logic is checked against a real return rather than a
   remembered one.
8. **What does the model actually write?** Look here first; it costs nothing and it decides
   what a generative bank can be claimed to have measured. Two distinct degeneracies show up
   in the text, and **both produce confident wrong numbers rather than errors.** A model that
   **echoes its prompt** passes 129 of IFEval's 511 vendored items and reports a high, tight
   theta at `ungradable.rate` 0.0 with no field indicating anything is wrong — which is why
   `ifeval` is blocked outright (Step 3). A model that **loops a clause** never reaches
   `\boxed{}` or a `Final Answer:` line, the only two forms MATH's grader reads, so every
   item scores 0 for a formatting reason rather than a mathematical one: on `run_019fe78d`,
   zero of 40 completions carried either. Both are written up in
   [`RESULT_CAVEATS.md`](../../../RESULT_CAVEATS.md); cite the entry rather than restating
   it. This checkpoint did all of it on 64 tokens — ifeval echoed its instruction, math
   looped, gpqa invented options (E) through (H) after the real four.

### End with a decision, not a log

The probe's output is only worth its card if it changes what you do next. Say which of the
above apply to *this* checkpoint and therefore what has to change before a real run: a
budget that must absorb an end-of-text token the model never emits, a generative bank not
worth submitting because the decode is degenerate, a `--dtype` that did not reach the
weights, a window or a tokenizer that will refuse the load outright. When the decode loops,
say before submitting that the run will establish the pipeline and not the checkpoint — it
is still worth doing for that, but it is a different claim than the user asked for.

**A clean probe is itself the result.** It licenses Step 3's preflight and a real
submission, and it is the evidence behind telling a user that nothing in the pipeline needed
changing for their model — which is a stronger statement than a green eval, because it names
what was checked. [`RUNNER_REQUEST.md`](../../../RUNNER_REQUEST.md) is the submitter-facing
artifact for the same information; hand them that when you want them to answer it themselves.

---

## Step 2 — read the inference example

Input 2 is not paperwork. Ask for the output more insistently than for the script: the
output is what shows where the model stops and what the call returns. Four facts come out
of it and each one changes the command — the same four Step 1 measures for itself, asked
here of the submitter's evidence rather than yours. When Step 1 has run you already hold
better answers than a pasted example can give, and what is left to read off theirs is what
no probe of yours can see: the library and commit it loaded under, and whether their runner
and yours agree about the model at all.

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
(1,024 for `leaderboard_math`, the only generative bank in the ready set, and a nominal cap
the context clamp can lower but never raise) — the difference between a generative CAT that
is cheap and one that is not worth submitting.

Two consequences of that token now land at checkpoint load rather than in the report. The
end-of-text token is resolved from the checkpoint's *own* tokenizer at run time, closes
MATH's few-shot exemplars (`exemplar_eos_token`) and is appended to that bank's effective
stop list (`eos_stop_sequences`), so a checkpoint shipping no tokenizer files and naming
one the Hub cannot serve is **refused** at load rather than degraded into scoring every
item at the flat cap — `require_live_tokenizer`, which probes the counter rather than
testing for the attribute. The same load reads the checkpoint's context window out of
`config.json`, trying `OLMO_CORE_HF_MAX_POSITION_EMBEDDINGS`, then olmo-eval's own
`model.*` spellings, then `dataset.sequence_length`; a checkpoint declaring none of them
is **refused** by `olmo_core_context_length` with a message naming every path it searched
and the override variable, because unclamped is not a degraded generative run but a wrong
one that looks right.

When a prompt does not fit, the clamp acts before the ladder: the generation budget is
reduced first, and whole exemplars are dropped only when the prompt itself will not fit,
flooring at 1 shot. That order is measured, not asserted — in `run_019fe78d` the clamp
fired on exactly one of 40 items, cutting a 1,645-token prompt's budget from 1,024 to 403
with `exemplars_dropped: 0`, and every one of the 40 ran at `num_fewshot: 4`. **So on a
2048-token window the exemplar ladder is measured inert and `num_fewshot` mixing is not a
hazard you will meet**; it becomes one on a smaller window or a bank with longer stems.
Read `num_fewshot` per response and the `context_fit` block rather than assuming either
way.

**Does the model recite its prompt, loop a clause, or otherwise decode degenerately?** Of
the four, look for this one first. It costs nothing and it decides what a generative bank
can be claimed to have measured. An IFEval instruction is verifiable precisely because it
names its own success token, so stating the constraint puts that token in the prompt, and a
response quoting the prompt satisfies the checker. Measured through this repo's own grading
path over the 511 vendored items, a verbatim echo passes 129 of them, 25.2%; simulated
against this style's own Fisher selection and EAP it reports theta +0.19 to +1.11 at
se 0.11–0.23 and stops on precision after 8 or 9 items — the highest and tightest number
in a sweep whose MCQ cells run -0.25 to -3.9, at `ungradable.rate` 0.0 with no field
indicating anything is wrong. That is why `ifeval` is blocked (Step 3); the evidence is in
[`RESULT_CAVEATS.md`](../../../RESULT_CAVEATS.md) and is not worth restating here. The
point for you is that a submitter's 200 tokens predict it for free, before anything is
spent: a model that repeats its prompt back will report inflated ability on a generative
bank rather than the floor it deserves. MCQ scoring is unaffected — it is forward-only and
decodes nothing.

**On MATH the same degeneracy fails in the opposite direction, and it is not free.** A
looping decode never reaches `\boxed{}` or `Final Answer:`, so the grader has nothing to
extract and scores every item 0 for a formatting reason. That is what happened on
`run_019fe78d`: 40 items, 40,339 tokens, 0 correct, no answer form in any completion. The
run still cost its GPU minutes and still produced a report a reader would take at face
value. So when the pasted output loops, say before submitting that the run will establish
the pipeline and not the checkpoint — it is still worth doing for that, and the header of
[`run-native-math.yaml`](../../../.edullm/run-native-math.yaml) says so in as many words,
but it is a different claim than the user probably asked for.

**What dtype do the weights load at?** Take it from the output, not from `config.json`.
The measured case here was 546.2 MB of weights for a 135M model — four bytes a parameter,
i.e. fp32 — from a command line that said `--dtype bfloat16`, because the flag did not
reach the loader at the time. It does now, and it reaches **both** configs the runner
builds — `InferenceConfig` for scoring and `GenerationConfig` for decoding — so a native
generative run is built at the precision the command names rather than at the
checkpoint's. Any theta recorded before 2026-08-08 says bfloat16 and was produced in fp32,
so it is not a baseline for a rerun.

Also record the library and commit the example was loaded with. A release wheel and `main`
disagree: PyPI `ai2-olmo-core` 2.4.0 has no schema for `sequence_mixer` or
`partial_rotary_factor` and cannot parse this config at all, while 2.5.0 — what the image
carries — can.

---

## Step 3 — preflight, free

`edullm check --json` costs a fraction of a second and reaches no network. Run the bank
resolution first, though: it is the only thing that will tell you a benchmark is not
runnable, and it is cheaper still.

**Query the ready set. Never recall it.** It changes as banks are vendored and as banks are
withdrawn, and a benchmark's *modality* moves too — `gpqa` finished moving from generative
to MCQ in `49d93da4`, which changed how many cells a native sweep can hold. Write this to a
temp file and run it from the repo root; the quoting does not survive most shells.

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

Three of the nine supported banks are blocked today — `winogrande`, `gsm8k` and, since
2026-08-08, `ifeval` — so `ready_names()` returns six. When a bank is blocked the exception
says specifically why, and it distinguishes no bank from a blocked bank from an unknown
name; relay that reason verbatim rather than paraphrasing it as "not supported".

**`ifeval` cannot be selected, whatever the user asks for — including by name, including
when they say they understand the caveat.** There is no flag that reaches it and no
correct way to point at its bank directly; `ready_names()` returns six names and `ifeval`
is not among them. Its block is unlike the other two, which are join-evidence questions:
this bank's join is fine and its grading is faithful, and that is exactly the problem — see
Step 1's last question and Step 2's third fact, and read the `blocked` reason out of
`datasets.py` for the authoritative wording to relay, because it is long and specific and
paraphrasing it as "not supported" loses the argument. The short of it is that a verbatim
prompt echo passes 129 of the 511 vendored items, so a reciting model reports a high, tight
theta at `ungradable.rate` 0.0 with no field indicating anything is wrong. What lifts it is the
echo-baseline guard: stamp each item offline with whether an echo passes it, then report
the share of a session's passes an echo would also have produced. **Re-checking a join
does not lift it**, and neither does converting the checkpoint or picking a better model;
the exploit belongs to the bank's verifiers rather than to any checkpoint. Say that to a
user who asks for it by name, and offer the five MCQ banks and MATH instead.

**Both modalities now run on the native path.** `LOADABLE_CHECKPOINT_KINDS` gives MCQ and
generative the same `("hf", "olmo_core")`, because `GENERATIVE_BACKENDS` registers
`"olmo_core": _load_olmo_core` beside `hf`. **This page said the opposite until
2026-08-09, and an agent reading the old sentence would have told a user that MATH cannot
be submitted natively.** Commit `0d25d4ba` wrote `_OlmoCoreCompleter` and registered it;
`run_019fe78d` then graded `leaderboard_math` on a raw sharded checkpoint end to end with
no conversion anywhere in it. So `--benchmark leaderboard_math --checkpoint-prep none
--checkpoint-kind olmo_core` is a submission you can make today, and Step 4 has the spec.

The two registries stay separate dicts rather than collapsing to one tuple, and the
docstrings say why: a backend registers per modality, the two legitimately diverged for
weeks while the MCQ scorer read this format and the generative stub refused it, and the
next backend will arrive the same way. **They agree today; that is a fact about today.**
Read them, do not recall them. `runner.run` calls `check_checkpoint_kind` before it
fetches anything, so a kind neither registry knows still fails in seconds having spent
nothing, ahead of the multi-gigabyte `resolve_checkpoint` that guard exists to run before.

Converting first (`--checkpoint-prep auto --checkpoint-kind hf`) still reaches the
generative bank and still has never once been executed. It is now the fallback rather than
the route, and the reason is generality rather than price: `save_hf_model` has to
understand each submitter's architecture — this checkpoint already needed the
`get_hf_config` patch because a plain pre-norm block raises upstream — while the native
reader takes whatever `TransformerConfig.from_dict` parses. Note the asymmetry if you ever
do take it: on the converted path the kind names the *backend*, so `--checkpoint-kind hf`
is correct and `olmo_core` is refused up front, which is the reverse of the native path.

---

## Step 4 — write the spec

Specs live in `.edullm/`. Start from the closest sibling rather than from nothing:
[`run-native-cat.yaml`](../../../.edullm/run-native-cat.yaml) is the single-benchmark
control, [`run-native-cat-sweep.yaml`](../../../.edullm/run-native-cat-sweep.yaml) is the
MCQ fan-out, and [`run-native-math.yaml`](../../../.edullm/run-native-math.yaml) is the
generative one — the spec `run_019fe78d` was submitted from. Their headers carry the
arguments for every flag below and are worth reading once.

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

### MATH is the same spec with two words changed

`leaderboard_math` is the one generative name in `ready_names()`, and on the native path
its spec differs from the MCQ control in the benchmark name and the reported estimator,
and in nothing structural:

```yaml
  --benchmark leaderboard_math
  --ability-estimator batch_eap+mwle
```

Everything else — `--checkpoint-prep none --checkpoint-kind olmo_core --dtype bfloat16`,
the clone, the pinned sha, `--s3-out "$EDULLM_OUTPUT_PREFIX"` — is unchanged, which is the
claim the per-modality registry makes: a backend is a completer plus a row in
`GENERATIVE_BACKENDS`, and everything between a prompt and a graded response is
backend-agnostic and already written.

**`batch_eap+mwle` is the standard, and ask for it explicitly.** An earlier version of this
page recommended plain `batch_eap` on a first run and justified it with the claim that
"both thetas land in the report either way." **That claim is false**, and the first native
MATH run proved it: `run_019fe78d` asked for `batch_eap` and its report carries
`theta_online` and `theta_batch` and **no `theta_mwle` at all**. The estimator you do not
ask for is not computed, so asking for the pair is the only way to get both.

Why the pair is worth having: EAP's standard-normal prior compresses every ability toward
zero, hardest at the tails, which is exactly where a small checkpoint sits. MWLE drops the
prior and penalises `1/2 ln I(theta)` instead. Reporting both lets a reader see how much of
a theta is prior and how much is evidence — on the MCQ sweep the two differed by 0.008 to
0.18, all smaller than the standard error except on musr.

The real caveat is about the SE, not the estimator, and it is a reading instruction rather
than a reason to avoid MWLE: under `batch_eap+mwle` the published
`ability.standard_error` is `se_mwle`, an asymptotic `1/sqrt(I(theta))`, while the stopping
rule acted on the posterior `se_online`. Quote `se_online` when discussing precision. See
`RESULT_CAVEATS.md`, which records how this inverted the apparent precision ordering across
the five MCQ cells.

**Keep it one cell.** There is one generative bank, so there is nothing to fan out over,
and a single cell is also what auto-approves — see Step 6, which is where that decision
actually costs you something.

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
reports. Those differ whenever a modality moves or a bank is withdrawn, and both happened
this week: `gpqa` became MCQ and gained a cell, `ifeval` was blocked and lost one.

Several checkpoints are the same mechanism with the array holding S3 prefixes and
`index_parameter: checkpoint`. Several checkpoints *and* several benchmarks is a product,
so it is one submission per checkpoint each fanning out over benchmarks — say the total
cell count to the user before submitting any of them, because `cost` is per submission.

---

## Step 5 — commit, push, repin

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

## Step 6 — check, then submit

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
  read `approving_environment`.
- **Shape decides approval, and price does not.** This is the single most useful
  operational fact when you are submitting one benchmark. The class comes out of
  `classify_request`, not out of a cost threshold: **a fan-out never auto-approves at any
  price, while a single cell routes to `automatic`.** Measured both ways on the same
  checkpoint and the same hardware — the five-cell MCQ sweep classified `routine` /
  `run-approval-lead` and sat about sixteen minutes waiting for a lead, and single-cell
  `run_019fe78d` was, in the CLI's words, "released automatically. Nothing is waiting on a
  person." Five separate single-cell submissions cost the same as a five-cell fan-out and
  wait for nobody. So fan out when the cells are one experiment, and do not fan out merely
  because several runs are convenient to launch together.
- **`approval_class: automatic` does not mean it will start.** Runs classified automatic
  have still parked at `PENDING_APPROVAL` behind the `run-approval-lead` GitHub
  environment, which the CLI cannot see, and the automatic class also closes once the
  day's unattended spend reaches its ceiling. Tell the user a run may wait; do not promise
  it will not.
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

## Step 7 — watch it

`edullm status --json` answers from GitHub, dispatches nothing, is free and **may be
polled**. Plain `edullm status`, `edullm status --ask-aws` and `edullm logs` all start a
workflow, so none of them belongs in a loop.

Real progress lives in CloudWatch, log group `/aws/batch/sbsandbox-intern-edullm-gpu`.
Reach it through the sb-aws broker, read-only — AGENTS.md rules out `boto3`, the `aws` CLI
and `curl` at an AWS endpoint, and a broker call is attributable where a laptop shell is
not. **Read the events, not `lastEventTimestamp`**: that field lags badly enough on a live
stream to make a working job look dead.

### A generative run in flight looks different, and the difference wastes checks

Three things about `run_019fe78d` that an MCQ run does not teach you, and each one cost a
check that returned nothing:

- **S3 is written once, at the end.** `runner._write_report` uploads `cat_report.json`
  after the session finishes; nothing is streamed and no partial artifact appears. So an
  absent output prefix at minute ten means the run has not finished, and nothing else. Do
  not read it as failure.
- **`edullm status` tops out at `ADMITTED`.** It did not report a terminal state for this
  run at any point, so waiting for one is waiting for something that does not arrive.
  Absence of `SUCCEEDED` is not evidence.
- **`edullm logs <full-run-id>` is the only live view of the container**, and it is not
  free in the way `status --json` is. It dispatches `cancel-run.yml`, which holds the
  `run-canceller` role — the one identity granted `batch:DescribeJobs` and
  `logs:GetLogEvents` on the two Batch log groups — and prints the stream's last 50 lines
  into the workflow's step summary. Nothing is cancelled unless you tick `stop`. It costs a
  GitHub runner minute per call, so **check S3 first and reach for logs only once absence
  has stopped being informative.**

**For a fan-out, `edullm logs` cannot read a cell at all.** The workflow resolves the run
id to one job and reads `container.logStreamName` off it; an array parent runs no
container, so that field is empty and the workflow correctly reports that there is no
stream yet. It is not a fault and there is no flag for it — a cell's output is reachable
only through the broker, by log stream.

---

## Step 8 — read the report

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
- `metadata.generation_runtime` and `metadata.generation_budget` — **generative runs only**,
  and they are how you check that the run was administered the way you think. Between them
  they record `context_length_declared` with the `context_length_source` that supplied it,
  `dtype_requested`, `kv_cache`, `eos_token` with `eos_token_id_passed_to_generate`,
  `exemplars_end_with_eos`, the set of `num_fewshot_used`, `items_over_context`, and a
  budget breakdown counting how many items came from the cascade against
  `context_clamped`. On `run_019fe78d` that read `dataset.sequence_length` / 2048,
  `num_fewshot_used: [4]`, and 39 cascade against 1 clamped.

Then read [`RESULT_CAVEATS.md`](../../../RESULT_CAVEATS.md), which is the companion to this
list and is not summarised here on purpose. Its entries name the run each was found in, so
cite the entry rather than restating it. The ones that change how a report is read today:
`gpqa`'s theta measures a constant-"A" responder rather than knowledge,
`ability.standard_error` is not the quantity the stopping rule compared against 0.3, the
`ifeval` echo, and the two `leaderboard_math` entries the next section acts on.

### A MATH theta is not a result until you have checked two things

An agent will otherwise report a number, and on this bank the number arrives looking
entirely healthy: no `ungradable`, no alert, an SE that would pass a glance.
`run_019fe78d` returned **theta -1.6523, se 0.5475, `observed_accuracy` 0.0 over 40 of
1,183 items, `stop_reason: max_items_reached`** — and that theta had been computed *before
the run, from the bank alone*, as -1.652 ± 0.547. The agreement to four significant
figures is the whole point.

- **Did it score anything?** On a checkpoint that scores zero, theta is the standard-normal
  prior conditioned on all-wrong — a statement about where this bank's difficulties sit, not
  a measurement of the model. The bank's information gives an SE floor of 1.01 at theta -2,
  so any all-wrong session returns approximately that same pair, and **two checkpoints that
  both score zero return the same theta and it will not order them.** Quote nothing.
- **Did the completions contain an answer at all?** `MathLatexEquivalence` reads exactly
  two forms, a `\boxed{}` expression or the Minerva `Final Answer:` line. **Zero of the 40
  completions contained either**; the decode looped a clause until the budget ran out, so
  every item scored 0 for a formatting reason rather than a mathematical one. Nothing in
  the report says so: `extract_math_answer` falls through to normalizing the whole
  completion, so `extracted_answer` reads `'Thefollowingissimplifiedsolutiontothe'` in the
  same field, with the same shape, that a genuine `\boxed{}` read would occupy. The
  runner does warn per item — "Repeated across a run this is the prompt convention failing
  to take, not a weak checkpoint" — but nothing aggregates it into the report, so the check
  is yours: grep the `completion` fields for `\boxed{` and `Final Answer` before quoting a
  MATH theta, and say the run measured extraction rather than mathematics if neither
  appears.

Both are written up in [`RESULT_CAVEATS.md`](../../../RESULT_CAVEATS.md); cite it rather
than restating it at length to a user.

---

## Worked example — the MCQ sweep

**The five inputs, as given.** (1) `s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/run_019fce1a-f393-70e3-ba0e-e2771c70f9c0/checkpoints/step305176/`.
(2) An inference example loaded with `ai2-olmo-core` 2.5.0, whose tokenizer
`HuggingFaceTB/SmolLM2-135M` reports `pad == eos == bos == 0`, and whose weights measured
546.2 MB for 135M parameters — fp32, against a command line that said bfloat16. The example
did not settle whether this checkpoint emits EOS, which is the case Step 1 exists for: the
64-token probe `run_019fe316-0e47` settled it, and settled the degeneracy question with it —
token 0 appeared in none of the 192 tokens it decoded, and all three decodes were
degenerate, ifeval echoing its own instruction back.
(3) All MCQ banks. (4) Defaults, with Warm's estimate reported instead of EAP.
(5) `gpu-1xl4`.

Input 2 settles two things before anything is written: the pad/eos collision is why
`--checkpoint-kind olmo_core` needs the fabricated pad id, and the fp32 measurement is why
`--dtype bfloat16` is a real change here rather than a restatement — a rerun is not
expected to reproduce the fp32 theta. Neither the EOS answer nor the echo costs anything on
this submission, because MCQ scoring is forward-only and never decodes; both are what price
a generative bank, and the echo is what withdrew `ifeval` outright. Input 3 plus Step 3's
preflight gives five MCQ banks at the pinned sha, so it becomes a five-cell fan-out rather
than five submissions.

The spec is [`.edullm/run-native-cat-sweep.yaml`](../../../.edullm/run-native-cat-sweep.yaml),
already committed, pinning `49d93da4…` — which is now several commits behind, so Step 5
applies before this is submitted again rather than after:

```yaml
schema_version: 1
workload_profile: olmo-core-check
suggested_compute: gpu-1xl4
fanout:
  size: 5
  index_parameter: benchmark
command: >-
  bash -lc 'BENCHMARKS=(arc_challenge hellaswag musr bbh gpqa)
  ; CELL="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"
  ; BENCH="${BENCHMARKS[$CELL]:-}"
  ; if [ -z "$BENCH" ]; then echo "cell $CELL has no benchmark; fanout.size must equal ${#BENCHMARKS[@]}" >&2; exit 2; fi
  ; echo "cell $CELL -> $BENCH"
  ; if ! command -v git >/dev/null; then apt-get update -qq && apt-get install -y -qq --no-install-recommends git; fi
  && git clone --filter=blob:none https://github.com/edu-llm/olmo-eval-full.git /opt/olmo-eval-full
  && cd /opt/olmo-eval-full
  && git checkout 49d93da4534169b57d4ff6252fdd954323ab6e9d
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
off this page** — they were recorded on 2026-08-08 against the four-cell version of this
spec, before `gpqa` was reclassified and the array grew, and they live in configuration
that moves. `cells` alone dates them:

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

## Worked example — the generative one, which has been run

Same checkpoint, same hardware, same five inputs except (3) `leaderboard_math` alone and
(4) defaults with `batch_eap` reported. The spec is
[`.edullm/run-native-math.yaml`](../../../.edullm/run-native-math.yaml), committed in
`a18d44f1`, and its header records the three things it does not share with the MCQ specs:
why one cell rather than a fan-out, where the 2048 window comes from, and why the reported
estimator is `batch_eap`. It pins `ff816927`, the first pushed commit carrying the native
completer, so Step 5 applies before it is submitted again.

```bash
cd ../OLMo-core
edullm submit --experiment native-olmo-core-math --dataset none \
  --workload olmo-core-check --compute gpu-1xl4 \
  --commit 08df5aa0142465c80b4ea48e84faa46117275d61 \
  --spec ../olmo-eval-full/.edullm/run-native-math.yaml
```

`--commit` is on that line and is **not** in the spec's header, which shows the command
without it. Omitted, the CLI takes the OLMo-core clone's current HEAD, and different
`edullm/*` branches build materially different images — the image is what supplies
`olmo_core` on the container's `PYTHONPATH`, so it decides whether
`TransformerConfig.from_dict` parses the checkpoint at all.

What happened, as expectations rather than as numbers to quote — read `check` for the
current ones:

- **It auto-approved.** One cell, so `classify_request` routed it to `automatic` and the
  CLI reported that it was "released automatically. Nothing is waiting on a person." The
  five-cell sweep above waited about sixteen minutes for a lead at comparable money.
- **About 22 minutes wall clock and roughly $0.30 against a $0.80 ceiling.** Decode ran at
  39.6 tokens/sec with no KV cache — 1,024 tokens in about 25.8s, so roughly 17 minutes for
  40 items — and the balance was the clone, the pip install and the 1.74 GB pull. Scale
  that rate, not the wall clock, when a submitter's checkpoint is larger: it is what the 1h
  per-cell bound binds against, and there is no second attempt.
- **The report landed at the run's S3 output prefix, once, at the end.** Nothing appeared
  there while it ran. A pulled copy sits beside this repository in
  `cat_runs/run_019fe78d-math/`, uncommitted.
- **The result was theta -1.6523 at se 0.5475 with 0 of 40 correct, and is not quotable**
  — Step 8 says why, and it is the part of this example most likely to be misreported. What
  the run establishes is that a raw training checkpoint can be graded generatively end to
  end, which is what a future submitter needs and what no offline test can show.

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
  meaningless. **`ifeval` in particular is not available on any request**, however the user
  frames it; relay the `blocked` reason and offer the six ready names.
- Do not compare theta across benchmarks. Predicted accuracy is the cross-benchmark
  quantity.
- Do not present a theta from a report carrying an `ungradable` alert as a measurement of
  the model.
- Do not submit a generative bank without having read a real decode from this checkpoint for
  an echo — the submitter's sample output, or Step 1's — and do not report the theta if you
  skipped it. That failure carries no alert, an `ungradable.rate` of 0.0 and the tightest SE
  in the sweep.
- Do not convert a checkpoint to get a first decode out of it. Step 1 is native by
  construction, and `save_hf_model` refusing an unfamiliar block is a fact about the exporter
  rather than about the checkpoint.
- Do not quote a `leaderboard_math` theta without checking that the completions contain
  `\boxed{}` or `Final Answer`, and do not quote one at all from a session that scored
  zero. Both failures produce a healthy-looking report; see Step 8.
- Do not poll `edullm logs` while a run is in flight. It dispatches a workflow and buys the
  last 50 lines; S3, which is free, answers the only question you usually have.
- Do not quote a price, a runtime bound, a cost ceiling or an approver from memory or from
  a document — including this one.
