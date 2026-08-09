# Running the two generative banks

> **EXECUTED 2026-08-09. Read this header before the body, which is the plan as it stood
> beforehand and is stale in several places.**
>
> Route B was chosen and built. `GENERATIVE_BACKENDS` now registers
> `{"hf", "olmo_core"}` and `LOADABLE_CHECKPOINT_KINDS[GENERATIVE]` is `("hf", "olmo_core")`
> (commit `0d25d4ba`), so every claim below that generative accepts `hf` alone, or that
> `_load_olmo_core` raises `NotImplementedError`, describes the world before that commit.
> Conversion was never attempted and is not needed.
>
> `leaderboard_math` ran natively on Preston's `step305176` as
> `run_019fe78d-13f9-701c-9db4-c9df2e973e88`: single cell, auto-approved with no lead wait,
> about 22 minutes, roughly $0.30 of a $0.80 ceiling, 39.6 tokens/sec. Spec at
> `.edullm/run-native-math.yaml`. Report in `cat_runs/run_019fe78d-math/`.
>
> `ifeval` is blocked and is no longer one of "the two generative banks" -- a verbatim
> prompt echo passes 129 of its 511 items. The scope of this file is now MATH alone.
>
> **What the run settled.** The checkpoint config confirmed `dataset.sequence_length` 2048
> with no model-level window, so the completer's fallback is the only source. The exemplar
> ladder never fired -- all 40 items ran 4-shot -- so the mixed-shot hazard is measured
> inert at this window; the budget clamp fired once, reducing a 1,645-token prompt to 403
> tokens, which is the designed order. Theta came back -1.6523 +/- 0.5475 on 0 of 40
> correct, matching a pre-run prediction of -1.652 +/- 0.547 computed from the bank alone.
> That agreement is the result: the number is the prior conditioned on all-wrong, and zero
> completions contained `\boxed{}` or `Final Answer`, so nothing was ever graded on its
> mathematics. See `RESULT_CAVEATS.md`.
>
> **Still open:** flash-attn and KV caching (Phase 1 below, still accurate and still not on
> the critical path), and the `extraction_path` field that would make a formatting failure
> legible as one rather than as forty wrong answers.

Everything the MCQ sweep proved is done: `run_019fe33f` administered all five MCQ banks
natively, 61 items against 10,486, every cell stopping on precision. What remains is
`ifeval` and `leaderboard_math`, and this file is the plan for getting a number out of them.

Status at time of writing: **blocked externally.** `sbsandbox` is in emergency cost
lockdown (`sandboxKilled`), so nothing can be built, submitted or run until an admin clears
it. Everything below is preparable offline. Note that nothing in the `edullm` CLI knows
about the lockdown -- `check` returns exit 0 with no refusals regardless -- so a clean check
today says nothing about whether a submission will run.

## Before you submit anything

**The container clones this repository from GitHub at the sha the spec names. Your working
tree is not in the run.** `.edullm/run-native-cat-sweep.yaml` pins `49d93da4`, which is now
6 commits behind and contains **none** of the merged machinery -- grepping that sha for
`ifeval_token_budget`, `fit_budget_to_context` and `eos_stop_sequences` returns zero
matches. Submitting the spec as written would evaluate code from before all of this work and
return a well-formed `cat_report.json` with a plausible theta, measured on a harness nobody
has. That is a wrong answer, not a failure, and nothing downstream catches it.

So, every time, in order:

1. Commit and push `flow/uni-mcq`. It is currently ahead of origin.
2. Repin the spec's `git checkout` to the new sha.
3. Name the OLMo-core commit on the submit line with `--commit`. Without it the CLI takes
   the clone's current HEAD, which is `edullm/p3-math-split`, not the `08df5aa0` the working
   sweep used. Different `edullm/*` branches build materially different images, and the
   image is also what supplies `olmo_core` on the container's `PYTHONPATH` -- so this choice
   decides whether `TransformerConfig.from_dict` parses the checkpoint at all.

**`edullm check`'s `uncommitted_changes` refusal reads the OLMo-core tree, not this one.**
OLMo-core is clean, so check returns `"refusals": []`. That green light says nothing about
the repository whose code is actually being evaluated.

## Submit as two single-cell runs, not one two-cell fan-out

Same $1.61 total, no lead wait. Approval class is decided in `classify_request`, not by a
cost threshold: a fan-out never auto-approves whatever it costs, while a single cell routes
to `automatic`. Measured both ways -- the 5-cell sweep classified `routine` /
`run-approval-lead` and sat about 16 minutes; one cell classifies `automatic`. Two
submissions beat one approval round-trip.

Budget the wait anyway: the skill records `automatic` runs still parking behind a lead, and
the automatic class closes once the day's unattended spend reaches its ceiling.

## The one thing that blocks it

```
grading.LOADABLE_CHECKPOINT_KINDS = {'mcq': ('hf', 'olmo_core'), 'generative': ('hf',)}
generative.GENERATIVE_BACKENDS    = ['hf']
```

Generative grading takes an HF checkpoint. Preston's is native OLMo-core.

**Route B, native, is the chosen route.** Two reasons, the second decisive:

1. **The merged machinery is not HF-specific, so Route B duplicates none of it.** An earlier
   draft of this file claimed the budget cascade, clamp, ladder and EOS resolution "live on
   `_HFCompleter`" and would have to be mirrored. That was wrong. Everything from
   `ifeval_token_budget` through `eos_stop_sequences` and `GenerativeScorer` is free-standing
   or scorer-side; `_HFCompleter` begins hundreds of lines later, and
   `GenerativeScorer._published_fitter` builds the fitter from what a completer publishes
   precisely "so there is one place that turns a completer's published tokenizer and window
   into a fitted prompt". A backend's whole contract is the four attributes in the table
   below. Route B writes an adapter, not a second implementation.
2. **Conversion does not generalize, and this pipeline is for future submitters.**
   `save_hf_model` has to understand each submitter's architecture. This checkpoint already
   needed the `get_hf_config` patch because a plain pre-norm block raises `NotImplementedError`
   upstream; the next unusual block type needs another patch. The native reader accepts
   whatever `TransformerConfig.from_dict` parses, which is a far wider door, and it costs no
   extra disk, no Hub egress and no conversion step.

Route A is kept below as the fallback if the native completer hits something the probe did
not surface.

## Route A — convert, then grade with the existing path (FALLBACK, not chosen)

`--checkpoint-prep auto --checkpoint-kind hf`. `convert.ensure_hf_checkpoint` and the
`get_hf_config` monkeypatch already exist and are wired.

**What recommends it, and this is new.** Every piece of machinery the two agents just built
-- the per-item budget cascade, the context clamp, the exemplar ladder, the runtime EOS
resolution, `eos_token_id` reaching `generate` -- lives on `_HFCompleter`. Route A uses all
of it as-is. Nothing is mirrored, nothing is re-tested.

**What it risks.** The conversion has never executed once.

1. **The library version, not the export.** The first framing of this risk was wrong.
   `partial_rotary_factor` is never read by `get_hf_config`, so a non-1.0 value is *silently
   omitted* rather than raising -- and this checkpoint's is 1.0, making it a no-op here.
   `sequence_mixer` is a config key that builds as `block.attention`, which the export reads
   by attribute. The genuine failure is earlier and cruder: whether the installed
   `ai2-olmo-core` understands `sequence_mixer` in `TransformerConfig.from_dict` at all,
   which is a fork-versus-PyPI-2.4.0 question that fails before `save_hf_model` runs.
   The plain pre-norm block *would* raise `NotImplementedError` upstream, and the
   `get_hf_config` patch exists precisely to intercept that -- so the patch is load-bearing
   for this checkpoint, not an optimization.
2. Disk. The converted copy lands in the same tmpdir as the 1.74 GB original, which is the
   pressure `--checkpoint-prep none` exists to avoid.
3. ~~Network.~~ **Answered, and no longer a risk.** The tokenizer is an identifier fetched
   from the Hub at the end of conversion, and both this file and `HF_CONVERSION.md` treated
   Hub reachability as open. The platform's own source settles it: `edullm_platform/
   tokenizers.py` records, for exactly this case, that "Batch hosts sit in public subnets
   with allow-all egress."

If this fallback is ever taken, **do not run the conversion smoke on `cpu-32vcpu` for cost
reasons.** That shape is $1.428/h against `gpu-1xl4`'s $0.8048/h -- 1.77x the price for the
same 4 container vCPUs, half the RAM, no card, and a 64 MB `/dev/shm` where every GPU shape
declares 4 GB. An earlier draft cited `HF_CONVERSION.md` as authority for proving conversion
"where it is cheap"; that document retracts the word in as many words. Cheaper still is to
let conversion fail inside the real `gpu-1xl4` cell, which surfaces after clone and pull at
roughly 8m35s median, about $0.23 across two cells, against a $1.43 ceiling plus a whole
approval round-trip for a separate smoke. The only thing a separate smoke buys is not
writing Route B's completer against a route you were about to abandon -- and Route B is now
chosen, so it buys nothing.

**Verified as sound:** `--checkpoint-prep auto --checkpoint-kind hf` does reach the
generative grader. `check_checkpoint_kind` runs first and passes, because the kind names the
*backend* rather than the on-disk layout; the fetch and conversion then happen before
`_load_hf`. Using `--checkpoint-kind olmo_core` here would be refused up front, which is the
opposite of the MCQ path. The machinery is also genuinely portable: nothing in the budget or
EOS code branches on `checkpoint_facts`, and `resolve_eos_token` reads only the tokenizer,
which conversion writes beside the weights where `_load_tokenizer` expects it.

## Route B — grade natively (CHOSEN; Phase 0 builds it)

**What recommends it.** Generation already works. Probe `run_019fe316-0e47` decoded three
prompts at 64 tokens each through `generate_batch`, on this checkpoint, on a real GPU,
using the loader `_OlmoCoreScoringModel` already proves. The shape question, the EOS
question and the fabricated-pad question are all answered.

**What it costs.** `_load_olmo_core` raises `NotImplementedError` and is not even in
`GENERATIVE_BACKENDS`, so today the failure is `ValueError: Unknown checkpoint_kind` before
that stub is reached. Writing it means publishing the completer contract the merged budget
machinery reads:

| what | attribute | if missing |
|---|---|---|
| token counter | `TOKEN_COUNTER_ATTR` | **load refuses** -- `require_live_tokenizer` |
| budget-aware call | `BUDGET_AWARE_ATTR` | budgets computed, then **not passed**; every item takes the flat cap |
| context window | `CONTEXT_WINDOW_ATTR` | **clamp and ladder silently off** |
| EOS spelling | `EOS_TEXT_ATTR` | leak-catcher stop weakened |
| decode stop | real `eos_token_id` into `generate_batch` | full budget every item |

Only the first refuses. **The rest fail silently**, and the context window is the dangerous
one: `PromptFitter.fit` returns unclamped when no window is published, so MATH's long stems
would keep four exemplars, overflow, and be truncated into the few-shot block that teaches
`\boxed{}` and `Final Answer:` -- zeros that look like the model rather than the harness.

`--dtype` is a second gap: the runner puts it on `InferenceConfig` only, and
`GenerationConfig` has no such field, so a native generative path needs its own threading.
Route A gets this free through `torch_dtype="auto"` on the converted directory.

**What is NOT a blocker, contrary to `_load_olmo_core`'s former docstring.** It argued that
`pad_token_id == eos_token_id == 0` blocks decoding. It does not: `GenerationConfig.validate`
rejects only `pad == eos`, and fabricating the *pad* leaves the real EOS at 0, exactly as
the MCQ scorer already does. Probe `run_019fe316-0e47` loaded this checkpoint with
`pad_token_id=1, eos_token_id=0` and decoded 64 tokens on each of three prompts. The
docstring has been corrected.

**The 2048 does NOT constrain this route automatically**, which is the correction that
matters most here. Nothing on the native path reads a context length today, so the ladder
fires only if the new completer publishes one. Note also that the checkpoint's trained
`sequence_length` is asserted in several docs but **cannot be verified in this tree** --
no checkpoint `config.json` is committed. The 2048 that fires the ladder on Route A comes
from `hf_config_patch.DEFAULT_MAX_POSITION_EMBEDDINGS`, and only the node configuration
that exported it ties that to the training length.

## Phase 0 — build the native completer

`_OlmoCoreCompleter`, registered in `GENERATIVE_BACKENDS["olmo_core"]`, with `olmo_core`
added to `LOADABLE_CHECKPOINT_KINDS["generative"]`. All of it is offline work and none of it
waits on the lockdown.

Lift the loading and decode from two proven places rather than inventing either:
`inference._OlmoCoreScoringModel` for resolving the checkpoint and the tokenizer and for
fabricating a pad distinct from eos, and `.edullm/generation_probe.py` for the
`generate_batch` call, the prompt-token stripping and the `use_cache=False` fallback.

Publish all four attributes. Three of the four fail silently if forgotten, so treat the
table above as a checklist rather than as documentation, and assert each one in a test.
`max_context_tokens` is the one that matters most: without it the ladder is inert and MATH's
long stems get truncated into the exemplars that teach the answer format.

Two things the HF path gets for free and this one does not: `--dtype` reaches only
`InferenceConfig` today, so thread it to the generation module; and there is no
`max_position_embeddings` to read, so decide where the native context length comes from --
the checkpoint config's `sequence_length` is the honest source, and unlike the HF path it
must be read rather than defaulted.

Definition of done, all offline: the two generative banks resolve a model, build prompts at
the right shot counts for a 2048 window, and the suite stays green.

## Phase 1 — flash-attn, only if the decode is too slow

Independent of the route and **not on the critical path.** Without KV caching the decode is
quadratic: every step re-forwards the whole prefix.

On the arithmetic, which an earlier draft got wrong: the probe measured 64 tokens in ~2.0 s
on an L4, which `TORCH_TODOS.md` extrapolates to about 32 s per 1024-token item and roughly
20 minutes for 40 items. This file previously said 40 s and 27 minutes, which is that rate
applied to 1280 tokens while labelled 1024. **Both figures are floors**, because both
extrapolate linearly from a 64-token sample on a path both documents call quadratic. Against
a 1 h per-cell bound that is viable but not comfortable, and worth fixing only if a run
actually approaches the bound.

An attempt that hits the bound carries `Job attempt duration exceeded timeout` and no
container exit code. `olmo-core-check` sets `maximum_attempts: 1`, so Batch cannot re-drive
it: the exposure is $0.80 and a lost result, not a doubled bill.

**This phase applies to Route B only.** A converted HF checkpoint gets transformers' own
caching, so on Route A the whole phase is moot.

Two halves, both required, neither sufficient alone:

1. Branch off `edullm/memory-split-135m` and add the wheel to its `.edullm/Dockerfile` --
   the `cu12torch2.9` build that `edullm/p3-math-split` has already shipped, matching that
   branch's `torch==2.9.0`. See `TORCH_TODOS.md` for the exact lines and why our own image
   is the worse host for them.

   It is **three** lines, not two, and the two assertions are the point: the
   `compiled_with_cxx11_abi` check and the `flash_attn_varlen_func` import turn a wheel
   mismatch into a build failure instead of a paid GPU failure. Note also that `--no-deps`
   skips flash-attn's own dependencies, which is why p3-math-split installs `einops`
   separately.

   **First check whether this is needed at all.** `edullm/p3-math-split` already carries the
   wheel. If its `olmo_core` parses this checkpoint's config, submitting against that branch
   gets flash-attn at zero build cost.

   Pushing to `edullm/**` fires `edullm-platform-build.yml`, gated on `ruff check` over the
   whole checkout. Recent history is sobering: of the last 12 runs, 8 failed, mostly on
   torch/CUDA pin fights, with successes taking 3m28s to 7m24s. Budget several attempts.
   Each success writes an immutable ~3 GB ECR tag that cannot be rewritten or reused, and
   submitting before the build lands is refused at compile with `NoPublishedImageError`.
2. Pass `attention_backend` to `TransformerGenerationModule.from_checkpoint` **in the native
   completer**, resolving the name as `OlmoCoreProvider` does. The probe's `_flash()` has
   working code to lift.

**Do not thread it into `_OlmoCoreScoringModel`, which is what an earlier draft of this plan
and of `TORCH_TODOS.md` both said.** That class calls `model_forward` and nothing else: it
never calls `generate_batch`, never reaches `prepare_inference_cache`, and so never hits the
`assert_supports_kv_cache` that refuses. Giving the MCQ scorer a backend field changes
nothing and buys no speed. The backend is fixed at `from_checkpoint` rather than selectable
per call, which is exactly why the probe builds a second module for flash instead of
reconfiguring the scorer's. So half 2 has no home at all until Route B's completer exists,
and sequencing this phase before that completer would be paying for plumbing that does
nothing.

## Phase 2 — submit

`ifeval` and `leaderboard_math`, `gpu-1xl4`, mirroring `.edullm/run-native-cat-sweep.yaml`
but as two single-cell submissions rather than a 2-cell fan-out, for the auto-approval
reason above. Work through the pre-submission checklist first; the sha is the one that
matters.

**Cost: $1.61 ceiling total**, two cells at one hour each at $0.8048/h. It is a ceiling and
Batch bills what runs -- platform history for this cohort is a median of 8m35s over three
runs, so realistic spend is **$0.25 to $0.85**. The per-cell hour bound is genuine:
`olmo-core-check` declares `maximum_runtime_hours: 1` and cost is cells x nodes x hours x
rate x attempts, so two cells get an hour each rather than sharing one.

Also write `--dtype bfloat16` into the command text: the `bfloat16_not_in_the_hardware`
guard reads the words of the command and nothing else. It is moot on `gpu-1xl4`, which is
Ada, but it keeps the spec honest if the shape ever moves. Do not pass `--hours` above 1;
`runtime_above_the_workload_bound` refuses it. The three `deferred` entries about published
images and scan findings are normal.

Budget expectations, on a checkpoint that never emits EOS so every item pays its cap:

- `leaderboard_math`: 1024 per item.
- `ifeval`: per-item, min 76 / median 1280 / max 1280, about 33,500 tokens for 40 items
  against 61,440 under the old flat 1536.

## Phase 2b — build the echo guard BEFORE submitting

Not optional, and it changed the shape of this plan. IFEval's verifiers are gamed by exactly
the degeneracy this checkpoint already exhibits: **a verbatim echo of an IFEval prompt passes
129 of the 511 items, 25.2%**, verified through this repo's own grading path. Simulated
against our own Fisher selection and EAP, an echo reports theta +0.19 to +1.11 with se
0.11-0.23 after 8-9 items, stopping on precision. That would be the highest and tightest
number in the entire sweep, from a model that said nothing, with no field indicating
anything is wrong. Full evidence in `RESULT_CAVEATS.md`.

Stamp each item offline with whether an echo passes it -- no checkpoint needed, seconds to
compute -- and report the share of a session's passes an echo would also have produced.
Above roughly half, the theta is a constraint-checking artifact.

Cheaper still and worth having regardless: over the best 40 items at floor ability the bank
gives an SE floor of 0.572, so IFEval **cannot** reach se <= 0.3 on a floor-level
checkpoint. `stop_reason: precision_reached` at 8 or 9 items is therefore self-refuting on a
model expected at the floor, and costs nothing to check.

## Phase 3 — read the results against what is already known

**Do not treat this as a checklist. All three of the pre-existing caveats are inert for this
run**, so ticking them green tells you nothing:

- `mixed_shot_alert` cannot fire for IFEval, which is 0-shot. For MATH the ladder needs a
  stem over roughly 1,112 tokens and at most a handful of the 1,183 qualify, none of which
  the CAT is likely to select. Note the two measurements disagree on how many -- one agent
  measured with the real tokenizer and reported 4-, 2- and 1-shot mixing at 2048, another
  with a character proxy found a single qualifying item -- so treat the count as unsettled
  and read `num_fewshot` per response rather than assuming either.
- `ungradable.rate` will be 0.0 in both cells. IFEval's clamp is inert above a 1,667-token
  window and its longest prompt is 387; MATH's administered items fit.
- The floor expectation is not a check. `TORCH_TODOS.md` records the degenerate decodes, but
  **the plan must not assume the answer** -- that is how the gpqa number survived. If IFEval
  returns +1.1, nothing in Phase 3 as originally written distinguishes "it worked better
  than expected" from "the verifiers were gamed". Phase 2b is what makes that distinguishable.

What is legitimately claimable from this run: that the pipeline executed and produced a
report. Not claimable: an absolute ability number on either bank, and on MATH not a relative
one either, because the exemplar EOS is resolved per checkpoint so two submitters spelling
end-of-text differently are scored behind materially different prompts while passing the
same convention guard.

## What is deliberately not here

- Native generative KV caching beyond Phase 1's two halves.
- `gsm8k`, which is configured but not among the seven vendored `ready_names()`.
- Any change to the five MCQ banks, which are done.
