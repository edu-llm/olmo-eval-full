# Plan — convert-then-score: replacing the native OLMo-core loader

Supersedes the OLMo-core spike (`.edullm/olmo_core_spike.py`) and the Phase 2 native
loader built against it. The checkpoint is converted to HuggingFace format on the node
and scored by the existing `_HFScoringModel`, rather than read natively by a second
scorer written against OLMo-core's internals.

## Design constraint: two seams, three futures

**Nothing below may hardwire "convert to HF and score with transformers" as the only
path.** Conversion is one policy and transformers is one backend, and the code says so
structurally, because two other configurations are live possibilities: running the native
OLMo-core checkpoint with no conversion at all, and serving the converted checkpoint
through vLLM. Both must be reachable by changing a flag and adding one function, never by
restructuring the runner.

The two things that vary are independent, so they get separate seams rather than one
combined mode switch.

**Seam 1 — preparation.** What happens to the staged directory before anything loads it.
`prepare_checkpoint(local_dir, work_dir, *, policy) -> Path` in
`diagnostics/mcq_cat/common/convert.py`, selected by `--checkpoint-prep`:

| Policy | Behavior |
|---|---|
| `auto` (default) | detect the layout; convert OLMo-core to HF, pass HF through |
| `none` | hand whatever is on disk to the backend untouched |

`none` exists from day one even though nothing needs it yet, because it is what makes the
native path a flag rather than a revert. A future policy — convert-and-quantize, or
stage-and-serve — is a new entry in this table, not a change to the runner.

**Seam 2 — backend.** What loads the prepared directory and grades items. This seam
already exists as the `checkpoint_kind` dispatch inside `load_scoring_model`; it is an
`if`/`elif` chain today and becomes a dict registry, mirroring `GRADERS` in `grading.py`,
which is already written that way.

Together they compose:

| Configuration | `--checkpoint-prep` | `--checkpoint-kind` | What it costs to enable |
|---|---|---|---|
| Convert-then-score | `auto` | `hf` | this plan |
| Already-HF checkpoint | `auto` (no-op) | `hf` | works today |
| Native OLMo-core, no conversion | `none` | `olmo_core` | `git stash pop`, one registry entry |
| vLLM | `auto` | `vllm` | one class implementing `score_items` |

Three rules follow, and they are called out again in the phases that implement them.

**The registry is data, not control flow.** `load_scoring_model` and
`load_generative_model` become `dict[str, Loader]` lookups whose `KeyError` names the
registered kinds. Adding a backend is a line in a dict; a kind registered for MCQ and not
for generative is expressible, which matters because that asymmetry is exactly the state
the native path was in.

**`LOADABLE_CHECKPOINT_KINDS` keeps its per-modality shape.** The parked work turned it
from `("hf",)` into `{MCQ: (...), GENERATIVE: (...)}` for good reasons that survive this
pivot even though its `olmo_core` entries do not. Keeping the dict with both entries at
`("hf",)` today means restoring the native path is a one-word data edit rather than
redoing that refactor. This revises what Phase 0 would otherwise have reverted.

**The runner closes what it opens.** `ScoringModel` is a `Protocol` with a single
`score_items` method and no lifecycle, which is correct for two in-process backends and
wrong for a served one — vLLM has a subprocess to tear down. Rather than widening the
protocol and forcing a no-op `close` onto both existing scorers, the runner calls `close()`
if the loaded object has one. Three lines now, and no runner change when vLLM arrives.

## Why this replaces the spike

The spike existed to discover OLMo-core's API contracts empirically, so a native loader
could be designed against a written record rather than against assumptions. The loader
was then built, and its own report lists roughly ten things that stay unverified until it
meets a GPU — what `model_forward` returns for a batch of one, what the real tokenizer's
defaults prepend, how bf16 log-softmax without an intermediate `.float()` compares to the
provider's cast. The last of those decides whether the ATLAS difficulties still apply to
the numbers coming out, which is the whole measurement.

Conversion removes the question instead of answering it. `_HFScoringModel` loads through
`AutoModelForCausalLM`, which is the path every bank in `calibrated_datasets/` was
calibrated behind, so there is no second arithmetic to reconcile and no tokenizer
divergence to characterize.

It is not free. A native checkpoint has to be unsharded to disk once per checkpoint,
against a CAT session that administers 13-40 items — the tradeoff `.edullm/Dockerfile`
argues against at line 127. That argument is correct about cost and silent about risk;
this plan takes the cost.

### The reference

`origin/SteeringVectors`, nine commits dated 2026-08-07, is a working platform job that
converts an OLMo-core checkpoint on the node and loads it with transformers. It is not a
vLLM run despite the image flag — `run_steering_eval.py` never imports vLLM, and
`--extra vllm` survives in the run command only because `uv sync` is exact and dropping it
would remove the baked torch. What transfers is the conversion function, the `run.yaml`
shape, and the fork pin. What does not transfer is `INSTALL_TORCH_AND_VLLM=1`: this branch
already has torch via `INSTALL_OLMO_CORE=1`, and the vllm extra costs 4.54 GiB without
bringing `ai2-olmo-core`.

### The no-vLLM decision, and when to revisit it

Decided 2026-08-08: no vLLM anywhere in this plan.

MCQ grading is one forward pass per choice with the log-probs read off the output, so
there is nothing for an inference server to do. Generative grading does decode — up to
`max_new_tokens` sequential steps through `model.generate` at batch size 1 — and that is
genuinely the workload vLLM is built for. What it is *not* built for is this access
pattern: a CAT chooses item *n+1* from the ability estimate that item *n* produced, so the
items cannot be batched, and continuous batching is where vLLM's order-of-magnitude comes
from. The remaining gain is kernel and memory-management efficiency, which is real but
unmeasured here.

Revisit if Phase 7 shows the three generative benchmarks dominating wall clock. Measure
first; the reference for that path already exists in `add-cat-evals`, which drives
`olmo-eval run-external` over a served checkpoint.

## Phases

Ordered so that each one is verifiable on its own, and so that everything checkable
without a GPU is checked before anything is billed. Only phases 5 and 7 need a machine;
the other nine run here.

Phase 8 comes after 7 rather than beside it on purpose: the skill tells an agent what to
do, and until one real submission has happened there is no way to know whether what it
says is true.

| # | Phase | Needs | Verified by |
|---|---|---|---|
| 0 | Park the native loader | nothing | baseline suite green |
| 1 | Checkpoint reconnaissance | S3 read via sb-aws | fork decision recorded |
| 2 | Preparation seam (`auto`/`none`) | nothing | unit tests, no torch installed |
| 3 | Conversion body | nothing | import-guard test |
| 3b | Vendor the `get_hf_config` patch | nothing | patch tests, no torch |
| 4 | Wire both seams + backend registry | nothing | full suite green |
| 5 | Conversion smoke | CPU box with torch, not the L4 | HF dir loads |
| 6 | `run.yaml` | nothing | `edullm check` prices it |
| 7 | One live run | GPU spend | `cat_report.json` |
| 8 | Rewrite the skill for submission | nothing | cold agent submits unaided |
| 9 | Plan and README hygiene | nothing | placeholders replaced |

---

### Phase 0 — Park the native loader

**Purpose.** Return the tree to a known-good baseline before anything is added to it.
Preserve the native loader rather than deleting it, because it is the fallback if
per-checkpoint conversion costs more than the sessions it serves.

The 478 uncommitted lines on `flow/uni-mcq` come off:

```bash
git stash push -m "phase2 olmo_core native loader" \
  diagnostics/mcq_cat/common/inference.py \
  diagnostics/mcq_cat/common/grading.py \
  diagnostics/mcq_cat/runner.py
```

Two things do not go with it.

`diagnostics/mcq_cat/styles/uni_mcq/tests/test_runtime_guards.py` carries a real fix that
is independent of the pivot. `test_the_runner_refuses_before_fetching_the_checkpoint` was
passing for the wrong reason: `main` returns 1 for every failure, so the test's
`AssertionError` sentinel could not be distinguished from the guard actually firing. The
replacement asserts both that no fetch was attempted and that the logged message is this
guard's. Keep it.

`LOADABLE_CHECKPOINT_KINDS` keeps the per-modality dict and loses only its `olmo_core`
entries, becoming `{MCQ: ("hf",), GENERATIVE: ("hf",)}`. Reverting it to a flat tuple
would be the more honest description of today's behavior and the wrong call under the
design constraint above: the native path has to come back as a data edit, not as a repeat
of that refactor. `check_checkpoint_kind` keeps reading only the half matching
`request.modality`, which is what makes a kind registered for one modality and not the
other expressible at all.

**Done — 2026-08-08.** Stash `fed3d92551a0bf8be4074ca5f13d24b0c76bedf5`, message
`phase2 olmo_core native loader (parked for HF_CONVERSION plan)`. A stash is reachable but
unreferenced, so that SHA is the only durable handle on the fallback; `git show` it
directly if `git stash list` has moved on.

All five files were stashed rather than three. The guard test turned out to be entangled
rather than separable — the diff rewrote the whole class to assert that `olmo_core`
*reaches* the MCQ loader, which stops being true here — so the two keepers were
re-derived by hand against the clean baseline instead of being carved out of it:

- `grading.py` now carries the per-modality `LOADABLE_CHECKPOINT_KINDS` with both entries
  at `("hf",)`, `_CHECKPOINT_SOURCES`, and a `check_checkpoint_kind` that reads only the
  half matching `request.modality`.
- `test_the_runner_refuses_before_fetching_the_checkpoint` now asserts that no fetch was
  attempted and that the logged message is this guard's. Verified by mutation: with the
  guard's condition forced false, `assert exit_code == 1` still passes and the new
  `assert not fetched` is what fails — which is precisely the hole the old test had.

941 tests pass, unchanged from baseline, and ruff is clean on both files.

---

### Phase 1 — Checkpoint reconnaissance

**Purpose.** Settle the facts about Preston's checkpoint that determine what the converter
must contain: which OLMo-core can parse its config, whether its block type needs the
`get_hf_config` patch, whether its attention forces conversion onto a GPU, and how many
bytes the node has to move. Two were answered from the previous sweep's own writeup and
needed only confirming; the directory size genuinely required a call.

Target, the final checkpoint of the dense arm:
`s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/run_019fce1a-f393-70e3-ba0e-e2771c70f9c0/checkpoints/step305176/`

The arm spans two run IDs because training was interrupted and resumed once:
`run_019fc8e7` holds steps 0 to 247910 and `run_019fce1a` holds 251724 to 305176, with the
latter declaring a `load_path` into the former. 305176 is off the uniform 3814-step grid
and is the last of the 82. The split arm is the `run_019fc8e9` / `run_019fce2f` pair and is
not in scope. `hf-converter-patch/iam_policy_proposed.json` already carries a read grant
for exactly this step's prefix.

The model is **SmolLM2-135M**, which sizes everything downstream: roughly 0.27 GB of
weights at bf16, against `CONVERSION.md`'s measured table of 4-6 GB of RAM for a 1B. That
is the reason Phase 5 can use this checkpoint directly rather than hunting for a smaller
one.

**The route is the sb-aws broker, read-only — not the `aws` CLI.** `AGENTS.md` is flat
about it: "Do not write a script that calls AWS. No `boto3`, no `aws` CLI, no `curl` at an
AWS endpoint." The reason it gives is not squeamishness about credentials. For the people
here who hold no AWS role the CLI simply fails, and for the few who do it succeeds and
leaves no run anybody can cite — and it is the second half that bites here. A broker call
is attributable to a person and a session and can be pointed at later; the same call typed
into a laptop shell is a fact that existed for one terminal and then did not, which for a
phase whose entire output is three claims written into this file is the difference between
a finding and an assertion.

Concretely: `accounts` and `whoami` once to fix identity, then `aws` against account
`sbsandbox` — the ready one, and the sb-aws skill says to stay inside it unless the user
names another — with the command passed as a structured argument array rather than a
string. Both calls this phase needs are a list and a get, which that skill classifies under
its read-only default: they observe metadata, create nothing and move no protected data, so
neither needs separate authorization. The phase as a whole still does; nothing below runs
until the user says to.

**The tension, stated rather than smoothed over.** `AGENTS.md`'s prohibition sits under a
heading about running things on a GPU, and its worked failure is a script reaching for the
cluster. A read-only fetch of one `config.json` is arguably outside what that rule was
written to stop, and someone could make that argument honestly. There is no reason to. The
broker is correct under both documents where the CLI is correct under at most one, the two
cost the same keystrokes, and when one of two equivalent routes is defensible either way it
is not a close call.

**Two of the three were already answered, from a document in this workspace.**
`FINAL_PRESTON_RESULTS_BASE/README.md` is the writeup of the 82-checkpoint sweep over
*this arm* — `mode: base`, both roots, ending at the same step 305176 — and its Method
section records what that sweep needed to convert these checkpoints. It is evidence from a
run that completed on all 82, not a plan's guess, so the fork and the patch went in as
settled and the broker call was a confirmation rather than a discovery. The size was open:
it is question 2 in `CAT_EVAL_HANDOFF.md` and had never been answered.

**Done — 2026-08-08.** Three read-only broker calls against `sbsandbox`: a
`list-objects-v2` with `--delimiter /` for the layout, a recursive `list-objects-v2` for
the size — recursive because the shards sit a level down and a summary of the prefix alone
would report a confidently wrong small number — and a `get-object` for `config.json`.
Everything below is read off those.

**The layout, and 1.74 GB of it.** At the top: `.metadata.json` (40 B), `config.json`
(6,960 B), `data_paths.txt` (1,124 B), and the prefixes `model_and_optim/` and `train/`.
Recursively and untruncated the prefix holds 1,744,814,262 bytes across 140 objects, of
which `model_and_optim/` is 1,744,683,602 across 129 and `train/` — the dataloader's
position — is 122,536. Everything except the shards is noise: all but about 130 KB of the
directory is one prefix.

**The config is the arm the sweep's README described, at this step.** `_CLASS_` is
`__main__.ExperimentConfig`; `mode` is `base` and `mask_dir` is empty, so this is the dense
arm and not the memory-split one; `max_duration` is `{value: 305176, unit: steps}`, so this
is the last step rather than a mid-run checkpoint that happens to sit at the end of a
listing; and `trainer.load_path` points into `run_019fc8e7-.../checkpoints`, which is the
resume that split this arm across two run IDs. Nothing disagrees, so there was nothing to
stop on.

The architecture is SmolLM2-135M, as the sizing in Phase 5 assumed: `d_model` 576, 30
layers, 9 attention heads over 3 KV heads — so GQA — at `head_dim` 64, feed-forward hidden
size 1536 with `silu`, RMS norm at eps 1e-05, `vocab_size` 49152, and
`tie_word_embeddings: true`. That is roughly 134.5M parameters.

**The fork is required, and now for a reason read off this checkpoint rather than
inherited.** `block.sequence_mixer` is present at all, and its `rope` block carries both
`no_global_rope: false` and `partial_rotary_factor: 1.0`. PyPI `ai2-olmo-core==2.4.0` has
no schema for those fields, so `TransformerConfig.from_dict` fails on this config before a
single weight is read. `edu-llm/OLMo-core@edullm/memory-split-135m` is the branch that
wrote them, which is why Phase 6's sync line takes it.

**The patch is required.** `block._CLASS_` is
`olmo_core.nn.transformer.config.TransformerBlockConfig` with `name: "default"` — a plain
pre-norm `TransformerBlock`, not a `ReorderedNormTransformerBlock`. Upstream's
`get_hf_config` accepts only the latter, verified unchanged at
`OLMo-core/src/olmo_core/nn/hf/config.py:96-101`, so without the patch the conversion
raises `NotImplementedError` *after* rebuilding the model and reading 1.7 GB of shards into
it. Phase 3b is therefore unconditional, and so is the fork install in Phase 6's sync line
that has to precede it.

**The CPU-only constraint survives, which was the one finding that could have broken Phase
5.** `sequence_mixer.name` is `"default"`, its `_CLASS_` is
`olmo_core.nn.attention.AttentionConfig`, and its `type` is `"attention"`: not fused, not
flash-attention. `CONVERSION.md`'s caveat that either of those forces `device=cuda`
therefore does not apply here. `init_device="cpu"` in Phase 3's body is correct as written,
`convert_olmo_core_to_hf` needs no `device` parameter, and Phase 5 can go to the cheapest
CPU shape the catalog offers.

**Disk: about 2 GB coexist under the run's tempdir, and 85% of what the sync pulls never
reaches the output.** The run trained with `dp_config.param_dtype: bfloat16` against a
model `dtype: float32` and `reduce_dtype: float32`, so what is on S3 is fp32 master weights
plus AdamW's `m` and `v` — about 13 bytes per parameter across 134.5M of them, which is the
1.74 GB. Roughly 538 MB of that is the model at fp32 and roughly 1.08 GB is the two
optimizer moments, which the converter never reads: Phase 3 hands
`load_model_and_optim_state` a model and no optimizer, so its read planner selects only the
model tensors. The HF output is those weights at bf16, about 269 MB, so peak occupancy
under `--checkpoint-prep auto` is a little over 2 GB.

The 1.08 GB is a cost to know about rather than a bug to fix. The read planner skips the
optimizer *tensors*, but the sync that puts shards on disk works at object granularity and
DCP interleaves model and optimizer state inside the same `.distcp` files — 129 objects,
none separable by prefix or by name. Avoiding the transfer would mean teaching the fetch to
do a planned partial read directly out of S3, which is a different program from
`resolve_checkpoint`.

**The tokenizer is an identifier, so conversion needs the HuggingFace Hub and not only
S3.** `dataset.tokenizer.identifier` is `HuggingFaceTB/SmolLM2-135M`, with
`eos_token_id == pad_token_id == bos_token_id == 0`. There are no tokenizer files in the
checkpoint, so `_resolve_tokenizer_id` returns that string and
`AutoTokenizer.from_pretrained` resolves it against the Hub over the network. This is a
different grant from the S3 read, and it is the *last* thing the conversion body does —
after the shards have loaded and the weights have been written — so on a node that cannot
reach `huggingface.co` the failure arrives at the end of an otherwise successful conversion
and reads like a converter bug. Phases 5 and 6 both depend on that reachability. If a venue
turns out to be locked down, the fix is to pre-populate `HF_HOME` in the image, not to
debug the converter.

One consequence of those three zeros worth carrying into Phase 5: the emitted
`config.json` sets `pad_token_id`, `bos_token_id` and `eos_token_id` to `None`, because
that is what both upstream's OLMo-2 path and the patch do, and `save_hf_model` only fills
them in when it is handed a tokenizer object, which Phase 3's call is not. The ids reach
the output directory through the saved `tokenizer_config.json` instead. That is fine for
MCQ scoring, which reads log-probs off a forward pass, and is the first thing to check if a
generative benchmark later decodes without stopping.

---

### Phase 2 — The preparation seam, torch-free

**Purpose.** Build seam 1: decide what happens to a staged checkpoint before a backend
sees it. Keeping this half free of torch is what makes it testable at all, since none of
torch, transformers or `ai2-olmo-core` is installed on the development machine.

New module `diagnostics/mcq_cat/common/convert.py`. Detection ported from
`run_steering_eval.py:246-278` and `:345-357` on `origin/SteeringVectors`; the policy
wrapper is ours.

- `_is_hf_checkpoint(dir)` — weights (`model.safetensors`, its index, or
  `pytorch_model.bin`) **and** a tokenizer file.
- `_is_olmo_core_checkpoint(dir)` — `model_and_optim/` or `.metadata`.
- `ensure_hf_checkpoint(local_dir, out_dir)` — the ladder. HF passes through untouched.
  OLMo-core converts. A directory with a `config.json` but neither signature is allowed
  through with a warning, because letting transformers produce its own error is more
  informative than guessing. Anything else raises naming the path.
- `prepare_checkpoint(local_dir, work_dir, *, policy)` — the seam itself. `auto` delegates
  to `ensure_hf_checkpoint`; `none` returns `local_dir` unchanged and logs that it did.

`none` is not a placeholder and should not be written as one. It is the whole mechanism
by which the native path returns, and it logs rather than passing silently because a run
that skipped preparation and then failed inside a loader should say so in its own output.

The fourth detection branch matters more than it looks. A partially-synced checkpoint has
a `config.json` and no weights, which is exactly the shape the warning branch admits — so
the warning text should say what was found, not just that the layout was unrecognized.

**Tests** in `diagnostics/mcq_cat/styles/uni_mcq/tests/test_convert.py`: one fixture
directory per detection branch, asserting the dispatch with the conversion body
monkeypatched to a sentinel. Add a partially-synced fixture (config, no weights) and pin
which branch it takes, since that is the case a real S3 interruption produces. Pin
`policy="none"` against an OLMo-core fixture too — that assertion is the one that fails if
someone later "simplifies" the policy away.

**Done — 2026-08-08.** `diagnostics/mcq_cat/common/convert.py`, 21 tests in
`tests/test_convert.py`, all passing with no torch installed.

Two things came out differently than written. The OLMo-core detector took the reference's
richer form -- `model_and_optim/.metadata` *or* a config carrying `model` and `dataset`
blocks and **no** `architectures` key -- and that last clause turned out to be
load-bearing rather than incidental: without it a freshly converted directory answers
`True` to both detectors and which branch runs depends on the order they happen to be
tested in. It has its own test. And `describe_layout` was added so the pass-through
warning and the refusal both report what was actually on disk; a message that says only
"unrecognized" leaves the reader unable to tell a mistyped prefix from a dead upload.

Six mutations tried, six caught. The one that initially survived is worth recording: with
the "already HF" branch disabled, an HF directory falls through to the pass-through branch
and is *still returned unchanged*, so every assertion about the return value still held.
Only the absence of the warning distinguishes them, which is why that test now asserts on
the log.

---

### Phase 3 — The conversion body

**Purpose.** Turn a sharded OLMo-core checkpoint into a directory
`AutoModelForCausalLM` can open. Port it rather than invent it, so the three non-obvious
details come along instead of being rediscovered on a billed box.

`convert_olmo_core_to_hf(local_dir, out_dir, dtype)` in the same module, from
`run_steering_eval.py:281-342`. Written here, exercised in Phase 5.

`load_model_and_optim_state` expects a process group. A single gloo rank is enough, and
the reference initializes one from `MASTER_ADDR`/`MASTER_PORT`/`RANK`/`WORLD_SIZE`
defaults and destroys it in a `finally`. Without it the call fails in a way that reads
like a checkpoint problem.

`save_hf_model` raises `FileExistsError` on a directory that already exists, and the
directory has to exist first for the tokenizer save. Hence `save_overwrite=True`, which is
safe only because the directory is one we just made under the run's tempdir.

The tokenizer comes from `config["dataset"]["tokenizer"]` through `TokenizerConfig`, with
a dolma2 fallback, and is saved into the output directory so the result loads with
`AutoTokenizer.from_pretrained(out_dir)`.

One deliberate divergence from the reference: **dtype is a parameter, defaulting to
bfloat16.** SteeringVectors hardcodes `DType.bfloat16`. That hardcode is the same one in
`eval-direct-gpu/scripts/run_eval_sweep.sh:796` that forced Preston's T4 fleet to `sed`
the sweep script on every box, because torch and vLLM both refuse bf16 on sm_75. An L4
keeps the default; a Turing card has a way out that is not a text substitution.

**Half-true when written; closed 2026-08-08.** For two phases the parameter existed on
`convert_olmo_core_to_hf` and on `prepare_checkpoint` and nothing reached either from the
command line: `runner.py` had no `--dtype`, and its `prepare_checkpoint` call passed none,
so every run took the default. The escape hatch was real in the library and unreachable
from the CLI, which is worse than the hardcode it replaced in one specific way — a
hardcode is visible in the text of the command the platform reads, and a default is not.
Phase 6 measured what that cost.

`runner.py` now takes `--dtype`, threads it through `prepare_checkpoint` into the body,
records it in `report["run"]` and prints it in the dry-run line. So the sentence above is
a description rather than an aspiration: the Turing card's way out is `--dtype float16` on
the command line, which is a flag and not a `sed` against a file on the box.

**The accepted set is narrower than `DType` on purpose.** `olmo_core.config.DType` also
names `float8_e4m3fn` and `float8_e5m2`, and `save_hf_model` would cast to either without
complaint. The `LlamaConfig` this pipeline emits carries no quantization block, so
`from_pretrained` cannot read back what that conversion would write — the conversion
succeeds and the load two lines later does not. `convert.CONVERSION_DTYPES` is therefore
`(bfloat16, float16, float32)`, argparse refuses the rest, and `_check_dtype` runs before
`_olmo_core_imports()` so a caller reaching the library directly is refused there too.
That ordering is Phase 3b's argument again: `DType(dtype)` is evaluated as an argument to
`save_hf_model`, which is after the model has been rebuilt and 1.74 GB of shards read into
it, and a refusal that cheap belongs before the work rather than after it.

**Under `--checkpoint-prep none` a non-default value warns, and the default does not.**
Nothing is converted, so the precision describes nothing either way. The default is what a
caller gets for not asking, and warning on it would fire on every `none` run while telling
nobody anything. The case worth a line is the one this whole change exists for: somebody
sets `--dtype float16` because the card they were given has no bfloat16, gets `none`, and
finds out that the weights loaded at whatever training wrote only when a kernel refuses
the format — which reads like a broken flag rather than an inapplicable one.

**The `get_hf_config` patch is required rather than conditional, and Phase 3b installs
it.** `FINAL_PRESTON_RESULTS_BASE/README.md` records that the 82-checkpoint sweep over
this arm needed "a patch to `get_hf_config` to emit a `LlamaConfig` for standard
`TransformerBlock` models, which upstream OLMo-core refuses to convert"; upstream still
refuses at `OLMo-core/src/olmo_core/nn/hf/config.py:96-101`, which accepts only
`ReorderedNormTransformerBlock`; and Phase 1 confirmed this checkpoint's block is the plain
kind. Without the patch this phase's function raises `NotImplementedError` on Preston's
checkpoint every time.

Bringing `hf-converter-patch/apply_llama_config_patch.py` into the repo is its own piece of
work with its own decision to make about *how* it is installed, so it is Phase 3b rather
than a paragraph here. This phase's only obligation is that the conversion body is the
thing that installs it.

Phase 1 does not decide whether this is needed — that is settled. What Phase 1's
`config.json` still decides is whether the attention config forces conversion onto a GPU,
which is a separate caveat recorded in Phase 5.

**Tests**: only that the module imports without torch, and that calling the body without
it raises a legible `RuntimeError` rather than an `ImportError` from three frames down, in
the style `generative.py`'s graders already use.

**Done — 2026-08-08.** Written and import-guarded; unexercised until Phase 5, as planned.

Two additions beyond the port. `_resolve_tokenizer_id` warns when it falls back to dolma2,
because the fallback is a guess about which vocabulary a model was trained against and a
wrong one scores every continuation against the wrong token ids -- producing a theta
rather than an error. And the function re-checks its own output with
`is_hf_checkpoint` before returning, so a conversion that wrote weights but no tokenizer
fails at the converter rather than in the loader after the model has been rebuilt once.

The dependency check runs before any filesystem work, so a run that cannot convert leaves
no half-written output directory behind. That ordering has a test.

---

### Phase 3b — Vendor the `get_hf_config` patch

**Purpose.** Make the conversion possible at all. Upstream's exporter builds an HF config
only for `ReorderedNormTransformerBlock` and refuses everything else at
`OLMo-core/src/olmo_core/nn/hf/config.py:96-101`; Phase 1 confirmed this checkpoint's block
is `TransformerBlockConfig` with `name: "default"`, a plain pre-norm `TransformerBlock`. So
Phase 3's body raises `NotImplementedError` on every run until this lands, and it raises it
*after* rebuilding the model and reading 1.74 GB of shards into it.

Only the config is missing. `convert_state_to_hf` dispatches on `config.model_type` and
already carries `llama` weight mappings in three tables, so returning a `LlamaConfig`
engages a path that exists.

**A runtime monkeypatch, not a rewrite of site-packages.** The source is
`hf-converter-patch/apply_llama_config_patch.py`, the script the memory-split node drivers
ran, which edits the installed `config.py` in place from a separate process. That version
works — 82 checkpoints came out of it — and its cost is two ordering hazards that live in
whatever shell drives the node. `hfsplit-full/node_driver.sh` shows both: the rewrite has
to land after `uv sync` and before the first `uv run` (`hfpatch_start`, lines 230-239), and
`UV_NO_SYNC=1` has to hold afterwards or `uv run` re-syncs, reinstalls PyPI
`ai2-olmo-core==2.4.0` and reverts the fork and the patch together — which that driver
cared about enough to spend a whole verification step on (`uvnosync_verified`, lines
241-268). Neither hazard is visible to anything that could check it; both are properties of
the order two shell lines happen to appear in.

Installing the patch from inside the converter deletes them rather than documenting them.
There is no artifact for a re-sync to revert, nothing to sequence against the install, and
no `config.py.orig` left in site-packages. Three further reasons, in descending order of
how much they matter:

It is testable. The textual patch can only be exercised by installing `ai2-olmo-core` and
rewriting a file, which is impossible on this machine and destructive on any other; the
monkeypatch is exercised by 34 tests with no torch installed. That is not a convenience —
it is the difference between a patch whose refusals are asserted and one whose refusals are
hoped for.

It is narrower against upstream drift. The script aborts unless a 14-line block of source
matches character for character, so any cosmetic edit upstream stops the pipeline. The
monkeypatch depends on the *types* it dispatches on — `TransformerBlock`,
`ReorderedNormTransformerBlock`, `Attention` — and on the function's name. Both fail loudly
rather than silently, which is the important property, but the second fails on a much
smaller class of change.

It travels with the repo. The node driver had to `aws s3 cp` the patch out of a bucket
before it could run; here it is an import.

**The hazard the monkeypatch takes on instead, and how it is paid.**
`olmo_core.nn.hf.checkpoint` does `from olmo_core.nn.hf.config import get_hf_config` at
module scope and calls it from `save_hf_model` at line 159, so rebinding the name in the
module that *defines* it does nothing for the module that *calls* it. That mistake reports
success and then raises exactly the error the patch exists to prevent. `apply()` therefore
imports `olmo_core.nn.hf` first, which loads every by-value holder, and then rebinds all of
them in one scan over `sys.modules` scoped to `olmo_core`. Anything imported afterwards
picks up the patched function from the defining module, so the result does not depend on
import order at all. This is the property the drivers could not have: the in-process
version is not merely easier to order correctly, it has no ordering to get wrong.

**Where it lives.** `diagnostics/mcq_cat/common/hf_config_patch.py`, beside `convert.py`
and importable without torch, exposing `apply()` (idempotent, returns whether it installed)
and `is_applied()`. `convert_olmo_core_to_hf` calls `apply()` immediately after
`_olmo_core_imports()` — after, because the rebinding scan can only reach modules that are
loaded, and immediately, because patching is free and reading the shards is not, so the
refusal belongs before the load rather than after it.

**A wrapper, not a rewritten guard.** The script edits upstream's block check and inserts
an early return. The monkeypatch instead intercepts and delegates: MoE models, normalized
transformers, reordered-norm blocks and every one of upstream's own refusals reach the
original function untouched, and only the case that used to raise is handled here. So it
cannot change the output for any model that converted before this existed, which is a
claim the textual patch cannot quite make.

The emitted `LlamaConfig` is the script's, field for field, including its refusals: QK-norm,
rope scaling, sliding windows, a non-`Attention` sequence mixer and mixed block types all
raise rather than being dropped, because a config that quietly omits one of them still
loads, still scores, and reports an ability estimate for a model that is not the one that
was trained. `hidden_act="silu"` stays hardcoded, as upstream hardcodes it and as the
validated script hardcodes it; Phase 1 confirms this config says `silu`, and detecting it
properly would need a feed-forward attribute `olmo_core` does not expose consistently.

**`tie_word_embeddings` is carried through, and this config sets it.** Phase 1 read
`tie_word_embeddings: true`, so it matters that the emitted config says so rather than
taking `LlamaConfig`'s default of `False`. It does, reading the flag off the rebuilt model,
and the weight path agrees with it: olmo-core's `_tie_weights` makes
`lm_head.w_out.weight` the same `Parameter` object as `embeddings.weight`, so `state_dict()`
carries both names at one tensor, `convert_state_to_hf` maps both, and `save_pretrained`
then drops the tied head from the safetensors because HF knows it is tied. Get this wrong
and nothing fails: the directory loads, scores correctly, and holds a second copy of a
28M-parameter embedding matrix described as an independent head. Two tests pin it, one per
value.

`OLMO_CORE_HF_MAX_POSITION_EMBEDDINGS` survives as the way to override
`max_position_embeddings`, defaulting to 2048 — the value the validated node configuration
exported, and the same as this checkpoint's `sequence_length`. Upstream writes `-1` here,
which is harmless for a config that is only round-tripped and not harmless for one a server
reads, since vLLM sizes its KV cache from it.

**Tests** in `diagnostics/mcq_cat/styles/uni_mcq/tests/test_hf_config_patch.py`, torch-free,
against a stand-in `olmo_core` module tree. Standing the package in is not a compromise:
every property under test is a property of the module graph or of the emitted kwargs, and a
stand-in is the only way to construct the by-value import — the thing most likely to be got
wrong — on purpose. The two the phase exists for are that `apply()` runs before the shards
are read and that its absence is detectable rather than silent; the rest pin each refusal,
each field, and that upstream's paths are unchanged.

**Done — 2026-08-08.** 34 tests, all passing with no torch installed. Six mutations tried,
six caught, including the two that would have shipped: rebinding only the defining module,
and validating only the first block. The one worth recording is the first — with the scan
narrowed to `olmo_core.nn.hf.config`, `is_applied()` still answers `True` and every test
about the emitted config still passes, because the only thing that breaks is a call site
that this repository does not own. That is precisely the failure a site-packages rewrite
cannot have and an in-process patch can, and it is why the scan has a test of its own
rather than being an implementation detail.

---

### Phase 4 — Wire both seams into the runner

**Purpose.** Connect preparation and backend selection so the three configurations differ
only by flags. This is where the design constraint either holds or quietly stops being
true.

In `diagnostics/mcq_cat/runner.py`, between the fetch and the load:

```python
checkpoint_dir = s3_io.resolve_checkpoint(...)
checkpoint_dir = convert.prepare_checkpoint(
    checkpoint_dir, Path(tmp) / "checkpoint-hf", policy=args.checkpoint_prep
)
model = grading.load_grader(request, checkpoint_dir, settings)
```

Also in this phase, and each small:

`--checkpoint-prep {auto,none}` on the parser, defaulting to `auto`, recorded in
`report["run"]` beside `checkpoint_kind`. A report that cannot say whether its checkpoint
was converted cannot be compared with one that was.

`load_scoring_model` and `load_generative_model` become dict-registry lookups. Today each
registry holds one entry, `"hf"`, and the `KeyError` handler names what is registered for
that modality. The point is not the one entry; it is that the second one is a line.

The runner calls `close()` on the loaded model if it has one, in a `finally`. Neither
current backend does, so this is inert — and it is the only speculative line in the plan,
justified because a served backend without it leaks a subprocess and adding it later means
touching the runner again.

Both directories live under the same `tempfile.TemporaryDirectory`, so under `auto` the
native checkpoint and its HF copy coexist for the run — a little over 2 GB for this
checkpoint, per Phase 1. Under `none` only one exists, which is a real reason to reach for
it on a disk-constrained shape.

**Four existing tests break** and each needs a decision rather than a blanket fix. They
monkeypatch `s3_io.resolve_checkpoint` to return a bare `tmp_path / "ckpt"`, which now
flows into `ensure_hf_checkpoint` and hits the unrecognized-layout branch:

| Test | Line |
|---|---|
| `test_grading_dispatch.py` | 355, 404, 479 |
| `test_runner_integration.py` | 75 |

Prefer giving them a fixture that sniffs as HF over monkeypatching the converter away —
these are the tests that cover the runner's ordering, and stubbing the new step out of
them means the ordering is no longer covered.

**Done — 2026-08-08.** 976 tests pass, up from the 941 baseline; ruff clean across
`diagnostics/mcq_cat`. Seven mutations tried against the wiring, seven caught, including
both "the flag is ignored" and "`close()` runs only on success".

**Six tests broke, not four.** `test_scoring_convention.py` has two more stubs of the same
shape at lines 449 and 467 that the plan's survey missed. All five were fixed the way the
plan preferred -- a `stage_hf_checkpoint` helper in `conftest.py` that writes the thinnest
directory the detector accepts -- so the runner's real ordering stays under test rather
than being stubbed out of it.

The sixth was not a stub at all.
`test_generative_grading.py::test_olmo_core_points_at_the_integration_point` asserted a
`NotImplementedError` naming the integration point, and under a registry `olmo_core` is
simply absent, so it now fails as any unknown kind does. That is the better behaviour --
a backend nobody registered and a backend nobody has heard of are the same state, and
reporting them differently made the table look longer than it was -- so the test was
rewritten to pin the new message rather than the old error type.

**One deliberate deviation.** `--dry-run` output is *not* byte-identical: it gained
`prep=%s`. The dry run prints the plan, and which preparation policy will run is part of
the plan; omitting it would mean `--checkpoint-prep none` and the default produce
identical dry-run output.

`_load_olmo_core` survives in both modules as an unregistered, documented landing site
naming stash `fed3d925` and the three steps to restore it, rather than being deleted.

---

### Phase 5 — Conversion smoke on a real checkpoint

**Purpose.** Prove the converter against real weights on a real machine. This is the first
step that can fail for reasons no local test would surface, so it is kept as small as
possible: convert one checkpoint, load it, score nothing.

**This does not run on `gpu-1xl4`.** Not a preference — a constraint, because the
conversion path has no GPU in it by construction and the L4 is the shape Phase 7 needs. The
reference says so three ways in `run_steering_eval.py:281-342`: the model is built with
`init_device="cpu"` (`:317`); the process group `load_model_and_optim_state` demands is a
single **gloo** rank at `WORLD_SIZE=1` (`:308-312`), gloo and not nccl, and nccl is the
backend that would require CUDA to initialize at all; and `save_hf_model` is handed
`model.state_dict()` off that CPU model (`:324-326`). Nothing between those lines imports
`torch.cuda` or moves a tensor to a device. The GPU enters that file exactly once, at
`_load_model:403`, in a `.to(cfg.device)` that happens after conversion has already
returned — which is Phase 7's line, not this phase's.

**So the smoke wants host RAM and disk, not VRAM.** `add-cat-evals`' `CONVERSION.md` puts
peak RAM at roughly the model size in bf16 — ~2 GB per 1B params, ~2× if validation runs,
which we skip — and disk at ~2 GB in plus ~2 GB out per 1B params, so about twice the model
size resident at once. Budget against its measured table rather than its rule of thumb,
since the two disagree and the table is the conservative one: 1B ≈ 4-6 GB, 7B ≈ 16-20 GB,
13B ≈ 28-32 GB. It also quotes ~30 s for 1B on local NVMe with validation skipped, plus S3
transfer and a 5-15 s cold torch/olmo_core import. Every one of those is a host number.
Read the rest of that document with one correction: it claims the conversion needs "no
`torch.distributed` init", and the working reference initializes a process group because
`load_model_and_optim_state` will not run without one, which is why Phase 3 ports it.

**What a CPU-only smoke proves, and what it does not.** It proves the whole conversion
contract: that the pinned fork parses this checkpoint's schema, that the DCP shards under
`model_and_optim/` load into the rebuilt `Transformer`, that `get_hf_config` accepts the
block type instead of raising `NotImplementedError`, that HF weights and a tokenizer are
written, and that the directory opens with `AutoModelForCausalLM` and `AutoTokenizer`. It
proves nothing GPU-specific — and no gap is being papered over by that, because conversion
never touches a GPU, so running it on one would exercise identical code beside an idle
card. The genuinely device-dependent question is whether the bf16 weights this writes load
and score correctly on an L4, and no conversion test answers it on any hardware. That
question is Phase 7's and cannot be pulled forward by choosing a more expensive venue here.

**The ordering benefit is the whole argument.** Proving conversion on CPU means Phase 7's
GPU time buys only the thing that needs a GPU. A converter bug — an unparseable schema, a
block `get_hf_config` refuses, a tokenizer identifier that will not resolve — then fails on
a shape Phase 7 is not competing for, rather than burning L4 minutes to surface a
`NotImplementedError` that a CPU would have raised identically. Note that this is an
argument about contention and about not paying for an idle card, not about the hourly rate,
which the venue list below now records as running the other way.

Venues, best first — which was written as "cheapest first" and is no longer the same
ordering, for the reason recorded under 1:

1. **A platform job on the catalog's one CPU-only compute.** The image already carries
   torch and `ai2-olmo-core` through `INSTALL_OLMO_CORE=1`, so there is no local resolution
   to do. **Settled 2026-08-08, and the answer is not the one the ordering below assumed.**
   `edullm check` names fourteen provisioned shapes when handed a `--compute` it does not
   recognize, and exactly one of them has no card: `cpu-32vcpu`. It is valid for
   `olmo-eval-sweep` — checked against this spec it prices, lands at `automatic` approval,
   and draws no refusal beyond `uncommitted_changes`, which is this plan's own working
   state. It also draws no bfloat16 refusal, so the smoke keeps the default precision.

   It is not, however, the cheap option. Read out of `edullm check --json` on 2026-08-08,
   `cpu-32vcpu` costs more per hour than `gpu-1xt4`, `gpu-1xl4` and `gpu-1xa10g` — a CPU
   shape with 32 vCPUs is a large machine, and the single-GPU shapes here are small ones.
   The numbers are deliberately not copied into this file, per `AGENTS.md`; re-run the
   check rather than quoting a plan.

   That does not overturn the choice, because Phase 5's argument was never that CPU is
   cheaper per hour. It was that conversion has no GPU in it by construction, so an L4 spent
   on it is a card sitting idle beside a CPU workload, and that a converter bug should
   surface on a shape Phase 7 is not competing for. Both still hold. What it does overturn
   is the word "cheapest": if the deciding criterion were the bill for one short run,
   `gpu-1xt4` would win it — and that shape is now refused outright, since the conversion
   writes bfloat16. Choose `cpu-32vcpu` for the reason it was always the right venue, not
   for a price it does not have.
2. **A small CPU EC2 box.** Bounded and disposable, but it is a machine to launch, pay for
   and remember to stop, and the sb-aws skill requires a written value assessment naming
   the instance type and its cost before one is even proposed. Reach for it if 1 refuses.
3. **A local `uv sync --extra olmo_core`.** Last. This machine has no torch, no
   transformers and no `ai2-olmo-core` today, so this pulls torch for the first time, and
   the lock's torch carries CUDA wheels gated on `sys_platform == 'linux'`; whether it
   resolves on Windows at all is unknown. Last not because it is slow but because a
   resolution failure here would say nothing about the converter.

Whichever venue: `suggested_compute` in `.edullm/run.yaml` is a suggestion. That file's own
header records that the machine is supplied at submit time, so "not the L4" is a choice
made on the submission and not a property of a committed file — it is entirely possible to
satisfy every sentence above and still land the smoke on a card by accepting a default.

**The one caveat that could have invalidated the CPU-only claim is cleared.**
`CONVERSION.md` notes that fused and flash-attention block configs force `device=cuda`.
Phase 1 read the `config.json` for exactly this and found `sequence_mixer.name: "default"`
under `AttentionConfig` with `type: "attention"` — neither of the two. The constraint
stands as written and needs no fallback.

What Phase 1 adds instead is a network requirement nothing above accounts for: the
tokenizer is resolved by identifier against the HuggingFace Hub, at the very end of the
conversion, so a venue that can reach S3 and not `huggingface.co` fails after doing all the
work. Whichever venue is chosen, confirm outbound Hub access before spending anything on
it.

**Use `step305176` itself — the hunt for a smaller checkpoint was a false economy.** An
earlier draft preferred a throwaway: `run_019fcf53`, twenty steps on a T4 on 2026-08-05,
referenced at `.edullm/Dockerfile:159`. Reading that reference kills the idea. What it
records is a run that logged all twenty steps and then died on the first line *after* the
loop with `MissingDependencyException: Using CRC32C requires an additional dependency` — a
failure in the checkpoint write, so there is most likely nothing there to convert.

It does not matter, because the premise was wrong. Preston's model is SmolLM2-135M: about
0.27 GB of weights at bf16, against the 4-6 GB `CONVERSION.md` measures for a 1B. The real
checkpoint *is* the tiny checkpoint. Converting anything else would exercise a different
architecture, prove less, and save nothing worth having — and the whole point of this phase
is that the thing it converts is the thing Phase 7 will convert.

Phase 1 measured what `model_and_optim/` weighs and the guess above was low: 1.74 GB, not
"under a gigabyte", because the checkpoint carries fp32 masters and both AdamW moments at
about 13 bytes per parameter. It does not change the choice — 1.74 GB in and 269 MB out is
still nothing next to a 1B — but it is the sync that pays it, and the shape needs a little
over 2 GB of scratch rather than the ~0.5 GB a bf16-weights estimate would suggest.

**Done when** a converted directory loads with `AutoModelForCausalLM.from_pretrained` and
`AutoTokenizer.from_pretrained` on a machine with no GPU, and the wall-clock and peak RSS
of the conversion are recorded here. Those two numbers decide whether the parked native
loader ever comes back.

---

### Phase 6 — `run.yaml`

**Purpose.** Express the working pipeline as a platform job spec. `edullm check` then
prices and validates it without dispatching anything, so submission errors surface for
free.

Replace the spike command with the runner, following the SteeringVectors shape. The two
constraints that shape it are both recorded on that branch: `edullm submit` cannot inject
environment variables, so the checkpoint path and every flag are baked into `command:`;
and the GPU workload role is scoped to `teams/<team>/runs/*`, so the checkpoint must be
staged under the outputs bucket rather than `edullm-checkpoints`. The current spike
`run.yaml` already reads from a compliant path.

Keep `workload_profile: olmo-eval-sweep` and `suggested_compute: gpu-1xl4`.

The sync line keeps `--extra olmo_core` and then swaps in the fork, which Phase 1 has
already named: `uv pip install --no-deps "ai2-olmo-core @
git+https://github.com/edu-llm/OLMo-core@edullm/memory-split-135m"`. `--no-deps` so it
cannot drag the pinned resolution around underneath it, and after the sync rather than
instead of it, because the extra is what brings the surrounding dependency set. It does
not need `--extra vllm`: that extra exists in the SteeringVectors command to protect a
baked torch that came from `INSTALL_TORCH_AND_VLLM=1`, and this branch's torch comes from
`INSTALL_OLMO_CORE=1` instead, which the sync must therefore preserve rather than the
vllm one.

Phase 3b changes what follows the fork install. There is no patch step, because the
converter installs the patch itself, so the only thing left on the command line is
`OLMO_CORE_HF_MAX_POSITION_EMBEDDINGS=2048` — redundant against the default and worth an
`export` anyway, because it makes the converted context length auditable from the spec
rather than from a Python default. `UV_NO_SYNC=1` stays, and its reason has changed: it no
longer protects the patch, since there is no longer anything on disk to revert, and this
command never runs `uv run`. It protects the fork pin against the next edit that does.

**Done — 2026-08-08.** `edullm check --experiment uni-mcq-arc-challenge --dataset none
--json` parses the spec, resolves the command, and prices it at `automatic` approval
against `gpu-1xl4` for a 2-hour ceiling. One refusal, `uncommitted_changes`, which is this
plan's own working state and not a property of the file; the three `deferred` entries are
all image-registry questions that only a pushed commit can answer. Nothing about the
command, the workload profile or the compute was refused.

**The `--dtype` comment did not survive, and what replaced it was a finding rather than a
rewording.** The plan above assumed the reasoning transferred unchanged. It did not:
`runner.py` had no `--dtype` flag at all. Precision was `prepare_checkpoint`'s bfloat16
default, chosen in code, which is exactly the thing the platform's
`bfloat16_not_in_the_hardware` guard cannot see, because that guard reads the text of
`command:`. Measured the same day: `edullm check --compute gpu-1xt4` against that spec
raised no bfloat16 refusal whatsoever. A Turing card priced, passed, was admitted, and
would then have failed on the first kernel that wanted the format — the precise failure
the old comment existed to prevent, unguarded.

Naming the flag anyway would have been worse than leaving the gap, and this is the reason
the gap sat in the yaml for a day rather than being papered over. The guard reads text and
the runner reads argv, so a fabricated `--dtype bfloat16` satisfies the first and makes the
second exit 2 on an unrecognized argument, converting a failure that happens on the wrong
card into one that happens on every card. Closing it properly was a change to `runner.py`,
which Phase 3 now records.

**Closed and verified — 2026-08-08.** With `--dtype bfloat16` written into `command:` and
the flag real behind it, `edullm check --compute gpu-1xt4` returns
`bfloat16_not_in_the_hardware`, and the detail quotes the command back: *"this command asks
for bfloat16 with `--dtype bfloat16`"*, followed by the provisioned shapes whose cards have
the format. The refusal appears on `gpu-1xt4`, `gpu-4xt4` and `gpu-8xt4` — every Turing
shape in the catalog, and no others. `gpu-1xl4` is unaffected, so `suggested_compute` still
prices as it did.

Two details of that output are worth keeping. `cpu-32vcpu` draws no bfloat16 refusal
either, because the guard is about cards and that shape has none — which matters to Phase
5, since its smoke keeps the default rather than needing an override. And the detail says
in as many words that it "read the words of your command and nothing else", so this is a
guard against a *stated* precision, not a verified one: `--dtype bfloat16` and a program
that ignored it would pass identically. The tests are what make the statement true, which
is why they pin the threading rather than the yaml.

---

### Phase 7 — One live run

**Purpose.** Get one real theta out of the full pipeline against Preston's checkpoint.
Establish the wall-clock numbers that every later estimate — and the vLLM question —
depends on.

One checkpoint, one benchmark. `arc_challenge` is the right first target: it is the only
bank whose bridge is independently attested by a shipped release, so a theta that looks
wrong is more likely to be the pipeline than the join.

Treat the first *generative* benchmark as a separate checkpoint in this phase rather than
folding it in. It is the first time this pipeline decodes anything against a real model,
and its cost profile is unrelated to the MCQ banks'.

**Done when** `cat_report.json` lands in S3 with a theta, a standard error at or under
0.3, and a `bank_provenance` block — and when conversion's share of
`experiment_duration_seconds` is written back into Phase 5's numbers.

---

### Phase 8 — Rewrite the skill around job submission

**Purpose.** Make `eval-cat` answer the question an agent is actually asked — *submit an
eval job for this model on this benchmark* — end to end, without the user knowing anything
about this pipeline. Today the skill describes driving the runner by hand on a GPU box,
which is a different task from submitting a job.

The exchange to design for:

> **User:** submit an eval for `s3://…/checkpoints/step305176` on `arc_challenge`
> **Agent:** reads `eval-cat`, prices it, submits it, says where results will land.

The agent should need nothing beyond the skill and `AGENTS.md`, and should ask the user
nothing the skill could have answered.

**What the current skill gets wrong for that workflow.** These are not stale sentences,
they are a different procedure:

| Current text | Under the submission workflow |
|---|---|
| asks the user for an output location | wrong — `$EDULLM_OUTPUT_PREFIX` is injected by the platform |
| tells the agent to run `python -m diagnostics.mcq_cat.runner` | that is the *on-node* command, and belongs baked in `run.yaml` |
| "requires torch and transformers, so it runs on a GPU box" | the agent never touches a box |
| "must be HuggingFace format" (line 23) | wrong after Phase 4 — preparation is automatic |
| "`--checkpoint-kind olmo_core` is a declared but unimplemented path" (line 166) | wrong framing — it is a registered backend selected by two flags |
| silent | must say: commit and push to `edullm/**` before submitting |
| silent | must say: `--dataset none` |

**The headline the rewrite exists for.** `edullm submit` cannot inject environment
variables, and per `AGENTS.md` the platform builds its image from *the last commit on an
`edullm/**` branch*, not from the working tree. So pointing a run at a different checkpoint
or benchmark means editing `.edullm/run.yaml`, committing, pushing to `edullm/uni-mcq`,
letting the image build, and only then submitting. An agent that misses this will edit
`run.yaml`, submit, and evaluate the *previous* checkpoint — the failure produces a
well-formed `cat_report.json` with a plausible theta and nothing downstream catches it.
That warning is the reason this phase is not a doc touch-up.

**Defer the platform half; do not restate it.** `AGENTS.md` already carries the submission
contract in a managed block, and says in as many words that there is deliberately no skill
for submitting because a table of refusal codes would duplicate what `edullm check --json`
prints. Link it. Copying exit codes or refusal codes into the skill would also rot against
a block that is distributed from `edu-llm/platform` and reverts local edits.

**Quote no prices.** `AGENTS.md` forbids stating a price, runtime bound, cost ceiling or
approver from memory or from a document. The skill instead tells the agent to read `cost`
and `approval_class` out of `edullm check --json`. The current Step 2's runtime prose goes.

**Keep what still works.** The `ready_names()` preflight is still correct, still free, and
is the only thing that will tell an agent that seven of nine benchmarks are runnable. The
Step 3 report-field list is still accurate and should survive intact, including the
`ungradable` alert and the instruction never to compare theta across benchmarks.

**Add the cost shape per modality.** Four of the seven runnable benchmarks are MCQ and
three are generative, and a generative CAT decodes up to `max_new_tokens` per item
sequentially. A user asking for "all benchmarks" is asking for seven submissions with two
very different cost profiles, and the agent should say so before submitting rather than
after.

**Resulting shape** — five steps, replacing the current three:

1. What to get from the user: checkpoint URI and benchmark. Explicitly *not* an output
   location, and explicitly not the checkpoint format.
2. Preflight, free: `ready_names()` for the benchmark, then `edullm check --json`.
3. Bake the run: edit `.edullm/run.yaml`, commit, push to `edullm/**`, wait for the image.
4. Submit: `edullm submit --dataset none`.
5. Poll `edullm status --json` (free, pollable — unlike `edullm status` and `edullm logs`),
   then read `cat_report.json` and report it with its caveats.

**Done when** an agent given only "submit an eval for `<s3 uri>` on `arc_challenge`", and
no other context, reaches a priced `edullm check` without asking a question the skill could
have answered and without reaching for AWS directly.

---

### Phase 9 — Plan and README hygiene

**Purpose.** Close the loop on the two documents this plan invalidated as it went. Small,
and easy to skip, which is why it is a phase.

This file records Phase 0's stash SHA and Phase 1's answers already; what is still owed is
Phase 5's two numbers and Phase 7's share of them, replacing the paragraphs that ask for
them. The one open item this plan created for itself — Phase 6's finding that `runner.py`
had no `--dtype`, leaving the conversion precision invisible to the platform's hardware
guard — was a code change rather than a documentation one and is closed; Phase 3 records
the flag and Phase 6 records the refusal it now produces.

The branch README's merge-conflict discipline — "never edit `base.py`, `registry.py`,
`runner.py`, `common/*`" — was already overtaken by `164ad8f0`, which put 2,386 lines into
exactly those files, and this plan touches three of them again. Correct it rather than
violate it a third time silently.

## What is not in scope — and what it costs to add later

Deliberately not built here, with the price of each recorded so the design constraint can
be checked against reality rather than asserted.

**vLLM**, per the decision above. To add: one class implementing `score_items` against a
served endpoint, registered under `"vllm"` in the MCQ registry and, if it generates, the
generative one. It gets `close()` for free from Phase 4. No runner change, no new flag —
`--checkpoint-prep auto --checkpoint-kind vllm`.

**The native OLMo-core path.** Parked in Phase 0's stash, not deleted. To restore:
`git stash pop`, register `_OlmoCoreScoringModel` under `"olmo_core"`, and add that string
to `LOADABLE_CHECKPOINT_KINDS[MCQ]`. Run it with `--checkpoint-prep none --checkpoint-kind
olmo_core`. Everything the parked work assumed about the runner stays true, which is the
point of keeping the dict shape in Phase 0.

**The generative path's `olmo_core` branch.** `generative.load_generative_model` still
refuses it. Under `auto` the refusal is unreachable, since the backend only ever sees HF;
under `none` it is the correct error. Nothing to do unless the native path returns.

**A sweep over all the benchmarks.** Seven of the nine are runnable — `winogrande` and
`gsm8k` are blocked pending stronger join verification — and of those seven, four are MCQ
(`arc_challenge`, `hellaswag`, `musr`, `bbh`) and three are generative (`ifeval`,
`leaderboard_math`, `gpqa`). Phase 7 runs one of each to establish both cost profiles; the
sweep across all seven is the phase after this plan, and under the no-env-injection
constraint it is seven submissions or one fan-out, not one job.

If any of these turns out to cost more than a paragraph above says, that is a finding
about the seams and belongs back in this file.
