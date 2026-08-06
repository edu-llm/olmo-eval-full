# Incorporating vLLM into the P3 math-split evaluation

Written after doing the same thing for a parallel workstream (Preston's
memory-split SmolLM2-135M sweep, 164 checkpoints across 16 boxes). This
documents what we did there, which parts transfer, and where the P3 harness
needs a different approach than ours did.

---

## TL;DR

1. **Your config problem doesn't exist.** Our hardest blocker was that OLMo-core
   refuses to synthesize an HF config for non-OLMo block types. You imported
   Qwen2.5-0.5B from the Hub, so `export_checkpoint.py` loads the pinned config
   and never calls `get_hf_config`. Skip our entire patch.
2. **Use L4, not T4.** Your harness enforces BF16 in three separate gates, and
   vLLM refuses BF16 outright on Turing. On T4 you must either fight all three
   gates or accept FP16. On L4 (compute capability 8.9) nothing needs to change.
   We believe this is what's currently failing.
3. **Only half your harness is a natural vLLM fit.** Generation is; the
   teacher-forced NLL pass is not, because it needs per-token logits. Details
   below — this is the main design decision.
4. **Validate before trusting.** We caught a genuinely broken generation path
   only because we ran the same checkpoint two ways and compared. Budget an hour
   for this.

---

## What we did for Preston, and what transfers

Our pipeline, per box: bootstrap → clone repo → install the OLMo-core fork →
patch the HF config generator → fetch checkpoint from S3 → convert OLMo-core to
HF → serve through vLLM → run benchmarks → sync results to S3 → self-terminate.

| Step | Transfers to you? |
|---|---|
| Patch `get_hf_config` for standard blocks | **No** — you have a pinned Qwen config |
| Force conversion dtype | **Maybe** — only if you go T4 |
| `UV_NO_SYNC=1` so `uv run` doesn't revert the installed fork | **Yes**, if you install a fork then invoke through `uv` |
| `export UV_PYTHON=3.12` before bootstrap | **Yes** — uv picks 3.14, `llvmlite` has no wheel, build fails |
| LF-only user-data | **Yes** if launching from Windows; CRLF breaks cloud-init |
| Shard across boxes by checkpoint | **Partly** — you have 2 checkpoints, not 82. See sharding below. |
| Validate the vLLM path against a non-vLLM path | **Yes, emphatically** |
| Self-termination: `shutdown -h +N`, `instance-initiated-shutdown-behavior terminate`, stall watchdog | **Yes** |

Cost reference: 82 checkpoints x 7 benchmarks x ~50k items each ran in 3h17m
across 8 `g4dn.xlarge` for about $11.

---

## Why your case is easier than ours

Our models were a Llama-family architecture trained **from scratch** inside
OLMo-core, so no HF config had ever existed and OLMo-core explicitly refused to
create one (there's a unit test asserting the `NotImplementedError`). We had to
patch the config generator.

Yours were initialized from pinned Qwen2.5-0.5B weights —
`export_checkpoint.py` verifies `base_model_weight_sha256` against a constant and
`load_pinned_hf_config()` pulls `config.json` from the exact Hub revision. That's
the supported round-trip path. Your exported directory should be an ordinary
`Qwen2ForCausalLM` that vLLM has served since forever.

**So if vLLM is rejecting your model, suspect the runtime, not the export.**

---

## Most likely cause of the current failure

You mentioned 8x T4 *or* 8x L4. If you're on T4, this is almost certainly it:

- `export_checkpoint.py:315` — `floating_state_dtype` raises unless every
  floating parameter is `torch.bfloat16`
- `export_checkpoint.py:401` and `run_eval.py:939` — safetensors validators
  reject anything but BF16 tensors
- `run_eval.py:1305` — loads with `dtype=torch.bfloat16` on CUDA

So your exported weights are BF16 by construction. **vLLM performs an explicit
compute-capability check and refuses BF16 below sm_80**, typically with
"Bfloat16 is only supported on GPUs with compute capability of at least 8.0."
The T4 is 7.5.

Note this differs from raw PyTorch, which *silently emulates* BF16 on Turing at
roughly a 10x slowdown rather than erroring. We measured 2.16 TFLOPS emulated
BF16 versus 23.01 for FP16 on a T4. So the same model may "work" under plain
transformers on a T4 and hard-fail under vLLM — which is a confusing signal if
you're bisecting.

**Recommendation: use `g6.xlarge` (L4, 24 GB, sm_89).** Native BF16, no gates to
fight, 24 GB instead of 16 GB for a 16,384-token context, and about $0.80/hr
against $0.526. For a run this size the difference is a few dollars.

If you must use T4, you need FP16 everywhere: relax the three BF16 gates, export
FP16, and pass `--dtype float16`. Both arms must use the identical dtype, since
dtype can move scores — but as long as it's identical the dense-vs-split
comparison stays fair.

---

## The part that actually needs thought: your harness has two halves

This is the important section. `run_eval.py` does two very different things, and
they have opposite affinities for vLLM.

### Half 1 — teacher-forced NLL and next-token accuracy (`target_nll`)

This is your **primary endpoint**, and your own docstring explains why:
generation has a floor problem, per-token loss doesn't. It works by calling the
model directly:

```python
output = model(input_ids=window, attention_mask=..., use_cache=False,
               logits_to_keep=n_score + 1)
prediction_logits = output.logits[:, -(n_score + 1):-1].float()
nll = cross_entropy(prediction_logits.reshape(...), labels.reshape(...), reduction="sum")
total_correct += int((prediction_logits.argmax(dim=-1) == labels).sum())
```

That needs **raw per-token logits over the full 151,936-token vocabulary**, plus
argmax comparison. vLLM's server does not hand you logits.

It *can* be done, via `prompt_logprobs`. Send `prompt + target` as the prompt
with `max_tokens=0` and `prompt_logprobs=1`, and vLLM returns, for every prompt
position, the logprob of the actual token plus the top-1 alternative. Summing the
actual-token logprobs over the target span gives you NLL; comparing the top-1
token id to the actual token id gives you next-token accuracy. Both metrics are
recoverable.

But be aware of what you're taking on:

- `prompt_logprobs` is memory-hungry and historically the slowest path in vLLM;
  on some versions it interacts badly with chunked prefill. Check your version's
  release notes.
- Your careful chunking policy (`iter_target_chunks`, sliding 16,384 window,
  every scored token retains its predecessor) exists to bound *logits memory* in
  a manual forward pass. vLLM manages that itself, so you'd drop your policy and
  adopt theirs — meaning the NLL numbers are no longer computed the way
  `NLL_CONTEXT_POLICY = "bounded_sliding_window_preserve_predecessor"` claims.
  That string is recorded in your results and checked by `compare_arms.py`.
- Numerical differences between vLLM's kernels and HF's will shift NLL slightly.
  Fine for a dense-vs-split *comparison* where both go through the same path;
  not fine if you want NLL comparable to previously reported numbers.

### Half 2 — greedy generation (`generate`)

This is a straightforward, large win. You're currently doing HF `model.generate`
at batch 8, up to 8,192 new tokens, 12,573 generations per arm. vLLM's continuous
batching is enormously better at exactly this. In our sweep, vLLM sustained ~62
items/sec with seven benchmarks running concurrently through one server.

### Recommendation: migrate generation first, keep NLL on HF

Generation is almost certainly your bottleneck — it's autoregressive, one forward
pass per token, up to 8,192 of them per row. The NLL pass is a handful of batched
forward passes over a sequence you already have.

So the highest-value, lowest-risk split is:

- **Generation → vLLM.** Big speedup, and you've already said you accept that the
  generation loop differs.
- **NLL and next-token accuracy → keep the existing HF path.** Your primary
  endpoint stays on code that's already correct, already chunked, and already
  described accurately by the policy strings in your results.

Run them as two phases rather than two live models, the way our sweep did two
`olmo-eval` invocations per checkpoint. Loading both an HF model and a vLLM
engine simultaneously will fight over memory, since vLLM preallocates KV cache.

If you later want NLL on vLLM too, do it as a deliberate migration with a
back-to-back comparison against the HF numbers, not as part of this run.

---

## Sharding across 8 GPUs

`run_eval.py` walks families and conditions sequentially in one process and
writes one JSON. You have 2 checkpoints, so the obvious "one checkpoint per box"
split we used doesn't apply.

Your row distribution is very uneven:

| Family | Rows | Share |
|---|---|---|
| mizar | 2,485 | 59% |
| isabelle | 590 | 14% |
| metamath | 494 | 12% |
| prf2 | 313 | 7% |
| enigma | 263 | 6% |
| thproofs | 46 | 1% |

**Do not shard by family** — mizar alone is 59% of the work, so your slowest box
determines wall clock and you'd get maybe 1.7x from 8 GPUs.

Better options, in order of preference:

1. **Add `--shard-index` / `--shard-count`** that slices the row list after
   `partition_context_eligible` and before scoring. Sixteen shards (8 boxes x 2
   arms, or 8 boxes each running both arms on 1/8 of rows) balances almost
   perfectly. Requires a merge step.
2. **Shard by family x condition** — 18 units, still imbalanced because mizar's
   three conditions are each larger than most whole families.
3. **One box per arm, 2 boxes total.** Simplest, no merge, no code change.
   If generation on vLLM is as fast as we saw, this may already be fast enough.

**Whichever you choose, check whether `compare_arms.py` accepts a merged result
file.** It validates matched controls and hashes across the pair, and
`build_evaluation_metadata` records per-family cohort counts. A naive merge might
fail its gate or, worse, pass while silently misreporting denominators.

---

## The validation gate we'd insist on

This is the part we'd most strongly urge you to copy, because it's what caught a
real bug for us.

We evaluated one checkpoint two independent ways — native OLMo-core reading the
raw checkpoint, and converted-plus-vLLM — and compared. Result: **all five
multiple-choice benchmarks identical to sixteen decimal places**, which validated
the conversion. But the generative benchmarks diverged badly, and when we
measured the actual generated text we found the *native* path was emitting
degenerate repetition loops while vLLM produced fluent output. Something in
OLMo-core's uncached generation path is broken on hardware without KV-cache
support.

We would never have found that by looking at scores alone. The lesson for you:

**Before the full run, take ~50 rows and score them both ways.** Same checkpoint,
same rows, HF `generate` versus vLLM. Compare:

- Exact-match counts. If vLLM's differs wildly, investigate before scaling.
- The actual generated strings, not just the aggregate. Sample a dozen and read
  them. Degenerate output is obvious to a human and invisible in a score.
- Mean generated length. A large gap here means the two paths are stopping
  differently, which for your whitespace-normalized `exact_target` matters.

Also confirm your stop behavior. Your gold targets are proofs; if vLLM doesn't
see an EOS it will run to `max_new_tokens` (8,192) on every row, which is both
slow and score-affecting. Check `generation_config.json` in the export carries a
real `eos_token_id`. Our OLMo-core export wrote `eos_token_id: null` — an
upstream convention, not a bug we introduced — and every generation ran to the
cap. Yours comes from a pinned Qwen config so it should be fine, but verify.

---

## Diagnostic checklist for the immediate failure

In order, because each isolates a different layer:

1. **Load the exported dir in plain transformers**, no vLLM:
   `AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda")`.
   If this fails, the problem is the export, not vLLM.
2. **Check the GPU.** `nvidia-smi --query-gpu=name,compute_cap --format=csv`.
   If it says 7.5, that's a T4 and BF16 will be refused by vLLM.
3. **Read the actual vLLM error.** "Bfloat16 is only supported on GPUs with
   compute capability of at least 8.0" confirms the diagnosis above.
4. **Check the dtype field name.** `write_hf_dir` sets `hf_config.dtype =
   source_dtype` (line 521). transformers v5 uses `dtype`; v4 used
   `torch_dtype`. If your vLLM links an older transformers that expects
   `torch_dtype`, the field may be ignored or mis-read. Inspect the emitted
   `config.json` and compare against the pinned Qwen one.
5. **Check tied embeddings.** Qwen2.5-0.5B ties embeddings, and `write_hf_dir`
   handles this explicitly (`allowed_missing = {"lm_head.weight"}`, then
   `model.tie_weights()`). Confirm the saved safetensors either contains
   `lm_head.weight` or the config says `tie_word_embeddings: true`. A model
   missing both will load but produce garbage.
6. **Try serving the stock Hub model** —
   `vllm serve Qwen/Qwen2.5-0.5B` — on the same box. If the stock model fails
   too, it's environment, not your export.
7. **Then reduce.** Drop `--max-model-len` to 2048 and batch to 1. If it works,
   it's a KV-cache sizing problem, not a model problem — 16,384 context on a
   16 GB T4 with a 151,936 vocab is tight.

---

## Environment notes from our run

- `export UV_PYTHON=3.12` before bootstrap. uv otherwise selects 3.14, where
  `llvmlite` has no wheel and the source build fails on a `spawn()` TypeError.
- `UV_NO_SYNC=1` if you install a fork with `uv pip install` and then invoke
  anything through `uv run` — otherwise the run re-syncs and silently reverts
  your fork. We verified this explicitly rather than assuming it.
- vLLM flags we used: `--dtype auto` (picks up dtype from the config),
  `--gpu-memory-utilization 0.55`, `--enable-prefix-caching`, no tensor
  parallelism. Prefix caching is cheap to enable but won't help you much — your
  prompts diverge right after the `I know these mathematical statements:` header.
- Do not use tensor parallelism. A 0.5B model gains nothing and pays all-reduce
  overhead. Parallelism belongs across rows, not within the model.
- Self-termination that actually worked: `sudo shutdown -h +N` on the box,
  `--instance-initiated-shutdown-behavior terminate` at launch, plus a watchdog
  that shuts down after 20 minutes with no local file activity. All 16 of our
  boxes self-terminated without ever reaching the backstop.

---

## Open questions

Answering these would let us be much more specific.

1. **What is the exact vLLM error and traceback?** Everything above is inference
   from reading your code. One traceback replaces all of it.
2. **Which GPU is attached when it fails — T4 or L4?** And what does
   `nvidia-smi --query-gpu=compute_cap` report?
3. **vLLM and transformers versions**, and whether you're using `vllm serve`
   (OpenAI-compatible HTTP) or the offline `LLM(...)` Python API. The
   `prompt_logprobs` route differs between them.
4. **Do you want vLLM for the NLL half too, or just generation?** Our
   recommendation is generation only for this run, but if the NLL pass is your
   measured bottleneck that changes the answer.
5. **Is the primary endpoint allowed to move?** Migrating NLL to vLLM will shift
   the numbers slightly and invalidates
   `NLL_CONTEXT_POLICY = "bounded_sliding_window_preserve_predecessor"` as a
   description of what was computed. Acceptable, or must NLL stay bit-comparable
   to previous runs?
6. **Will `compare_arms.py` accept a merged multi-shard result JSON?** If not,
   we're limited to one box per arm regardless of how many GPUs are available.
7. **How long did your reference run take, on what hardware?** With 12,573
   evaluations per arm, each up to 8,192 generated tokens, we can't size
   instances without an anchor.
8. **Do you want `--probe`?** Your docstring calls it the mechanism check, but
   it's absent from the README's execution order. It loads all 3.4 GB of train
   shards to classify fact-name visibility.
