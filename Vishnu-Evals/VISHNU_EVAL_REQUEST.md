# Files needed to run the P3 math-split evaluation

We have `Vishnu-Eval-Code/` (four files) and the portable `corpus-v3/` tree, and
we've verified both S3 checkpoints at step 23166 exist. The corpus checks out:
4,191 eval rows across the six families (mizar 2,485, isabelle 590, metamath
494, prf2 313, enigma 263, thproofs 46).

Neither `run_eval.py` nor `export_checkpoint.py` can currently be imported.
Both fail at module scope on dependencies that live outside the `evals/`
directory we received. `compare_arms.py` is pure stdlib and runs as-is.

Below is everything we need, in priority order.

---

## 1. BLOCKING — the `p3_math_split` parent package

Both scripts do `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`,
so they expect to sit at `src/scripts/train/p3_math_split/evals/` with these
modules one level up at `src/scripts/train/p3_math_split/`. We received only the
`evals/` subdirectory.

### `mm_verify.py`

`run_eval.py:55` imports it at module scope, so nothing runs without it.

| Symbol | Used at | Purpose |
|---|---|---|
| `norm` | `run_eval.py:57`, via `exact_target` (755) and `run_probe` (622) | whitespace normalization for exact match |
| `parse_proof` | `run_eval.py:823` | appears unused on the current main path, but imported |
| `VERIFIER_SCHEMA_VERSION` | `run_eval.py:794` (getattr) | gates Metamath validity reporting |
| `verify_proof_tristate` | `run_eval.py:795` (getattr) | ditto |

**Please send the real file rather than let us stub it.** `norm` defines the
normalization that decides exact-match scoring; a reimplementation that differs
even in whitespace handling would silently change every exact-match number
without failing anything.

### `provenance.py`

`export_checkpoint.py:61` imports it at module scope.

| Symbol | Purpose |
|---|---|
| `TOKENIZER_ARTIFACT_ID` | pinned tokenizer identity, checked against the checkpoint config |
| `TOKENIZER_ARTIFACT_VERSION` | ditto |
| `TOKENIZER_FILE_SHA256` | ditto |
| `TOKENIZER_COMPOSITE_SHA256` | ditto |
| `TOKENIZER_PAD_TOKEN_ID` | ditto |
| `TOKENIZERS_VERSION` | ditto |
| `fetch_tokenizer_artifact(artifact, cache_dir)` | downloads the sealed tokenizer |
| `seal_tokenizer_files(dir)` | re-seals and fingerprints the exported tokenizer |

---

## 2. BLOCKING — the OLMo-core fork and commit

`export_checkpoint.py:40` imports `olmo_core.nn.transformer.qwen`, which does
not exist in our checkout (`edu-llm/OLMo-core`, commit `fa6c501`). We searched
the whole workspace; there is no `qwen.py` anywhere.

Needed symbols:

```
HF_EOS_TOKEN_ID, HF_HIDDEN_SIZE, HF_INTERMEDIATE_SIZE,
HF_NUM_ATTENTION_HEADS, HF_NUM_KV_HEADS, HF_NUM_LAYERS,
HF_RMS_NORM_EPS, HF_ROPE_THETA, HF_VOCAB_SIZE,
QWEN2_0_5B_HF_ID, QWEN2_0_5B_HF_REVISION,
QWEN2_0_5B_HF_WEIGHTS_SHA256, QWEN2_0_5B_HF_WEIGHTS_SIZE,
export_to_hf_state_dict, resolve_pinned_hf_snapshot
```

**Please give us the exact repo URL and commit SHA** you ran the export from, not
just a branch name. `run_eval.py` records `source_commit` in the result
provenance and `compare_arms.py` cross-checks it, so a drifting branch would
produce results that fail their own comparison gate.

The other OLMo-core imports (`distributed.checkpoint.unshard_checkpoint`,
`io.*`, `train.checkpoint.Checkpointer`) look standard and are probably present
in any recent fork; please confirm the same commit supplies them.

---

## 3. BLOCKING — tokenizer artifact access

`fetch_tokenizer_artifact()` pulls `{TOKENIZER_ARTIFACT_ID}/{TOKENIZER_ARTIFACT_VERSION}`
from an artifact store we can't identify from the code we have. We need:

- Where it resolves from (S3 prefix, internal registry, HF Hub?)
- Whether our sandbox role can read it, and the exact ARN/prefix if it's S3

Separately, `resolve_pinned_hf_snapshot()` fetches `config.json` from
`QWEN2_0_5B_HF_ID` at `QWEN2_0_5B_HF_REVISION` on the HuggingFace Hub. Confirm
that revision is public, or supply a token.

---

## 4. NEEDED — the handoff document

`Vishnu-Eval-Code/README.md` refers to
`memorysplit-requery-exact/p3-math-split-evals.md` for "full path setup, S3
roots, smoke commands, and required output layout." We don't have it. It
appears to define the expected output directory structure, which matters if you
want results in a specific place.

## 5. NEEDED — the arm configs for `compare_arms.py`

The example invocation passes `--dense-config $EVAL_WORK/configs/dense.json`
and `--split-config .../split.json`. Are those the `config.json` files from the
checkpoint directories (both present in S3), or separate platform run configs
we'd need shipped to us? `compare_arms.py` uses them to verify training-control
equality, so the wrong file would fail the gate.

## 6. NEEDED — corpus validation script

Step 1 of your execution order is
`assemble_v3_evaluator_root.py --check-only` on `corpus-v3/`. That script isn't
in what we received. We can skip it, but then we're running unvalidated against
a corpus you expect to be checked first — your call.

---

## Not needed, for confirmation

**`mm_expand.py` and the Metamath databases.** `run_eval.py:781` imports
`mm_expand.MM` only when `--mm-dir` is passed. But
`metamath_verifier_availability()` reports `"unavailable"` unless `mm_verify`
exposes `VERIFIER_SCHEMA_VERSION == "p3-metamath-tristate-v1"` and a callable
`verify_proof_tristate`, and your README says validity is deliberately out of
scope. So we plan to run **without** `--mm-dir`, which still gives Metamath its
NLL, next-token accuracy and exact-match numbers. Tell us if that's wrong.

If we do end up needing them, `corpus-v3/metamath_sources.json` pins
`set.mm`, `iset.mm` and `nf.mm` at `metamath/set.mm` commit `82830c78...`, which
we can fetch ourselves.

---

## Questions about the run itself

**Expected runtime.** The full matrix is 4,191 rows x 3 conditions = 12,573
evaluations per arm, 25,146 across both. Each is a chunked teacher-forced NLL
pass plus a greedy generation of up to 8,192 new tokens through HF `generate` at
batch 8 — no vLLM, no continuous batching. How long did your runs take, on what
hardware? We'll run the `--limit 1` smoke and extrapolate before committing, but
a reference number would save us a wrong-sized instance.

**Hardware floor.** BF16 is enforced in three places (`floating_state_dtype`,
the safetensors dtype check, and the `torch.bfloat16` model load), so a T4 is
ruled out. We plan on L4 (`g6.xlarge`, compute capability 8.9). Confirm that's
sufficient for 16,384 context at batch 8 on a 0.5B model with a 151,936 vocab,
or tell us you needed more memory.

**Parallelism.** `run_eval.py` walks families and conditions sequentially in one
process and writes a single JSON, and `compare_arms.py` validates matched
controls across the pair. If we shard by family across boxes and merge the
JSONs, will `compare_arms.py` accept a merged file, or does something in the
provenance hashing assume a single process wrote it? If merging is unsafe we'll
run one box per arm and accept the wall-clock.

**`--probe`.** Worth running? It loads every train shard (3.4 GB) to classify
fact-name visibility. Your docstring calls it the mechanism check — "if the probe
does not separate the arms, the loss mask did not do what it was supposed to" —
which sounds like something you'd want, but it isn't in the README's execution
order.
