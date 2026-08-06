# P3 math-split evaluation: vLLM backend, sharding, and AWS run

Handoff for Vishnu. Everything below is either done or verified. Written
2026-08-06.

Goal: run `run_eval.py` on both arms at step 23166, using vLLM for the
generation pass to make it fast, sharded four ways per arm across 8 L4 GPUs.

---

## 1. Code — pushed, nothing to bundle

Repository `edu-llm/OLMo-core`, branch `edullm/p3-math-split`, commit
`edacafe458de23073140b85cd60fe9fc134ea1c3` ("vishnu eval"), confirmed on
origin. 1,177 insertions across three files:


| File                                                        | Change         |
| ----------------------------------------------------------- | -------------- |
| `src/scripts/train/p3_math_split/evals/run_eval.py`         | +486 lines     |
| `src/scripts/train/p3_math_split/evals/merge_shards.py`     | new, 300 lines |
| `src/scripts/train/p3_math_split/evals/shard_merge_test.py` | new, 399 lines |


```bash
git fetch origin edullm/p3-math-split
git checkout edacafe458de23073140b85cd60fe9fc134ea1c3
```



### New flags on `run_eval.py`

```
--generation-backend {hf,vllm}      default hf; switches the generation pass only
--vllm-gpu-memory-utilization F     default 0.55
--vllm-max-model-len N              default = --context-length
--shard-index N                     0-based
--shard-count N                     default 1
--s3-out S3_PREFIX                  incremental result sync
```



### What the vLLM backend does and does not touch

**Generation only.** The teacher-forced NLL pass always runs through the
HuggingFace forward path regardless of backend, so
`target_token_micro_nll_per_token`, next-token accuracy, and the
`NLL_CONTEXT_POLICY` string remain exactly what they were. vLLM cannot supply
raw per-token logits over the 151,936 vocabulary, which is what
`chunked_sequence_nll` needs; porting NLL would mean `prompt_logprobs` and
abandoning the declared chunking policy. Deliberately not done.

The engine is constructed **before** the HF model loads, because vLLM sizes its
KV cache against free GPU memory. Both stay resident.

`generate()` keeps its module-level positional signature so the existing
monkeypatch in `run_eval_test.py:797` still works.

### Sharding — one subtlety that matters

Slicing happens **after** `rows_for_condition`, using a strided split
(`condition_rows[shard_index::shard_count]`).

**A naive implementation is silently wrong here.** Under `facts_corrupted`,
replacement statements are drawn from a single RNG stream advanced once per row.
A shard that iterates only its own rows lands at a different point in that
stream and scores *different prompts* than an unsharded run, producing entirely
plausible wrong numbers. The fix is that every shard walks the full cohort and
builds every prompt, skipping only the expensive model work:

```python
# Built for every row, including rows another shard scores. Under
# facts_corrupted the replacement statements come off a shared RNG
# stream, so skipping a row early would shift every later draw and this
# shard would score different prompts than the unsharded run.
prompt = build_prompt(row, condition, rng, corrupt_pool)
if scored_positions is not None and (i - 1) not in scored_positions:
    continue
```

`shard_merge_test.py` includes
`test_naive_slicing_would_corrupt_the_corrupted_condition`, which asserts the
buggy version actually fails, so the guard cannot pass vacuously. Six checks
total, all passing, and they run with no torch, transformers, vLLM or GPU.

Striding rather than contiguous blocks because row cost varies enormously:
enigma and prf2 average ~2,900 and ~3,200 target tokens against isabelle's ~321.

### Merging

`merge_shards.py` recomputes aggregates from per-example sufficient statistics
rather than averaging shard aggregates — averaging is wrong for token-micro
endpoints whose denominators differ per shard. It refuses to merge unless every
`input_provenance` hash agrees, `evaluation_controls` match apart from shard
fields, indices cover 0..N-1 exactly once, and no example ID is duplicated. The
merged total is anchored to family-level `evaluated_examples`, which no shard can
influence, so a truncated final shard cannot pass.

### Incremental S3 sync

Uploads fire after each `(family, condition)` cell. Shard-scoped keys:

```
{s3_out}/{arm}/shard{index}/_IN_PROGRESS   refreshed every cell
{s3_out}/{arm}/shard{index}/partial.json   marked "partial": true
{s3_out}/{arm}/shard{index}/result.json    final
{s3_out}/{arm}/shard{index}/_READY | _FAILED
```

All uploads are best-effort; transport errors log a warning and the run
continues. `result.json` retries three times and is written locally first.

**NLL canary at 8.0 nats/token.** Uniform prediction over Qwen's 151,936-token
vocabulary is `ln(151936) = 11.9312`, so 8.0 sits well above any healthy value
and well below chance. Tripping prints a banner and records
`nll_canary_tripped`. It detects *collapse* only — a perturbation shifting NLL a
few percent passes cleanly while still invalidating a comparison.

---



## 2. Findings you may not have

`compare_arms.py` **wants the CHECKPOINT** `config.json`**, not the exported one.**
`--dense-config` takes the file inside `checkpoints/step23166/config.json`
(6.3 KiB). The exported HF directory also contains a `config.json`, and passing
that fails immediately — it has no `arm`, `train_module`, or `init_seed`.
Validated locally against both arms:

```
validate_training_configs: PASS
ALL flattened differences (4):
  [ALLOWED] arm                              dense / split
  [ALLOWED] train_module.arm                 dense / split
  [ALLOWED] trainer.callbacks.wandb.name     run_019fd409-1654… / run_019fd409-e826…
  [ALLOWED] trainer.save_folder              teams/platform/… / teams/memory-split/…
```

Both declare `init_seed: 42` and share `source_commit`
`d165d6cada69f8146df280d39ee7b46f5432f218`.

**Metamath parsing is cheap.** 6.19 s wall clock (`set.mm` 4.96, `iset.mm` 0.95,
`nf.mm` 0.28), peak RSS 1.13 GB. Pass `--mm-dir` everywhere; there is nothing to
optimize.

**flash-attn is not needed on the eval path.** Every top-level `olmo_core` import
is guarded; the two unguarded ones in `rope.py:604` and `layer_norm.py:335` sit
inside function bodies that only fire when constructing flash-attn-backed
modules. `run_eval.py` loads through `AutoModelForCausalLM` and
`export_checkpoint.py` only does state-dict conversion.

**Torch pins are compatible.** vLLM 0.19.1 requires `torch==2.10.0`; 0.21.0+
requires 2.11.0. OLMo-core requires `>=2.8.0`. No conflict, no need to split into
two virtualenvs.

**Cohort sizes.** `facts_present` at fraction 1.0, `facts_absent` and
`facts_corrupted` at 0.10 with ceil rounding, selected by
`deterministic-rank-sample-v1`. That is 4,191 + 422 + 422 = **5,035 evaluations
per arm**, not 12,573.

**Corpus shape.** 4,191 eval rows: mizar 2,485, isabelle 590, metamath 494,
prf2 313, enigma 263, thproofs 46.

`build_evaluation_metadata` **requires the raw train shards.** It hashes
`(corpus/"shards").glob("*.jsonl")` and raises if an evaluated family lacks one,
so all 3.38 GB must be present. The S3 copy at
`s3://edullm-data/pretrain/formal-proof-premises-500m/v3/` is *tokenized*
`.u32le.bin` and cannot substitute.

---



## 3. AWS

Account `sbsandbox` (**056956104102**), region **us-east-1**.

### Checkpoints under evaluation


| Arm   | Path                                                                                                                          |
| ----- | ----------------------------------------------------------------------------------------------------------------------------- |
| dense | `s3://sbsandbox-intern-edullm-outputs/teams/platform/runs/run_019fd409-1654-7068-aaf2-003c275e2556/checkpoints/step23166`     |
| split | `s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs/run_019fd409-e826-7024-b8b4-2cc03d1551d2/checkpoints/step23166` |




### Results location — this is what to watch

```
s3://sbsandbox-intern-edullm-outputs/teams/eval/vishnu-p3/
    {dense,split}/shard{0..3}/_IN_PROGRESS
    {dense,split}/shard{0..3}/partial.json
    {dense,split}/shard{0..3}/result.json
    {dense,split}/shard{0..3}/_READY | _FAILED
    bootstrap/                       staged corpus, mm databases, user-data
```

Monitor with `olmo-eval-full/Vishnu-Evals/poll_p3_run.py`:

```bash
python poll_p3_run.py --s3-out s3://sbsandbox-intern-edullm-outputs/teams/eval/vishnu-p3 --watch 60
```

```
arm    sh state    cells  rows   age  nll    gen   cap%  cell
dense  0  running   7/18   412   12s  1.842   613     4  metamath/facts_absent
split  2  FAILED    3/18    88    9m  9.912     -     -  enigma/facts_present  <-- NLL CANARY
```

It flags failed shards, tripped canaries, and any shard silent for over 90
seconds.

### Verified launch parameters


| Item             | Value                                                                                                    |
| ---------------- | -------------------------------------------------------------------------------------------------------- |
| Instance         | `g6.xlarge`, 1x L4 24 GB, 4 vCPU, 16 GiB, $0.8048/hr                                                     |
| AMI              | `ami-07bcf82131b289395` — DLAMI GPU PyTorch 2.10, Ubuntu 24.04                                           |
| Subnet           | `subnet-0fd5ed8accae254dc` — public, `rtb-09a566c0412dadcc1` routes 0.0.0.0/0 to `igw-01b5e62549ac297a7` |
| Security group   | `sg-087218d8c87aa8576`                                                                                   |
| Instance profile | `edullm-eval-smoke`                                                                                      |
| G-family quota   | 3,696 vCPU; our 32 takes usage to ~180                                                                   |


The PyTorch 2.10 AMI is chosen deliberately: it matches vLLM 0.19.1's hard
`torch==2.10.0` pin, so no torch is reinstalled at boot.

L4 is required, not preferred. The harness enforces BF16 in three places
(`floating_state_dtype`, the safetensors dtype validators, the `torch.bfloat16`
model load) and vLLM refuses BF16 below compute capability 8.0. T4 is ruled out.

### IAM

Inline policy `edullm-eval-vishnu-p3-s3` was added to the existing role
`edullm-eval-smoke`. Additive only — the two pre-existing policies
(`edullm-eval-smoke-s3`, `edullm-eval-split-s3`) were captured before and read
back byte-identical after. Grants: read on both checkpoint prefixes, read on
`s3://edullm-data` (required by `fetch_tokenizer_artifact` for
`tokenizer/qwen25-vendored/v1`), and read/write on `teams/eval/vishnu-p3/*`.

### Self-termination

Four layers, adapted from a fleet that ran successfully earlier the same day:

1. Sweep exit trap calls `shutdown -h now` on completion — the normal path.
2. Stall watchdog, `shutdown -h now` after 20 minutes with no local file
  activity, using `find "$WORK" -newermt`.
3. `sudo shutdown -h +480` scheduled as the first line of user-data, before
  `set -euo pipefail`, so it is armed before anything can fail.
4. `--instance-initiated-shutdown-behavior terminate`, so a halt terminates and
  releases EBS rather than merely stopping.

The 8-hour backstop is generous on purpose: EOS behaviour is unmeasured, so the
run could be ~30 minutes or ~4 hours per arm. A backstop that fires mid-sweep
loses everything; an over-generous one costs nothing because boxes self-terminate
on completion.

Verify after boot by reading `/run/systemd/shutdown/scheduled` on each box rather
than trusting the launch config.

### Cost

Expected **$13–15**, worst case **~$52** if every box rides the full backstop.
S3-to-EC2 is free within us-east-1.

---



## 4. Commands

Export, per box:

```bash
python src/scripts/train/p3_math_split/evals/export_checkpoint.py \
  --run s3://sbsandbox-intern-edullm-outputs/teams/${TEAM}/runs/${RUN}/checkpoints \
  --step 23166 --out /mnt/work/hf/${ARM} --work-dir /mnt/work/unshard
```

`TEAM` is `platform` for dense, `memory-split` for split.

Evaluate, one process per box, `SHARD` in 0..3:

```bash
python src/scripts/train/p3_math_split/evals/run_eval.py \
  --model /mnt/work/hf/${ARM} --arm ${ARM} \
  --corpus /mnt/work/corpus-v3 --mm-dir /mnt/work/mm \
  --generation-backend vllm \
  --vllm-gpu-memory-utilization 0.55 --vllm-max-model-len 16384 \
  --context-length 16384 --max-new-tokens 8192 --nll-chunk-size 256 \
  --batch-size 16 --seed 20260801 \
  --conditions facts_present facts_absent facts_corrupted \
  --shard-count 4 --shard-index ${SHARD} \
  --s3-out s3://sbsandbox-intern-edullm-outputs/teams/eval/vishnu-p3 \
  --out /mnt/work/results/${ARM}.shard${SHARD}.json
```

`--batch-size` is inert under the vLLM backend; it affects only the HF
generation path and the probe.

Merge and compare:

```bash
python src/scripts/train/p3_math_split/evals/merge_shards.py \
  --shards results/${ARM}.shard*.json --out results/${ARM}.json

python src/scripts/train/p3_math_split/evals/compare_arms.py \
  --dense results/dense.json --split results/split.json \
  --dense-config cfg/dense_config.json --split-config cfg/split_config.json \
  --out results/comparison.json
```

---



## 5. Risks accepted — no smoke test was run

The user chose to skip the smoke stage. These are unverified:

- **The vLLM generation backend has never executed.** Not once. First execution
is the full sweep.
- **NLL contamination is unchecked.** Nothing verifies that standing up the vLLM
engine leaves the shared HF forward path untouched. If it does not, the primary
endpoint is wrong and no downstream check reveals it — `compare_arms.py`
validates internal consistency, not correctness against a reference. The NLL
canary catches collapse only.
- **The sharded path is proven arithmetically, not against a real model.** The
merge logic and RNG-preservation property are covered by tests that run without
torch; `run_eval.py`'s sharded path against actual weights is untested.
- **EOS behaviour is unmeasured.** If generations run to the 8,192 cap rather
than stopping, the run is roughly 9x longer than estimated.

The intended mitigation is the incremental sync: all four become visible in the
first few cells rather than at the end. Watch the poller early.

---



## 6. Helper scripts

In `olmo-eval-full/Vishnu-Evals/` (not part of the OLMo-core commit):


| Script                      | Purpose                                                                       |
| --------------------------- | ----------------------------------------------------------------------------- |
| `poll_p3_run.py`            | live per-shard progress from S3                                               |
| `fetch_metamath_sources.py` | fetch and SHA-256 verify set.mm/iset.mm/nf.mm against `metamath_sources.json` |
| `measure_corpus.py`         | size both passes from the corpus                                              |
| `check_backend_dispatch.py` | verify the backend dispatch without a GPU                                     |
| `INCORPORATE_VLLM.md`       | design notes on the vLLM integration                                          |
| `VISHNU_EVAL_REQUEST.md`    | original dependency questions, now largely resolved                           |


