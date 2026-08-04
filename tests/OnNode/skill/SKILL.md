---
name: onnode-checkpoint-infer
description: >-
  Install and wire up the on-node checkpoint inference TEST (Flow 1) into a model
  training run. Downloads checkpoint_infer.py, its config template, and prompt pool
  from the olmo-eval-full repo, then adds a per-checkpoint call to the team's training
  script so each new checkpoint runs a few inferences and uploads outputs to S3. Use
  when a training team wants to run quick inference on every checkpoint saved to S3,
  or mentions checkpoint eval, on-node inference, or Flow 1.
---

# On-node checkpoint inference TEST (Flow 1)

Sets up a small, standalone script that runs a few inferences on each training
checkpoint (on the training node) and saves the outputs to S3. The script does not
depend on the `olmo_eval` package.

Design reference: `Plan/flow1_training_checkpoint/README.md` in
`edu-llm/olmo-eval-full` (branch `CheckpointFlows`).

## Setup workflow

Copy this checklist and track progress:

```
- [ ] Step 1: Download the script files into the training repo
- [ ] Step 2: Install dependencies
- [ ] Step 3: Configure via checkpoint_infer.env
- [ ] Step 4: Add the per-checkpoint call to the training script
- [ ] Step 5: Validate with --dry-run, then one real checkpoint
```

### Step 1: Download the files

Fetch the three files into a `tools/checkpoint_infer/` directory in the training repo:

```bash
BASE="https://raw.githubusercontent.com/edu-llm/olmo-eval-full/CheckpointFlows/tests/OnNode"
mkdir -p tools/checkpoint_infer
curl -fsSL "$BASE/checkpoint_infer.py"          -o tools/checkpoint_infer/checkpoint_infer.py
curl -fsSL "$BASE/checkpoint_infer.env.example" -o tools/checkpoint_infer/checkpoint_infer.env.example
curl -fsSL "$BASE/prompts.jsonl"                -o tools/checkpoint_infer/prompts.jsonl
```

### Step 2: Install dependencies

Always needed: `boto3`. Then one backend matching the checkpoint format:

- HuggingFace-format checkpoints (`CHECKPOINT_KIND=hf`): `transformers` + `torch`.
- Raw OLMo-core checkpoints (`CHECKPOINT_KIND=olmo_core`): the `ai2-olmo-core` package
  already used by the trainer (see Step 4 note).

```bash
pip install boto3 transformers torch   # hf backend
```

### Step 3: Configure

```bash
cd tools/checkpoint_infer
cp checkpoint_infer.env.example checkpoint_infer.env
```

Edit `checkpoint_infer.env` and set at minimum:

- `RESULTS_BUCKET` — S3 bucket for outputs.
- `RUN_NAME` — groups all steps of this run.
- `CHECKPOINT_KIND` — `hf` or `olmo_core`.

Keep `NUM_PROMPTS` and `MAX_NEW_TOKENS` small so the TEST shares the training node
without stalling the loop. Set `CUDA_VISIBLE_DEVICES` to pin it to a spare GPU.

### Step 4: Add the call to the training script

Find where the training loop finishes writing a checkpoint to S3. Right after that,
add a call passing the just-written checkpoint URI. Run it backgrounded (`&`) so
training is never blocked:

```bash
# after the checkpoint for $STEP is fully written to S3
source tools/checkpoint_infer/checkpoint_infer.env
CKPT="s3://${RESULTS_BUCKET}/checkpoints/${OWNER}/${RUN_NAME}/step${STEP}/"
python tools/checkpoint_infer/checkpoint_infer.py "$CKPT" &
```

Only the checkpoint URI changes per call; everything else comes from the env file.

**Wiring guidance:**
- Locate the checkpoint-save call (search the training code for `save`, `checkpoint`,
  `s3://`, or the step counter) and insert the hook immediately after the upload
  completes, not before.
- Match `${STEP}` to the training script's real step variable and the checkpoint URI
  to the exact layout the trainer writes (`step<N>/` vs `step<N>-hf/`).
- If checkpoints are raw OLMo-core, implement the `_load_olmo_core` function in
  `checkpoint_infer.py` using the run's config + tokenizer (it ships as a marked
  `NotImplementedError` integration point).

### Step 5: Validate

```bash
source tools/checkpoint_infer/checkpoint_infer.env
# dry run: checks config + prompts, prints the target S3 path, no model load
python tools/checkpoint_infer/checkpoint_infer.py s3://.../step1000/ --dry-run
# real run on one checkpoint
python tools/checkpoint_infer/checkpoint_infer.py s3://.../step1000/
```

Confirm `results.jsonl` and `manifest.json` appear under
`s3://$RESULTS_BUCKET/$RESULTS_PREFIX/$RUN_NAME/step<N>/`.

## Notes

- The script writes only to S3; the local checkpoint download uses a temp dir that is
  cleaned up after each run.
- Fixed `SEED` means the same prompts are used across checkpoints, so outputs are
  comparable step over step.
- Scoring / IRT / CAT diagnostics are out of scope for this TEST (later Flow branches).
