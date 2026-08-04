# Mock training run (Flow 1 validation)

Step 2 of the `CheckpointFlows` plan: a fake, CPU-only training loop that writes
a checkpoint every few steps and, right after each checkpoint lands in S3, fires
the standalone Flow 1 hook (`../checkpoint_infer.py`) to run inference on it. It
proves the whole flow end to end — each step produces a distinct set of inference
outputs in S3 and the loop is never blocked on inference.

See `Plan/flow1_training_checkpoint/README.md` (Validation) and
`../skill/SKILL.md` (Step 5) for the design this exercises.

## Files

- `mock_training_run.py` — the fake trainer. Builds a tiny GPT-2 model + byte-level
  tokenizer locally (no Hub download), perturbs the weights per step so each
  checkpoint differs, `save_pretrained`s it, uploads it to
  `s3://{bucket}/checkpoints/{owner}/{run}/step{N}/`, then fires the hook
  (`python ../checkpoint_infer.py "$CKPT" &`) non-blocking. Waits for the
  background inference jobs at the end.
- `validate_results.py` — lists the S3 keys the run produced and checks each step
  has `results.jsonl` + `manifest.json`, each manifest reports `success: true`
  with the expected `num_prompts`, and generations differ across steps.
- `run_mock_validation.sh` — orchestrator: starts a local `moto` S3 endpoint,
  sets dummy creds + env, creates the bucket, dry-runs the hook, runs the trainer,
  then runs the validator. No Docker, no GPU, no real AWS.

## Run the local validation

```bash
uv pip install "moto[server]"        # one-time: local S3, plus boto3/transformers/torch
bash tests/OnNode/mock_training/run_mock_validation.sh
```

Tunables (env overrides, all optional):

```bash
MOCK_NUM_STEPS=5 MOCK_STEP_SIZE=50 NUM_PROMPTS=4 MAX_NEW_TOKENS=16 MOTO_PORT=5001 \
  bash tests/OnNode/mock_training/run_mock_validation.sh
```

The script prints the dry-run target path, the per-step S3 key listing, an example
`manifest.json`, and a couple of `results.jsonl` rows from two different steps, and
ends with `RESULT: PASS`.

## Point the SAME trainer at REAL S3

The trainer and the hook read the same environment as `checkpoint_infer.py`. To run
against real S3, just change the environment — no code changes:

```bash
export AWS_ACCESS_KEY_ID=...            # real creds (or rely on ~/.aws / IAM role)
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
unset S3_ENDPOINT_URL                   # <-- drop the local moto endpoint
export RESULTS_BUCKET=my-real-bucket    # must already exist
export RUN_NAME=my-run
export CHECKPOINT_KIND=hf
export NUM_PROMPTS=8 MAX_NEW_TOKENS=64 NUM_GPUS=1

uv run --no-sync python tests/OnNode/mock_training/mock_training_run.py
uv run --no-sync python tests/OnNode/mock_training/validate_results.py   # optional check
```

Notes:

- With no `S3_ENDPOINT_URL`, boto3 talks to real AWS; the code path is otherwise
  identical to the local run.
- The results bucket must exist beforehand (the local orchestrator creates the moto
  bucket up front; real S3 buckets are assumed pre-provisioned).
- On a GPU node set `NUM_GPUS>=1` so the hook uses `device_map="auto"`; the local
  validation uses `NUM_GPUS=0` to stay on CPU.
