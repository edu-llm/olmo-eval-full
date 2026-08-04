# Flow 1: Training Checkpoint Eval (on-node, custom inference TEST)

Plan for a small **standalone** script that a training team drops into their run
script. Every time training writes a new checkpoint to S3, the script loads that
checkpoint **on the training node**, runs a few inferences directly, and writes the
outputs back to S3.

This is the concrete build plan for **Flow 1** in the parent [`Plan/README.md`](../README.md).
It stays at the "inference-only TEST" stage on purpose: prove a checkpoint can be
loaded, produce generations, and land them in S3. Scoring, IRT, and CAT diagnostics
are added later on branches off `CheckpointFlows` and are **out of scope here**.

> This script does **not** call `olmo-eval run` or any of the suite's CLI/runner
> code. It is a self-contained inference script (model library + boto3). The only
> shared concept is the parent plan's staging (prove inference first, add diagnostics
> later).

## Goal

One command the training team calls per checkpoint:

```bash
python tests/OnNode/checkpoint_infer.py s3://<bucket>/checkpoints/<owner>/<run>/step<N>/
```

Each invocation:

1. Takes a **different** checkpoint S3 URI (the one just written).
2. Downloads/loads that checkpoint on the local GPUs.
3. Runs a few random inferences (a handful of prompts through the model).
4. Uploads a single results file to a deterministic S3 location.
5. Exits without blocking the training loop.

## Script responsibilities (step by step)

1. **Parse + validate inputs:** checkpoint URI is `s3://...`; required config present;
   AWS credentials resolvable (env / `~/.aws` / IAM role). Fail fast with a clear message.
2. **Materialize the checkpoint locally.** Sync the checkpoint prefix from S3 to a
   per-step temp dir (e.g. `/tmp/ckpt-infer/step<N>/`) with boto3 (or `aws s3 sync`).
   Skip re-download if already present.
3. **Load the model** onto the local GPU(s) — see [Checkpoint loading](#checkpoint-loading).
4. **Select prompts.** Randomly sample `NUM_PROMPTS` prompts (fixed seed for
   reproducibility) from a bundled prompt pool (`scripts/prompts.jsonl`) or a
   caller-provided prompts file.
5. **Run generation** with fixed sampling params (max tokens, temperature, seed).
6. **Assemble a results record** (see [Output format](#output-format)).
7. **Upload results to S3** with boto3 to a deterministic per-step key.
8. **Clean up** the local temp dir and log the resolved S3 result URI.

## Checkpoint loading

The training team declares which format they save via `CHECKPOINT_KIND`:

- **`hf`** — a Hugging Face format directory. Load with `transformers`
  (`AutoModelForCausalLM.from_pretrained(local_dir)` + `AutoTokenizer`) or, for speed,
  `vllm.LLM(model=local_dir)`. Lowest-friction option.
- **`olmo_core`** — the raw native trainer checkpoint. Load with the **OLMo-core
  library that already exists in the training environment** (the training node can
  reconstruct the model + tokenizer from the run config and load the checkpoint
  weights). This avoids a separate HF conversion step on-node.

Loading is isolated behind one function (`load_model(local_dir, kind) -> (model, tokenizer)`)
so the two branches don't leak into the rest of the script.

## Inference

- **Prompts:** a small JSONL pool bundled with the script; `NUM_PROMPTS` are sampled
  at random with a fixed seed so runs are comparable across checkpoints.
- **Generation:** greedy or low-temperature, short `MAX_NEW_TOKENS`. The TEST is
  intentionally tiny so it can share the training node without stalling the loop.
- **Determinism:** seed the prompt sampler and the generator so step-to-step diffs
  reflect the model, not sampling noise.

## Output format

A single JSON/JSONL file per checkpoint, e.g. `results.jsonl`, with one row per
prompt plus a small run header:

```json
{"checkpoint": "s3://.../step1000/", "step": 1000, "run": "<run>",
 "prompt": "...", "output": "...", "gen_params": {"max_new_tokens": 64, "temperature": 0.0},
 "timestamp": "2026-08-04T18:30:00Z", "wall_time_s": 3.1}
```

Also emit a tiny `manifest.json` (checkpoint URI, step, prompt count, model load time,
success flag) so downstream tooling can tell a step succeeded at a glance.

## S3 output layout

Deterministic, keyed by run + step so nothing collides across invocations:

```
s3://{RESULTS_BUCKET}/{RESULTS_PREFIX}/{RUN_NAME}/step{N}/results.jsonl
s3://{RESULTS_BUCKET}/{RESULTS_PREFIX}/{RUN_NAME}/step{N}/manifest.json
```

The step number is parsed from the checkpoint URI (`.../step<N>/` or `.../step<N>-hf/`).

## Script interface

**Positional argument:** the checkpoint S3 URI (changes every call).

**Configuration (env vars or a small `checkpoint_infer.env` file):**

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `RESULTS_BUCKET` | yes | — | Bucket for results |
| `RESULTS_PREFIX` | no | `checkpoint-infer` | Prefix within the bucket |
| `RUN_NAME` | yes | — | Groups all steps of one training run |
| `CHECKPOINT_KIND` | no | `hf` | `hf` or `olmo_core` |
| `PROMPTS_FILE` | no | bundled `prompts.jsonl` | Prompt pool to sample from |
| `NUM_PROMPTS` | no | `8` | How many random prompts to run |
| `MAX_NEW_TOKENS` | no | `64` | Generation length cap |
| `TEMPERATURE` | no | `0.0` | Sampling temperature |
| `SEED` | no | `1234` | Prompt-sampling + generation seed |
| `NUM_GPUS` | no | `1` | GPUs / tensor-parallel degree |
| `CUDA_VISIBLE_DEVICES` | no | — | Pin the TEST to spare GPU(s) on the node |
| `AWS_REGION` | no | `us-east-1` | Region for checkpoint + result buckets |
| `S3_ENDPOINT_URL` | no | — | For S3-compatible stores |
| `BLOCKING` | no | `false` | If `false`, run in background so training is not blocked |

The training team sets these once; only the checkpoint URI changes per call.

## Dependencies

Kept minimal and installed in the training env (or a small sidecar venv):

- `boto3` — S3 download/upload.
- One inference backend, matching `CHECKPOINT_KIND`:
  - `hf`: `transformers` (+ `torch`, already present) or `vllm`.
  - `olmo_core`: the OLMo-core package already used by the trainer.

No dependency on this repo's `olmo_eval` package.

## Integration for the training team (the hook)

Minimal: one line after each checkpoint save, passing the new location.

```bash
# after the trainer finishes writing step $STEP
CKPT="s3://$RESULTS_BUCKET/checkpoints/$OWNER/$RUN_NAME/step${STEP}/"
python tests/OnNode/checkpoint_infer.py "$CKPT" &   # backgrounded so training is not blocked
```

## Deliverables

```
tests/OnNode/
  checkpoint_infer.py          # the standalone inference script (this plan's output)
  checkpoint_infer.env.example # documented config template
  prompts.jsonl                # default prompt pool for the TEST
  README.md                    # folder usage
  skill/SKILL.md               # skill for a training team's agent to install + wire up
Plan/flow1_training_checkpoint/
  README.md                    # this plan
```

## Validation

Follows Step 2 of the parent plan (mock training run):

1. **CPU/dry smoke test:** run against a tiny local checkpoint (or `--limit-prompts 1`)
   to confirm load → generate → S3 write end to end.
2. **Single real checkpoint:** point at one existing checkpoint on S3; confirm
   `results.jsonl` + `manifest.json` appear under the expected `step{N}/` key.
3. **Mock loop:** a fake training script that writes N dummy checkpoints and fires the
   hook each time; confirm each step produces a distinct S3 folder and the loop is
   never blocked (`BLOCKING=false`).
4. **Both formats:** exercise `CHECKPOINT_KIND=hf` and `olmo_core`.

## Design decisions

- **Standalone, no `olmo-eval` runtime.** Per the requirement, the script owns the
  full load → infer → upload path itself and depends only on a model backend + boto3.
- **On-node local inference.** Uses the training node's GPUs; `CUDA_VISIBLE_DEVICES`
  lets the team pin it to a spare device.
- **Non-blocking + tiny by default.** Small prompt count and short generations so it
  does not steal training throughput; backgrounded unless `BLOCKING=true`.
- **Deterministic S3 layout keyed by run + step.** Per-step separation and easy
  discovery, independent of any suite conventions.
- **Fixed seeds.** Same prompts across checkpoints so outputs are comparable step over
  step.

## Out of scope (later branches)

- Scoring, IRT, and CAT diagnostics (Uni-MCQ, MIRT-MCQ, Uni-FRQ, MIRT-FRQ).
- Benchmark selection and full-suite runs.
- The retroactive manager over existing checkpoints (that is Flow 2).

## Open questions for the training team

1. Checkpoint format actually saved: HF (`step<N>-hf/`) or raw OLMo-core (`step<N>/`)?
2. For `olmo_core`, what is needed to reconstruct the model on-node (run config path,
   tokenizer, expected checkpoint file layout)?
3. Spare GPU budget on the node for the TEST, or must it run only while training is
   paused at the checkpoint boundary?
4. Preferred prompt pool and `NUM_PROMPTS` / `MAX_NEW_TOKENS` for the smoke test.
5. Do checkpoint and result buckets share credentials/region, or differ?
