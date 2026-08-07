# SteeringVectors: trustworthiness steering over S3 checkpoints

This branch vendors [ChnQ/TracingLLM](https://github.com/ChnQ/TracingLLM)
("Towards Tracing Trustworthiness Dynamics: Revisiting Pre-training Period of
Large Language Models") and wraps its steering-vector method in a job that runs
against model checkpoints stored in **AWS S3** rather than HuggingFace
revisions.

## Layout

| Path | Purpose |
|---|---|
| `vendor/TracingLLM/` | Upstream code, datasets, and license, vendored verbatim (reference + CSV data). |
| `run_steering_eval.py` | The S3-aware job harness (checkpoint -> steering vector -> intervention eval -> results). |
| `run.env.example` | Config template for the harness / submission. |
| `../.edullm/run.yaml` | The platform job spec that invokes the harness. |

## What the job does

For one checkpoint (`CHECKPOINT_S3`):

1. **Materialize** the checkpoint from S3 onto the node (HF-format directory).
2. **Build steering vectors** — collect last-token residual-stream activations on
   a probing dataset (`STEERING_DATASET`) and form, per layer, the
   mean-difference direction between the two classes, unit-normalized and scaled
   by the projection standard deviation. This reproduces
   `vendor/TracingLLM/src/generate_steering_vector.py`.
3. **Evaluate** a self-contained discriminative trustworthiness task
   (`EVAL_DATASET` in `stereoset` / `sst2` / `confaide`) on the target model
   (`TARGET_S3`, default = the checkpoint) at **baseline** and again with the
   steering vector injected as a forward hook, for every `LAYERS` × `ALPHAS`
   cell. This reproduces the discriminative path of
   `vendor/TracingLLM/src/eval_trustworthiness.py`.
4. **Write** `metrics.json` (baseline vs. steered accuracy and deltas per cell)
   and `predictions.jsonl`, locally and to `RESULTS_S3/<run>/step<N>/`.

`truthfulqa` and `toxigen` are supported upstream but need an external judge
(OpenAI / toxigen-roberta), so they are out of scope for this self-contained
test. `truthfulqa` may still be used as a `STEERING_DATASET` (vector source).

The activation-capture, steering-vector, hook, and prompt logic are
reimplemented in the harness rather than imported: the vendored modules import
`openai`, `scikit-learn`, `tqdm`, and `matplotlib` at load, which the platform
image does not carry. Only the vendored CSV datasets are read directly.

## Run it

This is an eduLLM platform job. Do not run it against AWS from a laptop — the
platform holds the credentials and the record (see the repo `AGENTS.md`).

Validate the plan locally first (no torch, no S3, no network):

```bash
source SteeringVectors/run.env.example   # after filling in CHECKPOINT_S3 / RESULTS_S3
python SteeringVectors/run_steering_eval.py "$CHECKPOINT_S3" --dry-run
```

Price and submit through the platform (checkpoint eval reads no corpus, so
`--dataset none`):

```bash
edullm check  --experiment steering-vectors --dataset none
edullm submit --experiment steering-vectors --dataset none
```

The checkpoint and output locations are passed to the run through the submission
environment (`CHECKPOINT_S3`, `RESULTS_S3`, and the optional knobs in
`run.env.example`); `.edullm/run.yaml` reads them.

### Sweeping many checkpoints

`.edullm/run.yaml` uses the `olmo-eval-sweep` workload profile. Point one cell at
each checkpoint step with a fan-out (`edullm check --fanout-size N
--fanout-index-parameter ...`) or submit one run per step, grouped under a shared
`--experiment`.

## The one image decision

The image the platform builds for `olmo-eval-full` carries `transformers` but
**not** `torch` (`INSTALL_TORCH_AND_VLLM` defaults to `0` in
`.edullm/Dockerfile`). This pipeline needs torch, so the job command in
`.edullm/run.yaml` first runs `uv sync ... --extra vllm` to materialize torch
from the checked-in lockfile at the start of the run — the run-time alternative
the Dockerfile itself documents.

The trade-off: that download (~4.5 GiB) happens once per run on a billed GPU. If
this job is run often, flip `ARG INSTALL_TORCH_AND_VLLM=0` to `=1` in
`.edullm/Dockerfile` on this branch instead (a one-line change that bakes torch
into the image) and drop the `uv sync` prefix from the command. That changes the
image for every olmo-eval run built from this branch, which is why it is left as
a deliberate choice rather than made here.

## Attribution

Upstream method and datasets: Qian et al., *Towards Tracing Trustworthiness
Dynamics: Revisiting Pre-training Period of Large Language Models*, arXiv
2402.19465. Vendored under its Apache-2.0 license (`vendor/TracingLLM/LICENSE`).
