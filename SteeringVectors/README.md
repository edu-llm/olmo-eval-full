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

**One checkpoint is enough.** The harness self-steers: it extracts the steering
vector from `CHECKPOINT_S3` and applies it back to that same model as the eval
target, so a single checkpoint yields a complete baseline-vs-steered result. Set
`TARGET_S3` only to steer a *different* model with the vector.

For one checkpoint (`CHECKPOINT_S3`):

1. **Materialize** the checkpoint from S3 onto the node, converting it to an
   HF-format directory when it is a raw OLMo-core checkpoint (see below).
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

## Checkpoint format: OLMo-core -> HF

TracingLLM's method needs a HuggingFace model (`AutoModelForCausalLM`, forward
hooks on `model.model.layers`). The OLMo checkpoints in S3
(`s3://edullm-checkpoints/olmo-370m/...`) are **raw OLMo-core** checkpoints
(`model_and_optim/` distributed shards + an olmo-core `config.json`), so the
harness converts them on the node before loading:

- `ensure_hf_checkpoint` detects the layout. HF dirs are used as-is.
- For OLMo-core, `convert_olmo_core_to_hf` rebuilds the `Transformer` from
  `config.json`, loads the distributed checkpoint with
  `olmo_core.distributed.checkpoint.load_model_and_optim_state`, writes HF
  weights via `olmo_core.nn.hf.save_hf_model` (bf16), and saves the checkpoint's
  tokenizer (dolma2) so the directory loads with `AutoTokenizer`.

`ai2-olmo-core[transformers]==2.4.0` (already a project dependency) provides the
conversion; the conversion runs in a single-rank gloo process group.

The default `CHECKPOINT_S3` in `run.env.example` points at one such checkpoint:
`s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/`
(a 370M, 12-layer / 512-dim step).

## Run it

This is an eduLLM platform job. Do not run it against AWS from a laptop — the
platform holds the credentials and the record (see the repo `AGENTS.md`).
`edullm submit` cannot inject env vars, so each stage's checkpoints and sweep are
baked directly into the command in `.edullm/run.yaml` (the env knobs in
`run.env.example` drive local `--dry-run` only).

Validate the plan locally first (no torch, no S3, no network):

```bash
python SteeringVectors/run_probe_dynamics.py s3://b/run/step625/ --dry-run
python SteeringVectors/run_steering_eval.py s3://b/run/step750/ \
  --target s3://b/run/step1315/ --dimensions toxigen truthfulqa confaide --dry-run
```

Price and submit through the platform (these read checkpoints, not a corpus, so
`--dataset none`):

```bash
edullm check  --dataset none
edullm submit --dataset none
```

### The two stages

The replication runs as separate submissions that share this image and the
staged checkpoints. First stage the 10 `refhq-5p5b` steps (125, 250, 375, 500,
625, 750, 875, 1000, 1125, 1315) under
`s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/replication-staged/step<N>/`
(the GPU workload role can read `teams/<team>/runs/*` but not `edullm-checkpoints`).

**Smoke — what `.edullm/run.yaml` submits as shipped.** A one-checkpoint probing
run against the already-staged `step1315`, 2 datasets × 3 layers × 200 statements,
to validate the whole path (OLMo-core→HF conversion, activation capture, probe
training, S3 upload) cheaply before the full sweep. It needs no new staging.

**Stage A — probing dynamics (full).** Once the 10 steps are staged, swap the
command to `run_probe_dynamics.py` over all 10 steps × 5 datasets × every layer:

```bash
python SteeringVectors/run_probe_dynamics.py \
  s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/replication-staged/step125/ \
  s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/replication-staged/step250/ \
  ... step375 step500 step625 step750 step875 step1000 step1125 step1315 ... \
  --datasets truthfulqa toxigen confaide stereoset sst2 \
  --max-statements 1000 --test-ratio 0.2 \
  --results-s3 "$EDULLM_OUTPUT_PREFIX" --run-name probe-dynamics-refhq --device cuda:0
```

**Stage B — steering intervention.** Swap the command block in `.edullm/run.yaml`
to the following, re-push the `edullm/**` branch (rebuilds nothing that changed),
and submit again:

```bash
python SteeringVectors/run_steering_eval.py \
  s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/replication-staged/step750/ \
  --target s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/replication-staged/step1315/ \
  --dimensions toxigen truthfulqa confaide \
  --layers 8 --alphas -2 -1 1 2 \
  --max-statements 1000 --eval-limit 200 --toxigen-limit 150 --max-new-tokens 64 \
  --results-s3 "$EDULLM_OUTPUT_PREFIX" \
  --run-name steering-refhq-750to1315 --device cuda:0
```

The steering vector is built from the mid-training source step (750) and applied
to the latest step (1315); `--eval-limit` / `--toxigen-limit` bound the sweep to
fit one `olmo-eval-sweep` cell. Raise the alpha/layer grid or the limits once a
first cell confirms the wiring.

## The one image decision

The image the platform builds for `olmo-eval-full` carries `transformers` but,
by default, **not** `torch` (`INSTALL_TORCH_AND_VLLM` defaults to `0` upstream).
This branch flips that default to `1` in `.edullm/Dockerfile`, so torch + vLLM
are baked into the image and the ~4.5 GiB torch pull no longer happens on the
billed GPU at the top of every run. The trade-off is the one the Dockerfile
documents: you pay for it once per image build instead of once per run, which is
the cheaper half for a job run more than a handful of times.

The command in `.edullm/run.yaml` still keeps its `uv sync --frozen ...
--extra vllm ...` prefix on purpose. `uv sync` is exact: dropping the `vllm`
extra would make it *remove* the baked torch/vLLM, and the sync is also what
layers the small `olmo_core` dependency set on top before the fork git install
pins the OLMo-core commit. Against the baked image that sync is a fast no-op for
torch and only resolves the `olmo_core` additions.

## OLMoE-7B — three-checkpoint full protocol

Paper-faithful TracingLLM on **OLMoE-7B** with **exactly three** checkpoints
(early, Chinchilla-optimal, final). See `PLAN_OLMOE7B.md`.

| Role | Used for |
|------|----------|
| **early** | Probing + steering-vector source |
| **chinchilla** | Probing + steering-vector source (~20× non-embedding params) |
| **final** | Probing + **all steered eval** (target model) |

Single entry point: `run_tracingllm_olmoe7b.py` runs probing → MI → steering
(early & chinchilla → final) → all 5 trustworthiness benchmarks → ARC/MMLU/MathQA/RACE
→ figure PNGs.

**Dry-run** (no torch, no S3):

```bash
python SteeringVectors/run_tracingllm_olmoe7b.py --dry-run
```

**Platform job:** copy `SteeringVectors/platform-run-olmoe7b.yaml` to
`.edullm/run.yaml` on branch `edullm/steering-olmoe7b`, stage three checkpoints
to the URIs in `checkpoints_olmoe7b.json`, then:

```bash
edullm check --json --dataset none
edullm submit --dataset none
```

Suggested compute: `gpu-8xl40s` (7B MoE + OLMo-core→HF conversion).

## Smoke: stage + steering vector (370M)

**Staging is not a platform job.** Batch roles cannot read `edullm-checkpoints`
(run `run_019fec94-69ac` failed with `AccessDenied`). Copy the checkpoint with
`sb_aws` (or `./SteeringVectors/stage_checkpoints.sh`) before the GPU smoke:

```bash
sb_aws s3 sync \
  s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/ \
  s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/steering-smoke/step1315/
```

| Job | Spec | Compute | What it does |
|-----|------|---------|--------------|
| **Stage** | `stage_checkpoints.sh` / `sb_aws` (local) | — | S3 sync into `teams/.../steering-smoke/` |
| **Vector** | `platform-run-vector-smoke.yaml` | `gpu-1xl40s` | OLMo-core→HF, build stereoset vector at layer 6, upload `.pt` + JSON |

Dry-run locally:

```bash
python SteeringVectors/generate_steering_vector.py --dry-run \
  --checkpoint s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/steering-smoke/step1315/
```

Submit GPU smoke (after staging):

```bash
edullm submit --dataset none --experiment steering-smoke-vector \
  --compute gpu-1xl40s --spec SteeringVectors/platform-run-vector-smoke.yaml
```

## Attribution

Upstream method and datasets: Qian et al., *Towards Tracing Trustworthiness
Dynamics: Revisiting Pre-training Period of Large Language Models*, arXiv
2402.19465. Vendored under its Apache-2.0 license (`vendor/TracingLLM/LICENSE`).
