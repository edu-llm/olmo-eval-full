# ATLAS CAT checkpoint diagnostic (training hook)

Drop-in commands for a training job: after a checkpoint lands on S3, launch one
`g6.xlarge` (1× L4) worker that runs the online ATLAS CAT and writes results to a
**fixed S3 prefix**.

## Scope today: 4 benchmarks

This diagnostic wires **4 of the 5 ATLAS benchmarks**: `atlas_arc` (ARC-Challenge),
`atlas_hellaswag`, `atlas_winogrande` (MCQ log-likelihood), and `atlas_gsm8k`
(generative exact-match). Each is backed by a calibrated ATLAS 3PL bank vendored in
this repo under `AdaptiveTesting/Inputs/ATLAS/<benchmark>/` (params +
`atlas_idx_to_question_id.csv`).

**TruthfulQA is not yet wired** — it has mixed scoring and no olmo-eval base task, so
it stays out until a base task exists. That's the one remaining benchmark to reach 5/5.

Id-bridge note: HellaSwag joins on the native `ind`; WinoGrande and GSM8K use
positional indices, so their `atlas_idx_to_question_id.csv` bridges were generated
against the exact split ordering olmo-eval enumerates (see
`AdaptiveTesting/Inputs/ATLAS/scripts/build_atlas_idx_bridge.py`). The bank join keeps
only items present in both the bank and the task; a mismatch degrades to fewer items
rather than misaligning.

## Known output location

```
s3://edullm-adaptive-inference-056956104102/smoke/atlas_cat/<run_id>/
  atlas_arc.json           # theta, se, pirt_accuracy, n_items, selected ids
  atlas_hellaswag.json     # one JSON per eval in EVALS
  atlas_winogrande.json
  atlas_gsm8k.json
  pipeline_provenance.json # checkpoint URI, instance id, git sha, evals, args
  worker.log               # full worker stdout (best-effort)
```

Region: `us-east-1`. Account: `056956104102` (sbsandbox / edullm adaptive).

> The `smoke/` prefix is used because the worker's `EswManagedInstance` role only
> has `s3:PutObject` on `smoke/*`, `smoke_split/*`, `full200/*`. To write elsewhere,
> set `S3_OUT_ROOT` to a prefix the worker role can write to (or grant the role
> `s3:PutObject` on that prefix).

## Training-script snippet

```bash
# After you write the checkpoint (HF / OLMo-core layout) to S3:
CHECKPOINT_S3="s3://YOUR_BUCKET/checkpoints/${EXP}/${STEP}"
RUN_ID="${EXP}-step${STEP}"   # must be S3-key safe: [A-Za-z0-9._-]+

# From a checkout of edu-llm/olmo-eval-full @ AdaptiveEvals (or pin a SHA):
bash AdaptiveTesting/scripts/atlas_cat_diagnose/launch_g6.sh \
  --checkpoint "${CHECKPOINT_S3}" \
  --run-id "${RUN_ID}"
# Add DRY_RUN=1 to print the aws call without launching.
```

Optional flags / env:

| Flag / env | Default | Meaning |
|---|---|---|
| `--checkpoint` | required | `s3://` or HF id the worker should evaluate |
| `--run-id` | required | results subdirectory under `atlas_cat/` |
| `--se-stop` | `0.3` | CAT early-stop SE |
| `--max-items` | `40` | CAT cap |
| `--evals` | 4 ATLAS evals | space-separated online evals (share one vLLM boot) |
| `--branch` | `AdaptiveEvals` | git branch the worker clones |
| `DRY_RUN=1` | off | print `run-instances` only |
| `INSTANCE_TYPE` | `g6.xlarge` | preferred; script falls back to `g5.xlarge` if g6 has no capacity |
| `SUBNET` | (auto-rotate) | pin one subnet/AZ if you want; otherwise tries all VPC AZs |
| `S3_OUT_ROOT` | see script | override results bucket/prefix root |

## Local wiring test (no GPU / no AWS)

```bash
uv run pytest tests/adaptive/test_atlas_cat_checkpoint_pipeline.py -v
```

## Beaker alternative (AI2 clusters)

Requires `BEAKER_TOKEN` and workspace access. Even `--dry-run` currently talks to
the Beaker API for the username:

```bash
uv run olmo-eval beaker launch \
  -m "${CHECKPOINT_S3}" \
  -E atlas_arc -E atlas_hellaswag -E atlas_winogrande -E atlas_gsm8k \
  -c h100 -w ai2/oe-data -B ai2/oe-base \
  --aws-credentials --dry-run
```

Prefer the `g6.xlarge` path above when training already lives on AWS.
