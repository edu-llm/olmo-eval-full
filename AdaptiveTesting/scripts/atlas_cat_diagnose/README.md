# ATLAS CAT checkpoint diagnostic (training hook)

Drop-in commands for a training job: after a checkpoint lands on S3, launch one
`g6.xlarge` (1× L4) worker that runs online `atlas_arc` and writes results to a
**fixed S3 prefix**.

## Known output location

```
s3://edullm-adaptive-inference-056956104102/atlas_cat/<run_id>/
  atlas_arc_results.json   # theta, se, pirt_accuracy, n_items, selected ids
  pipeline_provenance.json # checkpoint URI, instance id, git sha, args
  worker.log               # full worker stdout (best-effort)
```

Region: `us-east-1`. Account: `056956104102` (sbsandbox / edullm adaptive).

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
| `--branch` | `AdaptiveEvals` | git branch the worker clones |
| `DRY_RUN=1` | off | print `run-instances` only |
| `INSTANCE_TYPE` | `g6.xlarge` | override (still 1× L4 family recommended) |
| `S3_OUT` | see script | override results bucket/prefix root |

## Local wiring test (no GPU / no AWS)

```bash
uv run pytest tests/adaptive/test_atlas_cat_checkpoint_pipeline.py -v
```

## Beaker alternative (AI2 clusters)

Requires `BEAKER_TOKEN` and workspace access. Even `--dry-run` currently talks to
the Beaker API for the username:

```bash
uv run olmo-eval beaker launch \
  -m "${CHECKPOINT_S3}" -E atlas_arc \
  -c h100 -w ai2/oe-data -B ai2/oe-base \
  --aws-credentials --dry-run
```

Prefer the `g6.xlarge` path above when training already lives on AWS.
