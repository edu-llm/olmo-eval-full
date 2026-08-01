# ATLAS CAT checkpoint diagnostic (training hook)

Drop-in commands for a training job: after a checkpoint lands on S3, launch one
`g6.xlarge` (1× L4) worker that runs online `atlas_arc` and writes results to a
**fixed S3 prefix**.

## Scope today: ARC-Challenge only

The full ATLAS suite is **5 benchmarks** — ARC-Challenge, HellaSwag, WinoGrande,
GSM8K, TruthfulQA — but this diagnostic currently wires **only ARC-Challenge**
(`atlas_arc`). Treat the ARC theta/SE as a single-benchmark ability signal for now,
not the full ATLAS profile.

To reach all 5 (future work, not yet done):

1. **Ship the banks.** Only the `arc` bank is committed; `hellaswag/ winogrande/
   gsm8k/ truthfulqa/` are `.gitignore`d, so a fresh worker can't fetch them. Commit
   them or stage them in S3 for the worker to pull.
2. **Wire MCQ benchmarks (low effort).** `atlas_hellaswag` + `atlas_winogrande` reuse
   the same log-likelihood MCQ scorer as ARC → gets to 3/5. (Watch the id bridge:
   WinoGrande uses positional indices, so `atlas_idx_to_question_id.csv` must match
   the task's `metadata["id"]` order.)
3. **Add a generative scorer (more effort).** GSM8K (generative) and TruthfulQA don't
   fit the argmax-over-choices scorer; they need a generate-and-match correctness
   adapter before CAT applies → gets to 5/5.
4. **Group into an `atlas` suite** so the worker runs all five behind one flag.

## Known output location

```
s3://edullm-adaptive-inference-056956104102/smoke/atlas_cat/<run_id>/
  atlas_arc_results.json   # theta, se, pirt_accuracy, n_items, selected ids
  pipeline_provenance.json # checkpoint URI, instance id, git sha, args
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
  -m "${CHECKPOINT_S3}" -E atlas_arc \
  -c h100 -w ai2/oe-data -B ai2/oe-base \
  --aws-credentials --dry-run
```

Prefer the `g6.xlarge` path above when training already lives on AWS.
