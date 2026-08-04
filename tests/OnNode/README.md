# OnNode checkpoint inference TEST (Flow 1)

Standalone, on-node inference script a training team runs once per checkpoint. It
loads a checkpoint from S3, runs a few random inferences on the local GPU(s), and
uploads the outputs back to S3. It does **not** depend on the `olmo_eval` package.

Design: [`Plan/flow1_training_checkpoint/README.md`](../../Plan/flow1_training_checkpoint/README.md).

## Files

| File | Purpose |
|---|---|
| `checkpoint_infer.py` | The inference script |
| `checkpoint_infer.env.example` | Config template (copy to `checkpoint_infer.env`) |
| `prompts.jsonl` | Default prompt pool sampled for the TEST |
| `skill/SKILL.md` | Skill for a training team's agent to install + wire this up |

## Quick start

```bash
cp checkpoint_infer.env.example checkpoint_infer.env
# edit checkpoint_infer.env: set RESULTS_BUCKET, RUN_NAME, CHECKPOINT_KIND
source checkpoint_infer.env

pip install boto3 transformers torch          # hf backend
python checkpoint_infer.py s3://bucket/checkpoints/owner/run/step1000/ --dry-run
python checkpoint_infer.py s3://bucket/checkpoints/owner/run/step1000/
```

Outputs land at:

```
s3://$RESULTS_BUCKET/$RESULTS_PREFIX/$RUN_NAME/step<N>/results.jsonl
s3://$RESULTS_BUCKET/$RESULTS_PREFIX/$RUN_NAME/step<N>/manifest.json
```

## Training-loop hook

After each checkpoint save, call the script with the new location (backgrounded so it
does not block training):

```bash
CKPT="s3://$RESULTS_BUCKET/checkpoints/$OWNER/$RUN_NAME/step${STEP}/"
python /path/to/checkpoint_infer.py "$CKPT" &
```

## Status

Starting implementation. The `hf` backend is functional; the `olmo_core` loader is an
integration point (`_load_olmo_core`) pending the run's config/checkpoint layout.
