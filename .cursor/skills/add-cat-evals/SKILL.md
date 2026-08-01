---
name: add-cat-evals
description: >-
  Add ATLAS CAT (Computerized Adaptive Testing) checkpoint diagnostics to an
  OLMo-core training run. Converts a native OLMo-core checkpoint to HF format,
  runs the adaptive ATLAS evals via vLLM, and writes theta/SE results to S3.
  Use when a training team wants to auto-run CAT diagnostics on checkpoints, add
  atlas / adaptive-testing evals to a training script, or asks how to score
  checkpoints with ATLAS CAT.
disable-model-invocation: true
---

# Add CAT evals to a training run

Runs the ATLAS adaptive tests on a checkpoint and writes results to a fixed S3
prefix. Designed to be called from an OLMo-core training script right after a
checkpoint is saved.

## What it does

1. Converts a native OLMo-core checkpoint (`model_and_optim/` + `config.json`) to
   HF format (`config.json` + `*.safetensors`). CPU-only for standard dense archs.
2. Boots a local vLLM server on the HF checkpoint.
3. Runs the adaptive CAT for each benchmark (each picks 8–40 items until SE ≤ stop),
   sharing one vLLM boot.
4. Writes per-eval `<eval>.json` (theta, se, pirt_accuracy, n_items, selected ids),
   `pipeline_provenance.json`, and `worker.log` to S3, plus a `_READY` marker.

Scope today: **4 benchmarks** — `atlas_arc` (ARC-Challenge), `atlas_hellaswag`,
`atlas_winogrande` (MCQ log-likelihood), and `atlas_gsm8k` (generative exact-match),
each backed by a calibrated ATLAS 3PL bank vendored in this repo. TruthfulQA is not
yet wired (no olmo-eval base task). Override the set with `--evals`.

## Requirements

**Conversion node** (the training node, CPU):
- `ai2-olmo-core==2.4.0` (already present in an OLMo-core env) + `transformers`
- RAM ≈ model size at bf16 (~2 GB per 1B params), ~2× if validation runs
- Disk ≈ 2× model size (input shards + HF output)
- Standard **dense** OLMo-2/OLMo-3 arch. MoE / fused / flash-attention checkpoints
  force conversion onto GPU and are not covered by the cheap path.

**Diagnostic (this repo's env, 1 GPU)**:
- `olmo-eval` installed with vLLM extras (`uv sync --extra vllm --extra hf`)
- 1× GPU with ~24 GB VRAM (fits ≤ ~8B at bf16). Larger models need a bigger GPU
  or tensor-parallel (`--tp N`).
- AWS creds with `s3:PutObject` on the target results prefix.

Nothing is uploaded to the HuggingFace Hub. "HF" means the on-disk file format;
weights and results stay local / in your S3.

## Quick start (in-process, from a training script)

After the checkpoint lands at `${CKPT_DIR}` (a dir containing `config.json` and
`model_and_optim/`):

```bash
bash .cursor/skills/add-cat-evals/scripts/run_cat_diagnostic.sh \
  --checkpoint "${CKPT_DIR}" \
  --run-id "${EXP}-step${STEP}" \
  --s3-out "s3://YOUR_BUCKET/atlas_cat"
```

Run it **backgrounded** so it doesn't block training:

```bash
nohup bash .cursor/skills/add-cat-evals/scripts/run_cat_diagnostic.sh \
  --checkpoint "${CKPT_DIR}" --run-id "${EXP}-step${STEP}" \
  --s3-out "s3://YOUR_BUCKET/atlas_cat" >/dev/null 2>&1 &
```

`--run-id` must match `[A-Za-z0-9._-]+`. Results land at `<s3-out>/<run-id>/`.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | required | OLMo-core checkpoint dir (or an HF dir / HF id; skips conversion) |
| `--run-id` | required | results subdirectory; `[A-Za-z0-9._-]+` |
| `--s3-out` | required | `s3://bucket/prefix` root for results |
| `--tokenizer` | (from config) | HF tokenizer id if not resolvable from checkpoint config |
| `--se-stop` | `0.3` | CAT early-stop standard error |
| `--min-items` | `8` | CAT floor |
| `--max-items` | `40` | CAT cap |
| `--tp` | `1` | vLLM tensor-parallel size (raise for large models) |
| `--evals` | `atlas_arc atlas_hellaswag atlas_winogrande atlas_gsm8k` | space-separated CAT evals to run (share one vLLM boot) |
| `--skip-convert` | off | checkpoint is already HF format / an HF id |
| `--keep-hf` | off | keep the converted `-hf` dir (default: temp, removed after) |
| `--dry-run` | off | print the plan without running |

## Output location

```
<s3-out>/<run-id>/
  atlas_arc.json            # theta, se, pirt_accuracy, n_items, selected_question_ids
  atlas_hellaswag.json      # (one JSON per --evals entry)
  atlas_winogrande.json
  atlas_gsm8k.json
  pipeline_provenance.json  # checkpoint, run_id, git_sha, evals, args
  worker.log
  _READY                    # written last; poll for this to know it's done
```

## Verify locally first (no GPU, no AWS)

```bash
uv run pytest tests/adaptive/test_atlas_cat_checkpoint_pipeline.py -v
```

## How it maps to olmo-eval

The diagnostic step is exactly:

```bash
uv run olmo-eval run-external \
  -m "${HF_CKPT}" \
  -e atlas_arc -e atlas_hellaswag -e atlas_winogrande -e atlas_gsm8k \
  --provider vllm_server \
  -a "se_stop=0.3" -a "min_items=8" -a "max_items=40" \
  -O "${OUT}"
```

These evals only run through `run-external` (vLLM), which needs HF-format weights —
hence the conversion step. Native OLMo-core weights are not loadable by vLLM directly.
Each `atlas_*` eval joins its calibrated 3PL bank
(`AdaptiveTesting/Inputs/ATLAS/<benchmark>/`) to the benchmark's questions via
`atlas_idx_to_question_id.csv`; only items present in both the bank and the task are
eligible for selection.

## Additional resources

- Conversion details, memory/timing, and architecture caveats: see [CONVERSION.md](CONVERSION.md)
- Separate-GPU-worker path (launch an EC2 box per checkpoint instead of in-process):
  `AdaptiveTesting/scripts/atlas_cat_diagnose/` (`launch_g6.sh`, `run_worker.sh`)
