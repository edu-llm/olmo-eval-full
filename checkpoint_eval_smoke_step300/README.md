# Checkpoint eval smoke test — what happened

This bundle contains the results of a **smoke test** that ran the `eval-checkpoints`
skill against a single model checkpoint stored in S3, on a rented AWS GPU, and wrote
the benchmark outputs back to S3. Everything under `results/` was downloaded verbatim
from S3; this README explains how it was produced.

Date: 2026-08-04. AWS account: `056956104102` (sbsandbox), region `us-east-1`.

---

## TL;DR

- The full checkpoint-eval pipeline ran end to end and succeeded: **bootstrap a bare GPU
  box -> pull the checkpoint from S3 -> boot vLLM -> evaluate 5 benchmarks -> upload
  results -> write a `_READY` marker.**
- **The accuracy numbers are meaningless.** The checkpoint under test is a *tiny,
  randomly initialized* GPT-2 (a placeholder from an earlier mock training run), and the
  smoke group only scores 2 instances per benchmark. This run proves the *plumbing*, not
  model quality.
- The GPU instance was terminated afterward. Total cost was a small fraction of the
  $15 cap.

---

## The bigger picture (how we got here)

This was the last step in a short chain of work:

1. **Resumed context** on how `olmo-eval` evaluates checkpoints (two inputs matter: `-m`
   the checkpoint and `-t` the tasks; `olmo-eval run` runs inline, `olmo-eval beaker
   launch` runs on a separate node).
2. **Built and ran a mock "training run"** that wrote checkpoints to S3 at
   `s3://edullm-adaptive-inference-056956104102/checkpoints/mock-owner/mock-20260804-142414/step{100,200,300}/`.
   To keep that run free/offline, the "model" is a **tiny random GPT-2** (HF format:
   `config.json` + `model.safetensors` ~170 KB + a byte-level tokenizer), not a real model.
3. **This smoke test:** pointed the `eval-checkpoints` skill at `step300/` of that mock run.

So the checkpoint evaluated here is that placeholder GPT-2 — hence meaningless scores.

---

## What exactly was run

**Skill:** `eval-checkpoints`, from the `p3-tickets` branch of the `olmo-eval-full` repo
(`.cursor/skills/eval-checkpoints/`). It discovers a checkpoint, stages the weights,
shells out to `uv run olmo-eval` once per checkpoint (one vLLM boot, one `-t` per
benchmark), parses the returned `metrics.json`, and writes accuracy tables. It does not
contain any eval logic of its own.

**Checkpoint (input):**
`s3://edullm-adaptive-inference-056956104102/checkpoints/mock-owner/mock-20260804-142414/step300/`
(already HF format, so no OLMo-core -> HF conversion was needed.)

**Results (output):**
`s3://edullm-adaptive-inference-056956104102/mock-checkpoint-eval/mock-20260804-142414/`

**Benchmarks:** the skill's `smoke` group, which is the default set at a 2-instance cap:
`csqa`, `hellaswag`, `piqa`, `socialiqa`, `arc_easy` (all multiple-choice). `SIQA` maps to
`socialiqa`. The `--limit 2` cap is why scores are not a measurement.

**Command (real run, on the GPU box):**

```bash
bash .cursor/skills/eval-checkpoints/scripts/run_eval_sweep.sh \
  --checkpoint s3://edullm-adaptive-inference-056956104102/checkpoints/mock-owner/mock-20260804-142414/step300/ \
  --s3-out    s3://edullm-adaptive-inference-056956104102/mock-checkpoint-eval/mock-20260804-142414/ \
  --group smoke \
  --bootstrap
```

A free `--dry-run` (S3 read only, no GPU, no spend) was done first to validate the
checkpoint path, the resolved benchmark list, and the cost estimate before launching
the GPU.

**Compute:** a single `g4dn.xlarge` (NVIDIA T4, 16 GB) in `us-east-1`, launched from a
GPU Deep Learning AMI (drivers preinstalled). `--bootstrap` installed the Python
environment (uv / vLLM / olmo-eval) on the fresh box. vLLM ran with
`tensor_parallel_size=1` and `gpu_memory_utilization=0.5`.

**Cost / cleanup:** hard cap was $15; actual usage was a handful of short T4 sessions,
well under $1. The GPU instance was terminated after the run and no key pairs or
security groups were left behind.

---

## Results

`results/accuracy_wide.csv` (one row per checkpoint, one column per benchmark):

```
run_id,step,checkpoint,arc_easy,csqa,hellaswag,piqa,socialiqa,status
step300,300,s3://.../step300,0.0,0.0,0.0,1.0,0.5,ok
```

Read these as **plumbing evidence only**. With a random model and 2 instances per task,
`piqa=1.0` and `socialiqa=0.5` are coin-flips on a tiny sample, and `0.0` elsewhere is
equally uninformative. `status=ok` and the `step300/_READY` marker are the meaningful
part: every stage of the pipeline completed for every benchmark.

`results/step300/run_provenance.json` records how the run was configured (checkpoint,
output, benchmark selection = "group 'smoke'", `limit=2`, `tensor_parallel_size=1`,
`gpu_memory_utilization=0.5`).

---

## What is in this zip

```
checkpoint_eval_smoke_step300/
  README.md                     <- this file
  results/                      <- downloaded verbatim from S3
    accuracy_wide.csv           <- one row per checkpoint (the deliverable table)
    accuracy.csv                <- long form: one row per checkpoint/benchmark/metric
    accuracy.json               <- same data as JSON
    sweep.log                   <- top-level sweep log
    _bootstrap/
      _BOOT_DONE                <- marker that box bootstrap finished
      userdata.log              <- full bootstrap log from the GPU box
    step300/
      _READY                    <- terminal success marker for this checkpoint
      metrics.json              <- olmo-eval's standard output, one entry per task
      run_provenance.json       <- checkpoint, benchmarks, settings, status
      logs/                     <- vLLM server logs
      metrics/                  <- per-run inference metric lines
      predictions/              <- per-instance model predictions, per benchmark
      requests/                 <- per-instance prompts sent to the model, per benchmark
```

(The `predictions/`, `requests/`, `logs/`, and `metrics/` folders appear under two
`_tmp_tmp.*` subdirectories because the checkpoint was staged to a fresh temp dir on
each of the two evaluation passes on the box. The scored result is the same either way.)

The large `_bootstrap/repo.tar.gz` (the shipped repo snapshot, ~1.9 MB) was intentionally
excluded to keep this bundle small.

---

## Important caveats

- **Not a real evaluation.** The checkpoint is a random placeholder model, and `--group
  smoke` scores only 2 instances per benchmark. Do not cite these numbers.
- To get a meaningful result, rerun the same command **without** `--group smoke` (full
  splits) against a **real** checkpoint.

## Suggested next steps

1. **Real model:** redo the mock training run with a genuine OLMo3 190M (OLMo-core native
   checkpoints), then run this same eval on a full benchmark split.
2. **No-AWS submitters:** wrap this skill in a job-submission layer (e.g. an AWS Batch
   job definition whose compute role has S3 access to the checkpoint/result buckets) so
   someone without direct AWS access can submit `{checkpoint URI, output prefix,
   benchmarks}` and never touch AWS. This matches the CheckpointFlows "Flow 2" plan.
