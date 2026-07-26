# AWS judge handoff

This folder is the blinded GPU-job bundle. It contains the 261 tutor-response/criterion cases and the runner for all five judges. It does **not** contain the human P/F labels, grader packets, human notes, or tutor-model identities.

## Contents

- `inputs/judge_cases.blinded.jsonl`: the tutor responses and criteria to grade;
- `scripts/run_judge_validation.py`: blinded judge runner and S3 checkpointing;
- `run_judge_suite.sh`: six-wave launcher for one judge/GPU job;
- `STUDY_DESIGN.json`: frozen wave definitions and acceptance thresholds;
- `requirements-aws.txt`: GPU and S3 Python dependencies;
- `SHA256SUMS`: transfer-integrity hashes for the bundle files.

## AWS setup

Use a Linux NVIDIA GPU instance with Python 3.10 or newer. An Ampere-or-newer
40–48 GB GPU is the simplest BF16 target for all five judges, including Gemma
3 12B. Use an NVIDIA driver/CUDA image compatible with the pinned vLLM 0.26.0
release. Smaller cards may require tensor parallelism or quantization; if used,
keep that setting identical across all six waves for a judge. From this folder:

```bash
python3 -m venv .venv-judge
source .venv-judge/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-aws.txt
python scripts/run_judge_validation.py models
```

This blinded copy is intended only for the `models` and `run` commands. Preparation and comparison remain with the evaluation team.

Attach an IAM role that can list the chosen bucket prefix and get/put objects. Do not place AWS keys in this folder. Gemma is gated, so accept its Hugging Face license and provide `HF_TOKEN` to that job.

## Study design

Each judge runs six waves over all 261 cases:

1. Three identical canonical runs: `canonical_r1`, `canonical_r2`, and `canonical_r3`.
2. Three controlled variants, each run once: `whitespace_r1`, `header_synonyms_r1`, and `instruction_politeness_r1`.

That produces 1,566 judgments per judge and 7,830 across all five judges. Temperature, seed, checkpoint, threshold, and task content stay frozen. The variants alter only fixed prompt scaffolding; they never alter the tutor response, criterion, reference material, or other case data. Each output records a frozen-configuration hash so the evaluation team can reject cross-wave setting drift.

## Submit the five jobs

Replace the example S3 URI with the real **blinded-output root**. Run one command per GPU job:

```bash
bash run_judge_suite.sh selene     s3://YOUR-BUCKET/edu-judge-validation/v2/blinded
bash run_judge_suite.sh flow       s3://YOUR-BUCKET/edu-judge-validation/v2/blinded
bash run_judge_suite.sh prometheus s3://YOUR-BUCKET/edu-judge-validation/v2/blinded
bash run_judge_suite.sh qwen       s3://YOUR-BUCKET/edu-judge-validation/v2/blinded
bash run_judge_suite.sh gemma      s3://YOUR-BUCKET/edu-judge-validation/v2/blinded
```

Run these as five separate GPU jobs, ideally in parallel. Each launcher processes its six waves sequentially. Model weights are cached after the first download, although the current runner reloads the cached model for each wave.

The launcher gives every judge and wave a distinct prefix: `S3_ROOT/JUDGE/WAVE`. Each completed batch is flushed locally, uploaded with a SHA-256 checksum, and followed by a manifest checkpoint. `--require-s3-upload` makes missing or failed S3 publishing fail the job.

Before a full launch, smoke-test each judge; for example, test Selene with three
cases per wave:

```bash
bash run_judge_suite.sh selene \
  s3://YOUR-BUCKET/edu-judge-validation/v2/blinded \
  --limit 3
```

Then rerun the same command without `--limit`; resume mode retains those three cases and evaluates the remaining 258.

To resume on a replacement instance, restore this folder and run the same command. The runner downloads and verifies the saved checkpoint. If it reports that the remote run may still be active, first confirm the old job has stopped, then append `--allow-s3-takeover` to the command.

Return the S3 study root to the evaluation team. They will download the 30 wave JSONLs and compare them locally against the separately retained `human_labels.csv`.

After transferring the folder, its contents can be checked with `sha256sum -c SHA256SUMS` on Linux.
