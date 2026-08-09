# RUNBOOK — Judge Grading Pipeline on AWS (for AWS first-timers)

This is a step-by-step guide to running the **multi-benchmark judge grading pipeline**
on an AWS GPU machine, written for someone who has **never launched AWS compute
before**. Follow it top-to-bottom. Copy-paste the commands. When in doubt, read the
⚠️ callouts — they are where people lose money or time.

---

## 1. What this does / prerequisites

**In plain English:** tutor models were asked a bunch of questions, and their answers
are saved as files. This pipeline hands those answers to a *frozen "judge" model*
(Qwen), which reads each answer against a grading rubric and marks it pass/fail. The
judge is a large model, so it needs a **GPU** to run at a reasonable speed — that's why
we run it on AWS.

You feed it tutor responses (`runs/responses/<Benchmark>/<model>.jsonl`) and it produces
graded verdicts and score matrices (`runs/judge/<Benchmark>/`), copied up to S3 so the
results survive even if you shut the machine down.

**What you need before starting:**

1. **AWS access** — either your own AWS account (you can launch a GPU instance), or SSH
   access to the team's existing GPU box (skip straight to step 3d if someone hands you
   a box that's already set up).
2. **An S3 bucket** you can read from and write to (this is where responses come from and
   where grades go). If you don't have one, create one (step 2 explains what it is).
3. **A Hugging Face token** (`HF_TOKEN`) — the judge model is a gated download, so you
   must be logged in. Get one at <https://huggingface.co/settings/tokens> (a read token
   is enough) and accept the model's terms on its HF page.

**Scope — which benchmarks get graded:** `TutorBench`, `TutorEval`, `InFoBench`,
`Bridge`, `BiGGen`, `WildBench`. (`IFEval` and `EduBench` are **excluded** — they're
graded by deterministic verifiers, not the LLM judge, so the driver skips them.)

**The judge model:** `Qwen/Qwen3.5-9B` in `bf16`, served with **vLLM** and **prefix
caching on**. You do not download or configure it by hand — the runner does that; you
just need `HF_TOKEN` set.

---

## 2. AWS concepts in 60 seconds

You'll see these words in the AWS Console. Here's the one-liner for each:

- **EC2 instance** — a rented computer in Amazon's data center. A "GPU instance" is one
  that has an NVIDIA GPU attached. You pay **per hour** while it's running.
- **AMI (Amazon Machine Image)** — the pre-installed operating system + drivers your
  instance boots from. Pick a **Deep Learning AMI** so NVIDIA drivers and CUDA are
  already there (no driver install headache).
- **Key pair** — an SSH key (a `.pem` file you download once) that lets you log into the
  instance. Lose it and you can't get in.
- **Security group** — a firewall for your instance. You'll open **port 22 (SSH)** so you
  can connect, ideally only from your own IP address.
- **IAM role** — permissions you attach to the instance so it can talk to other AWS
  services **without** you pasting secret keys onto the box. You'll give it **read/write
  to your S3 bucket** so `aws s3 sync` just works.
- **S3 bucket** — Amazon's file storage. Think of it as a shared cloud folder addressed
  like `s3://your-bucket/some/path`. Responses come from here; grades go back here.
- **Spot vs On-Demand** — two ways to pay. **On-Demand** = full price, never interrupted
  (use this for your first run). **Spot** = much cheaper, but AWS can reclaim the machine
  with ~2 minutes' notice (only for later, when you can tolerate interruptions).

---

## 3. Path A — single GPU box (recommended for first-timers)

This is the whole pipeline on **one** machine. Do this first.

### 3a. Launch an EC2 GPU instance (AWS Console)

1. Sign in to the AWS Console → search **"EC2"** → **Launch instance**.
2. **Name** it something like `judge-grading`.
3. **AMI:** choose an **AWS Deep Learning AMI** (Ubuntu, with NVIDIA drivers/CUDA
   preinstalled). Search the AMI catalog for "Deep Learning".
4. **Instance type:** pick a **GPU instance**. A single modern data-center GPU (e.g. an
   **L40S (`g6e.xlarge`, 48 GB) / A100-class** instance family) is plenty for one judge process. The judge
   is a 9B model in bf16 — a GPU with **≥24 GB** of memory is comfortable; more headroom
   is better.
   > ℹ️ Don't take my instance-family names as gospel or assume a price — availability and
   > cost change constantly. **Check current AWS GPU pricing** for your region and pick the
   > cheapest instance that has enough GPU memory.
5. **Key pair:** create a new key pair, name it (e.g. `judge-key`), and **download the
   `.pem` file**. Keep it safe. Then run locally: `chmod 400 judge-key.pem`.
6. **Network / security group:** create a security group that **allows inbound SSH
   (TCP 22)**. Set the source to **"My IP"** so only you can connect (safer than
   `0.0.0.0/0`).
7. **IAM role (instance profile):** attach a role that grants **read/write to your S3
   bucket** (an `AmazonS3` read+write policy scoped to your bucket is ideal). This is what
   lets `aws s3 sync` work on the box with no secret keys. If you can't create one now,
   you can still run and configure credentials another way, but the IAM role is the clean
   path.
8. **Storage:** the model + Python env + responses need room — give the root volume a
   healthy size (e.g. **100+ GB**) so downloads don't fill the disk.
9. Click **Launch instance**, then open the instance and copy its **Public IPv4 address**.

> ⚠️ **COST WARNING.** A GPU instance bills **every hour it is running** (and keeps
> billing even while idle). When you are done, **TERMINATE the instance** (EC2 →
> Instances → select → *Instance state* → *Terminate*). "Stop" still charges for storage;
> "Terminate" ends charges (and deletes the box). Set yourself a reminder.

### 3b. SSH into the instance

From your laptop, using the `.pem` you downloaded and the public IP from 3a:

```bash
ssh -i judge-key.pem ubuntu@<public-ip>
```

(The Deep Learning AMI's default user is usually `ubuntu`. If login is refused, check the
AMI's listed default username.)

### 3c. Clone the repo + one-time setup

On the instance:

```bash
git clone <your-repo-url> eduLLM-Evals
cd eduLLM-Evals

# Install uv (the project runs everything via `uv run`).
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"   # or restart your shell so `uv` is on PATH

# Sync dependencies INCLUDING the GPU extra (this pulls vLLM/torch/transformers/boto3).
uv sync --extra gen
```

> ⚠️ **Setup-script mismatch — read this.** There is a script,
> `scripts/aws/setup_respgen.sh`, that builds a **pip** virtualenv at `.venv` with the
> `[gen]` extra. **Do not rely on it for grading.** The grading wrapper runs everything
> through **`uv run`**, which uses uv's own project environment — *not* that pip `.venv` —
> so a `.venv` built by `setup_respgen.sh` is simply bypassed. On a fresh box, use
> `uv sync` as shown above.
>
> ⚠️ **Use `--extra gen`, not `--extra dev`.** The GPU dependencies (vLLM, torch,
> transformers, boto3) live in the **`gen`** extra. The `dev` extra is only `pytest` and
> will **not** install vLLM, so the judge step would crash with an import error. (You'll
> also need the AWS CLI and NVIDIA tools present — the Deep Learning AMI ships
> `nvidia-smi`, and `aws` is usually preinstalled; if `aws` is missing, install the AWS
> CLI v2.)

Now set your secrets/config **in the same shell** you'll launch from:

```bash
export HF_TOKEN=<your-hugging-face-token>          # required; gated model download
export S3_GRADING_PREFIX=s3://<your-bucket>/edu-tutor-grading   # your real bucket
```

> ⚠️ If you leave `S3_GRADING_PREFIX` unset it defaults to the placeholder
> `s3://YOUR-BUCKET/edu-tutor-grading`, and the wrapper prints a **WARNING** and cannot
> actually push/pull. Always set it to a real bucket/prefix.

### 3d. Make sure the tutor responses are present

The pipeline grades files at `runs/responses/<Benchmark>/<model>.jsonl` (these come from
a teammate's response-generation step). Check whether they're already on the box:

```bash
ls runs/responses/       # should list benchmark folders: TutorBench, BiGGen, ...
```

If they're missing, sync them down from S3 (adjust the source path to where your
teammate published them):

```bash
aws s3 sync s3://<your-bucket>/edu-tutor-responses runs/responses
```

> If a benchmark has no response files, the driver simply **skips** it — that's expected,
> not an error.

### 3e. Smoke test on ONE benchmark first

Before committing to a full run, prove the whole path works on the smallest slice.
`GPU=0` pins the judge to CUDA device 0 (a single-GPU box only has index 0):

```bash
ONLY=BiGGen GPU=0 bash scripts/aws/run_grading_gpu4.sh
```

This runs the full emit → push → judge → pull → ingest cycle for just BiGGen. If it
finishes and writes `runs/judge/BiGGen/verdicts.jsonl`, you're good.

> The wrapper is named `run_grading_gpu4.sh` because it defaults to GPU **index 4** on the
> team's 8-GPU box. On your own single-GPU instance you **must** pass `GPU=0`, or it will
> complain that GPU 4 doesn't exist.

### 3f. Full run (all in-scope benchmarks)

```bash
GPU=0 TP=1 bash scripts/aws/run_grading_gpu4.sh
```

- `GPU=0` — pin to your only GPU.
- `TP=1` — tensor-parallel size 1 (one GPU). Leave at 1 on a single-GPU box.

The wrapper loops every in-scope benchmark: it stages blinded cases, pushes them to S3,
runs the judge, pulls the verdicts back into a flat inbox, and finally ingests everything
into per-benchmark `verdicts.jsonl` + `response_matrix.csv`.

### 3g. Multi-GPU on one box (optional, faster)

If your instance has several GPUs, you can grade **shards in parallel**, one judge process
per GPU, then do a single merge/ingest at the end. The wrapper splits work by a stable
hash of each `(model, scenario)` block, so every shard is self-contained.

**Step 1 — launch one shard worker per GPU** (example for 4 GPUs; each pins a different
`GPU` index, all share the same `NUM_SHARDS`, and each takes a distinct `SHARD_INDEX`).
`SKIP_INGEST=1` tells each worker to emit + judge + upload its shard but **not** ingest:

```bash
for i in 0 1 2 3; do
  NUM_SHARDS=4 SHARD_INDEX=$i GPU=$i SKIP_INGEST=1 \
    HF_TOKEN="$HF_TOKEN" S3_GRADING_PREFIX="$S3_GRADING_PREFIX" \
    bash scripts/aws/run_grading_gpu4.sh > shard_$i.log 2>&1 &
done
wait   # block until all four shard workers finish
```

**Step 2 — final ingest pass** that pulls **all** shards' verdicts down and merges them
into the full matrix. Re-run the wrapper **without** `SKIP_INGEST`, keeping the same
`NUM_SHARDS` so its pull loop fetches every shard (the judge step is a fast no-op here
because it resumes already-graded cases):

```bash
NUM_SHARDS=4 SHARD_INDEX=0 GPU=0 \
  HF_TOKEN="$HF_TOKEN" S3_GRADING_PREFIX="$S3_GRADING_PREFIX" \
  bash scripts/aws/run_grading_gpu4.sh
```

> Why this shape: with `NUM_SHARDS>1` the wrapper names files `cases.shard<i>.jsonl` /
> `canonical_r1.shard<i>.jsonl` and, in step (d), downloads **every** shard `0..N-1` into
> the flat inbox `runs/judge/_verdicts_inbox/<Benchmark>/`; step (e) then ingests the whole
> inbox, merging all shards. Keeping `NUM_SHARDS=4` on the final pass is what makes it pull
> all four.

### 3h. Fetch results, then TERMINATE

Results are already mirrored to S3 by the wrapper, and they're on the box under
`runs/judge/`. To pull them to your laptop:

```bash
# From your laptop (or wherever you want the results):
aws s3 sync s3://<your-bucket>/edu-tutor-grading ./grading-results
```

Per benchmark you get `runs/judge/<Benchmark>/verdicts.jsonl`,
`runs/judge/<Benchmark>/response_matrix.csv`, `manifest.json`, plus a roll-up at
`runs/judge/_index.json`.

> ⚠️ **Now TERMINATE the instance** (EC2 → Instances → Terminate). Confirm it shows
> `terminated`. This is the single most important step for not getting a surprise bill.

---

## 4. Path B — AWS Batch fleet (scale later)

For grading at larger scale you'd normally fan out across many machines with **AWS
Batch**. The pipeline is already *sharding-ready*: the wrapper reads
`SHARD_INDEX` from **`AWS_BATCH_JOB_ARRAY_INDEX`** by default, so an N-task Batch **array
job** with `NUM_SHARDS=N` would have each task grade one shard automatically.

**However, the Batch scaffolding does not exist in this repo yet.** Running on Batch also
requires, none of which is currently checked in:

- a **container image** (a `Dockerfile` built and pushed to **ECR**),
- a Batch **compute environment**, **job queue**, and an **array job definition**,
- **IAM roles** for the job (task role with S3 access + execution role).

**Recommendation:** stick with **Path A** for now. When you genuinely need Batch scale,
**ask to have this scaffolding built** — it's a follow-up project, not something you can
copy-paste from here today.

---

## 5. Troubleshooting

| Symptom | Cause & fix |
| --- | --- |
| `set HF_TOKEN in the environment` / gated-repo 401/403 on model download | `HF_TOKEN` isn't exported (or you didn't accept the model terms). `export HF_TOKEN=...` in the same shell, and accept the Qwen model's terms on its HF page. |
| `WARNING: S3_GRADING_PREFIX is the placeholder (...YOUR-BUCKET...)` | You didn't set a real bucket. `export S3_GRADING_PREFIX=s3://<your-bucket>/edu-tutor-grading`. |
| `GPU <n> already has ... running compute process(es); refusing to start` | Safety guard: the pinned GPU is busy. Pick a free index with `GPU=<free index>` — **do not** kill the other job. |
| `This node has N GPU(s) ... GPU <n> does not exist` | You asked for a GPU index that isn't there. CUDA indices are **0-based** (8th GPU = index 7). On a single-GPU box use `GPU=0`. |
| `uv not found` | Install uv (see 3c) and make sure it's on `PATH` (`source "$HOME/.local/bin/env"` or restart the shell). |
| `aws CLI not found` | Install AWS CLI v2. Needed for the S3 push/pull. |
| `nvidia-smi not found — are you on the GPU node?` | You're not on a GPU box (or drivers aren't installed). Use a Deep Learning AMI on a GPU instance. |
| `SKIP: no responses dir ...` / a benchmark is skipped | No response files for it under `runs/responses/<Benchmark>/`. Sync them down (3d) or accept the skip. |
| `ImportError: vLLM is required ...` | You synced without the GPU deps. Run `uv sync --extra gen` (not `--extra dev`). |
| Out-of-memory (OOM) on a smaller GPU | Options in rough order: (1) use a GPU with more memory; (2) the underlying runner supports `--gpu-memory-utilization` (default 0.90), `--max-model-len` (default 8192), and `--quantization` — but ⚠️ **the wrapper only threads `TP` through today**, so to use those you'd invoke `aws_judge_handoff/scripts/run_judge_validation.py run ...` directly (or ask for the wrapper to expose them); (3) raising `TP` splits the model across multiple GPUs (needs a multi-GPU box). |
| Job ran but you still see it in EC2 | ⚠️ You forgot to **terminate**. Do it now — you're being billed. |

---

## 6. Quick reference — wrapper environment variables

All of these are read by `scripts/aws/run_grading_gpu4.sh`. Set them inline
(`VAR=value bash scripts/aws/run_grading_gpu4.sh`) or `export` them first.

| Variable | Default | What it controls |
| --- | --- | --- |
| `HF_TOKEN` | *(required)* | Hugging Face token for the gated judge model download. Wrapper aborts if unset. |
| `S3_GRADING_PREFIX` | `s3://YOUR-BUCKET/edu-tutor-grading` (placeholder → warning) | Base S3 prefix cases are pushed to and verdicts pulled from. **Set to a real bucket.** |
| `GPU` | `4` | CUDA device index to pin the judge to. Use `0` on a single-GPU box. Must exist and be free. |
| `TP` | `1` | Tensor-parallel size (how many GPUs the judge model is split across). |
| `JUDGE` | `qwen` | Frozen judge name (Qwen3.5-9B). |
| `ONLY` | *(empty = all)* | Comma-separated subset of in-scope benchmarks, e.g. `ONLY=TutorBench,Bridge`. |
| `RESPONSES_ROOT` | `runs/responses` | Where tutor response shards (`<Benchmark>/<model>.jsonl`) are read from. |
| `JUDGE_ROOT` | `runs/judge` | Where per-benchmark outputs (verdicts, matrices, manifests) are written. |
| `INBOX` | `<JUDGE_ROOT>/_verdicts_inbox` | Flat local folder verdicts are pulled into before ingest. |
| `NUM_SHARDS` | `1` | Number of shards to split grading into (for parallel/array runs). |
| `SHARD_INDEX` | `$AWS_BATCH_JOB_ARRAY_INDEX` or `0` | Which shard *this* run handles (0-based). Auto-set by AWS Batch array jobs. |
| `REPLICATE_ID` | `r1` | Tag for this grading replicate; appears in verdict filenames (`canonical_<id>`). |
| `PROMPT_VARIANT` | `canonical` | Judge prompt variant. |
| `SKIP_INGEST` | `0` | If `1`, emit + judge + upload only (used by per-shard workers); skip the ingest step. |

**Related scripts:**

- `scripts/aws/run_grading_gpu4.sh` — the grading wrapper you run (everything above).
- `scripts/run_all_judge_grading.py` — the driver it calls (emit cases / ingest verdicts;
  supports `--s3-prefix`, `--num-shards/--shard-index`, `--only`, `--keep-dead-models`).
- `aws_judge_handoff/scripts/run_judge_validation.py` — the frozen judge runner (Qwen +
  vLLM; flags `--tensor-parallel-size`, `--dtype`, `--quantization`,
  `--gpu-memory-utilization`, `--max-model-len`).
- `scripts/aws/chain_respgen_then_grade.sh` — **do-both-stages** option: waits for the
  response-generation run to finish, then automatically runs this grading pipeline on the
  same box (fully detached; logs + results synced to S3). Use this if you also need to
  generate the tutor responses first, rather than just grading existing ones.
- `scripts/aws/setup_respgen.sh` — the legacy pip-`.venv` setup script (see the mismatch
  callout in 3c; prefer `uv sync --extra gen`).
