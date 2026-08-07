# vLLM on AWS — GPU environment setup

This documents how to bring up the **vLLM inference environment** for `olmo-eval`
on a fresh AWS GPU instance, so a new teammate can reproduce it end to end.

It is distilled from a run that **actually completed on AWS**: the
`eval-direct-gpu` skill (`.cursor/skills/eval-direct-gpu/`), whose
`scripts/bootstrap.sh` installs the environment and `scripts/run_eval_sweep.sh`
boots vLLM to score checkpoints. That run happened on
a single NVIDIA **T4** GPU box in `us-east-1`, booted vLLM, served inference over
its OpenAI-compatible HTTP API, and uploaded results to S3. The downloaded
artifacts live under `checkpoint_eval_smoke_step300/` at the repo root (vLLM
server logs, `metrics.json`, `_READY`, and the full bootstrap log). See
[Source / provenance](#source--provenance) for the exact commands and evidence.

The commands below are the real ones from those two scripts. A second,
SSM-driven flow (`run_rung1_smoketest.sh` in this folder) installs the same vLLM
environment a slightly different way and is covered under
[Alternative](#alternative-ssm-driven-rung1-smoke-test).

This file is about the **vLLM environment** only. For launching, staging, and
tearing down the GPU instance itself, see
[`RUNBOOK_rung1_smoketest.md`](RUNBOOK_rung1_smoketest.md).

> **This does not carry over to platform Batch jobs, and that was checked rather
> than assumed.** Everything below depends on a Deep Learning AMI. A platform job
> runs a prebuilt OLMo-core container whose torch is a minor version behind what
> vLLM 0.19.1 links against, on a 30 GiB root volume, under a one-hour bound. The
> platform path therefore uses the `olmo_core` provider instead. See "Why the
> platform path cannot simply use vLLM too" in
> [`.cursor/skills/eval-platform/BENCHMARKS.md`](.cursor/skills/eval-platform/BENCHMARKS.md)
> for the evidence and for the pin set to start from if an eval image is ever built.

> **P3 math-split uses a separate pinned environment.** Its first L4 fleet
> reused the DLAMI's Python 3.13/PyTorch cu130 environment, while vLLM 0.19.1's
> extension required CUDA 12, and every shard failed on `libcudart.so.12`.
> Follow `Vishnu-Evals/HANDOFF_README.md` and run OLMo-core's
> `bootstrap_vllm_env.sh`; do not adapt the generic commands below by installing
> vLLM into the AMI interpreter.

---

## Why an AWS GPU box

vLLM and its torch build are **Linux + CUDA only** — the `pyproject.toml` `vllm`
extra is gated on `platform_system != 'Darwin'`, so it does not resolve on macOS.
You need a Linux GPU node with NVIDIA drivers and CUDA already present. AWS **Deep
Learning AMIs (DLAMIs)** ship those drivers + CUDA pre-baked, so there is no
driver install step — you go straight to installing the Python environment.

---

## Prerequisites

- A running GPU instance booted from a **GPU Deep Learning AMI** (NVIDIA drivers +
  CUDA preinstalled). The proven run used a `g4dn.xlarge` (T4, 16 GB); the team's
  documented DLAMI for smoke tests is `ami-0b6f2229ad14c9323` (Ubuntu 24.04 DLAMI)
  on `g5.xlarge` (A10G, 24 GB). Never exceed `g6.xlarge`.
- A checkout of this repository on the box (the skill shells out to
  `uv run olmo-eval` from the repo it lives in).
- The `aws` CLI on the box, plus credentials that can read the checkpoint prefix
  and `s3:PutObject` to the output prefix (on AWS this is the instance's IAM
  role; no keys are pasted).
- Outbound network to PyPI / GitHub / the Hugging Face Hub.
- A large scratch disk for caches. The DLAMI root volume is small (~19 GB) —
  torch + vLLM + model downloads do not fit there — so caches go on the DLAMI
  NVMe scratch at `/opt/dlami/nvme`.

---

## Step-by-step vLLM environment setup

These commands mirror `scripts/bootstrap.sh` verbatim; running that script does
all of them for you (`bash .cursor/skills/eval-direct-gpu/scripts/bootstrap.sh`).

```bash
# 1. Ensure HOME is set. Bare / cloud-init / SSM shells often run as root with no
#    HOME, which would send uv's install and caches to the wrong place.
export HOME="${HOME:-/root}"

# 2. Install uv (the package/Python manager). It lands in ~/.local/bin, which is
#    not on PATH by default, so add it in the same shell.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

# 3. Point the Hugging Face cache at the big NVMe scratch, not the tiny root disk.
#    (bootstrap.sh picks the first writable of /opt/dlami/nvme, /mnt, /data.)
export HF_HOME=/opt/dlami/nvme/hf-cache
mkdir -p "${HF_HOME}"

# 4. Install the environment from this repo. This pulls vLLM + torch (the
#    inference backend), transformers (tokenizers / HF checkpoints), and
#    boto3/smart_open (S3 read + result upload). olmo_core is only needed to
#    convert native OLMo-core checkpoints to HF format.
cd /path/to/olmo-eval-full
uv sync --extra vllm --extra hf --extra s3 --extra olmo_core
```

Notes grounded in the scripts and the proven run:

- **Python version.** `pyproject.toml` requires `>=3.12`. You do **not** need a
  separate `uv python install`: `uv sync` provisions the interpreter itself (the
  proven run auto-fetched CPython 3.12.13).
- **OLMo-core is optional and can conflict.** `ai2-olmo-core==2.4.0` conflicts
  with the `openhands` extra. If resolution fails, re-run without it:
  `bash scripts/bootstrap.sh --no-olmo-core` (equivalently drop
  `--extra olmo_core`) and handle any OLMo-core → HF conversion separately.
- **`aws` is not a Python dependency.** `uv sync` cannot supply it; the DLAMI
  already has it. Results always go to S3, so the box needs the `aws` CLI even
  when checkpoints are local.

---

## Running inference

The skill runs everything through one `olmo-eval` invocation per checkpoint so a
single vLLM boot serves every benchmark. The proven entry point (also does the
bootstrap above via `--bootstrap`):

```bash
bash .cursor/skills/eval-direct-gpu/scripts/run_eval_sweep.sh \
  --checkpoint s3://BUCKET/checkpoints/EXP/step300/ \
  --s3-out    s3://BUCKET/evals/EXP/ \
  --group smoke \
  --bootstrap
```

- Always `--dry-run` first (S3 read only, no GPU spend): it validates the
  checkpoint path, resolves the benchmark list, and prints a cost estimate.
- `--group smoke` caps to 2 instances per benchmark — a plumbing check, not a
  measurement. Drop it (and use a real checkpoint) for actual numbers.
- `--tp N` sets vLLM tensor-parallel size (default 1, correct for a single-GPU
  box). `--gpu-memory-utilization 0.N` lowers the VRAM fraction to fit a small
  GPU (the proven T4 run used `0.5`).

Under the hood the sweep issues a vLLM-backed run like this (the `vllm_server`
provider is what boots vLLM; overrides bind to the preceding `--harness` / `-t`):

```bash
uv run olmo-eval run \
  -m /local/path/to/checkpoint \
  --harness default -o provider.kind=vllm_server \
  -o provider.kwargs.gpu_memory_utilization=0.5 \
  -t hellaswag -o limit=2 \
  -t socialiqa -o limit=2 \
  -t arc_easy  -o limit=2 \
  -t piqa      -o limit=2 \
  -t csqa      -o limit=2 \
  -O /local/output/dir
```

vLLM cannot load an `s3://` path directly, so the sweep first `aws s3 sync`s the
weights to a local dir and passes that to `-m`.

---

## Verifying vLLM is working

1. **CLI present after sync:**

```bash
uv run olmo-eval --help | head -n 20
```

2. **vLLM actually booted and served inference.** The per-run vLLM server log
   (under `logs/vllm_server_*/`) shows the OpenAI-compatible API server coming up
   and answering completion requests:

```
(APIServer pid=...) INFO:     Application startup complete.
(APIServer pid=...) INFO:     127.0.0.1:... - "GET /v1/models HTTP/1.1" 200 OK
(APIServer pid=...) INFO:     127.0.0.1:... - "POST /v1/completions HTTP/1.1" 200 OK
```

3. **Success is confirmed by the written artifacts**, not by parsing accuracy:

- `metrics.json` — one entry per task, with `provider_init_seconds` and per-task
  durations.
- `run_provenance.json` — records `status: "ok"`, the benchmarks, and the vLLM
  settings (`tensor_parallel_size`, `gpu_memory_utilization`).
- `_READY` marker in the S3 output prefix (a `_FAILED` marker is written instead
  on failure), and `accuracy_wide.csv` / `accuracy.csv` at the sweep root.

---

## Troubleshooting / gotchas

These are the failure modes the scripts explicitly guard against:

- **`uv: command not found` right after install.** uv is at `~/.local/bin`, which
  is not on PATH by default. `export PATH="${HOME}/.local/bin:${PATH}"` in the
  same shell.
- **uv/caches land in the wrong place on a root box.** A bare or SSM shell may run
  as root with no `HOME`, so `export HOME="${HOME:-/root}"` before installing uv.
- **`No space left on device` during sync / model download.** The DLAMI root disk
  is tiny (~19 GB). Point caches at the NVMe scratch: `HF_HOME=/opt/dlami/nvme/...`
  (and, on the SSM flow, `UV_CACHE_DIR=/opt/dlami/nvme/...`). torch + vLLM + model
  weights far exceed the root volume.
- **`uv sync` fails on `ai2-olmo-core`.** It conflicts with `openhands`. Re-run
  with `--no-olmo-core` (drop `--extra olmo_core`) and convert checkpoints
  separately.
- **vLLM crashes on model load with architecture/config errors.** That model's
  architecture is unsupported by vLLM (the team has hit this with `gemma-3-*`,
  `OpenELM-*`, `mamba-*`). Use a vLLM-supported model; exotic architectures need
  an HF-transformers fallback, out of scope here.
- **Out of memory (OOM) on model load.** Lower `--gpu-memory-utilization`, or use
  a smaller model. **Do not upsize past `g6.xlarge`** — the team rule is to drop
  the model, not the instance.
- **`AccessDenied` on S3 `PutObject`.** The instance role lacks write to your
  output prefix. Add a *scoped* policy for that prefix; do not modify the shared
  role globally.
- **Older GPUs (T4) log a FlashAttention-2 warning.** `FA2 is only supported on
  devices with compute capability >= 8` — vLLM falls back automatically and still
  serves inference (the proven T4 run completed with this warning present).

---

## Alternative: SSM-driven "rung1" smoke test

`run_rung1_smoketest.sh` (and the on-node `rung1_smoketest.sh`) in this folder
set up the **same** vLLM environment through AWS Systems Manager (SSM) instead of
a boot-time script, then run `olmo-eval` on `Qwen/Qwen2.5-0.5B-Instruct` +
`arc_easy`. Its scripts are complete and well-tested in structure, but no result
artifacts from a live run are committed, so this file treats it as the secondary
reference. Two differences worth knowing:

- **Install via the frozen lockfile.** Instead of explicit extras it uses
  `uv python install 3.12 && uv sync --frozen`. That works because `pyproject.toml`
  sets `default-groups = ["dev", "vllm"]`, so the `vllm` group (which mirrors the
  `vllm` extra) — plus the `s3` extra via `dev` — is installed with no flags.
- **Root/SSM environment.** SSM's `AWS-RunShellScript` runs as root with no
  `HOME`, so the on-node commands `export HOME=/root`, put uv on PATH, and pin
  **both** `UV_CACHE_DIR` and `HF_HOME` onto `/opt/dlami/nvme/...`. The eval is
  launched detached with `setsid nohup ... &` because SSM stays `InProgress`
  until every child exits — you tail the log file instead of waiting on SSM.

For the full launch → stage → run → verify → teardown flow (including the team
DLAMI/subnet/security-group defaults and cost discipline), follow
[`RUNBOOK_rung1_smoketest.md`](RUNBOOK_rung1_smoketest.md).

---

## Source / provenance

- **Primary source:** the `eval-direct-gpu` skill —
  `.cursor/skills/eval-direct-gpu/scripts/bootstrap.sh` (environment) and
  `scripts/run_eval_sweep.sh` (vLLM inference), read here read-only via
  `git show p3-tickets:...`.
- **Proven on AWS.** A `--group smoke` run against `step300/` of a mock training
  run completed on a single NVIDIA **T4** (`g4dn.xlarge`, `us-east-1`), from a GPU
  Deep Learning AMI. Evidence downloaded verbatim from S3 lives under
  `checkpoint_eval_smoke_step300/`:
  - `results/_bootstrap/userdata.log` — bootstrap on the box: `aws-cli/2.36.15 …
    Linux/6.8.0-1061-aws`, `nvidia: Tesla T4, 15360 MiB`, uv 0.12.1, CPython
    3.12.13, `uv sync --extra vllm --extra hf --extra s3 --extra olmo_core`, and
    resolved versions **vLLM 0.19.1, torch 2.10.0+cu128, transformers 5.7.0,
    ai2-olmo-core 2.4.0, boto3 1.42.95**; ends `SWEEP_EXIT=0`.
  - `results/step300/logs/vllm_server_*/vllm_server_*.log` — the vLLM API server
    serving `/v1/models` and `/v1/completions` with `200 OK`.
  - `results/step300/metrics.json`, `run_provenance.json` (`status: "ok"`,
    `tensor_parallel_size: 1`, `gpu_memory_utilization: 0.5`), the `_READY`
    marker, and `results/accuracy_wide.csv`.
- **Dependency pins** cross-checked against `pyproject.toml`: the `vllm` extra
  (`vllm[runai]==0.19.1`, Darwin-excluded), the `s3` extra (`boto3`,
  `smart_open[s3]`), and `default-groups = ["dev", "vllm"]`.
- **Secondary reference:** `tests/aws/run_rung1_smoketest.sh` /
  `rung1_smoketest.sh`, added in commit `88e3f8c0` ("aws instead of beaker smoke
  run"); its `RUNBOOK_rung1_smoketest.md` is the authoritative launch/teardown
  guide.