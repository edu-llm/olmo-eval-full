# CPU sweep — 200 models × education / instruction suite

Parallel HF/`transformers` job (no GPU required) over `models_200.yaml`.
MCQ benches write `correct`/`wrong`; open benches write **responses only** (no judge).

## Suite

See [`DATASETS.md`](DATASETS.md) for Hugging Face download links.

Alias: `--benchmarks cpu_sweep`

## Quick start (one machine / one shard)

```bash
# 0) bootstrap (once per host)
ROOT=/opt/adaptive-cpu bash bootstrap_cpu.sh
source /opt/adaptive-cpu/venv/bin/activate
export VENV_PY=/opt/adaptive-cpu/venv/bin/python
export ROOT=/opt/adaptive-cpu
export HF_TOKEN=…   # gated models + pedagogy; or put it in AdaptiveTesting/.env

# 1) parallel dataset prefetch (all hosts can do this; safe / resumeable)
cd /opt/adaptive-cpu/code/AdaptiveTesting/Test/Inference
python cpu_sweep_200/prefetch_datasets.py --workers 8

# 2) run this machine's shard (example: machine 3 of 8)
export SHARD_INDEX=3 NUM_SHARDS=8
export S3_URI=s3://edullm-adaptive-inference-056956104102/cpu_sweep_200
export PROBE_QUESTIONS=100
export PROBE_MAX_SECONDS=900   # skip model if 100Q > 15 min
bash cpu_sweep_200/run_cpu_worker.sh
```

## Multi-machine split

Models are round-robin sharded (`--shard-index` / `--num-shards`).
Give every host the **same** `NUM_SHARDS` and a unique `SHARD_INDEX ∈ [0, NUM_SHARDS)`.

```bash
NUM_SHARDS=8 S3_URI=s3://…/cpu_sweep_200 ./launch_fleet.sh   # prints per-host commands
```

Continuous `aws s3 sync` of `AdaptiveTesting/Outputs/` runs every `S3_SYNC_SECONDS` (default 120) and on exit. Failures never block inference.

## Safety / non-blocking behavior

| Guard | Behavior |
|---|---|
| Model load failure | log `[skip model]`, continue |
| Per-benchmark exception | log `[skip]`, continue |
| Slow probe (`PROBE_QUESTIONS` / `PROBE_MAX_SECONDS`) | log `[skip slow model]`, abandon remaining benches for that model |
| Dataset load failure at start | drop that benchmark for the whole shard |
| S3 sync errors | logged; worker keeps going |
| Resume | per-(benchmark, model) CSV/JSONL + `.done` manifests |

Default probe: **100 questions / 900s**. Tune down for stricter skip (e.g. `600`).

## Cost-effective AWS recommendation

**Prefer a small GPU fleet over pure CPU for ≤7B.** CPU HF log-likelihood is ~10–50× slower than a single L4/A10G; wall-clock and \$ often favor GPU.

### Best \$ / throughput (recommended)

| Option | Instance | Why |
|---|---|---|
| **Primary** | **`g6.xlarge` (1× L4 24GB)** or **`g5.xlarge` (1× A10G 24GB)** | Spot-friendly; one 7B model fits; use existing vLLM path (`aws/run_parallel.sh`) with `models_200.yaml` |
| Fleet size | **8–16 shards** (one instance each) | Round-robin 200 models → ~12–25 models/host |
| Storage | **200–500GB gp3** + HF cache on disk | Avoid re-download |
| Spot | Yes (Batch / ASG with interruption resume — outputs already resume) | Cut ~60–70% compute \$ |

Wire GPU hosts to the same S3 prefix; keep `--no-judge` for this suite.

### If you must stay CPU-only

| Option | Instance | Notes |
|---|---|---|
| **Best CPU** | **`c7i.8xlarge` / `c6i.8xlarge` (32 vCPU, 64GB)** | High clock + enough RAM for ~7B fp32 (~28GB) or better: load in fp16 on CPU if available |
| Alt | **`c7i.4xlarge` (16 vCPU, 32GB)** | Only for ≤3B comfortably; 7B will swap/OOM in fp32 |
| Fleet | **16–32 shards** | CPU needs more parallelism to finish in reasonable time |
| Avoid | burstable `t3` / low-RAM | thrash on 7B |

CPU tips:
- `FORCE_CPU=1` (set by `run_cpu_worker.sh`)
- `OMP_NUM_THREADS=nproc` but **one model process per host** (multi-process on one box fights for RAM)
- Prefer **fp16/bf16 on GPU**; on CPU stick to models that fit in RAM

### Rough order-of-magnitude (200 models × ~12 benches × ≤2k items)

| Setup | Ballpark wall time | Ballpark \$ (spot) |
|---|---|---|
| 8× `g6.xlarge` + vLLM | 1–3 days | low–mid hundreds |
| 16× `c7i.8xlarge` CPU HF | 1–3 weeks | often higher than GPU for same finish time |

### S3 layout

```
s3://edullm-adaptive-inference-…/cpu_sweep_200/
  Outputs/mcq/<bench>/<org__model>.csv
  Outputs/open/<bench>/<org__model>.responses.jsonl
  Outputs/_manifests/
```

### IAM

Worker role needs: `s3:GetObject/PutObject/ListBucket` on that prefix, `secretsmanager:GetSecretValue` on `hf-token`, and (if using Batch) standard Batch/ECS exec roles.
