# Split-cost sweep: CPU download → GPU inference

Decouple the two phases of a sweep so each runs on the cheapest hardware that
can do it:

- **Weight download is pure network I/O** — no GPU, almost no CPU. Do it on a
  cheap CPU box (or a small fleet) and push a **warm Hugging Face cache to S3**.
- **Inference is GPU-bound.** GPU workers pull the warm cache and load weights
  from disk — no HF downloads on paid GPU time.

At fleet scale this converts N wasted "GPU-hours spent downloading" into ~1
cheap CPU-hour plus fast, free, in-region S3 transfers.

> This folder is **self-contained** and does not modify anything else. It reuses
> the existing registry (`models_registry.py`) and the existing sweep driver
> (`run_benchmark.py`) as-is.

## Why it saves money

| Phase | Bottleneck | Right hardware |
|---|---|---|
| Download hundreds of GB of weights | network / disk | cheap CPU (`c7i.large` ≈ \$0.09/hr) |
| MCQ scoring + FRQ generation | GPU compute | `g6.xlarge` L4 ≈ \$0.80/hr |

- **Download once, fan out to many GPUs.** All workers read the same S3 cache
  (multi-Gbps, free in-region) instead of each hitting the HF hub (slow,
  rate-limited, gated retries).
- **GPU boxes spend ~100% of paid time on inference.**

## Files

| File | Runs on | Purpose |
|---|---|---|
| `prefetch_weights.py` | CPU | `snapshot_download` every roster repo into `HF_HOME` (no torch/vllm) |
| `list_repos.py` | either | Map roster/shard → repo ids or `models--org--name` cache dirs |
| `bootstrap_downloader.sh` | CPU | Tiny venv: `huggingface_hub` + `hf_transfer` + `datasets` |
| `sync_cache_to_s3.sh` | CPU | Prefetch weights, then `aws s3 sync HF_HOME → S3`, write `_READY` |
| `pull_cache_from_s3.sh` | GPU | Sync cache from S3 (optionally only this shard's models) |
| `run_gpu_infer.sh` | GPU | Pull cache, then run the **existing** `run_benchmark.py` |
| `launch_split_aws.sh` | control | Launch CPU downloader → wait `_READY` → launch GPU fleet (DRY_RUN by default) |

## S3 layout

```
s3://…/hf-cache/
  hub/models--org--name/snapshots/<sha>/…   # standard HF cache (preserved intact)
  _manifests/prefetch_<ts>.json
  _READY                                     # marker: cache is complete
s3://…/split_infer/Outputs/…                 # inference outputs
```

Keeping the standard `hub/models--org--name/snapshots/<sha>/` layout is what lets
vLLM resolve each model by its repo id straight from the synced cache.

## Manual usage

### 1) On the cheap CPU box
```bash
export ROOT=/opt/adaptive-dl
ROOT=$ROOT REGION=us-east-1 bash bootstrap_downloader.sh
# download the whole roster (or filter), then push to S3:
ROOT=$ROOT S3_CACHE=s3://edullm-adaptive-inference-056956104102/hf-cache \
  MAX_PARAMS_B=1.5 \
  bash sync_cache_to_s3.sh
```

### 2) On each GPU worker (venv already has vLLM via `aws/node_bootstrap.sh`)
```bash
export ROOT=/opt/dlami/nvme/adaptive-infer
ROOT=$ROOT SHARD_INDEX=0 NUM_SHARDS=4 \
  S3_CACHE=s3://edullm-adaptive-inference-056956104102/hf-cache \
  S3_OUT=s3://edullm-adaptive-inference-056956104102/split_infer \
  MAX_PARAMS_B=1.5 \
  bash run_gpu_infer.sh
```

## One-shot AWS orchestration (optional)

```bash
# prints the exact AWS calls, spends nothing:
NUM_SHARDS=4 MAX_PARAMS_B=1.5 bash launch_split_aws.sh
# actually launch (CPU downloader → wait READY → GPU fleet):
DRY_RUN=0 NUM_SHARDS=4 MAX_PARAMS_B=1.5 bash launch_split_aws.sh
```

## Smoke test (`smoke_split.sh`)

End-to-end validation of the split path with 5 small, ungated, vLLM-friendly
models — download on a cheap CPU box, then a few real inferences on GPU pulling
weights straight from S3.

```bash
# on the CPU downloader box (parallel, workers=5):
bash smoke_split.sh cpu     # -> s3://…/smoke_split/hf-cache + _READY

# on the GPU box (after _READY), pulls only those 5 models, runs smoke_test.py:
bash smoke_split.sh gpu     # -> s3://…/smoke_split/results/<stamp>/{smoke_report.json,DONE}
```

Default models: `Qwen2.5-0.5B`, `Qwen2.5-0.5B-Instruct`, `SmolLM2-135M`,
`SmolLM2-360M`, `Qwen2.5-1.5B-Instruct` (≈3B params total, ~6 GB). The GPU side
runs `smoke_test.py --num-mcq 3 --num-frq 2` — a handful of inferences — and
proves the weights came from the S3 cache (it never invokes a downloader).
`launch_split_aws.sh` is the on-demand launcher for the full sweep; the smoke is
driven the same way (CPU first, then GPU after `_READY`).

## Notes & gotchas

- **Streaming push + free.** `prefetch_weights.py --s3-prefix … --delete-after` pushes
  each model to S3 the moment it finishes and deletes it locally, so peak disk stays
  flat (≈ concurrent workers × model size) and every completed model is persisted even
  if the box later dies. `sync_cache_to_s3.sh` and the smoke enable this by default.
- **Selective per-shard pull.** With `SHARD_INDEX`/`NUM_SHARDS`, `pull_cache_from_s3.sh`
  only syncs that shard's `models--org--name/*` folders (smaller disk, faster start).
  The same filter (`MAX_PARAMS_B`, `MODELS_YAML`) must be used on both sides.
- **Gated models** need `HF_TOKEN` only on the downloader (cached from Secrets
  Manager as `hf-token`). GPU workers don't need it once the cache is warm.
- **Datasets.** FRQ banks are local to `eduLLM-Evals`; MCQ datasets are small and
  fetched at run time. Set `OFFLINE=1` on the GPU only if you've also pre-cached
  datasets (otherwise MCQ loads would fail under strict offline).
- **Keep everything in one region** and add a **VPC S3 gateway endpoint** so cache
  transfer never incurs NAT/egress charges.
- **Alternatives to S3 warm-cache:** an EBS snapshot hand-off (download → snapshot
  → restore per worker) or a shared EFS mount. S3 is the cheapest, simplest, and
  resumable default; local NVMe restored from S3 loads faster than EFS.
