# EDLLM Adaptive Inference — Agent Runbook

Operating guide for running the adaptive-inference sweeps on AWS (`sbsandbox`). Read **Section 0** before touching `run-instances`.

## Table of Contents

- [0. Pre-Launch Checklist (mandatory)](#0-pre-launch-checklist-mandatory)
- [1. First-Time Setup (local machine)](#1-first-time-setup-local-machine)
- [2. Account & Region Defaults](#2-account--region-defaults)
- [3. Staging Code to a Node (S3 + SSM Pattern)](#3-staging-code-to-a-node-the-s3--ssm-pattern)
- [4. HuggingFace Token (Secrets Manager)](#4-huggingface-token-secrets-manager)
- [5. Launching EC2 Instances](#5-launching-ec2-instances)
- [6. Running the Inference Sweep on a Node](#6-running-the-inference-sweep-on-a-node)
- [7. The Split-Download-Infer Fleet Architecture](#7-the-split-download-infer-fleet-architecture)
- [8. Known Gotchas](#8-known-gotchas)
- [9. Syncing Results Back Locally](#9-syncing-results-back-locally)
- [10. Cost Reference](#10-cost-reference)

---

## 0. Pre-Launch Checklist (mandatory)

**Before every `run-instances` call, the agent must confirm all three of the following. Do not launch if any check fails.**

1. **Instance type is `g6.xlarge` or cheaper** — reject any other GPU type.
2. **Running GPU instance count ≤ 2** (i.e., launching this one won't exceed 3 total). Check with:
   ```bash
   aws ec2 describe-instances \
     --filters "Name=tag:Name,Values=edullm-gpu-worker" "Name=instance-state-name,Values=running,pending" \
     --query 'Reservations[].Instances[].InstanceId' \
     --profile sbsandbox --region us-east-1
   ```
3. **`user-data` includes a `shutdown -h +N` line**, with `N` calculated for this specific run (not a fixed default).

---

## 1. First-Time Setup (local machine)

```bash
# Install the broker (needs Node 20+)
npm install -g ./sb-aws-creds-0.2.1.tgz

# If `sb-aws-creds` isn't on PATH (homebrew node puts it under Cellar):
# find the absolute path, e.g. /opt/homebrew/Cellar/node@22/22.23.0/bin/sb-aws-creds
which sb-aws-creds || find /opt/homebrew -name sb-aws-creds 2>/dev/null

# Authenticate (opens browser → Google login with @alphaaiengineering.com)
sb-aws-creds login
```

**Register in Cursor (`~/.cursor/mcp.json`):**

```json
{
  "mcpServers": {
    "sb_aws": {
      "command": "/absolute/path/to/sb-aws-creds",
      "args": ["mcp"]
    },
    "TypeUI": { ... }   // preserve existing entries
  }
}
```

- Server name must be `sb_aws` (matches the `/Test` skill's `mcp__sb_aws__*` references)
- Use the absolute path — the binary may not be on PATH
- Restart Cursor after saving

**Verify setup:**

```
accounts → shows sbsandbox (ready), sbproduction, legacy
whoami   → shows arhant.choudhary@alphaaiengineering.com, refresh token expiry
```

Token is valid ~30 days. If `whoami` works, no re-login needed.

---

## 2. Account & Region Defaults

| Config | Value |
| --- | --- |
| Account | `sbsandbox` (056956104102) |
| Region | `us-east-1` |
| S3 bucket | `edullm-adaptive-inference-056956104102` |
| Instance profile | `EswManagedInstance` |
| IAM role | `EswManagedInstance` (also used by other projects — don't modify globally) |

The `sbproduction` account exists but interns have restricted access. Always use `sbsandbox` for inference work.

---

## 3. Staging Code to a Node (the S3 + SSM Pattern)

The standard pattern for getting code onto an EC2 node — you cannot `scp` or embed large blobs in SSM args.

```
Local machine → tar code → upload to S3 (via sb_aws) → presigned URL →
SSM run-command (curl + untar on node)
```

**Step by step:**

1. **Tar your code:** `tar czf code.tgz -C /path/to AdaptiveTesting/Test/Inference/`
2. **Create/use S3 bucket:** `aws s3 mb s3://edullm-adaptive-inference-056956104102 --profile sbsandbox` (bucket already exists)
3. **Upload:** `aws s3 cp code.tgz s3://edullm-adaptive-inference-056956104102/runs/code.tgz --profile sbsandbox`
4. **Presign (do this via `sb_aws` MCP, not local CLI — broker creds are longer-lived than node STS tokens):** `aws s3 presign s3://...code.tgz --expires-in 3600`
5. **Send to node via SSM:**
   ```bash
   aws ssm send-command \
     --instance-ids i-XXXXXXXXXXXXXXXXX \
     --document-name AWS-RunShellScript \
     --parameters 'commands=["mkdir -p /opt/dlami/nvme/adaptive-inference && cd /opt/dlami/nvme/adaptive-inference && curl -o code.tgz \"<PRESIGNED_URL>\" && tar xzf code.tgz && chmod +x code/Test/Inference/aws/*.sh"]' \
     --profile sbsandbox --region us-east-1
   ```

> **Always presign via the MCP broker, not the node's STS token** — the node's temp credentials expire faster.

---

## 4. HuggingFace Token (Secrets Manager)

Gated models (meta-llama, google, mistralai) need `HF_TOKEN`. Never commit it or pass it in plain env vars.

**Create the secret once:**

```bash
# From your .env file, using your own AWS CLI auth:
./AdaptiveTesting/Test/Inference/aws/put_hf_secret.sh
```

Or via `sb_aws` MCP directly:

```bash
aws secretsmanager create-secret --name hf-token \
  --secret-string '{"HF_TOKEN":"hf_..."}' \
  --region us-east-1 --profile sbsandbox
```

Secret ARN: `arn:aws:secretsmanager:us-east-1:056956104102:secret:hf-token-...`

**Grant node access (resource policy — don't touch the role itself):**

```bash
aws secretsmanager put-resource-policy \
  --secret-id hf-token \
  --resource-policy '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"arn:aws:iam::056956104102:role/EswManagedInstance"},"Action":"secretsmanager:GetSecretValue","Resource":"*"}]}' \
  --profile sbsandbox
```

The node's run script fetches it at runtime: `aws secretsmanager get-secret-value --secret-id hf-token`.

---

## 5. Launching EC2 Instances

> Also see the [Pre-Launch Checklist](#0-pre-launch-checklist-mandatory) — it applies to every launch below.

**Reusable launch parameters (from prior runs):**

| Parameter | Value |
| --- | --- |
| AMI | `ami-0b6f2229ad14c9323` (GPU DLAMI, Ubuntu 24.04) |
| Security group | `sg-087218d8c87aa8576` |
| Subnet (1b) | `subnet-0a4235fb98b63930f` (try 1b first, rotate on capacity issues) |
| Instance profile | `EswManagedInstance` |

**GPU instance types:**

| Type | GPU | Use case | Allowed? |
| --- | --- | --- | --- |
| `g6.xlarge` | L4 | All EDLM inference runs | ✅ Max allowed GPU |
| `g5.xlarge` | A10G | Smoke tests only | ✅ OK for smoke |
| `g6e.xlarge` | L40S | 7B / large models | ❌ Forbidden |
| `p6-b200.48xlarge` | 8× B200 | Shared training node only | ❌ Forbidden for new launches |

> **Hard rule: never launch an instance type above `g6.xlarge` (L4).** If a model requires more VRAM than an L4 (24GB) can provide, skip it — do not upgrade the instance type.

**CPU downloader instances:** `m7i.xlarge` (16GB RAM, 60GB gp3 root) — for the split-download-infer architecture.

**Launch command pattern (via `sb_aws` MCP) — always `g6.xlarge`, always ≤3 total:**

```bash
aws ec2 run-instances \
  --image-id ami-0b6f2229ad14c9323 \
  --instance-type g6.xlarge \
  --count 1 \
  --iam-instance-profile Name=EswManagedInstance \
  --security-group-ids sg-087218d8c87aa8576 \
  --subnet-id subnet-0a4235fb98b63930f \
  --user-data file://bootstrap.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=edullm-gpu-worker}]' \
  --profile sbsandbox --region us-east-1
```

> **Capacity:** If a zone is out of capacity, rotate subnets. `us-east-1b` has worked consistently.

---

## 6. Running the Inference Sweep on a Node

**On a shared node (e.g. the B200 training box), pick a free GPU:**

```bash
nvidia-smi  # GPUs 0-1 used by MemorySplit training — use 2+
```

**Launch via SSM (detached so it survives session close):**

```bash
aws ssm send-command \
  --instance-ids i-05de75630c4774cdd \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["export CUDA_VISIBLE_DEVICES=2 && export HF_TOKEN=$(aws secretsmanager get-secret-value --secret-id hf-token --query SecretString --output text | python3 -c \"import sys,json; print(json.load(sys.stdin)['\''HF_TOKEN'\''])\") && setsid nohup /opt/dlami/nvme/adaptive-inference/code/Test/Inference/aws/node_run_sweep.sh > /opt/dlami/nvme/adaptive-inference/sweep.log 2>&1 &"]' \
  --profile sbsandbox --region us-east-1
```

**Key env vars in the sweep script:**

- `CUDA_VISIBLE_DEVICES` — pin to a free GPU (e.g. `2`)
- `VLLM_ATTN_BACKEND` — set per model requirements
- `HF_TOKEN` — fetched from Secrets Manager at launch
- `GPU` defaults to `2` in `node_run_sweep.sh`; `tensor_parallel_size: 1` in `inference.node.yaml`

**Check logs:**

```bash
aws ssm send-command \
  --instance-ids i-XXXXXXXXXX \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["tail -50 /opt/dlami/nvme/adaptive-inference/sweep.log"]' \
  --profile sbsandbox
```

> **SSM InProgress hang:** SSM marks a command InProgress until all child processes exit. Use `setsid nohup ... &` to fully detach; check logs separately rather than waiting on the SSM status.

---

## 7. The Split-Download-Infer Fleet Architecture

For large runs (100–200 models), use the two-tier CPU+GPU fleet pattern:

- **16× `m7i.xlarge` CPU downloaders** — each instance learns its shard index from `ami-launch-index` (IMDSv2, see below), downloads its slice of models from HuggingFace, streams weights to S3 under `full200/hf-cache/`, writes `ready/<model>` markers. Shard 0 also builds MCQ dataset caches.
- **Up to 3× `g6.xlarge` GPU workers (hard cap)** — install vLLM offline from the S3 wheelhouse, poll `ready/` markers, claim models, run inference, upload CSV results under `full200/results/Outputs/`, self-terminate when queue is empty **or when the auto-shutdown timer fires** (see [Section 0](#0-pre-launch-checklist-mandatory)). Never launch more than 3 GPU instances in parallel.

**S3 structure:**

```
s3://edullm-adaptive-inference-056956104102/full200/
  hf-cache/hub/          # model weights
  ready/<model-slug>     # download-complete markers
  done/<model-slug>      # inference-complete markers
  results/Outputs/mcq/   # per-benchmark CSV results
  _WHEELS_READY          # vLLM offline wheelhouse built
  wheels/                # ~6GB of pre-built Python wheels
```

**Results format:** `Outputs/mcq/<benchmark>/<org>__<model>.csv` + `.done` manifests.

---

## 8. Known Gotchas

### IMDSv2 (critical for CPU shard scripts)

The GPU DLAMI enforces IMDSv2. Plain `curl http://169.254.169.254/latest/meta-data/ami-launch-index` returns blank. All metadata fetches need a token:

```bash
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
SHARD=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/ami-launch-index)
```

Without this, all 16 CPU boxes get `SHARD=""`, do nothing, and self-terminate.

### vLLM vs. HF Fallback

Not all models in `models.yaml` run on vLLM. Models with unsupported architectures must be flagged `backend: hf_fallback`:

- Currently flagged: `google/gemma-3-*`, `apple/OpenELM-*`, `state-spaces/mamba-*`
- Impact: ~5 models, ~30–50 extra minutes on the HF transformers path (scores one item at a time, no batching)
- Symptom if missing: vLLM crashes on model load with architecture config errors

### Loop Order (model-outer vs. benchmark-outer)

- **model-outer** (default): Load one model → run all 17 benchmarks → unload → next model. Correct for a single-worker fleet.
- **benchmark-outer**: Run all models through one benchmark → next benchmark. Results in up to 1,700 model loads on a single node — catastrophic for wall-clock time.

### S3 PutObject Permissions

The node's IAM role (`EswManagedInstance`) may not have write access to all S3 prefixes by default. If results aren't syncing, attach a resource-based policy to the bucket or a scoped inline policy to the role covering your specific S3 prefix.

### HF Token in SSM Commands

Don't inline the raw token in SSM parameter strings — it appears in CloudTrail and SSM history. Always fetch from Secrets Manager at runtime.

### GPU Safety Check

The `node_run_sweep.sh` launcher refuses to start if the target GPU has any running compute process. GPUs 0 and 1 on the shared B200 box run MemorySplit training — never use them.

---

## 9. Syncing Results Back Locally

```bash
# Sync all CSV outputs from S3 to local
aws s3 sync s3://edullm-adaptive-inference-056956104102/full200/results/Outputs/ \
  /Users/arhant/Documents/EDLM/olmo-eval-full/AdaptiveTesting/Outputs/_full200/ \
  --profile sbsandbox

# Or generate a presigned URL for a tarball and curl it:
aws ssm send-command \
  --instance-ids i-XXXXXXXXXX \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["tar czf /tmp/results.tgz /opt/dlami/nvme/adaptive-inference/Outputs/ && aws s3 cp /tmp/results.tgz s3://edullm-adaptive-inference-056956104102/results.tgz"]' \
  --profile sbsandbox
# Then presign and curl locally
```

---

## 10. Cost Reference

| Run | Instances | Duration | Cost |
| --- | --- | --- | --- |
| 56-model pedagogy backfill | Mix of g6/g6e | ~2 hrs | ~$4.60 |
| 200-model fleet (paused) | 8× g6e.xlarge + 16× m7i | partial | ~$0.05 at T+5min |

Always set a hard cost cap (e.g. `$15`) before launching a fleet. All GPU boxes self-terminate when the work queue empties.
