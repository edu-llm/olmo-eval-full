# RUNBOOK — Rung 1 Smoke Test (olmo-eval on the team's AWS, for AWS first-timers)

**Goal:** prove that `olmo-eval run` works on a team-provisioned GPU machine and uploads
its results to the team's S3 bucket — using a **built-in olmo-eval task** and a **tiny
model**. This de-risks the larger "replace Beaker with AWS" effort *before* anyone builds
a launcher. It is deliberately the cheapest, smallest thing that can possibly work.

This guide is written for someone who has **never used AWS**. Every AWS term gets a
one-line plain-English analogy the first time it appears. Follow it top-to-bottom and
copy-paste the commands. The ⚠️ callouts are where people lose money or time.

> **You are NOT creating an AWS account and you do NOT need a GPU-quota request.** The
> team (`sbsandbox` account) is already fully provisioned: bucket, GPU image, network,
> and permissions all exist. You get in through a **credential broker** (Section 1) and
> reuse the team defaults (Section 2). This runbook builds directly on the team's real
> AWS runbook by Arhant (`AWS run book.md`).

> ⚡ **Automated one-shot (scripted alternative).** Everything below (Sections 3–8) is
> also implemented as a single parameterized script, `run_rung1_smoketest.sh`, in this
> folder. It runs the whole flow from your laptop with the team defaults:
> ```bash
> ./run_rung1_smoketest.sh
> ```
> **Always preview first** — `--dry-run` prints every AWS command it would run and
> executes nothing:
> ```bash
> ./run_rung1_smoketest.sh --dry-run
> ```
> By default it **`git clone`s the public olmo-eval repo on the node** (the built-in
> `arc_easy` task lives there), so **no local repo path is needed**. To instead ship a
> local checkout via the S3 tar route (Section 5) — e.g. once you have uncommitted/private
> code — opt in with `STAGE_MODE=tar`:
> ```bash
> STAGE_MODE=tar ./run_rung1_smoketest.sh
> ```
> Prereq: `aws` CLI + `sb-aws-creds login` && `sb-aws-creds install-profiles` (so
> `aws --profile sbsandbox` works). Every value is overridable via env vars/flags; see
> `./run_rung1_smoketest.sh -h`. The manual steps below remain the source of truth.

---

## 0. Overview & definition of done

**In plain English:** we rent **one** small GPU computer in Amazon's cloud for a few
minutes, install `olmo-eval` on it, run a tiny evaluation (10 questions of a built-in
multiple-choice task with a 0.5B model), have it upload the scores to the team's cloud
folder (S3), then shut the computer down. That's it.

**The whole flow:**

```
sb-aws-creds (broker) → get AWS access
      │
      ▼
launch ONE g5.xlarge GPU instance (team AMI/SG/subnet, auto-shutdown timer)
      │
      ▼
stage olmo-eval onto the node (S3 + presigned URL + SSM)  →  uv sync (installs vLLM)
      │
      ▼
olmo-eval run  -m Qwen/Qwen2.5-0.5B-Instruct  -t arc_easy  -o limit=10  → uploads to S3
      │
      ▼
verify metrics.json in s3://edullm-adaptive-inference-056956104102/smoketest/…
      │
      ▼
TERMINATE the instance (belt-and-suspenders; the timer also self-terminates)
```

**✅ Definition of done:** you can run `aws s3 ls` on the team bucket and see a
`metrics.json` (plus `predictions/` and `requests/`) under the `smoketest/rung1/…`
prefix, containing an accuracy number for `arc_easy`. When you see your results in S3,
Rung 1 passes.

**A note on how you'll actually run AWS commands.** Once you register the `sb_aws` MCP
(Section 1), **your Cursor agent can run AWS commands on your behalf** through the
broker's `aws` tool — you can literally ask it to "launch the instance" or "list the
smoketest results" and it will call AWS with the `sbsandbox` credentials. Every AWS
command in this runbook is shown in raw `aws … --profile sbsandbox` form so it is
copy-pasteable, but the beginner-friendly path is to hand these to the agent.

---

## 1. Get access via the broker (`sb-aws-creds`)

AWS normally makes you paste temporary keys into a file every few hours. The team
replaced that with **`sb-aws-creds`** — a small command-line **credential broker**
(*broker = a helper that hands out short-lived AWS keys after you log in with Google, so
you never copy-paste secrets*). It also runs as an **MCP server** (*MCP = the plug-in
protocol Cursor uses to give the agent new tools*), which is what lets the agent make
AWS calls for you.

### 1a. Install it (needs Node 20+)

```bash
# From wherever you extracted the package (the .tgz the team gave you):
npm install -g ./sb-aws-creds-0.2.1.tgz

# Find the absolute path to the installed binary — you'll need it for the MCP config.
# (Homebrew's node hides binaries under Cellar, so `which` may not show it.)
which sb-aws-creds || find /opt/homebrew -name sb-aws-creds 2>/dev/null
```

> ℹ️ The `README.md` *inside* the package describes an older `pipx`/Python install with
> `install-profiles`. Ignore it — the shipped `sb-aws-creds@0.2.1` is the **Node** CLI
> the team runbook describes (`package.json` confirms `"bin": {"sb-aws-creds": …}`,
> `"engines": {"node": ">=20"}`, and a `mcp` subcommand). Use `npm install -g` as above.

### 1b. Log in

```bash
sb-aws-creds login
# Opens your browser → log in with your @alphaaiengineering.com Google account → approve.
# A refresh token is saved to your OS keychain. It lasts ~30 days.
```

### 1c. Register the `sb_aws` MCP server in Cursor

Edit `~/.cursor/mcp.json` and add an `sb_aws` entry. **Use the absolute path** from step
1a as `command`, and `["mcp"]` as `args`. Preserve any existing entries.

```json
{
  "mcpServers": {
    "sb_aws": {
      "command": "/absolute/path/to/sb-aws-creds",
      "args": ["mcp"]
    }
  }
}
```

- The server name **must** be `sb_aws` (tooling references `mcp__sb_aws__*`).
- **Restart Cursor** after saving so it picks up the new MCP server.

### 1d. Verify

Ask the agent (or use the MCP tools) to run:

- **`accounts`** → should list **`sbsandbox` (ready)**, plus `sbproduction`, `legacy`.
- **`whoami`** → should show your `…@alphaaiengineering.com` email and the token expiry.

If `whoami` works, you're authenticated and no re-login is needed for ~30 days. The
broker also exposes an **`aws`** tool — that's the one the agent uses to run any
`aws …` command below against `sbsandbox`.

> **Optional — run AWS locally too.** If you'd rather run `aws` yourself in a terminal
> instead of through the agent, run `sb-aws-creds install-profiles`, which writes a
> `credential_process` block into `~/.aws/config` so `aws --profile sbsandbox …` works
> transparently. Not required for this runbook.

---

## 2. Team account defaults (`sbsandbox`)

Everything below is **already created** — reuse these exact values. Do not create new
buckets/roles.

| Config | Value | One-line analogy |
| --- | --- | --- |
| Account | `sbsandbox` (`056956104102`) | the team's cloud "workspace" |
| Region | `us-east-1` | which data-center city your resources live in |
| S3 bucket | `edullm-adaptive-inference-056956104102` | a shared cloud folder (`s3://…`) |
| GPU AMI | `ami-0b6f2229ad14c9323` (Ubuntu 24.04 DLAMI) | pre-baked disk image w/ NVIDIA drivers + CUDA already installed |
| Security group | `sg-087218d8c87aa8576` | the instance's firewall rules |
| Subnet (1b) | `subnet-0a4235fb98b63930f` | which slice of the network / availability zone it sits in |
| Instance profile | `EswManagedInstance` | an ID badge attached to the machine that grants it AWS permissions (e.g. S3) with **no** pasted keys |

- **EC2** = a rented computer in Amazon's data center; a "GPU instance" has an NVIDIA GPU
  attached. You pay **per hour** while it runs.
- **AMI (Amazon Machine Image)** = the OS + drivers the instance boots from. The team's
  DLAMI already has GPU drivers + CUDA, so there's no driver-install pain.
- **IAM role / instance profile** = permissions the machine carries so it can talk to S3
  etc. without secret keys. `EswManagedInstance` is **shared with other projects — never
  modify the role globally.** If you need a permission it lacks, add a *scoped* policy
  for your prefix only (see Troubleshooting).

> Always use **`sbsandbox`** for this work. `sbproduction` exists but interns have
> restricted access.

---

## 3. Pre-launch checklist (MANDATORY)

**Before every `run-instances` call, confirm ALL THREE. Do not launch if any check fails.**
This is the team's hard rule — reproduced verbatim.

1. **Instance type is `g6.xlarge` or cheaper.** For a smoke test use **`g5.xlarge`**.
   ❌ Reject anything bigger (`g6e.xlarge`, `p6-b200.*`, etc.). If a model needs more than
   an L4's 24 GB, **skip the model — do not upsize the instance.**
2. **Running GPU instance count ≤ 2** (so launching this one won't exceed **3 total**).
   Check first:
   ```bash
   aws ec2 describe-instances \
     --filters "Name=tag:Name,Values=edullm-gpu-worker" "Name=instance-state-name,Values=running,pending" \
     --query 'Reservations[].Instances[].InstanceId' \
     --profile sbsandbox --region us-east-1
   ```
3. **`user-data` includes a `shutdown -h +N` line**, with `N` sized for *this* run (not a
   fixed default). For this smoke test, `N = 90` minutes is a safe cap (the eval itself
   takes only a few minutes; the timer is a safety net).

> ⚠️ **Cost discipline.** A GPU instance bills **every hour it runs**, even while idle.
> The auto-shutdown timer + explicit terminate at the end (Section 8) are what keep this
> test under a dollar. Always set a hard cost cap in your head before launching.

---

## 4. Launch ONE smoke-test instance

We launch a single **`g5.xlarge`** (1× NVIDIA **A10G**, 24 GB — the team's designated
"smoke tests only" type). A 0.5B model is *tiny*, so 24 GB is enormous headroom.

### 4a. Write the `user-data` bootstrap

**user-data** = a script AWS runs automatically the first time the instance boots. Ours
does the one mandatory thing: arm the auto-terminate timer. (We stage and run the eval
separately via SSM in Sections 5–6, mirroring the team pattern.)

Save this as `rung1_bootstrap.sh` locally:

```bash
#!/bin/bash
# Rung 1 smoke-test bootstrap. Runs once at first boot.
# MANDATORY safety net: halt the box 90 minutes from now no matter what.
# Combined with --instance-initiated-shutdown-behavior terminate (below),
# "halt" becomes "terminate", so the machine deletes itself and billing stops.
shutdown -h +90
```

### 4b. Run the pre-launch checklist (Section 3), then launch

```bash
aws ec2 run-instances \
  --image-id ami-0b6f2229ad14c9323 \
  --instance-type g5.xlarge \
  --count 1 \
  --iam-instance-profile Name=EswManagedInstance \
  --security-group-ids sg-087218d8c87aa8576 \
  --subnet-id subnet-0a4235fb98b63930f \
  --instance-initiated-shutdown-behavior terminate \
  --user-data file://rung1_bootstrap.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=edullm-gpu-worker},{Key=Purpose,Value=rung1-smoketest}]' \
  --profile sbsandbox --region us-east-1
```

- `--instance-initiated-shutdown-behavior terminate` — makes the `shutdown -h` timer
  **terminate** (fully delete + stop billing) instead of merely stopping. Important.
- `--iam-instance-profile Name=EswManagedInstance` — gives the box S3 access via its
  badge, so `olmo-eval` can upload with no pasted keys.
- Tag `Name=edullm-gpu-worker` — keeps it visible to the checklist's instance-count query.

Note the returned **InstanceId** (looks like `i-0abc123…`). Wait ~1–2 minutes, then
confirm it's running and SSM-reachable:

```bash
# State check
aws ec2 describe-instances --instance-ids i-XXXXXXXXXXXXXXXXX \
  --query 'Reservations[].Instances[].State.Name' --profile sbsandbox --region us-east-1

# SSM check — must show PingStatus "Online" before you can send commands
aws ssm describe-instance-information \
  --filters "Key=InstanceIds,Values=i-XXXXXXXXXXXXXXXXX" \
  --query 'InstanceInformationList[].PingStatus' --profile sbsandbox --region us-east-1
```

> **Capacity:** if the launch fails with an insufficient-capacity error, rotate the
> subnet/zone and retry. `us-east-1b` (the subnet above) has worked consistently.
>
> ℹ️ **Why no SSH / key pair?** The team's access pattern is **SSM**
> (*AWS Systems Manager — a way to send shell commands to an instance through AWS itself,
> no open SSH port and no `.pem` key needed*). So there's no key pair or port-22 rule to
> manage here; you drive the box entirely through `aws ssm send-command`.

---

## 5. Stage olmo-eval onto the node (S3 + presigned URL + SSM)

You can't `scp` to the box (no SSH) and you can't paste a whole repo into an SSM command.
The team's standard pattern is: **tar the code → upload to S3 → make a presigned URL →
tell the node (via SSM) to `curl` + untar it.** (*Presigned URL = a temporary, no-login
download link to a private S3 object.*)

```
Local → tar repo → aws s3 cp to bucket → aws s3 presign → SSM: curl + untar on node
```

**Step by step (run the presign via the `sb_aws` broker — its creds outlive the node's
short-lived STS token):**

```bash
# 1. Tar the olmo-eval repo locally (from the parent of the repo dir).
#    We only need built-in tasks, so the plain repo is enough.
tar czf olmo-eval.tgz -C /Users/cat/alpha-projects olmo-eval-full

# 2. Upload to the team bucket under a smoke-test staging path.
aws s3 cp olmo-eval.tgz \
  s3://edullm-adaptive-inference-056956104102/smoketest/staging/olmo-eval.tgz \
  --profile sbsandbox --region us-east-1

# 3. Presign a 1-hour download URL (do this via the sb_aws MCP / broker creds).
aws s3 presign \
  s3://edullm-adaptive-inference-056956104102/smoketest/staging/olmo-eval.tgz \
  --expires-in 3600 --profile sbsandbox --region us-east-1
#   → copy the long https://… URL it prints; use it as <PRESIGNED_URL> below.

# 4. Pull + unpack on the node via SSM.
aws ssm send-command \
  --instance-ids i-XXXXXXXXXXXXXXXXX \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["mkdir -p /opt/dlami/nvme/rung1 && cd /opt/dlami/nvme/rung1 && curl -sSL -o olmo-eval.tgz \"<PRESIGNED_URL>\" && tar xzf olmo-eval.tgz && ls olmo-eval-full"]' \
  --profile sbsandbox --region us-east-1
```

> ℹ️ **Simpler alternative if the node has GitHub egress:** instead of steps 1–4 you can
> just `git clone https://github.com/allenai/olmo-eval` on the node via SSM. The built-in
> `arc_easy` task exists in the public repo, so a clone is sufficient for Rung 1. The
> S3+presigned pattern above is the team-blessed default and always works, so it's shown
> first.

**Install olmo-eval with `uv`** (the project uses `uv` with a checked-in `uv.lock`;
Python **3.12+** required). The GPU inference provider (**vLLM**) lives in the `vllm`
dependency group, which is in the project's **default groups**, so a plain
`uv sync --frozen` installs it on Linux (vLLM is marked Linux/CUDA-only via PEP 508
markers). The default groups also pull in the `s3` extra (`boto3` + `smart_open`), which
is what lets olmo-eval upload results to S3.

```bash
aws ssm send-command \
  --instance-ids i-XXXXXXXXXXXXXXXXX \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["cd /opt/dlami/nvme/rung1/olmo-eval-full && export HOME=/root && export PATH=\"$HOME/.local/bin:$PATH\" && export UV_CACHE_DIR=/opt/dlami/nvme/rung1/uv-cache && export HF_HOME=/opt/dlami/nvme/rung1/hf && mkdir -p \"$UV_CACHE_DIR\" \"$HF_HOME\" && curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12 && uv sync --frozen && uv run olmo-eval --help | head -20"]' \
  --profile sbsandbox --region us-east-1
```

- `uv sync --frozen` — installs from the lockfile; default groups (`dev` + `vllm`) bring
  in **vLLM + torch + transformers + boto3/smart_open**. No extra flags needed.
- `uv run olmo-eval --help` — confirms the CLI entrypoint installed (`olmo-eval` is the
  console script defined in `pyproject.toml`). You should see the subcommands, including
  `run`.

> ⚠️ **SSM sets no `HOME`, and the DLAMI root disk is tiny (~19 GB).** `AWS-RunShellScript`
> runs as root but leaves `HOME` unset, so `$HOME/.local/bin` expands to `/.local/bin` and
> `uv` (installed at `/root/.local/bin`) isn't found — hence `export HOME=/root` first.
> The DLAMI's fast local NVMe is mounted at `/opt/dlami/nvme`; because torch + vLLM plus
> model downloads far exceed the root disk, point the caches there via
> `UV_CACHE_DIR=/opt/dlami/nvme/rung1/uv-cache` and `HF_HOME=/opt/dlami/nvme/rung1/hf`
> (and the eval's `-O` output dir), so nothing fills the root disk.

---

## 6. Run the smoke test via SSM

The recommended smoke test — **cheapest thing that works**:

| Choice | Value | Why |
| --- | --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` | ~0.5B params, **public** (no HF token/gating), runs on vLLM, trivially fits an A10G's 24 GB |
| Task | `arc_easy` (built-in) | small grade-school multiple-choice; **logprob-scored**, so no long text generation — fast + cheap |
| Cap | `-o limit=10` | evaluate only 10 instances → seconds of GPU time |

**The exact run command** (note `-o limit=10` comes **after** `-t arc_easy` — overrides
apply to the preceding `-t`):

```bash
uv run olmo-eval run \
  -m Qwen/Qwen2.5-0.5B-Instruct \
  -t arc_easy -o limit=10 \
  -O /opt/dlami/nvme/rung1/results \
  --s3-bucket edullm-adaptive-inference-056956104102 \
  --s3-prefix smoketest \
  --s3-group rung1 \
  --s3-region us-east-1
```

Flag reference (from `src/olmo_eval/cli/run/`):

- `-m/--model` — model name or HF path.
- `-t/--task` — task or suite; `-o/--override key=value` after it sets task config
  (here `limit=10`; there is **no** top-level `--limit` flag — it's a task override).
- `-O/--output-dir` — local output dir.
- `--s3-bucket` / `--s3-prefix` / `--s3-group` — **all three are required together**;
  the CLI errors out if you pass one without the others.
- `--s3-region` — defaults to `us-east-1` (also reads `AWS_REGION`); matches the team region.
- `--num-gpus` — defaults to **1**, so you don't need to pass it on a single-GPU box.

**Run it detached via SSM** so it survives the SSM session closing (the team's
"SSM InProgress hang" gotcha — SSM stays `InProgress` until *all* child processes exit,
so fully detach with `setsid nohup … &` and check the log separately):

```bash
aws ssm send-command \
  --instance-ids i-XXXXXXXXXXXXXXXXX \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["cd /opt/dlami/nvme/rung1/olmo-eval-full && export HOME=/root && export PATH=\"$HOME/.local/bin:$PATH\" && export UV_CACHE_DIR=/opt/dlami/nvme/rung1/uv-cache && export HF_HOME=/opt/dlami/nvme/rung1/hf && mkdir -p \"$UV_CACHE_DIR\" \"$HF_HOME\" && setsid nohup uv run olmo-eval run -m Qwen/Qwen2.5-0.5B-Instruct -t arc_easy -o limit=10 -O /opt/dlami/nvme/rung1/results --s3-bucket edullm-adaptive-inference-056956104102 --s3-prefix smoketest --s3-group rung1 --s3-region us-east-1 > /opt/dlami/nvme/rung1/smoketest.log 2>&1 &"]' \
  --profile sbsandbox --region us-east-1
```

**Watch the log** (poll this; do NOT wait on the SSM command status):

```bash
aws ssm send-command \
  --instance-ids i-XXXXXXXXXXXXXXXXX \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["tail -50 /opt/dlami/nvme/rung1/smoketest.log"]' \
  --profile sbsandbox --region us-east-1
```

**Expected runtime:** first run spends a minute or two downloading the model + dataset
and starting the vLLM engine, then scores 10 instances in seconds. Total: **a few
minutes**.

**What success looks like in the log:** a printed **Run Configuration** panel, a vLLM
startup banner, a line like `S3 uploads enabled: s3://edullm-adaptive-inference-056956104102/smoketest/rung1/...`,
and at the end an `arc_easy` accuracy number with no traceback. If the process exits 0
and you saw the "S3 uploads enabled" line, move to verification.

> The tiny model + `arc_easy` are **not gated**, so you do **not** need the team's
> `hf-token` Secrets Manager secret for this smoke test. (You'd only need it later for
> gated models like meta-llama/google/mistralai.)

---

## 7. Verify results in S3 (and sync back locally)

`olmo-eval` writes to this layout (from `S3Config`):

```
s3://{bucket}/{prefix}/{group}/{model}_{model_hash_last6}/{experiment_id}/
    ├── metrics.json                       ← the scores (your proof)
    ├── predictions/{task}-predictions.jsonl
    └── requests/{task}-requests.jsonl
```

So for this run, look under
`s3://edullm-adaptive-inference-056956104102/smoketest/rung1/`.

```bash
# List everything under the smoke-test group (expect a Qwen2.5-0.5B-Instruct_<hash>/… tree)
aws s3 ls s3://edullm-adaptive-inference-056956104102/smoketest/rung1/ \
  --recursive --profile sbsandbox --region us-east-1

# Pull the results down to your laptop to eyeball metrics.json
aws s3 sync s3://edullm-adaptive-inference-056956104102/smoketest/rung1/ \
  ./rung1-results --profile sbsandbox --region us-east-1

cat ./rung1-results/*/*/metrics.json
```

**✅ If you can see `metrics.json` with an `arc_easy` accuracy value, Rung 1 passes.**
The exact accuracy of a 0.5B model on 10 questions is irrelevant — the point is that the
end-to-end path (AWS GPU → olmo-eval → S3) works.

---

## 8. TEARDOWN (do this even though the timer exists)

The `shutdown -h +90` + `--instance-initiated-shutdown-behavior terminate` will
self-terminate the box, **but never rely on the timer alone.** Terminate explicitly the
moment you've got your results:

```bash
# Terminate (fully delete, stops all charges).
aws ec2 terminate-instances --instance-ids i-XXXXXXXXXXXXXXXXX \
  --profile sbsandbox --region us-east-1

# Confirm it's gone: state should progress to "terminated".
aws ec2 describe-instances --instance-ids i-XXXXXXXXXXXXXXXXX \
  --query 'Reservations[].Instances[].State.Name' --profile sbsandbox --region us-east-1
```

> ⚠️ **Stop ≠ Terminate.** *Stop* powers the box off but you **keep paying for its disk**
> and it still counts against the ≤3 GPU-instance cap. *Terminate* deletes it and ends
> all charges. For a smoke test, **always terminate.**
>
> ⚠️ Also clean up the staging object if you like:
> `aws s3 rm s3://edullm-adaptive-inference-056956104102/smoketest/staging/olmo-eval.tgz --profile sbsandbox`.

**Cost check.** From the team's cost reference, a single g5/g6-class box for a few minutes
is **well under ~$1** if you terminate promptly (their 56-model, ~2-hour mixed-GPU run
was only ~$4.60). **This smoke test should cost under ~$1** as long as you tear down when
done. If you ever see the instance still running later, terminate it immediately — you're
being billed.

---

## 9. Troubleshooting

| Symptom | Cause & fix |
| --- | --- |
| Metadata / `ami-launch-index` fetches return blank on the node | **IMDSv2 token required.** The DLAMI enforces IMDSv2, so metadata needs a token first: `TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")` then pass `-H "X-aws-ec2-metadata-token: $TOKEN"` on the metadata `curl`. (Not needed for the smoke test itself, but bites any node self-configuration script.) |
| SSM command sits at `InProgress` forever | SSM stays InProgress until **all** child processes exit. You launched a long-running eval — that's why we use `setsid nohup … &`. Don't wait on the SSM status; `tail` the log file instead (Section 6). |
| `--s3-bucket is required` / `--s3-prefix is required` / `--s3-group is required` | You passed some but not all three S3 flags. They are **required together**. Pass all of `--s3-bucket`, `--s3-prefix`, `--s3-group`. |
| `AccessDenied` on S3 PutObject when uploading results | The shared `EswManagedInstance` role may lack write to your exact prefix. **Don't modify the role globally.** Add a *scoped* permission for `…/smoketest/*` — a bucket policy allowing that role `s3:PutObject` on the `smoketest/*` prefix, or a scoped inline policy. Ask the agent (via `sb_aws`) to add it. |
| Out-of-memory (OOM) on model load | Almost impossible with a 0.5B model on a 24 GB A10G, but if you swapped in something bigger: **use a smaller model** — do **not** upsize past `g6.xlarge`. You can also cap context with `-o` overrides, but the right Rung-1 move is a smaller model. |
| vLLM crashes on model load with architecture/config errors | That model's architecture isn't supported by vLLM (team has hit this with `gemma-3-*`, `OpenELM-*`, `mamba-*`). For Rung 1, **stick with `Qwen/Qwen2.5-0.5B-Instruct`**, which vLLM supports. Exotic architectures need an HF-transformers fallback path, out of scope here. |
| `uv: command not found` on the node after install | `uv` installs to `~/.local/bin`. Ensure it's on PATH in the same command: `export PATH="$HOME/.local/bin:$PATH"`. |
| `uv: command not found` / the install command exits **127 immediately** under SSM | **SSM's `AWS-RunShellScript` runs as root but sets no `$HOME`,** so `$HOME/.local/bin` expands to `/.local/bin` and `uv` (at `/root/.local/bin`) is never found. **Fix:** `export HOME=/root` **before** any `$HOME`/`uv` use, then `export PATH="$HOME/.local/bin:$PATH"`. |
| `No space left on device` during `uv sync` (installing torch/vLLM) | The DLAMI root disk is only ~19 GB, but `uv` caches to `$HOME/.cache/uv` and HF models download to `$HOME/.cache/huggingface` — both on the root disk. **Fix:** put the caches on the big NVMe scratch: `export UV_CACHE_DIR=/opt/dlami/nvme/rung1/uv-cache` and `export HF_HOME=/opt/dlami/nvme/rung1/hf` (`mkdir -p` both) before installing/running. |
| Instance won't appear in SSM (`describe-instance-information` empty) | Give it 1–2 min after launch; confirm the instance profile is `EswManagedInstance` (SSM needs the role's SSM permissions). If still absent, the box may be in a subnet without SSM egress — rotate subnet and relaunch. |
| Launch fails: insufficient capacity | Rotate subnet/zone and retry; `us-east-1b` is the reliable default. |
| `whoami`/`accounts` fail or "Refresh token rejected" | Broker session expired/rotated. Re-run `sb-aws-creds login`, then restart Cursor so the `sb_aws` MCP reconnects. |

---

## 10. Quick reference — what to change per run

| Placeholder | Fill with |
| --- | --- |
| `i-XXXXXXXXXXXXXXXXX` | the InstanceId returned by `run-instances` (Section 4) |
| `<PRESIGNED_URL>` | the `https://…` URL from `aws s3 presign` (Section 5) |
| `N` in `shutdown -h +N` | minutes until auto-terminate (use `90` for this smoke test) |
| S3 smoke-test prefix | `smoketest` / group `rung1` → `s3://edullm-adaptive-inference-056956104102/smoketest/rung1/` |

**Fixed team values (don't change):** account `sbsandbox` (`056956104102`), region
`us-east-1`, bucket `edullm-adaptive-inference-056956104102`, AMI `ami-0b6f2229ad14c9323`,
SG `sg-087218d8c87aa8576`, subnet `subnet-0a4235fb98b63930f`, instance profile
`EswManagedInstance`. Instance type: **`g5.xlarge`** for smoke tests (never above
`g6.xlarge`).

**Recommended smoke-test choice, restated:**
`uv run olmo-eval run -m Qwen/Qwen2.5-0.5B-Instruct -t arc_easy -o limit=10 -O /opt/dlami/nvme/rung1/results --s3-bucket edullm-adaptive-inference-056956104102 --s3-prefix smoketest --s3-group rung1 --s3-region us-east-1`

A copy-pasteable on-node helper that does the install + run is provided alongside this
file: **`rung1_smoketest.sh`**.
