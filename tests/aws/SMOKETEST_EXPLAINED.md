# SMOKETEST_EXPLAINED — Understanding the Rung 1 AWS Smoke Test

*A plain-English companion to [`RUNBOOK_rung1_smoketest.md`](./RUNBOOK_rung1_smoketest.md).
The runbook tells you **what commands to type**. This document explains **what each of
those commands actually means and why we do it**, assuming you know Python and ML but
have never touched AWS.*

> **Read this first, run the runbook second.** Nothing here launches anything or spends
> money. It's the mental model you want in your head before you follow the runbook.

---

## 1. What is this test and why do we run it?

### The problem: we're leaving Beaker

Your team runs model evaluations today through **Beaker**, AI2's internal job scheduler.
Beaker is like *borrowing a warm desk in a big shared office*: the GPU machines are
already powered on, Beaker just finds you a free one, sits your job down, hands you a
shared network drive (Weka), and quietly injects your cloud credentials so your job can
read and write files. You don't have Beaker access, so you want to run the same
`olmo-eval` framework on **AWS** (Amazon's cloud) instead.

The catch: AWS gives you none of that comfort for free. There's no warm machine waiting,
no shared drive, and no auto-injected credentials. You have to (a) turn a machine on
yourself, (b) copy your code and model onto it over the network, (c) run the eval, (d)
save the results somewhere durable, and (e) turn the machine off so you stop paying.
That's a lot of new moving parts, and any one of them can be subtly broken.

### What "smoke test" means

In engineering, a **smoke test** is the smallest possible end-to-end check that a system
powers on without bursting into smoke. It doesn't try to be thorough or fast or
realistic — it just proves the wires are connected. (The name comes from hardware: plug
it in, and if no smoke comes out, you can start real testing.)

**Rung 1** is that smoke test for the "run olmo-eval on AWS instead of Beaker" effort. It
is deliberately the cheapest, tiniest thing that can possibly exercise the whole path:

- a **tiny model** — `Qwen/Qwen2.5-0.5B-Instruct` (~0.5 billion parameters, public, no
  access token needed),
- a **tiny standard task** — `arc_easy` (grade-school multiple-choice questions),
- **capped to 10 questions** (`limit=10`), so the actual evaluation takes *seconds*.

The model's accuracy is irrelevant. A 0.5B model answering 10 questions will score
whatever it scores — we don't care. What we care about is that a results file
(`metrics.json`) appears in the team's cloud storage at the end. If it does, the AWS path
works, and it's now safe to invest in the bigger effort (evaluating real training
checkpoints). This "prove the pipe before building the factory" is called **de-risking**.

### What success looks like

You run one command to list the team's cloud folder, and you see a file named
`metrics.json` with an `arc_easy` accuracy number in it, under a path ending in
`smoketest/rung1/…`. That's the entire definition of done. (Section 6 below is the exact
success criteria.)

---

## 2. The big picture in plain English

Here is the whole flow as a short story, with no AWS jargon yet:

1. **Get a key to the building.** You authenticate once through a small helper the team
   built, which hands you a temporary pass to their AWS account.
2. **Rent one GPU computer for a few minutes.** You ask AWS to power on a single machine
   that has an NVIDIA GPU. Crucially, you arm a self-destruct timer *at the same moment
   you turn it on* — so even if you walk away and forget it, it deletes itself.
3. **Mail your code to the machine.** There's no shared drive and no way to log in
   directly, so you zip up the `olmo-eval` code, drop the zip in cloud storage, mint a
   temporary download link, and tell the machine (through AWS's remote-control channel)
   to download and unzip it.
4. **Install dependencies.** You tell the machine to install olmo-eval's Python
   dependencies (vLLM, torch, transformers, etc.) using `uv`.
5. **Run the eval.** You tell the machine to run `olmo-eval run` on the tiny model + tiny
   task. When it finishes, olmo-eval itself uploads the results to the team's cloud
   storage.
6. **Check the results, then tear it down.** You list the cloud folder to confirm
   `metrics.json` is there, then explicitly delete the machine (belt-and-suspenders — the
   self-destruct timer would have done it anyway).

### ASCII diagram of the flow

```
  YOU (your laptop + Cursor agent)
      │
      │  (1) authenticate via the sb-aws-creds broker  →  temporary AWS access
      ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  AWS account "sbsandbox" (us-east-1)                                 │
  │                                                                     │
  │   (2) run-instances: launch ONE g5.xlarge GPU box                   │
  │        • boots from team DLAMI (drivers+CUDA preinstalled)          │
  │        • user-data arms `shutdown -h +90`  ⏱  self-destruct timer    │
  │        • --instance-initiated-shutdown-behavior terminate            │
  │              │                                                       │
  │              ▼                                                       │
  │      ┌──────────────────────────────┐                               │
  │      │  the GPU instance (the node) │                               │
  │      │                              │                               │
  │  (3) │  ← S3 + presigned URL + SSM: │   ← tar+upload code to S3,     │
  │      │    curl + untar olmo-eval    │     presign a link, SSM curl   │
  │  (4) │  uv sync --frozen (vLLM…)    │                               │
  │  (5) │  olmo-eval run … --s3-*  ────┼──►  writes metrics.json to S3  │
  │      └──────────────────────────────┘                               │
  │                                          s3://…/smoketest/rung1/…    │
  │   (6) terminate-instances  (timer would also self-terminate)        │
  └─────────────────────────────────────────────────────────────────────┘
      │
      ▼
  (6) aws s3 ls / sync  →  you see metrics.json locally  ✅ Rung 1 passes
```

Everything you do to the machine — install, run, check logs — happens by **sending it
commands through AWS**, not by logging in. That's the single biggest mental shift from a
normal Linux server, and it's explained in the SSM entry below.

---

## 3. The AWS building blocks, explained

This is a glossary. Each entry says what the thing is in plain terms and **why this
smoke test needs it**. The concrete values (bucket name, AMI id, etc.) come straight from
the runbook and are the team's real, already-provisioned resources — you reuse them, you
don't create them.

### AWS account & region
- **AWS account** — a company's isolated cloud "workspace" where all its resources live.
  The team's is nicknamed **`sbsandbox`** (numeric id `056956104102`). You don't own it;
  you're a guest with a temporary pass.
- **Region** — which physical data-center location your resources live in. The team uses
  **`us-east-1`** (Northern Virginia). All resources here (storage, machines) must be in
  the same region to talk to each other cheaply, so you always pass `--region us-east-1`.

### The credential broker (`sb-aws-creds`)
- **Credential broker** — normally AWS makes you copy-paste temporary secret keys into a
  config file every few hours. The team replaced that chore with **`sb-aws-creds`**, a
  small command-line tool: you log in once with your Google account, and it hands your
  tools short-lived AWS keys on demand (valid ~30 days before you re-login). *Why we need
  it:* it's how you get any access to `sbsandbox` at all, safely, without ever holding a
  raw secret.
- **MCP server** — MCP is the plug-in protocol Cursor uses to give the agent new tools.
  `sb-aws-creds` can run as an MCP server named **`sb_aws`**, which lets your Cursor agent
  run AWS commands for you. *Why we need it:* it's the beginner-friendly path — you can
  ask the agent "launch the instance" instead of memorizing AWS CLI syntax.

### S3 (storage)
- **S3 (Simple Storage Service)** — Amazon's cloud file storage. Think of it as *a giant
  shared online folder*. Files ("objects") live at addresses like
  `s3://bucket-name/some/path/file.json`. It's durable (files don't vanish when a machine
  is deleted) and every machine in the account can reach it. *Why we need it:* it's where
  we stage the code onto the machine, and where the eval's results land so they survive
  after the machine is gone.
- **Bucket** — the top-level named container in S3 (like a drive letter). The team's is
  **`edullm-adaptive-inference-056956104102`**.

### EC2 (the rented computer)
- **EC2 (Elastic Compute Cloud)** — a computer you rent by the minute in Amazon's data
  center. *You pay for every minute it's switched on*, so the golden rule is: turn it off
  the moment you're done.
- **Instance** — one running EC2 computer. Each has an id like `i-0abc123…`.
- **GPU instance** — an EC2 instance that also has an NVIDIA GPU attached (needed to run
  models fast). For this smoke test we use **`g5.xlarge`** (one NVIDIA A10G GPU, 24 GB of
  GPU memory). A 0.5B model is tiny, so 24 GB is enormous headroom. *Why this exact type:*
  the team designates `g5.xlarge` as the "smoke tests only" GPU (see the GPU cap in
  Section 5).

### AMI / DLAMI (the pre-baked disk image)
- **AMI (Amazon Machine Image)** — the operating-system-plus-software snapshot a new
  instance boots from. Like a factory disk image for a fresh laptop.
- **DLAMI (Deep Learning AMI)** — an AMI that already has the painful stuff installed:
  **NVIDIA GPU drivers and CUDA**. *Why we need it:* installing GPU drivers by hand is a
  notorious time sink; booting from the DLAMI skips it entirely. The team's is
  **`ami-0b6f2229ad14c9323`** (Ubuntu 24.04).

### Networking (security group & subnet)
- **Security group** — the instance's firewall: which network traffic is allowed in/out.
  The team's is **`sg-087218d8c87aa8576`**. You reuse it as-is.
- **Subnet** — which slice of the team's private network (and which availability zone, a
  distinct building within the region) the machine sits in. The team's is
  **`subnet-0a4235fb98b63930f`** in zone `us-east-1b`. *Why it matters here:* if that
  zone is temporarily out of GPU capacity, the launch fails and you rotate to another
  subnet/zone and retry.

### IAM role / instance profile (the machine's ID badge)
- **IAM role / instance profile** — a set of permissions *attached to the machine itself*,
  like an ID badge clipped to it. Because the badge says "this machine may read/write our
  S3 bucket," the machine can upload results **without anyone pasting secret keys onto
  it**. The team's badge is **`EswManagedInstance`**. *Why we need it:* it's how
  `olmo-eval` uploads to S3 with zero credentials in the code. ⚠️ This badge is *shared
  with other projects — never modify it globally*; if a permission is missing, add a
  narrow, prefix-scoped policy instead.

### SSM (remote control, instead of SSH)
- **SSH** — the traditional way to log into a remote server (open port 22, use a `.pem`
  key). We do **not** use it here.
- **SSM (AWS Systems Manager)** — a remote-control service: you send shell commands to a
  machine *through AWS itself* and get the output back, with no open SSH port and no key
  file. Think of it as *texting a command to the box and reading its reply*. *Why we need
  it:* the team's whole access pattern is SSM, so there's no key pair or port-22 firewall
  rule to manage. Every "do X on the machine" step is an `aws ssm send-command`.

### Presigned URL (a self-expiring download link)
- **Presigned URL** — a temporary, no-login web link to one private S3 file. Anyone with
  the link can download that one file until the link expires (we use 1 hour). *Why we need
  it:* the machine can't be `scp`'d to (no SSH) and you can't paste a whole code repo into
  an SSM command, so instead you upload the zipped repo to S3, mint a presigned link, and
  tell the machine to `curl` that link. Like *mailing the box a self-destructing download
  link.*

### user-data (the boot script)
- **user-data** — a script AWS runs automatically the first time an instance boots. *Why
  we need it:* it's where we arm the self-destruct timer, so the safety net is in place
  from the very first second the machine is alive.

### Self-terminating instance (`shutdown -h +N`)
- **Self-terminating instance** — a machine set up to turn *itself* off (and delete
  itself) after a fixed time, with no human involved. We put `shutdown -h +90` in the boot
  script (meaning "halt 90 minutes from now") and launch with a setting that makes "halt"
  mean "delete-and-stop-billing." Think of it as *an oven timer that not only turns the
  oven off but also hauls it to the curb.* *Why we need it:* it guarantees you can't leave
  a GPU running by accident and rack up charges.

### Secrets Manager (mentioned, not used here)
- **Secrets Manager** — a vault for passwords/tokens. The team keeps a HuggingFace token
  (`hf-token`) here for downloading *gated* models. *Why it's NOT used in this smoke
  test:* Qwen2.5-0.5B-Instruct and `arc_easy` are public/ungated, so no token is needed.
  You'd only touch this later for gated models (meta-llama, google, mistralai).

---

## 4. Step-by-step: what each command actually does

This mirrors the runbook's sections, but focuses on *intent, inputs/outputs, and what can
go wrong* rather than syntax. Refer to the runbook for the exact commands to copy.

### Step 0 — Get access (runbook §1)
- **Intent:** obtain temporary permission to act in the `sbsandbox` account.
- **What happens:** you install `sb-aws-creds`, run `sb-aws-creds login` (opens a browser,
  you log in with your `@alphaaiengineering.com` Google account), and register it as the
  `sb_aws` MCP server in Cursor so the agent can make AWS calls. Verify with `accounts`
  (should list `sbsandbox (ready)`) and `whoami` (should show your email + token expiry).
- **What could go wrong:** if `whoami` fails with "Refresh token rejected," your ~30-day
  session expired — re-run `sb-aws-creds login` and restart Cursor.

### Step 1 — Confirm the team defaults (runbook §2)
- **Intent:** reuse the already-created bucket, AMI, security group, subnet, and instance
  profile. **You create nothing new here.** These are the values listed in Section 3
  above.

### Step 2 — The mandatory pre-launch checklist (runbook §3)
This is the team's hard safety rule. **Before every launch, all three must be true:**
1. **Instance type is `g6.xlarge` or cheaper** — for a smoke test, `g5.xlarge`. Never
   bigger. If a model needs more than 24 GB, you *drop the model, not upsize the machine*.
2. **At most 2 GPU instances already running** (so this one won't exceed 3 total). You
   check by listing running instances tagged `edullm-gpu-worker`.
3. **The boot script contains a `shutdown -h +N` line** sized for this run (here `N=90`
   minutes).
- **What could go wrong:** skipping this is how people leak money. Don't launch if any
  check fails.

### Step 3 — Launch one instance (runbook §4)
- **Intent:** power on a single `g5.xlarge` with the self-destruct timer already armed.
- **Inputs:** the AMI, instance type, count=1, the instance profile (`EswManagedInstance`),
  security group, subnet, the `--instance-initiated-shutdown-behavior terminate` setting,
  and the `user-data` boot script (which contains `shutdown -h +90`). It's also tagged
  `Name=edullm-gpu-worker` so the checklist's counting query can find it.
- **Outputs:** an **InstanceId** like `i-0abc…`. You'll paste this into every later
  command (the runbook writes it as `i-XXXXXXXXXXXXXXXXX`, a placeholder for *your* id).
- **The two flags that matter most:**
  - `--instance-initiated-shutdown-behavior terminate` turns the boot script's "halt"
    into "delete + stop billing." Without it, the timer would only *stop* the machine (you'd
    keep paying for its disk).
  - `--iam-instance-profile Name=EswManagedInstance` clips on the S3 badge so the eval can
    upload results with no keys.
- **Then wait ~1–2 minutes** and confirm two things: the instance state is `running`, and
  SSM shows it `Online` (only then can you send it commands).
- **What could go wrong:**
  - *Insufficient capacity* → the zone is temporarily full; rotate subnet/zone and retry.
  - *Instance never appears in SSM* → give it a minute or two; if still absent, the
    instance profile or subnet may lack SSM reachability — relaunch.

### Step 4 — Stage the code onto the node (runbook §5)
- **Intent:** get the `olmo-eval` code onto a machine you can't SSH into.
- **The pattern:** tar the repo → `aws s3 cp` it to the bucket under a `smoketest/staging/`
  path → `aws s3 presign` a 1-hour download link → `aws ssm send-command` telling the node
  to `curl` the link and untar it into the fast local disk at `/opt/dlami/nvme/rung1`.
- **Important detail:** do the `presign` step through the `sb_aws` broker, because the
  broker's credentials outlive the node's own short-lived token.
- **Simpler alternative the runbook notes:** if the node has GitHub access, you can just
  `git clone https://github.com/allenai/olmo-eval` on the node instead. The S3+presign
  route is the team-blessed default that always works, so it's shown first.
- **What could go wrong:** a stale/expired presigned URL (mint a fresh one, 1-hour
  window); or writing to the root disk instead of `/opt/dlami/nvme` (use the NVMe path so
  model downloads don't fill the small root disk).

### Step 5 — Install dependencies with `uv` (runbook §5)
- **Intent:** install olmo-eval's Python dependencies reproducibly.
- **What happens:** install `uv` (a fast Python package manager), `uv python install 3.12`
  (olmo-eval needs Python 3.12+), then `uv sync --frozen` to install exactly the versions
  pinned in the checked-in `uv.lock`. The project's *default dependency groups* pull in
  **vLLM** (the GPU inference engine, Linux/CUDA-only) plus torch/transformers, and the
  `s3` extra (`boto3` + `smart_open`) that enables S3 uploads. No extra flags needed.
- **Confirm** with `uv run olmo-eval --help`, which should list subcommands including
  `run`.
- **What could go wrong:** `uv: command not found` → it installs to `~/.local/bin`, so
  make sure that's on `PATH` in the same command. But note SSM runs as root with **no
  `HOME` set**, so `$HOME/.local/bin` expands to `/.local/bin` and misses it — you must
  `export HOME=/root` *first*, then `export PATH="$HOME/.local/bin:$PATH"`. Also, torch +
  vLLM overflow the ~19 GB root disk, so point the caches at the NVMe with
  `export UV_CACHE_DIR=/opt/dlami/nvme/rung1/uv-cache` and
  `export HF_HOME=/opt/dlami/nvme/rung1/hf` or `uv sync` dies with `No space left on device`.

### Step 6 — Run the eval (runbook §6)
- **Intent:** run the tiny eval and have olmo-eval upload results to S3.
- **The command** (conceptually):
  `olmo-eval run -m Qwen/Qwen2.5-0.5B-Instruct -t arc_easy -o limit=10 -O <local out>
  --s3-bucket … --s3-prefix smoketest --s3-group rung1 --s3-region us-east-1`.
- **Key correctness details:**
  - `-o limit=10` must come **right after** `-t arc_easy` — the override applies to the
    preceding task. There is no top-level `--limit` flag.
  - The three S3 flags (`--s3-bucket`, `--s3-prefix`, `--s3-group`) are **required
    together**; passing one without the others errors out.
  - `--num-gpus` defaults to 1, so you don't pass it on a single-GPU box.
- **Run it detached** via `setsid nohup … &`. This is because of the team's "SSM
  InProgress hang" gotcha: an SSM command stays `InProgress` until *all* child processes
  exit, so a long-running eval would make SSM appear to hang. Detaching frees the SSM
  command immediately; you then watch progress by `tail`-ing the log file with a separate
  SSM command — **do not wait on the SSM command status.**
- **Expected runtime:** a minute or two to download the model + dataset and start vLLM,
  then a few seconds to score 10 questions. A few minutes total.
- **What success looks like in the log:** a "Run Configuration" panel, a vLLM startup
  banner, a line like `S3 uploads enabled: s3://…/smoketest/rung1/…`, and a final
  `arc_easy` accuracy number with no traceback.
- **What could go wrong:** vLLM crashing on an unsupported architecture (stick to Qwen for
  Rung 1); OOM (near-impossible with a 0.5B model on 24 GB — if it happens you swapped in
  something too big; use a smaller model, don't upsize); `AccessDenied` on S3 upload (the
  shared role may lack write to your exact prefix — add a *scoped* policy for
  `smoketest/*`, never modify the role globally).

### Step 7 — Verify results in S3 (runbook §7)
- **Intent:** confirm the results landed in cloud storage.
- **olmo-eval's output layout:**
  `s3://{bucket}/{prefix}/{group}/{model}_{hash}/{experiment_id}/` containing
  `metrics.json` (the scores), plus `predictions/` and `requests/` folders.
- **What you do:** `aws s3 ls …/smoketest/rung1/ --recursive` to see the tree, then
  `aws s3 sync` it to your laptop and `cat` the `metrics.json`.

### Step 8 — Teardown (runbook §8)
- **Intent:** delete the machine so charges stop, even though the timer would eventually
  do it.
- **What you do:** `aws ec2 terminate-instances --instance-ids i-…`, then confirm the
  state reaches `terminated`. Optionally delete the staging tarball from S3.
- ⚠️ **Stop ≠ Terminate.** *Stop* powers off but you keep paying for the disk and it still
  counts against the 3-instance cap. *Terminate* deletes it and ends all charges. For a
  smoke test, **always terminate.**

---

## 5. Safety, cost, and cleanup

Three independent safeguards keep this test cheap and prevent runaway spend:

1. **The self-destruct timer.** `shutdown -h +90` in the boot script + the launch flag
   `--instance-initiated-shutdown-behavior terminate` means the machine deletes itself 90
   minutes after boot no matter what — even if your laptop dies or you forget it. 90
   minutes is a generous safety net; the eval itself only needs a few minutes.
2. **The explicit terminate at the end.** You don't rely on the timer alone. The moment
   you've confirmed results, you run `terminate-instances`. Belt and suspenders.
3. **The pre-launch discipline (team hard rules):**
   - **GPU cap:** never launch anything bigger than `g6.xlarge` (24 GB L4). Smoke tests
     use `g5.xlarge` (24 GB A10G). If a model doesn't fit 24 GB, you drop the model — you
     do **not** upsize the instance.
   - **≤3 running GPU instances at once:** check the running count (instances tagged
     `edullm-gpu-worker`) before every launch, and don't launch if 3 are already up.

### Why cost is under ~$1
A single g5/g6-class GPU box costs on the order of a dollar-ish *per hour*, and you use it
for only a few minutes before terminating. For scale, the team's real 56-model, ~2-hour
mixed-GPU back-fill cost about **$4.60 total** — so a few minutes on one small box is well
under **$1**. The only way this test costs real money is if a machine is left running, and
the three safeguards above exist precisely to prevent that. *If you ever notice the
instance still running later, terminate it immediately — you're being billed.*

---

## 6. How we know it worked

Rung 1 **passes** when all of these are true:

- The `olmo-eval run` process exited cleanly (exit code 0), with **no Python traceback**
  in `smoketest.log`.
- The log contained the `S3 uploads enabled: s3://…/smoketest/rung1/…` line.
- `aws s3 ls s3://edullm-adaptive-inference-056956104102/smoketest/rung1/ --recursive`
  shows a tree under `Qwen2.5-0.5B-Instruct_<hash>/<experiment_id>/` containing:
  - **`metrics.json`** ← the actual proof (has an `arc_easy` accuracy value),
  - `predictions/arc_easy-predictions.jsonl`,
  - `requests/arc_easy-requests.jsonl`.

**The accuracy value itself does not matter.** A 0.5B model on 10 questions can score
anything; we're not measuring model quality. The presence of `metrics.json` in S3 is the
whole point: it proves the end-to-end path **AWS GPU → olmo-eval → S3** works without
Beaker.

---

## 7. What this does and does NOT prove

**It proves:**
- You can get access to `sbsandbox`, launch a self-terminating GPU instance, stage code
  onto it over SSM, install olmo-eval with `uv`, run a built-in eval on a GPU, and have
  olmo-eval upload results to the team S3 bucket — all without Beaker and for under ~$1.
- The plumbing (broker → EC2 → SSM → S3 → self-terminate) is sound.

**It does NOT prove (explicitly out of scope for Rung 1):**
- **The Qwen LLM-judge path.** Some real evals are model-graded (a ~9B Qwen judge scores a
  checkpoint's answers). Fitting two models under the 24 GB L4 cap is an open design
  question (sequential stages vs. a separate judge instance) — untouched here.
- **Multi-checkpoint back-fill / the shard-index fleet.** Scoring a whole folder of past
  checkpoints with several parallel workers (each reading its `ami-launch-index` via
  IMDSv2) is a separate, larger pattern — not exercised by a single-box smoke test.
- **Real `g6.xlarge` runs on actual training checkpoints.** Rung 1 uses a throwaway public
  model, not a real checkpoint pulled from S3, and uses `g5.xlarge`, not the `g6.xlarge`
  used for real runs.
- **Gated models / the HF-token secret.** Qwen2.5-0.5B is ungated, so Secrets Manager
  isn't touched. Gated models (meta-llama, google, mistralai) would add that step.

In short: Rung 1 de-risks the *pipe*, not the *factory*. Passing it is the green light to
build the launcher/hook and tackle those larger pieces.

---

## 8. FAQ / common confusions

**Q: Why not just SSH into the machine like a normal server?**
The team's access model is SSM, not SSH. SSM lets you run commands through AWS itself with
no open port 22 and no `.pem` key to manage or leak. It's more secure and there's nothing
to set up. So instead of `ssh` + `scp`, you use `aws ssm send-command` and the
S3-presigned-URL staging trick.

**Q: What's the difference between this and Beaker?**
Beaker is a *warm shared office*: machines already on, a shared drive, and credentials
auto-injected. AWS gives you none of that — you turn a machine on yourself, copy code over
the network, and turn it off yourself. Rung 1 is proving you can reproduce Beaker's
outcome (GPU → run eval → results saved) using raw AWS building blocks instead.

**Q: Do I need to babysit it while it runs?**
No. You launch it, kick off the eval detached (so it survives your session closing), and
periodically `tail` the log. The self-destruct timer means even total neglect can't run up
a bill. That said, you *should* explicitly terminate as soon as results land — it's good
hygiene and frees a slot against the ≤3-instance cap.

**Q: What if it doesn't shut down?**
Two things have to both fail for that to happen: the `shutdown -h +90` timer *and* your
explicit `terminate-instances`. If you ever see the instance still running later, just run
`terminate-instances` again — terminate is idempotent and immediately stops billing.
Remember *Stop ≠ Terminate*: terminate is the one that deletes the machine and ends
charges.

**Q: Why a 0.5B model and only 10 questions — isn't that useless?**
That's the point of a smoke test. Small + fast + cheap means you can iterate on the
*plumbing* in minutes for pennies. Real model quality is measured later, once the pipe is
proven.

**Q: Can the Cursor agent just do all of this for me?**
Largely yes — once the `sb_aws` MCP is registered, you can ask the agent to run the AWS
commands (launch, stage, run, verify, terminate) on your behalf through the broker. The
runbook still shows raw `aws …` commands so they're copy-pasteable, but the agent path is
the beginner-friendly one.

**Q: Do I need to request a GPU quota or create an AWS account?**
No. The `sbsandbox` account is already fully provisioned — bucket, GPU image, network,
permissions all exist. You just get in through the broker and reuse the team defaults.

---

## Appendix: fixed values used by this smoke test

All pulled directly from the runbook; these are real team resources, not placeholders:

| Thing | Value |
| --- | --- |
| Account | `sbsandbox` (`056956104102`) |
| Region | `us-east-1` |
| S3 bucket | `edullm-adaptive-inference-056956104102` |
| GPU AMI (DLAMI) | `ami-0b6f2229ad14c9323` (Ubuntu 24.04) |
| Security group | `sg-087218d8c87aa8576` |
| Subnet (zone 1b) | `subnet-0a4235fb98b63930f` |
| Instance profile | `EswManagedInstance` |
| Instance type | `g5.xlarge` (smoke tests only; never above `g6.xlarge`) |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (public, ungated) |
| Task | `arc_easy`, capped `-o limit=10` |
| S3 result path | `s3://edullm-adaptive-inference-056956104102/smoketest/rung1/…` |
| Self-terminate timer | `shutdown -h +90` (+ `--instance-initiated-shutdown-behavior terminate`) |

**Values that are genuinely per-run placeholders (you fill them in):**
- `i-XXXXXXXXXXXXXXXXX` — the InstanceId returned by `run-instances`.
- `<PRESIGNED_URL>` — the `https://…` link from `aws s3 presign` (valid 1 hour).
- `N` in `shutdown -h +N` — minutes until auto-terminate (use `90` for this smoke test).
