# Plan: Run Checkpoint Evals on AWS (instead of Beaker)

*A plain-English plan for someone new to AWS. Every AWS term is explained with a
simple analogy the first time it shows up. No code changes are proposed here — this
is a planning document.*

---

## 0. Reconciliation with the team's AWS setup (read this first)

**This plan has changed direction.** An earlier draft proposed building everything on
**AWS Batch** (a "temp agency for computers" that manages job queues and machine pools
for you). We are **no longer doing that.** The organization already has a working AWS
pipeline built on plain **EC2 + SSM**, and we are aligning with it rather than diverging.
This is non-negotiable: the team set this pipeline up, documented it, and runs real work
on it, so we reuse it exactly.

Concretely, this plan now targets the team's **`sbsandbox`** account and its established
**EC2 + SSM** pattern, exactly as described in the team's real runbook
(`AWS run book.md`) and the already-rewritten smoke-test guide
(`eduLLM-Evals/scripts/aws/RUNBOOK_rung1_smoketest.md`). You get access through the
team's **`sb-aws-creds` credential broker** (*broker = a small helper that hands out
short-lived AWS keys after a Google login, so you never copy-paste secrets*). The
account, S3 bucket, IAM role, GPU machine image, and HF-token secret **already exist** —
so most of the one-time AWS setup is already done, and the new work is small (a hook plus
an on-node run/staging script).

**What this means for the rest of the document:** wherever the old plan said "Batch job",
"job queue", "compute environment", "warm pool", "array job", or "container image on
ECR", read instead: **a self-terminating EC2 GPU instance launched with `run-instances`,
driven over SSM, staging its code from S3 via a presigned URL.** The sections below are
written that way.

---

## 1. Goal

We want to run `olmo-eval` on model **checkpoints** (the snapshots training saves as it
learns) using **AWS**, because we do not have access to AI2's "Beaker" system. The end
result should feel the same as today: point the evaluator at a checkpoint, run some
tasks, and get scores back — just powered by the team's AWS pipeline instead of Beaker.

---

## 2. How it works today (Beaker), in plain terms

Today the only way to launch a remote eval is through **Beaker**, AI2's internal job
scheduler. Think of Beaker as **borrowing a warm desk in a big shared office**: the
computers (GPU machines) are already powered on and waiting, Beaker just finds you a
free desk and sits your job down at it. It also automatically hands you a shared filing
cabinet called **Weka** (a shared network drive that every machine can see), and it
quietly slips your AWS keys into the job so it can read and write files in S3.

The catch: when we move to AWS, **we lose that shared office**. AWS does *not* keep
machines warm for free, and there is **no Weka** — so our AWS setup has to (a) get a
machine running itself, and (b) copy the code, dependencies, and checkpoint onto that
machine over the network, since there's no shared drive to lean on.

The good news: the team already solved this on `sbsandbox`. A machine boots from a
**pre-baked GPU image**, we **stage code from S3**, we drive it with **SSM**, and the
machine **turns itself off when done**. The rest of this plan just applies that same
recipe to checkpoint evals.

---

## 3. The AWS building blocks, explained simply

Here is every AWS piece we'll use, each with a one-line analogy. These are exactly the
pieces the team runbook uses — nothing Batch-specific.

- **S3** — Amazon's cloud storage. Think **a giant shared folder / online storage
  locker**. Files live at addresses like `s3://my-bucket/some/path`. This is where
  checkpoints come from and where results go. The team bucket is
  `edullm-adaptive-inference-056956104102`.
- **EC2 instance** — **a computer you rent by the minute** in Amazon's data center. You
  pay for every minute it's switched on, so the golden rule is *turn it off the moment
  you're done*.
- **GPU instance** — an EC2 instance that **also has a GPU** (the special chip needed to
  run large models quickly). Same idea, just a beefier rented computer. The team's
  workhorse is **`g6.xlarge`** (one NVIDIA L4, 24 GB) — see the hard ceiling in §4.
- **DLAMI (Deep Learning AMI)** — **a pre-baked machine image**: a disk snapshot the
  instance boots from that **already has the NVIDIA GPU drivers and CUDA installed.**
  Think of it as **a laptop that arrives with all the hard driver setup already done** —
  you skip the painful "install GPU drivers" step entirely. The team's is
  `ami-0b6f2229ad14c9323` (Ubuntu 24.04).
- **SSM (AWS Systems Manager)** — **a remote-control service.** It lets you **run shell
  commands on a machine through AWS itself, with no SSH and no password/key.** Think of
  it as **texting a command to the box and getting the output back** — no open port 22,
  no `.pem` file to manage. This is how we install and launch the eval on the node.
- **S3 + presigned URL staging** — how we get our code onto a box that has no shared
  drive and no SSH. We **tar the code, upload it to S3, mint a temporary no-login
  download link (a *presigned URL* = a private S3 file with a short-lived "anyone with
  this link can download" pass), then tell the node over SSM to `curl` + untar it.**
  Think of it as **mailing the box a self-expiring download link** instead of physically
  copying files over.
- **Secrets Manager** — **a vault for passwords.** We keep the HuggingFace token
  (`HF_TOKEN`, needed for gated models) here, and the box fetches it at runtime with its
  badge — the secret **never** appears in code or command history. Think **a key safe the
  box can open because of who it is, not because you handed it the key.**
- **Self-terminating instance / `shutdown -h +N`** — **the box turns itself off when
  done.** We put a line like `shutdown -h +90` in the boot script (halt 90 minutes from
  now) and launch with "on shutdown, terminate", so the machine **deletes itself and
  stops billing** without anyone babysitting it. Think of it as **an oven timer that not
  only turns the oven off but takes it to the curb.** This is what replaces the old idea
  of a "warm pool" — instead of keeping a machine on, we make each machine dispose of
  itself.
- **IMDSv2 shard index (`ami-launch-index`)** — **how one member of a fleet learns which
  slice of the work is its own.** When you launch N identical machines at once, AWS gives
  each one a number (0, 1, 2, …) it can read from its own metadata. Think of it as
  **numbered name-tags handed out at the door** so worker #3 knows to grab checkpoint #3.
  On the team's DLAMI you must read it through **IMDSv2** (a token-first metadata service —
  you fetch a short-lived token, then use it to read the number; without the token the
  read comes back blank). This is what replaces the old Batch "array job" idea.
- **Instance profile / IAM role** — **the box's S3 keycard.** It's a badge attached to the
  machine that says "allowed to read and write our S3 bucket (and read the HF secret),"
  so the box can fetch checkpoints and save results **without us pasting secret passwords
  onto it.** The team's badge is `EswManagedInstance`. It's **shared with other projects —
  never modify the role globally**; if a permission is missing, add a *scoped* policy for
  your prefix only.

---

## 4. The hard GPU ceiling (a first-class constraint)

**Never launch a GPU instance bigger than `g6.xlarge` (one NVIDIA L4, 24 GB VRAM).** This
is the team's hard rule, reproduced verbatim from the runbook, and it shapes every design
choice below:

- **`g6.xlarge`** (L4, 24 GB) — the maximum allowed for all real eval runs.
- **`g5.xlarge`** (A10G, 24 GB) — smoke tests only.
- **`g6e.xlarge`, `p6-b200.*`, and anything larger — forbidden.** There is **no** A100 /
  H100 / L40S option in this plan. If a model needs more than 24 GB, you **skip it or
  restructure the run — you do NOT upsize the instance.**

Two more standing rules from the team's mandatory pre-launch checklist:

- **At most 3 GPU instances running at once** (tag `Name=edullm-gpu-worker`; check the
  running count before every launch).
- **Every launch's `user-data` must include a `shutdown -h +N` line**, with `N` sized for
  *that* run — never a fixed default.

Everywhere this plan mentions a GPU box, assume `g6.xlarge` with these three rules
applied.

---

## 5. End-to-end flow

```
  [ Training run ]
        |
        |  (1) saves a checkpoint to S3
        v
  s3://<bucket>/checkpoints/step_50000/
        |
        |  (2) a small "hook" fires and calls `aws ec2 run-instances`,
        |      passing the checkpoint's S3 path into a user-data bootstrap
        v
  [ one self-terminating g6.xlarge GPU instance ]   (team AMI/SG/subnet/instance-profile)
        |
        |  (3) user-data arms `shutdown -h +N`; the node stages olmo-eval
        |      (S3 + presigned URL + SSM, or install-on-DLAMI) and installs deps
        v
  [ olmo-eval on the L4 GPU ]
        |  (4) `olmo-eval run -m <checkpoint> -t <tasks> --s3-*`
        |      - downloads the checkpoint from S3
        |      - runs the eval tasks
        |      - writes results back up to S3
        v
        |  (5) `shutdown -h` fires → instance self-terminates, billing stops
        v
  s3://<bucket>/olmo-eval-results/...
```

In one sentence: **training saves a checkpoint to S3 → a hook calls `run-instances` for
one self-terminating `g6.xlarge` → the node stages olmo-eval from S3 over SSM and runs
`olmo-eval run -m <checkpoint> -t <tasks> --s3-*` on the L4 → results are written back to
S3 → the box turns itself off.** No Batch, no queue, no warm pool, no container registry.

---

## 6. Flow 1 — live per-checkpoint eval (the hook)

**Idea in words:** training already writes each checkpoint to S3. Right after it does, we
add a tiny step that **launches one self-terminating GPU box to evaluate that checkpoint,
then goes away.** This is exactly how you get an "on-demand node that creates itself,
runs, and tears down" — the pay-per-use behavior — **without AWS Batch.** The single
`run-instances` call *is* the whole "temp agency": it hires one machine, and the machine
fires itself.

**What the training script adds (illustrative only — a few lines, not production code):**

```bash
# after training saves a checkpoint to s3://<bucket>/checkpoints/step_$STEP/
# user-data: arm the self-terminate timer, then stage + run the eval on the L4.
aws ec2 run-instances \
  --image-id ami-0b6f2229ad14c9323 \
  --instance-type g6.xlarge \
  --count 1 \
  --iam-instance-profile Name=EswManagedInstance \
  --security-group-ids sg-087218d8c87aa8576 \
  --subnet-id subnet-0a4235fb98b63930f \
  --instance-initiated-shutdown-behavior terminate \
  --user-data file://eval_bootstrap.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=edullm-gpu-worker},{Key=Step,Value='"$STEP"'}]' \
  --profile sbsandbox --region us-east-1
```

The `eval_bootstrap.sh` user-data does three things (mirroring the smoke-test runbook):
arm `shutdown -h +N`, stage `olmo-eval` onto the node (S3 + presigned URL + SSM, or a
plain install on the DLAMI), and run the eval:

```bash
#!/bin/bash
shutdown -h +90                         # self-terminate safety net (size N per run)
# ...stage olmo-eval (S3+presigned+SSM or install-on-DLAMI), then:
olmo-eval run -m "$CHECKPOINT_S3" -t "$EVAL_TASKS" \
  --s3-bucket edullm-adaptive-inference-056956104102 \
  --s3-prefix olmo-eval-results --s3-group "step_$STEP" --s3-region us-east-1
```

This mirrors what Beaker does today (get a GPU → run `olmo-eval run -m <checkpoint> -t
<tasks>` → results to S3), except the "get a GPU" step is a single `run-instances` call
and the machine disposes of itself afterward. Keep the hook to a few lines; the real work
lives in the bootstrap/staging script and in `olmo-eval` itself.

> Apply §4 before every launch: `g6.xlarge` only, ≤3 running GPU boxes, `shutdown -h +N`
> present.

---

## 7. Flow 2 — retroactive back-fill (the shard-index fleet)

**Idea in words:** sometimes you already have a folder full of past checkpoints and want
to score all of them. Instead of a Batch "array job", we use the team's **shard-index
fleet pattern**: launch several identical `g6.xlarge` workers at once, and each one reads
its own number from **`ami-launch-index`** (via IMDSv2) to decide **which checkpoints are
mine.**

**How it maps:**

- List the checkpoints you want to evaluate (e.g. `step_1000`, `step_2000`, …).
- Launch a small fleet with **one** `run-instances` call using `--count K`, where
  **`K ≤ 3` (the team's hard GPU cap — never more than 3 GPU workers in parallel).**
- Each worker fetches its shard index over **IMDSv2** and self-assigns a slice of the
  checkpoint list (worker *i* of *K* handles every *K*-th checkpoint), then runs the same
  `olmo-eval run -m <that checkpoint> -t <tasks> --s3-*` for each, and **self-terminates**
  when its slice is done (or when the `shutdown -h +N` timer fires).

Reading the shard index on the DLAMI **requires the IMDSv2 token dance** (fetch a token,
then read `ami-launch-index` with it) — without the token every worker gets a blank index,
does nothing, and shuts down:

```bash
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
SHARD=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/ami-launch-index)
```

This is the **exact** fleet fan-out the team already runs for the 200-model inference
sweep (16 CPU downloaders + ≤3 GPU workers, each keyed off `ami-launch-index`) — we're
reusing a proven approach, just indexing into "which checkpoint" instead of "which model."

> **Cap reminder:** the team runbook is explicit — **never more than 3 `g6.xlarge` GPU
> workers at once.** If you have more checkpoints than 3 workers can chew through quickly,
> let each worker loop over several checkpoints rather than launching more machines.

---

## 8. Two models on one GPU: the Qwen judge under the L4 cap

Some evals are **model-graded**: a checkpoint produces answers, and a separate **judge
model** (here an ~9B **Qwen** grader) scores them. That means two models are in play, and
the 24 GB L4 ceiling (§4) forces a decision the old plan dodged.

**The old plan's "Option A: put both models on one big GPU (A100/H100)" is off the
table** — those instance types are **forbidden** by the team rule. A checkpoint model
**plus** a ~9B judge **cannot co-reside** on a single 24 GB L4: the judge alone (~9B in
half precision ≈ 18 GB before any KV-cache/activation overhead) leaves no room for a
second model. So the judge must be handled one of two ways, both within the `g6.xlarge`
ceiling:

- **Option A (sequential stages on the same box):** run the checkpoint eval first, unload
  it, **then** load the Qwen judge and grade — one model resident at a time. Simplest;
  one instance; slower because you pay two model-load cold starts back-to-back. This
  matches the team's "model-outer" loop discipline (load → run → unload → next) and is the
  recommended default.
- **Option B (separate instance / endpoint for the judge):** run the judge on its **own**
  `g6.xlarge` (or as a standing inference endpoint), and have the checkpoint eval call it.
  More moving parts (a second box that also must obey the ≤3 cap and self-terminate), but
  lets generation and grading overlap. Consider only if sequential grading is too slow.

Either way, **do not** reach for a bigger GPU — that's the one lever the team rule removes.
Open question §11 tracks confirming the judge's real VRAM footprint on an L4.

---

## 9. What we need to build / set up (checklist)

Because we're reusing the team's `sbsandbox` pipeline, **most one-time AWS setup already
exists.** Access is simply the **`sb-aws-creds` broker** (Section 1 of the smoke-test
runbook). Below, "already exists" means the team provisioned it and we must not recreate
or modify it.

**Already done (reuse, don't recreate):**

1. **AWS account + access** — `sbsandbox` (`056956104102`), reached via the `sb-aws-creds`
   broker. ✅ Exists.
2. **S3 bucket** — `edullm-adaptive-inference-056956104102` for checkpoints in and results
   out. ✅ Exists.
3. **IAM role / instance profile** — `EswManagedInstance`, the box's S3 keycard. ✅ Exists.
   *Shared with other projects — never modify globally; add only scoped, prefix-specific
   policies if a write is denied.*
4. **GPU machine image (DLAMI)** — `ami-0b6f2229ad14c9323`, with GPU drivers + CUDA
   pre-installed. ✅ Exists. **This is what removes the need for a container image / ECR.**
5. **Network** — security group `sg-087218d8c87aa8576`, subnet `subnet-0a4235fb98b63930f`
   (`us-east-1b`; rotate on capacity errors). ✅ Exists.
6. **HF token secret** — in Secrets Manager (`hf-token`), fetched at runtime by the box's
   role. ✅ Exists (needed only for gated models).

**New work (small):**

7. **An on-node run/staging script.** The bootstrap that (a) stages `olmo-eval` onto the
   DLAMI via **S3 + presigned URL + SSM** (or a plain on-node install, since the DLAMI
   already has CUDA), (b) installs deps with `uv sync --frozen`, and (c) runs
   `olmo-eval run … --s3-*`. Model it on the smoke-test runbook's staging + run commands.
   - *Status:* New. The smoke-test runbook already proves every step of this on a single
     box — generalize it from one model to "the checkpoint I was handed."
8. **The per-checkpoint hook (Flow 1).** A few lines added to the training script that
   call `run-instances` (with the user-data bootstrap) each time a checkpoint is saved.
   - *Status:* New, small.
9. **The back-fill fan-out (Flow 2).** A launcher that starts ≤3 `g6.xlarge` workers with
   `--count K` and an on-node script that self-assigns checkpoints via the IMDSv2 shard
   index.
   - *Status:* New, but the shard-index pattern is already proven in the team's 200-model
     sweep — reuse it.
10. **(Later) olmo-eval-side wiring.** Any convenience flags/glue inside `olmo-eval` for
    checkpoint paths and S3 grouping. Deferred; not required to get Flow 1 working.

> **What we explicitly do NOT build anymore:** no Batch compute environment, no job queue,
> no job definition, no warm pool, and **no ECR container image.** The DLAMI + S3-staging
> pattern replaces the container entirely — removing the single biggest brand-new artifact
> the old plan required.

---

## 10. Cost & speed notes

- **No warm pool — self-termination gives pay-per-use for free.** The old plan weighed
  "keep a GPU warm (pay for idle) vs. eat a cold start each time." With the team pattern
  that tradeoff **disappears**: each eval launches a fresh box that **turns itself off
  when done**, so there is **zero idle GPU cost between checkpoints.** You only pay for the
  minutes an eval actually runs. There is nothing to "scale back down to zero" because
  nothing stays up.
- **Cold start is the accepted cost.** Each fresh box pays a **few-minute boot + model-load
  cold start** (instance boot, dependency install/cache, model download, vLLM engine
  start) before the eval proper begins. For per-checkpoint evals during training this is
  fine; it's the price of not keeping a machine idle. (If checkpoints ever come so fast
  and close together that cold starts dominate, revisit — but do **not** solve it by
  keeping a forbidden always-on GPU.)
- **Real cost numbers (from the team runbook).** A single g5/g6-class box for a few
  minutes is **well under ~$1** when you terminate promptly. For scale: the team's
  56-model pedagogy back-fill on a mix of g6-class GPUs ran ~2 hours for **~$4.60 total**,
  and a paused 200-model fleet was **~$0.05 at T+5min**. Per-checkpoint evals are far
  smaller than that.
- **Mandatory cost discipline (team rule).** Always set a **hard cost cap** in your head
  before launching (the runbook uses ~$15 for a whole fleet), keep the **`shutdown -h +N`
  self-terminate** line on every box, and **explicitly terminate** the moment results land
  — never rely on the timer alone, and never leave an idle GPU running (idle still bills,
  and it counts against the ≤3-instance cap).
- **Spot instances (mention only):** AWS also rents **"spot" machines** — the same
  computers at a discount, with the catch that Amazon can reclaim them on short notice.
  Potentially useful for the back-fill (Flow 2), where a lost worker can be relaunched,
  but out of scope for now and still subject to the `g6.xlarge` ceiling.

---

## 11. Open questions / decisions for the user

- **Two-model GPU fit under the L4 cap:** confirm the ~9B Qwen judge's real VRAM footprint
  on a 24 GB L4, and lock in **sequential stages (§8 Option A)** vs. **separate judge
  instance (§8 Option B)**. (The old "both on one A100/H100" option is ruled out by the
  team rule.)
- **Checkpoint size limits on an L4:** what is the largest checkpoint model we can eval on
  a single `g6.xlarge` (24 GB), and what's our policy when a checkpoint exceeds it (skip?
  shard? smaller-precision load?) — since upsizing the instance is not allowed.
- **Which tasks / suite per checkpoint:** the full eval suite every time, or a fast subset
  live (Flow 1) and the full suite in back-fill (Flow 2)?
- **S3 layout:** confirm the prefixes for checkpoints in vs. results out under
  `edullm-adaptive-inference-056956104102` (e.g. `checkpoints/…` and `olmo-eval-results/…`),
  matching the team's existing prefix conventions.
- **Back-fill concurrency:** with the hard **≤3 GPU workers** cap, how many checkpoints
  should each worker loop over, and is a spot-instance back-fill worth pursuing later?
- **Persistent polling worker (future only):** instead of a hook per checkpoint, a
  long-running worker could watch S3 for new checkpoints and launch evals. We're **not**
  designing this now — noting it as a possible future fallback (and it would still obey the
  `g6.xlarge` / ≤3 / self-terminate rules).
