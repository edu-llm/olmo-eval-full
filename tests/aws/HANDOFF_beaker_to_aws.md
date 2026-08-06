# Handoff: Beaker → AWS for olmo-eval Checkpoint Evaluations

> **Purpose of this doc:** Context dump for the next agent / teammate picking up the
> "run our evals on AWS instead of AI2 Beaker" effort. Everything here reflects the
> state as of **2026-08-04** on the `CheckpointFlows` branch of `olmo-eval-full`.

---

## 1. The big picture / goal

We want to evaluate **training checkpoints** of our model using the `eduLLM-Evals`
tasks (TutorEval, TutorBench, the Pedagogy MCQ benchmark) **plus a Qwen LLM judge**,
and we want to run this on **our team's AWS account**, NOT on **AI2 Beaker**.

> **Decision (2026-08-05): evals run retroactively, in a batch — not during training.**
> Training only writes checkpoints to S3; **all evals run afterward, batched over the saved
> `step_N` checkpoints** via the **retroactive back-fill fleet** (`AWS_CHECKPOINT_EVAL_PLAN.md`
> §7 / `CHECKPOINT_LAUNCHER_SCOPING.md` §3.5). The unit of work is unchanged — "eval one
> checkpoint on AWS → results to S3" — but it is driven by a **batch driver over saved
> checkpoints**, not an inline during-training hook. The inline per-checkpoint hook (Flow 1) is
> **de-emphasized and off the eval critical path** (kept for reference, not deleted).

- `olmo-eval` (this repo) already has a checkpoint-eval concept ("CheckpointFlows").
- Its two run paths today are:
  - `olmo-eval run ...`  → runs the eval pipeline **inline** on whatever machine you're on.
  - `olmo-eval beaker launch ...` → an **orchestration layer** that provisions an AI2
    **Beaker** node and ultimately calls `olmo-eval run` there. Results go to S3.
- **We do not have Beaker access.** So the plan is to replace the Beaker orchestration
  layer with our team's AWS pattern, while reusing the exact same `olmo-eval run` core.

---

## 2. Why AWS EC2 + SSM (not AWS Batch, not a persistent GPU)

Early exploration considered AWS Batch, EC2-direct, and SkyPilot/Ray. We **pivoted to
our team's existing EC2 + SSM pattern** because the org already has a battle-tested
setup and we shouldn't diverge from it. Key constraints that shaped everything:

- **Account:** `sbsandbox`.
- **GPU cap:** `g6.xlarge` (**NVIDIA L4, 24 GB VRAM**) is the ceiling we can launch.
  - Consequence: the checkpoint model and the Qwen judge **cannot co-reside** on one
    24 GB GPU. They must run **sequentially** (generate responses → then judge).
- **No persistent GPU** (cost). Instances are **launched on demand** and
  **self-terminate** via `shutdown -h +N` scheduled in their `user-data`.
- **No custom ECR container.** We **stage code onto the DLAMI at runtime** instead of
  baking an image (simpler; avoids ECR ownership).
- **Access is via a credential broker**, not static keys (see §4).

---

## 3. The two planning docs (repo root)

- **`AWS_CHECKPOINT_EVAL_PLAN.md`** — the plan for running checkpoint evals on AWS.
  Originally written for AWS Batch, then **fully rewritten** for the team's EC2+SSM
  pattern, the `g6.xlarge`/L4 constraint, no-ECR runtime staging, and self-terminating
  instances.
- **`OLMO_EVAL_INTEGRATION_PLAN.md`** — the plan for wiring our custom evals + Qwen judge
  into `olmo-eval`. Findings:
  - `olmo-eval` already supports **LLM-judge scorers** and **auxiliary model providers**,
    and has a **unidimensional CAT** implementation.
  - Our **multidimensional CAT** (the scenario-level TutorEval work) is the **only piece
    with no direct `olmo-eval` equivalent** → keep it **offline** for now.
  - The **Pedagogy Benchmark** is our custom **multiple-choice (MCQ)** task to register.

---

## 4. Team AWS access (credential broker + MCP)

- **Broker:** `sb-aws-creds` — a Node.js CLI (shipped as `sb-aws-creds-0.2.1.tgz`).
  It brokers temporary creds to the `sbsandbox` account.
- **Reference runbook:** the team's own `AWS run book.md` documents their EC2+SSM
  inference pattern (S3 staging, Secrets Manager for the HF token, auto-shutdown).
- **Local wiring that must exist for AWS CLI + agent use:**
  - `~/.cursor/mcp.json` has an `sb_aws` MCP server. **Note:** it registers with a
    `user-` prefix, so from an agent it's addressed as **`user-sb_aws`**.
  - `~/.aws/config` gets an **`sbsandbox`** profile via `sb-aws-creds install-profiles`.
- **Setup gotchas already hit & solved (so the next person doesn't repeat them):**
  1. `aws: command not found` → `brew install awscli`.
  2. `The config profile (sbsandbox) could not be found` → the profile isn't provisioned
     until the broker/MCP has run once. Order that works: `sb-aws-creds login` →
     **restart Cursor** so the MCP loads → then `sb-aws-creds install-profiles`.
  3. Agent MCP call failed with "server sb_aws not found" → use **`user-sb_aws`**.
  4. AWS CLI **"Not logged in"** → run `sb-aws-creds login` again (broker session expired).
  5. AWS CLI **"fetch failed"** → sandbox network blocked the broker endpoint; rerun the
     command with **full network** permission.

---

## 5. The smoke test ("Rung 1") — what it is and why

**Goal of the smoke test:** prove we can run a real `olmo-eval` job end-to-end on our AWS
setup **instead of Beaker** — i.e., validate the EC2+SSM path before wiring it into the
checkpoint flow. It deliberately goes through the **same `olmo-eval run` core** the
Beaker path uses.

**Rung 1 configuration (intentionally tiny/cheap):**
- Instance: `g5.xlarge` (small GPU; within the pattern).
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (tiny).
- Task: `arc_easy`.

**End-to-end flow the smoke test performs:**
1. Pre-flight checks (creds, profile, params).
2. Launch EC2 via `run-instances` with `user-data` that schedules **auto-termination**.
3. **Stage code** onto the node (default `git clone`; `tar` upload is opt-in — see §7).
4. Install deps with `uv`.
5. Run `olmo-eval` on the node via **SSM** (`send-command` + `get-command-invocation` polling).
6. Verify results in **S3**.
7. **Teardown** the instance.

### Smoke-test files (all in `tests/aws/`)
- **`run_rung1_smoketest.sh`** — the single, parameterized **orchestrator** that runs the
  whole thing end-to-end. Has `--dry-run` and `-h/--help`, `set -euo pipefail`, a
  `trap cleanup` for teardown, and a `STAGE_MODE` knob (`clone` default / `tar` opt-in).
  All key values are env-overridable.
- **`RUNBOOK_rung1_smoketest.md`** — the manual, step-by-step version of the same process
  (broker setup → launch → stage → install → run → verify → teardown) plus a
  troubleshooting table.
- **`SMOKETEST_EXPLAINED.md`** — beginner-friendly deep explanation of what the smoke test
  does and why.
- **`rung1_smoketest.sh`** — an older on-node helper (its commands are now inlined into the
  orchestrator; kept for reference).

> ⚠️ **Path note:** the scripts' internal defaults were originally written for the path
> `eduLLM-Evals/scripts/aws/`. The files now live at **repo-root `tests/aws/`**. If you run
> them from here, either override the path env vars or update the in-file defaults
> (esp. `STAGE_MODE=tar` local-repo path and the git-clone repo/ref defaults).

---

## 6. Smoke test — what actually happened when we ran it live

We **did run it live on a real GPU instance** and got `olmo-eval` to execute. Two on-node
bugs were found and fixed:

1. **`uv: command not found`** — `HOME` was **empty** in the SSM execution context, so
   `export PATH="$HOME/.local/bin:$PATH"` expanded wrong and `uv` (at `/root/.local/bin`)
   wasn't found. **Fix:** `export HOME=/root` before anything else. This is now baked into
   both the orchestrator's install/run commands and the runbook/troubleshooting.
2. **`No space left on device` during `uv sync`** — the DLAMI **root disk (~19 GB)** is too
   small for torch + vLLM (~11 GB) at the default `/root/.cache/uv`. **Fix:** redirect caches
   to the larger **NVMe scratch disk**: `UV_CACHE_DIR=/opt/dlami/nvme/...` and
   `HF_HOME=/opt/dlami/nvme/...` (create the dirs first).
3. **Orchestrator falsely "hangs" after a successful eval** (found 2026-08-04, second live
   run). The completion poll ran `tail -n 200 smoketest.log` via SSM, but olmo-eval's log is
   full of large Rich tables, so the output exceeds SSM's **~24 KB `StandardOutputContent`
   cap** and gets truncated **before** the trailing `RUNG1_DONE_EXIT=N` sentinel — so the
   loop never detects completion and spins to `EVAL_TIMEOUT`. The eval itself had *finished
   and uploaded to S3*. **Fix (in `run_rung1_smoketest.sh`):** the poll now emits the
   `grep`-ed sentinel line **first** (always within the first bytes), then a small
   `tail -c 1500` heartbeat. NOTE: this bit us on a run where the S3 upload had already
   succeeded — verify results in S3 directly (`aws s3 ls .../smoke/rung1/`) before assuming a
   spinning orchestrator means failure.

---

## 7. S3 `PutObject` → `AccessDenied` — RESOLVED (2026-08-04)

**Root cause: a PREFIX MISMATCH, not a missing capability.** The `EswManagedInstance`
role *already has* `s3:PutObject` on the results bucket — but only for the prefixes
`smoke/*`, `smoke_split/*`, `full200/*` (via the inline role policy
`AdaptiveSmokeTemp20260731`, and `smoke/*` also via the bucket policy). The smoke-test
script (`run_rung1_smoketest.sh`) writes to **`smoketest/…`** (`S3_PREFIX=smoketest`,
`S3_GROUP=rung1` → `s3://edullm-adaptive-inference-056956104102/smoketest/rung1/`).
`smoke/*` does **not** match `smoketest/…` (needs a literal `/` after `smoke`), so the
upload is `implicitDeny` → `AccessDenied`.

Verified with `aws iam simulate-principal-policy` on `role/EswManagedInstance`:
- `s3:PutObject` → `.../smoketest/rung1/metrics.json` = **implicitDeny**
- `s3:PutObject` → `.../smoke/rung1/metrics.json`     = **allowed**

**Target results location:**
- Bucket: `edullm-adaptive-inference-056956104102` (region `us-east-1`)
- Original (denied) prefix: `smoketest/rung1/` + `smoketest/staging/`.
- Current default (post-fix): `smoke/rung1/` (metrics.json etc.); tar-staging: `smoke/staging/`.

**Can our broker role apply the fix?** Yes. Our session assumes
`Intern-cathy.du-sbsandbox` (AdministratorAccess, capped by the `InternSandboxBoundary`).
Simulation confirms `s3:PutBucketPolicy` (on the bucket) and `iam:PutRolePolicy` (on
`EswManagedInstance`) are both **allowed**; `iam:CreateRole` is **explicitDeny** by the
boundary unless the new role also carries `InternSandboxBoundary` (so a brand-new
dedicated role is the most awkward option). No account admin needed for the two
policy-edit routes.

- **Workaround still valid:** pull `metrics.json` off the node via SSM.
- **RESOLUTION CHOSEN — the zero-AWS-change prefix alignment.** The smoke-test scripts now
  default `S3_PREFIX=smoke` (was `smoketest`), so results write to `smoke/rung1/` and
  tar-staging to `smoke/staging/` — both already covered by the ESW `smoke/*` grant. **No
  IAM or bucket-policy change was made.** Files updated (defaults only, still
  env-overridable): `tests/aws/run_rung1_smoketest.sh`, `tests/aws/rung1_smoketest.sh`, and
  the doc references in `RUNBOOK_rung1_smoketest.md` / `SMOKETEST_EXPLAINED.md`.
  Re-verified: `s3:PutObject` → `.../smoke/rung1/metrics.json` = **allowed** for
  `role/EswManagedInstance`.
- **Alternative still on the table (not applied):** if we later want a distinct
  `smoketest/` (or per-run) namespace, add a **scoped bucket-policy grant** giving ESW
  `s3:PutObject/GetObject/DeleteObject` on `.../smoketest/*` (+ `ListBucket` w/ prefix
  `smoketest/*`). Pre-verified via simulation with `--resource-policy` → **allowed**; our
  broker role can apply it (shared-infra mutation is gated by auto-review and needs
  approval). Extending the inline role policy `AdaptiveSmokeTemp20260731` is a third option
  (touches the shared ESW role; least preferred).
- **Applies to the newly-pulled diagnostic paths too (2026-08-04).**
  `diagnostics/mcq_cat/common/s3_io.py` and `tests/OnNode/checkpoint_infer.py` upload results
  with **raw `boto3` `put_object`** (not olmo-eval's storage layer, not the `--s3-*` flags), so
  they are subject to the **same `EswManagedInstance` grant**. Their write locations —
  `mcq_cat`'s `--s3-out` and `checkpoint_infer`'s default `checkpoint-infer/` prefix — are
  **not** under `smoke/*`, so a real run there will hit the same `AccessDenied` unless pointed
  at a granted prefix (e.g. `smoke/…`) or covered by the scoped bucket-policy grant above.

---

## 8. `STAGE_MODE`: clone vs tar (why local repo path mattered)

- **`clone` (default):** the node does `git clone <repo> && git checkout <ref>`. No dependency
  on your local checkout → this is why we made it the default (a teammate can run the script
  from anywhere).
- **`tar` (opt-in):** tars up your **local** repo and uploads it. Needed only for
  private/uncommitted code. This is the mode that requires a valid local repo path.

---

## 9. Current git state (as of handoff)

- **Branch:** `CheckpointFlows`, local == `origin/CheckpointFlows` tip, plus the staged
  changes below on top.
- **Staged / intended to commit** (nothing else):
  - `.gitignore` (added a guard — see below)
  - `AWS_CHECKPOINT_EVAL_PLAN.md`
  - `OLMO_EVAL_INTEGRATION_PLAN.md`
  - `tests/aws/RUNBOOK_rung1_smoketest.md`
  - `tests/aws/SMOKETEST_EXPLAINED.md`
  - `tests/aws/run_rung1_smoketest.sh`
  - `tests/aws/rung1_smoketest.sh`
  - (this file, `tests/aws/HANDOFF_beaker_to_aws.md`, if you stage it too)
- **`.gitignore` guard:** `eduLLM-Evals/` is now **fully ignored** on this branch. Reason: a
  stray "Stage All" previously swept the entire `eduLLM-Evals/` tree (~5,600 data/log/run
  files, 1M+ lines) into a commit. It was reset. The remote `CheckpointFlows` **does not
  contain `eduLLM-Evals/`** — that tree only exists on disk (from the `tutoreval` branch) and
  is intentionally not tracked here.

### Important gotchas for the next agent
- The smoke-test files originally came from the **`tutoreval`** branch (via `git checkout`).
  They are safe there in history if you ever need the originals.
- `tests/aws/` is at the **repo root**, which is **above** the Cursor workspace root
  (`eduLLM-Evals/`) — so these files **won't appear in the file explorer** unless you open
  the repo root as the workspace folder.
- **Do NOT `git add -A` / "Stage All" carelessly.** The `.gitignore` now blocks the big
  `eduLLM-Evals/` tree, but always sanity-check `git status` before committing.

---

## 10. Suggested next steps

1. **Close the S3 `PutObject` gap** (scoped bucket policy or dedicated IAM role) so results
   auto-upload — this unblocks unattended runs.
2. **Wire the AWS EC2+SSM launcher into the checkpoint flow** as the Beaker replacement
   (reuse `olmo-eval run` core; orchestrate launch/stage/run/teardown like the smoke test).
   *(2026-08-05: evals are now **retroactive/batched**, not during training. Build order:
   prove the **single-checkpoint atom** first, then promote the **batch / back-fill driver**
   (enumerate saved `step_N` checkpoints → sequential loop → ≤3-worker shard-index fleet) to the
   core mechanism. The inline during-training hook (Flow 1) is de-emphasized. See
   `CHECKPOINT_LAUNCHER_SCOPING.md` §3.5 / §9 and `AWS_CHECKPOINT_EVAL_PLAN.md` §7.)*
3. **Register the custom evals** per `OLMO_EVAL_INTEGRATION_PLAN.md`: Qwen judge as an
   LLM-judge scorer, Pedagogy MCQ as a task. Keep the **multidimensional CAT offline** for now.
   *(2026-08-04: a standalone offline MCQ CAT — uni + MIRT — is now taking shape in
   `diagnostics/mcq_cat/` (plan `Plan/mcq_cat_diagnostics/README.md`), decoupled from
   `olmo_eval`. It's a candidate home for MCQ/pedagogy CAT instead of the olmo-eval ATLAS path —
   see the `OLMO_EVAL_INTEGRATION_PLAN.md` reconciliation note.)*
4. Account for the **24 GB / sequential model-then-judge** constraint in the flow design.
5. Update the smoke-test scripts' in-file path defaults now that they live in `tests/aws/`.
