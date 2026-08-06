# Scoping: EC2+SSM Checkpoint-Eval Launcher (Beaker replacement)

**Status:** scoping / design only. No launcher code here, nothing committed. This
document maps the existing Beaker orchestration onto our validated EC2+SSM smoke-test
pattern and specifies exactly what to build next.

**Eval model (decided 2026-08-05): retroactive batch only.** Evals are **not** run during
training. Training only writes checkpoints to S3; full evals run **retroactively, batched over
all saved `step_N` checkpoints** once (or well after) training is done. The **atom** is still
"eval one checkpoint on AWS via the EC2+SSM launcher → results to S3" (§3 — this is the Beaker
replacement, and arguably a cleaner fit since Beaker also ran evals as discrete provisioned
jobs). The **core orchestration** on top is a **batch / back-fill driver** (§3.5) that
enumerates a run's checkpoints and evaluates each. There is **no inline, fire-during-training
hook** — that requirement is dropped entirely.

**Reads it builds on:**
- `tests/aws/run_rung1_smoketest.sh` — the proven launch → stage → install → run →
  poll-sentinel → verify-S3 → teardown orchestrator (validated end-to-end).
- `tests/aws/HANDOFF_beaker_to_aws.md` §6/§7 — S3 prefix/grant situation, the 24 KB
  SSM-output poll fix, on-node `HOME`/cache gotchas.
- `AWS_CHECKPOINT_EVAL_PLAN.md` — plain-English AWS plan (L4 ceiling, self-terminate,
  shard-index fleet, `olmo-eval-results` namespace).
- `OLMO_EVAL_INTEGRATION_PLAN.md` — how the Qwen judge maps onto olmo-eval's LLM-judge
  scorer + auxiliary provider.

---

## 1. Key findings

### 1.1 The Beaker seam (where orchestration ends and the reusable core begins)

The Beaker path is a thin orchestration wrapper around a single CLI invocation. Concretely
(`src/olmo_eval/launch/beaker/launcher.py`):

- A launch is fully described by a `BeakerJobConfig` whose **`command` is literally
  `["olmo-eval", "run", "-m", <model>, "-t", <task>, ...]`** (see the class docstring at
  `launcher.py:7-14` and `:328-335`, and `BeakerJobConfig.command` at `:359-360`).
- `BeakerLauncher.launch()` (`launcher.py:974-1105`) does **only orchestration**:
  1. resolves cluster aliases (`resolve_clusters`),
  2. builds an **install command** — `uv pip install -e '.[<extras>]' -c <cuda-constraints>`
     (`_build_install_cmd`, `launcher.py:792-972`),
  3. calls gantry's `launch_experiment(args=config.command, install=install_cmd, …)`
     (`launcher.py:1063-1092`), which **git-clones the repo to `/gantry-runtime`**, runs the
     install command, mounts **Weka** buckets for the HF cache
     (`HF_HOME=/weka/oe-eval-default/…`, `BeakerJobConfig.env_vars` at `:390-396`), optionally
     **injects AWS creds as Beaker env-secrets** (`inject_aws_credentials` →
     `ensure_aws_secrets`, `:1029-1034`), then **runs `config.command`**.
- Results reach S3 **inside `olmo-eval run`**, not in the launcher: the launcher just passes
  `--s3-bucket/--s3-prefix/…` through into `command`. The Beaker CLI options
  (`cli/beaker/launch.py:115-131`) are forwarded verbatim to the run.

**Seam:** everything above `olmo-eval run` is Beaker/gantry-specific (warm cluster node, git
clone to `/gantry-runtime`, `uv pip install`, Weka mount, AWS env-secret injection, log
follow). Everything at and below `olmo-eval run` is the **orchestrator-agnostic core** we
reuse unchanged.

### 1.2 The reusable core (`olmo-eval run`) is identical across orchestrators

`src/olmo_eval/cli/run/__init__.py` defines the core. Its contract:

- `-m/--model` (required): a preset name **or** an HF model id **or** a local dir **or** a
  remote URI. `get_provider_config` (`common/configs.py:152-195`) treats an unknown `-m` as a
  raw model path/id.
- `-t/--task` (repeatable), `-o/--override` (per-task, e.g. `-o limit=100`), `-H/--harness`
  (preset selecting provider + **auxiliary providers**), `-O/--output-dir`.
- **S3 output:** `--s3-bucket`, `--s3-prefix`, `--s3-group`, `--s3-region`,
  `--s3-endpoint-url` (`cli/run/__init__.py:86-90`, wired through `StorageSetup`, `:204-218`).
- `--store` for the Postgres backend (not needed for us initially).

This is the exact command our smoke test already runs over SSM (with `--s3-prefix smoke`), so
**the core is confirmed drop-in**: Beaker and our EC2+SSM launcher differ only in *how the
node is provisioned, code staged, and deps installed* — the eval invocation is byte-for-byte
the same shape.

### 1.3 How a checkpoint is consumed

- For **OLMo Core distributed checkpoints**, the `olmo_core` provider reads the checkpoint
  via AI2 `cached_path` (`inference/providers/olmo_core_utils.py`: `_read_checkpoint_config`,
  `_validate_olmo_core_checkpoint`, `_resolve_checkpoint` at `:204-390`). `cached_path`
  resolves `s3://`, `gs://`, `weka://`, HF, and local paths — so **`-m s3://<bucket>/checkpoints/step_N`
  works directly**, given the instance can read that S3 prefix (our `EswManagedInstance`
  profile already has S3 **read**).
- For **HF-format checkpoints / vLLM**, `-m` is an HF id or a local directory; the provider
  downloads/loads through the HF cache (redirect to NVMe on our box, see §3).
- Task/checkpoint config passes through `-o` overrides and/or a `-c` YAML config exactly as in
  Beaker (`cli/beaker/launch.py` forwards these into the same `run`).

### 1.4 S3 namespace

The checkpoint plan writes results to **`--s3-prefix olmo-eval-results`**
(`AWS_CHECKPOINT_EVAL_PLAN.md` §5–§6). That prefix is **not** covered by the
`EswManagedInstance` inline grant (which only allows `PutObject` under `smoke/*`,
`smoke_split/*`, `full200/*`). So a real checkpoint run **hits the same `AccessDenied` wall**
the smoke test hit — see §5 for the decision.

### 1.5 Reconciliation with the pulled `diagnostics/mcq_cat/` package (2026-08-04)

A teammate landed a **second, self-contained checkpoint-eval path** (`diagnostics/mcq_cat/`,
plan `Plan/mcq_cat_diagnostics/README.md`) that this launcher must also be able to drive. It
is **not** an `olmo-eval run` invocation and deliberately does **not** depend on the
`olmo_eval` package (same philosophy as `tests/OnNode/checkpoint_infer.py`, whose S3 IO +
checkpoint-load patterns it mirrors):

- **Entry point:** `python -m diagnostics.mcq_cat.runner --cat-style <name> --checkpoint
  <path|s3://…> --s3-out s3://… [--checkpoint-kind hf|olmo_core] [--benchmark …]
  [--irt-params …]` (`diagnostics/mcq_cat/runner.py`). Different CLI shape than `olmo-eval run`
  (`-m`, `-t`, `--s3-bucket/--s3-prefix/--s3-group`).
- **Checkpoint load — divergence from §1.3.** It **rolls its own** S3 prefix download via raw
  `boto3` (`common/s3_io.py: resolve_checkpoint`/`download_prefix`) into a tempdir, then loads
  with plain `transformers.AutoModelForCausalLM.from_pretrained` (`common/inference.py`). It
  does **not** use olmo-eval's `olmo_core` provider or `cached_path`. The `olmo_core` kind is a
  `NotImplementedError` integration point in **both** `mcq_cat` and `checkpoint_infer.py`, so
  today only **HF-format** checkpoints work on that path — an OLMo-Core *distributed*
  checkpoint only loads via the `olmo-eval` core path (§1.3).
- **Lighter deps.** The `mcq_cat` HF path needs only `torch` + `transformers` + `boto3` (boto3
  imported lazily), not the full `olmo-eval[vllm]` stack — so the stage/install phase differs
  by which on-node command runs.
- **Single GPU phase.** It scores MCQ by log-likelihood (one model resident, no LLM judge), so
  it **sidesteps the 24 GB two-model constraint** in §6 entirely.

**Net:** the launcher's orchestration pattern (launch/stage/install/run/poll/verify/teardown)
is agnostic to which on-node command runs; only the **staged deps** and the **result-key
convention** differ. See §3 phase 5, §5, and §6 for the specific reconciliations.

---

## 2. Beaker → EC2+SSM mapping

| Concern | Beaker path (today) | Our EC2+SSM launcher (to build) |
|---|---|---|
| Get a machine | gantry finds a **warm** cluster node | `aws ec2 run-instances` → one self-terminating `g6.xlarge` (L4, 24 GB) |
| Stage code | gantry **git-clones** repo → `/gantry-runtime` | SSM: `git clone <our repo>@<ref>` **or** tar+presigned-URL of the local tree → `/opt/dlami/nvme/olmo-eval` |
| Install deps | `uv pip install -e '.[extras]' -c constraints` | SSM: `uv sync --frozen` / `uv pip install -e '.[…]'` (DLAMI already has CUDA) |
| GPU driver/CUDA | Beaker image | **DLAMI** `ami-0b6f2229ad14c9323` (drivers/CUDA baked) |
| HF cache | Weka mount, `HF_HOME=/weka/…` | `HF_HOME`/`HF_HUB_CACHE`/`UV_CACHE_DIR` → **NVMe** `/opt/dlami/nvme/...` (root disk is tiny) |
| `HOME` | container `/root` | export `HOME=/root` explicitly (SSM runs as root but `HOME` may be unset) |
| S3 creds | injected AWS **env-secrets** | **instance profile** `EswManagedInstance` (no secrets on the box) |
| Run the eval | `olmo-eval run -m … -t … --s3-*` | **identical** `olmo-eval run -m … -t … --s3-*` over SSM |
| Progress/finish | Beaker log follow + status reporter | poll on-node log for **`<SENTINEL>=<exit>`**, 24 KB-safe (grep sentinel first, then bounded tail) |
| Results out | inside the run via `--s3-*` (+ `/results`) | inside the run via `--s3-*`; then **verify in S3** and **terminate** |
| Teardown | Beaker frees the node | `shutdown -h +N` safety net **+** explicit `terminate-instances` on success |

The right-hand column is exactly what `run_rung1_smoketest.sh` already does — the checkpoint
launcher is a **generalization of that script from "one fixed smoke model" to "the checkpoint
I was handed."**

---

## 3. Concrete design for the per-checkpoint eval (the atom)

This is the unit of work — evaluate **one** checkpoint — that the batch driver (§3.5) repeats
across a whole run. Mirror the smoke-test orchestrator's proven phases. Proposed script:
`tests/aws/run_checkpoint_eval.sh` (orchestrator, runs locally/off-box) plus an on-node
bootstrap it drives over SSM.

**Inputs (env-overridable, like the smoke script):**
- `CHECKPOINT` — `s3://…/checkpoints/step_N` (or HF id / local dir).
- `CHECKPOINT_KIND` — `hf` (default, validated) | `olmo_core` (optional native path). Selects the
  provider and the conditional `ai2-olmo-core` install (§3.6).
- `TASKS` — e.g. `tutorbench`, `pedagogy`, or a suite.
- `INSTANCE_TYPE=g6.xlarge` (real runs) / `g5.xlarge` (smoke). **Never larger** (§4 hard cap).
- `S3_BUCKET=edullm-adaptive-inference-056956104102`, `S3_PREFIX` (see §5), `S3_GROUP=step_N`.
- `SHUTDOWN_MIN` — sized per run (see §6 timeouts), armed as `shutdown -h +N`.
- `HARNESS` — preset selecting the checkpoint provider and (later) the judge (§6).

**Phases (reuse the smoke orchestrator verbatim where possible):**

1. **Preflight** — verify broker creds (`whoami` / `sts get-caller-identity`); enforce the
   **≤3 running `edullm-gpu-worker`** cap by counting tagged running instances before launch.
2. **Launch** — `run-instances` with DLAMI, `g6.xlarge`, `--iam-instance-profile
   Name=EswManagedInstance`, team SG/subnet, `--instance-initiated-shutdown-behavior
   terminate`, tags `Name=edullm-gpu-worker,Step=<N>`, and **user-data that arms
   `shutdown -h +${SHUTDOWN_MIN}`**.
3. **Stage** — over SSM: `export HOME=/root`; redirect `HF_HOME`, `HF_HUB_CACHE`,
   `UV_CACHE_DIR`, `TMPDIR` onto **`/opt/dlami/nvme`**; fetch the code (clone our repo at the
   eval ref, or untar a presigned bundle) into NVMe.
4. **Install** — `uv sync --frozen` against the DLAMI's CUDA (lean, HF-default), **conditionally
   adding `--extra olmo_core` only when `CHECKPOINT_KIND=olmo_core`** (§3.6); mirror Beaker's
   constraint trick (freeze `torch`/`nvidia-*` so uv can't downgrade the pre-baked CUDA stack —
   see `launcher.py:889-891`).
5. **Run** — over SSM, in a detached logged process, run the **same** core:
   `olmo-eval run -m "$CHECKPOINT" -t "$TASKS" -O <nvme-out> --s3-bucket "$S3_BUCKET"
   --s3-prefix "$S3_PREFIX" --s3-group "$S3_GROUP" --s3-region us-east-1`, echoing a trailing
   **`CHECKPOINT_DONE_EXIT=$?`** sentinel. For `CHECKPOINT_KIND=olmo_core`, prepend
   `-H default -o provider.kind=olmo_core` to select the native `OlmoCoreProvider` (§3.6); the HF
   default needs no provider override. **Skip-if-done:** before launching a box, check S3 for the
   step's result marker (`metrics.json` under `$S3_PREFIX/$S3_GROUP/…`) and skip the whole launch
   if present, so the atom is safe to re-run (§3.5 point 4).
   - **Pluggable on-node command (§1.5):** for MCQ CAT diagnostics the run is instead
     `python -m diagnostics.mcq_cat.runner --cat-style <name> --checkpoint "$CHECKPOINT"
     --s3-out s3://$S3_BUCKET/$S3_PREFIX/<group>`. Same sentinel + 24 KB-safe poll wrapper;
     only the staged deps (`torch`/`transformers`/`boto3`, no vLLM) differ.
6. **Poll** — reuse the **24 KB-safe** poll: emit the grepped `…_DONE_EXIT=<n>` line **first**
   (survives SSM's ~24 KB `StandardOutputContent` cap), then a small `tail -c 1500` heartbeat.
   (This is the exact bug/fix documented in `HANDOFF` §6.)
7. **Verify** — `aws s3 ls` / `s3api head-object` on the expected result keys under
   `s3://$S3_BUCKET/$S3_PREFIX/$S3_GROUP/…` before declaring success.
8. **Teardown** — on success/failure, `terminate-instances` explicitly (don't rely on the
   timer alone); confirm `shutting-down`/`terminated`.

**Reuse note:** phases 3, 4, 6, 7, 8 are effectively unchanged from `run_rung1_smoketest.sh`;
the new work is phase-2 sizing (bigger disk/timeout), phase-5 checkpoint `-m`, and the judge
phase (§6).

---

## 3.5 The retroactive batch / back-fill driver (core orchestration) — CONFIRMED DESIGN

Because evals run **retroactively over all checkpoints** (not per-checkpoint during training),
the launcher's top layer is a **batch driver**. **Confirmed design (2026-08-05):** a **small,
self-terminating fleet capped at ≤3 workers**, where **each worker installs the environment
ONCE and then loops over its index-sharded subset of checkpoints** (worker *i* takes steps
*i, i+K, i+2K, …*), running the §3 atom (`olmo-eval run`) per checkpoint and uploading results,
with an **idempotent skip-if-done** guard, then self-terminates.

**Why install-once-per-worker (the key rationale):** the expensive part of a run is the
**~10-minute torch + vLLM install** (plus first CUDA/model warmup) on the fresh DLAMI. Paying
that **once per worker** and amortizing it across **many checkpoints** — rather than once per
checkpoint (a fresh box each time) — is the whole point. A 3-worker fleet that each evaluates,
say, 10 checkpoints pays **3 installs, not 30**.

Driver shape:

1. **Enumerate** — list the run's checkpoints in S3 (e.g. `aws s3 ls
   s3://<bucket>/checkpoints/<run>/` → `step_1000/`, `step_2000/`, …). This ordered list is the
   shard input.
2. **Launch the fleet (≤3)** — one `run-instances --count K` (`K ≤ 3`, §4) of self-terminating
   `g6.xlarge` workers, tagged `Name=edullm-gpu-worker`. Each worker reads its shard index from
   **`ami-launch-index`** via **IMDSv2** (token dance — see `AWS_CHECKPOINT_EVAL_PLAN.md` §7).
3. **Per-worker: install ONCE, then loop its shard.** Worker *i* of *K* stages code + installs
   deps **one time** (§3 phases 3–4), then iterates **its slice** — steps *i, i+K, i+2K, …*
   (round-robin by `ami-launch-index`). For **each** assigned checkpoint it runs the §3 atom
   (`olmo-eval run -m s3://…/step_N … --s3-group step_N`), polls the 24 KB-safe sentinel, and
   verifies the S3 result — **reusing the one installed environment** across all of them.
4. **Idempotent skip-if-done.** Before evaluating a step, check S3 for its **results marker**
   (e.g. `…/<run>/step_N/metrics.json`, or a `_SUCCESS` object). If present, **skip** that step.
   This makes the fleet **safe to re-run / resume** after a crash, a spot reclaim, or a partial
   sweep — a relaunched worker picks up only the unfinished steps.
5. **Collect + self-terminate.** Each run writes to `…/<results-prefix>/<run>/step_N/`
   (`S3_GROUP = step_N`), a per-step layout ready for cross-checkpoint aggregation. When a worker
   finishes its slice (or the `shutdown -h +N` safety timer fires), it self-terminates; the
   driver confirms teardown.

**Constraints carried from the atom:** the §4 hard cap (`g6.xlarge` only, **≤3 workers at
once**, `shutdown -h +N` on every box) governs fleet width; the §5 grant applies to every
worker's writes; the §6 judge phase (if any) runs per checkpoint *inside* the atom. A
**sequential single-box loop is just the *K = 1* case** of this same driver (install once, loop
all checkpoints, skip-if-done) — the cheapest way to prove the loop before fanning out.

**What we are NOT building:** no inline "fire per checkpoint during training" hook, no coupling
to the training loop, no backgrounded/non-blocking launch from a trainer. Training's only job is
to write checkpoints to S3; the batch driver runs afterward. (A lighter
`checkpoint_infer.py`-style payload could optionally stand in as the per-checkpoint command, but
the eval path is `olmo-eval run` per checkpoint — see §1.5.)

### 3.5.1 Driver inputs / outputs

The batch entrypoint (`tests/aws/run_checkpoint_batch.sh`, beside the §3 atom) is env-overridable
in the same style as the atom, and takes:

- **`RUN_URI`** (required unless `CHECKPOINT_LIST` is given) — the run's checkpoint **root**: the
  S3 prefix that is the **direct parent of the `step*` dirs**. See the explicit-`RUN_URI` rule in
  §3.5.2.
- **`CHECKPOINT_LIST`** (optional override) — an explicit, space/newline-separated list of
  `s3://…/step_N` URIs that **bypasses enumeration** entirely (for re-running a hand-picked subset).
- **`CHECKPOINT_KIND`** (default `hf`) — the **per-batch** format (§3.5.5 / §3.6); auto-detect only
  warns on mismatch, it does not switch per checkpoint.
- **`TASKS`**, **`LIMIT`** — threaded verbatim into the §3 atom's `-t` / `-o limit=`.
- **`STEP_FILTER`** (optional) — restrict the enumerated steps, either a regex against `step_N`
  names or a numeric `MIN:MAX` window on `N` (§3.5.2).
- **`FLEET_SIZE`** (default `1`, **hard cap 3**, §4) — number of workers; `1` is the sequential
  *K = 1* case. Validated in P2.a, only *used* from P2.c.
- **S3 result-namespace inputs** — `S3_BUCKET` / `S3_PREFIX` / result `S3_GROUP` convention
  (`step_N` per checkpoint, §3.5.4), matching the atom's §5 grant story.
- **Flags** — `--dry-run` / `--profile` / `--region`, mirroring the atom.

**Outputs:** the ordered `(step, uri, kind)` enumeration, a per-worker manifest shard, and a
batch-level `_batch/summary.json` aggregation (§3.5.4). Under `--dry-run` it prints the resolved
config and the full ordered enumeration and what it *would* dispatch — and launches nothing.

### 3.5.2 Enumeration (the shard input)

Enumeration turns `RUN_URI` into the ordered list of checkpoints:

1. **Explicit-`RUN_URI` rule (approved).** `RUN_URI` must point **directly at the parent of the
   `step*` dirs**; the driver does **not** auto-descend into `checkpoints/` or hunt for a nested
   layout. It lists the **immediate child prefixes** of `RUN_URI` (`aws s3 ls "$RUN_URI"` with
   delimiter `/`) and nothing deeper. (Whether to later add auto-descend is an open question, §8.)
2. **Step matcher `^step_?\d+$`.** Keep only immediate children whose basename matches — real
   checkpoints are named **without** an underscore (e.g. `step7629`, `step940`), but the underscore
   form (`step_1000`) is also accepted.
3. **Numeric sort by the integer `N`.** Sort by the parsed integer, **not** lexically — lexical
   order breaks (`step1000` < `step125` < `step250`), numeric order is `125, 250, …, 1000, 1125`.
4. **`STEP_FILTER`.** Apply the optional regex / `MIN:MAX` window to the sorted list.
5. **Per-checkpoint list-only format detect (sanity-check).** For each surviving step, a cheap
   **list-only** `aws s3 ls` of its immediate children classifies it: `config.json` +
   `model.safetensors` and **no** `model_and_optim/` ⇒ `hf`; a `model_and_optim/` + `.metadata`
   layout ⇒ `olmo_core`. This is only a **warn-on-mismatch** check against the per-batch
   `CHECKPOINT_KIND` (§3.5.5) — it never rewrites the kind.

The result is the ordered list of `(step, uri, kind)` tuples that feeds the shard/dispatch layer.

**Layout-variance note (both are valid `RUN_URI`s, no auto-descend needed).** The `step*` dirs sit
at whatever prefix the run chose; the operator just points `RUN_URI` at that prefix:

- **Bucket-root layout:** `RUN_URI=s3://edullm-olmo-100m-bpe-ckpts/` → child `step7629/`.
- **Nested layout:** `RUN_URI=s3://edullm-checkpoints/olmo-370m/<run>/checkpoints/` → children
  `step125/ … step1315/` (e.g. `.../edullm-370M-refhq-5p5b/checkpoints/step940`).

Both are enumerated identically because `RUN_URI` already names the `step*` parent in each case.

### 3.5.3 Fleet model & install amortization

- **Fleet width ≤ 3 (§4 hard cap).** `FLEET_SIZE` workers via one `run-instances --count K`,
  `K ≤ 3`; `FLEET_SIZE=1` is the sequential single-box case that proves the loop first.
- **IMDSv2 shard index.** Each worker reads its `ami-launch-index` via **IMDSv2** (token dance,
  `AWS_CHECKPOINT_EVAL_PLAN.md` §7) and takes the static round-robin slice steps *i, i+K, i+2K, …*
  of the enumerated list (§3.5.2). Static partition, **no claim-queue** (deferred — see §8).
- **Install once per worker.** The ~10-minute torch + vLLM install (plus first CUDA/model warmup)
  is paid **once per worker** and amortized across that worker's whole slice, then the §3 atom
  (`olmo-eval run`) runs per checkpoint reusing the one installed environment. A 3-worker fleet over
  30 checkpoints pays **3 installs, not 30**.

### 3.5.4 Idempotency, failure handling, observability

- **`_SUCCESS` done-marker (explicit).** Each finished step writes an explicit `_SUCCESS` object
  under its result prefix; the skip-if-done guard checks for that marker (in addition to the atom's
  `metrics.json` check, §3 phase 5 / §3.5 point 4) so the fleet is safe to re-run / resume — a
  relaunched worker only picks up steps still missing `_SUCCESS`.
- **Per-worker manifest shards → `_batch/summary.json`.** Each worker writes a manifest shard of
  the steps it attempted and their outcomes; a final aggregation collects the shards into
  `…/<results-prefix>/<run>/_batch/summary.json` for a single cross-checkpoint view.
- **Log exfil on failure.** On a per-checkpoint failure the on-node eval log is exfiltrated to
  `…/step_N/eval.log` in S3, so a post-mortem needs no live box.

### 3.5.5 Format threading (per-batch kind)

`CHECKPOINT_KIND` is a **per-batch** input (default `hf`), **not** a per-checkpoint mixed setting —
every checkpoint in a batch is dispatched under the same kind, threaded into each atom invocation
exactly as §3.6 describes (lean HF install by default; conditional `olmo_core` extra + `-o
provider.kind=olmo_core` when `CHECKPOINT_KIND=olmo_core`). The list-only auto-detect (§3.5.2 step
5) only **warns** when a checkpoint's on-disk layout disagrees with the batch kind; it never
switches kind mid-batch. A run that genuinely mixes formats should be split into two batches.

### 3.5.6 Execution identity — the P2 decision

The batch driver is the natural home for the §3.7 execution-identity decision, and P2 bakes in the
**identity-agnostic entrypoint**:

- **Never hard-code a human AWS profile.** Credentials resolve **ambiently** via the standard AWS
  resolution chain; `AWS_PROFILE` is **optional** (just one of several sources) and passed through
  when set. The *same* entrypoint therefore runs **laptop-side today** and under a **service / CI
  role later** with **no code change** — which is exactly the §3.7.3 "move orchestration onto a
  machine identity" direction, realized at the entrypoint level from the start.
- **Launcher permission set (§3.7.4).** Whatever principal runs the driver needs the §3.7.4
  checklist: scoped `ec2:RunInstances`, `iam:PassRole` for `EswManagedInstance`, `ssm:SendCommand`
  (+ `ssm:GetCommandInvocation`), `ec2:TerminateInstances`, `ec2:DescribeInstances`, and S3
  read on the checkpoint source buckets + read/write on the results prefix.
- **Terminate tag-mismatch prerequisite (RECORD, do not fix here).** The node role's *self-terminate*
  grant is scoped to `Purpose=smoke-test`, but the checkpoint atom tags instances
  `Purpose=checkpoint-eval`, so **explicit teardown (§3 phase 8) is currently IAM-denied**. This
  must be reconciled — either widen the terminate grant to the `checkpoint-eval` tag or realign the
  atom's tag — **before P2 runs a real fleet**. It does not affect the no-GPU P2.a work.

Cross-reference: §3.7 (full execution-identity design), §3.7.1 (node/machine identity — already
fine), §3.7.4 (permission checklist + the tag mismatch).

---

## 3.6 Checkpoint format & the loader (decision — 2026-08-05)

**Question:** can the retroactive eval load real training checkpoints via the HF path (works
today), or does it need the currently-stubbed `olmo_core` loader / a conversion step?

**Decision (confirmed by team lead, 2026-08-05): HF is the primary, default, validated format;
native OLMo-Core is an optional safety valve behind a per-run flag.** Concretely:

- **PRIMARY — HF (default, validated path).** Training will run the **OLMo→HF converter**, so
  saved checkpoints are **mostly HuggingFace format** (`config.json` + `model.safetensors`).
  This is the path the atom exercises end-to-end and the one every consumer is covered for
  today, with **no new loader code** — the olmo-eval HF/vLLM provider (`-m` → HF dir/id) and the
  diagnostics HF path both load it.
- **OPTIONAL — native OLMo-Core (safety valve).** We still support raw, sharded OLMo-Core
  checkpoints (`model_and_optim/` + `.metadata`) in case **other teams** produce their own native
  checkpoints. This rides on the **real** `olmo-eval` `olmo_core` provider — **not** the stubbed
  diagnostics loader — so it is coherent today, just not yet validated on a real native run.

**How the two paths are threaded — the `CHECKPOINT_KIND` flag.** The launcher/atom takes a
per-run parameter **`CHECKPOINT_KIND`, defaulting to `hf`**:

- **`CHECKPOINT_KIND=hf`** (default): the olmo-eval HF/vLLM provider, **lean install** (no
  `ai2-olmo-core`). This is the common, validated path and stays light.
- **`CHECKPOINT_KIND=olmo_core`**: the install step **conditionally adds the `olmo_core` extra**
  (`ai2-olmo-core[torchao,transformers]==2.4.0`, `pyproject.toml:53-55`) and the run **selects the
  olmo_core provider** (`olmo-eval run … -o provider.kind=olmo_core`). The provider is
  `OlmoCoreProvider` (`src/olmo_eval/inference/providers/olmo_core.py:46`), validated by
  `olmo_core_utils.py:_validate_olmo_core_checkpoint` (expects `config.json` + a `model_and_optim/`
  distributed-checkpoint metadata layout). This keeps the common HF path lean while leaving the
  native path available as a real, non-stub escape hatch.

**Context / evidence behind the decision:**

- The Flow-1 plan and skill already carry the same **`CHECKPOINT_KIND` = `hf` | `olmo_core`**
  split; the **HF path is functional**, and the **diagnostics `olmo_core` path is a marked
  `NotImplementedError`** in both `tests/OnNode/checkpoint_infer.py` and
  `diagnostics/mcq_cat/common/inference.py`. Our atom deliberately routes the native path through
  the **real olmo-eval provider** instead of that stub (reuse over reinvent).
- **What real OLMo training emits natively:** OLMo / OLMo-Core trainers write **sharded,
  distributed OLMo-Core checkpoints** (`model_and_optim/` + `.metadata`), **not** HF — HF is
  produced only via an explicit `save_pretrained` / OLMo-Core→HF **conversion**. The confirmed
  strategy is that our training **runs that converter**, so our launcher mostly sees HF; the
  native path exists for checkpoints that skip it.
- **S3 has no real checkpoints yet** — `s3://edullm-adaptive-inference-056956104102/checkpoints/`
  contains only `mock-owner/mock-20260804-142414/step{100,200,300}/`, which is **HF format**
  (`config.json` + `model.safetensors` + `tokenizer.json` + `generation_config.json`) — consistent
  with the HF-primary decision, though not itself proof of the real run's layout.

**Open item to resolve later (not a blocker for P0): WHERE does the OLMo→HF conversion run?**
If training deposits HF copies in S3 (converts before upload), our launcher only ever sees HF and
the `olmo_core` branch is purely the "other teams" escape hatch. If the conversion instead happens
downstream of us — or not at all for some runs — the `olmo_core` branch becomes load-bearing.
Either way the atom is built HF-first with the flag wired; this only changes how often the native
branch is exercised. A **one-time OLMo-Core→HF conversion** right after each checkpoint is written
(so every downstream tool sees uniform HF) remains the clean long-term option (tracked in §8).

---

## 3.7 Execution identity / who-can-run the pipeline (design constraint)

**The goal this protects:** "**anyone on the team can run evals, not just people with AWS
access.**" Today running the pipeline keeps demanding the *human operator's* AWS credentials
(broker-minted, short-lived, MFA-gated). That is friction now — the constant re-auth/MFA prompts
during a run — and a real blocker for the goal above. The key move is to separate the **two
distinct credential surfaces** and treat them differently.

### 3.7.1 Node / instance identity — already fine (machine identity)

The EC2 worker assumes the **`EswManagedInstance`** role **automatically via its instance
profile** (§2, "S3 creds" row: instance profile, no secrets on the box). **Nothing human touches
the node** — the eval already runs under a **machine identity**, which is exactly the right
pattern. No change of shape is needed here.

- **Only gap:** the role policy must grant **S3 read on the checkpoint *source* buckets** (in
  addition to the results-prefix write covered in §5). This is the same class of finding as §1.4 /
  §5 — a **role-policy fix, not a per-user change**. Once granted, the node needs no human
  identity at all.

### 3.7.2 Launcher / orchestrator identity — this is the barrier (human identity today)

`run_checkpoint_eval.sh` (the §3 orchestrator) runs on the **operator's laptop under their own
AWS creds** and needs **privileged** permissions to drive the fleet:

- `ec2:RunInstances`, `iam:PassRole` (to attach the node role to the worker), `ssm:SendCommand`,
  `ec2:TerminateInstances`, plus S3 access.

The recurring MFA/re-auth prompts are a **symptom** of this surface: broker creds are short-lived,
so a long sweep outlives them. For the "anyone can run" goal this fails on two counts:

- **(a) Access.** Not everyone has — or *should* have — `RunInstances` + `PassRole` in the sandbox
  account. Gating eval runs on those grants defeats the goal.
- **(b) Brittleness.** The laptop must stay up and authenticated for the **whole** run, and hits
  **credential expiry** mid-sweep (worse the longer the batch, per §3.5).

### 3.7.3 Direction: move orchestration onto a machine identity

**Decouple "who *triggers* a run" from "who *holds* AWS creds"** by moving orchestration off the
operator's laptop and onto a **machine identity**. Options, cleanest first:

1. **Central service / CI trigger (preferred).** A control-plane — a CI job, a Lambda /
   Step Functions workflow, or a shared runner role — **launches instances under its OWN IAM
   role**. Users just **submit a job** (open a PR, drop a queue message, add a config entry) and
   need **ZERO AWS credentials**. This fully realizes "anyone can run."
2. **Dedicated assumable launcher role (half-measure).** A role scoped to *exactly* the launch
   permissions below, which authorized users `sts:AssumeRole` into. Removes the "privileges live on
   every laptop" problem but **still requires each user to have some AWS identity** to assume it —
   so it does not, by itself, reach "anyone can run."

**Why the retroactive-batch pivot makes the service model natural:** since evals now run
**retroactively over all checkpoints after training finishes** (§3.5, no fire-during-training
hook), a run *is* a **batch job kicked off once** — or **triggered on training completion** — under
a **service role**, not an interactive per-person laptop session. So the **P2 batch driver (§3.5,
§9) should be designed for a service identity FROM THE START**, rather than built for a laptop and
retrofitted later. This also composes cleanly with the dedicated results namespace (§5 / P3):
the service role is the single principal that writes there.

### 3.7.4 Launcher permission checklist (what the service / role needs)

Whichever identity model we pick, the launching principal needs exactly:

- **`ec2:RunInstances`** — ideally **scoped** to the specific AMI (the DLAMI), security group,
  subnet, and instance types (`g6.xlarge` / `g5.xlarge` per §4), not account-wide.
- **`iam:PassRole`** for **`EswManagedInstance`** — so it can attach the node role to workers
  (§3.7.1).
- **`ssm:SendCommand`** (+ **`ssm:GetCommandInvocation`** for the 24 KB-safe poll, §3 phase 6).
- **`ec2:TerminateInstances`** — **note the current tag mismatch:** the role's *self-terminate*
  permission is scoped to `Purpose=smoke-test`, but the checkpoint atom tags instances
  `Purpose=checkpoint-eval`. Either the launcher role's terminate grant must cover the
  `checkpoint-eval` tag, or the atom's tag/self-terminate scoping must be reconciled — otherwise
  explicit teardown (§3 phase 8) is denied.
- **`ec2:DescribeInstances`** — for the ≤3-worker cap preflight (§3 phase 1) and teardown
  confirmation.
- **S3** — **read** on the checkpoint source buckets and **read/write** on the results prefix
  (§5). This mirrors §3.7.1's node-role gap but for the launching principal.

**Not building here:** none of this changes the *atom* (§3) or the *node* identity (§3.7.1). It is
purely about which principal runs the orchestrator, and it is an explicit **P2 decision** (§8, §9).

---

## 4. Hard GPU ceiling (unchanged constraint)

`g6.xlarge` (one L4, 24 GB) is the maximum for real runs; `g5.xlarge` for smoke; anything
larger is forbidden (`AWS_CHECKPOINT_EVAL_PLAN.md` §4). All sizing below respects this — we
**restructure the run, never upsize the GPU**. Also: ≤3 GPU workers at once, and every launch
carries a `shutdown -h +N`.

---

## 5. S3 prefix / grant decision

**Problem:** the checkpoint flow's `olmo-eval-results/*` prefix is **not** in the
`EswManagedInstance` `PutObject` grant (`smoke/* | smoke_split/* | full200/*` only), so
auto-upload will `AccessDenied` exactly as the smoke test did.

**Options:**

- **(A) Reuse an already-granted prefix.** Point checkpoint results under `smoke/…`
  (e.g. `--s3-prefix smoke --s3-group checkpoints/step_N`). Zero AWS change, works today, but
  **pollutes the smoke namespace** with real results and muddies lifecycle/retention.
- **(B) Apply the pre-verified scoped bucket-policy grant** for a dedicated
  `olmo-eval-results/*` prefix. From `HANDOFF` §7, our broker role
  (`Intern-cathy.du-sbsandbox`, capped by `InternSandboxBoundary`) **can** apply a scoped
  bucket policy (`s3:PutBucketPolicy` = allowed; simulated PutObject with `--resource-policy`
  = allowed). It grants ESW `s3:PutObject/GetObject/DeleteObject` on
  `arn:aws:s3:::edullm-adaptive-inference-056956104102/olmo-eval-results/*` (+ `ListBucket`
  with prefix condition). Low blast radius (additive, prefix-scoped, our results bucket), but
  it **mutates shared infra** → gated by auto-review and needs explicit approval.

**Same wall for the `diagnostics/mcq_cat/` path (§1.5).** `common/s3_io.py` uploads results
with **raw `boto3` `put_object`** (not olmo-eval's storage layer, not `--s3-*` flags), so its
`--s3-out` prefix is subject to the **identical `EswManagedInstance` grant**. Likewise
`tests/OnNode/checkpoint_infer.py` defaults to a `checkpoint-infer/` prefix that is **not**
covered by `smoke/*`. Whatever decision we make below applies to **all three** write paths
(`olmo-eval run --s3-*`, `mcq_cat --s3-out`, `checkpoint_infer`): point them at a granted
prefix, or apply the scoped grant for the shared results namespace.

**Recommendation:** **(B) for production checkpoint runs** — the checkpoint flow is long-lived
and deserves a clean, dedicated `olmo-eval-results/` namespace, and the grant is a one-time,
pre-verified, reversible, prefix-scoped addition. **Interim:** while validating the launcher
itself, use **(A)** under a `smoke/checkpoints/…` subprefix so first tests are not blocked on
the approval. Do **not** modify the `EswManagedInstance` role's inline policy (shared role).

---

## 6. The 24 GB / L4 sequential judge constraint

Model-graded evals (TutorEval/TutorBench) need **two** models: the checkpoint (generation) and
the **~9B Qwen judge** (grading). On a 24 GB L4 they **cannot co-reside** (the 9B judge alone
≈ 18 GB in half precision before KV-cache), and upsizing is forbidden. So generation and
grading must run as **two sequential GPU phases on the one box** — one model resident at a
time.

**How this maps onto olmo-eval — and the gap it exposes:**

- olmo-eval's native judge path is **co-resident**: a harness preset lists the judge under
  `HarnessConfig.auxiliary_providers` (`OLMO_EVAL_INTEGRATION_PLAN.md` §2b), the runner starts
  it as a local vLLM server alongside the main provider (`runners/asynq/runner.py:653-666`),
  and `Task.score_responses` grades **immediately after generation, in the same `run`** via
  `provider_name="qwen_judge"`. That assumes enough GPU for *both* models — which we don't
  have on an L4.
- Therefore, honoring the L4 cap means splitting the single co-resident `run` into **two
  sequential `olmo-eval run` invocations on the same instance:**
  - **Phase A — generate.** `olmo-eval run -m <checkpoint> -t <tutorbench-gen>` with
    `--save-predictions` (predictions JSONL) and `--s3-*`. Only the checkpoint is resident.
    Use a task/harness variant whose scorer is a **no-op/deferred** stub so no judge is loaded.
  - **Phase B — judge.** After Phase A's process exits (VRAM freed), start **only** the Qwen
    judge and grade Phase A's saved predictions, writing verdicts back to `--s3-*`. Only the
    judge is resident.

- **This two-phase split is the design's main olmo-eval gap.** olmo-eval today couples
  generation+scoring inside one `run` with a co-resident aux provider; it has **no built-in
  "score previously-saved predictions with a judge-only run" entrypoint**. Options to close it:
  1. **Add a score-from-predictions mode** to olmo-eval (a run/subcommand that loads a judge as
     the *main* provider and scores an existing predictions JSONL). Cleanest long-term; needs a
     small core addition.
  2. **Reuse the eduLLM frozen judge stage** (`eduLLM-Evals/aws_judge_handoff/…run_judge_validation.py`)
     as Phase B over olmo-eval-saved predictions — matches current AWS judge handoff exactly,
     no core change, but keeps grading outside the harness.
  3. **External judge endpoint** (`judge_fn`, OpenAI-compatible): run the judge as a standing
     server (its own `g6.xlarge` or off-box) and let a single olmo-eval `run` call it during
     scoring (this is `AWS_CHECKPOINT_EVAL_PLAN.md` §8 Option B). Avoids the sequential split
     but adds a second box (still ≤3 cap, still self-terminating).

**Recommendation:** default to **sequential two-phase on one box** (§8 Option A of the plan) to
stay self-contained and cheap; pursue core option (1) if we want the judge inside `olmo-eval
run` proper. Confirm the judge's real L4 VRAM footprint before committing (open question).

**Does `diagnostics/mcq_cat/` help here?** Its CAT engine (`common/cat_loop.py`) and IRT code
are for **MCQ log-likelihood** scoring — a *different* eval family from the open-response
LLM-judge, so **do not** reuse `cat_loop`/`inference` for the judge phase. What the judge
phase (and the generate phase) **should** reuse are its plumbing utilities:
`diagnostics/mcq_cat/common/s3_io.py` (`resolve_checkpoint`, `download_prefix`, `upload_files`,
`is_s3_uri`) and the `--checkpoint`/`--s3-out`/`--checkpoint-kind` runner shape, so the two
flows handle S3 checkpoint download + result upload the same way rather than reinventing them.
Note also that MCQ CAT is a **single-model, single-phase** run (no Phase B), so it needs none
of the sequential split — it is effectively "Phase A only."

---

## 7. Gaps vs. the smoke test

| Dimension | Smoke test | Checkpoint eval (what changes) |
|---|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (tiny) | real checkpoint, up to L4-fittable size; **skip/restructure if >24 GB** |
| Model source | HF id, downloaded | `-m s3://…/checkpoints/step_N` via `cached_path` (needs `olmo_core` extra + S3 read) |
| Code staged | clone of **public** `allenai/olmo-eval` | our **fork/branch** that contains the checkpoint-eval tasks (tutorbench/pedagogy/judge) — public clone won't have them |
| Tasks | `arc_easy`, `limit` tiny | full task/suite; larger instance counts and generations |
| Disk | trivial | checkpoint + HF cache on **NVMe** (`/opt/dlami/nvme`, ~229 GB); root disk (~19 GB) is too small |
| Timeout | short | longer eval + **judge phase** → raise `EVAL_TIMEOUT` and size `shutdown -h +N` accordingly |
| GPU | `g5.xlarge` ok | `g6.xlarge` (real) |
| Scoring | metric only | **LLM-judge phase** (sequential, §6) |
| S3 prefix | `smoke/` (granted) | `olmo-eval-results/` (**needs grant**, §5) |

---

## 8. Open questions

1. **Judge VRAM on L4** — confirm the ~9B Qwen judge actually fits in 24 GB at the intended
   precision/context; locks in §6 (sequential vs. external endpoint).
2. **Largest evaluable checkpoint on an L4**, and the policy when a checkpoint exceeds 24 GB
   (skip / lower precision / shard) — upsizing is not allowed.
3. **Score-from-saved-predictions**: build the olmo-eval core entrypoint (§6 option 1) or keep
   Phase B as the eduLLM frozen judge stage (option 2)? This is the biggest core decision.
4. **Code staging**: clone our repo@ref vs. tar+presigned-URL of the working tree. Clone is
   simpler and matches Beaker; tar avoids needing the eval ref pushed. (The custom tasks don't
   exist yet — see `OLMO_EVAL_INTEGRATION_PLAN.md`; the launcher can be validated on stock
   tasks first.)
5. **S3 grant approval** — get sign-off to apply the scoped `olmo-eval-results/*` bucket-policy
   grant (§5 option B), or accept the `smoke/` interim.
6. **Task suite per checkpoint** — the full eval suite on every checkpoint, or a fast subset on
   most with the full suite on a sampled set, across the retroactive batch
   (`AWS_CHECKPOINT_EVAL_PLAN.md` §11).
7. **Which on-node command per eval family?** MCQ CAT via `diagnostics/mcq_cat/runner.py`
   (offline, no olmo_eval dep) vs. `olmo-eval run` core (tasks + judge). The launcher can drive
   both; do we standardize the result-key convention (`--s3-out` vs `--s3-prefix/--s3-group`)
   across them? See also `OLMO_EVAL_INTEGRATION_PLAN.md` (2026-08-04 MCQ CAT reconciliation note).
8. **Checkpoint format + `olmo_core` loading (decision recorded in §3.6).** **Resolved
   (2026-08-05):** HF is the **primary, default, validated** format (training runs the OLMo→HF
   converter); native OLMo-Core is an **optional safety valve** behind `CHECKPOINT_KIND=olmo_core`,
   which conditionally installs `ai2-olmo-core` and selects the **real** `OlmoCoreProvider` — not
   the diagnostics stub (`NotImplementedError` in `diagnostics/mcq_cat/common/inference.py` and
   `tests/OnNode/checkpoint_infer.py`), which stays unused on our path. **Still open:** (a) **where
   the OLMo→HF conversion runs** — if training deposits HF copies in S3, the launcher only ever
   sees HF and the `olmo_core` branch is purely the "other teams" escape hatch; if conversion is
   downstream of us, that branch becomes load-bearing; (b) whether to add a one-time OLMo-Core→HF
   conversion so all consumers see uniform HF.
9. **Execution / launch identity: service role vs per-user (decision due at P2, §3.7).** How does
   the *orchestrator* authenticate — a central **service / CI machine identity** (users submit a
   job, need zero AWS creds) or a **dedicated assumable launcher role** (users still need an AWS
   identity to assume it)? The node already runs under a machine identity (`EswManagedInstance`,
   §3.7.1); the open decision is the *launcher* surface (§3.7.2–§3.7.3). Must be settled when the
   P2 batch driver is scoped (§9) so it is designed for a service identity from the start, not
   retrofitted. Also folds in the `ec2:TerminateInstances` tag mismatch noted in §3.7.4.
10. **Task-suite granularity per checkpoint (batch driver, §3.5).** Beyond the whole-vs-subset
    question of Q6, is the eval unit **one task-suite per checkpoint** (the current §3.5 shape), or
    should the driver shard at a finer **(checkpoint × task)** granularity across the fleet — e.g.
    to parallelize a heavy suite over multiple workers for a single checkpoint? Not settled; the
    P2.a enumeration keeps the unit at the checkpoint level.
11. **Auto-descend for `RUN_URI` (enumeration, §3.5.2).** The approved rule requires `RUN_URI` to
    point **directly** at the `step*` parent (no descend). Should a later convenience mode
    optionally auto-descend (e.g. into a `checkpoints/` child) when no `step*` children are found at
    `RUN_URI`? Deferred — explicit-`RUN_URI` is the P2.a behavior; auto-descend would be additive.
12. **Claim-queue vs static shard-index partition (fleet, §3.5.3).** P2 uses a **static**
    `ami-launch-index` round-robin partition (simple, no coordination). At higher fleet widths or
    with very uneven per-checkpoint runtimes this can leave workers idle while others still churn; a
    **claim-queue** (workers pull the next unclaimed step) would rebalance. Deferred — revisit if
    the static partition proves too skewed as fleet width / checkpoint counts grow.

---

## 9. Phased implementation checklist (cheapest → fullest)

Evals run retroactively (§3.5), so the build order proves the **per-checkpoint atom** first,
then promotes the **batch driver** to core, then layers namespace and judge on top.

- **P0 — HF-first atom, `olmo_core` flag wired but not yet validated (cheap, no judge, stock
  task).**
  Copy `run_rung1_smoketest.sh` → `run_checkpoint_eval.sh`; parameterize `CHECKPOINT`/`TASKS` and
  add **`CHECKPOINT_KIND` (default `hf`)** per the §3.6 decision. The **HF path works end-to-end**
  — keep `g5.xlarge` + a tiny public model + `arc_easy` + `--s3-prefix smoke/checkpoints`, lean
  install, and prove the **single-checkpoint** launch/stage/install/run/poll/verify/teardown still
  passes with an **idempotent skip-if-done** guard. The **`olmo_core` branch is wired but NOT yet
  validated**: `CHECKPOINT_KIND=olmo_core` conditionally installs the `olmo_core` extra
  (`ai2-olmo-core`) and selects the real `OlmoCoreProvider` (`-o provider.kind=olmo_core`) — a
  coherent, runnable branch, left to be exercised in P1 against a real native checkpoint. **No AWS
  change.**
- **P1 — Real checkpoint load, generation only (still one checkpoint).**
  Point `-m` at a small **real `s3://…/checkpoints/step_N`** (or HF id), add the `olmo_core`
  extra, stage checkpoint + HF cache on NVMe, `g6.xlarge`, larger disk/timeout. Verify
  predictions land in S3 (still under a covered `smoke/` subprefix). Confirms checkpoint
  loading + sizing.
- **P2 — Batch / back-fill driver (CORE — §3.5, detailed in §3.5.1–§3.5.6).**
  Enumerate a run's `step_N` checkpoints in S3 (§3.5.2) and run the P0/P1 atom across **all** of
  them, threading the per-batch `CHECKPOINT_KIND` (§3.5.5), the `_SUCCESS`/manifest idempotency
  and observability (§3.5.4), and the identity-agnostic entrypoint (§3.5.6). Ordered sub-steps:
    - **P2.a — enumerate + identity-agnostic entrypoint (NO GPU).** ✅ **Implemented +
      offline-validated (2026-08-05)** in `run_checkpoint_batch.sh`. Parse inputs (§3.5.1), do the
      `^step_?\d+$` + numeric-sort + `STEP_FILTER` enumeration with list-only format detect
      (§3.5.2), resolve creds ambiently (§3.5.6), and under `--dry-run` print the resolved config +
      ordered `(step, uri, kind)` list and what it *would* dispatch — launching **no** instance.
      Validate `FLEET_SIZE ≤ 3` as an inert check. Enumeration proven on both real S3 layouts
      (bucket-root `step7629`; nested `.../checkpoints/step125…1315`), incl. the format-mismatch warn.
    - **P2.b — K=1 multi-checkpoint loop (amortized install).** ✅ **Implemented + live-validated
      (2026-08-05):** one `g6.xlarge`, install paid **once**, all 3 mock checkpoints
      (step100/200/300) evaluated with `_SUCCESS` + `metrics.json` (`done=3`), clean teardown (~$0.17).
      Also added a **pre-launch all-done gate** (zero-GPU skip, confirmed) and a **subnet/instance-type
      capacity fallback** + exit-code fix. One worker: install once, loop all enumerated checkpoints
      running the §3 atom, skip-if-done via `_SUCCESS` (§3.5.4).
    - **P2.c — ≤3 IMDSv2 shard-index fleet.** ✅ **Implemented + live-validated (2026-08-05):**
      K=2 fleet proved fan-out + install-once amortization (w0=step100+step300, w1=step200), all 3
      steps `_SUCCESS` (arc_easy 0.3/0.3/0.1), and clean teardown of both instances. (A pipefail
      bug found on the first run was fixed and the re-run confirmed `done=3` — see resolved note
      below.) Generalize to `FLEET_SIZE ≤ 3` workers, each reading `ami-launch-index` via IMDSv2 and
      looping its static *i, i+K, …* slice (§3.5.3). Kept as a **separate** node-driven path from the
      validated laptop-driven K=1 `dispatch_k1` (intentional, to avoid regressing K=1).
    - **P2.d — aggregation / reporting.** ✅ **Implemented + validated (2026-08-05):** merges
      per-worker shards into `_batch/summary.json` and correctly tallied `total=3, done=2, failed=1`
      in the K=2 run (robust to missing/partial shards). Collect per-worker manifest shards into
      `_batch/summary.json` and exfil per-checkpoint logs on failure (§3.5.4).
  Build the **sequential *K = 1* case first** (P2.b), then generalize to the **≤3-worker IMDSv2
  shard-index fleet** (P2.c) (`AWS_CHECKPOINT_EVAL_PLAN.md` §7). Confirm the ~10-min install is
  paid **once per worker**, not per checkpoint. This is the retroactive sweep — the central
  mechanism, not a late add-on. Test cheaply with 2–3 tiny checkpoints before a full run.
  - ⚠️ **MUST decide execution identity here (service role vs assumable launcher role) — see the
    Execution identity section (§3.7).** The batch driver is the natural home for a **machine /
    service identity** (a run is a batch job kicked off once, or on training completion — not an
    interactive laptop session), so design it for a service identity **from the start** rather than
    retrofitting one. This also settles the launcher permission checklist and the
    `ec2:TerminateInstances` tag mismatch (§3.7.4), and it feeds P3: the service role is the single
    principal that writes to the dedicated results namespace.
  - ✅ **Resolved bug — P2.c fleet `pipefail` SIGPIPE false-negative (found + fixed 2026-08-05).**
    On the first live K=2 run, `step300` (worker-0's *2nd* slice item) was misreported `failed`
    despite a fully-successful eval (`worker_0.json` showed `outcome:"failed"` with `exit_code:0`,
    and the exfil'd `eval.log` had metrics written + uploaded, no traceback). **Root cause:** the
    on-node loop's done-check `aws s3 ls "$prefix" --recursive | grep -q <marker>` runs under
    `set -o pipefail`; `grep -q` exits on first match and closes the pipe, the still-writing
    `aws s3 ls` takes SIGPIPE (141), and `pipefail` propagates it as a false "not-done" — timing-
    dependent, so it only bit a worker's 2nd+ checkpoint. **Fix:** capture the listing into a
    variable first, then `grep -q <marker> <<<"$listing"` (no live writer to SIGPIPE), in both the
    skip-check and post-eval verify. The laptop-side `verify_step` already captured-first, so K=1 was
    never affected. **Re-run confirmed `done=3`** (single attempt on step_300).
- **P3 — Dedicated results namespace.**
  Get approval and apply the scoped `olmo-eval-results/*` bucket-policy grant (§5 B); switch
  `--s3-prefix olmo-eval-results`; re-verify a `PutObject` succeeds. Once execution identity is
  settled (§3.7 / P2), grant the write on this prefix to the **service/launcher principal**, so the
  namespace has a single owning identity.
- **P3.5 — mcq_cat as second driver command (pluggable-hook proof).**
  Wire the `diagnostics/mcq_cat/` path (§1.5) into the batch driver's pluggable on-node command
  (`python -m diagnostics.mcq_cat.runner --cat-style <name> --checkpoint … --s3-out …`), reusing the
  same enumeration / fleet / skip-if-done / aggregation machinery. Validate on an **HF checkpoint**
  (native `olmo_core` load is still a `NotImplementedError` stub, §8 #7-8). Sequenced here because it
  depends on the P3 results namespace (mcq_cat's `common/s3_io.py` writes there too) and is the
  **cheapest** second eval family — single-model, single-phase, no judge — so it de-risks the
  driver's command-swap hook before the heavier P4 judge path. Bump ahead of the tutorbench/tutoreval
  integration if the team treats the Pedagogy MCQ benchmark as the priority deliverable.
- **P4 — Judge phase (sequential, per checkpoint).**
  Add Phase B (§6): after generation, load the Qwen judge and grade the saved predictions
  (via the chosen mechanism — core score-from-predictions or the eduLLM frozen stage). Validate
  verdict parity on a few cached cases. Requires the custom tutorbench/judge tasks from
  `OLMO_EVAL_INTEGRATION_PLAN.md`.

Test cheaply first: P0/P1 on `g5`/tiny model cost well under ~$1 per run; don't touch AWS
policy or the judge until the plumbing is proven.

---

## 10. Key file references

- Beaker seam / launcher: `src/olmo_eval/launch/beaker/launcher.py`
  (`BeakerJobConfig.command` `:359`, `launch()` `:974-1105`, `_build_install_cmd` `:792-972`,
  CUDA constraint trick `:889-891`, AWS secret inject `:1029-1034`).
- Beaker CLI (option → command forwarding): `src/olmo_eval/cli/beaker/launch.py` (S3 opts
  `:115-131`).
- Reusable core: `src/olmo_eval/cli/run/__init__.py` (S3 opts `:86-90`, `StorageSetup`
  `:204-218`).
- Model/checkpoint resolution: `src/olmo_eval/common/configs.py:152-195`;
  `src/olmo_eval/inference/providers/olmo_core_utils.py:204-390` (`cached_path` checkpoint read).
- Judge provider + scorer: `OLMO_EVAL_INTEGRATION_PLAN.md` §2; runner aux-provider startup
  `src/olmo_eval/runners/asynq/runner.py:653-666,803-821`; exemplar
  `src/olmo_eval/evals/tasks/harmbench.py:133-146`.
- Proven orchestrator to generalize: `tests/aws/run_rung1_smoketest.sh`; poll/24 KB fix and S3
  grant context: `tests/aws/HANDOFF_beaker_to_aws.md` §6–§7.
- Companion self-contained eval path (§1.5): `diagnostics/mcq_cat/runner.py` (CLI),
  `diagnostics/mcq_cat/base.py` (`CatStyle` contract), `diagnostics/mcq_cat/common/{s3_io,
  inference,cat_loop,irt_params,benchmark_download}.py`, plan
  `Plan/mcq_cat_diagnostics/README.md`; mirrored S3/checkpoint patterns in
  `tests/OnNode/checkpoint_infer.py`.
