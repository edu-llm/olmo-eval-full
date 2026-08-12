# MRBench Phase B on eduLLM — compliant plan (OLMoE-1B-7B-0125-Instruct)

Planning doc only. **Nothing here has been executed on a GPU or against AWS.** All
platform specifics that require the `edullm` CLI (not installed on this machine)
are marked **UNVERIFIED — confirm from `edullm check --json` / `edullm <verb> --help`**.

Goal: evaluate `allenai/OLMoE-1B-7B-0125-Instruct` on the `mrbench` task (Phase B —
the model-under-test generates 192 tutor remediations, which are judged on 8
dimensions → DAMR), the validated Claude-Haiku judge from Phase A.

Recommended shape: **generate on eduLLM (GPU, no secret, `--dataset none`), judge
locally** (reuse the already-validated Haiku judge + `.apienv`). Rationale below.

> **Status (admin-independent CODE prereqs): DONE + offline-validated.** All three
> code changes that need no GPU / AWS / edullm / judge-API are implemented on this
> branch and validated against the mock-model + stubbed-judge smoke: **(1)** the
> `MRBENCH_GENERATE_ONLY` gate (§2b-i), **(2)** the wheel-safe data-path resolver
> (`-o data_source=` / `MRBENCH_DATA_SOURCE` / checkout fallback, §2b-ii), and **(3)**
> the local join→judge→DAMR script `diagnostics/mrbench/phase_b_judge.py` (§2c). Still
> open (all gated on the admin/edullm, not on code): the **[BLOCKER]** torch/vLLM image
> flip (#1) and the **[UNVERIFIED]** edullm CLI surface, artifact fetch without AWS, and
> model registration (#4–#7).

---

## 0. AGENTS.md constraints as they apply to us

From `AGENTS.md` (the `edullm:begin…end` block, managed by edu-llm/platform):

- **Use `edullm`, never AWS.** No `boto3`, no `aws` CLI, no `curl` at an AWS
  endpoint — not to launch, not to fetch results. The only supported path to the
  cluster is `edullm`.
- **The platform takes a commit, not a working tree.** The image is built from the
  **last commit**; anything uncommitted is not in the run, and **only a push to a
  branch named `edullm/<something>` builds an image at all.** → our MRBench code must
  be committed onto an `edullm/<name>` branch.
- **`edullm check --json` first.** Free, no network, lists every refusal at once.
  **Match on `code`**, act on the `detail` (detail is human-worded, don't match it).
  Exit codes: `0` ok · `1` refused on the merits · `2` the command/install is wrong ·
  `3` platform unreachable (the only one worth retrying).
- **Read stdout on its own.** The first `check` in a repo with no `.edullm/run.yaml`
  writes one and says so **on stderr** — do **not** `2>&1 | …` that first call.
- **`--dataset none` ≠ omitting `--dataset`.** Pass the literal `none` when the run
  reads no corpus. MRBench Phase B reads no training corpus (it reads the vendored
  conversations file, not a platform dataset) → **`--dataset none`**.
- **Write the dtype into the command text.** The `bfloat16_not_in_the_hardware` guard
  reads the command string, not code. Name `bfloat16` literally in the command.
- **`edullm status --json` is free / pollable; `edullm status` (no json) and
  `edullm logs` are slow — never in a loop.**
- **Never quote a price / runtime bound / cost ceiling / approver from memory or a
  doc** (including this one). Read `cost` and `approval_class` out of
  `edullm check --json`.
- **Never** `--force` past a refusal, **never** edit `.edullm/run.yaml` to silence a
  refusal without reading it, **never commit a secret** (`.apienv` stays uncommitted;
  it is already gitignored).
- **`run` / `shell` are the exploration route** ("ships this tree to a machine of your
  own… no run anybody can cite"); **`submit` is the recorded/citable path.**
- Install/upgrade is exactly `uv tool install --force git+https://github.com/edu-llm/platform`
  (re-running it IS the upgrade). `edullm` needs `gh` logged in and an `origin` remote;
  no AWS profile/SSO/VPN.

## 0b. `.edullm/` findings (from `origin/main`)

- Only **`.edullm/Dockerfile`** is tracked; **there is no committed `.edullm/run.yaml`**
  on `origin/main` (the first `edullm check` will generate one — expected, on stderr).
- **Base image:** `nvidia/cuda:12.8.1-runtime-ubuntu24.04`, pinned by digest in the
  platform's `config/repositories.yaml` (passed in as `BASE_IMAGE`).
- Image installs extras **`beaker hf s3 clients`** with `--no-default-groups` (drops
  dev + vllm). `clients` = the OpenAI-compatible backend (`litellm`).
- **⚠️ The load-bearing finding — the registered default image has NO torch.**
  `ARG INSTALL_TORCH_AND_VLLM=0` is the default and "is what every build has produced
  so far": **only `mock` and `litellm` are runnable; `vllm`, `vllm_server` and
  `huggingface` (the three that load weights onto a card) are not.** Building at `1`
  adds torch+vLLM, but the comment states **"the platform's build workflow passes
  `BASE_IMAGE` and nothing else, so this cannot be set per build"** — flipping it is a
  **one-line default-branch (`main`) change + rebuild + re-register**. See Blocker #1.
- The image does `COPY . .` into `/opt/olmo-eval` **and** `uv pip install --no-deps .`.
  So the running task is the **installed wheel** under site-packages, while the repo
  tree (incl. `diagnostics/`) also sits at `/opt/olmo-eval`. This matters for the
  MRBench data/diagnostics path — see Blocker #3.
- The image asserts task `*.jsonl` data travels in the wheel, `olmo-eval --help`
  works, and `mock`/`litellm` construct; at `=1` it additionally asserts torch(+CUDA)
  and vLLM import.

## 0c. `edullm` CLI status

**Not installed** on this machine (`edullm --version` → not found). Per the task
constraints I did **not** install it. Every command/flag below is therefore written
from AGENTS.md and marked **UNVERIFIED**; ground them with `edullm check --json` and
`edullm <verb> --help` once installed.

---

## 1. Branch / commit workflow

The image = last commit on an `edullm/<name>` branch. Our MRBench code currently
lives on `frq/mrbench` and is **untracked** (`src/olmo_eval/evals/tasks/mrbench.py`,
the whole `diagnostics/mrbench/` tree incl. the CC BY-SA `MRBench_V1.json` + its
`NOTICE.md`, and the `mrbench` registry entry via `@register("mrbench")`).

Steps (no push executed here):
1. Create the platform branch off our work: `git switch -c edullm/mrbench-phaseb`
   (or branch from `frq/mrbench`).
2. **Commit the code that must be in the image:** `src/olmo_eval/evals/tasks/mrbench.py`,
   `diagnostics/mrbench/**` (including `data/MRBench_V1.json` + `data/NOTICE.md`), and
   any `pyproject.toml`/registry changes. **CC BY-SA note:** the dataset is
   redistributable under CC BY-SA 4.0 with attribution (already in `NOTICE.md`) — fine
   to commit; keep the NOTICE beside it.
3. **Do NOT commit `.apienv`** (gitignored; the judge key never enters the image).
4. Exclude the bulky local judge cache: `diagnostics/mrbench/runs/` is already
   gitignored, so the 10 MB Phase-A cache won't ride along.
5. Push `edullm/mrbench-phaseb` (this is what triggers the image build). **UNVERIFIED**
   whether the push alone builds, or `edullm submit` triggers the build for the pushed
   commit — confirm from `edullm submit --help`.

## 2. Recommended: generate on eduLLM, judge locally (the split)

**Why split:** an in-cluster full `mrbench` run would (a) need the judge key as a
secret in the cluster (AGENTS.md gives no sanctioned way to inject an arbitrary API
key — see §3), and (b) call the judge over the network from a GPU job that is billed
while it waits. Generating only, with **no secret** and **`--dataset none`**, keeps
the GPU job cheap and secret-free; the judge then runs locally against the already
validated Haiku config + `.apienv` (the exact Phase-A path).

### 2a. What the generation step must produce
`olmo-eval` writes per-instance predictions JSONL locally (and, on the platform, into
the run's committed output). Record shape (`runners/io/builders.py:103-117`,
`runners/io/writers.py`): each line has

```json
{"doc_id": 0, "native_id": "<conversation_id>",
 "model_output": [{"text": "<tutor response>", "extracted_answer": null, ...}],
 "final_output": "<tutor response>", "instance_metrics": {...}, "label": ...}
```

`native_id` = the MRBench `conversation_id` (the task sets `metadata["id"]`);
`final_output` (and `model_output[0].text`) = the generated tutor remediation. That is
all the local judge needs — history / ground-truth / data-source are re-joined from the
local `MRBench_V1.json` by `conversation_id`. (The **requests** JSONL additionally
embeds the full instance metadata, if a self-contained artifact is preferred.)

### 2b. Two code gaps that block a clean generate-only run today

The `mrbench` task **could not separate generate from score as originally written** —
`score_responses` **always** judged (imported `diagnostics.mrbench`, called the judge,
needed `OPENAI_API_KEY`) and ran automatically after generation. Both gaps below are
now **[CODE] DONE** on this branch — minimal, opt-in, default-OFF (validated offline
against the mock-model + stubbed-judge smoke; the default inline-judge path is
unchanged):

- **(i) Generate-only gate — [DONE].** Env flag **`MRBENCH_GENERATE_ONLY`** (truthy =
  `1`/`true`/`yes`/`on`). When set, `score_responses` returns immediately after answer
  extraction — **no diagnostics import, no judge call, no API, no network** — and the
  runner still writes the predictions JSONL with `native_id` (= `conversation_id`) and
  `final_output` (the tutor text), which is all the local judge needs. Default OFF ⇒
  behaviour is identical to the inline-judge path. (Implemented in
  `src/olmo_eval/evals/tasks/mrbench.py`.)
- **(ii) Data path under the installed wheel — [DONE].** The task now resolves the data
  file through a small precedence chain (`_resolve_data_path`): **(a)** explicit
  `-o data_source=/abs/MRBench_V1.json`; **(b)** env var **`MRBENCH_DATA_SOURCE`**;
  **(c)** fall back to the checkout path (`Path(__file__).parents[4]/diagnostics/...`).
  So both a source checkout (fallback path) and an installed wheel (pass an explicit
  path, e.g. `-o data_source=/opt/olmo-eval/diagnostics/mrbench/data/MRBench_V1.json`,
  which exists in the image via `COPY . .`) work; `/opt/olmo-eval` is **not** hardcoded.
  A missing file raises a clear error naming both override mechanisms. The top-level
  diagnostics import stays guarded (dimension-list fallback keeps task *discovery*
  resilient), and under generate-only the judge import is never reached.
  *(Durable alternative, still deferred: bundle `MRBench_V1.json` as package data and
  load via `importlib.resources` to drop the `parents[4]` dependency entirely.)*

  *(Note: the task-level `required_secrets=("OPENAI_API_KEY",)` is consumed by the
  **beaker launcher** to mount secrets, not enforced as a hard gate by the local
  `AsyncEvalRunner`; the real thing that needs the key is the judge call inside
  `score_responses`, which the generate-only gate skips. Confirm the edullm run path
  doesn't separately enforce task secrets — UNVERIFIED.)*

### 2c. Local judging — [DONE] `diagnostics/mrbench/phase_b_judge.py`
The **local** scorer is implemented at **`diagnostics/mrbench/phase_b_judge.py`** (not
needed in the image). It:
1. reads the generation predictions JSONL (from the generate-only run's output) via
   `--predictions <path>`,
2. for each record joins `conversation_id = native_id`, `answer = final_output`, and
   looks up `conversation_history` / `Ground_Truth_Solution` from the local
   `MRBench_V1.json`,
3. **reuses** `judge_prompt.build_messages` + `judge_client.score_messages` +
   `parse.parse_result` (the identical Figure-6 path — the judge is NOT reimplemented),
   maps labels to `reference.DESIRED_LABELS`, and computes per-dimension + aggregate
   **DAMR** via `metrics.compute_judge_damr` (with optional `--bootstrap` CIs from
   `metrics.bootstrap_ci_rate`),
4. caches every call idempotently via `judge_cache.JudgeCache` (key
   `(conversation_id, <model>, dimension)`) so re-runs skip completed calls, and can
   write the DAMR result JSON via `--out`.

Modes mirror `judge_run.py`: **default = dry run** (join stats + sample prompts + cost
estimate via `cost.py` rates, **ZERO network**); `--metrics` computes DAMR from an
existing cache (no network); `--live` is gated on a key + real model id in the env.
Judge env comes from `.apienv` (`source .apienv`) exactly as in Phase A
(`OPENAI_BASE_URL`, `MRBENCH_JUDGE_MODEL`, `OPENAI_API_KEY`) — consumed at runtime, never
read/quoted/hardcoded here. **Phase-B is DAMR only** (no AC / macro-F1 — there is no
human gold for a model-under-test's fresh responses).

```bash
# dry run (join + cost, no network)
uv run python -m diagnostics.mrbench.phase_b_judge --predictions <preds.jsonl>
# DAMR from an existing cache (no network)
uv run python -m diagnostics.mrbench.phase_b_judge --predictions <preds.jsonl> --metrics
# live (needs key + MRBENCH_JUDGE_MODEL in env)
uv run python -m diagnostics.mrbench.phase_b_judge --predictions <preds.jsonl> --live
```

## 3. Alternative: secret-in-cluster (only if the platform sanctions it)

AGENTS.md describes **no** general API-key injection: every AWS credential lives in a
workflow trust-pinned to one file on `main`, and **committing a secret is explicitly
forbidden**. `edullm add` teaches the platform about a *repository / dataset / shape /
model / person* (not secrets); `edullm ask` "files one ask for something you need."
So unless a platform skill documents a sanctioned secret mechanism (**UNVERIFIED** —
check edu-llm/platform `skills/README.md`), there is **no compliant way** to give the
judge key to an in-cluster run without committing it. **Recommendation: use the split
in §2.** If (and only if) a sanctioned per-user secret exists, the full in-cluster
`mrbench` run would additionally need fixes (ii) above and torch/vLLM in the image (#1).

## 4. Exact `edullm` invocations (UNVERIFIED — confirm via `--help`)

1. **Price + list refusals (do this first, read stdout alone):**
   ```bash
   edullm check --json            # first run writes .edullm/run.yaml (note on stderr)
   ```
   Read `cost` and `approval_class` from the JSON — **do not** quote them from memory.
2. **Generation submission (recorded path).** Conceptually a `submit` whose command
   runs olmo-eval in generate-only mode, names the dtype, and declares no corpus:
   ```bash
   edullm submit --dataset none \
     --model allenai/OLMoE-1B-7B-0125-Instruct \
     -- bash -lc 'MRBENCH_GENERATE_ONLY=1 olmo-eval run \
        -m allenai/OLMoE-1B-7B-0125-Instruct \
        -o provider.kind=vllm -o provider.dtype=bfloat16 \
        -t mrbench -o data_source=/opt/olmo-eval/diagnostics/mrbench/data/MRBench_V1.json'
   ```
   - `bfloat16` appears **literally** in the command (satisfies the dtype guard).
   - `provider.kind=vllm` **requires the torch/vLLM image** (Blocker #1).
   - `--dataset none` (no corpus).
   - **UNVERIFIED:** the exact `edullm submit` surface — whether model/dtype/dataset are
     flags, or all live in `.edullm/run.yaml`, and how the run command is expressed.
     Confirm with `edullm submit --help` and by reading the generated `run.yaml`.
3. **Hardware / dtype:** request a card that supports **bfloat16** — **A10G (Ampere
   sm_86)** and **L4 (Ada sm_89)** both support bf16, so naming `bfloat16` is safe on
   either. (Which card/`cost`/`approval_class` you get: read from `edullm check --json`.)
4. **Verb choice:** use **`submit`** for a citable generation run; use `run`/`shell`
   only for throwaway exploration (nothing checked/priced/recorded).
5. **Monitor:** `edullm status --json` (free, pollable). Avoid bare `edullm status` /
   `edullm logs` in a loop.

## 5. Refusals to expect (match on `code`; authoritative list = `check` output)

AGENTS.md names a few; the full, current set prints beside each refusal in
`edullm check --json` (do not hand-maintain a table here):

- `unregistered_repository` → **won't arise** (this repo is registered); if it ever
  does, the one-off **registering-a-repository** skill covers it.
- `bfloat16_not_in_the_hardware` → the command names bf16 but the priced card lacks it;
  pick a bf16 card (A10G/L4) or drop the dtype. Named-in-command turns a dead machine
  into a free refusal.
- `unregistered_stage_reference` → a Dockerfile `FROM`/stage not reachable from the
  pinned base; **not ours** (the committed `.edullm/Dockerfile` is compliant).
- A **`--dataset` refusal** if it's omitted → pass the literal `none`.
- Anything else → **match `code`, act on `detail`** (which names the field/file). Exit
  `2` = fix the command/install; exit `3` = retry (platform unreachable).

**Not a `check` refusal but a paid runtime failure:** the default (torch-less) image
silently cannot construct `vllm`/`hf` — `check` prices the commit and cannot see the
image contents, so this only surfaces after the machine is admitted. Resolve #1 before
submitting a GPU generation.

## 6. Open questions / blockers

1. **[BLOCKER] Image has no torch/vLLM by default.** The registered image is
   `INSTALL_TORCH_AND_VLLM=0` (mock/litellm only). Loading OLMoE weights (`vllm`/`hf`)
   needs `=1`, which **cannot be set per-build** and is a **one-line default change on
   `main` + rebuild + re-register** — a maintainer/admin action. Until then no GPU
   generation of OLMoE is possible via `submit`. *(Alternative that needs no image
   change: serve OLMoE behind an OpenAI-compatible endpoint and use `litellm` — but
   that is not "run the model on a GPU on eduLLM" and needs an endpoint + its own key.)*
2. **[CODE — DONE] Generate-only mode** (§2b-i): `MRBENCH_GENERATE_ONLY` gate added so
   the GPU run is secret-free and never imports the judge. Opt-in, default-OFF,
   validated offline against the mock smoke.
3. **[CODE — DONE] Data path under the installed wheel** (§2b-ii): `_resolve_data_path`
   precedence `-o data_source=` → `MRBENCH_DATA_SOURCE` env → checkout fallback. Works
   for checkout *and* wheel (pass `-o data_source=/opt/olmo-eval/...`); `/opt/olmo-eval`
   is not hardcoded. The local judge (§2c) is implemented as
   `diagnostics/mrbench/phase_b_judge.py`. (The package-data / `importlib.resources`
   variant is still deferred.)
4. **[PLATFORM/UNVERIFIED] Artifact retrieval without AWS.** How do we pull the 192
   generations (predictions JSONL) from a `submit` run back to the laptop for local
   judging, given "never AWS"? Need the sanctioned mechanism (an `edullm` fetch verb or
   a documented location) — confirm from `edullm <verb> --help` / platform skills.
5. **[PLATFORM/UNVERIFIED] Model registration.** Does `allenai/OLMoE-1B-7B-0125-Instruct`
   need `edullm add` (model/shape registration), or is it pulled from HF at runtime
   (the `hf` extra is present; OLMoE-Instruct is public/ungated)? Read from `check`
   refusals.
6. **[PLATFORM/UNVERIFIED] Secret mechanism.** Confirm whether any sanctioned per-user
   secret exists (it gates §3). Absent evidence, use the split in §2.
7. **[ENV] `edullm` not installed here.** Install via the AGENTS.md line before any
   `check`/`submit`; needs `gh` logged in + an `origin` remote.

---

*See also: [`PHASE_A_VALIDATION.md`](./PHASE_A_VALIDATION.md) (the validated Haiku
judge this reuses) and [`README.md`](./README.md) (§ olmo-eval integration).*
