# FRQ CAT diagnostic branches (Step 3, FRQ)

This is the build plan for the free-response diagnostics called out in Step 3 of the
parent [`Plan/README.md`](../README.md) (the `flow/uni-frq` and `flow/mirt-frq` branches).
It is the FRQ counterpart of [`Plan/mcq_cat_diagnostics/README.md`](../mcq_cat_diagnostics/README.md)
and follows the same rule: layer Computerized Adaptive Testing (CAT) on top of the
already-validated checkpoint inference plumbing, and build the styles in parallel so they
merge back with no conflicts. Inference on a checkpoint is assumed to work; this plan is
only about the FRQ scaffolding and how the two FRQ styles plug into it.

## Current state (read this first)

- The **MCQ** scaffolding already exists and is frozen at `diagnostics/mcq_cat/`
  (`base.py`, `registry.py`, `runner.py`, `common/`, empty `styles/`). Its
  [design doc](../mcq_cat_diagnostics/README.md) describes the interface + auto-discovering
  registry pattern this plan reuses verbatim. The `flow/uni-mcq` and `flow/mirt-mcq`
  branches already have a home and add only `styles/<name>/`.
- There is **no equivalent FRQ scaffolding on `CheckpointFlows`**. The
  `diagnostics/frq_cat/` files currently visible here (`full_bank_judge.py`,
  `full_bank_respgen.py`, `replay/`, `reports/`) are run helpers/artifacts that *import* a
  full `frq_cat` package (`registry`, `base`, `common/bank_loader`, `judge`, `checkpoint`,
  `styles/frq`) which lives on the `frq-lab` / `frq/*` branches, not here. So the FRQ
  scaffolding described below must be built and frozen on `CheckpointFlows` **before** the
  two FRQ branches cut.

## Problem

We want two FRQ CAT styles that share everything except their IRT model and item
selector: `uni-frq` (unidimensional 1PL/2PL/3PL) and `mirt-frq` (multidimensional MIRT).
Each is developed on its own branch and pull requested straight back into
`CheckpointFlows`.

The risk is merge conflicts, which come from two branches editing the same lines of the
same file. The whole strategy is therefore: make every style additive (new files only)
and freeze everything shared before parallel work starts.

## What makes FRQ different from MCQ

MCQ scoring is a single in-process step: batched log-likelihood over the answer choices
yields a correct/incorrect signal. FRQ grading is **two stages**:

1. **Response generation (respgen).** The checkpoint under test generates a free-response
   answer for each scenario. This uses the checkpoint served as a vLLM endpoint (the
   "tutor" endpoint), matching the existing `full_bank_respgen.py` convention
   (greedy/deterministic, `temperature=0`, fixed seed).
2. **LLM-as-a-judge.** A separate, frozen judge model (Qwen `generic-binary`) grades each
   response against every rubric criterion and emits a verdict, which the engine maps to
   the correct/graded signal IRT consumes. One judge call == one criterion, matching the
   existing `full_bank_judge.py` contract, including its calibration provenance
   (frozen HF revision, prompt/normalization/evidence policy versions).

Both stages are **shared** and belong in the frozen scaffolding. The only genuinely
style-specific pieces stay the same as MCQ: the IRT model (ability estimator) and the
item/criterion selector. The uni-vs-MIRT axis is expressed entirely through those two
pieces plus the IRT parameter files; the shared contract already supports scalar-or-tuple
ability and multi-dimensional discriminations.

## Decision record

These were confirmed before writing this plan:

- **Branches:** `flow/uni-mcq`, `flow/mirt-mcq`, `flow/uni-frq`, `flow/mirt-frq`.
- **Topology:** style branches PR straight into `CheckpointFlows` (scaffolding is frozen
  there); no per-modality integration branch.
- **Judge:** shared and frozen in `frq_cat/common/judge.py`; each FRQ style ships only a
  judge `config.yaml`, so the calibrated judge identity is defined once.
- **Invocation seam:** a new `tests/aws/run_checkpoint_diag.sh` that reuses the atom's
  launch/stage/install/teardown and swaps the run command for the diagnostics runner (see
  [`Plan/checkpoint_diagnostics_wiring/README.md`](../checkpoint_diagnostics_wiring/README.md)).
- **FRQ execution:** served vLLM endpoints (tutor = checkpoint under test, judge = frozen
  Qwen), like the existing `full_bank_*` helpers.

## Do not use fill-in placeholders in shared files

Same rule as the MCQ plan. Do not add a central `if style == ...` dispatch, a
`STYLES = [...]` list, a shared config block, or a `pyproject` entry-points table that
each branch fills in — every style branch would edit the same region and conflict on the
second merge. The correct extension point is an interface plus an auto-discovering
registry, where filling the slot means adding a new file, not editing a shared line.

## Architecture: interface + registry + per-style directories

Three tiers, identical in shape to `mcq_cat`:

1. **Stable contract (written once, then frozen).** An abstract base every FRQ style
   implements: `download_bank()`, `load_irt_params()`, `select_next_item()`,
   `estimate_ability()`, `stopping_rule()`, `report()`. Grading (respgen + judge) is driven
   by the shared engine, not reimplemented per style.
2. **Shared utilities (stable).** The ~80% every FRQ CAT reuses: scenario/rubric bank
   loading, respgen against the served checkpoint, the frozen LLM-as-a-judge client, IRT
   parameter loading, the generic FRQ CAT loop, and result/provenance writing to S3.
3. **Per-style modules (each on its own branch, its own directory).** Each style lives in
   `styles/<name>/` and self-registers. Adding a style means adding new files only.

Proposed layout:

```
diagnostics/frq_cat/
  base.py            # the FRQ interface + dataclasses every style implements   (frozen)
  registry.py        # register() + get_cat(name); auto-discovers styles/        (frozen)
  runner.py          # CLI: --cat-style NAME --checkpoint ... --tutor-endpoint   (frozen)
                     #      --judge-endpoint ... --s3-out ...
  common/            # shared, stable utilities
    bank_loader.py   # scenario + rubric/criterion bank load + sha (from frq-lab)
    respgen.py       # generate responses from the checkpoint-under-test (served vLLM)
    judge.py         # frozen LLM-as-a-judge client + verdict parsing (shared, calibrated)
    irt_params.py
    cat_loop.py      # generic engine: select -> respgen -> judge -> update theta -> stop
    s3_io.py
  styles/
    uni_frq/  __init__.py   # @register("uni_frq")   <- flow/uni-frq adds only this dir
    mirt_frq/ __init__.py   # @register("mirt_frq")  <- flow/mirt-frq adds only this dir
```

The registry discovers styles by scanning `styles/` and importing modules that call
`@register(...)`. There is no central list to edit, so `runner.py` resolves
`--cat-style mirt_frq` without being modified when a style is added.

## Strategy pattern for the styles

The FRQ styles differ on exactly two axes: the IRT model (unidimensional 1PL/2PL/3PL vs
multidimensional MIRT) and the item/criterion-selection policy (e.g. Fisher information).
Bank loading, respgen, judging, and the CAT loop are common. So the scaffolding owns the
generic engine and the shared judge, and each style supplies only its IRT model, its
selector, and its `config.yaml` + IRT parameter file. This shrinks each style to the
genuinely different bits, which minimizes the code that could ever conflict.

## Contents of a style branch: the calibration graduation package

What a style branch actually adds is modeled on a real finalized release: the TutorEval ->
`flow/uni-frq` **graduation package** (`eduLLM-Evals/data/TutorEval/FLOW_PACKAGE.md` on
`frq/tutoreval`). A calibration lab on `frq/<bench>` snapshots a finished calibration
(`<bench>/FLOW_PACKAGE.md` + a `*-cal-*` / `*-<N>models` tag), and only the
**strictly-necessary payload** graduates into the matching `flow/*-frq` branch. The
calibration scripts, the response matrix, staging, and non-canonical variants stay behind
(these releases deliberately do **not** include all the scripts).

Each style branch therefore adds two things, under one disjoint path
`diagnostics/frq_cat/styles/<name>/`:

**(A) The graduated calibration payload** (the data, ported into the frozen loader's
schema). Using TutorEval's package as the template:

| Artifact | Role | Style-branch home |
|---|---|---|
| Fitted IRT params bank (canonical) | the deployment bank: per-item discrimination `a` (keyed by modeled skill) + difficulty `b` + `q_modeled` — e.g. TutorEval's 1,186-item `rubrics_qmatrix_final_unidim_fitted.jsonl` | `styles/<name>/bank/params.jsonl` |
| Items / scenarios | the prompts the checkpoint-under-test responds to (`scenarios_final.jsonl`) | `styles/<name>/bank/scenarios.jsonl` |
| Judge config | selects the shared, frozen judge (`judge_frozen.yaml`); the judge code itself is NOT forked | `styles/<name>/config.yaml` |
| `FLOW_PACKAGE.md` (provenance) | snapshot tag, fit method, `n_persons`, matrix `sha256`, locked config, caveats, rerun/swap protocol | `styles/<name>/FLOW_PACKAGE.md` |
| Dimensionality evidence | why uni vs MIRT was chosen (skillfit figures + caption) | `styles/<name>/evidence/` |

**(B) The thin style code** (the genuinely style-specific behavior the scaffolding calls):

```
diagnostics/frq_cat/styles/<name>/
  __init__.py        # @register("<name>") — the only registration point
  style.py           # IRT model (estimate_ability) + selector (select_next_item) + stopping_rule
  config.yaml        # locked config: fit family + CAT stop/selection policy + judge selection
  bank/              # graduated payload: params.jsonl + scenarios.jsonl
  FLOW_PACKAGE.md    # graduation provenance (tag, n_persons, sha256, caveats, rerun protocol)
  evidence/          # dimensionality figures backing the uni/MIRT choice
  tests/             # style-owned tests (schema-load + a tiny CAT smoke)
```

Nothing outside `styles/<name>/` is touched, so the style PRs merge in any order.

### The bank must conform to the frozen loader schema

TutorEval's package flags its own layout as *provisional* and says to "conform the file
arrangement to the FRQ diagnostic loader (`diagnostics/frq_cat/common/irt_params.py`)"
before committing into `flow/uni-frq`. That makes the loader schema a hard dependency of
every graduation: `common/bank_loader.py` + `common/irt_params.py` define the on-disk bank
schema (skill-keyed `discrimination`, scalar `difficulty`, `q_modeled`), and each style
ports its lab artifacts into that schema. So the bank/judge loaders must be **frozen first**
(before any branch cuts); the port is data-shaping, never a loader edit.

The uni-vs-MIRT difference shows up here as a bank shape, not new engine code: unidim ships
`q_modeled = {"ability": 1}` with a scalar discrimination; MIRT ships a multi-skill
`q_modeled` with a per-skill discrimination vector. The frozen `IRTBank` already models both
(scalar-or-tuple discrimination, `dimensions`), so `mirt-frq` is a different payload + a
MIRT estimator/selector, not a different contract.

### Runtime data flow (scoring a new checkpoint), from the package

1. Generate tutor responses from the checkpoint on the style's `scenarios.jsonl` (respgen).
2. Grade per criterion with the style's `judge_frozen.yaml` selection (shared judge).
3. Feed graded outcomes + the fitted bank (a/b params) into the style's CAT loop to estimate
   ability (uni: scalar `ability`; MIRT: per-skill vector) with standard error.

### What does NOT graduate (stays on `frq/<bench>`)

- Calibration/fitting scripts and the EM export tooling.
- The response matrix (a calibration input; lands in S3, not the style branch).
- Non-canonical variants (e.g. the 2-skill bank when unidim is the shipped instrument).
- `staging/`, coverage reports, and other lab intermediates (referenced by `FLOW_PACKAGE.md`,
  not copied).

### MCQ parity

MCQ style branches ship the same package shape **minus the judge and respgen**: a fitted
MCQ IRT bank + the benchmark items + `FLOW_PACKAGE.md` + evidence, under
`diagnostics/mcq_cat/styles/<name>/`. MCQ grading is in-process log-likelihood over choices,
so there is no `judge_frozen.yaml` and no served tutor endpoint.

## Branch and PR workflow

1. Build and freeze the FRQ scaffolding (`base.py`, `registry.py`, `runner.py`,
   `common/`) on `CheckpointFlows`, alongside this plan. Stabilizing the interface before
   parallel work is the single most important step.
2. Cut `flow/uni-frq` and `flow/mirt-frq` from that frozen commit. Each adds only its
   `styles/<name>/` directory and self-registers.
3. Pull request both straight back into `CheckpointFlows`. Because paths are disjoint they
   merge in any order with no conflicts.
4. If a style needs a change to a shared utility or the interface, treat it as a smell:
   make it a small, separate PR to `CheckpointFlows` first, merged and serialized, so
   shared code is never edited inside two style branches at once.

## Merge-conflict avoidance rules

- Styles add files under `styles/<name>/` only. They never edit shared files.
- No central registry list, no shared dispatch: discovery is by directory scan.
- Per-style config and IRT params live under `styles/<name>/`, not one shared file.
- The judge is defined once in `common/judge.py`; styles select it via config, never fork it.
- Put `common/`, `base.py`, `registry.py`, `runner.py` under CODEOWNERS.
- The ruff config is already committed on `CheckpointFlows`, so formatters do not reformat
  shared files and create spurious conflicts.
- Rebase style branches on `CheckpointFlows` periodically; drift stays near zero because
  paths are disjoint.

## Deliverables

```
# Frozen scaffolding (this branch, before any style cuts)
diagnostics/frq_cat/base.py, registry.py, runner.py   # frozen scaffolding
diagnostics/frq_cat/common/*                           # shared utilities (incl. frozen judge +
                                                       #   the bank/IRT loader schema packages conform to)
Plan/frq_cat_diagnostics/README.md                     # this plan

# Added per style branch, as a graduated calibration package (disjoint path)
diagnostics/frq_cat/styles/<name>/style.py, __init__.py, config.yaml
diagnostics/frq_cat/styles/<name>/bank/{params.jsonl, scenarios.jsonl}
diagnostics/frq_cat/styles/<name>/FLOW_PACKAGE.md      # tag, n_persons, sha256, caveats, rerun protocol
diagnostics/frq_cat/styles/<name>/evidence/*           # dimensionality figures
```

## Sequence at a glance

1. Build and freeze the FRQ scaffolding: interface, auto-discovering registry, runner, and
   `common/` (bank loader, respgen, frozen judge, IRT params, CAT loop, S3 IO). **Freeze the
   bank/IRT loader schema explicitly** — it is the contract every graduation package conforms
   to at port time.
2. Wire the style-agnostic checkpoint invocation seam and CODEOWNERS (see the wiring plan).
3. Smoke both runners in `--dry-run` on a tiny checkpoint before cutting branches.
4. Cut `flow/uni-frq` and `flow/mirt-frq` from the frozen commit. Each ports its calibration
   graduation package (fitted bank + scenarios + judge config + `FLOW_PACKAGE.md` + evidence)
   into its `styles/<name>/` directory, conforming the layout to the frozen loader, and adds
   its thin `style.py`.
5. Pull request both back into `CheckpointFlows`; they merge without conflicts.

## Out of scope

- MCQ CAT styles (covered by the existing `mcq_cat` scaffolding and its plan).
- Benchmark/IRT calibration itself (done on `frq-lab` / `frq/*`); this plan consumes the
  fitted banks and parameters, it does not fit them.
- Full-suite FRQ scoring beyond what a CAT run needs.
