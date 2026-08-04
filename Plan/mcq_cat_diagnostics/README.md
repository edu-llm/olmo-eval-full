# MCQ CAT diagnostic branches (Step 3, MCQ)

This is the build plan for the multiple-choice diagnostics called out in Step 3 of the
parent [`Plan/README.md`](../README.md) (the `Uni-MCQ` and `MIRT-MCQ` branches, and any
other MCQ CAT styles). It layers Computerized Adaptive Testing (CAT) on top of the
already validated inference plumbing from Flow 1 and Flow 2. Inference is assumed to
work; this plan is about how several MCQ CAT styles are built in parallel and merged
back together without merge conflicts.

## Problem

We want several styles of MCQ CAT (for example unidimensional 1PL/2PL/3PL and
multidimensional MIRT, with different item-selection policies). Each style needs the
same three pieces: download the benchmark, load IRT parameters, and run inference /
scoring on a checkpoint. Each style will be developed on its own branch and pull
requested back into one shared integration branch.

The risk is merge conflicts. Conflicts come from two branches editing the same lines of
the same file. So the whole strategy is: make every style additive (new files only),
and freeze anything shared before parallel work starts.

## Do not use fill-in placeholders in shared files

A tempting approach is to put stubs in a shared file (a central `if style == ...`
dispatch, a `STYLES = [...]` list, a shared config block, a pyproject entry-points
table) that each branch fills in. Do not do this. Every style branch would edit the same
region of the same file, which conflicts on the second and later merges. A placeholder
that multiple branches fill in the same spot is a conflict generator, not a conflict
preventer.

The correct form of a placeholder here is a stable extension point: an interface plus an
auto-discovering registry, where filling the slot means adding a new file, not editing a
shared line.

## Architecture: interface + registry + per-style directories

Three tiers:

1. Stable contract (integration branch, written once, then frozen). An abstract base or
   Protocol that every style implements, for example `download_benchmark()`,
   `load_irt_params()`, `score()`, `select_next_item()`, `stopping_rule()`, `report()`.

2. Shared utilities (integration branch, stable). The roughly 80 percent every MCQ CAT
   reuses: S3 download, checkpoint load plus batched log-likelihood MCQ scoring, IRT
   parameter loading, the generic CAT loop (select, administer, update theta, check
   standard-error stop), and result writing to S3. Written once so styles neither
   reimplement nor edit them.

3. Per-style modules (each on its own branch, its own directory). Each style lives in
   `styles/<name>/` and self-registers. Adding a style means adding new files only.

Proposed layout:

```
diagnostics/mcq_cat/
  base.py            # the interface every style implements        (frozen)
  registry.py        # register() + get_cat(name); auto-discovers styles/  (frozen)
  runner.py          # CLI: --cat-style NAME --checkpoint ... --s3-out ...  (frozen)
  common/            # shared, stable utilities
    benchmark_download.py
    irt_params.py
    inference.py     # checkpoint load + batched loglikelihood MCQ scoring
    cat_loop.py      # generic select / administer / update / stop engine
    s3_io.py
  styles/
    uni_2pl/   __init__.py   # @register("uni_2pl")   <- one branch adds only this dir
    uni_3pl/   __init__.py   # @register("uni_3pl")   <- another branch, only this dir
    mirt/      __init__.py   # @register("mirt")        <- another branch, only this dir
```

The key detail that gives zero shared-file edits: the registry discovers styles by
scanning the `styles/` directory and importing modules that call `@register(...)`. There
is no central list to edit, so `runner.py` resolves `--cat-style mirt` without being
modified when a style is added. A hardcoded dispatch or a pyproject entry-points table
would make that single file the conflict point; directory auto-discovery avoids editing
any shared file.

## Strategy pattern for the styles

The MCQ CAT styles differ mainly on two axes: the IRT model (1PL/2PL/3PL,
unidimensional versus MIRT) and the item-selection policy (for example Fisher
information). Download, parameter loading, checkpoint inference, and the CAT loop are
common. So the integration branch owns the generic CAT engine, and each style supplies
only the IRT model and the selector. This shrinks each style to the genuinely different
bits, which both reduces duplication and reduces the code that could ever conflict.

## Branch and PR workflow

1. Create one MCQ CAT integration branch off `CheckpointFlows` that holds the frozen
   scaffolding (`base.py`, `registry.py`, `runner.py`, `common/`). All style branches
   pull request back into this branch. It later merges up to `CheckpointFlows`.
2. Land and freeze the scaffolding first, then cut every style branch from that commit.
   Stabilizing the interface before parallel work is the single most important step.
3. Each style branch adds only `styles/<name>/` (implementation, its own `config.yaml`,
   and its own tests) and self-registers. Because paths are disjoint, the style PRs
   merge in any order with no conflicts.
4. If a style needs a change to a shared utility or the interface, treat it as a smell.
   Make it a small, separate PR to the integration branch first, merged and serialized,
   so shared code is never edited inside two style branches at once.

## Merge-conflict avoidance rules

- Styles add files under `styles/<name>/` only. They never edit shared files.
- No central registry list, no shared dispatch: discovery is by directory scan.
- Per-style config lives in `styles/<name>/config.yaml`, not one shared config file.
- Put `common/` and `base.py` under CODEOWNERS so shared changes get review.
- Commit the lint and format config on the integration branch up front so formatters do
  not reformat shared files and create spurious conflicts.
- Rebase style branches on the integration branch periodically. Drift stays near zero
  because paths are disjoint.

## Deliverables

```
diagnostics/mcq_cat/base.py, registry.py, runner.py   # frozen scaffolding
diagnostics/mcq_cat/common/*                           # shared stable utilities
diagnostics/mcq_cat/styles/<name>/*                    # one per style, added on its branch
Plan/mcq_cat_diagnostics/README.md                     # this plan
```

## Sequence at a glance

1. Cut an MCQ CAT integration branch off `CheckpointFlows`.
2. Build and freeze the scaffolding: interface, auto-discovering registry, runner, and
   `common/` utilities (benchmark download, IRT params, checkpoint loglikelihood
   scoring, CAT loop, S3 IO).
3. Cut one branch per CAT style from that commit. Each adds only its `styles/<name>/`
   directory and self-registers.
4. Pull request every style back into the integration branch. They merge without
   conflicts because they touch disjoint paths.
5. Merge the integration branch up once the styles are in.

## Out of scope

- Free-response (FRQ) CAT styles, which follow the same pattern on their own branches.
- Benchmark selection and full-suite scoring beyond what a CAT run needs.
