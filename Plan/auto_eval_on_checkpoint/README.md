# Plan — evaluate checkpoints automatically as training writes them

Goal: a user starts a training run, the eval skill is invoked once at the start, and from
then on every Nth checkpoint is diagnosed without anybody typing anything. Not built yet.
This file defines the transition and names the decisions that have to be settled first.

Settled by the user, 2026-08-08:
- The watcher lives **on the training node**, beside training, and dies with the job.
- A **pre-approval path exists** for the platform's lead gate; details to be supplied.
- Cadence is **every Nth checkpoint**, N configurable.

## The fork that "on the training node" still contains

"On the training node" is two designs, not one, and they differ in almost every property.

**A — evaluate in process, on the training node's own GPU.** This is what Arhant's
`add-cat-evals` skill does: the training script calls a diagnostic right after a checkpoint
is saved, backgrounded so it does not block the next step.

**B — submit a platform job from the training node.** The node detects the checkpoint and
dispatches a submission, which runs elsewhere.

**B is probably not possible, and this is the first thing to verify.** `AGENTS.md` states
the platform holds every AWS credential in a workflow whose trust policy pins it to one
file on `main`, and a workload's role may write its own output prefix and read the team's
run space — nothing else. A training container therefore has no credential with which to
dispatch `submit-run.yml`, and `edullm` needs `gh` logged in and a clone with an origin
remote, neither of which a research image carries. If that is right, B collapses into "the
watcher runs on the user's laptop", which is not the training node and is a different plan.

**A looks strictly better here, for a reason specific to this workload.** An MCQ CAT is
about two seconds of GPU: run `run_019fe277` administered eight items between 18:28:34.9
and 18:28:35.6, against roughly a minute of setup. Everything expensive about the platform
path is setup — image pull, clone, pip install, and a 1.74 GB checkpoint sync that on the
training node is a local directory that was just written. A also sidesteps the lead gate
entirely, sidesteps the pinned-sha problem below, and costs no separate instance.

What A costs: the CAT briefly shares the training GPU. At two seconds per checkpoint
against a 3814-step save interval, that is not a real cost, but it is not zero and it must
be measured rather than assumed — the model has to be loaded from the checkpoint, which is
the part that takes seconds rather than milliseconds.

**Phase 1 is deciding this**, and it is a read plus one experiment rather than a debate.

## What A requires that we do not have

**A completion marker.** A checkpoint directory appears before it is finished. Preston's
`step305176` holds `.metadata.json` at 40 bytes beside `config.json` and
`model_and_optim/`, and whether that file is written last is unknown and decidable from
OLMo-core's checkpointer. Evaluating a half-written checkpoint produces either a crash or,
worse, a theta from partial weights. Do not infer completion from directory existence or
mtime.

**Idempotency.** The watcher must not re-evaluate what it has already done. The sweep
scripts already solve this with `_READY` / `_FAILED` markers per checkpoint, and the same
shape works here.

**Failure isolation, which is the highest-order requirement.** The watcher must never kill
training. Not on a bad checkpoint, not on an OOM, not on a missing bank, not on a network
failure reaching the HuggingFace Hub for a tokenizer. Every eval runs in a child process
whose death is logged and ignored. A training run lost to its own instrumentation is worse
than no instrumentation.

**A cadence rule.** Every Nth checkpoint, N configurable. The default should be argued from
the save interval rather than picked: Preston saved every 3814 steps and produced 82
checkpoints, so N=1 is 82 evals and N=5 is 17. Note that CAT's whole claim is cheapness —
eight items against a 650-item bank — so N=1 is more defensible here than it would be for a
full benchmark sweep.

## What changes if B turns out to be possible

Then the pinned-sha problem becomes load-bearing. The container clones olmo-eval-full at
one commit, so an automated submitter has to pin something, and the honest choice is the
sha as of training start — pinned once, reused for every checkpoint in the run, recorded in
every report. A watcher that resolved a branch tip per submission would silently change the
measuring instrument mid-run, which is the same class of error as re-vendoring a bank
between checkpoints.

The pre-approval path also becomes load-bearing rather than a convenience. Every run so far
has parked at `run-approval-lead` despite `edullm check` predicting `automatic`, and 82
checkpoints times four benchmarks is 328 approvals. Get the mechanism in writing before
designing around it.

## Phases

| # | Phase | Needs | Verified by |
|---|---|---|---|
| 1 | Decide A vs B | reading `AGENTS.md`, one credential probe on a node | a written answer, not an assumption |
| 2 | Completion marker | reading OLMo-core's checkpointer | the marker named, and its write order proven |
| 3 | Watcher, offline | nothing | unit tests over a fake directory that fills up |
| 4 | In-process eval hook | a GPU | one checkpoint diagnosed without disturbing training |
| 5 | Cadence, markers, isolation | nothing | a killed child leaves training running |
| 6 | Skill integration | nothing | the skill starts the watcher at training start |

Phases 1 and 2 are reads. Phase 3 is the whole watcher with the eval stubbed, which is
where the failure-isolation tests live and where a fake directory can be made to misbehave
in ways a real one will not on demand.

## Open questions

1. **Is B possible at all?** Does a research image carry `gh`, and does a workload role
   hold anything that can dispatch a workflow? Phase 1.
2. **What marks a checkpoint complete?** Phase 2, and it decides whether the watcher can be
   trusted.
3. **What is the pre-approval mechanism**, in writing? Only matters under B.
4. **Does the in-process eval disturb training?** Memory, and whether loading a second copy
   of the model beside an optimizer state fits. Measure on the smallest real case.
5. **Where do results go under A?** The training run's own output prefix is the obvious
   answer, since the workload role may write it, but that couples eval results to a
   training run's lifecycle and makes cross-run comparison a later problem.
6. **What happens to a checkpoint written while an eval is still running?** Queue, skip, or
   evaluate late. Skipping is defensible under a cadence rule and is the simplest.
