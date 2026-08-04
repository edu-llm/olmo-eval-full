# Checkpoint Flows

Tooling for running olmo-eval on model checkpoints, both live during a training run and retroactively on checkpoints already saved to S3. The work is staged on purpose: first prove that inference can run on a checkpoint and land its output in S3, then layer the IRT diagnostics on top, then package everything as skills that the training teams can use directly.

## Guiding principle

Prove inference before adding diagnostics. Every flow starts as a minimal TEST that only runs inference on a checkpoint and writes the output to S3. Full diagnostics (scoring, IRT and CAT, benchmark selection) are added only after that test flow works end to end.

## Step 0: Document the current olmo-eval flow

Before building anything, document exactly how the existing system works for model checkpoints:

- How olmo-eval is invoked for a checkpoint: entry points, commands, and arguments.
- Which scripts are involved along the full path, and what each one does.
- What already exists in the current evals suite: benchmarks, scoring, configs, and infrastructure.

Deliverable: a written reference of the current invocation path and a complete inventory of the suite, so the two flows below reuse what already exists instead of duplicating it.

## Step 1: Build the two base flows (inference only)

Both flows begin as inference-only tests. The only goal at this stage is to prove that a checkpoint can be loaded, run inference, and have its output saved to S3. No diagnostics yet.

### Flow 1: Training checkpointing flow (live)

A training team adds a few simple commands to their existing training run script. Those commands:

- Provide the S3 location where the run is saving checkpoints.
- After each checkpoint is created, invoke the eval script to run TEST inferences on that checkpoint and save the output to S3.

The integration for the training team should be minimal: a small hook that fires once per checkpoint.

### Flow 2: Retroactive checkpoint flow (existing checkpoints)

For checkpoints that already exist in S3:

- A per-checkpoint script that takes a single checkpoint, runs inference, and saves the output to S3.
- A manager script that runs the per-checkpoint script across all checkpoints, with the option to parallelize across multiple GPUs.

## Step 2: Validate both flows

- Run a mock training run to exercise Flow 1 from checkpoint creation through inference output in S3.
- Use existing checkpoints to exercise Flow 2, first on a single checkpoint and then with the manager across many.
- Confirm both flows reliably produce inference output in the expected S3 locations before moving on.

## Step 3: Diagnostic branches (branched off this branch)

Once the base flows are validated, create one branch off CheckpointFlows per diagnostic type. Each branch adds its scoring and IRT or CAT diagnostic on top of the shared inference flows:

- Uni-MCQ: unidimensional IRT on multiple-choice benchmarks.
- MIRT-MCQ: multidimensional IRT on multiple-choice benchmarks.
- Uni-FRQ: unidimensional IRT on free-response benchmarks.
- MIRT-FRQ: multidimensional IRT on free-response benchmarks.

## Step 4: Setup skills for training teams

Create a skill for each flow that helps a training team set it up without needing to know the internals. Each skill should let them specify:

- Exactly which diagnostics they want to run.
- Which full benchmarks they want.
- Location and details: S3 paths, checkpoint layout, GPU and parallelism settings, and any run-specific options.

## Sequence at a glance

1. Document the current olmo-eval checkpoint flow.
2. Build Flow 1 (training) and Flow 2 (retroactive) as inference-only tests.
3. Validate both with a mock training run and existing checkpoints.
4. Branch off for diagnostics: Uni-MCQ, MIRT-MCQ, Uni-FRQ, MIRT-FRQ.
5. Package each flow as a setup skill for training teams.
