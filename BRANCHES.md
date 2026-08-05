# Branch map

Fork of `allenai/olmo-eval`. Branches:

- `main` — upstream sync base + eduLLM platform registration. Default branch.
- `CheckpointFlows` — deliverable for training teams: run our evals on checkpoints
  (live-training + retroactive inference flows). Diagnostic sub-branches merge in:
  - `flow/uni-mcq`, `flow/mirt-mcq`, `flow/uni-frq`, `flow/mirt-frq`
- `frq-lab` — lean shared FRQ calibration toolchain (`tutor_cat/` + scripts). Parent of:
  - `frq/<bench>` — per-benchmark calibration labs: `tutorbench`, `tutoreval`, `biggen`,
    `bridge`, `dr-sci`, `wildbench`, `ap-ib`, `edubench`, `infobench`. Messy by design.
    A finalized calibration is snapshotted (`<bench>/FLOW_PACKAGE.md` + a `*-cal-*` tag)
    and its strictly-necessary bank graduates into the matching `flow/*-frq` branch.
- `Research` — MCQ calibration work.
