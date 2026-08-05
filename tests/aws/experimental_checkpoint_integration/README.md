# ⚠️ EXPERIMENTAL — AWS ↔ CheckpointFlows integration (NOT approved, may be discarded)

> **STATUS: EXPERIMENTAL / SPECULATIVE. Do not rely on anything in this folder.**
>
> - **Created:** 2026-08-04
> - **Origin:** Auto-generated proposal by an assistant subagent. It has **NOT** been
>   reviewed or discussed with the team.
> - **Decision:** Whether to adopt, change, or delete this is **pending a team discussion**.
>   It **might never be used.**
> - **Do NOT** wire this into the real `Plan/` or `tests/OnNode/` flow, import it from
>   production code, or run it against AWS until the team has signed off.

---

## What this folder is

A **sandbox / holding area** for a proposed way to connect our AWS EC2+SSM+S3+teardown
launch mechanism (from the Rung 1 smoke test, see `../`) to the `CheckpointFlows` Flow 1
checkpoint-eval workload (`tests/OnNode/checkpoint_infer.py`).

The idea being explored: generalize the smoke-test orchestrator so that instead of running
`olmo-eval run` on a public task, it can launch `checkpoint_infer.py` against a **training
checkpoint** on a GPU node (parameterized model/checkpoint path → stage → run → collect →
self-terminate), reusing the verified fixes (`HOME=/root`, NVMe caches, instance-type guard,
teardown trap).

## Why it is quarantined here

- It's an **early proposal**, not a decision.
- The real integration touches team-owned structure (`Plan/`, `tests/OnNode/`) and the
  open **S3 `PutObject` IAM** gap — both need team input first.
- Keeping it in one clearly-labeled folder means it can be **deleted in one step** if the
  team decides not to use it, with zero impact on the tracked CheckpointFlows layout.

## Contents

- `README.md` — this banner (do not remove the EXPERIMENTAL framing).
- _(other files below are the auto-generated proposal — treat as drafts):_

<!-- The subagent's proposed launcher/docs are (re)generated into this folder. -->

## If the team says NO

Delete the whole folder:

```bash
git rm -r tests/aws/experimental_checkpoint_integration/   # if committed
# or, if never committed:
rm -rf tests/aws/experimental_checkpoint_integration/
```

## If the team says YES

Promote the relevant pieces into the proper CheckpointFlows locations (see the proposal's
own notes and `../HANDOFF_beaker_to_aws.md` §10 "Suggested next steps"), then remove this
experimental folder.
