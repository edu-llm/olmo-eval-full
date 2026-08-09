# `Avaneesh-Test-Ticket` — runs

Preston's `step305176`, a 135M native OLMo-core checkpoint, evaluated to exercise the ticket
workflow itself. The submission was made by an agent following
[`eval-cat`](../../.cursor/skills/eval-cat/SKILL.md) with no prior knowledge of it, so this
ticket is evidence about the instructions as much as about the checkpoint.

A fan-out gets one line per cell: they share a run id and a harness sha, and the benchmark
column is what distinguishes them.

| run id | harness sha | OLMo-core sha | benchmark | outcome |
| --- | --- | --- | --- | --- |
| `run_019fe873` (probe) | `ff1188ec` | `08df5aa0` | — | Step 1 generation probe, one pass, 56 findings and 0 failed. `--dtype bfloat16` reaches the weights; decodes degenerately and never emits EOS, which licensed the MATH cell as a pipeline exercise rather than a measurement |
| `run_019fe87d` cell-0 | `ff1188ec` | `08df5aa0` | `arc_challenge` | theta -2.1189, se_mwle 0.2727, 8 items, `precision_reached`, 0 ungradable |
| `run_019fe87d` cell-1 | `ff1188ec` | `08df5aa0` | `hellaswag` | theta -3.0851, se_mwle 0.2343, 9 items, `precision_reached`, 0 ungradable |
| `run_019fe87d` cell-2 | `ff1188ec` | `08df5aa0` | `musr` | theta -0.6621, se_mwle 0.0600, 8 items, `precision_reached`, 0 ungradable |
| `run_019fe87d` cell-3 | `ff1188ec` | `08df5aa0` | `bbh` | theta -3.8557, se_mwle 0.2752, 24 items, `precision_reached`, 0 ungradable. Carries a `bank_caveat` — read it before quoting the theta |
| `run_019fe87d` cell-4 | `ff1188ec` | `08df5aa0` | `gpqa` | theta -0.2497, se_mwle 0.2534, 12 items, `precision_reached`, 0 ungradable. **Constant-"A" responder**: `chosen_index` 0 on all twelve, 6 correct by letter prior. Not knowledge |
| `run_019fe87d` cell-5 | `ff1188ec` | `08df5aa0` | `leaderboard_math` | theta -1.7024, se_mwle 0.7672, 40 items, `max_items_reached`, 0 ungradable. **Quote nothing from it** — see below |

## What these numbers do and do not support

**The five MCQ cells are real measurements** and two of them reproduce earlier runs to four
significant figures, which is stronger evidence that the pipeline is deterministic than a
merely green result would be. Thetas are low because a 135M checkpoint sits far below the
0.5B–7B population these banks were calibrated on; low is expected, and it is inversion or
structural impossibility that would have been interesting.

**`leaderboard_math` measures extraction, not mathematics.** Zero of 40 completions contain
`\boxed{` or `Final Answer`, observed accuracy is 0.0, and the decode loops one clause to the
cap. Its `theta_online`/`se_online` equal the bank-only prior to four significant figures — the
session learned nothing, and the estimate is the prior it started from.

**`ungradable` is 0 in all six, including MATH, and that is the trap the field is for.** The
count records items the grader could not score, not items the model failed to answer; a
completion that parses cleanly and contains no answer is gradable and wrong. A clean rate is
therefore not evidence that a generative cell measured anything.

**`ability.standard_error` here is `se_mwle`**, an asymptotic `1/sqrt(I(theta))`, because the
run asked for `batch_eap+mwle`. The stopping rule acted on `se_online`, a posterior SD. They
are not on one scale, so the column above must not be read against the 0.3 threshold. The
inversion is live in this run: `musr` looks sharpest on `se_mwle` at 0.060 but is 0.172 online,
while `bbh` is genuinely tightest online at 0.108 across its 24 items.

## Two notes on the record itself

**The harness sha is `ff1188ec`, which was a bad commit** — it reverted `SKILL.md` from 1019
lines to 903. It changed that file and nothing else, so the code this run executed is identical
to `fa70d46c`'s and the results stand. It is pinned here rather than corrected because the
point of a pin is to name what ran. The restore is `4e6b6e84`.

**The branch name does not conform to the convention.** `[a-z0-9][a-z0-9._-]*-ticket` is
lowercase, and `Avaneesh-Test-Ticket` is not; the preflight refused it and the trial reached for
`--any-branch` rather than lowercasing. That was the finding the name was chosen to produce, so
the branch is left as it is — renaming it now would orphan the sha this record pins.
