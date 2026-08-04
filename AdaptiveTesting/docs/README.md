# ATLAS × olmo-eval integration docs

Reference docs written to scope integrating **ATLAS adaptive testing (IRT-based
CAT)** into the **olmo-eval** suite so that a model launched on Beaker can be run
through an ATLAS eval and reported by ability (θ) + p-IRT accuracy instead of (or
alongside) full-benchmark accuracy.

**Current status: adaptive testing is on hold.** The client scrapped CAT in favour of
full-benchmark accuracy runs. Nothing was torn out — the adaptive core, the banks, and
the `atlas_*` tasks all still work — but if you are here to *use* this, read
[`05_cat_deferred.md`](05_cat_deferred.md) first: it records why CAT was viable for
HellaSwag only, the switch from 3PL to 2PL, and the measured limits of the item bank.

Docs 01–03 are the map and assessment written while scoping the work; doc 04
records the implementation that followed, which is **built and verified end to end
for ARC-Challenge**; doc 05 is the deferral record. Read them in order:

| Doc | What it covers |
|---|---|
| [`01_olmo_eval_architecture.md`](01_olmo_eval_architecture.md) | How the outer `olmo-eval` suite is built and how an eval actually runs — task registry, runner, providers, scoring, suites, Beaker launch, external evals, results DB. The integration surface. |
| [`02_atlas_and_adaptive_testing.md`](02_atlas_and_adaptive_testing.md) | What ATLAS is (3PL IRT + Fisher-info CAT + p-IRT), the vendored ATLAS repo, the teammate's calibration/validation experiments already in `AdaptiveTesting/`, the standalone inference harness, and all the data schemas + the item↔question_id bridge. |
| [`03_integration_assessment.md`](03_integration_assessment.md) | How hard the integration is, the exact seams to hook into, a recommended bare-bones first cut, alternatives, and the open risks. Feeds the plan. |
| [`04_implementation_and_phase3_handoff.md`](04_implementation_and_phase3_handoff.md) | **What is actually built (Phases 0–2), the exact commands to test/verify it, and the Phase 3 pickup guide** (multi-benchmark id-bridges + banks + `atlas` suite, generative-scoring caveat, scoring-parity recalibration). Start here to run or extend the integration. |
| [`05_cat_deferred.md`](05_cat_deferred.md) | **Why CAT is on hold and what to pick up.** Which benchmarks were viable and why only HellaSwag, the 3PL→2PL switch and its evidence, measured 2PL bank statistics including the SE floor that caps late-training usefulness, the calibration path and its real (statistical) blocker, how respgen relates to ATLAS calibration, and the hard-won operational details from the removed sweep skill. |

Caveats before you rely on the output. Doc 02 §2 was written when only ARC had a
calibrated bank; six now do (`hellaswag_2pl`, `arc`, `winogrande`, `gsm8k`, `ifeval`,
`math`), and doc 05 §1 has the current table. Still true: p-IRT accuracy is not
reportable as an absolute number, and θ ranking is the sound output. Also note θ is
anchored to the calibration population rather than being an absolute score, and θ from
the 2PL HellaSwag bank is **not** comparable to θ recorded against the old 3PL bank —
doc 05 §4 quantifies both.

## One-paragraph summary

`olmo-eval` (this repo's `src/olmo_eval/`) is a task-registry eval framework: each
benchmark is a `Task` that yields `Instance`s, formats log-likelihood requests,
and scores them; a suite of tasks is launched locally (`olmo-eval run`) or on
Beaker (`olmo-eval beaker launch`), and per-instance predictions can be persisted
to Postgres. ATLAS is a separate psychometric method (published as an R pipeline,
now **vendored + partly reimplemented in Python** under `AdaptiveTesting/`) that
fits **3PL IRT** item parameters on a big model×item response matrix, then does
**adaptive item selection by max Fisher information** to estimate a model's ability
θ from a small subset and reconstruct full-benchmark accuracy (p-IRT). The two fit
together because olmo-eval's MCQ tasks (ARC, HellaSwag, WinoGrande, …) already emit
a per-item correct/incorrect signal keyed by the **native `question_id`**, which is
exactly the key ATLAS's calibrated item bank is indexed by. The bare-bones
integration is: after a model's per-instance predictions exist, load the ATLAS 3PL
bank, run the (already-written) numpy Fisher-info CAT + p-IRT, and emit θ / p-IRT
accuracy as an olmo-eval result.
