# Data gaps vs. the outline

What `Outline.md` asks for but the repository does **not** (yet) fully support, with the
concrete action to close each gap. Ordered roughly by importance for the paper.

## FRQ

1. **Per-judge validation scoreboard is missing (highest priority).** The judge-selection
   *design, thresholds, 261 blinded cases, and frozen config* are in-repo (`02_FRQ_Judge/`),
   but the actual numbers — Macro-F1, critical-failure sensitivity, per-skill F1,
   test-retest agreement, prompt-flip rate for Prometheus/Flow/Selene/Qwen/Gemma — are in a
   gitignored `runs/` tree / S3, not here. The outline's *Results* table (the "H/H/H/H"
   placeholders and "no judge passed every threshold") cannot be filled from local data.
   *Action:* recover the six-wave verdict JSONLs + `human_labels.csv` and run
   `eduLLM-Evals/scripts/compare_judge_reliability.py` / `compare_reliability_suite.sh`.
2. **96-case zero/few-shot/countercheck pilot has no result file** — narrative only in the
   outline. *Action:* locate or re-run the pilot; store the three prompting conditions'
   metrics.
3. **TutorEval MIRT is not done.** The outline's feasibility mentions a 52-model run
   (19 overlapping) but there is **no TutorEval graded response matrix** locally (only a
   52-model list). All FRQ MIRT results here are **TutorBench only**. *Action:* run the
   emit→judge→ingest pipeline on TutorEval, then `calibrate_mirt.py` + `cat_eval` as in
   `03_FRQ_MIRT`.
4. **WildBench / EduBench / Bridge MIRT not fit.** Scenarios + rubrics/Q-matrix exist, but
   no calibrated banks or CAT results. The outline lists all five FRQ benchmarks; only
   TutorBench is calibrated.
5. **Matched-method parameter-SE at N=115 uses an asymptotic (observed-information)
   estimate**, not the bootstrap used for the N=82 `param_uncertainty` figure. The
   comparison in `03_FRQ_MIRT` is internally matched (same method both N), but to line up
   with the existing N=82 bootstrap number a bootstrap recompute at N=115 would be cleaner.
6. **"115 models" reconciliation.** We materialized 115 = 82 + 33 (disjoint) which matches
   the outline number. The separately-mentioned "52-model run, 19 overlapping" is a
   *different* (TutorEval) plan and is not the source of the 115 — worth stating precisely
   in the paper so the two "second run" descriptions aren't conflated.
7. **`cat_eval_tutorbench.py` hard-codes 82-run bank constants** (`3497/6180`,
   "scaffolding needs refit") in its `bank` manifest block and summary-table text, so the
   115-run `cat_metrics.json → bank` sub-block is misleading (the real fit used 3,669
   items). Recovery/CAT/pIRT numbers are correct; only that sub-block is stale. *Action:*
   parameterize those constants before publishing the table verbatim.

## MCQ / ATLAS

8. **Pedagogy IRT is now fit (new here), but only at 52 models / 2PL.** SE≤0.15 needs ~87%
   of the bank; a 3PL and/or more calibration models would make the tight-SE regime
   practical. *Action:* expand the pedagogy response matrix (more of the 156-model roster)
   and try a 3PL once the pool is large enough.
9. **Range-restricted ATLAS recalibration pool is 95% 7B** (`cal_models_0p5_7b.csv`), so the
   r≈0.59 "restricted" result partly reflects *size-skew*, not just range restriction. A
   uniform 0.5–7B calibration pool is the missing clean comparison.
10. **`musr` fails to link** (r≈−0.04) — a benchmark where the current ATLAS-3PL pipeline
    does not recover; flagged as a negative control, not yet diagnosed.
11. **SocialIQA barely discriminates** the in-house 52-model pool (r=0.27 @ SE≤0.3). Either
    a wider model pool or a note that CAT value is benchmark-dependent.

## Inference cost

12. **Hardware label mismatch.** The individual-latency table (≈4/10/20/35 s) is computed
    on / for an **L4** (the GPU full200 used); the outline captions it **L40S**. Re-label
    to L4, or re-measure single-stream latency on an actual L40S (≈2–3× faster decode).
13. **Individual latency is a roofline *estimate*, not a direct single-stream measurement.**
    All measured latencies are batched. A small single-request benchmark per band would
    replace the estimate with data.
14. **MCQ cost basis is a 24-model generative run**, because the full200 MCQ is
    loglikelihood-scored and records no latency. It is order-of-magnitude only and not the
    same model set as the open-ended cost.

## Rosters / bookkeeping

15. **`models_200_stats.txt` is stale** (reports 200 models; the current `models_200.yaml`
    has 156). Regenerate stats from the live YAML before quoting counts.
16. **full200 is ~33% complete** (52 of 156 models across 9 benchmarks). More completed
    models would strengthen every downstream MCQ/FRQ/cost result.
17. **OpenHelm is absent locally** although the outline cites it as a data source; either
    obtain it or drop the citation.
