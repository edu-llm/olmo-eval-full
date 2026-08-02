TutorBench-33 LLM-judge grading bundle
======================================

Judge:   Qwen/Qwen3.5-9B  (rev c202236235762e1c871ad0ccb60c8ee5ba337b9a)
Backend: vLLM 0.26.0 / torch 2.11 / bfloat16, tensor_parallel=1, max_model_len=32768
Prompt:  judge-validation-v3, canonical variant, replicate r1, temperature 0.0, seed 42
Scope:   33 models x 6,462 criteria = 213,246 cells
Coverage: 212,688 judged pass/fail + 542 auto-fail = 213,230 filled (99.992%);
          16 no-decision holes (judge parse errors), left blank (no_decision_policy=missing).

Contents
--------
response_matrix/
  response_matrix.csv    33 rows (models) x 6,462 cols (criteria); values y in {0,1}, blank = hole
  matrix_manifest.json   staging/ingest manifest for the matrix
  ingest_index.json      run roll-up (counts, coverage, judge config)

verdicts_merged/
  verdicts.jsonl         one row per graded cell, de-blinded (model, scenario, criterion_id -> verdict)

verdicts_raw_shards/
  canonical_r1.shard{0..4}.jsonl   raw judge outputs as returned by the 5 GPU workers
                                   (blinded case_id keys), 212,704 rows total
