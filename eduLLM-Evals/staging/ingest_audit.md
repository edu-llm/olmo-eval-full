# Calibration verdict ingest audit

Generated: 2026-07-29T16:38:37.542522+00:00
Source file: `run_data.jsonl`
Mapping source: `jsonl-deblinded:run_data.jsonl`

## Row-count reconciliation vs contract

- Source rows: 485,615
- Expected judge cases: 505,615
- Gap (expected - source): 20,000
- Expected cells: 506,760
- Expected missing cells: 1,145

## Verdict counts

- fail: 427,560
- no_decision: 6,475
- pass: 51,580

Conflicting case_ids (pass vs fail): 0

## Matrix coverage

- Shape: 6180 criteria x 82 models (total cells 506,760)
- Observed cells: 479,140
- Fill rate: 94.55%
- All-empty (all-NaN) criterion columns: 244
- Scenarios present / curated: 635 / 662 (absent: 27)
- tutor_model values unmatched to a v2 file: 0

Sample all-empty criterion columns:

- tb_0486_c03
- tb_0486_c04
- tb_0486_c05
- tb_0486_c06
- tb_0486_c07
- tb_0487_c01
- tb_0487_c02
- tb_0487_c03
- tb_0487_c04
- tb_0487_c05
- tb_0487_c06
- tb_0487_c07
- tb_0487_c08
- tb_0487_c09
- tb_0487_c10
- tb_0487_c11
- tb_0487_c12
- tb_0488_c01
- tb_0488_c02
- tb_0488_c03
- tb_0488_c04
- tb_0488_c05
- tb_0488_c06
- tb_0488_c07
- tb_0488_c08
- tb_0489_c01
- tb_0489_c02
- tb_0489_c03
- tb_0489_c04
- tb_0489_c05
- tb_0489_c06
- tb_0489_c07
- tb_0489_c08
- tb_0489_c09
- tb_0489_c10
- tb_0489_c11
- tb_0490_c01
- tb_0490_c02
- tb_0490_c03
- tb_0490_c04

Least-covered models (observed cells):

- Qwen/Qwen2.5-1.5B: 5,504
- HuggingFaceTB/SmolLM2-135M: 5,673
- mistralai/Mistral-7B-v0.3: 5,710
- mistralai/Mistral-7B-v0.1: 5,733
- meta-llama/Llama-3.2-1B: 5,771
- Qwen/Qwen2.5-1.5B-Instruct: 5,794
- Qwen/Qwen2.5-3B-Instruct: 5,795
- meta-llama/Llama-3.2-3B-Instruct: 5,806
- 01-ai/Yi-6B-Chat: 5,810
- ibm-granite/granite-3.0-2b-base: 5,812

## no_decision distribution

model_verdict is NaN on no_decision rows; those cells are NaN in the matrix.

By decision_source:

- none: 6,475

Top no_decision scenarios:

- tb_0363: 123
- tb_0529: 75
- tb_0371: 70
- tb_0482: 66
- tb_0357: 64
- tb_0381: 55
- tb_0362: 53
- tb_0330: 50
- tb_0355: 48
- tb_0561: 45
- tb_0598: 43
- tb_0343: 42
- tb_0407: 40
- tb_0285: 39
- tb_0375: 39

## Unresolved / off-bank

- Unresolved response_ids: 0
- Off-bank criterion_ids in source: 0
