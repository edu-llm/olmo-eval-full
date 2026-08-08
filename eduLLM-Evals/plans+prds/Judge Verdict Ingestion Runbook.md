# Judge Verdict Ingestion Runbook

How to audit, ingest, and verify the Qwen judge verdict shards into the MIRT
response matrix. Windows/PowerShell (use `;`, not `&&`).

**Interpreter note.** The `eduLLM-Evals\.venv` lacks pandas. Anything that needs
pandas/numpy/scipy must run with the repo-root interpreter
`..\.venv\Scripts\python.exe`. Both scripts below are import-light: the ingester
needs numpy (use repo-root venv); the auditor is pure stdlib (either venv works).

**Scope of this run.** 24 full shards + `shard_00025` (partial) are ready.
`shard_00019` DIED on a config error and is **excluded** here (re-ingest later,
see step 5). Do **not** modify data files, IRT params, or the Q-matrix.

---

## 0. Where the shards go

Put the verdict JSONLs somewhere like `grading/qwen/canonical_r1/shards/`, one
`shard_000NN.jsonl` per shard. Every command takes a directory, a glob, or an
explicit file list. To exclude `shard_00019`, list files explicitly or move it
out of the ingest set (globs cannot easily negate on Windows).

A convenience variable that excludes shard 19:

```powershell
$shards = Get-ChildItem grading\qwen\canonical_r1\shards\shard_*.jsonl |
    Where-Object { $_.Name -ne "shard_00019.jsonl" } |
    ForEach-Object { $_.FullName }
```

---

## 1. Audit the shard verdicts (read-only, before ingest)

```powershell
..\.venv\Scripts\python.exe scripts\audit_judge_verdicts.py grading\qwen\canonical_r1\shards\ `
    --out-json staging\audit_judge_verdicts.json `
    --out-csv  staging\audit_judge_verdicts.csv
```

Reports: total cells, decided, `no_decision` count + %, cause buckets
(native=pass/PF=fail; native=fail/PF=pass; truncation `finish_reason=length`;
truncation+disagreement combined), `finish_reason` distribution, per-shard and
per-model `no_decision` counts, and which **criteria set** the verdicts match
(curated vs the final bank at `data/TutorBench/rubrics_qmatrix_final.jsonl`),
with match % and any unknown criterion_ids.

Sanity checks against the known run: `no_decision` ≈ **1.33%** (~6,475 of
~485,615), the native=pass/PF=fail bucket should dominate (~96% of voids, ~128
in the reverse direction), truncation ~4%, and **criteria match should be
`final` at ~100%** with zero unknown ids.

### Confirm the field names first

Field names are **configurable** because the evolved two-signal gate schema is
not fully pinned in this checkout (see "Schema assumptions" below). If the audit
prints WARNINGs like "No P/F verdict signal found", the real field names differ
from the defaults — inspect one shard and re-run with the right flags, e.g.:

```powershell
..\.venv\Scripts\python.exe scripts\audit_judge_verdicts.py grading\...\shards\ `
    --native-field native_verdict --pf-field pf_verdict `
    --finish-reason-field finish_reason --shard-field shard_id
```

Per-model counts need a `model` field on the rows **or** the private de-blinding
index (`--cases-index staging\cases_index.jsonl`), because verdict rows are
blinded (they carry `response_id`/`case_id`, not the tutor id).

---

## 2. Ingest the shards into the response matrix (void → missing)

```powershell
..\.venv\Scripts\python.exe scripts\run_judge_grading.py grade `
    --mode ingest-verdicts `
    --ingest-file $shards `
    --no-decision-policy missing `
    --no-resume
```

`--no-decision-policy missing` (the **default**) maps every voided
(`no_decision`/unscorable/truncation) cell to **NaN (missing)** in the matrix,
and cells absent from the verdict files are **also** missing. Voids are still
recorded in `staging\verdicts.jsonl` (with the reason) for auditability but are
excluded from the matrix. Genuine `pass`→1 / `fail`→0 mapping is unchanged.

Do **not** use `--no-decision-policy fail` (legacy: void→0) for calibration; it
exists only to reproduce the old auto-fail behavior.

`--no-resume` forces a clean rebuild. Use `--resume` (default) only to append
new shards to an existing `staging\verdicts.jsonl` without reprocessing.

Outputs: `staging\response_matrix.csv`, `response_matrix.npy` (NaN = missing),
`response_matrix_manifest.json`.

---

## 3. Verify coverage / shape

```powershell
..\.venv\Scripts\python.exe -c "import json,numpy as np,pathlib; m=json.load(open('staging/response_matrix_manifest.json')); a=np.load('staging/response_matrix.npy'); print('dims', m['matrix']['rows_models'],'x',m['matrix']['cols_criteria']); print('filled', m['coverage']['n_filled'], 'holes(=missing)', m['coverage']['n_holes']); print('nan cells', int(np.isnan(a).sum()), 'of', a.size)"
```

Expect rows = usable models, cols = criteria (match
`staging\judge_inputs_manifest.json`). `n_holes` (= NaN) should ≈ voids +
absent cells (dominated by the excluded `shard_00019` block until step 5). The
run summary also prints `ingested no_decision : N (policy=missing: y=NaN
MISSING, excluded from matrix)` and an `ingest missing` count for holes.

Cross-check the manifest's `frozen_judge_observed` provenance against
`judge_frozen.yaml` (the ingester warns on drift).

---

## 4. Downstream open question (NOT resolved by ingestion)

Ingestion treats every void identically as **missing** — it does **not** pick a
winner between the NATIVE free-text verdict and the single-token P/F verdict.
Whether to re-decide voids by trusting one signal (the "frozen-judge authority"
question) is a **downstream analysis** decision, run against the audit's cause
buckets, not something ingestion should silently resolve. Keep voids missing;
revisit authority separately if the missingness rate materially affects MIRT
calibration.

---

## 5. Re-ingest after `shard_00019` is recovered

Once `shard_00019` is re-run and its verdict JSONL is back, drop it into the
shards directory and rebuild from the full set:

```powershell
# audit the full set (now including shard 19)
..\.venv\Scripts\python.exe scripts\audit_judge_verdicts.py grading\qwen\canonical_r1\shards\ `
    --out-json staging\audit_judge_verdicts.json --out-csv staging\audit_judge_verdicts.csv

# clean rebuild over ALL shards (no exclusion this time)
..\.venv\Scripts\python.exe scripts\run_judge_grading.py grade `
    --mode ingest-verdicts `
    --ingest-file grading\qwen\canonical_r1\shards\ `
    --no-decision-policy missing `
    --no-resume
```

A clean `--no-resume` rebuild is preferred over `--resume` here so the
previously-missing `shard_00019` cells are filled from the recovered verdicts
rather than left as stale holes. Re-verify with step 3; `n_holes` should now
drop to (voids + genuinely absent cells) only.

---

## Schema assumptions (confirm against the REAL shard files)

The canonical runner in this checkout
(`aws_judge_handoff/scripts/run_judge_validation.py`) emits per-verdict:
`case_id, response_id, scenario_id, criterion_id, verdict, native_score,
status, error, raw_output`, plus provenance fields. It does **not** (here)
carry explicit `native_verdict` / `pf_verdict` / `finish_reason` / `shard_id`
fields — those belong to the evolved two-signal consistency gate used for the
real run. Therefore:

- **Ingester** trusts only `verdict` (`pass`/`fail`) + `status` (`ok`). The gate
  is expected to already set `verdict = "no_decision"` for voids, so this is
  sufficient. Confirm the real rows use `verdict`/`status` this way.
- **Auditor** defaults: `--verdict-field verdict`, `--status-field status`,
  `--native-field native_verdict` (falls back to `--native-score-field
  native_score`), `--pf-field pf_verdict`, `--finish-reason-field finish_reason`,
  `--criterion-field criterion_id`, `--case-field case_id`, `--model-field
  model`, `--shard-field shard_id` (falls back to file stem), `--truncation-value
  length`. **Verify these against one real shard**; adjust flags if the two-signal
  gate named them differently. The audit prints WARNINGs for any signal it could
  not find rather than guessing.
