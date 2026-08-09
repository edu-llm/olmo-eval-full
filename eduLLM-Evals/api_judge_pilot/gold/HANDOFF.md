# Biggen gold set — handoff for judge-candidate scoring

## What this is
A **representative-core human-gold set** for biggen, built to measure judge candidates'
**false-pass rate** (and accuracy) on real graded cells, since biggen has no prior human labels.

- **Population:** 121,028 gradable cells (52 models × biggen criteria, auto-fail cells excluded).
- **Sample:** 100 cells, **stratified by `capability`** (biggen's `criticality`/`primary_skill` are null),
  proportional allocation, seed 20260809. Reference balance: 52 Qwen-fail / 48 Qwen-pass.
- **Labels:** two independent, blinded proposers (`claude-opus-5` + `openai-group/gpt-5.5`) each gave a
  structured PASS/FAIL with evidence; concordant → provisional gold; discordant → human-adjudicated.
  (`opus-4.8` and `sonnet-5` are catalog-listed but **denied by the gateway**; opus-5 substituted.)

## The gold artifact
`biggen_core/gold_labels.jsonl` — one row per cell:
```json
{"gold_case_id","model","scenario_id","criterion_id","capability",
 "gold_label": "pass"|"fail"|null, "provenance", "_qwen_verdict"}
```
`provenance`: `human_adjudicated` | `ai_concordant_spotchecked` | `ai_concordant_accepted`.
**Score only rows where `gold_label` is not null.** Status (see `gold_manifest.json`): **FINAL —
100/100 resolved, 0 pending** (7 human-adjudicated + 18 owner-spot-checked + 75 two-model-concordant;
55 fail / 45 pass).

## The cases to grade
`biggen_core/sample.jsonl` has the full case per `gold_case_id`:
`scenario_prompt`, `conversation_context`, `reference_solution`, `criterion`, `candidate_response`.
These are the exact fields `run_api_judge_pilot.build_messages` / `regrade_benchmark.build_cases` render.

## How to score the candidates
Grade the 100 gold cells with each candidate using the **same chosen judge config** as the pilot
(`--adapter generic-binary-strict`, JSON mode, temp 0), one call per (candidate, gold_case) = ~500 calls.
Candidates (gateway slugs):
- `claude-group/claude-sonnet-4-6`, `claude-group/claude-opus-4-6`
- `gemini-group/gemini-2.5-flash`, `gemini-group/gemini-3-flash-preview` (**use max_tokens ≥ 4096** — it truncates/over-fails otherwise, see FINDINGS.md)
- `openai-group/gpt-4.1`

Then join verdict → gold on `gold_case_id` and report **per candidate**:
- **false-pass rate** = P(judge=pass | gold=fail) — the headline; overall + per `capability`.
- accuracy, false-fail rate, balanced accuracy, coverage/unscorable.
- Also compute the same for the **frozen Qwen judge** using `_qwen_verdict` (free baseline).

Only join on `gold_label != null`. Report on the 93 provisional now; refresh when the 7 land.

## Caveats to carry into any report
- **Representative probability sample** → the FP rate is an unbiased estimate for the biggen grading
  matrix (weight by capability strata if needed; see `sample_manifest.json`).
- **Single-annotator gold**, AI-assisted (opus-5 + gpt-5.5 proposers, human-adjudicated disagreements).
  Concordant cells are two-model-agreed, owner-spot-checked, not each individually human-verified.
- Proposers are **independent of the candidate set** (opus-5 ≠ opus-4-6; gpt-5.5 ≠ gpt-4.1), so scoring
  candidates against this gold is not self-agreement.
- Only ~100 cells → per-capability rates are directional (small n per stratum).

## Reproduce / refresh
```bash
# finalize after the 7 verdicts (example):
uv run --no-project python api_judge_pilot/gold/adjudicate.py \
  --packets api_judge_pilot/gold/biggen_core/packets.jsonl \
  --sample  api_judge_pilot/gold/biggen_core/sample.jsonl \
  --out     api_judge_pilot/gold/biggen_core/gold_labels.jsonl \
  --set biggen__0021=fail --set biggen__0048=fail ...   # owner verdicts
```
Pipeline scripts: `sample_cells.py`, `make_packets.py`, `adjudicate.py`, `review.py`, `summarize_packets.py`.
