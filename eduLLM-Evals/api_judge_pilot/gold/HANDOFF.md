# Gold sets — handoff for judge-candidate scoring (biggen + tutoreval)

Two **representative-core human-gold sets** for measuring judge candidates' **false-pass rate**
(and accuracy) on real graded cells, since these benchmarks had no prior human labels.

## The two gold artifacts
| Benchmark | Gold file | Cases dir | Stratum | Qwen baseline |
|---|---|---|---|---|
| **biggen** | `biggen_core/gold_labels.jsonl` | `biggen_core/sample.jsonl` | `capability` (8 buckets) | `_qwen_verdict` present |
| **tutoreval** | `tutoreval_core/gold_labels.jsonl` | `tutoreval_core/sample.jsonl` | `primary_skill` (conceptual_understanding / quantitative_procedural / unknown) | `_qwen_verdict` = null (no matrix locally) |

Each is **100 cells, FINAL, 0 pending**. `gold_labels.jsonl` row schema:
```json
{"gold_case_id","model","scenario_id","criterion_id","stratum","criticality",
 "gold_label": "pass"|"fail", "provenance", "_qwen_verdict"}
```
`provenance`: `human_adjudicated` | `ai_concordant_spotchecked` | `ai_concordant_accepted`.

- **biggen:** 55 fail / 45 pass; 7 human-adjudicated + 18 spot-checked + 75 concordant. `criticality` is null
  (biggen rubrics carry none) → no critical-FP slice for biggen.
- **tutoreval:** 80 fail / 20 pass (fail-heavy — small models on hard tutoring criteria);
  9 human-adjudicated + 18 spot-checked + 73 concordant. `criticality` **present**
  (critical / not_critical) → **critical-FP is computable** for tutoreval.

## The cases to grade
`<bench>_core/sample.jsonl` has the full case per `gold_case_id`: `scenario_prompt`,
`conversation_context`, `reference_solution`, `criterion`, `candidate_response`.
These are the exact fields `run_api_judge_pilot.build_messages` / `regrade_benchmark.build_cases` render.

## How to score the candidates
Grade each gold set's 100 cases with each candidate using the **chosen judge config**
(`--adapter generic-binary-strict`, JSON mode, temp 0), one call per (candidate, gold_case).
Candidates (gateway slugs):
- `claude-group/claude-sonnet-4-6`, `claude-group/claude-opus-4-6`
- `gemini-group/gemini-2.5-flash`, `gemini-group/gemini-3-flash-preview` (**max_tokens ≥ 4096**, see FINDINGS.md)
- `openai-group/gpt-4.1`

Join verdict → gold on `gold_case_id`, then report **per candidate per benchmark**:
- **false-pass rate = P(judge=pass | gold=fail)** — headline; overall + per `stratum`
  (+ per `criticality` and a **critical-FP** for tutoreval).
- accuracy, false-fail rate, balanced accuracy, coverage/unscorable.
- For **biggen only**, compute the same for the frozen Qwen judge from `_qwen_verdict` (free baseline).

## Caveats to carry into any report
- **Representative probability samples** (weights in `<bench>_core/sample_manifest.json`) → the FP rate is
  an unbiased estimate for that benchmark's grading matrix. biggen pop = 121,028 cells / 52 models;
  tutoreval pop = 92,438 cells / 52 models.
- **Single-annotator**, AI-assisted gold: proposers **opus-5 + gpt-5.5** (independent of the candidate set —
  opus-5 ≠ opus-4-6, gpt-5.5 ≠ gpt-4.1), owner-adjudicated on disagreement, owner-spot-checked a random 18.
  (`claude-opus-4-8` / `sonnet-5` were denied by the gateway; opus-5 substituted.)
- ~100 cells each → per-stratum rates are directional (small n per bucket; tutoreval is 82% conceptual).

## Pipeline (reusable) / refresh
`sample_cells.py` (`--stratify-field`, `--benchmark`), `make_packets.py`, `adjudicate.py`
(`--set id=label`, `--spotchecked ...`), `review.py`, `dump_ids.py`, `dump_spotcheck.py`, `summarize_packets.py`.
tutoreval bank was extracted from git tag `tutoreval-unidim-52models:eduLLM-Evals/data/TutorEval/`
into `tutoreval_src/`.
