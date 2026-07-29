# Data drop point

> **Generating open-model responses over these benchmarks?** The candidate banks
> here (IFEval, InFoBench, TutorEval, WildBench, Bridge, plus TutorBench) are wired
> into the multi-benchmark response generator via [`benchmarks.yaml`](../benchmarks.yaml):
> `tutor-cat generate --benchmarks benchmarks.yaml` loads each model once and answers
> every enabled benchmark. See the "Generating open-model responses" section of the
> [top-level README](../README.md) for flags, per-benchmark system prompts, and the
> output layout. (Grading is per-benchmark and separate.)

## ⚠️ Use the curated bank

The **finalized rubric bank** lives in `data/curated/`:

- **`data/curated/rubrics_qmatrix_curated.jsonl`** — the criterion bank to grade against
- **`data/curated/scenarios_curated.jsonl`** — scenarios (renumbered `criterion_ids`)

The top-level `rubrics_qmatrix_final.jsonl` / `scenarios.jsonl` are the **untouched
TutorBench source**, kept for provenance only. The curated versions apply the
`curation_v1` fixes (atomized bundled criteria, softened rigid wording, consolidated/
normalized presentation criteria, conditional criteria marked optional). See
`curation/CHANGES.md` for a before→after summary and `curation/grading_notes.md` for
the bank-wide grading rules the judge must apply.

Note: curation edits the **criteria only** — it does not touch IRT/calibration params.
Grade with the curated bank first; (re-)derive IRT params on the frozen criterion set
afterward (split children currently carry copied placeholder params).

## Schemas

- `scenarios.jsonl` / `scenarios_curated.jsonl` — one Scenario Schema object per line
- `rubrics_*.jsonl` — one Rubric Schema object per line (includes the calibrated
  `discrimination` map, `difficulty`, `q_mapping`; curated records also carry a
  `curation` provenance block and, for presentation criteria, `optional` +
  `dimension: style_surface` + `judge_guidance`)

Check them with:

```
tutor-cat validate --config config.yaml
```

## Candidate benchmarks (not confirmed)

TutorBench (`scenarios.jsonl` + `rubrics_*.jsonl` here) is the committed bank pointed to by
`config.yaml`. **EduBench, IFEval, InFoBench, and TutorEval are candidate benchmarks under
evaluation — potential options, not confirmed for use, and still pending further review.** They
ship as on-disk artifacts only; none is wired into a default run, and choosing to include any of
them is a separate, still-open decision.

| Folder | What it is | Status / open question |
|--------|-----------|------------------------|
| `EduBench/` | 9 educational task types, judge-scored 12-metric rubric | candidate — first bank with a **pedagogical** rather than instruction-following axis; open question is whether its 9 task-type skills are the right axis or whether the 12 metrics should be |
| `IFEval/` | instruction-following (deterministic verifier) | candidate — **instruction-following only, no content-quality signal**; keep only if that dimension is wanted |
| `InFoBench/` | instruction-following (judge-scored) | candidate — same instruction-following-vs-content question as IFEval |
| `TutorEval/` | science-tutoring, offline skeleton | candidate — **not engine-loadable** (ships no per-criterion skill labels); needs a q-matrix source first |

Each folder's own `README.md` carries the same status note plus the specifics. (`WildBench/`
and `AP_IB/` are also present as separate on-disk artifacts; their inclusion status isn't
covered here.) Reminder on the standing constraint: any benchmark added must ship its **own
rubrics** (no hand-authoring) and be verified against the actual dataset artifacts, not docs.
