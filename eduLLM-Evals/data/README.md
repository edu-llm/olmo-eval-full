# Data drop point

Place the preprocessed TutorBench files here:

- `scenarios.jsonl` — one Scenario Schema object per line
- `rubrics.jsonl` — one Rubric Schema object per line (must include the
  calibrated `discrimination` map, `difficulty`, and `q_mapping`)

Then check them with:

```
tutor-cat validate --config config.yaml
```

## Candidate benchmarks (not confirmed)

TutorBench (`scenarios.jsonl` + `rubrics_*.jsonl` here) is the committed bank pointed to by
`config.yaml`. **IFEval, InFoBench, and TutorEval are candidate benchmarks under evaluation —
potential options, not confirmed for use, and still pending further review.** They ship as
on-disk artifacts only; none is wired into a default run, and choosing to include any of them
is a separate, still-open decision.

| Folder | What it is | Status / open question |
|--------|-----------|------------------------|
| `IFEval/` | instruction-following (deterministic verifier) | candidate — **instruction-following only, no content-quality signal**; keep only if that dimension is wanted |
| `InFoBench/` | instruction-following (judge-scored) | candidate — same instruction-following-vs-content question as IFEval |
| `TutorEval/` | science-tutoring, offline skeleton | candidate — **not engine-loadable** (ships no per-criterion skill labels); needs a q-matrix source first |

Each folder's own `README.md` carries the same status note plus the specifics. (`WildBench/`
and `AP_IB/` are also present as separate on-disk artifacts; their inclusion status isn't
covered here.) Reminder on the standing constraint: any benchmark added must ship its **own
rubrics** (no hand-authoring) and be verified against the actual dataset artifacts, not docs.
