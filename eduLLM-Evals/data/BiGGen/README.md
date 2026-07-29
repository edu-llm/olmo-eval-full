# BiGGen-Bench

General / diverse-capability evaluation, ingested into the repo Scenario + Rubric
schema.

- Source: https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench
- Paper: https://arxiv.org/abs/2406.05761
- Frozen snapshot: `biggen_source.json` (765 instances)
- Ingested: **695 scenarios / 695 criteria** (`scenarios.jsonl`, `rubrics.jsonl`,
  plus pretty-printed `.json` copies)

## ⚠️ Before grading: expand the criterion

Each BiGGen instance is currently ingested as **one holistic binary criterion**
per scenario (see "Polytomous → binary" below). This is a faithful
threshold-collapse of BiGGen's native 5-point rubric, but a single holistic
criterion is a coarse unit for judging and for MIRT.

**Expanding each instance's single criterion into multiple atomic binary
criteria (one checkable claim each) is an option that should be explored before
any responses to this benchmark are graded.** Atomic decomposition tends to give
more reliable judge verdicts and cleaner per-skill signal, and it matches the
atomic-checklist style the Qwen judge already uses. The `score_anchors` field
preserves the original 1–5 level descriptions, which is the natural raw material
for that decomposition. Treat the current one-criterion form as provisional
until this has been evaluated.

## Pipeline (deterministic, re-runnable)

```
uv run python scripts/edit_biggen_travel_plan.py
uv run python scripts/edit_biggen_item_recommendation.py
uv run python scripts/ingest_biggen.py
```

## Polytomous → binary (threshold-collapse)

BiGGen ships one 5-point (polytomous) instance-level rubric per prompt; this repo
scores binary. Each instance is collapsed into a single binary criterion:

- `criterion` — BiGGen's `score_rubric.criteria`.
- `score_anchors` — the five level descriptions are preserved, together with the
  binary pass rule:
  ```json
  {"scale_min": 1, "scale_max": 5, "pass_threshold": 4,
   "levels": {"1": "...", "2": "...", "3": "...", "4": "...", "5": "..."}}
  ```
  The judge grades on the 1–5 scale and thresholds at `pass_threshold` (4) → a
  response passes iff it reaches "all requirements met" (5) or "missing exactly
  one" (4). The global default lives in `config.yaml` (`result_pass_threshold`).

## Held / deferred at ingest

- **multilingual** — BiGGen exposes multilingual as a *capability* (70
  instances), not a per-row language tag, so it is **excluded** at ingest and
  held for later. The 70 rows remain in `biggen_source.json`.
- **reference answers** — held. `reference_solution` is null; the native BiGGen
  reference is preserved verbatim under `native_reference_answer` (provenance,
  not yet wired into judging).
- **skill axis** — undecided. `q_mapping` and `primary_skill` are null, and MIRT
  `difficulty`/`discrimination`/`irt_params` are **not** emitted yet (append with
  `assign_irt_params.py` once the skill axis is chosen), matching InFoBench.

## Controlled edits (`biggen_edit_v1`)

Two tasks were reframed so a binary ≥4 pass bar is semantically correct; every
edited row carries an `edit_provenance` block (with the pre-edit input, criteria,
and rubric) and is otherwise reversible.

- `planning/travel_plan` (8 rows) — folded each `Optional:` list into
  `Must Have:` and rewrote the anchors to a count-based template (5 = all
  required experiences, 4 = missing exactly one, …).
- `tool_usage/item_recommendation` (10 rows) — replaced the quality/expertise
  gradient at levels 2–5 with an enumerated count-based checklist over each
  prompt's stated requirements, removing the "expert extras / unrequested
  rationale" gate.

## Provenance fields

`system_prompt`, `capability`, and `task` are preserved on the ingested records
for provenance. `Scenario.from_json` / `Rubric.from_json` ignore unknown keys, so
these carry through without affecting loading (same pattern as Bridge).

## Schema mapping

Scenario (`scenarios.jsonl`):

```
scenario_id             -> bgb_<zero-based 4-digit index>
source_id               -> BiGGen "id" (join key back to HuggingFace)
use_case / subject      -> BiGGen "capability"
task                    -> BiGGen "task" (sub-capability, e.g. travel_plan)
system_prompt           -> BiGGen "system_prompt" (verbatim; provenance)
prompt                  -> BiGGen "input" (edited variant where applicable)
reference_solution      -> null (held)
native_reference_answer -> BiGGen "reference_answer" (verbatim; provenance)
criterion_ids           -> ["bgb_XXXX_c01"]  (always exactly one)
```

Rubric (`rubrics.jsonl`):

```
criterion_id   -> <sid>_c01
criterion      -> BiGGen "score_rubric.criteria"
scoring_type   -> "binary"
score_anchors  -> {scale_min, scale_max, pass_threshold, levels{1..5}}
primary_skill  -> null (skill axis TBD)
q_mapping      -> null (skill axis TBD)
capability     -> BiGGen "capability"
task           -> BiGGen "task"
edit_provenance-> biggen_edit_v1 block (edited rows only; null otherwise)
```
