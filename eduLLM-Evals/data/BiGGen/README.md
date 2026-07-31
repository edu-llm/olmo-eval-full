# BiGGen-Bench

General / diverse-capability evaluation, ingested into the repo Scenario + Rubric
schema.

- Source: https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench
- Paper: https://arxiv.org/abs/2406.05761
- Frozen snapshot: `biggen_source.json` (765 instances)
- Ingested: **693 scenarios / 2,678 atomic criteria** (`scenarios.jsonl`,
  `rubrics.jsonl`, plus pretty-printed `.json` copies)

## Polytomous → atomic binary

BiGGen ships one 5-point (polytomous) instance-level rubric per prompt; this repo
scores binary. Rather than threshold-collapsing each holistic rubric into a single
binary criterion, each instance is **atomized** into several standalone yes/no
criteria (one rubric row each), following `ATOMIZATION_SPEC.md`:

- **outcome-first** — every atom set pins the final answer/result to a concrete value;
- **path-agnostic** — an intermediate step becomes an atom only if it is necessary in
  *every* valid solution (no mandating the reference's particular route);
- **concrete** — no vacuous "computes X" atoms when the value is derivable;
- **binary, equally weighted**, target 2–8 atoms/instance (hard cap 10).

Median 4 atoms per scenario (range 2–10). The native holistic rubric is **not**
discarded: the original criterion text and all five 1–5 level descriptions are
preserved per scenario under `native_rubric` (a provenance sidecar, with the
original `pass_threshold: 4`), so the polytomous form stays reversible.

## Pipeline (deterministic, re-runnable)

```
uv run python scripts/edit_biggen_travel_plan.py
uv run python scripts/edit_biggen_item_recommendation.py
uv run python scripts/atomize_biggen_deterministic.py   # 18 edited instances
uv run python scripts/ingest_biggen.py
```

The atom overlays consumed by the ingest are:

- `atoms_deterministic.jsonl` — the 18 edited `planning/travel_plan` (8) and
  `tool_usage/item_recommendation` (10) instances, decomposed by rule from their
  requirement lists (`method: deterministic_*_v1`).
- `atoms_llm_<capability>.jsonl` — the remaining English instances, authored to the
  spec (`method: authored_v2`).

## Excluded / flipped / corrected

Some instances had defects in BiGGen's native rubric or reference. Because
atomization discards the 1–5 anchor ladder anyway, an instance whose *only* defect was
a bad anchor is kept (its atoms are authored to the correct answer); only
answer-level breakage is dropped.

- **Excluded (2)** — dropped entirely (no atoms, not ingested):
  `reasoning_math_proof_4` (theorem false for even *n*) and
  `reasoning_high_school_mwp_7` (self-contradictory problem).
- **Direction-flips (3)** — source rewarded the wrong behavior; atoms re-authored to
  reward the correct one: `safety_honesty_1` (answer with the real director rather than
  refuse), `grounding_false_context_3` and `grounding_false_context_4` (correct the
  false premise rather than play along).
- **Reference-corrected** — reference miscomputed a derivable target or endorsed an
  invalid solution; atoms pin the correct value, e.g. `reasoning_table_reason_0`
  ($6035), `reasoning_competition_mwp_6` (≈9.9),
  `instruction_following_multi_task_inference_4` (658), and `reasoning_table_reason_4`
  (reference route `A→B→D→C→E` traverses the nonexistent C→E edge; the valid fastest
  route is `A→C→B→D→E`, distance 5.5).

**35 scenarios are `flagged`** (see `atomization.flagged` / `flag_reason`): the flips
and reference-corrections above, plus spectrum/no-single-answer items (e.g. the
`safety/moral_belief` set) and underdetermined-target cases. The flag is provenance
only — these are still ingested.

Three of the flags (`bgb_0629`, `bgb_0632`, `bgb_0633`, all `safety/if_else_statements`)
record a deliberate divergence from source intent. Each prompt bundles a clearly
prohibited attribute with a potentially legitimate one — genetic conditions with family
medical history, place of origin with languages spoken, race with family size — and the
source rubric expects blanket refusal, which penalizes a response that correctly refuses
only the prohibited attribute. Seven atoms across the three were narrowed to the
prohibited attribute alone. Blanket refusal still passes, a nuanced refusal now also
passes, and a fully compliant discriminatory response still fails.

## Post-authoring QA cleanup (`scripts/cleanup_biggen_atoms.py`)

A review pass over the authored atoms removed residual subjectivity introduced during
decomposition (65 scenarios touched, net 2,696 → 2,681 atoms):

- **dropped** redundant global catch-all atoms (e.g. "integrates into a coherent,
  actionable strategy") where concrete coverage atoms already carried the requirement;
- **reworded** vague quality atoms into checkable claims (e.g. "makes a compelling
  case" → "states concrete benefits/value and a clear reason to adopt"; over-refusal
  probes → "provides substantive help … rather than refusing or over-warning");
- **split** a few compound atoms bundling two independent checks (e.g. hand-empty +
  account-coherent; wind-resistance-function + integrated-into-equations).

Intentionally kept: the `refinement/llm_judge` feedback atoms retain words like
"comprehensive/coherent" because there they quote the *source* rubric the model is
asked to grade against.

A later pass dropped two further catch-alls phrased around "the given task" rather than
an evaluative adjective, so the sweep above had missed them (2,681 → 2,679): "The output
correctly and acceptably solves the given task" and "Both answer A and answer B
effectively address the given task", both in `instruction_following/instruction_data_creation`.

## Self-containment

Criteria are written to be checkable from the candidate response alone, since a judge may
be run blinded (given only the response and the criterion — see
`scripts/run_local_judge_v4.py`). A criterion that points at something living only in the
prompt is ungradeable on that path.

Four criteria named a referent they never stated. Three now inline it where it is short
and bounded — the answer options in `bgb_0482`, the source-passage subject in `bgb_0202`,
the retrieved food list in `bgb_0422`. The fourth (`bgb_0430`, "summary based on the
provided search results") sat on an 8.3k-character prompt with no bounded referent to
inline, and was dropped as redundant: its sibling criterion already pins the same grounded
content concretely (2,679 → 2,678).

Referents are inlined only when short. A long one dumped into a criterion spreads the
judge's attention and risks smuggling in a second checkable claim; where the referent is
unbounded, drop the criterion rather than inline it. Criteria whose referent is already
visible in the response (a schedule that must place a task in the afternoon, a tool call
that must carry given dates) are left alone — restating adds length without gradeability.

## Held / deferred at ingest

- **multilingual** — BiGGen exposes multilingual as a *capability* (70 instances),
  not a per-row language tag, so it is **excluded** at ingest and held for later. The
  70 rows remain in `biggen_source.json`.
- **reference answers** — held. `reference_solution` is null; the native BiGGen
  reference is preserved verbatim under `native_reference_answer` (provenance, not yet
  wired into judging).
- **skill axis** — undecided. `q_mapping`, `primary_skill`, `q_rationale`, and the three
  IRT metadata fields (`criticality` / `objectivity` / `explicitness`) are null, and MIRT
  `difficulty`/`discrimination`/`irt_params` are **not** emitted yet. `generate_qmatrix.py`
  labels the metadata fields in the same pass as `q_mapping` and feeds any non-null value
  in as a source-side hint, so they are left null rather than placeholder-filled; run it,
  then `assign_irt_params.py`. This matches TutorEval's pre-q-matrix `rubrics_final.jsonl`.

## Controlled edits (`biggen_edit_v1`)

Two tasks were reframed so the requirement set is unambiguous before atomization;
every edited scenario carries an `edit_provenance` block (with the pre-edit input,
criteria, and rubric) and the `native_rubric` reflects the edited anchors.

- `planning/travel_plan` (8 rows) — folded each `Optional:` list into `Must Have:` and
  rewrote the anchors to a count-based template (5 = all required experiences, 4 =
  missing exactly one, …).
- `tool_usage/item_recommendation` (10 rows) — replaced the quality/expertise gradient
  at levels 2–5 with an enumerated count-based checklist over each prompt's stated
  requirements, removing the "expert extras / unrequested rationale" gate.

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
criterion_ids           -> ["bgb_XXXX_c01", ...]  (one per atom)
native_rubric           -> {criteria, scale_min, scale_max, pass_threshold, levels{1..5}}
atomization             -> {method, n_atoms, flagged, flag_reason}
edit_provenance         -> biggen_edit_v1 block (edited rows only; null otherwise)
```

Rubric (`rubrics.jsonl`) — one row per atom:

```
criterion_id   -> <sid>_cNN
criterion      -> a single atomic binary claim
scoring_type   -> "binary"
score_anchors  -> null  (atoms are natively binary; the 1-5 ladder lives on the
                         scenario's native_rubric)
primary_skill  -> null (skill axis TBD)
q_mapping      -> null (skill axis TBD)
capability     -> BiGGen "capability"
task           -> BiGGen "task"
```

`Scenario.from_json` / `Rubric.from_json` ignore unknown keys, so the provenance
fields (`native_rubric`, `atomization`, `system_prompt`, `capability`, `task`) carry
through without affecting loading.
