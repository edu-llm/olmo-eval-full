# Bridge

Bridge reformatted into the Scenario + Rubric schema. Source:
https://huggingface.co/datasets/rose-e-wang/bridge (splits: `train` / `validation` /
`test`, 700 tutor–student math remediation conversations). Paper: *Bridging the
Novice-Expert Gap via Models of Decision-Making: A Case Study on Remediating Math
Mistakes* (Wang et al.).

Built by [`scripts/ingest_bridge.py`](../../scripts/ingest_bridge.py). The records carry no
`difficulty` / `discrimination`: those are calibrated from real judge responses, and until
that fit exists the bank stays parameter-free rather than carrying synthetic stand-ins.

**Current artifact: 642 scenarios · 13,211 criteria · bank of 39 · 20–22 criteria per
scenario.**

> **Status: candidate benchmark — not confirmed for the bank.** Bridge is a *potential*
> option under evaluation, not a committed part of the calibration set. Its presence in
> `data/` does not mean it is wired into any run — the default bank in `config.yaml`
> remains TutorBench, and Bridge only loads when you explicitly flip `SKILLS` to its
> 5-skill axis (below) and point `config.yaml` at these files.

> **⚠️ Deliberate hand-authoring exception.** The standing rule (see
> [`data/README.md`](../README.md)) is that a benchmark "must ship its own rubrics — no
> hand-authoring." Bridge ships **no** per-response criterion checklist — its native rubric
> is a 4-dimension conversation-level scale. So the criterion bank here (and its
> criteria→skill q-matrix) is **hand-authored**. Only the **applicability** of those
> criteria is data-driven (from the expert error label `e` and the TEKS `lesson_topic`).

**Bridge is math-only.** Every `lesson_topic` is a TEKS math code (`grade.standard.Topic`),
spanning grades 1–8 plus a few Algebra 2 rows. A θ estimated on Bridge is *math-tutoring*
ability, not general tutoring ability.

## The 5 skills (q-matrix axis)

Bridge uses its own 5-skill axis, **not** the engine default
(`content` / `diagnosis` / `scaffolding`). This is an on-disk artifact only and is **not**
engine-loadable under the default 3-skill `tutor_cat.SKILLS`. The list order is the fixed
q-matrix column order.

| Skill | Meaning | Pure-loading anchors |
|-------|---------|----------------------|
| `diagnosis` | identifying the student's specific error and its cause | D1, D2 |
| `strategy` | choosing an effective remediation move | P1, P2, G2, R2, S2 |
| `math` | mathematical correctness of the tutor's own statements | M1, M4, M5 |
| `communication` | clarity, structure, audience-appropriateness | C1, Y1, Y2 |
| `affective` | supportive stance: tone, encouragement, agency | A1, A3, A4 |

Every skill carries **at least two pure (single-loading) criteria**. This is enforced in
`validate()` — a dimension anchored by one criterion text is not separably identified in a
confirmatory M2PL.

## Criterion bank: 39 criteria in 4 tiers

Each scenario draws its **core** criteria plus exactly one module from each conditional
tier, giving **20–22 criteria**.

```
core                     every scenario                    16 criteria
error-type module        canonical per conversation         2–3 criteria
topic-domain module      lesson_topic keywords              1–2 criteria
grade-band module        lesson_topic grade prefix          1 criterion
```

### Tier 1 — core (16)

D1 identifies the error · D2 doesn't invent an error · M1 all math correct · **M4 doesn't
endorse the student's wrong answer** · **M5 math is relevant to the task** · M3 moves toward
correct resolution · P1 guides, doesn't tell · P2 move fits situation · P3 keeps student
involved · P4 coheres with prior turns · C1 clear · C3 focused · A1 warm (not merely
neutral) · A2 constructive framing · A3 not discouraging · **A4 safe to be wrong**.

### Tier 2 — error-type modules

| `e` value(s) | Module | Codes |
|--------------|--------|-------|
| `misinterpret`, `diagnose` | `conceptual` | D3, D4, D5 |
| `guess` | `guess` | G1, G2 |
| `right-idea` | `right_idea` | R1, R2 |
| `careless` | `careless` | S1, S2 |
| `imprecise` | `imprecise` | I1, I2 |

### Tier 3 — topic-domain modules (first keyword match wins)

| Domain | Code(s) | Scenarios |
|--------|---------|-----------|
| `geometry_spatial` | V1, V2 | 204 |
| `operations_arithmetic` | O1 | 139 |
| `place_value_number` | N1 | 131 |
| `measurement_conversion` | U1 | 49 |
| `fractions` | F1 | 32 |
| `proportional_reasoning` | Z1 | 31 |
| `algebra_expressions` | X1 | 29 |
| `data_graphing` | B1 | 27 |

### Tier 4 — grade-band module

`Y1` (grades 1-3, 240) · `Y2` (4-5, 332) · `Y3` (6-12, 70). Derived from the TEKS prefix
(`3.6B…` → grade 3; `A2.7D…` → secondary).

## Three design rules worth knowing

**1. Conversation-level error gating.** Bridge re-annotates the same conversation by
different experts, who often disagree about `e` — 278 scenarios sit in a conversation whose
annotators disagreed. Keying the error module on the row's own label gave byte-identical
stimuli *mutually contradictory critical criteria* (42% of the pre-fix bank). The module is
now resolved once **per conversation** — the first annotation in split order wins — and every
row sharing that `c_id` draws it. Deterministic pre-pass, no API.

Taking the **union** of all annotators' modules was tried and rejected: it preserved every
expert's framing but gave disagreement-heavy conversations an extra module, so those
scenarios carried one more `critical` criterion (8 vs 7). Difficulty would then partly encode
whether two annotators happened to disagree rather than the tutoring task. With a single
canonical module the critical count is **7.00 for both** agreed and disagreed scenarios. The
discarded labels remain in `error_types` for provenance.

**2. One scenario per row — `source_id` is deliberately non-unique.** Every surviving row
becomes its own scenario, so each expert's revision stays a separate item and nothing is
merged away. The consequence is real and must be handled downstream: 424 scenarios share a
conversation with at least one other (430 unique conversations across 642 scenarios), and
those repeats have an identical stimulus *and* an identical criterion set,
differing only in `reference_solution`. That is perfect local dependence, and the judge sees
a different gold key for each copy — so the same item can acquire two difficulties. **Group
or hold out on `source_id` at calibration**; `validate()` only enforces that repeats agree
on their module set, not that they are unique.

**3. No `optional` criteria, and negative wording where content is not guaranteed.** The
repo's `optional: true` flag is **not** an N/A mechanism —
[`run_calibration_judging.py`](../../scripts/run_calibration_judging.py) *drops* optional
criteria from the run entirely and the judge prompt states "Never invent an N/A outcome"
(`judge_guidance` is read by no code at all). Every criterion is therefore answerable
pass/fail for any response. Criteria about content a response need not contain (V2, N1, U1,
B1, X1, Z1, I2, Y3, D4, G1, R1, S1, I1) are stated in **negative form** — fail on misuse,
pass on silence — so the empty case has a determinate verdict rather than a forced fail. M1
states its empty case explicitly.

## Exclusions

| Reason | Dropped | What it catches |
|--------|---------|-----------------|
| `no_clear_mistake` | 56 | `e` is free text ("no mistake" / "end session" / "unresponsive") |
| `empty_student_turn` | 2 | final student turn has no text |

700 source rows → 58 dropped → **642 scenarios**. All drops are logged to `dropped.jsonl`
with a reason.

### `visible_mistake` — flagged, not excluded

**91 scenarios (1,880 criteria) carry `visible_mistake: false`.** Their final student turn
is a bare acknowledgment ("yes" ×112, "no", "done", …) **and** no digit appears anywhere in
the conversation, so the transcript is only tutor turns plus an assent:

```
[tutor]   You are doing a good job.
[tutor]   So far so good. keep going.
[tutor]   Are you done with your answer?
[student] "yes"          ← the "mistake"
```

There is no student position to remediate, so D1/D2 (**both `critical`**) are effectively
unpassable and will read as floor items. They are **kept** so the bank stays complete, and
flagged so a calibration run can filter or model them explicitly — `[s for s in scenarios if
s["visible_mistake"]]` gives the 551 with a remediable error.

Bare acknowledgments **with** a digit somewhere are `visible_mistake: true`: the student's
answer is recoverable from an earlier turn or the tutor's restatement, or the assent is
itself the wrong answer to a yes/no question.

## Criterion ids

Ids use a **stable suffix per code** (`CODE_INDEX`): a given suffix denotes the same
criterion on every scenario. Because each scenario draws a subset of the bank, its
`criterion_ids` are intentionally **non-contiguous**. New codes must be appended at the end
of the bank to keep existing ids stable.

## How the source maps into the schema

| Bridge field | Schema field |
|--------------|--------------|
| `c_id` | `source_id` (**non-unique** — the conversation key; join key back to HuggingFace) |
| `c_h[-1].text` | `prompt` (the student's final turn); also derives `visible_mistake` |
| `c_h[:-1]` | `conversation_context` (`{role, content}`, role `student`/`tutor`) |
| `c_r_` (joined) | `reference_solution` (this row's expert revised reply = the gold key) |
| `c_r` (joined) | `novice_response` (original tutor reply; provenance) |
| `e` | `error_type` (this row) + `error_types` (all, provenance) → `error_module` (canonical) |
| `z_what` / `z_why` | `expert_strategy` / `expert_intention` (provenance) |
| `lesson_topic` | `lesson_topic`, and derives `grade_band` + `topic_domain` |
| HF split | `native_split` (provenance; `split` is the pipeline role `calibration`) |

Each rubric also carries `criterion_code` (D1, G2, …) and `applicability` (the tier that
attached it, e.g. `core` or `error_type:guess`), so the conditional structure survives into
calibration for grouping and diagnostics.

`use_case` is `mistake_remediation`; the response generator has no dedicated system prompt
for it, so it falls back to the `adaptive_explanation` prompt, which preserves the
multi-turn context.

## Known limitations

- **Judge grounding.** `tutor_cat/judge.py` sends the judge only `prompt` and
  `conversation_context` — not `reference_solution` or `error_type`. The production path
  (`run_local_judge_v4.py`) does render the reference. The diagnosis criteria (D1/D2) assume
  the judge can see what the student got wrong; verify that before a real run.
- **Prescriptiveness.** The error modules encode one remediation framing per error type.
  They have been reworded toward negative/falsifiable form to reduce unfair failure of
  legitimate alternative remediations, but this has not been validated against Bridge's own
  expert gold replies.
- **Topic gating is keyword-based** and coarse; `operations_arithmetic` remains a catch-all
  of 139 scenarios.
- **Floor items are retained.** The 91 `visible_mistake: false` scenarios (1,994 criteria)
  are kept deliberately, but D1/D2 are unpassable on them. Filter on the flag, or expect
  those items to carry no information.
- **Duplicate stimuli are retained.** 424 scenarios share a conversation with another and,
  byte-identical apart from `reference_solution`. At temperature 0 a
  tutor returns the same response to each, so these rows are near-copies in the response
  matrix while the judge scores them against different gold keys. Group on `source_id`.

## Rebuild

```
python scripts/ingest_bridge.py
```

The ingester is the whole rebuild. `Rubric.from_json` hard-requires `difficulty` and
`discrimination`, so loading this bank through `tutor_cat.dataio.load_bank` needs those
fields supplied first — from a calibration fit, or from
[`scripts/assign_irt_params.py`](../../scripts/assign_irt_params.py) if a synthetic
placeholder pass is wanted for a dry run:

```
python scripts/assign_irt_params.py \
    --input data/Bridge/rubrics.jsonl \
    --skills diagnosis,strategy,math,communication,affective \
    --log-dir data/Bridge/irt_logs --no-backup
```

Anything it writes is synthetic (`irt_params.source == "synthetic"`), so keep it out of the
committed artifact.
