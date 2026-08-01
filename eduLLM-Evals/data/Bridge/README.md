# Bridge

Bridge reformatted into the Scenario + Rubric schema. Source:
https://huggingface.co/datasets/rose-e-wang/bridge (splits: `train` / `validation` /
`test`, 700 tutor–student math remediation conversations). Paper: *Bridging the
Novice-Expert Gap via Models of Decision-Making: A Case Study on Remediating Math
Mistakes* (Wang et al.).

Built by [`scripts/ingest_bridge.py`](../../scripts/ingest_bridge.py). The records carry no
`difficulty` / `discrimination`: those are calibrated from real judge responses, and until
that fit exists the bank stays parameter-free rather than carrying synthetic stand-ins.

**Current artifact: 239 scenarios · 4,861 criteria · bank of 39 · 20–22 criteria per
scenario.**

> **⚠️ 403 of the original 642 scenarios were removed.** 353 because the problem statement
> is missing, then 50 more because the graded turn has no gradeable error at all. Bridge
> transcribes live sessions held over a shared whiteboard, and the worksheet was never
> captured. On 55% of the bank that left transcripts like *"Here comes the question. / Is
> that your final answer? / yes"* — a tutor asked to diagnose an error without being told
> what was asked. See [Missing problem statements](#missing-problem-statements). The
> excluded ids are committed in [`visual_exclusions.json`](visual_exclusions.json); delete
> that file and rebuild to restore the 642.

> **Status: candidate benchmark — not confirmed for the bank.** Bridge is a *potential*
> option under evaluation, not a committed part of the calibration set. Its presence in
> `data/` does not commit it to the calibration set. It **is** registered in
> [`benchmarks.yaml`](../../benchmarks.yaml) with `enabled: true`, so response generation
> covers it (see `bridge-modelresps/`), but the CAT bank in `config.yaml` remains
> TutorBench.

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
(`content` / `diagnosis` / `scaffolding`). `load_bank` infers the axis from the criteria's
own `q_mapping` key order, so the bank loads without being told; the list order below is
that fixed q-matrix column order and must not be reordered in place.

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

D1 identifies the error · D2 doesn't invent an error · M1 all math correct · M4 doesn't
endorse the student's wrong answer · M5 math is relevant to the task · M3 moves toward
correct resolution · P1 guides, doesn't tell · P2 move fits situation · P3 leaves a
substantive step for the student · P4 coheres with prior turns · C1 reasoning in a
followable order · C3 focused · A1 warm (not merely neutral) · A2 constructive framing ·
A3 not discouraging · A4 safe to be wrong.

### Tier 2 — error-type modules

| `e` value(s) | Module | Codes |
|--------------|--------|-------|
| `misinterpret`, `diagnose` | `conceptual` | D3, D4, D5 |
| `guess` | `guess` | G1, G2 |
| `right-idea` | `right_idea` | R1, R2 |
| `careless` | `careless` | S1, S2 |
| `imprecise` | `imprecise` | I1, I2 |

### Tier 3 — topic-domain modules (first keyword match wins)

Seven domains; `data_graphing` was **retired** — see [Retired criteria](#retired-criteria).

| Domain | Code(s) | Scenarios |
|--------|---------|-----------|
| `geometry_spatial` | V1, V2 | 49 |
| `operations_arithmetic` | O1 | 105 |
| `place_value_number` | N1 | 44 |
| `measurement_conversion` | U1 | 19 |
| `fractions` | F1 | 7 |
| `proportional_reasoning` | Z1 | 5 |
| `algebra_expressions` | X1 | 10 |

### Tier 4 — grade-band module

`Y1` (grades 1-3, 75) · `Y2` (4-5, 133) · `Y3` (6-12, 31). Derived from the TEKS prefix
(`3.6B…` → grade 3; `A2.7D…` → secondary).

## Retired criteria

`B1` (data-display reading) is **kept in the bank list but attached to nothing**, so
`CODE_INDEX` — and therefore every existing `criterion_id` — is unchanged; removing an entry
from the middle of the bank would renumber every code after it. `validate()` exempts retired
codes from the "never attached" check and asserts they stay unattached.

Its gate keyed on `lesson_topic`, which names the lesson rather than the graded turn. After
the visual cut only 6 scenarios carried it, and **four of those were arithmetic or fraction
addition** — `5+1+3+8+12+4+6`, `1/8+2/4+6/8+2/2+5/8` — inside lessons *titled* "Bar Graphs"
or "Line Plots with Fractions". B1 asks whether the response misreads a scale, key or axis,
which is unanswerable when no data display is involved. Dropping the gate lets those rows
route on their actual content: the two fraction ones now draw `F1`, the other four `O1`.

## Per-item audit — what each turn actually asks

The two cuts above were found by reading items rather than trusting their metadata, so the
same read was run over the whole bank:
[`scripts/audit_bridge_items.py`](../../scripts/audit_bridge_items.py) asks `gpt-5.5`, once
per scenario, what the graded turn actually asks and what the student actually did, then
judges the stored labels against that. Full results:
[`staging/bridge_item_audit/report.md`](../../staging/bridge_item_audit/report.md).

**50 scenarios excluded** — the graded turn has no gradeable error. These carried a *clean*
error label, which is exactly why Bridge's free-text `no_clear_mistake` filter missed them:

```
[tutor]   Please show your work on the whiteboard using the Pencil Tool
[student] i well be right back
[tutor]   Okay, Let me know when you are back?
[student] "i have to leeve"          ← the turn we graded
GOLD:     "We understand emergencies happen. I'll see you at your next session."
```

That one was labelled `error_module: conceptual`. Others answer *"Do you need any help?"* with
`"no"`, or *"Do you understand?"* with `"a little bit"`, and in one case the student was simply
right (the expert reply opens *"Correct!"*).

**83 `topic_domain` corrections** ([`topic_overrides.json`](topic_overrides.json)) — the
`data_graphing` defect at scale. A multi-turn dialogue drills into an arithmetic sub-step and
that sub-step is the graded turn:

| scenario | `lesson_topic` | what the turn asks | student |
|---|---|---|---|
| `bridge_0003` | Areas by Decomposition | `36+42` | `1,512` |
| `bridge_0085` | Geometric Lines | `2+0+1+1` | `3` |
| `bridge_0069` | Shapes and Area | `4 × 4` | `12` |

> **The audit's `error_module` disagreements were deliberately NOT acted on.** It contradicted
> the expert annotation on 142 scenarios, but 54 of those push `right_idea`/`careless` toward
> `guess` — precisely what the missing whiteboard produces, since the expert *saw* the
> student's working and a text-only reader cannot. It also disagrees with every annotator in
> 131 of 142 cases. The experts are better placed here; the label stands.

> **Scope of validation.** One classifier, no human raters. Precision was hand-checked on
> roughly a dozen items across categories and held; recall is unmeasured, and the 76 scenarios
> the audit called clean have not been reviewed.

## Three design rules worth knowing

**1. Conversation-level error gating.** Bridge re-annotates the same conversation by
different experts, who often disagree about `e` — 85 scenarios sit in a conversation whose
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
merged away. The consequence is real and must be handled downstream: 178 scenarios share a
conversation with at least one other (163 unique conversations across 239 scenarios), and
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
pass/fail for any response. Fourteen criteria about content a response need not contain
(A4, B1, D2, D4, G1, I1, I2, M4, M5, N1, P4, R1, S1, U1) are stated in **negative form** — fail
on misuse, pass on silence — so the empty case has a determinate verdict rather than a forced
fail. M1 states its empty case explicitly. This was originally applied to nineteen criteria;
the pilot below showed negative form inflates pass rates, so six were rewritten.

## Exclusions

| Reason | Dropped | What it catches |
|--------|---------|-----------------|
| `missing_problem_statement` | 263 | the question was on the whiteboard, never in the chat |
| `error_not_diagnosable_from_text` | 90 | a figure is referenced and the error is unreadable without it |
| `no_error_present` | 20 | the student is correct, acknowledging, or ending the session |
| `not_mathematics` | 18 | the graded turn is session admin or tool talk |
| `not_gradeable` | 12 | the error is real but not recoverable from the visible text |
| `no_clear_mistake` | 56 | `e` is free text ("no mistake" / "end session" / "unresponsive") |
| `empty_student_turn` | 2 | final student turn has no text |

700 source rows → 461 dropped → **239 scenarios**. All drops are logged to `dropped.jsonl`
with a reason; the two deterministic reasons carry a null `scenario_id` because they fail
before ids are assigned.

**Ids are non-contiguous, and deliberately so.** They are assigned over every row passing
the deterministic checks and *before* the exclusion list is applied, so a surviving
scenario's id never moves when that list changes. Response runs in `bridge-modelresps/`
and pilot judgments in `staging/` stay joinable across the cut.

### Missing problem statements

The 263 in the first round. These transcripts record the conversation *around* a problem that
only ever existed on the shared whiteboard:

```
[tutor]   Here is your first Exit Ticket question.
[tutor]   Please go ahead and give it a good try.
[tutor]   Is that your final answer?
[student] "yes"          ← the turn a tutor must respond to
```

Only the expert's reply reveals the task (*"Could you let me know why you think this shape
is a pentagon?"*) — nothing the tutor model sees names a shape at all. Elsewhere a student
answers `"90"` and the gold reply shows the question was "the 8th multiple of 10".

Identified by [`scripts/audit_bridge_visuals.py`](../../scripts/audit_bridge_visuals.py),
which asked `gpt-5.5` one question per scenario: can a tutor identify the student's error
from the text alone? 364 of 642 were flagged, 362 at high confidence, and only 18 of 212
repeated conversations drew inconsistent verdicts. Evidence behind the cut:
[`report_precut_642.md`](../../staging/bridge_visual_audit/report_precut_642.md); the same
audit re-aggregated over the bank as it now stands is in
[`report.md`](../../staging/bridge_visual_audit/report.md).

**Two related groups are still IN the bank**, pending a separate decision — they are listed
under `not_excluded_yet` in [`visual_exclusions.json`](visual_exclusions.json):

| kind | scenarios | what it is |
|---|---|---|
| `explicit_pointer` | 74 | names a shared artifact — "the pink rectangle", "look at the whiteboard" |
| `labelled_option` | 27 | the student answers with a label only a figure defines — `"b"`, `"picture 3"` |

> **The classification is a model's judgment, not a verified label.** Precision was
> hand-checked on ten cases and held; recall was not measured, and no random sample of the
> 278 it cleared has been reviewed. Treat 41% as an estimate with an unmeasured error bar.

### `visible_mistake` — flagged, not excluded

**2 scenarios (42 criteria) carry `visible_mistake: false`.** Their final student turn
is a bare acknowledgment ("yes", "done", "no", …) **and** no digit appears anywhere
in the conversation, so the transcript is only tutor turns plus an assent:

```
[tutor]   You are doing a good job.
[tutor]   So far so good. keep going.
[tutor]   Are you done with your answer?
[student] "yes"          ← the "mistake"
```

There is no student position to remediate, so D1/D2 (**both `critical`**) are effectively
unpassable and will read as floor items. They are **kept** so the bank stays complete, and
flagged so a calibration run can filter or model them explicitly — `[s for s in scenarios if
s["visible_mistake"]]` gives the 237 with a remediable error.

This group shrank from 91 to 2 across the cuts, and that overlap is worth
reading: **82 of the original 91 were also flagged as missing their problem statement.**
The flag was largely detecting a missing-context defect, not an affective one.

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

`use_case` is `mistake_remediation`, but the response generator keys its system prompt on
the **benchmark name** first, and `SYSTEM_PROMPTS_BY_BENCHMARK["Bridge"]`
([`tutor_cat/respgen/prompts.py`](../../tutor_cat/respgen/prompts.py)) supplies a dedicated
one, so the `use_case` path never fires. The multi-turn context is preserved either way.

> **The Bridge system prompt overlaps three criteria.** It instructs the model to "Identify
> the specific error", "guid[e] them toward the right approach rather than simply giving away
> the answer", and "Keep a supportive, encouraging tone" — close to a verbatim statement of
> **D1**, **P1** and **A1**, which are therefore marked `explicitness: "explicit"`; every
> other criterion stays `implicit`, since the scenario prompt (a student turn like `"4 m"`)
> asks for nothing. The expectation was that all three would sit at ceiling. The pilot below
> shows only A1 does — being *told* to identify the error turns out not to mean models do it.

## Pilot validation

The criterion wording was tuned against a real judging run rather than by argument.
**8 tutors spanning `gpt-4.1-nano` → `claude-opus-4-8` × 100 stratified scenarios = 790
graded responses / ~16,200 criterion judgments**, using the production Bridge system prompt
and `gpt-5.5` as judge. Harness and full report: `staging/bridge_pilot/`.

`discrimination` below is the corrected item–total correlation (how well a criterion tracks
overall response quality, self excluded). Healthy = pass rate between 0.05 and 0.95 with
discrimination ≥ 0.20; outside that a criterion carries little information for calibration.

| | baseline | after tuning |
|---|---|---|
| at ceiling (pass ≥ 0.95) | 8 | **1** |
| at floor (pass ≤ 0.05) | 0 | **0** |
| no discrimination (< 0.10) | 4 | 6 |
| strong (discrimination ≥ 0.30) | 17 | **21** |
| **healthy overall** | 23 / 39 | **28 / 39** |

Three findings overturned the design assumptions:

1. **D1 and P1 are among the best items, not freebies.** Despite the system prompt naming
   them, D1 passes only 0.218 (discrimination 0.499) and P1 0.594 (0.629). Only **A1** is
   the dead weight that was predicted (0.914 / 0.091).
2. **Negative wording did inflate pass rates** — 0.866 mean vs 0.646 for positive-form
   criteria. Six codes were rewritten toward a positive, specific demand; negative-form
   codes went 19 → 13.
3. **The sibling criteria are NOT redundant.** Highest within-family correlation is P2–P4 at
   phi 0.538; C1–C3 is 0.079. A1–A4 top out at 0.347. Nothing approaches the 0.7 threshold,
   so the four affective / four strategy / four math criteria measure distinct things.

### Criteria still carrying little information

| code | pass rate | disc | note |
|---|---|---|---|
| `A3` | 0.996 | 0.011 | **kept deliberately** — a tutor demeaning a child *is* a critical failure, so this stays as a safety tripwire. Exclude it from an IRT fit rather than delete the only check on it. |
| `P3` | 0.881 | 0.007 | two rewrites failed to move it; overlaps P1 conceptually |
| `F1` | 0.143 | 0.029 | variance recovered after an overcorrection, but discrimination did not |
| `A1` | 0.914 | 0.091 | named verbatim by the system prompt |
| `S1` | 0.863 | 0.092 | `critical` |
| `D5` | 0.596 | 0.093 | good variance, weak discrimination |

### Validation against Bridge's own experts

The strongest check available without human raters: grade each scenario's
`reference_solution` — what a **real expert tutor actually wrote** — as if a model had
produced it. A criterion the experts fail is far more likely to be a bad criterion than a
bad expert.

The first run failed that check.

| | mean pass rate |
|---|---|
| 8 AI tutors | 0.669 |
| Human experts | 0.681 |

**A tie.** Expert tutors scored no better than `gpt-4.1-nano`. Six criteria — A1, A2, A4,
C1, X1, Y3 — were failed *more often by the experts than by the models*, A4 by 0.456.

The cause was measurable: **expert replies have a median length of 83 characters; model
replies 380 (4.3×)**. Real expert tutoring reads like

> *"Great try! Can you explain how you got 21?"*  ·  *"Let's recheck your answer. Count up
> the sides again."*

Excellent tutoring — brief, targeted, hands the thinking back. But it never pauses to signal
psychological safety (A4), perform warmth (A1) or lay out a chain of reasoning (C1), so it
failed all three. **The rubric was partly rewarding verbosity over teaching.**

Those six were reworded so a terse reply can pass — two of them stated negatively, since
experts establish safety by simply carrying on helpfully rather than announcing it. Result:

| | before | after |
|---|---|---|
| experts − models | +0.012 (tie) | **+0.052 (experts ahead)** |
| criteria experts fail more | 6 | **3** (X1, I2, A1) |
| healthy criteria | 28/39 | 27/39 |
| at ceiling | 1 | 4 |

**This was a deliberate trade: validity over item statistics.** Letting brief answers pass
made those criteria easier for everyone, so ceiling counts rose. A rubric with sharper
discrimination that ranks chatty models above human teachers is measuring the wrong thing;
one point of "healthy criteria" is worth that.

Three criteria still favour models — X1 (−0.322), I2 (−0.240), A1 (−0.233). Some of this
may be genuine rather than a defect: expert tutors really do drop units (I2) and really do
sometimes skip encouragement entirely (A1). Left as-is and flagged.

**A methodological side finding:** showing the judge a reference answer inflates its
verdicts substantially — M3 0.870 → 0.570 and P2 0.980 → 0.740 when the reference is hidden.
The production judge does show it, so grading there is materially more lenient than a blind
read would be.

**Tuning stopped after three rounds.** Further wording changes fitted to 100 scenarios would
start tracking sample noise; the real calibration is the definitive read.

> **⚠️ The rubric separates models only weakly.** Mean pass rates span 0.699 → 0.788 across
> the 8 tutors, and the ordering is partly scrambled (`gpt-4.1-nano` outscored `gpt-4o`).
> Frontier models do land on top, but a 9-point spread is thin. Removing or fixing the
> remaining low-information criteria should widen it, since they contribute pass marks
> almost uniformly.

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
  of 105 scenarios. `lesson_topic` names the **lesson** a session belongs to, not what the
  graded turn asks, so a "Bar Graphs" lesson can hold a plain subtraction question. This
  is what retired `data_graphing`; the same mismatch may affect other strands and has not
  been audited.
- **Floor items are retained.** The 9 `visible_mistake: false` scenarios (184 criteria)
  are kept deliberately, but D1/D2 are unpassable on them. Filter on the flag, or expect
  those items to carry no information.
- **Duplicate stimuli are retained.** 152 scenarios share a conversation with another and,
  byte-identical apart from `reference_solution`. At temperature 0 a
  tutor returns the same response to each, so these rows are near-copies in the response
  matrix while the judge scores them against different gold keys. Group on `source_id`.
- **⚠️ Tier 3 may not earn its place.** After the topic corrections,
  `operations_arithmetic` holds **105 of 239 scenarios (44%)** while `fractions` has 7,
  `proportional_reasoning` 5 and `algebra_expressions` 10. Each instance is still answered by
  every model, so per-item difficulties remain estimable; what is not estimable is anything
  about a *strand* resting on 5 stimuli. The corrections did not cause this — they revealed
  it. Bridge's graded turns are mostly topic-neutral arithmetic sub-steps, so a topic tier has
  little to gate on. **Retiring Tier 3 entirely is a live option and has not been decided.**
- **The pilot statistics predate the cut.** Every pass rate and discrimination below was
  measured on a 100-scenario sample of the 642-scenario bank, and roughly 55% of that
  sample no longer exists. Directionally the numbers should hold or improve — the removed
  items are the ones a tutor could not answer — but they are not a measurement of the
  current bank.

## Rebuild

```
python scripts/ingest_bridge.py
```

The ingester plus [`visual_exclusions.json`](visual_exclusions.json) are the whole rebuild.
`load_bank` reads this bank cleanly — it infers the 5-skill axis from the criteria's own
`q_mapping` and reports every rubric as `calibrated: False`, which is what keeps an
uncalibrated bank out of CAT selection. To restore the pre-cut 642, delete or empty the
exclusion list and re-run.

[`scripts/assign_irt_params.py`](../../scripts/assign_irt_params.py) can append synthetic
placeholder parameters if some downstream step needs the fields populated for a dry run:

```
python scripts/assign_irt_params.py \
    --input data/Bridge/rubrics.jsonl \
    --skills diagnosis,strategy,math,communication,affective \
    --log-dir data/Bridge/irt_logs --no-backup
```

Anything it writes is synthetic (`irt_params.source == "synthetic"`), so keep it out of the
committed artifact.
