# Bridge

Bridge reformatted into the Scenario + Rubric schema. Source:
https://huggingface.co/datasets/rose-e-wang/bridge (splits: `train` / `validation` /
`test`, 700 tutor–student math remediation conversations). Paper: *Bridging the
Novice-Expert Gap via Models of Decision-Making: A Case Study on Remediating Math
Mistakes* (Wang et al.).

Built by [`scripts/ingest_bridge.py`](../../scripts/ingest_bridge.py). The records carry no
`difficulty` / `discrimination`: those are calibrated from real judge responses, and until
that fit exists the bank stays parameter-free rather than carrying synthetic stand-ins.

**Current artifact: 172 scenarios · 3,285 criteria · 30 live criteria of a 39-slot bank ·
19–20 criteria per scenario, 7 of them `critical`.**

> **⚠️ 470 of the original 642 scenarios were removed, and the topic tier retired.** Four
> audits, each finding what the last one missed — see [Audit trail](#audit-trail). Bridge
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

## Audit trail

Four passes, each finding what the previous one missed. The pattern is worth stating because
it is the reason for the last one: every pass but the final one asked whether an item *looked*
sound, which measures precision and says nothing about recall.

| # | pass | found | removed |
|---|---|---|---:|
| — | Bridge's own free-text filter | `e` is "no mistake" / "end session" | 58 |
| 1 | [`audit_bridge_visuals.py`](../../scripts/audit_bridge_visuals.py) | the problem statement lives only on the session whiteboard | 263 |
| 2 | same, second round | a figure is referenced and the error is unreadable without it | 90 |
| 3 | [`audit_bridge_items.py`](../../scripts/audit_bridge_items.py) | the graded turn has no gradeable error; 83 wrong topic labels | 50 |
| 4 | [`verify_bridge_items.py`](../../scripts/verify_bridge_items.py) | **adversarial** — wrong gold maths, malformed questions, unanswerable criteria | 67 |

**700 → 172.** The single most consequential fact about Bridge is that it transcribes live
sessions held over a **shared whiteboard that was never saved**, so most of what the tutor and
student were looking at is simply absent. That is not a packaging problem; the visual was
never digitised, and no amount of re-ingesting recovers it.

Pass 4 is the only one that measured recall, by inverting the burden of proof: three lenses
per scenario, each told to assume the item is unfit and prove it, majority of three required.
It found a defect class none of the first three looked for — **gold keys that state wrong
mathematics**. The judge is *shown* the reference, and the pilot separately established that
showing it makes the judge more lenient, so a wrong reference actively misleads grading:

> `bridge_0342` — the student answers **"hexagon"** for a five-sided shape, and the expert's
> gold reply says *"Try that again! The prefix for 5 sides is **hexa**."* It confirms the wrong
> answer and teaches the wrong prefix.

Others: *"a rectangle has two equal sides and two equal angles"*; speed's "formula" given as
*miles per hour*; commutativity offered as an *inverse* operation; `0.621 × 1000` given as
`6210`; and a gold key containing the student's next reply pasted after the tutor's.

**Two things the audits deliberately did NOT change.** The expert `error_module` annotation was
contradicted on 142 scenarios, but 54 of those push `right_idea`/`careless` toward `guess`,
which is exactly what the missing whiteboard produces — the expert saw the student's working
and a text-only reader cannot. The labels stand. Separately, the first version of the
adversarial lenses disqualified `bridge_0000` unanimously, an item deliberately kept in pass 2;
both settled points are now stated in that prompt so the instrument stops re-litigating them.

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

### Tier 3 — topic-domain modules · **RETIRED IN FULL**

All eight codes (V1, V2, F1, N1, U1, X1, Z1, O1, plus B1) are retired. See
[Retired criteria](#retired-criteria) for the measurements. `topic_domain` survives as
scenario metadata but attaches no criterion, so the tier list below is provenance only.

### Tier 4 — grade-band module

`Y1` (grades 1-3, 56) · `Y2` (4-5, 95) · `Y3` (6-12, 21). Derived from the TEKS prefix
(`3.6B…` → grade 3; `A2.7D…` → secondary).

## Retired criteria

Nine of the 39 bank slots attach to nothing. They are **kept in the bank list** so
`CODE_INDEX` — and therefore every existing `criterion_id` — is unchanged; removing an entry
from the middle would renumber every code after it. `validate()` exempts retired codes from
the "never attached" check and asserts they stay unattached.

**The whole of Tier 3.** Its gate keys on `lesson_topic`, which names the *lesson* rather
than the graded turn — a lesson titled "Bar Graphs" ends on `12 − 1`, and 83 of 289 scenarios
needed a hand correction. More decisively, the census pilot over all 239 scenarios found
7 of its 8 live criteria defective:

| code | domain | pass | disc | length-bias | experts − models | verdict |
|---|---|---:|---:|---:|---:|---|
| `V1` | geometry | 0.635 | **0.074** | **+0.291** | −0.105 | no discrimination, rewards length |
| `V2` | geometry | 0.936 | 0.513 | +0.194 | **−0.181** | anti-expert |
| `F1` | fractions | 0.464 | 0.331 | **+0.332** | **−0.179** | rewards length, anti-expert |
| `N1` | place value | **0.954** | 0.317 | +0.000 | −0.022 | ceiling |
| `U1` | measurement | 0.934 | 0.486 | +0.000 | −0.092 | *clean — the only one* |
| `X1` | algebra | **0.963** | 0.329 | +0.000 | **−0.263** | ceiling, anti-expert |
| `Z1` | proportional | 0.675 | **0.182** | +0.000 | **−0.475** | weak, worst anti-expert |
| `O1` | operations | 0.688 | **0.123** | +0.171 | **−0.345** | weak, anti-expert |

Five of the eight criteria human experts fail more often than models were Tier 3, as were two
of the three worst length-biased. The core and error-module criteria are by contrast healthy
and expert-neutral (D2 disc 0.720, P2 0.789, D3 0.783, all within 0.02 of the experts). So the
damage was concentrated in one tier, and keeping U1 alone would have meant retaining an
unreliable routing layer for a single criterion.

**`B1`** (data-display reading) was retired earlier for the same gating defect: after the
visual cut only 6 scenarios carried it and four of those were arithmetic or fraction addition
inside lessons merely *titled* "Bar Graphs".

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
conversation with at least one other (126 unique conversations across 172 scenarios), and
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
| `failed_adversarial_verification` | 67 | a majority of three adversarial lenses disqualified it |
| `no_clear_mistake` | 56 | `e` is free text ("no mistake" / "end session" / "unresponsive") |
| `empty_student_turn` | 2 | final student turn has no text |

700 source rows → 528 dropped → **172 scenarios**. All drops are logged to `dropped.jsonl`
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

**No scenario now carries `visible_mistake: false`** — all 91 were removed by the audits, 82
of them because the problem statement was missing rather than for the affective reason the
flag was built to catch. That is itself a finding: the flag was largely detecting a
missing-context defect, not an unremediable-assent one. The ingester still computes it, so a
future re-inclusion would carry it.

The flag fires when the final student turn is a bare acknowledgment ("yes", "done", …) **and**
no digit appears anywhere in the conversation, leaving only tutor turns plus an assent:

```
[tutor]   You are doing a good job.
[tutor]   So far so good. keep going.
[tutor]   Are you done with your answer?
[student] "yes"          ← the "mistake"
```

There is no student position to remediate, so D1/D2 (**both `critical`**) are unpassable.
`[s for s in scenarios if s["visible_mistake"]]` now returns all 172.

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

## Pilot validation — measured on the whole bank

The rubric was tuned against real judging runs rather than by argument, and re-measured after
every change to the bank. The current figures are a **census, not a sample**: all 172
scenarios × 8 tutors spanning `gpt-4.1-nano` → `claude-opus-4-8` = **1,376 graded responses /
26,280 criterion judgments**, judge `gpt-5.5`, production Bridge system prompt. Harness:
[`staging/bridge_pilot/pilot.py`](../../staging/bridge_pilot/pilot.py) with
`BRIDGE_PILOT_DIR` / `BRIDGE_PILOT_N=all`; report in `staging/bridge_pilot_v6/`.

Sampling was abandoned once the bank got small: a 100-scenario draw would have left the
conditional criteria with a handful of observations each.

### Where it stands

| | 642-bank sample | 239-bank census | **172-bank census** |
|---|---|---|---|
| healthy criteria | 27 / 39 | 22 / 36 | **17 / 30** |
| at ceiling (≥0.95) | 4 | 8 | **9** |
| at floor | 0 | 0 | **0** |
| discrimination ≥0.30 | 21 | 24 | **20** |
| model spread | 0.699–0.788 | 0.808–0.931 | **0.842–0.960** |
| experts − models | +0.052 | **−0.047** | **+0.031** |
| criteria experts fail more | 3 | 8 | **1** |

**Model separation and validity both improved; item difficulty is the cost.** Removing
unanswerable items raised every pass rate, so more criteria drifted to ceiling — 9 of 30. The
ordering is now correct (`claude-opus-4-8` 0.960 and `gpt-5.5` 0.943 on top; on the 642 bank
`gpt-4.1-nano` outscored `gpt-4o`).

### The expert check, which is the one that matters

Grading Bridge's own `reference_solution` — what a real expert tutor actually wrote — as if a
model had produced it. A criterion the experts fail is far more likely to be a bad criterion
than a bad expert.

On the 239-scenario bank this **regressed**: models 0.836 against experts 0.789, so the rubric
ranked eight AI tutors above human teachers, with 8 criteria failed more often by the experts.
Two things caused it, and both are now fixed:

- **Tier 3** supplied five of those eight criteria. Retired in full (above).
- **D1** asked the tutor to *identify* the error. Bridge's experts answer a wrong value with
  *"Can you explain how you got 21?"* — which locates the error precisely without naming it,
  and the old wording scored that as a miss. D1 now credits directing the student to the exact
  step that went wrong; a generic "try again" still fails. Its discrimination **rose** with the
  rewording, 0.333 → **0.548**.

Result: **experts 0.915 against models 0.884 (+0.031)**, and the list of criteria the experts
fail more often is down from 8 to **one**.

### The one remaining: A1

`A1` (conveys some encouragement) sits at pass 0.850, discrimination **−0.042**, experts
−0.269. Real expert tutors frequently skip encouragement entirely and get straight to the
maths, so this is arguably a genuine difference in style rather than a defect — but a
criterion with negative discrimination is measuring nothing either way.

### Criteria still carrying little information

| code | pass | disc | note |
|---|---:|---:|---|
| `A1` | 0.850 | −0.042 | above; named verbatim by the system prompt |
| `D5` | 0.838 | −0.028 | two rewrites have failed to move it |
| `A4` | 0.996 | −0.015 | ceiling |
| `A3` | 0.997 | 0.020 | **kept deliberately** — a tutor demeaning a child *is* a critical failure, so this stays as a safety tripwire. Exclude it from an IRT fit rather than delete the only check on it. |
| `P3` | 0.820 | 0.107 | overlaps P1 conceptually |
| `S1` | 0.879 | 0.124 | `critical` |

> **⚠️ The `affective` dimension has no informative pure anchor.** Its three pure-loading
> criteria are A1 (−0.042), A3 (0.020) and A4 (−0.015) — all three at or below zero
> discrimination. Only A2 (0.443) carries signal, and it is not pure. A confirmatory M2PL will
> struggle to identify this dimension from the data; treat any `affective` θ with suspicion
> until the criteria are rewritten.

**The strongest items** are the diagnosis and strategy core: P2 0.768, C3 0.708, M3 0.701,
D2 0.674, D3 0.667, S2 0.593 — all with healthy variance and all within 0.02 of the experts.

### Length bias

Mean length↔pass correlation across 30 criteria: **−0.085**, with 4 flagged beyond ±0.25
(Y3, S2, Y1, S1) — down from 7 on the previous bank. This is a standing check because the
rubric was once rewarding verbosity: expert replies have a median length of 83 characters
against the models' 380, and six criteria originally demanded something a terse reply could
not produce.

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
  of the bank. `lesson_topic` names the **lesson** a session belongs to, not what the
  graded turn asks, so a "Bar Graphs" lesson can hold a plain subtraction question. This
  is what retired `data_graphing`; the same mismatch may affect other strands and has not
  been audited.
- **Floor items are gone.** The `visible_mistake: false` group is now empty; D1/D2 are
  answerable on every remaining scenario.
- **Duplicate stimuli are retained.** 92 scenarios share a conversation with another and,
  byte-identical apart from `reference_solution`. At temperature 0 a
  tutor returns the same response to each, so these rows are near-copies in the response
  matrix while the judge scores them against different gold keys. Group on `source_id`.
- **Tier 3 is gone, and that was the right call.** The census pilot found 7 of its 8 live
  criteria defective: V1 (disc 0.074, length-bias +0.291), V2 (experts −0.181), F1
  (length-bias +0.332, experts −0.179), N1 (ceiling 0.954), X1 (ceiling 0.963, experts
  −0.263), Z1 (disc 0.182, experts −0.475), O1 (disc 0.123, experts −0.345). Only U1 was
  clean, and keeping one criterion did not justify retaining a routing layer whose gate had
  already proven unreliable twice.
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
