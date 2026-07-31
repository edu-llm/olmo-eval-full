# Gate A — Proposed Scenario / Question-Text Edits (FOR REVIEW) — **v2 (re-scoped)**

> ## ⚠️ THESE ARE PROPOSALS. NOTHING HAS BEEN APPLIED.
> - **No** edits were made to `data/TutorBench/curated/scenarios_curated.jsonl`.
> - **No** edits were made to the rubric bank (`rubrics_qmatrix_curated.jsonl`).
> - **Nothing** was committed.
> - The only files changed are this memo + the structured proposal (`staging/revision_evidence/proposed_scenario_edits.jsonl` / `.csv`).

**What v2 is.** v1 proposed 6 `reword_ambiguous` + 11 `mis_specified_fix` full rewrites + 15 scaffolding adds + 169 short suggestions. The reviewer pushed back. v2 **verifies every reword against the authoritative Hugging Face original (`ScaleAI/TutorBench`), matched by content** (our `tb_XXXX` ids do not exist upstream), and re-scopes hard. **Result: 2 edits survive** (two one-word typo fixes); everything else is dropped or skipped.

---

## Headline

- **Only 2 edits are still recommended**, both trivial source typos: `tb_0222` (`than than` → `than`) and `tb_0207` (`What do mean` → `What do you mean`). Nothing else.
- **Every "defect" we flagged in the reword set was verified against the HF original.** The two most important flags in v1 were **wrong**:
  - **`tb_0319`** — the "incorrect √34 chord" tutor turn is **in the HF original by design**. The rubric *explicitly grades the model for catching that √34 is wrong* and deriving the correct ≈11.19 m. v1 would have **handed over the graded answer.** → **DO NOT EDIT.**
  - **`tb_0347`** — the `word1.lenth()` typo is **a rubric target** (criterion 0 grades identifying the compile error). "Fixing" it would delete a criterion. → **DO NOT EDIT.**
- **No conversation-context turns are touched at all.** All three context edits flagged in v1 (`tb_0319`, `tb_0201`, `tb_0230`) are dropped.
- **All 11 `mis_specified_fix` full rewrites are SKIPPED.** Each student prompt is already coherent and directly answerable; the "unstated" content the rubric wants is **intentionally implicit** — the benchmark tests whether the tutor *proactively surfaces* it. Making the ask explicit weakens the criterion.
- **All 15 scaffolding adds dropped** (reviewer: unnatural student phrasing).
- **All 169 short `mis_specified` suggestions dropped** (folded into skip-by-default).

**Verification method.** HF dataset `ScaleAI/TutorBench` (renamed from `scaleai/tutorbench`; `tutorbench/tutorbench` is only a ~30-row sample and was **not** used). Rows fetched individually via the Dataset Viewer `/search` API and matched by **follow-up text** (the initial explanation is reused across sibling tasks, so matching on the problem/explanation alone returns the wrong sibling). Field map: `PROMPT`→`context.student[0]`, `UC1_INITIAL_EXPLANATION`→`context.tutor[1]`, `FOLLOW_UP_PROMPT`→our `prompt`. For all 6 rewords, **our ingested layer and curated layer are byte-identical**, and both match the HF original — i.e. **no ingestion/curation corruption exists** in any of them.

---

## (A) Verification table — 9 scenarios (6 reword + 3 context-turn)

| scenario | HF row | classification | evidence from original | recommended action |
|---|---|---|---|---|
| **tb_0319** (calc) | 462 | **INTENTIONAL_DESIGN** | √34≈5.83 m straight-chord answer is in the original `UC1_INITIAL_EXPLANATION`. Rubric **crit[2]** grades that √34 is *incorrect*; **crit[3]** grades the correct ≈**11.19 m** ((3^{2/3}+5^{2/3})^{3/2}); crit[6–9] grade the L=A/cosθ+B/sinθ derivation. The wrong tutor turn **is the item**. | **DROP.** Editing the context turn hands over the graded answer. |
| **tb_0347** (cs) | 830 | **INTENTIONAL_DESIGN** | `UC1` empty (feedback task). Rubric **crit[0]** *explicitly* grades identifying `word1.lenth()` as a compile error. The typo is a target, not noise. | **DROP.** Fixing the typo deletes a criterion. |
| **tb_0201** (chem) | 344 | **NOT_AN_ERROR** | Identical to HF. The tutor turn itself defines "increasing negative" = *becoming more negative*; rubric grades the correct order N<P<F<Cl. Standard terminology, already disambiguated in-context; not ungradable. | **DROP.** Do not alter the HF source problem. |
| **tb_0222** (phys) | 365 | **SOURCE_ERROR** (typo); scenario design intentional | Identical incl `greater than **than**`. Rubric grades that the torques **subtract** (τ_net = τ₁−τ₂ = 4.2 N·m) and that the tutor's 5.8 N·m is a **mistake to catch** — the wrong turn is intentional. | **KEEP — minimal.** Fix `than than` only; name the two referents without values/sign. |
| **tb_0207** (chem) | 350 | **SOURCE_ERROR** (typo) | Identical incl `What do mean by` (missing "you") and the "mole triangles" wording. Question is coherent and gradable. | **KEEP — minimal.** Fix `What do mean` → `What do you mean` only. |
| **tb_0230** (calc) | 373 | **SOURCE_ERROR** (benign) | Identical incl the malformed `r(\θ)` LaTeX in the tutor turn. It is in the **source** model output (not our ingestion), is **not graded** by any criterion, and does not block grading. | **DROP.** Not ungradable → leave as-is. |

*Context-turn scenarios (`tb_0319`, `tb_0201`, `tb_0230`) are the last-flagged subset; all three are dropped, so **no context turn is modified.***

### tb_0319 specifically (as requested)
The incorrect √34 straight-chord tutor turn is **present in the HF original by design**, not a parsing artifact of ours. The rubric is built *around* that wrong turn: it grades the model for stating √34 is incorrect and producing the true optimization (≈11.19 m). **We do not correct it.** v1's plan to replace the context turn with the correct set-up would have made the item trivially passable.

---

## (B) Re-scoped `reword_ambiguous` — minimal fixes only (2 survive)

Rule applied: smallest change that fixes the actual defect; **do not restate context** the model should pull from the conversation (pulling from context is part of what's evaluated).

### `tb_0222` — physics — edit: `prompt`
**BEFORE:** `What does it means when the net torque is greater than than each individual torque?`
**AFTER:** `What does it mean when the net torque is greater than each individual torque (the applied-force torque and the friction torque)?`
*Fix: the `than than` typo (and `means`→`mean`); a minimal, value-free referent for "each individual torque." **No** re-listing of 5 / 0.8 / 5.8 N·m, and nothing that reveals that the torques actually subtract.*

### `tb_0207` — chemistry — edit: `prompt`
**BEFORE:** `What do mean by 0.25 mol/L × 0.100 L = 0.025 mol?? Isn't moles calculated by mass not volume? …`
**AFTER:** `What do you mean by 0.25 mol/L × 0.100 L = 0.025 mol? Isn't moles calculated by mass not volume? I'm confused, are there more mole triangles used to calculate moles, if so, what are they?`
*Fix: `What do mean` → `What do you mean` (and the doubled `??`). The student's original phrasing is otherwise preserved; v1's added mass-route / when-to-use clauses are dropped.*

### Dropped from the reword set
`tb_0347` (intentional graded typo), `tb_0201` (not an error), `tb_0319` (intentional wrong-answer design), `tb_0230` (benign, ungraded source render glitch).

---

## (C) `mis_specified_fix` reconsidered — **all 11 SKIP**

The reviewer's concern is correct and is confirmed by the HF rubrics we read: TutorBench criteria are **overwhelmingly `implicit` / `subjective` by design** (e.g. tb_0201 has 15 of 16 criteria `implicit`; tb_0230 likewise). The benchmark's signal is precisely whether the tutor *proactively* surfaces unstated content. For each of the 11, the student prompt is coherent and directly answerable, and the content v1 wanted to make explicit is the implicit target itself → **editing weakens the criterion.**

| scenario | why the "unstated" content is intentionally implicit → SKIP |
|---|---|
| tb_0108 | Student literally asks "what a water shell is or how it has energy" — the definition/dehydration-energy items are the proactive-teaching target. |
| tb_0010 | "Why do we add the probabilities?" already invites naming the addition rule; naming it is the implicit test. |
| tb_0086 | "What would x and n be?" — answer is "there is no data set"; the when-it-applies contrast is proactive. |
| tb_0181 | Asks "why not Ka directly? can I use Henderson–Hasselbalch?" — species-in-solution / Ka-vs-Kb is what a good answer surfaces. |
| tb_0013 | Asks for the definition of "conditionally independent"; the not-independent counter-example is the proactive part. |
| tb_0124 | Asks the H-bond/paper-folding question directly; "a better analogy" is proactive. |
| tb_0063 | Asks "is that delay all known ahead of time?" — deterministic-vs-stochastic is already invited. |
| tb_0146 | Asks both "how does SMAD4 activate a gene?" and "why is losing a growth factor bad?" directly. |
| tb_0112 | Asks "why is one side favoured?" — carbocation stability is exactly the answer. |
| tb_0064 | Bundled + the "1 in 8 eventually = 12.5%" figure is the **student's misconception** (lifetime incidence vs point prevalence) for the tutor to correct — not a scenario defect. Answerable → SKIP. |
| tb_0279 | Student's interval implies they used t; diagnosing t-vs-z (population variances known → z) is a diagnosis test, answerable → SKIP. |

**Count: 0 survive, 11 skip.** None is literally unanswerable or misdirected as posed, so none clears the "minimal explicit ask justified" bar.

---

## (D) `add_scaffolding_question` (15) — **DROPPED**
Reviewer rejected all 15: the appended "give me a hint / let me try first" clause reads as **unnatural student phrasing**. Removed from the proposal (`change_group = dropped_scaffolding`). The underlying coverage gap (241/662 scenarios have no scaffolding criterion) is real but is a **Gate B** authoring problem, not a Gate A prompt-injection problem.

## (E) 169 short `mis_specified` suggestions — **DROPPED**
Folded into the skip-by-default reasoning in (C) (`change_group = skip_implicit`). "Hard ≠ broken"; implicit criteria are by design.

---

## What changed from v1

| | v1 | v2 |
|---|---:|---:|
| `reword_ambiguous` applied | 6 | **2** (typo-only) |
| context-turn edits | 3 | **0** |
| `mis_specified_fix` full rewrites | 11 | **0** (all skip) |
| short `mis_specified` suggestions | 169 | **0** |
| scaffolding adds | 15 | **0** |
| **total recommended edits** | **201+** | **2** |

Specific reversals:
- **tb_0319**: v1 "correct the √34 tutor turn" → v2 **DROP** (√34 is the graded error; correcting it trivializes the item).
- **tb_0347**: v1 "fix `lenth()`" → v2 **DROP** (`lenth()` is a graded criterion target).
- **tb_0201**: v1 "disambiguate the ordering in the source problem" → v2 **DROP** (tutor already defines the term; standard usage; not ungradable).
- **tb_0230**: v1 "fix `r(\θ)` LaTeX" → v2 **DROP** (benign, ungraded source glitch).
- **tb_0222 / tb_0207**: v1 expanded rewrites → v2 **shrunk to one-word typo fixes** with no context restatement.
- 11 `mis_specified` full rewrites → **all SKIP** (implicit-by-design).

---

## Confirmation (nothing applied)
- `data/TutorBench/curated/scenarios_curated.jsonl` — **UNTOUCHED** (git: clean).
- rubric bank `rubrics_qmatrix_curated.jsonl` — **UNTOUCHED** (git: clean).
- Nothing committed.
- Only files written: this memo and `staging/revision_evidence/proposed_scenario_edits.jsonl` / `.csv` (added columns `original_classification`, `verification_evidence`; re-scoped `change_group` values: `reword_ambiguous`×2, `dropped_intentional_design`×2, `dropped_not_an_error`×1, `dropped_source_error_benign`×1, `skip_implicit`×180, `dropped_scaffolding`×15, `defer_too_hard`×70).

## Output paths
- **This memo:** `plans+prds/Gate A - Proposed Scenario Edits (for review).md`
- **Structured proposal:** `staging/revision_evidence/proposed_scenario_edits.csv` / `.jsonl`
  - Columns: `scenario_id, scenario_type, change_group, priority, current_text_snippet, proposed_text, needs_gate_b_criterion, rationale, edit_field, original_classification, verification_evidence`

*Verification performed read-only against `ScaleAI/TutorBench` via the HF Dataset Viewer API (rows matched by content). No scenario/rubric files changed; nothing committed.*
