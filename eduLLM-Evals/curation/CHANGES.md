# Rubric curation — what changed and why (`curation_v1`)

**Use `data/curated/rubrics_qmatrix_curated.jsonl` + `data/curated/scenarios_curated.jsonl`.**
These are the finalized criterion bank (the originals in `data/*.jsonl` are the
untouched TutorBench source, kept for provenance). Nothing here changes IRT/
calibration params — it edits the **criteria** the judge grades. Run judges on the
curated bank first, then (re-)derive IRT params on the frozen criterion set.

This curation addressed four issues found during human grading of the TutorBench
criteria: criteria that were **non-atomic** (bundled several checks into one P/F),
**rigid/verbatim** (a correct-but-differently-worded answer would fail),
**irrelevant/mis-scoped** (graded presentation or restated already-given content),
and inconsistent **presentation** coverage.

## Summary

| Change | Count | Net Δ criteria |
|--------|------:|---------------:|
| Atomic **splits** (1 bundled criterion → N atomic ones) | 64 parents → 142 children | +78 |
| **Soften** wording (remove absolute-failure / reference-leak / prescribed phrasing) | 8 | 0 |
| **Presentation consolidation** (merge a scenario's style orphans → 1 optional criterion) | 357 orphans → 313 | −44 |
| **Presentation added** (give every scenario one optional style criterion) | 349 | +349 |
| **Rescope → optional** (conditional criteria the judge may mark N/A) | 3 | 0 |
| `restatement` / `numeric-tolerance` | reviewed, **kept** | 0 |
| Bank total | **6,462 → 6,845** | **+383** |

Every one of the 662 scenarios now has exactly one optional presentation criterion.
Full per-criterion before/after log: `curation/pilot_change_report.{md,csv}`.
Policy + decision rules: `curation/CURATION_POLICY.md`. Bank-wide grading rules
the judge applies: `curation/grading_notes.md`.

---

## 1. Atomic splits — non-atomic criteria

A criterion that bundled 2+ independently gradeable requirements is split into
atomic children (children inherit the parent's `criticality` / `q_mapping` /
skill unless overridden; `q_rationale` is regenerated per child).

**Example — `tb_0003_c09`:**
- **Before (1 criterion):** "The response must provide explanations … you made 3 errors …
  \(\frac{du}{dx}=\frac{dy}{dx}\), \(\int \frac{du}{\cos(u)}=\sin(u)\), and
  \(\int dx = Cx\) … lead to your incorrect final answer."
- **After (3 criteria):**
  - `tb_0003_c09`: identify & explain the **first** error \(\frac{du}{dx}=\frac{dy}{dx}\) is incorrect.
  - `tb_0003_c10`: identify & explain the **second** error \(\int \frac{du}{\cos(u)}=\sin(u)\) is incorrect.
  - `tb_0003_c11`: identify & explain the **third** error \(\int dx = Cx\) is incorrect.

Only genuinely bundled criteria were split; mechanically-linked chains, a single
concept, and definition+explanation were **kept** as one (see policy). Flag
precision was low (~7–15%), so most `non_atomic` flags were reviewed and kept.

## 2. Soften — rigid / over-specified wording

Removed phrasing that would fail a correct answer: "leads to failure" absolutes,
Golden/reference-solution references, and prescribed exact structure.

**Example — `tb_0019_c02`:**
- **Before:** "…define both \(n=10\) … and \(p=0.35\) … **Any missing label or undefined parameter leads to failure.**"
- **After:** "…define both \(n=10\) … and \(p=0.35\) … in context." (the absolute-failure clause dropped)

**Example — `tb_0011_c08`:** "…structure similar to the **Golden Response** (e.g. …)"
→ "…a pedagogically effective logical order (e.g. …)" (reference-leak removed).

The broader "accept semantically equivalent answers" concern (the bulk of the
`rigid_verbatim` flags were false positives — "state that <fact>" content
criteria) is handled **bank-wide** by `curation/grading_notes.md`, which the judge
loads, rather than by rewriting hundreds of criteria.

## 3. Presentation — consolidation + normalization

All-zero-Q style/persona criteria (second person, Markdown, LaTeX, code formatting)
are **not** skill signals — they're already excluded from MIRT calibration. We made
them consistent and non-gating:

- **Consolidation:** each scenario's style orphans merge into **one** `optional`
  criterion tagged `dimension: style_surface`, covering only the aspects present
  (persona / structure / math / code).
  **Example — `tb_0004`:** "…logically sequenced into labelled steps and format the
  equations in LaTeX" → "The response should follow tutoring presentation
  conventions: use clear Markdown structure …; render mathematical expressions in
  LaTeX. (Style/presentation only …)".
- **Normalization:** the 349 scenarios that had no style criterion each get one
  default (subject-appropriate: LaTeX for math subjects, code for CS), also
  `optional` + `style_surface`.

These carry `judge_guidance` marking them non-gating (mark N/A if inapplicable) and
`optional: true`.

## 4. Rescope → optional — conditional criteria

A few criteria demanded a formula only relevant if the tutor re-derives from scratch.
They're rewritten as conditional + `optional` with judge guidance to mark N/A when
the condition doesn't hold.

**Example — `tb_0001_c02`:**
- **Before:** "The response must include the formula … pH to [H⁺] conversion …"
- **After:** "**If the response re-derives** the ionization from scratch, it must
  include the pH→[H⁺] conversion formula …" — `judge_guidance`: "OPTIONAL. Only
  applies if the response chooses to re-derive … otherwise mark N/A (leave blank)."

## 5. Kept as-is (reviewed false positives)

- **`restatement_of_formula_or_answer`** (~90): the follow-up turn almost always
  *asks* for the answer/formula (student asks, asks to verify, or is confused), so
  these are legitimate content criteria — **kept**. Withhold variants ("must not
  give the answer") were never touched.
- **`numeric_no_tolerance`** (80): covered by the grading note (equivalent notation
  + rounding/sig-figs) — no per-criterion edit.

---

## Provenance

Every edited/added record carries a `curation` block (`policy_version`, `op`,
`from`, `original_criterion_id`, and for presentation criteria the matched
`aspects`). Regenerate the curated bank at any time with:

```
python scripts/apply_curation.py            # source + curation/pilot_edits.json -> data/curated/
python scripts/verify_curated.py            # integrity checks
```
