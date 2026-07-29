# Curation → Human Regrading Impact (analysis only)

**Question:** given the pending `curation_v1` (originals `data/rubrics_qmatrix_final.jsonl`
→ curated `data/curated/rubrics_qmatrix_curated.jsonl`), exactly which **human**
labels must be re-done, scoped to the criteria that actually changed?

**Bottom line (one paragraph):** The curation touches almost nothing that a human
already labeled. Of the two human sets, the Q-matrix human-review sample (25 criteria)
has **1** changed criterion and the judge-selection sample (87 criteria) has **15**,
of which only **6** are substantive (3 optional rescopes + 3 content/diagnosis splits)
and **9** are presentation/style orphans that are now `optional`/non-gating and already
excluded from the per-skill acceptance gate. **Minimal required human action:** none is
strictly required to trust prior conclusions; the highest-value optional re-grades are the
**3 split parents in the judge sample** (`tb_0003_c09`, `tb_0336_c03`, `tb_0340_c08`),
which produce **7 new atomic children** needing fresh P/F, plus the **3 tb_0001 rescopes**;
the single Q-matrix hit (`tb_0020_c07`) needs no q-label re-review because a split inherits
the parent's `q_mapping`. Everything else (414 of 430 changed criteria) is outside both
human samples and needs no human action.

> Provenance note: this is **not** a heuristic alignment. Every curated record carries a
> `curation` block with `original_criterion_id` / `from`, so originals and curated ids are
> mapped exactly. No fuzzy scenario+text matching was needed. Verified: renumbered/unchanged
> records have **0** text diffs and **0** q_mapping diffs vs their originals.

---

## Task 1 — Criterion-level delta (exact, id-keyed)

Source **6,462** → curated **6,845** (net **+383**). Full per-criterion table:
`staging/curation_delta.csv` (`original_id, curated_ids, scenario, change_type, note`).

| Bucket | Count | Invalidates prior human grade? |
|--------|------:|--------------------------------|
| **unchanged** (same id + text) | 2,987 | No |
| **renumbered_unchanged** (id changed, text & q_mapping identical) | 3,045 | No — content identical |
| **text-modified** (same criterion, text edited) | **9** | **Yes** (text edited; `q_mapping` unchanged) |
| **split** (1 parent → N atomic children) | **64 parents → 142 children** | **Yes** (prior single grade no longer maps) |
| **merged** (style orphans → 1 optional `style_surface`) | **357 orphans → 313** | **Yes** for the orphan (but presentation-only, all-zero-Q) |
| **added** (new optional presentation criterion) | **349** | n/a — never had a human label |
| **removed** (deleted, no successor) | **0** | — |
| **q_mapping-only change** (text same, Q changed) | **0** | — (would affect calibration only, not grades) |

- **text-modified = 9** = 6 `soften` + 3 `rescope_optional`:
  `tb_0001_c02, tb_0001_c07, tb_0001_c08` (rescope→conditional/optional),
  `tb_0011_c08, tb_0019_c02, tb_0019_c05, tb_0019_c08, tb_0187_c01, tb_0444_c01` (soften wording).
- There are **no pure removals** and **no q_mapping-only changes**: renumbering preserved
  `q_mapping` everywhere, so nothing is a "calibration-only" change that leaves grades valid.

**"Changed" set for human-impact purposes** = text-modified (9) + split parents (64) +
merged orphans (357) + removed (0) = **430 original criterion ids** (excludes unchanged and
renumbered). Sets saved to `staging/_changed_sets.json`.

---

## Task 2 — The two human-labeled samples

**(a) Q-matrix human review** — `qmatrix_human_review/coordinator_manifest.json`
(+ `qmatrix_review_A/B/C.csv`, `adjudications.csv`). This reviewed the **Q-matrix labels**
(content/diagnosis/scaffolding) of **25 unique criteria** (10 CORE ×3 reviewers, plus AB/AC/BC pairs).
IDs: `tb_0016_c04, tb_0020_c07, tb_0047_c07, tb_0056_c06, tb_0091_c06, tb_0113_c01,
tb_0120_c02, tb_0125_c01, tb_0129_c01, tb_0162_c06, tb_0215_c05, tb_0219_c09, tb_0240_c07,
tb_0268_c03, tb_0314_c04, tb_0378_c14, tb_0433_c05, tb_0546_c07, tb_0550_c04, tb_0554_c02,
tb_0569_c08, tb_0576_c05, tb_0582_c01, tb_0615_c06, tb_0620_c05`.

**(b) Judge-selection human grades** — `grader_packets/` (`sample_scenarios.jsonl`,
`sample_rubrics.jsonl`, `grader_01..06.csv`, human P/F + notes; workflow in
`docs/judge_validation.md`). These are the human pass/fail labels used to pick the judge:
**87 unique criteria** across **10 scenarios × 3 tutors = 261 graded cases**. (No
`runs/judge_validation_v1/` exists; the live artifacts are the grader packets, and the AWS
handoff bundle `aws_judge_handoff/`.) The 87 ids span scenarios `tb_0001, tb_0003, tb_0012,
tb_0102, tb_0335, tb_0336, tb_0340, tb_0355, tb_0497, tb_0500, tb_0507` (full list in
`staging/_crossref.json`).

---

## Task 3 — Cross-reference (changed ∩ each human sample)

### (a) Q-matrix human-review re-review needed — **1 criterion**
- `tb_0020_c07` — **split** into `tb_0020_c07 / c08 / c09`.
- Impact: split children **inherit the parent's `q_mapping`**, so the human's Q-label carries
  over unchanged. Q-label re-review is **optional/minimal** (confirm the inherited label still
  fits each of the 3 narrower children). **0** of the 25 had a text-only or q_mapping-only change.

### (b) Judge-selection human re-grade needed — **15 of 87 criteria** (only 6 substantive)
- **text-modified (3):** `tb_0001_c02, tb_0001_c07, tb_0001_c08` — rescoped to conditional/`optional`
  (judge may mark N/A). Prior P/F was against the old unconditional text. *(Note: `tb_0001_c02` is
  already an existing human-label adjudication candidate flagged in `docs/judge_validation.md`.)*
- **split parents (3):** `tb_0003_c09, tb_0336_c03, tb_0340_c08` → **7 new atomic children**
  needing fresh human P/F: `tb_0003_c09/c10/c11`, `tb_0336_c03/c04`, `tb_0340_c08/c09`.
- **merged presentation orphans (9):** `tb_0003_c10, tb_0012_c07, tb_0335_c08, tb_0336_c09,
  tb_0336_c10, tb_0340_c10, tb_0497_c05, tb_0500_c10, tb_0507_c06` — all-zero-Q style/persona
  criteria consolidated into one `optional` `style_surface` criterion. These are non-gating and
  already reported as **`unmapped`** (excluded from the per-skill acceptance gate) in the judge report.

**Do the judge-selection agreement metrics need recompute?** **No, not to trust the judge
choice.** Judge selection is a *relative* ranking of 5 judges over an *identical* case set, so
15 equally-affected criteria (of 87) cannot flip which judge wins — and 9 of the 15 are optional
presentation items that don't enter the per-skill gate at all. Only 6 substantive criteria moved.
Recommendation: treat this as a **stable** selection. If/when the curated bank becomes the
official instrument, optionally (i) re-grade the **7 new split children** + **3 tb_0001 rescopes**
against the curated text and (ii) regenerate `comparison_summary.json` as a robustness check — this
is a small handful, not a full re-run, and is not expected to change the ranking.

### New (added/split) criteria with no human labels
- **349 added** optional `style_surface` presentation criteria have no source and no prior human
  label. **12** fall in Sample-A scenarios and **2** fall in Sample-B scenarios
  (`tb_0001_c11, tb_0355_c10`). Because they are optional/non-gating, adding them to the human sets
  is **low priority** (only do so if you want presentation coverage in the human gold set).
- **7 split children** in Sample-B scenarios (listed above) are the genuinely new atomic items that
  *would* need human P/F if the curated bank is graded.

### Changed criteria in NEITHER human sample → **NO human action** — **414 of 430**
- Breakdown: text-modified **6**, split parents **60**, merged orphans **348**.
- These need only downstream **judge re-grade** on the curated text and (later) **Q-relabel**
  during calibration — **no human regrading**.

---

## Minimal human-action checklist
1. **Q-matrix labels to re-review: 1 (optional)** — `tb_0020_c07` (inherited label likely fine).
2. **Judge-selection items to re-grade: 6 substantive (optional)** —
   splits `tb_0003_c09, tb_0336_c03, tb_0340_c08` (→ 7 new children) + rescopes
   `tb_0001_c02, tb_0001_c07, tb_0001_c08`. The 9 presentation orphans are non-gating.
3. **Judge-selection metric recompute:** not required; selection is stable. Optional robustness
   re-run only if the curated bank becomes the graded instrument.
4. **Everything else (414/430 changed criteria):** **no human action.**

### Caveats
- Id alignment is **exact via `curation` provenance**, not heuristic — high confidence.
- "Invalidates prior grade" for splits/text-modified is by construction (the graded text changed);
  a human might judge some edits cosmetic, but they are flagged conservatively.
- Judge-selection stability argument assumes the standard relative-ranking use of the agreement
  metrics (as in `docs/judge_validation.md`); it is not a claim that absolute metric values are unchanged.

*Artifacts:* `staging/curation_delta.csv`, `staging/_changed_sets.json`, `staging/_crossref.json`;
reproduce with `scripts/_build_delta.py` then `scripts/_crossref.py`.
