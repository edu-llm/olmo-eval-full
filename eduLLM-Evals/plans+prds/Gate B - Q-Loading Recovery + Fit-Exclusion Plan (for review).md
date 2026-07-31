# Gate B — Q-Loading Recovery + Fit-Exclusion Plan (FOR REVIEW)

> ## ⚠️ NOTHING HAS BEEN APPLIED. THIS IS A REVIEW-ONLY PLAN.
> - **No** Q-loadings written to `data/TutorBench/curated/rubrics_qmatrix_curated.jsonl` (the bank).
> - **No** edits to `staging/response_matrix.csv` or `staging/calibration_mirt_full2skill.csv` (fitter inputs).
> - **No** items deleted from the bank. The exclusion list is a **psychometric fit-mask**, not a bank edit.
> - **Nothing** committed.
> - Files written by this task: this memo + `staging/revision_evidence/qloading_recovery_plan.csv` / `.jsonl` (both under gitignored `staging/`) + the read-only helper `staging/revision_evidence/_gateb_recovery.py`.
>
> **Bank git-state caveat (honest):** at the time of writing, `git status` shows the bank with **one pre-existing modified line** (the tb_0532_c06 reword + the tb_0370 "56→69 in²" fix — the two Gate B *text* edits). **This task did not make that change** — the helper opens the bank read-only. The change predates this plan; it is flagged here so the reviewer knows the bank is **not** pristine, but **nothing in this task touched it.**

**Scope.** Two difficulty-neutral, text-preserving psychometric actions carried forward from the Gate B proposal:
1. **Fit-exclusion list** — the 60 reverse-behaving (`a<0`) items to mask from the M2PL fit (kept in the bank), each tagged.
2. **Q-loading recovery plan** — a per-criterion axis assignment (`presentation` / `scaffolding` / `trim`) for all 640 `missing_Q_row` criteria that currently load no skill and therefore never enter the fit, each variance-checked against the 82-model response matrix.

Per the pre-registered design: modeled axes are **correctness** (= content ∪ diagnosis, collapsed), **scaffolding**, and **presentation** (optional 3rd axis). **No new "affect" axis** — affect/tone criteria map to **presentation**. **No criterion text is changed** (Gate A/Gate B lesson).

---

## Headline numbers

| Quantity | Value |
|---|---|
| Reverse items excluded from fit | **60** (degenerate **29** · low-info-fine **30** · judge-inverted **1**) |
| `missing_Q_row` criteria classified | **640** |
| → recover as **presentation** | **634** |
| → recover as **scaffolding** | **5** |
| → **trim** | **1** |
| Recovers passing the variance check (non-degenerate, usable in fit) | **612 / 639** |
| Recovers failing variance (all-pass/all-fail → add ~zero fit info) | **27** (all presentation) |
| Low-confidence classifications (no keyword; defaulted to presentation) | **108** |
| Could-not-classify (hard failures) | **0** |

**Bottom line up front:** presentation is the real prize — recovery **stands up a previously-empty pre-registered axis with 607 fit-usable items**. Scaffolding, the weak axis, is **NOT** rescued by this recovery: only **5** genuine scaffolding items exist in the Q-less set, and because 30 of the 60 excluded reverse artifacts were scaffolding-loaded, **scaffolding's fit-eligible count actually goes *down* slightly** (−25 net). The scaffolding rescue Gate B anticipated must come from the **3-skill re-estimation of the "FINE" items**, not from recovering no-skill criteria.

---

## TASK 1 — Fit-exclusion list (the 60 reverse-behaving items)

These carry a fitted `a<0`. They are **kept in the bank** and **masked from the M2PL fit** because a negative discrimination is either a numeric artifact (sparse cells / axis-collapse) or a single judge-inversion — not a signal that the criterion is broken. Evidence per row in the CSV includes fitted `a`, `b`, `pass_rate`, and the **recomputed raw point-biserial** of item-pass vs. model ability (mean pass across all items), plus the recomputed pass count `k/82`.

### Exclusion tags

| Tag | n | Meaning | Variance in resp. matrix |
|---|---:|---|---|
| **degenerate** | **29** | extreme fit (a≈−4.09, b≈18.3) or near-zero-variance / sparse-cell / no-data artifact | 25 have `0<k<82` but near-zero (mostly k=1), 3 truly zero-variance, 1 no-data |
| **low-info-fine** | **30** | spurious flag; recomputed raw r_pb ≥ 0 (abler models pass *more*) — negative `a` is a 2-skill collapse artifact | 30 have real variance |
| **judge-inverted** | **1** | judge scores it backwards (`tb_0532_c06`, r_pb=−0.57) | real variance |
| mis-keyed | 0 | rewards wrong behavior | — |

- **degenerate** examples: the 6 headline `a≈−4.09/b≈18.3` items (`tb_0112_c02`, `tb_0273_c03`, `tb_0274_c07`, `tb_0464_c01`, `tb_0641_c06`, `tb_0043_c01`) each pass **1/82** → the fit blew up on a near-empty cell; content is legitimate.
- **low-info-fine** examples: `tb_0617_c06`, `tb_0598_c05`, `tb_0185_c08`, `tb_0220_c05` — all recompute to strongly-positive r_pb. Legitimate hint criteria whose negative 2-skill loading is a **collapse artifact of the weak, correlated scaffolding axis**; the 3-skill run should re-estimate them cleanly.
- **judge-inverted**: `tb_0532_c06` only — implicit "must not falsely credit the student" guard; abler models proactively supply the fact (good tutoring) and the judge scores presence-of-fact as a fail. Excluded from fit; the corresponding judge-anchor reword is a separate Gate B *text* action (out of scope here).

**Net Task 1: 60 items masked from the fit, 0 dropped from the bank.**

---

## TASK 2 — Q-loading recovery for the 640 `missing_Q_row` criteria

### The rule (this is what the reviewer approves)

A transparent, ordered, keyword classifier over the **criterion text** (full text pulled from the bank, not the truncated preview). First match wins:

1. **trim** — restate/echo-the-problem noise (`^the (response|model) must (re)?state the given problem`, `restate/repeat/echo the problem/question`) **and** no scaffolding-behavior signal. → no gradable content; do not load.
2. **scaffolding (strong)** — genuine guiding/eliciting *behavior*: `socratic`, `guiding question`, `guide the student toward/through`, `without giving/revealing the answer`, `next step`, `elicit`, `check for understanding`, `follow-up/guiding/probing/clarifying question`, `encourage the student to try/think/solve`, `scaffold`, `define … jargon` / `first defining … term`, `one step at a time`, `guiding language`.
3. **presentation** — formatting/structure **and** affect/tone: `format`, `markdown`, `bold`, `bullet`, `latex`, `heading`, `section`, `concise`, `verbose`, `length`, `second person`, `tone`, `encourag`, `acknowledg`, `empath`, `warm`, `friendly`, `compliment`, `prais`, `support`, `reassur`, `feelings`, `overwhelm`, `frustrat`, `positive`, `readab`, `organized`, `conversational`, `invite further questions`, etc. (Per the pre-registered decision, **affect → presentation**.)
4. **scaffolding (weak)** — a bare mention of `hint`/`guide` with no presentation/affect signal. (Tagged `weak:` at medium confidence.)
5. **default** — no keyword → **presentation** at **low confidence** (affect/engagement is the dominant residual; flagged for reviewer spot-check).

**Ordering rationale:** presentation is checked **before** the weak "hint" mention on purpose — many Q-less items mention a hint but actually grade its *tone/format* (e.g. "use an encouraging tone to begin the hint", "organize hints into sections"). Those are presentation, not scaffolding. Only genuine guiding-*behavior* phrasing (tier 2) counts as scaffolding.

### Counts

| Assignment | n | variance-pass | variance-fail |
|---|---:|---:|---:|
| **presentation** | **634** | 607 | 27 |
| **scaffolding** | **5** | 5 | 0 |
| **trim** | **1** | — | — |

- **27 presentation recovers fail the variance check** — they are all-pass (or all-fail) across the 82 models (e.g. `tb_0486_c07` "use 2nd person language", `tb_0489_c07` "organized into clear sections", `tb_0491_c15` "positive and encouraging tone"). They can be loaded for coverage but will contribute **~zero information** to the fit; flagged `variance_ok=False` in the CSV.
- **108 low-confidence defaults** — no keyword matched; assigned presentation. Spot-checking confirms these are overwhelmingly genuine presentation (second-person POV, "invite further questions", "conversational atmosphere", "labeled sections", "suitable for high-school level"). A handful have mild scaffolding flavor (e.g. `tb_0088_c07` "follow-up to ensure the explanation helped") that a reviewer may reclassify.

### Representative samples

**Presentation (high-confidence):**
- `tb_0002_c09` — "must acknowledge the student's feelings of being overwhelmed with confusion." (affect)
- `tb_0005_c07` — "have an encouraging, positive, and empathetic tone throughout." (tone)
- `tb_0002_c01` — "begin with a compliment/acknowledgement to the student for their follow-up." (affect)

**Scaffolding (all 5):**
- `tb_0521_c03` — "hint should not use technical jargon without first defining it." (jargon-gating)
- `tb_0564_c05` — "hint must be written as though from a teacher to a student … guiding language." (guiding)
- `tb_0598_c09` — "organize its reasoning into … guiding questions." (eliciting)
- `tb_0006_c09` — "conclude by checking for understanding or offering further help." (check-for-understanding)
- `tb_0564_c14` — "hint should talk about this exact problem …" (weak hint mention; medium confidence)

**Trim (the 1):**
- `tb_0036_c07` — "The response must state the given problem Solve the differential equation …" (echoes the prompt; no gradable behavior).

---

## TASK 3 — Fit-eligibility delta (sanity check)

Fit-eligible on an axis = loads that axis **and** has non-degenerate variance (`0 < k < 82`) in the response matrix. Correctness = content ∪ diagnosis.

| Axis | Current fit-eligible | Excluded (reverse artifacts) | Recovered (variance-passing) | **Projected** | Δ |
|---|---:|---:|---:|---:|---:|
| **correctness** | 3,035 | −51 | +0 | **2,984** | **−51** |
| **scaffolding** | 738 | −30 | +5 | **713** | **−25** |
| **presentation** | 0 | 0 | +607 | **607** | **+607** |

*(Exclusion removals exceed 60 in aggregate because a reverse item can load more than one axis; the union of the 60 excluded items removes 51 correctness-loaded and 30 scaffolding-loaded rows.)*

### Honest read of the delta
- **Presentation is the win.** Recovery converts a previously-empty pre-registered axis into **607 fit-usable items** — enough to actually estimate the optional 3rd axis for the first time.
- **Scaffolding is *not* rescued by recovery.** The Q-less set contains only **5** genuine scaffolding criteria; everything else affect/tone-flavored is presentation by design. Because 30 of the excluded reverse artifacts were scaffolding-loaded, **net scaffolding fit-eligibility falls ~25** (738 → 713). This is the important, non-obvious finding: *do not expect the missing_Q recovery to fix the weak axis.* The scaffolding rescue Gate B pointed to comes from **re-estimating the 30 "low-info-fine" items in the 3-skill fit** (where their negative 2-skill loading is a collapse artifact), not from loading no-skill criteria.
- **Correctness dips slightly** (−51) purely from masking degenerate/sparse artifacts; this *improves* fit hygiene without meaningfully shrinking a large axis.
- **Quality caveat on presentation:** of the 607, some fraction are near-ceiling style items (27 already fail variance outright and were excluded from the +607; more sit just above the variance floor). The +607 should be read as "coverage now exists," not "607 highly-informative items."

---

## Deliverables
- **This memo:** `plans+prds/Gate B - Q-Loading Recovery + Fit-Exclusion Plan (for review).md`
- **Plan (700 rows = 60 exclude + 640 recover/trim):** `staging/revision_evidence/qloading_recovery_plan.csv` / `.jsonl`
  - Columns: `criterion_id, scenario_id, set (exclude_from_fit | recover_qloading | trim), proposed_axis, current_skills, variance_ok, rule_matched, evidence, confidence`
- **Helper (read-only):** `staging/revision_evidence/_gateb_recovery.py`

*All analysis read-only over the 82-model pilot + bank. This task wrote nothing to the bank or fitter inputs and committed nothing. The one modified bank line noted in the banner is a pre-existing Gate B text edit, not a product of this task.*
