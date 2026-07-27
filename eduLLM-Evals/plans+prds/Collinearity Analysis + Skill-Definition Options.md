# Collinearity Analysis + Skill-Definition Options (content ↔ diagnosis)

**Status:** decision memo / analysis. **This memo changes nothing.** It does not touch
`data/rubrics_qmatrix_final.jsonl`, the `q_mapping`s, or the skill definitions in
`Skill Definitions v2 (Q-matrix + Judge).md`. It exists to inform whether to collapse or
redefine the `content` / `diagnosis` skills, and to make the cost of each path explicit.

**TL;DR recommendation:** **(A) Keep the 3 skills as labeled and decide from full-data
evidence** — the M2PL latent correlation + EFA + a 3-dim-vs-collapsed-2-dim AIC/BIC
comparison. The 82% structural overlap is largely a *real* property of "correct the
student's specific error" criteria (which genuinely require both skills), not a labeling
artifact, and structural overlap ≠ empirical collinearity. The decision is fully
reversible after grading (Q-matrix regen is cheap; tutor responses and judge verdicts are
reused), so there is no reason to prematurely destroy the `diagnosis` label.

---

## 1. Structural analysis (exact, from `data/rubrics_qmatrix_final.jsonl`, 6462 criteria)

Marginals and combos (verified against the earlier full count; smallest cell
diagnosis+scaffolding = 30 re-confirmed by grep):

| skill | items (marginal) |
|---|---:|
| content | 4524 |
| diagnosis | 2372 |
| scaffolding | 1136 |

| combo (c,d,s) | count |
|---|---:|
| content-only (1,0,0) | 2096 |
| content+diagnosis (1,1,0) | 1724 |
| none (0,0,0) | 1107 |
| content+scaffolding (1,0,1) | 485 |
| scaffolding-only (0,0,1) | 402 |
| diagnosis-only (0,1,0) | 399 |
| content+diagnosis+scaffolding (1,1,1) | 219 |
| diagnosis+scaffolding (0,1,1) | 30 |

Co-occurrence: content∩diagnosis **1943**, content∩scaffolding 704, diagnosis∩scaffolding 249.

### Conditional probabilities — the (a)symmetry that matters

- **P(content | diagnosis) = 1943 / 2372 ≈ 81.9%.** Four out of five diagnosis items also
  load content.
- **P(diagnosis | content) = 1943 / 4524 ≈ 42.9%.** Content is the "base" skill; fewer than
  half of content items also load diagnosis.

This is the signature of **near-nesting, not symmetric entanglement**: `diagnosis` behaves
close to a *sub-region* of `content`. The instrument almost never asks the model to diagnose
without also being right about the domain. That is exactly what worsens identifiability — a
dimension that rarely varies independently of another is hard to separate at finite N.

### Is there a real "diagnosis-not-content" construct? — the 429 and the 399

**Diagnosis-without-content = 399 (diagnosis-only) + 30 (diagnosis+scaffolding) = 429 items
(18.1% of all diagnosis items).** A sample read of these criteria/`q_rationale` shows the
population is **not** homogeneous; it splits into two clearly distinguishable sub-constructs:

1. **Genuine error-*detection* (separable, content=0 is correct).** These require reading and
   flagging a *specific* thing the student wrote/used, without the tutor having to supply or
   verify the correct domain answer:
   - `tb_0017_c03` "identify that the student incorrectly uses the term *pre-equivalence*"
   - `tb_0017_c10` "identify that the student uses *0.0017* for the concentration of H₂O₂"
   - `tb_0017_c13` "identify that the student states the units … as *m/s*"
   - `tb_0028_c03` "identify that the student incorrectly states the parent isotope as U-235"
   - `tb_0027_c01` "identify that the student is missing the background knowledge …"

   These are a **coherent, separable signal**: a "did the model *notice what the student
   actually did*" ability, distinct from "can the model solve it." This is real evidence that
   `diagnosis` is not merely a relabeling of `content`.

2. **Acknowledgement / validation of state (borderline; near the v2 all-zero line).** These
   turn on recognizing/affirming the student's *expressed* confusion or a correct piece of
   their reasoning:
   - `tb_0001_c10` "acknowledge the student's confusion regarding the re-calculation of HA"
   - `tb_0005_c01` "acknowledge the student's expressed confusion regarding the formula …"
   - `tb_0010_c04` "explicitly validate that the student correctly understands the division
     formula"
   - `tb_0020_c12`, `tb_0024_c05` "acknowledge the student's confusion before explaining"

   Under the strict v2 rule ("merely acknowledging a feeling is **not** diagnosis → all-zero")
   several of these are arguably mislabeled and closer to all-zero. They are a *weaker,
   noisier* diagnosis signal than group 1 and inflate the diagnosis-only cell.

**Characterization:** the `diagnosis` dimension is **not independent** of `content` in the
instrument (82% of it co-fires with content, and it behaves like a near-subset), **but it is
not empty of separable content either** — the group-1 error-detection items are a real,
distinct construct. The open question the data must answer is whether that separable signal
is *strong and prevalent enough* to identify a third latent dimension, or whether it is too
thin (≈399 pure items, minus the borderline acknowledgement subset) to hold up as its own axis.

---

## 2. Causal attribution to the v2 broad "address-the-error" rule

The v2.1 rule (`Skill Definitions v2`, §`diagnosis`, "Broad address-the-error rule (adopted)")
says: *any* phrasing asking the tutor to correct/address **the student's** error loads
`diagnosis`, even without naming the specific misconception. The question: **how much of the
82% overlap is manufactured by this rule** vs. reflecting genuine specific-misconception
identification?

**Heuristic estimate (directional — free-text keyword pass, formalized in
`scripts/analyze_collinearity.py`, block 2):** I read a stratified sample of the 1724
content+diagnosis items' `q_rationale` diagnosis-justification blocks. The dominant pattern is
**genuine specific-misconception**, not broad-rule boilerplate. The justifications repeatedly
cite the student's concrete work:

- `tb_0002_c08` "state that the order proposed by the student as HCl>HF>HI is incorrect"
- `tb_0003_c03` "identify that the student's solution [du/dx = dy/dx] is incorrect"
- `tb_0067_c02` "the student's answer of 800 likely came from using absolute values …"
- `tb_0068_c01` "correct the misconception that U = ½kx² applies here …"
- `tb_0069_c03` "the student is missing key background on Mendel's Law of Segregation …
  (student wrote 'two gametes … (RW and WW)')"

Across the sample, the overwhelming majority named a specific student step/value/proposal/gap.
Only a minority were *framing-only* ("explain that in addition to electronegativity …",
`tb_0002_c04`) or acknowledgement-flavored, and even those usually referenced the specific
misconception.

**Directional bottom line (heuristic, must be confirmed by running the script):**
- Estimated **specific-misconception**: **~70–85%** of the 1724 content+diagnosis items.
- Estimated **broad-rule + acknowledgement (the part that would dissolve if `diagnosis` were
  narrowed to "explicit specific misconception only")**: **~15–30%** (roughly **~250–500
  items**).

So narrowing the diagnosis definition (Option C) would remove diagnosis from only a *minority*
of the currently-overlapping items. **Most of the 82% overlap is not a rule artifact — it is
intrinsic**: a criterion that says "correct the student's specific error" genuinely requires
*both* reading the error (diagnosis) *and* knowing the right answer (content). You cannot
define that entanglement away without also mislabeling the criterion.

> ⚠️ **Caveat:** these percentages are a keyword heuristic over free text, not adjudicated
> labels. Run `analyze_collinearity.py` for the exact machine counts (it dumps a per-item
> label CSV so every classification is auditable), and spot-check a sample by hand before
> quoting any number as fact. Grep-based counting of the full bank is unreliable here (the
> tool's count mode capped inconsistently at 100 during verification), which is precisely why
> the exact counts are delegated to the script.

---

## 3. Preliminary empirical read (`scripts/analyze_collinearity.py`) — DIRECTIONAL ONLY

`staging/response_matrix.csv` is **82 model rows × 6462 criteria**, but only **~20–23 persons
are observed per item** (median 20; fill rate 24.9%), and 3802 items are all-fail / 15
all-pass (zero-variance) at this N. The graded fleet is a **biased small-model subsample**
(e.g. `Yi-6B`, `pythia-1.4b`, …). **Nothing in this block is powered.** It exists only to see
whether the empirical signal *points the same way* as the structure.

The script computes, over the ~20 graded models:

1. **Per-skill-group item pass rates** (content-loaded, diagnosis-loaded, content-only,
   diagnosis-only, etc.).
2. **Item-level pass-pattern correlation** between content-**primary** and diagnosis-**primary**
   items — mean pairwise Pearson/phi and an **approximate tetrachoric** (Digby cosine
   approximation with 0.5 continuity correction; no extra dependency), over pairwise-complete
   respondents with a `--min-overlap` floor — compared against the within-content and
   within-diagnosis baselines. (If cross-group ≈ within-group correlation, the two item sets
   are behaving like one dimension.)
3. **Per-model pass-rate correlation** between the content-loaded and diagnosis-loaded item
   sets (one point per model). **This is the most robust read at N≈20** (≈20 points, not
   thousands of thin item pairs) and is the number to look at first.

**Run it:**

```bash
python scripts/analyze_collinearity.py
```

Structural + heuristic only (no matrix needed), or tuning the item-pair overlap floor:

```bash
python scripts/analyze_collinearity.py --structural-only
python scripts/analyze_collinearity.py --min-overlap 10 --min-items 4
```

Outputs: `staging/collinearity_report.json`, `staging/collinearity_report.csv` (per-item
heuristic labels), plus a printed summary.

> ⚠️ **Interpretation guardrail:** a high per-model content↔diagnosis correlation here is
> *expected even if the skills are truly separable*, because (a) the sample is dominated by
> weak models that fail broadly, and (b) 82% of diagnosis items share their content loading.
> Do **not** read a high empirical correlation as "collapse the skills." The powered evidence
> is the full-data M2PL latent correlation (§4A), not this. Treat block 3 as a sanity check
> that the machinery runs and the sign is plausible.

---

## 4. Options

For all three options, the re-work economics are the same and modest, because the expensive
artifacts are already produced and **reused**:

- **Tutor responses:** reused (no re-generation).
- **Judge verdicts:** reused (grading is per-criterion pass/fail, independent of the skill
  labels; changing `q_mapping` does not change whether a response satisfied a criterion).
- **What actually re-runs:** (i) Q-matrix regeneration/verification — a **frontier-LLM pass**
  over the same criteria, cheap and fast; and (ii) a **calibration re-fit**
  (`calibrate_mirt.py`), seconds-to-minutes of CPU. No human re-grading, no GPU fleet re-run.

### Option A — Keep 3 skills; decide empirically from the full-data fit (RECOMMENDED)

Keep `content` / `diagnosis` / `scaffolding` and their v2 labels exactly as-is. Once the full
matrix is graded, let the calibration decide dimensionality with three convergent tests:

1. **M2PL latent correlation** `R` (`calibrate_mirt.py --estimate-latent-corr`): read the
   content↔diagnosis off-diagonal. Near ±1 ⇒ collapsing toward unidimensionality; comfortably
   < ~0.9 ⇒ separable.
2. **EFA / scree** (`--efa`): does a genuine second/third factor emerge, or does content+diagnosis
   load one factor?
3. **3-dim vs collapsed-2-dim AIC/BIC:** fit the confirmatory 3-dim model and a 2-dim model
   with content+diagnosis merged; compare information criteria on the same likelihood.

**Proposed (do NOT build here):** add a **`--collapse content,diagnosis`** mode to
`calibrate_mirt.py`. Mechanics: before `align_q_rows`, OR the two named columns into a single
merged skill column (a merged item loads the combined dim iff it loaded either), reducing
`SKILLS` to 2 dims for that run; everything downstream (`fit_m2pl_em`, AIC/BIC, manifest) is
already dimension-agnostic. Then the manifest's existing `comparison` block gives a direct
3-dim-vs-collapsed AIC/BIC contrast, and `--estimate-latent-corr` on the 3-dim run gives the
latent correlation. This is a small, additive flag — no schema change, no data mutation.

- **Identifiability implication:** keeps the (potentially weak) 3rd dimension in play; if it is
  under-identified at the real N, tests 1–3 will *show* it and you drop to 2 dims *with evidence*.
  Conservative v2 labeling already minimizes false-1 contamination.
- **Re-work cost:** **lowest / none up front.** No Q-matrix change. Only the calibration run
  (which happens anyway). Fully reversible.
- **Pro:** decide from evidence, not a prior; preserves the separable error-detection signal
  (§1 group 1) until proven redundant; nothing is irreversibly lost.
- **Con:** you carry a possibly-thin 3rd dimension through one calibration before pruning it.

### Option B — Collapse content+diagnosis into one dimension (2-skill model)

Merge into **correctness/diagnosis** + **scaffolding**. `q_mapping` regenerated to two columns.

- **Identifiability implication:** **best** — removes the hardest-to-separate pair outright, so
  the remaining 2-dim model is comfortably identified even at modest N. Latent correlations
  can't blow up between two axes that are, a priori, well separated (scaffolding κ was 0.59 vs
  diagnosis 0.365).
- **Re-work cost:** low (Q-matrix regen + re-fit; responses/verdicts reused), but **the label
  loss is not recoverable without another regen** — you'd have discarded the diagnosis
  distinction and would need to re-regenerate to get it back.
- **Pro:** simplest, most stable instrument; honest about the 82% overlap; no more content↔diagnosis
  identifiability fights.
- **Con:** **throws away the genuine error-detection construct** (§1 group 1 — the ~399 pure
  diagnosis items, esp. the "notice what the student actually did" items) even though it is
  separable. Loses the ability to report "diagnosis" as a distinct tutoring competency, which is
  a core selling point of the benchmark. Premature if A hasn't been run — you'd be collapsing on
  *structure* before seeing whether the *empirical* latent correlation actually demands it.

### Option C — Narrow the `diagnosis` definition (revert/limit the broad rule)

Restrict `diagnosis=1` to **explicit specific-misconception identification only**; drop the
broad "address-the-error" loading and the acknowledgement-flavored items. Keeps 3 dims.

- **Identifiability implication:** **improves** separability by removing the broad-rule
  cross-loadings that push content↔diagnosis together — but per §2 this only touches an
  estimated **~15–30%** of the overlap (~250–500 items), so it **shrinks but does not fix** the
  entanglement. The remaining specific-misconception items still, by construction, load both.
- **Re-work cost:** low (Q-matrix regen with a tightened prompt + re-fit; responses/verdicts
  reused). Note this **directly contradicts the v2.1 rationale**, which adopted the broad rule
  *deliberately* to avoid false-0s on genuine error-engagement items (the asymmetry argument in
  §"1-placement policy"). Reverting re-opens that false-0 risk.
- **Pro:** yields a "purer," higher-precision diagnosis dimension; keeps 3 skills; aligns
  diagnosis with the clean group-1 construct.
- **Con:** re-introduces the false-0 problem v2.1 was designed to prevent; a **definition change
  is more consequential and less reversible than a fit choice** (it changes what the benchmark
  *means* by diagnosis, and re-labels many items); and it only partially addresses the overlap,
  so you may still end up needing A's evidence anyway.

---

## 5. Recommendation

**Do (A).** Keep the labels; add the `--collapse content,diagnosis` flag to `calibrate_mirt.py`;
grade the full matrix; then decide dimensionality from the M2PL latent correlation + EFA +
3-dim-vs-2-dim AIC/BIC. Rationale:

1. **Structural overlap ≠ empirical collinearity.** 82% co-occurrence is a labeling fact; whether
   the *latent abilities* are collinear is an empirical question only the full-data fit answers.
2. **The overlap is mostly real, not a rule artifact** (§2): narrowing (C) removes only a minority
   and re-opens the false-0 risk v2.1 closed on purpose.
3. **There is a genuine separable diagnosis signal** (§1 group 1) worth keeping until proven
   redundant.
4. **The choice is cheap and reversible post-grading** — regen is a frontier-LLM pass, the re-fit
   is seconds, and responses + judge verdicts are reused. Collapsing (B) or redefining (C) now
   would discard information (labels / a construct) *before* the evidence that would justify it.

If, after the full fit, the content↔diagnosis latent correlation is ≳0.9 **and** the collapsed
2-dim model wins on AIC/BIC **and** EFA shows no separate factor, **then** collapse to Option B —
with evidence, and at the same low re-work cost.

---

## Appendix — deliverables

- **This memo:** `plans+prds/Collinearity Analysis + Skill-Definition Options.md`
- **Analysis script:** `scripts/analyze_collinearity.py` → run with `python scripts/analyze_collinearity.py`
  (outputs `staging/collinearity_report.{json,csv}` + printed summary).
- **Proposed (not built):** `calibrate_mirt.py --collapse content,diagnosis` (Option A test).
- **Powered decision inputs (run on the FULL graded matrix):**
  `python scripts/calibrate_mirt.py --estimate-latent-corr --efa`.

---

## 6. Empirical results (1/4 matrix, DIRECTIONAL)

> ⚠️ **All numbers below are from runs already executed on a partial (~1/4) matrix and are
> NON-IDENTIFIABLE.** Only **N≈28 persons** of the 82 models carry usable signal on the
> relevant items; a powered latent-structure decision needs **≥150** persons. Read this section
> as *directional convergence*, not as a decision. The decision is **deferred to the full
> graded matrix** (§7).

### 6.1 Structural (exact, re-confirmed)

- **P(content | diagnosis) = 81.9%.** Four of five diagnosis items also load content.
- **P(diagnosis | content) = 43.0%.** Content is the base skill; fewer than half of content
  items also load diagnosis.
- **Diagnosis-without-content = 429 items (18.1%** of all diagnosis items).

This is the near-nesting signature from §1: `diagnosis` behaves like a near-subset of `content`.

### 6.2 Causal attribution (heuristic, from `scripts/analyze_collinearity.py`)

Of the **1943** content+diagnosis items, the script's keyword classifier assigns:

| class | count | share |
|---|---:|---:|
| specific-misconception | 1807 | 93.0% |
| broad-rule | 1 | 0.05% |
| acknowledgement | 32 | 1.6% |
| unclassified | 103 | 5.3% |

The part that would **dissolve if `diagnosis` were narrowed** to "explicit specific
misconception only" (broad-rule + acknowledgement) is only **33 items (1.7%)**.

⇒ **The overlap is REAL, not a v2-broad-rule artifact.** Narrowing the diagnosis definition
(the old **Option C**) is **off the table** — it would touch <2% of the overlap while
re-opening the false-0 risk v2.1 deliberately closed.

> ⚠️ **Caveat:** these are heuristic keyword counts over free text, not adjudicated labels. The
> classifier over-assigns "specific-misconception" (it is the catch-all bucket); the auditable
> per-item CSV (`staging/collinearity_report.csv`) should be spot-checked before any count is
> quoted as fact. Directionally, though, the broad-rule share is unambiguously tiny.

### 6.3 Empirical directional read (1/4 matrix, N≈28)

- **Per-model pass-rate correlation** (content-only vs diagnosis-only item sets, one point per
  model): **r = 0.9665**.
- **Item-level cross-skill phi ≈ 0.173**, which is **≈ the within-skill phi** — i.e. content and
  diagnosis items behave about as correlated *across* skills as *within* a skill.
- **M2PL latent correlation matrix** `[content, diagnosis, scaffolding]`:

```
[[ 1.000,  0.945, -0.201],
 [ 0.945,  1.000, -0.171],
 [-0.201, -0.171,  1.000]]
```

- **content ↔ diagnosis latent r = 0.945** (very high, points to collapse).
- **scaffolding ↔ {content, diagnosis} ≈ −0.19** (near-zero / separate — scaffolding is clearly
  a distinct axis; the negative sign is not meaningful at this N, read it as ~0/separate).

> ⚠️ **POWER-ARTIFACT WARNING — do not interpret this yet.** The dimensionality comparison at
> N=28 reports **"multi beats uni: AIC=False, BIC=False"** (the 3-dim model has **4370
> parameters**). At N=28 that is a **power artifact of massive over-parameterization**, **NOT**
> evidence for unidimensionality. It says nothing about the true latent structure and must be
> ignored until the full-matrix fit.

### 6.4 Convergence

Structural near-nesting (§6.1), causal attribution (§6.2, overlap is intrinsic), per-model
r = 0.9665, item-level cross ≈ within phi, and latent **r = 0.945** all point the **same
direction: content and diagnosis likely collapse into one dimension.** **scaffolding is clearly
distinct** (r ≈ −0.19). This is consistent with a ~2-dim "correctness/diagnosis + pedagogy"
structure — but every one of these signals is under-powered.

---

## 7. Conclusion / re-run plan

**Conclusion (provisional, directional):** The content ↔ diagnosis latent **r = 0.945**
converges with every other signal → the likely outcome is to **collapse content + diagnosis
into a single dimension** (→ Option B), with **scaffolding retained as a separate axis**.
**Option C (narrow diagnosis) is off the table** (§6.2: only 1.7% of overlap is rule-driven).

**But the DECISION IS DEFERRED to the full graded matrix.** Nothing changes now:
- **No Q-matrix change**, no skill-definition change, no data mutation.
- **Fully reversible** — the current fit is non-identifiable (N≈28 ≪ 150 needed).
- **Does NOT gate the $50 judge run** — grading is per-criterion pass/fail and skill-label
  independent; responses and judge verdicts are reused regardless of the eventual collapse.

### Re-run plan (exact commands, on the FULL matrix)

```bash
python scripts/calibrate_mirt.py --estimate-latent-corr --efa --matrix <FULL_MATRIX.csv>
python scripts/calibrate_mirt.py --collapse content,diagnosis --estimate-latent-corr --matrix <FULL_MATRIX.csv>
```

### Collapse decision rule (apply after the full-matrix fit)

**Collapse content + diagnosis → 2 dims IFF all three hold:**
1. latent **r ≳ 0.9** (content ↔ diagnosis), **AND**
2. the **collapsed 2-dim model wins on AIC/BIC** vs the 3-dim model, **AND**
3. **EFA finds no separate content/diagnosis factor**.

**Otherwise keep 3 dims.** (Scaffolding stays a distinct dimension in both branches.)
