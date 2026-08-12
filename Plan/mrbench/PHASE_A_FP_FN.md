# MRBench Phase A — Per-Class FP/FN Decision-Rate Reliability

**Status: complete (offline re-analysis, FREE).** No judge/API/network/GPU calls —
this re-analyses the already-cached temp-0 judgments from the completed Phase-A run
(Claude Haiku 4.5, 12,712 judge calls; 12,424 pair to gold = 1,553 pairs/dimension × 8).

- Code: `diagnostics/mrbench/reliability.py` (metrics + bootstraps),
  `diagnostics/mrbench/phase_a_fp_fn.py` (CLI).
- Output: `diagnostics/mrbench/runs/phase_a_haiku_4_5_2026-08-09/fp_fn_metrics.json`
  (written *next to*, never over, the existing `metrics.json`; `runs/` is gitignored).
- Reproduce (no API calls):

```bash
uv run python -m diagnostics.mrbench.phase_a_fp_fn --bootstrap 2000 --seed 42
```

## What this lens adds vs AC (Pearson)

The Phase-A validation ([PHASE_A_VALIDATION.md](PHASE_A_VALIDATION.md)) judges reliability
with **Pearson AC** — a single per-dimension correlation between the judge's ordinal score
(1–3) and the human gold score (1–3). AC answers *"do the judge and humans move together?"*
It does **not** tell you *which* mistakes the judge makes, or in which direction.

The FP/FN decision-rate lens reframes each dimension as a set of **classification decisions**
and asks, per label:

- **False-positive rate (fall-out)** — when the true label is *not* `c`, how often does the
  judge *assert* `c`? (over-calling)
- **False-negative rate (miss rate)** — when the true label *is* `c`, how often does the
  judge *miss* it? (`= 1 − recall`; under-calling)

This is exactly the framing a downstream acceptance gate needs (e.g. "how often does the
judge wrongly flag a good tutor turn as bad?"), which a correlation cannot express. It is a
**complement** to AC/DAMR, not a replacement: primary tutor-quality stays DAMR, primary
judge-reliability stays Pearson AC, and per-dimension macro-F1 stays the secondary
BEA-comparable number.

## Definitions (per-class one-vs-rest)

For each dimension `d` and each ordinal class `c ∈ {1, 2, 3}` (the Figure-6 rubric levels;
we always use the dimension's *label*, never a hard-coded score), treat `label == c` as the
positive:

| Cell | Condition |
|---|---|
| TP | judge == c AND gold == c |
| FP | judge == c AND gold != c |
| FN | judge != c AND gold == c |
| TN | judge != c AND gold != c |

- **FP rate** = FP / (# gold != c) = FP / (FP + TN)
- **FN rate** = FN / (# gold == c) = FN / (TP + FN) = **1 − recall**
- precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = harmonic mean
- **per-dimension macro-F1** = mean of the three per-class F1s

Judge↔gold pairing is **identical to `metrics.py`**: a judged cell pairs to gold via
`(conversation_id, tutor, dimension)`, and only successfully parsed records with a matching
gold score contribute. That identity is what makes the per-dimension macro-F1 a direct
cross-check against `metrics.compute_macro_f1` — and it matches (below).

## Confidence intervals — why cluster on `conversation_id`

Two 95% percentile bootstraps (2000 resamples, seed 42) are reported for every rate and F1:

- **case-level** — resample individual judged cells with replacement (reuses
  `metrics.bootstrap_ci_rate` for the two rates). This assumes cells are independent.
- **conversation-id-clustered** — resample whole `conversation_id`s with replacement, then
  recompute. Cells from the same dialogue (the same conversation scored across 9 tutors and
  8 dimensions) are **correlated**, so treating them as independent understates the variance.
  The clustered bootstrap is the honest interval; the case-level one is shown alongside only
  to demonstrate the gap.

**Clustering does widen the intervals** (mean CI width over classes, clustered ÷ case-level):

| Dimension | FP width ratio | FN width ratio |
|---|---|---|
| Mistake_Identification | 1.12 | 1.14 |
| Mistake_Location | 1.25 | 1.21 |
| Revealing_of_the_Answer | 1.05 | 1.13 |
| Providing_Guidance | 1.17 | 1.08 |
| Actionability | 1.06 | 1.05 |
| Coherence | 1.21 | 1.21 |
| Tutor_Tone | 1.20 | 1.23 |
| Humanlikeness | 1.04 | 1.02 |

Ratios are consistently > 1 (up to ~1.25), confirming clustering matters: a naive case-level
CI is ~5–25% too narrow. Report the clustered interval.

## Results — per dimension / per class (clustered 95% CIs)

Rates are fractions in [0, 1]. `supp` = gold support (# cells with gold == c). `*` marks the
DAMR-desired class. CIs are the **conversation-clustered** bootstrap. Full case-level CIs and
raw confusion counts are in `fp_fn_metrics.json`.

### Mistake_Identification — macro-F1 0.467 (desired: Yes)

| Class | supp | FP rate [95% CI] | FN rate [95% CI] | F1 [95% CI] |
|---|---|---|---|---|
| *1 Yes | 1238 | 0.121 [0.088, 0.160] | 0.397 [0.360, 0.433] | 0.739 [0.710, 0.767] |
| 2 To some extent | 91 | 0.189 [0.167, 0.213] | 0.758 [0.667, 0.851] | 0.113 [0.068, 0.155] |
| 3 No | 224 | 0.210 [0.182, 0.238] | 0.152 [0.102, 0.209] | 0.548 [0.502, 0.594] |

### Mistake_Location — macro-F1 0.443 (desired: Yes)

| Class | supp | FP rate [95% CI] | FN rate [95% CI] | F1 [95% CI] |
|---|---|---|---|---|
| *1 Yes | 999 | 0.173 [0.137, 0.211] | 0.485 [0.442, 0.526] | 0.639 [0.602, 0.675] |
| 2 To some extent | 119 | 0.060 [0.047, 0.073] | 0.882 [0.810, 0.950] | 0.128 [0.053, 0.201] |
| 3 No | 435 | 0.432 [0.390, 0.472] | 0.172 [0.132, 0.213] | 0.563 [0.525, 0.600] |

### Revealing_of_the_Answer — macro-F1 0.604 (desired: No)

| Class | supp | FP rate [95% CI] | FN rate [95% CI] | F1 [95% CI] |
|---|---|---|---|---|
| 1 Yes (answer correct) | 218 | 0.083 [0.067, 0.101] | 0.142 [0.088, 0.202] | 0.725 [0.667, 0.773] |
| 2 Yes (answer incorrect) | 25 | 0.018 [0.012, 0.025] | 0.840 [0.656, 0.962] | 0.140 [0.035, 0.269] **sparse** |
| *3 No | 1310 | 0.099 [0.062, 0.136] | 0.085 [0.067, 0.103] | 0.947 [0.936, 0.956] |

### Providing_Guidance — macro-F1 0.539 (desired: Yes)

| Class | supp | FP rate [95% CI] | FN rate [95% CI] | F1 [95% CI] |
|---|---|---|---|---|
| *1 Yes | 903 | 0.374 [0.327, 0.420] | 0.258 [0.224, 0.294] | 0.738 [0.706, 0.767] |
| 2 To some extent | 366 | 0.235 [0.207, 0.265] | 0.617 [0.562, 0.672] | 0.357 [0.307, 0.406] |
| 3 No | 284 | 0.070 [0.057, 0.084] | 0.535 [0.474, 0.591] | 0.523 [0.474, 0.573] |

### Actionability — macro-F1 0.526 (desired: Yes)

| Class | supp | FP rate [95% CI] | FN rate [95% CI] | F1 [95% CI] |
|---|---|---|---|---|
| *1 Yes | 861 | 0.377 [0.338, 0.418] | 0.276 [0.244, 0.310] | 0.714 [0.686, 0.742] |
| 2 To some extent | 189 | 0.192 [0.170, 0.215] | 0.561 [0.489, 0.635] | 0.311 [0.257, 0.363] |
| 3 No | 503 | 0.090 [0.073, 0.108] | 0.545 [0.499, 0.588] | 0.554 [0.515, 0.593] |

### Coherence — macro-F1 0.356 (desired: Yes) — weakest dimension

| Class | supp | FP rate [95% CI] | FN rate [95% CI] | F1 [95% CI] |
|---|---|---|---|---|
| *1 Yes | 1253 | 0.350 [0.293, 0.406] | 0.357 [0.314, 0.402] | 0.745 [0.710, 0.776] |
| 2 To some extent | 139 | 0.012 [0.006, 0.018] | 0.986 [0.961, 1.000] | 0.025 [0.000, 0.067] |
| 3 No | 161 | 0.364 [0.325, 0.407] | 0.273 [0.203, 0.348] | 0.298 [0.261, 0.338] |

### Tutor_Tone — macro-F1 0.448 (desired: Encouraging)

| Class | supp | FP rate [95% CI] | FN rate [95% CI] | F1 [95% CI] |
|---|---|---|---|---|
| *1 Encouraging | 496 | 0.478 [0.443, 0.514] | 0.056 [0.033, 0.082] | 0.637 [0.602, 0.671] |
| 2 Neutral | 1056 | 0.054 [0.031, 0.080] | 0.508 [0.473, 0.543] | 0.649 [0.617, 0.680] |
| 3 Offensive | 1 | 0.021 [0.014, 0.027] | 0.000 [0.000, 0.000] | 0.059 [0.000, 0.190] **sparse** |

### Humanlikeness — macro-F1 0.484 (desired: Yes) — excluded from the AC gate

| Class | supp | FP rate [95% CI] | FN rate [95% CI] | F1 [95% CI] |
|---|---|---|---|---|
| *1 Yes | 1370 | 0.530 [0.451, 0.604] | 0.163 [0.141, 0.186] | 0.878 [0.861, 0.892] |
| 2 To some extent | 95 | 0.126 [0.106, 0.145] | 0.832 [0.756, 0.901] | 0.109 [0.064, 0.156] |
| 3 No | 88 | 0.044 [0.034, 0.055] | 0.477 [0.369, 0.580] | 0.465 [0.374, 0.547] |

**Sparse classes** (support < 30) — Revealing class 2 (n=25) and Tutor_Tone class 3
"Offensive" (n=1) — have very wide/degenerate CIs and should not be read as reliable. The
`fp_fn_metrics.json` flags these with `"sparse": true`.

## Macro-F1 cross-check ✅

Recomputed per-dimension macro-F1 mean = **0.4834** vs the recorded **0.48** in
[PHASE_A_VALIDATION.md](PHASE_A_VALIDATION.md) → **MATCH** (|Δ| = 0.0034; the 0.48 is a
2-dp rounding of 0.4834). Every per-dimension value reproduces `metrics.json` exactly
(0.467 / 0.443 / 0.604 / 0.539 / 0.526 / 0.356 / 0.448 / 0.484). The `FN rate = 1 − recall`
identity holds for every cell. This confirms the gold/judge alignment is identical to the
validated pipeline — no re-judging, no alignment drift.

## Reading the numbers

- **The middle class "To some extent" is the judge's blind spot everywhere.** Its FN rate is
  0.62–0.99 across dimensions (it almost never *emits* the middle label when gold says so),
  with F1 0.03–0.36. This is the ordinal-classifier failure mode AC/DAMR hide: the judge
  collapses the 3-way scale toward the extremes.
- **Tone / Humanlikeness / Coherence over-call the desired class.** The desired class carries
  a high FP rate (Tone Encouraging 0.48, Humanlikeness Yes 0.53, Coherence Yes 0.35): the
  judge is generous, asserting the "good" label on turns humans did not.
- **Revealing_of_the_Answer is the strongest** — desired class "No" has FP 0.099 / FN 0.085 /
  F1 0.95, consistent with its high AC (0.72).
- **Coherence is weakest** (macro-F1 0.36, both extreme classes with ~0.35 FP), matching its
  failed AC gate — treat as low-confidence.

## Acceptance thresholds — TBD, to be recalibrated with Sameer / Lawrence

The teammate's **binary** judge gates (FP ≤ 0.05, macro-F1 ≥ 0.90, flip ≤ 0.10,
retest ≥ 0.90) were written for a **2-class** decision. They should be recalibrated before
being applied to this **3-class ordinal** task:

- **FP ≤ 0.05 is almost certainly too strict here.** In one-vs-rest over 3 classes, chance-level
  FP for a class is far above 0.05, and even the *strong* Revealing "No" class sits at ~0.10.
  A flat 0.05 would fail nearly every (dimension, class) cell for reasons that are structural,
  not quality-driven. It is only ~met by naturally rare classes (e.g. Tone class 2/3), where a
  low FP just reflects the judge rarely emitting that label at all.
- **macro-F1 ≥ 0.90 is unreachable for a 3-class ordinal with a hard-to-annotate middle band.**
  The validated judge's best dimension is 0.60; the mean is 0.48. Human–human agreement on
  these ordinal labels is itself modest.

**Concrete recommendation to discuss:**

1. **Judge the *decision that matters*, not all three classes.** Gate on the **desired-vs-rest
   binary** per dimension (desired label as positive) — that is the decision the downstream
   pipeline actually consumes, and it side-steps the middle-class sparsity.
2. **Set direction-specific, per-dimension targets from these baselines**, e.g. on the desired
   class: FP rate ≤ ~0.15 and FN rate ≤ ~0.30 as a *starting* band, tightened per dimension
   (Revealing can hold ~0.10; Tone/Humanlikeness/Coherence need looser FP or explicit caveats).
3. **Prefer clustered-CI lower/upper bounds, not point estimates, in the gate**, so within-
   dialogue correlation is accounted for.
4. **Exclude sparse classes** (support < 30) from any pass/fail and report them descriptively.
5. **Keep macro-F1 as a monitored secondary**, not a 0.90 gate; if a single-number gate is
   wanted, base it on the desired-vs-rest binary F1 with a realistic bar (~0.60–0.70), agreed
   with Sameer/Lawrence.

*The exact numeric thresholds are deliberately left open here — they are a policy decision for
the team, and these baselines are the input to that conversation, not a substitute for it.*

## Proposed gates for P2/P3 — pending Sameer / Lawrence sign-off

Decisions taken so far: **adopt the teammates' *benign, meaning-preserving* perturbations**
(whitespace / politeness-prefix / rubric-synonyms) rather than typo injection — typos conflate
"judge robustness" with "garbled-input handling" and measure the wrong thing.

### Absolute FP/FN bands (desired-vs-rest, per dimension) — starting proposal

Gate on the desired-vs-rest binary decision (positive = the DAMR-desired label), point estimate
with the clustered CI reported, and **exclude classes with support < 30**. Three tiers instead
of one global bound, because the "soft" dimensions are structurally leniency-prone:

| Band | Rule | Where each dimension lands (from the results above) |
|---|---|---|
| **Reliable** | desired FP ≤ 0.15 **and** FN ≤ 0.30 | Mistake_Identification (FN 0.40 just over), Revealing (0.10 / 0.09 ✓) |
| **Usable w/ caveat** | desired FP ≤ 0.40 **or** FN ≤ 0.55 | Mistake_Location, Providing_Guidance, Actionability, Coherence |
| **Report-only (too lenient to rank on)** | desired FP > 0.40 | Tutor_Tone (0.48), Humanlikeness (0.53) |

### Perturbation-robustness gates (P2) — about *change*, not absolutes

Measured as canonical vs each benign variant:

- **Desired-decision flip rate** (on the binarized decision, so it is binary-comparable):
  worst variant **≤ 0.10** (keep the teammates' bar here — it is a binary rate).
- **Raw 3-class label-flip rate**: reported but only loosely bounded at **≤ 0.15** and treated
  as diagnostic — a 3-class judge has more ways to move than a binary one.
- **|ΔAC| per dimension ≤ 0.05**, and — the gate that matters most — **the 6/7 AC-gate verdict
  must not change under any variant** (if a meaning-preserving reword flips a dimension's
  pass/fail, the validation is fragile).
- **|ΔDAMR| per dimension ≤ 0.05.**
- *Overall "robust" = every variant preserves the AC-gate verdict AND worst desired-decision
  flip ≤ 0.10 AND max |ΔAC| ≤ 0.05.*

### P2 — how to run (code ready, pending sign-off)

The code is built and offline-validated; only the paid re-judge remains. New files (none of
the existing Phase-A modules were touched): `diagnostics/mrbench/perturbations.py` (the benign
`whitespace` / `politeness` / `rubric_synonyms` variants, each a pure `messages -> messages`
transform of the Figure-6 prompt, with a meaning-preservation assert), `robustness.py` (the
metrics/gates above), and `phase_a_perturb.py` (the CLI). Each variant re-judges a **stratified
subsample** (default ~1.2k cells, `dimension × tutor × source`, fixed seed) and writes to its
own cache (`runs/<run>/perturb/<variant>.jsonl`), so it never collides with the canonical
Phase-A cache.

```bash
# (a) DRY-RUN + cost, zero network — prints subsample size, variant list, judge-call
#     count and the Haiku-rate estimate (default ~1,192 cells × 3 variants = 3,576 calls,
#     ≈ $2.9 at $1/$5 per MTok). Safe to run any time.
uv run python -m diagnostics.mrbench.phase_a_perturb

# (b) LIVE re-judge — GATED. Requires --live AND a TrueFoundry key (tfy_va_… in
#     TFY_API_KEY/OPENAI_API_KEY) AND the Haiku 4.5 id in MRBENCH_JUDGE_MODEL, and
#     should only run once teammates approve the cost above. Writes per-variant caches,
#     then prints the robustness gates and writes perturb_metrics.json.
uv run python -m diagnostics.mrbench.phase_a_perturb --live

# (c) Re-report from existing per-variant caches (offline, no spend):
uv run python -m diagnostics.mrbench.phase_a_perturb --metrics
```

### Test-retest gate (P3, optional)

Identical input at temperature 0 → expect ~0 drift; gate **desired-decision flip ≤ 0.05**
(tighter than perturbation). If near zero, close the item without further replicates.

*These are a starting proposal to ratify with Sameer / Lawrence, not final policy. Two points
to settle with them: (a) point-estimate vs conservative CI-bound gating, and (b) explicit
agreement that desired-vs-rest is the decision we gate on, given the middle-class blind spot.*

## Tests deliberately not added

Per `CLAUDE.md` (ask before adding tests), no test file was added. The math was verified with
an inline scratch check (confusion counts, FP/FN rates, `FN = 1 − recall`, NaN handling on
empty positives/negatives, and agreement with `metrics.f1_for_class`), and the macro-F1
cross-check against the recorded run serves as an end-to-end correctness check. **Warranted
but not added:** a small `tests/` unit for `reliability.py` covering (a) a hand-computed
confusion → FP/FN/F1, (b) `FN_rate == 1 − recall`, (c) macro-F1 equals `metrics.macro_f1` on a
fabricated record set, and (d) the clustered bootstrap collapsing to the case-level bootstrap
when every conversation has exactly one cell. Happy to add it on request.
