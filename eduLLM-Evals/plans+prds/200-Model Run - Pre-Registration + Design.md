# 200-Model Run — Pre-Registration + Design (Multidimensional IRT / M2PL Calibration)

**Status:** DRAFT for human review. This is a *pre-registration* — it fixes the analysis
and decision rules **before** any 200-run data is seen. It does **not** modify pipeline
code, does not run the sweep, does not touch the fitters (`calibrate_mirt.py`), the curated
rubric bank, or `response_matrix*.csv`. Nothing here is committed to the runtime yet.
**Date:** 2026-07-30
**Author:** (calibration/measurement design)
**Scope:** the *calibration & measurement* design for the upcoming 200-model sweep. Benchmark
*execution* is treated as already engineered (the `cpu_sweep_200` runner is wired and
described in §5 for grounding only).

**Provenance this memo builds on (all read-only under `eduLLM-Evals/`):**
- `plans+prds/Pilot Baseline Freeze (2 and 3 skill).md` — frozen pilot baseline (N=82).
- `plans+prds/Calibration Run 6 - Presentation Axis (complete supplement).md` — presentation earns a pre-registered optional 3rd axis.
- `plans+prds/CAT Evaluation - TutorBench 2-skill.md` / `... 3-skill (presentation).md` — CAT recovery.
- `plans+prds/Calibration - K-Fold Cross-Validation (2-skill).md` — person-level OOS honesty check.
- `data/TutorBench/rubrics_qmatrix_calibrated_2skill.jsonl` / `_3skill.jsonl` — runtime-consumable calibrated banks.
- Runner (parent repo, read-only): `AdaptiveTesting/Test/Inference/{run_benchmark,datasets_registry,engine}.py` + `cpu_sweep_200/`.

---

## 0. TL;DR — what is pre-registered

- **Primary instrument = 2-skill `[correctness, scaffolding]`.** Presentation is the
  pre-registered **OPTIONAL 3rd axis**, admitted/demoted by the **frozen accept/reject rule
  in §6.3**, evaluated once on the 200-run fit before any tuning.
- **The 200-run's headline job is to harden the *scaffolding* axis** (the weak, under-identified
  one: cross-fold `a_scaffolding` ≈0.48 at N=82) and to **confirm presentation loadings at N≥150**.
  Denser persons is the single biggest lever we have.
- **Anchoring:** carry over **every pilot model that also lives in `models_200.yaml`
  (~28 models, see §3.2)** as **common-person anchors** linking the 200-run scale back to the
  frozen pilot baseline. Target **≥25 usable anchors spanning the full ability range**;
  designate them *before* the run.
- **Cross-benchmark linking is via COMMON PERSONS** (all 200 models take all benchmarks),
  not common items — the benchmarks share no items. Latent metric fixed to `θ ~ N(0, I)` per
  benchmark; the confirmatory Q-matrix fixes MIRT rotation.
- **⚠ Biggest open decision (needs user):** the pilot's low-ability floor (GPT-2, Pythia,
  BLOOM(z), OPT, RedPajama, TinyLlama, phi-2) is **deliberately excluded from `models_200.yaml`**
  (vLLM-broken families). That removes ~40 of the 82 pilot models — including most of the
  low-ability anchors — and **restricts the low end of the ability range**. Because the CPU
  sweep uses the **HF backend** (not vLLM), those families *can* be reintroduced. **Decide
  before launch** (see §2.4 / §8).

---

## 1. Objective — what the 200-run buys over the 82-model pilot

The pilot (N=82) produced a *usable but honest-caveated* calibration: correctness is strong
and reproducible; **scaffolding is the weak axis** (cross-fold `a_scaffolding` ≈0.48 at k=5,
OOS CAT recovery ≈0.71); presentation earned an optional-axis verdict statistically but is
flagged "confirm at N≥150." Every one of those caveats is a **sample-size (persons)** limit,
not a structural one. The 200-run addresses them directly:

1. **Denser person matrix → identifiability.** M2PL discrimination is a *person-powered*
   estimate. The k-fold study already showed the diagnostic pattern: `a_scaffolding` cross-fold
   stability rose from **0.48 (k=5, ~65 train persons) → 0.73 (k=10, ~74 train persons)**.
   Going 82→200 real persons should push scaffolding into the "reproducible" regime and
   tighten every `b`/`a` estimate. This is the primary scientific payoff.
2. **Better scaffolding-axis spread.** The pilot's full-bank scaffolding abilities were
   *quantized toward the prior* (clustered at ≈−1.2 / 0 / +1.1) because few models expressed
   scaffolding variance. A 200-set chosen to **span scaffolding ability** (§2) supplies the
   between-model variance the second latent dimension needs to be identified.
3. **Presentation confirmation, not gating.** Run 6 admitted presentation on complete N=82
   data (3-dim beats both folds on AIC+BIC; cross-fold `a_presentation` 0.712 > scaffolding
   0.488). The 200-run is the pre-registered **confirmation** at N≥150 (§6.3), not a re-litigation.
4. **Cross-benchmark linking.** The pilot calibrated TutorBench in isolation. Running the same
   200 models across TutorBench + the MCQ/open suite lets us place benchmarks on a **comparable
   latent scale via the shared person population** (§4) — enabling statements like "a model at
   correctness θ=+1 on TutorBench sits at pedagogy-MCQ θ≈+X."
5. **A denser, less-circular OOS check.** Larger folds (bigger train N per fold) shrink the
   optimism gap and make the person-level k-fold honesty check (§6.4) more trustworthy.

**Non-goals (explicit):** the 200-run does **not** re-open the content+diagnosis→correctness
collapse (settled, reinforced by Run 5; content↔diagnosis ≈0.94), and does **not** rebuild the
runner. It re-estimates item params on more persons and confirms/locks the axis structure.

---

## 2. Model selection — how to choose the 200 to span ability

### 2.1 Guiding principle
IRT item parameters are identified by **between-person variance on the trait**. For the weak
axis, that means: **maximize spread on *scaffolding* ability**, and — critically —
**decorrelate scaffolding from correctness** so the two latent dimensions are separable. The
pilot's problem was not too few models overall; it was too few models that are *good tutors but
weak solvers* (or vice-versa) to break the correctness↔scaffolding relationship.

### 2.2 Concrete selection axes (pre-registered targets)
The existing `models_200.yaml` is a curated `[0.2B, 7.0B]` open-weights set (200 checkpoints,
truncated-normal size mix with both tails halved: 0.2–1B≈12, 6–7B≈13). Evaluated against the
calibration need, we pre-register these coverage targets:

| Axis | Target | Rationale (calibration) |
|---|---|---|
| **Correctness ability spread** | Full range floor→ceiling | Anchors the difficulty `b` scale; needs genuinely weak *and* strong solvers. |
| **Scaffolding ability spread** ⭐ | **Deliberately over-sample instruct/chat/DPO/RLHF checkpoints and their matched base models** | Instruct-tuning is the main source of scaffolding variance; base models supply the low-scaffolding end. This is the axis we are trying to identify. |
| **Instruct vs base pairing** | Include **matched base↔instruct pairs** wherever a family offers them | Within-family base/instruct pairs are the cleanest way to inject scaffolding variance *at fixed correctness*, which is exactly what decorrelates the two axes. `models_200.yaml` already has many (Llama-3.2-1B/-Instruct, Falcon3-1B-Base/-Instruct, Qwen2.5-1.5B/-Instruct, OLMo-2-1B/-Instruct/-DPO, Granite base/instruct, Bielik base/instruct, SmolLM2 base/instruct, EuroLLM base/instruct, salamandra base/instruct, etc.). **Keep and prioritize these pairs.** |
| **Size range** | Uniform-ish across 0.2–7B bands | Size correlates with correctness; even coverage prevents an ability gap that would leave `b` unanchored in a region. |
| **Family diversity** | ≥30 distinct families | Reduces family-specific DIF from dominating any latent dimension; guards against a single family's quirk being read as trait variance. |
| **Presentation spread** | Comes largely for free | Presentation pass-rates spread 0.012–0.699 across the pilot fleet; the size/instruct spread above supplies presentation variance too. |

### 2.3 Which pilot models carry over
**Only ~28 of the 82 pilot models are present in the current `models_200.yaml`.** The pilot
leaned heavily on families that `models_200.yaml` **explicitly excludes** as vLLM-broken
(pythia ×7, opt ×3, bloom/bloomz ×8, gpt2 ×4, TinyLlama, RedPajama ×2, phi-1_5/phi-2,
Mistral-7B ×3, Yi-6B ×2, deepseek-7B ×3, Qwen2/2.5-7B ×5, Falcon3-7B, h2o-danube3, Amber,
zephyr-7b, vicuna-7b, Qwen3-0.6B, SmolLM2-135M/360M, Qwen2.5-0.5B).

**Confirmed carryover (pilot ∩ `models_200.yaml`), ~28 models — the anchor pool of §3:**

```
HuggingFaceTB/SmolLM2-1.7B                 HuggingFaceTB/SmolLM2-1.7B-Instruct
Qwen/Qwen2-1.5B                            Qwen/Qwen2.5-1.5B
Qwen/Qwen2.5-1.5B-Instruct                 Qwen/Qwen3-1.7B
Qwen/Qwen3-4B                              allenai/OLMo-2-0425-1B
allenai/OLMo-2-1124-7B                     google/gemma-2-2b
ibm-granite/granite-3.0-2b-base            ibm-granite/granite-3.1-2b-base
ibm-granite/granite-3.1-2b-instruct        meta-llama/Llama-3.2-1B
meta-llama/Llama-3.2-1B-Instruct           meta-llama/Llama-3.2-3B
meta-llama/Llama-3.2-3B-Instruct           microsoft/Phi-3-mini-4k-instruct
microsoft/Phi-3.5-mini-instruct            stabilityai/stablelm-2-1_6b
stabilityai/stablelm-2-zephyr-1_6b         stabilityai/stablelm-3b-4e1t
stabilityai/stablelm-zephyr-3b             tiiuae/Falcon3-1B-Base
tiiuae/Falcon3-3B-Base                     openbmb/MiniCPM-2B-sft-bf16
nvidia/Nemotron-Mini-4B-Instruct           h2oai/h2o-danube2-1.8b-base
```

(Exact overlap must be re-confirmed programmatically at launch by intersecting the pilot
response-matrix row keys — see `reports/cat_eval_3skill/cat_per_model.csv` — with
`models_200.yaml` ids; the list above is the analyst's manual intersection and should be
treated as *the proposed* carryover, subject to a scripted check.)

### 2.4 Consequence — low-ability floor & the reintroduction decision (OPEN)
The excluded families were the pilot's **low-ability floor** (GPT-2, Pythia, BLOOM, OPT,
RedPajama all sit near the bottom of the correctness scatter). Dropping them **restricts the
ability range at the low end**, which (a) weakens anchoring in that region and (b) can inflate
apparent discrimination for easy items. Because the CPU sweep uses `--backend hf`
(`run_cpu_worker.sh` sets `FORCE_CPU=1`), the "vLLM-broken" exclusion **does not bind** here.

**Pre-registered recommendation:** add back **~10–15 low-ability anchors** (a subset of the
pilot's GPT-2 / Pythia / BLOOM / OPT models) to a `models_200.yaml`-derived launch list, *for
the CPU-HF sweep only*, to preserve the low end of the range and maximize common-person
anchor overlap. **This changes the final model list and needs user sign-off (§8).**

---

## 3. Common-person anchoring — linking the 200-run back to the frozen pilot

### 3.1 Design
The 200-run and the pilot share **no items** on the calibration side (same TutorBench rubric
bank), but they *can* share **persons**. Any model run in **both** the pilot and the 200-run is
a **common-person anchor**: its responses appear in both matrices, so its ability estimate must
agree (up to the linking transform) across the two calibrations. Common persons let us:
- put the 200-run latent scale into the **same metric** as the frozen pilot baseline
  (`rubrics_qmatrix_calibrated_2skill/3skill.jsonl`), and
- **detect drift** — if the item bank shifted, anchors' abilities won't line up.

### 3.2 How many anchors, and which
- **Proposed anchor set = the full pilot∩200 carryover (~28 models, §2.3)**, augmented to
  **≥25 *usable* anchors** if any carryover models fail to load or hit the probe-skip.
- **Spread requirement (pre-registered):** anchors must span the ability range, **not** cluster.
  Require **≥2 anchors in each correctness-ability quintile** of the pilot distribution, and at
  least a handful that differ on scaffolding (instruct vs base). The carryover set already
  includes both ends (SmolLM2-135M-class low ⟷ Qwen3-4B / OLMo-2-7B high) and base↔instruct
  pairs (Llama-3.2-1B/-Instruct, Granite base/instruct, StableLM base/zephyr), which is what we
  want. **If §2.4's low-ability reintroduction is approved, the anchor count rises to ~38–43**,
  materially strengthening the link.
- **Why ~25–30 and not 5–10:** linking error shrinks with anchor count and with anchor spread.
  For a 2–3-dim latent link, a **rule-of-thumb ≥20 well-spread common persons** gives a stable
  transform; ~28 (or ~40 with reintroduction) is comfortably above that and cheap here (the
  models are re-run anyway).

### 3.3 Linking procedure (pre-registered)
1. **Independent-then-align.** Fit the 200-run M2PL freshly (do **not** fix item params to the
   pilot). Then estimate the linking transform (mean/variance, per latent dimension) that best
   maps the **anchors'** 200-run abilities onto their **pilot** abilities (Haebara / Stocking-Lord
   style for the confirmatory MIRT metric; at minimum a per-dimension linear `θ_link = s·θ + t`).
2. **Metric convention.** Keep the fitter's existing convention (`θ ~ N(0, I)` prior, latent
   correlation estimable via `--estimate-latent-corr`); the link is applied *post hoc* so the
   200-run fit is not distorted by the pilot.

### 3.4 Drift detection & handling (pre-registered thresholds)
Compute, on the anchor set:
- **Anchor ability agreement:** Pearson `r(θ_pilot, θ_200)` per dimension. **Expect ≥0.90 for
  correctness.** `< 0.80` on correctness ⇒ investigate before trusting the link.
- **Anchor residual / DIF:** after linking, flag any anchor whose per-dimension ability residual
  exceeds **|Δθ| > 1.0** (≈1 prior SD) or a robust **z > 3** as an **unstable anchor**; drop it
  from the linking set (report which and why) and refit the transform.
- **Item drift:** for TutorBench items common to both fits, correlate `b` and `a_correctness`
  across pilot↔200. `b`/`a_correctness` should track (pilot k-fold vs-all-82 medians were
  0.91/0.88); a large drop signals a population or judge shift.
- **Scaffolding caveat:** `a_scaffolding` is *expected* to move (it should *improve*, not merely
  agree) — so scaffolding drift is read as **stabilization**, judged by cross-fold stability
  rising toward correctness's, not by pilot↔200 identity.

---

## 4. Cross-benchmark linking design

### 4.1 What is shared vs benchmark-specific
- **Shared:** the **person population** — the same 200 models answer TutorBench + every MCQ/open
  benchmark. This is the *only* linking substrate; the benchmarks share **no items**.
- **Benchmark-specific:** item parameters, item pool, and dimensionality. TutorBench is the
  **multidimensional** instrument (M2PL over rubric criteria, 2–3 latent axes). The MCQ
  benchmarks (openbookqa, socialiqa, piqa, pedagogy) are naturally **unidimensional** (one
  ability per benchmark, 2PL/Rasch on item correctness). The other open benchmarks
  (tutoreval, biggen, wildbench, infobench, ifeval, edubench, bridge) are judged-rubric or
  instruction-following and are, for linking, treated as **unidimensional latent traits** unless
  a per-benchmark dimensionality check says otherwise.

### 4.2 Linking structure — **common-person (concurrent) calibration**
Because linking is person-based, we pre-register a **two-stage common-person equating**, not a
single joint monolith:
1. **Per-benchmark calibration.** Fit each benchmark's own IRT model on the 200 persons
   (TutorBench = M2PL as today; MCQ = 2PL). This yields, per benchmark, a person-ability vector
   on that benchmark's own metric.
2. **Cross-benchmark placement via the shared persons.** Standardize each benchmark's abilities
   to `θ ~ N(0,1)` (per dimension) over the common 200-person population, then estimate the
   **cross-benchmark person-ability correlation matrix** (and, optionally, a joint hierarchical
   person prior). The shared population *is* the equating link: two benchmarks are on a
   comparable scale because their θ's are expressed over the identical 200 models.
   - TutorBench-**correctness** is the natural hub axis (it correlates with MCQ correctness);
     report each benchmark's correlation to the TutorBench correctness (and scaffolding /
     presentation) axes.

This deliberately **avoids** forcing all benchmarks into one latent — different benchmarks
measure different things; the deliverable is a *correlation/placement map*, not a single scalar.

### 4.3 Identifiability constraints (pre-registered)
- **Metric fixing (each latent, each benchmark):** mean 0, unit variance on the 200-person
  population (standard M2PL/2PL identification).
- **Rotation (MIRT):** fixed by the **confirmatory Q-matrix** (each criterion loads only its
  assigned skills). This is what makes correctness/scaffolding/presentation nameable rather than
  arbitrary rotations — keep the confirmatory Q, do **not** switch to exploratory MIRT.
- **Sign/orientation:** fix each axis so higher θ = more able, oriented by the designated
  high/low **anchor models** (§3) — anchors also pin orientation across benchmarks.
- **Slot layout unchanged:** keep the documented 3-slot `SKILLS=(content, diagnosis,
  scaffolding)` repurposing (2-skill and 3-skill slot maps as in the Pilot Baseline Freeze), so
  the 200-run banks stay runtime-loadable via `Rubric.from_json` with no schema change.
- **Minimum persons:** `calibrate_mirt.py`'s own `--min-persons-identifiable=150` heuristic is
  now **satisfied** (N≈200) for the full 3-dim model — a key reason the 200-run can *confirm*
  presentation rather than defer it.

---

## 5. Load-once execution note (grounded in `cpu_sweep_200`)

*Operational grounding only — the runner is already engineered; no new code is proposed.*

- **Fleet launch.** `cpu_sweep_200/launch_fleet.sh` prints one command per shard; every host
  gets the **same `NUM_SHARDS`** and a unique `SHARD_INDEX`. Models are **round-robin sharded**
  over `models_200.yaml` (`select_models(... i % num_shards == shard_index)`), so ~200/`NUM_SHARDS`
  models land on each host. `run_cpu_worker.sh` sets `FORCE_CPU=1`, `--backend hf`, `--no-judge`,
  `--max-samples 2000`, and a **probe guard** (`--probe-questions 100 --probe-max-seconds 900`:
  time the first 100 items and skip a model that would blow the wall-clock budget).
- **Dataset prefetch.** `cpu_sweep_200/prefetch_datasets.py` normalizes + caches all 12
  suite benchmarks (`CPU_SWEEP_BENCHMARKS`) to local JSONL once, in parallel; `run_benchmark.py`
  additionally pre-flights each benchmark and **drops any unloadable one** so a gated/moved
  dataset can't abort a multi-day sweep.
- **Per-model-once loop.** The efficient default in `run_benchmark.py` holds **one model
  resident** and iterates **all benchmarks** for it (`== model <id> == → for benchmark in
  benchmarks`), then `engine.close()`. `--resident-all` reproduces the exact benchmark-outer /
  model-inner order in one co-located process. Either way **a model is loaded exactly once** and
  reused across every question of every benchmark — the whole point of the sweep.
- **Output schema (feeds calibration).** Results stream to **one durable file per
  (benchmark, model)** with question-level resume and `.done` manifests
  (`Outputs/mcq/<bench>/<org__model>.csv` = `correct|wrong`; `Outputs/open/<bench>/<org__model>.responses.jsonl`
  = raw responses only, since `--no-judge`). Optional continuous `aws s3 sync` mirrors
  `Outputs/` to the S3 prefix; sync failures never block inference.
- **Downstream to the response matrix (the calibration input).** The sweep produces **raw open
  responses**, *not* judged scores. A **separate judging pass** (Prometheus judge over the
  TutorBench rubric bank, 6,845 criteria) converts responses → the `{0,1,NaN}` rubric
  **response matrix** (200 × criteria), which is what `calibrate_mirt.py` consumes. **The judge
  step is where calibration risk concentrates (§7), not the sweep.** MCQ `correct|wrong` feeds
  the per-benchmark 2PL fits of §4 directly.

---

## 6. Pre-registered analysis plan

All decision rules below are fixed **before** seeing 200-run data.

### 6.1 Primary instrument
**2-skill M2PL `[correctness, scaffolding]`.** Structure: content+diagnosis→correctness
(collapse settled; content↔diagnosis ≈0.94), scaffolding separate. Fit settings frozen to the
pilot's: Bock–Aitkin EM/MML, `ridge=1e-3`, `grid=7`, `tol=1e-4`, `max_iter=200`,
`--estimate-latent-corr`. Same variance/Q-row filters (`prepare_block`). Report the calibrated
bank in the existing 3-slot layout so it stays runtime-loadable.

### 6.2 Scaffolding (the weak axis) — pre-registered handling
- **Success criterion:** cross-fold (k=5) median `a_scaffolding` **≥ 0.65** at N≈200 (up from
  0.48 at N=82). If met ⇒ relabel scaffolding **healthy**. If **0.50–0.65** ⇒ keep "needs
  refit," report as lower-confidence. If **< 0.50** ⇒ scaffolding stays diagnostic-only and we
  escalate persons/regularization.
- **Extreme-loading guard:** the pilot's fast-but-fake scaffolding "convergence" came from a few
  extreme-`a` items (max ≈18). Pre-register a **report** of the count of `|a_scaffolding|>1.5`
  and negative loadings; if extremes persist at N≈200, apply the documented negative-loading
  clamp (as in the freeze) and flag over-fit items — **do not** silently let them drive CAT SE.

### 6.3 ⭐ Presentation axis — pre-registered ACCEPT/REJECT rule
Evaluated **once**, on the frozen 200-run fit (same-item-block model comparison + k-fold), on
the complete matrix (incl. `style_surface` optional criteria), **before any SE/budget tuning**.

**ACCEPT presentation as a reported 3rd axis iff ALL hold:**
1. **Model comparison (same item block):** full 3-dim `[corr, scaff, presentation]` beats
   **fold-into-scaffolding** on **both AIC and BIC** (ΔAIC>0 **and** ΔBIC>0), replicating Run 6
   (which had ΔAIC +1,238 / ΔBIC +1,217 at N=82).
2. **Clean optimum:** 3-dim loglik **above** both nested folds (no Run-4-style under-identification).
3. **Identifiability:** cross-fold (k=5) median `a_presentation` **≥ 0.65** **and** `≥` the same
   run's median `a_scaffolding` (presentation must be no worse-identified than the shipped axis;
   at N=82 it was 0.712 > 0.488).
4. **Independent recovery:** OOS (k-fold) presentation recovery **r ≥ 0.78** (N=82: 0.822) **and**
   CAT-ability `r(presentation, correctness) ≤ 0.90` (materially below latent collinearity;
   N=82 CAT r was 0.810).
5. **Item health:** ≥90% of assigned `style_surface` presentation items survive the variance
   filter; median `|a_presentation|` ∈ [0.5, 3.0]; ≤3% negative loadings.

**REJECT → demote presentation to diagnostic-only if ANY hold:**
- fold-into-scaffolding wins AIC, **or** 3-dim loglik below a nested fold, **or**
- cross-fold `a_presentation` **< 0.50**, **or** OOS recovery **r < 0.70**, **or**
- CAT `r(presentation, correctness) > 0.95` (effectively redundant with correctness).

**MIXED (some criteria pass, some fail):** presentation is reported as **EXPLORATORY /
secondary** — shown in tables and figures but **not** part of the primary shipped instrument,
pending a further N or a targeted presentation-item expansion.

> The primary 2-skill instrument does **not** depend on the presentation outcome; §6.3 only
> decides whether the optional 3rd axis is promoted, held exploratory, or demoted.

### 6.4 OOS honesty check — person-level k-fold
- **Method (frozen to the existing harness):** partition the **200 persons** into folds
  (seeded), fit M2PL on TRAIN persons, freeze items, **EAP-score** held-out persons from their
  own observed cells, compute pooled cell-level **log-loss / accuracy / AUC / Brier** and the
  **optimism gap** vs the all-200 in-sample fit.
- **k = 5 headline, k = 10 robustness** (as in the 2-skill k-fold memo). Larger train folds at
  N=200 (~160/180 train persons) should **shrink** the gap vs the pilot's already-small gap
  (AUC −0.027).
- **Item-param cross-fold stability** (`b`, `a_correctness`, `a_scaffolding`, and, if presentation
  accepted, `a_presentation`) reported as the honest identifiability read.
- **Pre-registered pass:** OOS AUC ≥ 0.88 pooled and optimism gap in AUC ≤ 0.04 ⇒ calibration
  "generalizes." (Pilot: 0.905 / 0.027 — the 200-run should match or beat.)

### 6.5 CAT policy to apply (pre-registered)
- **Selection/update/scoring:** unchanged multidimensional CAT — 3-D (or 2-D) Gauss–Hermite EAP,
  **multidimensional Fisher-info** next-item selection, exact MIRT Newton/Laplace update
  (`tutor_cat.mirt.update`), all imported (no re-implementation), via
  `scripts/cat_eval_tutorbench_multiskill.py --skills {2,3}`.
- **Stopping rule:** per-dim SE `< 0.30`, `min-items 1`, `max-items 100`, seed fixed.
- **Presentation budget (pre-registered *if* accepted):** because SE<0.30 on presentation
  roughly **doubled** test length at N=82 (only 30/82 converged), pre-register presentation with
  a **relaxed SE target (0.40) or an item cap**, reported alongside the strict-0.30 numbers.
  This is a *deployment budget* knob, decided from the recovery/exposure evidence, **not** a
  validity gate.
- Report CAT recovery **per dimension** (in-sample + OOS), CAT length, item-exposure by skill,
  and pIRT MAE — mirroring `reports/cat_eval*/`.

### 6.6 Metrics & tables to report (deliverables)
Reference-shaped, mirroring `reports/cat_eval/cat_summary_table.md`:
1. **Calibration summary** per instrument (2-skill primary, 3-skill optional): items kept
   (fit_block/total), bank health per axis, `a` median per axis, latent correlation matrix,
   AIC/BIC same-item-block model comparison (uni / fold-scaffolding / fold-correctness / full).
2. **k-fold OOS table:** pooled log-loss/acc/AUC/Brier + optimism gap; item-param cross-fold
   stability (`b`, `a_correctness`, `a_scaffolding`, `a_presentation`).
3. **CAT recovery table:** recovery r per dim (in/OOS), CAT items to converge, pIRT MAE,
   item-exposure by skill.
4. **Anchor/linking report:** anchor list, per-dim `r(θ_pilot, θ_200)`, linking transform,
   dropped/unstable anchors, item-drift correlations (§3.4).
5. **Cross-benchmark placement map:** cross-benchmark person-ability correlation matrix
   (TutorBench axes × MCQ/open benchmarks) (§4).
6. **Figures:** recovery scatters (per axis), CAT length hist, SE-reduction curve, pIRT
   calibration, item-info & item-exposure by skill, pIRT/anchor drift plots.
7. **Provenance:** N, `n_items_fit`, matrix sha256, seeds, fit settings, slot map, judge model +
   version, no_decision rate — stamped as in the Pilot Baseline Freeze.

---

## 7. Risks / threats to validity & mitigations

| # | Threat | Why it invalidates calibration | Pre-registered mitigation |
|---|---|---|---|
| 1 | **Judge `no_decision` / abstain rate** | Missing rubric cells → sparser matrix; if non-random (harder items abstained), biases `b`/`a` and inflates fit. | **Report the no_decision rate per benchmark/criterion up front.** Treat abstains as **NaN (marginalized)**, never as 0/1. Pre-register a **≤10% no_decision ceiling per criterion**; criteria above it are dropped from the fit block (recorded), not imputed. Fix judge model+version+seed; log it in provenance. |
| 2 | **Sparse cells / all-fail columns** | Zero-variance items can't be calibrated; thin cells give unstable `a`. | Keep the existing `prepare_block` filters (drop all-fail/zero-variance/all-zero-Q; retain their heuristic params, never zero them, as in the freeze). Report surviving-item counts per axis. |
| 3 | **Correctness↔scaffolding / ↔presentation collinearity** | Latent r(corr,pres)=0.945, r(corr,scaff)=−0.48; high collinearity → weakly-identified second/third axes; risk of a rotation reading correctness variance as scaffolding. | **Model selection (§2) deliberately decorrelates** via matched base↔instruct pairs (scaffolding variance at fixed correctness). Keep the **confirmatory Q** (fixes rotation). Report the full latent-corr matrix and the CAT-ability de-correlation (Run 6/CAT precedent: CAT r 0.81 ≪ latent 0.945). |
| 4 | **Ability-range restriction (low-end floor removed)** | `models_200.yaml` drops the pilot's GPT-2/Pythia/BLOOM/OPT floor → truncated correctness range → `b` unanchored at the low end, inflated discriminations. | **§2.4 reintroduction of ~10–15 low-ability HF models** for the CPU sweep. Report the correctness ability histogram; if the floor is thin, flag `b` estimates in that region as extrapolated. |
| 5 | **Anchor instability / linking error** | If common-person anchors don't agree pilot↔200, the link (and any "drift" claim) is untrustworthy. | **≥25 well-spread anchors (§3.2)**, per-dim agreement thresholds and residual/DIF drop rule (§3.4). Reintroduce low-ability models to grow anchor count/spread. Report the transform and every dropped anchor. |
| 6 | **Scaffolding over-fit (extreme-`a`)** | A few extreme-discrimination scaffolding items fake fast CAT convergence onto unstable estimates. | §6.2 extreme-loading report + negative-loading clamp; trust scaffolding *recovery r* (value) over *SE* (uncertainty). |
| 7 | **Presentation test-length cost** | SE<0.30 on presentation ≈doubles CAT length (30/82 converged) — an operational, not validity, failure that could be mistaken for the axis being bad. | §6.5 relaxed presentation SE/cap budget, reported next to strict numbers; §6.3 keeps the accept/reject rule on *validity* evidence (fit, recovery, identifiability), not test length. |
| 8 | **Population/judge drift vs pilot** | The 200-set is a different population; item `b`/`a` could shift for reasons unrelated to more data. | §3.4 item-drift correlations on common TutorBench items; fixed judge version; provenance stamping; the independent-then-align linking (§3.3) so the 200 fit is not distorted by the pilot. |
| 9 | **Model load failures / probe-skips shrink N** | If many carryover/anchor models fail to load on CPU-HF or trip the probe, effective N and anchor count fall below plan. | Runner already logs `[skip model]`/`[skip slow model]` and resumes; **pre-register a re-run of failed anchors** (tune `PROBE_MAX_SECONDS`) so anchor count stays ≥25 usable. Report realized N and skip list. |
| 10 | **Cross-benchmark over-claim** | Forcing all benchmarks onto one scalar would be invalid (they measure different things). | §4 keeps benchmarks on **their own** latents linked by **shared persons**; deliverable is a correlation/placement map, not a single number. |

---

## 8. Open decisions needing user input

1. **Final model list / low-ability reintroduction (blocking).** Approve adding ~10–15 pilot
   low-ability models (GPT-2 / Pythia / BLOOM(z) / OPT / RedPajama / TinyLlama / phi-2) back for
   the **CPU-HF** sweep to preserve the low ability floor and grow the anchor set from ~28 to
   ~40? Or keep the curated `models_200.yaml` exactly (accepting a restricted low end)? This
   changes both the model list and the anchor count.
2. **Anchor count target.** Confirm **≥25 usable common-person anchors** (default = full pilot∩200
   carryover, §2.3), or specify a different minimum / a hand-picked anchor subset.
3. **Presentation accept/reject thresholds (§6.3).** Confirm the exact cutoffs (esp.
   `a_presentation ≥ 0.65`, OOS `r ≥ 0.78`, CAT `r(pres,corr) ≤ 0.90`), or adjust.
4. **Scaffolding success cutoff (§6.2).** Confirm cross-fold `a_scaffolding ≥ 0.65` as the
   "healthy" bar, or set a different target.
5. **Judge for the calibration matrix.** Confirm the judge model + version + the `no_decision`
   ceiling (default ≤10%/criterion), since the calibration input depends entirely on it and the
   CPU sweep itself runs `--no-judge`.
6. **CAT presentation budget.** If presentation is accepted, confirm the deployment stopping
   rule (strict SE 0.30 vs relaxed 0.40 / item cap).

---

## 9. Reproducibility / read-only guarantees

This memo writes **nothing** to the runtime: no fitter edits, no `--write-params`, no changes to
`data/TutorBench/curated/*`, `response_matrix*.csv`, `config.yaml`, or the runner. The 200-run,
when executed, reuses the existing read-only harnesses (`calibrate_mirt.py`,
`cat_eval_tutorbench_multiskill.py`, `kfold_cv_mirt.py`) with the settings frozen above, and
writes new artifacts under `staging/` + tracked `reports/` copies, exactly as the pilot studies
did. All decision rules in §6 are fixed **before** 200-run data exists.

---

## 10. Decision Log — §8 RESOLVED (LOCKED 2026-07-30)

All §8 open decisions are signed off; the pre-registration is **locked prior to any 200-run data**.

1. **Model list (blocking):** KEEP `models_200.yaml` as-is — **no** low-ability reintroduction. Anchor pool = the ~28 pilot∩200 carryover; the restricted low correctness end is accepted (report the ability histogram and flag low-end `b` as extrapolated, per §7 risk #4).
2. **Anchor count:** **≥25 usable** common-person anchors (= the full ~28 carryover), spread across the ability range.
3. **Presentation accept/reject (§6.3):** confirmed as written — `a_presentation ≥ 0.65`, OOS `r ≥ 0.78`, CAT `r(pres,corr) ≤ 0.90`, beats fold-into-scaffolding on AIC+BIC.
4. **Scaffolding cutoff (§6.2):** confirmed — cross-fold median `a_scaffolding ≥ 0.65` = healthy; 0.50–0.65 = needs-refit / lower-confidence; < 0.50 = diagnostic-only.
5. **Judge / no_decision:** frozen Qwen judge; `no_decision` → **NaN** (marginalized, never 0/1); **≤10%/criterion** abstain ceiling, criteria above it dropped from the fit block (recorded, not imputed).
6. **CAT presentation budget:** report **both** strict SE 0.30 and relaxed 0.40/cap; deployment budget chosen later from recovery/exposure evidence (not a validity gate).

**Status: pre-registration LOCKED.** Remaining pre-launch work is engineering/analysis (CAT policy, scaffolding hygiene, baseline freeze) — none of which changes these rules.
