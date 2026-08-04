# Calibration Log (MIRT structure decisions, Runs 1-6 + k-fold + hygiene)

Consolidated chronological record of the TutorBench MIRT calibration: the dimensionality
decisions (Runs 1-6), the k-fold generalization study, the ridge-hygiene decision, and the
correctness leaderboard deliverable. Supersedes the nine separate memos (Calibration Run 1-6,
K-Fold CV, Scaffolding Hygiene, Correctness-Axis Leaderboard). For the frozen post-decision
bank see `Pilot Baseline Freeze (2 and 3 skill).md`.

## Shared method + standing caveats (apply to every run)

- Fitter: `scripts/calibrate_mirt.py` (Bock-Aitkin EM/MML, Gauss-Hermite quadrature, confirmatory
  M2PL). All runs are **read-only**: no `--write-params`, fitter and curated bank byte-for-byte
  unchanged, nothing committed. Candidate-axis runs use a **slot-repurposing trick**: the fitter
  reads `SKILLS=(content,diagnosis,scaffolding)` agnostically, so an experimental bank repurposes
  the 3 slots as `content->correctness (=content OR diagnosis)`, `diagnosis->scaffolding`,
  `scaffolding-><candidate>`. Under this, `--collapse diagnosis,scaffolding` folds the candidate
  INTO scaffolding and `--collapse content,scaffolding` folds it INTO correctness.
- **N = 82 persons < 150 identifiability floor** in every pilot run (`identifiable: false`).
  All absolute numbers are provisional; the durable fix for the weak axis is more *persons*, not
  more items or tweaks.
- **BIC prefers unidimensional in every run** - a program-wide parsimony artifact at ~270k-330k
  observed cells that also "rejects" the accepted correctness/scaffolding split. AIC (+ nested
  model comparisons + latent-r + cross-fold stability) is the operative criterion, not BIC.
- Cells in {0,1,NaN}: `pass=1, fail=0, no_decision->NaN, absent->NaN`. Matrix is fail-dominated.

## Final adopted structure

- **Collapse content + diagnosis -> "correctness"; keep scaffolding separate** (Run 1, reinforced
  Run 5).
- **Presentation adopted as an optional 3rd axis** -> instrument `[correctness, scaffolding,
  presentation]` (Run 6, on complete supplement).
- **`ridge = 1e-2`** is the M2PL fit default (adopted 2026-07-31, Scaffolding Hygiene).
- Rejected/deferred 3rd-axis candidates: motivation (DEFER), metacognition (reject), communication
  (defer - must grade the items first).

---

## Run 1 - Collapse decision (2026-07-29). Matrix `staging/response_matrix.csv`, 82 x 6,180.

- Fit block 3,347 items / 268,936 observed cells (dropped 244 all-NaN + 1,976 all-fail + 613
  missing-Q). 0 Q-matrix misalignment (all 6,180 columns in the curated bank).
- **content <-> diagnosis latent r = 0.946** (approx the synthetic prior 0.945) -> pre-registered
  collapse rule FIRES.
- Model comparison (same block): collapsed 2-skill AIC 131,201.85 (winner) < full 3-dim 133,594.53;
  ** collapsed beats full by ΔAIC +2,392.7 / ΔBIC +15,394.4**. Unidim AIC 132,391.
- Scaffolding is a distinct axis: r(content,scaff)=-0.170, r(diagnosis,scaff)=-0.181, collapsed
  r(correctness,scaff)=-0.418.
- **Decision: COLLAPSE content+diagnosis -> correctness; keep scaffolding separate (provisional,
  N=82).** Note: `--collapse` is a fit-time diagnostic; adopting the collapse is an upstream
  Q-matrix change, not a `--write-params` op. (Unidimensional `calibrate_partial.py` girth/rasch
  did not fit: no hole-free dense block + `girth` not installed; the authoritative unidim baseline
  is the numpy EM 1-dim inside `calibrate_mirt.py`.)

## Runs 2-3 - Candidate 3rd axes on the same 82 x 6,180 matrix (all rejected/deferred)

| Candidate | Run | fitted (assigned->survived) | r(correctness, cand) | same-set verdict | decision |
|---|---|---|---:|---|---|
| motivation / affect (`hit_affect_motivation`) | 2 | 605 -> 485 | 0.90 (identified) | beats both folds on AIC; BIC-vs-correctness margin thin (+460) | **DEFER** |
| metacognition (`hit_metacognitive`) | 3 | 34 -> 20 | 0.999 (clip) | folding INTO correctness *improves* AIC+BIC | **reject** (redundant/too sparse) |
| communication clarity (`hit_communication_clarity`) | 3 | 33 -> 24 | 0.999 (clip) | distinct from scaffolding; wash vs correctness | **defer** (575/586 tags land on ungraded optional criteria - grade them first) |

- Motivation is healthy (median a 0.98 approx correctness 1.03, > scaffolding 0.42; pass-rate spread
  0.01-0.75) and distinct from scaffolding (r=-0.47), but r=0.90 with correctness at non-identifiable
  N -> park for a powered re-test with more affect-capable (instruct/RLHF) tutors.
- Metacognition/communication pin to the +0.999 clip (only ~20-24 fitted items) = "no separable
  signal," a weaker basis than motivation's genuine 0.90. **Interim `[correctness, scaffolding]`
  stands after Runs 2-3.**

## Run 4 - Presentation on the ~50% partial supplement (2026-07-29) -> DEFER

- Combined matrix `staging/response_matrix_full.csv` (82 x 6,845) from `run_data.jsonl` +
  `run_data_supp.jsonl` (partial); only 427/662 presentation criteria graded. New helpers:
  multi-jsonl / `--include-optional` ingest, bank-`--select-dimension`.
- Presentation is a **healthy, high-variance cluster** (423 survive, pass-rate 0.01-0.66, median
  a=1.08) - categorically stronger than Run 3's probes. r(correctness,presentation)=+0.979
  (identified, not clipped); distinct from correctness (fold-into-correctness costs ΔAIC +561).
- BUT fold-into-scaffolding *beat* the 3-dim (ΔAIC +75 / ΔBIC +96), AND the 3-dim converged to a
  loglik *below* its own nested submodel (local optimum -> under-identified at N=82).
- **Verdict: DEFER** - real and distinct from correctness, but a separate axis not yet earned on
  partial, under-powered data. Finish grading presentation + N>=150, then re-test.

## Run 5 - Complete supplement (2026-07-30): presentation EARNS the axis; 2-skill reinforced

Complete supplement `supplement_v2/run_data_supp_v2.jsonl` (74,329 rows) grades all 662/662
presentation + 3 rescope_optional + 244 nonoptional hole-fills. `scripts/merge_full_matrix.py`
rebuilds the combined matrix: **82 x 6,845, 98.1% fill, 0 all-NaN columns**, 0 big-run<->supplement
overlap (0 disagreements). Nonoptional slice `response_matrix_full_nonopt.csv` = 82 x 6,180,
3,497 items / 280,943 cells.

### Presentation (item set 4,156 / 332,637 cells) - Run 4 verdict FLIPS
- 656/662 survive; pass-rate 0.01-0.70, median a_presentation=1.09. r(corr,pres)=**0.945** (lower
  than Run 4's 0.979, cleanly identified), r(scaff,pres)=-0.559.
- Full 3-dim is the **AIC winner**: beats fold-into-scaffolding by ΔAIC **+1,238**, fold-into-
  correctness by ΔAIC **+1,772**. Run 4's under-identification is **resolved** - 3-dim loglik
  (-78,679.83) now cleanly exceeds both nested folds. -> **presentation earns a separate 3rd axis
  under AIC.**

### 2-skill definitive + k-fold refresh (nonoptional slice)
- Collapsed 2-skill AIC 137,789.53; **beats full 3-way by ΔAIC +2,674.3 / ΔBIC +16,331** (larger
  margin than Run 1); the full 3-way loglik is now *worse* than the collapsed (stronger collapse
  evidence). content<->diagnosis r=0.942; collapsed correctness<->scaffolding r=**-0.462**.
- k-fold refresh on full slice: OOS AUC 0.905 (k=5) / 0.909 (k=10), small optimism gap - unchanged
  vs sparse. **Scaffolding NOT rescued by the extra items** (k=5 stability 0.48->0.495; k=10 fell
  0.725->0.637): the bottleneck is 82 persons, not item coverage. Add persons, not items.

## Run 6 - Presentation gate re-test on complete data (2026-07-30) -> ADOPT presentation

Confirmatory gate before the freeze (same complete matrix, same fit as Run 5).
- Full 3-dim beats both folds on **AIC and (nested) BIC**: vs scaffolding ΔAIC +1,238 / ΔBIC +1,217;
  vs correctness ΔAIC +1,772 / ΔBIC +1,750. Clean optimum (loglik above nested submodels).
- Health: 656/662 survive, median a_presentation 1.09. **Cross-fold stability (k=5):
  a_presentation median 0.712 (min 0.630) > a_scaffolding 0.488** and approaches a_correctness
  0.772 - presentation is *better* identified than the axis already in the instrument.
- Tension: r(corr,pres)=0.945 approx content<->diagnosis 0.942 (which we collapsed), but the
  confirmatory item-loading fold is decisive (folding presentation into correctness is the *worst*
  model), so high latent-r does not imply collapse.
- **Verdict: adopt presentation as an optional 3rd axis `[correctness, scaffolding, presentation]`.**
  Does not block freezing the correctness+scaffolding core; DOES block freezing a presentation-
  *excluding* instrument as final. Confirm/lock presentation params at N->200 rather than gating on it.

## K-Fold Cross-Validation study (2-skill, `staging/kfold/`)

Addresses the circularity that the bank was calibrated on the same 82 models the CAT scores.
Person-level folds (seed 20260729): fit on TRAIN models, freeze params, EAP-score HELD-OUT models.

| Metric | k=5 OOS | k=10 OOS | in-sample (all 82) | gap (k=5) |
|---|---:|---:|---:|---:|
| Log-loss | 0.2635 | 0.2548 | 0.2160 | +0.0475 |
| Accuracy | 0.8943 | 0.8966 | 0.9060 | -0.0118 |
| AUC | 0.9045 | 0.9087 | 0.9312 | -0.0267 |
| Brier | 0.0765 | 0.0748 | 0.0672 | +0.0093 |

- Small optimism gap -> Run-1 in-sample fit was NOT materially inflated by circularity.
- Item-param cross-fold stability (median pairwise Pearson): `b` 0.853 (k5)/0.919 (k10),
  `a_correctness` 0.800/0.888, **`a_scaffolding` 0.478/0.725** - scaffolding is the weak link at
  k=5 but stabilizes at k=10 (small-train-N effect, not structural non-identifiability).
- CAT pilot (fold 0, 17 unseen models): ~20 adaptive items recover predicted score to ~0.03 MAE;
  raw 2-d ability L2 approx 1.1 (looseness concentrated in scaffolding). Trust rank-order +
  pass-probability (driven by `b`, `a_correctness`); treat scaffolding ability as lower-confidence.

## Scaffolding Hygiene - ridge sweep (2026-07-31) -> ADOPT ridge = 1e-2

Audit (baseline ridge 1e-3): of 768 Q-scaffolding items, **293 (~38%) fit negative loadings** and
**22 blow up (|a|>=6)** as near-separation artifacts (pass-rate ~0.01-0.09, weak point-biserial) -
the scaffolding axis is not just thin, it is *noisy*. Ridge sweep (OOS CAT recovery, k=5):

| variant | OOS correctness | OOS scaffolding | extreme-a | scaff stability |
|---|---:|---:|---:|---:|
| baseline 1e-3 | 0.916 | 0.536 | 22 | 0.558 |
| **ridge 1e-2** | **0.931** | **0.667** | 6 | 0.618 |
| ridge 2e-2 | 0.929 | 0.670 | 1 | 0.631 |
| scaff-ridge 5e-2 | 0.927 | 0.663 | 0 | **0.777** |
| cap \|a\|<=4 | 0.929 | 0.532 | 0 | 0.722 |
| exclude 22 extreme | 0.930 | 0.620 | 19* | 0.523 |

- **Global ridge is a clean monotone win** across all three metrics; OOS scaffolding plateaus at
  ~1e-2. **Adopt 1e-2**: best OOS scaffolding, correctness *improves* (guardrail held), extreme-a
  22->6, one-line default change, no bank edits. Runner-up scaff-ridge 5e-2 only if maximizing
  cross-fold stability; cap and exclude not recommended.
- Caveats: N=82 is still the bottleneck (best OOS scaffolding ~0.67 vs correctness ~0.93); do NOT
  hard-code 1e-2 as the permanent default - re-run this sweep at the 200-run and pick the plateau
  empirically. (Fresh-baseline OOS scaffolding 0.536 is below the older stale-bank 0.644; the robust
  claims are the monotone trend + correctness improvement, independent of baseline.)

## Correctness-Axis Leaderboard (82 models) - deliverable

Scores the frozen 2-skill bank (ridge=1e-2, floor=15,
`data/TutorBench/rubrics_qmatrix_calibrated_2skill.jsonl`, 3,443 items, 60 `exclude_from_fit`);
no re-fit. Two abilities per model: full-response EAP (fine 41-node grid, the sort key) and CAT theta.
- **Recovery:** in-sample Pearson r=0.934 / Spearman 0.953 (fine grid), pilot grid-7 r=0.964;
  **honest OOS (k-fold) correctness r approx 0.929** - the number to quote for generalization.
- Face-valid ranking: top Qwen/Qwen3-4B (theta_full 3.46), Qwen3-1.7B/Qwen2.5-7B-Instruct/Falcon3-7B
  (2.45); bottom pythia-70m (-5.86), gpt2-medium, pythia-160m, bloom-560m.
- Full 2-skill test length is long (mean 47 / median 40 items) because the scaffolding floor of 15
  keeps it running; **correctness alone reaches SE<0.3 at ~24 items** (81/82 models). Tail shrinkage
  (weak models pulled toward -2 by the N(0,1) prior + short test) preserves rank (rho=0.95) but
  regularizes absolute CAT theta in the tails. Outputs: `reports/leaderboard_correctness_2skill/`.
