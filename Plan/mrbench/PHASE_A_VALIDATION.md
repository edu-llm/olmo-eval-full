# MRBench Phase A — Judge Validation Record

**Verdict: ✅ VALIDATED**

- **Judge model:** Claude Haiku 4.5 (`claude-group/claude-haiku-4-5`) via the TrueFoundry AI Gateway.
- **Date:** 2026-08-09
- **Scope:** full Phase-A live run — all 1,589 baked-in tutor responses × 8 dimensions = **12,712 judge calls** (temperature 0, single sample). 12,712/12,712 succeeded; **0 failures, 0 parse errors, 0 fallbacks**.
- **Actual cost:** **≈ $12.10** at Haiku rates ($1 / $5 per MTok in/out), from ~8.30M input + ~0.76M output tokens (avg 653 in / 60 out per call).
- **Full machine-readable metrics + provenance:** `diagnostics/mrbench/runs/phase_a_haiku_4_5_2026-08-09/` (`metrics.json`, `metrics.txt`, `run_provenance.json`) — under the gitignored `runs/` (local record only).

## Acceptance gate (decided)

A dimension **passes** if `AC ≥ 0.30` **and** its 95% bootstrap CI lower bound `> 0`.
The judge is **validated** if it passes **≥ 6/8** dimensions **and** beats **both** paper baselines (Prometheus2, Llama-3.1-8B) on **≥ 7/8**. Weak dimensions are caveated, not fatal. Human-likeness is reported but **excluded** from the gate. Self-bias is **N/A** for Haiku (not one of the 9 baked-in tutors).

**Result:** 6/7 gate-eligible dimensions pass (only Coherence fails) **and** beats both baselines on 7/8 → **VALIDATED**.

## Per-dimension AC (judge vs human gold), 95% CIs, macro-F1, gate

| Dimension | AC | 95% CI | macro-F1 | Gate |
|---|---|---|---|---|
| Mistake_Identification | +0.49 | [+0.46, +0.54] | 0.47 | **PASS** |
| Mistake_Location | +0.37 | [+0.32, +0.41] | 0.44 | **PASS** |
| Revealing_of_the_Answer | +0.72 | [+0.68, +0.77] | 0.60 | **PASS** |
| Providing_Guidance | +0.48 | [+0.43, +0.52] | 0.54 | **PASS** |
| Actionability | +0.42 | [+0.37, +0.46] | 0.53 | **PASS** |
| Coherence | +0.25 | [+0.20, +0.30] | 0.36 | **FAIL** (AC < 0.30) |
| Tutor_Tone | +0.44 | [+0.40, +0.47] | 0.45 | **PASS** |
| Humanlikeness | +0.38 | [+0.31, +0.46] | 0.48 | *excluded* |

n = 1,553 (judge, gold) pairs per dimension. **Macro-F1 mean (secondary, BEA-comparable) = 0.48** — reporting only; primary tutor-quality metric stays DAMR, judge-reliability stays Pearson AC.

## Beats both paper baselines on 7/8 dimensions

Per-dimension AC, mean across the 9 tutors (comparable to the paper's Tables 5/6):

| Dimension | Ours | Prometheus2 | Llama-3.1-8B | Beats both |
|---|---|---|---|---|
| Mistake_Identification | +0.38 | −0.16 | −0.16 | ✓ |
| Mistake_Location | +0.30 | −0.15 | −0.18 | ✓ |
| Revealing_of_the_Answer | +0.71 | −0.22 | −0.28 | ✓ |
| Providing_Guidance | +0.31 | −0.24 | −0.31 | ✓ |
| Actionability | +0.28 | −0.10 | −0.16 | ✓ |
| Coherence | +0.15 | −0.12 | −0.16 | ✓ |
| Tutor_Tone | +0.42 | −0.32 | −0.36 | ✓ |
| Humanlikeness | +0.04 | +0.08 | +0.08 | ✗ |

The paper's critics are near-zero/negative on almost every dimension; the Haiku judge is strongly positive on all except Human-likeness. **7/8** (the miss is Human-likeness, which is excluded from the gate anyway).

## Caveats

- **Coherence — weak.** AC 0.25 (below the 0.30 bar; CI upper bound sits at 0.30) and macro-F1 0.36 — the judge is a poor discriminator here. Caveated, **not fatal** per the decided policy; treat Coherence numbers as low-confidence.
- **Mistake_Location — moderate.** Passes but AC is only 0.37.
- **Human-likeness — excluded from the gate.** Its pooled AC (0.38) is inflated by cross-tutor variance; per-tutor it is ~0 and it is the one dimension that does **not** beat the baselines.
- **Self-bias — N/A.** Claude Haiku is not one of the 9 baked-in MRBench tutors, so there is no self-authored subset to exclude (the "Sonnet is one of the judged tutors" note in the tooling refers to the *tutor* Sonnet in the data, not the judge).

## Reproduce (from the existing cache; no API calls)

```bash
uv run python -m diagnostics.mrbench.judge_run --metrics --bootstrap 1000 --macro-f1
```
