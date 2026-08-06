# EAP-posterior stop rule vs online-SE stop rule (ADOPTED as of-record)

> **UPDATE — ADOPTED.** Following this prototype, the EAP-posterior stop is now the WildBench
> **of-record** CAT stop rule. The stop-dependent experiments (exp-04/05/06/09/10/11 + deployed
> SE) were re-run under it; the prior online-SE outputs are archived under
> `experiments/archive_onlineSE/`. See the top-level `README.md`. This file is kept for
> provenance of the original head-to-head that motivated the switch. The numbers below match the
> of-record re-run (recovery r≈0.975, precision-reached ~87%, order/seed θ-SD ~0.13).

**Original status (at prototype time): exploratory.** The of-record experiments were then
UNTOUCHED. This changes ONLY the CAT stop rule; it does not modify the production engine,
refit the bank, or re-run the SE_param bootstrap. The adaptive scenario *selection* is the
production engine's; we obtain each model's adaptive administration order by running the
engine to a forced length (L_max=40), then apply the EAP-posterior stop post-hoc:

> stop at the first scenario with `n_scenarios ≥ 8` AND `EAP posterior SD ≤ 0.12`
> (posterior ∝ L(administered responses | θ)·N(0,1) on a fine 321-node θ grid), else cap.

Both rules run at the locked settings (floor 8, target 0.12, MWLE, grid-7 catpool bank,
seed 20260729). SE_param is REUSED from the of-record deployed bootstrap (no re-bootstrap;
conservative for the longer EAP test).

## Head-to-head (WildBench, N=52)

| metric | online-SE rule | **EAP-posterior rule** |
|---|---|---|
| scenarios (median / mean) | 8 / 8.8 | 8 / **13.8** |
| criteria (median / mean) | 85 / 87 | 89 / **136** |
| % reaching SE_ability ≤ 0.12 | 31/52 (60%) | **48/52 (92%)**, 4 cap |
| SE_ability (median / mean) | 0.110 / 0.152 | 0.110 / **0.117** |
| SE_total (median / mean) | 0.131 / 0.179 | 0.128 / **0.150** |
| OOS recovery r [CI] | 0.957 | **0.975** [0.958, 0.986] |
| OOS recovery slope | 0.890 | **0.921** |
| OOS θ-MAE | 0.428 | **0.357** |
| OOS mean length | 9.0 scen / 86 crit | 13.2 scen / 129 crit |

(% reaching for the online rule is 31/52 here at seed 20260729 vs 34/52 in the of-record
seed-42 `se_post_vs_total.csv` — a seed difference; the honest metric is ~60% either way.)

## Findings

1. **Precision honesty ↑↑.** The EAP rule raises the fraction that actually reach the 0.12
   posterior-SD target from **60% → 92%**. The median test length is **unchanged (8
   scenarios)** — typical, already-precise models stop at the same place; only the borderline
   models extend (Q3 length 8 → 16 scenarios). Mean length rises 8.8 → 13.8 because the tail
   pays for its extra precision.
2. **Recovery improves** on all three: r 0.957 → **0.975**, slope 0.890 → **0.921** (closer to
   1), θ-MAE 0.428 → **0.357** — at ~+4 scenarios mean length. Not hurt; modestly better.
3. **The 4 that still cap are exactly the `weakly_identified` tail** (OLMo-1B-hf, mGPT,
   smol_llama-220M-GQA, smol_llama-220M-openhermes) — near-all-fail with essentially no
   informative items at their θ, so even 40 scenarios can't reach 0.12. Notably the three
   *non-flagged* near-all-fail models (SmolLM2-360M, EuroLLM-1.7B, Yi-6B-200K) that missed
   under the online rule **now close** under the EAP rule — they were never truly degenerate,
   just under-administered by the optimistic online-SE stop.

## Op-point

At floor 8 the EAP rule gives OOS r = 0.975, so the locked **8/0.12 still comfortably holds**
(r ≥ 0.95). The *shortest* r≥0.95 point could shift under the stricter EAP stop — a full
floor×SE re-sweep under the EAP rule is WARRANTED to re-derive it, but was **not run** (out of
scope for this prototype).

## Recommendation (for the user to review)

**Favorable — recommend adopting the EAP-posterior stop**, because it is the more honest
measurement (true posterior SD vs the engine's optimistic online normal-approx), it raises the
reach rate 60%→92%, improves recovery, keeps the median test length unchanged, and cleanly
isolates the 4 genuinely-degenerate models as the only caps. Before adoption:
- re-run a full floor×SE recovery sweep under the EAP stop to re-lock the operating point,
- **re-run the of-record exp-04/05/06/07/08 under the EAP stop** (and re-bootstrap SE_param on
  the EAP-administered sets — here it was reused),
- decide whether the ~+4-scenario mean length (borderline models only) is acceptable.

## Files
- `comparison.json` — full aggregate comparison.
- `per_model_comparison.csv` — per-model online vs EAP length / SE_ability / SE_total / reached.
- `figures/eap_vs_online_stop.png` — SE_ability at stop + length scatter.
