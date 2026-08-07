# 11 — order / seed stability

**Status: carried over (STOP-INDEPENDENT, legacy stop).** Path/seed dependence of the CAT
estimate across 8 seeds on the 2-skill/115 instrument. Measured under the legacy online-SE stop
(the fixed-seed deployment policy and the qualitative conclusion — variance concentrated in the
low-θ / near-all-fail tail — are stop-independent). A fresh EAP-stop re-run would tighten the
absolute spread (as it did for the sibling benches) but is not required to justify the fixed-seed
policy; see the caveat below.

## Numbers (8 seeds, MWLE)

- Across-seed θ spread (MWLE): mean SD **0.290** (corr) / **0.190** (scaff); median 0.254 / 0.146.
  Batch/online estimators are tighter (mean SD ~0.18 / ~0.14). Spread is concentrated in the
  low-θ tail.
- **Decision (locked): fixed production seed** so single-run CAT scoring is reproducible; the
  honest measurement SE is still reported via SE_total.

## Caveat (stop dependence)

These absolute SDs are under the legacy online-SE stop. Under the of-record EAP-posterior stop
the administration runs to a real posterior-SD target, which reduces order/seed variance (the
sibling WildBench/Bridge migrations halved it). The **direction of the conclusion is unchanged**;
if exact EAP-stop order/seed SDs are needed, re-run `scripts/scenario_order_experiment.py` under
the EAP stop. Marked as carried-over rather than TODO because it does not gate any headline number.

## Files

- `metrics.json` — across-seed spread (online/batch/MWLE), config (8 seeds).
- `per_model_spread.csv` — per-model θ spread across seeds.
- `figures/seed_spread_correctness.png`, `figures/seed_spread_scaffolding.png`.

## Provenance

`scripts/scenario_order_experiment.py`. Original:
`regenerated_figures/scenario_level_115/order/2_skills/`.
