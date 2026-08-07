# 04 — adaptive vs random efficiency / SE-vs-length

**Status: TODO — EAP 20/0.27 regeneration pending (STOP-DEPENDENT).**

This experiment (adaptive CAT vs random-order administration; recovery-vs-length and
SE-vs-length curves) is **stop-dependent** and is NOT yet available under the locked
EAP-posterior stop at floor=20 / SE=0.27. **No figure is dropped into this slot** to avoid
implying a legacy (online-SE) result is current.

The legacy online-SE version is preserved (clearly superseded) at
`../archive_onlineSE_legacy/frq_mae/` (`frq_adaptive_vs_random.csv`,
`figures/frq_adaptive_vs_random_efficiency.png`). Under that OLD stop: adaptive reached corr
r=0.982 in ~21.9 scenarios vs random r=0.972 in ~182 scenarios — i.e. adaptive selection matches
random at ~8× fewer scenarios. The qualitative advantage will hold under the EAP stop, but the
absolute lengths/SEs must be re-measured at 20/0.27 before quoting.

## How to regenerate @20/0.27

Reuse the EAP machinery in `scripts/eap_oos_grid_study.py` / `scripts/scenario_cat_lib.py`:
build full-exhaustion **adaptive** traces (already done in the OOS grid) plus matched
**random-order** traces per model, then compare per-skill recovery r and EAP posterior marginal
SD at matched lengths and at the 20/0.27 stop. Run with `--workers 6` (the machine OOMs at the
default worker count; each worker loads the matrix + dense grid).

Deferred as TODO (per the packaging brief's escape hatch) rather than shipping unvalidated
freshly-written numbers.
