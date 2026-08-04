# ARC item-matched (fixed test length) correlation comparison

Question: if you give 2PL and 3PL the same NUMBER of items that 1PL uses, what correlation do they reach, and where does 1PL overtake them?

## Method

The Fisher-information CAT fixes item ORDER; the SE target only decides where to stop. So the correlation at exactly N items is well defined: run one CAT per held-out model, then read that model's running prediction after N administered items and correlate against its actual full-ARC accuracy across models. We reuse the exact `se_sweep.py` machinery (Fisher-information selection, EAP theta and SE, and the `pred_meanprob` predictor for all three local banks, so the predictor is identical and the comparison is fair).

- Data: in-house ARC matrix, `arc_challenge`, seed 7, n_test 13, full train pool. Same 13 held-out models and same actual accuracies for 1PL, 2PL, and 3PL.

- Banks (all 759 items for the local set): 1PL via `load_local` method=rasch, 2PL via method=girth, 3PL via `se_sweep_small_pool.fit_bank(kept, "3PL")` on the SAME kept train matrix with its usability filter, using the c-aware CAT.

- Two readouts of the same traces: `hold_at_floor` (PRIMARY) runs each CAT to the SE=0.10 bank floor and holds the last prediction if the CAT ended before N (this mirrors `se_sweep.full_cat_traces`); `forced_N` (robustness) administers exactly N items with no early stop. They are identical for every N up to a bank's SE=0.10 stop and differ only past it (mostly 2PL beyond about 100 items).

## Validation against the SE sweep

- 2PL at N=8: r = 0.8615 (SE sweep SE0.30 uses exactly 8 items, r = 0.8615). Exact match expected and observed.

- 1PL at N=43: r = 0.9723 (SE sweep SE0.30 uses 43.4 items, r = 0.976). Close; small gap is the fixed-N vs fixed-SE slicing.

- 2PL at N=71: r = 0.9662 (SE sweep SE0.12 uses 71 items, r = 0.965).

## Matched-budget table (PRIMARY, hold_at_floor; Pearson r)

| N | 1PL | 2PL | 3PL | ATLAS_3PL | reference |
|---|---|---|---|---|---|
| 20 | 0.8811 | 0.8944 | 0.9234 | 0.9270 | headline |
| 43 | 0.9723 | 0.9555 | 0.9345 | 0.9388 | headline; 1PL_SE0.30 |
| 64 | 0.9680 | 0.9636 | 0.9439 | 0.9435 | 1PL_SE0.25 |
| 71 | 0.9721 | 0.9662 | 0.9430 | 0.9434 | headline |
| 100 | 0.9776 | 0.9688 | 0.9418 | 0.9452 | headline; 1PL_SE0.20 |
| 182 | 0.9871 | 0.9686 | 0.9321 | 0.9464 | 1PL_SE0.15 |
| 304 | 0.9848 | 0.9688 | 0.9285 | 0.9470 | 1PL_SE0.12 |

N=43 is 1PL's SE0.30 operating point (its headline budget). Reference notes tag the headline set {20, 43, 71, 100} and the 1PL SE operating points (SE0.30=43, SE0.25=64, SE0.20=100, SE0.15=182, SE0.12=304 items).

## Forced-N readout (robustness; Pearson r at the same budgets)

| N | 1PL | 2PL | 3PL | ATLAS_3PL | reference |
|---|---|---|---|---|---|
| 20 | 0.8811 | 0.8944 | 0.9234 | 0.9263 | headline |
| 43 | 0.9723 | 0.9555 | 0.9345 | 0.9382 | headline; 1PL_SE0.30 |
| 64 | 0.9680 | 0.9636 | 0.9428 | 0.9470 | 1PL_SE0.25 |
| 71 | 0.9721 | 0.9662 | 0.9428 | 0.9484 | headline |
| 100 | 0.9776 | 0.9689 | 0.9466 | 0.9536 | headline; 1PL_SE0.20 |
| 182 | 0.9871 | 0.9766 | 0.9368 | 0.9620 | 1PL_SE0.15 |
| 304 | 0.9848 | 0.9803 | 0.9352 | 0.9747 | 1PL_SE0.12 |

Forcing 2PL to actually take N items (instead of holding at its SE=0.10 floor) is the only place the two readouts differ; it shows whether extra items past 2PL's floor help or not.

## Crossover (smallest N above which 1PL wins for the rest of the range)

- 1PL overtakes 2PL for good at N = 30 items (hold_at_floor); N = 30 (forced_N).
- 1PL overtakes 3PL for good at N = 30 items (hold_at_floor); N = 30 (forced_N).

## Plain-language verdict

At 1PL's own SE0.30 budget of 43 items, 1PL reaches r = 0.9723 while 2PL reaches r = 0.9555 and 3PL reaches r = 0.9345, so at a matched 43-item budget the richer models do not beat 1PL. The order flips only at very short tests: at N=20 items the c-aware and discriminating banks lead, with 3PL highest among the local banks (r = 0.9234), 2PL next (r = 0.8944), and 1PL last (r = 0.8811), because those banks spread models apart with very few items while 1PL still needs items to accumulate the same information. Once you can afford a few dozen items 1PL pulls ahead and stays ahead: it passes 2PL for good at N = 30 items and passes 3PL for good at N = 30 items, and it keeps improving with test length while the others plateau. 3PL never leads again at any larger matched budget here, and it carries two extra caveats: its absolute calibration is poor (MAE about 0.140 at N=43 versus about 0.051 for 1PL, from the c-inflated mean-prob predictor), and its rank correlation actually drifts DOWN as you add items (r = 0.9285 at N=304) because many thin-sample items are pinned at the a and c bounds and inject noise. Bottom line: at 1PL's item budget the extra 2PL and 3PL parameters do not buy accuracy, and 1PL is the better bank at every budget beyond a very short screening test of about two dozen items.

## Files

- `item_matched_corr_vs_N.csv` primary curves (bank, N, r, mae, n_eval_models).
- `item_matched_corr_vs_N_forcedN.csv` robustness curves (forced-N readout).
- `matched_budget_table.csv` r at the reference budgets (primary readout).
- `matched_budget_table_forcedN.csv` same budgets, forced-N readout.
- `crossover.csv` crossover N for each pair and readout.
- `../../../figures/arc_item_matched_corr_vs_items.png` main figure.
- `../../../figures/arc_item_matched_floor_vs_forced.png` robustness figure.

## Secondary reference: ATLAS 3PL

The ATLAS 3PL line uses the published ATLAS ARC 3PL bank on ATLAS's OWN 60 held-out models and OWN response matrix with the `pred_pirt` predictor. It is a DIFFERENT test set (different models, different response matrix, different predictor), so it is NOT strictly item-matched to the local banks and is shown only as a reference line. Its numbers should not be read as a like-for-like comparison against local 1PL/2PL/3PL.
