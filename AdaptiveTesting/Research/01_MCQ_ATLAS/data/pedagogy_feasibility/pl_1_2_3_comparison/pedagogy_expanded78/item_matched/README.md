# Item-matched (fixed test length) 1PL vs 2PL vs 3PL on pedagogy

The SE-stopped comparison in `../run_pl_expanded78.py` asks "how good is each bank at its own
SE operating point", where every bank spends a different number of items (1PL SE0.3 takes
about 43 items, 2PL SE0.3 takes about 12). That is not an apples-to-apples budget. This
folder answers the budget-matched question instead:

> If you give 2PL and 3PL the same number of items that 1PL uses, what correlation do they
> reach, and where (if anywhere) does 2PL or 3PL overtake 1PL and stay above it?

## Method

In the Fisher-information CAT the item order is fixed and the SE target only decides where to
stop, so the correlation at exactly `N` items is a read-off, not a new run. For each held-out
model we take one full CAT trace (`se_sweep.full_cat_traces` with `pred_meanprob`), read the
running prediction after `N` administered items (holding the last prediction if that model's
CAT ended before `N`), and compute Pearson r and MAE against actual full-bank pedagogy
accuracy across the held-out models. Sweeping `N` over the integer grid from `MIN_ITEMS=8` to
the kept bank size gives one correlation-vs-length curve per bank. The same c-aware CAT is
used for all three banks so the comparison is fair.

- Data: expanded 78-model pedagogy matrix (`../../expanded_calibration/_mcq_data`, 920 items)
  via `tutor_cat.mcq_irt.matrix.load_benchmark`.
- Partition (CONTROLLED, shared with `run_expanded_calibration` / `run_pl_expanded78`):
  seed-7 split of the 52 OLD models gives 40 OLD-train and 12 OLD held-out. The primary
  calibration pool is `calib66` (those 40 plus the 26 newly scored models); the eval set is
  the fixed 12 OLD held-out models. The 40-model pool (`calib40`) is also computed for
  continuity with the original 52-model / 40-train baseline, on the identical 12-model eval
  set.
- Banks (reused fitters): 1PL girth `rasch_mml`, 2PL girth `twopl_mml`, 3PL
  `se_sweep_small_pool.fit_3pl_mml`, all via `se_sweep_small_pool.fit_bank` with the uniform
  usability filter (finite, a>0, 0<=c<1). Item filter `filter_items` on the train slice. Bank
  sizes: 366 items at 66 models, 318 items at 40 models (identical across model types).

## Validation (passed)

Reading each bank's trace with the SE-stopping rule reproduces the controlled SE-stopped
numbers already reported in `../pl_1_2_3_expanded_controlled.csv` (hard gate on 1PL/2PL,
tolerance 5e-3; 3PL reported as a soft check because its thin-pool EM fit is only reproducible
within one environment):

| pool | bank | SE | recomputed | reference |
|---|---|---|---|---|
| 66 | 1PL | 0.30 | 0.7340 | 0.7340 |
| 66 | 2PL | 0.30 | 0.9049 | 0.9049 |
| 66 | 1PL | 0.15 | 0.9255 | 0.9255 |
| 66 | 2PL | 0.15 | 0.9580 | 0.9580 |
| 40 | 1PL | 0.30 | 0.8903 | 0.8903 |
| 40 | 2PL | 0.30 | 0.7163 | 0.7163 |
| 66 | 3PL | 0.30 | 0.5067 | 0.5067 (soft) |
| 40 | 3PL | 0.30 | 0.8195 | 0.8195 (soft) |

The fixed-length read-offs also line up with the SE-stopped operating points, as required:
1PL at N=43 gives r=0.736 (SE0.3 mean 43.2 items, r=0.734) and 2PL at N=12 gives r=0.904
(SE0.3 mean 12.2 items, r=0.905).

## Matched-budget correlation (66-model pool, 12 held-out)

r of the CAT prediction after a fixed number of administered items:

| fixed length N | 1PL | 2PL | 3PL |
|---:|---:|---:|---:|
| 8 | 0.583 | 0.856 | 0.247 |
| 12 | 0.420 | 0.904 | 0.390 |
| 20 | 0.491 | 0.903 | 0.483 |
| 43 (1PL SE0.3 budget) | 0.736 | 0.919 | 0.528 |
| 100 | 0.884 | 0.944 | 0.669 |
| 288 (1PL SE0.15 budget) | 0.923 | 0.949 | 0.840 |
| 366 (full bank) | 0.944 | 0.958 | 0.840 |

At the budget 1PL needs to hit SE0.3 (about 43 items), 2PL already reaches r=0.919 while 1PL
is only at r=0.736. At a genuinely short test of 12 items, 2PL is at r=0.904 while 1PL has
collapsed to r=0.420. The full per-N curves are in `pedagogy_item_matched_corr_vs_N.csv`, and
the reference-point table is in `pedagogy_matched_budget_table.csv`.

## Crossover

66-model pool (primary):

- 2PL vs 1PL: no crossover. 2PL is at or above 1PL at every fixed length from 8 to 366
  (closest approach +0.009 near the full bank). 2PL never needs to "catch up".
- 3PL vs 1PL: never. 3PL stays below 1PL at every length; its best is r=0.840 (near the full
  bank) against a 1PL best of r=0.944.

40-model pool (secondary, the original baseline regime):

- 2PL vs 1PL: 2PL leads only at very short tests (N=8 to 30). 1PL overtakes 2PL at N=31 and
  stays above all the way to the full bank. At the 1PL SE0.3 budget (about 43 items) 1PL wins
  0.880 to 0.787.
- 3PL vs 1PL: never stays above. 3PL is highest at N=8 (0.834) then collapses as the unstable
  thin-pool fit adds noisy items.

## Verdict (plain language)

On pedagogy, the answer to "give 2PL the same number of items 1PL uses" depends entirely on
how many models the bank was calibrated on. With the expanded 66-model pool, 2PL wins the
budget-matched comparison outright: at 1PL's own 43-item SE0.3 budget 2PL reaches r=0.919
versus 1PL's r=0.736, and 2PL sits at or above 1PL at every test length from 8 items up, so
there is no crossover to wait for. The extra 26 calibration models sharpen the 2PL
discriminations enough that the second parameter pays for itself immediately, letting the CAT
front-load the most informative items and reach a strong correlation in a fraction of the
questions. 3PL never becomes competitive at this pool size and stays well below 1PL at every
budget.

## Contrast with ARC

On ARC (prior item-matched analysis), 1PL wins at its own item budget: matched on items, the
Rasch model recovers accuracy at least as well as 2PL, so the discrimination parameter does
not pay off at a fixed test length. Pedagogy reproduces that same ARC pattern on the small
40-model pool (1PL overtakes 2PL at N=31 and leads 0.880 to 0.787 at the 43-item budget), but
the pattern flips on the expanded 66-model pool, where 2PL is at or above 1PL at every budget
and beats it decisively at short tests. The item-matched winner is therefore a property of the
calibration pool size, not a fixed fact about the IRT model.

## Files

- `run_item_matched.py` produces everything here plus the figure. It touches nothing outside
  this folder and the shared figures directory.
- `pedagogy_item_matched_corr_vs_N.csv` (bank, N, r, mae, n_eval_models): 66-pool curves.
- `pedagogy_matched_budget_table.csv` (bank, reference_label, reference_N, r, mae): r at
  N in {12,20,43,100} plus the 1PL SE0.3 and SE0.15 operating points, 66-pool.
- `pedagogy_item_matched_corr_vs_N_pool40.csv`: 40-pool curves (secondary, continuity).
- `pedagogy_item_matched_summary.csv`: per (pool, bank) bank size, SE-stopped operating
  points, r at the reference lengths, and the 2PL/3PL-vs-1PL crossover verdict.
- Figure: `../../../figures/pedagogy_item_matched_corr_vs_items.png`.

## Reproduce

CPU-local, no network:

```bash
PYTHONPATH=eduLLM-Evals uv run python AdaptiveTesting/Research/01_MCQ_ATLAS/data/pedagogy_feasibility/pl_1_2_3_comparison/pedagogy_expanded78/item_matched/run_item_matched.py
```

Runtime is dominated by the two girth `twopl_mml` fits (about 50 seconds each); 1PL and the
3PL EM fit in a few seconds.
