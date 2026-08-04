# ATLAS accuracy-vs-ability examples (OpenLM)

Concrete cases where two models have **near-identical full-bank accuracy** but a
**meaningful gap in IRT ability** (full-bank EAP theta), plus a held-out test that
the ability gap is real signal rather than calibration noise.

## Method

- Banks: frozen ATLAS-style 3PL `irt_item_parameters_combined.csv` per benchmark;
  items with `a1 <= 0` dropped (`load_bank`), `(d, g)` -> `(a, b=-d/a, c=g)`.
- Response matrix: train + test rows stacked over the shared a>0 item columns.
- Per model: full-bank accuracy = mean correctness; ability = full-bank EAP theta
  (`eap_se`, Gaussian prior, 81 nodes) -- identical conventions to
  `atlas_error_plots_openlm.py`.
- Degenerate models removed: accuracy at/below the bank guessing floor `mean(c)`,
  accuracy >= 0.995, or near-constant response vectors (minority count < 5).

## Files

- `<bench>_examples.csv` -- curated illustrative pairs. Candidates are pairs with
  |accuracy diff| <= 0.005, both models in the central 10-90 percentile of
  accuracy (well-identified region), a theta gap exceeding ~2x its combined EAP
  posterior SD (`z_gap >= 2`, i.e. the gap is not measurement noise), and the
  mechanism holding (the higher-theta model's correct answers sit on more
  discriminating and/or harder items). Pairs are ranked by |theta gap| with
  distinct models. `model_a` is the higher-theta model, so `theta_gap > 0`.
  `heldoutB_gap` = model_a - model_b accuracy on a random held-out half (seed 7)
  of the a>0 items -- reported as-is, never selected on; positive means the
  higher-ability model also scores higher out of sample. `mean_disc_correct_*` /
  `mean_diff_correct_*` are the mean discrimination (a) / difficulty (b) of each
  model's correctly-answered items; `mechanism_confirmed` is True when the
  higher-theta model's correct items are more discriminating and/or harder.
- `validation_summary.csv` -- per-benchmark held-out predictiveness of theta.

## Held-out validation (Pass 2)

Split the a>0 bank into halves A/B (random, seed 7). Among **all** pairs with
near-equal **half-A** accuracy (|diff| <= 0.005; and the exact-tie subset),
the higher half-A-theta model is predicted to score higher on the untouched half
B. Half A sets both the accuracy match and the ability predictor; half B is never
seen, so this is a genuine train/test split of items (no filtering on outcome).
`frac_higher_theta_wins` is that fraction among decided (non-tied) pairs
(chance = 0.50); `mean_heldoutB_gap` is the mean half-B accuracy advantage of the
higher-theta model. Values > 0.50 and > 0 mean ability predicts held-out
performance at fixed accuracy.

### Summary

`_tol` = pairs within |accuracy diff| <= acc_tol; `_exact` = exactly-equal
accuracy; `_midband` = the |diff|<=tol pairs restricted to the central 10-90
percentile of half-A accuracy (where theta is best identified).

| benchmark | n_models | n_items | guess_floor | frac_higher_theta_wins_tol | mean_heldoutB_gap_tol | frac_higher_theta_wins_exact | mean_heldoutB_gap_exact | frac_higher_theta_wins_midband | mean_heldoutB_gap_midband |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ifeval | 1095 | 511 | 0.0520 | 0.6130 | 0.0105 | 0.6036 | 0.0096 | 0.6235 | 0.0114 |
| math | 768 | 1183 | 0.0027 | 0.5187 | 0.0015 | 0.5208 | 0.0012 | 0.5543 | 0.0022 |
| gpqa | 1102 | 579 | 0.0712 | 0.5092 | 0.0006 | 0.5140 | 0.0007 | 0.5023 | 0.0003 |
| musr | 1099 | 432 | 0.0442 | 0.6093 | 0.0123 | 0.6042 | 0.0121 | 0.6068 | 0.0125 |
| bbh | 1102 | 3965 | 0.0645 | 0.5427 | 0.0014 | 0.5487 | 0.0012 | 0.5329 | 0.0011 |

### Findings

- **ifeval** (STRONG): higher-theta model wins held-out half B in 61.3% of equal-accuracy pairs (mean held-out gap +0.0105).
- **math** (WEAK/ABSENT): higher-theta model wins held-out half B in 51.9% of equal-accuracy pairs (mean held-out gap +0.0015).
- **gpqa** (WEAK/ABSENT): higher-theta model wins held-out half B in 50.9% of equal-accuracy pairs (mean held-out gap +0.0006).
- **musr** (STRONG): higher-theta model wins held-out half B in 60.9% of equal-accuracy pairs (mean held-out gap +0.0123).
- **bbh** (MODEST): higher-theta model wins held-out half B in 54.3% of equal-accuracy pairs (mean held-out gap +0.0014).

Caveats: on `bbh` the bank is very large (~4k a>0 items) so the fixed 81-node EAP
grid resolves theta only to ~0.1 steps; its near-equal-accuracy theta gaps are
coarse and its held-out effect is modest. On `math` and `gpqa` most models sit near
the accuracy/guessing floor, so theta gaps at fixed accuracy carry little held-out
signal (gpqa is at chance; its largest in-sample gaps even reverse out of sample --
the reason the held-out test matters).
