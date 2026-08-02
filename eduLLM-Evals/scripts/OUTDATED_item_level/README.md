# OUTDATED - item-level CAT harnesses (superseded)

**Do not use these for new results.** Every script in this folder selects the next test
item at the **individual-criterion** level. The production CAT engine
(`tutor_cat/engine.py` + `tutor_cat/selector.py`) selects at the **scenario** level: it
commits to a whole scenario and grades all of its criteria at once. An online CAT can never
pick a single criterion in isolation, so results produced by these scripts do not reflect
how the real system behaves.

This mattered in practice. A few conclusions from these scripts were **artifacts of the
item-level granularity** and were overturned once the same analyses were run through the real
scenario-level engine:

- D-optimality "roughly halves test length" -> at the scenario level it does **not** shorten
  tests (it is slightly longer), and for the 3-skill bank it actually **lowers** precision.
- "Presentation is starved by the selector" -> at the scenario level presentation rides along
  with ~99% of scenarios; it is slow only because each scenario carries ~1 presentation
  criterion, not because selection skips it.
- The online estimator "swings up to 12 logits with item order" -> at the scenario level the
  seed-to-seed spread is ~0.2 logits and online is comparable to batch, because the dominant
  variation is which scenarios are sampled, not sequential item order.

## Replacements (use these instead)

| Outdated (here) | Scenario-level replacement (in `scripts/`) |
|---|---|
| `cat_selection_experiment.py` | `scenario_selection_experiment.py` |
| `random_order_experiment.py` | `scenario_order_experiment.py` |
| `kfold_estimator_cv.py` | `scenario_kfold_estimator_cv.py` |
| `leaderboard_param_uncertainty.py` | `scenario_param_uncertainty.py` |

All replacements drive the real engine via `scripts/scenario_cat_lib.py` and parallelise
across models. Results live under `regenerated_figures/scenario_level/`.

## Notes

- These files were left runnable (their `ROOT` was repointed to `parents[2]` for the new
  depth) purely for historical reference; they still import shared math from
  `scripts/regen_cat_figures.py`, `scripts/kfold_cv_mirt.py`, and `scripts/calibrate_mirt.py`,
  which remain in `scripts/`.
- `regen_cat_figures.py` was intentionally **not** moved: it is teammate-owned and is also a
  math source (dense-grid EAP / MWLE), but its item-level CAT *results* are likewise
  superseded by the scenario-level runs.
