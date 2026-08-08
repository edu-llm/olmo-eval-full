# Plan — the metrics and reporting layer for `uni_mcq`

What a CAT run should record, and what it records today. The gap is narrower than it
looks in one place and wider in another: the standard-error and ability-trajectory graphs
are *reconstructible* from a report already written, while the headline reduction number
cannot be computed honestly from anything in the repo. Its denominator exists — every one of
the nine values is attested somewhere — but never as a field that means only that. It is a
docstring in three cases, a manifest note in two, and in the remaining four a field whose
value happens to coincide with the right answer for reasons that do not hold everywhere.

This plan does not change any code. It names the file and the dataclass each change
belongs in, and orders the work so that everything checkable without a GPU or an AWS call
is checked first.

## What `cat_report.json` records today

Verified by reading, not by trusting the docstrings. Three layers assemble it:
`UniMcqStyle.report` builds the `metadata` dict at
[`style.py:570-611`](../../../diagnostics/mcq_cat/styles/uni_mcq/style.py), `CATReport.to_dict`
serializes the envelope at [`base.py:154-176`](../../../diagnostics/mcq_cat/base.py), and
`runner.run` bolts a `run` block on afterwards at
[`runner.py:212-220`](../../../diagnostics/mcq_cat/runner.py).

| Path | Source | Notes |
|---|---|---|
| `cat_style`, `benchmark` | `base.py:157-158` | |
| `ability.theta`, `ability.standard_error` | `base.py:160-161` | final estimate only |
| `ability.metadata` | `base.py:162` | always `{}` — `estimate_ability` never populates it (`style.py:498`) |
| `num_items_administered` | `base.py:164` | |
| `responses[]` | `base.py:165-174` | `item_id`, `chosen_index`, `correct`, `choice_logprobs`, `metadata`, **in administration order** |
| `metadata.theta`, `metadata.standard_error` | `style.py:572-573` | duplicates of `ability.*` |
| `metadata.fit_family` | `style.py:571` | from the manifest, authoritative |
| `metadata.pirt_accuracy` | `style.py:575` | |
| `metadata.observed_accuracy` | `style.py:575` | |
| `metadata.pirt_accuracy_denominator` | `style.py:576-579` | a sentence, not a number |
| `metadata.n_items_administered` | `style.py:581` | duplicates `num_items_administered` |
| `metadata.bank_size` | `style.py:581` | `len(self._order)` — the run-time selectable set |
| `metadata.stop_reason` | `style.py:582` | one of `bank_exhausted`, `max_items_reached`, `precision_reached`, `stopped_early` (`style.py:688-696`) |
| `metadata.ungradable` | `style.py:583` | `count`, `rate`, `item_ids`, `reasons`, and `alert` above 20% |
| `metadata.selected_item_ids` | `style.py:584` | ordered, and therefore load-bearing — see below |
| `metadata.cat_settings` | `style.py:585-591` | `se_threshold`, `min_items`, `max_items`, `max_items_pinned_by_style`, `max_items_is_pinned_value` |
| `metadata.bank_provenance` | `style.py:598` | the manifest subset `resolve.py:86-105` allows through |
| `metadata.modality` | `style.py:599` | |
| `metadata.scoring_note` | `style.py:600-605` | |
| `metadata.prompt_style`, `metadata.score_normalization` | `style.py:607-608` | MCQ banks only |
| `metadata.bank_caveat` | `style.py:611` | only on a bank whose spec carries one |
| `run.*` | `runner.py:213-219` | `cat_style`, `checkpoint`, `checkpoint_kind`, `checkpoint_prep`, `modality`, `grader`, `timestamp` |

More is written than is documented. The eval-cat skill's Step 3 field list
(`.cursor/skills/eval-cat/SKILL.md:123-141`) names nine of these and misses eight metadata
keys entirely — `fit_family`, `bank_size`, `selected_item_ids`, `cat_settings`, `modality`,
`prompt_style`, `score_normalization`, `bank_caveat` — two of which, `bank_size` and
`selected_item_ids`, turn out to be the two the graphs depend on. `test_style.py:309-324`
pins thirteen as required, so any addition below has to be added there too.

`bank_provenance` is richer than it sounds and carries most of the denominator argument
already: `upstream_bank_rows`, `bridge_rows`, `items`, and the full `dropped` breakdown
come through. Two things do not: `sha256`, which the manifest holds
(`vendor_bank.py:1447-1450`) and `provenance()` filters out, and any statement of how
large the benchmark's own split is.

What is missing, in order of how much it costs to add: the theta and SE trajectory; a
reduction metric of any kind; the number of items the benchmark itself would have run;
and a hash pinning the bank the report was computed against.

## The trajectory: not recorded, but exactly reconstructible

This is the central question, because both requested graphs are functions of it.

**It is not recorded.** `run_cat` re-estimates ability after every response and assigns
it over the previous value at [`cat_loop.py:68`](../../../diagnostics/mcq_cat/common/cat_loop.py);
`CATState` (`base.py:115-140`) has fields for `administered` and `ability` and none for a
history. The intermediate estimates do reach the log — `cat_loop.py:69-75` emits
`Administered %s (step %d): theta=%s se=%s` at INFO on every item — so the trajectory of
any run that has already happened is recoverable from its container logs. That is not a
plan; a log line on a node that has been reclaimed is not a metric.

**The stack we ported from does not record it, which is why we do not.** `origin/Research`
carries three CAT implementations and we inherited from the one that discards the history.
`CatResult` in `src/olmo_eval/adaptive/cat.py:49-60` carries `theta`, `se`, `order`,
`scores`, `selected_question_ids`, `pirt_accuracy`, `n_items`, `bank_version`, and
`CatSession.record` overwrites `self.theta, self.se` at `:111` on every call, exactly as ours
does. The olmo-eval reports built on it inherit the same shape: `evals/tasks/atlas.py:157-162`
emits `atlas_theta`, `atlas_se`, `atlas_n_items`, `atlas_pirt_accuracy` and `atlas_bank_version`
with no trace, and the online path at `evals/external/benchmarks/atlas/eval.py:202-223` adds
`bank_size` and per-item `{question_id, score}` predictions but still no per-step estimate.

**The other two stacks both record it, in two different shapes, both worth knowing.**
`eduLLM-Evals/tutor_cat/mcq_irt/cat.py:19-27` declares `theta_trace` and `se_trace` as fields
on its own `CATResult` and appends to both at `:62-63` — an MCQ CAT, on the same branch,
keeping exactly what ours drops. The production MIRT engine takes the sidecar route instead:
`eduLLM-Evals/tutor_cat/engine.py:13` documents `criterion_updates.jsonl` as a "per-criterion
theta/U trace" and `:256-257` writes `theta_after` and `se_after` on every update, with the
summary landing in a separate `final_result.json`.

**And where a trace was used for exactly our two figures, it was then thrown away.**
`cat_eval_tutorbench.py:143-144` builds `se_trace` and `theta_trace`, `:159-160` appends per
item, `:282-287` draws the SE reduction curve from them — and `:614-617`, under the comment
`write per-model CSV (drop bulky traces)`, filters both columns out before anything is
persisted:

```python
csv_cols = [c for c in df.columns if c not in ("se_trace", "theta_trace")]
```

The published SE curve therefore cannot be regenerated from the artifact sitting beside it.
That is the precise failure this plan exists to avoid, and it is the argument for putting the
trajectory in `cat_report.json` rather than in whatever the plotting process holds in memory.

**The reconstruction that makes this cheap.** EAP is memoryless. `estimate_ability`
conditions on the entire response pattern each time and ignores `previous` — the docstring
says so at `style.py:488-489`, and `eap_theta_se` (`irt.py:62-81`) is a pure function of
`(resp, a, b, c)` with a fixed 81-node grid and no state. So

> theta_k and SE_k for k = 1..n are exactly recomputable from the ordered
> `selected_item_ids`, the ordered `responses[].correct`, and `params.json`.

Both are already in `cat_report.json`, in order. Nothing is approximated and nothing is
re-simulated: the selection that produced the sequence is not being replayed, only the
estimator re-run on prefixes of the sequence that actually occurred. The step-0 point is
well-defined too — `estimate_ability` with no responses returns exactly `(0.0, 1.0)`
(`style.py:491-492`), which is where the SE curve should start so the fall from the prior
is visible.

This is what makes the ordering below possible: the graphs can be built and validated
before any contract change, on a report from a run that has already happened, and the
replay then serves as the parity oracle for the inline recorder.

The one thing reconstruction needs that the report does not carry is a guarantee that
`params.json` has not moved since. `bank_provenance` gives `source_commit` and
`generated_at` but not `sha256`. Arhant's own metrics files solve this the right way —
`eduLLM-Evals/regenerated_figures/2_skills/metrics.json` carries a `provenance` block with
`expected_sha256`, `actual_sha256` and `aligned: true` — and the same pattern applies here
for one extra key in `resolve.ResolvedBank.provenance`.

## The denominator

Three candidates were proposed. Two of them are the same number, and the third is not
recorded anywhere machine-readable.

The claim is already being made, which is the reason to get this right rather than to get
it done. `.cursor/skills/eval-cat/SKILL.md:11` tells every agent that a run administers
"13-40 items instead of the benchmark's full 1,000-5,000" — a reduction claim, in prose,
with no field behind it and no per-dataset number anywhere for a reader to check it
against.

**Candidates (b) and (c) coincide.** The `a > 0` filter and the bridge join are applied
*offline, at vendoring time* — `vendor_bank.py` writes only surviving rows, and the
manifest's accounting identity (kept items plus every drop reason equals
`upstream_bank_rows`) is asserted by a test, per `calibrated_datasets/README.md:401-402`.
So `items.jsonl` line count, `len(params.json)`, `manifest.items`, and the run-time
`len(self._order)` that becomes `metadata.bank_size` are one number. Confirmed equal on
all nine banks. `_build_arrays` (`style.py:320-328`) re-intersects at run time and warns
if they ever diverge, which is the only way `bank_size` could differ from `manifest.items`.

**Candidate (a) is not a field.** `vendor_bank.load_task_items` enumerates
`get_task(...).instances` (`:847`) and knows the total — `len(items) + ungradable` at
`:860` — and the manifest it writes at `:1417-1452` does not include it. The number
survives only as prose in `manifest.notes`, and only for some banks.

| Dataset | Full split, as the task enumerates it | Where that number is written | Bank (`items.jsonl` = `params.json` = `bank_size`) | Bank as share of split |
|---|---|---|---|---|
| `arc_challenge` | 1,172 | `bridge_rows`; corroborated by README:109 | 650 | 55.5% |
| `hellaswag` | 10,042 (9,609 distinct ids) | `bridge_rows`; and pinned as data in `hf-converter-patch/aggregate_results.py:20` | 5,044 | 50.2% |
| `musr` | 756 (250 + 256 + 250) | `src/olmo_eval/evals/tasks/musr.py:5-7`; `notes` agrees | 432 | 57.1% |
| `bbh` | 5,761, over the 24 MCQ subtasks the spec declares | `notes`, which calls the per-subtask counts "true full sizes upstream rather than truncations" (`datasets.py:1384-1388`); the three generative subtasks are absent from bank and task module alike (`:1388-1391`) | 3,965 | 68.8% |
| `ifeval` | 541 | `src/olmo_eval/evals/tasks/ifeval.py:3`, `:20`, `:54` | 511 | 94.5% |
| `leaderboard_math` | 1,324 | `notes`: subtask counts "summing to 1,324"; `datasets.py:862-863` | 1,183 | 89.4% |
| `gpqa` | 1,192 instances over 546 distinct questions | `src/olmo_eval/evals/tasks/gpqa.py:8-10`; nesting argued at README:158-163 | 395 | 33.1% of instances |
| `winogrande` (blocked) | 1,267 | `notes`: "Bank max index, bridge rows and enumerated instances all agree at 1,267" | 865 | 68.3% |
| `gsm8k` (blocked) | 1,319 | `notes`: same sentence, at 1,319 | 1,298 | 98.4% |

Every one of these nine numbers is written down somewhere in the tree, and no two are
written down in the same *kind* of place: four in `bridge_rows`, three in a task module's
docstring, two only in a manifest note. That is the case for phase 3 in one sentence.

`bridge_rows` looks like it could stand in for the split, and the reason to refuse it is
not that it is wildly wrong. It equals the split by construction for the four Route C
banks, whose bridge maps the whole evaluation order. For the five Route B banks
`bridge_rows == upstream_bank_rows`, because their bridge *is* the bank's own index map —
and those harvests came from near-complete leaderboard evaluations, so it lands close
anyway: 754 against MuSR's 756, 535 against IFEval's 541, exact on GPQA and BBH, and
1,206 against `leaderboard_math`'s 1,324.

That last one is the whole argument. A denominator that means "the split" on four banks,
"the harvest" on five, is silently 9% short on one of them, and carries nothing in the
report to say which reading applies is not a denominator — it is a coincidence accurate
enough to be trusted and wrong without warning. `full_split_items` has to be a field that
means one thing.

**Upstream never had to make this choice, which is why there is no precedent to copy.**
Arhant's published reductions are computed as `mean_CAT_items / bank_size` and stated in
prose rather than recorded in any field — "21.1 of 1,172", "under 2% of items" for ATLAS ARC,
"roughly 30x" for FRQ TutorBench, "~80x" for TutorEval scenarios. There is no `items_saved`
key anywhere in the artifacts. And in the ATLAS ARC case the two candidate denominators
*coincide*: 1,172 is both the published bank size and the full split, because no `a > 0`
filter stood between them at report time. Ours does — 650 of 839 calibrated of 1,172 — so the
question that the precedent never had to answer is forced here, and copying the formula would
silently pick the bank while quoting a number that reads like the split.

**Which denominator is honest.** Neither alone. The composed reduction — administered
items over the full split — is what a user means by "CAT saved you this much", and
reporting it bare attributes to adaptivity a factor that is mostly not adaptivity. On
HellaSwag, 24 of 10,042 is 99.8%, and half of that 10,042→24 collapse is the bank simply
not having 4,998 of those items: they failed the `a > 0` filter, and no amount of adaptive
selection was ever going to administer them. Calibration attrition is a *coverage loss*
being counted as a *saving*.

So the report should carry the decomposition rather than a single ratio, and let the
headline be the composed number with its two factors beside it:

```
full split  ──(bank coverage)──▶  calibrated bank  ──(adaptive stopping)──▶  administered
  10,042            50.2%              5,044               99.5%                  24
```

The second arrow is the CAT's contribution and the only one that changes when the stopping
rule changes. The first is a property of the bank and is constant across every run of that
dataset. Proposed field, a new `metadata.reduction` block:

```json
"reduction": {
  "items_administered": 24,
  "bank_items": 5044,
  "full_split_items": 10042,
  "full_split_source": "manifest.task_instances",
  "reduction_vs_bank": 0.99524,
  "reduction_vs_full_split": 0.99761,
  "bank_coverage_of_split": 0.50229,
  "summary": "24 items administered of the 5,044 in the calibrated bank (99.5% fewer). The bank covers 5,044 of the 10,042 instances the hellaswag task enumerates, so 24 of 10,042 overall (99.8% fewer) — the first factor is the adaptive stopping, the second is the bank's calibration coverage and not a saving."
}
```

Two rules on that block. `full_split_items` is `null`, and the three fields that depend on
it are omitted rather than defaulted, when the manifest does not record it — a run must
never substitute `bridge_rows` and call it a split. And `full_split_source` names where
the number came from, because the four Route C banks will read it from a field that agrees
with `bridge_rows` and the Route B banks will not, and a reader has to be able to tell
which without opening the manifest.

`pirt_accuracy_denominator` already exists as a sentence saying the same thing about a
different quantity (`style.py:576-579`), and stays. It answers "what is predicted accuracy
over?" and the reduction block answers "how much less did we run?"; conflating them is how
a predicted-accuracy figure ends up captioned as a coverage claim.

## Where the plotting happens

**Matplotlib is not in the runtime image, and neither is `scipy`.** `pyproject.toml:70-74`
puts `matplotlib`, `seaborn` and `scipy` together in an optional extra named `analysis`, and
lists `matplotlib.**` and `scipy.**` under the mypy ignore-missing-imports overrides at
`:240-241`, which is what a dependency nobody expects to be installed looks like. The extra
is reachable only through the `dev` group (`:147`, via `olmo-eval[storage,beaker,hf,analysis]`),
and `.edullm/Dockerfile:54-55` syncs with `--no-default-groups`, which drops `dev` by name,
then adds `--extra beaker --extra hf --extra s3 --extra clients` and later `--extra olmo-core`
at `:143-145`. `analysis` appears nowhere. The repo's `matplotlib` mentions are all outside
this pipeline: two are strings injected into sandboxed solution code
(`evals/tasks/bigcodebench.py:129`, `evals/tasks/ds1000.py:71-72`) and one is a `scipy.stats`
import in `src/olmo_eval/analysis/eval_power.py:9`. Nothing under `diagnostics/` imports
either.

`numpy` and `pandas` *are* present — numpy as a base dependency (`pyproject.toml:27`) and
pandas transitively through `datasets` — which matters twice. It is why the trajectory costs
nothing to record at run time, since `irt.py` already runs on numpy alone. And it is the line
the offline tooling should stay on the far side of: an aggregation step may use pandas freely,
and anything reaching for `scipy` is as unavailable on the node as a plot is.

Adding it would mean a new extra in a GPU image whose size is already argued over line by
line in that Dockerfile — `:115-121` weighs 4.54 GiB of vLLM against 141 packages of
olmo-core — to render two PNGs per run.

**The stronger argument is that a single-run plot is the wrong artifact anyway.** Every
figure in `hf-converter-patch/plot_curves.py` is a *cross-checkpoint* figure: it loads
`results/{full,split}/accuracy_consolidated.csv`, intersects the step sets so both arms are
sampled at identical points (`:67`, and the rationale in the module docstring), and plots
accuracy against training step. Arhant's SE curve is the same shape one level up — a mean
over 115 models' `se_trace` at each item index, drawn only where at least five models are
still administering (`cat_eval_tutorbench.py:282-287`). Neither of those can be produced
by the process that generates one run's data. Aggregation is inherently a step that runs
after the runs.

So: **record the trajectory as data in `cat_report.json`; plot offline.** The plotter is a
script under `diagnostics/mcq_cat/styles/uni_mcq/scripts/`, beside `vendor_bank.py` and
`check_bridge_alignment.py`, which are already the precedent for "offline tooling that is
never imported at run time" (README:189-191, and `vendor_bank.py`'s own note that it is
never imported at run time). It reads one or more `cat_report.json`, needs the `analysis`
extra, and runs on a laptop.

Conventions to inherit rather than reinvent, all from `plot_curves.py`: `matplotlib.use("Agg")`
before importing pyplot (`:15-18`), `dpi=150` on save (`:124`), the item count in the
figure title (`:113`, `f"{title}  ·  {n_items:,} items"`), a dashed reference line for the
threshold, and a printed consistency check at the end that says in words whether the series
are comparable (`:137-145`). From `cat_eval_tutorbench.py`: x-axis "items administered",
y-axis "mean posterior SE (logits)", and an `axhline` at the SE target (`:288-291`).

## Arhant's metrics, in three buckets

The distinction that governs everything below: a *recovery* metric needs something to
recover, and for a live checkpoint there is no ground-truth theta. A calibration study has
one because it holds a response matrix over a population of models and can fit the whole
bank for each; a single run of an unknown checkpoint does not, and cannot be made to have
one except by administering the whole bank, which is bucket (ii).

**"pIRT" names two different quantities upstream and they must not be merged.** What we
ported is the ATLAS *blend* (`cat.py:131-150` → `pirt.py`): observed 0/1 on administered
items, model-predicted probability on the rest, weighted by the fraction seen, denominator
the bank. TutorBench's `pred_acc_cat` is not that — it is the mean predicted success
probability at theta over **every observed cell of the response matrix**, and `pirt_mae_cat`
is its absolute error against `obs_acc` computed over the same full set
(`cat_eval_tutorbench.py:354-404`). The first is computable from one run because the unseen
items contribute a prediction rather than an observation; the second needs the full observed
matrix, which for a live checkpoint means a full-bank run. Same three letters, different
bucket. Any port of the TutorBench figures has to carry its definition with it.

### (i) Computable from one live CAT run — add these now

| Metric | What it measures | Precedent |
|---|---|---|
| `n_items` / `cat_n_items` | test length | `cat.py:59`; we have it as `num_items_administered` |
| `se_trace` | posterior SE after each item | `cat_eval_tutorbench.py:143,159` |
| `theta_trace` | posterior mean after each item | same, `:144,160` |
| `n_cross_<dim>` | index of the first item at which SE crosses the target, or null if it never does | `cat_eval_tutorbench.py:174-176`; the x-axis of `cat_length_hist.png` |
| `stop_reason` / `precision_reached` | why the session ended | `tutor_cat/engine.py:174-194`; we already have it |
| `pirt_accuracy` | blended predicted accuracy over the bank | `cat.py:131-150`, ported verbatim to `pirt.py` |
| `obs_acc` | observed accuracy over administered items | `cat_eval_tutorbench.py:297`; we have it as `observed_accuracy` |
| bank health: median non-zero `a`, count of items loading | whether the bank can discriminate at all | `cat_metrics.json` → `bank.a_correctness_median_nonzero`, `a_correctness_n_loading` |
| `n_items_fit` / `n_items_total` | calibration coverage | `cat_metrics.json` → `bank`; ours is `manifest.items` vs `upstream_bank_rows` |

`n_cross` is the one genuinely new single-run metric worth adding beyond the trajectory
itself, and it is nearly free once the trajectory exists. It is not the same as
`num_items_administered`: a session that hits `min_items` before SE ever crossed, or one
that crossed at item 6 and kept going to the floor of 8, differ in a way `stop_reason`
does not express. On BBH the gap should be the largest of any bank: its floor is 24 and
README:288-292 describes a posterior that collapses at once on a median `a` of 3.99, so the
crossing will land well short of the stop. That distance is the honest measure of how much
of BBH's test length is precision and how much is the subtask-coverage floor argued for at
`style.py:170-178`, and it is currently unmeasured.

The `stop_reason` row is the one place the current pipeline is *ahead* of what it was ported
from. `CatResult` has no stop reason at all — stopping is implicit in `next_item()` returning
`None` at `adaptive/cat.py:92-98` — while the production engine names three of them,
`precision_reached`, `max_scenarios_reached` and `bank_exhausted` (`engine.py:175-194`). Our
vocabulary is the engine's with the scenario noun swapped for items, plus `stopped_early` for
a case neither upstream expresses. The fleet-level `stop_reasons` histogram in bucket (iii) is
then just a count over runs we can already emit, which makes it the cheapest bucket (iii)
metric on the list.

Bank health is worth stamping into the report even though it is a property of the bank and
not the run, for the reason `bank_caveat` is: BBH's median `a` of 3.99 is why its floor is
24, and a reader comparing two datasets' SE curves without that number will conclude BBH
converges faster when what it does is collapse.

### (ii) Needs a full-bank reference run on the same checkpoint

| Metric | What it measures | Precedent |
|---|---|---|
| `theta_full` | ability from administering every bank item | `cat_eval_tutorbench.py:242` (`theta_full_{dim}`) |
| `abs(theta_cat - theta_full)` / `ability_l2_err` | this checkpoint's CAT error, one number | the per-model term inside `theta_mae`; `kfold_cat_pilot.py` names it `ability_l2_err` |
| `obs_acc` | true accuracy over the whole bank | `cat_eval_tutorbench.py:297` |
| `pred_acc_full`, and `pirt_mae_full` vs `pirt_mae_cat` | how much of the predicted-accuracy error is the CAT and how much is the IRT model | `cat_metrics.json:in_sample`; note the definition above, not ours |
| `pred_score_abs_err` | `abs(p_cat − p_full)` for one model | `kfold_cat_pilot.py` |
| SE at the CAT's stopping point vs SE at full bank | whether the reported interval is honest | implied by `se_reduction_curve.png` read to its right edge |
| items to reach the full-bank theta within a tolerance, vs random selection | whether Fisher selection is doing the work | ATLAS replication reports Fisher reaching the same SE in 1.6×-5.6× fewer items than random |

These are single-model quantities. The published versions are aggregates over a population
and belong in bucket (iii); the per-model term is computable here and is the only thing
that turns the reduction claim from a count into a validated claim.

The last row is the one worth flagging as optional-but-cheap. A random-selection arm over
the same bank and the same checkpoint is a second run, not a population, so it is bucket (ii)
— and it answers a question the full-bank run does not: whether the 40-item estimate is good
because the bank is easy to estimate from or because Fisher selection chose well. Upstream
quotes 1.6×-5.6× as a comparative sweep; one arm on one checkpoint would not reproduce that
number, but it would say which end of the range this bank sits at.

### (iii) Needs many checkpoints or many models — do not promise these

| Metric | Why it is out of reach | Precedent |
|---|---|---|
| `recovery_r` | a correlation needs a population to correlate over; `n = 115` in the published run | `cat_metrics.json:in_sample.recovery_r`, plotted as `recovery_scatter_<dim>.png` |
| `theta_mae` | a mean absolute error over models; the 0.684 quoted for BBH at README:293 is a calibration-study number | README:293 |
| `pirt_mae_cat` / `pirt_mae_full` as published | means over 82-115 models | `regenerated_figures/*/metrics.json` |
| `pirt_calibration.png` | scatter of predicted against actual across models | `cat_eval_tutorbench.py:295-300` |
| `cat_length_hist.png` | a histogram over models; one run is one bar | `:259-274` |
| `se_reduction_curve.png` **as published** | mean SE across models, gated on ≥5 models still administering | `:276-293` |
| k-fold / held-out recovery | needs a bank recalibrated without each model; `n = 1,102` in GPQA's | `gpqa` manifest notes |
| `bank_health` cross-fold `a` correlation | needs the calibration folds | `cat_metrics.json:bank.bank_health` |
| estimator comparison (EAP vs MWLE vs batch) | needs the population and a second estimator | `regenerated_figures/estimator_comparison/` |
| floor sweep | one CAT run per floor per model | `regenerated_figures/floor_sweep/2skill_floor{10,11,...}` |

The SE graph asked for here is a **bucket (i)** figure that happens to share a filename
with a bucket (iii) one. Ours is a single trace for one checkpoint; Arhant's is a mean over
115 models. Both are legitimate and they are not the same figure, and the plotter should
title them differently so nobody reads one as the other.

## Making the reduction claim defensible

Reporting "24 of 10,042" is a count. It becomes a claim only when someone shows that the
24-item estimate is the same estimate the 5,044-item one would have given. That requires
one full-bank reference run per (checkpoint, dataset), and it is the only route available
without a population of models.

The run is not a new code path. Set `--max-items` to at least the bank size and
`--se-threshold 0` and the session runs to `bank_exhausted` (`style.py:690-691`), with
`max_items_is_pinned_value: false` already recorded so it can never be mistaken for a
normal run. Its `metadata.theta` is `theta_full`. What comes out is `abs(theta_cat -
theta_full)` for that checkpoint, the two SEs, and the difference in `pirt_accuracy` — and
a full-bank `se_trace`, which is the *only* way to see where on the SE curve the stopping
rule actually landed rather than assuming it landed where the threshold says.

The cost is the honest objection. It is `bank_size / n_administered` times a normal run,
and over the 13-40 item range the README quotes that is 16-50× on ARC's 650, 126-388× on
HellaSwag's 5,044, and 99-165× on BBH's 3,965 (whose floor of 24 puts it at the low end).
On the MCQ banks that is a forward pass per choice with no decoding and is plausibly
affordable once per dataset on one checkpoint. On the three generative banks it is not — those decode
up to `max_new_tokens` sequentially at batch size 1 and cannot be batched
(`HF_CONVERSION.md:104-108`), so a 395-item GPQA reference run is a different order of
spend from a 13-item session.

The proportionate version: **one full-bank reference run, on `arc_challenge`, on whichever
checkpoint Phase 7 of `HF_CONVERSION.md` uses.** ARC is the smallest MCQ bank at 650 items
and the only one whose bridge is independently attested by a shipped release
(`HF_CONVERSION.md:511-513`), so a discrepancy there is the pipeline rather than the join.
One number from one dataset does not validate the other six, and the plan should say so in
the report rather than let a single `theta_error` field imply a general result. Anything
more than that is a sweep and belongs after this plan, not inside it.

## Phases

Ordered so nothing needs a card until phase 7. Phases 0 through 2 run here with nothing
installed. Phase 3 needs `olmo_eval` and reach to HuggingFace but no GPU and no AWS. Phase
5 can be built against the stubbed-scorer report the CLI smoke test already writes
(README:339) and only becomes interesting once `HF_CONVERSION.md`'s phase 7 has produced a
real one; phase 6 needs several of those and is therefore the last free phase in practice
even though it is free in principle.

| # | Phase | Needs | Verified by |
|---|---|---|---|
| 0 | Pin the report schema | nothing | a test that fails on any unplanned key |
| 1 | Offline trajectory replay | nothing | replay's last point equals the report's final theta |
| 2 | The reduction block | nothing | per-bank unit test; `null` where the split is unknown |
| 3 | Record the split size at vendoring | `olmo_eval` + HF network, CPU | all nine manifests carry `task_instances` |
| 4 | Record the trajectory in the run | nothing | replay parity test against phase 1 |
| 5 | The offline plotter | one report | two PNGs from a committed fixture |
| 6 | Cross-checkpoint aggregation | several reports | one figure over ≥3 checkpoints |
| 7 | The full-bank reference run | GPU spend | `theta_error` recorded for `arc_challenge` |

---

### Phase 0 — Pin the report schema

**Purpose.** Make the current shape of `cat_report.json` a thing a test knows, so every
addition below is deliberate. Today thirteen keys are asserted present and nothing asserts
that no others appear.

`test_style.py:309-324` checks membership one key at a time. Replace the loop with an
equality assertion against the full expected key set, at both the top level and inside
`metadata`, with the two conditional keys (`prompt_style`/`score_normalization` on MCQ
banks, `bank_caveat` where the spec has one) handled explicitly rather than by omission.

The value is not the coverage, it is the failure mode. A key added to `report()` without a
corresponding decision here is currently invisible; after this, it is a red test naming
the key. Every phase below adds keys.

While here, record the two duplications rather than removing them. `metadata.theta` and
`metadata.n_items_administered` restate `ability.theta` and `num_items_administered`, and
the eval-cat skill's field list reads the metadata copies. Removing them is a separate
decision from adding anything, and doing both at once means a broken skill looks like a
metrics bug.

**Done when** the schema test fails if a key is added to `report()` and not to the test.

---

### Phase 1 — Offline trajectory replay

**Purpose.** Produce both requested graphs' underlying data from reports that already
exist, without changing the runtime. This also builds the oracle that phase 4's inline
recorder will be checked against.

New script `diagnostics/mcq_cat/styles/uni_mcq/scripts/replay_trajectory.py`. Input: a
`cat_report.json`. Output: the per-item sequence, as JSON on stdout or to a file.

The whole method is four lines of `numpy` around existing code. Read
`metadata.selected_item_ids` and the aligned `responses[].correct`, resolve the bank
through `resolve.resolve(report["benchmark"])`, load `params.json` through the same
`load_irt_params` the style uses, and call `irt.eap_theta_se` on each prefix. Prepend the
step-0 point `(0.0, 1.0)`, which is what `estimate_ability` returns on an empty response
list (`style.py:491-492`).

Two guards, both cheap and both load-bearing. Compare the replay's final `(theta, se)`
against `ability.theta` and `ability.standard_error` and fail on any disagreement beyond
floating-point noise — that is the assertion that the reconstruction is exact rather than
approximately right. And compare the bank's `sha256` against `bank_provenance` when it is
present, so a replay against a re-vendored bank is an error rather than a quiet
disagreement. It will not be present until phase 2 adds it; until then the script warns.

Use Arhant's names for the output arrays, `theta_trace` and `se_trace`
(`cat_eval_tutorbench.py:143-144`), so his plotting code ports without renaming.

**Done when** replaying a report reproduces its final theta and standard error to within
1e-9, on the report the existing CLI smoke test writes against the 650-item ARC bank
(README:339).

---

### Phase 2 — The reduction block

**Purpose.** Add the headline "CAT saved you this much" number, with a denominator that can
be defended. Get the honest structure in place before the missing input arrives, so phase 3
fills a field rather than reshaping a block.

In `UniMcqStyle.report` (`style.py:570-611`), a `metadata.reduction` dict with the fields
listed above. `items_administered` and `bank_items` are `state.step` and
`len(self._order)`, both already in hand. `full_split_items` reads a new manifest key
through `bank_provenance`; until phase 3 writes it, every bank reports `null` and the three
dependent fields are absent.

Also in this phase, and small: add `sha256` to the tuple in
`resolve.ResolvedBank.provenance` (`resolve.py:86-104`). It is one string in the keys
tuple. It is what lets phase 1's replay assert it is reading the bank the run read, and
what lets a figure regenerated in six months say so.

The `summary` string is not decoration. Every other number in this report that a reader
could misread carries a sentence — `pirt_accuracy_denominator`, `scoring_note`,
`ungradable.alert` — and a bare `reduction_vs_full_split: 0.998` is the single most
misreadable number the pipeline will emit. It should be a sentence that names both factors
in the same breath as the headline.

**Done when** a unit test over all nine vendored banks asserts the identity
`bank_coverage_of_split × (1 − reduction_vs_bank) = 1 − reduction_vs_full_split` wherever
`full_split_items` is non-null, and asserts the three dependent fields are absent wherever
it is null.

---

### Phase 3 — Record the split size at vendoring

**Purpose.** Turn the one denominator that lives in prose into a field. It is already
computed and then discarded, so this is a line in a dict rather than new work.

`load_task_items` returns `(items, ungradable)` and knows the enumeration total as
`len(items) + ungradable` (`vendor_bank.py:860`); `load_multi_task_items` sums the same
across subtasks. Add `task_instances` and `task_instances_distinct` to the manifest at
`vendor_bank.py:1417-1452`, next to `upstream_bank_rows` and `bridge_rows`, and to the
`provenance()` tuple in `resolve.py`.

Both counts, not one, because for HellaSwag they differ and the difference is documented:
the validation split concatenates two sub-splits whose native `ind` values each restart, so
10,042 instances answer to 9,609 ids (`vendor_bank.py:804-810`). A single field would have
to pick, and a reduction reported against 9,609 for a benchmark everyone quotes at 10,042
is a number nobody can check.

This means re-running the vendoring per dataset. It needs `olmo_eval` and network access to
HuggingFace but no GPU and no AWS, and `--dry-run` already logs the counts without writing
(`vendor_bank.py:1547-1551`), so the numbers can be read and checked against the table
above before anything is regenerated. Two banks need a `DataSource` override the notes
record — `allenai/winogrande` and `openai/gsm8k`, both because the bare repo id is no
longer accepted by the Hub — and both are blocked datasets, so they can go last or not at
all.

Regenerating a manifest changes its `sha256` block and its `generated_at`, and phase 2 just
made `sha256` something a report carries. Re-vendor all nine in one pass rather than
piecemeal, so no two banks in a comparison disagree about which vendoring they came from.

Once the field exists, pin the expected values as data and check them, rather than trusting
whatever the enumeration returned on the day. `hf-converter-patch/aggregate_results.py:15-23`
is the precedent: an `EXPECTED_INSTANCES` dict — `"hellaswag": 10042` at `:20` — that
`validate` compares every run's `num_instances` against at `:38-49`, marking the row `INVALID`
rather than plotting it. The equivalent here is the table in this document, committed beside
the allowlist, and a vendoring that enumerates a different count failing instead of writing a
new denominator. A split size that silently moves is exactly the failure the bridges were
rebuilt to prevent, one level up.

**Done when** all nine manifests carry `task_instances`, every value matches the table in this
document, and the three sourced only from a task-module docstring (`ifeval` 541, `musr` 756,
`gpqa` 1,192) plus BBH's 5,761 have been confirmed by the enumeration rather than by prose. A
disagreement is a finding, not a bug in the enumeration: correct the table here and say which
way it moved.

---

### Phase 4 — Record the trajectory in the run

**Purpose.** Stop depending on reconstruction. The replay is exact for this style because
EAP is memoryless, and that is a property of `uni_mcq`'s estimator rather than of the
contract — a style with a sequential estimator would make it silently wrong.

This is the one change that touches the frozen scaffolding, in three places:

- `base.py`: a `TrajectoryPoint` dataclass — `step: int`, `item_id: str | None`,
  `correct: bool | None`, `theta: Ability`, `standard_error: Ability` — with `item_id` and
  `correct` nullable so the step-0 prior point has a home. `theta` typed as `Ability`
  (`base.py:24`) rather than `float`, so a MIRT style gets the same field for free.
- `base.py`: `trajectory: list[TrajectoryPoint]` on `CATState` (`:115-140`) and
  `trajectory: tuple[TrajectoryPoint, ...]` on `CATReport` (`:143-152`), serialized in
  `to_dict` (`:154-176`).
- `cat_loop.py`: append one point after the estimate at `:68`, and seed the step-0 point
  when the state is constructed at `:48-52`.

`style.py:613-620` then passes `tuple(state.trajectory)` into the report, and its length is
checked against `state.step + 1` — a trajectory that has drifted from the response list is
worse than none, because it plots.

**The frozen-scaffolding rule this violates is already void.** README:381-385 says never to
edit `base.py`, `runner.py` or `common/*`, and commit `164ad8f0` put changes into
`base.py` (16 lines), `runner.py` (45) and six files under `common/` in the same commit
that wrote that README. `HF_CONVERSION.md:607-610` already schedules the correction and
notes that its own plan touches three of those files again. This is the fourth. The
correct move is the one that file names: raise the contract change on `CheckpointFlows`
so it lands once for all styles, rather than working around it.

Two alternatives, both rejected, both for the record.

`AbilityEstimate.metadata` (`base.py:101`) is an existing dict that `to_dict` already
serializes (`:162`) and that nothing ever populates, so a style could stash the trajectory
there with no contract change at all. It would appear under `ability.metadata.trajectory` —
the history of the estimate, filed as metadata about its last point — and every future style
would have to rediscover the convention. Available if the contract cannot be reopened in
time; not the design.

A **sidecar file** is the other, and it is what the production MIRT engine does:
`tutor_cat/engine.py:158` opens `criterion_updates.jsonl` and `:256-257` streams
`theta_after`/`se_after` per update, with the summary going to a separate `final_result.json`.
That shape earns its keep where the trace is long and streamed — a scenario CAT emits one
record per criterion, many per item. Ours is 13-40 records of five scalars, which fits
comfortably in the report that already carries a `responses` array of the same length, and a
second S3 object is a second thing to find, keep together, and lose. Revisit if a future
style administers enough items that the report stops being readable.

The naming is worth settling here rather than by accident. `tutor_cat/mcq_irt/cat.py:26-27`
uses `theta_trace`/`se_trace`; our records-of-structs form keeps `item_id` and `correct`
aligned to each point, which two parallel arrays cannot guarantee. Emit the struct form as
canonical and let the plotter derive the two arrays, so the precedent's names survive where
they are used without the failure mode they permit.

**Done when** the inline trajectory and phase 1's replay agree point-for-point on the same
run, and the schema test from phase 0 has been updated to expect the new key.

---

### Phase 5 — The offline plotter

**Purpose.** Emit the two requested figures. One script, reading reports, writing PNGs,
running nowhere near a GPU.

`diagnostics/mcq_cat/styles/uni_mcq/scripts/plot_cat_report.py`, needing the `analysis`
extra (`pyproject.toml:70-74`) that the runtime image deliberately does not have. It takes
one or more `cat_report.json` paths and an output directory, and falls back to phase 1's
replay for any report predating phase 4 — which, since phase 4 has to wait on
`CheckpointFlows`, is how the first figures will actually be produced.

Two figures per report:

`se_curve.png` — standard error against items administered, starting at the step-0 prior
point so the fall from SE = 1 is visible, with a dashed `axhline` at `cat_settings.se_threshold`
and a marker where the trace first crosses it. Title carries the dataset, the checkpoint,
and the bank size in `plot_curves.py:113`'s format. If the run stopped on `min_items`
rather than precision, the crossing marker and the endpoint are at different x, and that
distance is the most informative thing on the figure.

`theta_trajectory.png` — theta against items administered, with a ±1 SE band from the same
trace, and a rug or color along the x-axis marking which items were answered correctly. The
band is what stops the figure being read as a converging point estimate: theta wandering
inside a shrinking envelope is the expected picture, and without the band a two-logit swing
at item 3 looks like instability rather than a wide prior.

Print a consistency summary at the end in the style of `plot_curves.py:137-145`, naming the
dataset, the bank sha256, whether the trajectory was recorded or replayed, and — once phase
2 lands — the reduction line. A figure whose provenance is only in the filename gets
separated from it.

**Done when** both PNGs are produced from a committed fixture report and the script refuses
a report whose bank sha256 does not match the bank on disk.

---

### Phase 6 — Cross-checkpoint aggregation

**Purpose.** Produce the figure that is actually useful: how a checkpoint's ability moves
over training. A single run's theta is uninterpretable in isolation — the scoring note says
so in every report — and only a series makes it a measurement.

`aggregate_cat_reports.py` beside the plotter, following the two-stage shape
`hf-converter-patch` uses throughout: `aggregate_results.py:70-98` walks a results tree and
writes one `accuracy_consolidated.csv`, and both `plot_curves.py:64-65` and
`compare_arms.py:74-75` read that CSV rather than the raw results. The separation is what
let `plot_curves.py` intersect two arms' step grids before plotting (`:67`) instead of
comparing series sampled at different points, and the same hazard exists here the moment
two checkpoints have been run on different subsets of the seven datasets.

One row per (checkpoint, dataset, run): training step parsed from the checkpoint URI,
theta, SE, `pirt_accuracy`, `observed_accuracy`, `n_items_administered`, `n_cross`,
`stop_reason`, `bank_size`, `full_split_items`, the reduction figures, `ungradable.rate`,
and the bank sha256. Keep `aggregate_results.py`'s `status` and `notes` columns
(`:94`): the plotter skips any row not marked `ok` (`plot_curves.py:41`), which is how a
reference run, an `ungradable`-alerted run, or a half-synced result gets excluded at plot
time by a rule a reader can see rather than by the aggregator dropping it silently. Then
`theta_vs_step.png` per dataset with SE error bars, and a grid across datasets in
`plot_curves.py`'s 3×3 layout (`:80`).

Refuse to plot two rows whose bank sha256 differs, and refuse to plot theta across
datasets on shared axes. The second is the rule the eval-cat skill already states in
words — never compare theta across benchmarks — and a plotting script is where a stated
rule becomes an enforced one.

**Done when** one figure covers at least three checkpoints of the same run on the same
dataset, and the aggregator refuses a mixed-bank input.

---

### Phase 7 — The full-bank reference run

**Purpose.** Turn the reduction count into a validated claim, once, on the cheapest bank.
Every metric above describes what the CAT did; this is the only one that says whether it
was right.

`arc_challenge`, 650 items, on the checkpoint `HF_CONVERSION.md`'s phase 7 uses, invoked
with `--max-items 650 --se-threshold 0` so the session runs to `bank_exhausted`. No code
change: `max_items_is_pinned_value: false` already lands in the report (`style.py:590`) and
is exactly the marker that keeps a reference run from being aggregated as a normal one —
phase 6's aggregator should skip any row carrying it.

Record `theta_full`, `se_full`, `pirt_accuracy_full` and the full-bank `se_trace` into a
sibling file rather than into the adaptive run's report, and add a `reference_run` block to
the adaptive report pointing at it with `theta_error = abs(theta_cat − theta_full)`. Two
files because they are two runs; one pointer because a `theta_error` with no way back to
the run that produced it is a number that will be quoted out of context within a week.

State the scope in the report text, not just here. One dataset, one checkpoint, one seed's
worth of evidence. It says whether the ARC pipeline recovers ARC's full-bank theta; it says
nothing about BBH, whose `theta_mae` is flat at 0.684 whatever the stopping rule
(README:293), and nothing about the three generative banks, whose reference runs are
unaffordable for the reason recorded above.

**Done when** `theta_error` for `arc_challenge` is written into this file, alongside the
wall-clock ratio between the reference run and the adaptive one — the second number being
what decides whether this is ever worth doing on a second dataset.

## What else has to change

Folded here rather than edited in place, since `HF_CONVERSION.md` and the branch README are
being worked on elsewhere.

`HF_CONVERSION.md` phase 7's **Done when** currently accepts a `cat_report.json` with "a
theta, a standard error at or under 0.3, and a `bank_provenance` block" (`:519-521`). Once
phases 2 and 4 land, that acceptance criterion should also require the `reduction` block
and a trajectory whose length is `n_items_administered + 1`; otherwise the first live run
produces a report that cannot be plotted and nobody notices until someone tries.

`HF_CONVERSION.md` phase 8 says the eval-cat skill's "Step 3 report-field list is still
accurate and should survive intact" (`:574-576`). It will not survive intact — it gains
`reduction` and `trajectory`, and the guidance around it should say that the trajectory is
data for the offline plotter rather than something to read in the terminal. The instruction
never to compare theta across benchmarks stays and gets a second home in phase 6's
aggregator.

That skill's opening line needs a correction of its own. `SKILL.md:11` says a run
administers 13-40 items "instead of the benchmark's full 1,000-5,000", and the table above
puts the actual range at 756 for MuSR to 10,042 for HellaSwag — wrong at both ends, and
low at the top by a factor of two on the largest bank. Once `reduction` exists the sentence
should quote nothing and point at the field.

`HF_CONVERSION.md` phase 9 records that the README's merge-conflict discipline was
overtaken by `164ad8f0` (`:607-610`). Phase 4 above is a fourth touch of `base.py` and a
second of `cat_loop.py`, which is the first change to that file at all; whoever writes the
correction should describe the rule as "changes to the contract go through
`CheckpointFlows`" rather than as a prohibition nobody has observed.

The branch README needs two corrections this plan tripped over, both independent of it.
`README:56` says the admission criteria leave "eight" datasets and the table beneath it
lists nine, which `README:77` then calls nine. And `README:61-62` says "Nothing carries
[a `blocked` reason] today", while `datasets.py:683` and `:763` both do — `winogrande` and
`gsm8k` — which is why seven of nine are runnable and why the two blocked banks are
excluded from phase 3's re-vendoring.

## What is unverified

All nine split sizes in the denominator table are attested somewhere in the tree, but "in the
tree" is doing a lot of work: three come from a task module's prose docstring
(`ifeval.py:3`, `musr.py:5-7`, `gpqa.py:8-10`), and a docstring is not a value any code reads.
Phase 3 does not discover these numbers so much as promote them from comments to data, and the
enumeration it runs is what actually confirms them. If a `full_split_items` disagrees with this
table, trust the enumeration and correct the table.

BBH is the one whose provenance is thinnest: 5,761 is recorded as the count of evaluated
documents whose position a fresh enumeration reproduces, and the argument that this equals the
enumeration total is the manifest note calling the per-subtask counts "true full sizes upstream
rather than truncations" (`datasets.py:1384-1388`). That reads correct and is not the same as
having been counted.

No live `cat_report.json` exists on this branch to check the inventory table against — the
only reports written so far come from the CLI smoke run with the scorer stubbed
(README:339). The table is read off the code that writes it, and the first real run may
surface a key this document does not list.

The claim that phase 1's replay is exact rests on `estimate_ability` being pure and on the
bank being unchanged. The first is verified by reading `irt.py:62-81` and the second is
what phase 2's `sha256` addition exists to enforce; until that lands, a replay against a
re-vendored bank fails silently, which is the reason the guard is in phase 2 rather than
later.
