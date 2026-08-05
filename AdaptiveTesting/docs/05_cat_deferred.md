# CAT deferred: what was built, what was learned, what to pick up

**Status: on hold.** The client scrapped adaptive testing in favour of running full
benchmarks and reporting accuracy. This document exists so that revisiting CAT is a
read rather than a re-investigation.

Nothing in `src/olmo_eval/adaptive/`, the vendored banks, or the `atlas_*` tasks and
evals was removed. What was removed is one skill,
`.cursor/skills/run-checkpoint-evals/`, which orchestrated a per-checkpoint CAT
sweep. That directory was never tracked in git, so this document is the only record
of what it knew. Its replacement is a pair of skills that run full benchmarks and
report accuracy: `.cursor/skills/eval-direct-gpu/` on a GPU box you control, and
`.cursor/skills/eval-platform/` as an AWS Batch job through the eduLLM platform.

Read [`04_implementation_and_phase3_handoff.md`](04_implementation_and_phase3_handoff.md)
first for what the integration is. This doc covers what was learned after it.

## 1. The decision that shapes everything: only HellaSwag was viable

A CAT selects items by Fisher information from a calibrated IRT item bank, and a bank
is two CSVs under `Inputs/ATLAS/<subdir>/` that `olmo_eval.adaptive.bank.load_bank`
both requires:

```
irt_item_parameters_combined.csv   # a1, d, g per item, linked onto one scale
atlas_idx_to_question_id.csv       # bank index -> the task's metadata["id"]
```

Item parameters cannot be derived from a benchmark. They are fit from how *many other
models* answered each item. Of the seven benchmarks the client asked for, exactly one
had a bank.

| Benchmark | Registered in `benchmarks.py` | Bank on disk | Binary per-item score | What is missing |
|---|---|---|---|---|
| `hellaswag` | yes | **yes** | yes | nothing |
| `piqa` | yes | no | yes | calibration only |
| `csqa` | yes | no | yes | calibration only |
| `arc_easy` | no | no | yes | registration + calibration |
| `socialiqa` | no | no | yes | registration + calibration |
| `naturalqs` | no | no | **no** | switch to `naturalqs:mc`, then both |
| `jeopardy` | no | no | **no** | switch to `jeopardy:mc`, then both |

Two traps in that table are worth stating outright.

`piqa` and `csqa` are the closest to viable and also the most dangerous. They are
registered with a `bank_subdir` that does not exist on disk, so `--evals atlas_piqa`
passes CLI validation and then fails at run time with "ATLAS bank unavailable". Under
the old skill's failure semantics that sank the entire run.

`arc_easy` cannot borrow the `arc/` bank. ATLAS calibrated ARC-**Challenge**; the two
splits share no items, so the id join would find zero overlap.

`naturalqs` and `jeopardy` are F1-scored, and IRT needs a dichotomous 0/1 cell. F1 is
continuous partial credit, so these cannot feed a CAT in their default form regardless
of whether a bank exists. Their `:mc` variants are log-likelihood scored with native
ids and are the variants to calibrate if these benchmarks matter.

## 2. Why no upstream bank exists for the other six

Every public large-scale IRT calibration derives from **Open LLM Leaderboard**
response matrices. v1 covered ARC-Challenge, HellaSwag, MMLU, TruthfulQA, WinoGrande
and GSM8K; v2 covered IFEval, BBH, MATH, GPQA, MuSR and MMLU-Pro. PIQA, CSQA,
SocialIQA, NaturalQS and Jeopardy were never leaderboard tasks, so nobody has
assembled a thousands-of-models response matrix for them.

Verified against three sources: the upstream ATLAS GitHub tree
(`Peiyu-Georgia-Li/ATLAS`, full recursive listing, 464 entries) contains
`irt_item_parameters_combined.csv` for exactly five benchmarks and zero references to
any of the six; the paper states it directly in section 3.2 ("The item pool spans five
benchmarks: ARC, GSM8K, HellaSwag, TruthfulQA, and WinoGrande"); and HuggingFace hosts
no ATLAS item banks. tinyBenchmarks and MetaBench cover the same leaderboard set.

This is not a gap ATLAS happened to leave. The raw material has never been published.

## 3. The 2PL decision

The project switched from 3PL to 2PL. The registry now resolves `hellaswag` to
`hellaswag_2pl`, which moves the offline task and the online eval together.
`tests/adaptive/test_benchmarks.py` pins that decision and asserts the sibling banks
were not swapped. Docs [02](02_atlas_and_adaptive_testing.md) and
[04](04_implementation_and_phase3_handoff.md) carry supersession notes above their
original 3PL rationale.

**The evidence supports the switch independently of anyone's preference.** The 3PL
HellaSwag bank's fitted `g` has a p95 of 0.834 on a four-choice benchmark where chance
is 0.25. A lower asymptote that high is not modelling guessing; it is the third
parameter absorbing misfit. That is almost certainly why upstream published 2PL refits
at `experiments/*_2pl/` for all five of its benchmarks.

**The runtime needed no change, and this was verified rather than assumed.** Upstream's
2PL CSV carries the identical `X,a1,d,g,u` header with `g` present and exactly `0.0` in
all 5,595 rows, which is precisely the case `load_bank` already handles since
`np.clip(0.0, 0.0, 0.999)` is a no-op. `prob()` is bit-identical to a hand-written
logistic at `c = 0`; `fisher_info` differs by a uniform `2e-9` relative scale factor
that cannot change an `argmax`; EAP differs by about `1e-16`. If a future `mirt` export
ever omits the `g` column, the minimal fix is `g = float(row.get("g") or 0.0)` in
`load_bank`, which also covers empty cells.

### Do not zero out `g` on a 3PL bank

Someone will propose reusing the 3PL parameters with `c` forced to zero instead of
using a real 2PL fit. It is not an approximation; it is a different and worse model,
because `a1` and `d` were estimated *jointly* with `g`.

Measured on responses generated from true 3PL item behaviour, 40 items, 400
replications:

| true theta | correct 3PL params | `g` forced to 0 |
|---|---|---|
| −2.0 | bias +0.392, RMSE 0.564 | bias **+1.584**, RMSE 1.649 |
| −1.0 | bias +0.120, RMSE 0.420 | bias +0.915, RMSE 1.010 |
| 0.0 | bias −0.017, RMSE 0.340 | bias +0.418, RMSE 0.548 |

The +0.39 at theta −2 under correct scoring is ordinary EAP shrinkage toward the
N(0,1) prior. The shortcut adds roughly **1.2 logits of spurious bias on top of it**,
and the direction confirms the mechanism: with `c = 0` the model asserts a low-ability
taker essentially cannot answer correctly, so every correct answer it does observe must
mean higher ability. Weak models get systematically flattered — which is exactly the
ability region an early training checkpoint occupies.

Per-item, one representative case: item 9204 has 3PL `(a, b, c) = (4.53, −0.80, 0.845)`
giving `P(correct | theta = −2) = 0.846`. Force `c = 0` and it becomes 0.004, a ~1,300x
odds change. The honest 2PL refit instead moves `b` to −1.94, giving 0.460.

## 4. The HellaSwag 2PL bank, measured

All figures reproduce `load_bank`'s filter exactly: keep an item when discrimination is
positive and finite and its index is in the bridge, then take `a = a1`, `b = -d/a1`,
`c = g`.

Of **5,595** calibrated rows, **590** are dropped for non-positive discrimination,
leaving **5,005** selectable. The bridge covers every survivor, so nothing is lost to a
failed join. The 3PL bank yields 5,044 for comparison; the 2PL index set is a strict
subset, with five indices (9843, 9844, 9846, 9848, 9849) present only in 3PL.

### The well-behaved criterion, stated

A percentage without its threshold is unreproducible, which is how the old "roughly
63%" figure became impossible to check. Here it means

```
0.3 <= a <= 4   and   |b| <= 6
```

On that criterion **3,750 of 5,005 (74.9%)** are well behaved. The 3PL bank scores
**3,668 of 5,044 (72.7%)** on the *identical* criterion — that is the honest
comparison, and the banks are close. The old 63% additionally required `c <= 0.5`,
which is vacuous under 2PL, so it is not like-for-like.

The items that `c` filter used to catch have not vanished. Under 3PL about 1,002
near-always-correct items appeared as `c > 0.5`; under 2PL the same easiness is
expressed as a very negative `b`.

The remaining 1,255 are fitting artifacts, failing at least one clause and overlapping
rather than partitioning: **943** with discrimination above 4 (effectively step
functions), **163** below 0.3 (nearly uninformative), and **284** with `|b| > 6`, where
a near-zero `a` sends `b = -d/a` to absurd values (range −448 to +957). These are
largely self-limiting, since Fisher information drives selection and such items either
carry none or carry it in a band so narrow the CAT must already be at the right theta.

### The bank skews easy, and that caps late-training usefulness

Median difficulty over all 5,005 usable items is **−2.42**, quartiles −3.04 and −1.25.
Do not quote a standard deviation over the whole bank: the `|b| > 6` artifacts make it
23.8, which describes the artifacts. Among well-behaved items, **2,117** sit below −2,
**1,559** in the −2..+2 band, and only **74** above +2.

Taking the 40 most informative items at a fixed theta and summing their Fisher
information gives an optimistic floor on the standard error a 40-item test could reach.
Optimistic because it assumes the CAT already knows where to look and ignores the prior:

| theta | SE floor, 40 items (2PL) | (3PL, reference) |
|---|---|---|
| −2 | 0.03 | 0.11 |
| −1 | 0.08 | 0.08 |
| 0 | 0.10 | 0.06 |
| +1 | 0.12 | 0.04 |
| +2 | **0.16** | 0.12 |
| +3 | **0.27** | 0.22 |

At theta >= +2 the bank is thin enough that the 40 most informative items are all
well-behaved ones, with no high-discrimination artifacts left to lean on, and the floor
at +3 sits just inside the default `--se-stop 0.3`. Since the floor is optimistic, a
converged checkpoint out there will in practice exhaust `--max-items` without reaching
the threshold.

**This is the concrete reason the diagnostic was sharpest early and mid-training.** It
is not a tuning problem; it is the item bank running out of hard items.

### Theta is anchored, and not portable across banks

Theta is anchored to ATLAS's calibration population (~3,467 leaderboard models), so it
is comparable across checkpoints of one run and against those models, but it is not an
absolute score.

**Thetas from the 2PL bank are not comparable to any theta previously recorded against
the 3PL bank.** Different calibrations, not a rescaling. On the 4,503 items usable in
both with `a >= 0.3` and `|b| <= 6`, difficulty holds up in rank (Pearson r = 0.84) but
its scale compresses (sd 2.08 -> 1.49, median −2.70 -> −2.39). Discrimination is
effectively re-estimated: Pearson r = 0.02, Spearman 0.31. **Discrimination is what
moves item selection**, since Fisher information scales with `a` squared. Rank each
bank by Fisher information and the top-100 sets overlap **62/100 at theta −1**, 46/100
at 0, and **43/100 at +1**. Re-baseline rather than splicing old and new numbers into
one curve.

`ItemBank.version` records the provenance
(`hellaswag_2pl:irt_item_parameters_combined.csv:n5005`) and surfaces as
`atlas_bank_version` offline and `bank_version` online. Use it to tell results apart
after the fact.

## 5. Calibrating a new bank: the path and the real blocker

The procedure works and is proven in this repo — the `ifeval` and `math` banks came
from exactly this route via `Experiments/openlm_atlas_3pl/`.

1. **Collect responses.** Score many models on the full benchmark, reduced to a
   models x items 0/1 matrix. `Test/Inference/` writes per-model MCQ CSVs;
   `Experiments/openlm_gpqa_atlas_3pl/prepare_matrix.py` is a working template for
   stacking them into the wide format the R fitter wants. Score items **exactly** the
   way olmo-eval will at eval time or the parameters will not transfer.
2. **Fit in chunks.** `Inputs/ATLAS/scripts/01_fit_irt.r` (R + `mirt`), one call per
   chunk of >= 100 items. `01_fit_irt_custom.r` is the variant for a locally assembled
   matrix. Both default to 3PL, so select 2PL explicitly.
3. **Link chunks.** `02_link_chunks_custom.r` mean-sigma links onto one scale and emits
   `irt_item_parameters_combined.csv`. Chunk parameters are on different scales until
   this runs; never use unlinked output.
4. **Build the id bridge.** `scripts/build_atlas_idx_bridge.py` enumerates the
   olmo-eval base task and writes `atlas_idx_to_question_id.csv`. It reads
   `bank_subdir` from the registry, so it follows a registry swap automatically, and it
   aborts if the task's instance count does not match the bank's max index — the guard
   against a silently misaligned positional join. Both HellaSwag banks max out at
   10,042, the split size, so regenerating is safe.
5. **Register** in `src/olmo_eval/adaptive/benchmarks.py`: name, `base_task`,
   `bank_subdir`, `scoring`, `positional_id`.

**The blocker is statistical, not engineering.** Upstream ATLAS used 3,467-4,680 models
per benchmark:

| ATLAS benchmark | Models | Items |
|---|---|---|
| ARC | 4,162 | 842 |
| GSM8K | 4,195 | 1,307 |
| HellaSwag | 3,467 | 5,608 |
| TruthfulQA | 4,635 | 628 |
| WinoGrande | 4,680 | 1,046 |

The local roster is 100 models (`Inputs/Models/models.yaml`), or 200 via
`Experiments/models_200/`. Fitting is cheap once you have a matrix; collecting the
matrix is not, and response matrices are not committed here. `01_fit_irt_custom.r` also
drops constant columns, so at n=100 many items are all-correct or all-wrong and vanish
entirely.

2PL materially eases this. It estimates two parameters per item instead of three, and
the parameter it drops — the guessing asymptote — is the least identifiable of the
three at small n. Any revisit should reassess feasibility under 2PL rather than
inheriting the 3PL pessimism.

### The positional-id trap

Benchmarks whose base task emits a positional index as `metadata["id"]` (`winogrande`,
`piqa`, `gsm8k`, `socialiqa`) join the bank by split position, so the bridge must be
generated against the exact ordering olmo-eval enumerates. If the ordering shifts the
join finds no overlap and the adaptive report is skipped rather than scoring the wrong
items — a degradation, not a corruption. HellaSwag is immune, joining on the native
`ind`, as do `csqa` and the `arc_*` tasks.

### Scoring-parity caveat on the existing banks

The `ifeval` and `math` banks were calibrated on Open LLM Leaderboard v2 responses
under lm-eval-harness scoring, not olmo-eval's, and `gsm8k` carries the same caveat
from ATLAS's own scoring. This affects the absolute scale rather than whether the
pipeline runs, and for tracking relative progress across checkpoints of one run a
consistent bias matters far less than it would for cross-model comparison. HellaSwag is
the least exposed of the set: MCQ log-likelihood, native id join.

## 6. respgen and the CAT are two halves of a pipeline never connected

Worth recording because it is the single most likely thing to be rediscovered.

`eduLLM-Evals/tutor_cat/respgen/` (entry point `tutor-cat generate`) is an 8-way
data-parallel vLLM fleet that runs 100-200 open-weight models over the tutor
benchmarks. It emits **free text**, not a binary matrix; a separate judging pass
produces `runs/judge/<Benchmark>/response_matrix.csv` with `{0,1,NaN}` cells, which
feeds `calibrate_mirt.py` (multidimensional M2PL) and drives tutor_cat's own CAT — a
separate implementation from olmo-eval's.

`AdaptiveTesting/Test/Inference/` **is a port of respgen**, module for module, and says
so in its headers (`frq_generate.py` "Ported from eduLLM-Evals `respgen/runner.py`";
`engine.py` annotated "respgen parity" in four places). The port adds an MCQ
log-likelihood scorer the original never had, writing
`Outputs/mcq/<bench>/<model_slug>.csv` with `question_id,model,benchmark,predicted,gold,result`
— which is the binary grid calibration needs, in the id space `load_bank` expects. A
real row from a prior sweep:

```
Mercury_7175875,01-ai/Yi-6B-Chat,arc_challenge,C,C,correct,loglikelihood
```

`Mercury_7175875` is exactly the ARC id `bank.py` names as the join key. The repo
already holds 63 models x arc_challenge, 63 x arc_easy, 63 x openbookqa and 62 x sciq
in that shape.

**No bank on disk traces back to this.** Every vendored bank came from upstream ATLAS
or an Open LLM Leaderboard v2 scrape. The handoff artifact exists, in the right format,
and has never been walked across — the only thing between it and a bank is step 1 to 4
above, plus enough models.

There are also **three separate IRT implementations** in this workspace, which is worth
knowing before adding a fourth: ATLAS 3PL (R/`mirt` fitting, `olmo_eval/adaptive/irt.py`
inference), tutor_cat's multidimensional M2PL (`tutor_cat/mirt.py`), and
`tutor_cat/mcq_irt/` doing 2PL via `girth`, whose docstring says item filtering follows
the ATLAS rules and which reads the `Outputs/mcq/` schema directly. That third one
already runs calibrate -> EAP -> CAT -> theta recovery end to end on this repo's own
arc/openbookqa/sciq data with k-fold validation.

## 7. What the deleted skill knew

`.cursor/skills/run-checkpoint-evals/` was verified working before removal. If a CAT
sweep is rebuilt, these were the non-obvious parts, all of which the replacement skill
`.cursor/skills/eval-direct-gpu/` also needs and therefore carries. They are all sweep
concerns, which is why they live there and not in the single-checkpoint platform skill:

- **S3 checkpoint enumeration does not exist in olmo-eval.** Discovery was
  `aws s3 ls` filtered to `PRE` lines. `data/backends/s3.py` has listing code but it is
  for eval datasets, not checkpoints.
- **Order checkpoints by trailing step integer, anchored to the final path segment.**
  Matching anywhere in the path makes `s3://` parse as step 3. Strip `\r` too, or
  Python's CRLF output on a Windows host corrupts every path.
- **vLLM cannot load an `s3://` path.** olmo-eval has no S3 download on that code path,
  so weights must be synced locally first. Passing an `s3://` URI to `-m` fails inside
  the model loader rather than at argument parsing.
- **Native OLMo-core checkpoints need conversion** via `$OLMO_CORE_CONVERT` pointing at
  OLMo-core's `convert_checkpoint_to_hf.py`. Detection is the presence of
  `model_and_optim/` or `.metadata`.
- **Isolate per checkpoint.** The predecessor `add-cat-evals` skill aborted the whole
  run when any single eval failed, before the S3 sync and before writing `_READY`,
  leaving a poller waiting forever on a prefix that would never appear. Every checkpoint
  must end with `_READY` or `_FAILED`.
- **Validate numeric flags up front.** A non-numeric `--se-stop` reached a
  `float()` call in the provenance writer *after* the eval and before the marker, so it
  burned GPU time and then stranded the result.
- **The two runners both write `metrics.json`**, so an adaptive group and a full group
  must write to separate output directories or one clobbers the other.
- **Online eval results are `<eval_name>_results.json`**, not `<eval_name>.json`.
  `ExternalEval._save_results` writes `f"{self.name}_results.json"`. Both
  `add-cat-evals/SKILL.md` and the `atlas_cat_diagnose` README documented the wrong
  name.
- **A failed eval writes no results file at all.** `AtlasExternalEval.execute` returns
  `_error_result` without calling `_save_results`, so a summariser keyed on result files
  silently drops failed checkpoints. Emit a placeholder row instead.
- **`DEFAULT_MAX_ITEMS` is 200** in `adaptive/cat.py`; the skill overrode it to 40. Runs
  are not comparable across different caps unless the flag is passed explicitly.

## 8. If you pick this back up

The cheap path, in order:

1. Decide whether CAT is worth it given section 4 — the bank cannot resolve a converged
   checkpoint, which is arguably when the client cares most.
2. If yes, `hellaswag` works today with no calibration work. Five more benchmarks
   (`arc_challenge`, `winogrande`, `gsm8k`, `ifeval`, `math`) have banks and need only
   resolver changes, subject to the parity caveat in section 5.
3. For the client's actual list, calibration is unavoidable. Start with `piqa` and
   `csqa`, which are already registered, and reassess sample-size feasibility under 2PL
   before committing compute.
4. The offline `atlas_*` tasks are the cheapest way to sanity-check a bank: `run -t
   atlas_hellaswag` gives full-split accuracy *and* a post-hoc theta on the same scale
   as an online CAT, at no extra inference cost, so an adaptive estimate can be
   validated against ground truth on the same checkpoint.
