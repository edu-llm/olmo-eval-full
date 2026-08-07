# Plan — `flow/uni-mcq`: the unidimensional MCQ CAT style

Branch goal: add ONE registered style, `uni_mcq`, under
`diagnostics/mcq_cat/styles/uni_mcq/`, that runs a unidimensional (1PL/2PL/3PL)
adaptive multiple-choice diagnostic on a checkpoint and lands a `cat_report.json`
in S3. Everything lands under one disjoint directory so this branch PRs back into
`CheckpointFlows` without conflicts. Nothing in the frozen scaffolding
(`diagnostics/mcq_cat/{base,registry,runner}.py`, `common/*`) is edited.

MCQ grading is **in-process log-likelihood scoring** — there is *no* LLM judge and
*no* served endpoint. So on this branch "judging script" == the in-process scorer
already frozen in `diagnostics/mcq_cat/common/inference.py`; the real work is the
**CAT script** (IRT model + selector + stopping) plus **porting a calibrated IRT
bank** into the frozen loader schema.

## Source materials

**None of this is on `flow/uni-mcq`.** This branch has no `AdaptiveTesting/`, no
`src/olmo_eval/adaptive/`, and no `eduLLM-Evals/`; it is `main` plus the `diagnostics/`
scaffolding. Every path below lives on `origin/Research` and is read with
`git show <ref>:<path>`, which is why the bank must be **vendored** rather than
imported, and why the CAT math is **ported** rather than reused.

| Artifact | Path on `origin/Research` | Schema |
|---|---|---|
| ATLAS 3PL params | `AdaptiveTesting/Inputs/ATLAS/<bench>/irt_item_parameters_combined.csv` | `X,a1,d,g,u` |
| ATLAS idx→id bridge | `AdaptiveTesting/Inputs/ATLAS/<bench>/atlas_idx_to_question_id.csv` | `atlas_idx,question_id[,question]` |
| OpenLM v2 GPQA params (3PL and 2PL) | `AdaptiveTesting/Experiments/openlm_gpqa_atlas_3pl/{calibration,calibration_2pl}/irt_item_parameters_combined.csv` | `X,a1,d,g,u` |
| GPQA idx→id map | `AdaptiveTesting/Experiments/openlm_gpqa_atlas_3pl/data/item_id_map.csv` | `atlas_idx,item_id,subtask,question_id` |

One class of artifact is **not** upstream: the bridges for `hellaswag`, `winogrande` and
`gsm8k` live in this repo, at
`diagnostics/mcq_cat/styles/uni_mcq/bridges/<dataset>.csv`, because the upstream ones
were wrong and had to be rebuilt. See below.

Item stems and choices are **not** stored anywhere upstream. The one near-miss is
ARC's bridge, which uniquely carries a third `question` column, but it is stem-only
with no choices or gold index and so cannot score an item. Research solves this two
ways: its offline simulators need no items at all (they replay a saved response
matrix), and its live path enumerates `get_task(...).instances` at run time. We do the
latter **once, offline**, and commit the result.

Reference CAT implementation, ported verbatim: `src/olmo_eval/adaptive/irt.py` and
`cat.py` on `origin/Research` (3PL `prob` / `fisher_info`, EAP on 81 nodes over
[-4, 4], and the `pirt_accuracy` blend).

**ATLAS → standard IRT conversion** (from `adaptive/bank.py`): `a = a1`,
`b = -d / a1`, `c = clip(g, 0, 0.999)`, dropping items with `a <= 0`, non-finite `a`,
or no bridge entry. 2PL banks have `g = 0` → `c = 0`.

## Supported datasets

Admission requires all four: the bank came from **Route B** (Open LLM Leaderboard v2
harvest) or **Route C** (ATLAS published); a **grader exists for how the benchmark is
answered** (`modality`, multiple choice or generative); a **task definition exists on
this branch**; and the bank is **usable**. That leaves eight, encoded in
[`datasets.py`](../../../diagnostics/mcq_cat/styles/uni_mcq/datasets.py) along with the
exclusions and their reasons.

Supported is not the same as runnable, and the `blocked` field is what separates them.
Nothing carries one today; `resolve()` still checks it ahead of looking at the disk,
because the situation it guards is a dataset whose committed bank loads perfectly and
whose join was afterwards shown wrong.

| Dataset | Route | Bank | Modality | Id kind | Status |
|---|---|---|---|---|---|
| `arc_challenge` | C | `Inputs/ATLAS/arc` | mcq | native strings | ready |
| `ifeval` | B | `openlm_atlas_3pl/ifeval` | generative | content hash | ready — re-keyed, old key confirmed correct |
| `leaderboard_math` | B | `openlm_atlas_3pl/math` | generative | content hash | ready — re-keyed, **old key was wrong** |
| `hellaswag` | C | `Inputs/ATLAS/hellaswag` | mcq | content hash | ready — bridge rebuilt, verified |
| `winogrande` | C | `Inputs/ATLAS/winogrande` | mcq | content hash | ready — bridge rebuilt, weakly corroborated |
| `gsm8k` | C | `Inputs/ATLAS/gsm8k` | generative | content hash | ready — bridge rebuilt, unverifiable |
| `gpqa` | B | `openlm_gpqa_atlas_3pl/calibration` | generative | `<subtask>\|<doc_id>`, verified | ready — 579 of 1,192 survive `a > 0`, 395 after deduplication |
| `musr` | B | `openlm_atlas_3pl/musr` | mcq | content hash | ready — `acc_norm`, bridge re-keyed from v2 details |
| `bbh` | B | `openlm_atlas_3pl/bbh` | mcq | `<subtask>\|<doc_id>`, verified | ready — read predicted accuracy, not theta |

All nine are **3PL** fits. 2PL is supported without code changes because it is the
`c = 0` case of 3PL exactly — both the response function and Fisher information
reduce — so one implementation serves both families. GPQA and MuSR each have a 2PL refit
upstream that keeps far more items and, for MuSR, fits far more stably across folds; we
stay on 3PL by policy and the tradeoff is recorded in each spec's notes.

`musr` is the one MCQ bank not scored by an unnormalized sum. Its task declares
`LogprobPerCharMCAccuracyMetric` and leaderboard v2 reported it as `acc_norm`, so
normalization is a per-dataset setting — `MCQ_SCORE_NORMALIZATIONS`, selected from
`config.yaml` beside `prompt_style`, recorded in the manifest and checked at startup.
The other three resolve the default and are byte-for-byte unaffected.

### The three ATLAS bridges that had to be rebuilt

**Do not regenerate these with `Inputs/ATLAS/scripts/build_atlas_idx_bridge.py`.** That
script assumes ATLAS index *k* is the *k*-th instance `olmo_eval` enumerates from the
HuggingFace split, and the assumption is false. ARC is the only benchmark where it can be
checked directly, because it is the only one carrying a bridge the release *shipped*: the
generated and shipped bridges agree on 2 of 1,172 positions. The release's order is Open
LLM Leaderboard v1 example order — same items, permuted.

A permutation is the failure that hides. It joins every row, reports 100% overlap, clears
the positional-id floor, converges the CAT and collapses the standard error on schedule;
the theta is noise with a healthy interval beside it. The external check is to correlate
each item's p-value against the bank's implied `b = -d/a1`, which should be strongly
negative, and it is run by
[`scripts/check_bridge_alignment.py`](../../../diagnostics/mcq_cat/styles/uni_mcq/scripts/check_bridge_alignment.py).

The replacement reads the order off the v1 leaderboard's own per-model detail repos
(`open-llm-leaderboard-old/details_*`, one row per example in evaluation order) rather
than inferring it, via
[`scripts/build_leaderboard_bridge.py`](../../../diagnostics/mcq_cat/styles/uni_mcq/scripts/build_leaderboard_bridge.py).
The same run on ARC reproduces the shipped bridge on 1,172 of 1,172 question stems, which
is the control that licenses applying it to the three that have no shipped bridge.

| Bank | Spearman(p, b), old bridge | new bridge | Scrambled control | Items before → after |
|---|---|---|---|---|
| `hellaswag` | −0.036 | **−0.538** | −0.00 ± 0.03 | 4,840 → 5,044 |
| `winogrande` | +0.037 | **−0.177** | +0.00 ± 0.04 | 865 → 865 |
| `gsm8k` | no response data exists | no response data exists | — | 1,298 → 1,298 |

For scale, ARC's independently attested bridge reads −0.319 over the same four
checkpoints and −0.857 over the 3,747-model calibration matrix. HellaSwag clears that
reference; WinoGrande is the right sign and about five standard deviations off the null
but below the ≈−0.27 its lower p-value reliability predicts, so read its standard error
rather than its point estimate. GSM8K rests on the ARC control alone.

### Every Route B join has now been measured

The Route B banks looked lower risk and were not exempt. Their `<subtask>|<doc_id>`
composite names its own subtask, so any scramble is confined within one, but `doc_id` is
lm-eval's enumeration position at harvest time and rebuilding it by counting a fresh
enumeration is the same assumption at a smaller scale. All five have now been checked
against Open LLM Leaderboard **v2**'s per-example detail files
(`open-llm-leaderboard/<model>-details`), which record the document each `doc_id` named,
cross-checked against three further models spanning September 2024 to February 2025.

| Bank | Documents | Agreeing with the assumed position | Outcome | Items before → after |
|---|---|---|---|---|
| `musr` | 756 | 756 | re-keyed by content hash | 432 → 432 |
| `ifeval` | 541 | 541 | re-keyed by content hash | 511 → 511 |
| `leaderboard_math` | 1,324 | **8** | re-keyed by content hash | 1,183 → 1,183, **1,178 renamed** |
| `gpqa` | 1,192 | 1,192 | keeps its verified composite | 579 → 395, **184 nested repeats removed** |
| `bbh` | 5,761 | 5,761 | keeps its verified composite | 3,965 → 3,965 |

**`leaderboard_math` is the one that was wrong, and it was wrong almost everywhere.** Its
task numbers each subject's Level-5 subsequence of `DigitalLearningGmbH/MATH-lighteval`
from zero while lm-eval's `doc_id` counts the pre-filtered `lighteval/MATH-Hard` split,
so the two are unrelated orderings of the same 1,324 problems. The bank still holds 1,183
items at 100% overlap; 123 problems left it, 123 entered, and only 5 columns still
describe the question they described before. Every guard passed before and after, which
is the point.

`gpqa` and `bbh` keep a composite key deliberately. Their joins came out verified on
every document, and what stops the re-key is the **id**: GPQA's subsets are nested, so
448 questions appear under two or three subtask labels and 32 pairs hash alike once the
per-subset choice shuffle coincides, while BBH ships four literally duplicated items that
one hardcoded choice set per subtask cannot separate. Under a content hash both are
dropped as ambiguous, so the composite is the better key for the one reason a position
ever is — these really are distinct bank rows.

Distinct bank rows are not distinct questions, and on `gpqa` that distinction is the whole
story. Its three subsets are nested quality filters over one pool of 546 questions, so 184
of the 579 rows surviving `a > 0` are a second or third calibration of a question the bank
already holds, and the copies disagree wildly because diamond was fit in its own chunks of
the calibration. Vendoring keeps one per question — extended, else main, else diamond —
leaving 395 items, and the alignment statistic improves to −0.754 rather than degrading.
The argument is in
[`calibrated_datasets/README.md`](../../../calibrated_datasets/README.md).

These banks are checked against their own calibration response matrices rather than a
per-question harvest, and that check cannot see the join either — the matrix columns *are*
the parameter rows, so it attests the fit and the `b = -d/a1` conversion. `ifeval` reads
−0.888, `gpqa` −0.754, `leaderboard_math` −0.731, `bbh` −0.629 and `musr` −0.616, each
against a scrambled control within 0.003 of zero. `leaderboard_math` read −0.731 both
before and after a re-key that renamed 1,178 of its 1,183 items, which is the sharpest
available demonstration of what that statistic does not attest. See
[`bridges/README.md`](../../../diagnostics/mcq_cat/styles/uni_mcq/bridges/README.md).

## What this branch adds (all under `diagnostics/mcq_cat/styles/uni_mcq/`)

```
diagnostics/mcq_cat/styles/uni_mcq/
  __init__.py        # @register("uni_mcq") — the ONLY registration point
  datasets.py        # the supported-dataset allowlist + exclusions, pure data
  irt.py             # 3PL prob / fisher_info / eap_theta_se, ported verbatim
  pirt.py            # the pIRT blend, ported verbatim
  resolve.py         # allowlist check + the three-branch ladder
  style.py           # the seven CatStyle methods
  config.yaml        # se_threshold, min_items, max_items (NO fit_family — see below)
  bridges/           # the three bridges rebuilt from leaderboard v1 order, + provenance
  scripts/
    vendor_bank.py              # offline: upstream CSVs + task enumeration -> artifacts
    build_leaderboard_bridge.py # offline: recovers a bridge from v1 example order
    check_bridge_alignment.py   # offline: the p-value vs difficulty acceptance test
  tests/             # loader, parity, reduction, resolver, and a 5-item CAT smoke

calibrated_datasets/<dataset>/
  params.json        # IRT bank in (a, b, c) form — JSON, NOT JSONL
  items.jsonl        # question stems + choices + gold_index
  manifest.json      # provenance + drop accounting; fit_family lives here
```

Full artifact spec: [`calibrated_datasets/README.md`](../../../calibrated_datasets/README.md).
Per-dataset divergences from the olmo-eval task, and the ones that are deliberate:
[`calibrated_datasets/DEVIATIONS.md`](../../../calibrated_datasets/DEVIATIONS.md).

### 1. Vendor the bank → `calibrated_datasets/<dataset>/params.json`

Target schema is `diagnostics/mcq_cat/common/irt_params.py` (`item_id`, `difficulty`
(`b`), `discrimination` (`a`, scalar for unidim), `guessing` (`c`)).

**It is `params.json`, not `params.jsonl`.** `load_irt_params` calls `json.loads` on
the whole file and accepts a list or an object keyed by item id — it does not read
line-delimited JSON. Only `items.jsonl` is line-delimited.

```json
[{"item_id": "Mercury_417466", "difficulty": -0.119675, "discrimination": 1.098289, "guessing": 0.0}]
```

`scripts/vendor_bank.py` is the one-shot offline port: it reads the source CSV from
`origin/Research`, applies the guard and the (a, b, c) conversion, joins the bridge
for stable `item_id`s, and emits the artifacts. It is never imported at run time.

### 2. Vendor the items → `calibrated_datasets/<dataset>/items.jsonl`

Conform to the frozen `BenchmarkItem` (`id`, `question`, `choices`, `gold_index`).
The same script enumerates `get_task(<task>).instances`, keys them by
`metadata["id"]`, and intersects with the bank ids — the same restriction the live
Research path performs at run time. Only surviving items are written, so
`items.jsonl` and `params.json` always describe the same set.

### 3. The CAT script — `style.py` (implements `CatStyle` from `base.py`)

- `download_benchmark(benchmark)` → resolve the dataset name through `resolve.py`,
  then load `calibrated_datasets/<name>/items.jsonl` via the frozen
  `common/benchmark_download.load_items_from_jsonl`.
- `load_irt_params(source)` → delegate to `common/irt_params.load_irt_params` on
  `params.json` (scalar discrimination → `dimensions=1`).
- `estimate_ability(bank, responses, previous)` → **EAP** on the fixed 81-node grid
  over θ∈[-4,4] with a standard-normal prior; returns
  `AbilityEstimate(theta, standard_error)`.
- `select_next_item(bank, state)` → **max Fisher information** at current θ among
  un-administered items.
- `stopping_rule(state)` → `step >= min_items and se <= se_threshold`. Initial state
  is θ=0, SE=1, so the `min_items` floor always binds first.
- `report(state)` → `CATReport` with theta, SE, administered items, `pirt_accuracy`,
  `observed_accuracy`, and the bank provenance from `manifest.json`.

**One implementation serves both fit families.** 2PL is the `c = 0` case of 3PL
exactly: the response function reduces to the bare sigmoid, and Fisher information
reduces from `a²·((1-P)/P)·((P-c)/(1-c))²` to `a²·P·(1-P)`. There is no 2PL branch,
and a test pins that reduction numerically.

Grading itself is the in-process scorer (`common/inference.py`, log-likelihood over
choices) driven by `common/cat_loop.run_cat`. The scorer's *shape* did change once: it
takes a `(prompt, continuation)` pair per choice rather than one prompt and many
continuations, because WinoGrande's task substitutes the choice into the prompt and
shares the continuation. Which layout a dataset gets is data, in `MCQ_PROMPT_STYLES`,
selected from `config.yaml` — see
[`calibrated_datasets/DEVIATIONS.md`](../../../calibrated_datasets/DEVIATIONS.md).

### 4. `config.yaml`

```yaml
style: uni_mcq
se_threshold: 0.3
max_items: 40
min_items: 8
datasets:
  bbh:
    min_items: 24
```

**`fit_family` is deliberately absent.** It is a property of how a particular bank's
parameters were estimated, not a style-wide setting, so it lives in each bank's
`manifest.json` and is authoritative from there. The toggle exists at *vendoring*
time (`vendor_bank.py --fit-family {2pl,3pl}` selects the upstream calibration
directory); at run time a supplied value that disagrees with the manifest raises.

`max_items` is pinned at 40, matching both the runner's default and the seam's
`MAX_ITEMS`, so an ordinary invocation is comparable without passing an extra flag. 40
is what Research's live CAT administers: its library default is `DEFAULT_MAX_ITEMS =
200` in `src/olmo_eval/adaptive/cat.py`, but all three launchers that drive that CAT —
`AdaptiveTesting/scripts/atlas_cat_diagnose/launch_g6.sh`, its `run_worker.sh`, and
`.cursor/skills/add-cat-evals/scripts/run_cat_diagnostic.sh` — override it with
`se_stop=0.3 min_items=8 max_items=40`. The value is still recorded per run rather than
assumed, because estimates under the 200 default are not comparable with these.

`min_items` is per dataset, folded on by the same mechanism as `prompt_style`, and 8
everywhere but `bbh`. A shared floor is a claim about a bank's discriminations rather
than about the harness: eight items reach SE ≤ 0.3 where the median `a` is around 1.4,
and on BBH's inflated 3.99 the posterior collapses at once and the session ends before it
has touched a third of 24 unrelated subtasks. 24 is the smallest floor under which
touching all of them is arithmetically possible — it does not deliver coverage, because
Fisher selection clusters, and it does not improve BBH's estimate, whose `theta_mae` is
flat at 0.684 whatever the stopping rule. It is deliberately not in the recorded scoring
convention: it cannot change whether an item is answered correctly, so a bank vendored
under one floor and run under another stays comparable.

## How it runs (unchanged seam)

```bash
DIAGNOSTIC_MODALITY=mcq_cat CAT_STYLE=uni_mcq \
  CHECKPOINT=s3://.../step_1000 CHECKPOINT_KIND=hf \
  BENCHMARK=arc_challenge \
  tests/aws/run_checkpoint_diag.sh
# → uv run python -m diagnostics.mcq_cat.runner --cat-style uni_mcq \
#      --checkpoint ... --s3-out ... --benchmark arc_challenge --max-items 40
```

MCQ needs no tutor/judge endpoints; the box loads the checkpoint and scores in-process.

Two env vars matter more than they look:

**`BENCHMARK` is effectively required.** The seam forwards it only when set, and this
style has no default dataset — measuring something the user did not ask for is worse
than failing. Omitting it produces an error naming the ready datasets.

**`MAX_ITEMS` needs no override.** The seam defaults it to 40, the runner's own default
is 40, and the style pins 40, so all three agree and an ordinary run is comparable. The
CLI value still wins if supplied, as an explicit flag should; a run that used some other
cap logs a warning and records `max_items_is_pinned_value: false` in its report, so it
never looks interchangeable with one that did not.

## Verification before PR

Status as implemented:

1. **Done.** `ruff check` and `ruff format --check` both clean over
   `diagnostics/mcq_cat/styles/uni_mcq`.
2. **Done.** `python -m diagnostics.mcq_cat.runner --list-styles` prints `uni_mcq`.
3. **Done.** `--dry-run` resolves the style and prints the plan without loading a model.
4. **Done.** 683 tests pass under `diagnostics/` with `PYTHONPATH=src` and 589 without,
   the rest skipping on `olmo_eval`: loader round-trips through the frozen loaders,
   numerical parity pinned against the Research implementation, the 2PL-as-`c=0`
   reduction, manifest authority over fit family, the full ladder, prompt and metric
   parity against each task's own `format_request`, and theta recovery over every
   committed bank with only the forward pass simulated.
5. **Done.** `bash tests/aws/run_checkpoint_diag.sh --dry-run` with
   `DIAGNOSTIC_MODALITY=mcq_cat CAT_STYLE=uni_mcq BENCHMARK=arc_challenge` constructs the
   expected on-node command. A full CLI run against the real 650-item ARC bank (with the
   scorer stubbed) writes a valid `cat_report.json`.

### What a live GPU run additionally needs

The dry-run surfaces three prerequisites that are easy to miss:

**Point the staging at this fork.** `GIT_REPO_URL` defaults to
`https://github.com/allenai/olmo-eval` -- AI2 upstream, which contains no `diagnostics/`
package. A run with the default clones upstream and then fails with
`ModuleNotFoundError: diagnostics`. Set both:

```bash
GIT_REPO_URL=https://github.com/edu-llm/olmo-eval-full GIT_REF=flow/uni-mcq
```

**Commit and push `calibrated_datasets/` first.** The node gets the bank by cloning, so
an unpushed bank is an absent bank.

**For `ifeval`, install `ifbench` and pre-fetch its NLTK corpora.** The verifier registry
is a declared git-URL dependency and is not in the default install; several of its
verifiers additionally tokenize with NLTK and download `punkt_tab` and
`averaged_perceptron_tagger_eng` on first use. Where that download cannot reach the
network the failure is a `LookupError` mid-session, on whichever item selected such a
verifier, after the checkpoint has been staged.

```bash
uv pip install "ifbench @ git+https://github.com/allenai/IFBench.git@finbarr/clean-install"
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

Then, with authorization for the EC2 spend:

```bash
DIAGNOSTIC_MODALITY=mcq_cat CAT_STYLE=uni_mcq \
  GIT_REPO_URL=https://github.com/edu-llm/olmo-eval-full GIT_REF=flow/uni-mcq \
  CHECKPOINT=s3://.../step_1000 CHECKPOINT_KIND=hf \
  BENCHMARK=arc_challenge \
  bash tests/aws/run_checkpoint_diag.sh
```

## Merge-conflict discipline

- Touch **only** `diagnostics/mcq_cat/styles/uni_mcq/**` and this `Plan/flows/uni_mcq/`.
- Never edit `base.py`, `registry.py`, `runner.py`, `common/*`, or the seam. If the
  frozen loader can't represent a needed field, raise it on `CheckpointFlows` (change
  the contract once, for all styles) rather than editing shared code here.
- Registration happens exactly once, in `styles/uni_mcq/__init__.py`.
