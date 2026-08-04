# Benchmarks: what they score, what they cost, what they report

This file owns every benchmark-specific fact about the skill. `SKILL.md`
deliberately contains none, so that adding or removing a benchmark never means
editing it.

The short version of the recurring question: the skill runs the complete
designated **evaluation** split for every benchmark, and the parts it leaves out
are left out because they are unlabeled or because they are training data.

## Scope: what ships today

The registry holds seven benchmarks, organized into groups. **Running with no
flags gets the `default` group: the five multiple-choice reasoning tasks.** They
need no setup beyond the skill itself — no API keys, no new task code, no data
sourcing.

| Group | Benchmarks | Status |
|---|---|---|
| `default` / `reasoning` | `csqa` `hellaswag` `piqa` `socialiqa` `arc_easy` | ready, API-free |
| `fact_proxy` | `naturalqs` `jeopardy` | ready, API-free; generative, see the metric notes |
| `all` | all seven | ready |

`fact_proxy` is separated because those two are *proxies* for fact recall rather
than purpose-built fact benchmarks. They work today and cost nothing extra, so
they are a reasonable first read on whether a model recalls facts, but they are
not what a fact-recall study would report.

### What is deliberately not here yet

A fact-recall suite (TriviaQA, PopQA, SimpleQA, T-REx exact-match, FactScore) is
the eventual target. None of it is wired, and the gap is larger than it looks:

- **TriviaQA, PopQA, T-REx, FactScore have no task file in olmo-eval at all.**
  Each needs writing, and T-REx additionally needs a data source chosen, since
  there is no canonical evaluation split.
- **SimpleQA exists but the bare task scores nothing.** `Task.metrics` defaults to
  `()` and `simpleqa.py` never sets it; the only metric is attached by the
  `simpleqa:judge` variant. Running `-t simpleqa` would do full inference and
  report no score. Use `-t simpleqa:judge`.
- **SimpleQA and FactScore are LLM-judge graded**, via `build_openai_judge_fn`, so
  they need an API key in the container, outbound network access, and a per-call
  budget. FactScore additionally needs a retrieval corpus. This is a different
  operational posture from the API-free tasks above.
- **The prompt-count heuristic below does not transfer to them.** Generative tasks
  send one prompt per instance rather than one per answer choice, but generate
  many tokens instead of scoring a fixed continuation, so they are cheaper in
  prompt count and considerably slower per prompt.

When those land, add registry entries and extend a `factual` group. Until then
they can be run with `--allow-any-task`, which skips validation and gives no cost
estimate.

## The registry

`scripts/benchmarks.json` is the only place benchmark names appear in the skill's
code. `run_eval_sweep.sh` reads it to resolve `--group` and `--benchmarks`, reject
typos before any spend, and total up the cost estimate.

Each entry carries:

- `instances` — size of the scored split, for cost estimation.
- `choices` — vLLM prompts issued per instance. Multiple-choice log-likelihood
  scoring sends one prompt per answer choice; generative tasks send one per
  instance, so `choices` is 1 for them.
- `split` — which split the olmo-eval task scores, recorded so the estimate can
  be checked against the task definition.
- `metrics` — the metric keys the task emits. Entries with more than one get
  qualified column names in `accuracy_wide.csv`.
- `kind` — `mcq` or `generative`. Affects how to read the score and how the prompt
  count relates to the instance count.

Three side tables: `groups` names selectable sets (`default` is what runs when
neither `--group` nor `--benchmarks` is given), `aliases` maps common wrong names
to the real task name purely to produce a helpful error, and `limit_unsafe` flags
tasks whose split selection changes when `--limit` is set.

**To add a benchmark**, add an entry and put it in a group. Nothing else changes —
no shell edits, no `SKILL.md` edits. To run a task that is not in the registry,
pass `--allow-any-task`; it will run, but no cost estimate is available for it.

Which selection was used is recorded in each run's `run_provenance.json` as
`benchmark_selection`, and printed in the sweep log, so a result set is always
traceable to how it was requested.

## Task names

These are olmo-eval task names, not HuggingFace dataset names. The distinction
bites people writing the invocation from memory, so the registry rejects the
aliases with a pointer to the right name.

| Use this | Not this | Dataset it loads |
|---|---|---|
| `hellaswag` | `hella_swag` | `allenai/hellaswag` |
| `piqa` | — | `piqa` (via `refs/convert/parquet`) |
| `arc_easy` | `arc-easy` | `allenai/ai2_arc`, subset `ARC-Easy` |
| `csqa` | `commonsense_qa`, `commonsenseqa` | `commonsense_qa` |
| `socialiqa` | `social_i_qa`, `social_iqa`, `siqa` | `social_i_qa` (via `refs/convert/parquet`) |
| `naturalqs` | `natural_questions`, `nq`, `nq_open` | `google-research-datasets/nq_open` |
| `jeopardy` | — | `soldni/jeopardy`, subset `mosaicml_gauntlet` |

`piqa` and `socialiqa` pin the `refs/convert/parquet` revision because those
datasets originally shipped Python loading scripts that current `datasets`
versions refuse to execute. It works, but it is an auto-generated branch rather
than `main`, so pin a revision hash if you need strict reproducibility across a
long training run.

Benchmark data downloads from HuggingFace on first use. No gated repos are
involved, but the eval box needs Hub access and an `HF_HOME` with room.

## What each benchmark scores and reports

| Benchmark | Scored split | Instances | Choices | Metrics reported |
|---|---|---|---|---|
| `hellaswag` | validation | 10,042 | 4 | `accuracy` |
| `piqa` | validation | ~1,838 | 2 | `accuracy` |
| `arc_easy` | test | 2,376 | ~4 | `accuracy` |
| `csqa` | validation | 1,221 | 5 | `accuracy` |
| `socialiqa` | validation | ~1,954 | 3 | `accuracy` |
| `naturalqs` | validation | 3,610 | 1 | `f1` (primary) + `accuracy` |
| `jeopardy` | train | ~2,117 | 1 | `f1` (primary) + `accuracy` |

Five of the seven are multiple choice and report a single accuracy. Two are
generative and report **two** scores, because F1 and exact match answer different
questions — "how close was the answer" versus "was it right".

### What is excluded, and why

| Benchmark | Train split | Test split |
|---|---|---|
| `hellaswag` | 39,905, labeled, unused | 10,003, **unlabeled** |
| `piqa` | ~16,113, labeled, unused | **unlabeled** |
| `arc_easy` | 2,251, labeled, unused | *(this is the scored split)* |
| `csqa` | 9,741, labeled, unused | 1,140, **unlabeled** |
| `socialiqa` | ~33,410, labeled, unused | **unlabeled** |
| `naturalqs` | ~87k, labeled, unused | *(none exists)* |
| `jeopardy` | *(this is the scored split)* | *(none exists)* |

Five use validation, two use test. That pattern follows the OLMES convention,
visible in the `olmes_*_fixed` few-shot source names throughout the task files.
`arc_easy` uses test because ARC publishes labeled test data; `jeopardy` uses
train because the `mosaicml_gauntlet` subset has only that split.

## Unlabeled test data would score as all-wrong, silently

This is the reason "just run the whole dataset" is not an option. HellaSwag's
test split ships with `label: ""` on all 10,003 rows — the answers are withheld
for the leaderboard. olmo-eval maps a missing label to `gold_idx = -1` and still
returns the instance:

```python
label = int(doc["label"]) if doc.get("label", "") != "" else -1
```

Since a model's argmax is always in `0..3`, every such item scores as wrong. No
error is raised. Folding HellaSwag's test split in would cap a perfect model at
83.3% and look like a plausible result.

`Split.ALL` exists in the enum, so requesting every split is mechanically
possible. Do not use it for `hellaswag`, `csqa`, `piqa` or `socialiqa`. If more
items are ever wanted, name the train split explicitly per task.

## Train splits are excluded on purpose

They are labeled and scoreable, so this is a choice rather than a limitation. Two
reasons.

Comparability: published numbers for these benchmarks are on the evaluation
split, so mixing train in makes the result incomparable to anything.

Contamination, which matters more for a training sweep. Train portions are the
part most likely to appear in a pretraining corpus — they are published, widely
mirrored, and used for fine-tuning. Scoring checkpoints on data they may have
memorized inflates the numbers, and inflates them *progressively* as training
proceeds, which would read as genuine capability gain on a training curve.

### The few-shot split is loaded but never scored

All seven declare `fewshot_split = "train"`, so train is referenced. It is only
used to build demonstration examples. The five multiple-choice tasks default to
`num_fewshot = 0`, so it is not even loaded. `jeopardy` sets
`fewshot_source = "jeopardy_fixed"` and routes to a hardcoded constant instead of
the dataset.

`naturalqs` is the exception: `num_fewshot = 5` with no `fewshot_source` on the
base class, so it genuinely downloads the nq_open train split to draw five
examples. That is a download-size surprise, not a correctness problem.

## Metrics

| Benchmark | Metric | Scorer | Notes |
|---|---|---|---|
| `hellaswag` | `accuracy` | `logprob` | log-likelihood argmax over 4 choices |
| `piqa` | `accuracy` | `logprob` | per-token normalized, 2 choices |
| `arc_easy` | `accuracy` | `logprob` | ~4 choices |
| `csqa` | `accuracy` | `logprob` | 5 choices |
| `socialiqa` | `accuracy` | `logprob` | per-char normalized, 3 choices |
| `naturalqs` | `f1` (primary) | `drop_f1` | generative |
| | `accuracy` | `drop_exact_match` | |
| `jeopardy` | `f1` (primary) | `f1` | generative, SQuAD-style |
| | `accuracy` | `squad_exact_match` | |

All five multiple-choice tasks serialize under the metric name `accuracy`
regardless of their normalization variant. `LogprobMCAccuracyMetric`,
`LogprobPerTokenMCAccuracyMetric` and `LogprobPerCharMCAccuracyMetric` all set
`name = "accuracy"` and apply normalization inside `compute()`, so no per-task
key mapping is needed when reading results.

**Do not look for `acc_norm`.** That is lm-eval and Open LLM Leaderboard
nomenclature. It appears in `AdaptiveTesting/Inputs/OpenLM/download_openlm_responses.py`,
which is why it turns up in greps, but olmo-eval never emits it.

### F1 versus exact match on the generative pair

F1 measures token overlap between the generated answer and the reference:
precision is the fraction of generated tokens appearing in the reference, recall
the reverse, and F1 their harmonic mean. It gives partial credit — answering
"Shakespeare" against a reference of "William Shakespeare" earns 0.67.

Exact match asks a stricter question: did the answer match after normalization.
Both use the same SQuAD-style normalization (lowercase, strip punctuation, drop
`a`/`an`/`the`, collapse whitespace) and both take the best result over multiple
reference answers, so they differ only in strictness rather than in
preprocessing.

**An F1 of 0.62 does not mean 62% of questions were answered correctly.** It is
the mean partial-credit overlap. Reporting both is deliberate: F1 alone hides
whether a model is nearly right or verbosely wrong, and exact match alone
discards the difference between a close answer and nonsense. They are kept in
separate output columns because putting the two in one "accuracy" column would
misread.

`jeopardy` gained its exact-match metric in this repo — upstream shipped
`SQuADF1Metric` alone. The scorer is `SQuADExactMatchScorer` in
`src/olmo_eval/common/scorers/base.py`, the exact-match companion to the existing
`SQuADF1Scorer`, sharing its normalization and multi-reference handling. Plain
`ExactMatchScorer` was not used: it compares raw strings and so would penalize
the article and punctuation differences SQuAD scoring is defined to ignore,
disagreeing with the F1 sitting beside it. `squad.py` and `coqa.py` have the same
gap if anyone wants exact match there too.

Adding a second metric to a task means it also needs an explicit
`primary_metric`. `TaskConfig.get_primary_metric()` returns `None` when several
metrics exist without one, which would drop the task from the `summary` block of
`metrics.json`.

## Cost

A full seven-benchmark sweep is roughly **22,200 instances but about 71,000 vLLM
prompts** per checkpoint. The gap is because multiple-choice scoring sends one
prompt per answer choice, and HellaSwag alone accounts for about 40,000 of the
total (10,042 instances at four choices). The prompt count is what drives
runtime, and it multiplies by every checkpoint in the sweep.

`--dry-run` prints both figures, per checkpoint and for the whole sweep, before
anything spends. Use `--latest N` to cap a trial run.

## The `--limit` trap

`--limit N` is **not** a safe way to shrink a trial run. On `hellaswag` and
`socialiqa` a limit switches the task to loading validation *and* train and
sampling from the union, so a limited run scores a different population than an
unlimited one — in both directions, since the sampled set is neither the
evaluation split nor a superset of it.

Those two are flagged in the registry's `limit_unsafe` table, and the script
names them in its output whenever `--limit` is combined with an affected
benchmark. Use `--latest 1` to pilot instead.

## Reading the output

`accuracy_wide.csv` has one row per checkpoint and one column per score.
Benchmarks reporting a single metric get a bare column name; the generative pair
get qualified ones, so the columns for a full sweep are:

```
hellaswag  piqa  arc_easy  csqa  socialiqa
naturalqs.f1  naturalqs.accuracy  jeopardy.f1  jeopardy.accuracy
```

`accuracy.csv` is the same data in long form, with the `scorer` that produced
each number, an `is_primary` flag, and `num_instances`. Watch `num_instances`: it
is `len(responses)`, the count of successfully scored responses rather than the
number queued. Materially below the split size in the table above means instances
were dropped and the accuracy is over a smaller denominator.

Failed checkpoints appear with a `status` other than `ok` and no score, rather
than being omitted, so a checkpoint missing from the curve should always be
explainable from the table.

## The command this produces

For reference, a full sweep runs this per checkpoint — one vLLM boot covering
every benchmark:

```bash
uv run olmo-eval run -m "${HF_CKPT}" \
  --harness default -o provider.kind=vllm_server \
  -t hellaswag -t piqa -t arc_easy -t csqa -t socialiqa -t naturalqs -t jeopardy \
  -O "${OUT}"
```
