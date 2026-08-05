# Benchmarks: what they score, what they cost, what they report

This file owns every benchmark-specific fact about the skill. `SKILL.md`
deliberately contains none, so that adding or removing a benchmark never means
editing it.

The short version of the recurring question: the skill runs the complete
designated **evaluation** split for every benchmark, and the parts it leaves out
are left out because they are unlabeled or because they are training data.

## Scope: what ships today

The registry holds twelve benchmarks, organized into groups. **Running with no
flags gets the `default` group: the five multiple-choice reasoning tasks.** They
need no setup beyond the skill itself — no API keys, no new task code, no data
sourcing.

| Group | Benchmarks | Status |
|---|---|---|
| `default` / `reasoning` | `csqa` `hellaswag` `piqa` `socialiqa` `arc_easy` | ready, API-free |
| `fact_proxy` | `naturalqs` `jeopardy` | ready, API-free; generative, see the metric notes |
| `factual` | `popqa` `triviaqa` | ready, API-free; purpose-built fact recall |
| `all` | the nine above | ready |
| `smoke` | same as `default`, capped at 2 instances each | plumbing check only — **not a measurement** |
| `mc_format` | `arc_easy:mc` `csqa:mc` `socialiqa:mc` | ready, API-free; three of the above re-asked in the A/B/C/D format — a cross-check, not part of a sweep |

`all` is the nine, not the twelve. The three `mc_format` entries are the same
questions on the same splits as `arc_easy`, `csqa` and `socialiqa`, asked a
different way, so folding them into `all` would inflate a sweep's cost by a third
to score three benchmarks twice. Ask for them by name. See
[`rc` and `mc`](#rc-and-mc-two-ways-to-put-the-same-question-to-a-model).

`fact_proxy` is separated because those two are *proxies* for fact recall rather
than purpose-built fact benchmarks. They work today and cost nothing extra, so
they are a reasonable first read on whether a model recalls facts, but they are
not what a fact-recall study would report.

### What is deliberately not here yet

`popqa` and `triviaqa` have both landed and are in the `factual` group above. The
rest of the fact-recall suite (SimpleQA, T-REx exact-match, FactScore) has not,
and the gap is larger than it looks:

- **T-REx and FactScore have no task file in olmo-eval at all.** Each needs
  writing, and T-REx additionally needs a data source chosen, since there is no
  canonical evaluation split. TriviaQA used to be listed here as the easy one, on
  the grounds that it is the same shape as `popqa` and its `answer.aliases` field
  maps straight onto the multi-reference handling the SQuAD-family scorers
  already do. That turned out to be true, and it is now written: see
  [`triviaqa`](#triviaqa-closed-book-and-a-deliberate-mirror-of-co-lmlm).
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

When those land, add registry entries and extend the `factual` group, which is
what `popqa` and `triviaqa` did.

Until then all three — `simpleqa`, `trex` and `factscore` — are listed in the
registry's `unsupported` table and **refused outright**, with the reason
printed. That refusal survives `--allow-any-task` on
purpose: the flag skips this registry, not olmo-eval's task registry, so it
cannot run a task that has no task file. Allowing it through would only move the
failure to after the checkpoint had been fetched, converted and booted — and
because all benchmarks share one `olmo-eval run` invocation, it would take every
valid benchmark in the same request down with it. When a request mixes supported
and unsupported names, the error prints the runnable subset ready to paste back.

## The registry

`scripts/benchmarks.json` is the only place benchmark names appear in the skill's
code. **Both entry points read it**, through the one shared
`scripts/resolve_benchmarks.py`: `run_eval_sweep.sh` to resolve `--group` and
`--benchmarks`, reject typos before any spend, and total up the cost estimate,
and `submit_eval_run.sh` to do the same before a submission costs a queue slot.
A name valid on one path is valid on the other because there is only one
resolver.

Each entry carries:

- `instances` — size of the scored split, for cost estimation.
- `choices` — inference requests issued per instance, on either provider. On the
  platform path these are `olmo_core` forward passes rather than vLLM prompts.
  Multiple-choice log-likelihood scoring sends one per answer choice; generative
  tasks send one per instance, so `choices` is 1 for them.
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

A group is either a plain list of benchmark names or an object, which lets a
group carry settings as well as membership:

| Key | Meaning |
|---|---|
| `benchmarks` | the list, same as the plain-list form |
| `like` | inherit another group's list, so the two cannot drift apart |
| `limit` | instances per benchmark, applied unless `--limit` is passed |
| `description` | appended to the resolver's source line, which both paths print |

`smoke` uses `like: default` rather than repeating the five names, so editing
`default` keeps the smoke check honest automatically.

**To add a benchmark**, add an entry and put it in a group. Nothing else changes —
no shell edits, no `SKILL.md` edits. To run a task that is not in the registry,
pass `--allow-any-task`; it will run, but no cost estimate is available for it.

Which selection was used is traceable afterwards, though the two paths record it
differently. The sweep writes `run_provenance.json` with a `benchmark_selection`
key and prints the same string in `sweep.log`. The platform path writes
`eval_provenance.json`, which has no `benchmark_selection` key at all. It records
the resolved list under `benchmarks` and the invocation under
`olmo_eval_command`, so what ran is recoverable there; how it was *asked for* is
recoverable only from the submitted command, which the submitter prints and the
platform stores in the run's manifest.

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
| `popqa` | test | 14,267 | 1 | `accuracy` under two scorers: `containment` (primary) + `squad_exact_match` |
| `triviaqa` | validation | 17,944 | 1 | `accuracy` under two scorers: `windowed_containment` (primary) + `squad_exact_match` |
| `arc_easy:mc` | test | 2,376 | ~4 | `accuracy` |
| `csqa:mc` | validation | 1,221 | 5 | `accuracy` |
| `socialiqa:mc` | validation | 1,954 | 3 | `accuracy` |

Five of the nine are multiple choice and report a single accuracy. Four are
generative and report **two** scores each, because a single number cannot say
both "was it right" and "how close was it".

The three `:mc` rows repeat their base task's split, instance count and choice
count exactly, because the variant overrides the prompt format and nothing else.
That is asserted against the real `register_variant` call in
`tests/python/test_mc_variants.py`, so the day someone adds a `limit=` upstream
the registry is told rather than left quietly wrong.

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
| `popqa` | *(none exists)* | *(this is the scored split)* |

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

Seven of the nine declare `fewshot_split = "train"`, so train is referenced. It
is only used to build demonstration examples. The five multiple-choice tasks
default to `num_fewshot = 0`, so it is not even loaded. `jeopardy` sets
`fewshot_source = "jeopardy_fixed"` and routes to a hardcoded constant instead of
the dataset.

`naturalqs` is the exception: `num_fewshot = 5` with no `fewshot_source` on the
base class, so it genuinely downloads the nq_open train split to draw five
examples. That is a download-size surprise, not a correctness problem.

`popqa` declares no `fewshot_split` at all, because it has no train split to
declare. Its 15 demonstrations are hardcoded in `constants/popqa.py`, so it
never reads anything but the split it scores.

`triviaqa` declares none either, for a different reason: it is zero-shot, so
there are no demonstrations to source. TriviaQA does publish a train split of
138,384 questions; this task simply never touches it.

## Metrics

| Benchmark | Metric | Scorer | Notes |
|---|---|---|---|
| `hellaswag` | `accuracy` | `logprob` | log-likelihood argmax over 4 choices |
| `piqa` | `accuracy` | `logprob` | per-token normalized, 2 choices |
| `arc_easy` | `accuracy` | `logprob` | ~4 choices |
| `csqa` | `accuracy` | `logprob` | 5 choices |
| `socialiqa` | `accuracy` | `logprob` | per-char normalized, 3 choices |
| `arc_easy:mc` | `accuracy` | `logprob` | argmax over one label per option, usually `A`–`D`; inherits unnormalized |
| `csqa:mc` | `accuracy` | `logprob` | labels `A`–`E`; inherits unnormalized |
| `socialiqa:mc` | `accuracy` | `logprob` | labels `A`–`C`; inherits per-char, which the equal-length labels make a no-op |
| `naturalqs` | `f1` (primary) | `drop_f1` | generative |
| | `accuracy` | `drop_exact_match` | |
| `jeopardy` | `f1` (primary) | `f1` | generative, SQuAD-style |
| | `accuracy` | `squad_exact_match` | |
| `popqa` | `accuracy` (primary) | `containment` | generative; gold answer appearing anywhere in the generation |
| | `accuracy` | `squad_exact_match` | same metric name, second scorer |
| `triviaqa` | `accuracy` (primary) | `windowed_containment` | generative; gold answer in the first 100 characters, lowercase only |
| | `accuracy` | `squad_exact_match` | same metric name, second scorer |

Every multiple-choice task here serializes under the metric name `accuracy`
regardless of its normalization variant, and under the scorer name `logprob`
regardless of which. `LogprobMCAccuracyMetric`, `LogprobPerTokenMCAccuracyMetric`
and `LogprobPerCharMCAccuracyMetric` all set `name = "accuracy"`, all use
`LogprobScorer`, and apply their normalization inside `compute()`, so no per-task
key mapping is needed when reading results. The benchmark name is therefore the
only thing separating one MCQ column from another — which is why the `:mc`
entries keep their full name, colon included, all the way into the CSV header.

**Do not look for `acc_norm`.** That is lm-eval and Open LLM Leaderboard
nomenclature. It appears in `AdaptiveTesting/Inputs/OpenLM/download_openlm_responses.py`,
which is why it turns up in greps, but olmo-eval never emits it.

### `rc` and `mc`: two ways to put the same question to a model

Everything above scores what olmo-eval calls the **`rc`** format, after reading
comprehension, and what the literature calls cloze scoring. The options never
appear in the prompt. Each one is scored as a separate continuation of the
question, in full, and the answer is whichever continuation the model found most
likely. Four options means four requests, each carrying the whole option text.

The **`mc`** format asks the same question the way a person would see it on a
test paper: the options are listed in the prompt, labelled `A.`, `B.`, `C.`, and
the request scores a single label token. Four options still means four requests —
this is argmax over `" A"`, `" B"`, `" C"`, `" D"`, not one generation. What
shrinks is only the continuation; the prompt grows to hold every option, so a
request is *longer* than its cloze counterpart. But a single-token label is
dropped from the model input entirely, which leaves all four requests carrying
byte-identical input, and the provider forwards it once — so the longer prompt is
paid for once rather than four times. See
[`mc_format` costs less than its request count says](#mc_format-costs-less-than-its-request-count-says).
The switch is the `is_mc` branch in each task's `format_request`, reached by the
variant setting a `MultipleChoiceFormatter`.

This is *multiple choice prompting* from Robinson & Wingate, **"Leveraging Large
Language Models for Multiple Choice Question Answering"** (ICLR 2023), which also
names the ability it depends on: **multiple choice symbol binding**, associating
a bare symbol with the option text sitting beside it. Its failure mode is
documented by Zheng et al., **"Large Language Models Are Not Robust Multiple
Choice Selectors"** (ICLR 2024): a model that cannot bind falls back on a prior
over the labels themselves, so the score reports where the answer was placed as
much as what it was.

| Name | Select it with |
|---|---|
| `arc_easy:mc` | `--group mc_format`, or `--benchmarks "arc_easy:mc csqa:mc socialiqa:mc"` |
| `csqa:mc` | as above, or on its own with `--benchmarks "csqa:mc"` |
| `socialiqa:mc` | as above |

**These are a cross-check, not a training curve.** Symbol binding is emergent:
it appears at some scale and some amount of training and is simply absent before
that, so an early pretraining checkpoint reads at or near chance — 25%, 20%, 33%
for these three — while its `rc` score is already moving. A flat `mc` line over
the first half of a run is the expected result and says nothing about the model
beyond "not yet". Plot the default group as the curve; run `mc_format` at a
handful of steps to see whether the format has started to work at all, and near
the end to check that an `rc` gain is a real gain rather than an artifact of
continuation scoring.

All three inherit `num_fewshot = 0` from their base task, so this is zero-shot
multiple choice prompting, which is a harder ask than the setting the result was
published in — nothing in the prompt demonstrates that a bare letter is the
expected answer. Expect these to lag a few-shot number from the literature, and
do not read a low score as the model failing the questions until it has answered
a few in the format. Neither of those is a reason to distrust multiple choice
prompting itself; both are consequences of asking for it zero-shot, this early.

**The answer for a first sweep is still no, but for one reason rather than two.**
The group is expected to read at or near chance until symbol binding appears, so on
an early checkpoint it buys a number that says nothing about the model. Cost is no
longer part of the argument — one forward pass per instance makes `mc` the cheaper
of the two formats. Run it once binding is plausible, or near the end of a run as a
check that an `rc` gain is real, and leave it out of the first pass.

Stacking a shot count on is not available: `arc_easy:mc:full`
parses, but `arc_easy` has no five-shot `mc` variant at all, and the two that do
exist — `csqa:mc_olmo3base` and `socialiqa:mc_olmo3base` — also switch to
`validation+train` and subsample to 10,000, so they are not the same population
as the entries here and are not offered as if they were.

What `mc` is good for is that it **cannot be gamed by option length or option
prior**, because every continuation is one letter. Those two biases are what the
normalized metrics in the table above exist to correct, and they are only a
problem because cloze scoring compares whole option texts against each other:
`LogprobPerCharMCAccuracyMetric` divides by continuation length so a long option
is not penalised for having more tokens to be unlikely about, and
`LogprobUncondMCAccuracyMetric` — which `csqa:rc` uses, though the registered
`csqa` does not — subtracts each option's unconditional logprob so a phrase the
model finds plausible on its own does not win on that alone, at the cost of a
second request per option. Under `mc` both corrections are no-ops: the labels are
the same length and, being bare letters, carry no content prior. What replaces
them is label-position bias, which is Zheng et al.'s subject and which no metric
here corrects for. The bias does not disappear; it changes shape.

The metric key does not change. All three inherit their base task's metric, and
every one of those serializes as `accuracy` under the `logprob` scorer, so
`accuracy_wide.csv` gains three columns named for the benchmarks —
`arc_easy:mc`, `csqa:mc`, `socialiqa:mc` — sitting beside the base task's own.
`socialiqa:mc` inherits `LogprobPerCharMCAccuracyMetric` specifically, which is
harmless rather than wrong: dividing every option by the same two characters
leaves the argmax where it was.

**`hellaswag:mc` and `piqa:mc` exist upstream and are deliberately not
registered.** Both are one `register_variant` call away, in
`evals/tasks/hellaswag.py` and `evals/tasks/piqa.py`, and nothing here changes
their behaviour — they are simply not offered, because the format does not suit
them:

- **HellaSwag has no question.** Its loader builds the query from
  `activity_label + ": " + ctx` and the options from `endings`, so the four
  options are sentence continuations of that context rather than answers to
  anything. Labelling them `A.` to `D.` and asking for a letter discards the one
  signal the task is built on — which ending reads as a natural continuation —
  and replaces it with a matching exercise the dataset was not written for.
- **PIQA is binary, and its stem is a goal rather than a question.** Two options
  means chance is 50%, so the range a real result has to stand out from is half
  as wide, and label bias eats a larger share of what is left. A 3-point move on
  a 50-point range is much harder to distinguish from a preference for `A` than
  the same move on `arc_easy`'s 75.

Both would still run under `--allow-any-task`, without a cost estimate, if
someone wants to look. Registering them would mean standing behind the number,
which is a different thing.

### F1 versus exact match on the generative pair

F1 measures token overlap between the generated answer and the reference:
precision is the fraction of generated tokens appearing in the reference, recall
the reverse, and F1 their harmonic mean. It gives partial credit — answering
"Shakespeare" against a reference of "William Shakespeare" earns 0.67.

Exact match asks a stricter question: did the answer match after normalization.
Within a task the two metrics share their normalization and both take the best
result over multiple reference answers, so they differ only in strictness rather
than in preprocessing.

The two tasks do not share that normalization with *each other*, which the scorer
column above records and which matters if you are tempted to compare them.
`jeopardy` scores SQuAD-style: lowercase, strip punctuation, drop `a`/`an`/`the`,
collapse whitespace. `naturalqs` uses DROP scoring (`drop.py`), which on top of
that splits on hyphens, canonicalizes numbers so `5` and `5.0` match, compares
token *sets* so repeated tokens collapse, scores a reference containing a number
as 0 unless the answer contains that number, and rounds F1 to two decimals. Track
`naturalqs` against `naturalqs` across checkpoints, not against `jeopardy`.

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

### `popqa`: aligned to the paper

`popqa` follows the setup in Mallen et al. 2023, *When Not to Trust Language
Models* ([arXiv:2212.10511](https://arxiv.org/abs/2212.10511)), which introduced
the dataset. Comparability to published numbers is the main reason to run PopQA
rather than one of the fact proxies, so each of these is the paper's choice
rather than ours:

| Setting | Value | Source |
|---|---|---|
| Prompt | `Q: <question>\nA:` | §4.1, "a simple template 'Q: A:'" |
| Shots | 15 | §4.1, 15-shot for GPT-Neo and OPT |
| Split | all 14,267 of `test` | the only split published |
| Primary metric | accuracy by substring containment | §3.1 |

The metric, quoted: *"We mark a prediction as correct if any substring of the
prediction is an exact match of any of the gold answers."* That is containment,
and it is deliberately lenient — it is robust to a base model answering "The
capital of France is Paris" rather than "Paris", which exact match reads as
flatly wrong.

**`squad_exact_match` is reported beside it, and is not from the paper.** It is
there because containment is inflated by two things, and the gap between the two
scores is what exposes them:

- The rule is raw substring, not token boundaries. Several PopQA answer sets list
  short aliases — `politician` also lists `pol` — so a prediction of "policy"
  scores correct. This is a known false positive of the published metric, kept
  and asserted in the tests rather than fixed, because tightening it would leave
  published numbers unreachable.
- A model that hedges by listing candidates ("Paris, London, Rome") scores
  correct because one of them hits.

So containment far above exact match means either padding or hedging, not
necessarily recall. Report containment for comparison with the literature; watch
exact match to know whether to trust it.

No F1: the answer is a single entity, so partial token overlap is not a
meaningful quantity.

Both scores land under the metric name `accuracy` with different scorer keys, so
`accuracy_wide.csv` gives them the three-level names
`popqa.accuracy.containment` and `popqa.accuracy.squad_exact_match`.

PopQA has no train split, so demonstrations cannot be sampled. The 15 in
`constants/popqa.py` are written by hand, since the paper does not publish its
own. Every question uses a relation template taken verbatim from the dataset —
PopQA phrases these precisely, "Who *was* the director of X?" against "Who *is*
the author of X?", and a mismatched demonstration would teach a format the scored
questions never use. Subjects are high-popularity, so a few may also appear among
the scored questions; that is the same contamination the paper's own
demonstrations would carry, bounded at roughly 0.1% of the split.

The `popqa:zeroshot` variant reproduces the paper's other arm, which it used for
GPT-3 only to hold down API cost.

### Popularity is carried through to the predictions

PopQA exists to separate head from long-tail factual recall, so a single flat
score throws away the point of it. Each instance therefore carries the dataset's
own popularity fields into `predictions/popqa-predictions.jsonl` under
`instance_attributes`:

| Field | Meaning |
|---|---|
| `s_pop` | subject entity's monthly Wikipedia pageviews — the axis to bucket by |
| `o_pop` | answer entity's pageviews |
| `prop` | relation type, e.g. `occupation` |
| `subj` | subject entity name |

These are copied from the input dataset and are **never** produced by the model
being evaluated. They sit on the same JSONL row as that instance's scores, so
accuracy can be broken down by popularity without joining against
`requests.jsonl`.

The summarizer still reports one number per scorer across the whole split;
bucketing by popularity is a downstream step on the predictions file.

### `triviaqa`: closed-book, and a deliberate mirror of Co-LMLM

TriviaQA is titled a reading-comprehension dataset and ships evidence documents
with every question. **The config choice is what decides whether this measures
recall or comprehension**, so it is the first thing to check if the numbers ever
look wrong. This task loads `rc.nocontext`: the same 17,944 questions as `rc`
with the evidence stripped. The two are otherwise identical, but `rc` validation
is 936 MB against 7.3 MB — so if a run suddenly downloads a gigabyte, the wrong
config is wired up.

The setup mirrors **Co-LMLM** (arXiv:2607.07707, appendix A.6), which is where
each choice comes from rather than from our own judgement:

| Setting | Value | Source |
|---|---|---|
| Config / split | `rc.nocontext` / `validation` | their `prepare_popqa_prompts.py` loader |
| Instances | all 17,944 | A.6, "the full TriviaQA evaluation set (17,944 examples)" |
| Shots | 0 | their pipeline has no few-shot machinery |
| Prompt | `{question}\nThe answer is` | their `append_answer_stub.py` |
| Decoding | greedy, 32 tokens | A.6, "greedy decoding with a maximum of 32 tokens" |
| References | raw `value` + `aliases` | their loader; *not* the `normalized_*` fields |
| Metric | gold in the first 100 chars, case-insensitive | A.6 |

Two consequences worth internalizing. **Zero-shot means nothing demonstrates the
answer format**, so a model that knows the fact may still answer in a sentence;
their answer cue nudges toward brevity but does not enforce it. That is exactly
what containment absorbs, and why exact match sits beside it. And **the paper
calls its metric "Exact Match" when it is substring containment** — it scores
strictly higher, so do not compare their TriviaQA column against an exact-match
number from anywhere else.

`WindowedContainmentScorer` implements their rule rather than reusing
`ContainmentScorer`, because the two differ in ways that move the score:
lowercasing is the only normalization, so a gold of "the Beatles" does not match
a prediction of "Beatles" where SQuAD normalization would; and only the first
100 characters are searched, so an answer after a long preamble does not count.

**What is ours, not theirs:** the `squad_exact_match` companion. The paper
reports no second metric for TriviaQA. It changes no published-comparable
number, and it is the thing that reveals when containment is being inflated by a
rambling answer.

One caveat on the benchmark itself. `rc` is the *reading-comprehension* subset,
filtered so evidence documents contain the answer, and stripping the evidence
does not undo that filtering — the question set still skews toward what was
answerable from retrieved text. TriviaQA's authors suggest `unfiltered` for
open-domain use. Matching the paper is the reason to run this, so `rc.nocontext`
stands, but `unfiltered.nocontext` (11,313 questions) is the purer read if a
second opinion is ever wanted.

Finally, `triviaqa` and `popqa` are complements rather than substitutes, which
is why both sit in `factual`. TriviaQA skews toward well-known entities and
measures head knowledge; PopQA is built for the long tail. A mid-training
checkpoint should show a wide gap between them, and that gap closing is more
informative than either number alone.

## Cost

A full nine-benchmark sweep is roughly **55,400 instances and about 103,300
inference requests** per checkpoint — vLLM prompts on the sweep, `olmo_core`
forward passes on the platform. Requests exceed instances because multiple-choice
scoring sends one per answer choice, and HellaSwag alone accounts for about
40,000 of the total (10,042 instances at four choices). The request count is what
drives runtime, and it multiplies by every checkpoint in the sweep.

The two `factual` benchmarks are the largest by instance count — `triviaqa` at
17,944 and `popqa` at 14,267 — but being generative they issue one prompt each,
so they add less runtime than their share of the instances suggests. `--group
factual` runs just those two, at 32,211 instances and the same 32,211 prompts.

### `mc_format` costs less than its request count says

`--group mc_format` is 5,551 instances and 21,471 requests: the same instances and
the same request count as `arc_easy`, `csqa` and `socialiqa` under `rc`, because
a variant changes the prompt and not the number of options to score.

**Read as cost, that parity is wrong, and it is wrong in the cheap direction.** The
four requests of an `mc` instance are identical to one another. A row's model input
is `(context + continuation)[:-1]`, and for a single-token label that expression
drops the label, so every row of a four-option instance holds byte-identical input:
the same question asked four times. `OlmoCoreProvider` forwards each distinct input
once and reads all four labels out of the one distribution, so 21,471 requests are
**5,551 forward passes**, one per instance.

That makes `mc` the cheaper format rather than the more expensive one. Cloze cannot
share, because each of its requests appends a different option and the inputs
genuinely differ. A four-option instance forwards roughly four questions and four
options under `rc`, against one question and four options under `mc`.

**The sharing is exact, not an approximation.** Identical input means identical
logits at every position the scorer reads, and the per-label log-probs come out
bit-identical to the unshared path — it is the same arithmetic, done once. It
applies to `olmo_core` and `huggingface`, which compute a full-vocabulary
distribution. The vLLM, vLLM-server and LiteLLM providers request a bounded number
of prompt log-probs and so cannot read an arbitrary label out of a single pass; they
still forward per request. `mc` is cheap on `vllm_server` for a different reason,
prefix caching, which is on by default there.

Two consequences. A `--batch-size` tuned against the old profile is now conservative
for this group, since its effective batches are several times smaller — safe, but
stale. And a request count is no longer a proxy for forward passes here, which
matters when comparing observed runtime against the estimate.

Generative requests are not directly comparable to multiple-choice ones either:
each one generates tokens rather than scoring a fixed continuation, so it is
slower per request. Treat the request count as a within-kind comparison.

`--dry-run` prints both figures, per checkpoint and for the whole sweep, before
anything spends. Use `--latest N` to cap a trial run.

## The `--limit` trap

`--limit N` is **not** a safe way to shrink a trial run. On `hellaswag` and
`socialiqa` a limit switches the task to loading validation *and* train and
sampling from the union, so a limited run scores a different population than an
unlimited one — in both directions, since the sampled set is neither the
evaluation split nor a superset of it.

`socialiqa:mc` is affected too, and is flagged separately: a variant changes the
prompt, not the loader, and the registry matches on the exact name it was given.

Those three are flagged in the registry's `limit_unsafe` table, and the script
names them in its output whenever a limit is combined with an affected
benchmark. Use `--latest 1` to pilot a real measurement instead.

### Why `smoke` exists anyway

`--group smoke` deliberately walks into this trap: it is `default` with a limit
of 2, so it triggers exactly the warning above on `hellaswag` and `socialiqa`.
That is the intended behavior. Its purpose is to prove the machinery works —
checkpoint fetched, converted, vLLM booted, all five tasks scored, results
uploaded, summary written — at roughly 10 instances and 36 prompts per
checkpoint instead of 17,431 and 65,315.

The scores it produces are not measurements of anything and must never be
reported or plotted. Use it to answer "does this run at all", then re-run
without it to answer "how good is this checkpoint".

**A limit shrinks inference, not the download.** The two mechanisms differ per
task, and neither avoids reading the dataset:

| Benchmark | How the limit is applied | Sampled from | Rows read to pick 2 |
|---|---|---|---|
| `csqa` | generic sampler in `runners/asynq/preparation.py`, seed 42 | validation | 1,221 |
| `piqa` | same generic sampler | validation | 1,838 |
| `arc_easy` | same generic sampler | test | 2,376 |
| `hellaswag` | in-task, `random.Random(1234)` | validation **+ train** | 49,947 |
| `socialiqa` | in-task, `random.Random(1234)` | validation **+ train** | 35,364 |

So a smoke run issues 36 prompts but still downloads and processes roughly 90k
rows on a cold `HF_HOME`, most of it HellaSwag and SocialIQA train data that
exists only to be sampled away. Budget for the download on a fresh box; it is
cached for subsequent runs. Both samplers are seeded, so repeated smoke runs
score the same instances.

## Reading the output

`accuracy_wide.csv` has one row per checkpoint and one column per score. Column
names are qualified only as far as they need to be: a bare benchmark name when it
reports a single score, `<benchmark>.<metric>` when it reports several metrics,
and `<benchmark>.<metric>.<scorer>` when one metric name carries more than one
scorer. Nothing shipping today needs that third level — `Task.compute_metrics`
allows it, so the summarizer names for it rather than letting two scorers collide
into one column. The columns for a full sweep are:

```
hellaswag  piqa  arc_easy  csqa  socialiqa
naturalqs.f1  naturalqs.accuracy  jeopardy.f1  jeopardy.accuracy
popqa.accuracy.containment  popqa.accuracy.squad_exact_match
triviaqa.accuracy.windowed_containment  triviaqa.accuracy.squad_exact_match
```

A `mc_format` run adds `arc_easy:mc`, `csqa:mc` and `socialiqa:mc`, each a bare
benchmark name like the other single-score tasks. The colon is part of the
benchmark name rather than a separator, so it needs no escaping in the header and
cannot collide with the `.metric.scorer` qualification, which only ever appends.

`accuracy.csv` is the same data in long form, with the `scorer` that produced
each number, an `is_primary` flag, and `num_instances`. Watch `num_instances`: it
is `len(responses)`, the count of successfully scored responses rather than the
number queued. Materially below the split size in the table above means instances
were dropped and the accuracy is over a smaller denominator.

Failed checkpoints appear with a `status` other than `ok` and no score, rather
than being omitted, so a checkpoint missing from the curve should always be
explainable from the table.

## Two inference backends, and their numbers are not interchangeable

The same benchmark scored through a different provider can give a different
number, so which one produced a score belongs beside it.

| Path | Provider | Why |
|---|---|---|
| `submit_eval_run.sh` (platform) | `olmo_core` | The OLMo-core image carries CUDA torch and `ai2-olmo-core` and not vLLM |
| `run_eval_sweep.sh` (own machine) | `vllm_server` | vLLM is installed there, and is far faster per prompt |

For the five multiple-choice benchmarks the risk is low: scoring is
log-likelihood argmax over fixed continuations, and two correct implementations
should agree. For the four generative ones it is real — sampling, stop-sequence
handling and tokenisation details all differ between backends, and every metric
here is computed on the generated string.

So compare a checkpoint against itself across steps on **one** path, and treat a
cross-path comparison as a different experiment. `eval_provenance.json` records
which provider ran, which is what makes that checkable after the fact.

The `olmo_core` provider also reads a native OLMo-core checkpoint directly, so
the platform path never converts to HF. That removes a step, and it means a
checkpoint that only exists in HF format is not evaluable on that path.

### Why the platform path cannot simply use vLLM too

This gets asked, reasonably, because `VLLM_SETUP.md` at the repo root documents a
vLLM run that provably worked on AWS. It worked on a *Deep Learning AMI*, which is
a different thing from a Batch container: drivers and CUDA pre-baked, `uv`
available, a repo checkout on disk, and a large NVMe scratch. Three things block
carrying it into a platform job, and each is sufficient alone.

**Torch.** The OLMo-core image ships torch 2.9.0+cu128 on Python 3.12.13. Our
`uv.lock` resolves `vllm 0.19.1` against torch 2.10.0. vLLM's wheels link against
a specific torch C++ ABI, so a minor-version gap means pip must replace the
image's torch — gigabytes, and it overwrites the CUDA build the image exists to
provide.

**Disk.** `gpu-1xa10g` has no launch template, so it takes the ECS GPU AMI default
of 30 GiB and has roughly 13 GiB free once the OS and the 4.4 GB image are
accounted for. torch plus vLLM plus a native checkpoint plus its HF conversion
does not fit. Only `gpu-8xa100` and `gpu-8xh100` carry 500 GiB, and `capacity.yaml`
marks both as placing unreliably.

**Time.** `olmo-core-check` is bounded at one hour, and the platform's own guidance
for `olmo-eval-full` warns that installing a backend at runtime spends that hour
on a download.

Building an eval image with vLLM baked in is the only route past these, and it is
not free either: the ECR scan gate lists `image_scan_findings_unreviewed` under
`denied_outright`, so every unreviewed critical in the newly added dependency tree
refuses the run until someone writes a per-vulnerability justification.

Worth keeping for whoever revisits this. The argument *for* vLLM is real and
specific: multiple-choice scoring issues one request per option over a shared
question prefix, `vllm_server` enables prefix caching by default, and caching
applies to the log-likelihood path because both providers score through
`prompt_logprobs` with `max_tokens=1` rather than by generating. The pin set known
to work together is vLLM 0.19.1, torch 2.10.0+cu128, transformers 5.7.0,
ai2-olmo-core 2.4.0, CPython 3.12.13. And vLLM does not conflict with `olmo_core`
— `pyproject.toml`'s `conflicts` block names `openhands` on one side of every pair.

## The command this produces

For reference, a full sweep runs this per checkpoint — one vLLM boot covering
every benchmark:

```bash
uv run olmo-eval run -m "${HF_CKPT}" \
  --harness default -o provider.kind=vllm_server \
  -t hellaswag -t piqa -t arc_easy -t csqa -t socialiqa \
  -t naturalqs -t jeopardy -t popqa -t triviaqa \
  -O "${OUT}"
```
