# Benchmarks: what they score, what they cost, what they report

This file owns every benchmark-specific fact about the skill. `SKILL.md`
deliberately contains none, so that adding or removing a benchmark never means
editing it.

The short version of the recurring question: the skill runs the complete
designated **evaluation** split for every benchmark, and the parts it leaves out
are left out because they are unlabeled or because they are training data.

## Scope: what ships today

The registry holds nine benchmarks, organized into groups. **Running with no
flags gets the `default` group: the five multiple-choice reasoning tasks.** They
need no setup beyond the skill itself — no API keys, no new task code, no data
sourcing.

| Group | Benchmarks | Status |
|---|---|---|
| `default` / `reasoning` | `csqa` `hellaswag` `piqa` `socialiqa` `arc_easy` | ready, API-free |
| `fact_proxy` | `naturalqs` `jeopardy` | ready, API-free; generative, see the metric notes |
| `factual` | `popqa` `triviaqa` | ready, API-free; purpose-built fact recall |
| `all` | all nine | ready |
| `smoke` | same as `default`, capped at 2 instances each | plumbing check only — **not a measurement** |

`fact_proxy` is separated because those two are *proxies* for fact recall rather
than purpose-built fact benchmarks. They work today and cost nothing extra, so
they are a reasonable first read on whether a model recalls facts, but they are
not what a fact-recall study would report.

### What is deliberately not here yet

`popqa` has landed; the rest of the fact-recall suite (TriviaQA, SimpleQA, T-REx
exact-match, FactScore) has not, and the gap is larger than it looks:

- **TriviaQA, T-REx and FactScore have no task file in olmo-eval at all.** Each
  needs writing, and T-REx additionally needs a data source chosen, since there
  is no canonical evaluation split. TriviaQA is the easy one — it is the same
  shape as `popqa`, and its `answer.aliases` field maps straight onto the
  multi-reference handling the SQuAD-family scorers already do.
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

When those land, add registry entries and extend a `factual` group.

Until then they are listed in the registry's `unsupported` table and **refused
outright**, with the reason printed. That refusal survives `--allow-any-task` on
purpose: the flag skips this registry, not olmo-eval's task registry, so it
cannot run a task that has no task file. Allowing it through would only move the
failure to after the checkpoint had been fetched, converted and booted — and
because all benchmarks share one `olmo-eval run` invocation, it would take every
valid benchmark in the same request down with it. When a request mixes supported
and unsupported names, the error prints the runnable subset ready to paste back.

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

A group is either a plain list of benchmark names or an object, which lets a
group carry settings as well as membership:

| Key | Meaning |
|---|---|
| `benchmarks` | the list, same as the plain-list form |
| `like` | inherit another group's list, so the two cannot drift apart |
| `limit` | instances per benchmark, applied unless `--limit` is passed |
| `description` | appended to the log line and to `benchmark_selection` |

`smoke` uses `like: default` rather than repeating the five names, so editing
`default` keeps the smoke check honest automatically.

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
| `popqa` | test | 14,267 | 1 | `accuracy` under two scorers: `containment` (primary) + `squad_exact_match` |
| `triviaqa` | validation | 17,944 | 1 | `accuracy` under two scorers: `windowed_containment` (primary) + `squad_exact_match` |

Five of the nine are multiple choice and report a single accuracy. Four are
generative and report **two** scores each, because a single number cannot say
both "was it right" and "how close was it".

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
| `naturalqs` | `f1` (primary) | `drop_f1` | generative |
| | `accuracy` | `drop_exact_match` | |
| `jeopardy` | `f1` (primary) | `f1` | generative, SQuAD-style |
| | `accuracy` | `squad_exact_match` | |
| `popqa` | `accuracy` (primary) | `containment` | generative; gold answer appearing anywhere in the generation |
| | `accuracy` | `squad_exact_match` | same metric name, second scorer |
| `triviaqa` | `accuracy` (primary) | `windowed_containment` | generative; gold answer in the first 100 characters, lowercase only |
| | `accuracy` | `squad_exact_match` | same metric name, second scorer |

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

A full nine-benchmark sweep is roughly **55,400 instances and about 103,300 vLLM
prompts** per checkpoint. Prompts exceed instances because multiple-choice scoring
sends one prompt per answer choice, and HellaSwag alone accounts for about 40,000
of the total (10,042 instances at four choices). The prompt count is what drives
runtime, and it multiplies by every checkpoint in the sweep.

The two `factual` benchmarks are the largest by instance count — `triviaqa` at
17,944 and `popqa` at 14,267 — but being generative they issue one prompt each,
so they add less runtime than their share of the instances suggests. `--group
factual` runs just those two, at 32,211 instances and the same 32,211 prompts.

Generative prompts are not directly comparable to multiple-choice ones, though:
each one generates tokens rather than scoring a fixed continuation, so it is
slower per prompt. Treat the prompt count as a within-kind comparison.

`--dry-run` prints both figures, per checkpoint and for the whole sweep, before
anything spends. Use `--latest N` to cap a trial run.

## The `--limit` trap

`--limit N` is **not** a safe way to shrink a trial run. On `hellaswag` and
`socialiqa` a limit switches the task to loading validation *and* train and
sampling from the union, so a limited run scores a different population than an
unlimited one — in both directions, since the sampled set is neither the
evaluation split nor a superset of it.

Those two are flagged in the registry's `limit_unsafe` table, and the script
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
  -t hellaswag -t piqa -t arc_easy -t csqa -t socialiqa \
  -t naturalqs -t jeopardy -t popqa -t triviaqa \
  -O "${OUT}"
```
