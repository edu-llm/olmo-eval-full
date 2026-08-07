# `calibrated_datasets/` — vendored unidimensional IRT banks

One directory per dataset, each holding everything the `uni_mcq` CAT style needs to
administer an adaptive test with **no network access and no conversion at run time**.

```
calibrated_datasets/<dataset>/
  params.json     # IRT item parameters, already in (a, b, c) form
  items.jsonl     # question stems, choices, gold index
  manifest.json   # provenance and drop accounting
```

Produced by
[diagnostics/mcq_cat/styles/uni_mcq/scripts/vendor_bank.py](../diagnostics/mcq_cat/styles/uni_mcq/scripts/vendor_bank.py),
run **offline by a developer**, never on the eval box. The artifacts are committed.

Where this harness presents or grades a benchmark differently from the `olmo_eval` task
its items came from, the difference is recorded per dataset in
[DEVIATIONS.md](DEVIATIONS.md). Read that before comparing a theta against a published
one.

## Three of these were rebuilt, and they are not equally trustworthy

`hellaswag/`, `winogrande/` and `gsm8k/` were blocked until recently. Their parameters
were always fine; the bridge that decided which item each parameter row describes was
not. All three came from `Inputs/ATLAS/scripts/build_atlas_idx_bridge.py`, which assumes
ATLAS index *k* is the *k*-th instance `olmo_eval` enumerates from the HuggingFace split,
and that assumption is false — the ATLAS release's indices are Open LLM Leaderboard v1
example order, a different order over the same items.

All three now use bridges rebuilt from that order, committed under
[`styles/uni_mcq/bridges/`](../diagnostics/mcq_cat/styles/uni_mcq/bridges/) and read at
vendoring time through `DatasetSpec.bridge_in_repo`. The recovery was validated on
ARC-Challenge first, the one benchmark with a bridge the release shipped and therefore
the only known answer available: it reproduces that bridge on 1,172 of 1,172 question
stems. `arc_challenge` keeps the shipped bridge regardless, because it is the reference
every other bank is judged against and the reproduction would gain it one item in 650.

**How well each repair is attested differs, and the difference matters more than the
fact of it.** The acceptance test is the p-value-against-difficulty correlation described
below, run by
[check_bridge_alignment.py](../diagnostics/mcq_cat/styles/uni_mcq/scripts/check_bridge_alignment.py):

| Bank | Old bridge | Rebuilt | Scrambled control | Reading |
|---|---|---|---|---|
| `hellaswag` | −0.036 | **−0.538** | −0.00 ± 0.03 | clears ARC's own −0.319 on the same four checkpoints |
| `winogrande` | +0.037 | **−0.177** | +0.00 ± 0.04 | right sign, ~5σ, but below the ≈−0.27 predicted |
| `gsm8k` | untestable | untestable | — | rests on the ARC control alone; no responses exist |

For `winogrande`, prefer the standard error and the stop reason to the point estimate.
For `gsm8k`, nothing about *this bank* has been measured — the ARC control says the
procedure that produced its bridge is sound, and that is a different claim.

The per-dataset evidence is in
[datasets.py](../diagnostics/mcq_cat/styles/uni_mcq/datasets.py). `resolve()` still
consults a spec's `blocked` **before** it looks for artifacts, and the order still
matters: these files parse and load, so a resolver that checked the disk first would run
a bank that had been taken out of service.

## GPQA's three subsets are one question pool, so its bank is deduplicated

GPQA's `diamond`, `main` and `extended` are not different content. Per the GPQA paper
([arXiv 2311.12022](https://arxiv.org/abs/2311.12022), §2.3 and Table 2) they are
**nested quality filters over a single pool of 546 questions**:

| Subset | Questions | What the filter does |
|---|---|---|
| `extended` | 546 | everything that survived expert validation |
| `main` | 448 | removes questions both experts answered incorrectly and all three non-experts answered correctly, adding back some where the second expert made a demonstrable mistake |
| `diamond` | 198 | requires both experts correct and a majority of non-experts incorrect |

So diamond ⊂ main ⊂ extended, and 198 + 448 + 546 = **1,192** — exactly the row count of
the calibrated bank, which is arithmetic rather than coincidence. Enumerating the three
registered tasks confirms it exactly: 546 distinct questions over 1,192 instances, 198
appearing three times, 250 twice and 98 once, with no repeat inside any one subset.

ATLAS calibrated that union as though the three subsets held independent items, so every
diamond question was calibrated three times and every main-only question twice. After the
`a > 0` filter, **579 rows are only 395 distinct questions**: 237 calibrated once, 132
twice, 26 three times, so 184 of them are repeat calibrations.

**Why this is worse than redundancy.** The CAT masks administered items by id, and the
copies carry different ids. One session can therefore administer the same question two or
three times, score it each time, and hand EAP each scoring as independent evidence about
the checkpoint. Fisher information scales with `a²`, so a copy that happened to get an
inflated discrimination is *preferentially* selected: `diamond|100` carries `a = 28.020`
against its `main` twin's `0.993`.

**The copies disagree, and systematically.** `atlas_idx` was assigned over the
lexicographically sorted composite ids, so all of `diamond|…` sorts first and occupies
indices 1–198. Calibration ran in twelve ~100-item chunks
(`calibration/irt_item_parameters_101.csv`, `201`, … `1101`, `1193`), which means diamond
was fit almost entirely inside its own first two chunks, separately from its extended and
main twins several chunks away, with mean-σ linking only approximately reconciling the
chunk scales afterwards. The wild value is nearly always the diamond copy:

| Same question | difficulty `b` |
|---|---|
| `diamond\|91`, `extended\|314` | **182.58** vs 1.85 |
| `diamond\|76`, `extended\|265`, `main\|200` | **43.99** vs 0.52 vs 0.21 |
| `diamond\|117`, `extended\|400`, `main\|313` | **41.23** vs 1.25 vs 0.75 |
| `extended\|384`, `main\|298` | 35.33 vs 4.95 |

**The rule: keep Extended, else Main, else Diamond.** For each distinct question,
vendoring keeps the extended calibration if it survived the `a > 0` filter, else the main
one, else the diamond one. That prefers the fit estimated over the widest chunk span and
the largest response set, and it is deterministic and recorded rather than a judgement
made per item. The precedence lives in `DatasetSpec.duplicate_question_precedence`, is
declared for no other dataset, and is copied into `manifest.json` beside the distribution
it produced. The 184 removed rows are accounted for as a `duplicate_question` drop, so
the manifest still balances against all 1,192 upstream rows.

| | `extended` | `main` | `diamond` | total |
|---|---|---|---|---|
| bridge rows | 546 | 448 | 198 | 1,192 |
| after `a > 0` | 252 | 228 | 99 | 579 |
| after deduplication | 252 | 119 | 24 | **395** |

`extended` is unchanged across the last two rows because it is the most preferred tier.
The 24 surviving `diamond` items matter most: they are questions whose extended *and*
main calibrations both failed `a > 0`, so the precedence keeps the diamond copy rather
than losing the question.

**This is deliberately a different rule from the one HellaSwag got**, and the
inconsistency is the point rather than an oversight. There, two *different* questions
collided on one bridge id and there was no basis to prefer either, so every claimant was
dropped — see `ambiguous_item_id` below. Here it is the *same* question calibrated
repeatedly, the copies are distinguishable by how they were fit, and dropping all of them
would discard the question along with them.

Deduplication runs *after* the overlap guards, not before. Those guards measure whether
the bridge still names the questions it was built against, and every removed row joined
perfectly well; running the rule first would push `diamond` below its own per-subtask
floor and abort a healthy join.

Removing the repeats **sharpened** the alignment statistic rather than weakening it,
which is what a bank shedding redundancy without shedding information should do:
`check_bridge_alignment` reads Spearman −0.754 over the 395 items against a scrambled
control of +0.002 ± 0.047, where the 579-row bank read −0.746 against −0.001 ± 0.042.
Pearson moved much further, −0.26 to −0.39, because the copies removed are the ones
carrying the implausible `b` magnitudes.

**Detecting the repeats needs the dataset row, not the rendered stem.** A content hash
over question-plus-choices finds only 7 of the 184, because `GPQATask.process_doc`
reshuffles the choices per subset, so the same question under two labels usually hashes
differently. Splitting the rendered stem on its first option letter happens to give the
right answer on GPQA and is still the wrong tool: on BBH the same regex reports 2,957
phantom duplicates, because BBH's frozen few-shot exemplars contain option letters of
their own. Vendoring instead records each instance's own `question` while it is
enumerating — before any choice block or system prompt is rendered into the stem — and
groups on that, asserting that no two rows of one subset ever share a question.

## Why this exists

The upstream calibration output is not directly usable. It is in mirt's
parameterization, keyed by a positional column index, and carries no question text.
Three fragile transformations stand between it and a runnable bank: converting the
parameters, joining the bank index to the task's item ids, and enumerating the item
stems and choices.

Doing that once, offline, means it can be inspected, diffed and reviewed. Doing it on
every GPU run would mean re-downloading datasets, re-deriving the join, and
discovering a zero-overlap failure only after a checkpoint had been staged.

## Formats

The formats are dictated by the frozen loaders in `diagnostics/mcq_cat/common/`.
Both artifacts load through that code **unmodified**.

### `params.json` — JSON, not JSONL

[`load_irt_params`](../diagnostics/mcq_cat/common/irt_params.py) calls `json.loads`
on the whole file, so this is a single JSON document: either a list of records or an
object keyed by item id. This project writes a list.

```json
[
  {
    "item_id": "Mercury_7175875",
    "difficulty": -0.727,
    "discrimination": 3.925,
    "guessing": 0.771,
    "metadata": {"atlas_idx": 1}
  }
]
```

| Key | Meaning | Notes |
|---|---|---|
| `item_id` | Joins to `items.jsonl` and to the task's `metadata["id"]` | required |
| `difficulty` | IRT `b` | required |
| `discrimination` | IRT `a`, scalar for unidimensional | defaults to `1.0` |
| `guessing` | IRT `c`, the lower asymptote | defaults to `0.0`, so a 2PL bank omits it |
| `metadata` | Free-form; carries the upstream `atlas_idx` | optional |

### `items.jsonl` — JSONL, one record per line

[`load_items_from_jsonl`](../diagnostics/mcq_cat/common/benchmark_download.py)
parses one record per line.

```json
{"id": "Mercury_7175875", "question": "...", "choices": ["...", "..."], "gold_index": 2}
```

| Key | Meaning | Notes |
|---|---|---|
| `id` | Must match an `item_id` in `params.json` | defaults to the row index |
| `question` | The rendered stem | required |
| `choices` | List of answer strings | required |
| `gold_index` | Index of the correct choice | required (or `answer`) |
| `metadata` | Free-form | optional |

`items.jsonl` contains **only** items that survived the bank filter and the id join,
so the two files always describe the same item set.

#### Generative items

A generative benchmark has no choices to rank, so its records look different:

```json
{"id": "0", "question": "Natalia sold clips to 48 of her friends...", "choices": [], "gold_index": -1,
 "metadata": {"gold_answer": "72", "answer_type": "numeric", "modality": "generative"}}
```

The empty `choices` and `gold_index` of `-1` are **load-bearing**. `load_items_from_jsonl`
copies the record into a `BenchmarkItem` verbatim, and an item with no choices is how
the run time knows to sample a completion and match an extracted answer rather than
rank continuation log-likelihoods over a choice set that does not exist.

| Metadata key | Meaning |
|---|---|
| `answer_type` | Which entry of `ANSWER_GRADERS` decides the item; declared per dataset |
| `modality` | Always `generative` on these records; MCQ records carry no `metadata` at all |
| `gold_answer` | The string a gold-matched grader compares against; absent for IFEval |

`answer_type` is a property of the **benchmark**, not of the individual answer. GSM8K
items carry `numeric`, MATH items `math_latex` — including the ones whose gold happens
to read as a bare number, since grading half a bank by exact match and half by symbolic
equivalence would put two conventions inside one calibrated scale — and IFEval items
`ifeval_strict`.

**IFEval items carry no gold answer at all.** Its prompts state verifiable
instructions, and a response is correct when every one of them passes, so the record
carries the constraints instead:

```json
{"id": "0", "question": "Write a 300+ word summary ... Do not use any commas ...", "choices": [], "gold_index": -1,
 "metadata": {"answer_type": "ifeval_strict", "modality": "generative",
              "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
              "kwargs": [{}, {"relation": "at least", "num_words": 300}]}}
```

Which extra keys a record needs is declared by the grader itself, in
`ItemGrader.required_metadata`, and vendoring copies exactly those — so a grader cannot
acquire a new input without the bank that feeds it carrying that input too. Grading
IFEval needs the `ifbench` verifier registry installed; vendoring it does not.

The IRT layer is unchanged in every case: it only ever sees a binary
correct/incorrect, so `modality` selects a grading scheme and `answer_type` selects a
grader within it, and nothing else changes. Both are declared in
[datasets.py](../diagnostics/mcq_cat/styles/uni_mcq/datasets.py) and stamped into
`manifest.json`.

### `manifest.json` — provenance

Not read by the frozen loaders; read by the style so provenance reaches the report.

```json
{
  "dataset": "arc_challenge",
  "task": "arc_challenge",
  "route": "C",
  "modality": "mcq",
  "answer_type": null,
  "fit_family": "3pl",
  "source_ref": "origin/Research",
  "source_commit": "6998270f51af47ffc5623ced960461ef28557118",
  "bank_dir": "AdaptiveTesting/Inputs/ATLAS/arc",
  "bridge_path": "AdaptiveTesting/Inputs/ATLAS/arc/atlas_idx_to_question_id.csv",
  "bridge_kind": "atlas",
  "bridge_in_repo": false,
  "bridge_provenance": null,
  "upstream_bank_rows": 839,
  "bridge_rows": 1172,
  "items": 650,
  "dropped": {
    "non_positive_discrimination": 189,
    "non_finite_discrimination": 0,
    "not_in_bridge": 0,
    "not_in_task": 0,
    "ambiguous_item_id": 0
  },
  "ungradable_instances": 0,
  "positional_ids": false,
  "scoring_convention": {
    "recorded_by": "vendor_bank",
    "runtime": {
      "modality": "mcq",
      "prompt_style": "question_answer",
      "num_fewshot": 0,
      "score_normalization": "unnormalized_sum_of_continuation_logprobs",
      "max_length": null
    },
    "calibration": {
      "prompt_style": "unrecorded",
      "num_fewshot": 25,
      "metric": "unrecorded",
      "note": "..."
    }
  },
  "sha256": {"params.json": "...", "items.jsonl": "..."},
  "generated_at": "2026-08-06T00:00:00+00:00"
}
```

`modality` and `ungradable_instances` were added when generative datasets arrived, and
`answer_type` when the second and third of them did, so banks vendored before each do
not carry them. Nothing requires any of the three: the absence of `modality` means
`mcq`, and the absence of `answer_type` means the numeric default. `answer_type` is
`null` on an MCQ bank, which has no generative grader to name.

`frozen_prompt` and `bank_caveat` arrived with `bbh` on the same terms. The first says the
stems are whole rendered prompts rather than bare questions; the second is what a reader
has to know before using the numbers, and it is `null` on every bank whose measurements
speak for themselves. Only `bbh` carries a caveat today, and it says to read the predicted
accuracy and distrust the theta.

`duplicate_question_precedence` and `subtask_items` arrived with GPQA's deduplication, and
are `null` on every bank that does not deduplicate. They are recorded together because
either alone invites the wrong reading: the first says which calibration of a repeated
question was preferred, the second is what that rule left behind, and a subtask count that
has fallen is otherwise indistinguishable from a subtask that joined badly.

`fit_family` is **authoritative**. It records the family the parameters were
*estimated under*, and the style reads it from here rather than from any flag. A
supplied value that disagrees raises.

### `scoring_convention` — what this bank is to be scored under, and what it was fit under

The two halves are different claims and the block keeps them apart, because conflating
them is what lets a wrong number look right.

`runtime` is what this harness does, resolved from
[config.yaml](../diagnostics/mcq_cat/styles/uni_mcq/config.yaml) by the same code a run
resolves it with. It is **checked at startup**: the runner compares it against the
settings the scorer is about to be built from, before the checkpoint is fetched, and a
difference raises. That covers the presentation and grading fields — prompt style, shot
count, few-shot block, chat format, stop sequences, token budget, prompt-length cap,
grader, score normalization — on the rule that each of them can change whether an
individual item is answered correctly. `max_items`, `min_items`, `se_threshold`,
`batch_size`, `device_map`, `checkpoint_kind` and `seed` are deliberately absent: they
move how precisely an ability is measured and on what hardware, not the scale it is
measured on, and the report already records the first three. `min_items` is per dataset —
8 everywhere except `bbh`, which asks for 24 — and keeping it out of the checked half is
what lets that floor move without invalidating a bank vendored under the old one.

`calibration` is what the bank's difficulties were estimated against, and it is **never
checked**, because a difference there is a property of the bank rather than something a
run can fix. Where nothing upstream records a fact the field says `"unrecorded"` rather
than a plausible reconstruction, which is the useful answer: it means a theta from this
bank cannot be reconciled with a published one. Only `arc_challenge` has a recorded shot
count (25), only `leaderboard_math`, `ifeval`, `musr` and `bbh` a recorded metric — the
first a live deviation, the other three matched exactly. `DEVIATIONS.md` is the prose
reading of the same facts.

`score_normalization` is in the checked half and is per dataset, which is easy to miss
because three of the five MCQ banks take the same value. `arc_challenge`, `hellaswag` and
`winogrande` are ranked by the unnormalized sum their tasks declare; `musr` and `bbh` are
ranked by `acc_norm`, the same sum divided by the continuation's length, because that is
the binary Open LLM Leaderboard v2 scored them under and the binary their banks were fit
on. The two rules order a choice set differently, so a MuSR run resolving the default is
refused at startup rather than reported.

`num_fewshot` is in the checked half too, and it is per prompt style rather than fixed at
0. Four of the five MCQ banks are scored behind a style that frames the item's stem and
nothing else, so they record 0 and always have. `bbh` records 3, because its stems are
whole rendered prompts: `leaderboard_bbh` is 3-shot behind a description that differs per
*subtask*, which no per-dataset setting can express, so vendoring calls the task's own
`format_request` and freezes description, exemplars and answer cue into the stem — the
same treatment `gpqa` gives its shuffled choice block, and visible in the manifest as
`frozen_prompt`. Recording 0 there would describe a prompt no model is shown, and
recording 3 is what refuses a config edit that swapped the dataset onto a stem-framing
style and quietly dropped the prefix.

`recorded_by` distinguishes a block written alongside the artifacts from one added
afterwards by
[migrate_manifests.py](../diagnostics/mcq_cat/styles/uni_mcq/scripts/migrate_manifests.py),
which is how the six banks vendored before the block existed acquired one. A migrated
block describes the config as it stands now, since the config in force when those banks
were produced was never written down.

A bank with no block at all is refused rather than run, and the message points at the
migration.

## Reading these numbers

**`upstream_bank_rows` is smaller than `bridge_rows`, and that is expected.**
Calibration drops items that fail to converge, leaving sparse `X` indices, while the
bridge covers the whole evaluation split. ARC is 839 calibrated of 1,172 in the split.

**`items` is smaller again, and the `dropped` breakdown must account for the gap.**
Kept items plus every drop reason always equals `upstream_bank_rows`; a test asserts it.
The two drop reasons that carry real weight:

- `non_positive_discrimination` — a negative or zero `a1` means the item scores
  backwards. Research's own loader drops these, and it is not a rounding error: 189 of
  ARC's 839 rows and 556 of HellaSwag's 5,600. Since Fisher information scales with `a`
  squared, leaving one in would actively corrupt item selection.
- `ambiguous_item_id` — **a bridge is not guaranteed to be injective.** Two separately
  calibrated rows, with different `(a, b, c)`, can claim the same id, and nothing
  downstream would complain: `load_irt_params` builds a dict keyed by item id, so the
  later record silently overwrites the earlier one, and `BenchmarkBank.get` returns the
  first match — the CAT would score one item using another's parameters and still report
  a confident ability. Since no row can be shown to own a contested id, **every claimant
  is dropped**. This is currently zero everywhere. It used to cost HellaSwag 204 rows
  across 102 ids, when its bridge keyed on the native `ind`: the validation split
  concatenates two sub-splits whose `ind` values each restart, so 10,042 rows carry only
  9,609 distinct values and `ind == 180` names two different questions. Its rebuilt
  bridge keys on a content hash instead, which is injective by construction, and the 204
  came back.
- `duplicate_question` — **the same question calibrated more than once**, under a
  different id each time, which is the opposite failure and takes the opposite
  resolution: one copy is kept by a declared precedence rather than all being dropped.
  Only GPQA has it, for 184 of its 579 filtered rows, and the section above is why.

So HellaSwag lands at 5,044 usable items of 5,600 calibrated, ARC at 650 of 839,
WinoGrande at 865 of 1,045, GSM8K at 1,298 of 1,306, MATH-Hard at 1,183 of 1,206, IFEval
at 511 of 535, GPQA at 395 of 1,192, MuSR at 432 of 754 and BBH at 3,965 of 5,761. Every
bank but GPQA loses items only to non-positive discriminations: each bridge is injective
and every surviving bank row found its item.

GPQA and MuSR are the outliers and the number is worth pausing on. Under the `a > 0`
filter alone GPQA discards 51% and MuSR 43%, which is not attrition, it is most of the
benchmark failing to discriminate under a 3PL fit, and both have a 2PL refit upstream that
keeps far more (GPQA's keeps 873 of 1,192). We stay on 3PL by policy; the consequence is a
smaller bank with a shorter dynamic range, recorded per dataset in `datasets.py`. GPQA
then loses another 184 to deduplication and lands at 395, and those two reductions should
be read differently: the first is items whose difficulty the data does not support, the
second is items that were never separate questions. BBH's 31% sits between those two and
the ATLAS banks, and on BBH the filter is the thing that makes the bank work at all:
full-bank theta recovers accuracy at r = 0.69 unfiltered and 0.86 filtered.

**`ungradable_instances` is not part of that sum.** It counts *task* instances skipped
during enumeration — an MCQ instance with no choices, a generative one with no gold
answer — not calibrated bank rows. A skipped instance already reappears in `dropped` as
`not_in_task` if the bank happened to calibrate it, so adding the two would double-count.

**A join that names nothing about the item is the fragile one**, and two kinds do.
When `positional_ids` is true the bank is keyed to a position in the task's enumeration,
so a change in enumeration order silently rekeys every item rather than failing; GPQA and
BBH are the two that still join that way, and both match at 100%. A `content_hash` join is
keyed to the item's own text, which cannot be rekeyed by a reorder but *can* miss
wholesale if the task changes how it renders an item. Both need the same guard, which
`needs_overlap_floor` supplies: the bank's max index must equal the bridge's row count,
and at least 90% of bank items must find a task item.

**A bank spanning several tasks gets that guard per subtask.** GPQA, MuSR and BBH each
draw from three or more registered tasks, so one task drifting while the others hold would
leave the dataset-wide overlap comfortably above 90% — `check_subtask_overlap` applies the
floor to each subtask separately for that reason, and it is the real guard on these three.
It matters most on BBH, whose 24 subtasks mean any one of them holds under a fifteenth of
the bank and could vanish entirely inside a dataset-wide 90%.

**How a subtask is named depends on which key the bridge uses.** GPQA and BBH are keyed
by `<subtask>|<position>`, rebuilt by counting each registered task's enumeration from
zero, so the label falls out of the id and a second rule applies beside the floor: a
subtask's positions must be distinct and the task must enumerate exactly one past the
highest of them. That rule used to demand a gapless `0..n-1` run, which MuSR's
`object_placements` cannot satisfy — 254 bridge rows over a 256-long split, positions 136
and 140 dropped during calibration — and sparseness there is upstream's right rather than
a fault. MuSR, IFEval and MATH-Hard are keyed by content hash instead, which removes the
position from the key and the subtask with it: the label comes from a `subtask` column
their bridges carry and is copied onto each `params.json` record, and there is no
enumeration count to check because a task that dropped a row now loses that row rather
than rekeying everything after it. MATH-Hard needs that column even though its spec names
a single task, because one `leaderboard_math` task concatenates seven MATH subjects and
the composite id used to be the only thing naming which. See
`styles/uni_mcq/bridges/README.md` for why GPQA and BBH keep the composite.

**Full overlap is not evidence that the join is right, and on four banks here it was
not.** What the guard cannot catch is a join against a *differently ordered copy* of the
same items: every row matches something, the floor is cleared with room to spare, and
every item is the wrong one. That is precisely the failure `hellaswag`, `winogrande`,
`gsm8k` and `leaderboard_math` turned out to have — all four cleared the floor
throughout. It is also invisible
downstream. A permuted bridge selects items by a Fisher information computed from the
wrong difficulties, collapses the standard error on schedule, and reports a theta that is
noise with a healthy interval printed beside it. So the bridge a spec names matters more
than the floor it clears, and the check that actually settles it is external: correlate
each item's p-value against the bank's implied `b = -d/a1` and expect it strongly
negative. ARC's verified join gives −0.86; a scrambled one gives ~0. That check is
[check_bridge_alignment.py](../diagnostics/mcq_cat/styles/uni_mcq/scripts/check_bridge_alignment.py),
and it is what a rebuilt bridge has to pass before its bank is used.

The Route B composite key is the same hazard scoped to a subtask, and all five have now
been measured against it: each bank's `doc_id` was matched back to the document Open LLM
Leaderboard v2 recorded evaluating. MuSR, IFEval, GPQA and BBH came out agreeing with the
positional assumption on every document, so those four banks are unchanged and what
improved is the evidence. `leaderboard_math` did not — its composite named the right
question on 8 of 1,324 positions — so its 1,183 items are the same count over largely
different problems, and anything computed from it before 2026-08-07 was measured against
difficulties belonging elsewhere.

Note what `check_bridge_alignment` can and cannot say for these banks. The calibration
matrix's columns are the same positional index as the parameter rows, so it attests the
difficulties against the responses they were fit from and says nothing about which
question a column is. `leaderboard_math` read −0.731 both before and after its re-key,
which is that limitation demonstrated rather than argued.

IFEval is the near miss that was avoided rather than discovered late. The `ife_NNNN` ids
in the eduLLM-Evals scenario dump were assigned after sorting the prompts by integer key,
while the dataset's native order is lexicographic by the key's string form, so item `0`
is key 1000 and is `ife_0049`. Both id spaces are dense 0–540 over the same 541 prompts,
so that join would have found ~100% overlap and produced a well-formed bank of mismatched
items. It is vendored from a content-hashed bridge instead, under which the wrong
ordering joins nothing rather than joining everything wrongly.

**The pIRT denominator is this bank, not the benchmark.** Predicted accuracy is
computed over the items in `params.json`. For HellaSwag that is 5,044 items, not the
10,042 in the split, so it is predicted accuracy over the usable calibrated subset.

**Theta is anchored to the calibration population**, so it compares across checkpoints
of one run and against that population, but is not an absolute score. It is also not
comparable across banks — `manifest.json` records which bank produced it.

## Adding a dataset

A dataset must first be listed in
[datasets.py](../diagnostics/mcq_cat/styles/uni_mcq/datasets.py), which encodes the
four admission criteria: the bank came from Route B or C, a grader exists for how the
benchmark is answered (`modality`), a task definition exists on this branch, and the
bank is usable. Its entry also carries a `CalibrationConvention` recording what is known
about how its responses were prompted and scored, `"unrecorded"` where nothing is, and
`config.yaml` must pin the presentation it will be run under. Then:

```bash
python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.vendor_bank \
    --dataset arc_challenge --source-ref origin/Research
```

If the dataset's bank is keyed by nothing but an ATLAS column index and no bridge shipped
with it, build one from Open LLM Leaderboard v1 example order rather than by walking the
HuggingFace split — see
[styles/uni_mcq/bridges/](../diagnostics/mcq_cat/styles/uni_mcq/bridges/) — and run the
alignment check before trusting the result.
