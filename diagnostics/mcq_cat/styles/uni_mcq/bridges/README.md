# `bridges/` — index bridges rebuilt from what the leaderboard recorded evaluating

A **bridge** answers the only question a calibration output does not: column `X<k>` holds
a difficulty, and which question is that? Get it wrong by a permutation and every row
still joins, overlap still reads 100%, the CAT still converges, the standard error still
collapses on schedule, and the reported ability is noise.

Every bridge here replaces one whose key was a **position in an enumeration performed
somewhere else** — either a bare column index, or a `<subtask>|<doc_id>` composite that
is the same thing scoped to a subtask. A position is only a usable key if the enumeration
can be reproduced exactly, and that is an assumption rather than a measurement. It has
now been measured on all eight banks that had one, and it was false on two: a positional
bridge disagrees with the one ARC shipped on 1,170 of 1,172 positions, and
`leaderboard_math`'s composite named the right question on 8 of 1,324.

The replacement key is a content hash, so a re-rendered or reordered split **misses**
rather than silently naming its neighbour's question, and vendoring's overlap floor turns
a miss into an abort. Seven of the nine banks use it. The two that do not — `gpqa` and
`bbh` — keep a composite because a hash cannot separate their duplicated items, not
because their joins are unchecked; both were verified document by document against the
same leaderboard records, and the section below records what that found.

```
bridges/
  <dataset>.csv                # atlas_idx -> item_id, the artifact vendoring reads
  <dataset>.json               # provenance: which run the order came from, how ids derive
  arc_challenge.control.json   # the known-answer run that validates the v1 procedure
```

Written by
[`../scripts/build_leaderboard_bridge.py`](../scripts/build_leaderboard_bridge.py),
read by [`../scripts/vendor_bank.py`](../scripts/vendor_bank.py) through
`DatasetSpec.bridge_in_repo`.

## Two generations, two recoveries

| | v1 (Route C, ATLAS) | v2 (Route B, local fits) |
|---|---|---|
| Datasets | `hellaswag`, `winogrande`, `gsm8k` | `musr`, `ifeval`, `leaderboard_math`; `gpqa` and `bbh` keep a verified composite |
| Bad key | bare `atlas_idx` | composite `<subtask>` + `<doc_id>` |
| Details repo | `open-llm-leaderboard-old/details_<model>` | `open-llm-leaderboard/<model>-details` |
| File | one parquet per harness config | one JSONL per leaderboard task |
| Order comes from | **row position** in the parquet | the record's explicit **`doc_id` field** |
| Declared in | `LEADERBOARD_ORDER` + `LEADERBOARD_RECIPES` | `LEADERBOARD_V2` |

The row-position/`doc_id` difference is the trap. v2 sample files are written in
*completion* order, not document order — MuSR's `murder_mysteries` file opens with
`doc_id` 0, 8, 16, 24 — so a reader that took row order would produce a scrambled bridge
that passed every count and bijection check in the module. `read_details` keys on the
explicit field and refuses a record that lacks one.

## Why these are committed rather than regenerated

Regenerating one needs the Hub, an archived details file and the HuggingFace split all
reachable at the same moment, and the v2 repos are gated: an authenticated user has to
have accepted each repo's terms once before any file in it can be read. A join that can
only be reproduced under those conditions is a join nobody will reproduce, so the
generator stays committed beside the artifacts and the sidecar records everything needed
to re-run it.

## The control

`arc_challenge` has no CSV here on purpose. It is the only benchmark ATLAS shipped a
bridge for, so it is the only place the v1 recovery can be checked against a known answer,
and it keeps the shipped bridge — the reproduction exists to validate the procedure, not
to replace an independently attested artifact.

`arc_challenge.control.json` records that run: the recovered order reproduces the shipped
bridge on **1,172 of 1,172** question stems. Its only two id-level differences are
positions where the shipped bridge is itself wrong, ATLAS having resolved ids by matching
stem text onto two questions whose stems recur under a second id.

There is no v2 equivalent, because no Route B bank ships a bridge to be checked against.
What stands in its place is the **cross-model check**, which the build performs and will
not skip: the pinned run's `doc_id -> document` map is compared against three other
models' runs, and any disagreement aborts. `doc_id` is a position in an enumeration
lm-eval performed, so runs months apart agreeing on it is what makes it a property of the
benchmark rather than of one evaluation.

## Format

| Column | Meaning |
|---|---|
| `atlas_idx` | 1-based bank column; the bank's `X<k>` |
| `item_id` | The joinable key, `datasets.content_item_id` over the task instance |
| `subtask` | Which subtask the calibration drew the row from. v2 bridges only |
| `split_index` | 0-based row in the split the task enumerates, within the subtask; audit only |
| `question` | Truncated, whitespace-collapsed stem; audit only, **never an item source** |

Only `atlas_idx`, `item_id` and — for a multi-task bank — `subtask` are read.
`split_index` is what makes a permutation visible to a reader.

`subtask` is not decoration on a v2 bridge. Hashing the item is what removes the position
from the key, and it removes the subtask with it, so without this column the per-subtask
overlap floor has nothing to group by and one drifted subtask averages into the healthy
ones. Vendoring copies it onto each `params.json` record for the same reason.

### Why `item_id` is a hash

Because for HellaSwag no field of the dataset works. Its validation split concatenates
the `zeroshot` and `indomain` sub-splits, whose native `ind` values each restart, so
10,042 rows carry only 9,609 distinct values: `ind == 180` is both "Sharpening knives" at
position 8 and "How to become a sports announcer" at position 3260. Keying on it cost 204
items to the ambiguous-id drop.

WinoGrande, GSM8K and the Route B banks *do* have a unique field — a bare or composite
split position — and deliberately do not use it. A position is exactly the key whose
silent drift produced this whole situation, and the bridge build already establishes the
content match, so recording a content-addressed id costs nothing and makes the join
self-verifying.

The derivation is `datasets.content_item_id` — SHA-256 over the instance's question and
choices, each NUL-terminated, first 16 hex digits. Question *and* choices, because 22
HellaSwag contexts recur with different endings; NUL-terminated, so `("ab", "c")`
and `("a", "bc")` cannot collide. It hashes the **task's** rendering, which means a change
to how the task presents an item changes the id and drops the row rather than quietly
scoring it behind a prompt it was not matched under.

## Re-running an existing bridge

```bash
# The v1 control. Writes only arc_challenge.control.json; ARC keeps the shipped bridge.
python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.build_leaderboard_bridge \
    --dataset arc_challenge

# A v1 bridge. Aborts rather than writing if the match is not a bijection over the split.
python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.build_leaderboard_bridge \
    --dataset hellaswag

# A v2 bridge. Also aborts if two leaderboard runs disagree about the ordering.
python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.build_leaderboard_bridge \
    --dataset musr

# Then re-vendor and re-run the acceptance test.
python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.vendor_bank --dataset musr
python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.check_bridge_alignment --dataset musr

# gpqa and bbh have a LEADERBOARD_V2 entry and deliberately no CSV. Running either
# performs the whole verification and then aborts on the colliding ids, which is how
# their composite key is re-checked.
python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.build_leaderboard_bridge \
    --dataset gpqa --dry-run
```

Expected results, and what a rebuild has to reproduce before it is believed:

| Dataset | Rows | Positions matching enumeration order | Spearman(p, b) |
|---|---|---|---|
| `arc_challenge` | 1,172 | 2 | −0.319 over the four shared checkpoints, −0.857 over ATLAS's matrix |
| `hellaswag` | 10,042 | 3 | **−0.538** over 910 items, null −0.00 ± 0.03 |
| `winogrande` | 1,267 | 1 | **−0.177** over 865 items, null +0.00 ± 0.04 |
| `gsm8k` | 1,319 | 2 | no response data exists on any ref |
| `musr` | 754 | 756 of 756 | **−0.616** over 432 items, null +0.00 ± 0.05 |
| `ifeval` | 535 | 541 of 541 | **−0.888** over 511 items, null −0.00 ± 0.04 |
| `leaderboard_math` | 1,206 | **8 of 1,324** | −0.731 over 1,183 items, null +0.00 ± 0.03 |

MuSR and IFEval are the ones where the recovery agreed with the assumption it replaced:
every `doc_id` equals the position a fresh enumeration assigns, so the re-keyed bridges
name exactly the items the old keys did and both banks are unchanged, at 432 and 511.
That is a result, not a formality — the same recovery on ARC found the assumption wrong
on 1,170 of 1,172 positions — and it is why the count is reported rather than tuned to.

**`leaderboard_math` is the one that was wrong.** Its task numbers each subject's
Level-5 subsequence of `DigitalLearningGmbH/MATH-lighteval` from zero while lm-eval's
`doc_id` counts the pre-filtered `lighteval/MATH-Hard` split, so the two are unrelated
orderings of the same 1,324 problems and the composite key agreed with the truth on 8 of
them. The bank still holds 1,183 items at 100% overlap and only **5 of those 1,183**
describe the question they described before: 123 problems left the bank, 123 entered it,
and 1,055 moved to different columns. Its Spearman is unmoved at −0.731 across that
change, which is the sharpest available demonstration of what that statistic does not
attest — see the warning under point 5 below.

Read the per-dataset entries in [`../datasets.py`](../datasets.py) before quoting any of
these: the banks are not equally well attested.

## The two banks that keep a composite key

`gpqa` and `bbh` were taken through the same recovery and keep
`item_id_map.csv`. That is a stop rather than an omission, and the reason is the **id**
rather than the join — both joins came out verified:

| Dataset | Documents | Positions matching enumeration order | Why the content key fails |
|---|---|---|---|
| `gpqa` | 1,192 | 1,192 of 1,192 | Nested subsets: 448 questions appear under two or three subtask labels, and `process_doc` reshuffles the choices per subset, so 32 pairs hash alike |
| `bbh` | 5,761 | 5,761 of 5,761 | Four items are literal duplicates within their subtask, and one hardcoded choice set per subtask cannot separate them |

Both were cross-checked against the same three models spanning September 2024 to
February 2025, agreeing on every document. So `<subtask>|<doc_id>` on these two names
the question the leaderboard evaluated, measured rather than assumed — which is exactly
the standing `leaderboard_math`'s composite turned out not to have.

Under a content hash both collisions are dropped as ambiguous at vendoring time, so
re-keying either would silently lose calibrated items to an accident of a shuffle or of
upstream's duplicates. A composite distinguishes them by position, which here is the one
thing that does.

Keying `gpqa` this way is not the same as shipping its repeats, and the two decisions are
worth keeping apart. Its subsets are nested quality filters over one pool, so 184 of the
579 rows that survive the `a > 0` filter are a second or third calibration of a question
another row already carries — a hash of the *rendered* item catches only 7 of them,
because the per-subset shuffle usually separates them. Vendoring removes all 184 by
grouping on the dataset row's own question text and keeping one calibration per question,
which leaves a 395-item bank; `calibrated_datasets/README.md` sets out the precedence and
why it differs from HellaSwag's rule. The composite is what lets those rows be told apart
in the first place.

## Bringing a new Route B benchmark through

Each is one entry in `LEADERBOARD_V2` and a CLI invocation. Nothing else in the module
should need editing; if it does, that is a sign the entry is describing something the
table cannot express, and the table is the thing to extend. The five brought through so
far needed one shared helper each at most, and the exception is worth knowing about:
`from_instance` defaults to `question_and_choices`, and the two benchmarks whose choices
are not a statement about which question an instance is — `gpqa`, reshuffled per subset,
and `bbh`, one constant set per subtask — use `question_only` instead.

### 1. Look before declaring

```bash
python -m diagnostics.mcq_cat.styles.uni_mcq.scripts.build_leaderboard_bridge \
    --inspect --details-task leaderboard_bbh_snarks
```

Reports the resolved file, the document count, whether `doc_id` is dense, the record's
`doc` keys and a truncated first document. **Run this before writing anything**, because
what a `doc` holds is whatever lm-evaluation-harness handed the leaderboard and is not
guessable: gpqa's carries seventy-odd annotation columns beside `Question`, ifeval's
carries `prompt` next to its verifier arguments, and none of them is the prompt the model
saw — that is in the record's `arguments`, behind the model's own chat template.

The leaderboard task names, all confirmed present in the pinned run, and the index map
each bank joins through:

| Dataset | Leaderboard task | Index map on `origin/Research` | `doc` keys |
|---|---|---|---|
| `musr` | `leaderboard_musr_<split>` | `Experiments/openlm_atlas_3pl/musr/data/item_id_map.csv` | `narrative`, `question`, `choices`, `answer_choice`, `answer_index` |
| `ifeval` | `leaderboard_ifeval` | `Inputs/ATLAS/ifeval/atlas_idx_to_question_id.csv` | `prompt`, `key`, `instruction_id_list`, `kwargs` |
| `leaderboard_math` | `leaderboard_math_<subject>_hard` | `Experiments/openlm_atlas_3pl/math/data/item_id_map.csv` | `problem`, `solution`, `answer`, `type`, `level` |
| `gpqa` | `leaderboard_gpqa_<subset>` | `Experiments/openlm_gpqa_atlas_3pl/data/item_id_map.csv` | `Question`, `Correct Answer`, `Incorrect Answer 1..3`, `choice1..4`, `answer`, and ~60 more |
| `bbh` | `leaderboard_bbh_<subtask>` | `Experiments/openlm_atlas_3pl/bbh/data/item_id_map.csv` | `input`, `target` |

All paths are under `AdaptiveTesting/`. The four `item_id_map.csv` files carry
`atlas_idx,item_id,subtask,question_id`, which is what the defaults expect; ifeval's is
`atlas_idx,question_id` and needs the two overrides below.

### 2. Add the entry

```python
"bbh": LeaderboardV2Recipe(
    task_prefix="leaderboard_bbh_",
    index_map="AdaptiveTesting/Experiments/openlm_atlas_3pl/bbh/data/item_id_map.csv",
    doc_fields=("input",),
    from_record=lambda doc: (str(doc["input"]),),
    from_instance=question_only,
    enumeration=PER_SUBTASK_TASK,
    normalization="...",
),
```

The knobs, and what each is deciding:

- **`task_prefix`** — joined to a bridge subtask label to give the leaderboard task name.
  A benchmark evaluated as one task (`ifeval`) spells `leaderboard_` here and puts the
  label in `single_subtask`.
- **`index_map`** — the upstream file mapping `atlas_idx` to a subtask and a `doc_id`.
  Usually the fit's own `item_id_map.csv`. **`ifeval` is the odd one out**: its spec joins
  through `Inputs/ATLAS/ifeval/atlas_idx_to_question_id.csv`, whose `question_id` is a
  bare integer, so set `subtask_column=""` and `single_subtask="ifeval"`. Its ids run
  0..540 with six absent, because 541 prompts were harvested and 535 calibrated.
- **`from_record`** — the identity of one evaluated example, built from the details
  record's `doc`. It has to reproduce the **olmo-eval task's** rendering, not the
  leaderboard's prompt.
- **`from_instance`** — the same identity from an olmo-eval instance.
  `question_and_choices` covers a plain MCQ task; `question_only` is for the two whose
  choices say nothing about which question an instance is. `gpqa` shuffles its four
  options per question from the *task's* seed while the details record carries the
  leaderboard's own shuffle in `doc["choice1..4"]`, so the two orderings are unrelated
  and including either leaves every document unmatched. `bbh` attaches one hardcoded
  choice set to every item of a subtask, so within the split being matched the choices
  are constant; its vendored stem is separately the frozen 3-shot prompt while
  `doc["input"]` is the bare item, so the identity has to be the *instance's* `question`
  rather than its rendered stem. Neither is a loosened match — the bijection is required
  over the whole split, so a recurring stem aborts instead of picking a candidate — and
  the item id is computed separately from the matched instance, so it can be wider than
  the identity without either side being weakened.
- **`enumeration`** — `PER_SUBTASK_TASK` where `spec.subtasks` names one registered task
  per subtask (`musr`, `gpqa`, `bbh`); `WHOLE_TASK` for a single-task bank (`ifeval`);
  `COMPOSITE_ID` for `leaderboard_math`, whose one task concatenates seven subjects and
  emits `<subject>|<position>` as its own id.
- **`normalization`** — prose, copied verbatim into the sidecar. Say "none" if the two
  renderings match exactly. This is the one part of a bridge a later reader cannot
  recover by re-running anything, because a recipe that quietly loosened a comparison and
  one that reproduced a conversion look identical when both pass.

**Normalize toward the source's convention; never loosen the match.** The v1 work found
olmo-eval and lm-eval render HellaSwag stems differently by a single substitution, and
naive matching left 777 of 10,042 unmatched; reproducing lm-eval's own preprocessing gave
a perfect 10,042. The bijection guard is what makes that difference visible, so it stays
strict.

### 3. Build, vendor, verify

```bash
python -m ...build_leaderboard_bridge --dataset bbh --dry-run   # every check, no write
python -m ...build_leaderboard_bridge --dataset bbh
python -m ...vendor_bank --dataset bbh
python -m ...check_bridge_alignment --dataset bbh
```

Then flip the spec in [`../datasets.py`](../datasets.py): `bridge_path` to
`{_BRIDGES}/<dataset>.csv`, `bridge_in_repo=True`, `bridge_kind=CONTENT_HASH`, and
`positional_ids=False` — the last one matters, because `needs_overlap_floor` reaches the
floor through the `CONTENT_HASH` branch instead and leaving both set claims the join is
positional when it is not.

Budget a few minutes for `bbh`: the cross-check reads 24 subtasks across four models.

### What to check before believing the output

1. **`documents_matched == documents_evaluated`** in the sidecar. A partial match aborts,
   so this is a statement that no normalization was needed to reach it, not a rate.
2. **Every cross-checked model agrees on every document**, and their runs span more than
   one period. Two evaluations weeks apart agreeing says less than two a year apart.
3. **`positions_agreeing_with_enumeration_order`** against the document count. Equal
   means the old positional join happened to be right and the bank should come out
   unchanged; far below means it was wrong and the bank *will* change. Either is a
   legitimate result. What is not legitimate is adjusting anything to hold a count.
4. **The vendored item count, reported not targeted.** Reproducing the previous number
   exactly is worth investigating rather than celebrating — check the per-`atlas_idx`
   questions actually match before concluding the join was already right.
   `leaderboard_math` is why: it came out at 1,183 items before and after, and 1,178 of
   those 1,183 columns changed which question they describe.
5. **`check_bridge_alignment` similar or better**, with its scrambled control beside it.
   For a Route B bank this attests the parameters against the responses they were fit
   from and *not* the join, since the matrix columns are the same positional index as the
   parameter rows; the join is attested by the details match instead. Treat that as a
   warning rather than a caveat: `leaderboard_math` read −0.731 before its re-key and
   −0.731 after, across a change that replaced 123 of its items and moved almost all the
   rest, because both sides of the correlation move together under any permutation of the
   join.
6. **Theta still recovers monotonically** over the re-vendored bank. The absolute scale
   is the bank's; the ordering is what a checkpoint comparison rests on. Each bank's
   recovery test runs a real CAT over `calibrated_datasets/` with only the forward pass
   simulated, so re-vendoring is enough to re-check it.

### When to stop instead

A bridge that will not build as a clean bijection, or whose ids collide, leaves the
benchmark on its current key. That is what `gpqa` and `bbh` did, and the section above
records the evidence for both. A known-positional bank whose join has been checked
against the leaderboard's own records beats a content-hashed one that quietly dropped
the items its key could not separate.
