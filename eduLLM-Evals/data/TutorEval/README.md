# TutorEval

TutorEval (from *Language Models as Science Tutors*) reformatted into the Scenario +
Rubric schema. Source:
https://huggingface.co/datasets/princeton-nlp/TutorEval (one `train` split, revision
`aafd8c96c506f11755c005842c893677b76f20ca`).

Built by [`scripts/ingest_tutoreval.py`](../../scripts/ingest_tutoreval.py).

> **Status: candidate benchmark — not confirmed for the bank.** TutorEval is a *potential*
> option under evaluation, not a committed part of the calibration set. Whether it is used
> still needs further review. Its presence in `data/` does not mean it is wired into any
> run — the default bank in `config.yaml` remains TutorBench. Beyond the usual review, note
> the hard blocker below: TutorEval ships no per-criterion skill labels, so it is **not even
> engine-loadable** as-is.

> **This is an offline skeleton, not a finished bank.** Unlike WildBench/InFoBench,
> TutorEval ships **no per-criterion skill labels**, so there is no offline source for a
> q-matrix. `q_mapping`, `primary_skill`, and the `criticality`/`objectivity`/
> `explicitness` triple are left **null**, and no `difficulty`/`discrimination`/
> `irt_params` are written. The file is therefore **not engine-loadable** yet
> ([`Rubric.from_json`](../../tutor_cat/schemas.py) hard-requires those). See
> [What's null / deferred](#whats-null--deferred).

## What's in this folder

| File | Rows | What it is |
|------|------|------------|
| `scenarios.jsonl` / `.json` | 828 | one tutoring question each (`.json` is a pretty-printed twin) |
| `rubrics.jsonl` / `.json`   | 1793 | one binary criterion per retained `key_points` bullet |
| `dropped.jsonl`             | 52 | every row removed at ingest, with a `reason` (see below) |

- **828 scenarios, 1793 criteria** after drops (from 834 questions / 1846 bullets).
- `.jsonl` is the canonical artifact; the `.json` twin holds identical records
  pretty-printed for reading.

## Two conditions, partitioned

TutorEval has two official evals. Each question is assigned to **exactly one** condition
(never graded twice — double-grading the same `key_points` would break local
independence in the MIRT model):

| `book_condition` | Kept | Prompt |
|------------------|-----:|--------|
| `closed_book` | 368 | the `question` verbatim, chapter withheld |
| `open_book`   | 460 | the `question` with the `chapter` embedded via the authors' template |

The open-book template is verbatim from the upstream `generation_template.txt`. The
chapter goes in `prompt`, never in `conversation_context` (these are single-turn).
`scenarios.json` is ~6 MB because 460 chapters are embedded — expected.

## How the source maps into the schema

| TutorEval field | Schema field |
|-----------------|--------------|
| *(no id field)* | `source_id` = `null` |
| `question` (+ `chapter`) | `prompt` (chapter embedded only for open-book) |
| `domain` | `subject` (math / computer_science / physics / environmental_science / life_sciences) |
| `difficulty` | `subset` (native `easy`/`hard` band — kept out of the IRT `difficulty`) |
| `key_points[i]` | `criterion` (one binary rubric row each; `- ` stripped, whitespace collapsed) |
| `closed_book` | `book_condition` (`closed_book`/`open_book`) |
| `misleading_question` | `misleading_question` (provenance extra; 112 kept) |
| `answer_in_chapter` | `answer_in_chapter` (provenance extra) |

`book_condition`, `misleading_question`, and `answer_in_chapter` are TutorEval-native
provenance fields beyond the base schema. Like InFoBench's `source_id`/`subset`, they are
dropped on load by `Scenario.from_json` until the dataclass gains the field — they exist
so the later q-matrix / IRT passes can use them (e.g. `misleading_question` → `criticality`).

## Dropped items

Bad items are removed at ingest (not flagged, not hand-fixed), each logged to
`dropped.jsonl`:

| Level | Reason | Count | Why |
|-------|--------|------:|-----|
| scenario | `equation_stripped_chapter` | 4 | chapter's equations reduced to bare `(2)`/`(3)` placeholders |
| scenario | `no_criteria_left` | 2 | every bullet was dropped by the criterion-level rules |
| criterion | `too_terse` | 34 | closed-book bullet shorter than `TERSE_MIN_CHARS` (20) |
| criterion | `defers_to_chapter` | 12 | closed-book bullet points at the withheld chapter |

`figure_reference` (questions about an absent figure) is a documented rule but matches 0
rows on the pinned revision. Criterion-level rules apply to **closed-book only** — in
open-book the chapter is in the prompt, so chapter-referencing bullets are gradable.

> **Note on `too_terse`:** many short bullets (`"Answer is radiation"`) are in fact
> gradable given the question. They were dropped per the ingest decision; to keep them,
> set `TERSE_MIN_CHARS = 0` at the top of the ingester and re-run. Every drop is logged,
> so this is fully reversible.

## Criterion quality — read this before trusting the bank

The retained criteria are **weaker as standalone binary items** than WildBench's or
InFoBench's, for a structural reason: TutorEval's `key_points` were authored to be handed
to a grader **as a group and scored holistically** (PRESENTATION 0–3 / CORRECTNESS 0–3),
**not** atomized into independent pass/fail items. Our schema atomizes them (one binary
criterion per bullet), so the mismatch shows up as terse, fragmentary, or directive
phrasing. By contrast InFoBench's `decomposed_questions` already ship as standalone yes/no
questions and WildBench's `checklist` items as standalone checks.

Measured over the 1793 kept criteria:

| pattern | share | example | verdict |
|---|--:|---|---|
| context-dependent (`this`/`the student's`) | 53% | `"this is not necessarily true"` | **fine** — see below |
| lowercase sentence fragments | 19% | `"to show this, we can conjugate by a permutation matrix"` | mostly fine with context |
| imperative tutor-directives | 25% | `"Give student hint that they need to read each line"` | gradable; phrasing only |
| bare verdicts / answer tokens | 7% | a scenario whose 4 criteria are `CIFAR-10 / CIFAR-100 / SVHN / MNIST` | **genuinely weak** |

**Why most of this is not as bad as it looks in `rubrics.json`.** The judge never scores a
criterion blind: [`judge.py`](../../tutor_cat/judge.py) passes `scenario.prompt` with every
criterion, and for open-book items the chapter is embedded in that prompt. So a
context-dependent bullet like `"this is not necessarily true"` is graded alongside its
question (*"Is it valid to state schizophrenic people have depression?"*) and resolves
fine. Reading criteria in isolation (as `rubrics.json` shows them) overstates the problem.
Note also that the judge wraps each criterion as *"Does the tutor's response satisfy this
criterion: {criterion}"*, so imperative phrasing (`"Explain that X"`) already reads
correctly — a mechanical imperative→question rewrite would buy almost nothing.

**What is genuinely weak** is the ~7% of bare answer-token / holistic-list items (the
`CIFAR-10 / …` case): atomized, each becomes a low-information "did the response mention
X?" check that saturates and carries little discrimination.

**Deliberately not hand-fixed.** Rewriting 1793 criteria would replace source-faithful
data with paraphrase and needs the manpower/rubric-authoring we don't have. Options, in
order of preference:

1. **Leave as-is; filter at calibration.** Once real judge responses exist, the weak
   items show near-zero empirical discrimination and are dropped then — the same
   principled, data-driven filter used elsewhere. Costs nothing now.
2. **Narrow to a high-value subset.** The integration spec (§3) judges TutorEval a
   content probe largely redundant with the existing bank, **except** the 112
   `misleading_question` items (false-premise / anti-sycophancy → `diagnosis`), which it
   calls genuinely valuable. Keeping only those trades quantity for quality.
3. **LLM rewrite pass (bigger scope).** Extend the deferred q-matrix pass to atomize/
   rephrase each bullet into a standalone binary predicate. This works but crosses from
   *formatting source data* into *LLM-authored rubrics*, and would need the same
   judge-validation as any generated rubric.

No path is applied yet — the bank ships as the faithful atomization (option 0). Revisit
once there is calibration signal, or switch to the subset if a clean contribution is
wanted sooner.

## What's null / deferred

- **Real (from the dataset):** `prompt`, `criterion`, `subject`, `subset`,
  `book_condition`, `misleading_question`, `answer_in_chapter`.
- **Null in this skeleton (no offline source):** `q_mapping`, `primary_skill`,
  `q_rationale`, `criticality`, `objectivity`, `explicitness`.
- **Absent entirely:** `difficulty`, `discrimination`, `irt_params`.

TutorEval's natural skill axis *is* the engine's own `content / diagnosis / scaffolding`
(it's science tutoring), so once populated it can pool with TutorBench **without**
de-globalizing the axis — unlike InFoBench (5 axes) or WildBench (11).

To finish the bank later:

```bash
# 1. fill q_mapping / primary_skill / q_rationale (+ the metadata triple) via the LLM pass
python scripts/generate_qmatrix.py   # content/diagnosis/scaffolding; needs API access
# 2. synthesize difficulty / discrimination
python scripts/assign_irt_params.py \
    --input data/TutorEval/rubrics.jsonl \
    --skills content,diagnosis,scaffolding \
    --log-dir data/TutorEval/irt_logs --no-backup
```

## Rebuilding

```bash
python scripts/ingest_tutoreval.py
```

Offline, no API. Deterministic: ids are assigned over rows sorted by
`(path_to_chapter, question)`, so they are stable across reruns.
