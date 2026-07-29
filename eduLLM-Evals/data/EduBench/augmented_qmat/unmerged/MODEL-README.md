# EduBench

EduBench reformatted into the Scenario + Rubric schema. Source:
https://huggingface.co/datasets/DirectionAI/EduBench (`en_data/`, nine JSONL files, one
per educational task type). Paper: *EduBench: A Comprehensive Benchmarking Dataset for
Evaluating Large Language Models in Diverse Educational Scenarios* (arXiv 2505.16160).

Built by [`scripts/ingest_edubench.py`](../../scripts/ingest_edubench.py). **This bank ships
uncalibrated** — `difficulty` and `discrimination` are explicit `null`, and no IRT
parameters are included. [`scripts/assign_irt_params.py`](../../scripts/assign_irt_params.py)
can fill them, but its output is synthetic and is deliberately not committed here; see
*Placeholders* below.

> **Status: candidate benchmark — not confirmed for the bank.** EduBench is a *potential*
> option under evaluation, not a committed part of the calibration set. Whether it is used
> still needs further review. Its presence in `data/` does not mean it is wired into any
> run — the default bank in `config.yaml` remains TutorBench, and EduBench only loads when
> you explicitly flip `SKILLS` to its 9-task axis and point `config.yaml` at these files.
> Unlike IFEval and InFoBench, EduBench is a **pedagogical** benchmark rather than an
> instruction-following one; the open question is not whether that dimension belongs, but
> whether the 9 task types are the right skill axis (see *Skill axis* below).

## What's in this folder

| File | Rows | What it is |
|------|------|------------|
| `scenarios.jsonl` / `.json` | 9163 | one educational task instance each (`.json` is a pretty-printed twin for reading) |
| `rubrics.jsonl` / `.json`   | 55231 | one binary criterion per Table 7 metric allocated to the scenario's task |

There is no `irt_logs/` here, unlike the sibling folders: nothing has been calibrated.

- **9163 scenarios, 55231 criteria.** Each record in `en_data/<STEM>.jsonl` becomes one
  scenario; each metric Table 7 allocates to that task type becomes one binary criterion
  (between 3 for PCC and 8 for IP).
- `.jsonl` is the canonical artifact; the `.json` twin holds the identical records
  pretty-printed. Note that `rubrics.json` is ~97 MB — large enough to be awkward to open;
  prefer the `.jsonl` for anything but targeted inspection.

**The published artifact is smaller than the paper reports.** `en_data/` totals 9163
records against the 9680 of the paper's Table 2 — every file ships fewer than claimed:

| file | shipped | Table 2 | file | shipped | Table 2 |
|------|--------:|--------:|------|--------:|--------:|
| `Q&A` | 1285 | 1328 | `QG`  | 1288 | 1338 |
| `EC`  | 1301 | 1350 | `AG`  | 1042 | 1073 |
| `IP`  | 1301 | 1350 | `TMG` | 1185 | 1347 |
| `PLS` | 448  | 561  | `PCC` | 252  | 259  |
| `ES`  | 1061 | 1074 | **total** | **9163** | **9680** |

Verified by counting raw lines and comparing against parsed records — they match exactly
per file, so nothing is being dropped at parse time. The ingester treats the **artifact**
as authoritative and keeps the paper's figures only as a reference column, so future drift
in the published files trips validation rather than passing silently.

## How the source maps into the schema

Every record has exactly three top-level keys: `information`, `prompt`,
`model_predictions`.

| EduBench field | Schema field |
|----------------|--------------|
| file stem | `use_case` (slug) + `source` (`Edubench_<STEM> JSONL`) |
| `information["Subject"]` | `subject` (25 distinct; 14 in PCC) |
| `information[<alias>]` | `grade_band` — see the alias note below (6 distinct: Elementary School, Middle School, High School, Undergraduate, Master, PhD) |
| `prompt` | `prompt`, verbatim |
| `information["Answer"]` (Q&A) / `["Corrected Answer"]` (EC) | `reference_solution` |
| Table 7 row | `criterion` + `q_mapping` (one rubric row per checkmark) |

Scenarios are emitted in the concatenation order `Q&A, EC, IP, PLS, ES, QG, AG, TMG, PCC`,
in file order within each, with `scenario_id = eb_<i>` counting from 0 across the whole
concatenation. Criteria are numbered `c01`…`cNN` within each scenario in the paper's
H.1.1 → H.3.4 row order, so `c01` is always IFTC — the only metric allocated to all nine
tasks.

EduBench records carry no native identifier, so unlike InFoBench there is no `source_id`
to preserve. `model_predictions` (five model responses per record) is discarded at parse
time; re-downloading is ~100 MB if real IRT calibration needs it later.

**Grade-band aliasing is required, not cosmetic.** The same semantic field carries three
different key names across the nine files: `Education Level` (Q&A, PLS, ES, QG), `Level`
(AG, TMG, PCC), `Difficulty` (EC, IP). EC/IP's `Difficulty` holds a grade band
("High School"), not an easy/medium/hard rating — **no difficulty field exists anywhere in
the source data.** The ingester reports which alias each file used and fails if a record
has none, so a silent `null` cannot slip through.

## Skill axis (q-matrix)

The axis is EduBench's own nine task types, slugged from the Table 7 column headers:

`answering_questions, error_correction, idea_provision, learning_support, mental_health,
question_generation, grading, material_generation, personalized_content_creation`

Three of the nine slugs differ from the file stem they come from, because Table 7's column
headers and the JSONL filenames disagree: `Answering Questions` → `Q&A`,
`Learning Support` → `PLS` (Personalized Learning Support), `Mental Health` → `ES`
(Emotional Support).

Table 7 of the paper, recovered from the arXiv HTML's LaTeXML cell IDs (empty cells are
emitted with IDs, so column positions are exact):

```text
                                        Q&A  EC  IP PLS  ES  QG  AG TMG PCC │  n
IFTC  Instruction Following & Task Compl. 1   1   1   1   1   1   1   1   1 │  9
RTC   Role & Tone Consistency             .   .   .   .   1   .   .   1   . │  2
CRSC  Content Relevance & Scope Control   1   .   1   1   .   1   1   1   . │  6
SEI   Scenario Element Integration        .   1   1   1   1   .   .   .   1 │  5
BFA   Basic Factual Accuracy              1   1   1   .   .   1   1   1   . │  6
DKA   Domain Knowledge Accuracy           .   .   1   .   .   1   .   1   . │  3
RPR   Reasoning Process Rigor             1   1   1   .   .   .   1   .   . │  4
EICP  Error Identification & Correction   .   1   .   .   .   .   1   .   . │  2
CSI   Clarity, Simplicity & Inspiration   .   1   1   .   .   1   .   1   . │  4
MGP   Motivation, Guidance & Pos. Fdbk    .   1   .   .   1   .   1   .   . │  3
PAS   Personalization, Adapt. & Learn Sup .   .   .   1   1   .   .   .   1 │  3
HOTS  Higher-Order Thinking & Skill Dev   .   .   1   1   .   1   .   1   . │  4
                                         ──  ──  ──  ──  ──  ──  ──  ──  ── │ ──
criteria per scenario                     4   7   8   5   5   6   6   7   3 │ 51
```

**`q_mapping` is a Table 7 row, not a per-response loading.** Under the default
`--q-mapping row`, a criterion's `q_mapping` is its full Table 7 row over the nine slugs,
so the artifact holds only **12 distinct `q_mapping` values** — one per metric. The
consequence is deliberate and should not be quietly patched: an EC scenario's IFTC
criterion carries `grading: 1` and `mental_health: 1`, because a Table 7 row states where
a metric is *used*, never what a given response *exercises*. Combined with the absent
`primary_skill` (below), IFTC therefore claims equal discrimination on all nine
dimensions.

`--q-mapping onehot` is the alternative: it intersects the row with the scenario's own
task, giving a one-hot loading and **9 distinct values** — the between-item
multidimensional reading, where each item measures exactly the task it came from.
Switching is a re-run, not a rewrite.

Criteria marking each skill (`q_mapping` load) under each mode:

| skill | `row` load | `onehot` load |
|-------|-----------:|--------------:|
| answering_questions | 28043 | 5140 |
| error_correction | 36679 | 9107 |
| idea_provision | 45477 | 10408 |
| learning_support | 26058 | 2240 |
| mental_health | 20937 | 5305 |
| question_generation | 36185 | 7728 |
| grading | 33790 | 6252 |
| material_generation | 38431 | 8295 |
| personalized_content_creation | 15287 | 756 |

Under `onehot` the loads are exactly *records × criteria-per-task* and sum to 55231; under
`row` they sum to far more, since each criterion loads on 2–9 skills.

**Do not co-load this axis with another benchmark's.** These nine slugs are EduBench's own
task vocabulary and share no meaning with InFoBench's 5 constraint types or TutorBench's 3
pedagogical skills — the same warning InFoBench's README carries about its `content` slug.
`tutor_cat.SKILLS` is a hardcoded 3-tuple, so loading this bank means rebinding `SKILLS`
on `tutor_cat`, `tutor_cat.schemas` and `tutor_cat.dataio`, or editing that constant.

## Criterion text

Each criterion is three parts joined by literal newlines:

```text
Criteria: <Title> (<ABBR>).
General Description: <the paper's "Description:" text, referents resolved>
Passing Description: <9-10 anchor> or <7-8 anchor>
```

The two top anchors are concatenated with the literal word "or", which is what makes the
criterion binary: **a pass is a response that would have scored at least 7-8** on
EduBench's 10-point scale.

The paper's generic referents are resolved to the current task, so the text is
scenario-specific: the implied subject becomes "the tutoring model", and
`the core task (e.g., solving problems, error correction, question generation)` becomes the
actual task, e.g. for EC:

```text
Criteria: Instruction Following & Task Completion (IFTC).
General Description: Did the tutoring model fully understand and execute the user's
instruction? Was the core task (correcting the student's answer and explaining the error)
completed? Is the output formatting correct?
Passing Description: The tutoring model fully understood and precisely executed all
instructions; achieved core task with perfect accuracy; output format is fully compliant.
or the tutoring model accurately understood main instructions and correctly completed the
task; core goals are well achieved; format is mostly correct with only minor omissions or
deviations.
```

Task-type references in SEI, EICP and MGP ("In error correction scenarios", "In answering
or tutoring") are resolved the same way.

## Placeholders — what is real vs. synthetic

- **Real (from the dataset):** `prompt`, `subject`, `grade_band`, `use_case`,
  `reference_solution`.
- **Real, but from the paper rather than the HF artifact:** `criterion` and `q_mapping`.
  See *Standing constraint* below.
- **Placeholder (uniform, matching InFoBench/WildBench):** `criticality="critical"`,
  `objectivity="objective"`, `explicitness="explicit"`. EduBench has no native source for
  these.
- **Explicit `null` — uncalibrated, and shipped that way:** `difficulty`,
  `discrimination`. No IRT values are committed, synthetic or otherwise.
- **Empty by design:** `expected_evidence`, `conversation_context`, `score_anchors`.
- **Absent by design:** `primary_skill`.

**Why no synthetic IRT values are committed here.** Running `assign_irt_params.py` on this
bank does work, but it would add no information. The heuristic derives `difficulty` from
`explicitness`/`criticality` and `discrimination` from
`objectivity`/`criticality`/`primary_skill` — and all four inputs are constant across all
55231 rows. A trial run confirmed the consequence exactly: difficulty came out as pure
jitter around `B_BASE + B_EXPLICITNESS[explicit] = -0.4` (mean -0.4015) and discrimination
as jitter around `A_BASE × A_OBJECTIVITY[objective] × A_CRITICALITY[critical] = 1.38`
(mean 1.395, identical to three decimals for all nine skills). That is weaker than
InFoBench, whose real multi-label `question_label` at least fed a varying `primary_skill`.
An explicit `null` states "uncalibrated" honestly; a per-row draw from one distribution
would have looked like item parameters without being any.

`primary_skill` is omitted because the schema the ingester targets does not include it.
This is safe: `assign_irt_params.py` reads it with `record.get`, so `None` falls through to
`A_NEUTRAL = 1.0` and every loaded skill is weighted equally instead of getting the
`A_PRIMARY = 1.20` / `A_SECONDARY = 0.85` split. With a `row` q-mapping there is no
defensible primary anyway.

## Notes

- `conversation_context` is empty for **all** rows, including the 1061 ES records that
  contain a multi-turn student dialogue — that dialogue is already inside `prompt`
  (`information["Dialogue with Student"]`), so nothing is lost by leaving the field empty.
- `reference_solution` is populated for **2586 rows only** (1285 Q&A + 1301 EC), empty for
  the other seven task types, which ship no gold answer. This deviates from a literal
  reading of the schema spec, by decision: EC's gold `Corrected Answer` appears nowhere in
  `prompt` — the prompt holds the student's *wrong* answer — so pinning the field to `""`
  would have discarded real data. (Q&A's answer does often appear in `prompt`, since its
  questions are multiple-choice and the gold is one of the listed options.)
- **The shipped artifact does not load, by design.** With `difficulty` and `discrimination`
  null, `load_bank()` returns `report.ok == False` with one error per rubric row —
  [`schemas.py`](../../tutor_cat/schemas.py) does `float(obj["difficulty"])`, and
  [`dataio.py`](../../tutor_cat/dataio.py) catches per-row parse failures into
  `ValidationReport` rather than raising. A null is a louder failure than a plausible
  placeholder, so the bank refuses to load until someone consciously supplies parameters.
  The round-trip has been verified once: calibrating with the nine slugs and reloading with
  `SKILLS` rebound gives 0 errors and 0 warnings across all 9163 scenarios and 55231
  criteria, so the only thing standing between this file and a loadable bank is the
  calibration step.
- `split: "calibration"` is a **pipeline-role label**, matching InFoBench and WildBench —
  not an HF split name. EduBench ships no train/test division.
- The Chinese `zh_data/` split is out of scope, and is not a mirror of `en_data/`: it ships
  8 files, missing `ES.jsonl`.
- `modality` is `"text"` for every row; EduBench ships no image or multimodal content.

**Standing constraint** (from [`data/README.md`](../README.md)): *any benchmark added must
ship its own rubrics (no hand-authoring) and be verified against the actual dataset
artifacts, not docs.* EduBench meets this **only partly**, and the gap should be weighed
before committing to it:

- *Own rubrics* — yes. The 12 metrics, their descriptions and their score anchors are
  EduBench's own rubric, quoted from the paper.
- *From the artifact* — **no.** The rubric lives in the paper's Appendix H and Table 7, not
  in the HF files, which contain only `information`/`prompt`/`model_predictions`. The
  scenario-specific substitutions and the >= 7-8 dichotomization are authored here.
- *Verified against artifacts* — yes for the scenario side. Record counts, `information`
  key sets, grade-band aliases and subject vocabularies were all read off the downloaded
  files, which is how the 9163-vs-9680 discrepancy surfaced.

## Rebuilding

```bash
python scripts/ingest_edubench.py
```

That reproduces exactly what is committed here, nulls included.

Calibration is a **separate, opt-in step** and its output is not committed — see
*Placeholders* for why. If you do want to load the bank:

```bash
python scripts/assign_irt_params.py \
    --input data/EduBench/rubrics.jsonl \
    --skills answering_questions,error_correction,idea_provision,learning_support,mental_health,question_generation,grading,material_generation,personalized_content_creation \
    --log-dir data/EduBench/irt_logs --no-backup
```

Note that this **overwrites `rubrics.jsonl` in place** (`--no-backup`), replacing the nulls
and appending `irt_params`. Re-running the ingester resets it to the committed null state.

Useful flags: `--dry-run` downloads and validates without writing, and prints per-file
record counts, grade-band aliases, subject/grade vocabularies and q-matrix loads.
`--q-mapping onehot` switches the loading as described above. `--out DIR` redirects output.

If HTTPS to HuggingFace fails with `unable to get local issuer certificate` or
`Basic Constraints of CA cert not marked critical`, add `--os-trust-store`. That routes
`huggingface_hub` through the operating system trust store instead of certifi, which is
needed behind a TLS-inspecting antivirus or corporate proxy that re-signs traffic with a
locally generated root. Certificates are still verified and hostnames still checked.
