# Merged EduBench skill axis

Built by [`scripts/build_edubench_merged.py`](../../../../scripts/build_edubench_merged.py) from `ingest_edubench.py`'s current (already-augmented) `TASKS`/`METRICS`.

## Merge groups

- **`answering_questions+error_correction+idea_provision+grading`** (Question Answering + Error Correction + Idea Provision + Automatic Grading) = `answering_questions`, `error_correction`, `idea_provision`, `grading`
- **`question_generation+material_generation`** (Question Generation + Teaching Material Generation) = `question_generation`, `material_generation`

## Rule

- **q-matrix**: a merged skill's applicability for each of the 12 metrics is the **union** of its members' applicability -- if either member matched a criterion, the merged skill matches it.
- **scenarios/rubrics**: the members' raw records are concatenated; every resulting scenario, regardless of which member file it came from, gets the merged skill's (union) criteria set uniformly.
- `source` keeps each record's ORIGINAL per-stem value; only `use_case` changes.
- `scenario_id`/`criterion_id` numbers are positional (`eb_<i>` over the whole concatenation), so a merge shifts the numbering of any skill sequenced after it -- content for every unmerged skill is otherwise unchanged.
- Only `--q-mapping row` semantics are supported.

See `skills_meta.json` for the full slug -> {label, display, members} mapping.
