This "unmerged" folder doesn't recursively merge skills based on their criteria overlap scores (see merged/README.md for more information about this).

We went through and manually grouped the 9 sub-skills across 4 broader skills (that will be termed "skills" for the rest of the README). The skill mapping is as follows:



## Skill Groups (9 EduBench task types → 4 groups)

| Group | Original task types |
| --- | --- |
| **academic assistance** | `answering_questions`, `idea_provision` |
| **emotional assistance** | `learning_support`, `mental_health` |
| **checking-student-work** | `error_correction`, `grading` |
| **content generation** | `question_generation`, `material_generation`, `personalized_content_creation` |



The q-matrix for this can be found in manual_groups_qmat.csv (12 EduBench rubric metrics × the 4 groups above). This q matrix is represented in the rubric jsons: every criterion's `q_mapping` is its metric's row from manual_groups_qmat.csv (the 4-group vector), replacing EduBench's original 9-task-slug axis. `ungrouped_qmat.csv` keeps the pre-grouping 12 × 9 matrix for reference.

Bank size (after cleaning — see below): **7,040 scenarios** and **50,341 rubric criteria** (one criterion per Table 7 metric allocated to each scenario's task type), across all 9 task types / 4 groups. The `.jsonl` files are canonical; the `.json` twins hold the identical records pretty-printed. `difficulty` and `discrimination` ship as `null` (uncalibrated). (Raw ingest was 9,163 / 67,823 before dedup + the removals below.)

## Cleaning / deviations from raw EduBench

All applied via reproducible scripts in `scripts/` (each dry-run-first + idempotent):

- **Deduplication** of near-identical scenarios.
- **Removed dropped-option MCQ items** — question stem says "which of the following …" but no options are shown, so the item is unanswerable/ungradeable. Removed from `error_correction`, `grading`, and `answering_questions`.
- **Removed all multiple-choice `idea_provision` items** — IP asks the model to reason "without giving the answer," which is contradictory when the answer is a visible MCQ option.
- **ES restructured** — the multi-turn dialogue was moved from the `prompt` into `conversation_context`; the ES `prompt` now holds only the instruction (so all ES prompts are intentionally identical strings).
- **Stripped the "respond in JSON format" wrapper** from every prompt (task descriptions/deliverables unchanged).
- **Metadata fixes** — relabeled grade-band mismatches to match each embedded student profile, collapsed doubled "Student Profile:" labels, and restructured one free-text profile (eb_8976) into dict form.
- **q_mapping re-expressed** over the 4 groups (see above), replacing the original 9-task-slug axis.

Important notes: 

- ES (Emotional Support) is the only one that populates conversation context, since in EduBench it has conversations about the Agent and Student speaking about the student's feelings. During calibration, we need to pass this context into our calibration model.
- Q&A (Answering Questions) and EC (Error Correction) are the only ones that populate the reference solution field inside scenarios.json.

