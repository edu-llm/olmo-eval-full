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

Bank size: **9,163 scenarios** and **67,823 rubric criteria** (one criterion per Table 7 metric allocated to each scenario's task type), across all 9 task types / 4 groups. The `.jsonl` files are canonical; the `.json` twins hold the identical records pretty-printed. `difficulty` and `discrimination` ship as `null` (uncalibrated).

Important notes: 

- ES (Emotional Support) is the only one that populates conversation context, since in EduBench it has conversations about the Agent and Student speaking about the student's feelings. During calibration, we need to pass this context into our calibration model.
- Q&A (Answering Questions) and EC (Error Correction) are the only ones that populate the reference solution field inside scenarios.json.

