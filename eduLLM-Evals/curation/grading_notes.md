# Bank-wide grading notes (curation_v1)

These instructions apply to **every** criterion in the curated rubric bank
(`data/curated/rubrics_qmatrix_curated.jsonl`) unless a criterion's own text
explicitly says otherwise. They are meant to be loaded by the judge/grader
harness as global grading guidance.

## 1. Grade on meaning, not wording (semantic equivalence)
A response satisfies a criterion when it conveys the **required idea, quantity,
fact, or step**, regardless of:
- different phrasing, synonyms, or sentence structure,
- different but mathematically equivalent notation (e.g. `1/2` vs `0.5`,
  `x^2` vs `x\cdot x`, fractions vs decimals),
- different ordering of independent points,
- rounding / significant-figure differences that do not change correctness.

Do **not** fail a response solely because it words a correct answer differently
from how the criterion phrases it. Criteria that say "state that …", "must
provide the answer …", etc. are requiring the **fact/answer**, not a verbatim
string.

## 2. Error identification (verbatim-negation)
When a criterion asks the response to identify that something "is incorrect,"
accept **either** an explicit "incorrect/wrong" label **or** an equivalent
correction that unambiguously conveys the same thing (e.g. supplying the correct
version in a way that makes clear the student's was wrong).

## 3. Conditional / optional criteria
Criteria marked `optional: true` (or carrying `judge_guidance`) apply only under
the stated condition; when the condition does not hold, mark them N/A (leave
blank) rather than failing the response.

## 4. Withhold / prohibition criteria
Criteria phrased "the response must NOT …" are satisfied when the response
avoids the prohibited content; they are single prohibitions and are not graded
as partial credit.

## How this is enforced (wiring)
These rules are not just documentation — they are enforced in every judge path:

| Note item | Canonical frozen judge | Local smoke judge | Human graders |
|-----------|------------------------|-------------------|---------------|
| #1 semantic equivalence | `EVIDENCE_DECISION_POLICY` item 7 (`aws_judge_handoff/scripts/run_judge_validation.py`, `PROMPT_VERSION = judge-validation-v3`) | `_GRADING_POLICY` in `tutor_cat/judge.py` (`PROMPT_VERSION = judge-v3`) | this file |
| #2 error identification | covered by policy item 7 ("equivalent … work is acceptable") | `_GRADING_POLICY` (explicit) | this file |
| #4 withhold / prohibition | `EVIDENCE_DECISION_POLICY` item 4 | `_GRADING_POLICY` | this file |
| #3 optional / conditional | upstream: `optional`/`judge_guidance` on the criterion record controls inclusion | same | this file |

The canonical judge (the calibration path) **already** enforced semantic
equivalence before this tranche; this file formalizes it and the smoke judge was
brought into alignment (`judge-v2` → `judge-v3`). When adding a new judge runner,
load this file's rules into its prompt policy.

## Scope note
The `rigid_verbatim` audit flag is ~95% false positives (mostly "state that
<fact>" content criteria, which require the fact, not a verbatim string). A full
scan for criteria that genuinely *forbid a correct equivalent* or demand a
mandatory verbatim phrase returned only substantive requirements (show-your-work,
real conceptual distinctions), so **no per-criterion text edits were required** —
the bank-wide policy above governs all of them.
