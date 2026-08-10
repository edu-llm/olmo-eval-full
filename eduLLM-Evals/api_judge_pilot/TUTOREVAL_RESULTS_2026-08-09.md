# TutorEval frontier-judge production results

The production grading run completed successfully on 2026-08-09 using the
frozen TutorEval judge configuration.

## Configuration

- Judge: `gemini-group/gemini-3-flash-preview`
- Adapter: `generic-binary-strict`
- Prompt version: `generic-binary-strict-v1`
- Output format: binary `pass`/`fail` JSON with evidence and rationale
- Temperature: `0.0`
- Maximum completion tokens: `6144`
- Frozen configuration fingerprint:
  `e2e977f54277a24d5889ed7d3b6baef0e84cdee24589bea31562087f0eaf14b6`

## Results

| Measure | Result |
|---|---:|
| Tutor models | 52 |
| Scenarios per model | 828 |
| Criteria per model | 1,786 |
| Expected and recorded verdict cells | 92,872 |
| Gemini API calls | 92,438 |
| Blank-response automatic fails | 434 |
| Pass verdicts | 12,923 |
| Fail verdicts | 79,949 |
| Overall pass rate | 13.9149% |
| Missing or duplicate cells | 0 |
| API errors | 0 |
| Unparseable/truncated judge outputs | 61 |
| Coverage | 100% |
| Wall-clock runtime | 7,467.6 seconds (about 2 h 4 min) |

The 61 outputs that could not be parsed were retained with
`reason=unparseable_judge_output` and graded as fail under the frozen
fail-closed policy. They were not API request failures. The separate
`errors.jsonl` file is empty.

Token usage reported by the provider was 294,879,823 prompt tokens and
80,966,825 completion tokens across all 92,438 API judgments.

## Artifacts and integrity

The full generated artifacts are in the ignored local directory
`api_judge_pilot/grading_tutoreval/`:

| Artifact | SHA-256 |
|---|---|
| `verdicts.jsonl` | `ea8cd542553342df013f07e65a77e0dc33da420d6787bcf9834000e1e87be2f7` |
| `summary.json` | `c11710a93ebcd7257e7037d963c6cad386627dbd2904d5cfe07317006612042d` |
| `manifest.json` | `9e7664d6dfc7ee12a7962c0864d8e168b51e950f4217fc04f45e3cf6ad477824` |

The manifest records the exact input hashes, runtime hashes, selection
provenance, judge settings, parser policy, and final cell counts. Raw judge
outputs and prompt hashes are retained in every API-graded verdict row.

## Selection-provenance caveat

The frozen judge was selected on the earlier 100-cell sample from
`origin/frq-lab`. A later source audit found legacy text-extraction corruption
in 49 sampled prompts and one sampled criterion, so those selection metrics
should be treated as directional. This production run did not reuse that
copied bank: it used the clean finalized TutorEval scenario and rubric files
whose hashes are recorded in `manifest.json`.
