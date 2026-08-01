# Bridge visual-dependency audit

Classifier `openai-group/gpt-5.5`, one call per scenario, 379/379 scenarios in the current bank judged. Question: can a tutor identify the student's error from the text alone?


> **263 audited scenarios have since been excluded from the bank** and are not counted below. The audit as it stood before that cut is in [`report_precut_642.md`](report_precut_642.md).


## Headline


| | scenarios | conversations | criteria |
|---|---:|---:|---:|
| requires a visual | 101 | 73 | 2094 |
| ...of those, high confidence | 101 | 73 | 2094 |
| self-sufficient (would remain) | 278 | — | — |

Share of the bank flagged: **26.6%**


## Confidence


| confidence | flagged |
|---|---:|
| high | 101 |


## Why they fail


| kind | scenarios |
|---|---:|
| `explicit_pointer` | 74 |
| `labelled_option` | 27 |


## Damage by topic domain


| topic_domain | flagged | total | share |
|---|---:|---:|---:|
| geometry_spatial | 43 | 116 | 37% |
| data_graphing | 3 | 9 | 33% |
| place_value_number | 25 | 80 | 31% |
| fractions | 5 | 19 | 26% |
| algebra_expressions | 5 | 22 | 23% |
| measurement_conversion | 6 | 33 | 18% |
| operations_arithmetic | 13 | 82 | 16% |
| proportional_reasoning | 1 | 18 | 6% |


## Damage by grade band


| grade_band | flagged | total | share |
|---|---:|---:|---:|
| 1-3 | 39 | 128 | 30% |
| 4-5 | 51 | 205 | 25% |
| 6-12 | 11 | 46 | 24% |


## Interaction with `visible_mistake`

`visible_mistake: false` already flags scenarios with no remediable error. If the two overlap little, this audit is finding a genuinely separate defect.


| | flagged visual | not flagged |
|---|---:|---:|
| visible_mistake=True | 77 | 269 |
| visible_mistake=False | 24 | 9 |


## Conversation-level consistency

Repeats of one conversation share a stimulus, so they should be classified alike. Splits are classifier noise and mark the borderline cases.


- conversations with >1 scenario: **115**
- of those, classified inconsistently: **2**


## Sample of flagged items


**bridge_0000** · `3.6D.Decomposing Figures` · explicit_pointer
> What is the area of the pink rectangle?
> student's final turn: `4 m`

**bridge_0012** · `4.2A.Comparing Decimals` · explicit_pointer
> Is that picture visible to you?
> student's final turn: `uhh yes`

**bridge_0018** · `5.3B.Word Problems with Standard Measurement` · explicit_pointer
> that question
> student's final turn: `yes`

**bridge_0025** · `5.7A.Standard Conversions` · explicit_pointer
> can i you see my answer
> student's final turn: `yes`

**bridge_0038** · `5.2A.Place Value (Review) - 1` · explicit_pointer
> Here is the place value chart.
> student's final turn: `Done`

**bridge_0039** · `3.4G.Properties of Multiplication` · labelled_option
> Student answers only “A” / “B”
> student's final turn: `A`

**bridge_0040** · `8.5G.Intro to Functions` · explicit_pointer
> top part of the chart
> student's final turn: `26`

**bridge_0048** · `7.1C.Additive Inverses` · labelled_option
> Which point on the number line... student: b
> student's final turn: `b`

**bridge_0064** · `4.2A.Place Value (Review) - 1` · explicit_pointer
> Is that number visible now?
> student's final turn: `eight thousand`

**bridge_0075** · `4.2A.Comparing Decimals` · explicit_pointer
> Is that picture visible to you?
> student's final turn: `uhh yes`

**bridge_0076** · `4.5C.Perimeter` · explicit_pointer
> Which shape is given in this question?
> student's final turn: `squre`

**bridge_0097** · `2.2A.Place Value` · explicit_pointer
> Can you use the whiteboard
> student's final turn: `i did`
