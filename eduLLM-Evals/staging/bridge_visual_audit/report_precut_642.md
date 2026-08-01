# Bridge visual-dependency audit — the pre-cut 642-scenario bank

> Snapshot of the audit as it stood before the 263 `missing_data` scenarios were removed.
> This is the evidence behind that cut; `report.md` regenerates against the current bank.

Classifier `openai-group/gpt-5.5`, one call per scenario, 642/642 scenarios judged. Question: can a tutor identify the student's error from the text alone?


## Headline


| | scenarios | conversations | criteria |
|---|---:|---:|---:|
| requires a visual | 364 | 255 | 7534 |
| ...of those, high confidence | 362 | 254 | 7492 |
| self-sufficient (would remain) | 278 | — | — |

Share of the bank flagged: **56.7%**


## Confidence


| confidence | flagged |
|---|---:|
| high | 362 |
| medium | 2 |


## Why they fail


| kind | scenarios |
|---|---:|
| `missing_data` | 263 |
| `explicit_pointer` | 74 |
| `labelled_option` | 27 |


## Damage by topic domain


| topic_domain | flagged | total | share |
|---|---:|---:|---:|
| data_graphing | 21 | 27 | 78% |
| geometry_spatial | 131 | 204 | 64% |
| place_value_number | 76 | 131 | 58% |
| fractions | 18 | 32 | 56% |
| operations_arithmetic | 70 | 139 | 50% |
| proportional_reasoning | 14 | 31 | 45% |
| measurement_conversion | 22 | 49 | 45% |
| algebra_expressions | 12 | 29 | 41% |


## Damage by grade band


| grade_band | flagged | total | share |
|---|---:|---:|---:|
| 1-3 | 151 | 240 | 63% |
| 4-5 | 178 | 332 | 54% |
| 6-12 | 35 | 70 | 50% |


## Interaction with `visible_mistake`

`visible_mistake: false` already flags scenarios with no remediable error. If the two overlap little, this audit is finding a genuinely separate defect.


| | flagged visual | not flagged |
|---|---:|---:|
| visible_mistake=True | 282 | 269 |
| visible_mistake=False | 82 | 9 |


## Conversation-level consistency

Repeats of one conversation share a stimulus, so they should be classified alike. Splits are classifier noise and mark the borderline cases.


- conversations with >1 scenario: **212**
- of those, classified inconsistently: **18**


## Sample of flagged items


**bridge_0000** · `3.6D.Decomposing Figures` · explicit_pointer
> What is the area of the pink rectangle?
> student's final turn: `4 m`

**bridge_0001** · `5.2A.Decimal Numbers` · missing_data
> Here comes the question.
> student's final turn: `0.8`

**bridge_0002** · `4.2G.Convert Fractions to Decimal Form` · missing_data
> No fraction or student answer appears in the chat.
> student's final turn: `yes`

**bridge_0007** · `3.1G.Shapes and Area (Review) - 2` · missing_data
> How long is the wood the carpenter wants to break into pieces?
> student's final turn: `2 inches`

**bridge_0010** · `3.6D.Decomposing Figures` · missing_data
> area of the top rectangle
> student's final turn: `es 50`

**bridge_0011** · `4.6D.Classifying 2D Figures` · missing_data
> this shape is a pentagon
> student's final turn: `yes`

**bridge_0012** · `4.2A.Comparing Decimals` · explicit_pointer
> Is that picture visible to you?
> student's final turn: `uhh yes`

**bridge_0015** · `3.2B.Multiples of Ten` · missing_data
> Problem asks for 8th multiple of 10, absent from chat
> student's final turn: `90`

**bridge_0016** · `5.2A.Decimal Numbers (Review) - 1` · missing_data
> student asks “is it right” but answer is absent
> student's final turn: `is it right`

**bridge_0017** · `8.4A.What is Slope?` · missing_data
> What is the value of y1? [student] 5?
> student's final turn: `5?`

**bridge_0018** · `5.3B.Word Problems with Standard Measurement` · explicit_pointer
> that question
> student's final turn: `yes`

**bridge_0019** · `3.4H.Understanding Division` · missing_data
> Only 'Is that your final answer?' and 'yes'; problem/answer absent
> student's final turn: `yes`
