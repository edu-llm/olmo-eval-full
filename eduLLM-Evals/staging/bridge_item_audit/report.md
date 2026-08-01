# Bridge per-item audit

Classifier `openai-group/gpt-5.5`, one call per scenario, 289/289 scenarios read for what the graded turn actually asks. Labels judged against that reading.


## Headline


| finding | scenarios | share |
|---|---:|---:|
| student is not actually wrong | 35 | 12% |
| not gradeable from visible text | 35 | 12% |
| topic_domain mismatched | 110 | 38% |
| ...of which not about maths at all (`none_fit`) | 18 | 6% |
| error_module contradicted by the evidence | 142 | 49% |
| problem statement still unstated | 23 | 8% |
| other concern flagged | 90 | 31% |
| **clean on every check** | **76** | **26%** |


## topic_domain: assigned vs read


| assigned | classifier's reading | n |
|---|---|---:|
| geometry_spatial | operations_arithmetic | 18 |
| operations_arithmetic | place_value_number | 12 |
| algebra_expressions | operations_arithmetic | 11 |
| measurement_conversion | operations_arithmetic | 11 |
| place_value_number | operations_arithmetic | 9 |
| fractions | operations_arithmetic | 8 |
| operations_arithmetic | none_fit | 5 |
| place_value_number | none_fit | 5 |
| proportional_reasoning | operations_arithmetic | 3 |
| geometry_spatial | measurement_conversion | 3 |
| proportional_reasoning | fractions | 3 |
| geometry_spatial | none_fit | 2 |
| operations_arithmetic | algebra_expressions | 2 |
| measurement_conversion | none_fit | 2 |
| operations_arithmetic | measurement_conversion | 2 |
| proportional_reasoning | algebra_expressions | 2 |
| place_value_number | algebra_expressions | 2 |
| proportional_reasoning | none_fit | 2 |
| geometry_spatial | fractions | 2 |
| algebra_expressions | place_value_number | 1 |
| operations_arithmetic | fractions | 1 |
| operations_arithmetic | geometry_spatial | 1 |
| algebra_expressions | none_fit | 1 |
| fractions | none_fit | 1 |
| proportional_reasoning | measurement_conversion | 1 |


## error_module: assigned vs suggested


| assigned | suggested | n |
|---|---|---:|
| right_idea | guess | 30 |
| careless | guess | 24 |
| conceptual | ? | 13 |
| conceptual | guess | 12 |
| conceptual | imprecise | 10 |
| imprecise | ? | 10 |
| right_idea | conceptual | 8 |
| right_idea | ? | 7 |
| right_idea | imprecise | 7 |
| guess | ? | 5 |
| guess | conceptual | 4 |
| careless | conceptual | 4 |
| guess | imprecise | 3 |
| imprecise | conceptual | 2 |
| imprecise | guess | 1 |
| imprecise | right_idea | 1 |
| careless | imprecise | 1 |

### Students who are not actually wrong — 35 of 289 (12%)

D1/D2 ask the tutor to identify an error. Where there is none they are unpassable, and both are `critical`.

- **bridge_0004** · student said `a little bit` — whether the student understands the tutor's comparison showing 16.50 and 12.95 are less than 20
- **bridge_0006** · student said `no mam` — whether the student's prior answer is final
- **bridge_0024** · student said `i have to leeve` — show work on the whiteboard using the Pencil Tool
- **bridge_0037** · student said `no` — whether the student needs help explaining why a place value cannot contain a two-digit number
- **bridge_0072** · student said `the dimensional area` — UNSTATED: identify that he should measure the area
- **bridge_0094** · student said `yes` — whether the tutor's multiplication/final-answer step makes sense
- **bridge_0096** · student said `90` — UNSTATED: give an example or another example of a multiple of 10
- **bridge_0111** · student said `can i go` — confirm understanding of the addition problem 1440 + 1440 + 1440
- **bridge_0135** · student said `no` — whether the student needs help answering the place-value question
- **bridge_0171** · student said `yes!!!!!!!` — to confirm whether the final answer is 1/5 < 1/2
- **bridge_0264** · student said `avarage speed` — identify the unit rate/quantity to find, the average speed in miles per hour
- **bridge_0270** · student said `how long it is` — define measuring length
- **bridge_0273** · student said `mmmm they have many sides` — state what the student knows about polygons
- **bridge_0278** · student said `seeeeee` — whether the student understands why 1/4 equals 2/8 from the models
- **bridge_0280** · student said `ok i thought that we turn the whole numb` — whether the student understands why 2,458 rounded to the nearest hundred is 2,500
- **bridge_0287** · student said `no` — whether the student needs anything else before ending the session
- **bridge_0295** · student said `yes` — confirm whether the student is present
- **bridge_0313** · student said `it can be big` — say what the student knows about a graph
- **bridge_0344** · student said `im done` — the value of 6 + 8
- **bridge_0383** · student said `yes` — whether the student is aware of the whiteboard tools

### Still not gradeable — 35 of 289 (12%)

Residual missing context the two earlier cuts did not catch.

- **bridge_0004** · student said `a little bit` — whether the student understands the tutor's comparison showing 16.50 and 12.95 are less than 20
- **bridge_0006** · student said `no mam` — whether the student's prior answer is final
- **bridge_0008** · student said `4  times in` — UNSTATED: determine how many times 4 can go into 27
- **bridge_0013** · student said `928` — UNSTATED: identify that the question asks for the expanded notation for 928
- **bridge_0024** · student said `i have to leeve` — show work on the whiteboard using the Pencil Tool
- **bridge_0036** · student said `2 equal angles` — UNSTATED: explain how to check whether the answer about a quadrilateral/trapezoid is correct
- **bridge_0037** · student said `no` — whether the student needs help explaining why a place value cannot contain a two-digit number
- **bridge_0041** · student said `so it would be 7 + 5` — UNSTATED: identify that the task is to find the perimeter, likely of a figure with side lengths 7 and 5
- **bridge_0072** · student said `the dimensional area` — UNSTATED: identify that he should measure the area
- **bridge_0094** · student said `yes` — whether the tutor's multiplication/final-answer step makes sense
- **bridge_0096** · student said `90` — UNSTATED: give an example or another example of a multiple of 10
- **bridge_0111** · student said `can i go` — confirm understanding of the addition problem 1440 + 1440 + 1440
- **bridge_0278** · student said `seeeeee` — whether the student understands why 1/4 equals 2/8 from the models
- **bridge_0280** · student said `ok i thought that we turn the whole numb` — whether the student understands why 2,458 rounded to the nearest hundred is 2,500
- **bridge_0295** · student said `yes` — confirm whether the student is present
- **bridge_0307** · student said `a spear` — UNSTATED: name the displayed shape
- **bridge_0313** · student said `it can be big` — say what the student knows about a graph
- **bridge_0344** · student said `im done` — the value of 6 + 8
- **bridge_0383** · student said `yes` — whether the student is aware of the whiteboard tools
- **bridge_0398** · student said `yea kinda.` — whether the student understands the explanation that 4 place-value jumps gives an exponent of 10^4

### Other concerns — 90 of 289 (31%)

- **bridge_0004** — The student is only expressing partial understanding, not making a mathematical error.
- **bridge_0005** — The underlying math problem is not visible, and the student's response appears to be gibberish or a typo rather than a diagnosable math error.
- **bridge_0006** — No mathematical task or student error is visible; the student is only saying the answer is not final.
- **bridge_0008** — The visible tutor prompt is garbled/reversed, so the intended division question is not clear from the visible text alone.
- **bridge_0009** — The tutor's question is imprecise/ambiguous because many quadrilaterals could have a longer length than width, though the intended answer appears to be rectangle.
- **bridge_0013** — The actual problem is not visible, so the imprecision cannot be identified from the visible transcript alone.
- **bridge_0024** — Student is ending the session; there is no mathematical error to address.
- **bridge_0036** — The actual task is not visible, so the tutor model could not know from the transcript alone that '2 equal angles' is an insufficient trapezoid criterion.
- **bridge_0037** — The student is merely declining help, not making a mathematical attempt or error.
- **bridge_0041** — The original problem or diagram is missing, so the tutor cannot verify from visible text alone that 7 + 5 is incomplete for the perimeter.
- **bridge_0072** — The student appears to be correct; there is no visible mistake for a tutor to address, and the referent of 'he' is under-specified in the visible text.
- **bridge_0094** — The tutor's statement is mathematically inconsistent/wrong: 4 times 1 is 4, not 48; the student only says yes.
- **bridge_0096** — The visible transcript does not include the actual problem, and 90 appears to be a valid multiple of 10 rather than an error.
- **bridge_0098** — expert reply asserts a specific subtraction error not supported by the visible work
- **bridge_0101** — The student response is incoherent/unclear, so the exact mathematical error is not identifiable beyond not answering the question.
- **bridge_0104** — The student response is a non-answer rather than evidence of a conceptual geometry misconception.
- **bridge_0108** — The tutor's wording is slightly imprecise: 'place value of ten' likely means 'tens place.'
- **bridge_0111** — The student is asking to leave rather than making a mathematical error, so this is not a math-tutoring error turn.
- **bridge_0128** — Student response is misspelled/nonstandard ('hudris'), likely intended as 'hundreds'.
- **bridge_0135** — The student is merely declining help, not making a mathematical error.
- **bridge_0144** — The tutor's wording is confusing: borrowing from the hundreds column would normally make the tens digit 4 become 14, while the expert reply appears to expect one less tha
- **bridge_0163** — Open-ended prior-knowledge prompt; the response is fragmentary rather than a clear math error.
- **bridge_0171** — Student is correct; there is no visible mistake to remediate.
- **bridge_0181** — The tutor's prior wording is awkward and somewhat incoherent, but the mathematical task is still visible.
- **bridge_0189** — Stored topic is about measurement conversion, but this turn is about the meaning of multiplication.
- **bridge_0202** — The expert reply addresses the 5 being in the hundreds place, which does not match the visible question about the digit 2.
- **bridge_0205** — The object/image is not visible, so the exact count is unavailable, but the student's error is identifiable.
- **bridge_0208** — The pictured shape is not visible, so the exact intended answer cannot be confirmed from the transcript alone.
- **bridge_0212** — The tutor's prompt appears to misuse 'inverse operation' when the expert reply expects a commutative/reversed multiplication fact, making the intended task ambiguous.
- **bridge_0228** — The prompt wording 'hundreds values' is somewhat ambiguous, but the intended answer is still inferable from the visible text.


## Criteria at stake


47 scenarios fail the two checks that make an item unanswerable (no error present, or not gradeable), carrying **968 criteria** and **329 critical judgments**.

