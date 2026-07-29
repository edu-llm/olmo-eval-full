# Bridge criterion-bank pilot — results

8 tutors x 100 scenarios; **790 graded responses**, 16,177 criterion judgments. Judge: `openai-group/gpt-5.5`. Tutor system prompt: the real Bridge one.


## Model spread (sanity check)

If the tutors do not differ, nothing below can be read as a property of the criteria.


| model | mean pass rate |
|---|---|
| `openai-group/gpt-4.1-nano` | 0.732 |
| `openai-group/gpt-5.4-nano` | 0.690 |
| `llama3-3-70b` | 0.686 |
| `qwen3-32b` | 0.708 |
| `openai-group/gpt-4o` | 0.678 |
| `mistral-large-3` | 0.733 |
| `openai-group/gpt-5.5` | 0.776 |
| `claude-group/claude-opus-4-8` | 0.763 |


## Q1 — are D1 / P1 / A1 at ceiling?

The system prompt names these three almost verbatim.


| code | pass rate | discrimination | criticality |
|---|---|---|---|
| **D1** | 0.222 | 0.486 | critical |
| **P1** | 0.618 | 0.594 | critical |
| **A1** | 0.803 | 0.030 | standard |


## Q2 — do the 19 negative-form criteria vacuously pass?


- negative-form (14 codes): mean pass rate **0.831**
- positive-form (25 codes): mean pass rate **0.608**


## Q3 — redundancy within a skill family

Phi correlation between sibling criteria on the same responses. Above ~0.7 they are close to one criterion asked twice.


**affective**

| pair | phi | n |
|---|---|---|
| A3 – A4 | +0.754 | 790 |
| A1 – A3 | +0.099 | 790 |
| A1 – A4 | +0.089 | 790 |
| A2 – A3 | -0.020 | 790 |
| A2 – A4 | -0.026 | 790 |
| A1 – A2 | -0.061 | 790 |

**strategy**

| pair | phi | n |
|---|---|---|
| P2 – P4 | +0.542 | 790 |
| P1 – P2 | +0.516 | 790 |
| P1 – P4 | +0.359 | 790 |
| P1 – P3 | +0.234 | 790 |
| P3 – P4 | +0.021 | 790 |
| P2 – P3 | -0.012 | 790 |

**math**

| pair | phi | n |
|---|---|---|
| M3 – M5 | +0.363 | 790 |
| M1 – M4 | +0.335 | 790 |
| M4 – M5 | +0.294 | 790 |
| M3 – M4 | +0.178 | 790 |
| M1 – M5 | +0.134 | 790 |
| M1 – M3 | +0.025 | 790 |

**communication**

| pair | phi | n |
|---|---|---|
| C1 – C3 | +0.103 | 790 |


## Length bias — is a criterion rewarding words rather than teaching?

Correlation between response length and passing. Positive means longer answers pass more often. Anything beyond ±0.25 is flagged: it is likely measuring verbosity, and a terse expert reply will fail it unfairly.


- mean across 38 criteria: **-0.054**
- flagged (|r| ≥ 0.25): **8** — V1, C1, B1, Y2, Y1, Y3, D4, S1

| code | length↔pass | reads as |
|---|---|---|
| V1 | +0.271 | rewards length |
| D5 | +0.247 | roughly neutral |
| V2 | +0.215 | roughly neutral |
| D3 | +0.210 | roughly neutral |
| A2 | +0.153 | roughly neutral |
| A1 | +0.148 | roughly neutral |
| Y3 | -0.363 | rewards brevity |
| D4 | -0.387 | rewards brevity |
| S1 | -0.486 | rewards brevity |


## Dead weight — criteria carrying little or no information

Pass rate above 0.95 or below 0.05 means almost no variance to measure; discrimination near zero means the criterion does not track overall quality.


| code | pass rate | disc | flags |
|---|---|---|---|
| A3 | 0.995 | 0.072 | CEILING no-disc |
| A4 | 0.991 | 0.059 | CEILING no-disc |
| C1 | 0.982 | 0.148 | CEILING |
| Y3 | 0.976 | 0.067 | CEILING no-disc |
| G1 | 0.943 | 0.098 | no-disc |
| P3 | 0.878 | 0.030 | no-disc |
| S1 | 0.855 | 0.095 | no-disc |
| A1 | 0.803 | 0.030 | no-disc |
| D5 | 0.592 | 0.082 | no-disc |


## All criteria


| code | pass rate | disc | crit | form | explicitness |
|---|---|---|---|---|---|
| A3 | 0.995 | 0.072 | critical | pos | implicit |
| A4 | 0.991 | 0.059 | standard | neg | implicit |
| C1 | 0.982 | 0.148 | standard | pos | implicit |
| Y3 | 0.976 | 0.067 | standard | pos | implicit |
| G1 | 0.943 | 0.098 | critical | neg | implicit |
| M1 | 0.942 | 0.159 | critical | pos | implicit |
| N1 | 0.938 | 0.212 | standard | neg | implicit |
| M4 | 0.937 | 0.273 | critical | neg | implicit |
| A2 | 0.930 | 0.281 | standard | pos | implicit |
| I1 | 0.927 | 0.375 | critical | neg | implicit |
| Y2 | 0.923 | 0.284 | standard | pos | implicit |
| B1 | 0.917 | 0.343 | standard | neg | implicit |
| I2 | 0.906 | 0.115 | standard | neg | implicit |
| P3 | 0.878 | 0.030 | standard | pos | implicit |
| S1 | 0.855 | 0.095 | critical | neg | implicit |
| D4 | 0.854 | 0.242 | standard | neg | implicit |
| M5 | 0.847 | 0.445 | standard | neg | implicit |
| G2 | 0.833 | 0.444 | standard | pos | implicit |
| X1 | 0.822 | 0.511 | standard | pos | implicit |
| U1 | 0.807 | 0.540 | standard | neg | implicit |
| A1 | 0.803 | 0.030 | standard | pos | explicit |
| P4 | 0.753 | 0.511 | standard | neg | implicit |
| P1 | 0.618 | 0.594 | critical | pos | explicit |
| Y1 | 0.615 | 0.336 | standard | pos | implicit |
| D5 | 0.592 | 0.082 | standard | pos | implicit |
| V2 | 0.591 | 0.581 | standard | pos | implicit |
| R1 | 0.545 | 0.477 | critical | neg | implicit |
| C3 | 0.527 | 0.737 | standard | pos | implicit |
| S2 | 0.524 | 0.366 | standard | pos | implicit |
| P2 | 0.494 | 0.773 | standard | pos | implicit |
| M3 | 0.441 | 0.750 | standard | pos | implicit |
| D2 | 0.409 | 0.771 | critical | neg | implicit |
| V1 | 0.409 | 0.416 | standard | pos | implicit |
| R2 | 0.388 | 0.496 | standard | pos | implicit |
| O1 | 0.257 | 0.517 | standard | pos | implicit |
| D1 | 0.222 | 0.486 | critical | pos | explicit |
| F1 | 0.179 | 0.132 | standard | pos | implicit |
| D3 | 0.141 | 0.582 | critical | pos | implicit |
| Z1 | 0.109 | 0.283 | standard | pos | implicit |
