# Bridge criterion-bank pilot — results

8 tutors x all scenarios; **1376 graded responses**, 26,280 criterion judgments. Judge: `openai-group/gpt-5.5`. Tutor system prompt: the real Bridge one.


## Model spread (sanity check)

If the tutors do not differ, nothing below can be read as a property of the criteria.


| model | mean pass rate |
|---|---|
| `openai-group/gpt-4.1-nano` | 0.886 |
| `openai-group/gpt-5.4-nano` | 0.842 |
| `llama3-3-70b` | 0.896 |
| `qwen3-32b` | 0.907 |
| `openai-group/gpt-4o` | 0.888 |
| `mistral-large-3` | 0.926 |
| `openai-group/gpt-5.5` | 0.943 |
| `claude-group/claude-opus-4-8` | 0.960 |


## Q1 — are D1 / P1 / A1 at ceiling?

The system prompt names these three almost verbatim.


| code | pass rate | discrimination | criticality |
|---|---|---|---|
| **D1** | 0.901 | 0.548 | critical |
| **P1** | 0.838 | 0.453 | critical |
| **A1** | 0.850 | -0.042 | standard |


## Q2 — do the 19 negative-form criteria vacuously pass?


- negative-form (11 codes): mean pass rate **0.911**
- positive-form (19 codes): mean pass rate **0.869**


## Q3 — redundancy within a skill family

Phi correlation between sibling criteria on the same responses. Above ~0.7 they are close to one criterion asked twice.


**affective**

| pair | phi | n |
|---|---|---|
| A1 – A3 | +0.091 | 1376 |
| A1 – A4 | +0.034 | 1376 |
| A3 – A4 | -0.004 | 1376 |
| A2 – A3 | -0.009 | 1376 |
| A2 – A4 | -0.011 | 1376 |
| A1 – A2 | -0.069 | 1376 |

**strategy**

| pair | phi | n |
|---|---|---|
| P2 – P4 | +0.578 | 1376 |
| P1 – P3 | +0.385 | 1376 |
| P1 – P2 | +0.345 | 1376 |
| P1 – P4 | +0.288 | 1376 |
| P2 – P3 | +0.078 | 1376 |
| P3 – P4 | +0.043 | 1376 |

**math**

| pair | phi | n |
|---|---|---|
| M3 – M5 | +0.516 | 1376 |
| M3 – M4 | +0.465 | 1376 |
| M1 – M3 | +0.357 | 1376 |
| M1 – M4 | +0.301 | 1376 |
| M1 – M5 | +0.227 | 1376 |
| M4 – M5 | +0.216 | 1376 |

**communication**

| pair | phi | n |
|---|---|---|
| C1 – C3 | +0.221 | 1376 |


## Length bias — is a criterion rewarding words rather than teaching?

Correlation between response length and passing. Positive means longer answers pass more often. Anything beyond ±0.25 is flagged: it is likely measuring verbosity, and a terse expert reply will fail it unfairly.


- mean across 30 criteria: **-0.085**
- flagged (|r| ≥ 0.25): **4** — Y3, S2, Y1, S1

| code | length↔pass | reads as |
|---|---|---|
| D3 | +0.118 | roughly neutral |
| A1 | +0.110 | roughly neutral |
| A2 | +0.085 | roughly neutral |
| G2 | +0.077 | roughly neutral |
| D1 | +0.076 | roughly neutral |
| M4 | +0.060 | roughly neutral |
| S2 | -0.315 | rewards brevity |
| Y1 | -0.350 | rewards brevity |
| S1 | -0.480 | rewards brevity |


## Dead weight — criteria carrying little or no information

Pass rate above 0.95 or below 0.05 means almost no variance to measure; discrimination near zero means the criterion does not track overall quality.


| code | pass rate | disc | flags |
|---|---|---|---|
| A3 | 0.997 | 0.020 | CEILING no-disc |
| A4 | 0.996 | -0.015 | CEILING no-disc |
| I1 | 0.983 | 0.279 | CEILING |
| Y2 | 0.983 | 0.145 | CEILING |
| C1 | 0.977 | 0.265 | CEILING |
| A2 | 0.974 | 0.443 | CEILING |
| Y3 | 0.969 | 0.340 | CEILING |
| G2 | 0.958 | 0.377 | CEILING |
| M5 | 0.952 | 0.453 | CEILING |
| A1 | 0.850 | -0.042 | no-disc |
| D5 | 0.838 | -0.028 | no-disc |


## All criteria


| code | pass rate | disc | crit | form | explicitness |
|---|---|---|---|---|---|
| A3 | 0.997 | 0.020 | critical | pos | implicit |
| A4 | 0.996 | -0.015 | standard | neg | implicit |
| I1 | 0.983 | 0.279 | critical | neg | implicit |
| Y2 | 0.983 | 0.145 | standard | pos | implicit |
| C1 | 0.977 | 0.265 | standard | pos | implicit |
| A2 | 0.974 | 0.443 | standard | pos | implicit |
| Y3 | 0.969 | 0.340 | standard | pos | implicit |
| G2 | 0.958 | 0.377 | standard | pos | implicit |
| M5 | 0.952 | 0.453 | standard | neg | implicit |
| M4 | 0.945 | 0.504 | critical | neg | implicit |
| M1 | 0.922 | 0.412 | critical | pos | implicit |
| D4 | 0.919 | 0.442 | standard | neg | implicit |
| G1 | 0.918 | 0.205 | critical | neg | implicit |
| P4 | 0.916 | 0.580 | standard | neg | implicit |
| I2 | 0.908 | 0.374 | standard | neg | implicit |
| M3 | 0.904 | 0.701 | standard | pos | implicit |
| D1 | 0.901 | 0.548 | critical | pos | explicit |
| C3 | 0.888 | 0.708 | standard | pos | implicit |
| S1 | 0.879 | 0.124 | critical | neg | implicit |
| P2 | 0.852 | 0.768 | standard | pos | implicit |
| A1 | 0.850 | -0.042 | standard | pos | explicit |
| D2 | 0.847 | 0.674 | critical | neg | implicit |
| D5 | 0.838 | -0.028 | standard | pos | implicit |
| P1 | 0.838 | 0.453 | critical | pos | explicit |
| P3 | 0.820 | 0.107 | standard | pos | implicit |
| Y1 | 0.788 | 0.316 | standard | pos | implicit |
| D3 | 0.787 | 0.667 | critical | pos | implicit |
| R1 | 0.760 | 0.419 | critical | neg | implicit |
| R2 | 0.661 | 0.425 | standard | pos | implicit |
| S2 | 0.596 | 0.593 | standard | pos | implicit |
