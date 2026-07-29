# Bridge criterion-bank pilot — results

8 tutors x 100 scenarios; **790 graded responses**, 16,177 criterion judgments. Judge: `openai-group/gpt-5.5`. Tutor system prompt: the real Bridge one.


## Model spread (sanity check)

If the tutors do not differ, nothing below can be read as a property of the criteria.


| model | mean pass rate |
|---|---|
| `openai-group/gpt-4.1-nano` | 0.729 |
| `openai-group/gpt-5.4-nano` | 0.687 |
| `llama3-3-70b` | 0.674 |
| `qwen3-32b` | 0.687 |
| `openai-group/gpt-4o` | 0.676 |
| `mistral-large-3` | 0.712 |
| `openai-group/gpt-5.5` | 0.780 |
| `claude-group/claude-opus-4-8` | 0.752 |


## Q1 — are D1 / P1 / A1 at ceiling?

The system prompt names these three almost verbatim.


| code | pass rate | discrimination | criticality |
|---|---|---|---|
| **D1** | 0.218 | 0.499 | critical |
| **P1** | 0.594 | 0.629 | critical |
| **A1** | 0.914 | 0.091 | standard |


## Q2 — do the 19 negative-form criteria vacuously pass?


- negative-form (13 codes): mean pass rate **0.817**
- positive-form (26 codes): mean pass rate **0.595**


## Q3 — redundancy within a skill family

Phi correlation between sibling criteria on the same responses. Above ~0.7 they are close to one criterion asked twice.


**affective**

| pair | phi | n |
|---|---|---|
| A2 – A4 | +0.369 | 790 |
| A1 – A4 | +0.323 | 790 |
| A1 – A2 | +0.270 | 790 |
| A1 – A3 | +0.128 | 790 |
| A2 – A3 | +0.088 | 790 |
| A3 – A4 | +0.071 | 790 |

**strategy**

| pair | phi | n |
|---|---|---|
| P1 – P2 | +0.561 | 790 |
| P2 – P4 | +0.499 | 790 |
| P1 – P4 | +0.382 | 790 |
| P1 – P3 | +0.174 | 790 |
| P2 – P3 | +0.001 | 790 |
| P3 – P4 | -0.021 | 790 |

**math**

| pair | phi | n |
|---|---|---|
| M1 – M4 | +0.379 | 790 |
| M3 – M5 | +0.332 | 790 |
| M4 – M5 | +0.281 | 790 |
| M1 – M5 | +0.175 | 790 |
| M3 – M4 | +0.155 | 790 |
| M1 – M3 | +0.063 | 790 |

**communication**

| pair | phi | n |
|---|---|---|
| C1 – C3 | +0.188 | 790 |


## Dead weight — criteria carrying little or no information

Pass rate above 0.95 or below 0.05 means almost no variance to measure; discrimination near zero means the criterion does not track overall quality.


| code | pass rate | disc | flags |
|---|---|---|---|
| A3 | 0.996 | 0.011 | CEILING no-disc |
| A1 | 0.914 | 0.091 | no-disc |
| I2 | 0.896 | -0.080 | no-disc |
| P3 | 0.881 | 0.007 | no-disc |
| S1 | 0.863 | 0.092 | no-disc |
| D5 | 0.596 | 0.093 | no-disc |
| F1 | 0.143 | 0.029 | no-disc |


## All criteria


| code | pass rate | disc | crit | form | explicitness |
|---|---|---|---|---|---|
| A3 | 0.996 | 0.011 | critical | pos | implicit |
| C1 | 0.947 | 0.252 | standard | pos | implicit |
| M4 | 0.942 | 0.257 | critical | neg | implicit |
| A4 | 0.941 | 0.235 | standard | pos | implicit |
| N1 | 0.938 | 0.102 | standard | neg | implicit |
| M1 | 0.937 | 0.156 | critical | pos | implicit |
| G1 | 0.927 | 0.115 | critical | neg | implicit |
| B1 | 0.927 | 0.255 | standard | neg | implicit |
| Y2 | 0.918 | 0.307 | standard | pos | implicit |
| A1 | 0.914 | 0.091 | standard | pos | explicit |
| I1 | 0.906 | 0.331 | critical | neg | implicit |
| I2 | 0.896 | -0.080 | standard | neg | implicit |
| P3 | 0.881 | 0.007 | standard | pos | implicit |
| D4 | 0.873 | 0.181 | standard | neg | implicit |
| S1 | 0.863 | 0.092 | critical | neg | implicit |
| G2 | 0.854 | 0.368 | standard | pos | implicit |
| A2 | 0.847 | 0.371 | standard | pos | implicit |
| M5 | 0.844 | 0.421 | standard | neg | implicit |
| U1 | 0.818 | 0.591 | standard | neg | implicit |
| P4 | 0.763 | 0.486 | standard | neg | implicit |
| X1 | 0.756 | 0.536 | standard | pos | implicit |
| Y3 | 0.636 | 0.628 | standard | pos | implicit |
| Y1 | 0.630 | 0.256 | standard | pos | implicit |
| D5 | 0.596 | 0.093 | standard | pos | implicit |
| P1 | 0.594 | 0.629 | critical | pos | explicit |
| V2 | 0.561 | 0.483 | standard | pos | implicit |
| C3 | 0.538 | 0.747 | standard | pos | implicit |
| S2 | 0.524 | 0.423 | standard | pos | implicit |
| R1 | 0.521 | 0.532 | critical | neg | implicit |
| P2 | 0.501 | 0.759 | standard | pos | implicit |
| M3 | 0.441 | 0.731 | standard | pos | implicit |
| D2 | 0.405 | 0.748 | critical | neg | implicit |
| R2 | 0.339 | 0.486 | standard | pos | implicit |
| V1 | 0.323 | 0.292 | standard | pos | implicit |
| O1 | 0.239 | 0.458 | standard | pos | implicit |
| D1 | 0.218 | 0.499 | critical | pos | explicit |
| F1 | 0.143 | 0.029 | standard | pos | implicit |
| Z1 | 0.109 | 0.274 | standard | pos | implicit |
| D3 | 0.099 | 0.575 | critical | pos | implicit |
