# Bridge criterion-bank pilot — results

8 tutors x all scenarios; **1903 graded responses**, 38,702 criterion judgments. Judge: `openai-group/gpt-5.5`. Tutor system prompt: the real Bridge one.


## Model spread (sanity check)

If the tutors do not differ, nothing below can be read as a property of the criteria.


| model | mean pass rate |
|---|---|
| `openai-group/gpt-4.1-nano` | 0.817 |
| `openai-group/gpt-5.4-nano` | 0.808 |
| `llama3-3-70b` | 0.837 |
| `qwen3-32b` | 0.861 |
| `openai-group/gpt-4o` | 0.815 |
| `mistral-large-3` | 0.873 |
| `openai-group/gpt-5.5` | 0.931 |
| `claude-group/claude-opus-4-8` | 0.923 |


## Q1 — are D1 / P1 / A1 at ceiling?

The system prompt names these three almost verbatim.


| code | pass rate | discrimination | criticality |
|---|---|---|---|
| **D1** | 0.456 | 0.333 | critical |
| **P1** | 0.822 | 0.494 | critical |
| **A1** | 0.827 | -0.014 | standard |


## Q2 — do the 19 negative-form criteria vacuously pass?


- negative-form (13 codes): mean pass rate **0.903**
- positive-form (25 codes): mean pass rate **0.801**


## Q3 — redundancy within a skill family

Phi correlation between sibling criteria on the same responses. Above ~0.7 they are close to one criterion asked twice.


**affective**

| pair | phi | n |
|---|---|---|
| A3 – A4 | +0.151 | 1903 |
| A1 – A4 | +0.024 | 1903 |
| A1 – A3 | +0.018 | 1903 |
| A2 – A4 | -0.009 | 1903 |
| A2 – A3 | -0.010 | 1903 |
| A1 – A2 | -0.068 | 1903 |

**strategy**

| pair | phi | n |
|---|---|---|
| P2 – P4 | +0.578 | 1903 |
| P1 – P2 | +0.447 | 1903 |
| P1 – P4 | +0.334 | 1903 |
| P1 – P3 | +0.328 | 1903 |
| P2 – P3 | +0.065 | 1903 |
| P3 – P4 | -0.004 | 1903 |

**math**

| pair | phi | n |
|---|---|---|
| M3 – M5 | +0.500 | 1903 |
| M3 – M4 | +0.386 | 1903 |
| M1 – M4 | +0.294 | 1903 |
| M1 – M3 | +0.285 | 1903 |
| M4 – M5 | +0.206 | 1903 |
| M1 – M5 | +0.202 | 1903 |

**communication**

| pair | phi | n |
|---|---|---|
| C1 – C3 | +0.186 | 1903 |


## Length bias — is a criterion rewarding words rather than teaching?

Correlation between response length and passing. Positive means longer answers pass more often. Anything beyond ±0.25 is flagged: it is likely measuring verbosity, and a terse expert reply will fail it unfairly.


- mean across 37 criteria: **-0.052**
- flagged (|r| ≥ 0.25): **7** — F1, V1, C3, U1, Y1, S2, S1

| code | length↔pass | reads as |
|---|---|---|
| F1 | +0.332 | rewards length |
| V1 | +0.291 | rewards length |
| V2 | +0.194 | roughly neutral |
| O1 | +0.171 | roughly neutral |
| D3 | +0.158 | roughly neutral |
| A1 | +0.135 | roughly neutral |
| Y1 | -0.327 | rewards brevity |
| S2 | -0.339 | rewards brevity |
| S1 | -0.497 | rewards brevity |


## Dead weight — criteria carrying little or no information

Pass rate above 0.95 or below 0.05 means almost no variance to measure; discrimination near zero means the criterion does not track overall quality.


| code | pass rate | disc | flags |
|---|---|---|---|
| A4 | 0.997 | 0.025 | CEILING no-disc |
| A3 | 0.996 | -0.014 | CEILING no-disc |
| C1 | 0.978 | 0.235 | CEILING |
| Y2 | 0.978 | 0.128 | CEILING |
| I1 | 0.974 | 0.207 | CEILING |
| A2 | 0.973 | 0.361 | CEILING |
| X1 | 0.963 | 0.329 | CEILING |
| Y3 | 0.959 | 0.400 | CEILING |
| N1 | 0.954 | 0.317 | CEILING |
| P3 | 0.827 | 0.061 | no-disc |
| A1 | 0.827 | -0.014 | no-disc |
| V1 | 0.635 | 0.074 | no-disc |


## All criteria


| code | pass rate | disc | crit | form | explicitness |
|---|---|---|---|---|---|
| A4 | 0.997 | 0.025 | standard | neg | implicit |
| A3 | 0.996 | -0.014 | critical | pos | implicit |
| C1 | 0.978 | 0.235 | standard | pos | implicit |
| Y2 | 0.978 | 0.128 | standard | pos | implicit |
| I1 | 0.974 | 0.207 | critical | neg | implicit |
| A2 | 0.973 | 0.361 | standard | pos | implicit |
| X1 | 0.963 | 0.329 | standard | pos | implicit |
| Y3 | 0.959 | 0.400 | standard | pos | implicit |
| N1 | 0.954 | 0.317 | standard | neg | implicit |
| G2 | 0.940 | 0.413 | standard | pos | implicit |
| M4 | 0.939 | 0.409 | critical | neg | implicit |
| V2 | 0.936 | 0.513 | standard | pos | implicit |
| U1 | 0.934 | 0.486 | standard | neg | implicit |
| M5 | 0.931 | 0.510 | standard | neg | implicit |
| M1 | 0.916 | 0.354 | critical | pos | implicit |
| G1 | 0.915 | 0.279 | critical | neg | implicit |
| I2 | 0.906 | 0.248 | standard | neg | implicit |
| P4 | 0.885 | 0.590 | standard | neg | implicit |
| D4 | 0.872 | 0.505 | standard | neg | implicit |
| S1 | 0.865 | 0.189 | critical | neg | implicit |
| M3 | 0.845 | 0.692 | standard | pos | implicit |
| C3 | 0.844 | 0.693 | standard | pos | implicit |
| P3 | 0.827 | 0.061 | standard | pos | implicit |
| A1 | 0.827 | -0.014 | standard | pos | explicit |
| P1 | 0.822 | 0.494 | critical | pos | explicit |
| P2 | 0.799 | 0.789 | standard | pos | implicit |
| D5 | 0.796 | 0.170 | standard | pos | implicit |
| D2 | 0.783 | 0.720 | critical | neg | implicit |
| Y1 | 0.779 | 0.310 | standard | pos | implicit |
| R1 | 0.777 | 0.409 | critical | neg | implicit |
| O1 | 0.688 | 0.123 | standard | pos | implicit |
| D3 | 0.676 | 0.783 | critical | pos | implicit |
| Z1 | 0.675 | 0.182 | standard | pos | implicit |
| R2 | 0.654 | 0.387 | standard | pos | implicit |
| V1 | 0.635 | 0.074 | standard | pos | implicit |
| S2 | 0.597 | 0.572 | standard | pos | implicit |
| F1 | 0.464 | 0.331 | standard | pos | implicit |
| D1 | 0.456 | 0.333 | critical | pos | explicit |
