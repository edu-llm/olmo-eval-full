# TutorBench scale comparison @ floor 20 / SE_ability 0.27

| metric | 2-skill correctness | 2-skill scaffolding | correctness-only unidim |
|---|---:|---:|---:|
| SE_ability (median, param-unc. bootstrap) | 0.3337 | 0.3073 | 0.3258 |
| SE_param (median) | 0.2074 | 0.1749 | 0.1126 |
| SE_total (median) | 0.4117 | 0.3592 | 0.3441 |
| OOS recovery r @20/0.27 | 0.9667 | 0.9321 | 0.9550 |
| slope @20/0.27 | 1.0187 | 0.9527 | 1.0055 |
| theta MAE @20/0.27 | 0.4368 | 0.2564 | 0.4159 |
| median test length (scenarios) | 25.0000 | 25.0000 | 20.0000 |
| %reach SE_ability<=0.27 | 43.3628 | 69.0265 | 59.6491 |

Notes:
- SE_ability / SE_param / SE_total are medians from the scenario-level parameter-uncertainty bootstrap (N=115, eap-grid 61, n-boot 150, min-evals-per-skill 12, max-se 0.30). SE_ability == posterior SE; SE_total = sqrt(SE_ability^2 + SE_param^2).
- OOS r / slope / θMAE / length / %reach are from the k=5 refit-per-fold OOS grids at floor 20 / SE_ability 0.27 (headline pool excludes flagged models).
- median test length is the JOINT administered test; for the 2-skill scale one test measures both skills simultaneously (25 scenarios), unidim measures the single axis (20).
- %reach shown is SE_ability<=0.27 per axis. 2-skill BOTH-skills reach = 33.6%; unidim combined (SE_ability<=0.27 AND SE_total<=0.30) = 57.9%.
