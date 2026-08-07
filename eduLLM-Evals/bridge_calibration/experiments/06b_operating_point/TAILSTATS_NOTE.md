# Tail-aware per-cell SE stats (EAP-stop floor x SE grid)

**Re-summarize only** -- frozen bank reused verbatim (no refit), deployed CAT on the full 51-model data. SE_ability = fine-grid EAP posterior SD (321 nodes / +/-8) at the stop; SE_param = observed-info bootstrap over the administered set; SE_total = sqrt(SE_ability^2 + SE_param^2). `recovery_r` is carried over (read-only) from the grid CSVs (OOS k-fold; not recomputed). No of-record file was modified.

## Why the median hid the SE-target benefit
On Bridge's heavy ~18-criterion testlets the MEDIAN model is already well below any SE target once the min_scenarios floor is met, so `se_total_median` is ~flat across SE targets. The SE target only helps the TAIL (low-information / extreme-ability models). The columns below expose that tail: as the SE target tightens, `se_total` mean / SD / max and the counts above 0.15 / 0.20 drop, at the cost of rising mean / max length.

## EAP stop -- floor 12
| SE target | mean scen (max) | recovery r | SE_total mean | SD | median | max | #>0.15 | #>0.20 |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 12.1 (16) | 0.9517 | 0.137 | 0.059 | 0.116 | 0.370 | 13 | 5 |
| 0.25 | 12.2 (19) | 0.9517 | 0.134 | 0.051 | 0.115 | 0.312 | 13 | 4 |
| 0.20 | 12.6 (29) | 0.9517 | 0.133 | 0.046 | 0.115 | 0.272 | 13 | 4 |
| 0.15 | 14.1 (47) | 0.9515 | 0.126 | 0.033 | 0.117 | 0.203 | 14 | 1 |
| 0.12 | 17.1 (60) | 0.9579 | 0.118 | 0.023 | 0.117 | 0.195 | 2 | 0 |
| 0.10 | 22.6 (60) | 0.9639 | 0.109 | 0.019 | 0.106 | 0.194 | 2 | 0 |
| 0.08 | 31.5 (60) | 0.9758 | 0.098 | 0.021 | 0.092 | 0.196 | 2 | 0 |

## EAP stop -- floor 8
| SE target | mean scen (max) | recovery r | SE_total mean | SD | median | max | #>0.15 | #>0.20 |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 8.2 (16) | 0.9374 | 0.162 | 0.062 | 0.131 | 0.363 | 20 | 15 |
| 0.25 | 8.4 (19) | 0.9374 | 0.160 | 0.055 | 0.130 | 0.297 | 19 | 15 |
| 0.20 | 9.0 (29) | 0.9393 | 0.154 | 0.046 | 0.131 | 0.265 | 19 | 14 |
| 0.15 | 11.2 (47) | 0.9388 | 0.140 | 0.026 | 0.131 | 0.207 | 18 | 1 |
| 0.12 | 14.8 (60) | 0.946 | 0.129 | 0.016 | 0.127 | 0.193 | 3 | 0 |
| 0.10 | 21.1 (60) | 0.9559 | 0.114 | 0.017 | 0.110 | 0.194 | 2 | 0 |
| 0.08 | 31.5 (60) | 0.9728 | 0.098 | 0.021 | 0.093 | 0.198 | 2 | 0 |

## Online stop -- floor 8 (side-by-side)
| SE target | mean scen (max) | recovery r | SE_total mean | SD | median | max | #>0.15 | #>0.20 |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 8.0 (8) | 0.937 | 0.166 | 0.076 | 0.131 | 0.475 | 20 | 14 |
| 0.25 | 8.0 (8) | 0.937 | 0.166 | 0.074 | 0.131 | 0.467 | 19 | 14 |
| 0.20 | 8.0 (8) | 0.937 | 0.166 | 0.074 | 0.131 | 0.470 | 20 | 15 |
| 0.15 | 8.8 (14) | 0.9401 | 0.157 | 0.069 | 0.131 | 0.478 | 19 | 8 |
| 0.12 | 11.9 (26) | 0.9378 | 0.136 | 0.035 | 0.124 | 0.302 | 10 | 3 |
| 0.10 | 18.6 (60) | 0.9525 | 0.119 | 0.021 | 0.114 | 0.198 | 5 | 0 |
| 0.08 | 30.5 (60) | 0.9693 | 0.099 | 0.022 | 0.093 | 0.203 | 2 | 1 |

## Online stop -- floor 12 (side-by-side)
| SE target | mean scen (max) | recovery r | SE_total mean | SD | median | max | #>0.15 | #>0.20 |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 12.0 (12) | 0.9519 | 0.139 | 0.070 | 0.116 | 0.482 | 13 | 4 |
| 0.25 | 12.0 (12) | 0.9519 | 0.139 | 0.070 | 0.117 | 0.477 | 13 | 3 |
| 0.20 | 12.0 (12) | 0.9519 | 0.139 | 0.070 | 0.116 | 0.477 | 13 | 4 |
| 0.15 | 12.1 (14) | 0.9519 | 0.139 | 0.070 | 0.116 | 0.480 | 13 | 5 |
| 0.12 | 14.3 (26) | 0.9532 | 0.126 | 0.042 | 0.116 | 0.319 | 6 | 3 |
| 0.10 | 20.1 (60) | 0.961 | 0.112 | 0.022 | 0.105 | 0.200 | 4 | 0 |
| 0.08 | 30.5 (60) | 0.9728 | 0.099 | 0.022 | 0.092 | 0.195 | 2 | 0 |

## Reading the tradeoff
- `se_total_median` ~flat across SE targets => the median model does not benefit.
- `se_total_mean`, `se_total_sd`, `se_total_max`, `#>0.15`, `#>0.20` FALL as SE tightens => the tail is pulled in.
- `scen_mean (max)` RISES as SE tightens => the cost is paid only by the tail (more scenarios for the few under-measured models); the floor-bound majority is unchanged.
- Net: the SE target is a TAIL-precision guarantee, not a median-precision lever. Pick it by how many tail models you are willing to leave above a given SE_total, versus the extra length those models incur.
