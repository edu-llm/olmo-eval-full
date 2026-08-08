# Calibration study summary

> **Provisional result.** The study selected a one-dimensional
> `instruction_following` structure and exported 2,096 fitted criteria. The CAT
> efficiency comparison used five-point EAP reference scores, which collapse ability
> estimates onto five theta nodes. Dense one-dimensional rescoring and an
> InFoBench-specific human audit of the Qwen judge remain pending; do not treat the
> current recovery or CAT-efficiency values as final production claims.

## Skill-structure decision

The automatic one-standard-error rule recommends **overall_1d**. The reported selection is **overall_1d** (1 latent dimension(s)).

The best mean held-out log_loss came from **overall_1d** at 0.4070. Its fold standard error was 0.0135, giving a one-SE boundary of 0.4205. Structures inside that boundary: overall_1d, correlation_2d, correlation_3d, merge_content_style_4d, merge_format_linguistic_4d, merge_number_linguistic_4d, full_5d.

The rule first keeps structures whose held-out performance is within one standard error of the best, then chooses the one with the fewest dimensions. AIC/BIC and parameter stability remain supporting diagnostics.

## CAT sweep aggregation

Aggregated 12 CAT run(s) using paired model-bootstrap 95% confidence intervals. Recovery is reported as correlation, slope, and MAE; test length is reported as mean scenarios and criteria where available.

The study-selected provisional CAT configuration used a minimum of **0 scenarios**, an SE target of **0.25**, and the **trace** selector.
- **selected_configuration** used -18.40 scenarios versus **random_baseline** (95% CI -21.46 to -15.58).

## Held-out estimator recovery

- **eap**: r 0.902 (95% CI 0.834 to 0.950); slope 0.777 (95% CI 0.697 to 0.869); mae 0.498 (95% CI 0.329 to 0.670).
- **mwle**: r 0.924 (95% CI 0.876 to 0.958); slope 0.845 (95% CI 0.763 to 0.936); mae 0.562 (95% CI 0.446 to 0.687).
- **online**: r 0.910 (95% CI 0.854 to 0.950); slope 0.595 (95% CI 0.539 to 0.664); mae 0.724 (95% CI 0.572 to 0.879).

### p-IRT pass-rate calibration

- **eap**: mae 0.077; bias -0.003; r 0.929.
- **full_eap**: mae 0.044; bias -0.004; r 0.983.
- **mwle**: mae 0.069; bias -0.001; r 0.944.
- **online**: mae 0.092; bias 0.003; r 0.937.

MAE is absolute predicted-vs-observed pass-rate error; bias is predicted minus observed; r measures whether models are ranked similarly by predicted and observed pass rate.

These are honest held-out-person results: each model was excluded from the item-parameter fit used to score it. Confidence intervals resample tutor models.

## Grid and ridge sensitivity

Compared 9 supplied sensitivity runs. The lowest held-out fold mean was **grid7_ridge0p1** at 0.3919 (grid=7, ridge=0.1).
Sensitivity results are diagnostics; this report does not silently replace the predeclared selected structure or settings.

## Scenario-order stability

Across the supplied seeds, MWLE's median across-model theta SD was 0.351 across dimensions; the largest reported model SD was 0.827.
Seeds tested: 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007; run failures: 0.

## Item-parameter uncertainty and total SE

- **multi_target** at requested SE 0.20: median MWLE total SE across dimensions 0.400; at least 52 reliable models per dimension; mean scenarios 7.7; precision reached 100.0%.
- **multi_target** at requested SE 0.25: median MWLE total SE across dimensions 0.537; at least 52 reliable models per dimension; mean scenarios 5.3; precision reached 100.0%.
- **multi_target** at requested SE 0.30: median MWLE total SE across dimensions 0.584; at least 52 reliable models per dimension; mean scenarios 4.8; precision reached 100.0%.
- **multi_target** at requested SE 0.35: median MWLE total SE across dimensions 0.606; at least 52 reliable models per dimension; mean scenarios 4.8; precision reached 100.0%.

Total SE combines conditional ability uncertainty and item-parameter uncertainty in quadrature; it is not the same as the CAT stopping SE.

## Figures

- Structure Cv: `figures/structure_cv.png`
- Cat Length Precision Tradeoff: `figures/cat_length_precision_tradeoff.png`
- Estimator Recovery: `figures/estimator_recovery.png`
- Total Se Vs Target: `figures/total_se_vs_target.png`

## Interpretation guardrail

This report compares the supplied candidate structures; it does not prove that an untested structure is worse. Fold uncertainty is based on a small number of folds, while CAT confidence intervals resample tutor models. Recovery against the coarse five-node EAP reference and CAT settings chosen using it remain provisional. The CAT replay also reuses the calibration cohort rather than an independent external cohort.
