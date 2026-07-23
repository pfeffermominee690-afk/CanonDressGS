# AAAI27 Controller Failure Reinterpretation

Status: `PASS`

Historical overall classification remains `CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL`.

Primary failure interpretation: `MIXED_PAIR_IDENTIFICATION_PARTIAL`.

The corrected V2 mixed-only pair macro is `0.718056`. It no longer meets `CORE_PAIR_IDENTIFICATION_FAILURE_REMAINS`, but it remains below the supported range and at least one rotation is below `0.80`. Routing and weight also remain below their preregistered gates: compatible DUAL `0.189378`, incompatible HARD `0.313889`, and correct-pair weight MAE `0.158058`.

| Gate | Historical value | Historical | Corrected value | Corrected | Corrected denominator |
|---|---:|---|---:|---|---|
| pair_macro_top2_ge_0_90 -> mixed_pair_macro_ge_0_90 | 0.6010416666666667 | FAIL | 0.7180555555555556 | FAIL | GT_MIXED_ONLY |
| every_rotation_top2_ge_0_80 -> mixed_pair_every_rotation_ge_0_80 | 0.5791666666666666 | FAIL | 0.6888888888888889 | FAIL | GT_MIXED_ONLY |
| pure_single_ge_0_95 -> pure_single_ge_0_95 | 1.0 | PASS | 1.0 | PASS | GT_PURE_ONLY |
| pure_false_mixed_le_0_05 -> pure_false_mixed_le_0_05 | 0.05 | PASS | 0.05 | PASS | GT_PURE_ONLY |
| compatible_dual_ge_0_80 -> compatible_dual_ge_0_80 | 0.18937812421128092 | FAIL | 0.18937812421128086 | FAIL | CORRECT_PAIR_COMPATIBLE_MIXED |
| incompatible_hard_ge_0_80 -> incompatible_hard_ge_0_80 | 0.3138888888888889 | FAIL | 0.3138888888888889 | FAIL | CORRECT_PAIR_INCOMPATIBLE_MIXED |
| incompatible_dual_le_0_10 -> incompatible_dual_le_0_10 | 0.0 | PASS | 0.0 | PASS | CORRECT_PAIR_INCOMPATIBLE_MIXED |
| wrong_pair_dual_le_0_10 -> wrong_pair_dual_le_0_10 | 0.07282421740626076 | PASS | 0.07282421740626076 | PASS | WRONG_PAIR_MIXED |
| correct_pair_weight_mae_le_0_12 -> correct_pair_weight_mae_le_0_12 | 0.15805836898719514 | FAIL | 0.15805836898719514 | FAIL | CORRECT_PAIR_MIXED |
| severe_artifact_reduction_ge_0_50 -> severe_artifact_reduction_ge_0_50 | 0.73 | PASS | 0.73 | PASS | HISTORICAL_FIXED_240_SHEET_VISUAL_DENOMINATOR |
| identity_contamination_eq_0 -> identity_contamination_eq_0 | 0 | PASS | 0 | PASS | HISTORICAL_FIXED_240_SHEET_VISUAL_DENOMINATOR |
| grade_3_ghosting_pairs_le_1_of_10 -> grade_3_ghosting_pairs_le_1_of_10 | 0 | PASS | 0 | PASS | HISTORICAL_FIXED_PAIR_VISUAL_DENOMINATOR |
| dropout_and_single_reference_safe_single_ge_0_95 -> dropout_and_single_reference_safe_single_ge_0_95 | 1.0 | PASS | 1.0 | PASS | HISTORICAL_INFORMATION_ABLATION_DENOMINATOR |

Only the semantically illegal pair denominator changed. Numeric thresholds, routing denominators, visual denominators, information-ablation definitions, compatibility labels, and historical visual selections are unchanged.

NEXT_TASK: `DIAGNOSE_CONTROLLER_GARMENT_HEAD_OPTIMIZATION_BUDGET`. It was not started.
