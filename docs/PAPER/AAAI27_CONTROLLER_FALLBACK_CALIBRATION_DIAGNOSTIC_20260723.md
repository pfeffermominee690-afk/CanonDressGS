# Controller fallback and calibration diagnostic

Status: **diagnostic complete; no repair applied**. Final classification: `MULTIPLE_FACTORS`. PAPER_FINAL=0.

## Formal-result reaggregation

The frozen formal archive reproduced 3 seeds, 240 mixed records per seed (720 total), 80 pure protocol records per seed, 320 protocol records per seed, 10 unordered pairs, AAB/ABB, three assignment positions and four target-view folds. Formal top-2 pair accuracy, ordering, activation, calibration and visual counts reaggregated exactly.

## Fallback decomposition

| Outcome | Count | Rate |
|---|---:|---:|
| DUAL_SUPPORT | 366 | 50.83% |
| LOW_TOP2_MASS | 115 | 15.97% |
| LOW_SECONDARY_WEIGHT | 239 | 33.19% |

The logical both-condition hit count is 1 and remains assigned to LOW_TOP2_MASS by the frozen priority. Correct-pair fallback occurs in 330 records; 306 are also ordering-correct. Wrong-pair DUAL exposure is 3; wrong-ordering DUAL exposure is 50.

## Pure/mixed separability and frozen grid

Rank separation is strong: entropy AUROC/AUPRC=0.976065/0.997900, secondary normalized weight=0.984398/0.998647, top-1 probability=0.977477/0.998031, and top-2 mass=0.930093/0.993287. Nevertheless, none of the 49 preregistered grid combinations satisfies the joint pure, mixed, wrong-pair and per-seed constraints: `SIMPLE_GATE_NOT_SEPARABLE`. No threshold was selected; 0.90/0.10 remains unchanged.

The closest high-activation diagnostic point (top2 mass 0.50, secondary weight 0.05) has pure SINGLE=100%, mixed DUAL=81.94%, but wrong-pair DUAL=66.67%, violating the <=10% constraint. Thus good marginal separability does not imply a safe simple deployment gate.

## Weight calibration

On 693 correct-pair queries, correct-order records have MAE=0.173971; order-wrong records have MAE=0.266158. DUAL-active MAE=0.137719, while fallback MAE=0.239180. Absolute weight error and LPIPS correlate at r=0.583129. No temperature fit, calibration model, or post-hoc replacement was performed.

## Decision

Fallback calibration, weight calibration, pair compatibility and reference robustness make independent contributions. Pair selection has only a small aggregate causal gain. Exact oracle pair/weight still has grade-3 failures for O01_O03 and O02_O03, so the diagnosis is `MULTIPLE_FACTORS`, with pair compatibility the largest residual visual source. This is not a Controller repair, threshold optimization, continuous-control pass, or unseen/cross-identity/novel-view claim.

Manual review is complete: 60/60 causal sheets and 11/11 diagnostic sheets. No scientific failure was rerun.
