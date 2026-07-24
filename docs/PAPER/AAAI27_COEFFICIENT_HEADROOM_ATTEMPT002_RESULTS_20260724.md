# Coefficient Headroom Attempt 002 Results

Execution is a held-out refinement-condition evaluation for subject02 and the closed set O01/O02/O03/O04/O08. It is not strict unseen-view, unseen-garment, open-world, or cross-identity generalization. Teacher Endpoint is an initialization and comparison endpoint, not an upper bound.

Final classification: `TEACHER_SPAN_AT_LOCAL_OPTIMUM`. `PAPER_FINAL=false`.


## Primary test metrics

| Method | lpips | rgb_mae | psnr | ssim | silhouette_iou | boundary_f | identity_metric |
|---|---|---|---|---|---|---|---|
| Full-Residual Equal-Step | 0.118421 | 0.078893 | 17.415982 | 0.540771 | 0.851443 | 0.464767 | 0.048736 |
| Full-Residual Equal-Wall-Time | 0.118421 | 0.078893 | 17.415982 | 0.540771 | 0.851443 | 0.464767 | 0.048736 |
| Render-Refined Coefficient | 0.038982 | 0.019202 | 25.751950 | 0.877422 | 0.913312 | 0.716581 | 0.010453 |
| SVD Endpoint | 0.038766 | 0.019046 | 25.780054 | 0.878517 | 0.913184 | 0.715816 | 0.010473 |
| Teacher Endpoint | 0.038768 | 0.019046 | 25.780061 | 0.878517 | 0.913184 | 0.715817 | 0.010473 |
| UNREGULARIZED_DIAGNOSTIC | 0.038981 | 0.019208 | 25.750880 | 0.877381 | 0.913312 | 0.716629 | 0.010450 |

## Registered gains

- COEFFICIENT_HEADROOM_GAIN LPIPS improvement: -0.000217
- TEACHER_HEADROOM_GAIN LPIPS improvement: -0.000214
- FULL_RESIDUAL_GAIN LPIPS improvement: -0.079653
