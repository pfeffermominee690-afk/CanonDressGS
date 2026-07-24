# Full-Residual Attempt 002 Comparison

Execution is a held-out refinement-condition evaluation for subject02 and the closed set O01/O02/O03/O04/O08. It is not strict unseen-view, unseen-garment, open-world, or cross-identity generalization. Teacher Endpoint is an initialization and comparison endpoint, not an upper bound.

Final classification: `TEACHER_SPAN_AT_LOCAL_OPTIMUM`. `PAPER_FINAL=false`.


| Method | lpips | rgb_mae | psnr | ssim | silhouette_iou | boundary_f | identity_metric |
|---|---|---|---|---|---|---|---|
| Teacher Endpoint | 0.038768 | 0.019046 | 25.780061 | 0.878517 | 0.913184 | 0.715817 | 0.010473 |
| Render-Refined Coefficient | 0.038982 | 0.019202 | 25.751950 | 0.877422 | 0.913312 | 0.716581 | 0.010453 |
| Full-Residual Equal-Step | 0.118421 | 0.078893 | 17.415982 | 0.540771 | 0.851443 | 0.464767 | 0.048736 |
| Full-Residual Equal-Wall-Time | 0.118421 | 0.078893 | 17.415982 | 0.540771 | 0.851443 | 0.464767 | 0.048736 |

Equal-wall-time used the largest registered full-residual checkpoint whose cumulative optimizer-section time did not exceed its selected coefficient run. No interpolation or test-metric selection was used. Cells: 20.


Result HEAD: `85e5d4a3b928ae2566272e4cda2d2e594cf9f39c`. Execution HEAD: `3fb53786a4af264f117701a01e8559c4b2bfafd1`.
