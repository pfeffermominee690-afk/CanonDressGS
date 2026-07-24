# Render-Refined Coefficient Attempt 002

Execution is a held-out refinement-condition evaluation for subject02 and the closed set O01/O02/O03/O04/O08. It is not strict unseen-view, unseen-garment, open-world, or cross-identity generalization. Teacher Endpoint is an initialization and comparison endpoint, not an upper bound.

Final classification: `TEACHER_SPAN_AT_LOCAL_OPTIMUM`. `PAPER_FINAL=false`.


Selected positive anchors: R0=0.1, R1=0.1, R2=0.1, R3=0.1.

Selection used only each rotation's calibration fold, the five-garment macro render objective, step 300, and the frozen absolute tie tolerance 1e-4. The stronger lambda wins a tie. Test observations were not used.

| Method | lpips | rgb_mae | psnr | ssim | silhouette_iou | boundary_f | identity_metric |
|---|---|---|---|---|---|---|---|
| Teacher Endpoint | 0.038768 | 0.019046 | 25.780061 | 0.878517 | 0.913184 | 0.715817 | 0.010473 |
| SVD Endpoint | 0.038766 | 0.019046 | 25.780054 | 0.878517 | 0.913184 | 0.715816 | 0.010473 |
| Render-Refined Coefficient | 0.038982 | 0.019202 | 25.751950 | 0.877422 | 0.913312 | 0.716581 | 0.010453 |
| UNREGULARIZED_DIAGNOSTIC | 0.038981 | 0.019208 | 25.750880 | 0.877381 | 0.913312 | 0.716629 | 0.010450 |


Result HEAD: `85e5d4a3b928ae2566272e4cda2d2e594cf9f39c`. Execution HEAD: `3fb53786a4af264f117701a01e8559c4b2bfafd1`.
