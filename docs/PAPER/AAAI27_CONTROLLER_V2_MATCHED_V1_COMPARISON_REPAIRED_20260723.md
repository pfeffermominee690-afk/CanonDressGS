# Repaired Controller V2 versus Matched V1

Both families used the same four cross-fit rotations, three seeds, frozen data
order, 150-step budget, final-checkpoint-only rule, and clean record exposure.
V2 added the preregistered nuisance augmentation stream and used 9,232
trainable parameters; matched V1 used 3,589.

| Metric | V2 | Matched V1 | Gate interpretation |
|---|---:|---:|---|
| protocol macro unordered top-2 pair | 0.601042 | 0.631250 | V2 fails 0.90 |
| protocol macro top-1 | 0.846875 | 0.861458 | diagnostic |
| mixed AUPRC | 0.962465 | 0.958126 | diagnostic |
| mixed AUROC | 0.874722 | 0.891389 | diagnostic |
| mixed Brier | 0.143964 | 0.492819 | diagnostic |
| pure false-mixed | 0.050000 | 0.100000 | V2 passes |
| pure SINGLE | 1.000000 | 0.900000 | V2 safe |
| compatible DUAL | 0.189378 | 0.523007 | V2 fails 0.80 |
| incompatible HARD | 0.313889 | 0.000000 | V2 fails 0.80 |
| incompatible DUAL | 0.000000 | 0.501533 | V2 passes 0.10 |
| wrong-pair DUAL | 0.072824 | 0.131513 | V2 passes 0.10 |
| correct-pair weight MAE | 0.158058 | 0.203023 | V2 fails 0.12 |
| correct-pair weight RMSE | 0.198143 | 0.236070 | diagnostic |

V2 therefore improves safety routing over matched V1: it eliminates
correct-incompatible DUAL exposure and lowers wrong-pair DUAL. The cost is
over-conservative fallback: mixed false-SINGLE is 0.830556 and compatible
DUAL is only 0.189378. This behavior is not a successful soft controller.

On the 240 fixed visual representatives, V2 used SINGLE/HARD on 208 sheets
and DUAL on 32; matched V1 used SINGLE on 135 and DUAL on 105. Grade-3 core
artifact sheets fell from 100 to 27 (73%), but V2 still showed severe
patch/cloud/mottle/full-body contamination whenever the exposed DUAL route
used a visually unstable combination. Both families retained moderate edge
scatter and silhouette discontinuity. Identity contamination, double outline,
and ghosting had maximum grade 0.

Primary-test render means were not uniformly better for V2. V2 versus matched
V1 respectively measured garment RGB MAE 0.080504 versus 0.066015, garment
LPIPS 0.052099 versus 0.039666, silhouette IoU 0.706995 versus 0.711309, and
boundary F-score 0.339466 versus 0.344219. These results retain the visual
cost of the controller's routing and endpoint decisions.

The comparison does not support promotion to formal evaluation.
`PAPER_FINAL=0`.
