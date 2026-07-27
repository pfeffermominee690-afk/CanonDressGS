# Reference-Conditioned Dual-Support Controller: Failure Analysis

Task: `AAAI27-REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-FORMAL-002`

Classification: `REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER_PARTIAL`

## Primary failure: fallback calibration

Protocol-weighted DUAL_SUPPORT activation is only 0.508333; seed rates are 0.504167, 0.491667, and 0.529167. The preregistered macro minimum is 0.80 and per-seed floor is 0.70. Consequently, many AAB/ABB sheets collapse to SINGLE_ENDPOINT. These outputs can look cleaner than FULL_LINEAR or Oracle blends, but they do not implement the requested composition and are graded as endpoint mismatch/wrong-garment mixture.

## Residual active-DUAL artifacts

When DUAL_SUPPORT activates, the review preserves patch, cloud, mottle, edge scatter, silhouette discontinuity, and full-surface contamination. O01_O03 and O02_O03 are the worst surface-contamination pairs; O03_O04, O03_O08, and O04_O08 retain moderate failures. No review record was removed because it was unfavorable.

Primary-sheet maximum grades: patch=3, cloud=3, mottle=3, edge scatter=2, silhouette discontinuity=2, full-body contamination=3, endpoint mismatch=3, and wrong-garment mixture=3. Manual identity-contamination maximum is 0, double-outline maximum is 0, and worst-seed grade-3 ghosting pair count is 0/10.

## Calibration and robustness

Protocol macro weight MAE=0.187604, RMSE=0.212702, Brier=0.063698, cross-entropy=0.677116, and ECE=0.114918. Assignment/view inconsistency remains. Perturbation assignment permutation is stable, but single-reference, blur, mask erosion/dilation, and reference dropout often change top-1/pair, fallback reason, mode, and rendered appearance.

## Exclusions and scientific boundary

There was no additional training, best-seed/checkpoint selection, temperature or threshold change, pair-specific exception, ground-truth pair/alpha correction, geometry interpolation, ghosting post-processing, or result-driven rerun. Target-forward leakage=0, ground-truth ID/pair/alpha inference use=0, frozen mutation=0, and PAPER_FINAL=0.

The next task is `DIAGNOSE_CONTROLLER_FALLBACK_CALIBRATION_AND_RESIDUAL_GHOSTING`. It was not started.
