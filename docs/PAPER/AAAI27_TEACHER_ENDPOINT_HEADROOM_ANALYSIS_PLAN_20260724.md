# Teacher Endpoint Headroom Analysis Plan

Task: `AAAI27-COEFFICIENT-HEADROOM-PROTOCOL-001`

## Endpoint comparisons

For every garment and rotation, evaluate the frozen Teacher Endpoint and SVD
Endpoint on all optimize, calibration, and test conditions before reading any
refinement result. Verify garment order, normalization SHA, basis rank, and the
stored SVD reconstruction error. Teacher/SVD differences are reported rather
than silently treated as exact parity.

The selected Render-Refined Coefficient is the positive-anchor candidate chosen
from calibration at step 300. `UNREGULARIZED_DIAGNOSTIC` is reported in a
separate table. Full-Residual step 300 supplies the equal-step outside-span
reference, and the preregistered time-matched checkpoint supplies the
equal-wall-time reference.

## Signed gains

The three named gains retain the requested raw metric convention: method minus
baseline. Thus a negative LPIPS/MAE gain is favorable, while a positive
IoU/boundary-F gain is favorable. Also report a separate direction-normalized
improvement (positive is favorable) for thresholding. Report
coefficient-versus-SVD, coefficient-versus-Teacher, and
full-residual-versus-Teacher separately. Do not call an SVD recovery gain a
Teacher headroom gain.

For each error metric, compute

`(Teacher error - refined error) / (Teacher error - full-residual error)`.

If the denominator is non-positive, emit a null ratio and a reason. Report the
macro ratio only when its denominator is positive, plus the number of valid
per-garment ratios. Ratios above one are retained rather than clipped, because
they are diagnostic estimates rather than probabilities.

## Trajectory diagnosis

At 0/20/50/100/150/200/250/300, report render terms, coefficient displacement,
residual distance to Teacher, gradient norm, and safety metrics. The trajectory
can diagnose overshoot or local curvature, but only step 300 enters the primary
classification. No trajectory observation can add steps or select a checkpoint.

Compare optimize, calibration, and test changes to distinguish generalizing
refinement from optimize-only fitting. Because Teacher initialization was
created with all four conditions, use "held-out refinement condition" and not
"unseen view" in reports.

## Classification

Apply `coefficient_headroom_success_gates.json` mechanically. Useful headroom
requires safe LPIPS improvement for at least four garments, both macro LPIPS
magnitude thresholds, a companion RGB/boundary/silhouette gain, and material
span recovery. Stable but sub-threshold gains are small headroom. A safe strong
Full-Residual gain without a coefficient gain is capacity-limited. Remaining
safe or unsafe non-improvements are local optimum. Missing required evidence is
protocol incomplete and is checked before scientific labels.

No result is available in this protocol-freeze task. `PAPER_FINAL=false`.
