# AAAI27 Hard-Lookup Fair Comparison Protocol

## Primary Control

`Reference Nearest Hard Lookup` is the deployable control. For each LOO split,
its bank contains exactly the four basis garments. It maps the held-out
adaptation reference feature to the nearest train-only frozen-F2 garment
centroid and returns that known endpoint residual.

It cannot output the held-out endpoint and cannot create a new coefficient.
A ground-truth outfit-ID oracle is not applicable because the held-out garment
does not exist in the bank.

The primary claim comparison is therefore:

`Few-View Low-Dimensional Adaptation` versus
`Reference Nearest Hard Lookup` on the exact held-out test view.

## Fairness Requirements

Both methods share target garment, rotation, K manifest, frozen MMLP-Human,
renderer, target pose/camera, metric implementation, and test denominator.
The hard lookup uses only the four-garment train centroid bank. The adaptation
starts from that same selected endpoint.

Full-residual optimization additionally shares adaptation views, initial rendered
state, rendering loss, 300-step budget, calibration/test folds, and Teacher
exclusion with low-dimensional adaptation. Equal-wall-time results are secondary.

## Oracle Diagnostics

`Residual Nearest Oracle`, `Oracle Projection into LOO Basis`, the optional
convex-combination oracle, and the `Held-Out Teacher Endpoint` are non-deployable.
They use the held-out Teacher only offline and appear in separate diagnostic
columns.

Oracle projection reports residual error, render metrics, span distance, and
projection ratio. It may diagnose basis capacity but may not initialize or
supervise deployable adaptation.

## Frozen Success Thresholds

At K=4, at least four of five garments must improve over hard lookup by at least
0.005 LPIPS and 0.002 RGB MAE in at least three of four rotations, with zero
identity contamination. At least four garments must avoid systematic K=4
regression relative to K=1 under the frozen macro tolerances.

Low-dimensional adaptation must use at most three trainable scalars, recover at
least 75 percent of full-residual gains for at least four garments, and satisfy
all safety gates. Oracle projection failure is classified as basis-capacity
failure, not optimizer failure.

These thresholds are frozen but cannot be evaluated while 45 required task
manifests are blocked.
