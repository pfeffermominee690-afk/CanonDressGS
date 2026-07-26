# Tri-Mode Garment Composition Contract

**RESEARCH METHOD DESIGN — NOT PAPER FINAL**

## Decision order

The controller receives only reference-derived frozen-F2 features and validity.
It predicts garment probabilities, mixedness, and all ten pair weights. Stable
ranking uses frozen outfit order for exact ties. The unordered top-2 prediction
indexes both the compatibility prior and the pair-specific weight; GT pair and
GT weight are absent from prediction forward.

Routing priority is:

1. fewer than two valid references → `SINGLE_ENDPOINT`;
2. mixedness below the calibration-frozen threshold → `SINGLE_ENDPOINT`;
3. top2-vs-top3 margin below threshold → `SINGLE_ENDPOINT`;
4. compatible predicted pair → `DUAL_SUPPORT`;
5. otherwise → `HARD_GEOMETRY_SOFT_VA`.

## SINGLE_ENDPOINT

This mode constructs one predicted-top-1 geometry support and one
visibility/appearance source. A second Gaussian branch is not constructed.
Pure references, dropout, single-reference, low mixedness, and low pair
confidence use this safe path.

## DUAL_SUPPORT

This mode constructs two immutable endpoint geometry supports for the predicted
pair. Opacity weights come from the selected predicted-pair weight head.
Geometry interpolation, geometry averaging, and basis-coefficient interpolation
are all false.

## HARD_GEOMETRY_SOFT_VA

This compatibility-safe mixed fallback constructs exactly one geometry support:
the predicted dominant garment endpoint. It still composes visibility and
appearance from both predicted endpoints using the pair-conditioned weight.
It therefore preserves a mixed output contract without interpolating geometry,
forcing Dual-Support, or silently returning a pure top-1 image.

The adapter has no pair-specific rendering branch. O01_O03 and O02_O03 reach
this mode only because their calibration manifests fail the same uniform rule
used for every pair.

## Information boundary

Target pose and camera are renderer inputs only after the controller decision.
They are not controller inputs. GT garment ID, pair, composition, alpha,
target RGB/mask, teacher residual, target render, and visual artifact label are
forbidden. The audited target-forward leakage count is 0.

## Interface-only evidence

The no-render smoke reached all three modes and verified geometry-support
counts of 1/2/1 and visibility/appearance source counts of 1/2/2. Endpoint
paths and the frozen pair orientation are retained in every runtime request.
No formal render, metric, or visual review was executed.
