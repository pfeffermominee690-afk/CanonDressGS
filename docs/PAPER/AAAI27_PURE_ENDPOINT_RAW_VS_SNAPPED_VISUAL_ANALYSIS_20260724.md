# AAAI27 Pure Endpoint Raw-vs-Snapped Visual Analysis

## Registered Semantics

- `RAW_PREDICTED_COEFFICIENT`: output of the shared coefficient predictor.
- `RAW_COEFFICIENT_ERROR`: error before endpoint realization.
- `REALIZED_SNAPPED_ENDPOINT_COEFFICIENT`: coefficient of the selected closed-bank endpoint.
- `REALIZED_ENDPOINT_ERROR`: error after nearest-endpoint realization.
- `ENDPOINT_EXACT_MATCH`: whether the realized discrete endpoint equals the target.

The generic label "coefficient error" is not used because it would merge two
different quantities.

## Sealed Result

For 60 registered clean queries, the Linear Coefficient Predictor has raw
coefficient MAE `71.44514598846436`, raw coefficient RMSE
`83.44418888092041`, and endpoint exact match `0.0`. CanonDressGS-Endpoint uses
the same predictor family but realizes the nearest closed-bank endpoint,
yielding realized endpoint MAE `0.0`, realized endpoint RMSE `0.0`, and endpoint
exact match `1.0`.

The result does not show exact continuous coefficient prediction. It shows that
closed-bank endpoint success is obtained through correct endpoint realization
despite large raw continuous coefficient error.

## Visual Selection

The contact sheet uses rotation 0, seed 0, all five garments, and both the
Linear Coefficient Predictor and CanonDressGS-Endpoint. Selection is by the
frozen manifest order, not by error magnitude or visual quality. The aggregate
plot and its PDF/SVG/PNG/source JSON are generated directly from sealed JSON.

## Claim Boundary

Safe wording: "The continuous predictor does not accurately reconstruct the
target coefficient; closed-bank success is obtained through correct
nearest-endpoint realization."

Forbidden interpretations include precise coefficient regression, continuous
control, arbitrary garment editing, unseen-garment generalization, and
hard-lookup superiority.
