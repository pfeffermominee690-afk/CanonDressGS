# CanonDressGS Main Method Freeze (2026-07-25)

## Formal role

`EXPLICIT_TEACHER_DERIVED_ENDPOINT_COORDINATE_SYSTEM`

## Offline pipeline

Frozen Animatable Gaussian Avatar -> Per-Garment Multi-View Optimization -> Valid Canonical Teacher Endpoints -> Explicit Endpoint Coordinate System.

## Inference pipeline

Reference RGB / Mask -> Frozen F2 -> Mean/Max Set Aggregation -> Small Endpoint Controller -> Nearest Valid Endpoint Realization -> Frozen Deformation / LBS / Renderer.

## Coefficient semantics

`RAW_PREDICTED_COEFFICIENT` and `REALIZED_SNAPPED_ENDPOINT_COEFFICIENT` are distinct. The paper method realizes a registered endpoint after prediction; it does not claim that the raw continuous coefficient itself is an exact endpoint.

## Excluded from the current main pipeline

- Render-Refined Coefficient.
- LOO Few-View Adaptation.
- Full-Residual Optimization.
- Spatially Distributed Clothing Coefficients.
- Automatic Dual-Support Controller.

Dual-Support is frozen as `ORACLE_OR_USER_SPECIFIED_GEOMETRY_SAFE_EXTENSION`.

`PAPER_FINAL=0`.
