# LOO Calibration Information Boundary Audit

## Authorized Boundary

Frozen F2 is authorized for train-only hard lookup, deployable nearest-known
initialization, K1/K2 adaptation preparation, and the initialization of internal
basis-garment-only calibration simulations. It is forbidden as the calibration loss,
for test evaluation, and for any test or held-out Teacher selection signal.

Calibration selects preregistered regularization settings once per split and method.
Its objective remains render-based common loss subject to zero identity contamination.
The F2 record initializes the simulation only; it is not a calibration metric. The
interfaces `F2ReferenceFeatureSet` and
`CALIBRATION_RENDER_OBSERVATION` are separate.

Counts: held-out Teacher reads `0`, test-target reads
`0`, test-metric reads `0`,
unauthorized F2 reads `0`. Calibration/test and
adaptation/calibration separation both PASS.
