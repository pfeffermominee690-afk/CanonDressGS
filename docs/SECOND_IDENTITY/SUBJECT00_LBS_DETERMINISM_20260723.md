# Subject00 LBS Determinism — 2026-07-23

## Final result

`FAIL` → `SUBJECT00_BLOCKED_LBS_NONDETERMINISTIC`.

Each run independently copied subject00 SMPL parameters and the sealed PointInterpolant binary, verified hashes, computed the 55 SMPL-X weight/gradient fields, and executed 55 solver calls at depth 7, gradient weight 0.05 and 12 threads. No run read or copied the other run's grid.

Both individual grids pass their basic numerical contract:

- shape `[128,128,128,55]`, dtype float32;
- finite, NaN=0, Inf=0, zero-sum voxels=0, unknown joints=0;
- minimum=0; maximum=`0.9984235168`;
- run_a weight-sum max error=`4.9674782e-8`;
- run_b weight-sum max error=`4.9493337e-8`;
- identical bbox/metadata;
- 55 solver grids per run;
- wall clock approximately 118 s / 120 s.

## Repeatability failure

| Metric | Observed | Frozen threshold | Result |
|---|---:|---:|---|
| Grid max abs | `2.2351369e-4` | `≤1e-6` | FAIL |
| Grid mean abs | `1.1395203e-6` | `≤1e-8` | FAIL |
| Argmax-joint agreement | `0.9999976158` | `1.0` | FAIL |
| Metadata exact | true | true | PASS |

run_a grid-array SHA is `d35fd769985f48ca668786ef1ca59680fa34d6a75b8644f01baea7911d09b0f6`; run_b is `56fadfe501d8c9f3fcb5ac4a17f8cf003742e7cd0eb04afe3dc14386a7579d82`. NPZ file hashes also differ. The task pre-registered a numerical fallback for non-bitwise solver output, but the observed differences exceed every repeatability allowance. No run was selected as “better,” and no threads/bounds/resolution/padding were changed.

## Geometry failure

Template coverage itself passes: extrapolated vertices=0, grid coordinates finite, interpolated sums have maximum error `1.8021565e-7`, and direct SMPL-X weight MAE is `0.0056667261` (within 0.02).

Two frozen geometry criteria fail:

| Metric | Observed | Frozen threshold | Result |
|---|---:|---:|---|
| Interpolated weight max abs | `0.7210046649` | `≤0.5` | FAIL |
| Dominant-joint agreement | `0.9010978520` | `≥0.95` | FAIL |

SMPL-X forward checks for frames 0, 1250 and 2499 are finite, have zero degenerate faces, finite normals, bounded coordinates and near-unit surface-area ratios. Those successes do not override the failed grid repeatability/geometry gates.

Consequences: no deterministic canonical NPZ was serialized, no atomic publication occurred, no canary-ready config was created, and no runtime smoke was allowed. Full run_a/run_b assets and logs remain in the external attempt for the next diagnosis task.
