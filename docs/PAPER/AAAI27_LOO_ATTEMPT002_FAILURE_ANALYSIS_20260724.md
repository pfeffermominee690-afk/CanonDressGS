# AAAI27 LOO Attempt 002 Failure Analysis

## Classification

`LOO_ADAPTATION_EXECUTION_INVALID`

## Failure

The basis-garment-only regularization calibration assembled four frozen conditions and passed them to `_reference_feature` as one F2 batch. The frozen F2 runtime permits only `K in {1,2,3}` and raised before the first optimizer was created.

- Root cause: `CALIBRATION_F2_BATCH_CARDINALITY_EXCEEDS_FROZEN_RUNTIME`
- Calibration conditions: `cond_000017`, `cond_000347`, `cond_000000`, `cond_000318`
- Implementation callsite: `tools/paper/run_loo_basis_adaptation_experiment.py:2178`
- Runtime guard: `scene/frozen_f2_linear_coefficient_control.py:62`
- Optimizer creations/steps/checkpoints: `0/0/0`
- Formal metrics/visual sheets: `0/0`
- Attempt 001 mutation count: `0`
- Attempt 003 created: `false`
- Automatic rerun performed: `false`

## Next Task

`REPAIR_LOO_ATTEMPT002_EXECUTION_FAILURE`
