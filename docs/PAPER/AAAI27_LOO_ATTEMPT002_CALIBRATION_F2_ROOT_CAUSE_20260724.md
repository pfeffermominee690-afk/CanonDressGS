# LOO Attempt 002 Calibration F2 Root Cause

## Classification

`LOO_CALIBRATION_F2_INTERFACE_REPAIR_READY`

## Exact Failure

Attempt 002 raised `ValueError: frozen F2 control supports K in {1,2,3}` in
`REGULARIZATION_CALIBRATION_BEFORE_OPTIMIZER`. The formal caller was `tools/paper/run_loo_basis_adaptation_experiment.py:2178` and the
frozen guard was `scene/frozen_f2_linear_coefficient_control.py:62`. The split-scoped failure occurred before
the formal task loop; deterministic ownership is `LOO-O01-R0-K1`
for held-out O01, R0, K1. The first internal simulation held out O02 and first requested
an O03 F2 feature.

## Root Cause

The caller assembled `cond_000017, cond_000347, cond_000000, cond_000318` as one K=4
adaptation reference set. Those values are four independent
`CALIBRATION_RENDER_OBSERVATION` units. Frozen F2 accepts K in {1,2,3}, so it returned
no payload and the backbone was never called. Root-cause classes are
`C. SINGLE_VIEW_MULTI_VIEW_AGGREGATION_MISMATCH`, `D. CALIBRATION_CONSUMER_EXPECTS_WRONG_INTERFACE`. Preflight validated 40 K1/K2 task
descriptors but omitted the shared four-observation calibration-initialization descriptor.

All 40 tasks depended on split calibration and would therefore have been blocked. K1/K2
adaptation F2 and test evaluation were not incompatible.
