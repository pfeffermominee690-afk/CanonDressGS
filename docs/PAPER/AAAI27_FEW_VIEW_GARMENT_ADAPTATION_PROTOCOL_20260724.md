# AAAI27 Few-View Garment Adaptation Protocol

## Frozen Inputs

The target garment may provide adaptation RGB/masks and adaptation pose/camera.
It may not provide its Teacher residual, endpoint label, test view, future
metric, or any full-five-garment basis information to the deployable forward or
loss.

The semantic selection rule is deterministic:

| K | Required views |
|---:|---|
| 1 | front |
| 2 | front, back |
| 4 | front, left, back, right |

Image quality and result-dependent selection are forbidden.

## Rotation Feasibility

| Rotation | Train | Calibration | Test | K=1 | K=2 | K=4 |
|---|---|---|---|---|---|---|
| R0 | front, back | left | right | valid | valid | blocked |
| R1 | back, left | right | front | blocked | blocked | blocked |
| R2 | left, right | front | back | blocked | blocked | blocked |
| R3 | right, front | back | left | valid | blocked | blocked |

Across five garments this yields 15 valid and 45 blocked planned tasks. Blocked
tasks remain in the manifest with exact conflicting IDs and may not execute.

## Primary Adaptation

For a valid task, the deployable variable is only `c_new in R^r`, where
`r<=3`. The primary initialization is the coefficient of the nearest known
garment. Nearest is determined by Euclidean distance from the held-out adaptation
reference feature to four train-only frozen-F2 garment centroids.

`ZERO_COEFFICIENT_INITIALIZATION` is a mandatory diagnostic. It cannot replace
the primary initialization after results are known.

Only adaptation-view rendering loss may update `c_new`. The held-out Teacher
residual is excluded from initialization, forward, and loss.

## Shared Rendering Loss

Both low-dimensional and full-residual adaptation use
`LOW_DIMENSIONAL_RENDER_ADAPTATION_LOSS_CONTRACT`:

| Component | Weight |
|---|---:|
| garment RGB L1 | 1.0 |
| VGG LPIPS v0.1 | 0.1 |
| mask L1 | 0.5 |
| boundary RGB L1 | 0.25 |
| protected identity RGB L1 | 10.0 |
| protected identity alpha L1 | 5.0 |

Regularization candidates and tie-breaking are frozen. Selection uses only the
four basis garments on the calibration fold through internal leave-one-basis-
garment-out simulation. The target garment and test fold contribute zero
information.

## Optimizer and Budget

Every garment x rotation x K task uses Adam, LR 0.02, betas 0.9/0.999,
epsilon 1e-8, no weight decay, 300 steps, gradient clipping at 5.0, constant
LambdaLR, seed 20260724, no retries, and checkpoints at
0/20/50/100/200/300. Step 300 is final; there is no best-checkpoint or early-
stopping selection.

Required telemetry includes trainable scalars, wall time, peak VRAM, convergence,
coefficient trajectory, and checkpoint size.

## Full-Residual Comparator

The primary full-residual comparator starts from the same nearest-known residual,
uses the same views, loss, poses/cameras, 300-step budget, and test denominator,
and receives no Teacher target. It optimizes 4,400,000 bound-normalized residual
scalars for 200,000 Gaussians, with protected gradients masked to zero.

An equal-wall-time diagnostic is reported separately and cannot replace the
equal-step primary comparison.

## Current Status

No adaptation is authorized until the view-fold manifest is repaired. Execution
counts are all zero.
