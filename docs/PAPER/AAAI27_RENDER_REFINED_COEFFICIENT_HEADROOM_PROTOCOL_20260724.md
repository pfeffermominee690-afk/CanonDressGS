# Render-Refined Coefficient Headroom Protocol

Task: `AAAI27-COEFFICIENT-HEADROOM-PROTOCOL-001`

Source: `research/pure-endpoint-execution-contract-repair-20260724` at `ff56ebfaf7b733adbb41799e248d01d0e7801ea8`

Classification: `COEFFICIENT_HEADROOM_PROTOCOL_READY`

## Scientific question

For fixed subject02 and the closed wardrobe O01/O02/O03/O04/O08, does
render-objective optimization of four coefficients inside the frozen
Teacher-derived affine residual span improve held-out condition renders beyond
both the SVD Endpoint and the Teacher Endpoint? The experiment also measures
what fraction of safe Full-Residual improvement is recovered inside the span.

The four names are frozen: **Teacher Endpoint**, **SVD Endpoint**,
**Render-Refined Coefficient**, and **Full-Residual Render Optimization**.
Teacher is an endpoint and initialization, not a capability bound.

## Frozen assets and variables

The subject02 base, five Teacher checkpoints, rank-4 basis, basis mean,
coefficient normalization, MMLP-Human, renderer, camera, pose, RGB/masks,
protected regions, white background, and render settings are immutable. The
coefficient method has one four-dimensional degree of freedom. Its
implementation stores standardized `z=(c-mean)/std`, initialized exactly from
the frozen SVD coefficient; this is a one-to-one numerical parameterization of
`c`, not an extra model. No random or cross-garment initialization is allowed.

Full-Residual uses the historical Rung-2 direct canonical field. Only
`raw_xyz`, `raw_log_scaling`, `raw_rotvec`, `raw_opacity`, and `raw_sh0` are
trainable; `raw_shN` remains the frozen zero schema buffer. Base avatar,
deformation, renderer, F2, and reference predictor remain frozen.

## Rotations and isolation

| Rotation | Optimize | Calibration | Test |
|---|---|---|---|
| R0 | 000000, 000318 | 000017 | 000347 |
| R1 | 000318, 000017 | 000347 | 000000 |
| R2 | 000017, 000347 | 000000 | 000318 |
| R3 | 000347, 000000 | 000318 | 000017 |

Each garment is optimized independently. Steps 1-300 alternate the two
optimize observations, so each receives 150 updates. The exact target assets,
camera/pose hashes, partition memberships, source-reference overlap, and query
order hashes are frozen in `coefficient_headroom_rotation_manifests.json`.

The target observations are disjoint across optimize/calibration/test within a
rotation. Closed-wardrobe reference images overlap across the inherited pure
endpoint query folds, but reference assets are not consumed by endpoint
optimization. A crucial limitation remains: each Teacher Endpoint was
historically optimized on all four conditions. Therefore the test fold is
isolated from refinement updates and selection, but initialization is not
target-naive. This is an endpoint-refinement diagnostic, not strict novel-view
generalization.

## Objective and regularization

The only historically supported rendering objective is
`CAPACITY_ORACLE_LOSS_V1`: garment RGB L1 (1.0), foreground alpha L1 (0.5),
new-silhouette alpha L1 (1.0), boundary RGB L1 (0.25), protected RGB L1 against
the Base Avatar (10.0), and protected alpha L1 against Base (5.0). LPIPS, SSIM,
and Dice are not Teacher training terms and stay at weight zero; LPIPS and SSIM
are evaluation metrics only.

The primary coefficient strategy is L2 anchoring in standardized coefficient
space with the preregistered positive grid `1e-4/1e-3/1e-2/1e-1`. One lambda
is selected per rotation from the step-300 five-garment macro calibration
render objective, shared by all five garments. Candidates within `1e-4` of the
minimum tie in favor of stronger regularization. Test metrics never select a
lambda. Lambda zero is run only as `UNREGULARIZED_DIAGNOSTIC` and cannot replace
the primary result.

## Budget and final rule

Coefficient optimization uses Adam, lr 0.02, no weight decay, fixed LR,
gradient clip 5.0, and 300 steps. Full-Residual uses the historical two-rate
Adam contract, geometry lr 0.001, appearance lr 0.002, no weight decay, fixed
LR, clip 1.0, and the same 300 steps. Checkpoints are fixed at
0/20/50/100/150/200/250/300. Step 300 is the only primary endpoint. There is no
early stopping, best checkpoint, seed selection, garment-specific budget, or
result-driven extension.

The formal platform is the historical CUDA/RTX 4090 renderer path. Four-scalar
Adam arithmetic is CPU-feasible, but end-to-end CPU rendering is not the frozen
formal path.

## Evaluation and decisions

Every garment and rotation reports optimize, calibration, and held-out test
RGB MAE, PSNR, SSIM, VGG LPIPS, silhouette IoU, boundary F, protected LPIPS,
identity metric, residual distances, train-test gap, safety grades, parameters,
time, VRAM, and storage. Primary aggregation has 20 equally weighted
garment-rotation test cells.

Headroom gains and the span recovery ratio are frozen in the evaluator
contract. A non-positive full-residual denominator produces `null` with a
reason, never an invented ratio. Numerical success thresholds and the complete
classification decision order are in `coefficient_headroom_success_gates.json`.

`Outfit-ID Refined Lookup` is an evaluation-only lookup using ground-truth
garment ID. The future `Reference-to-Refined-Coefficient` predictor targets
`c_i^R`; `Refined Hard Lookup` remains the fair garment-classifier-to-lookup
baseline. Endpoint optimization gains may not be attributed to that predictor.

## Claim boundary and execution boundary

The strongest allowed positive statement is: "Teacher-derived basis supports
low-dimensional render-objective refinement within its affine residual span."
This does not establish new-garment adaptation, arbitrary garments, continuous
reference control, cross-identity transfer, or performance beyond full-residual
capability.

This freeze created zero optimizers, steps, forward/backward calls, checkpoints,
renderer runs, or renders. `PAPER_FINAL=false`. The authorized next task is
`RUN_RENDER_REFINED_COEFFICIENT_HEADROOM_EXPERIMENT`; it was not started.
