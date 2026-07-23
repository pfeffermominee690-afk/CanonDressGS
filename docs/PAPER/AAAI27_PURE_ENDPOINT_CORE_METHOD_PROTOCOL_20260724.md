# AAAI-27 Pure Endpoint Core Method Protocol

Task: AAAI27-PURE-ENDPOINT-CORE-METHOD-PROTOCOL-001
Scientific protocol: PURE-ENDPOINT CONDITION-FOLD CROSS-FIT
Paper method: CanonDressGS-Endpoint
Classification: PURE_ENDPOINT_CROSSFOLD_PROTOCOL_READY

## Scope

This protocol fixes subject02 and the closed wardrobe O01/O02/O03/O04/O08. A
query contains only pure-garment reference RGB and clothing masks. The method
selects a discrete canonical garment endpoint, then uses frozen MMLP-Human
deformation and rendering. AAB, ABB, pair labels, mixture alpha, compatibility
labels, Dual-Support, HARD_GEOMETRY_SOFT_VA, and compatibility routing are out
of scope.

This is condition-fold cross-fit. It is not unseen-garment, unseen-identity,
strict novel-view, or unseen-reference generalization. Exact reference assets
overlap because target-excluded sets reuse the same closed wardrobe.

## Frozen Source

- Branch: research/controller-v2-crossfit-micro-pilot-from-repaired-contract-20260723
- HEAD: a8557b461f301c19a0f24eb17d924ddd9159a580
- Source records: paper_protocol/reviewer_risk/dual_support_controller_training_manifest.json
- Source rotations: paper_protocol/reviewer_risk/controller_v2_micro_pilot_rotation_manifests.json
- Historical Controller failures remain immutable.

## Cross-Fit Records

| Rotation | Train folds | Calibration | Test | Records T/C/V | Unique T/C/V | RGB overlap T-C/T-V/C-V |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 0,1 | 2 | 3 | 40/20/20 | 10/5/5 | 15/15/10 |
| 1 | 1,2 | 3 | 0 | 40/20/20 | 10/5/5 | 15/15/10 |
| 2 | 2,3 | 0 | 1 | 40/20/20 | 10/5/5 | 15/15/10 |
| 3 | 3,0 | 1 | 2 | 40/20/20 | 10/5/5 | 15/15/10 |

Exact record IDs are in pure_endpoint_rotation_manifests.json. Each condition
has five unique pure queries, one per garment. Historical pair containers
repeat each query four times. Protocol-weighted and unique-query metrics are
both reported; training uses each logical query once. Record IDs and logical
hashes are disjoint. RGB/mask overlap is disclosed, not hidden.

The retained formal_pure 20-record set is a separate safety audit. It never
enters training, calibration, the primary denominator, or the global macro.

## Model Contract

The recovered contract is:

reference RGB/mask -> frozen F2 -> deterministic mean/max -> LayerNorm(512)
-> Linear(4) -> nearest frozen endpoint -> rank-4 basis -> frozen renderer.

- Input/output dimensions: 512 to 4.
- Parameters: 3076.
- Initialization: LayerNorm 1/0; Linear 0/0.
- Target: standardized rank-4 teacher endpoint coefficient.
- Loss: mean SmoothL1 beta 1, coefficient only.
- Optimizer plan: Adam, lr 0.02, no weight decay, fixed LR.
- Schedule: 300 steps, batch 5, fixed garment order.
- Data order: two train folds in registered order, round-robin.
- Checkpoints: 0/20/50/100/200/300; evaluate step 300 only.
- No early stopping, checkpoint selection, seed selection, or thresholds.

Linear Coefficient Predictor renders the continuous prediction.
CanonDressGS-Endpoint snaps to a frozen endpoint before rendering.

## Gates

- macro five-way top-1 >= 0.90;
- every rotation >= 0.80;
- every garment recall >= 0.75;
- endpoint parity PASS;
- identity contamination count = 0;
- severe wrong-outfit endpoint rate <= 0.05.

Single-reference behavior is fully reported. Complete reference dropout must
safely abstain and use Base Avatar; garment recovery is not required.

## Execution Boundary

This freeze performs no training, training forward batch, backward, optimizer
creation, checkpoint write, formal render, threshold selection, or upstream
asset mutation. PAPER_FINAL=false. The next task is TRAIN_AND_EVALUATE_PURE_ENDPOINT_CORE_METHOD_CROSSFIT and was not
started.
