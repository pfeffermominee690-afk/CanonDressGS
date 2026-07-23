# Compatibility-Gated Controller V2 Design

**RESEARCH METHOD DESIGN — NOT PAPER FINAL**

- Task: `AAAI27-COMPATIBILITY-GATED-CONTROLLER-V2-DESIGN-001`
- Final design classification: `CONTROLLER_V2_DESIGN_READY`
- Source: `research/controller-calibration-compatibility-diagnostic-20260723` at
  `25318857b4cc4ee11011cd852c5cfc237a8b3cee`
- Formal V1 ancestor: `f45a518f330fb407942055756373e83f65717853`
- Design branch: `research/compatibility-gated-controller-v2-design-20260723`
- Scope: closed seen-garment wardrobe only.

## Inherited causal evidence

The inherited formal diagnosis is `MULTIPLE_FACTORS`. The primary residual
source is `DUAL_SUPPORT_COMPATIBILITY_DOMINANT`; secondary sources are weight
calibration, fallback calibration, and reference robustness. Pair selection is
high-accuracy and is not the primary failure source.

V1 evidence is preserved without alteration: top-2 pair accuracy 0.962500,
ordering accuracy 0.897222, mixed Dual-Support activation 0.508333, weight MAE
0.187604, weight RMSE 0.212702, pure top-1 20/20 per seed, pure
`SINGLE_ENDPOINT` 100%, identity contamination 0, target-forward leakage 0,
and GT pair/weight inference use 0.

The causal gains remain: pair-fix mean LPIPS 0.000350 (31.81% improved),
weight-fix 0.021878 (99.71%), fallback-removal 0.018765 (55.97%), and
full-oracle 0.041626. `SIMPLE_GATE_NOT_SEPARABLE` is retained: the illustrative
0.50 top-2-mass / 0.05 secondary-weight gate reaches pure SINGLE 1.0 and mixed
DUAL 0.819444 but wrong-pair DUAL 0.666667.

The design does not erase the archived scientific failures. O01_O03 and
O02_O03 both have pair accuracy 1.0 yet retain exact-pair/exact-weight grade-3
contamination and archived Oracle severe count 8. These failures motivate the
compatibility gate; they are not treated as classifier errors, insufficient
training, or tuned away.

## V2 factorization

Frozen F2 spatial maps are reduced by masked mean/max into 256-dimensional
reference rows. Deterministic validity-aware set mean/max aggregation produces
a 512-dimensional vector, followed by LayerNorm and L2 normalization. Three
independent linear heads then emit:

- five garment logits for top-1, unordered top-2 pair, and dominant ordering;
- one mixedness logit for pure versus mixed;
- ten pair-conditioned weight logits in frozen unordered-pair order.

The random-init adapter has 9,232 parameters: 1,024 normalization parameters,
2,565 garment-head parameters, 513 mixedness-head parameters, and 5,130
pair-weight-head parameters. Every forward computes all ten weights. Inference
selects a scalar only with the predicted pair; the V1 probability ratio is
forbidden as the final mixture weight.

## Compatibility prior

The prior is a garment-bank property for a closed wardrobe. It uses only
canonical/intrinsic proxies and frozen Oracle records from the rotation's
calibration fold. Test-fold data, current-query GT, target pose/camera/RGB/mask,
current-query renders, and pair blacklists are excluded. Missing canonical bbox
ratios remain explicit nulls with `UNAVAILABLE...NOT_FABRICATED` provenance.

The fixed uniform rule requires maximum core grade <=2, zero severe
patch/cloud/mottle/full-body records, identity grade 0, zero grade-3 ghosting,
and endpoint parity PASS.

| Rotation | Compatible | Incompatible | Rule-generated incompatible pairs |
|---:|---:|---:|---|
| 0 | 8 | 2 | O01_O03, O02_O03 |
| 1 | 8 | 2 | O01_O03, O02_O03 |
| 2 | 8 | 2 | O01_O03, O02_O03 |
| 3 | 8 | 2 | O01_O03, O02_O03 |

All four rotations retain their calibration-fold records and SHA-256 values
even though the observed labels are the same. Explicit pair blacklist count is
0. The repository archive contains 40 pair
records.

## Cross-fit and routing

The frozen scientific name is **CONDITION-FOLD CROSS-FIT EVALUATION**, not
novel-view evaluation. Each of four rotations uses two train folds, one
calibration fold, and one held-out test fold. Thresholds and compatibility are
calibration-only; all held-out folds must be aggregated without best-rotation,
best-seed, or best-checkpoint selection.

Pair confidence is uniquely defined as the top-2 versus top-3 probability
margin. Calibration grids are mixedness 0.10–0.90 in steps of 0.10 and pair
confidence {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}.

The three reachable runtime modes are `SINGLE_ENDPOINT`, `DUAL_SUPPORT`, and
`HARD_GEOMETRY_SOFT_VA`. The safe mixed fallback uses one dominant predicted
geometry support and two visibility/appearance sources; it never interpolates
geometry and never silently substitutes a pure top-1 image.

## No-training dry-run

The smoke exercised five pure, three AAB, three ABB, and seven boundary cases
(compatible, incompatible, low mixedness, low pair confidence, dropout,
single-reference, and exact tie). All three modes were reached. Seed 0 was
identical across two fresh processes, while seeds 0/1/2 had distinct parameter
hashes.

No formal image or metric was produced. Counts are: training
0, training-forward batches
0, backward 0, optimizer
creation 0, optimizer step
0, scheduler 0, checkpoint
load/write 0/0, renderer
0, formal renders/metrics/reviews
0/0/0,
and PAPER_FINAL 0.

## Scientific boundary

This design authorizes no empirical V2 quality claim. A future passing pilot
may claim only compatibility-aware routing within a closed seen-garment
wardrobe. It may not claim unseen-pair compatibility, unseen garment
generation, arbitrary synthesis, cross-identity generalization, novel-view
success, or novel-pose success.
