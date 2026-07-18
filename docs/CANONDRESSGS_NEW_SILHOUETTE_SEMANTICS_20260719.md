# CanonDressGS new-silhouette semantics closure — 2026-07-19

## Scope and provenance

- Task: `SUBJECT02-NEW-SILHOUETTE-SEMANTICS-001`
- Frozen predecessor: `cee8fc51b5039a102ef7e2c31632e348ae3b99a1`
- Historical objective run commit: `7cfaacd7dc15f063fc67b4aaa466440f27299479`
- Historical T5 output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-OBJECTIVE-RESIDUAL-REDESIGN-001/attempt_004`
- V6.1 formal run commit: `c01682775f294b534d2ff1b3ed0d05aaca4fc2c9`
- Formal output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-NEW-SILHOUETTE-SEMANTICS-001/attempt_002`
- Preserved zero-step tool failure: `attempt_001/S1_v6_1_semantics/O01`. The audit and one-time calibration were hash-copied to attempt 002 and were not repeated.
- Protocol: O01/O08 and `cond_000000/front`, `cond_000318/back`, `cond_000017/left`, `cond_000347/right` only.

S0 was citation-only. T5 was not rerun and historical Case F was not rewritten. No target was generated, no seven-outfit or image-conditioned job was started, and the T5 bounds, residual composition, target-progress RGB, protected semantics, renderer, R2, MMLP-Human base, and historical outputs were unchanged.

## Pixel audit and causal decision

T5 O01 passed. T5 O08 failed only raw silhouette IoU: back `0.874600`, right `0.848158`. Persisted alpha was thresholded with the unchanged `alpha >= 0.5`; PNG-derived IoU matched historical metrics within `2e-6`.

Every FN/FP pixel received exactly one label, in this priority: protected identity, artifact, trusted garment expansion/old removal, transition uncertainty, segmentation uncertainty, then body/pose drift. The frozen data contains raw and safe **binary** garment masks but no soft probability tensor; none was guessed or regenerated.

| O08 view | Raw FN | Raw FP | Trusted expansion | Transition | Protected | Body/pose drift | Artifact | Segmentation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| back | 4,674 | 28,497 | 585 (12.516%) | 1,197 (25.610%) | 281 (6.012%) | 2,611 (55.862%) | 0 | 0 |
| right | 4,858 | 35,206 | 669 (13.771%) | 795 (16.365%) | 777 (15.994%) | 2,617 (53.870%) | 0 | 0 |

| View | trusted FN fraction | drift FN fraction | Decision |
|---|---:|---:|---|
| back | 0.125160 | 0.618742 | `S-DRIFT` |
| right | 0.137711 | 0.698641 | `S-DRIFT` |

Most raw back/right FN therefore measures synthetic-target body/proportion/pose or protected-boundary drift, not reliable garment support. Raw full-foreground IoU remains mandatory as a diagnostic, but not as the sole garment-representation gate.

## V6.1 mask and loss contract

The objective is `SUPPORT_AWARE_REGION_TRUSTED_OBJECTIVE_V6_1`. Only silhouette-related regions changed.

The support radius is `ceil(diagonal(base_foreground_bbox) * (5 / hypot(1024,1536)))`, minimum one pixel. The ratio was frozen from the existing 1024×1536 resolution and 5-pixel V5/AAAI boundary contract before V6.1 outcomes. Its evidence is target edit core, safe target clothing, old clothing, and the adjacent base-foreground neighborhood.

- `M_expand_raw = target_fg ∧ ¬base_fg`
- `M_expand_trusted = M_expand_raw ∧ safe_clothing ∧ ¬protected ∧ ¬artifact ∧ support_band`
- `M_remove_trusted = base_fg ∧ ¬target_fg ∧ old_clothing ∧ ¬protected`
- `M_uncertain = (M_expand_raw ∪ M_remove_raw) \ (M_expand_trusted ∪ M_remove_trusted ∪ protected)`
- trusted background is V6 background excluding transition and artifact neighborhood.

No RGB/color rule, outfit-specific body part, O08-only range, hand polygon, teacher signal, target field in forward, or reference image was used.

V6 edit RGB, target progress, identity, neutral preserve, background, residual, stability, and cached transition formulas are unchanged. Underfill is `relu(target_alpha-pred_alpha)` on trusted expansion; removal is `relu(pred_alpha-target_alpha)` on trusted removal. Both use active-pixel-normalized SmoothL1. Empty masks return finite zero and uncertain pixels receive no hard target-alpha supervision.

## One-time gradient calibration

Exactly 20 optimizer steps ran, ten per outfit, with one shared weight set and no visual tuning.

| Group | Median gradient | Cap vs edit | Scale |
|---|---:|---:|---:|
| edit | 0.033320811 | — | 1.0 |
| silhouette | 0.177103880 | 0.5 | 0.094071374 |
| transition | 0.025556104 | 0.5 | 0.651914914 |
| regularization | 1.79113e-6 | 0.25 | 1.0 |

Frozen weights: edit/target-progress `1/1`; trusted underfill/removal `0.0470356872/0.0470356872`; transition `0.1629787286`; identity/background/neutral `2/1/0.5`; residual/stability `1e-4/1e-4`.

## S1: O01

| View | Active expand/remove | Raw IoU | Expand recall | Removal recall | Trusted IoU | Edit reduction | Target-closer | Protected | Background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| front | 11,448 / 12,740 | 0.957219 | 0.951607 | 0.905652 | 0.861186 | 0.903145 | 0.924323 | 0.005882 | 0.008664 |
| back | 12,771 / 22,298 | 0.913278 | 0.839480 | 0.808324 | 0.628982 | 0.894602 | 0.963614 | 0.006192 | 0.013876 |
| left | 6,202 / 935 | 0.966958 | 0.804257 | 0.996791 | 0.803868 | 0.834958 | 0.927107 | 0.004484 | 0.003728 |
| right | 6,539 / 14,204 | 0.919129 | 0.855635 | 0.654534 | 0.488817 | 0.908037 | 0.967756 | 0.003961 | 0.012510 |

Mean edit reduction/target-closer are `0.885186/0.945700`. Raw IoU, target-closer and all safety checks pass, but trusted IoU fails. Numeric status is `FAIL`; visual status `WARN`. O01 did not suffer a raw/safety regression: base was bitwise exact and base gradient count zero.

## S1: O08

| View | Active expand/remove | Raw IoU | Expand recall | Removal recall | Trusted IoU | Edit reduction | Target-closer | Protected | Background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| front | 8,572 / 19,607 | 0.959583 | 0.934671 | 0.859234 | 0.707024 | 0.877804 | 0.959367 | 0.005976 | 0.007613 |
| back | 6,227 / 28,863 | 0.885256 | 0.920507 | 0.734019 | 0.412255 | 0.845426 | 0.938918 | 0.006272 | 0.018634 |
| left | 5,308 / 13,167 | 0.955681 | 0.899774 | 0.706919 | 0.520999 | 0.885383 | 0.965985 | 0.004374 | 0.005741 |
| right | 3,722 / 30,595 | 0.860987 | 0.839334 | 0.447001 | 0.151349 | 0.900230 | 0.977513 | 0.004070 | 0.022678 |

Mean edit reduction/target-closer are `0.877211/0.960446`. Back/right expand recall improves from S0 `0.906054/0.820258` by `0.014453/0.019076`, but right remains below 0.92 and trusted IoU fails. Protected/background checks pass; abnormal Gaussian and maximum bound-hit fractions are both zero; base is bitwise exact with zero base gradients. Numeric and visual status are both `FAIL`.

## S2 eligibility

S2 did not run. Recall gain, remaining threshold failure, and safety conditions passed, but remaining-FN trusted fraction failed (`max=0.090923 < 0.70`). Coefficient update count and S2 optimizer steps are zero.

## Actual visual acceptance

The milestone/final/error sheets, O08 crops, and original-resolution O08 back/right target/prediction/alpha were opened with `view_image`.

- O01 `WARN`: garment semantics/color and raw contour remain, with recognizable protected areas; minor edge speckle and imperfect old-clothing removal remain.
- O08 `FAIL`: gray outfit semantics remain, but true back/right underfill remains. Back has scalloped sleeve opacity and internal alpha holes; right has a conspicuous translucent cloud behind shoulder, torso, arm and leg. These are visible floating-splat/raster failures even though mean background leakage passes.

## Tests and final adjudication

New V6.1 tests `24/24`, existing V6 `27/27`, dual-target/V5.3 `28/28`, fixed-episode `3/3`, R2 CUDA/autograd `12/12`, renderer unit, checkpoint, dataset, py_compile, and `git diff --check` all pass.

Final status: **FAIL — Case SC**. Mask semantics is not the sole issue; true garment underfill, trusted removal, and raster/alpha coverage remain unresolved.

- Seven-outfit re-adjudication: not allowed.
- More target generation: not allowed.
- Image-conditioned training: not allowed.
- Benchmark: garment-trusted silhouette becomes the primary clothing-region metric; raw full-foreground IoU remains diagnostic with an explicit synthetic-target drift limitation. Do not claim complete human-silhouette matching.
- Next and only task: `AUDIT_ALPHA_COVERAGE_AND_GAUSSIAN_RASTERIZATION`.
