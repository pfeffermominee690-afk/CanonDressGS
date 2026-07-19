# CanonDressGS instrumented projection/admission audit

## Decision

- Task: `SUBJECT02-INSTRUMENTED-PROJECTION-ADMISSION-001`
- Formal static-run commit: `3316995223f486a28527f20954e3c5979260860e`
- Formal output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-INSTRUMENTED-PROJECTION-ADMISSION-001/attempt_002`
- Status: **PASS (diagnostic audit complete)**
- Final case: **IP — placement failure**
- Unique next task: `REDESIGN_SCREEN_SPACE_COVERAGE_PLACEMENT_OBJECTIVE`
- Optimizer steps: **0**

The independent/backend projection contract agrees within every preregistered threshold. At the fixed O08 back/right trusted-FN pixels, P2 and P3 retain only 25.25% and 22.03% of P1 pre-tile theoretical support, while every surviving theoretical contributor reaches the matching backend tile. The dominant failure is therefore learned/canonical-to-posed screen-space support placement, not gsplat pre-tile admission.

## Why this audit followed Case RF

The frozen alpha/raster audit (`SUBJECT02-ALPHA-RASTER-AUDIT-001/attempt_004`, final `PASS / Case RF`) had already shown that:

- production and the float64 reference compositor agree;
- alpha cutoff and early termination are not the primary underfill cause;
- antialiasing and 2× supersampling do not restore coverage;
- autograd agrees with finite differences;
- surviving P2/P3 projected radii and center alpha are valid;
- projected-center support in the trusted expansion falls sharply from P1 to P2/P3.

The public high-level renderer did not then distinguish absent projected support from a hidden pre-tile rejection. This audit closes that distinction without changing training or rendering.

## Frozen inputs and attempts

The audit reused the exact P1/P2/P3 checkpoint paths and SHA256 values frozen by the alpha audit, with O08 `cond_000318` back and `cond_000347` right as focus views and O08 front plus O01 back/right as controls. Camera, pose, covariance, opacity and base state came from the same pipeline and fixed data contract.

- `attempt_001`: preserved tool failure before projection; a dict was passed where the fingerprint helper required `(name, tensor)` pairs. Optimizer steps: 0.
- `attempt_002`: valid formal audit. Static run completed in 104.05 seconds on an RTX 4090 with Python 3.10.20, PyTorch 2.4.1+cu121 and CUDA 12.1.
- Parquet postprocess attempt 1: read-only tool failure because `pyarrow` required a list rather than tuple for `columns`.
- Parquet postprocess attempt 2: PASS over 3,000,000 Gaussian-state-view records.
- Visual finalization attempt 1: shell JSON quoting failure, with no acceptance output written.
- Visual finalization attempt 2: PASS using base64 JSON transport.

No failed attempt ran a renderer optimization or modified a checkpoint, renderer, loss, bound, source output or frozen ref.

## Debug implementation

`scene/instrumented_projection_admission.py` implements two isolated paths:

1. `INDEPENDENT_GAUSSIAN_PROJECTION_ORACLE_V1`: target-independent PyTorch float64 camera transform, pinhole Jacobian, full covariance projection, `eps2d`, opacity-aware radius, image overlap and tile-bbox calculation.
2. An opt-in repo-local wrapper over the installed gsplat 1.5.3 `fully_fused_projection` and `isect_tiles` low-level operations.

Instrumentation defaults to off. When enabled, it runs only after the production render and receives detached read-only tensors. It does not replace the production import path, alter ordering, change floating-point operations in the production render, or participate in training. Installed `site-packages/gsplat` was not edited.

Each Gaussian retains its frozen base `gaussian_index`. The Parquet artifact records S0–S6 state, camera coordinates, projected mean/covariance/conic/radius, image and tile bounds, emitted intersections, first rejection stage and reason. Pixel JSONL records S7 depth-sorted, S8 alpha-eligible and S9 contributed index sets for every fixed trusted-FN pixel.

## Production regression

All 15 registered state/outfit/view combinations passed:

- debug off vs production RGB max absolute difference: `0`
- debug off vs production alpha max absolute difference: `0`
- debug on vs production RGB max absolute difference: `0`
- debug on vs production alpha max absolute difference: `0`
- RGB bitwise equality: PASS
- alpha bitwise equality: PASS
- binary alpha equality: PASS

The frozen base fingerprint was unchanged before and after both outfit audits:

`65973ab809d5bacc70cf4ad56462039e4057f3a0079d2253adc0b7715693a48a`

## Independent projection validation

Worst values across all 15 comparisons were:

| Quantity | Worst observed | Preregistered threshold | Result |
| --- | ---: | ---: | --- |
| projected mean median absolute difference | `1.3715933e-05 px` | `<= 0.01 px` | PASS |
| projected mean p99 absolute difference | `8.7739844e-05 px` | `<= 0.10 px` | PASS |
| depth relative p99 | `5.1571665e-08` | `<= 1e-5` | PASS |
| radius p99 absolute difference | `0 px` | `<= 1 px` | PASS |
| radius p99 relative difference | `0` | `<= 2%` | PASS |
| projected covariance p99 absolute difference | `7.5660890e-06` | diagnostic | recorded |
| conic p99 absolute difference | `6.3514866e-07` | diagnostic | recorded |

`PROJECTION_CONTRACT_DISAGREEMENT` is therefore false; Case IC is rejected.

## P1/P2/P3 support funnel

The following numbers use the same fixed P3 trusted-FN pixels for all states. Each value is the mean contributor count per pixel.

| State | View | FN pixels | N_pre | N_emitted | N_active | admission retention | pixel retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P1 | back | 495 | 30.997980 | 30.997980 | 30.698990 | 1.000000 | 0.993628 |
| P1 | right | 598 | 30.275920 | 30.275920 | 30.153846 | 1.000000 | 0.998000 |
| P2 | back | 495 | 7.216162 | 7.216162 | 7.216162 | 1.000000 | 1.000000 |
| P2 | right | 598 | 8.254181 | 8.254181 | 8.254181 | 1.000000 | 1.000000 |
| P3 | back | 495 | 6.836364 | 6.836364 | 6.836364 | 1.000000 | 1.000000 |
| P3 | right | 598 | 6.663880 | 6.663880 | 6.663880 | 1.000000 | 1.000000 |

Using the preregistered mean-of-view aggregation:

- P1 N_pre/N_emitted/N_active: `30.636950 / 30.636950 / 30.426418`
- P2 N_pre/N_emitted/N_active: `7.735171 / 7.735171 / 7.735171`
- P3 N_pre/N_emitted/N_active: `6.750122 / 6.750122 / 6.750122`
- P2 placement ratio vs P1: `0.2524785`
- P3 placement ratio vs P1: `0.2203262`
- P1/P2/P3 admission retention: `1.0 / 1.0 / 1.0`
- P1/P2/P3 pixel retention: `0.9958142 / 1.0 / 1.0`

Both P2 and P3 satisfy the Case-IP placement threshold (`<= 0.35`) and admission threshold (`>= 0.90`).

## Rejection reasons and backend admission

Across 3,000,000 Gaussian-state-view records:

- emitted normally: `2,999,985`
- S3 opacity below `1/255`: `15` records
- backend unexposed rejection: `0`
- intersection allocation omission: `0`
- nonfinite, near/far, covariance/determinant, radius, image-overlap, tile-overlap and backend-internal rejections: `0`

The 15 opacity-below-cutoff records are distributed across registered controls and states and do not appear among the missing trusted-FN theoretical contributors. No single backend rejection is dominant. Case IA is rejected.

## Same-index displacement

Matching is direct by base Gaussian index; no nearest-neighbour remapping is used.

| Target state | View | matched P1 supporters | mean screen displacement | moved out of expansion | moved to trailing cloud | mean opacity change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2 | back | 2,112 | 18.1417 px | 50.66% | 0% | -0.01811 |
| P3 | back | 2,112 | 18.1805 px | 50.90% | 0% | -0.01846 |
| P2 | right | 1,106 | 23.9430 px | 60.67% | 3.98% | -0.04379 |
| P3 | right | 1,106 | 24.6297 px | 61.03% | 0% | -0.04725 |

The displacement maps show coherent flows on sleeves, torso-side support and lower-body groups. P2 and P3 are visually similar. A small P2-right subset reaches the trailing-cloud region, but the missing support is not predominantly transferred there; P3 shows zero matched-support transfer into that region. Opacity decreases are modest relative to the large screen-space displacement and cannot explain the support collapse alone.

## Pre-tile counterfactual

There are no Gaussians that the independent oracle identifies as contributors at the fixed trusted-FN pixels but the backend omits from the matching tile:

- rejected Gaussian count: `0`
- trusted FN recovered: `0 / 495` back and `0 / 598` right (`0%`)
- new background FP: `0 / 512` sampled pixels in each view
- counterfactual cloud: false

The counterfactual cannot recover underfill because there is no hidden pre-tile rejection to restore. This is positive evidence against backend admission as the primary cause, not evidence that the missing placement is harmless.

## Synthetic unit scenes

All ten deterministic scenes passed: image center, image boundary, tile boundary, extremely thin minor radius, major/minor anisotropy, near-plane vicinity, multi-Gaussian alpha accumulation, equal-depth ordering, large tile bbox, and a deterministic 256-Gaussian O08 reduction. Independent projection agreed with the backend, and production alpha agreed with the float64 compositor at the tested pixels. No target RGB or training data was used to optimize these scenes.

## Visual inspection

The following images were actually opened at original resolution and as contact sheets:

- `projection_density_contact_sheet.png`
- `placement_flow_maps.png`
- `final_visual_contact_sheet.png`

Observed evidence:

- P1 has visibly broader/denser O08 back and right projected-center support than P2/P3.
- P2/P3 are spatially continuous and mutually similar; no tile-shaped or random dropout pattern is visible.
- Same-index flow is coherent and large, consistent with the measured 18–25 px displacement.
- O01 controls remain continuous and body-aligned.
- No dominant trailing-cloud transfer or backend-admission pattern is visible.

Visual acceptance status: **PASS** for the diagnostic claim.

## Test matrix

- new projection/admission contract: 21/21 PASS (including real CUDA backend)
- prior alpha/raster audit: 20/20 PASS
- V6.1 silhouette semantics: 24/24 PASS
- V6 objective/residual: 27/27 PASS
- V5.3 dual-target supervision: 28/28 PASS
- V5.2 fixed episode: 3/3 PASS
- R2 rotation autograd: 12/12 PASS, CUDA finite path tested
- differentiable renderer unit checks: PASS
- full training checkpoint checks: PASS
- image-conditioned dataset checks: 20/20 PASS
- `py_compile`: PASS
- `git diff --check`: PASS

## Proven, not proven, and permissions

Proven:

- the repo-local instrumentation does not change production RGB/alpha;
- independent and backend mean/depth/radius/covariance contracts agree;
- P2/P3 theoretical support at fixed trusted-FN pixels collapses before backend admission;
- all surviving theoretical contributors are emitted and active at those pixels;
- same-index support undergoes large coherent screen-space displacement;
- backend rejection cannot recover the missing silhouette support.

Not proven by this zero-step task:

- which revised placement objective will generalize;
- whether old-garment removal or multi-view coverage should receive the dominant new term;
- whether a future placement fix will avoid all trailing-cloud tradeoffs;
- whether the eventual seven-outfit or image-conditioned pipeline will pass.

Permissions after adjudication:

- modify production renderer: **NO**
- change coverage-placement objective in the next separately authorized task: **YES**
- rerun seven outfits: **NO**
- generate more targets: **NO**
- start formal training: **NO**

The next and only task is `REDESIGN_SCREEN_SPACE_COVERAGE_PLACEMENT_OBJECTIVE`. This audit must not be used to justify another silhouette-threshold sweep.
