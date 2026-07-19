# CanonDressGS alpha coverage and Gaussian rasterization audit — 2026-07-19

## Decision

- Task: `SUBJECT02-ALPHA-RASTER-AUDIT-001`
- Source V6.1 result: `FAIL / Case SC`
- Source tag: `v6-1-silhouette-case-sc-20260719`
- Audit branch: `research/alpha-raster-audit-20260719`
- Formal static run commit: `51ff263c87a8a9963313f4b44750f0b77b13384d`
- Formal accepted output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-ALPHA-RASTER-AUDIT-001/attempt_004`
- Optimizer steps: `0`
- Final status: **PASS (audit complete)**
- Final case: **RF — no unique attribution with the installed public backend interface**
- Next unique task: `BUILD_INSTRUMENTED_RASTERIZER_DEBUG_PATH`

Case SC means the V6.1 trusted silhouette semantics corrected part of the synthetic body/pose drift in the raw metric, but did not close the real visible underfill, internal holes, scalloped sleeve boundary, trailing cloud, or floating-splat evidence. It did not authorize further mask, loss, residual-bound, or renderer-default tuning.

Append-only attempt history is preserved. `attempt_001` failed at the clean-clone frozen-ref resolver, `attempt_002` failed at the Gaussian-state fingerprint adapter, and `attempt_003` failed at the CPU/CUDA device boundary of the standalone finite-difference probe. Each stopped at zero optimizer steps. The fixes were confined to audit tooling; `attempt_004` is the first complete formal result.

## Frozen inputs

Only O01 and O08 and the registered conditions `cond_000000` (front), `cond_000318` (back), `cond_000017` (left), and `cond_000347` (right) were read. P0 is the original 200k base. The frozen checkpoint states are:

| State | O01 SHA256 | O08 SHA256 |
|---|---|---|
| P1, representation Rung 2 | `af730d138697ab9c7a29f17603303ae41dfa36047b8cee59e2ee89328655ce56` | `4b0d113cf2e42904ec4f96833e5440fa4f6e6ab013e090be514ef3d8d3dd354e` |
| P2, V6 T5 | `0b9227df2fce94ea530be88fd2ac46438562e3e6338c2f4b9c3bd222ab225fcf` | `9d74a163a973c92f81cc4c827dc7508b9945632f87e70fe43e9ae1319a8ae6e1` |
| P3, V6.1 S1 | `46bf0dffbd6fc512e753cce7be7f895f7ca448568a100238fc1e02753b1ba111` | `7c0e5baf7b876b9ba40884f0db0ac8a70aeabdb24cff332fcad9b620ce8a5852` |

The base fingerprint is `65973ab809d5bacc70cf4ad56462039e4057f3a0079d2253adc0b7715693a48a` for both outfit loads. The production-path regression is bitwise exact for RGB and alpha (`max_abs_diff=0`). Historical outputs were read-only.

## Production renderer contract

The production path is `GaussianModel.render` to `gsplat.rasterization`, using `gsplat 1.5.3+pt24cu121`, PyTorch `2.4.1+cu121`, CUDA 12.1, and an RTX 4090. The complete 30-item, source-line-backed contract is in `renderer_static_audit/RENDERER_STATIC_CONTRACT_AUDIT.md`.

Key rules are:

- quaternion order is `wxyz`; the residual composition is `q_base * q_delta`;
- scaling is activated with `exp` before canonical covariance construction;
- opacity is `sigmoid(logit)`;
- per-Gaussian pixel alpha is `min(0.999, opacity * exp(-sigma))`;
- front-to-back composition uses `T_next = T * (1 - alpha)` and final alpha `1 - T`;
- contributions with `alpha < 1/255` are skipped;
- the loop terminates exclusively when the next transmittance is `<= 1e-4`;
- `near_plane=0.1`, `far_plane=1e10`, `eps2d=0.3`, `radius_clip=0`, tile size 16;
- integer projected radii use `ceil`; there is no independent maximum radius;
- classic rendering, `packed=False`, `sparse_grad=False`, float32 production tensors;
- RGB and alpha use the same accepted contribution loop;
- the formal `alpha >= 0.5` threshold is evaluator-only and unchanged.

The public API does not expose safe switches for R1 early termination, R2 alpha cutoff, R3 a radius below the default zero clip, or R7 tile/culling bypass. They were recorded as unsupported rather than emulated by changing production code.

## P1/P2/P3 parameter and coverage comparison

For O08 trusted-expansion pixels, projected-center support collapses from P1 to P2/P3, while the surviving P2/P3 Gaussians do not have smaller center alpha or smaller radii:

| View/state | Gaussian centers in region | major radius p50/p95/max | minor radius p50/p95/max | effective center alpha p50/p95/max |
|---|---:|---:|---:|---:|
| back P1 | 2346 | 8/17/59 | 5/12/26 | 0.8130/0.9035/0.9669 |
| back P2 | 410 | 10/18.55/48 | 7/13/17 | 0.8594/0.9272/0.9532 |
| back P3 | 372 | 9/19/54 | 6/13/17 | 0.8552/0.9251/0.9554 |
| right P1 | 1075 | 8/26/119 | 5/12/31 | 0.8305/0.9207/0.9715 |
| right P2 | 198 | 8/25/62 | 6/13/18 | 0.8679/0.9320/0.9565 |
| right P3 | 195 | 9/25.3/60 | 7/12/18 | 0.8630/0.9330/0.9500 |

This refutes a generic “P3 opacity is too low” or “P3 projected radius collapsed” explanation. The narrowest supported distinction is sparse or misplaced projected support in the trusted expansion. The static public outputs cannot uniquely separate learned center placement, canonical support topology, and pre-tile backend rejection.

## Alpha-threshold sweep

All thresholds were applied only to persisted soft alpha. No renderer or formal threshold changed.

| O08 view/state | recall at 0.01 | recall at 0.50 | raw IoU at 0.01 | raw IoU at 0.50 |
|---|---:|---:|---:|---:|
| back P1 | 1.0000 | 1.0000 | 0.8077 | 0.8808 |
| back P2 | 1.0000 | 0.9061 | 0.8228 | 0.8746 |
| back P3 | 1.0000 | 0.9205 | 0.8393 | 0.8853 |
| right P1 | 1.0000 | 1.0000 | 0.7986 | 0.8643 |
| right P2 | 1.0000 | 0.8203 | 0.8080 | 0.8482 |
| right P3 | 1.0000 | 0.8393 | 0.8212 | 0.8610 |

Lower thresholds expose low accumulated alpha in the target expansion, but simultaneously lower raw IoU and retain cloud/leakage structure. Therefore threshold 0.5 is not the sole failure and lowering it is not a renderer fix.

## Contributor and culling evidence

Deterministic P3 FN samples have only 8.44 active contributors on O08 back and 4.32 on O08 right, versus 53.32/45.96 in correct-coverage pixels. The combined hole average is 6.38 contributors.

For P3 trusted FN, `no_nearby_gaussian_support` accounts for 478/495 (`96.57%`) back samples and 501/512 (`97.85%`) right samples. Radius, frustum, near/far, cutoff, and early-termination categories are zero under the exported candidate evidence. This classification is diagnostic: production export cannot reveal Gaussians removed before its tile list.

The trailing cloud is not created by the alpha metric threshold. Registered cloud pixels contain real projected candidates and contributions; the right trailing-cloud crop averages 664 tile candidates and 44 active contributors. Production/reference comparison retains the same cloud. R5/R6 reduce cloud area slightly only while reducing trusted recall, so this is not an acceptable cleanup.

## Float64 reference compositor

The crop-only reference compositor uses production-exported projected means, conics, depths, opacities, and tile lists, but composes in float64 without the `1/255` cutoff or `1e-4` early termination.

| Crop | mean abs diff | p95 diff | trusted FN recovered at alpha >= 0.5 |
|---|---:|---:|---:|
| O01 normal boundary | 0.000260 | 0.000607 | 0.0% |
| O08 back max hole | 0.001210 | 0.007625 | 3.3% |
| O08 right max hole | 0.000541 | 0.003678 | 0.0% |
| O08 back sleeve scallop | 0.000980 | 0.005465 | 1.1% |
| O08 right trailing cloud | 0.002114 | 0.014362 | 0.0% |

No registered disagreement threshold is met. The production and reference crops are visually near-identical. Cutoff and early termination are not the dominant explanation for the measured underfill; the reference path also cannot recover pre-export omissions.

## Gradient checks

The scoped one-Gaussian pixel-alpha check produced 224 rows across two epsilon scales. All values are finite, gradient signs agree, and the maximum relative autograd-versus-centered-finite-difference error is `9.73539e-4`. Discrete tile/radius selection remains explicitly non-differentiable and was not mislabeled as a smooth parameter Jacobian.

## Static render variants

R0 production, R4 forced float32, R5 antialiased mode, and R6 deterministic 2x supersampling ran. R4 is identical. Relative to R0, R5 changes O08 back/right expansion recall by `-0.01831/-0.02176`; R6 changes it by `-0.01317/-0.01558`. No variant achieves positive recall gain or a qualifying hole reduction; all fail `VARIANT_COVERAGE_IMPROVEMENT`. No ninth variant was introduced.

## Visual acceptance

The original-resolution P1/P2/P3 O08 back/right alpha images, threshold plot, all five reference-compositor crops, and R0/R4/R5/R6 contact sheets were actually opened with the image viewer. P2/P3 show insufficient trusted-expansion coverage; P1 preserves recall but has ragged edge splats and cloud structure. The no-cutoff reference does not fill the holes or remove the cloud. Antialiasing/supersampling changes edge appearance but does not restore coverage. O01 does not show the same catastrophic failure.

The P3 threshold-0.5 background leakage is at most `0.002686`, below the frozen `0.03` limit. The frozen V6.1 protected checks remain valid, the base is bitwise unchanged, and this zero-step audit introduces no protected-region or checkpoint mutation.

## Final adjudication and permissions

Final case is **RF**. There is no defensible unique renderer setting or alpha parameterization change: the evidence localizes the bottleneck to insufficient accumulated alpha from sparse/misplaced projected support, but the installed public gsplat API cannot uniquely distinguish the remaining support-placement and pre-export culling mechanisms.

- Proof Probe: **NOT RUN**; there is no unique static candidate.
- Modify production renderer: **not authorized**.
- Modify alpha parameterization: **not authorized**.
- Re-run seven outfits: **not authorized**.
- Generate more targets: **not authorized**.
- Start image-conditioned training: **not authorized**.
- Modify V6/V6.1 mask, loss, bounds, target-progress, residual composition, R2, base, checkpoints, or frozen outputs: **not authorized**.

For paper writing, state that trusted-region semantics reduce synthetic full-body silhouette confounding, but alpha-consistent garment coverage remains limited for O08 back/right. Do not claim that the production rasterizer is defective, that antialiasing fixes coverage, or that raw synthetic silhouette is a direct garment-quality measure. Report the reference-compositor agreement and the backend instrumentation limitation.

The only next task is `BUILD_INSTRUMENTED_RASTERIZER_DEBUG_PATH`: expose pre-tile projected candidates, explicit rejection reasons, and safe toggles for cutoff/termination/tile admission while retaining bitwise default-path regression. No optimization or broader experiment is authorized by this audit.
