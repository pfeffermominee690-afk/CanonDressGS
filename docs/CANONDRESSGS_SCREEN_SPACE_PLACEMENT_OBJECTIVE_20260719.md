# CanonDressGS Screen-Space Placement Objective — 2026-07-19

## 1. Final status

- Task: `SUBJECT02-SCREEN-SPACE-PLACEMENT-001`
- Formal run commit: `431b6349e1a526c90798d98cf57e88b33faea83d`
- Output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-SCREEN-SPACE-PLACEMENT-001/attempt_001`
- Status: **`SUPPORT_PROXY_INVALID`**
- Optimizer created / optimizer steps: `false / 0`
- G1 / G2: `NOT_RUN / NOT_RUN`
- V6.2 candidate: `false`
- Next task: `REDESIGN_DIFFERENTIABLE_SUPPORT_PROXY`

This is the preregistered static qualification stop. It is not a renderer failure, checkpoint failure, V6.1 failure, or failed training run.

## 2. Meaning of Case IP

The inherited instrumented audit passed and concluded Case IP. Independent projection agrees with gsplat; `N_pre=N_emitted`, admission retention is 1.0, pixel retention is approximately 1.0, and no cutoff, tile-admission, or early-termination mechanism explains the missing O08 support. P2/P3 moved the same base Gaussian indices out of the trusted expansion region. Therefore production renderer, gsplat, alpha thresholds, R2 rotation, and residual bounds were intentionally left unchanged.

## 3. Target-independent editable pool

`G_editable` is constructed only from the frozen 200k subject02 base, registered base foreground, base old-clothing mask, base protected mask, four-view base visibility, and formal anchor graph. A projected center must be alpha-visible in a base view and land in old clothing or the nonprotected base garment envelope. Dominant anchors are expanded by one formal anchor-graph hop. Gaussians stably assigned to protected face/hair/hand/shoe evidence are excluded.

The O01/O08 source tensors are bitwise identical for all four conditions. Target RGB, target clothing/silhouette, outfit ID, P1 parameters, and color rules are not read by pool construction.

- pool: `169106 / 200000`
- seed: `169106`
- old-clothing source: `169103`
- nonprotected envelope source: `169106`
- graph-only additions: `0` (all one-hop candidates were already seeds)
- protected exclusions: `4919`
- expanded anchors: `8704 / 10000`
- indices SHA256: `232124458848a684cd98bef1b162ec889f1e3e4c80c0ef1a5caebcceb4edb011`
- pool shared across O01/O08: `true`

The broad `169106` pool is a recorded property of this deterministic definition, not a target-specific selection.

## 4. Fixed-attribute support render

`FIXED_ATTRIBUTE_SUPPORT_RENDER_V1` renders only `G_editable` with current posed means and current posed covariance. The covariance is detached, so scale and rotation cannot receive gradients. Opacity is replaced by a constant `0.05`, colors are constant one, SH is absent, and production opacity is not read. Only the current xyz path remains differentiable. This auxiliary branch does not replace production RGB/alpha rendering and is disabled at inference.

For each view it produces `A_support`. The proposed loss-only masks were frozen as:

```text
M_place_target = (target_safe_clothing ∪ M_expand_trusted)
                 ∩ ¬protected ∩ ¬silhouette_uncertain
M_place_forbidden = M_remove_trusted ∪ trusted_background
```

These masks do not enter Oracle forward or inference.

## 5. Proposed placement objective

The implementation freezes the requested equations for a future proxy that passes qualification:

```text
L_coverage = active_normalized SmoothL1(relu(0.65 - A_support), 0)
L_spill    = active_normalized SmoothL1(relu(A_support - 0.10), 0)
L_flow     = SmoothL1((delta_xyz_i - delta_xyz_j) / xyz_bound, 0)
L_placement = λcoverage L_coverage + λspill L_spill + λflow L_flow
```

Coverage and spill are normalized independently within each view and then averaged with equal view weights. Empty regions return finite zero. The flow term uses a fixed graph and only `delta_xyz`.

## 6. Static P1/P2/P3 result

No optimization was performed.

| State | View | Mean trusted support | Hole ratio (`A<0.65`) | Spill active (`A>0.10`) | Instrumented N_pre |
|---|---:|---:|---:|---:|---:|
| P1 | back | 0.456702 | 0.755426 | 0.004246 | 30.997980 |
| P1 | right | 0.594045 | 0.528849 | 0.012010 | 30.275920 |
| P2 | back | 0.472390 | 0.753101 | 0.007137 | 7.216162 |
| P2 | right | 0.628625 | 0.462961 | 0.015622 | 8.254181 |
| P3 | back | 0.472311 | 0.748871 | 0.004970 | 6.836364 |
| P3 | right | 0.625423 | 0.462949 | 0.011821 | 6.663880 |

Pre-registered qualification:

- P1−P3 mean-support margin back: `-0.015610`, required `>=0.15` — FAIL.
- P1−P3 mean-support margin right: `-0.031378`, required `>=0.15` — FAIL.
- Spearman correlation with instrumented N_pre: `-0.314286`, required `>=0.80` — FAIL.
- P2/P3 back/right classified as underfill — PASS.
- P2/P3 back/right spill detected without threshold tuning — PASS.

The fixed-opacity alpha proxy measures broad accumulated coverage but does not preserve the known P1/P2/P3 placement ordering. Its failure is particularly clear on right view, where P2/P3 have more support and fewer proxy holes despite instrumented N_pre being approximately four times lower than P1.

## 7. Actual visual inspection

`visual_acceptance/support_proxy_contact_sheet.png` was actually opened at original detail. P1 back/right are visibly more fragmented and irregular around the silhouette; P2/P3 are broader and more uniformly filled. The visual evidence therefore confirms that this proxy cannot stand in for the instrumented per-pixel placement statistic. It is not a garment-quality or production-render acceptance.

## 8. Gradient, calibration, G1/G2, and instrumentation

The proxy gate failed, so the following were deliberately not run:

- gradient-direction audit;
- one-time 20-step gradient calibration;
- G1 400/1000-step optimization;
- G2 coverage-first warmup;
- candidate instrumented projection/admission rerun;
- O08 final acceptance and O01 regression.

Consequently there are no frozen placement weights, candidate N_pre, coverage/cloud/removal changes, checkpoint, or final Case CP/CW/CG/CR/CF. The CP/CW/CG/CR/CF matrix was not reached because its mandatory pretraining proxy gate failed.

## 9. Integrity and regression

- base fingerprint before/after: `de312ccbcabaa37cde63cd318d087b69e830ed5148779ca40456784b964f80d9`
- base bitwise exact: `true`
- base gradient count: `0`
- source instrumented evidence unchanged: `true`
- frozen branches unchanged: `true`
- production renderer / gsplat / bounds / R2 / V6 / V6.1 changed: `false`
- generated targets / seven-outfit rerun / image-conditioned training: `false`

Regression passes include: placement contract 28 tests; instrumented projection 21; alpha audit 20; V6.1 24; V6 27; V5.3 dual target 28; R2 CUDA 12; differentiable renderer 2; checkpoint; and the 7-outfit × 4-condition regression dataset fixture. The latter remains a fixture, not the formal 12×200 dataset.

## 10. Adjudication

The formal result is `SUPPORT_PROXY_INVALID`. `SUPPORT_AWARE_REGION_TRUSTED_OBJECTIVE_V6_2` is not formed. Seven-outfit re-adjudication, more target generation, and formal image-conditioned training remain unauthorized. The unique next task is:

`REDESIGN_DIFFERENTIABLE_SUPPORT_PROXY`

That task must find a differentiable support measure which reproduces the frozen P1>P2/P3 placement ordering before any optimizer is created; it must not return to renderer or alpha-threshold modification.
