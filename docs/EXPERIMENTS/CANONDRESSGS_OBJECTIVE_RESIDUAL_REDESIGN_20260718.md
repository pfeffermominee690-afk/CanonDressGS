# CanonDressGS objective / residual redesign closure — 2026-07-18

## 1. Final decision

- Task: `SUBJECT02-OBJECTIVE-RESIDUAL-REDESIGN-001`
- Branch: `research/objective-residual-redesign-20260718`
- Frozen representation-triage source: `45539af725dcff6188a326ef732614acfc09cab5`
- Formal causal-matrix run commit: `7cfaacd7dc15f063fc67b4aaa466440f27299479`
- Final adjudication implementation commit: `6422ea8`
- Formal output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-OBJECTIVE-RESIDUAL-REDESIGN-001/attempt_004`
- Causal decision: **Case I** — objective and residual parameterization interact; neither T1 nor T2 passes both outfits.
- Final decision: **Case F** — T5 passes O01 but fails O08 on the preregistered per-view silhouette threshold.
- Seven-outfit gate rerun: **not authorized**.
- More target generation: **not authorized**.
- Image-conditioned training: **not authorized**.
- Unique next task: `REFINE_NEW_SILHOUETTE_MASK_SEMANTICS`.

`attempt_001` is a zero-step analysis-tool failure, `attempt_002` is invalid after a post-training persistence error, and `attempt_003` is an explicitly stopped performance-failure attempt. They remain append-only history. Only `attempt_004` is adjudicated here.

## 2. Meaning of Representation Triage Case A

Representation Triage showed that the original 200,000 Gaussian support can express O01 and O08 under independent direct optimization and under a shared four-view canonical solution. Canonical deformation, LBS, and the four frozen cameras were therefore not the demonstrated bottleneck. Case A did not prove that the formal bounded parameterization or V5.3 objective could find the same solution, and it did not authorize a garment Gaussian layer, clean-body rebuild, target generation, or image-conditioned training.

## 3. Questions answered

1. **Did V5.3 wrongly preserve old-garment change regions?** Yes, the previous preserve semantics could include base-foreground pixels that must change. V6 explicitly removes `old_garment_removal` from `neutral_preserve`. This defect was real, but T1/T2 show it was not the only bottleneck.
2. **Were current residual bounds smaller than successful-fit requirements?** Yes for xyz, log-scaling, rotation, and SH0 at the registered `1.25 × pooled |residual| p99.5` rule. Opacity-logit and SHN remain covered by current bounds.
3. **Did residual regularization suppress necessary clothing changes?** It contributed: T2 and T3 strongly outperform T1 in edit reduction. However, zero bound hits and the surviving O08 silhouette failure show that formal regularization/bounds are not the only remaining cause.
4. **Did staged optimization alone create a local minimum?** No isolated schedule failure was demonstrated. All enabled residual groups received nonzero gradients, last-100 edit/clothing slopes are negative, and T5 bound-hit/abnormal fractions are zero.
5. **Can V6 replace the old garment while protecting identity?** Yes for O01. O08 achieves strong RGB/target progress and protected/background constraints, but fails back/right silhouette coverage, so V6 is not yet closed globally.

## 4. Frozen inputs and execution contract

Only O01/O08 and four shared conditions were used: `cond_000000` front, `cond_000318` back, `cond_000017` left, and `cond_000347` right. The manifest SHA256 is `49bee929edc236af8f37d66be50eb7c44f4011d49e435af903c0e15918a83bbf`.

The Oracle forward receives only pose, `Rh`, `Th`, and camera state. Reference images are not used. Target RGB/masks are loss/evaluation fields after forward. Base Gaussians stay bitwise exact with zero base gradients in all T1–T5 runs. SHN is disabled.

Environment: Python 3.10.20, PyTorch 2.4.1+cu121, CUDA 12.1, NVIDIA GeForce RTX 4090.

## 5. Rung-2 residual statistics and bounds

Pooled absolute component statistics from successful O01/O08 Rung-2 residuals:

| Attribute | p50 | p75 | p90 | p95 | p99 | p99.5 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| xyz | 0.012915 | 0.023575 | 0.036015 | 0.044898 | 0.065983 | 0.075254 | 0.206349 |
| log_scaling | 0.031014 | 0.082172 | 0.177341 | 0.260538 | 0.458540 | 0.542432 | 1.624159 |
| rotation | 0.017594 | 0.057360 | 0.140603 | 0.219884 | 0.418270 | 0.499311 | 1.328043 |
| opacity_logit | 0.062356 | 0.178108 | 0.388369 | 0.562566 | 0.962706 | 1.113247 | 2.994730 |
| sh0 | 0.072894 | 0.188879 | 0.377932 | 0.541591 | 0.977844 | 1.194515 | 3.104405 |

Calibration ignores the most extreme 0.5% by construction; the maxima are evidence, not calibration targets. Per-outfit/all/garment/protected distributions and parameter-correlation matrices are preserved in `rung2_parameter_analysis/rung2_required_residual_statistics.{json,csv}`. Old-clothing and new-silhouette regions remain per-view image-space evidence rather than invented canonical hard partitions.

| Attribute | Current | Required = 1.25 × p99.5 | Candidate | Safety cap | Assessment |
|---|---:|---:|---:|---:|---|
| xyz | 0.050000 | 0.094068 | 0.094068 | 0.250000 | current tight |
| log_scaling | 0.350000 | 0.678040 | 0.678040 | 2.079442 | current tight |
| rotation | 0.261799 | 0.624138 | 0.624138 | 3.141593 | current tight |
| opacity_logit | 2.000000 | 1.391559 | 2.000000 | 6.000000 | current sufficient |
| sh0 | 0.250000 | 1.493144 | 1.493144 | 2.000000 | current tight |
| shN | 0.100000 | not calibrated | 0.100000 | 0.100000 | disabled/frozen |

`BOUND_CONTRACT_INCOMPATIBLE = false`; no candidate exceeds its safety cap. These remain research candidates and do not overwrite the production configuration.

## 6. V6 support-aware region semantics

`SUPPORT_AWARE_REGION_TRUSTED_OBJECTIVE_V6` constructs one deterministic priority-disjoint partition:

1. `protected_identity`: face, hair, hands, shoes; supervised to base RGB/alpha.
2. `new_silhouette`: target foreground outside base foreground, excluding protected pixels; supervised to target alpha.
3. `target_garment`: safe target clothing, excluding protected and new-silhouette pixels; supervised to edit target.
4. `old_garment_removal`: old clothing that lies in the edit region but not target clothing, excluding protected/new-silhouette pixels; supervised to edit target, never base.
5. `transition`: uncertain boundary pixels not occupied above; soft boundary-aware alpha target with SmoothL1.
6. `neutral_preserve`: stable base foreground that is neither old clothing nor edit/protected/occupied; supervised to base.
7. `background`: outside both target and base foreground after higher-priority regions; supervised to base/background.

Runtime assertions reject region overlap, protected pixels in garment supervision, and old-garment-removal pixels in neutral preserve.

## 7. V6 objective

The frozen objective is

`L = Σ_g λ_g L_g`,

with multi-scale masked Charbonnier edit/identity/background/preserve terms, target-progress margin loss, new-silhouette BCE+Dice, transition SmoothL1, bound-normalized residual regularization, and stability penalties for bound edges, opacity saturation, extreme scale, and floating splats.

For changed trusted-edit pixels:

`d_target = mean_c |prediction - target|`

`d_base = mean_c |prediction - base|`

`L_target_progress = mean[relu(d_target - d_base + 0.02)]`, active only where `mean_c |target-base| > 0.01`.

This formula contains no purple, hoodie, outfit-ID, or outfit-specific margin rule.

Frozen weights after the single preregistered 20-step calibration:

| Group | Weight |
|---|---:|
| edit_rgb | 1.0 |
| target_progress | 1.0 |
| new_silhouette | 0.0054805561326566115 |
| transition_alpha | 0.0027402780663283058 |
| identity | 2.0 |
| background | 1.0 |
| neutral_preserve | 0.5 |
| residual | 0.0001 |
| stability | 0.0001 |

Median gradient norms were edit `0.0327960763`, alpha `1.4960195415`, and regularization `1.7977376e-06`. The frozen scales are alpha `0.0109611123` and regularization `1.0`, satisfying alpha/edit ≤ 0.5 and regularization/edit ≤ 0.25. No visual tuning was used.

## 8. T0–T5 causal matrix

T0 is the immutable historical V5.3 failure and was not rerun.

| Run | Definition | Steps | Mean edit reduction | Mean target-closer | Min silhouette IoU | Max protected MAE | Max background | Numeric | Visual | Final |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| T1/O01 | capacity loss + formal/current bounds/current regularization | 800 | 0.648522 | 0.850328 | 0.894968 | 0.005095 | 0.021541 | FAIL | FAIL | FAIL |
| T1/O08 | same | 800 | 0.609540 | 0.700349 | 0.848014 | 0.005744 | 0.029173 | FAIL | FAIL | FAIL |
| T2/O01 | V5.3 + unbounded direct + minimal stability | 800 | 0.875059 | 0.915793 | 0.917918 | 0.007448 | 0.012586 | PASS | WARN | PASS |
| T2/O08 | same | 800 | 0.853961 | 0.910802 | 0.883044 | 0.007963 | 0.017379 | FAIL | WARN | FAIL |
| T3/O01 | capacity loss + formal/candidate bounds/bound-normalized reg | 800 | 0.816034 | 0.920065 | 0.899219 | 0.004509 | 0.021433 | FAIL | WARN | FAIL |
| T3/O08 | same | 800 | 0.851826 | 0.936413 | 0.848304 | 0.004707 | 0.029742 | FAIL | WARN | FAIL |
| T4/O01 | V6 + formal/current bounds/bound-normalized reg | 1000 | 0.755873 | 0.912498 | 0.897705 | 0.007614 | 0.017823 | FAIL | WARN | FAIL |
| T4/O08 | same | 1000 | 0.721096 | 0.853743 | 0.846048 | 0.007702 | 0.026223 | FAIL | WARN | FAIL |
| T5/O01 | V6 + formal/candidate bounds/bound-normalized reg | 1000 | 0.875152 | 0.947128 | 0.904302 | 0.006095 | 0.017322 | PASS | WARN | PASS |
| T5/O08 | same | 1000 | 0.872734 | 0.961704 | 0.848158 | 0.006411 | 0.026937 | FAIL | WARN | FAIL |

All last-100 edit/clothing slopes are negative after the append-only derived-metric repair. The original metrics SHA256, original zero V6 clothing slope, corrected values, and immutable loss-curve SHA256 are recorded in `comparisons/DERIVED_SLOPE_REPAIR_SUMMARY.json`. No optimization was rerun for that repair.

## 9. T5 detailed result

| Outfit/view | Edit reduction | Target closer | Silhouette IoU | New-silhouette recall | Base/purple retention | Protected MAE | Background leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| O01 front | 0.896954 | 0.926880 | 0.949975 | 0.825563 | 0.061047 | 0.005904 | 0.012152 |
| O01 back | 0.885362 | 0.964689 | 0.904302 | 0.780162 | 0.029152 | 0.006095 | 0.017322 |
| O01 left | 0.813163 | 0.930218 | 0.959019 | 0.758985 | 0.060370 | 0.004529 | 0.005820 |
| O01 right | 0.905126 | 0.966725 | 0.912554 | 0.692243 | 0.028161 | 0.004344 | 0.014959 |
| O08 front | 0.870982 | 0.960686 | 0.948875 | 0.828109 | 0.036216 | 0.005731 | 0.012347 |
| O08 back | 0.840813 | 0.940338 | **0.874600** | 0.765639 | 0.053046 | 0.006411 | 0.022401 |
| O08 left | 0.876599 | 0.968250 | 0.940233 | 0.859770 | 0.027623 | 0.004361 | 0.009002 |
| O08 right | 0.902541 | 0.977542 | **0.848158** | 0.660443 | 0.019103 | 0.004211 | 0.026937 |

T5/O01 slopes are `-1.2196091e-05`; T5/O08 slopes are `-2.5686032e-05`. Both have abnormal Gaussian fraction `0`, maximum bound-hit fraction `0`, base bitwise equality, and zero base gradients.

## 10. Actual visual inspection

Codex actually opened all ten persisted bundles at original detail. Each bundle contains step 0/200/480/final four-view panels and final prediction, protected overlay, head/face/hair crop, hands/arms crop, shoes crop, garment-boundary overlay, and absolute-error/splat overlay.

- T1 O01/O08: FAIL. Old purple/mottled appearance remains dominant or incomplete; O08 also has dense surface speckling.
- T2 O01/O08: WARN. Target semantics form, with localized edge voids/speckling and stable protected regions.
- T3 O01/O08: WARN. Target semantics form; O08 has visible surface/rim artifacts.
- T4 O01/O08: WARN. Old purple is no longer dominant, but mottling/speckling and silhouette erosion remain.
- T5 O01: WARN. Clear shared target garment, stable protected regions, only minor/local boundary defects.
- T5 O08: WARN. Target garment forms, but back/right underfill and surface speckling match the numeric silhouette failure.

The signed observations and source SHA256 values are in `visual_acceptance/visual_decisions.json` and `VISUAL_BUNDLE_MANIFEST.json`.

## 11. Causal conclusion

T1 fails both outfits. T2 passes O01 but fails O08 silhouette. Therefore neither “objective only” nor “bounds/parameterization only” explains both outfits; the primary adjudication is **Case I**.

T5 proves that V6 plus calibrated bounds restores the formal canonical representation for O01. It does not close O08: only `all_view_silhouette` fails, specifically back/right, while edit reduction, target-closer, protected, background, negative slopes, abnormal-Gaussian, and bound-truncation checks all pass. Consequently:

- bounds are not the remaining proven blocker (zero bound hits);
- residual regularization is not the remaining proven blocker (negative slopes and no truncation);
- splat control is not the remaining proven blocker (zero abnormal fraction and no large opacity cloud);
- optimization schedule is not proven to be the blocker (all enabled groups receive gradients and losses continue decreasing);
- the only isolated failure is O08 new-silhouette coverage/mask semantics.

Final state is **Case F**, not `V6_FORMAL_ORACLE_PASS` for both outfits.

## 12. Regression evidence

- Objective/residual contract: 27/27 PASS.
- Rotation autograd R2: 12/12 PASS, CUDA finite test executed.
- V5.3 dual-target supervision: 28/28 PASS.
- Differentiable renderer unit checks: PASS.
- Full training checkpoint checks: PASS.
- AAAI 7-outfit × 4-condition fixture dataset regression: PASS (fixture, not official 12×200 dataset); projection/real-model smoke are intentionally not part of this read-only regression.
- Core Python `py_compile`: PASS.
- `git diff --check`: PASS.

## 13. Permissions and next task

No seven-outfit rerun, new target generation, image-conditioned training, garment Gaussian layer, or production-config replacement is authorized by this result.

The only next task is:

`REFINE_NEW_SILHOUETTE_MASK_SEMANTICS`

It must start as a separate, preregistered task and must not be launched automatically from this closure.
