# R3 O00 Under-Clothes Arm Support Closure

## Final adjudication

The formal candidate is **`ARM_SUPPORT_FAIL / Case C`**. The single preregistered 12,000-Gaussian Medium arm support is finite, subject02-only, formally skinned, and frozen, but none of the nine frozen two-parameter calibration combinations jointly satisfies covered-state occlusion and revealed-arm repair. The mandatory stop fired before the fixed-open Gaussian Oracle; optimizer steps are **0**.

The formal result is `SUBJECT02-O00-ARM-SUPPORT-CLOSURE-001/attempt_003` at run commit `5bc0726cc0f5c4d34e8d9f846c582e29d1235cfa`. `attempt_001` and `attempt_002` are preserved zero-step tool failures: the former lost narrow transition labels during barycentric-owner transfer, and the latter compared CUDA tensors with CPU validator references. Neither reached a calibration render or produced candidate metrics.

## Frozen inputs and scope

- Base: formal subject02 200,000-Gaussian checkpoint; canonical tensor fingerprint before/after `de312ccbcabaa37cde63cd318d087b69e830ed5148779ca40456784b964f80d9`.
- Conditions: `cond_000000` front, `cond_000318` back, `cond_000017` left, `cond_000347` right.
- Data: existing O00 raw edit/base targets, masks, pose, `Rh/Th`, `K`, and `w2c`; no condition or outfit image was regenerated.
- Support config: `configs/audit/o00_arm_support_closure_v1.yaml`, SHA256 `9dca0e5d8317fef473ee1d40966e88582e9cf940edaa2861ab1bdb2bef8cdf53`.
- Oracle config was preregistered but not executed: `configs/oracle/o00_fixed_open_gaussian_v1.yaml`, SHA256 `ca6b9727ed961788b8e28a7ef50e4aa63d246af0d432e509730e4a3277371c97`.
- No formal model, renderer, V5.3 loss, residual bound, anchor mapping, checkpoint, legacy repository, condition, or outfit sheet was modified.

## Arm support construction

The support contains exactly `12,000` frozen surface-aligned Gaussians on formal SMPL-X parts 16–19 only. Hands/fingers 20–54, torso, face, hair, legs, and shoes are excluded. Each point stores its subject02 canonical surface/barycentric binding, 55-joint interpolated LBS, inward normal displacement, anisotropic tangent-plane scaling, normalized local-frame `wxyz` rotation, opacity, degree-0 SH0, disabled SHN, body part, side, skin region, and provenance.

The skin field revalidates all `63,445` trusted subject02 pixels (`18,588 / 16,843 / 13,366 / 14,648` across front/back/left/right) and maps them onto `1,262` observed surface vertices. It distinguishes:

1. left upper arm;
2. left lower arm;
3. right upper arm;
4. right lower arm;
5. shoulder transition;
6. wrist transition.

All six regions have direct subject02 observations. Only subject02 observations, left/right symmetry, same-region geodesic propagation, and bounded within-region low-frequency interpolation are allowed. Jay, Rose, generic skin, clay color, external textures, and generated textures are absent.

The frozen asset is `support/o00_arm_support_12000.pt`, SHA256 `4119d447a072936d240d7f68e1f70a9e993f8e9abf4501fc1e818fc4cb314541`, tensor fingerprint `7e713654be972c493cb7544dbb6cb5ed05e3ecc55f4dfefa88e730046c6a8f15`. It has zero trainable parameters and remains unchanged.

## Two-parameter calibration

The complete preregistered grid was evaluated once:

- inward offset: `0.004 / 0.007 / 0.010 m`;
- global opacity scale: `0.25 / 0.45 / 0.65`;
- total combinations: `9`;
- target RGB used to select calibration: `false`.

The deterministic priority chose `inward_offset=0.010 m`, `opacity_scale=0.25`, corresponding to final per-Gaussian opacity `0.1625`. It is the least covered-visible candidate, not a passing candidate.

| View | Hole repair recall | Background leakage | Covered support visibility |
|---|---:|---:|---:|
| front | 0.870485 | 0.129515 | 0.009694 |
| back | 0.862733 | 0.137267 | 0.007265 |
| left | 0.979153 | 0.020847 | 0.035961 |
| right | 0.782226 | 0.217774 | 0.024312 |
| mean | 0.873649 | 0.126351 | 0.019308 |

The anatomical-envelope outside fraction is `0.0`, and all four views are finite with no pose explosion. The frozen acceptance requires repair `min>=0.90, mean>=0.95`, leakage `each<=0.05, mean<=0.03`, and covered visibility `each<=0.01`; all three metric groups fail.

## Actual visual inspection

`visuals/o00_arm_support_four_view_contact_sheet.png` was opened at original resolution with the local image viewer. P2 repairs large P1 arm holes and follows all four formal poses without left/right exchange or explosion. Regional color is less uniform than the historical R3 median-color probe. However, shoulder caps and wrist joins remain abrupt, side views are locally bulky, and P3 visibly changes intact sleeves. Shoulder and wrist are both `WARN`; overall visual status is `WARN`. There is no gross disconnected brown tube, but the representation is not acceptable because the quantitative occlusion/repair tradeoff remains far outside its frozen limits.

## Oracle stop and decision matrix

`ARM_SUPPORT_PASS` is a hard precondition for the 480-step O00 fixed-open Gaussian Oracle. Because it failed, no Oracle directory, optimizer, checkpoint, residual history, or render milestone was created. Consequently edit/clothing reductions, last-80 slopes, short-sleeve formation, long-sleeve removal, and T-shirt/jeans four-view fit are **not evaluated**, not zero and not failed training metrics.

The final matrix case is **Case C**: the current SMPL-X-to-Gaussian arm support representation is still invalid. The next work, if separately authorized, must improve the support representation while retaining the frozen acceptance contract. Building a formal full-body support base and starting formal image-conditioned training are both **not allowed** from this result.

## Commands and checks

Formal commands used the clean cloud worktree and `/root/autodl-tmp/conda_envs/mmlphuman/bin/python`:

```text
python tools/run_o00_arm_support_closure.py --phase preflight --expected-head 5bc0726... --output .../attempt_003
python tools/run_o00_arm_support_closure.py --phase calibrate --output .../attempt_003
python tools/run_o00_arm_support_closure.py --phase finalize --output .../attempt_003 --inspection-method "actual image opening with local view_image" ...
```

The new O00 suite passes `12/12` locally and on CUDA. Existing R2 `12/12`, R3 `8/8`, R3-CLEAN `12/12`, GEOMCAM `11/11`, Module 4B `12/12`, Module 4B-R `10/10`, and V5.3/dual-target `28/28` checks pass locally; cloud R2 `12/12` also passes with CUDA coverage. Python compilation and `git diff --check` pass.

