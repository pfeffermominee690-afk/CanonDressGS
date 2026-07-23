# AvatarReX Base-Avatar Preflight

Task: `AAAI27-AVATARREX-BASE-AVATAR-PREFLIGHT-001`

## Result

Runtime status: `DATASET_AND_PARAMETER_ADAPTER_SMOKE_PASS`. The 16-camera calibration audit, 3-camera x 3-frame loader smoke, SMPL-X schema, neutral SMPL-X forward, strict split regeneration, config parse, and model/renderer interface parse passed. Raw tree metadata was unchanged before and after the smoke.

This task does not claim a complete model or render pass. `/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/avatarrex_lbn1/gaussian/template.ply` and `/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/avatarrex_lbn1/gaussian/lbs_weights_grid.npz` are absent, so the explicit blocker is `TEMPLATE_AND_LBS_ASSETS_REQUIRED`. Calling `Scene` was intentionally forbidden because the missing-template fallback writes derived files.

## Calibration And Pose

- Raw/runtime extrinsics: `x_camera = R @ x_world + T`; camera center `C=-R^T T`.
- `imgSize=[1500,2048]` means width then height; decoded arrays are `2048 x 1500`.
- All distortion coefficients are zero, so the loader's in-memory undistortion branch is a no-op for this capture.
- SMPL-X fields exactly match the eight expected float32 arrays. Gender remains `UNKNOWN`.
- Neutral compatibility passed for frames `0/950/1900` with `10475` vertices and `20908` faces; this does not establish gender metadata.
- `scene.dataset.get_scene_scale` must be corrected or overridden to use camera centers before canary admission.

## Runtime Boundary

The current MMLP path accepts `AVRexDataset` items and statically exposes the expected gsplat `w2c/K/width/height` interface. Model construction, zero-step render, checkpoint runtime restoration, optimizer creation, backward, and training remain pending until private derived assets pass their own task.

Final classification: `AVATARREX_BASE_AVATAR_PREFLIGHT_READY_FOR_DERIVED_ASSETS`. This admits the next private derived-assets task only; it does not admit canary or formal training.
