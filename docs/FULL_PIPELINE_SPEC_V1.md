# CanonDressGS Full Pipeline Specification v1

Status: **FROZEN (2026-07-16)**. This freeze defines interfaces and acceptance; it does not authorize training.

## Pipeline

`K reference RGB + foreground/clothing masks + pose/camera` → frozen observation encoder → posed-anchor projection → visibility-weighted aggregation → global/local clothing features → HyperNetwork/Anchor MLP → six typed anchor residuals → Gaussian interpolation → canonical full-attribute composition → frozen MMLP-Human pose deformation → target-camera render.

The conditioning boundary excludes outfit ID/cloth ID, teacher data, target RGB and target masks. Training alone may read target RGB/masks for loss after prediction. Teacher is optional supervision/evaluation evidence and must never construct inference conditioning.

## Frozen geometry and camera convention

- `pose[165]`: 55 SMPL-X joint rotations, axis-angle, flattened. It excludes global `Rh` and translation `Th`.
- Source `Rh` is stored both as `Rh_raw[3]` axis-angle and `R_global[3,3]`. The current MMLP-Human runtime consumes the matrix.
- Posed anchors are produced by pose-dependent MMLP-Human/LBS, then `x_world = R_global @ x_posed + Th`.
- Camera is OpenCV-style world-to-camera: `x_cam = w2c[:3,:3] @ x_world + w2c[:3,3]`; positive camera z is in front. Pixel projection is `u=fx*x/z+cx`, `v=fy*y/z+cy`. `c2w = inverse(w2c)`.
- No axis flip is permitted between the manifest, anchor projector and gsplat adapter.

## Dataset and split freeze

The official release is exactly 12 outfits and the same 200 condition IDs for each outfit: 2400 RGB, 2400 foreground masks and 2400 clothing masks. Outfit-disjoint splits are O00–O07 train, O08–O09 validation, O10–O11 test. A condition is shared geometry/camera state, not an image file and not an outfit identity.

Reference and target observations must belong to the same outfit and have different condition IDs. A training sample exposes target images only to the loss path. An inference sample exposes target pose/camera only.

## Acceptance

The checker enforces zero missing/duplicates/leakage/overlap/forbidden inference fields, identical condition sets, mask-outside-foreground ≤0.5%, median posed-anchor foreground hit ≥90%, every sample ≥80%, a loader batch, and a real-model smoke command. Projection evidence must come from the actual pose deformation/projector; canonical-only projection is not acceptable.

## Audited implementation conflicts

1. `ImageConditionedEpisodeDataset` uses cloth names/IDs and random episode sampling. Minimal adaptation: use `FullDressableTrainingDataset`; never feed `outfit_id` to the model.
2. The current independent inference helper creates a zero `target_rgb` to communicate H/W. Minimal adaptation: consume `target_camera.width/height` from `FullDressableInferenceDataset`.
3. Current model rendering infers H/W from `target_rgb`. Minimal adaptation: accept explicit target dimensions before formal full-dataset inference.
4. Existing Gate 4 manifest has one synthetic outfit and three conditions. It is regression evidence only.
5. No formal 200-condition generated asset directory or generator was found. The verified upstream candidates are `subject02_formal_800k/poses.json` (2115 records) and `cameras.json` (24 cameras). A reviewed 200-row selection is therefore required before export.

These conflicts do not change this specification.
