# Full Dataset Field Consumer Matrix

| Contract field | Loader | Encoder | Projector/deformation | Renderer/loss | Inference allowed |
|---|---|---|---|---|---|
| reference RGB | CHW float | yes | no | no | yes |
| reference foreground mask | CHW float | audit/visibility | projection QA | no | yes |
| reference clothing mask | CHW float | masked global/local feature | sampled with projected anchors | no | yes |
| reference pose[165] | float | no | MMLP-Human pose/LBS | no | yes |
| reference R_global, Th | float | no | posed→world | no | yes |
| reference K, w2c | float | no | world→camera→pixel | depth visibility | yes |
| target RGB | CHW float | forbidden | forbidden | training loss only | **no** |
| target masks | CHW float | forbidden | forbidden | training loss/metrics only | **no** |
| target_edit_rgb (optional dual mode) | CHW float | forbidden | forbidden | edit/transition/clothing loss only | **no** |
| target_base_rgb (optional dual mode) | CHW float | forbidden | forbidden | preserve/protected loss only | **no** |
| target_edit/core/transition masks | CHW float | forbidden | forbidden | region-aware loss/checker only | **no** |
| target_preserve/protected masks | CHW float | forbidden | forbidden | region-aware loss/checker only | **no** |
| target_foreground_mask (`A_edit`) | CHW float | forbidden | forbidden | edit/transition alpha loss only | **no** |
| target_base_foreground_mask (`A_base`) | CHW float | forbidden | forbidden | preserve/protected alpha loss only | **no** |
| target_clothing_mask_raw | not loaded; path/checksum retained | forbidden | forbidden | builder/checker traceability only | **no** |
| target_clothing_mask (safe) | CHW float | forbidden | forbidden | clothing RGB loss; exactly `raw ∩ foreground ∩ ¬protected` | **no** |
| target_old_clothing/revealed_skin masks | CHW float | forbidden | forbidden | audit/checker only | **no** |
| target pose/R/Th | float | forbidden as clothing condition | target deformation | renderer | yes |
| target K/w2c | float | forbidden as clothing condition | no | target camera | yes |
| teacher | optional descriptor | forbidden | forbidden for gate | optional training/eval | **no** |
| outfit ID/metadata | training routing/audit | **forbidden as condition** | no | grouping | **not returned** |
| cloth embedding | absent | forbidden | forbidden | forbidden | **no** |

Current-code aliases are `reference_cloth_masks` (model) versus contract loader `reference_clothing_masks`, and `Rh` versus contract `R_global`. The v1 loader provides `reference_Rh`/`target_Rh` aliases while retaining the explicit canonical names. A training adapter must rename `reference_clothing_masks` to `reference_cloth_masks`; this freeze does not silently change either meaning.

`supervision_mode` defaults to `single_target`. `dual_target_region_aware_v1` is opt-in and adds training-only dual RGB/mask fields. `select_forward_conditioning_fields()` is the explicit allowlist boundary; no dual-target field is a model conditioning input.

Camera source audit: `scene/anchor_image_projector.py` directly consumes K and w2c using positive-z OpenCV projection; `utils/dressable_camera_utils.py` passes the same matrices to gsplat without axis flips. `utils/mmlphuman_anchor_deformation.py` applies pose/LBS, then R and Th. `train_dressable.py` currently requires pose `[K,165]`, R `[K,3,3]`, Th `[K,3]`, camera dictionaries, and target image-derived H/W. `tools/infer_image_conditioned_dressable.py` correctly avoids target image reads at storage level but constructs a zero target RGB tensor, which must be removed by the minimal adapter described in the specification.
