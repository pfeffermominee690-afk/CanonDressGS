# Online Gate and Completion Audit

Baseline: `ffa1fd198e1e05393ca845f89a0a2785af8a82ea`.

## Projection and visibility

`AnchorImageProjector` returns sampled features `[K,A,C]`, projected pixels/grid/depth, in-frame and positive-depth masks `[K,A,1]`, cloth-mask confidence, foreground hit, view-angle confidence, visibility confidence, and optional sampled surface depth/alpha diagnostics. Depth confidence is continuous within the configured tolerance; in-frame and positive-depth are hard masks. The legacy `visibility_confidence` multiplies clothing confidence and is retained only for Gate-4 compatibility.

Module 3 adds separate `per_view_cloth_probability` and `per_view_visibility`. The latter is exactly reference-valid (applied by aggregator) × positive-depth × in-frame × depth-visibility × bilinear foreground confidence × optional view-angle confidence. Clothing probability is not part of visibility.

The encoder masks RGB by the clothing mask before feature extraction, so non-clothing pixels do not contribute clothing features. Existing aggregation projects sampled features, normalizes visibility × global view weight, and substitutes a learned unknown feature for unseen anchors. Module 3 instead exposes strict-zero surface/clothing features plus explicit observation masks for unobserved anchors.

## Graph/topology audit

The formal base exposes `xyz_vt [10000,3]` and `nbr_vt [10000,7]`; every row includes self and six stored neighbors. This is the highest-quality available anchor topology and is deterministically expanded through two-hop adjacency to K=8. `xyz_ft [300,3]` is floating point and is not a face-index tensor. Canonical normals, SMPL-X face IDs, body-part IDs and anchor barycentrics are unavailable. Canonical xyz chunked kNN is therefore only a fallback when `nbr_vt` is absent.

Graph weights use `exp(-d²/(2 sigma_i²))`, row-normalized, where sigma is the row median neighbor distance. Metadata contains no machine paths.

## Temporary gate

The previous formal path loaded `reference_region_path`, thresholded it in `train_dressable.py`, passed `reference_only_gate` into `forward_episode`, and multiplied all legacy offsets. `TemporaryReferenceGateAdapter` remains debug-only. Formal Module-3 entry `compute_online_six_channel_residuals` rejects teacher, target image/masks, cloth ID and temporary gate fields; its gate source is `online_reference_mask_completion`.
