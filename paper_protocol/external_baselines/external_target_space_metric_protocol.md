# External Target-Space Metric Protocol

Task: `AAAI27-CANONDRESSGS-EXTERNAL-BASELINE-FEASIBILITY-FROM-BUNDLE-001`

## Scope and denominators

- The evaluation unit is one frozen subject02 garment/target-condition record with ground-truth RGB, foreground mask, garment mask, target pose, and target camera.
- The target RGB denominator is every preregistered test record for which all compared methods produced an output. Failures remain failures and are not removed; a method-level failure count is reported separately.
- The mask denominator is the same preregistered record list, with no method-specific filtering.
- Render every method at the exact target camera, pose when supported, native target resolution, color space, and background convention. A method without pose control is evaluated only in the one fixed-pose canary and is labeled as such.
- Freeze a crop from the ground-truth foreground bounding box plus a 10% margin. Use the same crop for all methods. Pixels outside this crop are excluded. No prediction-dependent crop is allowed.

## Regions

- Full-image metrics use the fixed target crop, including its in-crop background.
- Garment-region metrics use the ground-truth garment mask inside the fixed crop.
- Protected-region metrics use `ground_truth_foreground AND NOT dilate(ground_truth_garment_mask, 5 px)`. Background is excluded.
- Silhouette IoU and Boundary F use ground-truth foreground and predicted alpha thresholded at 0.5. Boundary F tolerance is exactly 2 pixels at native resolution.

## Implementations

- RGB is sRGB in `[0,1]`; no per-method color correction is allowed.
- LPIPS: `lpips==0.1.4`, AlexNet backbone, inputs mapped to `[-1,1]`, spatial reduction disabled.
- PSNR: RGB MSE with data range 1.0. Infinite values are retained and separately counted.
- SSIM: `skimage.metrics.structural_similarity`, `channel_axis=-1`, `data_range=1.0`, `gaussian_weights=True`, `sigma=1.5`, `use_sample_covariance=False`.
- Report full-image LPIPS/PSNR/SSIM; garment-region LPIPS/PSNR/SSIM; silhouette IoU; Boundary F; protected-region LPIPS and RGB MAE.
- Also report adaptation time, trainable parameters, incremental storage, peak VRAM, rendering time/FPS, animation compatibility, and frozen-backbone preservation.

## Human review

- Use three reviewers, blinded randomized method columns, the same fixed crop, and no cherry-picking.
- Identity contamination is binary per record and decided by majority vote. Severe artifacts use grades 0-3; grades 2-3 count as severe.
- All missing outputs count as failures. Review all preregistered records or a preregistered uniform subset shared by every method.

## Exclusions

- Endpoint LPIPS, exact endpoint match, coefficient MAE/RMSE, endpoint-coordinate distance, and Teacher snapping parity are internal diagnostics and are forbidden in the external target-space table.
- `FACE_ID_PROTOCOL_STATUS=NOT_USED_DUE_TO_UNVALIDATED_FACE_ID_PROTOCOL`. No face-identity score may be added without a public license-compatible model, frozen crop/denominator/threshold, and independent threshold validation for all methods.
