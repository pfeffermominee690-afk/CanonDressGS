# Subject00 Resolution Contract Options

## Option A: STRICT_NATIVE_RESOLUTION

- Keep native 1024x1536 outputs: 5
- Regenerate non-A outputs after explicit authorization: 43
- Pixel transformations: none

## Option B: DETERMINISTIC_RESIZE_ONLY

- Keep native A outputs: 5
- Eligible strict 2:3 portrait B outputs for future Lanczos resize: 0
- Total potentially retained after an explicit resize contract: 5
- Regenerate C/D/E/F/G outputs after explicit authorization: 43
- Crop, pad, rotate, and aspect distortion remain forbidden

No B candidates exist in this audit, so Option B currently retains no additional image over Option A. This task does not select an option, resize an image, or authorize regeneration.
