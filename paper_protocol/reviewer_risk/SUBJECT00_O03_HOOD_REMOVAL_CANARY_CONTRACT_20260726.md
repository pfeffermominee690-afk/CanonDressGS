# Subject00 O03 Hood-Removal Canary Contract

- Task: `AAAI27-SUBJECT00-1349-HUMAN-SELECTION-AND-O03-CANARY-PREP-001`
- Canary: `SUBJECT00_O03_HOOD_REMOVAL_TARGETED_CANARY`
- Attempt namespace: `attempt_004_o03_hood_removal_targeted_canary`
- Generation authorized: `false`
- Request count: `4`
- Expected output: native `1349x1166` landscape PNG

## Purpose

Test whether a targeted edit can replace the complete O01 hoodie with the frozen O03 formal suit while preserving the registered Subject00 condition. This is an image-generation candidate canary, not a Base Avatar representation test.

## Requests

- `subject00_O03_slot00_canary_attempt004_cand00`
- `subject00_O03_slot04_canary_attempt004_cand00`
- `subject00_O03_slot05_canary_attempt004_cand00`
- `subject00_O03_slot07_canary_attempt004_cand00`

The fixed cells are O03/slot00/cam17/front, O03/slot04/cam11/right, O03/slot05/cam02/back-left, and O03/slot07/cam05/back. Each request is bound to the original registered condition path and SHA in `subject00_o03_hood_removal_canary_manifest_draft_20260726.json`.

## Prompt Contract

- `Replace the entire source hoodie with a complete formal suit.`
- `Remove the original blue-and-white hoodie completely.`
- `Remove the source hood from the head and neck region.`
- `The final outfit must contain no hood.`
- `The head, hair, face, ears, and neck must remain naturally visible according to the source identity and camera direction.`
- `Preserve identity, face structure, skin tone, body proportions, pose, hands, feet, camera, framing, lighting, and background.`
- `Generate one person only.`
- `Do not retain blue hoodie fabric around the head, neck, shoulders, torso, sleeves, or waist.`
- `Do not add a coat hood, sweatshirt hood, scarf-like hood, or head covering.`
- `Produce a complete suit jacket, shirt, trousers, and formal styling under the frozen O03 garment contract.`

The execution prompt must not request face beautification, a different hairstyle, a camera change, subject movement, zoom, or background reconstruction.

## Execution Contract

- Exactly four generation calls, one per request, only after explicit user authorization.
- Zero retries and one candidate per cell; no generate-many-then-select behavior.
- No resize, crop, padding, re-encoding repair, portrait canvas, output postprocessing, or background outpainting.
- No external API, API-key read, cloud image write, automatic acceptance, accepted promotion, or Teacher-target promotion.
- Output must parse as PNG at exactly 1349x1166.

## Human Gate

All four outputs must pass every gate: exact_resolution, similarity_registration, complete_hood_removal, correct_O03_suit, identity_consistency, pose_camera_preservation, hands_feet_completeness, background_geometry, garment_boundary, single_person. If fewer than 4/4 pass, the remaining six O03 missing cells must not be batch-generated.

## Status

`generation_authorized = false`. This task drafts and freezes the contract only; it performs no image generation.
