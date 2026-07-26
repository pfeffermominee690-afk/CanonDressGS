# Subject00 24-Cell Mask Generation Contract

This is a non-executable draft. `MASK_GENERATION_AUTHORIZED=false`.
`SEGMENTATION_METHOD=UNRESOLVED_PENDING_FORMAL_PREFLIGHT`; no model or project
pipeline is selected by this task.

The manifest binds all 24 accepted raw paths and SHA256 values, garment,
slot/camera/direction and native resolution. It expects 24 person/foreground
masks and 24 garment-region masks (48 total), stored as 8-bit single-channel
PNG at the accepted raw resolution under `E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\subject00_24_cell_masks_pending_authorization`. The root was not
created.

Raw images are immutable. Mask QA and human review are mandatory.
`TEACHER_TARGET_CREATION_AUTHORIZED=false` and
`TEACHER_ENDPOINT_OPTIMIZATION_AUTHORIZED=false`. Teacher targets require
accepted promotion, mask generation, mask QA, raw/mask binding, separate user
authorization and a Teacher-target registry.
