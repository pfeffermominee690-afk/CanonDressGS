# Subject00 Portrait Canary V2 Visual Fail Report

- Task: `AAAI27-SUBJECT00-PORTRAIT-CANARY-V2-FAIL-AND-VALID-REGION-AUDIT-001`
- Reviewer: `USER_AND_GPT_MANUAL_REVIEW`
- Contact sheet SHA-256: `cde7de35fb0a18a6c2c719fc1c7cd7ad78921545f7be9bf46a70ca7a6abe7f2a`
- Human visual result: `0 PASS / 4 FAIL`
- Classification: `SUBJECT00_PORTRAIT_CANARY_V2_TECHNICAL_PASS_VISUAL_FAIL_BACKGROUND_OUTPAINT`

| Request | Garment | Slot | Camera | Decision | Primary failure |
| --- | --- | --- | --- | --- | --- |
| `subject00_O01_slot00_cand00` | O01 | slot_00 | cam17 | FAIL | FAIL_WARPED_CEILING_AND_UNNATURAL_CARPET_PERSPECTIVE |
| `subject00_O03_slot03_cand00` | O03 | slot_03 | cam23 | FAIL | FAIL_SEVERE_CEILING_AND_CARPET_OUTPAINT_ARTIFACTS |
| `subject00_O04_slot02_cand00` | O04 | slot_02 | cam14 | FAIL | FAIL_LARGE_CEILING_WEDGE_AND_CARPET_POLYGON |
| `subject00_O01_slot06_cand00` | O01 | slot_06 | cam09 | FAIL | FAIL_ENLARGED_REPEATED_CARPET_TEXTURE_AND_PERSPECTIVE |

## Adjudication

All four requests passed the registered-crop preflight and native `1024x1536` output check. They also resolved the V1 subject-shrink and gray-canvas failure. These are engineering passes only.

All four fail visual acceptance because generated ceiling and carpet regions do not preserve registered room geometry. The failures include warped or wedge-shaped ceiling structures, repeated or enlarged carpet blocks, and inconsistent perspective. The generated top and bottom bands are not valid multiview observations and cannot be used as Teacher supervision.

The original review manifest remains unchanged. This report and its JSON adjudication are sidecars. `accepted=0`, `Teacher targets=0`, remaining-39 authorization is `DENIED`, Formal Base is `PENDING`, and `PAPER_FINAL=false`.
