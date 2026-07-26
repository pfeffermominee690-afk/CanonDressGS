# Subject00 24-cell Mask Generation Execution Report

Task: `AAAI27-SUBJECT00-24-CELL-MASK-GENERATION-EXECUTION-001`

Authorization: `AUTHORIZE_SUBJECT00_24_CELL_MASK_GENERATION_FROM_FROZEN_CONTRACT`

## Outcome

The frozen local SegFormer executor processed all 24 accepted Subject00 cells
in manifest order. It generated exactly 24 person masks and 24 garment masks.
All formal masks passed structural technical QA (native resolution, PNG mode
`L`, strict binary values, non-empty masks, and garment subset of person).
Human decisions remain null and no mask was promoted to accepted.

## Execution boundary

- Source: `research/subject00-24-cell-mask-generation-preflight-20260726` at `7cdcb4148c40222bad2798b977eae6db74fa03ba`
- Execution branch: `research/subject00-24-cell-mask-generation-execution-20260726`
- Attempt root: `E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-MASKS-001\attempt_001_subject00_24_cell_person_garment_masks`
- Fixed implementation SHA256: `b602df821f30046941bd98956d5afc552f36c7c72f1fe4af5c58621713a36693`
- Fixed model: `mattmdjaga/segformer_b2_clothes` revision `584abc1e1d260e23c0fc627c5217a09b2b461046`
- Fixed weights SHA256: `8f86fd90c567afd4370b3cc3a7e81ed767a632b2832a738331af660acc0c4c68`
- Model download bytes: `0`
- Environment install calls: `0`
- Automatic retries: `0`

## Counts

- Accepted/raw: `24/24`
- Inference images / forward batches: `24/24`
- Person masks: `24`
- Garment masks: `24`
- Total masks: `48`
- Person technical QA: `24/24`
- Garment technical QA: `24/24`
- Pair technical QA: `24/24`
- Garment-outside-person pixels: `0`
- Protected-region non-empty: `24/24`
- Disclosed limitations propagated: `9/9`
- Registration-override review pages: `2/2`

## Review boundary

The review layout is labeled `DISPLAY_ONLY_MASK_REVIEW_LAYOUT`. The required package contains
`90` artifacts; two additional
aggregate pages emitted by the frozen executor are also retained. All 24
person, garment, and pair human decisions remain `null`; `MASK_ACCEPTED_COUNT`
and `TEACHER_TARGET_COUNT` remain `0`.

## Immutability and classification

All `320` bound legacy files matched
their pre-execution byte counts and SHA256 values. Attempts 001-005 and the 24
accepted raws have zero mutations. No paper body was modified.

Final classification:
`SUBJECT00_24_CELL_MASK_GENERATION_TECHNICAL_PASS_PENDING_HUMAN_REVIEW`

Next unique task: `USER_REVIEW_SUBJECT00_24_CELL_GENERATED_MASKS`

## Git synchronization

- Execution artifact commit: `2ceed50ed1599622d9544e10aff78fc3eba2d304`
- Origin: `PUSHED_AND_VERIFIED`
- Cloud Git: `NOT_PUSHED_HOSTNAME_RESOLUTION_FAILED`
- Final reporting worktree: `CLEAN_AFTER_FINAL_REPORTING_COMMIT_VERIFIED_EXTERNALLY`
