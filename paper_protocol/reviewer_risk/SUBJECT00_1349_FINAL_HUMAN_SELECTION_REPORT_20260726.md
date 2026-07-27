# Subject00 1349 Final Human Selection Report

- Task: `AAAI27-SUBJECT00-1349-HUMAN-SELECTION-AND-O03-CANARY-PREP-001`
- Reviewer: `USER_AND_GPT_MANUAL_REVIEW`
- Source: `research/subject00-1349-cohort-correction-human-review-prep-20260726` at `f6f634c029abc72191ce1edbc99e56fe781f0bfd`
- Old review manifest evidence SHA256: `18d0076ad10cf739eeb9f150e4e65d5841be570de0935a63612c391da3cdf47c`
- Repository worktree copy SHA256: `50f4d3080589bcb0bbfc7a666589312e21e3da0e46217f3799c628f5e2dacd41`
- Canonical content SHA256: `fa8c6eca1fc3650c3a8637ffe9e8a8352f9e3a3f110ca5236c262223ddd12fb6`
- Final classification: `SUBJECT00_1349_HUMAN_SELECTION_FROZEN_O03_CANARY_PENDING_AUTHORIZATION`

## Manual Decision Freeze

The 31 high-resolution candidates are frozen as 23 visual passes and 8 failures, with no uncertain decisions. Machine registration PASS is supporting geometric evidence and is not equivalent to human Visual PASS.

- Selected for cell: 14
- Pass but not selected: 9
- Failed: 8
- Accepted: 0
- Teacher target: 0

### Selected Requests

- `subject00_O01_slot00_cand01`
- `subject00_O01_slot01_cand01`
- `subject00_O01_slot02_cand01`
- `subject00_O01_slot03_cand01`
- `subject00_O01_slot05_cand00`
- `subject00_O01_slot06_cand01`
- `subject00_O03_slot02_cand00`
- `subject00_O03_slot03_cand01`
- `subject00_O04_slot00_cand01`
- `subject00_O04_slot01_cand01`
- `subject00_O04_slot02_cand00`
- `subject00_O04_slot03_cand01`
- `subject00_O04_slot04_cand01`
- `subject00_O04_slot07_cand01`

### Failed Requests

- `subject00_O03_slot00_cand00`
- `subject00_O03_slot00_cand01`
- `subject00_O03_slot01_cand00`
- `subject00_O03_slot01_cand01`
- `subject00_O03_slot02_cand01`
- `subject00_O03_slot03_cand00`
- `subject00_O03_slot04_cand00`
- `subject00_O03_slot05_cand00`

All eight failures use `SOURCE_GARMENT_HOOD_RESIDUAL_IN_O03_SUIT`. `subject00_O03_slot02_cand01` is revised from preliminary `PASS_CANDIDATE` to final `FAIL` because high-resolution side-by-side review revealed the dark-blue source hood.

## Missing Cells

- `O01/slot_04`
- `O01/slot_07`
- `O03/slot_00`
- `O03/slot_01`
- `O03/slot_04`
- `O03/slot_05`
- `O03/slot_06`
- `O03/slot_07`
- `O04/slot_05`
- `O04/slot_06`

All ten cells require a rerun, but rerun authorization remains false.

## O03 Finding

O03 hoodie-removal failure is systematic in the existing candidate set: 2/8 cells are selected, 6/8 are missing, and 8 candidate images have explicit hood contamination. Managed Image Edit frequently changes the suit torso while retaining the source O01 hood around the head. This is an image-generation candidate failure, not evidence that O03 is unrepresentable, that MMLP-Human cannot represent suits, or that an O03 Teacher must fail.

## Canary

The four-request `SUBJECT00_O03_HOOD_REMOVAL_TARGETED_CANARY` contract is drafted but not authorized. It covers front, right, back-left, and back views at native 1349x1166. No image generation, retry, external API call, API-key read, accepted promotion, Teacher-target promotion, Formal Base training, Teacher optimization, or paper modification occurred.

## Next Task

`USER_AUTHORIZE_SUBJECT00_O03_HOOD_REMOVAL_CANARY_4_REQUESTS`
