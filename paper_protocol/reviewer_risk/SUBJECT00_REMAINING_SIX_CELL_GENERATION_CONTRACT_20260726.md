# Subject00 Remaining-Six-Cell Generation Contract

- Task: `AAAI27-SUBJECT00-O03-CANARY-HUMAN-REVIEW-FREEZE-001`
- Planned namespace: `attempt_005_subject00_remaining_six_cell_generation`
- Generation authorized: `false`
- Request count: `6`
- Retry authorized: `false`
- Accepted promotion: `false`
- Teacher-target promotion: `false`

## Scope

This draft covers only the six cells listed below. It prepares source, mask, camera, pose, prompt, and SHA bindings. It does not execute image generation and does not create the planned attempt namespace.

- `O01/slot04/cam11/right` -> `subject00_O01_slot04_remaining_attempt005_cand00`
- `O01/slot07/cam05/back` -> `subject00_O01_slot07_remaining_attempt005_cand00`
- `O03/slot01/cam21/front-left` -> `subject00_O03_slot01_remaining_attempt005_cand00`
- `O03/slot06/cam09/back-right` -> `subject00_O03_slot06_remaining_attempt005_cand00`
- `O04/slot05/cam02/back-left` -> `subject00_O04_slot05_remaining_attempt005_cand00`
- `O04/slot06/cam09/back-right` -> `subject00_O04_slot06_remaining_attempt005_cand00`

## Frozen Prompt Bindings

- `O01`: `subject00_O01_positive_v1` / `a8ab791e57a57fd2c63bff333bdd9937e07b61f8a43e40018215d7c39df085c7`
- `O03`: `subject00_O03_positive_v1` / `d92a379a9450f2a5f1d092537c723f22f24a9c4ec01e4f9bc5a7496f02369f35`
- `O04`: `subject00_O04_positive_v1` / `ac71757350a278dc4b28d5cb7bc6abe7417f39e9a8e2beec579fa25b9b729473`

The full positive and negative prompt text is frozen per request in `subject00_remaining_six_cell_generation_manifest_draft_20260726.json`. O03 execution must retain the proven complete hood-removal clauses from the four-cell canary in addition to the frozen O03 garment prompt.

## Output Gate

- Native PNG only.
- Allowed native resolution family: `1348x1167`, `1349x1166`, or `1350x1165` landscape.
- No resize, crop, padding, outpainting, recomposition, re-encoding repair, or other postprocessing.
- One generation call and one candidate per cell after separate explicit authorization.
- No retry and no generate-many-then-select behavior.
- Every output remains a candidate pending technical validation and human review.
- No accepted or Teacher-target promotion is authorized by this contract.

## Status

`generation_authorized = false`. The only next task is `USER_AUTHORIZE_SUBJECT00_REMAINING_SIX_CELL_GENERATION`.
