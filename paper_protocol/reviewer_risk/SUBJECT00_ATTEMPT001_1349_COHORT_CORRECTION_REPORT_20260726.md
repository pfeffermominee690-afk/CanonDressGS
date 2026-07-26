# Subject00 Attempt 001 1349x1166 Cohort Correction and High-Resolution Review Pack

- Task: `AAAI27-SUBJECT00-1349x1166-COHORT-CORRECTION-HUMAN-REVIEW-PREP-001`
- Source HEAD: `e8af4f852fb08021d4dd7ae070bf196c7bedc589`
- Actual PNGs reparsed: `48/48`
- Recomputed `1349x1166` outputs: `34`
- Recomputed raw cell coverage: `20/24`
- Recomputed machine-pass candidates: `31` across `18/24` cells
- Review candidates: `31`; multi-candidate cells: `13`
- Final classification: `SUBJECT00_1349_COHORT_CORRECTED_READY_FOR_HIGH_RES_USER_REVIEW`

## Adjudication

The proposed O03/O04 cell swap is rejected by direct evidence. `subject00_O03_slot07_cand01` is `1350x1165`, while both `subject00_O04_slot01_cand00` and `subject00_O04_slot01_cand01` are `1349x1166`. All three PNG IHDR values agree with the PNG SHA-bound provenance, old inventory, registration metrics, request-level review panels, and registry generation code.

The old missing-cell list is therefore retained unchanged: `O01/slot_04, O01/slot_07, O03/slot_06, O03/slot_07, O04/slot_05, O04/slot_06`. The root cause of the apparent contradiction was visual misreading of the dense multi-column contact sheet, not an underlying JSON, indexing, or final-response transcription defect. Old evidence remains preserved and unmodified.

## Review State

The high-resolution review pack contains all 31 eligible candidates. Every garment PDF contains eight cell pages; dual candidates share one page, missing cells use a text-only `NO_1349_PASS_CANDIDATE` page, and no placeholder candidate image is inserted. The high-risk PDF ranks candidates for review priority only. All human decision fields remain null; `accepted=0`, `teacher_target=0`, targeted rerun authorization remains denied, and `PAPER_FINAL=false`.
