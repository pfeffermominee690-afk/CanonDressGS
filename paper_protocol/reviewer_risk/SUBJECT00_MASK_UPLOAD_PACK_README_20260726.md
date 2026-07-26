# Subject00 Mask Human-Review Upload Pack

Task: `AAAI27-SUBJECT00-24-CELL-MASK-HUMAN-REVIEW-PACK-001`

All pages are `DISPLAY_ONLY_MASK_REVIEW_LAYOUT`. They are not formal masks.

Human review fields remain pending/null. This pack does not authorize mask
acceptance, Teacher targets, Teacher Endpoint optimization, or training.

## Upload order

1. `subject00_masks_master_24cell_overview.png` — Master 24-cell overview grouped by O01/O03/O04 — 7200×4800 — SHA256 `3a90cb94baaf2fb06b30ec9dc91427175a4aff5178eb28986f97022ecdc6d857`
2. `subject00_O01_masks_slots00_03_review.png` — O01 slots 00-03 six-column detail review — 7200×4400 — SHA256 `449669aeac38e376450a3ce099a917887105d6c851bf92cc83b2cf6157fd3b30`
3. `subject00_O01_masks_slots04_07_review.png` — O01 slots 04-07 six-column detail review — 7200×4400 — SHA256 `abc2d3fb050f15094530bec4bc634e042d8a4f079113c085e51f9015afc5d485`
4. `subject00_O03_masks_slots00_03_review.png` — O03 slots 00-03 six-column detail review — 7200×4400 — SHA256 `c29e0dd207448f0c9d04e95e67ca3ead102f93c1cc5a81b2d1aa3067a3e0bd4c`
5. `subject00_O03_masks_slots04_07_review.png` — O03 slots 04-07 six-column detail review — 7200×4400 — SHA256 `428b190e7d9b80da4eb785e0dd5608a78a4a7303d67afa3ed3838472d15dc562`
6. `subject00_O04_masks_slots00_03_review.png` — O04 slots 00-03 six-column detail review — 7200×4400 — SHA256 `ec37b801b7eacdc7f57ed96025032f0376d5257e586c926d40db9c5694553ef0`
7. `subject00_O04_masks_slots04_07_review.png` — O04 slots 04-07 six-column detail review — 7200×4400 — SHA256 `7cc22fb8b629c183aff8ec5225d2efee192c338199d5ea1d062253b66c0dfae3`
8. `subject00_masks_risk_and_limitations_review.png` — All 9 limitations and both registration overrides with local crops — 7200×6400 — SHA256 `3b6f4677611875e62ce330eb638a9168e4550c511a5b83103a28f176f892588a`

## Review checklist

- Person: head/hair/face/hands/feet/shoes, full foreground, background leakage.
- Garment: upper/lower garment, sleeves/trousers, protected skin/face/hair/hands/shoes.
- Pair: garment must remain inside person; protected region must remain non-empty.
- O01/O03/O04: check eight-view semantic consistency.
- Page 8: inspect all disclosed limitations and both registration overrides.

Record decisions outside this task; do not alter the PNG pages or formal masks.
