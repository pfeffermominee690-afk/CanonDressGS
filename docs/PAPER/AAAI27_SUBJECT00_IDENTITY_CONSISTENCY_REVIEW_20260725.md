# AAAI-27 Subject00 Identity Consistency Review

Task: `AAAI27-SUBJECT00-MINIMAL-SECOND-IDENTITY-DATASET-CONTRACT-001`

## Hard identity invariants

Every generated candidate and accepted endpoint must preserve Subject00 face identity, hair identity, skin tone, body shape, height/proportions, pose skeleton and semantics, camera perspective and position, hand/foot anatomy, background convention, and lighting convention.

A garment edit must not introduce age or sex-characteristic changes, face-shape or hairstyle changes, body-mass changes, limb-count changes, pose-semantic changes, camera drift, copied donor identity/body pixels, cropping, an extra person, text, numbering, logo, watermark, collage, or border.

Garment semantics do not override identity safety. A visually accurate O01/O03/O04 candidate with identity, anatomy, pose, camera, mask, or edge failure is rejected.

## Review record

Each append-only review event records:

- Candidate ID/SHA, identity, garment, condition, reviewer, stage, and timestamp.
- `identity_match`, `face_match`, `hair_match`, `body_shape_match`, `pose_match`, and `camera_match`.
- `garment_match`, `mask_quality`, and `edge_quality`.
- Artifact grade 0 through 3.
- `ACCEPT`, `REJECT`, or `MAYBE` plus a controlled reject reason and free-text comment.
- Previous-event SHA-256 for hash-chain continuity.

The machine-readable schema is `paper_protocol/reviewer_risk/subject00_identity_consistency_review_schema.json`; the blank event template is `subject00_human_review_manifest_template.json`.

## Decision workflow

Automatic checks may reject or warn but cannot accept. Reviewer A and Reviewer B work independently and cannot inspect one another's initial decision. Two `ACCEPT` votes are required. Any initial `REJECT` resolves to reject. Any other complete pair requires a third independent human adjudicator; a VLM cannot break the tie.

Candidate review is necessary but not sufficient. Each garment must also pass an eight-slot group review for cross-view identity, garment color/material, sleeve/hem semantics, left/right and front/back consistency, and absence of random accessories. Rejected candidates, disagreement, adjudication, and sibling attempts remain retained.

No review occurred in this task because no candidate was generated. Review events, accepted endpoints, and candidate-pixel mutations are all zero.
