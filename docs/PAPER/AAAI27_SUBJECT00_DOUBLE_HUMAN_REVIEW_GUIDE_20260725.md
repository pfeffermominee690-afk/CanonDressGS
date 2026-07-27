# Subject00 Double Human Review Guide

Use the four labeled contact sheets under `E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001\attempt_001\05_human_review\resolution_adjudication` for orientation, then inspect each original candidate at its registered path.

Reviewer 1 and Reviewer 2 must work independently. All visual fields and decisions start as null. Review identity, face, hair, skin tone, body shape, pose, camera, garment semantics, slot match, complete hands and feet, full-body framing, background, lighting, edges, and artifacts.

Two ACCEPT decisions produce `VISUAL_ACCEPT`; any REJECT produces `VISUAL_REJECT`; all other combinations require human adjudication. Dataset eligibility additionally requires `TECHNICAL_DATASET_ELIGIBILITY=PASS`. A category B image, if one exists, remains `VISUAL_ACCEPT_PENDING_RESOLUTION_CONTRACT` until the user explicitly authorizes a resize contract.

Do not copy any candidate into `06_accepted_rgb` or `09_teacher_targets` during this review preparation task.
