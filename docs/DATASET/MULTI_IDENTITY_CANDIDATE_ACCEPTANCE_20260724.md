# Multi-Identity Candidate Acceptance Contract

Task: `AAAI27-MULTI-IDENTITY-GENERATION-BACKEND-001`

## Naming and manifests

Candidate names are `{identity_id}__{garment_id}__{semantic_pose_slot}__candidate{candidate_index:02d}__{request_id}.png`. A neighboring sealed manifest links source/input hashes, prompt hashes, backend parameters, request events, automatic screening records, human review records, and the candidate SHA-256. Bare sequence numbers are forbidden.

Accepted endpoint records reference the original candidate path and SHA-256. Copying an accepted image into an untraceable location is forbidden. Review history and rejected siblings remain append-only even after one candidate is accepted.

## Formal acceptance

For each identity x garment, all eight endpoint slots must be accepted and all four reference slots (front, left, back, right) must be among them. The eight accepted records must have eight distinct candidate SHA-256 values; a repeated image cannot fill a missing view. Group review must PASS. Identity contamination, severe anatomy failures, and text/watermark counts must each be zero, and every slot must pass full-body completeness.

A candidate cannot reach ACCEPT from automatic screening alone. It must satisfy the independent Reviewer A/B rule or a valid third-person adjudication. A garment cannot reach formal acceptance until the candidate decisions and eight-slot group decision are both resolved.

If any slot lacks a passing candidate, the garment status is `GARMENT_SLOT_GENERATION_INCOMPLETE`. Processing stops at the frozen N. Additional generation requires user authorization and cannot be silently allocated to the identity with the higher observed failure rate.

## State transitions

The manifest lifecycle is `CANDIDATE_SEALED`, automatic rejection or human-review-required, initial-pair resolution or adjudication, group resolution, then accepted-reference or rejected. Every transition is a new event. No prior event or candidate file is edited in place.

This task contains zero generated, candidate, accepted, or rejected images. `PAPER_FINAL=0`.
