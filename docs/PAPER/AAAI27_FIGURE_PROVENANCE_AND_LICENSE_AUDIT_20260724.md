# AAAI27 Figure Provenance And License Audit

Task: `AAAI27-PAPER-FIGURE-EVIDENCE-BANK-001`

## Provenance contract

The full external registry records source category, absolute/relative path,
branch, HEAD, output root, task, attempt, experiment, method/ablation, identity,
garment/pair, references, pose, camera, split, rotation, seed, checkpoint and
renderer fields, original SHA-256, dimensions/channels, transforms, output
SHA-256, claim status, license status, eligibility, and notes.

Original source media was never modified. A path-token parser repair changed
12,263 garment fields after visual QA; it read zero media files and preserved the
v1 registry. The v2 registry is the active provenance source.

Eleven assets lack a unique attempt and are exactly classified
`UNKNOWN_PROVENANCE_DO_NOT_USE`. They do not block the bank because they are not
eligible candidates and the provenance-complete source assets remain indexed.

## Transform contract

- 9 review contact-sheet transforms use deterministic containment, white
  background, equal tile geometry, and no crop.
- 8 metric transforms use structured JSON and deterministic matplotlib output.
- Each metric plot has PNG, SVG, PDF, and source JSON with matching hashes.
- AI-generated figures: 0; retouched figures: 0; asymmetric crops: 0;
  method-specific enhancements: 0.

## License audit

AvatarReX is a metadata-only `LICENSE_RESTRICTED_DO_NOT_EXPORT` slot. Exported
AvatarReX RGB, masks, derived images, templates, LBS data, or renders: 0. Raw
dataset copies: 0. The repo and external cache contain no AvatarReX media.

Credential and personal-data scans are mandatory final checks. No API key,
checkpoint, raw dataset, proprietary model, or unlicensed third-party image is
an allowed committed artifact.

`PAPER_FINAL=0`.
