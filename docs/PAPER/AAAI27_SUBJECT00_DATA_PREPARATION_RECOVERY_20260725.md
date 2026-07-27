# Subject00 Data Preparation Recovery Audit

Task: `AAAI27-SUBJECT00-DATA-PREPARATION-GENERATION-READY-001`

## Recovery gate

The exact source is `research/subject00-minimal-second-identity-dataset-contract-20260725` at `9da04d923c7c6c03c6a6f7c732d8d9c58d59c88d`. The Git commit, Task B final summary, and Task B handoff agree on `SUBJECT00_CONDITION_RESOURCE_GAP`, `SECOND_IDENTITY_REPLICATION`, three garments, eight slots per garment, 12 planned references, 24 planned Teacher targets, 48 planned candidates, zero API calls, zero materialized images, zero endpoint/controller runs, and `PAPER_FINAL=false`.

## Current dependencies

The Storage branch `research/mmlphuman-subject00-storage-migration-adjudication-20260725` at `34e91445ef45c001cebcd789a4a6a88cad7b9ad8` reports `SUBJECT00_STORAGE_MIGRATION_PLAN_READY` with the live storage gate still blocked. The actual Formal Base artifacts contain no attempt, no `EXECUTION_HEAD`, no sealed checkpoint, and no sealed downstream manifest. `FORMAL_BASE_DEPENDENCY=PENDING` remains mandatory.

## Recovered assets

The cloud raw root contains the frozen 24-camera/2500-frame Subject00 capture, official masks, `calibration.json`, and `smpl_params.npz`. The published Subject00 template and surface-LBS support files are present. Historical canary and medium-pilot visuals are preview-only and are forbidden as formal generation inputs. No Formal Base condition-render root exists.

The selected identity source is pose 0 across eight strict-train cameras. The source images were opened during this audit and are also covered by prior canary/medium-pilot original-detail review. Full-body completeness, hands, feet, pose, camera consistency, and official masks pass. The raw hood partially occludes hairstyle evidence; the contract preserves only the visible hairline and forbids hairstyle invention.

## Mutation boundary

No Task B, Formal Base, Storage, raw, derived, or historical output asset was modified. No renderer, training, image/VLM API, or endpoint/controller execution occurred. Large assets remain outside Git.

`PAPER_FINAL=false`.
