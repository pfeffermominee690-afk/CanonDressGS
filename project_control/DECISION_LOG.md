# CanonDressGS Decision Log

## 2026-07-22 — Create a unique control center

- Decision: create `project/canondressgs-control-center-20260722` from sealed HEAD `371d812864614cd561e33edfe3f6c043b38415d2` in `E:\model_train\canondressgs_project_control`.
- Scope: only `project_control/` may be changed in this worktree.
- Rationale: a single writer prevents concurrent experiment/paper/data branches from editing the same project-status documents.

## 2026-07-22 — Treat Git metadata as authoritative for worktrees

- Decision: `git worktree list --porcelain` defines the Git-worktree inventory. Name-matched directories absent from it are recorded separately as non-Git directories.
- Observation: 29 Git worktrees and 24 non-Git directories were discovered after creating the control-center worktree.
- Consequence: no branch or worktree identity is inferred from a folder name alone.

## 2026-07-22 — Preserve expected/observed HEAD comparison

- Decision: retain both expected and observed values for the seven required milestones.
- Observation: all seven observed local HEADs match the task-provided expected HEADs.
- Important remote difference: P0 formal candidate local/origin HEAD is `32e6058…`, while the cached cloud ref is `9adc6b9…`. This difference is reported, not silently normalized.

## 2026-07-22 — Demote the previous causal interpretation to historical

- Decision: classify `research/continuous-control-artifact-root-cause-20260722` at `68623c36…` as `SEALED_HISTORICAL`.
- Evidence: the addendum states that at alpha 0.5 selected and non-selected channels all become the same six-channel midpoint; 10/10 reproduction is therefore non-identifying.
- Boundary: the engineering execution is preserved, but channel attribution with midpoint degeneracy is not a final causal conclusion.

## 2026-07-22 — Current causal attribution remains active

- Decision: classify `research/continuous-control-causal-attribution-20260722` at `67a8b08…` as the single `ACTIVE` mainline research task.
- Attempt state: `attempt_001` is preserved as a zero-result alpha-grid asset mismatch; the next append-only attempt is `attempt_002`.
- Boundary: no local causal final summary exists, so no micro-pilot or final method is selected by this control-center task.

## 2026-07-22 — Teaser remains documentation-active

- Decision: classify `paper/aaai27-real-teaser-figure-20260722` at `1f77b4e…` as `DOCUMENTATION_ACTIVE`.
- Evidence: formal source assets were hash-stable and focused checks passed; visual review remains `MANUAL_REVIEW_REQUIRED` with author signoff required.
- Boundary: `PAPER_FINAL=0`; the figure does not establish novel view, novel pose, interpolation, or FPS claims.

## 2026-07-22 — Cloud verification is explicitly incomplete

- Decision: record cached cloud refs and report-derived cloud paths separately from live verification.
- Observation: `git ls-remote origin` passed; `git ls-remote cloud` failed because host `canondress-cloud` could not be resolved.
- Boundary: `clean_cloud`, live cloud artifact presence, and cloud upload are not claimed.

## 2026-07-22 — Cleanup defaults to prohibited

- Decision: set `safe_to_remove=false` for every worktree.
- Observation: one branch is classified `SUPERSEDED`, but it lacks a verified origin/upstream and therefore fails the cleanup gate.
- Consequence: there are zero current cleanup candidates and no directory was removed, moved, or renamed.

## 2026-07-22 — Registry snapshot semantics

- Decision: the control-center row records the HEAD visible when the registry is generated.
- Rationale: a tracked registry cannot contain the hash of the commit that contains itself; committing the refreshed registry necessarily advances the branch.
- Validation: an ancestor-only self-row lag is a documented warning. Every non-control worktree remains an exact branch/HEAD check.

## 2026-07-23 — Refresh after dual-support controller design

The following decision chain is append-only and supersedes earlier current-state descriptions without deleting their historical record:

1. The former midpoint channel attribution was found degenerate at alpha=`0.5`; it remains a `SEALED_HISTORICAL` engineering record.
2. The mainline adopted endpoint-anchored `2^3` factorial causal attribution.
3. The sealed causal conclusion is `GEOMETRY_MAIN_EFFECT` at `8c43524b7ca0ee8b3c795dbe349466f30b29354f`.
4. Appearance-only and visibility-only remedies are excluded as the mainline response by the formal causal evidence.
5. `Dual-Support Geometry Blend` was proposed to avoid direct geometry interpolation while preserving endpoint behavior.
6. The geometry dual-support micro-pilot passed at `b216acd02323c822a7aca12f424efed6ba2e0a81`.
7. The all-10-pair dual-support evaluation passed at `d802f427f1e9c23595e1bdb4f10135f7de2c3f08`, within the closed seen-garment wardrobe boundary.
8. The mainline decision advanced to a reference-conditioned controller.
9. The controller predicts a five-class garment endpoint distribution, stable top-2 endpoints, mixture weights, and a single-endpoint fallback; it does not predict the former four-dimensional basis coefficients.
10. subject00 raw data is ready in the cloud, but second-identity training remains prohibited until an MMLP-Human preflight establishes template/LBS/loader readiness.

## 2026-07-23 — Controller design evidence passes, archive seal remains open

- Evidence: `dual_support_controller_dry_run_summary.json` reports PASS for `CONTROLLER_ADAPTER`, `SOFT_TARGET_DATASET`, `TOP2_SELECTION`, `SINGLE_ENDPOINT_FALLBACK`, `DUAL_SUPPORT_RUNTIME_INTERFACE`, `FORWARD_BOUNDARY`, and `NO_FORMAL_TRAINING_GATE`.
- Git state: local and live origin equal `174655ce6aabcae5d60ce45f3b4be319eb71e914`; the worktree is clean.
- Conservative classification: keep the controller worktree `ACTIVE`, because live cloud HEAD equality is a required seal gate and `canondress-cloud` DNS is unreachable. Do not infer cloud PASS from an absent cached ref.
- Next task while this gate is open: `COMPLETE_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER_DESIGN`.
- Conditional successor after sealing and explicit authorization: `TRAIN_AND_EVALUATE_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER`.

## 2026-07-23 — subject00 raw data ready is not model readiness

- Decision: replace `DOWNLOADED_LOCAL_PENDING_VERIFICATION` with `SUBJECT00_CLOUD_DATA_READY` for the raw dataset.
- Evidence: 119,412 files, 26,459,647,641 bytes, 24 cameras, 2,500 frames, matching image/mask missing-entry sets of 296, complete decode/calibration/SMPL audits, and tree fingerprint `2c0f894f70d944fa8d6ebd48ab1188b78cb19b673f0b7c006f1922a592b1ea7b`.
- Boundary: no `research/mmlphuman-subject00-preflight-20260722` branch, report, summary, or handoff was found. MMLP-Human status is therefore `QUEUED_NOT_STARTED`, not ready.

## 2026-07-23 — Cloud and cleanup remain conservative

- Cloud: record `CLOUD_MIRROR_NOT_VERIFIED_DNS_UNREACHABLE`; cached refs are evidence only, not a live sync claim.
- Unknown worktrees: retain three `UNKNOWN_REQUIRES_REVIEW` rows after rechecking Git common-dir, branch, HEAD, reports, remotes, and artifact registration; no unsupported PASS was assigned.
- Cleanup: zero candidates satisfy every gate. No worktree, branch, directory, or large artifact was removed or moved.
