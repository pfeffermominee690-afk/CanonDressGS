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
