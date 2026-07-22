# CanonDressGS Project Control Center

This directory is the unique project-management source of truth for CanonDressGS / AAAI-27. Its sole writer is the worktree `E:\model_train\canondressgs_project_control` on `project/canondressgs-control-center-20260722`.

Experiment, paper, figure, and data tasks must not edit these main status files. They publish their own formal report, summary JSON, and final handoff; the control center incorporates those outputs by scanning Git metadata and reports.

## Files

- [PROJECT_STATUS.md](PROJECT_STATUS.md): current source of truth, gates, blockers, next tasks, and risks.
- [WORKTREE_REGISTRY.yaml](WORKTREE_REGISTRY.yaml): all Git worktrees plus separately classified non-Git directories.
- [MILESTONE_REGISTRY.yaml](MILESTONE_REGISTRY.yaml): required canonical/historical milestone verification.
- [ARTIFACT_REGISTRY.yaml](ARTIFACT_REGISTRY.yaml): formal output roots, counts, fingerprints, and copy/freeze state.
- [DATASET_REGISTRY.yaml](DATASET_REGISTRY.yaml): identity, reference/target, teacher-bank, and basis-asset state.
- [TASK_QUEUE.md](TASK_QUEUE.md): active/blocked/queued work and dependency order.
- [DECISION_LOG.md](DECISION_LOG.md): durable governance and evidence decisions.
- [CLAIM_STATUS.md](CLAIM_STATUS.md): confirmed, partial, unconfirmed, and failed/limited claims.
- [templates/TASK_HANDOFF_TEMPLATE.json](templates/TASK_HANDOFF_TEMPLATE.json): producer handoff contract.

The `.yaml` registries intentionally use JSON syntax. JSON is a strict subset of YAML 1.2 and can be parsed with the Python standard library, avoiding an undeclared PyYAML dependency.

## Refresh and validation

Run from the control-center worktree:

```powershell
python project_control/scripts/refresh_project_control.py --verify-remotes
python project_control/scripts/validate_project_control.py
git diff --check
```

`refresh_project_control.py` performs only Git/file metadata reads and rewrites `WORKTREE_REGISTRY.yaml`. It never trains, evaluates, renders, deletes a directory, removes a worktree, deletes a branch, or mutates an artifact.

The registry's control-center row is a pre-commit snapshot. A Git commit hash depends on the tree containing the registry, so the registry cannot self-contain that same commit hash. Validation permits only this row to lag as an ancestor; all other branch/HEAD rows must match exactly. The final handoff reports the actual committed/pushed control-center HEAD.

## Cleanup policy

1. Never manually delete a Git worktree in File Explorer.
2. A worktree can become a cleanup candidate only when `status_class=SUPERSEDED` and all of the following are true:
   - push status is `PASS`;
   - origin HEAD is verified;
   - local clean state is `true`;
   - a formal report exists;
   - artifacts are independently registered.
3. Approved cleanup must use `git worktree remove`; manual directory deletion is forbidden.
4. Branch deletion is a separate decision after worktree removal. It is never implied by cleanup candidacy.
5. `SEALED_CANONICAL`, `FAILED_PRESERVED`, `ACTIVE`, and `DOCUMENTATION_ACTIVE` worktrees must not be deleted.
6. `safe_to_remove` defaults to `false` for every entry. Changing it requires a separate reviewed control-center decision.

The current registry has zero cleanup candidates. No worktree was deleted, moved, or renamed while building this control center.

## Cloud semantics

An origin/cloud HEAD field is not a generic synchronization claim. `origin` is verified live during each remote refresh. The `cloud` host did not resolve during refresh `CANONDRESSGS-PROJECT-CONTROL-REFRESH-002`, so cloud values are cached refs or report-derived paths and are labeled accordingly. `CLOUD_MIRROR_NOT_VERIFIED_DNS_UNREACHABLE` is not a sync PASS; `clean_cloud`, live artifact presence, and controller-branch parity remain unknown until successful read-only verification.

## Handoff ingestion

Each producing task should fill `TASK_HANDOFF_TEMPLATE.json`, commit its report/summary on its own branch, and return a short handoff containing branch, exact HEAD, clean state, output root, fingerprints, attempt history, scientific boundary, and next task. The control center, not the producer task, edits the global registries.
