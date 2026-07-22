# CanonDressGS Claim Status

This file records the claim boundary as of 2026-07-22. It is not a paper-final declaration.

## CONFIRMED

- Discrete garment switching.
- Ours-v2 deterministic endpoint performance.
- Hard lookup reaches the teacher endpoint.
- Linear fusion stability.
- SmoothL1 supervision advantage.

## PARTIAL

- Reference soft control: intermediate appearances exist, but only 3/10 pairs satisfy the frozen stable non-endpoint rule and visible artifacts remain.
- Numerical basis continuity: coefficient/basis interpolation is numerically complete with endpoint parity, but visual continuity is not reliable.

## NOT_CONFIRMED

- Visually reliable continuous control.
- Novel view.
- Novel pose.
- Novel pose + novel view.
- Cross-identity generalization.
- Unseen garment generation.

## FAILED / LIMITATION

- M3/M4: engineering execution completed, but scientific/visual behavior failed with collapse and severe artifacts.
- O07 held-out basis coverage: held-out projection/reference prediction is not supported.
- Current continuous interpolation artifacts: patching, mottle, cloud, edge scatter, and silhouette discontinuity remain visible.

## Use rule

Only a new sealed report and summary JSON may move a claim between sections. Fixed-view, subject02, seen-outfit evidence must not be generalized to new identities, poses, views, or garments. `PAPER_FINAL=0` remains in force.
