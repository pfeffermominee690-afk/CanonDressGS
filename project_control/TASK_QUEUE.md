# CanonDressGS Task Queue

The control-center worktree is the only writer. This refresh did not start any queued task.

## Active

| Priority | Task | Branch / location | Stage | Blocker | Completion signal |
|---|---|---|---|---|---|
| P0 | `COMPLETE_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER_DESIGN` | `research/reference-conditioned-dual-support-controller-20260722` at `174655ce6aabcae5d60ce45f3b4be319eb71e914` | seven classifications PASS; `29 passed` focused and `70 passed` broader compatibility tests; origin matches; clean | live `canondress-cloud` DNS failure prevents local/origin/cloud HEAD equality verification | live cloud branch equals local/origin, all strict gates remain true, then seal `REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER_DESIGN_READY` |
| P1 | `AAAI27-CANONDRESSGS-TEASER-REAL-ASSETS-001` | `paper/aaai27-real-teaser-figure-20260722` | deterministic PNG/PDF built and source-audited | author signoff | visual record leaves `MANUAL_REVIEW_REQUIRED` without changing formal source pixels |

## Queued by dependency

| Order | Task | Depends on | Gate |
|---|---|---|---|
| 1 | `TRAIN_AND_EVALUATE_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER` | sealed controller design and explicit authorization | controller design strict archive gate closes; training is not auto-started |
| 2 | `RUN_SECOND_IDENTITY_MMLPHUMAN_PREFLIGHT_WITHOUT_TRAINING` | subject00 raw data ready | branch, deterministic data/template/loader checks, formal readiness summary; no training |
| 3 | second-identity avatar training | subject00 preflight readiness | only if the preflight explicitly returns a training-ready classification and authorization is given |
| 4 | manuscript method/experiment refresh | sealed dual-support evidence and controller state | preserve claim boundaries; do not create `PAPER_FINAL` |
| 5 | strict novel-view/pose canary | final controller plus a ready second identity/avatar and garment benchmark | separately preregistered held-view/held-pose protocol |
| 6 | paper result freeze | all required experiments | independent evidence review and explicit result-freeze decision |

## Next Three Tasks

1. Restore read-only cloud reachability and complete the controller-design seal check.
2. After that seal and explicit authorization, train and evaluate the reference-conditioned dual-support controller.
3. Run the subject00 MMLP-Human preflight without training and publish its real readiness classification.

## Blocked / held

- Controller formal training/evaluation is held behind the incomplete controller-design seal and explicit authorization.
- subject00 avatar training is held because no MMLP-Human preflight branch/report exists; raw-data readiness alone is insufficient.
- Final abstract/results are held behind formal controller results.
- Strict novel-view/pose work is held behind a trained final controller, second-identity avatar, and garment benchmark.
- Cleanup is held because no worktree satisfies every cleanup gate and live cloud verification is unavailable.

## Preserved history

- The midpoint-degenerate root-cause archive remains `SEALED_HISTORICAL`; it does not override `GEOMETRY_MAIN_EFFECT` from the sealed causal attribution.
- Failed attempts remain append-only evidence and are not cleanup candidates.
- Cleanup candidates: `0`; no worktree, branch, or artifact was deleted.
