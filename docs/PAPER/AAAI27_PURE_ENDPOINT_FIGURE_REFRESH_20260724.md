# AAAI27 Pure Endpoint Figure Refresh

## Classification

`PURE_ENDPOINT_FIGURE_REFRESH_READY`

## Provenance Gate

- Figure Bank parent: `research/paper-figure-evidence-bank-20260724@06261aea35d26006db250df65707d1a09a0bf098`.
- Pure Endpoint result: `research/pure-endpoint-core-method-crossfit-amended-20260724@ce110887a942cf8db082ba688c8d36d2433bfdbe`.
- Execution HEAD: `195fb887f2cac8a72920b499c44bb66700a97e25`.
- Attempt: `PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001/attempt_001`.
- Source status: `SEALED`; classification: `PURE_ENDPOINT_CORE_METHOD_SUPPORTED`.
- Local, origin, and cloud source result HEADs matched before refresh.
- Source worktree was clean; `attempt_002` and active Pure Endpoint processes were absent.

## Integrity Counts

The sealed execution-count verifier reports 7 formal methods, 2 independently
trainable families, 24 training runs, 7,200 optimizer steps, 144 checkpoints,
4,620 logical renders, 1,460 physical renders, 3,160 cache reuses, 60 main
sheets, and 60 formal-pure sheets. Every expected/actual delta is zero.

The refresh inventories 3,046 visual records: 2,920 RGB/alpha render records,
120 visual sheets, and 6 overviews. These resolve to 492 unique SHA values and
2,554 duplicate records. SHA-addressed reuse avoids 652,123,884 duplicate bytes.

## Figure Outputs

- Figure 1: 60-sheet `FIGURE1_ENDPOINT_ONLY_TEASER_CANDIDATE_V1`, manual adjudication required.
- Figure 5: four-panel `FIGURE5_HARD_LOOKUP_RELATION_CANDIDATE_V1`.
- Figure 6: `FIGURE6_PURE_ENDPOINT_PERTURBATION_LIMITATION_V1`.
- Supplementary: `SUPPLEMENTARY_PURE_ENDPOINT_PARITY_AND_SAFETY_V1`.
- Metric plots: 8 groups, each with PDF, SVG, PNG, and source JSON.
- Contact sheets: 8, all marked review-only and not final paper figures.

## Scientific Boundary

Clean evaluated top-1 is 1.0. Canon endpoint exact match is 1.0 with realized
endpoint MAE/RMSE 0. Linear continuous realization has endpoint exact match 0,
raw MAE `71.44514598846436`, and raw RMSE `83.44418888092041`. Registered
mild blur has 39/60 Canon flips; single reference has 3/60; complete dropout
safely abstains in 80/80 records. Hard-lookup relations remain descriptive.

## Execution Boundary

This refresh used CPU/low-I/O registry reads, source-image decoding,
deterministic Pillow composition, and deterministic matplotlib plotting only.
Optimizer creation/steps, forward/backward, model inference, renderer runs,
new scientific renders, checkpoint writes, image APIs, AI-generated figures,
retouching, Headroom files consumed, AvatarReX media exports, and PAPER_FINAL
are all zero.

## Pending State

Figure 2 remains `METHOD_FREEZE_PENDING_HEADROOM_AND_LOO`. The Headroom attempt
was absent at refresh preflight and existed at final observation, with no
matching process; it remains unadjudicated and active-output-excluded with
`HEADROOM_ACTIVE_OUTPUT_FILES_READ=0`. LOO uses
the repaired source `research/loo-few-view-fold-manifest-repair-20260724@2c7c748026307e82c88b7f96bf2dc41a79ba7b6f`
and remains results-pending; no K=4 formal panel is retained. Figure 3 and
Figure 4 remain ready from historical evidence.

## Next Task

`WAIT_FOR_SEALED_HEADROOM_RESULT`
