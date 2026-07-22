# Subject00 Derived-Asset Runtime Smoke — 2026-07-23

## Status

`NOT_RUN_PRECONDITION_FAILED`.

The protocol permits dataset/model/renderer smoke only after template determinism, template compatibility, LBS determinism, LBS geometry and atomic publication all pass. LBS repeatability and geometry failed before publication, so no validated formal template/LBS pair existed for runtime smoke.

Accordingly:

- dataset smoke: not run in this stage;
- subject00 model construction: not run;
- 200,000-Gaussian initialization: not run;
- model forward/renderer/depth: not run;
- zero-step loss: not run;
- optimizer/backward/scheduler/checkpoint: 0/0/0/0.

The GPU initially hosted two formal processes and was left untouched. It later became available during CPU LBS work, but availability did not waive the failed asset precondition. This is not `GPU_SMOKE_DEFERRED`; it is a scientific/engineering LBS gate failure.

Machine record: `paper_protocol/second_identity/subject00_derived_asset_runtime_smoke.json`.
