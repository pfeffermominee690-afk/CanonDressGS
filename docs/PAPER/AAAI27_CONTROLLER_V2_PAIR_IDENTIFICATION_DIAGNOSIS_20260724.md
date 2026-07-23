# Controller V2 Pair Identification Diagnosis

Status: `PASS`

Historical result remains: `CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL`.

Primary diagnosis: `SOFTMAX_PAIR_PARAMETERIZATION_FAILURE`.
Representation: `REFERENCE_REPRESENTATION_RECOVERABLE`.

## Result

Formal V2 pair macro was `0.601042` versus matched V1 `0.631250`. The best legal probe was selected on calibration, not test: `CURRENT_SET_DIRECT_PAIR_10WAY` with held-out test macro `0.987500`.

The split is condition-fold held-out but has exact reference-asset overlap. It does not support unseen-reference generalization. The historical V1 96.25% is only a **closed-wardrobe protocol-fit result**, not strict cross-fold generalization.

No Controller/F2 training, optimizer step, threshold change, compatibility change, renderer run, new formal render, or PAPER_FINAL artifact occurred.

## Decision

Secondary factors: `none`.

NEXT_TASK: `DESIGN_DIRECT_PAIR_HEAD_CONTROLLER_V3`. It was not started.
