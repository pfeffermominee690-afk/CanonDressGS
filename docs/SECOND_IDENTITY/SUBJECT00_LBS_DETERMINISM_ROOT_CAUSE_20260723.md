# Subject00 LBS Determinism Root Cause — 2026-07-23

## Decision

The attempt_001 repeatability failure begins in the PointInterpolant raw numeric payload, not in parsing, normalization or NPZ serialization.

`NONDETERMINISM_CLASSIFICATION=MULTITHREAD_ONLY_NONDETERMINISM`

## Preserved source evidence

attempt_001 was snapshotted before repair work: 370 files, 4,171,424,719 bytes, content-tree SHA256 `1ba3a129cb35572edb8922524d8740e16c1cf46e4ef3553221ae707e99c0c8f9`. The source and final snapshots are required to match exactly.

The old wrapper recorded `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `PYTHONHASHSEED=0`, but the frozen `script/gen_weight_volume.py` passed `--threads 12` directly to every PointInterpolant invocation. Thus the actual solver thread count was 12. OPENBLAS/NUMEXPR/VECLIB/BLIS values were not persisted by attempt_001 and are reported as historically unrecoverable rather than guessed.

attempt_001 run_a/run_b gradient-input aggregate hashes are exact. Their raw solver payloads differ in 38/55 channels: max abs `8.3297901e-4`, mean abs `3.5914290e-6`. The parser reproduces that same difference. After clipping and normalization the difference becomes max `2.2351415e-4`, mean `1.1395205e-6`; the canonical float32 arrays reproduce the archived max `2.2351369e-4`, mean `1.1395203e-6`, argmax agreement `0.9999976158`. Reconstructed canonical arrays exactly match both archived grids. Therefore neither parser nor normalizer introduced the failure.

## Thread matrix

All valid matrix runs use the same frozen 110-file solver input pack, aggregate SHA256 `df0fabd1c34d2851ece81b400447a0e250f34c5694efc99806f3cdc0addd76e0`, the same template, bbox, model, PointInterpolant binary and scientific parameters.

| Pair | Raw | Parsed | Normalized | Canonical grid | NPZ | Result |
|---|---:|---:|---:|---:|---:|---|
| T1_A / T1_B | exact | exact | exact | exact | exact | PASS |
| T12_A / T12_B | different | different | different | different | different | FAIL |

T1 canonical grid SHA256 is `749599cb9380b8a9f81bac97d838528e939b60fb343fa4c08b75c9fe28dbf3c0`; deterministic NPZ SHA256 is `8c2c06b4e1154e361432f5bc115ecb875502e5bcdd1f6e3a5e2f459ecb9a0971`. T1 max/mean are zero and argmax agreement is 1.0. Mean wall time is about 313.39 seconds.

T12 canonical max is `6.8317214e-4`, mean `1.2601305e-6`, argmax agreement `0.9999928474`; only 18/55 raw payload channels are exact. Mean wall time is about 79.73 seconds.

The deterministic serializer has fixed key order, ZIP timestamp and compression settings. It produces identical bytes whenever the arrays are identical. `NPZ_CONTAINER_ONLY_DIFFERENCE` is therefore rejected.

## Matrix-input erratum

An initial T12_A diagnostic generated its gradient text under a 12-thread numeric-library environment. Those input bytes differed from T1, so the run was excluded before causal comparison and preserved at `attempt_002/attempt_history/T12_A_INPUT_CONFOUND`. It was not deleted or selected by result. The valid T12 pair uses the attempt_001/T1-verified common input pack. No threshold or scientific parameter changed.

## Root cause

The evidence supports `POINTINTERPOLANT_MULTITHREAD_REDUCTION_ORDER`: the 12-thread solver produces run-dependent floating-point reductions. Fixing the solver to one thread repairs repeatability with a roughly fourfold wall-time cost. This does not repair the separate geometry-fidelity failure.
