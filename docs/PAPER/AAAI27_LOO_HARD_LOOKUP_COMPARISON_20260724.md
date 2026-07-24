# AAAI27 LOO Hard-Lookup Comparison

## Runtime Status

The runtime hard-lookup phase was not reached because the attempt failed at the first basis renderer-parity gate. Consequently, no deployable K=1 or K=2 hard-lookup inference, no low-dimensional adaptation comparison, and no per-garment or per-rotation metric exists. These values must not be inferred from preflight artifacts.

## Frozen Preflight Replay

The repaired preflight replay remained valid before materialization: 15 garment-rotation pairs selected the same endpoint for K=1 and K=2, while five selected different endpoints. The five differences were `O01-R0: O04 -> O02`, `O01-R1: O04 -> O02`, `O02-R2: O08 -> O01`, `O04-R0: O03 -> O01`, and `O08-R3: O01 -> O02`. Held-out Teacher reads, test-metric reads, and optimizer reads were all zero during this replay.

This replay established cache-key and endpoint-selection consistency only. It is not a substitute for the frozen runtime lookup registry and cannot populate LPIPS, RGB MAE, IoU, boundary, win/tie/loss, or gate fields.

## Conclusion

The primary K=2 low-dimensional adaptation versus K=2 hard-lookup comparison is unavailable. The hard-lookup outperformance gate is `null`, not false, because evaluation never ran. The overall execution classification is `LOO_ADAPTATION_EXECUTION_INVALID`; no scientific hard-lookup conclusion is claimed.
