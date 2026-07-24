# AAAI27 LOO Basis Adaptation Results

## Status

The first cache-repaired attempt is scientifically invalid. Execution stopped in the first basis-construction split with `LOO_BASIS_CONSTRUCTION_PARITY_FAILURE`, before hard lookup, regularization calibration, oracle diagnostics, optimization, evaluation, or visual review. The formal classification is `LOO_ADAPTATION_EXECUTION_INVALID`.

The source HEAD is `d595264dd0bc9f8b19077955e9ba605f0d25fdd5`; the frozen execution HEAD is `a868df3c3811483c5dd07456d8cabfdc3577d5f4`; the result HEAD is `8436a6187da02ca3c21ef546f0940208a02fdac3`. `attempt_001` is preserved and `attempt_002` was not created.

## Observed Failure

For `LOO-O01`, the four training garments were `O02/O03/O04/O08`. The selected rank was 3 and the singular values were `879.335121`, `806.740809`, `697.199460`, and `0.0000519599`. Residual reconstruction parity passed: the largest normalized reconstruction RMSE was `2.6056503e-7`, and the repeated basis fingerprint matched.

Renderer parity failed for all four reconstructed basis garments under the frozen `1e-5` maximum absolute error threshold. The largest observed RGB error was `0.0029371381`; the largest alpha error was `0.0038456917`. No result-based threshold change was made.

## Result Boundary

There are no valid per-garment, per-rotation, LPIPS, RGB MAE, PSNR, SSIM, silhouette, boundary, protected-region, identity, recovery, or view-scaling results. All such fields are reported as `null` with explicit phase-not-reached reasons. This attempt supports no method claim. `PAPER_FINAL=false` and `paper_final_count=0`.
