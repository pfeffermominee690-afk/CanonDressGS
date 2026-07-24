# AAAI27 LOO Full-Residual Comparison

## Contract Preservation

The frozen full-residual schema remained `200,000 x 22 = 4,400,000` trainable scalars, with Adam, learning rate `0.02`, 300 steps, seed `20260724`, and checkpoints at `0/20/50/100/200/300`. No contract value was changed after the execution HEAD was frozen.

## Execution Status

The full-residual phase was not reached. Expected versus actual full-residual runs were `40/0`; optimizer creations were `120/0` across all method families; optimizer steps, forward calls, and backward calls were `36,000/0`; checkpoint writes were `720/0`.

Equal-step and equal-wall-time results are unavailable. LPIPS and RGB MAE recovery ratios are `null` with reason `HARD_LOW_FULL_METRICS_NOT_AVAILABLE`. This differs from a nonpositive denominator: the denominator itself was never observed, so no ratio or pass/fail value may be computed.

## Conclusion

The `>=0.75` recovery gate is not evaluated. Full residual provides neither recovery evidence nor a capacity/optimization diagnosis in this attempt. The failure occurred earlier at basis renderer parity and is classified `LOO_ADAPTATION_EXECUTION_INVALID`.
