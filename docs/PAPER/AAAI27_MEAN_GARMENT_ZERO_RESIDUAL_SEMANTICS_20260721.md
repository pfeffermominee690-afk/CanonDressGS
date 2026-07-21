# Mean-Garment and Zero-Residual Semantics

Date: 2026-07-21  
Task: `AAAI27-DETERMINISTIC-INITIALIZATION-PROTOCOL-001`

## Adjudication

The frozen coefficient normalization, rank-4 basis, evaluator implementation, and real frozen CUDA tensors establish the following meanings:

| Object | Exact meaning |
| --- | --- |
| Standardized coefficient zero, `z_std = [0,0,0,0]` | `MEAN_COEFFICIENT`. De-standardization applies `c_raw = z_std * train_std + train_mean`, so the result is the training coefficient mean, not raw zero. |
| Raw coefficient zero, `c_raw = [0,0,0,0]` | `MEAN_GARMENT_RESIDUAL`. In the explicit-basis parameterization, zero coefficients reproduce the stored basis mean residual bitwise. |
| Basis reconstruction at raw coefficient zero | The basis mean residual, not a physical zero residual. |
| Physical Gaussian residual zero | `BASE_AVATAR`: all six residual channels are exactly zero and no clothing residual is applied. |

Therefore an Ours-v2 zero-initialized output must be described as a **MEAN-GARMENT INITIAL PREDICTION**. The phrase **ZERO-RESIDUAL INITIAL PREDICTION** is incorrect for this endpoint.

## Frozen normalization

The real normalization values are:

- `train_mean = [-1.220703143189894e-05, -3.051757857974735e-06, 2.441406286379788e-05, -6.103515625e-05]`
- `train_std = [412.4079895019531, 364.3363037109375, 313.2566223144531, 297.9817810058594]`

Thus standardized zero maps exactly to `train_mean`. The standardized-zero endpoint has SHA256 `a4a413d2be70aa355a5a588ce4f948f522daf98547e9f61eee926317683e20b4`.

## Frozen tensor evidence

All residual objects use CUDA float32 tensors over 200,000 Gaussians with shapes:

- `delta_xyz`, `delta_log_scaling`, `delta_rotvec`: `[200000, 3]`
- `delta_opacity_logit`: `[200000]`
- `delta_sh0`: `[200000, 1, 3]`
- `delta_shN`: `[200000, 3, 3]`

Raw coefficient zero reconstructs residual SHA256 `618b965c09cdb86e99b8932915fca4ef78a16531cb3ce6fd2f13e3968235aef2`. It is bitwise equal to the basis mean residual. Its difference from physical zero has L1 `116054.95505340592`, L2 `153.3511138811272`, and Linf `1.6304153203964233`, so these objects cannot be conflated.

The physical zero residual has SHA256 `ce6fa8b65ef053e3e0e52c9316f69ce24725d2a84caed30d223a7ceaeb610311` and L1/L2/Linf norms all equal to zero.

The five-teacher residual mean has SHA256 `77eddfdee1b324768f571e04ddc19196ef7bdb85eeb29e957f31f0aa6ecebfe5`. It is numerically very close to, but not bitwise identical to, the stored basis mean residual; the difference has L1 `0.002697700288071092`, L2 `5.304314010471153e-06`, and Linf `1.7881393432617188e-07`.

## Evaluator replacement semantics

The evaluator operations are separate interventions:

- Zero replacement replaces reference RGB with `torch.zeros_like(reference_images)`; masks and target remain unchanged.
- Base replacement replaces reference RGB with `target_base_rgb` expanded to the reference count; masks and target remain unchanged.

Across all 20 frozen episodes, their reference-feature hashes differ (`0/20` bitwise-equal episodes). Therefore `ZERO_REPLACEMENT_EQUALS_BASE_REPLACEMENT = FALSE`. The equality of zero-head step-0 outputs does not make the evaluator inputs equivalent.

No render, evaluator metric, checkpoint, backward pass, or optimizer step was produced for this audit. The full machine-readable record is `paper_protocol/reviewer_risk/mean_garment_zero_residual_semantics_audit.json`.
