# Zero-Initialized Rotation Autograd Closure R2

Date: 2026-07-18

Status: **PASS**

## Commits and output

- Formal repair: `1386a42`.
- Regression and real-smoke tools: `f88dca8`.
- Real smoke run: `SUBJECT02-ROTATION-AUTOGRAD-R2-001/attempt_001`.
- Output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-ROTATION-AUTOGRAD-R2-001/attempt_001`.
- Run commit: `f88dca830b5201999caff3e17c6f3aa12c758663`.

## Numerical closure

- Random normalized base quaternion plus exactly zero trainable rotvec matches the base at `atol=rtol=1e-12` in float64.
- The composed quaternion has a real `grad_fn`; rotvec gradient is present, finite, and nonzero for a non-degenerate vector/covariance objective.
- Small-angle quaternion vector part matches `0.5 * rotvec` for `1e-8`, `1e-6`, and `1e-4` inputs.
- CPU float32 and CUDA float32 zero/small batched conversions are finite in forward and backward.
- Anisotropic covariance autograd and centered finite difference agree at zero with relative error below `1e-6` in the deterministic unit test.
- Isotropic covariance is allowed to have zero rotation sensitivity while preserving a finite, present gradient tensor.

## Shared-path closure

- Gaussian Oracle zero rotation: connected and nonzero on a rotation-sensitive covariance objective.
- Anchor Oracle zero rotation: interpolation and composition remain connected and nonzero.
- Image-conditioned six-head rotation head: exact zero output remains connected through bounding, interpolation, composition, and covariance to `rotation_head.weight/bias`.
- No teacher, target gate, or target image/mask is used to construct these paths.

## Real O00 renderer smoke

The smoke used `O00/cond_000000`, the frozen subject02 base, the formal V5.3 region-aware loss, formal renderer, formal gate/bounds, and Gaussian Oracle at exact zero residual initialization. It created no optimizer and executed zero optimizer steps.

- Raw rotvec exact zero: true.
- Composed rotation grad_fn: `DivBackward0`.
- Raw RGB/alpha grad_fn: `CloneBackward0` / `PermuteBackward0`.
- Asymmetric renderer-probe rotation gradient norm: `2.47606044468e-06`; nonzero elements: `499131`.
- Formal V5.3 objective: `1.00821352005`.
- Formal V5.3 rotation gradient norm: `6.84699552949e-05`; nonzero elements: `499131`.
- Other trainable raw gradient norms: xyz `0.00216466864`, scaling `0.000140519245`, opacity `0.000193614367`, SH0 `0.000108956134`.
- SHN remains formally disabled at SH degree 0.
- Base gradient names: none.
- Frozen base fingerprint before/after: `de312ccbcabaa37cde63cd318d087b69e830ed5148779ca40456784b964f80d9`; bitwise exact.

Zero-forward renderer comparison uses mean-error tolerance `1e-3` because independent CUDA gsplat forwards have known sparse max differences. Observed RGB mean/max absolute difference is `0.000384133/0.193263`; alpha is `0.000253932/0.204902`. The canonical quaternion difference from normalized base is `1.78814e-7`. This matches the previously measured independent-forward CUDA noise floor and is not a semantic output change.

Classification: `PATH_FIXED_REAL_BATCH_GRADIENT_NONZERO`.

## Regression scope

- New rotation closure tests: 12/12 PASS, including CUDA.
- Historical Module 4B-R regression tests: 10/10 PASS with the old detector converted to a closure regression.
- Module 4B contract: 12/12 PASS.
- Full Gaussian residual contract: PASS.
- Full-attribute Oracle: PASS.
- Dressable model: PASS.
- Full-dataset V5.3 fixture regression: PASS.
- Image-conditioned dataset: 20 checks PASS.
- Dual-target/region-aware supervision: 28/28 PASS.
- Full-training checkpoint: PASS.
- py_compile and `git diff --check`: PASS.

No data, mask, loss, gate, residual bound, interpolation map, base checkpoint, MMLP-Human, renderer, or historical Module 4B output was modified.

## Adjudication

R2 is closed. Full Module 4B was not rerun. The project may now proceed to a separately authorized R3 base-support representation design task. Existing evidence still prohibits formal image-conditioned training and does not authorize a garment Gaussian layer.
