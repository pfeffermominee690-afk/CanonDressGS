# Full Pipeline Checkpoint Reproducibility Policy V1

This policy distinguishes exact checkpoint restoration from numerical
reproducibility of a full CUDA raster-training step.

## A. State restoration determinism

The following remain strict and are never covered by a CUDA noise allowance:

- model tensors before a training step;
- optimizer state and parameter-group fingerprints;
- scheduler and AMP state;
- Python, NumPy, Torch CPU, and Torch CUDA RNG;
- sampler position, global step, and fixed-batch fingerprint;
- residual contract, bounds, SH degree, graph, and interpolation metadata;
- bounded residuals, gates, and composed canonical attributes before rasterization;
- a deterministic no-raster optimizer step.

These values must be bitwise equal where representable, otherwise satisfy
`atol=1e-7`, `rtol=1e-6`.

## B. Full CUDA render-training reproducibility

Some custom CUDA raster backward kernels use parallel accumulation and do not
guarantee bitwise equality across independent processes. A bounded numerical
allowance is valid only after an experiment establishes all of the following:

1. independent control processes started from the same checkpoint exhibit a
   nonzero empirical noise floor;
2. the first difference occurs in raster forward/backward or later;
3. no-raster checkpoint/resume parity passes the strict rule above;
4. resume-control maximum parameter difference is no greater than
   `max(1.25 * D_control_max, D_control_max + 1e-7)`;
5. the absolute parameter cap is `1e-6` and relative L2 cap is `1e-6`;
6. post-step RGB/alpha use `atol=1e-6`, `rtol=1e-5`;
7. post-step loss absolute difference is at most `1e-6`;
8. a three-step fixed-input check stays within `1e-6` without divergent growth.

This bounded rule does not weaken checkpoint loading, metadata validation,
optimizer restoration, RNG/sampler restoration, graph/interpolation validation,
or deterministic no-raster testing.
