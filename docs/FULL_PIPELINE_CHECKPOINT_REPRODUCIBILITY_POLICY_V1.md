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

Some custom CUDA raster backward kernels use nondeterministic parallel
reductions. Cross-process training trajectories therefore are not required to
remain bitwise identical when this backend participates in backward.

This limitation may be recorded only after all of the following are established:

1. the strict state-restoration requirements in Section A pass;
2. the deterministic no-raster optimizer step is bitwise equal or within the
   strict `atol=1e-7`, `rtol=1e-6` rule;
3. pre-raster residuals, gates, and composed attributes are exact;
4. trace evidence locates the first difference in a known nondeterministic CUDA
   raster backward stage or later;
5. independent control processes started from the same checkpoint exhibit a
   comparable numerical noise floor;
6. a resumed single step does not exhibit an additional anomaly beyond that
   independent-control noise;
7. restored optimizer moments and step counters are not reset;
8. RNG, sampler, data position, graph, and interpolation state are not reset;
9. resumed execution remains finite and its loss/evaluation sequence is
   continuous, without a restart signature, sudden discontinuity, or NaN/Inf;
10. the backend, environment, deterministic warnings, and reproducibility
    limitation are recorded with the run.

Longer independent CUDA trajectories may separate because each step feeds a
new reduction-order perturbation into Adam. Such separation is evidence that
the backend does not support exact trajectory reproduction; it is not, by
itself, evidence of approximate checkpoint loading or optimizer restoration.

The approved wording is:

> Cross-process training trajectories are not bitwise reproducible because the
> CUDA raster backward contains nondeterministic reductions.

Do not describe this condition as checkpoint approximate restore, optimizer
restore error, or resume failure.

This CUDA limitation never weakens checkpoint metadata validation, model and
optimizer loading, parameter-group validation, RNG/sampler restoration,
graph/interpolation validation, global-step restoration, pre-raster parity, or
the deterministic no-raster optimizer-step test.

## Module 4A Final Adjudication

The Module 4A evidence establishes:

- checkpoint state restoration: **PASS**;
- functional interrupted resume: **PASS**;
- exact cross-process CUDA training trajectory reproduction: **unsupported by
  the current gsplat backend**.

Accordingly, Module 4A is frozen as
`PASS_WITH_CUDA_NONDETERMINISM`. This is neither a checkpoint failure nor a
representation-method failure.
