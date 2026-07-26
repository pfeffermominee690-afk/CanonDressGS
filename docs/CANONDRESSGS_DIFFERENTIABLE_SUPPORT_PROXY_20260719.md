# CanonDressGS Differentiable Support Proxy Adjudication — 2026-07-19

## Final status

- Task: `SUBJECT02-DIFFERENTIABLE-SUPPORT-PROXY-002`
- Formal run commit: `3b674e442f473d4bb0594f52b14fc7ad28aead23`
- Formal evidence: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-DIFFERENTIABLE-SUPPORT-PROXY-002/attempt_002`
- Historical first attempt: `attempt_001` is preserved. It completed with zero optimizer steps, but did not persist the required region-level N_pre table. No result or artifact was overwritten; the reporting-only correction was committed before `attempt_002`.
- Final case: `PF`
- Frozen proxy: none
- Screen-space placement experiment resume: not allowed
- Training: not allowed
- Next unique task: `AUDIT_EDITABLE_GAUSSIAN_POOL_SPECIFICITY`

The final PF result is not a renderer, checkpoint, gradient, or training failure. It means none of the three preregistered proxy families satisfied every static gate under the unchanged 169,106-Gaussian editable pool.

## Why the previous proxy failed

`FIXED_ATTRIBUTE_SUPPORT_RENDER_V1` accumulated opacity through front-to-back alpha compositing. Overlap drove this quantity toward saturation, so it did not represent contributor count. It ranked P1 below P3 on O08 back/right and had Spearman correlation `-0.314286` with instrumented N_pre. No optimizer was created in that experiment.

## Candidate formulas and gradient contract

### A — `SOFT_PRE_CONTRIBUTOR_COUNT_V1`

For projected center `mu_i`, detached conic/covariance and detached opacity, the implementation reproduces official single-Gaussian `log_alpha_i(p)` and computes:

`w_i(p) = sigmoid((log_alpha_i(p) - log(1/255)) / temperature)`

`C_soft(p) = sum_i w_i(p)`

The only temperatures are `0.10`, `0.20`, and `0.40`. The operation is additive: there is no transmittance, alpha compositing, depth ordering, or early termination.

### B — `FIXED_RADIUS_CENTER_DENSITY_V1`

`D_r(p) = sum_i exp(-||p-mu_i||^2 / (2r^2))`

The only radii are `2`, `4`, and `8` pixels. This proxy does not accept scale, rotation, or opacity.

### C — `SOFT_KNN_SUPPORT_DISTANCE_V1`

The only k values are `4`, `8`, and `16`. A full differentiable field at the registered quarter-resolution grid would contain `16,623,796,224` distances. One float32 matrix is `61.9285 GiB`; the minimum estimated forward plus differentiable ranking workspace is `185.7854 GiB`, before autograd safety margin. All k variants are therefore `NOT_FEASIBLE_FOR_FORMAL_TRAINING`. Detached nearest-neighbor indices were not substituted.

For A and B, covariance, scale, rotation, opacity and appearance attributes cannot receive gradients. Only projected means retain gradients; the intended formal caller connects them to canonical `delta_xyz`. Target masks are absent from all proxy forward interfaces and are used only for offline region sampling or a future loss.

## Candidate grid and sparse regression

- Support grid: fixed original-resolution stride `4`.
- Coarse cell size: `16` pixels.
- Grid input: detached projected means and preregistered support extents only.
- Proxy A extent: detached covariance/opacity ellipse with log-alpha tail margin `8.0`.
- Proxy B extent: fixed six-sigma extent for maximum registered radius `8` pixels.
- Candidate indices are rebuilt per state/view without target fields.
- Exact distances from gathered candidates to pixels retain gradients.
- No `169106 × 1536 × 1024` tensor is constructed.
- Sparse-vs-dense regression: `144/144 PASS`; maximum absolute difference `6.7358e-7`, registered limit `1e-3`.

## Static qualification protocol

The audit evaluates `O01` and `O08`, four fixed views (`front/back/left/right`), and P1/P2/P3: 24 state-view combinations. Region coordinates are derived once from the frozen P3 evidence and reused across all three states. The region table contains:

- trusted expansion;
- P3 trusted-expansion underfill;
- correctly covered garment;
- trusted removal;
- trailing cloud;
- normal background.

Instrumented N_pre is recomputed using the independent float64 projection oracle. A vectorized implementation is checked against `independent_pixel_support` on every state/view before its values are accepted. Target data is never used to form projected Gaussian attributes or the candidate grid.

## State and pixel correlation results

| Candidate | O08 back P1/P3 | O08 right P1/P3 | Global Spearman | O01 Spearman | O08 Spearman | Median pixel Spearman | Hard status |
|---|---:|---:|---:|---:|---:|---:|---|
| A, T=0.10 | 4.712522 | 4.593608 | 0.999130 | 1.000000 | 1.000000 | 0.998141 | FAIL: cloud |
| A, T=0.20 | 4.675992 | 4.590437 | 0.999130 | 1.000000 | 1.000000 | 0.997998 | FAIL: cloud |
| A, T=0.40 | 4.624961 | 4.569640 | 0.998261 | 1.000000 | 0.993007 | 0.997618 | FAIL: cloud |
| B, r=2 | 17.829328 | 13.660922 | 0.960870 | 0.895105 | 0.972028 | 0.903743 | FAIL: cloud |
| B, r=4 | 6.124077 | 7.170202 | 0.964348 | 0.902098 | 0.944056 | 0.921866 | FAIL: cloud |
| B, r=8 | 1.732604 | 1.966941 | 0.927826 | 0.818182 | 0.923077 | 0.892801 | FAIL: cloud |

Every A/B candidate passed:

- O08 back/right state ordering and P1/P3 ratio;
- global Spearman;
- per-outfit Spearman;
- median per-view pixel Spearman;
- P2/P3 low-support separation from P1;
- O01 normal-view stability.

Every A/B candidate failed the same trailing-cloud gate. The decisive P3/O08/back region has instrumented N_pre mean `78.166667`, while proxy means are:

| Candidate | P3 O08 back cloud proxy mean |
|---|---:|
| A, T=0.10 | 1.36845e-30 |
| A, T=0.20 | 1.72747e-16 |
| A, T=0.40 | 2.16510e-9 |
| B, r=2 | 3.81054e-29 |
| B, r=4 | 9.21930e-9 |
| B, r=8 | 0.00781058 |

Thus the real projection contains many cloud contributors, but the unchanged G_editable proxy pool contains essentially none of them in the failed back-view region. This is direct evidence for the preregistered PF handoff to editable-pool specificity audit. Cloud support is reported only as an anomaly; it is never interpreted as correct garment coverage.

## Anti-saturation, gradients and performance

The selected-candidate rule first excludes every hard-gate failure. Since no candidate survived the cloud gate:

- no proxy was selected or frozen;
- anti-saturation was not used to rescue a failed candidate;
- gradient-direction audit was correctly not run;
- protected Gaussian gradient count is recorded as `0`;
- non-xyz gradient count is recorded as `0`;
- full quarter-resolution performance audit was correctly not run.

The six A/B forward timing totals accumulated across the 24 sampled state/views range from `0.0336` to `0.0567` seconds, but these are static sampled-query timings, not the registered full-view forward/backward performance measurement. They must not be reported as the 1.5-second full-view benchmark.

## Synthetic scenes

All `10/10` registered scenes passed:

1. center-density difference with equal covariance;
2. low-density state with larger covariance;
3. equal centers with different opacity for A;
4. B invariance to scale/rotation/opacity;
5. one point moving toward a target;
6. one point moving away from forbidden support;
7. additive multi-point overlap;
8. wide-Gaussian alpha-saturation counterexample;
9. tile-boundary continuity;
10. reduced O08 ordering reproduction.

These synthetic passes establish that the proxy implementations have the intended mathematics. They do not override the failed real-data cloud gate.

## Safety and reproducibility

- G_editable count: `169106 / 200000`.
- G_editable file SHA256 before/after: `232124458848a684cd98bef1b162ec889f1e3e4c80c0ef1a5caebcceb4edb011` / identical.
- Frozen base fingerprint before/after, both outfits: `de312ccbcabaa37cde63cd318d087b69e830ed5148779ca40456784b964f80d9` / identical.
- Frozen base gradient count: `0`.
- Optimizer created: `false`.
- Optimizer steps: `0`.
- Source evidence unchanged: `true`.
- Frozen branches unchanged: `true`.
- Previous output trees were not modified or deleted.

Regression results: new proxy tests `23/23`, placement `28/28`, instrumented projection `21/21`, alpha raster `20/20`, V6.1 `24/24`, V6 `27/27`, V5.3 `28/28`, R2 CUDA `12/12`, differentiable renderer unit checks PASS, full checkpoint checks PASS, image-conditioned dataset checks PASS, `py_compile` PASS, and `git diff --check` PASS.

## Final adjudication

`Case PF` is frozen for this task. No candidate is admitted for a placement loss, so screen-space placement must not resume, seven-outfit readjudication must not start, more targets must not be generated, and formal image-conditioned training must not start.

The only next task is:

`AUDIT_EDITABLE_GAUSSIAN_POOL_SPECIFICITY`
