# Module 4B-R Oracle Root-Cause Audit

Date: 2026-07-18

Run ID: `SUBJECT02-MODULE4B-ROOT-CAUSE-001/attempt_001`

Status: **COMPLETE / FAIL**

Root-cause cases: **R2 + R3**

## Scope and provenance

- Frozen formal baseline: `e01daa19cd134bce9a2d96bb793eccf097d9c833`.
- Diagnostic implementation commit: `5b7031b018aa74649ac3ea815b47e9566b38a3bb`.
- Evidence-completeness supplement commit: `1ec490f0f3454b168ff9f9303f0302f03e94ce5f`.
- Sealed Module 4B input: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MODULE4B-MICROPILOT-001/attempt_001`.
- Output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MODULE4B-ROOT-CAUSE-001/attempt_001`.
- Base checkpoint SHA256: `abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`.
- Environment: Python 3.10.20, PyTorch 2.4.1+cu121, CUDA 12.1, NVIDIA GeForce RTX 4090.
- The audit added **zero optimizer steps**. It did not rerun the six 480-step experiments, did not run Anchor Oracle, and did not alter V5.3 data, masks, loss, renderer, MMLP-Human, interpolation, or formal gate/composition semantics.

The four phases were executed with `tools/run_module4b_oracle_root_cause_audit.py --phase preflight|audit|gate-open|finalize`. The later `--phase supplement` only recomputed frozen-base scale quantiles to add the preregistered p90 field; it repeated no renders and no optimization.

## Gate path audit

- Initial gate probability/logit: `0.05` / `-2.944438979`.
- Activation: sigmoid.
- Opacity gate: `max(geometry_gate, appearance_gate)`.
- Gaussian Oracle gates a bounded Gaussian residual once before composition.
- Anchor Oracle gates at anchor level and interpolates the already gated residual. Separately interpolated gates are reporting fields and are not multiplied into the rendered residual again.
- Formal composition and renderer do not reapply gates. Regularization reads gate values but does not modify render composition.
- `DOUBLE_GATE_COMPOSITION_BUG=false`.
- Geometry and appearance gate parameter groups are in the optimizer at LR `0.001`; the scheduler is disabled. Stage 1 trains appearance gate only; stages 2 and 3 train both.

At step 0, every activated geometry/appearance/opacity gate is `0.05`, with zero fractions at thresholds `0.10`, `0.25`, and `0.50`. At step 480, O00 Gaussian means are geometry `0.0582719`, appearance `0.0539787`, opacity `0.0589246`; their maxima are `0.0932324`, `0.0988758`, and `0.0988758`. Across all six final runs every gate maximum remains below `0.10`. O00 Gaussian step-480 logit RMS updates are geometry `0.206259` and appearance `0.152770`; logit gradient norms are `8.38416e-4` and `1.93585e-4`. The gates are optimized but remain strongly suppressive.

This audit does **not** adjudicate R1. The preregistered technical stop prevented D1 from running after R2 was detected.

## Attribute and rotation diagnostics

Deterministic front/back probes used edit-region and preserve/protected control Gaussians, formal bounds, and gates `0.05`, `0.5`, and `1.0`, without changing base tensors.

| Attribute | Gate-1 diagnostic response | Interpretation |
|---|---|---|
| xyz | RGB max `0.826222`; alpha max `0.879211`; autograd/finite-difference agreement 2/2 | Strong renderer path |
| log scaling | RGB max `0.009485`; alpha max `0.011115` | Weak but nonzero path; direction checks are noise-sensitive at this amplitude |
| rotvec | RGB max `0.193263`; alpha max `0.204902`; dedicated covariance probe is authoritative | Nonzero downstream covariance path, but zero-init graph is broken |
| opacity logit | RGB max `0.002352`; alpha max `0.002762` | Weak but nonzero path |
| SH0 | RGB max `0.010659`; alpha exactly `0`; autograd/finite-difference agreement 2/2 | Expected color-only response |

SHN is exactly zero because the formal checkpoint uses SH degree 0; this is expected and is not a root cause.

Activated scale anisotropy over 200,000 base Gaussians is: p50 `2.704064`, p90 `10.773348`, p95 `19.216963`, p99 `73.724037`, max `241462.0`; fractions at ratios `>=1.05/1.10/1.25/1.50` are `0.997515/0.990875/0.951855/0.858455`. Rotation is therefore not generally degenerate through isotropy.

The formal `compose_canonical_gaussian_overrides` zero shortcut returns base rotation when all rotvec elements are exactly zero. That removes the trainable zero-initialized rotvec from autograd. Evidence:

- zero-initialization autograd connected: `false`;
- high-anisotropy zero-point covariance finite difference: `-6.530317e-4`;
- nonzero high-anisotropy rotvec changes normalized wxyz quaternion and covariance;
- high-anisotropy autograd and finite-difference directions agree on all three axes.

Classification: **ROTATION_PATH_BROKEN**, root-cause case **R2**. The formal composition was deliberately not fixed in this audit.

## D0/D1 technical-stop result

D0 learned-gate O00 Gaussian front-view metrics changed from step 0 to 160 as follows: total `1.0063953 -> 0.9992509`, edit `0.2647330 -> 0.2634108`, clothing `0.2371967 -> 0.2357674`, and alpha-edit `0.7905126 -> 0.7859680`. The opened step-0/step-160 panel remains visually close to the base long-sleeve hoodie.

D1 fixed-open was **SKIPPED_TECHNICAL_STOP / ROTATION_PATH_BROKEN**, with zero optimizer steps. Consequently there are no D1 losses, visible-change claims, residual magnitudes, gate statistics, bound-hit rates, or abnormal-Gaussian statistics. R1 gate bottleneck remains unadjudicated.

## Old-sleeve base support

- Projected old-sleeve Gaussian union: `17,241`.
- LBS arm candidates (SMPL-X shoulder/elbow/wrist joints 16-21): `12,931`, or `75.00145%` of the union. No explicit body-part labels exist.
- Skin-like base DC color fraction at RGB distance `<=0.15`: `4.25729%`.
- Old-sleeve activated opacity: mean `0.684649`, p50 `0.711921`, p95 `0.846502`, max `0.984687`.
- Base DC RGB distance to skin: mean `0.295599`, p50 `0.284075`.
- Front B probe (old-sleeve opacity down): alpha MAE `0.812495`, RGB MAE `0.515537` in the old-sleeve region.
- Back B probe: alpha MAE `0.857410`, RGB MAE `0.216063`.
- C probe skin-reference RGB MAE: front `0.042832`, back `0.167486`; alpha remains near one (`0.999544` / `0.992356`).

Actual opening of the A/B/C contact sheet shows that B creates holes/background penetration and only fragmented dark curves. C recolors the outer sleeve shell into bulky tubular brown sleeves rather than revealing a continuous inner arm. Classification: **BASE_SUPPORT_MISSING**, root-cause case **R3**.

## Objective and mask audit

The old-sleeve region is not accidentally preserve-only:

- front old-sleeve pixels `18,588`: edit-core overlap `18,235` (`98.1009%`), transition overlap `353` (`1.8991%`), preserve/protected/alpha-base pixel overlap all `0`;
- back old-sleeve pixels `16,843`: edit-core overlap `15,258` (`90.5896%`), transition overlap `1,585` (`9.4104%`), preserve/protected/alpha-base pixel overlap all `0`.

Old-sleeve effective-xyz gradient norms are nonzero: edit RGB front/back `0.0110941/0.0376841`, alpha-edit front/back `0.213107/0.186314`. Tiny preserve/protected residual gradients on some Gaussian centers arise from splat support outside the center pixel; the pixel masks themselves have zero old-sleeve overlap. `objective_mask_conflict=false`; R4 is rejected.

## Final adjudication

- Root causes: **R2 + R3**.
- Formal composition fix required: **yes**, in a separate authorized task with a minimal zero-init regression test.
- Redefine Oracle gate policy now: **no**; R1 was not adjudicated because D1 correctly stopped.
- Rebuild or replace the base human representation: **yes**, at least for reliable under-clothes body/skin support; this does not by itself authorize a garment Gaussian layer.
- Rerun Module 4B: **no**, until R2 is fixed and minimally verified; after that, R3 still requires an explicit representation decision.
- Formal image-conditioned training: **not allowed**.
- Final status: **FAIL**.
- Immediate sequential blocker: the zero-initialized rotation residual is disconnected by the formal composition zero shortcut.

## Verification

- Root-cause tests: 10/10 PASS.
- Existing Module 4B tests: 12/12 PASS.
- Full-attribute Oracle unit checks: PASS.
- Full-training checkpoint checks: PASS.
- Python compilation: PASS.
- `git diff --check`: PASS.
- Frozen base fingerprints remain bitwise exact.
