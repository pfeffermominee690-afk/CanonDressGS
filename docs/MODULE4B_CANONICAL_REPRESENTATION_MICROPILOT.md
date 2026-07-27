# Module 4B Canonical Representation Oracle Micro-pilot

## Final adjudication

- Run ID: `SUBJECT02-MODULE4B-MICROPILOT-001/attempt_001`
- Formal candidate-run commit: `dce29e089cc422abd661c622800e2d4c435bce26`
- Evidence-seal code commit before this report: `a5c5c02055dd5df7e91acd44e604e2c4a3b84b8b`
- Branch: `pipeline/full-dressable-20260715`
- Status: **FAIL**
- Decision matrix: **D**
- Next-stage permission: **DENIED**
- Bottleneck classification: **optimization/objective/composition or data-chain audit required**

This is a representation-capacity oracle experiment, not an image-conditioned inference result. It does not use reference images, teacher residuals, condition-specific residuals, or an image backbone. Target fields are consumed only after rendering by the V5.3 loss and evaluator.

## Frozen protocol

- Outfits: `O00`, `O01`, `O05`.
- Shared conditions per outfit: `cond_000000` (front), `cond_000318` (back), `cond_000017` (left), `cond_000347` (right).
- Runs: Gaussian-level and Anchor-level oracle for each outfit, six runs total.
- Steps: 480 per run, fixed front/back/left/right round-robin, 120 updates per condition.
- Schedule: Module 4A formal policy takes precedence: appearance initialization 1–80, geometry plus appearance 81–160, joint refinement 161–480.
- Loss: V5.3 `dual_target_region_aware_v1`, including the boundary-aware soft alpha target, protected base-only behavior, and transition SmoothL1.
- Base: 200,000 Gaussians, frozen and fingerprinted.
- Anchor representation: 10,000 anchors and the formal `[200000,3]` normalized interpolation mapping.
- Formal SH degree: 0; SHN remains a strict-zero disabled compatibility tensor.
- Bounds: xyz `0.05`, log-scaling `0.35`, rotation `0.2617993878` rad, opacity logit `2.0`, SH0 `0.25`, SHN `0.10` (disabled).

The 12-sample input fixture and V5.3 loss configuration were reused without modification. The pre/post aggregate input fingerprint is `457b4f93c9c04c337e7222750ca0ad8f9b6cf4f388775de1f10a18c640dba6fc`; post-run verification reports every protected input unchanged.

## Execution

Environment:

- Python `3.10.20`
- PyTorch `2.4.1+cu121`
- CUDA runtime `12.1`
- GPU `NVIDIA GeForce RTX 4090`
- Conda Python `/root/autodl-tmp/conda_envs/mmlphuman/bin/python`

The preflight, smoke, formal runs, preview, finalization, and evidence seal used `tools/run_module4b_canonical_oracle_micropilot.py` with the corresponding `--phase`, the frozen manifest, `configs/oracle/module4b_canonical_capacity_v1.yaml`, and the unique output root. Each formal run used `--phase run --oracle-kind <gaussian|anchor> --outfit <O00|O01|O05>` and restarted from zero residual state.

Two early smoke attempts are retained as zero-optimizer-step tool-interface failures: one encountered legacy base parameter enumeration and one encountered an unmaterialized base buffer. They are not candidate results. The corrected Gaussian smoke and Anchor smoke passed; their updates were discarded before the six formal runs.

## Quantitative results

| Outfit | Oracle | Numerical fit | Visual | Edit reduction | Clothing reduction | Edit slope (401–480) | Clothing slope (401–480) | Time (s) | Peak GPU bytes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| O00 | Gaussian | PARTIAL_FIT | FAIL | 2.8830% | 3.2996% | -6.01525e-05 | -3.73493e-05 | 210.69 | 1,863,985,664 |
| O00 | Anchor | NO_FIT | FAIL | 1.6392% | 1.9987% | -4.76288e-05 | -2.56823e-05 | 204.54 | 1,768,028,160 |
| O01 | Gaussian | PARTIAL_FIT | FAIL | 6.5065% | 7.3434% | -5.95549e-05 | -4.88301e-05 | 227.78 | 1,861,228,032 |
| O01 | Anchor | PARTIAL_FIT | FAIL | 3.8509% | 4.4951% | -4.53840e-05 | -3.45754e-05 | 208.37 | 1,768,028,672 |
| O05 | Gaussian | PARTIAL_FIT | FAIL | 3.1991% | 3.3741% | -1.11490e-04 | -1.21814e-04 | 213.82 | 1,861,227,520 |
| O05 | Anchor | PARTIAL_FIT | FAIL | 2.0634% | 2.2282% | -9.96359e-05 | -1.09923e-04 | 207.64 | 1,768,028,672 |

Gaussian Oracle has 4,800,000 trainable scalar parameters; Anchor Oracle has 240,000. All values remained finite. Protected last-40 means were `0.00302–0.00331`, below the preregistered absolute limit `0.005`; preserve last-40 means were `0.000160–0.000185`.

Anchor improvement retention relative to Gaussian was insufficient for every outfit:

| Outfit | Edit retention | Clothing retention | Required |
|---|---:|---:|---:|
| O00 | 0.5686 | 0.6057 | 0.75 |
| O01 | 0.5919 | 0.6121 | 0.75 |
| O05 | 0.6450 | 0.6604 | 0.75 |

The retention result does not establish an anchor bottleneck because the Gaussian upper-bound representation itself failed visual acceptance, including O00.

## Gradient, stability, and freeze evidence

- All six formal runs executed exactly 480 optimizer steps and remained finite.
- Frozen base state stayed bitwise exact with fingerprint `de312ccbcabaa37cde63cd318d087b69e830ed5148779ca40456784b964f80d9`; base gradient count was zero.
- The image backbone was not instantiated; its parameter, gradient, and update counts were zero.
- Disabled SHN stayed exactly zero.
- All 24 outfit/oracle/condition deterministic gradient-audit cases were finite and retained the exact base state.
- Rotation residual gradients were zero in every formal run. This violates the expected all-head gradient contract and is evidence for an optimization/composition audit; it must not be hidden by a relaxed threshold.
- O05 Gaussian and Anchor had zero residual-bound-hit/extreme-Gaussian fractions and zero outside-body-radius fraction. No opacity cloud, floating Gaussian, or large-scale distortion was observed.
- Gate probabilities stayed low: the O05 Gaussian geometry/appearance means were `0.05687/0.05321`; the interpolated O05 Anchor means were `0.06551/0.06854`; active fraction at `>=0.5` was zero.

## Checkpoint and resume

O00 Gaussian step 40 checkpoint SHA256 is `f7c527598c5f1eec01df952b3918558af26b7b7f504eff1c0642660e8666ddc6`. Model tensors, optimizer state, global step, sampler position, and next condition were restored exactly. The controlled resume test passed without repeating an optimizer step.

## Visual acceptance

Codex actually opened the six run step-0/final four-view sheets, every O05 requested intermediate stage, final residual/gate panels, protected-error panels, and the Gaussian-versus-Anchor comparison. All six runs are `VISUAL_FAIL`:

- O00: both representations remain the purple long-sleeve hoodie. Short sleeves, exposed upper arms, and the target jeans do not form.
- O01: the target gray hoodie appearance and convincing loose hoodie structure do not form; the purple base color and graphic remain.
- O05: neither representation forms the black long coat, hem extension, or exterior silhouette at steps 0, 80, 160, 320, or 480.
- Face, hair, hands, white shoes, and background remain mostly stable, with small but nonzero protected-region error.
- No run depends on abnormal Gaussian clouds to reduce its loss.

## Tests and decision

- Python compilation: PASS.
- Module 4B contract tests: 12/12 PASS.
- Full-attribute Oracle unit checks: PASS.
- Full-training checkpoint checks: PASS.
- `git diff --check`: PASS.

The final decision is case D: Gaussian O00 reached only `PARTIAL_FIT` numerically and `VISUAL_FAIL`. Therefore this experiment cannot adjudicate a body-Gaussian topology limit and cannot justify a garment Gaussian layer. Formal image-conditioned training is blocked. The single remaining blocker is that the direct Gaussian-level shared canonical oracle does not yet form the target garment, so the optimization/objective/composition/data chain must be audited before representation or method expansion.

## Evidence locations

- Output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MODULE4B-MICROPILOT-001/attempt_001`
- Final status: `MODULE4B_FINAL_STATUS.json`
- Final adjudication: `MODULE4B_FINAL_ADJUDICATION.md`
- Visual acceptance: `MODULE4B_VISUAL_ACCEPTANCE.md`
- Input fingerprints: `module4b_input_fingerprint.json`
- Oracle contract: `module4b_oracle_contract.json`
- Comparison tables and contact sheets: `comparisons/`
