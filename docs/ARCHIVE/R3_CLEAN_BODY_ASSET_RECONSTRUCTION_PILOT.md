# Subject02 Clean Body Asset Reconstruction Pilot R3-CLEAN-001

## Final decision

The diagnostic pilot stopped at its preregistered geometry gate and is sealed as **`CLEAN_BODY_ASSET_PILOT_FAIL / FAIL_GEOMETRY_ALIGNMENT`**. This is not a training failure: no optimizer was created and no optimizer step ran. No shell decomposition, skin appearance field, 60k/120k support asset, clean-foundation render, O00 Oracle, Module 4B rerun, or image-conditioned training was executed.

## Reproducibility contract

- Run ID: `SUBJECT02-CLEAN-BODY-ASSET-PILOT-001/attempt_001`
- Formal run commit: `9afac2548f15dd9bb0692415384a6b2f6510cfe5`
- Config: `configs/audit/r3_clean_body_asset_pilot_v1.yaml`
- Config SHA256: `7531889b68678942c223dfa46f3dd8327cb3168ad99e4cacb09584bce857c7df`
- Output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-CLEAN-BODY-ASSET-PILOT-001/attempt_001`
- Base checkpoint SHA256: `abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`
- Base Gaussian count/fingerprint: `200000` / `de312ccbcabaa37cde63cd318d087b69e830ed5148779ca40456784b964f80d9`
- Environment: Python `3.10.20`, PyTorch `2.4.1+cu121`, CUDA runtime `12.1`, NVIDIA GeForce RTX 4090.
- Condition protocol: the frozen 12 V3-ready conditions, with front/back/left/right represented three times each.
- Optimizer created/steps: `false` / `0`.

Commands:

```bash
/root/autodl-tmp/conda_envs/mmlphuman/bin/python \
  tools/run_r3_clean_body_asset_pilot.py \
  --phase preflight \
  --expected-head 9afac2548f15dd9bb0692415384a6b2f6510cfe5 \
  --condition-bundle /root/autodl-tmp/canondressgs_work/tmp/r3_clean_001_inputs

/root/autodl-tmp/conda_envs/mmlphuman/bin/python \
  tools/run_r3_clean_body_asset_pilot.py \
  --phase run \
  --expected-head 9afac2548f15dd9bb0692415384a6b2f6510cfe5 \
  --condition-bundle /root/autodl-tmp/canondressgs_work/tmp/r3_clean_001_inputs
```

## Geometry result

The reconstructed mesh uses subject02 `smpl_params.npz` beta values, not mean beta. It has `10,475` vertices, `20,908` faces, no degenerate faces, and the formal 55-joint SMPL-X LBS topology. The coordinate contract is the MMLP-Human canonical big pose in meters with the formal axes.

Across the frozen 12 conditions, all posed vertices were finite, bbox height/width ratios remained close to one, and no global left/right or up/down flip was found. Silhouette IoU was nevertheless insufficient:

- minimum: `0.7668237096`
- mean: `0.8233781843`
- below the frozen `0.80` gate: `cond_000714=0.7996358786`, `cond_000113=0.7668237096`, `cond_000439=0.7962255062`

The condition assets preserve their source PyTorch3D R/T/FOV. The pilot derives OpenCV projection and performs deterministic isotropic bbox fitting to the supplied V3-ready foreground masks. It does not hand-author camera values. Independent 2D joint labels are absent, so projected-joint residual is explicitly marked `NOT_AVAILABLE_NO_2D_JOINT_LABELS` rather than reported as zero.

Actual opening of the 12-condition contact sheet and the `cond_000113`/`cond_000439` overlays confirmed residual contour mismatch around the head, chest/abdomen, buttocks, feet, and pose-dependent arm/hand regions. This is not a simple global flip or scale-only error.

## Integrity and regression

Post-failure verification recomputed every sealed asset, evidence marker, R3 inventory, and condition-bundle SHA256 and reloaded the formal base. `all_file_sha256_exact=true`, `base_gaussian_bitwise_exact=true`, and `formal_inputs_unchanged=true`.

Regression results before the pilot:

- R3-CLEAN `12/12` PASS
- R2 `12/12` PASS, including CUDA
- R3 `8/8` PASS
- Module 4B `12/12` PASS
- Module 4B-R `10/10` PASS
- V5.2 `3/3` PASS
- V5.3/region-aware dual-target `28/28` PASS
- image-conditioned dataset and clothing-loss checks PASS
- full-attribute Oracle, full Gaussian residual, six-channel decoder unit, and checkpoint checks PASS
- `py_compile` and `git diff --check` PASS

The parameterized `check_real_six_channel_decoder.py` acceptance entrypoint was not rerun with invented inputs; its missing-argument exit is not recorded as a regression failure.

## Downstream authorization

- Build formal clean-body base pilot: **NO**
- Minimal O00 Gaussian Oracle: **NO**
- Full Module 4B: **NO**
- Formal image-conditioned training: **NO**

Only remaining blocker: obtain or define a subject02 clean-body geometry/camera alignment contract that passes the frozen 12-condition gate and includes independent joint/camera evidence, rather than relying on silhouette bbox fitting alone.
