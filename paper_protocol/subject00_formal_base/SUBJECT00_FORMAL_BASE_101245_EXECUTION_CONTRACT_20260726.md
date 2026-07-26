# Subject00 Formal Base 101245 Execution Contract

Task ID: `AAAI27-SUBJECT00-FORMAL-BASE-EXECUTION-PREFLIGHT-001`

This is a preflight and freeze contract only. It authorizes no training, no optimizer step, no output root creation, no attempt creation, no checkpoint write, and no dataset mutation.

## Source

- Source branch: `research/subject00-storage-reclamation-execution-20260725`
- Source HEAD: `d541e46a7963502d1e6613cbdfa477fdb4e82d6c`
- Preflight branch: `research/subject00-formal-base-execution-preflight-20260726`
- Windows worktree: `E:\model_train\canondressgs_subject00_formal_base_execution_preflight`
- Cloud worktree: `/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_subject00_formal_base_execution_preflight`
- Cloud identity: `autodl-container-ef19489c10-464381bb`, user `root`, project root `/root/autodl-tmp/canondressgs_work`

## Subject And Data

- Subject: `THuman4.0 Subject00`
- Identity: `subject00`
- Raw data root: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00`
- Processed data root: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets/subject00`
- Data manifest: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/reports/SUBJECT00_CLOUD_DATA_MANIFEST.json`
- Data manifest SHA256: `e61ca6061f0a85be9c5c6e1d9163341c10749bc1a4ff1b14d27f64b5705aa99e`
- Data tree SHA256: `2c0f894f70d944fa8d6ebd48ab1188b78cb19b673f0b7c006f1922a592b1ea7b`
- Availability manifest: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/reports/SUBJECT00_VALID_FRAME_CAMERA_MANIFEST.json`
- Availability file SHA256: `da9cbce12b6d0fa9a1fddced662eefa9d2f4f331011fbbf3a9944fc878011c7a`
- Availability canonical SHA256: `cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e`
- Frames: `2500`
- Cameras: `24`
- RGB files: `59704`
- Mask files: `59704`
- Pairing: `PASS_59704_VALID_PAIRS_296_IDENTICAL_MISSING_IMAGE_MASK_ENTRIES`
- Resolution: `1330x1150`

Protected inputs:

- Calibration: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00/calibration.json`
- Calibration SHA256: `4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7`
- Body parameters: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00/smpl_params.npz`
- Body parameter SHA256: `ac2738c308ad1a9cc02e7b63323c75e0eab28bddc88c57bb5887e07bf8ea28d2`
- Template: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets/subject00/template/template_smplx_body_surface.ply`
- Template SHA256: `f10a3b516e2b3a2ad38dc4924a3692b2f3e72a6cc9e66f3c0063c4e9cd210031`
- Surface LBS weights: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets/subject00/surface_lbs/lbs_weights.npy`
- Surface LBS file SHA256: `56aa68a9d4baade67621fa2bfac462ac88074eeaf7c9bfdbe86f2360b91261a1`
- Surface LBS array SHA256: `5176b447e1213e9275e08633589ef32ba4841825a05957d77b2f2c4ce18dc899`
- Surface attachment manifest SHA256: `de6cd51fe3f81e29139b45494860b186038b75a378b101e7428252b3b66c1af8`

## Initialization

Only legal initialization:

- Policy: `SUBJECT00_SHORT_CANARY_STEP0_ONLY`
- Path: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/checkpoints/step_000000.pth`
- Step: `0`
- Bytes: `701938720`
- SHA256: `29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a`
- Provenance status: `PASS_UNIQUE_SUBJECT00_SHORT_CANARY_STEP0_ONLY`

Forbidden: Subject02 checkpoints, non-step0 Subject00 canary checkpoints, optimizer-step checkpoints, garment-specific checkpoints, Full Avatar checkpoints, Teacher Endpoint, CanonDressGS endpoint, and any historical or unsealed checkpoint.

## Training Contract

- Config: `config/subject00_surface_lbs_formal_strict_split.yaml`
- Config SHA256: `be28dacbea5a3e4b35336556f752b1548ebb8c3c0b355be8348e405df49bd356`
- Total optimizer steps: `101245`
- Batch size: `1`
- Gradient accumulation: `1`
- Random background: enabled
- Precision: `DISABLED_NO_AUTOCAST_OR_GRADSCALER`
- Renderer: `scene.gaussian_model.GaussianModel.render`
- Rasterizer: `gsplat.rasterization`
- Seed: `0`
- Checkpoint cadence: `0, 20249, 40498, 60747, 80996, 100000, 101245`
- Evaluation cadence: fixed-96 trajectory at milestones; full four-quadrant streaming at final step `101245`

Trainable groups are explicitly frozen in the manifest: `dxyz`, `scales`, `quats`, `opacities`, `sh0`, `shN`, `dxyz_bs`, `dscales_bs`, `dquats_bs`, `dopacities_bs`, `dsh0_bs`, `dshN_bs`, `encoder_feat_params`, and `xyz_offset`. The trainable scalar count is `156097000`; selected frozen scalar count is `11631120`.

## Loss Optimizer Scheduler Seed

Loss:

- `l1`: weight `1.0`, masked/background-replaced full image
- `lpips`: weight `0.1`, active after step `6000`, crop size `512`, random patch active at step `300000`
- `dxyz_smooth`: weight `0.1`
- `gaussian_scaling`: weight `1.0`, threshold `0.01`

Optimizer:

- Adam/AdamW
- Betas `(0.9, 0.999)`
- Epsilon `1e-15`
- Default weight decay `0.0`
- Encoder feature weight decay `0.001`

Scheduler:

- `ExponentialLR`
- Horizon `800000`
- Step timing: after every optimizer step
- Unscheduled groups: `dscales_bs`, `dquats_bs`, `dopacities_bs`, `dsh0_bs`, `dshN_bs`

Seed `0` covers Python, NumPy, PyTorch CPU, PyTorch CUDA, frozen schedule order, dataloader generator semantics, camera/frame schedule, and renderer random-background sampling.

## Output And Resume

- Frozen output root: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-BASE-101245-001`
- Attempt ID: `attempt_001`
- Output root preexisted: `false`
- Output root created by preflight: `false`

Prior formal protocol evidence used `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001`. This contract freezes the requested `SUBJECT00-FORMAL-BASE-101245-001` root and records the prior alias history.

Resume policy:

- Resume only from nearest complete checkpoint within the same attempt.
- Restore model, optimizer, scheduler, Python RNG, NumPy RNG, Torch RNG, CUDA RNG states, and data-order position.
- No restart from step 0 after any optimizer step.
- No silent retry, output overwrite, attempt substitution, or checkpoint substitution.

## Storage GPU Environment

Live storage:

- Filesystem: `/dev/md0`, `xfs`, mounted at `/root/autodl-tmp`
- Total bytes: `268435456000`
- Used bytes: `222268727296`
- Free bytes: `46166728704`
- Projected steady-state bytes: `12280653758`
- Projected peak-write bytes: `14231374378`
- Projected final free bytes: `33886074946`
- Projected minimum free bytes: `31935354326`
- Formal storage gate: `32212254720`
- Status: `PASS_LIVE_FREE_BYTES_EXCEED_FORMAL_PRE_ATTEMPT_GATE_WITH_TIGHT_CONSERVATIVE_PEAK_MARGIN`

GPU/environment:

- GPU: `NVIDIA GeForce RTX 4090`
- Free VRAM: `24081 MiB`
- Utilization: `0%`
- Active compute processes: none
- Driver: `580.76.05`
- CUDA driver: `13.0`
- Python: `/root/autodl-tmp/conda_envs/mmlphuman/bin/python 3.10.20`
- PyTorch: `2.4.1+cu121`
- PyTorch CUDA: `12.1`
- gsplat: `1.5.3+pt24cu121`
- PyTorch3D: `0.7.8`
- Environment status: `PASS_MMLPHUMAN_ENV_IMPORTS`

## Authorization

- `TRAINING_AUTHORIZED = false`
- `OPTIMIZER_STEPS = 0`
- `OUTPUT_ROOT_CREATED = false`
- `DATA_MUTATIONS = 0`
- `CHECKPOINT_MUTATIONS = 0`
- `PAPER_MODIFICATIONS = 0`
- `PAPER_FINAL = false`

Final classification: `SUBJECT00_FORMAL_BASE_EXECUTION_CONTRACT_READY_PENDING_USER_AUTHORIZATION`

Next task: `USER_AUTHORIZE_SUBJECT00_FORMAL_BASE_101245_STEP_EXECUTION`
