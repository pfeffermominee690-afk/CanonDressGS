# Full Avatar O03 Equal-Step Execution Contract

Status: `FROZEN_STORAGE_RESOLUTION_PENDING`. Training and generation are not authorized.

## Identity, Views, and Data

- Base: `/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/chkpnt100000.pth`, SHA256 `abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`, step `100000`.
- Target manifest: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/aaai_gate_28_manifest.json`, SHA256 `49bee929edc236af8f37d66be50eb7c44f4011d49e435af903c0e15918a83bbf`.
- Protocol: `VIEW_TRANSDUCTIVE_MATCHED_ADAPTATION`. All four registered O03 views are used for adaptation and matched target-space evaluation; overlap is `4/4`.
- Disclosure: `INTENTIONAL_PRIVILEGED_ADAPTATION_ON_REGISTERED_TARGET_VIEWS_DISCLOSED`. This is a privileged per-garment adaptation control, not reference-controlled deployment, novel-view, leave-one-view-out, or target-independent adaptation.
- Different-pose and different-camera renders are animation-compatibility review only and do not enter target-space means without real targets.

## Objective

Use `EXACT_O03_TEACHER_OBJECTIVE_MATCH`: `CAPACITY_ORACLE_LOSS_V1` with weights `{"alpha_foreground": 0.5, "boundary_rgb": 0.25, "garment_rgb": 1.0, "new_silhouette_alpha": 1.0, "protected_alpha": 5.0, "protected_rgb": 10.0, "stability": 0.0001}`. The formal Teacher objective has no SSIM or perceptual component; exact matching keeps both absent. Mask roles, active-pixel denominators, stability definition, implementation paths, file hashes, and resolved config SHA are frozen in the manifest.

## Parameters and Optimization

- Policy: `ALL_FORMAL_BASE_OPTIMIZER_GROUPS`; 14 groups, 23 tensors, `156,097,000` float32 parameters (`624,388,000` bytes).
- Fresh optimizer state: `FRESH_RESET_NO_BASE_MOMENTUM_RESTORE`. Base momentum, scheduler state, scaler, loader cursor, and RNG state are not restored.
- Adam is used for 13 groups; `encoder_feat_params` uses AdamW with weight decay `0.001`. Betas are `(0.9, 0.999)`, epsilon is `1e-15`, and Base trainer gradient clipping is absent.
- Initial per-group LRs are the exact effective values stored at Base step 100000. Nine original ExponentialLR groups restart locally with a 1200-step horizon and the original family terminal factors (`0.01` for `dxyz`, `0.1` for the other eight); five originally unscheduled groups stay constant.
- Seed is `0` for Python, NumPy, PyTorch CPU/CUDA, loader generator, and the fixed four-condition sampling order.

## Steps, Checkpoints, and Resume

Exactly `1200` optimizer steps are required. Atomic checkpoints are fixed at local steps `0, 300, 600, 900, 1200`, with no best-checkpoint selection. After any optimizer step, only exact resume from the latest complete checkpoint is allowed; restarting from zero is forbidden.

## Storage Gate

The append-only correction changes the old storage classification from `PASS` to `FAIL_BELOW_SAFETY_GATE`. The current byte-exact conservative reservation is `10,680,535,158` bytes at steady state and `12,387,145,460` bytes during an atomic checkpoint write. From the captured `46,450,196,480` free bytes, minimum projected free space is `34,063,051,020`, short of the `40,802,189,312` formal gate by `6,739,138,292` bytes.

The Windows AvatarReX 7z and cloud 7z both rehash to `531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1` with `12,569,755,256` bytes, and Windows also contains a fully verified 60,834-file extraction. Current cloud Plan B staging is incomplete, so the allowed recommendation is `PLAN_A_PENDING_AVATARREX_PLAN_B_COMPLETION`. No deletion occurred. If Plan B completes and the user separately authorizes deletion, projected minimum free during training becomes `46,632,806,276` bytes.

## Authorization

Output root: `/root/autodl-tmp/canondressgs_work/outputs/FULL-AVATAR-FINETUNE-O03-EQUALSTEP-MICROPILOT-001`; attempt: `attempt_001`. Both remain uncreated. `training_authorized=false`, `generation_authorized=false`, optimizer steps `0`, and `PAPER_FINAL=false`.
