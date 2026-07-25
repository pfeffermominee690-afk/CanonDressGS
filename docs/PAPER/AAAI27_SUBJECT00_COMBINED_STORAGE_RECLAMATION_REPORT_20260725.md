# Subject00 Combined Storage Reclamation Report

Task: `AAAI27-SUBJECT00-COMBINED-STORAGE-RECLAMATION-EXECUTION-001`

## Decision

The authorized `PLAN_C` plus `PLAN_CKPT_D` execution reached the recommended storage target.

- Final classification: `SUBJECT00_COMBINED_RECLAMATION_45GIB_TARGET_PASS`
- Initial cloud free: `17,016,557,568` bytes
- Final cloud free: `59,215,609,856` bytes (`55.149 GiB`)
- Net free increase: `42,199,052,288` bytes (`39.301 GiB`)
- Margin over 45 GiB: `10,897,227,776` bytes
- `PAPER_FINAL=false`

## Archive Results

AvatarReX was archived under `E:\canondressgs_archive\cloud_datasets\AvatarReX`.

- 3 items
- 60,854 regular files
- 31,835,709,969 logical bytes
- File count, total bytes, and per-file SHA256: `PASS`
- Source unchanged before retirement: `PASS`

Three complete old checkpoint experiments were archived under `E:\canondressgs_archive\cloud_checkpoints`.

- `diffusion_piper_50k`
- `diffusion_30k`
- `irregular_diffusion_30k`
- 28 regular files, 3 preserved `checkpoints/last` symlinks
- 9,735,450,405 logical bytes
- File count, total bytes, per-file SHA256, and symlink mapping: `PASS`

## Deletion Results

- Regular files deleted: 60,952
- Logical file bytes deleted: 42,082,833,064
- Duplicate-safe files: 70 / 70, 293,877,819 bytes
- Optimizer-only files: 1 / 1, 219,104,837 bytes
- Unlisted deletions: 0
- Source-changed candidates skipped: 0

All six archive-source deletions passed the per-item release-delta threshold. Five source roots are fully absent. The AvatarReX `avatarrex_lbn1` root is intentionally present with one protected file after the correction below.

## Gate Correction

The original pre-deletion gate returned `PASS`, but its protected-inventory intersection included checkpoint archive roots and omitted AvatarReX roots. The post-reclamation protection audit detected that:

`/root/autodl-tmp/datasets/avatarrex_second_dataset_staging/avatarrex_lbn1/smpl_params.npz`

was also classified `SEALED_PROVENANCE_KEEP`.

The exact file was restored from the verified Windows archive to its original cloud path:

- Bytes: 1,309,966
- SHA256: `6ed3b3877d3895999e2636990bd417783328d695412b45d0679773e34b54613b`
- Corrected protection audit: `PASS`

The gate implementation now includes AvatarReX roots in the protected checkpoint inventory intersection. The initial failed protection audit remains preserved in the external execution logs.

## Protection Results

- `CRITICAL_ACTIVE_KEEP`: 670, all SHA-matched
- `SEALED_PROVENANCE_KEEP`: 2,344, all SHA-matched
- `UNIQUE_FINAL_KEEP`: 15, all SHA-matched
- `UNKNOWN_KEEP`: 622, all SHA-matched
- Paper or figure checkpoints: 669, intact
- Figure Bank checkpoints: 343, intact
- Subject02 images: 74,640 files, exact tree stats
- Subject00 derived assets: 1,634 files, exact tree stats
- Subject00 raw: 119,412 files, exact tree stats
- Short-canary SHA: `29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a`

No Formal attempt was created, no training or optimizer step ran, and no paper result was modified.

## Git State

- Execution branch: `research/subject00-storage-reclamation-execution-20260725`
- Reclamation result head: `3183cbedf338d9bbe50b41110daa0e256adc4809`
- Final reporting head: `SELF_ON_FINAL_REPORTING_COMMIT`
- Final local/origin/cloud equality is checked after the reporting commit.

## Next Task

`RUN_SUBJECT00_FORMAL_BASE_FULL_PREFLIGHT_AND_FREEZE_EXECUTION_HEAD`
