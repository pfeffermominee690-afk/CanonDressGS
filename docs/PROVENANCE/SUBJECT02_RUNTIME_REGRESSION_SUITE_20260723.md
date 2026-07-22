# Subject02 Runtime Regression Suite

Date: 2026-07-23
Mode: inference only

## Frozen input contract

- Frame IDs: `0..49`
- Camera: `18`
- Resolution: `1150 x 1330`
- White background
- Seed: `0`
- Frame-manifest SHA256: `9e17a2a0eeb4010f9c4b74d6030b219c2463c796186a74483e1da2bdab339731`
- Checkpoint SHA256: `abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`

## Checkpoint contract

Both the isolated cloud dirty candidate and V2 passed config parsing, dataset mapping, strict checkpoint restore, 42-key/version-2/iteration-100000 checks, 200,000-Gaussian count, parameter shape checks, and finite checks.

## Dirty-vs-clean 50-frame result

- RGB exact: 50/50
- Alpha exact: 50/50
- Projected-depth auxiliary exact: 50/50
- Gaussian count exact: 50/50
- Visible-Gaussian count exact: 50/50
- Output shapes exact: 50/50
- Warning set exact: PASS
- Maximum and median per-frame L1 difference: 0
- Maximum and median per-frame PSNR difference: 0
- Status: PASS

Per-frame dirty and V2 hashes are retained in `paper_protocol/second_identity/subject02_runtime_50frame_regression.json`; full process records and logs remain in the external attempt directory.

## Archived scalar regression

- Archived L1: `0.007736183702945709`
- Observed L1: `0.007752127721905708`
- Absolute delta: `1.5944018959998876e-05`
- Frozen tolerance: `5e-5`
- Archived PSNR: `24.59844398498535`
- Observed PSNR: `24.582090644836427`
- Absolute delta: `0.016353340148924644 dB`
- Frozen tolerance: `0.05 dB`
- Gaussian count: 200000 exact
- Status: PASS

Historical RGB/alpha tensors were not archived. The scalar test demonstrates tolerance parity, not exact historical render identity.

## Fresh-process determinism

Two new V2 Python processes produced identical checkpoint hash, warning set, Gaussian/visible-Gaussian counts, and all 50 RGB/alpha/depth hashes. This establishes same-environment fresh-process determinism. It does not assert gsplat CUDA bitwise determinism across different GPU, driver, Torch, CUDA, or gsplat versions.

## Mutation accounting

Training, backward, optimizer construction, optimizer step, scheduler step, and checkpoint write counts are all zero. Data and formal-output mutation counts are zero.
