# Subject00 MMLP-Human Cost Estimate V2 — 2026-07-23

## Evidence base

This is a bounded estimate, not a launch authorization. The historical reference is the **subject02 formal MMLP-Human run**, not an “800k-step training.” Its configured output directory contains an archived iteration-100,000 checkpoint. The run used the same 1,330×1,150 image resolution and 200,000 initial Gaussians on the same RTX 4090 environment.

Observed subject02 evidence:

- output tree: 5,838,497,130 bytes;
- checkpoint sizes: 710,856,844 bytes at iteration 2,000 and approximately 1.7066 GB each at iterations 6,000, 50,000 and 100,000;
- run start log: 2026-07-13 05:18; iteration-100,000 checkpoint mtime: 2026-07-14 02:38, an elapsed wall-clock upper-bound observation of about 21 h 20 min;
- early iterations ran much faster than the post-LPIPS regime, so linear extrapolation is uncertain;
- subject00 proposed training uses 18 cameras and 1,130 poses, versus historical subject02's 24 cameras and 2,115 configured frames, but one training iteration still renders one full-resolution sample.

## Bounds

| Component | Lower bound | Upper bound | Unknowns |
|---|---:|---:|---|
| Data/schema/loader validation | 20 s observed | 2 min | filesystem cache state |
| SMPL-X fallback template | <1 min observed in memory | 5 min | serialization and repeat validation |
| LBS grid, one run | unknown | unknown | no sealed generation log; 55 external solves |
| LBS repeatability gate | 2 complete runs | 2 complete runs | solver runtime and 12-thread byte determinism |
| Formal 100k-equivalent training | ~21 GPU-h historical observation | 30 GPU-h planning bound | validation/checkpoint cadence and contention |
| Configured 800k horizon | ~170 GPU-h linear lower planning bound | 240 GPU-h planning bound | later-stage loss cost and evaluations |
| Strict quadrant rendering | 30,120 valid-grid candidates before missing filtering | unknown | measured render throughput is pending derived assets |

No exact peak-VRAM measurement survived. The historical run completed checkpoints on a 24,564 MiB RTX 4090, so the defensible statement is `peak VRAM < device capacity`; a lower bound and safety margin remain unknown until the post-asset forward/render smoke.

## Storage planning

- Preserve at least 10 GiB free before derived-asset staging; two independent LBS staging roots and solver text/grid intermediates are required.
- The observed subject02 output is 5.84 GB through its surviving checkpoint set. At ~1.71 GB per mature checkpoint, retain only explicitly scheduled checkpoints and budget at least 15–25 GB for model checkpoints, logs and validation products at a longer horizon.
- Raw subject00 already occupies 26,459,647,641 bytes and must not be duplicated casually.
- Current cloud free space at audit time was 79,049,592,832 bytes; recheck immediately before preprocessing/training.

## Resume policy

Resume is allowed only from a checkpoint whose source HEAD, config, split hashes, raw fingerprint, template/LBS hashes, Gaussian count, optimizer keys and scheduler state all match. Never relabel the historical subject02 checkpoint as a subject00 initialization or reproduction. Unknown components must be measured in the next authorized preprocessing task before a training budget is accepted.
