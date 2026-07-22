# Subject00 Strict Novel-View / Novel-Pose Protocol V2 — 2026-07-23

## Status

`FROZEN_PROPOSED_NOT_EXECUTED`. This document freezes the protocol before any subject00 preprocessing or training. It does not claim that any quadrant has passed.

## Novel-view split

For each calibrated camera, compute `C=-R^T T`, subtract the mean of all 24 centers, compute azimuth/elevation, sort by `(azimuth,camera_id)`, and hold out ranks 0, 4, 8, 12, 16 and 20. The resulting azimuth order is `12..23,0..11`.

- Train cameras (18): `[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23]`
- Held-out cameras (6): `[0,4,8,12,16,20]`
- Overlap: 0
- Split SHA256: `8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44`

Held-out cameras are forbidden for avatar training, garment teacher training, garment basis construction and controller references. They are target-only for strict novel-view evaluation. Historical subject02 camera 18 was view-transductive because it participated in training; it is not reused as a privileged subject00 split rule.

## Novel-pose split

Convert body and global axis-angle rotations to rotation-6D. Standardize body dimensions separately; standardize global orientation separately and concatenate it with weight 0.25. With seed `20260723`, apply greedy farthest-point sampling for 125 held-out poses, enforcing at least 11 frames between held-out selections. Exclude a ±5 temporal buffer around each held-out frame from training.

- Total/valid frames: 2,500 / 2,500
- Train frames: 1,130
- Held-out frames: 125
- Buffer-only excluded frames: 1,245
- Train/held-out overlap: 0
- Train/buffer overlap: 0
- Minimum held-out pair temporal distance: 11
- Minimum held-out-to-train descriptor distance: 1.5116555691
- Split SHA256: `c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06`

The exact frame lists are sealed in `paper_protocol/second_identity/subject00_novel_pose_split_v2.json`.

## Four evaluation quadrants

All requested targets are filtered through the 60,000-entry availability manifest; an official missing pair is excluded with its recorded reason, never substituted.

| Quadrant | Target pose | Target camera | Avatar/teacher/reference boundary | Ground truth |
|---|---|---|---|---|
| A `SEEN_POSE_SEEN_VIEW` | train frames | train cameras | only train frames/cameras | available valid pairs |
| B `SEEN_POSE_NOVEL_VIEW` | train frames | held-out cameras | held-out camera pixels/calibration targets forbidden upstream | available valid pairs |
| C `NOVEL_POSE_SEEN_VIEW` | held-out frames | train cameras | held-out and buffer poses forbidden upstream | available valid pairs |
| D `NOVEL_POSE_NOVEL_VIEW` | held-out frames | held-out cameras | both pose and view boundaries apply | available valid pairs |

Garment teacher, garment basis and controller reference inputs must use only train frames and train cameras. Target-pose parameters and target-camera calibration may be supplied at inference because they define the requested output; target RGB/mask may be used only for evaluation. No held-out RGB/mask can drive optimization, reference selection or model choice.

Primary metrics are masked/full-image PSNR, SSIM and LPIPS, reported per quadrant with counts and missing-pair exclusions. Visual failure review must preserve silhouette, cloud/mottle, color contamination and temporal/pose discontinuity evidence. Metric or renderer changes require a new protocol version.
