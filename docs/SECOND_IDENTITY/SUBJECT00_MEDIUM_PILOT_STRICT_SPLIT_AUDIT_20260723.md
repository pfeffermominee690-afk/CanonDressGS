# Subject00 medium-pilot strict-split audit

Status: `PASS`

`PAPER_FINAL=0`

- Train cameras: `[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]`.
- Held-out cameras: `[0, 4, 8, 12, 16, 20]`.
- Train poses: 1130.
- Held-out poses: 125.
- Buffer-excluded poses: 1245.
- Camera split SHA256: `8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44`.
- Pose split SHA256: `c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06`.
- Availability SHA256: `cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e`.
- Evaluation-order SHA256: `38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a`.
- Record-manifest SHA256: `865118c2f216046008925ef14b049db6e1d2921922117f6e3c3e3e2fdacd3537`.
- Data-order SHA256: `0f0e7463d9f9066f54e19ec31f85f722afd58e58aba123af144918fdc97639f8`.
- Theoretical/valid/official-missing records: 20,340/20,249/91.
- Unique/exposed-once/repeated: 20,249/20,249/0.
- Held-out-camera/held-out-pose/buffer intersections: 0/0/0.

Per-camera valid counts:

`1:1130, 2:1124, 3:1120, 5:1130, 6:1124, 7:1130, 9:1124, 10:1111, 11:1130,
13:1130, 14:1130, 15:1130, 17:1130, 18:1124, 19:1130, 21:1130, 22:1111, 23:1111`.

The record manifest preserves every exact record ID and every per-pose valid-camera count. No
missing pair was synthesized or replaced; no held-out or buffer record entered training.
