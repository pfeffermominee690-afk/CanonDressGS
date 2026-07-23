# Subject00 canary strict-split audit (2026-07-23)

## Result

`PASS`. The 384 training exposures exactly match the frozen record manifest
(`a64a8d40876946b8f0919a4761e7b814e9ca7182cd0434b0cf97a6b2737386ca`), with 384 unique pose-camera records and no repeat,
replacement, deletion or shuffle.

## Training exposure

- Poses: `[0, 17, 35, 75, 93, 122, 151, 169, 220, 260, 300, 329, 358, 397, 426, 477, 517, 546, 564, 593, 644, 673, 702, 775, 815, 855, 883, 923, 974, 1047, 1087, 1149, 1167, 1218, 1258, 1298, 1349, 1389, 1417, 1435, 1497, 1526, 1555, 1573, 1613, 1675, 1693, 1755, 1806, 1868, 1919, 1947, 2031, 2082, 2100, 2151, 2169, 2198, 2249, 2289, 2362, 2413, 2464, 2493]`
- Cameras: `[1, 5, 10, 14, 19, 23]`
- Held-out camera exposure: `0`
- Held-out pose exposure: `0`
- Buffer-pose exposure: `0`

## Fixed evaluation

- Train-eval poses: `[0, 673, 1555, 2493]`
- Heldout-eval poses: `[56, 931, 1763, 2499]`
- Train cameras: `[1, 5, 10, 14, 19, 23]`
- Heldout cameras: `[0, 4, 8, 12, 16, 20]`
- Four quadrants: `24 + 24 + 24 + 24 = 96`
- Query-order SHA256: `38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a`
- Availability SHA256: `cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e`
- Step 0 exact order: `PASS`
- Step 384 exact order: `PASS`

Pose 44 remains a held-out split member. It is absent only from this fixed
canary evaluation because `CAMERA_FRAME_INPUT_UNAVAILABLE`; no replacement
pose was selected.

Held-out results were not used for checkpoint selection, hyperparameter
changes, early stopping, extra steps or rerun decisions. `PAPER_FINAL=0`.
