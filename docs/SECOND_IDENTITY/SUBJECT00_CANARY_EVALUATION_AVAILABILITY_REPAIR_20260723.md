# Subject00 canary evaluation availability repair

Task: `MMLPHUMAN-SUBJECT00-CANARY-CONTRACT-REPAIR-001`

The repair uses `AVAILABILITY_FILTER_THEN_EQUAL_SPACING`. It does not use image
quality, masks sizes, model outputs, future renders, or hand-selected
replacement poses.

The frozen availability authority is `/root/autodl-tmp/datasets/thuman4_second_identity_staging/reports/SUBJECT00_VALID_FRAME_CAMERA_MANIFEST.json`. Its formal SHA256
is `cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e`, computed from the `entries` array using canonical
compact sorted-key JSON. The whole-file raw SHA is recorded separately and is
not the protocol identity because the file also contains its summary and the
canonical hash declaration.

Evaluation cameras are the exact union
`[0, 1, 4, 5, 8, 10, 12, 14, 16, 19, 20, 23]`. A pose is eligible only when RGB exists, mask exists,
and `valid_pair=true` for all 12 cameras.

- Eligible train-evaluation poses: 64
- Eligible held-out-evaluation poses: 118
- Repaired train poses: `[0, 673, 1555, 2493]`
- Repaired held-out poses: `[56, 931, 1763, 2499]`
- Equal-spacing indexes: train `[0, 21, 42, 63]`, held-out `[0, 39, 78, 117]`
- Four quadrant counts: `24/24/24/24`
- Total availability: `96/96`, missing `0`, duplicate `0`
- Query-order SHA256: `38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a`

Pose 44 remains a member of the frozen held-out pose split. It is not invalid,
deleted, or reclassified. It is excluded only from this short canary fixed
96-query evaluation set because cameras 0, 10, and 23 have official missing RGB
and mask inputs (`CAMERA_FRAME_INPUT_UNAVAILABLE`).

No optimizer, training forward, backward, checkpoint, renderer, or visual sheet
was created by this repair.
