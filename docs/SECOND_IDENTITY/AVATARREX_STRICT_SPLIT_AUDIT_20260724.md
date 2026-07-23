# AvatarReX Strict Split Audit

Task: `AAAI27-AVATARREX-BASE-AVATAR-PREFLIGHT-001`

## Camera Split

The frozen camera split regenerated exactly. Train cameras: `camera_00, camera_03, camera_04, camera_05, camera_06, camera_07, camera_08, camera_10, camera_12, camera_13, camera_14, camera_15`. Held-out cameras: `camera_01, camera_02, camera_09, camera_11`. Counts are `12/4`; overlap is zero. The camera split SHA256 is `cb4f4cab2ca7fb91f3f34d788159d5ab6aba2a52e235f29561014cbe5dddc5f2`.

Held-out cameras remain the fixed azimuth ranks `0/4/8/12`: `camera_01, camera_02, camera_09, camera_11`. No training result was used to choose or modify them.

## Pose Split

The frozen pose split regenerated exactly: `856` train, `95` held-out, and `950` buffer-excluded frames. Train/held-out and train/buffer overlaps are zero; temporal leakage is zero; minimum held-out temporal distance is `11`. The pose split SHA256 is `ec2197322e8743764e8a642c2d66c70460dbc94f8fb09494c2fefdaa111eee9b`.

Every camera has `1901` valid RGB/mask pairs and every admitted pose has `16` valid cameras. Available strict quadrant counts are `10272` train-view/train-pose, `3424` held-out-view/train-pose, `1140` train-view/held-out-pose, and `380` held-out-view/held-out-pose.

Combined frozen split content SHA256: `ec810c3a3d53681b2b8d646b74b67084d71eecfa073e03a228309e3458f9a6f1`. Future result-driven split changes are forbidden.
