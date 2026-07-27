# AvatarReX Zero-Copy Standardization

The adapter uses `READ_ONLY_ZERO_COPY_LOADER_ADAPTER` over `avatarrex_lbn1`. It resolves 16 camera directories, 1901 RGB frames and 1901 `mask/pha/*.jpg` masks per camera, reads calibration and SMPL-X arrays in memory, and decoded only nine smoke records. Loader smoke status is `PASS`.

Camera centers use `C=-R^T T`; cameras are centered on the rig mean and sorted by `(azimuth,camera_index)`. Held-out ranks 0/4/8/12 yield `camera_01, camera_02, camera_09, camera_11`. The pose split uses seed 20260723, standardized rotation-6D descriptors, 95 farthest-point held-out frames, at least 11-frame held-out spacing, and radius-5 buffers. Counts are 856 train, 95 held-out, and 950 excluded buffer frames with zero overlap.

No RGB, masks, calibration, or SMPL-X data were copied or rewritten. Upstream license confirmation is required before redistribution or derived generation.

After the loader smoke, all 60834 raw files (19135049684 bytes) were rehashed. The full-content tree fingerprint remained `00482b7c98f6f46773fd13a3f33ebe278b9353e09fdb51fa9ed72583f7b27b15`; the retained archive and both metadata files also matched their frozen SHA256 values.

Readiness: `AVATARREX_ZERO_COPY_STANDARDIZATION_READY`; base-avatar status: `AVATARREX_BASE_AVATAR_PREPARATION_REQUIRED`.
