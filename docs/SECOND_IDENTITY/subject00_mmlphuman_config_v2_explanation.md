# Subject00 preflight V2 config explanation

`config/subject00_preflight_v2.yaml` is deliberately blocked for formal training.

It records the subject00 raw root and fingerprint, image/mask/calibration/SMPL paths, the availability manifest, frozen camera and pose split manifests, exact V2 source provenance, and the pending template/LBS/initial-point outputs. Its safety sentinels are:

- `preflight_only: true`
- `training_enabled: false`
- `derived_asset_state: PENDING_DETERMINISTIC_PREPROCESSING`
- `iterations: 0`
- `num_train_frame: 0`
- three `PENDING_MANUAL_CONFIRMATION` gates

The dedicated validator rejects formal launch whenever the preflight sentinels or pending asset fields are present; that rejection test passed. The frozen legacy `train.py` does **not** inspect these fields and therefore must not be invoked with this draft. This limitation is explicit because modifying `train.py` would violate the V2 15-file runtime closure.

The camera lists are complete: 18 train cameras and six geometric held-out cameras. Exact train/held-out/buffer pose IDs live in `paper_protocol/second_identity/subject00_novel_pose_split_v2.json` because the legacy config model only supports contiguous ranges. A future runnable config must be created only after deterministic assets exist and after a split-aware training entry has been authorized; silently converting the 1,130 non-contiguous train poses into a contiguous range is forbidden.

Current decision: `BLOCK_FORMAL_TRAINING`. Next authorized task: `PREPARE_SUBJECT00_MMLPHUMAN_DERIVED_ASSETS_WITHOUT_TRAINING`.
