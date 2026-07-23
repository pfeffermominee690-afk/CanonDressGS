# Subject00 Surface-LBS Checkpoint Contract (2026-07-23)

The surface runtime checkpoint persists canonical xyz, all Gaussian attributes, cached per-Gaussian LBS weights, attachment-valid flags, face IDs, barycentric coordinates, component/region IDs, source labels, surface distances, runtime mode flags, attachment provenance, and runtime counters. The smoke wrapper additionally records the source branch, HEAD at write (`ce9cc076377c3224c12d9caa3ff7772cc5d8dbbf`), template SHA256, sampler SHA256, attachment-manifest SHA256, and `lbs_mode=surface_attachment_cached`. The later loader-only validation fix is commit `5aaa8cc73ba1d64df5c9a974ee209c891f57c204`; it did not rewrite the checkpoint.

The checkpoint is explicitly tagged `SMOKE_CHECKPOINT_WRITE` and `training_checkpoint=false`. It is not a training checkpoint and contains no optimizer or scheduler state.

A fresh process followed the production inference contract by initializing the frozen SMPL T-pose/big-pose constants and then calling strict `GaussianModel.restore`. The result contained exactly 200,000 Gaussians and 200,000 attachments. Cached weights and all attachment tensors were exact to checkpoint state. No legacy grid was loaded and no spatial weight query occurred.

The 12-view RGB/alpha/depth suite generated 36 arrays. Before/after checkpoint values were bitwise exact with `max_abs=0`, tighter than the frozen `1e-6` tolerance. The checkpoint is 701,865,298 bytes and has SHA256 `96b8091e1a23067eb7588499914a18a2c447c1333786aaa39a6cbe86d7e1d4a8`.

One failed fresh-process harness attempt is preserved: it omitted `init_smpl_pose` and stopped before render with no checkpoint mutation or additional checkpoint write. The correction mirrors `visualize.py` and `test.py`; it does not alter checkpoint tensors or scientific data.
