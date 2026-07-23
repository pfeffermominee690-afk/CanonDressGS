# Subject00 Surface-LBS Runtime Smoke (2026-07-23)

The final smoke loaded the atomically published formal assets, not a private candidate directory. It constructed the actual `Scene` and `GaussianModel`, loaded 200,000 canonical surface Gaussians and 200,000 fixed 55-channel attachments, and did not load `lbs_weights_grid.npz`.

The 3×4 matrix was:

- canonical pose, frame-0 global placement;
- frame 1250;
- strict held-out frame 44;
- cameras 1, 4, 8, and 12, of which 4, 8, and 12 are strict held-out cameras.

All 12 RGB, alpha, depth, posed-xyz, and raster means2d tensors were finite. Alpha was nonempty in every view. No body explosion, bin overflow, warning, illegal-memory error, legacy-grid load, or forward spatial-weight query occurred. The minimum component-centroid separation was 0.0622617 m and the maximum posed bounding-box extent was 1.689003 m.

The frozen surface audit retained head/eye leakage=0 and hand/finger leakage=0. A manual visual spot check opened the held-out frame 44/camera 12 RGB, alpha, and depth images and confirmed coherent silhouette/depth mechanics. These observations are runtime-contract evidence, not a claim about trained appearance or reconstruction quality.

Pre-publish and post-publish renders were bitwise exact (`max_abs=0`). The post-publish model construction took 5.49 s, its warm 12-render suite took 3.76 s, and peak allocated VRAM was 800,973,824 bytes.

The zero-step L1 values are recorded only to prove a loss forward can be computed. No backward, optimizer, scheduler, training loop, or training checkpoint was executed.
