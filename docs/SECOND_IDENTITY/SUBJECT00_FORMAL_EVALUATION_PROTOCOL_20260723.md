# Subject00 formal strict evaluation protocol

This protocol freezes all evaluation choices before formal training results exist. Split hashes are camera `8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44`, pose `c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06`, and availability `cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e`.

## Complete availability-filtered sets

| Quadrant | Theoretical | Valid | Missing | Role |
| --- | ---: | ---: | ---: | --- |
| TRAIN_FIT | 20340 | 20249 | 91 | separate fit audit |
| STRICT_NOVEL_VIEW | 6780 | 6749 | 31 | primary strict result |
| STRICT_NOVEL_POSE | 2250 | 2217 | 33 | primary strict result |
| STRICT_NOVEL_POSE_AND_VIEW | 750 | 739 | 11 | primary strict result |

Every exact ID, missing record, per-camera count, per-pose count, query-order SHA, and manifest SHA is in `subject00_formal_evaluation_manifests.json`. Buffer-pose and replacement counts are zero. TRAIN_FIT is never pooled into the three primary strict averages.

Full four-quadrant streaming evaluation occurs only at fixed final step 101,245. The secondary step100,000 evaluation is uniquely frozen to fixed96 because the complete set contains 29,954 queries and full-final evidence has priority; this resource choice cannot change after results. The fixed96 trajectory covers steps `[0, 20249, 40498, 60747, 80996, 100000, 101245]` with source order SHA `38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a`. Reuse of step0 is hash-gated; medium-step reuse is permitted only under exact model-state, renderer, asset, and query parity.

For each full query, render, compute RGB MAE, PSNR, SSIM, LPIPS, silhouette IoU, boundary F, alpha occupancy, depth finite ratio, foreground coverage, render time, and peak VRAM, append a lightweight record, then release tensors. Report mean, median, standard deviation, p05/p25/p75/p95, per-camera, per-pose, query records, and invalid/failed counts. Full-evaluation PNG count is zero except the separately frozen visual subset.

All 96 original-detail review sheets must later be opened and persisted. Each compares GT, step0, medium20249, formal100000, formal101245, alpha, depth, metrics, and identifiers. Extra cases use only preregistered numeric sorts for silhouette, LPIPS, body-conforming bias, sleeve/bulk, hem, and face/hand; post-hoc human cherry-picking is forbidden.
