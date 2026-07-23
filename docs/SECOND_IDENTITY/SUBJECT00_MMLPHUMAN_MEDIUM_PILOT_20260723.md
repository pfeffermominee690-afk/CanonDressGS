# Subject00 MMLPHuman one-pass medium pilot

Task: `MMLPHUMAN-SUBJECT00-ONE-PASS-MEDIUM-PILOT-001`

Decision: `SUBJECT00_MMLPHUMAN_MEDIUM_PILOT_PASS`

Body-surface representation: `APPEARANCE_AND_SILHOUETTE_EMERGING`

`PAPER_FINAL=0`

## Scope and provenance

This archive records one completed `ONE_FULL_VALID_STRICT_TRAIN_PASS`. It began from the frozen
canary step-0 checkpoint, not canary step 384, and it did not select a best checkpoint. The formal
source is `8c69cdce0139532a33b1d4f839da7467784270e6` on `research/mmlphuman-subject00-short-canary-from-repaired-contract-20260723`; the canary runtime-code
head is `b0e8096589fe18069d95d2137e1db3979b4fe89f`, and the medium execution source head is
`36c43d9dd29abfe546a5352b46378244f3bd9f1b`. Historical provenance remains `LIMITED_HISTORICAL_PROVENANCE`.

The frozen record set contains 20,249 unique valid records out of 20,340 theoretical combinations;
91 pairs are official-missing. Every valid record was exposed exactly once, with no shuffle,
oversampling, early stopping, second pass, result-conditioned extension, or training resume.

## Initialization and strict splits

- Step-0 checkpoint: `29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a`, 701,938,720 bytes.
- Step/data-order position: 0/0.
- Gaussian/attachment/LBS: 200000/200000/[200000, 55].
- Camera split SHA256: `8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44`.
- Pose split SHA256: `c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06`.
- Availability SHA256: `cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e`.
- Evaluation-order SHA256: `38476b7d9f6a012e0e21d7f188c5db5131506ec2231e8a7b50a389c7c92ad98a`.
- Record-manifest SHA256: `865118c2f216046008925ef14b049db6e1d2921922117f6e3c3e3e2fdacd3537`.
- Data-order SHA256: `0f0e7463d9f9066f54e19ec31f85f722afd58e58aba123af144918fdc97639f8`.
- Held-out-camera/held-out-pose/buffer intersections: 0/0/0.

## Training evidence

- Forward/backward/optimizer: 20249/20249/20249.
- Total-loss first/last 5% medians: 0.008396284189075 /
  0.007115250686184 (15.257148% reduction).
- L1 first/last 5% medians: 0.008318810723722 /
  0.002171005820855
  (73.902450% reduction).
- LPIPS training loss first became nonzero at step 6001;
  it was nonzero for 14,249 steps.
- All loss, gradients, intended-gradient, and optimizer-state checks were finite/nonzero as required.
- Topology mutation, legacy-grid load, spatial-LBS query, and off-surface rebind counts were all zero.
- Wall time: 4338.664056 seconds; peak VRAM: 1,710,931,456 bytes.

## Evaluation

Step0 global: RGB MAE=0.012293901076, PSNR=23.287589, SSIM=0.970493633921, LPIPS=0.137394524412, IoU=0.837127586182, boundary F=0.502679918935, alpha=0.041629045440, depth-finite=1.000000, foreground=0.039630931949.

Canary-step384 global: RGB MAE=0.010658793627, PSNR=24.376180, SSIM=0.973509568100, LPIPS=0.121645711906, IoU=0.883148662472, boundary F=0.676065805752, alpha=0.041879549417, depth-finite=1.000000, foreground=0.039801140078.

Medium-final global: RGB MAE=0.003577645001, PSNR=30.588665, SSIM=0.981692799677, LPIPS=0.063285196394, IoU=0.915549589663, boundary F=0.803278516131, alpha=0.045975502615, depth-finite=1.000000, foreground=0.041880543751.

Medium final improved against step0 for 96/96 queries:

- LPIPS 0.137394524412 -> 0.063285196394
  (-53.939070% relative change).
- RGB MAE 0.012293901076 ->
  0.003577645001
  (-70.899026% relative change).

Medium final also improved against the secondary canary-step384 reference for 96/96 queries:

- LPIPS relative change: -47.975810%.
- RGB MAE relative change: -66.434804%.

All four quadrants improved mean LPIPS against step0; TT improved for 24/24 queries. Held-out
relative LPIPS changes were HH=-51.673866%,
HT=-51.894109%, and
TH=-55.491666%. All
96 final outputs were finite and alpha-nonempty, with zero body explosions.

### Step0 quadrants

| quadrant | RGB MAE | PSNR | SSIM | LPIPS | silhouette IoU | boundary F | alpha | depth finite | foreground |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TT | 0.011807970179 | 23.442235 | 0.971340062718 | 0.137431931061 | 0.831840447736 | 0.490933412500 | 0.040417783589 | 1.000000 | 0.038482401656 |
| TH | 0.012222244559 | 23.211664 | 0.970723723372 | 0.140407407967 | 0.832276605554 | 0.490558142644 | 0.040946469434 | 1.000000 | 0.038970415168 |
| HT | 0.012393549706 | 23.306286 | 0.970305137336 | 0.134843640961 | 0.840374285262 | 0.514365151318 | 0.042276588210 | 1.000000 | 0.040242399477 |
| HH | 0.012751839861 | 23.190172 | 0.969605612258 | 0.136895117660 | 0.844019006177 | 0.514862969278 | 0.042875340525 | 1.000000 | 0.040828511496 |

### Canary-step384 quadrants

| quadrant | RGB MAE | PSNR | SSIM | LPIPS | silhouette IoU | boundary F | alpha | depth finite | foreground |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TT | 0.010213863881 | 24.530204 | 0.974413285653 | 0.121086820339 | 0.879926152803 | 0.672178613428 | 0.040626975046 | 1.000000 | 0.038623896698 |
| TH | 0.010628040550 | 24.263237 | 0.973463989794 | 0.125204590770 | 0.876049622478 | 0.648939477773 | 0.041204478588 | 1.000000 | 0.039142802659 |
| HT | 0.010724651821 | 24.427771 | 0.973435506225 | 0.119086650821 | 0.887445933207 | 0.693930459470 | 0.042542116160 | 1.000000 | 0.040415113872 |
| HH | 0.011068618255 | 24.283508 | 0.972725490729 | 0.121204785692 | 0.889172941401 | 0.689214672337 | 0.043144627874 | 1.000000 | 0.041022747085 |

### Medium-final quadrants

| quadrant | RGB MAE | PSNR | SSIM | LPIPS | silhouette IoU | boundary F | alpha | depth finite | foreground |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TT | 0.003263216131 | 31.194267 | 0.982947659989 | 0.059623935415 | 0.920359711266 | 0.827293437108 | 0.044568241255 | 1.000000 | 0.040604010025 |
| TH | 0.003425096574 | 30.745172 | 0.982366723319 | 0.062492998317 | 0.917554977251 | 0.811317777215 | 0.045295357960 | 1.000000 | 0.041223983873 |
| HT | 0.003771929614 | 30.169449 | 0.980986965199 | 0.064867734288 | 0.910810651678 | 0.786621973382 | 0.046673014057 | 1.000000 | 0.042531437289 |
| HH | 0.003850337685 | 30.245773 | 0.980469850202 | 0.066156117556 | 0.913473018456 | 0.787880876818 | 0.047365397189 | 1.000000 | 0.043162743816 |

## Representation decision

The automatic evidence and 96/96 original-detail manual review support
`APPEARANCE_AND_SILHOUETTE_EMERGING`. Medium outputs consistently show a blue/white hoodie and dark
jeans/shoes rather than the nearly uniform gray step0/canary body. Mean silhouette IoU is
0.915549589663, mean boundary F-score is
0.803278516131, mean foreground RGB variance is
0.083380051112, and mean foreground chroma is
0.070652318459.

The limitation is material and is not hidden: loose hoodie bulk, sleeve volume, and hem offset
remain underrepresented by the body-surface fallback. Garment-proxy undercoverage averages
0.051245250758; fine face/eye and hand/finger detail is limited.
This is emerging appearance and silhouette, not proof of complete loose-clothing geometry.

Manual review found zero severe head/eye contamination, hand/finger contamination, detached
clouds, empty/black renders, full-frame opacity, camera mismatch, component separation,
silhouette collapse, or body explosion.

## Checkpoint and accounting

The final checkpoint is `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-MEDIUM-PILOT-001/attempt_001/checkpoints/step_020249.pth` with SHA256
`480805878f1dba3d0748ab0b31d1e18c68fdd3f42267d92cf4763f97c0d1f443`. Fresh-process roundtrip passed: 8 queries/24 arrays were
exact, with maximum absolute difference 0.0.

Counts: one training run; one optimizer; 20,249 forwards/backwards/steps; one reused step-0 pointer;
four new checkpoints; 216 primary logical renders; 96 secondary references; 120 new renders; 192
reused renders; 96 manual comparison sheets. The separate roundtrip rendered 8 fixed queries. The
sealed external attempt contains 250 files totaling 3,229,325,143 bytes.

All protected raw data, derived assets, subject02 assets, AvatarReX archives, canary outputs,
historical attempts, splits, template, attachment, cached LBS, and frozen parameters remained
unchanged. No formal long training or next task was started.

## Next task

`NEXT_TASK=DESIGN_SUBJECT00_FORMAL_STRICT_SPLIT_TRAINING_PROTOCOL`. This mapping is recorded only; it was not started.
