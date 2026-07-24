# AAAI-27 Subject00 Second-Identity Evaluation Plan

Task: `AAAI27-SUBJECT00-MINIMAL-SECOND-IDENTITY-DATASET-CONTRACT-001`

Status: `FROZEN_BLOCKED_CONDITION_RESOURCE_AND_FORMAL_BASE`

## Endpoint and coordinate-system contract

O01, O03, and O04 each receive an independent 1,200-step Teacher Endpoint from exact-zero residuals on the same frozen Subject00 formal support. Each teacher consumes eight accepted strict-train targets and must independently pass finite numerics, endpoint parity, identity contamination zero, component contamination zero, and visual review.

The three Teacher residual vectors are converted to an `ENDPOINT_COORDINATE_SYSTEM`, not a garment manifold. Mean, centering, SVD, projection, and reconstruction use float64. Centering applies strict zero-sum closure. The rank is exactly `r=min(numerical_rank,2)`. Every reconstructed endpoint must pass channel-wise reconstruction reporting and renderer parity against its full Teacher Endpoint. LOO adaptation and continuous-manifold claims are forbidden.

## Subject00 endpoint controller

The controller retains the Subject02 pure-endpoint structure without transferring Subject02 state:

`reference RGB/mask -> frozen F2 -> per-reference masked mean/max -> validity-aware set mean/max -> LayerNorm(512) -> Linear(512,r) -> nearest Subject00 endpoint -> frozen Subject00 renderer`

F2 has 128 spatial channels. Mean/max produces 256 dimensions per reference; set mean/max produces 512 dimensions. The only architecture change is coefficient dimension 4 to at most 2. At rank 2 the controller has 2,050 trainable parameters. LayerNorm initializes to weight 1/bias 0 and Linear to weight 0/bias 0, with no output activation.

The target is the train-only standardized Subject00 endpoint coordinate. Loss is mean SmoothL1 beta 1 on coefficients only; classification, pairwise geometry, and render losses are zero. Snapping uses squared L2 to O01/O03/O04 endpoint coordinates with frozen garment-order tie-breaking. Continuous predictions are not rendered by the primary method.

The planned optimizer is Adam, learning rate 0.02, no weight decay, fixed LR, gradient clip 5, and 300 steps. Four condition rotations times seeds 0/1/2 yield 12 planned runs and 3,600 optimizer steps. Milestones are 0/20/50/100/200/300; only step 300 is evaluated. Early stop, threshold selection, best checkpoint, and best seed are forbidden.

Complete reference dropout, nonfinite features/predictions, shape/fingerprint mismatch, failed endpoint parity, or failed identity safety cannot silently choose a garment. The safe outcome is stop or `ABSTAIN` with Base Avatar.

## Split principle and current gap

The structural rotations are train folds `[0,1]`, calibration `2`, test `3`, then cyclic shifts. Exact Subject00 fold records are not frozen. The same image or SHA may not appear in adaptation and test or in calibration and test, no split may move after quality inspection, and all garments must share one protocol.

The sealed three-garment protocol supplies only four cardinal references per garment, and those are the same assets used by the endpoint bank. It does not supply separate SHA-disjoint controller train/calibration/test pools. Therefore controller training/evaluation remains blocked as `SUBJECT00_CONDITION_RESOURCE_GAP`.

## Metrics and baselines

Base Avatar reports RGB MAE, PSNR, SSIM, LPIPS, and silhouette IoU under the strict formal quadrants. Teacher Endpoints add garment render metrics, boundary F1, endpoint parity, identity safety, component safety, and full visual review.

The endpoint controller reports clean top-1, per-garment precision/recall, exact endpoint match, coefficient diagnostics, identity contamination, component contamination, wrong-outfit endpoint, and empty render per rotation and replicate.

The required comparators are Reference Classifier Lookup, Nearest-Centroid Lookup, Outfit-ID Oracle, and full Teacher Endpoint. The first two use the same F2 inputs and folds; Oracle and Teacher are non-deployable upper bounds. The paper must report whether CanonDressGS-Endpoint beats, matches, or loses to hard lookup. Superiority is not assumed, and historical Subject02 numbers cannot substitute for Subject00 results.

Qualitative figures cover Base Avatar, every Teacher Endpoint, both hard lookups, CanonDressGS-Endpoint, identity/component overlays, reference swap/dropout, and all sealed visualization conditions without cherry-picking or retouching.

## Resource forecast

The standard candidate plan is 48 calls for 24 accepted slots under an explicit 50% planning-yield assumption. It requires at least 96 independent initial review decisions plus possible adjudication and three garment-group reviews. Decoded RGB storage for 48 1024 x 1536 uint8 images is 216 MiB; decoded accepted masks are 36 MiB, or 72 MiB if masks for all candidates are retained. PNG compression, provider response storage, checkpoint storage, renderer count, and GPU hours remain unmeasured. API cost is `PRICE_NOT_VERIFIED`.

No generation, review, training, rendering, or evaluation occurred in this task.
