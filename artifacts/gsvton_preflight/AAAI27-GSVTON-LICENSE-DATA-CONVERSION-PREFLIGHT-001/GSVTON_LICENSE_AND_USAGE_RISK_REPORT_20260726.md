# GS-VTON License And Usage Risk Report

Task ID: `AAAI27-GSVTON-LICENSE-DATA-CONVERSION-PREFLIGHT-001`

Audited official repository: `https://github.com/yukangcao/GS-VTON`

Pinned official commit: `96964b0a6528089123cc27a3ff3e3eb46505cf6e`

## Decision

`LICENSE_STATUS = LICENSE_NOT_SPECIFIED`

`EXECUTION_AUTHORIZED_BY_LICENSE = false`

No GS-VTON micro-canary execution, checkpoint download, dataset download, training, rendering, or data conversion is authorized by this preflight. Author permission or a different baseline path is required before any run.

## Evidence

- The official repository root at the pinned commit contains `README.md`, `main.py`, `requirements.txt`, `docs/`, `stage1/`, and `stage2/`, but no root `LICENSE`, `COPYING`, or `NOTICE`.
- The official README documents Python 3.8, CUDA 11.8, PyTorch 2.2.1, xformers 0.0.25, dependency installation, pretrained weights, data preparation, and the `main.py` command, but it does not grant a code license.
- GitHub repository license metadata was checked and did not identify a license.
- The author project page links paper and code and labels the work IJCV 2026, but does not include a code license statement.
- The arXiv page documents the paper and authorship, but no software usage grant was found.
- Nested licenses were found for vendored components. These do not license the GS-VTON repository as a whole and include restrictive upstream licenses for Gaussian Splatting and OpenPose.

## Dependency And Weight Risk

The official pipeline is a heavy diffusion plus 3DGS stack: IDM-VTON, BLIP2, RealFill LoRA on Stable Diffusion 2 inpainting, Stable Diffusion 1.5, Stable Diffusion 2.1 base, ControlNet/EditAnything, SD VAE, Detectron/DensePose, OpenPose, human parsing, CUDA rasterizers, tiny-cuda-nn, nvdiffrast, and Gaussian Splatting submodules.

Known public weight bytes from metadata-only checks are at least `51,622,353,397` bytes. This excludes unresolved gated/unauthenticated model sizes for `stabilityai/stable-diffusion-2-inpainting`, `stabilityai/stable-diffusion-2-1-base`, and the SharePoint `Self_Correction_Human_Parsing/logits.pt` file. No model body was downloaded.

## Subject02 Conversion Risk

The subject02 O03 micro-canary field mapping is draft-complete and read-only. It binds one O03 garment reference and one target condition (`cond_000347`) with source hashes. It does not copy, mutate, or convert data.

The target RGB and target mask are evaluation/target-space inputs only. The garment reference comes from the O03 reference-set cell and is not the target RGB or mask.

## Static 3DGS Initialization Risk

GS-VTON expects an initialized vanilla static 3D Gaussian Splatting data root with `point_cloud/iteration_30000/point_cloud.ply` plus COLMAP/Nerfstudio-style camera data. The MMLP-Human Base Avatar checkpoint and LBS grid are dynamic/avatar artifacts, not a direct replacement for that static 3DGS PLY and data root.

`MMLPHUMAN_INITIALIZATION_COMPATIBILITY = RETRAIN_STATIC_3DGS_REQUIRED`

## Final Classification

`GSVTON_MICRO_CANARY_PREFLIGHT_MULTIPLE_BLOCKERS`

The license gate blocks execution, and the static 3DGS initialization contract also blocks a direct subject02 run.

Unique next task:

`USER_OBTAIN_GSVTON_AUTHOR_PERMISSION_OR_SELECT_FULL_AVATAR_ONLY`

