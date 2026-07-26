# AAAI-27 Subject00 Three-Garment Dataset Contract

Task: `AAAI27-SUBJECT00-MINIMAL-SECOND-IDENTITY-DATASET-CONTRACT-001`

Status: `FROZEN_WITH_CONDITION_RESOURCE_GAP`

## Identity and wardrobe

The identity is THuman4.0 `subject00`. Raw Subject00 data are immutable and cannot be copied into Git. The wardrobe is exactly `O01/O03/O04`; all semantics below are copied from the sealed three-garment/shared-wardrobe protocol.

### O01

Light heather-gray pullover hooded sweatshirt with hood, drawstrings, long sleeves, ribbed cuffs and waist hem, paired with black full-length straight trousers. The upper fit is regular-to-relaxed and the lower fit is regular straight. The visual material evidence is sweatshirt knit/fleece above and matte woven below. A zip-front hoodie, short sleeves, blue trousers, double hood, or retained old cuffs is a semantic failure.

### O03

Dark navy tailored single-breasted two-button business suit with matching full-length tailored straight trousers, a white dress shirt, and a dark navy tie. The jacket is hip-length and all sleeves are long. A casual jacket, missing tie, mismatched trousers, open bare chest, or short sleeves is a semantic failure.

### O04

Black waist-length zip-front bomber jacket with long sleeves over a light-gray crew-neck shirt, paired with medium-blue full-length straight denim jeans. A hood, long coat, black trousers, formal blazer, or distressed/torn jeans is a semantic failure.

The garment evidence grade is `VISUAL_APPEARANCE_AUDIT_ONLY; FIBER_CONTENT_NOT_VERIFIED`. Reference evidence is the frozen mapping `E:/data_pre/audit_subject02_outfit_conversion_pilot_v1/outfit_cell_mapping_v2_normalized.json` and two historical front sheets under `E:/data_pre/outputs/gpt_image2_pose_outfit_sheets/Jay/`. Only garment semantics, prompts, view slots, screening rules, and endpoint count are shared; donor identity pixels, donor face/body shape, target images, and raw frame IDs are never shared.

The sealed protocol does not define a per-garment local-occlusion whitelist. This field is recorded as `NOT_SPECIFIED_BY_SEALED_PROTOCOL` and is a blocking gap, not filled by prompt inference.

## Exact slot decomposition

Each garment has eight endpoint target slots in this order:

1. `front` at 0 degrees, also a garment reference.
2. `front_left_three_quarter` at 45 degrees.
3. `front_right_three_quarter` at 315 degrees.
4. `left` at 90 degrees, also a garment reference.
5. `right` at 270 degrees, also a garment reference.
6. `back_left_three_quarter` at 135 degrees.
7. `back_right_three_quarter` at 225 degrees.
8. `back` at 180 degrees, also a garment reference.

This yields 24 planned Teacher targets and 12 cardinal references. Cardinal references are a disclosed subset of the same endpoint assets, not 12 additional images. Every logical ID is `subject00/{garment_id}/{semantic_pose_slot}`. All 24 entries are currently `PENDING_FORMAL_BASE` and none is materialized.

## Observation modalities and roles

Every formal observation must bind identity, garment, condition ID, role, RGB and SHA-256, mask and SHA-256, camera and SHA-256, pose/SMPL-X and SHA-256, resolution, background, source SHA-256, backend metadata, human review, split, and provenance.

The role enum is `IDENTITY_SOURCE`, `GARMENT_REFERENCE`, `TEACHER_TARGET`, `CONTROLLER_REFERENCE`, `CALIBRATION_CONDITION`, `TEST_CONDITION`, and `VISUAL_ONLY_ASSET`. A multi-role asset must disclose every role. A role may not be silently inferred from a directory name.

The normalized image contract is 1024 x 1536 portrait PNG, fixed full-body framing, complete hands and feet, fixed identity-specific camera scale, and light-gray or white background.

## Formal condition boundary

Teacher targets use strict-train poses and cameras only. Train cameras are `1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23`; held-out cameras `0,4,8,12,16,20` are forbidden as Teacher targets. The frozen pose split contains 1,130 train, 125 held-out, and 1,245 temporal-buffer frames.

The eight selected train pose/camera pairs are still pending. The source also lacks materialized `CONTROLLER_REFERENCE`, `CALIBRATION_CONDITION`, `TEST_CONDITION`, and `VISUAL_ONLY_ASSET` records. Existing cardinal endpoint references cannot be relabeled into train/calibration/test pools because the present contract forbids the same image or SHA from serving adaptation and test. No post-hoc quality-based split movement is allowed.

## Directory plan

The planned cloud root is `/root/autodl-tmp/canondressgs_work/datasets/subject00_three_garment/`; the planned Windows mirror is `E:/model_train/canondressgs_work/datasets/subject00_three_garment/`. Neither was created by this task.

The fixed subdirectories are `00_contract`, `01_identity_sources`, `02_garment_references`, `03_generation_requests`, `04_generation_responses`, `05_human_review`, `06_accepted_rgb`, `07_accepted_masks`, `08_camera_pose`, `09_teacher_targets`, `10_controller_splits`, `11_provenance`, and `12_final_verification`.

Git may contain manifests, metadata, hashes, review results, and license-permitted small thumbnails. Raw identity data, raw generated images, and formal dataset image copies remain outside Git.
