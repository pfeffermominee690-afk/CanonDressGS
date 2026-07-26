#!/usr/bin/env python3
"""Generate the external-baseline feasibility audit from the frozen bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-CANONDRESSGS-EXTERNAL-BASELINE-FEASIBILITY-FROM-BUNDLE-001"
SOURCE_BRANCH = "research/external-baseline-multi-source-evidence-freeze-20260726"
SOURCE_HEAD = "397a72df68bdfd1f1a1cc67d02ca8648586133b7"
NEW_BRANCH = "research/external-baseline-feasibility-from-bundle-20260726"
BUNDLE_SHA256 = "f12e5bb8ce2c36e12bdb911fdae41cd7e3e2ca0e9d2520dc2c57d2263ffd9e61"
FINAL_CLASSIFICATION = "EXTERNAL_BASELINE_PROTOCOL_READY_FOR_USER_EXECUTION_SELECTION"
NEXT_TASK = "USER_AUTHORIZE_FULL_AVATAR_FINETUNE_OR_GSVTON_CANARY"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_protocol" / "external_baselines"
BUNDLE = OUT / "source_bundle" / "canondressgs_external_baseline_multi_source_evidence_bundle.json"
REPORT = ROOT / "docs" / "PAPER" / "EXTERNAL_BASELINE_FEASIBILITY_AUDIT_REPORT_20260726.md"
HANDOFF = ROOT / "project_control_handoff" / "external_baseline_feasibility_from_bundle_handoff.json"


def source(url: str, kind: str, observation: str) -> dict[str, str]:
    return {"url": url, "kind": kind, "observation": observation, "accessed": "2026-07-26"}


def method(
    method_id: str,
    title: str,
    authors: list[str],
    status: str,
    venue: str,
    year: int | None,
    paper_url: str | None,
    project_url: str | None,
    repo_url: str | None,
    code_status: str,
    readme_status: str,
    custom_data_docs: str,
    checkpoints: str,
    checkpoint_size: str,
    required_dataset: list[str],
    environment: dict[str, str],
    weights: list[str],
    vram: str,
    storage: str,
    runtime: str,
    inference_only: str,
    full_training: str,
    sources: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "title": title,
        "authors": authors,
        "current_publication_status": status,
        "venue": venue,
        "year": year,
        "official_paper_url": paper_url,
        "official_project_url": project_url,
        "official_repository_url": repo_url,
        "official_code_status": code_status,
        "readme_status": readme_status,
        "custom_data_documentation": custom_data_docs,
        "pretrained_checkpoint_availability": checkpoints,
        "checkpoint_size": checkpoint_size,
        "required_dataset": required_dataset,
        "environment": environment,
        "additional_model_weights": weights,
        "expected_vram": vram,
        "estimated_storage": storage,
        "estimated_runtime": runtime,
        "inference_only_possibility": inference_only,
        "full_training_requirement": full_training,
        "official_source_evidence": sources,
    }


OFFICIAL_SOURCES = {
    "schema_version": "canondressgs.external_baseline.official_source_registry.v1",
    "task_id": TASK_ID,
    "official_source_only": True,
    "methods": [
        method(
            "FULL_AVATAR_FINETUNING",
            "Full Avatar Fine-tuning",
            [],
            "NOT_APPLICABLE_INTERNAL_CONTROL",
            "Same-backbone strong control",
            None,
            None,
            None,
            None,
            "INTERNAL_CONTROL_NO_EXTERNAL_CODE",
            "Contract defined by this audit",
            "Uses the existing subject02 registered target contract",
            "Uses the formal Base Avatar initialization; no external checkpoint",
            "Bound at execution from the formal Base Avatar checkpoint",
            ["subject02 O01/O02/O03/O04/O08 registered multi-view targets"],
            {"python": "CanonDressGS formal environment", "pytorch": "CanonDressGS formal environment", "cuda": "CanonDressGS formal environment"},
            [],
            "Same as formal CanonDressGS training environment; measure peak VRAM",
            "Five independent avatar checkpoints plus optimizer states; measure bytes",
            "Two preregistered budgets: equal-step and equal-wall-time",
            "NO",
            "YES_PER_GARMENT",
            [source(str(BUNDLE.relative_to(ROOT)), "frozen local contract", "Only frozen bundle facts may define CanonDressGS numbers and scope")],
        ),
        method(
            "GS_VTON",
            "GS-VTON: Controllable 3D Virtual Try-on with Gaussian Splatting",
            ["Yukang Cao", "Masoud Hadi", "Liang Pan", "Ziwei Liu"],
            "PREPRINT",
            "arXiv 2024; submitted to ICLR 2025, with no official acceptance evidence found",
            2024,
            "https://arxiv.org/abs/2410.05259",
            "https://yukangcao.github.io/GS-VTON/",
            "https://github.com/yukangcao/GS-VTON",
            "OFFICIAL_CODE_AVAILABLE",
            "Runnable main.py, custom-data pointer, dependency-weight instructions, and fixed YAML config are present",
            "Uses official Nerfstudio custom-data and 3DGS initialization instructions; no subject02 adapter is supplied",
            "No GS-VTON method checkpoint; public third-party preprocessing and diffusion weights are required",
            "Aggregate bytes not published by authors; must be resolved by a download manifest before execution",
            ["target-subject video or posed multi-view images", "initialized static 3DGS PLY", "one garment image", "calibrated cameras"],
            {"python": "3.8", "pytorch": "2.2.1", "cuda": "11.8", "xformers": "0.0.25"},
            ["IDM-VTON dependencies", "DensePose", "OpenPose", "human parsing", "self-correction parsing", "Stable Diffusion dependencies"],
            "Not published; use a 24 GB-class preflight recommendation, explicitly not an author claim",
            "Audit estimate 15-30 GB including environment, public weights, one 3DGS, and outputs",
            "Author runtime not published; config fixes stage-2 max_steps=10000, plus preprocessing, LoRA, and 3DGS initialization",
            "NO",
            "YES_SCENE_EDITING_AND_LORA",
            [
                source("https://arxiv.org/abs/2410.05259", "official arXiv", "Image-prompted 3D VTON from multi-view human images and a garment image"),
                source("https://yukangcao.github.io/GS-VTON/", "author project page", "Links the paper and official repository"),
                source("https://github.com/yukangcao/GS-VTON", "author repository", "Official implementation and custom-data instructions"),
                source("https://openreview.net/forum?id=8eenzfwKqU", "official submission page", "Submitted to ICLR 2025; not evidence of acceptance"),
            ],
        ),
        method(
            "GAUSSIANVTON",
            "GaussianVTON: 3D Human Virtual Try-ON via Multi-Stage Gaussian Splatting Editing with Image Prompting",
            ["Haodong Chen", "Yongle Huang", "Haojian Huang", "Xiangsheng Ge", "Dian Shao"],
            "PREPRINT",
            "arXiv technical report",
            2024,
            "https://arxiv.org/abs/2405.07472",
            "https://haroldchen19.github.io/gsvton/",
            "https://github.com/HaroldChen19/GaussianVTON",
            "PARTIAL_CODE_ONLY",
            "README states partial code was released and full code is coming soon",
            "No complete official custom-data execution path",
            "No complete method checkpoint contract",
            "NOT_PUBLISHED",
            ["existing 3DGS scene", "multi-view images", "garment image"],
            {"python": "3.8", "pytorch": "2.1.0", "cuda": "11.8"},
            ["Ladi-VTON", "GaussianEditor", "diffusion and segmentation dependencies"],
            "NOT_PUBLISHED",
            "NOT_RELIABLY_ESTIMABLE_FROM_PARTIAL_CODE",
            "NOT_RELIABLY_ESTIMABLE_FROM_PARTIAL_CODE",
            "NO",
            "BLOCKED_BY_INCOMPLETE_CODE",
            [
                source("https://arxiv.org/abs/2405.07472", "official arXiv", "Image-prompted multi-stage 3DGS editing"),
                source("https://haroldchen19.github.io/gsvton/", "author project page", "Links paper, code, and data"),
                source("https://github.com/HaroldChen19/GaussianVTON", "author repository", "README explicitly marks code as partial"),
            ],
        ),
        method(
            "GAUSSIAN_WARDROBE",
            "Gaussian Wardrobe: Compositional 3D Gaussian Avatars for Free-Form Virtual Try-On",
            ["Zhiyi Chen", "Hsuan-I Ho", "Tianjian Jiang", "Jie Song", "Manuel Kaufmann", "Chen Guo"],
            "PUBLISHED",
            "3DV 2026 Poster",
            2026,
            "https://openreview.net/forum?id=sncanvgvUn",
            "https://eth-ait.github.io/GaussianWardrobe/",
            "https://github.com/eth-ait/GaussianWardrobe",
            "OFFICIAL_CODE_AVAILABLE",
            "Training, animation, transfer, evaluation, and checkpoint instructions are present",
            "GEN_DATA.md targets 4D-DRESS and ActorsHQ acquisition contracts",
            "Selected 4D-DRESS checkpoints are linked",
            "Official aggregate checkpoint bytes not stated",
            ["4D-DRESS or ActorsHQ dynamic multi-view video", "SMPL-X registration", "layer labels and masks"],
            {"python": "NOT_PINNED", "pytorch": "NOT_PINNED", "cuda": "NOT_PINNED"},
            ["SMPL-X", "LPIPS weights", "PyTorch3D"],
            "NOT_PUBLISHED",
            "Audit estimate 20-60 GB excluding licensed datasets",
            "Full avatar and garment-layer training required; author runtime not published",
            "ONLY_WITH_MATCHED_PRETRAINED_4D_DRESS_SUBJECTS",
            "YES_FOR_CUSTOM_SUBJECTS",
            [
                source("https://openreview.net/forum?id=sncanvgvUn", "official 3DV publication page", "3DV 2026 Poster"),
                source("https://eth-ait.github.io/GaussianWardrobe/", "author project page", "Dynamic multi-view layered avatar acquisition"),
                source("https://github.com/eth-ait/GaussianWardrobe", "official lab repository", "MIT code and 4D-DRESS/ActorsHQ preparation"),
            ],
        ),
        method(
            "LAYGA",
            "LayGA: Layered Gaussian Avatars for Animatable Clothing Transfer",
            ["Siyou Lin", "Zhe Li", "Zhaoqi Su", "Zerong Zheng", "Hongwen Zhang", "Yebin Liu"],
            "PUBLISHED",
            "SIGGRAPH 2024 Conference Papers",
            2024,
            "https://arxiv.org/abs/2405.07319",
            "https://jsnln.github.io/layga/index.html",
            None,
            "OFFICIAL_CODE_NOT_FOUND",
            "Project page links paper and supplement but no code repository",
            "Paper/supplement describe data, not an executable custom-data pipeline",
            "NOT_FOUND",
            "NOT_APPLICABLE_WITHOUT_CODE",
            ["multi-view videos", "body/clothing segmentation", "source and target layered avatars"],
            {"python": "NOT_AVAILABLE", "pytorch": "NOT_AVAILABLE", "cuda": "NOT_AVAILABLE"},
            [],
            "NOT_AVAILABLE",
            "NOT_ESTIMABLE_WITHOUT_CODE",
            "Supplement reports 200k single-layer and 550k multi-layer iterations",
            "NO",
            "YES_BUT_CODE_UNAVAILABLE",
            [
                source("https://arxiv.org/abs/2405.07319", "official arXiv", "Layered Gaussian avatars from multi-view videos"),
                source("https://jsnln.github.io/layga/index.html", "author project page", "Paper and supplement links; no code link"),
                source("https://jsnln.github.io/layga/assets/supp.pdf", "author supplement", "Reports 200k and 550k training stages"),
            ],
        ),
        method(
            "DAMA",
            "DAMA: Disentangled Body-Anchored Gaussians for Controllable Multi-Layered Avatars",
            ["Daniel Eskandar", "Berna Kabadayi", "Garvita Tiwari", "Gerard Pons-Moll"],
            "PUBLISHED",
            "PhysHuman Workshop at CVPR 2026 (Oral)",
            2026,
            "https://arxiv.org/abs/2605.21001",
            "https://danieleskandar.github.io/dama/",
            "https://github.com/danieleskandar/DAMA-code",
            "OFFICIAL_CODE_AVAILABLE",
            "Complete 4D-DRESS preprocessing, three-stage training, evaluation, and applications are documented",
            "Official pipeline is specific to 4D-DRESS scans, SMPL-X, semantic masks, and Blender rendering",
            "No pretrained avatar checkpoints documented",
            "SMPL-X v1.1 is documented as 830 MB; no method checkpoint size published",
            ["4D-DRESS scan and SMPL-X meshes", "multi-view RGB and semantic masks", "camera parameters"],
            {"python": "3.9", "pytorch": "2.7.1", "cuda": "11.8", "blender": "3.6.2"},
            ["SMPL-X v1.1 (830 MB)", "Blender 3.6.2"],
            "Tested by authors on RTX 2080 Ti with 11 GB VRAM",
            "Audit estimate 10-30 GB excluding 4D-DRESS",
            "Full three-stage reconstruction is required; runtime not published",
            "NO_FOR_NEW_SUBJECT",
            "YES",
            [
                source("https://arxiv.org/abs/2605.21001", "official arXiv", "Body-anchored layered Gaussian reconstruction"),
                source("https://danieleskandar.github.io/dama/", "author project page", "CVPR 2026 workshop oral and official code link"),
                source("https://github.com/danieleskandar/DAMA-code", "author repository", "MIT code and tested environment"),
            ],
        ),
        method(
            "SEMANTICGARMENT",
            "SemanticGarment: Semantic-Controlled Generation and Editing of 3D Gaussian Garments",
            ["Ruiyan Wang", "Zhengxue Cheng", "Zonghao Lin", "Jun Ling", "Yuzhou Liu", "Yanru An", "Rong Xie", "Li Song"],
            "PUBLISHED",
            "ACM Multimedia 2025",
            2025,
            "https://doi.org/10.1145/3746027.3755136",
            None,
            None,
            "OFFICIAL_CODE_NOT_FOUND",
            "No author project or official repository was located from the paper metadata",
            "NOT_FOUND",
            "NOT_FOUND",
            "NOT_APPLICABLE_WITHOUT_CODE",
            ["text or image prompt", "3D semantic clothing model", "structural human prior"],
            {"python": "NOT_AVAILABLE", "pytorch": "NOT_AVAILABLE", "cuda": "NOT_AVAILABLE"},
            [],
            "NOT_AVAILABLE",
            "NOT_ESTIMABLE_WITHOUT_CODE",
            "NOT_ESTIMABLE_WITHOUT_CODE",
            "UNKNOWN",
            "UNKNOWN",
            [
                source("https://arxiv.org/abs/2509.16960", "official arXiv", "Text/image prompted 3D garment generation and semantic editing"),
                source("https://doi.org/10.1145/3746027.3755136", "ACM DOI", "Proceedings article in ACM Multimedia 2025, pages 9793-9802"),
            ],
        ),
    ],
}


PINS = {
    "schema_version": "canondressgs.external_baseline.repository_pin_registry.v1",
    "task_id": TASK_ID,
    "pins": [
        {"method_id": "GS_VTON", "repository": "https://github.com/yukangcao/GS-VTON", "default_branch": "main", "commit": "96964b0a6528089123cc27a3ff3e3eb46505cf6e", "commit_date": "2026-03-27T11:05:44+08:00", "shallow_clone_clean": True},
        {"method_id": "GAUSSIANVTON", "repository": "https://github.com/HaroldChen19/GaussianVTON", "default_branch": "main", "commit": "db5008cf68861b7863f7fe43d2a7453d3a2b9543", "commit_date": "2024-06-04T20:23:37+08:00", "shallow_clone_clean": True},
        {"method_id": "GAUSSIAN_WARDROBE", "repository": "https://github.com/eth-ait/GaussianWardrobe", "default_branch": "main", "commit": "af9ff485602b55f8125651e48803b1448b67c0b8", "commit_date": "2026-07-05T16:46:34+02:00", "shallow_clone_clean": True},
        {"method_id": "DAMA", "repository": "https://github.com/danieleskandar/DAMA-code", "default_branch": "main", "commit": "b2a6b233c4fc3c938024a8adc76571b4709880ba", "commit_date": "2026-06-16T09:20:33+02:00", "shallow_clone_clean": True},
        {"method_id": "LAYGA", "repository": None, "default_branch": None, "commit": None, "commit_date": None, "shallow_clone_clean": None},
        {"method_id": "SEMANTICGARMENT", "repository": None, "default_branch": None, "commit": None, "commit_date": None, "shallow_clone_clean": None},
    ],
    "external_source_root": r"E:\model_train\_external_baseline_source_audit_20260726",
    "checkpoint_download_bytes": 0,
    "dataset_download_bytes": 0,
}


LICENSES = {
    "schema_version": "canondressgs.external_baseline.license_audit.v1",
    "task_id": TASK_ID,
    "entries": [
        {"method_id": "GS_VTON", "license": "NO_ROOT_LICENSE_DETECTED", "status": "EXECUTION_REQUIRES_LICENSE_REVIEW", "evidence": "No top-level license file and GitHub API license=null at pinned commit; bundled third-party licenses do not license original GS-VTON code"},
        {"method_id": "GAUSSIANVTON", "license": "NO_ROOT_LICENSE_DETECTED", "status": "BLOCKED_BY_CODE_AND_LICENSE", "evidence": "No top-level license file and GitHub API license=null"},
        {"method_id": "GAUSSIAN_WARDROBE", "license": "MIT", "status": "PASS", "evidence": "Top-level LICENSE at pinned commit"},
        {"method_id": "LAYGA", "license": "NOT_APPLICABLE_NO_CODE", "status": "OFFICIAL_CODE_NOT_FOUND", "evidence": "Official project page has no code repository link"},
        {"method_id": "DAMA", "license": "MIT", "status": "PASS", "evidence": "Top-level LICENSE at pinned commit"},
        {"method_id": "SEMANTICGARMENT", "license": "NOT_APPLICABLE_NO_CODE", "status": "OFFICIAL_CODE_NOT_FOUND", "evidence": "Official code repository not found"},
    ],
}


QUESTION_KEYS = [
    "garment_image_input", "garment_image_count", "donor_garment_video", "target_subject_multiview_video",
    "dynamic_sequence", "garment_mesh", "smpl_or_smplx", "garment_segmentation", "foreground_masks",
    "existing_3dgs", "starts_from_existing_avatar", "freezes_avatar", "canonical_garment_state",
    "preserves_original_deformation", "animatable_avatar_output", "target_pose", "target_camera",
    "fixed_identity_suitable", "closed_wardrobe_suitable", "subject02_five_garment_compatible",
    "actorshq_or_4d_dress_required", "same_target_rgb_mask_evaluation", "equal_reference_protocol",
    "native_input_protocol",
]


def q(**values: str) -> dict[str, str]:
    missing = set(QUESTION_KEYS) - set(values)
    extra = set(values) - set(QUESTION_KEYS)
    if missing or extra:
        raise ValueError(f"compatibility keys missing={missing} extra={extra}")
    return values


COMPATIBILITY = {
    "schema_version": "canondressgs.external_baseline.task_compatibility_matrix.v1",
    "task_id": TASK_ID,
    "question_order": QUESTION_KEYS,
    "methods": [
        {"method_id": "FULL_AVATAR_FINETUNING", "classification": "MATCHED_EXECUTABLE", "answers": q(garment_image_input="NO", garment_image_count="0", donor_garment_video="NO", target_subject_multiview_video="YES_REGISTERED_TARGETS", dynamic_sequence="USES_EXISTING_AVATAR_CONTRACT", garment_mesh="NO", smpl_or_smplx="USES_BASE_AVATAR_CONTRACT", garment_segmentation="YES_TARGET_MASK", foreground_masks="YES", existing_3dgs="YES_BASE_AVATAR", starts_from_existing_avatar="YES", freezes_avatar="NO", canonical_garment_state="YES", preserves_original_deformation="NOT_GUARANTEED_MUST_MEASURE", animatable_avatar_output="YES", target_pose="YES", target_camera="YES", fixed_identity_suitable="YES", closed_wardrobe_suitable="YES", subject02_five_garment_compatible="YES", actorshq_or_4d_dress_required="NO", same_target_rgb_mask_evaluation="YES", equal_reference_protocol="NOT_APPLICABLE_NO_REFERENCE", native_input_protocol="YES")},
        {"method_id": "GS_VTON", "classification": "DIFFERENT_ASSUMPTIONS_EXECUTABLE", "answers": q(garment_image_input="YES", garment_image_count="1", donor_garment_video="NO", target_subject_multiview_video="YES_VIDEO_OR_POSED_MULTIVIEW", dynamic_sequence="NO_STATIC_SCENE_EDIT", garment_mesh="NO", smpl_or_smplx="NO", garment_segmentation="GENERATED_BY_PREPROCESSORS", foreground_masks="GENERATED_OR_SUPPLIED", existing_3dgs="YES_STATIC_3DGS_PLY", starts_from_existing_avatar="NO_STARTS_FROM_STATIC_3DGS", freezes_avatar="NO", canonical_garment_state="NO", preserves_original_deformation="NOT_APPLICABLE", animatable_avatar_output="NO", target_pose="NO_FIXED_POSE_ONLY", target_camera="YES", fixed_identity_suitable="YES_FIXED_SCENE", closed_wardrobe_suitable="YES_PER_GARMENT_EDIT", subject02_five_garment_compatible="CONDITIONAL_DATA_CONVERSION_AND_3DGS_INIT", actorshq_or_4d_dress_required="NO", same_target_rgb_mask_evaluation="YES_ONE_FIXED_TARGET_CONDITION", equal_reference_protocol="SUPPORTED_ONE_GARMENT_IMAGE", native_input_protocol="YES_DIFFERENT_ASSUMPTIONS")},
        {"method_id": "GAUSSIANVTON", "classification": "BLOCKED_BY_CODE", "answers": q(garment_image_input="YES", garment_image_count="1", donor_garment_video="NO", target_subject_multiview_video="YES", dynamic_sequence="NO_STATIC_SCENE_EDIT", garment_mesh="NO", smpl_or_smplx="NO", garment_segmentation="YES_OR_GENERATED", foreground_masks="YES_OR_GENERATED", existing_3dgs="YES", starts_from_existing_avatar="NO_STARTS_FROM_STATIC_3DGS", freezes_avatar="NO", canonical_garment_state="NO", preserves_original_deformation="NOT_APPLICABLE", animatable_avatar_output="NO", target_pose="NO_FIXED_POSE_ONLY", target_camera="YES", fixed_identity_suitable="YES_FIXED_SCENE", closed_wardrobe_suitable="YES_PER_EDIT", subject02_five_garment_compatible="UNKNOWN_PARTIAL_CODE", actorshq_or_4d_dress_required="NO", same_target_rgb_mask_evaluation="CONDITIONAL", equal_reference_protocol="CONCEPTUALLY_YES_NOT_EXECUTABLE", native_input_protocol="BLOCKED_BY_CODE")},
        {"method_id": "GAUSSIAN_WARDROBE", "classification": "BLOCKED_BY_DATA", "answers": q(garment_image_input="NO", garment_image_count="0", donor_garment_video="YES", target_subject_multiview_video="YES", dynamic_sequence="YES", garment_mesh="NO_INPUT_TEMPLATE_BUT_REGISTERED_BODY", smpl_or_smplx="YES", garment_segmentation="YES_LAYER_LABELS", foreground_masks="YES", existing_3dgs="NO", starts_from_existing_avatar="NO", freezes_avatar="NO", canonical_garment_state="YES_LAYERED", preserves_original_deformation="NO_OWN_DEFORMATION", animatable_avatar_output="YES", target_pose="YES", target_camera="YES", fixed_identity_suitable="YES", closed_wardrobe_suitable="YES", subject02_five_garment_compatible="NO_CURRENT_DYNAMIC_LAYERED_CONTRACT", actorshq_or_4d_dress_required="OFFICIAL_PIPELINE_EXPECTS_4D_DRESS_OR_ACTORSHQ", same_target_rgb_mask_evaluation="CONDITIONAL_AFTER_NEW_ACQUISITION", equal_reference_protocol="UNSUPPORTED_NO_GARMENT_IMAGE_INPUT", native_input_protocol="YES_BUT_DATA_BLOCKED")},
        {"method_id": "LAYGA", "classification": "BLOCKED_BY_CODE", "answers": q(garment_image_input="NO", garment_image_count="0", donor_garment_video="YES", target_subject_multiview_video="YES", dynamic_sequence="YES", garment_mesh="NO_EXTERNAL_GARMENT_TEMPLATE", smpl_or_smplx="NO_PARAMETRIC_TEMPLATE_CLAIM", garment_segmentation="YES_RECONSTRUCTED", foreground_masks="YES", existing_3dgs="NO", starts_from_existing_avatar="NO", freezes_avatar="NO", canonical_garment_state="YES_LAYERED", preserves_original_deformation="NO_OWN_MODEL", animatable_avatar_output="YES", target_pose="YES", target_camera="YES", fixed_identity_suitable="YES", closed_wardrobe_suitable="YES", subject02_five_garment_compatible="NO_CODE_AND_NO_LAYERED_CAPTURE", actorshq_or_4d_dress_required="NO_SINGLE_MANDATED_DATASET_BUT_MULTIVIEW_VIDEO_REQUIRED", same_target_rgb_mask_evaluation="CONDITIONAL", equal_reference_protocol="UNSUPPORTED_NO_GARMENT_IMAGE_INPUT", native_input_protocol="BLOCKED_BY_CODE")},
        {"method_id": "DAMA", "classification": "BLOCKED_BY_DATA", "answers": q(garment_image_input="NO", garment_image_count="0", donor_garment_video="NO_USES_SOURCE_LAYER_RECONSTRUCTIONS", target_subject_multiview_video="YES_RENDERED_FROM_SCAN", dynamic_sequence="NO_FOR_RECONSTRUCTION", garment_mesh="YES_4D_DRESS_SCAN_AND_SMPLX", smpl_or_smplx="YES", garment_segmentation="YES_SEMANTIC_MASKS", foreground_masks="YES", existing_3dgs="NO", starts_from_existing_avatar="NO", freezes_avatar="NO", canonical_garment_state="YES_LAYERED", preserves_original_deformation="NO_OWN_SMPLX_ANIMATION", animatable_avatar_output="YES", target_pose="YES", target_camera="YES", fixed_identity_suitable="YES", closed_wardrobe_suitable="YES_LAYER_RECOMBINATION", subject02_five_garment_compatible="NO_CURRENT_4D_DRESS_SCAN_SMPLX_CONTRACT", actorshq_or_4d_dress_required="YES_OFFICIAL_CODE_USES_4D_DRESS", same_target_rgb_mask_evaluation="CONDITIONAL_AFTER_NEW_ACQUISITION", equal_reference_protocol="UNSUPPORTED_NO_GARMENT_IMAGE_INPUT", native_input_protocol="YES_BUT_DATA_BLOCKED")},
        {"method_id": "SEMANTICGARMENT", "classification": "RELATED_WORK_ONLY", "answers": q(garment_image_input="YES_OR_TEXT", garment_image_count="1_OR_TEXT", donor_garment_video="NO", target_subject_multiview_video="NO", dynamic_sequence="NO_REQUIRED_INPUT", garment_mesh="NO_EXISTING_TEMPLATE", smpl_or_smplx="STRUCTURAL_HUMAN_PRIOR_UNSPECIFIED_EXECUTION", garment_segmentation="SEMANTIC_MODEL", foreground_masks="UNKNOWN", existing_3dgs="GARMENT_GAUSSIANS", starts_from_existing_avatar="NO", freezes_avatar="NOT_APPLICABLE", canonical_garment_state="STANDALONE_GENERATED_GARMENT", preserves_original_deformation="NO", animatable_avatar_output="NOT_SAME_AVATAR_CONTRACT", target_pose="NOT_ESTABLISHED_FOR_SUBJECT02", target_camera="YES_FOR_RENDERED_GARMENT", fixed_identity_suitable="NO_MATCHED_IDENTITY_CONTRACT", closed_wardrobe_suitable="NO_OPEN_GENERATION_EDITING_TASK", subject02_five_garment_compatible="NO", actorshq_or_4d_dress_required="NO", same_target_rgb_mask_evaluation="NO_MATCHED_TARGET_CONTRACT", equal_reference_protocol="NOT_SCIENTIFICALLY_COMPARABLE", native_input_protocol="RELATED_WORK_ONLY")},
    ],
}


DATA_DEPENDENCIES = {
    "schema_version": "canondressgs.external_baseline.data_dependency_matrix.v1",
    "task_id": TASK_ID,
    "subject02_frozen_scope": {"identity": "subject02", "garments": ["O01", "O02", "O03", "O04", "O08"], "scope": "fixed identity; seen garments; closed wardrobe; view-transductive", "positive_subject00": False, "positive_actorshq": False},
    "methods": [
        {"method_id": "FULL_AVATAR_FINETUNING", "available": ["formal Base Avatar initialization", "registered multi-view garment targets", "frozen train/calibration/test split"], "missing": [], "decision": "READY"},
        {"method_id": "GS_VTON", "available": ["one garment reference can be selected", "registered target RGB/masks and cameras are expected by the CanonDressGS protocol"], "missing": ["official-format video/multi-view conversion", "vanilla static 3DGS initialization", "resolved public dependency weights", "license clearance"], "decision": "CONDITIONAL_CANARY_READY"},
        {"method_id": "GAUSSIANVTON", "available": ["conceptual garment image and 3DGS inputs"], "missing": ["complete official code", "complete custom-data documentation", "license"], "decision": "BLOCKED_BY_CODE"},
        {"method_id": "GAUSSIAN_WARDROBE", "available": ["five garment endpoints exist in the CanonDressGS study"], "missing": ["4D-DRESS/ActorsHQ-style dynamic multi-view videos", "SMPL-X registration", "layer semantics"], "decision": "BLOCKED_BY_DATA"},
        {"method_id": "LAYGA", "available": [], "missing": ["official code", "layered multi-view videos for donor and target", "executable preprocessing contract"], "decision": "BLOCKED_BY_CODE"},
        {"method_id": "DAMA", "available": ["multi-view target imagery exists conceptually"], "missing": ["4D-DRESS raw scans", "SMPL-X mesh and texture", "semantic garment masks in official format"], "decision": "BLOCKED_BY_DATA"},
        {"method_id": "SEMANTICGARMENT", "available": ["garment images could be used as prompts"], "missing": ["official code", "matched fixed-identity avatar output contract"], "decision": "RELATED_WORK_ONLY"},
    ],
}


INTERNAL_EVIDENCE = {
    "pure_endpoint": {"classification": "PURE_ENDPOINT_CORE_METHOD_SUPPORTED", "training_runs": 24, "optimizer_steps": 7200, "checkpoints": 144, "clean_top1": 1.0, "canon_exact_endpoint": 1.0, "linear_exact_endpoint": 0.0, "linear_lpips": 0.067913006991148, "canon_blur_top1": 0.35, "canon_one_reference_top1": 0.95, "hard_lookup": "CLEAN_FUNCTIONAL_EQUIVALENCE", "identity_contamination": 0, "severe_wrong_outfit_failure": 0},
    "geometry": {"classification": "GEOMETRY_MAIN_EFFECT", "main_effect": 0.9959405426885113, "sufficient": "10/10", "necessary": "10/10"},
    "dual_support": {"classification": "DUAL_SUPPORT_ALL_PAIR_PASS", "pair_weight_source": "EXTERNALLY_SPECIFIED", "linear": {"lpips": 0.085303, "iou": 0.710052, "boundary_f": 0.300263}, "dual": {"lpips": 0.056985, "iou": 0.725290, "boundary_f": 0.379103}, "hard_lpips": 0.0318124, "artifacts_improved": "10/10", "automatic_controller": False},
    "headroom": {"classification": "TEACHER_SPAN_AT_LOCAL_OPTIMUM", "improved_garments": "0/5"},
    "loo": {"classification": "LOO_BASIS_CAPACITY_LIMITED", "capacity_pass": "0/5", "unseen_garment_adaptation": False},
}


METRIC_PROTOCOL = """# External Target-Space Metric Protocol

Task: `AAAI27-CANONDRESSGS-EXTERNAL-BASELINE-FEASIBILITY-FROM-BUNDLE-001`

## Scope and denominators

- The evaluation unit is one frozen subject02 garment/target-condition record with ground-truth RGB, foreground mask, garment mask, target pose, and target camera.
- The target RGB denominator is every preregistered test record for which all compared methods produced an output. Failures remain failures and are not removed; a method-level failure count is reported separately.
- The mask denominator is the same preregistered record list, with no method-specific filtering.
- Render every method at the exact target camera, pose when supported, native target resolution, color space, and background convention. A method without pose control is evaluated only in the one fixed-pose canary and is labeled as such.
- Freeze a crop from the ground-truth foreground bounding box plus a 10% margin. Use the same crop for all methods. Pixels outside this crop are excluded. No prediction-dependent crop is allowed.

## Regions

- Full-image metrics use the fixed target crop, including its in-crop background.
- Garment-region metrics use the ground-truth garment mask inside the fixed crop.
- Protected-region metrics use `ground_truth_foreground AND NOT dilate(ground_truth_garment_mask, 5 px)`. Background is excluded.
- Silhouette IoU and Boundary F use ground-truth foreground and predicted alpha thresholded at 0.5. Boundary F tolerance is exactly 2 pixels at native resolution.

## Implementations

- RGB is sRGB in `[0,1]`; no per-method color correction is allowed.
- LPIPS: `lpips==0.1.4`, AlexNet backbone, inputs mapped to `[-1,1]`, spatial reduction disabled.
- PSNR: RGB MSE with data range 1.0. Infinite values are retained and separately counted.
- SSIM: `skimage.metrics.structural_similarity`, `channel_axis=-1`, `data_range=1.0`, `gaussian_weights=True`, `sigma=1.5`, `use_sample_covariance=False`.
- Report full-image LPIPS/PSNR/SSIM; garment-region LPIPS/PSNR/SSIM; silhouette IoU; Boundary F; protected-region LPIPS and RGB MAE.
- Also report adaptation time, trainable parameters, incremental storage, peak VRAM, rendering time/FPS, animation compatibility, and frozen-backbone preservation.

## Human review

- Use three reviewers, blinded randomized method columns, the same fixed crop, and no cherry-picking.
- Identity contamination is binary per record and decided by majority vote. Severe artifacts use grades 0-3; grades 2-3 count as severe.
- All missing outputs count as failures. Review all preregistered records or a preregistered uniform subset shared by every method.

## Exclusions

- Endpoint LPIPS, exact endpoint match, coefficient MAE/RMSE, endpoint-coordinate distance, and Teacher snapping parity are internal diagnostics and are forbidden in the external target-space table.
- `FACE_ID_PROTOCOL_STATUS=NOT_USED_DUE_TO_UNVALIDATED_FACE_ID_PROTOCOL`. No face-identity score may be added without a public license-compatible model, frozen crop/denominator/threshold, and independent threshold validation for all methods.
"""


FULL_AVATAR_CONTRACT = """# Full Avatar Fine-tuning Contract

Status: `READY_FOR_USER_AUTHORIZED_MICRO_PILOT`

## Fixed inputs

- Start each run from the exact formal subject02 Base Avatar initialization.
- Run O01/O02/O03/O04/O08 independently with the same registered multi-view garment targets and frozen train/calibration/test split used by the Teacher Endpoint protocol.
- Do not read reference images. Garment identity selects the training target only; no target-test RGB, mask, pose, camera, or Teacher residual may enter optimization or tuning.
- The complete avatar backbone may update. Test-set tuning, best-seed selection, hidden identity drift, and cross-garment resume are forbidden.
- Every formal run uses an independent attempt directory. Resume is allowed only from a recorded optimizer-step checkpoint with model, optimizer, scheduler, RNG, and elapsed-time state.

## Budgets

- Equal-step: for garment `g`, use exactly `N_teacher(g)`, the optimizer-step count of its formal Teacher Endpoint construction. The frozen bundle does not export this count, so execution preflight must bind it from sealed Teacher metadata; it must not be guessed or replaced by the 300 controller steps.
- Equal-wall-time: for garment `g`, stop at the first completed optimizer step at or beyond the measured formal Teacher construction wall time `T_teacher(g)`. Report any overshoot. The controller cost is not hidden: report Teacher construction, controller training amortized over five garments, and total five-garment CanonDressGS onboarding separately.
- Use fixed preregistered seeds and aggregate all seeds. Never select the best seed.

## Required reporting

- Trainable parameters, optimizer steps, wall time, peak VRAM, checkpoint bytes, optimizer-state bytes, and five-garment incremental storage.
- All target-space and protected-region metrics in `external_target_space_metric_protocol.md`.
- Identity contamination, catastrophic drift, original animation compatibility, original pose/camera rendering, and exact resume behavior.
- A run is catastrophic drift if protected-region failure is grade 2-3, original animation cannot render, or the Base Avatar identity review fails.

No training is authorized by this document. `TRAINING_STEPS=0` for this audit.
"""


GSVTON_CANARY = """# GS-VTON Micro-Canary Contract

Status: `CONTRACT_READY_WITH_LICENSE_AND_DATA_PREFLIGHT_GATES`

This is a native-input, different-assumptions canary. It is not a strict apples-to-apples avatar comparison and does not establish animation or canonical-state compatibility.

## Single fixed case

- Exactly one garment, one target condition, one garment image, one official initialization path, one fixed config, and one attempt.
- Select the garment and target condition before any output exists. No sweep, automatic retry, seed selection, or result-dependent substitution is allowed.
- Use official repository commit `96964b0a6528089123cc27a3ff3e3eb46505cf6e` and preserve its native stage contracts.

## Preflight gates

1. Resolve the missing top-level GS-VTON license through institutional review or written author permission. Public source availability alone is not a license.
2. Convert the preregistered subject02 target-subject views to the official Nerfstudio/GS-VTON camera and image layout without altering RGB content.
3. Build the required vanilla static 3DGS through the official 3DGS initialization path. The CanonDressGS animatable Base Avatar is not silently treated as a compatible PLY.
4. Supply the same one garment reference used by the CanonDressGS one-reference path. CanonDressGS comparison value remains frozen at one-reference top-1 `0.95`; clean multi-reference `1.0` may not replace it.
5. Freeze masks, camera, target condition, output resolution, config, dependency-weight manifest, expected download bytes, available storage, and a 24 GB-class GPU preflight before execution.

## Execution cap and output

- Official config only; stage-2 `max_steps=10000`. Record stage-1 processing, LoRA adaptation, 3DGS initialization, and editing time separately.
- Expected output is one edited static 3DGS and target-camera RGB/alpha renders for the selected fixed pose.
- Evaluate only with `external_target_space_metric_protocol.md`; report native-input differences beside every result.
- Fail on missing output, camera mismatch, unresolved masks, non-finite render, severe identity contamination, or any unregistered retry.

## Claim boundary

Even a successful canary supports only feasibility for one subject02 garment and one fixed target condition under GS-VTON native inputs. It does not support a paper claim, an animatable-avatar claim, a canonical endpoint claim, or a five-garment result.

No download, conversion, training, or inference is authorized by this document.
"""


TABLE_PLAN = """# External Baseline Table Plan

## Table A: External target-space and adaptation cost

Candidate rows: Base Avatar, Full Avatar Fine-tuning, GS-VTON, CanonDressGS. GaussianVTON may enter only after complete official code and a sealed canary. Gaussian Wardrobe, LayGA, DAMA, and SemanticGarment may enter only after their native data/output contracts become executable and scientifically aligned.

Columns: protocol type, full/garment/protected target-space metrics, silhouette IoU, Boundary F, adaptation time, trainable parameters, peak VRAM, incremental storage, rendering time/FPS, animation compatibility, and frozen-backbone preservation. No Endpoint LPIPS, exact endpoint, coefficient, coordinate, or snapping-parity columns.

## Table B: Internal endpoint mechanism

Rows: Outfit-ID Oracle, Reference Classifier, Nearest Centroid, Linear Coefficient Predictor, CanonDressGS. This table retains the frozen internal endpoint diagnostics and one-reference robustness where applicable.

## Table C: Composition

Rows: Linear Geometry, Hard Geometry, Dual-Support. Reuse frozen all-pair results. Pair and weights remain externally specified. `TARGET_MIXTURE_WEIGHT_ERROR` is forbidden without an independent real mixed-garment target. Mixedness metrics describe support contribution or perceptual plausibility only, never true garment-mixture reconstruction accuracy.

The three evidence blocks must remain separate.
"""


PRIORITY = """# External Baseline Execution Priority

1. `Full Avatar Fine-tuning micro-pilot`: highest task match, same backbone/data/renderer, no external license or representation conversion, and strong value as a retraining-cost/identity-drift control.
2. `GS-VTON micro-canary`: highest external-paper relevance and one-image garment input, but native assumptions differ. Execute only after license clearance, subject02 conversion, vanilla 3DGS initialization, and dependency-weight/storage preflight.

GaussianVTON is the fallback only if its authors release complete official code and a license. Gaussian Wardrobe and DAMA are blocked by dynamic layered or 4D-DRESS-specific data. LayGA is blocked by absent official code. SemanticGarment remains Related Work because it is an open generation/editing task rather than a matched fixed-identity endpoint method.

No third execution candidate is recommended.
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def validate_bundle() -> dict[str, Any]:
    if not BUNDLE.exists() or sha256(BUNDLE) != BUNDLE_SHA256:
        raise RuntimeError("EXTERNAL_BASELINE_AUDIT_BLOCKED_BY_FROZEN_BUNDLE_MISMATCH")
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    gates = [
        bundle["final_bundle_classification"] == "CANONDRESSGS_MULTI_SOURCE_EVIDENCE_BUNDLE_FROZEN_FOR_EXTERNAL_BASELINE_AUDIT",
        bundle["all_classifications_match"], bundle["all_numeric_checks_pass"],
        bundle["all_claim_boundaries_pass"], bundle["paper_evidence_conflict_count"] == 0,
        bundle["dirty_worktree_excluded"], bundle["export_count"] == 53,
    ]
    if not all(gates):
        raise RuntimeError("EXTERNAL_BASELINE_AUDIT_BLOCKED_BY_FROZEN_BUNDLE_MISMATCH")
    for entry in bundle["evidence_entries"]:
        for export in entry["exports"]:
            path = ROOT / export["export_path"]
            if not path.exists() or sha256(path) != export["export_sha256"]:
                raise RuntimeError("EXTERNAL_BASELINE_AUDIT_BLOCKED_BY_FROZEN_BUNDLE_MISMATCH")
    return bundle


def main() -> None:
    bundle = validate_bundle()
    write_json(OUT / "external_baseline_official_source_registry.json", OFFICIAL_SOURCES)
    write_json(OUT / "external_baseline_repository_pin_registry.json", PINS)
    write_json(OUT / "external_baseline_license_audit.json", LICENSES)
    write_json(OUT / "external_baseline_task_compatibility_matrix.json", COMPATIBILITY)
    write_json(OUT / "external_baseline_data_dependency_matrix.json", DATA_DEPENDENCIES)
    write_text(OUT / "external_target_space_metric_protocol.md", METRIC_PROTOCOL)
    write_text(OUT / "full_avatar_finetuning_contract.md", FULL_AVATAR_CONTRACT)
    write_text(OUT / "gsvton_canary_contract.md", GSVTON_CANARY)
    write_text(OUT / "external_baseline_table_plan.md", TABLE_PLAN)
    write_text(OUT / "external_baseline_execution_priority.md", PRIORITY)

    summary = {
        "schema_version": "canondressgs.external_baseline.feasibility_final_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": NEW_BRANCH,
        "bundle_path": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "bundle_sha256": BUNDLE_SHA256,
        "bundle_gate_status": "PASS",
        "source_contract_status": "PASS",
        "internal_evidence_freeze_status": "PASS",
        "internal_evidence": INTERNAL_EVIDENCE,
        "method_decisions": {row["method_id"]: row["classification"] for row in COMPATIBILITY["methods"]},
        "gs_vton": {"publication_status": "PREPRINT", "official_code_status": "OFFICIAL_CODE_AVAILABLE", "pinned_commit": PINS["pins"][0]["commit"], "license": "NO_ROOT_LICENSE_DETECTED", "task_compatibility": "DIFFERENT_ASSUMPTIONS_EXECUTABLE", "data_compatibility": "CONDITIONAL_SUBJECT02_CONVERSION_REQUIRED", "canary_status": "CONTRACT_READY_WITH_LICENSE_AND_DATA_PREFLIGHT_GATES"},
        "gaussianvton": {"publication_status": "PREPRINT", "code_status": "PARTIAL_CODE_ONLY", "task_compatibility": "BLOCKED_BY_CODE"},
        "full_avatar": {"status": "READY_FOR_USER_AUTHORIZED_MICRO_PILOT", "equal_step_status": "READY_BOUND_TO_N_TEACHER_PER_GARMENT", "equal_wall_time_status": "READY_BOUND_TO_T_TEACHER_PER_GARMENT"},
        "equal_reference_protocol_status": "SUPPORTED_FOR_GS_VTON_WITH_CANONDRESSGS_ONE_REFERENCE_TOP1_0.95",
        "native_input_protocol_status": "NATIVE_INPUT_DIFFERENT_ASSUMPTIONS_COMPARISON",
        "target_space_metric_protocol_status": "FROZEN",
        "face_id_protocol_status": "NOT_USED_DUE_TO_UNVALIDATED_FACE_ID_PROTOCOL",
        "direct_dense_predictor_status": "HISTORICAL_DIAGNOSTIC_ONLY",
        "composition_mixedness_metric_status": "SUPPORT_CONTRIBUTION_ONLY_REUSE_FROZEN_DUAL_SUPPORT_NO_NEW_RENDER",
        "table_a_plan": ["Base Avatar", "Full Avatar Fine-tuning", "GS-VTON", "CanonDressGS"],
        "table_b_plan": ["Outfit-ID Oracle", "Reference Classifier", "Nearest Centroid", "Linear Coefficient Predictor", "CanonDressGS"],
        "table_c_plan": ["Linear Geometry", "Hard Geometry", "Dual-Support"],
        "immediate_execution_candidates": ["FULL_AVATAR_FINETUNING_MICRO_PILOT", "GS_VTON_MICRO_CANARY_AFTER_PREFLIGHT_GATES"],
        "blocked_methods": {"GAUSSIANVTON": "BLOCKED_BY_CODE", "GAUSSIAN_WARDROBE": "BLOCKED_BY_DATA", "LAYGA": "BLOCKED_BY_CODE", "DAMA": "BLOCKED_BY_DATA", "SEMANTICGARMENT": "RELATED_WORK_ONLY"},
        "required_data": {"full_avatar": "existing subject02 registered targets", "gs_vton": "official-format target views, cameras, one garment image, initialized static 3DGS"},
        "required_checkpoints": {"full_avatar": "formal Base Avatar", "gs_vton": "third-party preprocessing and diffusion weights; no method checkpoint"},
        "estimated_storage": {"full_avatar": "measure five checkpoints plus optimizer states", "gs_vton": "15-30 GB audit estimate"},
        "estimated_compute": {"full_avatar": "equal-step and equal-wall-time budgets", "gs_vton": "10k stage-2 steps plus preprocessing, LoRA, and 3DGS initialization; official runtime not published"},
        "cloud_sync_recheck": "FAILED_NON_BLOCKING",
        "operations": {"training_steps": 0, "renderer_inferences": 0, "checkpoint_download_bytes": 0, "dataset_download_bytes": 0, "data_mutations": 0, "paper_modifications": 0, "bundle_mutations": 0},
        "official_sources_only": True,
        "external_repositories_modified": False,
        "paper_final": False,
        "final_classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "frozen_bundle_export_count": bundle["export_count"],
    }
    write_json(OUT / "external_baseline_feasibility_final_summary.json", summary)

    report = f"""# CanonDressGS External Baseline Feasibility Audit

Task: `{TASK_ID}`

## Decision

`{FINAL_CLASSIFICATION}`

The frozen bundle passed every source, hash, classification, numeric, boundary, conflict, and export-integrity gate. CanonDressGS remains a fixed-subject02, five-seen-garment, closed-wardrobe, view-transductive result. No paper or scientific result was modified.

The primary same-backbone control is Full Avatar Fine-tuning under equal-step and equal-wall-time budgets. The external micro-canary candidate is GS-VTON at official commit `96964b0a6528089123cc27a3ff3e3eb46505cf6e`. It accepts one garment image and has a runnable official pipeline, but it is a static 3DGS editing method with different inputs. It requires license clearance, subject02 conversion, a vanilla 3DGS initialization, and a dependency-weight manifest before execution.

GaussianVTON is blocked by its official partial-code status. Gaussian Wardrobe and DAMA are blocked by dynamic layered or 4D-DRESS-specific acquisition requirements. LayGA has no official code link. SemanticGarment is a published open generation/editing method and is Related Work only for the current endpoint experiment.

## Comparison contract

External methods are compared only in target RGB/mask space. Endpoint LPIPS, exact endpoint match, coefficient errors, coordinate distance, and Teacher snapping parity remain internal diagnostics. CanonDressGS uses the frozen one-reference top-1 value `0.95` in any equal-reference discussion; clean multi-reference `1.0` cannot replace it.

Table A contains external target-space and cost evidence. Table B contains internal endpoint mechanisms. Table C contains frozen composition evidence. These blocks must not be merged.

Direct Dense Predictor evidence remains `HISTORICAL_DIAGNOSTIC_ONLY`: the frozen bundle records zero Direct V7 executions in the formal pure endpoint run and preserves prior blocked classifications, so historical unmatched values cannot enter Table A.

Composition mixedness is limited to support contribution or perceptual plausibility. `TARGET_MIXTURE_WEIGHT_ERROR` is forbidden without an independent real mixed-garment target. Existing Dual-Support results are reused without new rendering.

## Operational result

- Training steps: `0`
- Renderer inferences: `0`
- Checkpoint download bytes: `0`
- Dataset download bytes: `0`
- Data mutations: `0`
- Paper modifications: `0`
- Bundle mutations: `0`
- Cloud recheck: `FAILED_NON_BLOCKING` because `canondress-cloud` DNS still did not resolve
- Paper final: `false`

Unique next task: `{NEXT_TASK}`
"""
    write_text(REPORT, report)

    handoff = {
        "schema_version": "canondressgs.project_control.external_baseline_feasibility_from_bundle.v1",
        "task_id": TASK_ID,
        "branch": NEW_BRANCH,
        "base_head": SOURCE_HEAD,
        "summary": str((OUT / "external_baseline_feasibility_final_summary.json").relative_to(ROOT)).replace("\\", "/"),
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
        "classification": FINAL_CLASSIFICATION,
        "next_task": NEXT_TASK,
        "next_task_started": False,
        "paper_final": False,
        "operations": summary["operations"],
    }
    write_json(HANDOFF, handoff)


if __name__ == "__main__":
    main()
