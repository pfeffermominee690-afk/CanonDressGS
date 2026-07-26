#!/usr/bin/env python3
"""Build the Subject00 V2 visual-fail and valid-region audit sidecars."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
HANDOFF = ROOT / "project_control_handoff"
SOURCE_WORKTREE = Path(r"E:\model_train\canondressgs_subject00_portrait_canary_v2_registered_outpaint")
ATTEMPTS_ROOT = Path(r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-GENERATION-001")
ATTEMPTS = {
    "attempt_001": ATTEMPTS_ROOT / "attempt_001",
    "attempt_002": ATTEMPTS_ROOT / "attempt_002_portrait_canary",
    "attempt_003": ATTEMPTS_ROOT / "attempt_003_portrait_canary_v2_registered_outpaint",
}

TASK_ID = "AAAI27-SUBJECT00-PORTRAIT-CANARY-V2-FAIL-AND-VALID-REGION-AUDIT-001"
SOURCE_BRANCH = "research/subject00-portrait-canary-v2-registered-outpaint-20260726"
SOURCE_HEAD = "95370bc256c2e98c849471481fd6248579584f44"
BRANCH = "research/subject00-v2-visual-fail-valid-region-audit-20260726"
REVIEWER = "USER_AND_GPT_MANUAL_REVIEW"
REVIEW_TIMESTAMP = "2026-07-26T09:16:49+08:00"
CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_V2_VISUAL_FAIL_RECORDED_VALID_REGION_PROTOCOL_AUDITED"
VISUAL_CLASSIFICATION = "SUBJECT00_PORTRAIT_CANARY_V2_TECHNICAL_PASS_VISUAL_FAIL_BACKGROUND_OUTPAINT"
NEXT_TASK = "USER_SELECT_SUBJECT00_VALID_REGION_OR_NATIVE_LANDSCAPE_PROTOCOL"
REQUEST_IDS = [
    "subject00_O01_slot00_cand00",
    "subject00_O03_slot03_cand00",
    "subject00_O04_slot02_cand00",
    "subject00_O01_slot06_cand00",
]

PREFLIGHT_PATH = RISK / "subject00_portrait_canary_v2_registration_preflight.json"
OUTPUTS_PATH = RISK / "subject00_portrait_canary_v2_output_registry.json"
RESOLUTION_PATH = RISK / "subject00_managed_output_resolution_distribution.json"
CONTACT_PATH = ATTEMPTS["attempt_003"] / "07_human_review" / "subject00_portrait_canary_v2_contact_sheet.png"
REVIEW_MANIFEST_PATH = ATTEMPTS["attempt_003"] / "07_human_review" / "subject00_portrait_canary_v2_review_manifest.json"


def canonical_sha256(value: Any) -> str:
    if isinstance(value, dict):
        value = dict(value)
        value.pop("content_sha256", None)
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip() + "\n")


def run(*command: str, cwd: Path) -> str:
    return subprocess.run(
        list(command), cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def inventory(root: Path) -> dict[str, Any]:
    files = [
        {
            "relative_path": item.relative_to(root).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in sorted(path for path in root.rglob("*") if path.is_file())
    ]
    return {
        "root": str(root),
        "file_count": len(files),
        "tree_sha256": canonical_sha256(files),
        "files": files,
    }


def validate_source_gate() -> None:
    facts = {
        "source_branch": run("git", "branch", "--show-current", cwd=SOURCE_WORKTREE),
        "source_head": run("git", "rev-parse", "HEAD", cwd=SOURCE_WORKTREE),
        "source_status": run("git", "status", "--short", cwd=SOURCE_WORKTREE),
        "target_branch": run("git", "branch", "--show-current", cwd=ROOT),
    }
    expected = {
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "source_status": "",
        "target_branch": BRANCH,
    }
    if facts != expected:
        raise RuntimeError(f"source gate mismatch: {facts!r}")


def manual_specs() -> dict[str, dict[str, Any]]:
    return {
        "subject00_O01_slot00_cand00": {
            "subject_scale_visual_result": "PASS_BASIC_PIXEL_SCALE_PRESERVED",
            "pose_direction_visual_result": "PASS_FRONT_BASICALLY_PRESERVED",
            "garment_correctness_visual_result": "PASS_O01_LIGHT_WHITE_HOODIE_AND_TROUSERS_BASICALLY_CORRECT",
            "human_completeness_visual_result": "PASS_ONE_PERSON_LIMBS_BASICALLY_COMPLETE",
            "background_extension_visual_result": "FAIL_WARPED_CEILING_AND_UNNATURAL_CARPET_PERSPECTIVE",
            "identity_reviewability": "INSUFFICIENT_FOR_HIGH_CONFIDENCE_PASS_FACE_HAND_DETAIL",
            "garment_boundary_reviewability": "LOCALLY_REVIEWABLE_NOT_FORMAL_TEACHER_GRADE",
            "local_passes": [
                "subject pixel scale basically preserved",
                "front pose basically preserved",
                "one person and limbs basically complete",
                "O01 light/white hoodie and trousers basically correct",
                "V1 severe subject shrink and gray canvas resolved",
            ],
            "failure_reasons": [
                "ceiling extension geometry is visibly warped",
                "lower carpet blocks and perspective are unnatural",
                "top and bottom extensions are not a continuous registered scene",
                "face, hands, and details are insufficient for a high-confidence identity pass",
            ],
        },
        "subject00_O03_slot03_cand00": {
            "subject_scale_visual_result": "PASS_BASIC_PIXEL_SCALE_PRESERVED",
            "pose_direction_visual_result": "PASS_LEFT_ORIENTATION_BASICALLY_PRESERVED",
            "garment_correctness_visual_result": "PASS_O03_SUIT_BASICALLY_CORRECT",
            "human_completeness_visual_result": "PASS_ONE_PERSON_LIMBS_BASICALLY_COMPLETE",
            "background_extension_visual_result": "FAIL_SEVERE_CEILING_AND_CARPET_OUTPAINT_ARTIFACTS",
            "identity_reviewability": "INSUFFICIENT_FOR_FORMAL_IDENTITY_PASS",
            "garment_boundary_reviewability": "LOCALLY_REVIEWABLE_NOT_FORMAL_TEACHER_GRADE",
            "local_passes": [
                "subject scale basically preserved",
                "left orientation basically preserved",
                "O03 suit semantics basically correct",
                "one person and limbs basically complete",
            ],
            "failure_reasons": [
                "ceiling contains arrow-like and mirror-like structures",
                "top scene geometry is not continuous with the real room",
                "carpet perspective and texture are deformed",
                "background outpainting artifacts are severe",
                "output is unusable as a registered Teacher target",
            ],
        },
        "subject00_O04_slot02_cand00": {
            "subject_scale_visual_result": "PASS_BASIC_PIXEL_SCALE_PRESERVED",
            "pose_direction_visual_result": "PASS_FRONT_RIGHT_BASICALLY_PRESERVED",
            "garment_correctness_visual_result": "PASS_O04_JACKET_AND_JEANS_BASICALLY_CORRECT",
            "human_completeness_visual_result": "PASS_ONE_PERSON_BODY_COMPLETE",
            "background_extension_visual_result": "FAIL_LARGE_CEILING_WEDGE_AND_CARPET_POLYGON",
            "identity_reviewability": "INSUFFICIENT_FOR_FORMAL_IDENTITY_PASS",
            "garment_boundary_reviewability": "LOCALLY_REVIEWABLE_NOT_FORMAL_TEACHER_GRADE",
            "local_passes": [
                "subject scale basically preserved",
                "front-right orientation basically preserved",
                "O04 jacket and jeans basically correct",
                "one person and body complete",
            ],
            "failure_reasons": [
                "top region contains a huge unnatural wedge-shaped ceiling structure",
                "bottom region contains a huge carpet polygon and abnormal perspective",
                "background spatial continuity failed",
                "generated regions are invalid for registered multiview supervision",
            ],
        },
        "subject00_O01_slot06_cand00": {
            "subject_scale_visual_result": "PASS_BASIC_PIXEL_SCALE_PRESERVED",
            "pose_direction_visual_result": "PASS_BACK_RIGHT_BASICALLY_PRESERVED",
            "garment_correctness_visual_result": "PASS_O01_LIGHT_WHITE_HOODIE_BASICALLY_CORRECT",
            "human_completeness_visual_result": "PASS_ONE_PERSON_LIMBS_COMPLETE",
            "background_extension_visual_result": "FAIL_ENLARGED_REPEATED_CARPET_TEXTURE_AND_PERSPECTIVE",
            "identity_reviewability": "INSUFFICIENT_FOR_FORMAL_IDENTITY_PASS",
            "garment_boundary_reviewability": "LOCALLY_REVIEWABLE_NOT_FORMAL_TEACHER_GRADE",
            "local_passes": [
                "subject scale basically preserved",
                "back-right orientation basically preserved",
                "O01 light/white hoodie basically correct",
                "one person and limbs complete",
            ],
            "failure_reasons": [
                "lower carpet texture is abnormally enlarged",
                "carpet blocks repeat and their shapes and perspective change",
                "top and bottom extensions do not preserve registered background continuity",
                "output is unusable as a formal Teacher target",
            ],
        },
    }


def code_audit() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.subject00.valid_region_code_path_audit.v1",
        "task_id": TASK_ID,
        "audit_mode": "READ_ONLY_STATIC_CODE_AUDIT",
        "conclusions": {
            "arbitrary_resolution_support": "PARTIAL",
            "per_view_resolution_support": "PARTIAL_BASE_PIPELINE_BUT_NO_FORMAL_TEACHER_DATASET",
            "camera_intrinsics_override_support": "YES_IF_MATERIALIZED_IN_CALIBRATION_OR_PER_RECORD_CAMERA_JSON",
            "validity_mask_support": "NO_INDEPENDENT_HxW_PIXEL_VALIDITY_MASK",
            "validity_mask_loss_coverage": "NONE_COMPLETE",
            "validity_mask_evaluation_coverage": "NONE",
            "registered_1024x1150_support": "TECHNICALLY_SUPPORTED_IF_COMMON_SIZE_AND_CORRECTED_K_ARE_MATERIALIZED; BACKEND_UNPROVEN",
        },
        "evidence": [
            {
                "question": "configured_resize",
                "path": "config/subject00.yaml",
                "symbols_or_lines": ["image_scaling lines 16 and 24"],
                "finding": "Subject00 uses image_scaling=1; no 1024x1536 constant is configured.",
            },
            {
                "question": "base_image_size",
                "path": "scene/dataset.py",
                "symbols_or_lines": ["resize_image lines 63-72", "ThumanDataset.__getitem__ lines 281-312"],
                "finding": "At scaling=1 source HxW is preserved and sample height/width come from the loaded image. The non-unit path has a dormant NameError because line 68 references msk instead of mask.",
            },
            {
                "question": "base_batching",
                "path": "scene/scene.py",
                "symbols_or_lines": ["Scene.__init__ lines 31-63", "camera collection lines 68-82"],
                "finding": "Global image_scaling is shared, while DataLoader batch_size=1 permits varying sample HxW technically. Camera K is collected per camera rather than per frame.",
            },
            {
                "question": "formal_teacher_size",
                "path": "scene/dressable_dataset.py",
                "symbols_or_lines": ["RenderingDressableDataset lines 132-239"],
                "finding": "A common arbitrary HxW is accepted, but line 239 rejects rendering samples that do not all share one image size.",
            },
            {
                "question": "camera_override",
                "path": "utils/dressable_camera_utils.py",
                "symbols_or_lines": ["validate_camera_data lines 8-35", "build_mmlphuman_camera lines 38-78"],
                "finding": "Per-record K or intrinsics and w2c or R/T are accepted and passed with explicit H/W. Crop/outpaint formulas are not applied automatically; corrected K must be written into the consumed camera data.",
            },
            {
                "question": "renderer_viewport",
                "path": "scene/gaussian_model.py",
                "symbols_or_lines": ["GaussianModel.render lines 588-610"],
                "finding": "gsplat rasterization consumes cam K, width, and height directly; there is no separate repository NDC transform to update.",
            },
            {
                "question": "base_losses_and_evaluation",
                "path": "train.py",
                "symbols_or_lines": ["training lines 70-87", "training_report lines 140-179"],
                "finding": "L1 and evaluation L1/PSNR do not accept an independent validity mask. LPIPS uses a person-mask-derived crop, not a registered central validity region.",
            },
            {
                "question": "base_lpips_crop",
                "path": "utils/image_utils.py",
                "symbols_or_lines": ["crop_image lines 59-95"],
                "finding": "The crop is based on foreground bbox and resized to 512; exclusion of portrait bands is incidental, not a validity contract.",
            },
            {
                "question": "rendering_losses",
                "path": "utils/rendering_loss_utils.py",
                "symbols_or_lines": ["rgb_reconstruction_loss lines 16-47", "alpha_mask_loss lines 49-66", "combined_rendering_loss lines 103-135"],
                "finding": "RGB L1/SSIM use the supplied foreground mask, alpha loss covers the full image, and LPIPS is unmasked. No separate validity mask reaches every loss.",
            },
            {
                "question": "valid_mask_name_collision",
                "path": "scene/dressable_dataset.py; scene/clothing_observation_encoder.py; train_dressable.py",
                "symbols_or_lines": ["dressable_dataset lines 54-61", "clothing_observation_encoder lines 159-179", "train_dressable lines 259-270 and 1393-1499"],
                "finding": "Existing valid_mask is per-anchor and reference_valid_mask is [K] or [K,1] per view. Neither is an HxW pixel-validity region.",
            },
            {
                "question": "oracle_losses_and_evaluation",
                "path": "utils/oracle_loss_utils.py; train_full_attribute_oracle.py",
                "symbols_or_lines": ["oracle_rendering_loss lines 58-124", "mask_iou lines 127-132", "_evaluate lines 73-93"],
                "finding": "Foreground/clothing masks drive losses and IoU, but no independent validity region is consumed by losses or metrics.",
            },
        ],
        "technical_scientific_distinction": {
            "technically_parsable": "Base MMLP-Human can parse/render arbitrary HxW at scaling=1; formal Teacher rendering can use one common arbitrary HxW.",
            "geometrically_registered": "Requires RGB/masks and a consumed K containing cx_new=cx-x_left and, only for a portrait canvas, cy_new=cy+top_extension.",
            "scientifically_valid": "Requires verified scene registration and complete invalid-region exclusion from optimization and evaluation; current code does not satisfy this for portrait bands.",
        },
        "camera_registration": {
            "cx_new": "Supported only after materializing cx-x_left into consumed K.",
            "cy_new": "Supported only after materializing cy+top_extension into consumed K for the full portrait viewport.",
            "per_view_override": "YES in per-record dressable camera JSON; base calibration is per camera, not per frame.",
            "renderer_consumes_updated_K": True,
            "viewport_or_ndc_update": "Width/height and K define the gsplat viewport; no additional repository NDC update was found.",
            "manifest_binding": "Possible only when the manifest is converted into the actual dataset/camera records; V2 preflight is not read automatically.",
            "central_only_supervision_with_full_viewport": "NOT_CURRENTLY_SUPPORTED because no independent pixel validity reaches all losses and evaluation.",
        },
        "validity_mask": {
            "independent_from_clothing_and_foreground": False,
            "image_l1": "NO independent validity; base full image, dressable uses foreground mask.",
            "ssim": "NO independent validity; zeroing outside foreground can contaminate windows at boundaries.",
            "lpips": "NO independent validity; dressable full image, base person crop, oracle foreground-multiplied.",
            "alpha_silhouette": "NO independent validity; invalid bands would still affect alpha BCE/dice/IoU.",
            "regularization": "Parameter regularizers are not pixel-local and require an explicit scientific decision rather than implicit masking.",
            "clothing_mask_can_substitute": False,
            "required": "A new HxW validity_mask aligned through every transform, loss, and metric.",
            "full_output_provenance": "YES; retain the full output/contact sheet for provenance while excluding invalid bands from supervision.",
        },
    }


def protocol_audit(resolution: dict[str, Any]) -> dict[str, Any]:
    required_changes = [
        "Add a dedicated HxW validity_mask path to dataset schemas/manifests and load it aligned with RGB and foreground masks.",
        "Apply every resize, crop, and augmentation identically to validity_mask using nearest-neighbor interpolation.",
        "Carry validity_mask through samples, episodes, and camera bindings without reusing anchor/view valid_mask fields.",
        "Mask full-scene RGB/L1 by validity and define SSIM on valid crops or validity-aware windows without zero-boundary contamination.",
        "Compute LPIPS on the registered valid crop rather than the full portrait canvas.",
        "Mask alpha BCE/dice, silhouette losses, and IoU by validity.",
        "Apply the same validity contract to evaluation L1, PSNR, SSIM, LPIPS, and IoU.",
        "Add tests that perturb invalid bands and prove unchanged losses/metrics plus zero invalid-region gradients.",
        "Materialize crop-updated K, including cx_new and any cy_new, in calibration/per-record camera JSON and test raster alignment.",
    ]
    return {
        "schema_version": "canondressgs.subject00.next_data_protocol_audit.v1",
        "task_id": TASK_ID,
        "execution_authorized": False,
        "recommended_protocol": "NO_EXECUTABLE_PROTOCOL_CURRENTLY",
        "engineering_preference": "Candidate A only after complete validity-mask implementation and tests; Candidate C if exact native 1024x1150 backend support is proven; Candidate B only after per-image registration audit; Candidate D rejected.",
        "candidates": {
            "A": {
                "name": "REGISTERED_VALID_REGION_PORTRAIT",
                "classification": "BLOCKED_BY_MISSING_END_TO_END_PIXEL_VALIDITY_SUPPORT",
                "recommendation_level": "CONDITIONAL_PREFERRED_RESEARCH_DIRECTION",
                "current_code_direct_support": False,
                "scientific_contract": "Potentially valid only after all optimization and evaluation paths ignore top/bottom invalid bands.",
                "required_code_changes": required_changes,
            },
            "B": {
                "name": "NATIVE_REGISTERED_LANDSCAPE",
                "classification": "CONDITIONAL_AUDIT_REQUIRED_HOLD",
                "recommendation_level": "BACKUP_ONLY_AFTER_REGISTRATION_AUDIT",
                "pipeline_support": "Base path partial; formal Teacher dataset rejects per-view HxW.",
                "scientific_contract": "Dimensions do not establish registration; every image needs camera/framing/pose/background audit.",
            },
            "C": {
                "name": "REGISTERED_CROP_NATIVE_1024x1150",
                "classification": "PIPELINE_TECHNICALLY_SUPPORTED_GENERATION_BACKEND_UNPROVEN_HOLD",
                "recommendation_level": "CONDITIONAL_ON_NATIVE_BACKEND_PROOF",
                "pipeline_support": "A common 1024x1150 dataset with corrected K is technically parsable/renderable.",
                "scientific_contract": "Strong if output is truly native registered 1024x1150 with no recomposition; managed backend capability is not established.",
            },
            "D": {
                "name": "CONTINUE_VERTICAL_OUTPAINT",
                "classification": "REJECT",
                "recommendation_level": "REJECT",
                "evidence": "V2 is 0/4 visual pass with systematic scene-geometry failure.",
                "scientific_contract": "Invalid; prompt tuning or candidate selection cannot hide a systematic registration failure.",
            },
        },
        "attempt_001_landscape_reuse": {
            "classification": "CONDITIONAL_AUDIT_REQUIRED",
            "non_exact_output_count": 43,
            "landscape_count": resolution["observed_patterns"]["landscape_count"],
            "incompatible_portrait_count": resolution["aspect_ratio_category_distribution"]["D_PORTRAIT_INCOMPATIBLE_ASPECT"],
            "dimension_distribution": {
                key: count for key, count in resolution["exact_dimension_distribution"].items() if key != "1024x1536"
            },
            "near_original_1330x1150": "YES for the 38 landscape outputs, especially dominant 1349x1166, as dimensions only.",
            "framing_observation": "Reviewed examples often retain subject scale/framing better than V1 portrait outputs, but no frozen all-43 quantitative registration audit exists.",
            "intrinsics_update_feasibility": "Only if each native output is proven to be a registered viewport transform. A changed size alone is insufficient to derive K.",
            "recomposition_risk": "HIGH; existing generated outputs can change camera composition and background geometry rather than only image dimensions.",
            "reliable_registered_subset": "NOT_ESTABLISHED",
            "required_next_evidence": "Per-image camera, framing, pose, subject-scale, and background-registration audit for all 43; no image may be accepted by this audit.",
        },
        "required_code_changes": required_changes,
        "scientific_risks": [
            "Generated top/bottom pixels can inject mutually inconsistent multiview scene geometry.",
            "Foreground or clothing masks cannot represent whether background pixels are registered observations.",
            "Unmasked alpha, silhouette, LPIPS, and evaluation metrics can optimize or reward invalid bands.",
            "Per-view native dimensions do not imply a known camera transform when the generator recomposes framing.",
            "A dormant non-unit resize NameError and one-size Teacher constraint limit claimed arbitrary-resolution support.",
        ],
        "paper_final": False,
        "next_task": NEXT_TASK,
    }


def main() -> int:
    validate_source_gate()
    preflight = load_json(PREFLIGHT_PATH)
    outputs = load_json(OUTPUTS_PATH)
    resolution = load_json(RESOLUTION_PATH)
    review_manifest = load_json(REVIEW_MANIFEST_PATH)
    preflight_by_id = {item["request_id"]: item for item in preflight["records"]}
    output_by_id = {item["request_id"]: item for item in outputs["records"]}
    specs = manual_specs()

    if list(preflight_by_id) != REQUEST_IDS or list(output_by_id) != REQUEST_IDS or list(specs) != REQUEST_IDS:
        raise RuntimeError("V2 request set/order mismatch")
    if review_manifest["visual_decision_count"] != 0:
        raise RuntimeError("frozen review manifest has been populated")

    records: list[dict[str, Any]] = []
    for request_id in REQUEST_IDS:
        camera = preflight_by_id[request_id]
        output = output_by_id[request_id]
        spec = specs[request_id]
        records.append({
            "request_id": request_id,
            "garment": camera["garment_id"],
            "slot": camera["slot_id"],
            "camera": camera["camera_name"],
            "source_sha256": camera["source_sha256"],
            "derived_canvas_sha256": camera["derived_canvas_sha256"],
            "output_sha256": output["output_sha256"],
            "native_resolution_pass": output["status"] == "NATIVE_RESOLUTION_PASS",
            "registration_preflight_pass": camera["registration_decision"] == "REGISTRATION_PREFLIGHT_PASS",
            **{key: spec[key] for key in [
                "subject_scale_visual_result", "pose_direction_visual_result",
                "garment_correctness_visual_result", "human_completeness_visual_result",
                "background_extension_visual_result", "identity_reviewability",
                "garment_boundary_reviewability",
            ]},
            "local_passes": spec["local_passes"],
            "human_visual_decision": "FAIL",
            "failure_reasons": spec["failure_reasons"],
            "accepted": False,
            "teacher_target": False,
            "reviewer": REVIEWER,
            "review_timestamp": REVIEW_TIMESTAMP,
        })

    manual = {
        "schema_version": "canondressgs.subject00.portrait_canary_v2_manual_visual_adjudication.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "reviewer": REVIEWER,
        "review_timestamp": REVIEW_TIMESTAMP,
        "contact_sheet_path": str(CONTACT_PATH),
        "contact_sheet_sha256": file_sha256(CONTACT_PATH),
        "original_review_manifest_path": str(REVIEW_MANIFEST_PATH),
        "original_review_manifest_sha256": file_sha256(REVIEW_MANIFEST_PATH),
        "original_review_manifest_mutated": False,
        "request_count": 4,
        "human_visual_pass_count": 0,
        "human_visual_fail_count": 4,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "remaining_39_authorization": "DENIED",
        "remaining_39_generated_count": 0,
        "formal_base": "PENDING",
        "subject00_paper_positive_claim_count": 0,
        "paper_final": False,
        "classification": VISUAL_CLASSIFICATION,
        "records": records,
        "shared_conclusions": [
            "registered horizontal cropping preserves subject pixel scale",
            "crop-updated camera intrinsics are geometrically viable",
            "native portrait output is stable and V1 subject shrink/gray canvas is resolved",
            "unrestricted vertical outpainting systematically changes scene geometry",
            "native portrait resolution is not equivalent to registered multiview supervision",
            "generated background must not become valid Teacher supervision",
        ],
    }
    write_json(RISK / "subject00_portrait_canary_v2_manual_visual_adjudication_20260726.json", manual)

    evidence = {
        "schema_version": "canondressgs.subject00.portrait_canary_v2_visual_fail_evidence_registry.v1",
        "task_id": TASK_ID,
        "evidence_mode": "READ_ONLY_INPUTS_AND_TEXT_ONLY_SIDECARS",
        "source_gate": {
            "branch": SOURCE_BRANCH,
            "head": SOURCE_HEAD,
            "worktree": str(SOURCE_WORKTREE),
            "clean_at_build": True,
        },
        "fixed_evidence": {
            "contact_sheet": {"path": str(CONTACT_PATH), "sha256": file_sha256(CONTACT_PATH)},
            "review_manifest": {"path": str(REVIEW_MANIFEST_PATH), "sha256": file_sha256(REVIEW_MANIFEST_PATH)},
            "registration_preflight": {"path": str(PREFLIGHT_PATH), "sha256": file_sha256(PREFLIGHT_PATH)},
            "output_registry": {"path": str(OUTPUTS_PATH), "sha256": file_sha256(OUTPUTS_PATH)},
            "resolution_distribution": {"path": str(RESOLUTION_PATH), "sha256": file_sha256(RESOLUTION_PATH)},
        },
        "attempt_snapshots": {name: inventory(path) for name, path in ATTEMPTS.items()},
        "manual_record_count": len(records),
        "manual_records": records,
        "prohibited_action_counts": {
            "new_generation_calls": 0,
            "external_api_calls": 0,
            "api_key_reads": 0,
            "cloud_image_writes": 0,
            "attempt_001_mutations": 0,
            "attempt_002_mutations": 0,
            "attempt_003_mutations": 0,
            "teacher_bank_mutations": 0,
            "training_runs": 0,
            "paper_modifications": 0,
        },
        "classification": VISUAL_CLASSIFICATION,
        "paper_final": False,
    }
    write_json(RISK / "subject00_portrait_canary_v2_visual_fail_evidence_registry_20260726.json", evidence)

    code = code_audit()
    write_json(RISK / "subject00_valid_region_teacher_pipeline_code_audit_20260726.json", code)
    protocol = protocol_audit(resolution)
    write_json(RISK / "subject00_valid_region_protocol_comparison_20260726.json", protocol)

    report_rows = "\n".join(
        f"| `{item['request_id']}` | {item['garment']} | {item['slot']} | {item['camera']} | FAIL | {item['background_extension_visual_result']} |"
        for item in records
    )
    report = f"""# Subject00 Portrait Canary V2 Visual Fail Report

- Task: `{TASK_ID}`
- Reviewer: `{REVIEWER}`
- Contact sheet SHA-256: `{manual['contact_sheet_sha256']}`
- Human visual result: `0 PASS / 4 FAIL`
- Classification: `{VISUAL_CLASSIFICATION}`

| Request | Garment | Slot | Camera | Decision | Primary failure |
| --- | --- | --- | --- | --- | --- |
{report_rows}

## Adjudication

All four requests passed the registered-crop preflight and native `1024x1536` output check. They also resolved the V1 subject-shrink and gray-canvas failure. These are engineering passes only.

All four fail visual acceptance because generated ceiling and carpet regions do not preserve registered room geometry. The failures include warped or wedge-shaped ceiling structures, repeated or enlarged carpet blocks, and inconsistent perspective. The generated top and bottom bands are not valid multiview observations and cannot be used as Teacher supervision.

The original review manifest remains unchanged. This report and its JSON adjudication are sidecars. `accepted=0`, `Teacher targets=0`, remaining-39 authorization is `DENIED`, Formal Base is `PENDING`, and `PAPER_FINAL=false`.
"""
    write_text(RISK / "SUBJECT00_PORTRAIT_CANARY_V2_VISUAL_FAIL_REPORT_20260726.md", report)

    comparison_rows = "\n".join(
        f"| {key} | `{value['name']}` | `{value['classification']}` | `{value['recommendation_level']}` |"
        for key, value in protocol["candidates"].items()
    )
    comparison = f"""# Subject00 Valid-Region Protocol Comparison

| Candidate | Protocol | Classification | Recommendation |
| --- | --- | --- | --- |
{comparison_rows}

## Decision Boundary

No candidate is executable in this task. Candidate A is the preferred engineering direction only after an independent HxW validity mask reaches every relevant loss and evaluation metric. Candidate C becomes viable if the generation backend proves native registered `1024x1150` output. Candidate B remains a backup pending per-image registration and pipeline work. Candidate D is rejected by the fixed `0/4` V2 result.

The unique next action is `{NEXT_TASK}`.
"""
    write_text(RISK / "SUBJECT00_VALID_REGION_PROTOCOL_COMPARISON_20260726.md", comparison)

    summary = {
        "schema_version": "canondressgs.subject00.v2_visual_fail_valid_region_audit_summary.v1",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "new_branch": BRANCH,
        "worktree": str(ROOT),
        "contact_sheet_sha256": manual["contact_sheet_sha256"],
        "request_count": 4,
        "human_visual_pass_count": 0,
        "human_visual_fail_count": 4,
        "accepted_count": 0,
        "teacher_target_count": 0,
        "remaining_39_authorization": "DENIED",
        "remaining_39_generated_count": 0,
        "formal_base": "PENDING",
        "new_generation_calls": 0,
        "external_api_calls": 0,
        "api_key_reads": 0,
        "cloud_image_writes": 0,
        "paper_modifications": 0,
        "paper_final": False,
        "visual_classification": VISUAL_CLASSIFICATION,
        "final_classification": CLASSIFICATION,
        "next_task": NEXT_TASK,
    }
    write_json(RISK / "subject00_portrait_canary_v2_visual_fail_valid_region_audit_final_summary_20260726.json", summary)

    handoff = {
        **summary,
        "schema_version": "canondressgs.subject00.v2_visual_fail_valid_region_handoff.v1",
        "manual_adjudication": "paper_protocol/reviewer_risk/subject00_portrait_canary_v2_manual_visual_adjudication_20260726.json",
        "evidence_registry": "paper_protocol/reviewer_risk/subject00_portrait_canary_v2_visual_fail_evidence_registry_20260726.json",
        "code_audit": "paper_protocol/reviewer_risk/subject00_valid_region_teacher_pipeline_code_audit_20260726.json",
        "protocol_comparison": "paper_protocol/reviewer_risk/subject00_valid_region_protocol_comparison_20260726.json",
        "execution_status": "WAITING_FOR_USER_PROTOCOL_SELECTION",
    }
    write_json(HANDOFF / "subject00_v2_visual_fail_valid_region_audit_handoff_20260726.json", handoff)

    print(json.dumps({
        "status": "BUILT",
        "request_count": len(records),
        "visual_pass": 0,
        "visual_fail": 4,
        "attempt_file_counts": {key: value["file_count"] for key, value in evidence["attempt_snapshots"].items()},
        "classification": CLASSIFICATION,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
